import argparse
import gc
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from src.stage0.build_qwen_embeddings import last_token_pool, pick_device, pick_dtype
from src.stage0.evaluate_stage0_policy_recall import (
    LungCache,
    bbox_for_mode,
    bbox_volume,
    policy_to_bbox,
    union_bboxes,
)
from src.stage0.anatomy_expert import TotalSegLobeCache, anatomy_record, anatomy_to_bbox, parse_anatomy
from src.stage0.predict_router import load_head, predict_rows
from src.stage0.router_utils import read_jsonl, write_jsonl


def resolve_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    for base in (Path.cwd(), Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent, Path(__file__).resolve().parent.parents[1]):
        candidate = (base / p).resolve()
        if candidate.exists():
            return candidate
    return (Path(__file__).resolve().parent / p).resolve()


def resolve_output_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (Path.cwd() / p).resolve()


def read_prompt_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            if "prompt" not in row:
                raise ValueError(f"missing prompt in line {idx + 1}: {path}")
            row.setdefault("id", f"finding_{idx:04d}")
            rows.append(row)
    return rows


def parse_margin(text):
    vals = [float(x) for x in text.split(",")]
    if len(vals) != 3:
        raise ValueError("margin must be H,W,D")
    return vals


def crop_nifti(nii, bbox):
    h0, h1, w0, w1, d0, d1 = bbox
    arr = np.asanyarray(nii.dataobj)
    cropped = arr[h0:h1, w0:w1, d0:d1]
    affine = nii.affine.copy()
    affine[:3, 3] = nib.affines.apply_affine(nii.affine, [h0, w0, d0])
    header = nii.header.copy()
    return nib.Nifti1Image(cropped, affine, header)


def load_router(checkpoint_path: Path, device):
    head, ckpt = load_head(checkpoint_path, device)
    cfg = ckpt["config"]
    qwen_cfg = cfg["qwen"]
    model_path_override = os.environ.get("QWEN_MODEL_PATH")
    model_path = Path(model_path_override) if model_path_override else resolve_path(qwen_cfg["model_path"])
    dtype = pick_dtype(qwen_cfg.get("dtype", "auto"), device)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Qwen model path not found: {model_path}. "
            "Set config.yaml models.qwen_embedding or QWEN_MODEL_PATH."
        )
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    embedder = AutoModel.from_pretrained(
        str(model_path), trust_remote_code=True, torch_dtype=dtype
    ).to(device)
    embedder.eval()
    return tokenizer, embedder, head, cfg


def bbox_for_anatomy_groups(case_preds, prompt_bboxes, mode):
    if mode in {"case_union", "finding"}:
        return bbox_for_mode(case_preds, prompt_bboxes, mode)
    by_group = {}
    for p, bbox in zip(case_preds, prompt_bboxes):
        by_group.setdefault(p["anatomy_group"], []).append((p, bbox))
    mapping = {}
    groups = []
    for group_id, items in sorted(by_group.items()):
        group_bbox = union_bboxes([bbox for _, bbox in items])
        for p, _ in items:
            mapping[p["id"]] = group_bbox
        groups.append({"group_id": group_id, "bbox": group_bbox, "n_findings": len(items)})
    return mapping, groups


def process_one_case(args_dict: dict, model_bundle: dict | None = None):
    """Run Stage0 for one CT case. If model_bundle is provided, reuse pre-loaded models.

    args_dict: {image, prompts_jsonl, out_dir, checkpoint, crop_mode,
                spatial_mode, mapping_mode, totalseg_device, totalseg_fast,
                conservative_margin, moderate_margin, aggressive_margin,
                router_predictions_jsonl (optional)}
    model_bundle: {device, tokenizer, embedder, head, head_cfg (optional for router),
                   lung_cache (optional), totalseg_cache (optional)}
    Returns: (preds, crop_groups, summary) dicts
    """
    image_path = resolve_path(args_dict["image"])
    prompts_path = resolve_path(args_dict["prompts_jsonl"])
    out_dir = resolve_output_path(args_dict["out_dir"])
    crop_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    rows = read_prompt_rows(prompts_path)
    if not rows:
        raise SystemExit(f"no prompts found: {prompts_path}")

    if model_bundle and model_bundle.get("embedding_cache") is not None:
        # Cache mode: use precomputed embeddings + head only, no tokenizer/embedder
        device = model_bundle["device"]
        preds = predict_rows(rows, None, None,
                            model_bundle["head"], model_bundle["head_cfg"], device,
                            embedding_cache=model_bundle["embedding_cache"])
    elif model_bundle and all(k in model_bundle for k in ("tokenizer", "embedder", "head", "head_cfg")):
        device = model_bundle["device"]
        preds = predict_rows(rows, model_bundle["tokenizer"], model_bundle["embedder"],
                            model_bundle["head"], model_bundle["head_cfg"], device)
    elif args_dict.get("router_predictions_jsonl"):
        preds = read_jsonl(resolve_path(args_dict["router_predictions_jsonl"]))
    else:
        device = pick_device("auto")
        checkpoint = resolve_path(args_dict.get("checkpoint", "artifacts/models/router_head_best.pt"))
        tokenizer, embedder, head, cfg = load_router(checkpoint, device)
        preds = predict_rows(rows, tokenizer, embedder, head, cfg, device)
        del tokenizer, embedder, head, cfg
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    nii = nib.load(str(image_path))
    shape = nii.shape
    zooms = nii.header.get_zooms()[:3]
    margins = {
        "conservative": parse_margin(args_dict.get("conservative_margin", "60,60,50")),
        "moderate": parse_margin(args_dict.get("moderate_margin", "40,40,30")),
        "aggressive": parse_margin(args_dict.get("aggressive_margin", "20,20,20")),
    }

    spatial_mode = args_dict.get("spatial_mode", "anatomy")
    mapping_mode = args_dict.get("mapping_mode", "fixed")
    crop_mode = args_dict.get("crop_mode", "policy_group")

    # Lung cache (try to load, fall back to stub if model unavailable)
    lung_info = None
    try:
        if model_bundle and "lung_cache" in model_bundle:
            lung_cache = model_bundle["lung_cache"]
        else:
            lung_cache = LungCache()
        lung_info = lung_cache.get(image_path, shape, nii.affine, mapping_mode)
    except Exception as exc:
        print(f"  [WARN] lungmask unavailable: {exc}")
        print(f"  [WARN] falling back to whole-CT bounding box for safety")
        lung_cache = None
        lung_info = None

    # TotalSegmentator cache
    totalseg_cache = None
    if spatial_mode == "anatomy":
        if model_bundle and "totalseg_cache" in model_bundle:
            totalseg_cache = model_bundle["totalseg_cache"]
        else:
            totalseg_cache = TotalSegLobeCache(
                fast=args_dict.get("totalseg_fast", True),
                device=args_dict.get("totalseg_device", "gpu"))
        seg = totalseg_cache.get_seg(image_path, shape)

    prompt_bboxes = []
    for p in preds:
        if lung_info is not None:
            fallback_bbox = policy_to_bbox(p["final_policy"], lung_info, shape, zooms, margins)
        else:
            fallback_bbox = [0, shape[0], 0, shape[1], 0, shape[2]]  # whole CT
        p.update(anatomy_record(p["prompt"], p["laterality"], p["final_tightness"]))
        if spatial_mode == "anatomy":
            decision = parse_anatomy(p["prompt"], p["laterality"])
            bbox = anatomy_to_bbox(decision, seg, shape, zooms, margins[p["final_tightness"]])
            if bbox is None:
                p["anatomy_fallback_reason"] = "empty_totalseg_lobe_mask"
                bbox = fallback_bbox
            else:
                p["anatomy_fallback_reason"] = None
        else:
            bbox = fallback_bbox
        prompt_bboxes.append(bbox)

    write_jsonl(preds, out_dir / "stage0_router_predictions.jsonl")
    if args_dict.get("spatial_mode", "anatomy") == "anatomy":
        finding_bboxes, groups = bbox_for_anatomy_groups(preds, prompt_bboxes, crop_mode)
    else:
        finding_bboxes, groups = bbox_for_mode(preds, prompt_bboxes, crop_mode)

    group_records = []
    stem = image_path.name
    if stem.endswith(".nii.gz"):
        stem = stem[:-7]
    else:
        stem = Path(stem).stem

    for group in groups:
        bbox = group["bbox"] or [0, shape[0], 0, shape[1], 0, shape[2]]
        group_id = group["group_id"].replace(":", "_").replace("/", "_")
        crop_name = f"{stem}__{group_id}.nii.gz"
        crop_path = crop_dir / crop_name
        nib.save(crop_nifti(nii, bbox), str(crop_path))

        group_prompt_ids = [
            p["id"] for p in preds if finding_bboxes.get(p["id"]) == bbox
        ]
        group_records.append({
            "group_id": group["group_id"],
            "image": str(crop_path),
            "source_image": str(image_path),
            "bbox_hwd": bbox,
            "volume_ratio": bbox_volume(bbox) / max(1, int(np.prod(shape))),
            "n_findings": len(group_prompt_ids),
            "finding_ids": group_prompt_ids,
            "crop_mode": crop_mode,
            "spatial_mode": spatial_mode,
            "mapping_mode": mapping_mode,
            "lungmask_transform": lung_info.get("transform") if lung_info else None,
            "right_label": lung_info.get("right_label") if lung_info else None,
            "left_label": lung_info.get("left_label") if lung_info else None,
        })

    write_jsonl(group_records, out_dir / "stage0_crop_groups.jsonl")
    summary = {
        "image": str(image_path),
        "prompts": str(prompts_path),
        "out_dir": str(out_dir),
        "crop_mode": crop_mode,
        "spatial_mode": spatial_mode,
        "n_findings": len(preds),
        "n_crop_groups": len(group_records),
        "mean_group_volume_ratio": float(np.mean([r["volume_ratio"] for r in group_records])),
        "crop_groups": group_records,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return preds, group_records, summary


def main():
    parser = argparse.ArgumentParser(description="Run current Stage0 router + lungmask policy crops for one CT.")
    parser.add_argument("--image", required=True, help="Input CT .nii/.nii.gz")
    parser.add_argument("--prompts-jsonl", required=True, help="JSONL with at least {id,prompt}")
    parser.add_argument("--router-predictions-jsonl", help="Existing router predictions; skips Qwen router inference")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--crop-mode", choices=["policy_group", "finding", "case_union"], default="policy_group")
    parser.add_argument("--spatial-mode", choices=["lungmask", "anatomy"], default="anatomy")
    parser.add_argument("--mapping-mode", choices=["fixed", "auto"], default="fixed")
    parser.add_argument("--totalseg-device", default="gpu")
    parser.add_argument("--totalseg-fast", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--conservative-margin", default="60,60,50")
    parser.add_argument("--moderate-margin", default="40,40,30")
    parser.add_argument("--aggressive-margin", default="20,20,20")
    args = parser.parse_args()

    kwargs = {
        "image": args.image, "prompts_jsonl": args.prompts_jsonl,
        "out_dir": args.out_dir, "checkpoint": args.checkpoint,
        "crop_mode": args.crop_mode, "spatial_mode": args.spatial_mode,
        "mapping_mode": args.mapping_mode, "totalseg_device": args.totalseg_device,
        "totalseg_fast": args.totalseg_fast,
        "conservative_margin": args.conservative_margin,
        "moderate_margin": args.moderate_margin,
        "aggressive_margin": args.aggressive_margin,
    }
    if args.router_predictions_jsonl:
        kwargs["router_predictions_jsonl"] = args.router_predictions_jsonl

    _, _, summary = process_one_case(kwargs)
    with (Path(kwargs["out_dir"]) / "stage0_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
