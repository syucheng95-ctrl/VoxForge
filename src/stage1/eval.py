"""Sliding-window evaluation for a VoxTell checkpoint."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import nibabel as nib

from src.stage1.config import build_voxtell_from_checkpoint


from batchgenerators.utilities.file_and_folder_operations import load_json  # noqa: E402
from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image  # noqa: E402
from nnunetv2.preprocessing.normalization.default_normalization_schemes import ZScoreNormalization  # noqa: E402
from src.voxtell.inference.predictor import VoxTellPredictor  # noqa: E402
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient  # noqa: E402


class EvalPredictor(VoxTellPredictor):
    def __init__(
        self,
        model_dir: str,
        device: torch.device = torch.device("cuda"),
        text_encoding_model: str = "Qwen/Qwen3-Embedding-4B",
        embedding_dir: Path | None = None,
    ) -> None:
        self.embedding_dir = embedding_dir
        self.current_finding_ids = None

        if embedding_dir is None:
            super().__init__(model_dir=model_dir, device=device, text_encoding_model=text_encoding_model)
            return

        self.device = device
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        self.normalization = ZScoreNormalization(intensityproperties={})
        self.tile_step_size = 0.5
        self.perform_everything_on_device = True
        self.tokenizer = None
        self.text_backbone = None
        self.max_text_length = 8192
        self.patch_size = load_json(str(Path(model_dir) / "plans.json"))["configurations"]["3d_fullres"]["patch_size"]
        self.network = None

    @torch.inference_mode()
    def embed_text_prompts(self, text_prompts):
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]
        if self.embedding_dir is not None:
            if self.current_finding_ids is None:
                raise RuntimeError("current_finding_ids must be set when using --embedding-dir")
            if len(self.current_finding_ids) != len(text_prompts):
                raise RuntimeError(
                    f"finding id count {len(self.current_finding_ids)} != prompt count {len(text_prompts)}"
                )
            embeddings = []
            for fid in self.current_finding_ids:
                path = self.embedding_dir / f"{fid}.pt"
                if not path.exists():
                    raise FileNotFoundError(f"Missing text embedding: {path}")
                embeddings.append(torch.load(path, map_location="cpu", weights_only=True).float())
            return torch.stack(embeddings, dim=0).view(1, len(embeddings), -1).to(self.device)

        from src.voxtell.utils.text_embedding import last_token_pool, wrap_with_instruction
        text_prompts = wrap_with_instruction(text_prompts)
        tokens = self.tokenizer(text_prompts, padding=True, truncation=True, max_length=self.max_text_length, return_tensors="pt")
        tokens = {k: v.to("cpu") for k, v in tokens.items()}
        self.text_backbone = self.text_backbone.to("cpu")
        emb = self.text_backbone(**tokens)
        return last_token_pool(emb.last_hidden_state, tokens["attention_mask"]).view(1, len(text_prompts), -1).to(self.device)

    @torch.inference_mode()
    def predict_single_image_logits(self, data, text_prompts) -> np.ndarray:
        data, bbox, orig_shape = self.preprocess(data)
        embeddings = self.embed_text_prompts(text_prompts)
        logits = self.predict_sliding_window_return_logits(data, embeddings).detach().cpu().float().numpy()
        logits_reverted_cropping = np.zeros([logits.shape[0], *orig_shape], dtype=np.float32)
        return insert_crop_into_image(logits_reverted_cropping, logits, bbox)


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def metrics(pred, gt):
    pb = (pred > 0).astype(np.uint8)
    gb = (gt > 0).astype(np.uint8)
    inter = (pb & gb).sum()
    ps, gs = pb.sum(), gb.sum()
    d = (2 * inter + 1e-6) / (ps + gs + 1e-6)
    r = (inter + 1e-6) / (gs + 1e-6)
    p = (inter + 1e-6) / (ps + 1e-6) if ps > 0 else 0.0
    return d, r, p, int(ps), int(gs)


def sigmoid_np(logits: np.ndarray):
    logits = np.clip(logits, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-logits))


def parse_thresholds(value: str):
    thresholds = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        threshold = float(item)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        thresholds.append(threshold)
    if not thresholds:
        raise ValueError("--thresholds cannot be empty")
    return thresholds


def empty_bucket():
    return {"dice": 0.0, "recall": 0.0, "precision": 0.0, "pred_voxels": 0, "gt_voxels": 0, "n": 0}


def add_metric(bucket, d, r, p, pv, gv):
    bucket["dice"] += d
    bucket["recall"] += r
    bucket["precision"] += p
    bucket["pred_voxels"] += pv
    bucket["gt_voxels"] += gv
    bucket["n"] += 1


def summarize_bucket(bucket):
    n = max(bucket["n"], 1)
    return {
        "n_findings": bucket["n"],
        "dice": bucket["dice"] / n,
        "recall": bucket["recall"] / n,
        "precision": bucket["precision"] / n,
        "pred_voxels": bucket["pred_voxels"],
        "gt_voxels": bucket["gt_voxels"],
        "vol_ratio": bucket["pred_voxels"] / max(bucket["gt_voxels"], 1),
    }


def load_gt_mask(mask_path: Path, reader: NibabelIOWithReorient, check_stage2_order: bool = True):
    gt, _ = reader.read_seg(str(mask_path))
    if gt.ndim == 4 and gt.shape[0] == 1:
        gt = gt[0]
    gt = (gt > 0).astype(np.uint8)

    # Stage2 uses raw NIfTI arrays in xyz/HWD order and explicitly transposes to zyx/DWH.
    # For these Stage1 masks this should match nnUNet's canonical read_seg output.
    if check_stage2_order:
        raw = np.asarray(nib.load(str(mask_path)).dataobj)
        raw_zyx = (np.transpose(raw, (2, 1, 0)) > 0).astype(np.uint8)
        if raw_zyx.shape != gt.shape or not np.array_equal(raw_zyx, gt):
            print(
                f"  [WARN] GT orientation differs from raw transpose(2,1,0): "
                f"{mask_path.name} read_seg={gt.shape} raw_t210={raw_zyx.shape}"
            )
    return gt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit-cases", type=int, default=None)
    parser.add_argument("--text-model", type=str, default=None)
    parser.add_argument("--embedding-dir", type=Path, default=None,
                        help="Use precomputed embeddings/*.pt instead of online Qwen embeddings.")
    parser.add_argument("--thresholds", default="0.5",
                        help="Comma-separated sigmoid thresholds for full-volume sweep.")
    parser.add_argument("--per-finding-output", type=Path, default=None,
                        help="Optional JSONL file with per-finding diagnostics.")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    thresholds = parse_thresholds(args.thresholds)

    # Load model directly from checkpoint
    model = build_voxtell_from_checkpoint(args.model_dir, device=torch.device("cpu"))
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(device).eval()

    text_model_name = args.text_model or "Qwen/Qwen3-Embedding-4B"
    predictor = EvalPredictor(
        model_dir=str(args.model_dir),
        device=device,
        text_encoding_model=text_model_name,
        embedding_dir=args.embedding_dir,
    )
    predictor.network = model

    reader = NibabelIOWithReorient()

    # Group by case
    rows = load_jsonl(args.manifest)
    case_map = {}
    for r in rows:
        case_map.setdefault(r["case_name"], []).append(r)

    cases = sorted(case_map.items())
    if args.limit_cases:
        cases = cases[:args.limit_cases]

    buckets = {str(t): empty_bucket() for t in thresholds}
    per_finding_rows = []
    skipped_images, missing_masks, failed_cases = 0, 0, 0

    for ci, (cn, case_rows) in enumerate(cases, 1):
        ipath = args.image_root / cn
        if not ipath.exists():
            print(f"  [SKIP] {cn}")
            skipped_images += 1
            continue

        t0 = time.time()
        img, props = reader.read_images([str(ipath)])
        prompts = [r["prompt"] for r in case_rows]
        predictor.current_finding_ids = [r["id"] for r in case_rows]
        try:
            logits = predictor.predict_single_image_logits(img, prompts)
        except Exception as e:
            print(f"  [ERR] {cn}: {e}")
            failed_cases += 1
            continue

        for i, row in enumerate(case_rows):
            mp = args.label_root / row["label"].replace("labels_finding/", "").replace("\\", "/")
            if not mp.exists():
                print(f"  [SKIP] missing mask: {mp}")
                missing_masks += 1
                continue
            gt = load_gt_mask(mp, reader)
            prob = sigmoid_np(logits[i])
            finding_diag = {
                "case_name": cn,
                "id": row["id"],
                "category": row.get("category", ""),
                "prompt": row["prompt"],
                "gt_voxels": int(gt.sum()),
                "max_prob": float(prob.max()),
                "mean_prob": float(prob.mean()),
                "gt_max_prob": float(prob[gt > 0].max()) if gt.any() else 0.0,
                "gt_mean_prob": float(prob[gt > 0].mean()) if gt.any() else 0.0,
                "thresholds": {},
            }
            for threshold in thresholds:
                key = str(threshold)
                d, r, pr, pv, gv = metrics(prob > threshold, gt)
                add_metric(buckets[key], d, r, pr, pv, gv)
                finding_diag["thresholds"][key] = {
                    "dice": d,
                    "recall": r,
                    "precision": pr,
                    "pred_voxels": pv,
                }
            per_finding_rows.append(finding_diag)

        print(f"[{ci}/{len(cases)}] {cn} ({len(prompts)} prompts, {time.time()-t0:.0f}s)")

    n_evaluated = sum(bucket["n"] for bucket in buckets.values()) // max(len(buckets), 1)
    if n_evaluated == 0:
        raise RuntimeError(
            "No findings were evaluated. "
            f"skipped_images={skipped_images}, missing_masks={missing_masks}, failed_cases={failed_cases}"
        )

    threshold_results = {key: summarize_bucket(bucket) for key, bucket in buckets.items()}
    default_key = str(thresholds[0])
    default_res = threshold_results[default_key]
    res = {
        "checkpoint": str(args.checkpoint),
        "n_findings": default_res["n_findings"],
        "dice": default_res["dice"], "recall": default_res["recall"], "precision": default_res["precision"],
        "pred_voxels": default_res["pred_voxels"],
        "gt_voxels": default_res["gt_voxels"],
        "vol_ratio": default_res["vol_ratio"],
        "thresholds": threshold_results,
        "skipped_images": skipped_images,
        "missing_masks": missing_masks,
        "failed_cases": failed_cases,
    }
    print(f"\nResults:")
    for key, item in threshold_results.items():
        print(
            f"  th={key}: Dice={item['dice']:.4f} Recall={item['recall']:.4f} "
            f"Precision={item['precision']:.4f} VolRatio={item['vol_ratio']:.2f}"
        )

    out_path = args.output or args.checkpoint.parent.parent / "eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved to {out_path}")

    per_finding_out = args.per_finding_output or out_path.with_suffix(".per_finding.jsonl")
    with per_finding_out.open("w", encoding="utf-8") as f:
        for row in per_finding_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved per-finding diagnostics to {per_finding_out}")


if __name__ == "__main__":
    main()
