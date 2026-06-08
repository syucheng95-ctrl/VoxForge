import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np

from src.stage0.router_utils import write_csv, write_json
from src.stage0.anatomy_expert import TotalSegLobeCache, anatomy_record, anatomy_to_bbox, parse_anatomy

try:
    import SimpleITK as sitk
    from lungmask import LMInferer
except Exception as exc:
    sitk = None
    LMInferer = None
    LUNGMASK_IMPORT_ERROR = repr(exc)
else:
    LUNGMASK_IMPORT_ERROR = None


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_predictions(path: Path):
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    return read_csv(path)


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


def bbox_from_mask(mask):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    h0, w0, d0 = coords.min(axis=0)
    h1, w1, d1 = coords.max(axis=0) + 1
    return [int(h0), int(h1), int(w0), int(w1), int(d0), int(d1)]


def bbox_volume(bbox):
    if bbox is None:
        return 0
    h0, h1, w0, w1, d0, d1 = bbox
    return max(0, h1 - h0) * max(0, w1 - w0) * max(0, d1 - d0)


def margin_mm_to_vox(margin_mm, zooms):
    return [int(math.ceil(float(m) / float(z))) for m, z in zip(margin_mm, zooms)]


def expand_bbox(bbox, shape, margin_vox):
    if bbox is None:
        return None
    h0, h1, w0, w1, d0, d1 = bbox
    mh, mw, md = margin_vox
    H, W, D = shape
    return [
        max(0, h0 - mh),
        min(H, h1 + mh),
        max(0, w0 - mw),
        min(W, w1 + mw),
        max(0, d0 - md),
        min(D, d1 + md),
    ]


def union_bboxes(bboxes):
    valid = [b for b in bboxes if b is not None]
    if not valid:
        return None
    return [
        min(b[0] for b in valid),
        max(b[1] for b in valid),
        min(b[2] for b in valid),
        max(b[3] for b in valid),
        min(b[4] for b in valid),
        max(b[5] for b in valid),
    ]


def bbox_for_mode(case_preds, prompt_bboxes, mode):
    if mode == "case_union":
        case_bbox = union_bboxes(prompt_bboxes)
        return {p["id"]: case_bbox for p in case_preds}, [{"group_id": "case_union", "bbox": case_bbox, "n_findings": len(case_preds)}]
    if mode == "finding":
        groups = []
        mapping = {}
        for p, bbox in zip(case_preds, prompt_bboxes):
            group_id = p["id"]
            mapping[p["id"]] = bbox
            groups.append({"group_id": group_id, "bbox": bbox, "n_findings": 1})
        return mapping, groups
    if mode == "policy_group":
        by_policy = defaultdict(list)
        for p, bbox in zip(case_preds, prompt_bboxes):
            by_policy[p["final_policy"]].append((p, bbox))
        mapping = {}
        groups = []
        for policy, items in sorted(by_policy.items()):
            group_bbox = union_bboxes([bbox for _, bbox in items])
            group_id = f"policy:{policy}"
            for p, _ in items:
                mapping[p["id"]] = group_bbox
            groups.append({"group_id": group_id, "bbox": group_bbox, "n_findings": len(items)})
        return mapping, groups
    raise ValueError(f"unknown crop mode: {mode}")


def bbox_for_anatomy_groups(case_preds, prompt_bboxes, mode):
    if mode in {"case_union", "finding"}:
        return bbox_for_mode(case_preds, prompt_bboxes, mode)
    by_group = defaultdict(list)
    for p, bbox in zip(case_preds, prompt_bboxes):
        by_group[p["anatomy_group"]].append((p, bbox))
    mapping = {}
    groups = []
    for group_id, items in sorted(by_group.items()):
        group_bbox = union_bboxes([bbox for _, bbox in items])
        for p, _ in items:
            mapping[p["id"]] = group_bbox
        groups.append({"group_id": group_id, "bbox": group_bbox, "n_findings": len(items)})
    return mapping, groups


def load_sitk_image(image_path):
    try:
        return sitk.ReadImage(str(image_path))
    except Exception:
        nii = nib.load(str(image_path))
        arr_hwd = np.asanyarray(nii.dataobj).astype(np.float32)
        arr_dwh = np.transpose(arr_hwd, (2, 1, 0))
        img = sitk.GetImageFromArray(arr_dwh)
        zooms = nii.header.get_zooms()[:3]
        img.SetSpacing((float(zooms[0]), float(zooms[1]), float(zooms[2])))
        return img


def mask_center(mask):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return coords.mean(axis=0)


def fixed_lungmask_to_hwd(seg, target_shape):
    arr = np.transpose(seg, (2, 1, 0))
    if arr.shape != target_shape:
        return None
    return arr


def lungmask_to_hwd(seg, target_shape, affine):
    candidates = [
        ("identity", seg),
        ("d_hw_to_hwd", np.transpose(seg, (1, 2, 0))),
        ("d_hw_to_whd", np.transpose(seg, (2, 1, 0))),
        ("d_hw_to_dwh", np.transpose(seg, (0, 2, 1))),
    ]
    valid = []
    for name, arr in candidates:
        if arr.shape != target_shape:
            continue
        center_1 = mask_center(arr == 1)
        center_2 = mask_center(arr == 2)
        if center_1 is None or center_2 is None:
            continue
        world_1 = nib.affines.apply_affine(affine, center_1)
        world_2 = nib.affines.apply_affine(affine, center_2)
        # In nibabel's RAS world coordinates, left/right separation is the x
        # axis. This resolves the common square-slice ambiguity where both
        # HWD and WHD transposes have the same shape.
        score = abs(float(world_1[0] - world_2[0]))
        valid.append((score, name, arr))
    if not valid:
        return None, None
    valid.sort(key=lambda item: item[0], reverse=True)
    _, name, arr = valid[0]
    return arr, name


def lungmask_label_info(seg_hwd, affine):
    label_1_bbox = bbox_from_mask(seg_hwd == 1)
    label_2_bbox = bbox_from_mask(seg_hwd == 2)
    label_1_center = mask_center(seg_hwd == 1)
    label_2_center = mask_center(seg_hwd == 2)
    if label_1_center is None or label_2_center is None:
        raise RuntimeError("lungmask must contain both label 1 and label 2")
    label_1_world = nib.affines.apply_affine(affine, label_1_center)
    label_2_world = nib.affines.apply_affine(affine, label_2_center)
    if label_1_world[0] >= label_2_world[0]:
        right_bbox, left_bbox = label_1_bbox, label_2_bbox
        right_label, left_label = 1, 2
    else:
        right_bbox, left_bbox = label_2_bbox, label_1_bbox
        right_label, left_label = 2, 1
    return {
        "both": bbox_from_mask(seg_hwd > 0),
        "right": right_bbox,
        "left": left_bbox,
        "right_label": right_label,
        "left_label": left_label,
        "label_1_world_x": float(label_1_world[0]),
        "label_2_world_x": float(label_2_world[0]),
    }


class LungCache:
    def __init__(self):
        if LMInferer is None:
            raise RuntimeError(f"lungmask unavailable: {LUNGMASK_IMPORT_ERROR}")
        self._inferer = None
        self.cache = {}

    @property
    def inferer(self):
        if self._inferer is None:
            modelpath = os.environ.get("LUNGMASK_MODEL_PATH")
            self._inferer = LMInferer(modelname="R231", modelpath=modelpath)
        return self._inferer

    def get(self, image_path: Path, shape, affine, mapping_mode="fixed"):
        key = str(image_path)
        if key in self.cache:
            return self.cache[key]
        sitk_img = load_sitk_image(image_path)
        seg = self.inferer.apply(sitk_img)
        auto_seg_hwd, auto_transform_name = lungmask_to_hwd(seg, shape, affine)
        if mapping_mode == "auto":
            seg_hwd = auto_seg_hwd
            transform_name = auto_transform_name
        else:
            seg_hwd = fixed_lungmask_to_hwd(seg, shape)
            transform_name = "fixed_d_hw_to_whd"
        if seg_hwd is None:
            raise RuntimeError(f"lungmask shape mismatch: seg={seg.shape}, target={shape}")
        info = lungmask_label_info(seg_hwd, affine)
        info["transform"] = transform_name
        info["auto_transform"] = auto_transform_name
        info["mapping_mode"] = mapping_mode
        info["mapping_warning"] = None
        if mapping_mode == "fixed" and auto_transform_name != "d_hw_to_whd":
            info["mapping_warning"] = f"auto selected {auto_transform_name}, fixed uses d_hw_to_whd"
        self.cache[key] = info
        return info


def policy_to_bbox(policy, lung_info, shape, zooms, margins):
    if policy == "both_conservative":
        return expand_bbox(lung_info["both"], shape, margin_mm_to_vox(margins["conservative"], zooms))
    if policy == "left_moderate":
        return expand_bbox(lung_info["left"], shape, margin_mm_to_vox(margins["moderate"], zooms))
    if policy == "right_moderate":
        return expand_bbox(lung_info["right"], shape, margin_mm_to_vox(margins["moderate"], zooms))
    if policy == "left_aggressive":
        return expand_bbox(lung_info["left"], shape, margin_mm_to_vox(margins["aggressive"], zooms))
    if policy == "right_aggressive":
        return expand_bbox(lung_info["right"], shape, margin_mm_to_vox(margins["aggressive"], zooms))
    return expand_bbox(lung_info["both"], shape, margin_mm_to_vox(margins["conservative"], zooms))


def label_path_for(record, data_root: Path):
    label = str(record["label"]).replace("\\", "/")
    if Path(label).is_absolute():
        return Path(label)
    direct = data_root / label
    if direct.exists():
        return direct
    cleaned = label.replace("labels_finding/", "")
    alt = data_root / "labels" / cleaned
    if alt.exists():
        return alt
    alt = data_root / "labels_finding" / cleaned
    if alt.exists():
        return alt
    raw_alt = data_root.parent / "raw" / label
    if raw_alt.exists():
        return raw_alt
    return direct


def image_path_for(record, data_root: Path):
    image = Path(str(record["image"]))
    if image.is_absolute() and image.exists():
        return image
    if not image.is_absolute():
        direct = data_root / image
        if direct.exists():
            return direct
    case_name = record.get("case_name") or image.name
    candidates = [
        data_root / "images" / case_name,
        data_root / "images_flat" / case_name,
        data_root.parent / "raw" / "images_flat" / case_name,
        data_root.parent / "ct_cache" / "images_flat" / case_name,
        data_root.parent / "raw" / "images_2d" / case_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return image if image.is_absolute() else data_root / image


def summarize(values):
    arr = np.array(values, dtype=float)
    return {
        "n": int(len(arr)),
        "mean": float(arr.mean()) if len(arr) else None,
        "min": float(arr.min()) if len(arr) else None,
        "p05": float(np.quantile(arr, 0.05)) if len(arr) else None,
        "median": float(np.median(arr)) if len(arr) else None,
        "full_rate_ge_0_999": float(np.mean(arr >= 0.999)) if len(arr) else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="")
    parser.add_argument("--predictions", default="")
    parser.add_argument("--out-dir", default="artifacts/reports/stage0_policy_recall")
    parser.add_argument("--limit-cases", type=int)
    parser.add_argument("--conservative-margin", default="60,60,50")
    parser.add_argument("--moderate-margin", default="40,40,30")
    parser.add_argument("--aggressive-margin", default="20,20,20")
    parser.add_argument("--mapping-mode", choices=["fixed", "auto"], default="fixed")
    parser.add_argument("--crop-mode", choices=["case_union", "policy_group", "finding"], default="case_union")
    parser.add_argument("--spatial-mode", choices=["lungmask", "anatomy"], default="lungmask")
    parser.add_argument("--totalseg-device", default="gpu")
    parser.add_argument("--totalseg-fast", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    manifest_path = resolve_path(args.manifest)
    pred_path = resolve_path(args.predictions)
    out_dir = resolve_output_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def parse_margin(text):
        vals = [float(x) for x in text.split(",")]
        if len(vals) != 3:
            raise ValueError("margin must be H,W,D")
        return vals

    margins = {
        "conservative": parse_margin(args.conservative_margin),
        "moderate": parse_margin(args.moderate_margin),
        "aggressive": parse_margin(args.aggressive_margin),
    }

    manifest = {r["id"]: r for r in read_jsonl(manifest_path)}
    preds = read_predictions(pred_path)
    preds = [p for p in preds if p["id"] in manifest]
    if args.limit_cases is not None:
        selected_cases = []
        seen = set()
        for p in preds:
            case = manifest[p["id"]]["case_name"]
            if case not in seen:
                if len(seen) >= args.limit_cases:
                    continue
                seen.add(case)
                selected_cases.append(case)
        selected_cases = set(selected_cases)
        preds = [p for p in preds if manifest[p["id"]]["case_name"] in selected_cases]

    by_case = defaultdict(list)
    for p in preds:
        by_case[manifest[p["id"]]["case_name"]].append(p)

    data_root = manifest_path.parent
    lung_cache = LungCache()
    totalseg_cache = TotalSegLobeCache(fast=args.totalseg_fast, device=args.totalseg_device) if args.spatial_mode == "anatomy" else None
    rows = []
    case_rows = []
    group_rows = []

    for idx, (case_name, case_preds) in enumerate(by_case.items(), start=1):
        first_record = manifest[case_preds[0]["id"]]
        image_path = image_path_for(first_record, data_root)
        nii = nib.load(str(image_path))
        shape = nii.shape
        zooms = nii.header.get_zooms()[:3]
        lung_info = None
        seg = totalseg_cache.get_seg(image_path, shape) if totalseg_cache is not None else None

        per_prompt_bboxes = []
        policies = []
        for p in case_preds:
            policy = p["final_policy"]
            policies.append(policy)
            fallback_bbox = None
            if args.spatial_mode == "lungmask":
                if lung_info is None:
                    lung_info = lung_cache.get(image_path, shape, nii.affine, args.mapping_mode)
                fallback_bbox = policy_to_bbox(policy, lung_info, shape, zooms, margins)
            p.update(anatomy_record(p.get("prompt"), p.get("laterality"), p.get("final_tightness")))
            if args.spatial_mode == "anatomy":
                decision = parse_anatomy(p.get("prompt"), p.get("laterality"))
                bbox = anatomy_to_bbox(decision, seg, shape, zooms, margins[p["final_tightness"]])
                if bbox is None:
                    if lung_info is None:
                        lung_info = lung_cache.get(image_path, shape, nii.affine, args.mapping_mode)
                    fallback_bbox = policy_to_bbox(policy, lung_info, shape, zooms, margins)
                    p["anatomy_fallback_reason"] = "empty_totalseg_lobe_mask"
                    bbox = fallback_bbox
                else:
                    p["anatomy_fallback_reason"] = None
            else:
                bbox = fallback_bbox
            per_prompt_bboxes.append(bbox)
        if args.spatial_mode == "anatomy":
            finding_bboxes, groups = bbox_for_anatomy_groups(case_preds, per_prompt_bboxes, args.crop_mode)
        else:
            finding_bboxes, groups = bbox_for_mode(case_preds, per_prompt_bboxes, args.crop_mode)
        for g in groups:
            if g["bbox"] is None:
                g["bbox"] = [0, shape[0], 0, shape[1], 0, shape[2]]
            g["volume_ratio"] = bbox_volume(g["bbox"]) / max(1, int(np.prod(shape)))
            group_rows.append({
                "case_name": case_name,
                "crop_mode": args.crop_mode,
                "spatial_mode": args.spatial_mode,
                "group_id": g["group_id"],
                "n_findings": g["n_findings"],
                "group_volume_ratio": g["volume_ratio"],
                "group_bbox_hwd": json.dumps(g["bbox"]),
            })

        effective_case_bbox = union_bboxes([g["bbox"] for g in groups])
        if effective_case_bbox is None:
            effective_case_bbox = [0, shape[0], 0, shape[1], 0, shape[2]]
        effective_case_ratio = bbox_volume(effective_case_bbox) / max(1, int(np.prod(shape)))
        mean_group_ratio = float(np.mean([g["volume_ratio"] for g in groups])) if groups else None
        print(f"[{idx}/{len(by_case)}] {case_name} findings={len(case_preds)} policies={sorted(set(policies))} mode={args.crop_mode} mean_group_ratio={mean_group_ratio:.3f}", flush=True)

        case_recalls = []
        for p in case_preds:
            record = manifest[p["id"]]
            label_path = label_path_for(record, data_root)
            label_nii = nib.load(str(label_path))
            label = np.asanyarray(label_nii.dataobj) > 0
            total = int(label.sum())
            finding_bbox = finding_bboxes[p["id"]]
            if finding_bbox is None:
                finding_bbox = [0, shape[0], 0, shape[1], 0, shape[2]]
            h0, h1, w0, w1, d0, d1 = finding_bbox
            inside = int(label[h0:h1, w0:w1, d0:d1].sum())
            recall = 1.0 if total == 0 else inside / total
            case_recalls.append(recall)
            volume_ratio = bbox_volume(finding_bbox) / max(1, int(np.prod(shape)))
            rows.append({
                "id": p["id"],
                "case_name": case_name,
                "crop_mode": args.crop_mode,
                "spatial_mode": args.spatial_mode,
                "true_category": p.get("true_category"),
                "raw_pred_category": p.get("raw_pred_category"),
                "final_policy": p.get("final_policy"),
                "final_tightness": p.get("final_tightness"),
                "laterality": p.get("laterality"),
                "anatomy_target": p.get("anatomy_target"),
                "anatomy_lobes": "|".join(p.get("anatomy_lobes") or []),
                "anatomy_reason": p.get("anatomy_reason"),
                "anatomy_fallback_reason": p.get("anatomy_fallback_reason"),
                "gt_recall": recall,
                "gt_voxels": total,
                "inside_voxels": inside,
                "case_volume_ratio": volume_ratio,
                "case_bbox_hwd": json.dumps(finding_bbox),
                "prompt": p.get("prompt"),
            })

        case_rows.append({
            "case_name": case_name,
            "spatial_mode": args.spatial_mode,
            "n_findings": len(case_preds),
            "policies": "|".join(sorted(set(policies))),
            "case_volume_ratio": effective_case_ratio,
            "mean_group_volume_ratio": mean_group_ratio,
            "n_crop_groups": len(groups),
            "case_recall_min": min(case_recalls) if case_recalls else None,
            "case_recall_mean": float(np.mean(case_recalls)) if case_recalls else None,
            "case_bbox_hwd": json.dumps(effective_case_bbox),
            "mapping_mode": lung_info.get("mapping_mode") if lung_info else None,
            "lungmask_transform": lung_info.get("transform") if lung_info else None,
            "auto_transform": lung_info.get("auto_transform") if lung_info else None,
            "right_label": lung_info.get("right_label") if lung_info else None,
            "left_label": lung_info.get("left_label") if lung_info else None,
            "label_1_world_x": lung_info.get("label_1_world_x") if lung_info else None,
            "label_2_world_x": lung_info.get("label_2_world_x") if lung_info else None,
            "mapping_warning": lung_info.get("mapping_warning") if lung_info else None,
        })

    recalls = [r["gt_recall"] for r in rows]
    ratios = [r["case_volume_ratio"] for r in rows]
    summary = {
        "predictions": str(pred_path),
        "manifest": str(manifest_path),
        "margins_mm": margins,
        "mapping_mode": args.mapping_mode,
        "crop_mode": args.crop_mode,
        "spatial_mode": args.spatial_mode,
        "n_findings": len(rows),
        "n_cases": len(case_rows),
        "n_crop_groups": len(group_rows),
        "gt_recall": summarize(recalls),
        "case_volume_ratio_weighted_by_findings": summarize(ratios),
        "crop_group_volume_ratio": summarize([r["group_volume_ratio"] for r in group_rows]),
        "policy_counts": dict(sorted({p: sum(1 for r in rows if r["final_policy"] == p) for p in {r["final_policy"] for r in rows}}.items())),
        "worst_findings": sorted(rows, key=lambda r: r["gt_recall"])[:20],
    }

    stem = pred_path.stem.replace("_predictions", "")
    write_csv(rows, out_dir / f"{stem}_finding_recall.csv")
    write_csv(case_rows, out_dir / f"{stem}_case_recall.csv")
    write_csv(group_rows, out_dir / f"{stem}_crop_group_recall.csv")
    write_json(summary, out_dir / f"{stem}_summary.json")
    print(json.dumps(summary["gt_recall"], ensure_ascii=False, indent=2))
    print(json.dumps(summary["case_volume_ratio_weighted_by_findings"], ensure_ascii=False, indent=2))
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
