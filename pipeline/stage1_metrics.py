"""Evaluate current pipeline Stage1 masks against GT and VoxTell baseline.

For each finding, verified proposal coarse masks are unioned in full-CT HWD
space before computing micro-aggregated Dice/Recall/Precision.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np

from src.utils import load_config, load_jsonl, resolve


def add_counts(bucket: dict, tp: int, pred: int, gt: int) -> None:
    bucket["tp"] += int(tp)
    bucket["pred"] += int(pred)
    bucket["gt"] += int(gt)
    bucket["n"] += 1


def metrics(bucket: dict) -> dict:
    tp = bucket["tp"]
    pred = bucket["pred"]
    gt = bucket["gt"]
    return {
        "n": bucket["n"],
        "dice": (2 * tp) / max(1, pred + gt),
        "recall": tp / max(1, gt),
        "precision": tp / max(1, pred),
        "tp": tp,
        "pred_voxels": pred,
        "gt_voxels": gt,
    }


def reader_mask_to_raw_hwd(mask_reader: np.ndarray) -> np.ndarray:
    """Convert saved reader-space DWH mask crop back to raw HWD crop."""
    return mask_reader.transpose(2, 1, 0)[::-1, ::-1, :]


def paste_verified_mask(full_mask: np.ndarray, entry: dict, gate_bbox: list[int]) -> None:
    path = entry.get("coarse_mask_path")
    if not path:
        return

    mask_reader = np.asanyarray(nib.load(path).dataobj) > 0
    if mask_reader.sum() == 0:
        return

    gh0, gh1, gw0, gw1, gd0, gd1 = gate_bbox
    raw_h = gh1 - gh0
    raw_w = gw1 - gw0
    raw_d = gd1 - gd0

    ph0, ph1, pw0, pw1, pd0, pd1 = entry["proposal_bbox_hwd"]
    ch0 = max(0, ph0 - gh0)
    ch1 = min(raw_h, ph1 - gh0)
    cw0 = max(0, pw0 - gw0)
    cw1 = min(raw_w, pw1 - gw0)
    cd0 = max(0, pd0 - gd0)
    cd1 = min(raw_d, pd1 - gd0)
    if ch0 >= ch1 or cw0 >= cw1 or cd0 >= cd1:
        return

    mask_raw = reader_mask_to_raw_hwd(mask_reader)
    expected = (ch1 - ch0, cw1 - cw0, cd1 - cd0)
    if mask_raw.shape != expected:
        # Clip defensively if a bbox was clipped by the verifier.
        h = min(mask_raw.shape[0], expected[0])
        w = min(mask_raw.shape[1], expected[1])
        d = min(mask_raw.shape[2], expected[2])
        mask_raw = mask_raw[:h, :w, :d]
        ch1, cw1, cd1 = ch0 + h, cw0 + w, cd0 + d

    full_mask[gh0 + ch0:gh0 + ch1, gw0 + cw0:gw0 + cw1, gd0 + cd0:gd0 + cd1] |= mask_raw


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--verified-proposals", default="verified_proposals.jsonl",
                        help="VP filename inside stage1 dir (default: verified_proposals.jsonl)")
    args, _ = parser.parse_known_args()
    config = load_config(args.config)
    data_root = Path(resolve(config, "data.ct_images")).parent  # data/
    manifest = load_jsonl(resolve(config, "manifests.default"))
    manifest_by_id = {r["id"]: r for r in manifest}

    stage1_dir = Path(resolve(config, "outputs.stage1"))
    verified_rows = load_jsonl(stage1_dir / args.verified_proposals)

    pipeline_total = defaultdict(int)
    pipeline_by_cat: dict[str, dict] = defaultdict(lambda: defaultdict(int))
    per_finding = []

    for row in verified_rows:
        fid = row["finding_id"]
        mrow = manifest_by_id[fid]
        label_rel = mrow["label"].replace("\\", "/")
        gt_path = data_root / label_rel
        gt = np.asanyarray(nib.load(str(gt_path)).dataobj) > 0

        pred = np.zeros(gt.shape, dtype=bool)
        for entry in row["verified"]:
            if entry.get("source_expert") == "full_ct_voxtell":
                mask_path = entry.get("coarse_mask_path")
                if mask_path and Path(mask_path).exists():
                    mask_reader = np.asanyarray(nib.load(mask_path).dataobj) > 0
                    # full_ct mask is in VoxTell reader DWH space; convert to raw HWD
                    mask_raw = reader_mask_to_raw_hwd(mask_reader)
                    h = min(mask_raw.shape[0], pred.shape[0])
                    w = min(mask_raw.shape[1], pred.shape[1])
                    d = min(mask_raw.shape[2], pred.shape[2])
                    pred[:h, :w, :d] |= mask_raw[:h, :w, :d]
            else:
                paste_verified_mask(pred, entry, row["gate_bbox_hwd"])

        tp = int((pred & gt).sum())
        pred_vox = int(pred.sum())
        gt_vox = int(gt.sum())
        cat = mrow["category"]
        add_counts(pipeline_total, tp, pred_vox, gt_vox)
        add_counts(pipeline_by_cat[cat], tp, pred_vox, gt_vox)
        per_finding.append({
            "finding_id": fid,
            "category": cat,
            **metrics({"tp": tp, "pred": pred_vox, "gt": gt_vox, "n": 1}),
        })

    # S1 is now hybrid-routed:
    #   1a/1b/1c/2f -> FullCT VoxTell (th=0.3, via stage1_verify.py full_ct_voxtell branch)
    #   all others   → Pipeline S1 (Stage0 crop + Stage0.5 proposals + verifier coarse paste)
    # The "pipeline" metrics below already reflect this routing.
    FULLCT_ROUTED_CATEGORIES = {"1a", "1b", "1c", "2f"}
    HYBRID_COARSE_THRESHOLD = 0.3  # VoxTell probability threshold for binary coarse mask

    result = {
        "pipeline": {
            "overall": metrics(pipeline_total),
            "by_category": {cat: metrics(bucket) for cat, bucket in sorted(pipeline_by_cat.items())},
        },
        "hybrid_routing": {
            "fullct_categories": sorted(FULLCT_ROUTED_CATEGORIES),
            "pipeline_categories": sorted(set(pipeline_by_cat.keys()) - FULLCT_ROUTED_CATEGORIES),
            "coarse_threshold": HYBRID_COARSE_THRESHOLD,
            "note": "Pipeline S1 = FullCT VoxTell (th=0.3) for fullct_categories + Stage0/crop/proposal/verify for others",
        },
        "per_finding_pipeline": per_finding,
    }

    out_json = stage1_dir / "metrics_vs_baseline.json"
    out_csv = stage1_dir / "metrics_vs_baseline_by_category.csv"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    cats = sorted(pipeline_by_cat.keys())
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "category", "n",
            "S1_hybrid_dice", "S1_hybrid_recall", "S1_hybrid_precision",
        ])
        for cat in cats:
            pm = result["pipeline"]["by_category"].get(cat, metrics(defaultdict(int)))
            writer.writerow([
                cat, pm["n"],
                pm["dice"], pm["recall"], pm["precision"],
            ])

    print("S1 Hybrid Pipeline (FullCT th=0.3 for 1a/1b/1c/2f, Pipeline for others)")
    print(json.dumps(result["pipeline"]["overall"], indent=2))
    print("saved", out_json)
    print("saved", out_csv)


if __name__ == "__main__":
    main()
