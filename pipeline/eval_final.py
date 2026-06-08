"""Final evaluation: compute all micro-averaged Dice metrics from gate decisions.

Reads per-finding S1/S2 metrics from eval outputs, applies gate decisions,
and produces final_metrics.json + gate_decisions.csv.  Gate inference uses
only prediction-derived features (no GT); this eval step is separate.

Usage:
  python pipeline/eval_final.py --config configs/pipeline.yaml

Inputs (via config):
  outputs/final/gate_decisions_raw.json   — per-finding use_s2 decisions from apply_gate.py
  outputs/stage2/eval_finding_level.json  — S2 raw per-finding TP/pred/dice
  outputs/stage1/metrics_vs_baseline.json — S1 per-finding TP/pred/dice

Outputs:
  outputs/final/final_metrics.json   — micro Dice + comparison baselines
  outputs/final/gate_decisions.csv   — per-finding decisions with metrics
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from src.utils import load_config, resolve


def micro_dice(tp, pred, gt):
    return (2 * tp) / max(1, pred + gt)


def main():
    parser = argparse.ArgumentParser(description="Final eval: compute Dice metrics")
    parser.add_argument("--config", default="configs/pipeline.yaml", help="Config YAML path")
    args = parser.parse_args()

    config = load_config(args.config)

    decisions_path = Path(resolve(config, "outputs.final")) / "gate_decisions_raw.json"
    s2_eval_path = Path(resolve(config, "outputs.stage2")) / "eval_finding_level.json"
    s1_metrics_path = Path(resolve(config, "outputs.stage1")) / "metrics_vs_baseline.json"

    for p, name in [(decisions_path, "gate decisions"),
                    (s2_eval_path, "S2 eval"),
                    (s1_metrics_path, "S1 metrics")]:
        if not p.exists():
            print(f"[ERROR] Missing {name}: {p}")
            sys.exit(1)

    # Load gate decisions
    with open(decisions_path, encoding="utf-8") as f:
        decisions = json.load(f)
    use_s2_map = {d["finding_id"]: d["use_s2"] for d in decisions["decisions"]}
    p_s2_map = {d["finding_id"]: d.get("p_use_s2", 0.5) for d in decisions["decisions"]}
    threshold = decisions["threshold"]
    n_total = len(decisions["decisions"])

    # Load S2 per-finding metrics
    with open(s2_eval_path, encoding="utf-8") as f:
        s2_eval = json.load(f)
    s2_per_finding = {r["finding_id"]: r for r in s2_eval["per_finding"]}

    # Load S1 per-finding metrics
    with open(s1_metrics_path, encoding="utf-8") as f:
        s1_metrics = json.load(f)
    s1_pf_data = s1_metrics.get("per_finding_pipeline", s1_metrics.get("per_finding", []))
    s1_per_finding = {r["finding_id"]: r for r in s1_pf_data}

    # Gather per-finding data
    tp_s1, pred_s1, tp_s2, pred_s2, gt_vox = [], [], [], [], []
    finding_ids = []
    categories = []

    for fid in sorted(use_s2_map.keys()):
        s1 = s1_per_finding.get(fid, {})
        s2 = s2_per_finding.get(fid, {})
        gt = s2.get("gt_voxels", s1.get("gt_voxels", 0))

        finding_ids.append(fid)
        categories.append(s2.get("category", s1.get("category", "")))
        tp_s1.append(s1.get("tp", 0))
        pred_s1.append(s1.get("pred_voxels", 0))
        tp_s2.append(s2.get("s2_raw_tp", 0))
        pred_s2.append(s2.get("s2_raw_pred_voxels", 0))
        gt_vox.append(gt)

    tp_s1 = np.array(tp_s1, dtype=np.float64)
    pred_s1 = np.array(pred_s1, dtype=np.float64)
    tp_s2 = np.array(tp_s2, dtype=np.float64)
    pred_s2 = np.array(pred_s2, dtype=np.float64)
    gt_vox = np.array(gt_vox, dtype=np.float64)
    use_s2 = np.array([use_s2_map.get(fid, 0) for fid in finding_ids], dtype=bool)

    # Compute micro Dice
    s1_tp_sum, s1_pred_sum = tp_s1.sum(), pred_s1.sum()
    s2_tp_sum, s2_pred_sum = tp_s2.sum(), pred_s2.sum()
    gt_sum = gt_vox.sum()

    gate_tp = np.where(use_s2, tp_s2, tp_s1).sum()
    gate_pred = np.where(use_s2, pred_s2, pred_s1).sum()

    always_s1_dice = micro_dice(s1_tp_sum, s1_pred_sum, gt_sum)
    always_s2_dice = micro_dice(s2_tp_sum, s2_pred_sum, gt_sum)
    gate_dice = micro_dice(gate_tp, gate_pred, gt_sum)

    n_use_s2 = int(use_s2.sum())

    print(f"Gate micro Dice: {gate_dice:.4f}  (n_use_s2={n_use_s2}, threshold={threshold})")
    print(f"  Always S1:     {always_s1_dice:.4f}")
    print(f"  Always S2_raw: {always_s2_dice:.4f}")

    # Write decisions CSV
    out_dir = Path(resolve(config, "outputs.final"))
    out_dir.mkdir(parents=True, exist_ok=True)
    decisions_csv = out_dir / "gate_decisions.csv"
    with open(decisions_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["finding_id", "category", "p_use_s2", "use_s2",
                          "s1_dice", "s2_raw_dice", "s1_pred_voxels", "s2_raw_pred_voxels"])
        for i, fid in enumerate(finding_ids):
            s1_d = micro_dice(tp_s1[i], pred_s1[i], gt_vox[i])
            s2_d = micro_dice(tp_s2[i], pred_s2[i], gt_vox[i])
            writer.writerow([
                fid, categories[i],
                round(float(p_s2_map.get(fid, 0.5)), 4),
                int(use_s2[i]),
                round(s1_d, 4), round(s2_d, 4),
                int(pred_s1[i]), int(pred_s2[i]),
            ])
    print(f"Decisions -> {decisions_csv}")

    # Write final metrics
    metrics_out = out_dir / "final_metrics.json"
    final_metrics = {
        "gate_micro_dice": round(gate_dice, 4),
        "always_s1_micro_dice": round(always_s1_dice, 4),
        "always_s2_raw_micro_dice": round(always_s2_dice, 4),
        "n_use_s2": n_use_s2,
        "n_total": n_total,
        "threshold": threshold,
        "tp": int(gate_tp),
        "pred_voxels": int(gate_pred),
        "gt_voxels": int(gt_sum),
    }
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2, ensure_ascii=False)
    print(f"Metrics -> {metrics_out}")


if __name__ == "__main__":
    main()
