"""Apply learned gate: load model, predict use_s2 from prediction-derived features.

Uses ONLY feature__ columns (no GT / target columns).  Final Dice evaluation
is a separate step in eval_final.py.  All config read from pipeline.yaml.

Usage:
  python pipeline/apply_gate.py --config configs/pipeline.yaml
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from src.gate.classifier import load_gate_model, predict_gate
from src.utils import load_config, resolve


def main():
    parser = argparse.ArgumentParser(description="Apply learned gate")
    parser.add_argument("--config", default="configs/pipeline.yaml", help="Config YAML path")
    args = parser.parse_args()

    config = load_config(args.config)

    model_path = Path(resolve(config, "gate.model"))
    metadata_path = Path(resolve(config, "gate.metadata"))
    table_path = Path(resolve(config, "outputs.root")) / "gate_feature_table.csv"
    threshold = config["gate"]["threshold"]

    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        sys.exit(1)
    if not metadata_path.exists():
        print(f"[ERROR] Metadata not found: {metadata_path}")
        sys.exit(1)
    if not table_path.exists():
        print(f"[ERROR] Feature table not found: {table_path}")
        sys.exit(1)

    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)
    kept = metadata["feature_cols"]

    print(f"Loading model from {model_path}...")
    preprocessor, model = load_gate_model(str(model_path))

    # Load feature columns only
    rows = []
    with open(table_path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    if not rows:
        print("[ERROR] Feature table is empty — check build_gate_table stage")
        sys.exit(1)

    missing = [c for c in kept if c not in rows[0]]
    if missing:
        print(f"[ERROR] Table is missing {len(missing)} model features, first: {missing[:5]}")
        sys.exit(1)
    print(f"Table: {len(rows)} rows, {len(kept)} model features")

    # Predict
    use_s2, p_use_s2 = predict_gate(model, preprocessor, rows, kept, threshold)

    n_use_s2 = int(np.sum(use_s2))
    print(f"Gate inference: {n_use_s2}/{len(rows)} use_s2 (threshold={threshold})")

    # Write decisions
    out_dir = Path(resolve(config, "outputs.final"))
    out_dir.mkdir(parents=True, exist_ok=True)
    decisions_out = out_dir / "gate_decisions_raw.json"
    decisions = {
        "threshold": threshold,
        "decisions": [
            {
                "finding_id": r["meta__finding_id"],
                "category": r["meta__category"],
                "p_use_s2": round(float(p_use_s2[i]), 4),
                "use_s2": int(use_s2[i]),
            }
            for i, r in enumerate(rows)
        ],
    }
    with open(decisions_out, "w", encoding="utf-8") as f:
        json.dump(decisions, f, indent=2, ensure_ascii=False)
    print(f"Decisions -> {decisions_out}")


if __name__ == "__main__":
    main()
