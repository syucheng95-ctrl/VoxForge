"""Build S1 no-safety verified_proposals: XGB selector, no safety fallback.

Key differences from apply_fallback_selector.py:
  - hu -> ONLY source_expert == "hu" (no diffuse mixed in)
  - If predicted branch has no proposals: verified = [], NO fallback
  - Prints target distribution for verification

Input:
  outputs/stage1/verified_proposals.jsonl (raw VP from Stage1)
  models/fallback_selector_xgb_4feat.pkl

Output:
  outputs/stage1/verified_proposals_s1_nosafety.jsonl
"""

import argparse
import csv
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_features(analysis_dir):
    """Load the 4 features from pre-computed CSVs.
    Reuses the exact same logic as apply_fallback_selector.py."""
    features = {}
    csvs = [
        analysis_dir / "prompt_signals.csv",
        analysis_dir / "ct_signals.csv",
        analysis_dir / "proposal_signals.csv",
    ]
    for csv_path in csvs:
        with open(csv_path, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                fid = r["finding_id"]
                if fid not in features:
                    features[fid] = {}
                for k, v in r.items():
                    if k not in ("finding_id", "category", "case_name"):
                        try:
                            features[fid][k] = float(v)
                        except (ValueError, TypeError):
                            features[fid][k] = 0.0
    return features


def load_model_and_encoder(model_path):
    """Flexible model loading. Handles three known formats:
    1. {model, features, label_encoder, ...}
    2. {model, features, classes, ...}
    3. bare model with model.classes_
    """
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    if isinstance(bundle, dict):
        model = bundle["model"]
        feature_names = bundle.get("features") or bundle.get("feature_names")

        if "label_encoder" in bundle:
            le = bundle["label_encoder"]
            return model, feature_names, le, bundle

        if "classes" in bundle:
            # Adapter: wrap list-of-strings as a simple indexer
            class_list = bundle["classes"]
            class DummyLE:
                def inverse_transform(self, indices):
                    return [class_list[i] for i in indices]
            return model, feature_names, DummyLE(), bundle

    # Bare model
    if hasattr(bundle, "classes_"):
        class_list = bundle.classes_
        class DummyLE:
            def inverse_transform(self, indices):
                return [class_list[i] for i in indices]
        return bundle, None, DummyLE(), {}

    raise RuntimeError(
        f"Cannot determine model format. Bundle type: {type(bundle)}. "
        f"If dict, keys: {list(bundle.keys()) if isinstance(bundle, dict) else 'N/A'}"
    )


def main():
    ap = argparse.ArgumentParser(description="Build S1 no-safety VP")
    ap.add_argument("--config", default="configs/pipeline.yaml", help="Config YAML (for output paths)")
    ap.add_argument("--input", default=None, help="Raw VP path (overrides config)")
    ap.add_argument("--output", default=None, help="Output VP path (overrides config)")
    ap.add_argument("--model", default=None, help="Selector model path (overrides config)")
    ap.add_argument("--analysis-dir", default=None, help="Analysis CSV directory (overrides config)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # ── Paths ──
    from src.utils import load_config, resolve
    config = load_config(args.config)

    if args.input:
        input_path = Path(args.input)
        stage1_dir = input_path.parent
    else:
        stage1_dir = Path(resolve(config, "outputs.stage1"))
        input_path = stage1_dir / "verified_proposals.jsonl"

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = stage1_dir / "verified_proposals_s1_nosafety.jsonl"

    model_path = Path(args.model) if args.model else Path(resolve(config, "models.fallback_selector"))
    if not model_path.exists():
        alt = Path(config.get("models", {}).get("fallback_selector_rf", "")) if not args.model else model_path.parent / "fallback_selector_rf_4feat.pkl"
        if alt.exists():
            model_path = alt
        else:
            print(f"[ERROR] Model not found: {model_path}")
            sys.exit(1)

    # ── Load model ──
    model, feature_names, le, bundle = load_model_and_encoder(model_path)
    if feature_names is None:
        feature_names = getattr(model, "feature_names_in_", [])
    print(f"[Loaded] {model_path.name}: {type(model).__name__}, "
          f"{len(feature_names)} features: {feature_names}, "
          f"classes: {le.classes_.tolist() if hasattr(le, 'classes_') else '?'}, "
          f"dice={bundle.get('dice', '?')}")

    # ── Load features (reuse apply_fallback_selector.py logic) ──
    try:
        analysis_default = resolve(config, "outputs.analysis")
    except (KeyError, TypeError):
        analysis_default = str(Path(config["_upload_dir"]) / "outputs" / "analysis")
    analysis_dir = Path(args.analysis_dir) if args.analysis_dir else Path(analysis_default)
    all_features = load_features(analysis_dir)
    print(f"[Features] Loaded {len(all_features)} findings from analysis CSVs")

    # ── Load raw VP ──
    vp_rows = read_jsonl(input_path)
    print(f"[Input] {len(vp_rows)} findings from {input_path}")

    # ── Build S1 no-safety VP ──
    filtered_rows = []
    expert_wins = Counter()
    empty_count = 0

    for vp in vp_rows:
        fid = vp["finding_id"]

        # Predict using pre-computed features (exact same column order)
        feat_dict = all_features.get(fid, {})
        X = np.array([[feat_dict.get(fn, 0.0) for fn in feature_names]], dtype=np.float32)
        pred_idx = model.predict(X)[0]
        pred_exp = le.inverse_transform([pred_idx])[0]

        expert_wins[pred_exp] += 1

        # Filter proposals: keep ONLY the predicted expert
        if pred_exp == "full_ct":
            keep_experts = {"full_ct_voxtell"}
        elif pred_exp == "hu":
            keep_experts = {"hu"}  # ONLY hu, no diffuse
        elif pred_exp == "nodule_detector":
            keep_experts = {"nodule_detector"}
        elif pred_exp == "diffuse":
            keep_experts = {"diffuse"}
        else:
            keep_experts = {pred_exp}

        filtered_proposals = [e for e in vp.get("verified", [])
                              if e.get("source_expert") in keep_experts]
        filtered_rejected = [e for e in vp.get("rejected", [])
                             if e.get("source_expert") in keep_experts]

        # NO safety fallback — if empty, stay empty
        is_empty = len(filtered_proposals) == 0
        if is_empty:
            empty_count += 1

        use_fullct = pred_exp == "full_ct" and len(filtered_proposals) > 0
        has_fullct = any(e.get("source_expert") == "full_ct_voxtell"
                          for e in vp.get("verified", []))

        new_vp = dict(vp)
        new_vp["verified"] = filtered_proposals
        new_vp["rejected"] = filtered_rejected
        new_vp["n_verified"] = len(filtered_proposals)
        new_vp["n_rejected"] = len(filtered_rejected)
        new_vp["fallback_selector_xgb_prediction"] = pred_exp
        new_vp["fallback_selector_actual"] = pred_exp
        new_vp["fallback_selector_no_safety_empty"] = is_empty
        new_vp["use_full_ct_voxtell"] = use_fullct
        new_vp["has_full_ct_voxtell"] = has_fullct

        filtered_rows.append(new_vp)

    # ── Summary ──
    total_verified_before = sum(len(vp["verified"]) for vp in vp_rows)
    total_verified_after = sum(len(vp["verified"]) for vp in filtered_rows)

    print(f"\n[Summary]")
    print(f"  Findings:             {len(filtered_rows)}")
    print(f"  Expert selections:    {dict(expert_wins)}")
    print(f"  Empty (no fallback):  {empty_count}")
    print(f"  Verified before:      {total_verified_before}")
    print(f"  Verified after:       {total_verified_after}")
    print(f"  Reduction:            {total_verified_before - total_verified_after} proposals removed")

    # Expected distribution for verification
    print(f"\n[Target distribution check]")
    for exp_name in ["full_ct", "nodule_detector", "hu"]:
        count = expert_wins.get(exp_name, 0)
        print(f"  {exp_name:<20} {count:>3}")

    if not args.dry_run:
        write_jsonl(output_path, filtered_rows)
        print(f"\n[Saved] {output_path}")
    else:
        print(f"\n[Dry run] No files modified.")


if __name__ == "__main__":
    main()
