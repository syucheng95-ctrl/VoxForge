"""Build S2 C-method verified_proposals: S1 no-safety + empty filled with hu/diffuse.

Key properties:
  - Starts from S1 no-safety VP
  - If verified non-empty: keep as-is
  - If verified empty: fill from raw VP using source_expert in {"hu", "diffuse"}
  - NEVER uses full_ct or nodule_detector to fill empty

Input:
  outputs/stage1/verified_proposals.jsonl (raw VP)
  outputs/stage1/verified_proposals_s1_nosafety.jsonl (S1 no-safety VP)

Output:
  outputs/stage1/verified_proposals_s2_c.jsonl
"""

import argparse
import json
from collections import Counter
from pathlib import Path


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


def main():
    ap = argparse.ArgumentParser(description="Build S2 C-method VP")
    ap.add_argument("--config", default="configs/pipeline.yaml", help="Config YAML (for output paths)")
    ap.add_argument("--raw-vp", default=None, help="Raw VP path (overrides config)")
    ap.add_argument("--s1-vp", default=None, help="S1 no-safety VP path (overrides config)")
    ap.add_argument("--output", default=None, help="Output VP path (overrides config)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # ── Paths ──
    if args.raw_vp and args.s1_vp:
        raw_path = Path(args.raw_vp)
        s1_path = Path(args.s1_vp)
        stage1_dir = raw_path.parent
    else:
        from src.utils import load_config, resolve
        config = load_config(args.config)
        stage1_dir = Path(resolve(config, "outputs.stage1"))
        raw_path = stage1_dir / "verified_proposals.jsonl"
        s1_path = stage1_dir / "verified_proposals_s1_nosafety.jsonl"

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = stage1_dir / "verified_proposals_s2_c.jsonl"

    # ── Load ──
    raw_rows = read_jsonl(raw_path)
    raw_by_id = {r["finding_id"]: r for r in raw_rows}
    s1_rows = read_jsonl(s1_path)

    if not s1_rows:
        print("[ERROR] S1 no-safety VP is empty — run build_s1_nosafety_vp.py first")
        return

    print(f"[Input] Raw: {len(raw_rows)} findings, S1 no-safety: {len(s1_rows)} findings")

    # ── Build S2 C-method ──
    output_rows = []
    filled_empty = 0
    stayed_empty = 0
    fill_sources = Counter()

    for s1_row in s1_rows:
        fid = s1_row["finding_id"]
        s1_verified = s1_row.get("verified", [])
        raw_row = raw_by_id.get(fid, {})

        if len(s1_verified) > 0:
            # S1 no-safety has proposals — keep as-is
            new_row = dict(s1_row)
            new_row["s2_c_filled_empty"] = False
            new_row["s2_c_fill_sources"] = []
            output_rows.append(new_row)
        else:
            # S1 no-safety is empty — fill from raw VP with hu/diffuse
            raw_verified = raw_row.get("verified", []) if raw_row else []
            raw_rejected = raw_row.get("rejected", []) if raw_row else []

            fill_verified = [e for e in raw_verified
                             if e.get("source_expert") in ("hu", "diffuse")]
            fill_rejected = [e for e in raw_rejected
                             if e.get("source_expert") in ("hu", "diffuse")]

            if fill_verified:
                filled_empty += 1
                for e in fill_verified:
                    fill_sources[e.get("source_expert", "?")] += 1

                # Build new row: merge fill into verified
                new_row = dict(raw_row)
                new_row["verified"] = fill_verified
                new_row["rejected"] = [e for e in raw_verified + raw_rejected
                                       if e not in fill_verified]
                new_row["n_verified"] = len(fill_verified)
                new_row["n_rejected"] = len(new_row["rejected"])
                new_row["s2_c_filled_empty"] = True
                new_row["s2_c_fill_sources"] = list(set(
                    e.get("source_expert") for e in fill_verified
                ))
                # Preserve S1 selector fields
                for key in ("fallback_selector_xgb_prediction",
                            "fallback_selector_actual",
                            "fallback_selector_no_safety_empty",
                            "use_full_ct_voxtell",
                            "has_full_ct_voxtell"):
                    if key in s1_row:
                        new_row[key] = s1_row[key]
                output_rows.append(new_row)
            else:
                # No hu/diffuse in raw either — stay empty
                stayed_empty += 1
                new_row = dict(s1_row)
                new_row["s2_c_filled_empty"] = True
                new_row["s2_c_fill_sources"] = []
                output_rows.append(new_row)

    # ── Summary ──
    empty_from_s1 = sum(1 for r in s1_rows if r.get("fallback_selector_no_safety_empty"))
    total_verified = sum(len(r["verified"]) for r in output_rows)
    print(f"\n[Summary]")
    print(f"  Total findings:         {len(output_rows)}")
    print(f"  S1 empty:               {empty_from_s1}")
    print(f"  Filled from hu/diffuse: {filled_empty}")
    print(f"  Stayed empty:           {stayed_empty}")
    print(f"  Total S2 verified:      {total_verified}")
    print(f"  Fill sources:           {dict(fill_sources)}")

    if not args.dry_run:
        write_jsonl(output_path, output_rows)
        print(f"\n[Saved] {output_path}")
    else:
        print(f"\n[Dry run] No files modified.")


if __name__ == "__main__":
    main()
