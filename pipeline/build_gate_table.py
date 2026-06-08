"""Build per-finding gate feature table.

Joins data sources by finding_id, computes all inference-time features
(prediction-derived only, no GT), and outputs a CSV with two column families:
  meta__xxx    — identity/grouping (NOT for model)
  feature__xxx — inference-time features (model input)

Categorical features are stored as raw strings; one-hot encoding is done
via ColumnTransformer in the gate model bundle.  Final Dice evaluation is a separate
step in eval_final.py.
"""

import argparse
import csv
import json
import math
import sys
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from src.utils import load_config, resolve


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Keyword extraction ──
KEYWORD_PATTERNS = {
    "kw_bilateral": re.compile(r"both\s+lungs|bilateral", re.IGNORECASE),
    "kw_unilateral": re.compile(r"\b(left|right)\b", re.IGNORECASE),
    "kw_diffuse": re.compile(r"diffuse|throughout|extensive|scattered", re.IGNORECASE),
    "kw_focal": re.compile(r"focal|localized|solitary", re.IGNORECASE),
    "kw_multiple": re.compile(r"multiple|multifocal|several|numerous|many", re.IGNORECASE),
    "kw_small": re.compile(r"small|tiny|subcentimeter|micronodule|\d+\s*mm", re.IGNORECASE),
    "kw_mass": re.compile(r"\bmass\b|large|bulky", re.IGNORECASE),
    "kw_cavity": re.compile(r"cavity|cavitary", re.IGNORECASE),
    "kw_tree_in_bud": re.compile(r"tree.in.bud", re.IGNORECASE),
    "kw_honeycomb": re.compile(r"honeycomb|honeycombing", re.IGNORECASE),
    "kw_ggo": re.compile(r"ground.glass|GGO|opacit", re.IGNORECASE),
    "kw_consolidation": re.compile(r"consolidation|pneumonic", re.IGNORECASE),
    "kw_atelectasis": re.compile(r"atelectasis|collapse", re.IGNORECASE),
    "kw_nodule": re.compile(r"nodule|nodular", re.IGNORECASE),
    "kw_bronchial": re.compile(r"bronchial|peribronchial|bronchiectasis", re.IGNORECASE),
    "kw_pleural": re.compile(r"pleural|effusion|pneumothorax", re.IGNORECASE),
    "kw_subpleural": re.compile(r"subpleural|peripheral", re.IGNORECASE),
    "kw_lobe_specified": re.compile(r"\b(upper|lower|middle)\s*lobe\b|lobe\b|segment", re.IGNORECASE),
    "kw_size_mentioned": re.compile(r"\d+\s*(mm|cm)|diameter", re.IGNORECASE),
}


def extract_keywords(prompt: str) -> dict:
    feats = {}
    for name, pat in KEYWORD_PATTERNS.items():
        feats[f"feature__{name}"] = int(bool(pat.search(prompt)))
    feats["feature__prompt_len_chars"] = len(prompt)
    feats["feature__prompt_len_words"] = len(prompt.split())
    return feats


def safe_mean(vals: list[float]) -> float:
    return float(np.mean(vals)) if vals else 0.0


def safe_median(vals: list[float]) -> float:
    return float(np.median(vals)) if vals else 0.0


def safe_max(vals: list[float]) -> float:
    return float(max(vals)) if vals else 0.0


def safe_min(vals: list[float]) -> float:
    return float(min(vals)) if vals else 0.0


def safe_std(vals: list[float]) -> float:
    return float(np.std(vals)) if len(vals) > 1 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Build gate training table")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--manifest", default=None,
                        help="Manifest JSONL (default: config manifests.default)")
    parser.add_argument("--verified-proposals", default="verified_proposals.jsonl",
                        help="VP filename inside stage1 dir (default: verified_proposals.jsonl)")
    parser.add_argument("--finding-ids-from-vp", default=None,
                        help="Restrict rows to finding IDs present in this VP filename inside stage1 dir")
    args = parser.parse_args()

    # ── 1. Load all sources ──
    print("Loading data sources...")
    config = load_config(args.config)
    stage0 = Path(resolve(config, "outputs.stage0"))
    stage0_5 = Path(resolve(config, "outputs.stage0_5"))
    stage1 = Path(resolve(config, "outputs.stage1"))
    stage2 = Path(resolve(config, "outputs.stage2"))
    out_dir = Path(resolve(config, "outputs.root"))
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest) if args.manifest else Path(resolve(config, "manifests.default"))
    manifest_rows = load_jsonl(manifest_path)
    manifest = {r["id"]: r for r in manifest_rows}

    router_rows = load_jsonl(stage0 / "stage0_router_predictions.jsonl")
    router = {r["id"]: r for r in router_rows}

    crop_rows = load_jsonl(stage0 / "stage0_crop_groups.jsonl")
    # Build finding_id -> crop_group mapping
    finding_to_crop = {}
    for cg in crop_rows:
        for fid in cg["finding_ids"]:
            finding_to_crop[fid] = cg

    p05_rows = load_jsonl(stage0_5 / "stage0_5_proposals.jsonl")
    p05 = {r["finding_id"]: r for r in p05_rows}

    vp_rows = load_jsonl(stage1 / args.verified_proposals)
    vp = {r["finding_id"]: r for r in vp_rows}

    if args.finding_ids_from_vp:
        filter_rows = load_jsonl(stage1 / args.finding_ids_from_vp)
        requested_fids = {r["finding_id"] for r in filter_rows}
        print(f"[Filter] Restricting gate table to {len(requested_fids)} findings from {args.finding_ids_from_vp}")
    else:
        requested_fids = set(manifest.keys())

    s1_metrics_file = stage1 / "metrics_vs_baseline.json"
    if not s1_metrics_file.exists():
        print(f"[WARN] {s1_metrics_file} not found — run 1_metrics stage first")
        s1_pf = {}
    else:
        s1_data = load_json(s1_metrics_file)
        s1_pf_data = s1_data.get("per_finding_pipeline", s1_data.get("per_finding", []))
        s1_pf = {pf["finding_id"]: pf for pf in s1_pf_data}

    s2_file = stage2 / "eval_finding_level.json"
    s2_pf = {}
    if s2_file.exists():
        s2_data = load_json(s2_file)
        s2_pf = {pf["finding_id"]: pf for pf in s2_data.get("per_finding", [])}

    roi_rows = load_jsonl(stage2 / "roi_manifest.jsonl")
    roi_by_fid = defaultdict(list)
    for r in roi_rows:
        fid = r.get("parent_sample_id") or r.get("finding_id")
        roi_by_fid[fid].append(r)

    # ── 2. Build rows ──
    all_fids = sorted(requested_fids & set(manifest.keys()))
    rows = []
    skipped = 0

    for fid in all_fids:
        m = manifest.get(fid)
        if m is None:
            skipped += 1
            continue

        row: dict = {}

        # ── meta__ ──
        rr = router.get(fid, {})
        row["meta__finding_id"] = fid
        row["meta__case_name"] = m["case_name"]
        row["meta__category"] = rr.get("pred_category", m.get("category", "?"))

        # ── feature__ keywords ──
        prompt = m.get("prompt", "")
        row.update(extract_keywords(prompt))

        # ── feature__ Stage0 Router ──
        row["feature__pred_category"] = rr.get("pred_category", "?")
        row["feature__category_confidence"] = rr.get("category_confidence", 0)
        row["feature__anatomy_target"] = rr.get("anatomy_target", "?")
        row["feature__laterality"] = rr.get("laterality", "?")
        row["feature__anatomy_group"] = rr.get("anatomy_group", "?")
        row["feature__final_policy"] = rr.get("final_policy", "?")
        row["feature__final_tightness"] = rr.get("final_tightness", "?")
        row["feature__tightness_confidence"] = rr.get("tightness_confidence", 0)
        row["feature__fail_open_reason_is_null"] = int(rr.get("fail_open_reason") is None)
        row["feature__anatomy_fallback_reason_is_null"] = int(rr.get("anatomy_fallback_reason") is None)
        lobes = rr.get("anatomy_lobes") or []
        row["feature__n_anatomy_lobes"] = len(lobes)
        row["feature__is_both_lungs"] = int(rr.get("anatomy_target") == "both_lungs")
        row["feature__is_conservative_crop"] = int(rr.get("final_tightness") == "conservative")

        # ── feature__ Crop geometry ──
        cg = finding_to_crop.get(fid)
        full_shape = m.get("mask_shape_hwd", [512, 512, 200])
        if cg is not None:
            bbox = cg["bbox_hwd"]  # [h0, h1, w0, w1, d0, d1]
            crop_h = bbox[1] - bbox[0]
            crop_w = bbox[3] - bbox[2]
            crop_d = bbox[5] - bbox[4]
            crop_vol = crop_h * crop_w * crop_d
            full_h, full_w, full_d = full_shape
            full_vol = full_h * full_w * full_d
            row["feature__crop_volume_ratio"] = round(crop_vol / max(1, full_vol), 6)
            row["feature__crop_h"] = crop_h
            row["feature__crop_w"] = crop_w
            row["feature__crop_d"] = crop_d
            row["feature__crop_volume"] = crop_vol
            row["feature__crop_center_h_norm"] = round((bbox[0] + bbox[1]) / 2 / max(1, full_h), 4)
            row["feature__crop_center_w_norm"] = round((bbox[2] + bbox[3]) / 2 / max(1, full_w), 4)
            row["feature__crop_center_d_norm"] = round((bbox[4] + bbox[5]) / 2 / max(1, full_d), 4)
            row["feature__crop_touches_h0"] = int(bbox[0] == 0)
            row["feature__crop_touches_h1"] = int(bbox[1] >= full_h)
            row["feature__crop_touches_w0"] = int(bbox[2] == 0)
            row["feature__crop_touches_w1"] = int(bbox[3] >= full_w)
            row["feature__crop_touches_d0"] = int(bbox[4] == 0)
            row["feature__crop_touches_d1"] = int(bbox[5] >= full_d)
            row["feature__crop_is_fullish"] = int(row["feature__crop_volume_ratio"] > 0.7)
            row["feature__n_findings_in_crop_group"] = cg.get("n_findings", 1)
        else:
            for key in ["feature__crop_volume_ratio", "feature__crop_h", "feature__crop_w",
                        "feature__crop_d", "feature__crop_volume",
                        "feature__crop_center_h_norm", "feature__crop_center_w_norm",
                        "feature__crop_center_d_norm",
                        "feature__crop_touches_h0", "feature__crop_touches_h1",
                        "feature__crop_touches_w0", "feature__crop_touches_w1",
                        "feature__crop_touches_d0", "feature__crop_touches_d1",
                        "feature__crop_is_fullish", "feature__n_findings_in_crop_group",
                        "feature__proposal_max_volume_ratio_to_crop",
                        "feature__proposal_total_volume_ratio_to_crop"]:
                row[key] = 0

        # ── feature__ Stage0.5 proposals ──
        p05_row = p05.get(fid, {})
        row["feature__expert"] = str(p05_row.get("expert", "?"))
        row["feature__s05_fallback"] = int(p05_row.get("fallback", False))
        proposals = p05_row.get("proposals", [])
        row["feature__n_proposals"] = len(proposals)
        pvols = [p.get("volume_voxels", 0) for p in proposals]
        row["feature__proposal_total_volume"] = sum(pvols)
        row["feature__proposal_max_volume"] = safe_max(pvols)
        row["feature__proposal_mean_volume"] = safe_mean(pvols)
        row["feature__proposal_median_volume"] = safe_median(pvols)
        row["feature__proposal_volume_std"] = safe_std(pvols)
        crop_vol = row.get("feature__crop_volume", 1)
        row["feature__proposal_max_volume_ratio_to_crop"] = round(
            row["feature__proposal_max_volume"] / max(1, crop_vol), 6)
        row["feature__proposal_total_volume_ratio_to_crop"] = round(
            row["feature__proposal_total_volume"] / max(1, crop_vol), 6)
        row["feature__proposal_source_hu_count"] = sum(
            1 for p in proposals if p.get("source_expert") == "hu")
        row["feature__proposal_source_nodule_count"] = sum(
            1 for p in proposals if p.get("source_expert") == "nodule")
        detector_scores = [p.get("detector_score", 0) or 0 for p in proposals
                          if p.get("detector_score") is not None]
        row["feature__proposal_detector_score_max"] = safe_max(detector_scores)
        row["feature__proposal_detector_score_mean"] = safe_mean(detector_scores)
        hu_contrasts = [p.get("hu_contrast", 0) or 0 for p in proposals
                       if p.get("hu_contrast") is not None]
        row["feature__proposal_hu_contrast_max"] = safe_max(hu_contrasts)

        # ── feature__ Stage1 verifier ──
        vp_row = vp.get(fid, {})
        verified = vp_row.get("verified", [])
        rejected = vp_row.get("rejected", [])
        n_verified = len(verified)
        n_rejected = len(rejected)
        row["feature__n_verified"] = n_verified
        row["feature__n_rejected"] = n_rejected
        row["feature__verified_ratio"] = round(n_verified / max(1, len(proposals)), 4)
        row["feature__s1_fallback"] = int(vp_row.get("fallback", False))

        v_maxprobs = [v.get("max_prob", 0) for v in verified]
        v_meanprobs = [v.get("mean_prob", 0) for v in verified]
        v_fgratios = [v.get("fg_ratio", 0) for v in verified]
        v_scores = [v.get("score", 0) for v in verified]
        r_scores = [r.get("score", 0) for r in rejected]

        row["feature__verified_max_prob_max"] = safe_max(v_maxprobs)
        row["feature__verified_max_prob_mean"] = safe_mean(v_maxprobs)
        row["feature__verified_mean_prob_mean"] = safe_mean(v_meanprobs)
        row["feature__verified_fg_ratio_max"] = safe_max(v_fgratios)
        row["feature__verified_fg_ratio_mean"] = safe_mean(v_fgratios)
        row["feature__verified_score_max"] = safe_max(v_scores)
        row["feature__verified_score_mean"] = safe_mean(v_scores)
        row["feature__rejected_score_max"] = safe_max(r_scores)
        row["feature__verified_rejected_score_gap"] = round(
            safe_max(v_scores) - safe_max(r_scores), 6)
        row["feature__coarse_mask_count"] = sum(
            1 for v in verified if v.get("coarse_mask_path"))
        row["feature__s1_is_fullct_voxtell"] = int(
            str(p05_row.get("expert", "")).lower() in ("full_ct_voxtell", "fullct_voxtell"))

        # ── feature__ Stage1 coarse mask ──
        s1_row = s1_pf.get(fid, {})
        s1_pred = s1_row.get("pred_voxels", 0)
        row["feature__s1_pred_voxels"] = s1_pred
        row["feature__s1_pred_log_voxels"] = math.log1p(s1_pred)
        row["feature__s1_pred_volume_ratio_to_crop"] = round(
            s1_pred / max(1, crop_vol), 6)
        row["feature__s1_pred_is_empty"] = int(s1_pred == 0)
        row["feature__s1_pred_is_huge"] = int(s1_pred > 50_000_000)

        # ── feature__ Stage2 ROI ──
        fid_rois = roi_by_fid.get(fid, [])
        n_rois = len(fid_rois)
        row["feature__n_rois"] = n_rois
        if n_rois > 0:
            rvols = [r.get("voxel_count", 0) for r in fid_rois]
            row["feature__roi_total_volume"] = sum(rvols)
            row["feature__roi_max_volume"] = safe_max(rvols)
            row["feature__roi_mean_volume"] = safe_mean(rvols)
            row["feature__roi_median_volume"] = safe_median(rvols)
            row["feature__roi_volume_std"] = safe_std(rvols)
            row["feature__roi_max_volume_ratio_to_crop"] = round(
                safe_max(rvols) / max(1, crop_vol), 6)
            row["feature__roi_total_volume_ratio_to_crop"] = round(
                sum(rvols) / max(1, crop_vol), 6)
            row["feature__n_coarse_rois"] = sum(
                1 for r in fid_rois if r.get("bbox_source") == "coarse_mask")
            row["feature__n_proposal_rois"] = sum(
                1 for r in fid_rois if r.get("bbox_source") == "proposal")
            row["feature__roi_coarse_fraction"] = round(
                row["feature__n_coarse_rois"] / max(1, n_rois), 4)
            # Average depth (z-dimension span)
            d_spans = [(r.get("orig_bbox_hwd", [0,0,0,0,0,0])[5] -
                        r.get("orig_bbox_hwd", [0,0,0,0,0,0])[4]) for r in fid_rois]
            row["feature__roi_shape_d_mean"] = safe_mean(d_spans)
        else:
            for key in ["feature__roi_total_volume", "feature__roi_max_volume",
                        "feature__roi_mean_volume", "feature__roi_median_volume",
                        "feature__roi_volume_std", "feature__roi_max_volume_ratio_to_crop",
                        "feature__roi_total_volume_ratio_to_crop",
                        "feature__n_coarse_rois", "feature__n_proposal_rois",
                        "feature__roi_coarse_fraction", "feature__roi_shape_d_mean"]:
                row[key] = 0.0

        # ── feature__ Stage2 raw prediction ──
        s2_row = s2_pf.get(fid, {})
        s2_pred = s2_row.get("s2_raw_pred_voxels", 0)
        row["feature__s2_raw_pred_voxels"] = s2_pred
        row["feature__s2_raw_log_pred_voxels"] = math.log1p(s2_pred) if s2_pred >= 0 else 0
        row["feature__s2_s1_volume_ratio"] = round(
            s2_pred / max(1, s1_pred), 6)
        roi_total = row.get("feature__roi_total_volume", 0)
        row["feature__s2_roi_pred_density"] = round(
            s2_pred / max(1, roi_total), 6) if n_rois > 0 else 0.0
        row["feature__s2_pred_is_empty"] = int(s2_pred == 0)
        row["feature__s2_pred_is_huge"] = int(s2_pred > 100_000_000)
        row["feature__s2_overseg_ratio_gt5"] = int(
            row["feature__s2_s1_volume_ratio"] > 5.0)
        row["feature__s2_underseg_ratio_lt02"] = int(
            0 < row["feature__s2_s1_volume_ratio"] < 0.2)

        # ── feature__ S1-S2 overlap (from eval script, no GT needed) ──
        row["feature__s1_s2_intersection_voxels"] = s2_row.get("s1_s2_intersection_voxels", 0)
        row["feature__s2_overlap_with_s1"] = s2_row.get("s2_overlap_with_s1", 0.0)
        row["feature__s1_covered_by_s2"] = s2_row.get("s1_covered_by_s2", 0.0)
        row["feature__s1_s2_dice_proxy"] = s2_row.get("s1_s2_dice_proxy", 0.0)
        row["feature__s2_outside_s1_fraction"] = s2_row.get("s2_outside_s1_fraction", 0.0)
        row["feature__s1_outside_s2_fraction"] = s2_row.get("s1_outside_s2_fraction", 0.0)

        # ── feature__ S2 confidence (from STU-Net logits) ──
        for key in ["s2_conf_mean_prob", "s2_conf_std_prob", "s2_conf_p10_prob",
                     "s2_conf_p90_prob", "s2_conf_high_conf_frac_09",
                     "s2_conf_low_conf_frac_05_07", "s2_conf_soft_hard_ratio",
                     "s2_conf_mean_margin", "s2_conf_p10_margin",
                     "s2_conf_mean_entropy", "s2_roi_conf_spread"]:
            row[f"feature__{key}"] = s2_row.get(key, 0.0)
        row["feature__s2_n_low_conf_rois"] = s2_row.get("s2_n_low_conf_rois", 0)

        rows.append(row)

    print(f"Built {len(rows)} finding rows (skipped {skipped})")

    if not rows:
        print("[ERROR] No rows built — check upstream stages")
        sys.exit(1)

    # ── 3. Ensure all rows have same columns & order ──
    all_keys = list(rows[0].keys())
    for r in rows:
        for k in all_keys:
            if k not in r:
                r[k] = ""

    # Sort columns: meta__ first, feature__ second
    meta_cols = sorted([k for k in all_keys if k.startswith("meta__")])
    feat_cols = sorted([k for k in all_keys if k.startswith("feature__")])
    ordered_cols = meta_cols + feat_cols

    # ── 4. Write CSV ──
    out_csv = out_dir / "gate_feature_table.csv"
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=ordered_cols)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_csv}")
    print(f"  meta__ cols:    {len(meta_cols)}")
    print(f"  feature__ cols: {len(feat_cols)}")


if __name__ == "__main__":
    main()
