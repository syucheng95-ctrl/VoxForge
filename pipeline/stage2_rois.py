"""Stage2 ROI maker: reads Stage1 verified proposals, crops proposal-level ROIs.

Produces roi_images/ + roi_masks/ + roi_manifest.jsonl compatible with
eval_stratified.py --mode sliding-window.
With --no-labels, only roi_images/ + roi_manifest.jsonl are produced.

Margin logic for ROI cropping:
  small/medium → fixed (16, 16, 4)
  large/xlarge → proportional max(16, lesion_size * 0.25)
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

from src.utils import load_config, load_jsonl, resolve, write_jsonl

# ── Hybrid policy ──
VOXTELL_MASK_CATEGORIES = {"1e", "2a", "2b", "2d", "2g"}
MIN_MASK_FG = 50

# ── High-threshold component extraction ──
ROI_MASK_THRESHOLD = 0.6  # requires prob map from Stage1; ignored for old binary masks

# ── Component filtering ──
MIN_COMPONENT_VOXELS = 500   # discard components smaller than this
MIN_COMPONENT_EXTENT = 4     # discard components thinner than this on any axis

# ── Top-K (category-dependent) ──
TOP_K_FOCAL = 5
TOP_K_DIFFUSE = 10
TOP_K_FULLCT = 5
DIFFUSE_CATEGORIES = {"1d", "1e", "1f", "2a"}
# Note: 1a/1b/1c/2f moved to full_ct_voxtell; 1e moved to diffuse.

# ── Proposal fallback volume filter ──
PROPOSAL_MAX_VOXELS = 50_000_000  # absolute fallback; proposal ROI > this is skipped

# ── Margin constants (aligned with recrop_v3_rois.py) ──
FIXED_MARGIN_SMALL_MEDIUM = (16, 16, 4)
MIN_MARGIN_LARGE_XLARGE = (16, 16, 16)
MARGIN_RATIO = 0.25


def get_group(hwd):
    """Group ROI by HWD shape (same as recrop_v3_rois.py)."""
    h, w, d = [int(v) for v in hwd]
    if h <= 64 and w <= 64 and d <= 32:
        return "small"
    if d > 48:
        return "xlarge"
    if h > 96 or w > 96:
        return "large"
    return "medium"


def bbox_size(bbox):
    """bbox [h0,h1,w0,w1,d0,d1] → (h, w, d) size tuple."""
    return (bbox[1] - bbox[0], bbox[3] - bbox[2], bbox[5] - bbox[4])


def proportional_margin(box_hwd, min_margin_hwd, ratio):
    """margin = max(min_margin, lesion_size * ratio) per axis.

    box_hwd is pipeline format: [h0, h1, w0, w1, d0, d1].
    """
    h0, h1, w0, w1, d0, d1 = box_hwd
    lesion = (h1 - h0, w1 - w0, d1 - d0)
    return tuple(
        max(int(min_margin_hwd[i]), int(round(lesion[i] * ratio)))
        for i in range(3)
    )


def expand_box(box_hwd, shape_hwd, margin_hwd):
    """Expand bbox by margin, clamped to CT shape.

    box_hwd is pipeline format: [h0, h1, w0, w1, d0, d1].
    Returns same format.
    """
    h0, h1, w0, w1, d0, d1 = box_hwd
    mh, mw, md = margin_hwd
    return [
        max(0, h0 - mh),
        min(shape_hwd[0], h1 + mh),
        max(0, w0 - mw),
        min(shape_hwd[1], w1 + mw),
        max(0, d0 - md),
        min(shape_hwd[2], d1 + md),
    ]


def crop_roi(data, affine, bbox, out_path, dtype):
    """Crop 3D array to bbox [h0,h1,w0,w1,d0,d1], update affine, save."""
    H, W, D = data.shape[:3]
    h0, h1 = max(0, bbox[0]), min(H, bbox[1])
    w0, w1 = max(0, bbox[2]), min(W, bbox[3])
    d0, d1 = max(0, bbox[4]), min(D, bbox[5])
    if h0 >= h1 or w0 >= w1 or d0 >= d1:
        raise ValueError(f"bbox {bbox} outside array shape ({H},{W},{D})")
    arr = data[h0:h1, w0:w1, d0:d1].copy()
    new_affine = affine.copy()
    new_affine[:3, 3] = nib.affines.apply_affine(affine, [h0, w0, d0])
    nib.save(nib.Nifti1Image(arr.astype(dtype), new_affine), str(out_path))


def map_mask_bbox_to_ct(mask_bbox_dwh, proposal_bbox_hwd):
    """Map a VoxTell mask bbox back to original CT HWD coords.

    Mask is in VoxTell reader DWH space. Mapping:
      mask(d,w,h) → orig_d = pd0 + d, orig_w = pw1 - w, orig_h = ph1 - h.
    """
    d0, d1, w0, w1, h0, h1 = [int(v) for v in mask_bbox_dwh]
    ph0, ph1, pw0, pw1, pd0, pd1 = proposal_bbox_hwd
    # Map from mask coords to original CT (H/W axes flipped vs reader)
    orig_h0 = ph1 - h1
    orig_h1 = ph1 - h0
    orig_w0 = pw1 - w1
    orig_w1 = pw1 - w0
    orig_d0 = pd0 + d0
    orig_d1 = pd0 + d1
    if orig_h0 >= orig_h1 or orig_w0 >= orig_w1 or orig_d0 >= orig_d1:
        return None
    return [orig_h0, orig_h1, orig_w0, orig_w1, orig_d0, orig_d1]


def _load_mask_data(mask_path):
    """Load mask array, preferring high-threshold prob map if available.

    Returns (binary_mask, used_threshold) or (None, None) on failure.
    """
    # Try probability map first (saved by updated _verifier.py)
    prob_path = Path(str(mask_path).replace("_coarse.nii.gz", "_prob.nii.gz"))
    if prob_path.exists():
        try:
            prob = np.asanyarray(nib.load(str(prob_path)).dataobj).astype(np.float32)
            if np.any(prob > 0):
                return prob > ROI_MASK_THRESHOLD, f"prob>{ROI_MASK_THRESHOLD}"
        except Exception:
            pass
    # Fallback: binary mask (compatible with old Stage1 outputs)
    try:
        mask = np.asanyarray(nib.load(str(mask_path)).dataobj) > 0
        return mask, "binary>0"
    except Exception:
        return None, None


def load_mask_component_bboxes(mask_path, proposal_bbox_hwd, min_fg=MIN_MASK_FG):
    """Read coarse mask and return one original-CT bbox per connected component.

    If a probability map exists, thresholds at ROI_MASK_THRESHOLD (0.6) for
    tighter components.  Falls back to binary mask for old Stage1 outputs.

    ROIs are component-level crops. Keeping VoxTell coarse-mask ROIs
    component-level avoids turning multi-focal masks into one large union bbox.
    """
    try:
        mask, used_th = _load_mask_data(mask_path)
        if mask is None or int(mask.sum()) < min_fg:
            return []

        labeled, n_components = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
        component_sizes = np.bincount(labeled.ravel())
        component_slices = ndimage.find_objects(labeled)
        boxes = []
        for component_id, slc in enumerate(component_slices, start=1):
            if slc is None:
                continue
            component_size = int(component_sizes[component_id])
            if component_size < MIN_COMPONENT_VOXELS:
                continue
            d0, d1 = int(slc[0].start), int(slc[0].stop)
            w0, w1 = int(slc[1].start), int(slc[1].stop)
            h0, h1 = int(slc[2].start), int(slc[2].stop)
            # Discard components that are too thin/flat on any axis
            if (d1 - d0) < MIN_COMPONENT_EXTENT or (w1 - w0) < MIN_COMPONENT_EXTENT or (h1 - h0) < MIN_COMPONENT_EXTENT:
                continue
            mapped = map_mask_bbox_to_ct((d0, d1, w0, w1, h0, h1), proposal_bbox_hwd)
            if mapped is not None:
                boxes.append((component_size, mapped))

        boxes.sort(key=lambda item: item[0], reverse=True)
        return [(size, box) for size, box in boxes]
    except Exception:
        return []


def main():
    parser = argparse.ArgumentParser(description="Stage2 ROI maker")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--manifest", help="Manifest JSONL (default.jsonl)")
    parser.add_argument("--limit-cases", type=int, default=0, help="Limit N cases")
    parser.add_argument("--categories", default="",
                        help="Comma-separated categories to include (empty = all)")
    parser.add_argument("--include-rejected-topk", type=int, default=0,
                        help="Include top-K rejected Stage1 proposals as weak ROIs")
    parser.add_argument("--include-rejected-min-rejected", type=int, default=0,
                        help="Only include rejected proposals when finding has at least this many rejected proposals")
    parser.add_argument("--include-rejected-categories", default="2c,2d",
                        help="Comma-separated categories allowed to use rejected proposals")
    parser.add_argument("--verified-proposals", default="verified_proposals.jsonl",
                        help="VP filename inside stage1 dir (default: verified_proposals.jsonl)")
    parser.add_argument("--no-labels", action="store_true",
                        help="Inference mode: do not require or crop GT labels")
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = Path(resolve(config, "outputs.stage2"))
    image_dir = out_dir / "roi_images"
    mask_dir = out_dir / "roi_masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_labels:
        mask_dir.mkdir(parents=True, exist_ok=True)

    ct_images_dir = Path(resolve(config, "data.ct_images"))
    label_root = Path(resolve(config, "data.ct_images")).parent  # data/

    # ── Load inputs ──
    manifest_path = args.manifest or resolve(config, "manifests.default")
    manifest_rows = load_jsonl(manifest_path)
    manifest_by_id = {r["id"]: r for r in manifest_rows}

    stage1_out = Path(resolve(config, "outputs.stage1"))
    verified_rows = load_jsonl(stage1_out / args.verified_proposals)

    rejected_cats = {c.strip() for c in args.include_rejected_categories.split(",") if c.strip()}
    if args.include_rejected_topk > 0:
        patched_rows = []
        added = 0
        for row in verified_rows:
            cat = row.get("category", "")
            if cat in rejected_cats:
                rejected = list(row.get("rejected", []))
                rejected.sort(key=lambda x: float(x.get("score", 0) or 0), reverse=True)
                extra = []
                for rj in rejected[:args.include_rejected_topk]:
                    rj = dict(rj)
                    rj["verified"] = True
                    rj["weak_stage1_rejected"] = True
                    rj["bbox_source_hint"] = "stage1_rejected_topk"
                    extra.append(rj)
                if extra:
                    row = dict(row)
                    row["verified"] = list(row.get("verified", [])) + extra
                    added += len(extra)
            patched_rows.append(row)
        verified_rows = patched_rows
        print(f"Included {added} rejected top proposals as weak ROIs for categories {sorted(rejected_cats)}")

    wanted_cats = None
    if args.categories.strip():
        wanted_cats = {c.strip() for c in args.categories.split(",") if c.strip()}

    # ── Build finding → source CT / label index ──
    # Stage0 crop_groups has source_image per finding.  We also need it from manifest.
    # For robustness, build from manifest first, then override from crop_groups.
    stage0_out = Path(resolve(config, "outputs.stage0"))
    crop_groups = load_jsonl(stage0_out / "stage0_crop_groups.jsonl")
    fid_to_source = {}
    for g in crop_groups:
        src = g.get("source_image", "")
        for fid in g.get("finding_ids", []):
            fid_to_source[fid] = src

    # ── Process findings ──
    if args.limit_cases:
        # Limit by distinct case_name in verified_rows
        seen_cases = set()
        limited_verified = []
        for vrow in verified_rows:
            fid = vrow["finding_id"]
            mrow = manifest_by_id.get(fid)
            if mrow is None:
                continue
            case = mrow["case_name"]
            if len(seen_cases) >= args.limit_cases:
                break
            seen_cases.add(case)
            limited_verified.append(vrow)
        verified_rows = limited_verified
        print(f"Limited to {len(seen_cases)} cases, {len(verified_rows)} findings")

    # ── Pre-resolve CT keys and sort by CT so same-CT findings are consecutive ──
    # This way we only keep ONE CT in memory at a time.
    resolved_rows = []
    for vrow in verified_rows:
        fid = vrow["finding_id"]
        cat = vrow.get("category", "?")
        if wanted_cats and cat not in wanted_cats:
            continue
        mrow = manifest_by_id.get(fid)
        if mrow is None:
            continue
        source_ct = fid_to_source.get(fid)
        if not source_ct:
            source_ct = str(ct_images_dir.parent / mrow["image"])
        source_ct_path = Path(source_ct)
        if not source_ct_path.exists():
            source_ct_path = ct_images_dir / mrow["case_name"]
        if not source_ct_path.exists():
            continue
        label_path = None
        if not args.no_labels:
            label_rel = mrow["label"].replace("\\", "/").replace("labels_finding/", "")
            label_path = label_root / label_rel
            if not label_path.exists():
                continue
        resolved_rows.append((str(source_ct_path), str(label_path) if label_path else "", vrow))
    resolved_rows.sort(key=lambda x: x[0])  # sort by CT path

    roi_rows = []
    stats = defaultdict(lambda: {"n": 0, "source": defaultdict(int)})
    skipped_finding_ids = []
    comp_counter = 0
    t0 = time.time()

    # Single-CT cache: only keep the current CT in memory, drop when CT changes
    current_ct_key: str | None = None
    current_ct: tuple[np.ndarray, np.ndarray] | None = None

    for ct_key, label_key, vrow in resolved_rows:
        fid = vrow["finding_id"]
        cat = vrow.get("category", "?")
        mrow = manifest_by_id[fid]

        # Load CT only if it changed (findings are sorted by CT)
        if ct_key != current_ct_key:
            ct_nii = nib.load(ct_key)
            ct_data = np.asanyarray(ct_nii.dataobj)
            if ct_data.ndim == 4:
                ct_data = ct_data[..., 0]
            current_ct_key = ct_key
            current_ct = (ct_data.astype(np.float32), ct_nii.affine.copy())
        ct_arr, ct_affine = current_ct
        ct_shape = ct_arr.shape[:3]

        if not args.no_labels:
            # Label: always load fresh (uint8, small; caching not worth the OOM risk)
            lb_nii = nib.load(label_key)
            lb_data = np.asanyarray(lb_nii.dataobj)
            if lb_data.ndim == 4:
                lb_data = lb_data[..., 0]
            lb_arr = lb_data.astype(np.uint8)
            lb_affine = lb_nii.affine.copy()

        # ── Collect all lesion_boxes for this finding (with size for global sort) ──
        finding_lesions = []  # (lesion_box, bbox_source, component_size) list

        is_fullct = (
            vrow.get("use_full_ct_voxtell")
            or vrow.get("expert") == "full_ct_voxtell"
        )

        for ventry in vrow["verified"]:
            bbox_source = "proposal"
            lesion_entries = []  # (lesion_box, component_size)
            if cat in VOXTELL_MASK_CATEGORIES or is_fullct:
                mask_path = ventry.get("coarse_mask_path")
                if mask_path and Path(mask_path).exists():
                    mask_data = np.asanyarray(nib.load(str(mask_path)).dataobj) > 0
                    if mask_data.sum() >= MIN_MASK_FG:
                        # Returns list of (component_size, lesion_box)
                        lesion_entries = load_mask_component_bboxes(
                            mask_path, ventry["proposal_bbox_hwd"], min_fg=MIN_MASK_FG)
                        if lesion_entries:
                            bbox_source = "coarse_mask"

            if not lesion_entries:
                if is_fullct:
                    continue  # fullCT categories: no fallback to full-CT bbox
                # Proposal fallback: filter by volume
                prop_box = list(ventry["proposal_bbox_hwd"])
                prop_vol = (prop_box[1] - prop_box[0]) * (prop_box[3] - prop_box[2]) * (prop_box[5] - prop_box[4])
                # Only filter huge proposal fallback for coarse-mask categories.
                # Proposal-only categories have no other ROI source; skipping = 0 ROIs.
                if prop_vol > PROPOSAL_MAX_VOXELS and cat in VOXTELL_MASK_CATEGORIES:
                    continue
                # Proposal boxes get a nominal size of 1 (always last after coarse components)
                lesion_entries = [(1, prop_box)]
                bbox_source = "proposal"

            for comp_size, lesion_box in lesion_entries:
                finding_lesions.append((lesion_box, bbox_source, comp_size))

        # ── Global sort + Top-K per finding ──
        # Sort all components across proposals by component size descending,
        # so larger components from later proposals can displace smaller ones.
        finding_lesions.sort(key=lambda x: x[2], reverse=True)
        if is_fullct:
            top_k = TOP_K_FULLCT
        elif cat in DIFFUSE_CATEGORIES:
            top_k = TOP_K_DIFFUSE
        else:
            top_k = TOP_K_FOCAL
        if len(finding_lesions) > top_k:
            finding_lesions = finding_lesions[:top_k]

        # ── Track skipped findings ──
        if len(finding_lesions) == 0:
            skipped_finding_ids.append(fid)
            continue

        for lesion_box, bbox_source, _comp_size in finding_lesions:
            comp_id = f"comp_{comp_counter:08d}"
            comp_counter += 1

            # Apply margin
            lesion_hwd = bbox_size(lesion_box)
            group = get_group(lesion_hwd)
            if group in ("small", "medium"):
                margin = FIXED_MARGIN_SMALL_MEDIUM
            else:
                margin = proportional_margin(lesion_box, MIN_MARGIN_LARGE_XLARGE, MARGIN_RATIO)
            roi_box = expand_box(lesion_box, ct_shape, margin)

            # Crop from cached arrays
            img_out = image_dir / f"{comp_id}_image.nii.gz"
            crop_roi(ct_arr, ct_affine, roi_box, img_out, np.float32)

            roi_shape = bbox_size(roi_box)
            roi_row = {
                "component_sample_id": comp_id,
                "parent_sample_id": fid,
                "prompt": vrow.get("prompt", mrow.get("prompt", "")),
                "roi_image_path": f"roi_images/{comp_id}_image.nii.gz",
                "roi_shape_hwd": [int(v) for v in roi_shape],
                "voxel_count": int(np.prod(roi_shape)),
                "category": cat,
                "case_name": mrow["case_name"],
                "source_image": ct_key,
                "array_axis_order": "xyz",
                "orig_bbox_hwd": [int(v) for v in roi_box],
                "bbox_source": bbox_source,
            }
            if not args.no_labels:
                msk_out = mask_dir / f"{comp_id}_mask.nii.gz"
                crop_roi(lb_arr, lb_affine, roi_box, msk_out, np.uint8)
                roi_row["roi_mask_path"] = f"roi_masks/{comp_id}_mask.nii.gz"
                roi_row["source_label"] = label_key
            roi_rows.append(roi_row)

            stats[cat]["n"] += 1
            stats[cat]["source"][bbox_source] += 1

            if len(roi_rows) % 10 == 0:
                elapsed = time.time() - t0
                print(f"  [{len(roi_rows)} ROIs, {elapsed:.0f}s]", flush=True)

    # ── Write outputs ──
    manifest_out = out_dir / "roi_manifest.jsonl"
    write_jsonl(manifest_out, roi_rows)
    print(f"\nPrepared {len(roi_rows)} ROIs → {manifest_out}")
    if skipped_finding_ids:
        print(f"Skipped {len(skipped_finding_ids)} findings (no valid ROIs): "
              f"{skipped_finding_ids[:10]}{'...' if len(skipped_finding_ids) > 10 else ''}")

    print("\nPer-category ROI stats:")
    for cat in sorted(stats):
        s = stats[cat]
        print(f"  {cat}: n={s['n']}, sources={dict(s['source'])}")

    summary = {
        "n_findings_with_rois": len({r["parent_sample_id"] for r in roi_rows}),
        "n_rois": len(roi_rows),
        "n_skipped_findings": len(skipped_finding_ids),
        "skipped_finding_ids": skipped_finding_ids,
        "per_category": {cat: dict(s) for cat, s in stats.items()},
        "total_time_s": round(time.time() - t0, 1),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary → {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
