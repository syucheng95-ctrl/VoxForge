"""Stage2 finding-level eval: STU-Net per-ROI inference → paste-back → S2 raw metrics + features.

Reads roi_manifest.jsonl (must include orig_bbox_hwd), runs STU-Net
sliding-window on each ROI, pastes predictions back to full-CT finding-level,
unions all ROIs per finding, and outputs S2 raw metrics with overlap/confidence
features for downstream gate training.

Usage:
  python pipeline/stage2_eval.py --config configs/pipeline.yaml
  python pipeline/stage2_eval.py --config configs/pipeline.yaml --categories 1e,2b,2d
  python pipeline/stage2_eval.py --config configs/pipeline.yaml --limit-findings 10
  python pipeline/stage2_eval.py --config configs/pipeline.yaml --no-gt
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from src.utils import load_config, load_jsonl, resolve



def reader_mask_to_raw_hwd(mask_reader: np.ndarray) -> np.ndarray:
    """Convert saved reader-space DWH mask crop back to raw HWD crop."""
    return mask_reader.transpose(2, 1, 0)[::-1, ::-1, :]


def paste_coarse_mask(full_mask: np.ndarray, entry: dict, gate_bbox: list[int]) -> None:
    """Paste a single Stage1 coarse mask into full_mask (HWD, boolean)."""
    path = entry.get("coarse_mask_path")
    if not path:
        return
    try:
        mask_reader = np.asanyarray(nib.load(path).dataobj) > 0
    except (FileNotFoundError, OSError):
        return
    if mask_reader.sum() == 0:
        return
    gh0, gh1, gw0, gw1, gd0, gd1 = gate_bbox
    raw_h, raw_w, raw_d = gh1 - gh0, gw1 - gw0, gd1 - gd0
    ph0, ph1, pw0, pw1, pd0, pd1 = entry["proposal_bbox_hwd"]
    ch0 = max(0, ph0 - gh0); ch1 = min(raw_h, ph1 - gh0)
    cw0 = max(0, pw0 - gw0); cw1 = min(raw_w, pw1 - gw0)
    cd0 = max(0, pd0 - gd0); cd1 = min(raw_d, pd1 - gd0)
    if ch0 >= ch1 or cw0 >= cw1 or cd0 >= cd1:
        return
    mask_raw = reader_mask_to_raw_hwd(mask_reader)
    expected = (ch1 - ch0, cw1 - cw0, cd1 - cd0)
    if mask_raw.shape != expected:
        h = min(mask_raw.shape[0], expected[0])
        w = min(mask_raw.shape[1], expected[1])
        d = min(mask_raw.shape[2], expected[2])
        mask_raw = mask_raw[:h, :w, :d]
        ch1, cw1, cd1 = ch0 + h, cw0 + w, cd0 + d
    full_mask[gh0 + ch0:gh0 + ch1, gw0 + cw0:gw0 + cw1, gd0 + cd0:gd0 + cd1] |= mask_raw


def compute_dice(tp, pred, gt):
    return (2 * tp) / max(1, pred + gt)


def resolve_ct_path(mrow: dict, ct_images_dir: Path) -> Path | None:
    """Resolve a manifest row to a CT image path."""
    candidates = []
    if mrow.get("image"):
        image_rel = mrow["image"].replace("\\", "/")
        candidates.extend([
            ct_images_dir.parent / image_rel,
            ct_images_dir / image_rel,
        ])
    if mrow.get("case_name"):
        candidates.append(ct_images_dir / mrow["case_name"])
    for p in candidates:
        if p.exists():
            return p
    return None


def window_starts(length: int, patch: int, stride: int):
    """Return start positions for sliding windows."""
    if length <= patch:
        return [0]
    starts = list(range(0, length - patch + 1, stride))
    last = length - patch
    if starts[-1] != last:
        starts.append(last)
    return starts


def main():
    parser = argparse.ArgumentParser(description="Stage2 finding-level eval with paste-back")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--manifest", default="", help="Manifest JSONL (default: config manifests.default)")
    parser.add_argument("--categories", default="", help="Comma-separated; empty = all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overlap", type=float, default=0.5, help="Sliding-window overlap")
    parser.add_argument("--mode", choices=["sliding-window", "full-roi"], default="sliding-window",
                       help="Inference mode: sliding-window or full-roi")
    parser.add_argument("--limit-findings", type=int, default=0, help="Limit N findings (0=all)")
    parser.add_argument("--verified-proposals", default="verified_proposals.jsonl",
                        help="VP filename inside stage1 dir (default: verified_proposals.jsonl)")
    parser.add_argument("--no-gt", action="store_true",
                        help="Inference mode: do not require GT labels or compute metrics")
    args = parser.parse_args()
    force_sliding_max_voxels = int(os.environ.get('STAGE2_FORCE_SLIDING_MAX_VOXELS', '0'))

    config = load_config(args.config)
    out_dir = Path(resolve(config, "outputs.stage2"))

    wanted = {c.strip() for c in args.categories.split(",") if c.strip()} or None

    # ── Load inputs ──
    roi_rows = load_jsonl(out_dir / "roi_manifest.jsonl")
    print(f"Loaded {len(roi_rows)} ROIs from roi_manifest.jsonl")
    if wanted:
        roi_rows = [r for r in roi_rows if r["category"] in wanted]
        print(f"Filtered to {len(roi_rows)} ROIs (categories: {sorted(wanted)})")

    missing_bbox = [r for r in roi_rows if "orig_bbox_hwd" not in r]
    if missing_bbox:
        print(f"[ERROR] {len(missing_bbox)}/{len(roi_rows)} ROIs missing orig_bbox_hwd.")
        print("  Re-run: python pipeline/stage2_rois.py --config configs/pipeline.yaml")
        sys.exit(1)

    manifest_path = args.manifest or resolve(config, "manifests.default")
    manifest = {r["id"]: r for r in load_jsonl(manifest_path)}
    label_root = Path(resolve(config, "data.ct_images")).parent  # data/
    ct_images_dir = Path(resolve(config, "data.ct_images"))

    # ── Load verified proposals for S1 coarse masks (used by gate) ──
    stage1_out = Path(resolve(config, "outputs.stage1"))
    verified_rows = load_jsonl(stage1_out / args.verified_proposals)
    fid_to_stage1 = {r["finding_id"]: r for r in verified_rows}

    # ── Group ROIs by finding_id ──
    finding_rois = defaultdict(list)
    for r in roi_rows:
        fid = r.get("parent_sample_id") or r.get("finding_id")
        finding_rois[fid].append(r)

    # Iterate findings produced by the current upstream run. This keeps smoke tests
    # with --limit-cases from scanning the full manifest, while still
    # preserving no-ROI fallback for Stage1 findings that produced no Stage2 ROI.
    finding_ids = sorted(set(fid_to_stage1.keys()) | set(finding_rois.keys()))
    if not finding_ids:
        finding_ids = sorted(set(manifest.keys()))
    # When --categories is set, only evaluate findings of those categories (others excluded, not fallback)
    if wanted:
        finding_ids = [fid for fid in finding_ids
                       if manifest.get(fid, {}).get("category", "") in wanted]
    if args.limit_findings:
        finding_ids = finding_ids[: args.limit_findings]
    total_rois = sum(len(finding_rois.get(fid, [])) for fid in finding_ids)
    print(f"Processing {len(finding_ids)} findings ({total_rois} ROIs)")

    # ── Load STU-Net model (via src adapter) ──
    from src.stage2.inference import load_stunet_model

    ckpt_path = resolve(config, "models.stunet")
    print(f"Loading STU-Net from {ckpt_path}...")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, gps, helpers = load_stunet_model(ckpt_path, args.device)
    get_group, pad_to_shape_centered, prepare_image = helpers
    print(f"Loaded. Patch shapes: {gps}")

    # ── Results buckets (micro-aggregated, S2 raw only; skipped in --no-gt) ──
    bucket_overall = {"tp": 0, "pred": 0, "gt": 0, "n": 0}
    bucket_by_cat = defaultdict(lambda: {"tp": 0, "pred": 0, "gt": 0, "n": 0})
    per_finding = []

    mask_out_dir = out_dir / "finding_preds"
    mask_out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    for fi, fid in enumerate(finding_ids):
        rlist = finding_rois.get(fid, [])
        mrow = manifest.get(fid)
        if mrow is None:
            print(f"  [SKIP] {fid}: not in manifest")
            continue

        cat = mrow["category"]

        if args.no_gt:
            ct_path = resolve_ct_path(mrow, ct_images_dir)
            if ct_path is None:
                print(f"  [SKIP] {fid}: CT not found")
                continue
            ref_nii = nib.load(str(ct_path))
            ref_shape = ref_nii.shape[:3]
            ref_affine = ref_nii.affine
            gt = None
            gt_vox = 0
        else:
            # Load full finding GT
            label_rel = mrow["label"].replace("\\", "/").replace("labels_finding/", "")
            gt_path = label_root / label_rel
            if not gt_path.exists():
                print(f"  [SKIP] {fid}: GT not found: {gt_path}")
                continue
            ref_nii = nib.load(str(gt_path))
            gt = np.asanyarray(ref_nii.dataobj) > 0
            ref_shape = gt.shape
            ref_affine = ref_nii.affine
            gt_vox = int(gt.sum())

        # Build S1 coarse mask for gate (paste verified coarse masks to full CT)
        coarse_full = np.zeros(ref_shape, dtype=bool)
        s1_row = fid_to_stage1.get(fid)
        if s1_row is not None:
            gate_bbox = s1_row.get("gate_bbox_hwd")
            if gate_bbox is not None:
                for entry in s1_row["verified"]:
                    paste_coarse_mask(coarse_full, entry, gate_bbox)

        # ── No-ROI fallback ──
        if len(rlist) == 0:
            if not args.no_gt:
                bucket_overall["tp"] += 0
                bucket_overall["pred"] += 0
                bucket_overall["gt"] += gt_vox
                bucket_overall["n"] += 1
                bucket_by_cat[cat]["tp"] += 0
                bucket_by_cat[cat]["pred"] += 0
                bucket_by_cat[cat]["gt"] += gt_vox
                bucket_by_cat[cat]["n"] += 1

            empty_pred = np.zeros(ref_shape, dtype=np.uint8)
            pred_path = mask_out_dir / f"{fid}_pred.nii.gz"
            nib.save(nib.Nifti1Image(empty_pred, ref_affine), str(pred_path))

            per_finding.append({
                "finding_id": fid,
                "category": cat,
                "n_rois": 0,
                "s2_raw_dice": 0.0,
                "s2_raw_recall": 0.0,
                "s2_raw_precision": 0.0,
                "s2_raw_pred_voxels": 0,
                "s2_raw_tp": 0,
                "gt_voxels": gt_vox,
                "s2_pred_path": str(pred_path),
                "s1_s2_intersection_voxels": 0,
                "s2_overlap_with_s1": 0.0,
                "s1_covered_by_s2": 0.0,
                "s1_s2_dice_proxy": 0.0,
                "s2_outside_s1_fraction": 0.0,
                "s1_outside_s2_fraction": 0.0,
                "s2_conf_mean_prob": 0.0, "s2_conf_std_prob": 0.0,
                "s2_conf_p10_prob": 0.0, "s2_conf_p90_prob": 0.0,
                "s2_conf_high_conf_frac_09": 0.0, "s2_conf_low_conf_frac_05_07": 0.0,
                "s2_conf_soft_hard_ratio": 0.0,
                "s2_conf_mean_margin": 0.0, "s2_conf_p10_margin": 0.0,
                "s2_conf_mean_entropy": 0.0,
                "s2_roi_conf_spread": 0.0, "s2_n_low_conf_rois": 0,
            })
            continue

        # ── Union all ROI predictions on full CT ──
        pred_full = np.zeros(ref_shape, dtype=bool)
        roi_conf_list = []  # per-ROI confidence stats

        for r in rlist:
            bbox = [int(v) for v in r["orig_bbox_hwd"]]
            h0, h1, w0, w1, d0, d1 = bbox

            img_path = out_dir / r["roi_image_path"]
            if not img_path.exists():
                print(f"  [SKIP] {r['component_sample_id']}: image missing: {img_path}")
                continue

            img_roi = np.asanyarray(nib.load(str(img_path)).dataobj)
            if img_roi.ndim == 4:
                img_roi = img_roi[..., 0]
            img_roi = img_roi.astype(np.float32)

            # Inference (sliding-window or full-roi)
            orig_shape = img_roi.shape
            group = get_group(tuple(orig_shape))
            ph, pw, pd = gps[group]
            target = (
                max(int(np.ceil(orig_shape[0] / 32) * 32), ph),
                max(int(np.ceil(orig_shape[1] / 32) * 32), pw),
                max(int(np.ceil(orig_shape[2] / 32) * 32), pd),
            )
            img_pad, (h0_pad, w0_pad, d0_pad), (h_off, w_off, d_off) = \
                pad_to_shape_centered(img_roi, target, cval=-1024.0)
            img_norm = prepare_image(img_pad)
            img_t = torch.from_numpy(img_norm.transpose(2, 1, 0).copy()).float()
            img_t = img_t.unsqueeze(0).unsqueeze(0)  # (1, 1, D, W, H)
            _, _, dd, ww, hh = img_t.shape

            used_sliding = False
            try:
                with torch.no_grad():
                    with torch.autocast(device_type=device.type,
                                        dtype=torch.float16,
                                        enabled=(device.type == "cuda")):
                        force_sliding_large_roi = (
                            args.mode == "full-roi"
                            and force_sliding_max_voxels > 0
                            and int(np.prod(orig_shape)) > force_sliding_max_voxels
                        )
                        if force_sliding_large_roi:
                            print(f"  [force sliding] {r['component_sample_id']}: shape={orig_shape},voxels={int(np.prod(orig_shape))} > {force_sliding_max_voxels}")
                        if args.mode == "full-roi" and not force_sliding_large_roi:
                            # Full-ROI: process entire padded image at once
                            logits = model(img_t.to(device))  # (1, 2, D, W, H)
                            n_windows = 1
                        else:
                            # Sliding-window: average logits from overlapping patches
                            stride_d = max(1, int(round(pd * (1.0 - args.overlap))))
                            stride_w = max(1, int(round(pw * (1.0 - args.overlap))))
                            stride_h = max(1, int(round(ph * (1.0 - args.overlap))))
                            d_starts = window_starts(dd, pd, stride_d)
                            w_starts = window_starts(ww, pw, stride_w)
                            h_starts = window_starts(hh, ph, stride_h)

                            logits_sum = torch.zeros((1, 2, dd, ww, hh), dtype=torch.float32, device=device)
                            counts = torch.zeros((1, 1, dd, ww, hh), dtype=torch.float32, device=device)
                            used_sliding = True

                            for ds in d_starts:
                                for ws in w_starts:
                                    for hs in h_starts:
                                        patch = img_t[:, :, ds:ds + pd, ws:ws + pw, hs:hs + ph]
                                        patch = patch.to(device)
                                        patch_logits = model(patch)
                                        logits_sum[:, :, ds:ds + pd, ws:ws + pw, hs:hs + ph] += patch_logits.float()
                                        counts[:, :, ds:ds + pd, ws:ws + pw, hs:hs + ph] += 1.0

                            logits = logits_sum / counts.clamp_min(1.0)
                            n_windows = len(d_starts) * len(w_starts) * len(h_starts)
            except (torch.OutOfMemoryError, RuntimeError) as exc:
                if args.mode == "full-roi" and isinstance(exc, RuntimeError) and "OOM" not in str(exc):
                    # Full-roi failed (likely input too large for model) → fallback to sliding-window
                    print(f"  [full-roi FAIL] {r['component_sample_id']}: shape={orig_shape} → sliding-window fallback")
                    # Re-run with sliding-window
                    stride_d = max(1, int(round(pd * (1.0 - args.overlap))))
                    stride_w = max(1, int(round(pw * (1.0 - args.overlap))))
                    stride_h = max(1, int(round(ph * (1.0 - args.overlap))))
                    d_starts = window_starts(dd, pd, stride_d)
                    w_starts = window_starts(ww, pw, stride_w)
                    h_starts = window_starts(hh, ph, stride_h)
                    logits_sum = torch.zeros((1, 2, dd, ww, hh), dtype=torch.float32, device=device)
                    counts = torch.zeros((1, 1, dd, ww, hh), dtype=torch.float32, device=device)
                    try:
                        with torch.no_grad():
                            for ds in d_starts:
                                for ws in w_starts:
                                    for hs in h_starts:
                                        patch = img_t[:, :, ds:ds + pd, ws:ws + pw, hs:hs + ph]
                                        patch = patch.to(device)
                                        with torch.autocast(device_type=device.type,
                                                            dtype=torch.float16, enabled=(device.type == "cuda")):
                                            patch_logits = model(patch)
                                        logits_sum[:, :, ds:ds + pd, ws:ws + pw, hs:hs + ph] += patch_logits.float()
                                        counts[:, :, ds:ds + pd, ws:ws + pw, hs:hs + ph] += 1.0
                        logits = logits_sum / counts.clamp_min(1.0)
                        used_sliding = True  # override for del cleanup
                    except (torch.OutOfMemoryError, RuntimeError):
                        print(f"  [FALLBACK OOM SKIP] {r['component_sample_id']}: shape={orig_shape}")
                        del img_t, logits_sum, counts
                        torch.cuda.empty_cache()
                        continue
                else:
                    print(f"  [OOM SKIP] {r['component_sample_id']}: shape={orig_shape} group={group} "
                          f"mode={args.mode}")
                    torch.cuda.empty_cache()
                    continue

            if used_sliding:
                logits = logits_sum / counts.clamp_min(1.0)
            pred_pad = torch.argmax(logits, dim=1).cpu().numpy()[0]  # (D, W, H)
            # ── Confidence features: binary softmax prob, margin, entropy ──
            # For two classes, P(fg) = sigmoid(logit_fg - logit_bg). This avoids
            # materializing a full 2-channel softmax tensor for large ROIs.
            margin_pad_t = logits[0, 1] - logits[0, 0]
            prob_fg_pad = torch.sigmoid(margin_pad_t).cpu().numpy().astype(np.float32)  # (D, W, H)
            margin_pad = margin_pad_t.cpu().numpy().astype(np.float32)
            # Crop back to original ROI shape
            pred_roi = pred_pad[d_off:d_off + d0_pad,
                                w_off:w_off + w0_pad,
                                h_off:h_off + h0_pad]
            pred_roi = pred_roi.transpose(2, 1, 0)  # DWH → HWD
            pred_roi = pred_roi[:orig_shape[0], :orig_shape[1], :orig_shape[2]].astype(bool)
            # Crop prob/margin same way
            prob_fg_roi = prob_fg_pad[d_off:d_off + d0_pad,
                                      w_off:w_off + w0_pad,
                                      h_off:h_off + h0_pad].transpose(2, 1, 0)
            prob_fg_roi = prob_fg_roi[:orig_shape[0], :orig_shape[1], :orig_shape[2]]
            margin_roi = margin_pad[d_off:d_off + d0_pad,
                                    w_off:w_off + w0_pad,
                                    h_off:h_off + h0_pad].transpose(2, 1, 0)
            margin_roi = margin_roi[:orig_shape[0], :orig_shape[1], :orig_shape[2]]
            # Per-ROI confidence stats
            roi_pred_vox = int(pred_roi.sum())
            if roi_pred_vox > 0:
                fg_prob = prob_fg_roi[pred_roi]
                fg_margin = margin_roi[pred_roi]
                # entropy: -sum(p*log(p)), per-voxel from prob_fg
                eps = 1e-8
                bg_prob = 1.0 - fg_prob
                fg_entropy = -(fg_prob * np.log(fg_prob + eps) + bg_prob * np.log(bg_prob + eps))
                roi_conf = {
                    "roi_pred_vox": roi_pred_vox,
                    "mean_prob": float(fg_prob.mean()),
                    "std_prob": float(fg_prob.std()),
                    "p10_prob": float(np.percentile(fg_prob, 10)),
                    "p90_prob": float(np.percentile(fg_prob, 90)),
                    "high_conf_frac_09": float((fg_prob > 0.9).mean()),
                    "low_conf_frac_05_07": float(((fg_prob > 0.5) & (fg_prob < 0.7)).mean()),
                    "soft_hard_ratio": float(fg_prob.sum() / roi_pred_vox),
                    "mean_margin": float(fg_margin.mean()),
                    "p10_margin": float(np.percentile(fg_margin, 10)),
                    "mean_entropy": float(fg_entropy.mean()),
                }
            else:
                roi_conf = {
                    "roi_pred_vox": 0,
                    "mean_prob": 0.0, "std_prob": 0.0, "p10_prob": 0.0, "p90_prob": 0.0,
                    "high_conf_frac_09": 0.0, "low_conf_frac_05_07": 0.0,
                    "soft_hard_ratio": 0.0,
                    "mean_margin": 0.0, "p10_margin": 0.0, "mean_entropy": 0.0,
                }
            roi_conf_list.append(roi_conf)
            del img_t, logits, pred_pad
            if used_sliding:
                del logits_sum, counts
            del margin_pad_t, prob_fg_pad, margin_pad
            del prob_fg_roi, margin_roi
            # empty_cache moved to per-finding level for efficiency

            # Paste into full CT
            h0c, h1c = max(0, h0), min(pred_full.shape[0], h1)
            w0c, w1c = max(0, w0), min(pred_full.shape[1], w1)
            d0c, d1c = max(0, d0), min(pred_full.shape[2], d1)
            if h0c < h1c and w0c < w1c and d0c < d1c:
                crop_h, crop_w, crop_d = h1c - h0c, w1c - w0c, d1c - d0c
                pred_full[h0c:h1c, w0c:w1c, d0c:d1c] |= \
                    pred_roi[:crop_h, :crop_w, :crop_d]

        # Free GPU memory per finding
        torch.cuda.empty_cache()

        # ── Aggregate per-ROI confidence to finding level ──
        conf_feats = {}
        if roi_conf_list:
            weights = np.array([x["roi_pred_vox"] for x in roi_conf_list], dtype=np.float64)
            wt = weights.sum()
            if wt > 0:
                for key in ["mean_prob", "std_prob", "p10_prob", "p90_prob",
                             "high_conf_frac_09", "low_conf_frac_05_07",
                             "soft_hard_ratio", "mean_margin", "p10_margin", "mean_entropy"]:
                    conf_feats["s2_conf_" + key] = float(
                        sum(x[key] * w for x, w in zip(roi_conf_list, weights)) / wt)
            else:
                for key in ["mean_prob", "std_prob", "p10_prob", "p90_prob",
                             "high_conf_frac_09", "low_conf_frac_05_07",
                             "soft_hard_ratio", "mean_margin", "p10_margin", "mean_entropy"]:
                    conf_feats["s2_conf_" + key] = 0.0
            # ROI-level spread
            roi_means = [x["mean_prob"] for x in roi_conf_list if x["roi_pred_vox"] > 0]
            conf_feats["s2_roi_conf_spread"] = float(np.std(roi_means)) if len(roi_means) > 1 else 0.0
            conf_feats["s2_n_low_conf_rois"] = sum(
                1 for x in roi_conf_list if x["roi_pred_vox"] > 0 and x["mean_prob"] < 0.7)
        else:
            for key in ["mean_prob", "std_prob", "p10_prob", "p90_prob",
                         "high_conf_frac_09", "low_conf_frac_05_07",
                         "soft_hard_ratio", "mean_margin", "p10_margin", "mean_entropy"]:
                conf_feats["s2_conf_" + key] = 0.0
            conf_feats["s2_roi_conf_spread"] = 0.0
            conf_feats["s2_n_low_conf_rois"] = 0

        s2_pred = int(pred_full.sum())
        if args.no_gt:
            s2_tp = 0
            s2_dice = 0.0
            s2_rec = 0.0
            s2_prec = 0.0
        else:
            # ── Raw S2 metrics ──
            s2_tp = int((pred_full & gt).sum())
            s2_dice = compute_dice(s2_tp, s2_pred, gt_vox)
            s2_rec = s2_tp / max(1, gt_vox)
            s2_prec = s2_tp / max(1, s2_pred)

            bucket_overall["tp"] += s2_tp
            bucket_overall["pred"] += s2_pred
            bucket_overall["gt"] += gt_vox
            bucket_overall["n"] += 1
            bucket_by_cat[cat]["tp"] += s2_tp
            bucket_by_cat[cat]["pred"] += s2_pred
            bucket_by_cat[cat]["gt"] += gt_vox
            bucket_by_cat[cat]["n"] += 1

        # ── S1-S2 overlap features (for gate training, no GT needed) ──
        s1_mask = coarse_full > 0
        s2_mask = pred_full > 0
        s1_v = int(s1_mask.sum())
        s2_v = int(s2_mask.sum())
        inter = int(np.logical_and(s1_mask, s2_mask).sum())
        s2_overlap = inter / s2_v if s2_v > 0 else 0.0
        s1_covered = inter / s1_v if s1_v > 0 else 0.0

        per_finding.append({
            "finding_id": fid,
            "category": cat,
            "n_rois": len(rlist),
            "s2_raw_dice": round(s2_dice, 4),
            "s2_raw_recall": round(s2_rec, 4),
            "s2_raw_precision": round(s2_prec, 4),
            "s2_raw_pred_voxels": s2_pred,
            "s2_raw_tp": s2_tp,
            "gt_voxels": gt_vox,
            "s2_pred_path": str(mask_out_dir / f"{fid}_pred.nii.gz"),
            "s1_s2_intersection_voxels": inter,
            "s2_overlap_with_s1": round(s2_overlap, 6),
            "s1_covered_by_s2": round(s1_covered, 6),
            "s1_s2_dice_proxy": round((2 * inter) / (s1_v + s2_v) if (s1_v + s2_v) > 0 else 0.0, 6),
            "s2_outside_s1_fraction": round(1.0 - s2_overlap if s2_v > 0 else 0.0, 6),
            "s1_outside_s2_fraction": round(1.0 - s1_covered if s1_v > 0 else 0.0, 6),
        })
        # Add confidence features
        per_finding[-1].update({k: round(v, 6) if isinstance(v, float) else v
                               for k, v in conf_feats.items()})

        # Save S2 raw prediction mask (same affine as GT in eval mode, CT in inference mode)
        nib.save(nib.Nifti1Image(pred_full.astype(np.uint8), ref_affine),
                 str(mask_out_dir / f"{fid}_pred.nii.gz"))

        if (fi + 1) % 5 == 0:
            elapsed = time.time() - t0
            print(f"  [{fi+1}/{len(finding_ids)}] findings done ({elapsed:.0f}s)",
                  flush=True)

    # ── Aggregate ──
    def micro_metrics(b):
        tp, pred, gt = b["tp"], b["pred"], b["gt"]
        return {
            "n": b["n"],
            "dice": round(compute_dice(tp, pred, gt), 4),
            "recall": round(tp / max(1, gt), 4),
            "precision": round(tp / max(1, pred), 4),
            "tp": tp,
            "pred_voxels": pred,
            "gt_voxels": gt,
        }

    if args.no_gt:
        s2_overall = {
            "n": len(per_finding),
            "pred_voxels": int(sum(r.get("s2_raw_pred_voxels", 0) for r in per_finding)),
            "gt_available": False,
        }
        s2_by_cat = {}
    else:
        s2_overall = micro_metrics(bucket_overall)
        s2_by_cat = {cat: micro_metrics(b) for cat, b in sorted(bucket_by_cat.items())}

    result = {
        "overall": s2_overall,
        "by_category": s2_by_cat,
        "per_finding": per_finding,
        "gt_available": not args.no_gt,
        "total_time_s": round(time.time() - t0, 1),
    }

    out_json = out_dir / "eval_finding_level.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # ── Print ──
    print(f"\n{'=' * 70}")
    if args.no_gt:
        print("Stage2 Inference Results (no GT)")
        print(f"{'=' * 70}")
        print(f"Findings: {s2_overall['n']}, pred_voxels: {s2_overall['pred_voxels']}")
    else:
        print("Stage2 Finding-Level Results (S2 raw, micro-averaged)")
        print(f"{'=' * 70}")
        print(f"\n{'cat':>6}  {'n':>4}  {'Dice':>8}  {'Recall':>8}  {'Precision':>8}")
        print("-" * 48)
        for cat in sorted(s2_by_cat.keys()):
            r = s2_by_cat[cat]
            print(f"{cat:>6}  {r['n']:>4}  "
                  f"{r['dice']:>8.4f}  {r['recall']:>8.4f}  {r['precision']:>8.4f}")
        print(f"{'OVERALL':>6}  {s2_overall['n']:>4}  "
              f"{s2_overall['dice']:>8.4f}  {s2_overall['recall']:>8.4f}  {s2_overall['precision']:>8.4f}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s → {out_json}")
    print(f"Pred masks → {mask_out_dir}/")


if __name__ == "__main__":
    main()
