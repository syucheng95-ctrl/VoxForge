import argparse
import gc
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from .input_parser import parse_inputs
from .hu_expert import run_hu_expert
from .nodule_detector_expert import run_nodule_detector_expert
from .diffuse_expert import propose_gate
from .coordinate_mapper import crop_to_original
from .proposal_fusion import fuse_proposals


def load_nifti_cache(path: Path, cache: dict) -> nib.Nifti1Image:
    key = str(path)
    if key not in cache:
        cache[key] = nib.load(str(path))
    return cache[key]


def get_zooms(cropped_nii: nib.Nifti1Image) -> tuple:
    return tuple(float(z) for z in cropped_nii.header.get_zooms()[:3])


def process_finding(finding: dict, nifti_cache: dict, nodule_bundle_dir: str) -> dict:
    """Multi-expert exploration: run nodule + HU + diffuse on every finding.

    Each expert produces proposals independently with source_expert tag.
    Analysis phase will determine which experts matter for which categories.
    """
    fid = finding["finding_id"]
    category = finding["category"]
    gate_bbox = finding["gate_bbox_hwd"]
    crop_path = finding["cropped_roi_path"]
    source_path = finding["source_image_path"]

    cropped_nii = load_nifti_cache(crop_path, nifti_cache)
    zooms = get_zooms(cropped_nii)
    source_nii = load_nifti_cache(source_path, nifti_cache)
    original_shape = tuple(source_nii.shape)
    cropped_hu = np.asanyarray(cropped_nii.dataobj).astype(np.float32)

    expert_details = {}
    all_proposals = []
    experts_order = ["nodule_detector", "hu", "diffuse"]

    # ── 1. Nodule Detector ──
    nodule_bboxes = None
    nodule_scores = None
    nodule_fallback = False
    nodule_reason = None

    # Skip nodule detector entirely if it previously broke the CUDA context
    if not getattr(process_finding, "_nodule_broken", False):
        try:
            detections = run_nodule_detector_expert(cropped_nii, category, nodule_bundle_dir, cache_key=str(crop_path))
        except RuntimeError as exc:
            err_msg = str(exc).lower()
            if "out of memory" in err_msg or "invalid resource handle" in err_msg:
                print(f"[run_stage0_5] WARN {fid}: nodule detector CUDA error, disabling for rest of run")
                detections = None
                nodule_fallback = True
                nodule_reason = f"nodule_detector_cuda_error:{type(exc).__name__}"
                process_finding._nodule_broken = True
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                raise
    else:
        nodule_fallback = True
        nodule_reason = "nodule_skipped_after_cuda_error"

    if detections:
        nodule_bboxes = [crop_to_original(d["bbox"], gate_bbox) for d in detections]
        nodule_scores = [d["score"] for d in detections]

    if nodule_bboxes is None:
        nodule_fallback = True
        nodule_reason = nodule_reason or "no_detections"
        nodule_props = []
    else:
        nodule_props = fuse_proposals(
            nodule_bboxes, gate_bbox, original_shape, zooms,
            source_expert="nodule_detector", cropped_hu=cropped_hu,
            bbox_scores=nodule_scores,
        )
    expert_details["nodule_detector"] = {
        "fallback": nodule_fallback,
        "fallback_reason": nodule_reason,
        "n_raw": len(nodule_props),
    }
    all_proposals.extend(nodule_props)

    # ── 2. HU Expert ──
    hu_bboxes_crop = run_hu_expert(cropped_nii, category)
    hu_fallback = False
    hu_reason = None
    if hu_bboxes_crop:
        hu_bboxes = [crop_to_original(b, gate_bbox) for b in hu_bboxes_crop]
        hu_props = fuse_proposals(
            hu_bboxes, gate_bbox, original_shape, zooms,
            source_expert="hu", cropped_hu=cropped_hu,
            bbox_scores=None,
        )
    else:
        hu_fallback = True
        hu_reason = "no_hu_components"
        hu_props = []
    expert_details["hu"] = {
        "fallback": hu_fallback,
        "fallback_reason": hu_reason,
        "n_raw": len(hu_props),
    }
    all_proposals.extend(hu_props)

    # ── 3. Diffuse Gate ──
    diffuse_bboxes = propose_gate(gate_bbox)
    diffuse_props = fuse_proposals(
        diffuse_bboxes, gate_bbox, original_shape, zooms,
        source_expert="diffuse", cropped_hu=cropped_hu,
        bbox_scores=None,
    )
    expert_details["diffuse"] = {
        "fallback": False,
        "fallback_reason": None,
        "n_raw": len(diffuse_props),
    }
    all_proposals.extend(diffuse_props)

    # Assign proposal IDs
    for i, p in enumerate(all_proposals):
        p["proposal_id"] = f"{fid}_p{i:02d}"

    experts_used = [e for e in experts_order if expert_details[e]["n_raw"] > 0]

    return {
        "finding_id": fid,
        "prompt": finding["prompt"],
        "category": category,
        "anatomy_target": finding["anatomy_target"],
        "gate_bbox_hwd": gate_bbox,
        "expert": "+".join(experts_used) if experts_used else "none",
        "expert_details": expert_details,
        "n_proposals": len(all_proposals),
        "proposals": all_proposals,
    }


def main():
    parser = argparse.ArgumentParser(description="Stage0.5 MoE proposal generator")
    parser.add_argument("--predictions", required=True, help="stage0_router_predictions.jsonl")
    parser.add_argument("--crop-groups", required=True, help="stage0_crop_groups.jsonl")
    parser.add_argument("--out-dir", required=True, help="output directory")
    parser.add_argument("--limit", type=int, default=0, help="limit findings for smoke test")
    parser.add_argument("--nodule-detector", required=True, help="path to MONAI nodule detector bundle")
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    crop_path = Path(args.crop_groups)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    failed_file = out_dir / "failed_findings.jsonl"
    if failed_file.exists():
        failed_file.unlink()

    findings = parse_inputs(pred_path, crop_path)
    if args.limit > 0:
        findings = findings[:args.limit]
        print(f"[run_stage0_5] Limited to {args.limit} findings")

    nifti_cache = {}
    results = []
    failed = []

    for i, finding in enumerate(findings):
        try:
            result = process_finding(finding, nifti_cache, args.nodule_detector)
            results.append(result)
        except Exception as exc:
            import traceback
            err_msg = f"{type(exc).__name__}: {exc}"
            print(f"[run_stage0_5] ERROR {finding['finding_id']}: {err_msg}")
            failed.append({
                "finding_id": finding["finding_id"],
                "category": finding.get("category", "?"),
                "error": err_msg,
                "traceback": traceback.format_exc(),
            })
            continue
        if (i + 1) % 10 == 0:
            print(f"[run_stage0_5] {i + 1}/{len(findings)} findings done")

    output_file = out_dir / "stage0_5_proposals.jsonl"
    with output_file.open("w", encoding="utf-8") as f:
        for r in results:
            record = {k: v for k, v in r.items() if k != "prompt"}
            record["prompt"] = r["prompt"]
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    print(f"[run_stage0_5] Saved {len(results)} findings → {output_file}")

    if failed:
        with failed_file.open("w", encoding="utf-8") as f:
            for item in failed:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"[run_stage0_5] {len(failed)} failed findings → {failed_file}")

    # Summary (multi-expert mode)
    total_proposals = sum(r["n_proposals"] for r in results)

    # Per-expert stats
    expert_counts = {"nodule_detector": 0, "hu": 0, "diffuse": 0}
    expert_fallback = {"nodule_detector": 0, "hu": 0, "diffuse": 0}
    expert_props = {"nodule_detector": 0, "hu": 0, "diffuse": 0}
    for r in results:
        for exp_name in ("nodule_detector", "hu", "diffuse"):
            d = r.get("expert_details", {}).get(exp_name, {})
            if d:
                expert_counts[exp_name] += 1
                expert_props[exp_name] += d.get("n_raw", 0)
                if d.get("fallback"):
                    expert_fallback[exp_name] += 1

    print(f"[run_stage0_5] Summary: {len(results)} findings, "
          f"{total_proposals} total proposals ({total_proposals / max(1, len(results)):.1f} avg)")
    print(f"  nodule_detector: {expert_counts['nodule_detector']} findings, "
          f"{expert_props['nodule_detector']} proposals, "
          f"{expert_fallback['nodule_detector']} fallback")
    print(f"  hu:              {expert_counts['hu']} findings, "
          f"{expert_props['hu']} proposals, "
          f"{expert_fallback['hu']} fallback")
    print(f"  diffuse:         {expert_counts['diffuse']} findings, "
          f"{expert_props['diffuse']} proposals, "
          f"{expert_fallback['diffuse']} fallback")


if __name__ == "__main__":
    main()
