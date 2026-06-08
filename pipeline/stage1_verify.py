"""Stage1 runner: Per-proposal VoxTell text-consistency verifier.

Reads Stage0's cropped ROIs + Stage0.5's proposals, runs VoxTell on each
cropped ROI with the finding text, and scores each proposal by activation
inside its bounding box. Outputs verified_proposals.jsonl + coarse masks.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from src.stage1.inference import get_nibabel_io_with_reorient
NibabelIOWithReorient = get_nibabel_io_with_reorient()

from src.utils import load_config, load_jsonl, resolve, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage1 VoxTell verifier runner")
    parser.add_argument("--config", default="configs/pipeline.yaml", help="Path to pipeline config")
    parser.add_argument("--limit-findings", type=int, default=0, help="Limit N findings (0=all)")
    parser.add_argument("--finding-ids", default="", help="Comma-separated finding IDs to run")
    parser.add_argument("--append-existing", action="store_true",
                        help="Merge results into an existing verified_proposals.jsonl")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-processed findings in verified_proposals.jsonl")
    args = parser.parse_args()

    config = load_config(args.config)
    upload_dir = Path(config["_upload_dir"])
    out_dir = Path(resolve(config, "outputs.stage1"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Resource check before loading VoxTell (~8 GB GPU + CPU buffers) ──
    import ctypes
    import ctypes.wintypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.wintypes.DWORD),
                    ("dwMemoryLoad", ctypes.wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    MIN_RAM_GB = 3.0
    MIN_GPU_GB = 9.0

    def _check_resources():
        mem = MEMORYSTATUSEX()
        mem.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
        free_ram = mem.ullAvailPhys / (1024 ** 3)
        total_ram = mem.ullTotalPhys / (1024 ** 3)
        gpu_free = 0.0
        if torch.cuda.is_available():
            gpu_free_mem, gpu_total_mem = torch.cuda.mem_get_info(0)
            gpu_free = gpu_free_mem / (1024 ** 3)
        return free_ram, total_ram, gpu_free

    free_ram, total_ram, gpu_free = _check_resources()
    print(f"[Resource] System RAM: {free_ram:.1f} GB free / {total_ram:.0f} GB total "
          f"(min {MIN_RAM_GB:.0f} GB)", flush=True)
    print(f"[Resource] GPU memory: {gpu_free:.1f} GB free (min {MIN_GPU_GB:.0f} GB)", flush=True)

    wait_count = 0
    while free_ram < MIN_RAM_GB or gpu_free < MIN_GPU_GB:
        reasons = []
        if free_ram < MIN_RAM_GB:
            reasons.append(f"RAM {free_ram:.1f} < {MIN_RAM_GB:.0f} GB")
        if gpu_free < MIN_GPU_GB:
            reasons.append(f"GPU {gpu_free:.1f} < {MIN_GPU_GB:.0f} GB")
        wait_count += 1
        print(f"[Resource] WAITING ({', '.join(reasons)}) — retry #{wait_count} "
              f"in 30s (Ctrl+C to abort)", flush=True)
        time.sleep(30)
        free_ram, total_ram, gpu_free = _check_resources()

    if wait_count > 0:
        print(f"[Resource] OK — resources freed after {wait_count} retries, proceeding", flush=True)

    # Inputs
    stage0_out = Path(resolve(config, "outputs.stage0"))
    stage0_5_out = Path(resolve(config, "outputs.stage0_5"))
    crop_groups_file = stage0_out / "stage0_crop_groups.jsonl"
    proposals_file = stage0_5_out / "stage0_5_proposals.jsonl"

    if not proposals_file.exists():
        print(f"[ERROR] Proposals not found: {proposals_file}")
        print("  Run Stage0.5 first.")
        raise SystemExit(1)

    # Load data
    crop_groups = load_jsonl(crop_groups_file)
    proposals = load_jsonl(proposals_file)

    # Build index: finding_id → crop info
    finding_to_crop: dict[str, dict] = {}
    for g in crop_groups:
        for fid in g.get("finding_ids", []):
            finding_to_crop[fid] = {
                "cropped_roi_path": g["image"],
                "gate_bbox_hwd": g["bbox_hwd"],
                "source_image": g.get("source_image", ""),
            }

    # Build finding index
    prop_by_finding: dict[str, dict] = {}
    for p in proposals:
        prop_by_finding[p["finding_id"]] = p

    # Filter to findings that have both crops and proposals
    findings = [(fid, prop_by_finding[fid])
                for fid in prop_by_finding
                if fid in finding_to_crop]
    print(f"Matched {len(findings)} findings with crops + proposals "
          f"({len(prop_by_finding) - len(findings)} missing crops)")

    # ── Checkpoint: skip already-processed findings (only with --resume) ──
    existing_out = out_dir / "verified_proposals.jsonl"
    if not args.resume and existing_out.exists():
        existing_out.unlink()
        print("[Checkpoint] Fresh run, removed previous verified_proposals.jsonl")
    processed_ids: set[str] = set()
    if args.resume and existing_out.exists():
        for row in load_jsonl(existing_out):
            processed_ids.add(row["finding_id"])
        if processed_ids:
            n_before = len(findings)
            findings = [(fid, row) for fid, row in findings if fid not in processed_ids]
            print(f"[Checkpoint] Skipping {n_before - len(findings)} already-processed "
                  f"findings, {len(findings)} remaining")

    if args.finding_ids:
        wanted = {fid.strip() for fid in args.finding_ids.split(",") if fid.strip()}
        findings = [(fid, row) for fid, row in findings if fid in wanted]
        missing = sorted(wanted - {fid for fid, _ in findings})
        if missing:
            print(f"[WARN] Requested finding IDs not found in crops + proposals: {missing}")
        print(f"Filtered to {len(findings)} requested findings")

    if args.limit_findings:
        findings = findings[:args.limit_findings]

    if not findings:
        print("[Checkpoint] All findings already processed, nothing to do")
        raise SystemExit(0)

    # Load verifier (once)
    from _verifier import Verifier
    verifier = Verifier(
        model_dir=resolve(config, "models.voxtell"),
        text_model=resolve(config, "models.qwen_embedding"),
        device=args.device,
        project_root=upload_dir,
    )

    verified_rows = []
    total_t0 = time.time()

    # ── Process findings (pipeline verifier) ──
    ct_images_dir = Path(resolve(config, "data.ct_images"))
    reader_fc = NibabelIOWithReorient()  # shared across FullCT VoxTell side path
    for i, (fid, prop_row) in enumerate(findings):
        crop_info = finding_to_crop[fid]
        roi_path = Path(crop_info["cropped_roi_path"])
        gate_bbox = crop_info["gate_bbox_hwd"]

        if not roi_path.exists():
            print(f"  [SKIP] {fid}: cropped ROI missing: {roi_path}")
            continue

        # Load with NibabelIOWithReorient to match VoxTell training orientation
        reader = NibabelIOWithReorient()
        raw_nii = nib.load(str(roi_path))
        raw_shape = raw_nii.shape  # (H, W, D) — needed for bbox transform
        reoriented, _ = reader.read_images([str(roi_path)])
        cropped_roi = reoriented[0].astype(np.float32)  # (D, W, H) in reader space
        prompt = prop_row["prompt"]
        mask_out_dir = out_dir / "masks" / fid

        result = verifier.verify_finding(
            cropped_roi=cropped_roi,
            prompt=prompt,
            proposals=prop_row["proposals"],
            gate_bbox_hwd=gate_bbox,
            raw_shape=raw_shape,
            mask_out_dir=mask_out_dir,
        )

        # ── FullCT VoxTell side path ──
        source_ct = crop_info.get("source_image", "")
        ct_full_path = Path(source_ct)
        if not ct_full_path.exists():
            ct_full_path = ct_images_dir / (prop_row.get("case_name", "") + ".nii.gz")
        if not ct_full_path.exists():
            ct_full_path = ct_images_dir / (prop_row.get("case_name", ""))

        fullct_proposal = None
        if ct_full_path.exists():
            torch.cuda.empty_cache()
            raw_nii_fc = nib.load(str(ct_full_path))
            raw_shape_fc = raw_nii_fc.shape[:3]
            reoriented_fc, _ = reader_fc.read_images([str(ct_full_path)])
            logits_fc = verifier.predictor.predict_single_image_logits(
                reoriented_fc.copy(), [prompt],
            )
            clipped_fc = np.clip(logits_fc[0], -50, 50)
            prob_fc = 1.0 / (1.0 + np.exp(-clipped_fc))
            binary_fc = (prob_fc > 0.3).astype(np.uint8)

            mask_path_fc = mask_out_dir / "full_ct_coarse.nii.gz"
            prob_path_fc = mask_out_dir / "full_ct_prob.nii.gz"
            nib.save(nib.Nifti1Image(binary_fc, np.eye(4)), str(mask_path_fc))
            nib.save(nib.Nifti1Image(prob_fc.astype(np.float32), np.eye(4)), str(prob_path_fc))

            full_bbox_fc = [0, int(raw_shape_fc[0]), 0, int(raw_shape_fc[1]), 0, int(raw_shape_fc[2])]
            fullct_proposal = {
                "proposal_id": f"{fid}_fullct",
                "source_expert": "full_ct_voxtell",
                "proposal_bbox_hwd": full_bbox_fc,
                "coarse_mask_path": str(mask_path_fc),
                "coarse_prob_path": str(prob_path_fc),
                "score": 1.0,
                "max_prob": float(prob_fc.max()),
                "mean_prob": float(prob_fc.mean()),
                "fg_ratio": float(binary_fc.mean()),
                "volume_voxels": int(binary_fc.sum()),
                "verified": True,
            }
            torch.cuda.empty_cache()

        normal_verified = result["verified"]
        if fullct_proposal is not None:
            normal_verified = normal_verified + [fullct_proposal]

        has_full_ct_voxtell = any(
            p.get("source_expert") == "full_ct_voxtell"
            for p in normal_verified
        )

        verified_rows.append({
            "finding_id": fid,
            "prompt": prompt,
            "category": prop_row["category"],
            "verified": normal_verified,
            "rejected": result["rejected"],
            "fallback": result["fallback"],
            "gate_bbox_hwd": gate_bbox,
            "has_full_ct_voxtell": has_full_ct_voxtell,
            "use_full_ct_voxtell": False,
        })

        # ── Incremental save for checkpoint/resume ──
        write_jsonl(existing_out, [verified_rows[-1]], append=True)

        torch.cuda.empty_cache()

        if (i + 1) % 5 == 0:
            elapsed = time.time() - total_t0
            print(f"  [{i+1}/{len(findings)}] {elapsed:.0f}s")

    # ── All results already saved incrementally to existing_out ──

    # Summary
    n_verified = sum(len(r["verified"]) for r in verified_rows)
    n_rejected = sum(len(r["rejected"]) for r in verified_rows)
    n_fallback = sum(1 for r in verified_rows if r["fallback"])
    n_masks = sum(1 for r in verified_rows
                  for v in r["verified"]
                  if "coarse_mask_path" in v)

    summary = {
        "n_findings": len(verified_rows),
        "n_verified_proposals": n_verified,
        "n_rejected_proposals": n_rejected,
        "n_fallback_findings": n_fallback,
        "n_coarse_masks": n_masks,
        "total_time_s": round(time.time() - total_t0, 1),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nStage1 done: {len(verified_rows)} findings, "
          f"{n_verified} verified, {n_rejected} rejected, "
          f"{n_fallback} fallback")
    print(f"Output: {existing_out}")


if __name__ == "__main__":
    main()
