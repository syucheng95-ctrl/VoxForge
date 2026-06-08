"""End-to-end pipeline orchestrator.

Chains all stages sequentially:
  Stage0      → Router + Anatomy gate + CT cropping
  Stage0.5    → MoE proposal generation
  Stage1      → VoxTell baseline verification
  Stage1_metrics → S1 metrics evaluation
  Stage2_roi  → STU-Net ROI generation
  Stage2_eval → STU-Net finding-level eval (with overlap + confidence features)
  gate_table  → Build gate training table
  gate_apply  → Apply learned gate, output final decisions
  export_final → Export final inference masks

Usage:
  python run_full.py                          # default: default manifest
  python run_full.py --config config.yaml     # custom config
  python run_full.py --manifest path/to.jsonl # custom manifest
  python run_full.py --stages 0,0.5           # run only specific stages
  python run_full.py --inference-only         # no-GT inference path
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("STAGE2_FORCE_SLIDING_MAX_VOXELS", "30000000")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from src.utils import load_config, resolve


def main() -> None:
    parser = argparse.ArgumentParser(description="Full pipeline orchestrator")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--manifest", help="Manifest JSONL (default: config manifests.default)")
    parser.add_argument(
        "--stages", default=None,
        help="Comma-separated stages to run",
    )
    parser.add_argument("--inference-only", action="store_true",
                        help="Run the no-GT inference path and export final masks")
    parser.add_argument("--limit-cases", type=int, default=0, help="Limit N CT cases")
    parser.add_argument("--stage2-mode", choices=["sliding-window", "full-roi"], default="sliding-window",
                       help="Stage2 inference mode")
    parser.add_argument("--no-clean", action="store_true", help="Keep intermediate NIfTI files (default: auto-clean)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-processed findings in Stage 1/2")
    args = parser.parse_args()

    config = load_config(args.config)
    python = config["python"]
    upload_dir = Path(config["_upload_dir"])
    outputs_root = Path(resolve(config, "outputs.root"))
    config_path = Path(config["_upload_dir"]) / args.config if not Path(args.config).is_absolute() else Path(args.config)
    if args.stages is None:
        if args.inference_only:
            stages_arg = "0,0.5,1,s1_nosafety,s2_c,2_roi,2_eval,gate_table,gate_apply,export_final"
        else:
            stages_arg = "0,0.5,1,s1_nosafety,s2_c,1_metrics,2_roi,2_eval,gate_table,gate_apply,eval_final"
    else:
        stages_arg = args.stages
    stages = [s.strip() for s in stages_arg.split(",")]

    manifest = args.manifest or resolve(config, "manifests.default")

    total_t0 = time.time()
    child_env = os.environ.copy()

    stage_scripts = {
        "0":           Path(config["_upload_dir"]) / "pipeline" / "stage0_crop.py",
        "0.5":         Path(config["_upload_dir"]) / "pipeline" / "stage0_5_proposals.py",
        "1":            Path(config["_upload_dir"]) / "pipeline" / "stage1_verify.py",
        "1_metrics":    Path(config["_upload_dir"]) / "pipeline" / "stage1_metrics.py",
        "2_roi":       Path(config["_upload_dir"]) / "pipeline" / "stage2_rois.py",
        "2_eval":      Path(config["_upload_dir"]) / "pipeline" / "stage2_eval.py",
        "gate_table":  Path(config["_upload_dir"]) / "pipeline" / "build_gate_table.py",
        "gate_apply":         Path(config["_upload_dir"]) / "pipeline" / "apply_gate.py",
        "eval_final":        Path(config["_upload_dir"]) / "pipeline" / "eval_final.py",
        "export_final":      Path(config["_upload_dir"]) / "pipeline" / "export_final.py",
        "s1_nosafety":    Path(config["_upload_dir"]) / "pipeline" / "build_s1_vp.py",
        "s2_c":           Path(config["_upload_dir"]) / "pipeline" / "build_s2_vp.py",
    }

    for stage in stages:
        if stage not in stage_scripts:
            print(f"Unknown stage: {stage}. Choose from: {sorted(stage_scripts.keys())}")
            sys.exit(1)

        script = stage_scripts[stage]

        cmd = [
            python, str(script),
            "--config", str(config_path),
        ]

        # Per-stage extra args
        if stage == "0":
            cmd += ["--manifest", manifest]
            if args.limit_cases:
                cmd += ["--limit-cases", str(args.limit_cases)]
        elif stage == "0.5":
            pass  # Stage0 has already limited cases; process all findings it produced.
        elif stage == "1":
            if args.resume:
                cmd += ["--resume"]
        elif stage == "1_metrics":
            cmd += ["--verified-proposals", "verified_proposals_s1_nosafety.jsonl"]
        elif stage == "2_roi":
            cmd += ["--manifest", manifest,
                    "--verified-proposals", "verified_proposals_s2_c.jsonl"]
            if args.inference_only:
                cmd += ["--no-labels"]
            if args.limit_cases:
                cmd += ["--limit-cases", str(args.limit_cases)]
        elif stage == "2_eval":
            cmd += ["--manifest", manifest, "--mode", args.stage2_mode,
                    "--verified-proposals", "verified_proposals_s2_c.jsonl"]
            if args.inference_only:
                cmd += ["--no-gt"]
        elif stage == "gate_table":
            cmd += ["--manifest", manifest,
                    "--verified-proposals", "verified_proposals_s1_nosafety.jsonl"]
            if args.limit_cases:
                cmd += ["--finding-ids-from-vp", "verified_proposals_s1_nosafety.jsonl"]
        elif stage == "s1_nosafety":
            pass  # uses --config to resolve paths
        elif stage == "s2_c":
            pass  # uses --config to resolve paths
        elif stage == "export_final":
            cmd += ["--manifest", manifest,
                    "--verified-proposals", "verified_proposals_s1_nosafety.jsonl"]

        print(f"\n{'#' * 60}")
        print(f"# STAGE {stage}")
        print(f"{'#' * 60}")
        t0 = time.time()
        result = subprocess.run(cmd, env=child_env)
        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"\nPipeline FAILED at Stage {stage} (exit {result.returncode})")
            sys.exit(result.returncode)

        print(f"\nStage {stage} completed in {elapsed:.0f}s")

        # ── Auto-clean intermediate NIfTI (keep JSON/JSONL/CSV + final preds) ──
        if not args.no_clean and not args.inference_only:
            import shutil

            cleanup_map = {
                "1": [outputs_root / "stage0"],            # Stage1 has consumed crop NIfTI
                "2_eval": [outputs_root / "stage1" / "masks"],  # coarse masks are needed by Stage2 eval
                "gate_table": [
                    outputs_root / "stage0",               # final cleanup
                    outputs_root / "stage0_5",
                    outputs_root / "stage1",
                    outputs_root / "stage2" / "roi_images",
                    outputs_root / "stage2" / "roi_masks",
                ],
            }

            WIPE_DIRS = {
                outputs_root / "stage2" / "roi_images",
                outputs_root / "stage2" / "roi_masks",
            }

            for target_dir in cleanup_map.get(stage, []):
                if target_dir.exists() and target_dir.is_dir():
                    if target_dir in WIPE_DIRS:
                        # These contain only flat .nii.gz files — rmtree the whole dir
                        try:
                            shutil.rmtree(target_dir)
                            print(f"  [clean] Removed: {target_dir.relative_to(upload_dir)}")
                        except Exception as e:
                            print(f"  [clean] Failed to remove {target_dir.relative_to(upload_dir)}: {e}")
                    else:
                        # Only delete subdirectories (NIfTI), keep JSON/JSONL/CSV at root
                        for item in list(target_dir.iterdir()):
                            if item.is_dir():
                                try:
                                    shutil.rmtree(item)
                                    print(f"  [clean] Removed: {item.relative_to(upload_dir)}")
                                except Exception as e:
                                    print(f"  [clean] Failed to remove {item.relative_to(upload_dir)}: {e}")

    total_elapsed = time.time() - total_t0
    print(f"\n{'=' * 60}")
    print(f"  Pipeline complete! All stages passed in {total_elapsed:.0f}s")
    print(f"{'=' * 60}")

    # Collect summaries
    summaries = {}
    stage_dirs = {
        "0": "stage0", "0.5": "stage0_5", "1": "stage1",
        "1_metrics": "stage1",
        "s1_nosafety": "stage1", "s2_c": "stage1",
        "2_roi": "stage2",
        "gate_table": "gate",
        "eval_final": "final", "export_final": "export_final",
    }
    for s in stages:
        name = stage_dirs.get(s, s)
        if s == "2_eval":
            summary_path = Path(resolve(config, "outputs.stage2")) / "eval_finding_level.json"
        elif s == "gate_apply":
            summary_path = Path(resolve(config, "outputs.final")) / "gate_decisions_raw.json"
            if summary_path.exists():
                with open(summary_path, encoding="utf-8") as f:
                    gate_data = json.load(f)
                decisions = gate_data.get("decisions", [])
                summaries[f"stage_{s}"] = {
                    "threshold": gate_data.get("threshold"),
                    "n_total": len(decisions),
                    "n_use_s2": sum(int(d.get("use_s2", 0)) for d in decisions),
                }
            continue
        elif name == "final":
            summary_path = Path(resolve(config, "outputs.final")) / "final_metrics.json"
        elif name == "export_final":
            summary_path = Path(resolve(config, "outputs.final")) / "export_summary.json"
        elif name == "gate":
            summary_path = outputs_root / "gate_feature_table.csv"
            if summary_path.exists():
                summaries[f"stage_{s}"] = {"table_path": str(summary_path), "status": "ok"}
            continue
        else:
            summary_path = Path(resolve(config, f"outputs.{name}")) / "summary.json"
        if summary_path.exists():
            with open(summary_path, encoding="utf-8") as f:
                summaries[f"stage_{s}"] = json.load(f)

    summary_file = outputs_root / "pipeline_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    pipeline_summary = {
        "manifest": manifest,
        "stages_run": stages,
        "total_time_s": round(total_elapsed, 1),
        "stage_summaries": summaries,
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(pipeline_summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary: {summary_file}")


if __name__ == "__main__":
    main()
