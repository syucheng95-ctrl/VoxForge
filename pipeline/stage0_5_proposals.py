"""Stage0.5 runner: MoE proposal generator on Stage0 pre-cropped ROIs.

Runs src.proposal_generator.run_stage0_5 as a module (needs relative imports).
"""

import argparse
import os
from pathlib import Path

from src.utils import load_config, resolve, run_step


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage0.5 proposal generator runner")
    parser.add_argument("--config", default="configs/pipeline.yaml", help="Path to pipeline config")
    parser.add_argument("--limit", type=int, default=0, help="Limit N findings (0=all)")
    args = parser.parse_args()

    config = load_config(args.config)
    python = config["python"]
    upload_dir = Path(config["_upload_dir"])
    out_dir = Path(resolve(config, "outputs.stage0_5"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stage0 outputs -> Stage0.5 inputs
    stage0_out = Path(resolve(config, "outputs.stage0"))
    predictions = str(stage0_out / "stage0_router_predictions.jsonl")
    crop_manifest = stage0_out / "stage0_crop_groups.jsonl"
    if not crop_manifest.exists():
        crop_groups_files = list(stage0_out.glob("**/stage0_crop_groups.jsonl"))
        if crop_groups_files:
            crop_groups = str(crop_groups_files[0])
        else:
            print("[ERROR] No crop_groups.jsonl found from Stage0")
            raise SystemExit(1)
    else:
        crop_groups = str(crop_manifest)

    nodule_bundle = resolve(config, "models.nodule_detector")

    cmd = [
        python, "-m", "src.proposal_generator.run_stage0_5",
        "--predictions", predictions,
        "--crop-groups", crop_groups,
        "--out-dir", str(out_dir),
        "--nodule-detector", nodule_bundle,
    ]
    if args.limit:
        cmd += ["--limit", str(args.limit)]

    run_step(cmd, "Stage0.5: Proposal Generation", cwd=str(upload_dir))


if __name__ == "__main__":
    main()
