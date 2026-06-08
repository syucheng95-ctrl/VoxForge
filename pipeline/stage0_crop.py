"""Stage0 batch runner: Router + Anatomy gate + crop per CT case.

Reads a manifest JSONL (default.jsonl style), groups findings by case_name,
calls run_stage0_policy_crop.py for each CT, and merges outputs into a
single crop_manifest.jsonl.
"""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import torch

from src.utils import load_config, load_jsonl, resolve, run_step, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage0 batch runner")
    parser.add_argument("--config", default="configs/pipeline.yaml", help="Path to pipeline config")
    parser.add_argument("--manifest", help="Manifest JSONL (default: config manifests.default)")
    parser.add_argument("--limit-cases", type=int, default=0, help="Limit N cases (0=all)")
    args = parser.parse_args()

    config = load_config(args.config)
    upload_dir = Path(config["_upload_dir"])
    ct_images_dir = Path(resolve(config, "data.ct_images"))
    out_dir = Path(resolve(config, "outputs.stage0"))
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.manifest or resolve(config, "manifests.default")
    rows = load_jsonl(manifest_path)
    print(f"Loaded {len(rows)} findings from manifest")

    # Group by case_name
    case_map: dict[str, list[dict]] = {}
    for row in rows:
        case_map.setdefault(row["case_name"], []).append(row)

    cases = sorted(case_map.items())
    if args.limit_cases:
        cases = cases[: args.limit_cases]

    crop_manifest_rows: list[dict] = []
    all_predictions: list[dict] = []
    tmp_dir = Path(tempfile.mkdtemp(prefix="stage0_tmp_"))

    # ── Pre-load Stage0 models once (in-process) ──
    from src.stage0.run_stage0_policy_crop import process_one_case, load_router, pick_device
    from src.stage0.predict_router import load_head
    from src.stage0.anatomy_expert import TotalSegLobeCache
    from src.stage0.evaluate_stage0_policy_recall import LungCache

    checkpoint = resolve(config, "models.stage0_router")

    # ── Check for precomputed Qwen embeddings ──
    embedding_cache = None
    emb_path = Path(resolve(config, "outputs.root")) / "embeddings" / "qwen_embeddings.pt"
    if emb_path.exists():
        print("Loading precomputed Qwen embeddings (skipping Qwen model load)...")
        data = torch.load(str(emb_path), map_location="cpu")
        embedding_cache = {fid: emb for fid, emb in zip(data["ids"], data["embeddings"])}
        print(f"  Loaded {len(embedding_cache)} embeddings, input_dim={data['input_dim']}")

    device = pick_device("auto")

    if embedding_cache is not None:
        # Cache mode: only load Router Head, skip Qwen entirely
        print(f"Loading Router Head only (cache mode, skipping Qwen 4B)...")
        head, router_ckpt = load_head(Path(checkpoint), device)
        head_cfg = router_ckpt["config"]
        tokenizer = None
        embedder = None
    else:
        # Normal mode: load Qwen + Router Head
        qwen_model_path = resolve(config, "models.qwen_embedding")
        os.environ["QWEN_MODEL_PATH"] = qwen_model_path
        print(f"Loading Qwen + Router models (once for all cases)...")
        print(f"  Qwen model: {qwen_model_path}")
        tokenizer, embedder, head, head_cfg = load_router(Path(checkpoint), device)

    lung_cache = LungCache()
    totalseg_cache = TotalSegLobeCache(
        fast=config["stage0"].get("totalseg_fast", True),
        device=config["stage0"]["totalseg_device"])

    base_kwargs = {
        "checkpoint": checkpoint,
        "crop_mode": config["stage0"]["crop_mode"],
        "spatial_mode": config["stage0"]["spatial_mode"],
        "mapping_mode": config["stage0"]["mapping_mode"],
        "totalseg_device": config["stage0"]["totalseg_device"],
        "totalseg_fast": config["stage0"].get("totalseg_fast", True),
        "conservative_margin": config["stage0"]["conservative_margin"],
        "moderate_margin": config["stage0"]["moderate_margin"],
        "aggressive_margin": config["stage0"]["aggressive_margin"],
    }
    model_bundle = {
        "device": device, "tokenizer": tokenizer, "embedder": embedder,
        "head": head, "head_cfg": head_cfg,
        "embedding_cache": embedding_cache,
        "lung_cache": lung_cache, "totalseg_cache": totalseg_cache,
    }

    for ci, (case_name, case_rows) in enumerate(cases, 1):
        # Find CT image
        ct_path = ct_images_dir / case_name
        if not ct_path.exists():
            ct_path = ct_images_dir.parent / case_rows[0].get("image", "")
            if not ct_path.exists():
                print(f"  [SKIP] {case_name}: CT not found")
                continue

        # Write temp prompts JSONL for this CT
        tmp_prompts = tmp_dir / f"{case_name}_prompts.jsonl"
        prompts_rows = [{"id": r["id"], "prompt": r["prompt"]} for r in case_rows]
        write_jsonl(tmp_prompts, prompts_rows)

        case_out = out_dir / case_name.replace(".nii.gz", "").replace(".nii", "")
        case_out.mkdir(parents=True, exist_ok=True)

        kwargs = dict(base_kwargs, image=str(ct_path),
                      prompts_jsonl=str(tmp_prompts), out_dir=str(case_out))

        t0_case = time.time()
        print(f"\nStage0 [{ci}/{len(cases)}] {case_name}", flush=True)
        preds, groups, _ = process_one_case(kwargs, model_bundle)
        print(f"  Stage0 [{ci}/{len(cases)}] done in {time.time() - t0_case:.0f}s")

        for p in preds:
            p["_case_out_dir"] = str(case_out)
            all_predictions.append(p)
        for g in groups:
            g["_case_out_dir"] = str(case_out)
            crop_manifest_rows.append(g)

        # Clean up temp prompts
        tmp_prompts.unlink(missing_ok=True)

    # Merge and save predictions + crop groups
    merged_preds = out_dir / "stage0_router_predictions.jsonl"
    write_jsonl(merged_preds, all_predictions)
    print(f"\nMerged {len(all_predictions)} predictions → {merged_preds}")

    merged_groups = out_dir / "stage0_crop_groups.jsonl"
    write_jsonl(merged_groups, crop_manifest_rows)
    print(f"Merged {len(crop_manifest_rows)} crop groups → {merged_groups}")

    # Summary stats
    n_cases = len({r.get("source_image", "") for r in crop_manifest_rows})
    n_findings = sum(len(r.get("finding_ids", [])) for r in crop_manifest_rows)
    summary = {
        "n_input_cases": len(cases),
        "n_output_cases": n_cases,
        "n_output_findings": n_findings,
        "n_crop_groups": len(crop_manifest_rows),
        "n_predictions": len(all_predictions),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary: {summary}")

    # Clean temp dir
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
