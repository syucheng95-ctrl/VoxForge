import sys
from pathlib import Path
from typing import Dict, List, Optional

from src.stage0.router_utils import read_jsonl


def build_finding_to_group_index(crop_groups: List[dict]) -> Dict[str, dict]:
    index = {}
    for group in crop_groups:
        for fid in group.get("finding_ids", []):
            index[fid] = group
    return index


def resolve_crop_path(image_path_str: str) -> Optional[Path]:
    p = Path(image_path_str)
    if p.is_absolute() and p.exists():
        return p
    return None


def parse_inputs(
    predictions_path: Path,
    crop_groups_path: Path,
) -> List[dict]:
    predictions = read_jsonl(predictions_path)
    crop_groups = read_jsonl(crop_groups_path)

    finding_to_group = build_finding_to_group_index(crop_groups)
    results = []
    skipped_no_group = 0
    skipped_no_crop = 0

    for pred in predictions:
        fid = pred["id"]
        group = finding_to_group.get(fid)
        if group is None:
            skipped_no_group += 1
            continue

        crop_path = resolve_crop_path(group["image"])
        if crop_path is None:
            skipped_no_crop += 1
            continue

        source_path = Path(group["source_image"])

        results.append({
            "finding_id": fid,
            "prompt": pred["prompt"],
            "category": pred["pred_category"],
            "laterality": pred.get("laterality", "unknown"),
            "anatomy_target": pred.get("anatomy_target", "unknown"),
            "gate_bbox_hwd": group["bbox_hwd"],
            "cropped_roi_path": crop_path,
            "source_image_path": source_path,
            "tightness": pred.get("final_tightness", "conservative"),
            "anatomy_group": pred.get("anatomy_group", ""),
        })

    if skipped_no_group:
        print(f"[input_parser] {skipped_no_group} findings not found in any crop group — skipped")
    if skipped_no_crop:
        print(f"[input_parser] {skipped_no_crop} findings with missing cropped ROI — skipped")
    print(f"[input_parser] {len(results)} findings loaded")
    return results
