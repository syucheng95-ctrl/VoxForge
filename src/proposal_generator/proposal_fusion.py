import sys
from pathlib import Path
from typing import List, Optional

import nibabel as nib
import numpy as np

from src.stage0.evaluate_stage0_policy_recall import (
    bbox_from_mask,
    bbox_volume,
    expand_bbox,
    margin_mm_to_vox,
    union_bboxes,
)

from .config import EXPAND_MARGIN_MM, UNION_IOU_THRESHOLD, EXPERT_PRIORITY
from .coordinate_mapper import original_to_crop


def iou_3d(a: List[int], b: List[int]) -> float:
    """Compute IoU between two bboxes [h0,h1,w0,w1,d0,d1]."""
    h0 = max(a[0], b[0]); h1 = min(a[1], b[1])
    w0 = max(a[2], b[2]); w1 = min(a[3], b[3])
    d0 = max(a[4], b[4]); d1 = min(a[5], b[5])
    if h0 >= h1 or w0 >= w1 or d0 >= d1:
        return 0.0

    inter = (h1 - h0) * (w1 - w0) * (d1 - d0)
    vol_a = bbox_volume(a)
    vol_b = bbox_volume(b)
    union = max(1, vol_a + vol_b - inter)
    return inter / union


def union_overlapping(bboxes: List[List[int]], iou_threshold: float) -> List[List[int]]:
    """Merge overlapping bboxes. O(n^2) but n is small."""
    if len(bboxes) <= 1:
        return bboxes

    sorted_boxes = sorted(bboxes, key=lambda b: bbox_volume(b))
    merged = []
    for bbox in sorted_boxes:
        found = False
        for i, existing in enumerate(merged):
            if iou_3d(bbox, existing) >= iou_threshold:
                merged[i] = union_bboxes([bbox, existing])
                found = True
                break
        if not found:
            merged.append(bbox)
    return merged


def compute_compactness(bbox_crop: List[int], mask_crop: np.ndarray) -> float:
    """Ratio of foreground voxels to bbox volume. 1 = perfectly compact."""
    h0, h1, w0, w1, d0, d1 = bbox_crop
    bbox_vol = max(1, (h1 - h0) * (w1 - w0) * (d1 - d0))
    fg = mask_crop[h0:h1, w0:w1, d0:d1].sum() if mask_crop is not None else 0
    return min(1.0, fg / bbox_vol)


def compute_hu_stats(bbox_crop: List[int], hu_data: np.ndarray) -> dict:
    """Compute HU statistics within a bbox in cropped ROI coordinates."""
    h0, h1, w0, w1, d0, d1 = bbox_crop
    sub = hu_data[h0:h1, w0:w1, d0:d1]
    flat = sub.ravel()
    if flat.size == 0:
        return {"hu_mean": None, "hu_std": None}
    return {
        "hu_mean": float(flat.mean()),
        "hu_std": float(flat.std()),
    }


def compute_hu_contrast(bbox_crop: List[int], hu_data: np.ndarray,
                        expand_vox: int = 5) -> Optional[float]:
    """Contrast between inside bbox and surrounding ring."""
    H, W, D = hu_data.shape
    h0, h1, w0, w1, d0, d1 = bbox_crop
    h0e = max(0, h0 - expand_vox); h1e = min(H, h1 + expand_vox)
    w0e = max(0, w0 - expand_vox); w1e = min(W, w1 + expand_vox)
    d0e = max(0, d0 - expand_vox); d1e = min(D, d1 + expand_vox)

    inner = hu_data[h0:h1, w0:w1, d0:d1]
    outer = hu_data[h0e:h1e, w0e:w1e, d0e:d1e]
    mask = np.ones_like(outer, dtype=bool)
    mask[h0 - h0e:h1 - h0e, w0 - w0e:w1 - w0e, d0 - d0e:d1 - d0e] = False
    surround = outer[mask]
    if inner.size == 0 or surround.size == 0:
        return None
    return abs(float(inner.mean()) - float(surround.mean()))


def clip_to_gate(bbox: List[int], gate_bbox: List[int]) -> List[int]:
    """Clamp a bbox to lie within the gate_bbox."""
    return [
        max(gate_bbox[0], bbox[0]), min(gate_bbox[1], bbox[1]),
        max(gate_bbox[2], bbox[2]), min(gate_bbox[3], bbox[3]),
        max(gate_bbox[4], bbox[4]), min(gate_bbox[5], bbox[5]),
    ]


def fuse_proposals(
    bboxes_original: List[List[int]],
    gate_bbox: List[int],
    original_shape: tuple,
    zooms: tuple,
    source_expert: str,
    cropped_hu: Optional[np.ndarray] = None,
    bbox_scores: Optional[List[float]] = None,
) -> List[dict]:
    """
    1. clip to gate
    2. expand margin
    3. union overlapping (not NMS)
    4. attach signal fields
    No top-k truncation.
    """
    margin_vox = margin_mm_to_vox(EXPAND_MARGIN_MM, zooms)
    H, W, D = original_shape

    # Step 1 + 2: clip to gate, expand margin, clamp to original shape
    processed = []
    processed_scores = []
    for idx, b in enumerate(bboxes_original):
        # First clip to gate bounds, then expand, then clip to CT shape
        gate_clipped = clip_to_gate(b, gate_bbox)
        expanded = expand_bbox(gate_clipped, original_shape, margin_vox)
        processed.append(expanded)
        if bbox_scores is not None and idx < len(bbox_scores):
            processed_scores.append(float(bbox_scores[idx]))
        else:
            processed_scores.append(None)

    # Step 3: union overlapping
    merged = union_overlapping(processed, UNION_IOU_THRESHOLD)

    # Step 4: attach signals
    results = []
    for bbox_orig in merged:
        bbox_crop = original_to_crop(bbox_orig, gate_bbox)
        vol = bbox_volume(bbox_orig)

        hu_stats = {"hu_mean": None, "hu_std": None}
        hu_contrast = None
        compactness = 0.0
        if cropped_hu is not None:
            hu_stats = compute_hu_stats(bbox_crop, cropped_hu)
            hu_contrast = compute_hu_contrast(bbox_crop, cropped_hu)
            # also compute compactness using thresholded data placeholder
            compactness = 0.0

        proposal = {
            "proposal_bbox_hwd": bbox_orig,
            "source_expert": source_expert,
            "volume_voxels": vol,
            "hu_mean": hu_stats.get("hu_mean"),
            "hu_std": hu_stats.get("hu_std"),
            "hu_contrast": hu_contrast,
            "compactness": compactness,
            "expert_priority": EXPERT_PRIORITY.get(source_expert, 2),
        }
        if bbox_scores is not None:
            matching_scores = [
                s for b, s in zip(processed, processed_scores)
                if s is not None and iou_3d(bbox_orig, b) > 0
            ]
            proposal["detector_score"] = max(matching_scores) if matching_scores else None
        results.append(proposal)

    return results
