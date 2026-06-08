from typing import List

import numpy as np

from .config import MIN_COMPONENT_VOXELS, MAX_ELONGATION


def elongation(bbox: List[int]) -> float:
    """Ratio of max dimension to min dimension. >1 = elongated."""
    h = bbox[1] - bbox[0]
    w = bbox[3] - bbox[2]
    d = bbox[5] - bbox[4]
    dims = [x for x in [h, w, d] if x > 0]
    if len(dims) < 2:
        return 1.0
    return max(dims) / min(dims)


def touches_gate_edge(bbox_crop: List[int], gate_shape: tuple) -> bool:
    """True if the bbox touches the edge of the cropped ROI (possibly truncated)."""
    H, W, D = gate_shape
    h0, h1, w0, w1, d0, d1 = bbox_crop
    return h0 <= 0 or h1 >= H or w0 <= 0 or w1 >= W or d0 <= 0 or d1 >= D


def filter_bboxes(
    bboxes: List[List[int]],
    mask: np.ndarray,
    category: str,
    crop_shape: tuple,
) -> List[List[int]]:
    """
    Filter candidate bboxes (in cropped ROI coordinates):
    1. Volume check — reject too small per category
    2. Elongation check — reject too flat/elongated
    3. Gate-edge flagging — note but don't drop

    Returns filtered bbox list.
    """
    min_voxels = MIN_COMPONENT_VOXELS.get(category, MIN_COMPONENT_VOXELS["default"])

    kept = []
    for bbox in bboxes:
        h0, h1, w0, w1, d0, d1 = bbox
        vol = (h1 - h0) * (w1 - w0) * (d1 - d0)

        if vol < min_voxels:
            continue

        if elongation(bbox) > MAX_ELONGATION:
            continue

        kept.append(bbox)

    return kept
