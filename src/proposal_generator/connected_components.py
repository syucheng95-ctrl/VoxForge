import sys
from pathlib import Path
from typing import List

import numpy as np
from scipy.ndimage import label

from src.stage0.evaluate_stage0_policy_recall import bbox_from_mask

from .config import CC_STRUCTURE


def make_structure_3d(connectivity: int = 6) -> np.ndarray:
    if connectivity == 6:
        s = np.zeros((3, 3, 3), dtype=bool)
        s[1, 1, 1] = True
        s[0, 1, 1] = True; s[2, 1, 1] = True
        s[1, 0, 1] = True; s[1, 2, 1] = True
        s[1, 1, 0] = True; s[1, 1, 2] = True
        return s
    elif connectivity == 26:
        return np.ones((3, 3, 3), dtype=bool)
    else:
        raise ValueError(f"unsupported connectivity: {connectivity}")


def extract_components(
    mask: np.ndarray,
    min_voxels: int = 27,
) -> List[np.ndarray]:
    """
    Run connected-component labeling on a binary mask.
    Returns a list of boolean masks, one per component above min_voxels.
    """
    structure = make_structure_3d(CC_STRUCTURE)
    labeled, n_features = label(mask, structure=structure)
    components = []
    for i in range(1, n_features + 1):
        comp = (labeled == i)
        if comp.sum() >= min_voxels:
            components.append(comp)
    return components


def extract_component_bboxes(
    mask: np.ndarray,
    min_voxels: int = 27,
) -> List[List[int]]:
    """Run connected-component labeling and return component bboxes directly."""
    structure = make_structure_3d(CC_STRUCTURE)
    labeled, n_features = label(mask, structure=structure)
    if n_features == 0:
        return []

    counts = np.bincount(labeled.ravel())
    objects = []
    from scipy.ndimage import find_objects

    slices = find_objects(labeled)
    for label_id, obj in enumerate(slices, start=1):
        if obj is None or counts[label_id] < min_voxels:
            continue
        h, w, d = obj
        objects.append([h.start, h.stop, w.start, w.stop, d.start, d.stop])
    return objects


def components_to_bboxes(components: List[np.ndarray]) -> List[List[int]]:
    """Convert each component mask to [h0,h1,w0,w1,d0,d1] in array coordinates."""
    bboxes = []
    for comp in components:
        b = bbox_from_mask(comp)
        if b is not None:
            bboxes.append(b)
    return bboxes
