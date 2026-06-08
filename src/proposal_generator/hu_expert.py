from typing import List, Optional

import nibabel as nib
import numpy as np

from .config import HU_THRESHOLDS, MIN_COMPONENT_VOXELS
from .connected_components import extract_component_bboxes
from .morphology_filter import filter_bboxes


def apply_hu_threshold(cropped_nii: nib.Nifti1Image, category: str) -> np.ndarray:
    """Apply HU window for the given category. Returns boolean mask."""
    hu_data = np.asanyarray(cropped_nii.dataobj).astype(np.float32)
    thresholds = HU_THRESHOLDS[category]
    lower = thresholds["lower"]
    upper = thresholds["upper"]

    if lower is not None and upper is not None:
        mask = (hu_data >= lower) & (hu_data <= upper)
    elif lower is not None:
        mask = hu_data >= lower
    elif upper is not None:
        mask = hu_data <= upper
    else:
        mask = np.zeros_like(hu_data, dtype=bool)

    return mask


def run_hu_expert(
    cropped_nii: nib.Nifti1Image,
    category: str,
) -> Optional[List[List[int]]]:
    """
    Full HU expert pipeline:
      0. Guard: if category not in HU_THRESHOLDS → return None
      1. Apply HU threshold → binary mask
      2. Connected components → per-component masks
      3. Components to bboxes (in cropped ROI coords)
      4. Morphology filter
    Returns list of bboxes in cropped ROI coordinates, or None if no candidates.
    """
    if category not in HU_THRESHOLDS:
        return None

    mask = apply_hu_threshold(cropped_nii, category)

    if mask.sum() == 0:
        return None

    min_vox = MIN_COMPONENT_VOXELS.get(category, MIN_COMPONENT_VOXELS["default"])
    bboxes = extract_component_bboxes(mask, min_voxels=min_vox)
    if not bboxes:
        return None

    shape = cropped_nii.shape  # H, W, D
    bboxes = filter_bboxes(bboxes, mask, category, shape)

    if not bboxes:
        return None

    return bboxes
