from typing import List, Optional


def crop_to_original(
    bbox_crop: List[int],
    gate_bbox: List[int],
    original_shape: Optional[tuple] = None,
) -> List[int]:
    """Map a bbox from cropped ROI coordinates to original CT coordinates."""
    H0, H1, W0, W1, D0, D1 = gate_bbox
    h0, h1, w0, w1, d0, d1 = bbox_crop
    return [H0 + h0, H0 + h1, W0 + w0, W0 + w1, D0 + d0, D0 + d1]


def original_to_crop(
    bbox_original: List[int],
    gate_bbox: List[int],
) -> List[int]:
    """Map a bbox from original CT coordinates to cropped ROI coordinates."""
    H0, H1, W0, W1, D0, D1 = gate_bbox
    h0, h1, w0, w1, d0, d1 = bbox_original
    return [h0 - H0, h1 - H0, w0 - W0, w1 - W0, d0 - D0, d1 - D0]


def clamp_to_shape(bbox: List[int], shape: tuple) -> List[int]:
    H, W, D = shape
    h0, h1, w0, w1, d0, d1 = bbox
    return [
        max(0, h0), min(H, h1),
        max(0, w0), min(W, w1),
        max(0, d0), min(D, d1),
    ]
