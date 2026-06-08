from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import zoom

from .config import (
    NODULE_DETECTOR_MAX_DETECTIONS,
    NODULE_DETECTOR_SCORE_THRESHOLDS,
)

# MONAI bundle target spacing (LUNA16 training spacing)
TARGET_SPACING: Tuple[float, float, float] = (0.703125, 0.703125, 1.25)

_DETECTOR_CACHE: Dict[str, object] = {}


def _get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _load_detector(bundle_dir: Path):
    cache_key = str(bundle_dir.resolve())
    if cache_key in _DETECTOR_CACHE:
        return _DETECTOR_CACHE[cache_key]

    from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
    from monai.apps.detection.networks.retinanet_network import (
        RetinaNet,
        resnet_fpn_feature_extractor,
    )
    from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape
    from monai.networks.nets.resnet import resnet50

    device = _get_device()
    bundle_dir = bundle_dir.resolve()
    ckpt_path = bundle_dir / "models" / "model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"MONAI nodule detector checkpoint not found: {ckpt_path}")

    backbone = resnet50(
        spatial_dims=3,
        n_input_channels=1,
        conv1_t_stride=[2, 2, 1],
        conv1_t_size=[7, 7, 7],
    )
    feature_extractor = resnet_fpn_feature_extractor(backbone, 3, False, [1, 2], None)
    network = RetinaNet(
        spatial_dims=3,
        num_classes=1,
        num_anchors=3,
        feature_extractor=feature_extractor,
        size_divisible=[16, 16, 8],
        use_list_output=False,
    ).to(device)

    state = torch.load(str(ckpt_path), map_location=device)
    network.load_state_dict(state)
    network.eval()

    anchor_generator = AnchorGeneratorWithAnchorShape(
        feature_map_scales=[1, 2, 4],
        base_anchor_shapes=[[6, 8, 4], [8, 6, 5], [10, 10, 6]],
    )
    detector = RetinaNetDetector(
        network=network,
        anchor_generator=anchor_generator,
        debug=False,
        spatial_dims=3,
        num_classes=1,
        size_divisible=[16, 16, 8],
    )
    detector.set_target_keys(box_key="box", label_key="label")
    detector.set_box_selector_parameters(
        score_thresh=0.001,
        topk_candidates_per_level=2000,
        nms_thresh=0.22,
        detections_per_img=200,
    )
    detector.eval()

    loaded = {"detector": detector, "device": device}
    _DETECTOR_CACHE[cache_key] = loaded
    return loaded


def _resample_to_target(
    hu: np.ndarray, src_spacing: Tuple[float, float, float]
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Resample HU volume to MONAI target spacing. Returns (resampled, scale_factors)."""
    scale = tuple(s / t for s, t in zip(src_spacing, TARGET_SPACING))
    if all(abs(s - 1.0) < 0.05 for s in scale):
        return hu, (1.0, 1.0, 1.0)
    resampled = zoom(hu, scale, order=1, mode="constant", cval=-1024.0)
    return resampled.astype(np.float32), scale


def _preprocess_hu(
    cropped_nii: nib.Nifti1Image,
) -> Tuple[torch.Tensor, Tuple[float, float, float]]:
    """Normalise HU and optionally resample to target spacing."""
    hu = np.asanyarray(cropped_nii.dataobj).astype(np.float32)
    zooms = tuple(float(z) for z in cropped_nii.header.get_zooms()[:3])
    hu_resampled, scale = _resample_to_target(hu, zooms)
    scaled = np.clip((hu_resampled + 1024.0) / 1324.0, 0.0, 1.0)
    return torch.from_numpy(scaled).unsqueeze(0), scale


def _box_to_hwd(
    box: np.ndarray, shape: tuple, scale: Tuple[float, float, float]
) -> Optional[List[int]]:
    """
    Convert MONAI RetinaNet detection box to HWD bbox in original voxel coords.

    MONAI outputs boxes in xyzxyz format matching the input tensor's spatial dims.
    Our tensor is (B, H, W, D), so box[i] maps to:
      box[0]->h0, box[1]->w0, box[2]->d0, box[3]->h1, box[4]->w1, box[5]->d1

    If resampling was applied, we scale coordinates back by 1/scale.
    """
    h0 = int(np.floor(box[0] / scale[0]))
    w0 = int(np.floor(box[1] / scale[1]))
    d0 = int(np.floor(box[2] / scale[2]))
    h1 = int(np.ceil(box[3] / scale[0]))
    w1 = int(np.ceil(box[4] / scale[1]))
    d1 = int(np.ceil(box[5] / scale[2]))

    H, W, D = shape
    bbox = [
        max(0, h0), min(H, h1),
        max(0, w0), min(W, w1),
        max(0, d0), min(D, d1),
    ]
    if bbox[0] >= bbox[1] or bbox[2] >= bbox[3] or bbox[4] >= bbox[5]:
        return None
    return bbox


_RAW_DETECTION_CACHE: Dict[str, List[dict]] = {}


def run_nodule_detector_expert(
    cropped_nii: nib.Nifti1Image,
    category: str,
    bundle_dir: str,
    cache_key: Optional[str] = None,
) -> Optional[List[dict]]:
    """
    Run the MONAI LUNA16 lung nodule detector on a cropped ROI.

    Args:
        bundle_dir: path to MONAI bundle directory (contains models/model.pt)

    Returns detections in cropped ROI voxel coordinates:
      {"bbox": [h0, h1, w0, w1, d0, d1], "score": float}

    If cache_key is provided, raw detections (before thresholding) are cached
    per crop. Per-category threshold and max-detections filters are applied
    on each call, so 1e and 2d sharing the same crop get correct behaviour.
    """
    score_threshold = NODULE_DETECTOR_SCORE_THRESHOLDS.get(category, 0.05)

    if cache_key is not None and cache_key in _RAW_DETECTION_CACHE:
        raw_dets = _RAW_DETECTION_CACHE[cache_key]
    else:
        loaded = _load_detector(Path(bundle_dir))
        detector = loaded["detector"]
        device = loaded["device"]

        image, scale = _preprocess_hu(cropped_nii)
        try:
            image = image.to(device)
            with torch.inference_mode():
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                    pred = detector([image], use_inferer=False)[0]

            boxes = pred.get("box")
            scores = pred.get("label_scores")
            boxes_np = None if boxes is None else boxes.float().cpu().numpy()
            scores_np = None if scores is None else scores.float().cpu().numpy()
            del pred, boxes, scores
        finally:
            del image
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if boxes_np is None or scores_np is None or len(boxes_np) == 0:
            return None

        raw_dets = []
        for box, score in zip(boxes_np, scores_np):
            bbox = _box_to_hwd(box, cropped_nii.shape, scale)
            if bbox is None:
                continue
            raw_dets.append({"bbox": bbox, "score": float(score)})

        if not raw_dets:
            return None

        raw_dets.sort(key=lambda item: item["score"], reverse=True)

        if cache_key is not None:
            _RAW_DETECTION_CACHE[cache_key] = raw_dets

    # Filter by category-specific threshold
    detections = [d for d in raw_dets if d["score"] >= score_threshold]
    if not detections:
        return None

    return detections[:NODULE_DETECTOR_MAX_DETECTIONS]
