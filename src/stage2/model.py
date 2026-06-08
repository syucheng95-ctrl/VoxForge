from pathlib import Path
from typing import Literal


# Overridden at runtime by MEDIM_CKPT_DIR env var set from config
DEFAULT_CKPT_DIR = Path("models/medim_ckpt")


def create_stunet_model(
    variant: Literal["STU-Net-S", "STU-Net-B"] = "STU-Net-S",
    pretrained_dataset: str = "TotalSegmentator",
    in_channels: int = 1,
    out_channels: int = 2,
):
    """
    Create a MedIM STU-Net model and adapt the segmentation head to downstream binary segmentation.

    Notes:
    - Pretrained weights are loaded through MedIM.
    - The pretrained segmentation head is task-specific, so the final segmentation layer may
      be replaced for the target class count.
    """
    import os
    # MedIM reads MEDIM_CKPT_DIR when its registry module is imported, so set it before importing medim.
    # Respect an externally configured cache path (upload/external/stunet_inference.py sets it to models/medim_ckpt).
    if "MEDIM_CKPT_DIR" not in os.environ:
        DEFAULT_CKPT_DIR.mkdir(parents=True, exist_ok=True)
        os.environ["MEDIM_CKPT_DIR"] = str(DEFAULT_CKPT_DIR.resolve())
    import medim

    model = medim.create_model(variant, dataset=pretrained_dataset)

    # Keep the pretrained encoder/decoder weights but reset the segmentation head for downstream classes.
    if hasattr(model, "seg_outputs") and len(model.seg_outputs) > 0:
        head = model.seg_outputs[-1]
        head_in_channels = head.in_channels
        import torch.nn as nn

        model.seg_outputs[-1] = nn.Conv3d(head_in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    return model
