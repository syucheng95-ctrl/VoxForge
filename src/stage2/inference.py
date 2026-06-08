"""Adapter for STU-Net inference (used by Stage2 evaluation).

Provides: load_stunet_model(checkpoint_path, device) -> (model, gps, helpers)
"""

import os
from pathlib import Path

import torch


def load_stunet_model(checkpoint_path: str, device: str = "cuda"):
    """Load STU-Net model and return (model, gps, (get_group, pad_to_shape_centered, prepare_image))."""
    from src.stage2.eval_stratified import get_group, load_group_patch_shapes, pad_to_shape_centered, prepare_image
    from src.stage2.model import create_stunet_model

    # Set medim checkpoint dir
    upload_root = Path(__file__).resolve().parent.parent.parent
    upload_medim = upload_root / "models" / "medim_ckpt"
    if upload_medim.exists():
        os.environ["MEDIM_CKPT_DIR"] = str(upload_medim)
    else:
        os.environ.setdefault("MEDIM_CKPT_DIR", "")

    # Force offline mode
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"STU-Net checkpoint not found: {ckpt_path}")

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = create_stunet_model(variant="STU-Net-S", pretrained_dataset=None,
                                out_channels=2)
    ckpt_state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt_state["model_state_dict"])
    model = model.to(device).eval()
    gps = load_group_patch_shapes(ckpt_state)

    return model, gps, (get_group, pad_to_shape_centered, prepare_image)
