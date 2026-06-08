"""Adapter for VoxTell inference (used by Stage1 verification).

Provides: build_voxtell_model(model_dir, device) -> model
          NibabelIOWithReorient (image I/O matching VoxTell expected orientation)
          get_nibabel_io_with_reorient() -> NibabelIOWithReorient
"""

from pathlib import Path

import torch


def get_nibabel_io_with_reorient():
    """Get NibabelIOWithReorient for VoxTell-compatible image loading."""
    from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
    return NibabelIOWithReorient


def build_voxtell_model(model_dir: str, device: str = "cuda:0"):
    """Load VoxTell model from checkpoint."""
    from src.stage1.config import build_voxtell_from_checkpoint
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = build_voxtell_from_checkpoint(Path(model_dir), device=torch.device("cpu"))
    return model.to(device).eval()
