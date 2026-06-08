"""VoxTell model builder."""

import json
import pydoc
from pathlib import Path

import torch


# ======================================================================
# Model builder
# ======================================================================

def build_voxtell_from_checkpoint(model_dir, device=torch.device("cpu")):
    model_dir = Path(model_dir)
    from src.voxtell.model.voxtell_model import VoxTellModel

    with open(model_dir / "plans.json", "r", encoding="utf-8") as f:
        plans = json.load(f)
    arch = dict(**plans["configurations"]["3d_fullres"]["architecture"]["arch_kwargs"])
    for k in plans["configurations"]["3d_fullres"]["architecture"]["_kw_requires_import"]:
        if arch[k] is not None:
            arch[k] = pydoc.locate(arch[k])

    net = VoxTellModel(input_channels=1, **arch, decoder_layer=4,
                       text_embedding_dim=2560, num_maskformer_stages=5,
                       num_heads=32, query_dim=2048, project_to_decoder_hidden_dim=2048,
                       deep_supervision=False).to(device)

    ckpt = torch.load(model_dir / "fold_0" / "checkpoint_final.pth", map_location=device, weights_only=False)
    net.load_state_dict(ckpt["network_weights"])
    net.eval()
    return net
