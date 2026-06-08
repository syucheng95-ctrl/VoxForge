"""VoxTell per-proposal verifier.

Loads VoxTell once, verifies proposals by running text-conditional
inference on cropped ROIs and scoring activation inside each proposal box.
"""

import sys
import time
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import torch


DEFAULT_THRESHOLD = 0.3  # prob > this → foreground for coarse mask


def _build_predictor(model_dir: str, device: torch.device, text_model: str):
    """Build an EvalPredictor for inference."""
    from src.stage1.eval import EvalPredictor
    return EvalPredictor(model_dir=model_dir, device=device, text_encoding_model=text_model)


class Verifier:
    """Per-proposal VoxTell text-consistency verifier."""

    def __init__(self, model_dir: str, text_model: str, device: str = "cuda:0",
                 project_root: Optional[Path] = None):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"[Verifier] Loading VoxTell from {model_dir}...")
        t0 = time.time()
        self.predictor = _build_predictor(model_dir, self.device, text_model)
        print(f"[Verifier] Loaded in {time.time() - t0:.0f}s")

        # Cache text embeddings keyed by prompt
        self._text_cache: dict[str, torch.Tensor] = {}

    def _get_text_embedding(self, prompt: str) -> torch.Tensor:
        if prompt not in self._text_cache:
            self._text_cache[prompt] = self.predictor.embed_text_prompts([prompt])
        return self._text_cache[prompt]

    @torch.inference_mode()
    def verify_finding(
        self,
        cropped_roi: np.ndarray,
        prompt: str,
        proposals: list[dict],
        gate_bbox_hwd: list[int],
        raw_shape: Optional[tuple] = None,
        mask_out_dir: Optional[Path] = None,
    ) -> dict:
        """Verify proposals + save coarse masks for verified ones.

        Args:
            cropped_roi: reoriented 3D array (D,W,H) from NibabelIOWithReorient
            prompt: finding text
            proposals: list of proposal dicts with proposal_bbox_hwd in
                       ORIGINAL CT coordinates [h0,h1,w0,w1,d0,d1]
            gate_bbox_hwd: gate bbox in original CT coords [gh0,gh1,gw0,gw1,gd0,gd1]
            raw_shape: raw HWD shape of the cropped ROI before reorientation
            mask_out_dir: if set, save binary coarse masks for verified proposals
        """
        t0 = time.time()

        logits = self.predictor.predict_single_image_logits(
            cropped_roi.copy(), [prompt],
        )
        clipped = np.clip(logits[0], -50, 50)
        prob = 1.0 / (1.0 + np.exp(-clipped))  # shape (D, W, H) in reader space
        binary = (prob > DEFAULT_THRESHOLD).astype(np.uint8)

        gh0, gh1, gw0, gw1, gd0, gd1 = gate_bbox_hwd
        if raw_shape is not None:
            raw_H, raw_W, raw_D = raw_shape
        else:
            raw_H, raw_W, raw_D = cropped_roi.shape[::-1]  # best guess

        verified = []
        rejected = []

        for prop in proposals:
            ph0, ph1, pw0, pw1, pd0, pd1 = prop["proposal_bbox_hwd"]

            # Step 1: map from original CT HWD → cropped ROI raw HWD
            ch0 = max(0, ph0 - gh0)
            ch1 = min(raw_H, ph1 - gh0)
            cw0 = max(0, pw0 - gw0)
            cw1 = min(raw_W, pw1 - gw0)
            cd0 = max(0, pd0 - gd0)
            cd1 = min(raw_D, pd1 - gd0)

            if ch0 >= ch1 or cw0 >= cw1 or cd0 >= cd1:
                rejected.append({**prop, "score": 0.0, "reason": "bbox outside crop"})
                continue

            # Step 2: raw HWD → reader DWH (NibabelIOWithReorient transform)
            # raw[h,w,d] = reader[d, raw_W-1-w, raw_H-1-h]
            # → reader_d0=cd0, reader_d1=cd1, reader_w0=raw_W-cw1, reader_w1=raw_W-cw0,
            #   reader_h0=raw_H-ch1, reader_h1=raw_H-ch0
            rd0, rd1 = cd0, cd1
            rw0, rw1 = raw_W - cw1, raw_W - cw0
            rh0, rh1 = raw_H - ch1, raw_H - ch0

            rd0, rd1 = max(0, rd0), min(prob.shape[0], rd1)
            rw0, rw1 = max(0, rw0), min(prob.shape[1], rw1)
            rh0, rh1 = max(0, rh0), min(prob.shape[2], rh1)

            if rd0 >= rd1 or rw0 >= rw1 or rh0 >= rh1:
                rejected.append({**prop, "score": 0.0, "reason": "bbox outside reoriented crop"})
                continue

            region = prob[rd0:rd1, rw0:rw1, rh0:rh1]
            max_prob = float(region.max())
            mean_prob = float(region.mean())
            fg_ratio = float((region > DEFAULT_THRESHOLD).mean())
            score = 0.5 * max_prob + 0.3 * mean_prob + 0.2 * fg_ratio

            entry = {
                **{k: v for k, v in prop.items()},
                "score": round(score, 4),
                "max_prob": round(max_prob, 4),
                "mean_prob": round(mean_prob, 4),
                "fg_ratio": round(fg_ratio, 4),
            }

            if max_prob > 0.3 or score > 0.25:
                entry["verified"] = True
                verified.append(entry)
            else:
                entry["verified"] = False
                rejected.append(entry)

        fallback = len(verified) == 0
        if fallback and rejected:
            best = max(rejected, key=lambda x: x["score"])
            rejected.remove(best)
            best["verified"] = True
            best["fallback_promoted"] = True
            verified.append(best)

        # Save coarse masks for verified proposals
        if mask_out_dir is not None and verified:
            mask_out_dir.mkdir(parents=True, exist_ok=True)
            for entry in verified:
                pid = entry.get("proposal_id", "unknown")
                mask_path = mask_out_dir / f"{pid}_coarse.nii.gz"

                ph0, ph1, pw0, pw1, pd0, pd1 = entry["proposal_bbox_hwd"]
                ch0 = max(0, ph0 - gh0); ch1 = min(raw_H, ph1 - gh0)
                cw0 = max(0, pw0 - gw0); cw1 = min(raw_W, pw1 - gw0)
                cd0 = max(0, pd0 - gd0); cd1 = min(raw_D, pd1 - gd0)
                rd0, rd1 = cd0, cd1
                rw0, rw1 = raw_W - cw1, raw_W - cw0
                rh0, rh1 = raw_H - ch1, raw_H - ch0
                rd0, rd1 = max(0, rd0), min(prob.shape[0], rd1)
                rw0, rw1 = max(0, rw0), min(prob.shape[1], rw1)
                rh0, rh1 = max(0, rh0), min(prob.shape[2], rh1)

                if rd0 < rd1 and rw0 < rw1 and rh0 < rh1:
                    mask = binary[rd0:rd1, rw0:rw1, rh0:rh1].copy()
                    mask_nii = nib.Nifti1Image(mask, np.eye(4))
                    nib.save(mask_nii, str(mask_path))
                    entry["coarse_mask_path"] = str(mask_path)

                    # Also save probability map for high-threshold Stage2 ROI extraction
                    prob_path = mask_out_dir / f"{pid}_prob.nii.gz"
                    prob_roi = prob[rd0:rd1, rw0:rw1, rh0:rh1].copy().astype(np.float32)
                    prob_nii = nib.Nifti1Image(prob_roi, np.eye(4))
                    nib.save(prob_nii, str(prob_path))
                    entry["coarse_prob_path"] = str(prob_path)

        elapsed = time.time() - t0
        print(f"  [{proposals[0].get('proposal_id', '?')[:20]}...] "
              f"{len(verified)}V/{len(rejected)}R "
              f"(max_prob={verified[0]['max_prob'] if verified else 0:.3f}, "
              f"{elapsed:.1f}s)")

        return {
            "verified": verified,
            "rejected": rejected,
            "fallback": fallback,
        }
