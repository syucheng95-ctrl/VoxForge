"""Stratified evaluation for STU-Net checkpoint.

Modes:
  full-roi       Process each full ROI once after pad-to-32, no crop.
  fixed-patch    Center crop/pad to per-group patch.
  sliding-window Run per-group patch windows over the full ROI and average logits.

Usage: python eval_stratified.py --checkpoint <CKPT>.pt --manifest manifest.jsonl --data-root <ROOT>
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.stage2.dataset import GROUP_ORDER, ReXStage2ROIPatchDataset, get_group, load_jsonl
from src.stage2.model import create_stunet_model


DEFAULT_GROUP_PATCH_SHAPES = {
    "small": (64, 64, 32),
    "medium": (96, 96, 48),
    "large": (224, 224, 64),
    "xlarge": (256, 256, 160),
}

DEFAULT_GROUP_BATCHES = {
    "small": 24,
    "medium": 12,
    "large": 4,
    "xlarge": 1,
}


def batch_segmentation_metrics(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6):
    preds = torch.argmax(logits, dim=1).float()
    targets = targets.float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    pred_sum = preds.sum(dim=(1, 2, 3))
    target_sum = targets.sum(dim=(1, 2, 3))
    union = pred_sum + target_sum - intersection

    dice = (2 * intersection + eps) / (pred_sum + target_sum + eps)
    iou = (intersection + eps) / (union + eps)
    recall = (intersection + eps) / (target_sum + eps)
    precision = (intersection + eps) / (pred_sum + eps)

    return {
        "dice": float(dice.mean().item()),
        "iou": float(iou.mean().item()),
        "recall": float(recall.mean().item()),
        "precision": float(precision.mean().item()),
    }


def pad_to_shape_centered(volume: np.ndarray, target_shape: tuple[int, int, int], cval: float):
    """Pad symmetrically to at least target_shape, without cropping."""
    h, w, d = volume.shape
    th, tw, td = target_shape
    if (th, tw, td) == (h, w, d):
        return volume, (h, w, d), (0, 0, 0)
    out = np.full((th, tw, td), cval, dtype=volume.dtype)
    h_off = (th - h) // 2
    w_off = (tw - w) // 2
    d_off = (td - d) // 2
    out[h_off:h_off+h, w_off:w_off+w, d_off:d_off+d] = volume
    return out, (h, w, d), (h_off, w_off, d_off)


def pad_to_multiple_centered(volume: np.ndarray, multiple: int, cval: float):
    """Pad symmetrically so the original ROI stays centered (pad-only, no crop)."""
    h, w, d = volume.shape
    target_shape = (
        int(np.ceil(h / multiple) * multiple),
        int(np.ceil(w / multiple) * multiple),
        int(np.ceil(d / multiple) * multiple),
    )
    return pad_to_shape_centered(volume, target_shape, cval)


def prepare_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32, copy=False)
    image = np.clip(image, -1000.0, 400.0)
    mean = float(image.mean())
    std = float(image.std())
    image = image - mean
    if std > 0:
        image = image / std
    return image


def clip_image(image: np.ndarray) -> np.ndarray:
    return np.clip(image.astype(np.float32, copy=False), -1000.0, 400.0)


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32, copy=False)
    mean = float(image.mean())
    std = float(image.std())
    image = image - mean
    if std > 0:
        image = image / std
    return image


def load_roi_pair(row, data_root: Path):
    image_path = data_root / row["roi_image_path"]
    mask_path = data_root / row["roi_mask_path"]

    image = np.asarray(nib.load(str(image_path)).dataobj)
    mask = np.asarray(nib.load(str(mask_path)).dataobj).astype(np.uint8)
    # Handle 4D NIfTI: take first 3 dims only
    if image.ndim == 4:
        image = image[..., 0]
    if mask.ndim == 4:
        mask = mask[..., 0]
    mask = (mask > 0).astype(np.uint8)
    return image, mask


def update_metric_buckets(group_metrics, all_metrics, group, metrics):
    all_metrics.append(metrics)
    group_metrics[group]["dice"].append(metrics["dice"])
    group_metrics[group]["recall"].append(metrics["recall"])
    group_metrics[group]["precision"].append(metrics["precision"])


def summarize_metrics(group_metrics, all_metrics):
    results = {}
    for g in GROUP_ORDER:
        gm = group_metrics[g]
        n = len(gm["dice"])
        if n == 0:
            continue
        results[g] = {
            "n": n,
            "dice": float(np.mean(gm["dice"])),
            "recall": float(np.mean(gm["recall"])),
            "precision": float(np.mean(gm["precision"])),
        }

    overall = {
        "n": len(all_metrics),
        "dice": float(np.mean([m["dice"] for m in all_metrics])) if all_metrics else 0.0,
        "recall": float(np.mean([m["recall"] for m in all_metrics])) if all_metrics else 0.0,
        "precision": float(np.mean([m["precision"] for m in all_metrics])) if all_metrics else 0.0,
    }
    return results, overall


def load_group_patch_shapes(ckpt: dict):
    raw_shapes = ckpt.get("group_patch_shapes") or DEFAULT_GROUP_PATCH_SHAPES
    shapes = {}
    for group in GROUP_ORDER:
        shape = raw_shapes.get(group, DEFAULT_GROUP_PATCH_SHAPES[group])
        shapes[group] = tuple(int(v) for v in shape)
    return shapes


def load_group_batches(ckpt: dict):
    raw_configs = ckpt.get("group_configs") or {}
    batches = {}
    for group in GROUP_ORDER:
        cfg = raw_configs.get(group, {})
        batches[group] = int(cfg.get("batch", DEFAULT_GROUP_BATCHES[group]))
    return batches


def eval_full_roi(
    model,
    manifest_path,
    data_root,
    tag,
    device,
    group_patch_shapes,
    max_voxels=0,
    progress_interval=100,
    norm_scope="input",
    min_patch=False,
    skip_oom=True,
):
    """Evaluate all samples WITHOUT cropping: pad to 32x then crop output back."""
    rows = load_jsonl(manifest_path)
    data_root = Path(data_root)

    group_metrics = defaultdict(lambda: {"dice": [], "recall": [], "precision": []})
    all_metrics = []
    skipped = 0

    for idx, row in enumerate(rows, start=1):
        if progress_interval > 0 and idx % progress_interval == 0:
            print(f"    {tag}: {idx}/{len(rows)}", flush=True)

        image, mask = load_roi_pair(row, data_root)
        orig_shape = image.shape  # (H, W, D)

        # Skip if padded volume exceeds memory limit
        if max_voxels > 0:
            h, w, d = orig_shape
            th = int(np.ceil(h / 32) * 32)
            tw = int(np.ceil(w / 32) * 32)
            td = int(np.ceil(d / 32) * 32)
            if th * tw * td > max_voxels * 1e6:
                skipped += 1
                continue

        group = get_group(tuple(row["roi_shape_hwd"]))
        if min_patch:
            ph, pw, pd = group_patch_shapes[group]
            target_shape = (
                max(int(np.ceil(orig_shape[0] / 32) * 32), ph),
                max(int(np.ceil(orig_shape[1] / 32) * 32), pw),
                max(int(np.ceil(orig_shape[2] / 32) * 32), pd),
            )
            image_padded, (h0, w0, d0), (h_off, w_off, d_off) = pad_to_shape_centered(image, target_shape, cval=-1024.0)
            mask_padded, _, _ = pad_to_shape_centered(mask, target_shape, cval=0.0)
        else:
            # Pad symmetrically (no crop) to nearest multiple of 32 so UNet can process it
            image_padded, (h0, w0, d0), (h_off, w_off, d_off) = pad_to_multiple_centered(image, 32, cval=-1024.0)
            mask_padded, _, _ = pad_to_multiple_centered(mask, 32, cval=0.0)

        if norm_scope == "input":
            image_tensor = prepare_image(image_padded)
        elif norm_scope == "global":
            image_tensor = clip_image(image)
            image_tensor = normalize_image(image_tensor)
            image_tensor, _, _ = pad_to_shape_centered(image_tensor, image_padded.shape, cval=0.0)
        else:
            raise ValueError("full-roi supports --norm-scope input or global")

        image_tensor = np.transpose(image_tensor, (2, 1, 0))  # HWD -> DWH
        image_tensor = torch.from_numpy(image_tensor.copy()).float()
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, D, W, H)
        mask_tensor = np.transpose(mask_padded, (2, 1, 0))
        mask_tensor = torch.from_numpy(mask_tensor.copy()).long()
        mask_tensor = mask_tensor.unsqueeze(0)  # (1, D, W, H)

        x = image_tensor.to(device)
        y = mask_tensor.to(device)

        try:
            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                    logits = model(x)
        except torch.OutOfMemoryError:
            if not skip_oom:
                raise
            skipped += 1
            print(
                f"    skipped OOM: {row.get('component_sample_id', '<unknown>')} "
                f"group={group} padded_shape={image_padded.shape}",
                flush=True,
            )
            del x, y
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        # Crop output back to original ROI (offsets in D, W, H order after zyx transpose)
        logits_cropped = logits[:, :, d_off:d_off+d0, w_off:w_off+w0, h_off:h_off+h0]
        y_cropped = y[:, d_off:d_off+d0, w_off:w_off+w0, h_off:h_off+h0]

        m = batch_segmentation_metrics(logits_cropped, y_cropped)
        update_metric_buckets(group_metrics, all_metrics, group, m)

    results, overall = summarize_metrics(group_metrics, all_metrics)
    return results, overall, skipped


def eval_fixed_patch(model, manifest_path, data_root, tag, device, group_patch_shapes, group_batches, progress_interval=0):
    """Evaluate with per-group center crop/pad patches."""
    group_metrics = defaultdict(lambda: {"dice": [], "recall": [], "precision": []})
    all_metrics = []

    for group in GROUP_ORDER:
        ds = ReXStage2ROIPatchDataset(
            manifest_path=manifest_path,
            data_root=data_root,
            target_shape=group_patch_shapes[group],
            augment=False,
            group_filter=group,
            crop_mode="center",
        )
        if len(ds) == 0:
            continue

        loader = DataLoader(
            ds,
            batch_size=max(1, group_batches[group]),
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

        seen = 0
        with torch.no_grad():
            for batch in loader:
                x = batch["image"].to(device, non_blocking=True)
                y = batch["mask"].to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                    logits = model(x)

                for j in range(x.size(0)):
                    m = batch_segmentation_metrics(logits[j:j + 1], y[j:j + 1])
                    update_metric_buckets(group_metrics, all_metrics, group, m)

                seen += x.size(0)
                if progress_interval > 0 and seen % progress_interval == 0:
                    print(f"    {tag}/{group}: {seen}/{len(ds)}", flush=True)

    results, overall = summarize_metrics(group_metrics, all_metrics)
    return results, overall, 0


def window_starts(length: int, patch: int, stride: int):
    if length <= patch:
        return [0]
    starts = list(range(0, length - patch + 1, stride))
    last = length - patch
    if starts[-1] != last:
        starts.append(last)
    return starts


def eval_sliding_window(
    model,
    manifest_path,
    data_root,
    tag,
    device,
    group_patch_shapes,
    overlap=0.5,
    max_voxels=0,
    progress_interval=20,
    norm_scope="global",
    skip_oom=True,
):
    """Evaluate full ROI by averaging logits from per-group patch sliding windows."""
    rows = load_jsonl(manifest_path)
    data_root = Path(data_root)

    group_metrics = defaultdict(lambda: {"dice": [], "recall": [], "precision": []})
    all_metrics = []
    skipped = 0

    for idx, row in enumerate(rows, start=1):
        if progress_interval > 0 and idx % progress_interval == 0:
            print(f"    {tag}: {idx}/{len(rows)}", flush=True)

        group = get_group(tuple(row["roi_shape_hwd"]))
        ph, pw, pd = group_patch_shapes[group]

        image, mask = load_roi_pair(row, data_root)
        h, w, d = image.shape
        target_shape = (
            max(int(np.ceil(h / 32) * 32), ph),
            max(int(np.ceil(w / 32) * 32), pw),
            max(int(np.ceil(d / 32) * 32), pd),
        )

        if max_voxels > 0 and np.prod(target_shape) > max_voxels * 1e6:
            skipped += 1
            continue

        image_padded, (h0, w0, d0), (h_off, w_off, d_off) = pad_to_shape_centered(image, target_shape, cval=-1024.0)
        mask_padded, _, _ = pad_to_shape_centered(mask, target_shape, cval=0.0)

        if norm_scope == "global":
            image_tensor = prepare_image(image_padded)
        elif norm_scope == "input":
            image_tensor = clip_image(image)
            image_tensor = normalize_image(image_tensor)
            image_tensor, _, _ = pad_to_shape_centered(image_tensor, target_shape, cval=0.0)
        elif norm_scope == "window":
            image_tensor = clip_image(image_padded)
        else:
            raise ValueError("--norm-scope must be global, input, or window")

        image_tensor = np.transpose(image_tensor, (2, 1, 0))  # HWD -> DWH
        image_tensor = torch.from_numpy(image_tensor.copy()).float().unsqueeze(0).unsqueeze(0)
        mask_tensor = np.transpose(mask_padded, (2, 1, 0))
        mask_tensor = torch.from_numpy(mask_tensor.copy()).long().unsqueeze(0)

        x = image_tensor.to(device)
        y = mask_tensor.to(device)
        _, _, dd, ww, hh = x.shape

        patch_d, patch_w, patch_h = pd, pw, ph
        stride_d = max(1, int(round(patch_d * (1.0 - overlap))))
        stride_w = max(1, int(round(patch_w * (1.0 - overlap))))
        stride_h = max(1, int(round(patch_h * (1.0 - overlap))))

        d_starts = window_starts(dd, patch_d, stride_d)
        w_starts = window_starts(ww, patch_w, stride_w)
        h_starts = window_starts(hh, patch_h, stride_h)

        logits_sum = torch.zeros((1, 2, dd, ww, hh), dtype=torch.float32, device=device)
        counts = torch.zeros((1, 1, dd, ww, hh), dtype=torch.float32, device=device)

        try:
            with torch.no_grad():
                for ds in d_starts:
                    for ws in w_starts:
                        for hs in h_starts:
                            patch = x[:, :, ds:ds + patch_d, ws:ws + patch_w, hs:hs + patch_h]
                            if norm_scope == "window":
                                mean = patch.mean()
                                std = patch.std()
                                patch = patch - mean
                                if float(std.item()) > 0:
                                    patch = patch / std
                            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                                patch_logits = model(patch)
                            logits_sum[:, :, ds:ds + patch_d, ws:ws + patch_w, hs:hs + patch_h] += patch_logits.float()
                            counts[:, :, ds:ds + patch_d, ws:ws + patch_w, hs:hs + patch_h] += 1.0
        except torch.OutOfMemoryError:
            if not skip_oom:
                raise
            skipped += 1
            print(
                f"    skipped OOM: {row.get('component_sample_id', '<unknown>')} "
                f"group={group} padded_shape={image_padded.shape} windows={len(d_starts) * len(w_starts) * len(h_starts)}",
                flush=True,
            )
            del x, y, logits_sum, counts
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        logits = logits_sum / counts.clamp_min(1.0)
        logits_cropped = logits[:, :, d_off:d_off + d0, w_off:w_off + w0, h_off:h_off + h0]
        y_cropped = y[:, d_off:d_off + d0, w_off:w_off + w0, h_off:h_off + h0]

        m = batch_segmentation_metrics(logits_cropped, y_cropped)
        update_metric_buckets(group_metrics, all_metrics, group, m)

    results, overall = summarize_metrics(group_metrics, all_metrics)
    return results, overall, skipped


def print_results(title, group_metrics, overall, skipped):
    print(f"\n========== {title} ==========")
    if skipped:
        print(f"  (skipped {skipped} too-large ROI(s))")
    if overall["n"] == 0:
        print("  (no rows found)")
        return
    for g in GROUP_ORDER:
        if g in group_metrics:
            b = group_metrics[g]
            print(f"  {g:>7}: n={b['n']:5d}  Dice={b['dice']:.4f}  Recall={b['recall']:.4f}  Prec={b['precision']:.4f}")
    print(f"  {'OVERALL':>7}: n={overall['n']:5d}  Dice={overall['dice']:.4f}  Recall={overall['recall']:.4f}  Prec={overall['precision']:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--mode", choices=["full-roi", "fixed-patch", "sliding-window"], default="full-roi")
    p.add_argument("--splits", nargs="+", default=["eval"], choices=["eval"],
                   help="Data tags to evaluate.")
    p.add_argument("--overlap", type=float, default=0.5,
                   help="Sliding-window overlap fraction, only used by --mode sliding-window.")
    p.add_argument("--norm-scope", choices=["global", "window", "input"], default=None,
                   help=(
                       "Normalization ablation. full-roi: input=current padded input stats, "
                       "global=ROI stats before pad. sliding-window: global=current padded input stats, "
                       "window=per-window stats, input=ROI stats before pad."
                   ))
    p.add_argument("--no-full-roi-min-patch", action="store_true",
                   help="In full-roi mode, only pad to 32x multiple instead of at least the checkpoint's per-group patch shape.")
    p.add_argument("--fail-on-oom", action="store_true",
                   help="Raise CUDA OOM instead of skipping that ROI and continuing.")
    p.add_argument("--progress-interval", type=int, default=100,
                   help="Print progress every N samples (0 disables progress prints).")
    p.add_argument("--max-voxels", type=float, default=0,
                   help="Skip ROIs whose padded volume (M voxels) exceeds this (0=no limit).  e.g. --max-voxels 15 for 12GB GPU.")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = create_stunet_model(variant="STU-Net-S", pretrained_dataset="TotalSegmentator", out_channels=2)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    group_patch_shapes = load_group_patch_shapes(ckpt)
    group_batches = load_group_batches(ckpt)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"  eval mode: {args.mode}")
    print(f"  tags: {', '.join(args.splits)}")
    print(f"  group patch shapes: {group_patch_shapes}")
    norm_scope = args.norm_scope
    if norm_scope is None:
        norm_scope = "input" if args.mode == "full-roi" else "global"
    print(f"  norm scope: {norm_scope}")
    full_roi_min_patch = args.mode == "full-roi" and not args.no_full_roi_min_patch
    if args.mode == "full-roi":
        print(f"  full-roi min patch: {'enabled' if full_roi_min_patch else 'disabled'}")

    for tag in args.splits:
        if args.mode == "full-roi":
            group_metrics, overall, skipped = eval_full_roi(
                model,
                args.manifest,
                args.data_root,
                tag,
                device,
                group_patch_shapes,
                args.max_voxels,
                args.progress_interval,
                norm_scope,
                full_roi_min_patch,
                not args.fail_on_oom,
            )
        elif args.mode == "fixed-patch":
            group_metrics, overall, skipped = eval_fixed_patch(
                model, args.manifest, args.data_root, tag, device, group_patch_shapes, group_batches, args.progress_interval
            )
        else:
            group_metrics, overall, skipped = eval_sliding_window(
                model,
                args.manifest,
                args.data_root,
                tag,
                device,
                group_patch_shapes,
                args.overlap,
                args.max_voxels,
                args.progress_interval,
                norm_scope,
                not args.fail_on_oom,
            )
        print_results(tag.upper(), group_metrics, overall, skipped)

    print("\nDone.")


if __name__ == "__main__":
    main()
