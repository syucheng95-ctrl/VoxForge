import json
import random
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


GROUP_ORDER = ["small", "medium", "large", "xlarge"]

GROUP_PATCH_SHAPES = {
    "small": (64, 64, 32),
    "medium": (96, 96, 48),
    "large": (160, 160, 48),
    "xlarge": (192, 192, 96),
}


def get_group(hwd: tuple[int, int, int]) -> str:
    """Classify an ROI into a size group based on its original H×W×D dimensions."""
    h, w, d = hwd
    if h <= 64 and w <= 64 and d <= 32:
        return "small"
    if d > 48:
        return "xlarge"
    if h > 96 or w > 96:
        return "large"
    return "medium"


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class ReXStage2ROIPatchDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        data_root: str | Path,
        intensity_clip: tuple[float, float] | None = (-1000.0, 400.0),
        normalize: bool = True,
        tensor_order: str = "zyx",
        target_shape: tuple[int, int, int] | None = None,
        augment: bool = False,
        group_filter: Optional[str] = None,
        crop_mode: str = "center",
        foreground_crop_prob: float = 0.0,
    ):
        self.manifest_path = Path(manifest_path)
        self.data_root = Path(data_root)
        self.intensity_clip = intensity_clip
        self.normalize = normalize
        self.tensor_order = tensor_order
        self.target_shape = target_shape
        self.augment = augment
        self.group_filter = group_filter
        self.crop_mode = crop_mode
        self.foreground_crop_prob = float(foreground_crop_prob)

        rows = load_jsonl(self.manifest_path)
        if group_filter is not None:
            rows = [row for row in rows if get_group(tuple(row["roi_shape_hwd"])) == group_filter]

        self.rows = rows

        if self.target_shape is None:
            raise ValueError("target_shape must be provided")

    def __len__(self):
        return len(self.rows)

    def _load_nifti(self, path: Path) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(f"NIfTI file is missing: {path}")
        if path.stat().st_size == 0:
            raise nib.filebasedimages.ImageFileError(f"Empty file: '{path}'")
        return np.asarray(nib.load(str(path)).dataobj)

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32, copy=False)
        if self.intensity_clip is not None:
            lo, hi = self.intensity_clip
            image = np.clip(image, lo, hi)
        if self.normalize:
            mean = float(image.mean())
            std = float(image.std())
            image = image - mean
            if std > 0:
                image = image / std
        return image

    def _to_tensor(self, volume: np.ndarray, is_mask: bool) -> torch.Tensor:
        if self.tensor_order == "zyx":
            volume = np.transpose(volume, (2, 1, 0))
        elif self.tensor_order != "xyz":
            raise ValueError("tensor_order must be 'zyx' or 'xyz'")

        if is_mask:
            return torch.from_numpy(volume.copy()).long()
        return torch.from_numpy(volume[None, ...].copy()).float()

    def _center_crop_or_pad(self, volume: np.ndarray, cval: float = 0.0) -> np.ndarray:
        th, tw, td = self.target_shape
        out = np.full((th, tw, td), cval, dtype=volume.dtype)
        sh, sw, sd = volume.shape
        ch, cw, cd = min(th, sh), min(tw, sw), min(td, sd)
        sh1, sw1, sd1 = (sh - ch) // 2, (sw - cw) // 2, (sd - cd) // 2
        th1, tw1, td1 = (th - ch) // 2, (tw - cw) // 2, (td - cd) // 2
        out[th1:th1 + ch, tw1:tw1 + cw, td1:td1 + cd] = volume[sh1:sh1 + ch, sw1:sw1 + cw, sd1:sd1 + cd]
        return out

    def _crop_or_pad_with_params(
        self,
        volume: np.ndarray,
        starts_offsets_lengths: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
        cval: float = 0.0,
    ) -> np.ndarray:
        th, tw, td = self.target_shape
        out = np.full((th, tw, td), cval, dtype=volume.dtype)
        (h_start, h_off, ch), (w_start, w_off, cw), (d_start, d_off, cd) = starts_offsets_lengths
        out[h_off:h_off + ch, w_off:w_off + cw, d_off:d_off + cd] = \
            volume[h_start:h_start + ch, w_start:w_start + cw, d_start:d_start + cd]
        return out

    def _random_crop_or_pad_params(self, shape: tuple[int, int, int]):
        """Generate random crop params once so image and mask stay aligned."""
        th, tw, td = self.target_shape
        sh, sw, sd = shape

        if sh <= th:
            h_start, h_off, ch = 0, (th - sh) // 2, sh
        else:
            h_start, h_off, ch = random.randint(0, sh - th), 0, th
        if sw <= tw:
            w_start, w_off, cw = 0, (tw - sw) // 2, sw
        else:
            w_start, w_off, cw = random.randint(0, sw - tw), 0, tw
        if sd <= td:
            d_start, d_off, cd = 0, (td - sd) // 2, sd
        else:
            d_start, d_off, cd = random.randint(0, sd - td), 0, td

        return (h_start, h_off, ch), (w_start, w_off, cw), (d_start, d_off, cd)

    def _random_crop_or_pad_pair(self, image: np.ndarray, mask: np.ndarray):
        params = self._random_crop_or_pad_params(image.shape)
        return (
            self._crop_or_pad_with_params(image, params, cval=-1024.0),
            self._crop_or_pad_with_params(mask, params, cval=0.0),
        )

    def _foreground_crop_or_pad_params(self, mask: np.ndarray):
        """Generate crop params that include a randomly selected foreground voxel."""
        fg = np.argwhere(mask > 0)
        if fg.size == 0:
            return self._random_crop_or_pad_params(mask.shape)

        th, tw, td = self.target_shape
        sh, sw, sd = mask.shape
        ch0, cw0, cd0 = fg[random.randrange(len(fg))]

        def choose_start(size: int, target: int, coord: int):
            if size <= target:
                return 0, (target - size) // 2, size
            start_min = max(0, coord - target + 1)
            start_max = min(coord, size - target)
            if start_min > start_max:
                return random.randint(0, size - target), 0, target
            return random.randint(start_min, start_max), 0, target

        return (
            choose_start(sh, th, int(ch0)),
            choose_start(sw, tw, int(cw0)),
            choose_start(sd, td, int(cd0)),
        )

    def _foreground_random_crop_or_pad_pair(self, image: np.ndarray, mask: np.ndarray):
        if random.random() >= self.foreground_crop_prob:
            return self._random_crop_or_pad_pair(image, mask)
        params = self._foreground_crop_or_pad_params(mask)
        return (
            self._crop_or_pad_with_params(image, params, cval=-1024.0),
            self._crop_or_pad_with_params(mask, params, cval=0.0),
        )

    def _augment_geometry(self, image: np.ndarray, mask: np.ndarray):
        """In-place random flip + 90° rotation on H-W plane. Applied to both image and mask."""
        for axis in range(3):
            if random.random() < 0.5:
                image = np.flip(image, axis=axis)
                mask = np.flip(mask, axis=axis)

        k = random.randint(0, 3)
        if k > 0:
            image = np.rot90(image, k=k, axes=(0, 1))
            mask = np.rot90(mask, k=k, axes=(0, 1))

        return image, mask

    def _augment_intensity(self, image: np.ndarray) -> np.ndarray:
        """Gaussian noise + contrast jitter + gamma correction. Applied to normalized image."""
        if random.random() < 0.5:
            image = image + np.random.normal(0, 0.03, image.shape).astype(np.float32)
        if random.random() < 0.5:
            image = image * float(np.random.uniform(0.85, 1.15))
        if random.random() < 0.3:
            # Gamma: shift positive, apply gamma, shift back
            vmin = float(image.min())
            shifted = image - vmin + 0.01
            gamma = np.random.uniform(0.7, 1.4)
            image = np.power(shifted, gamma)
            image = image + vmin
        return image

    def __getitem__(self, index: int):
        row = self.rows[index]
        image_path = self.data_root / row["roi_image_path"]
        mask_path = self.data_root / row["roi_mask_path"]

        image = self._load_nifti(image_path)
        mask = self._load_nifti(mask_path).astype(np.uint8)
        if image.shape != mask.shape:
            raise ValueError(
                f"Image/mask shape mismatch for {row['component_sample_id']}: "
                f"image={image.shape} mask={mask.shape} "
                f"image_path={image_path} mask_path={mask_path}"
            )
        mask = (mask > 0).astype(np.uint8)

        if self.crop_mode == "random":
            image, mask = self._random_crop_or_pad_pair(image, mask)
        elif self.crop_mode == "foreground_random":
            image, mask = self._foreground_random_crop_or_pad_pair(image, mask)
        else:
            image = self._center_crop_or_pad(image, cval=-1024.0)
            mask = self._center_crop_or_pad(mask, cval=0.0)

        if self.augment:
            image, mask = self._augment_geometry(image, mask)

        image = self._prepare_image(image)

        if self.augment:
            image = self._augment_intensity(image)

        return {
            "image": self._to_tensor(image, is_mask=False),
            "mask": self._to_tensor(mask, is_mask=True),
            "id": row["component_sample_id"],
            "voxel_count": row["voxel_count"],
            "roi_image_path": str(image_path),
            "roi_mask_path": str(mask_path),
            "patch_size_xyz": list(self.target_shape),
        }
