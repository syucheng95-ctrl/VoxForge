import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np


LOBE_LABELS = {
    "left_upper_lobe": 10,
    "left_lower_lobe": 11,
    "right_upper_lobe": 12,
    "right_middle_lobe": 13,
    "right_lower_lobe": 14,
}

SIDE_TARGETS = {
    "left_lung": ["left_upper_lobe", "left_lower_lobe"],
    "right_lung": ["right_upper_lobe", "right_middle_lobe", "right_lower_lobe"],
    "both_lungs": [
        "left_upper_lobe",
        "left_lower_lobe",
        "right_upper_lobe",
        "right_middle_lobe",
        "right_lower_lobe",
    ],
}

BILATERAL_RE = re.compile(r"\b(bilateral|bilaterally|both|diffuse|throughout)\b", re.I)
LEFT_RE = re.compile(r"\b(left|lt\.?|left-sided)\b", re.I)
RIGHT_RE = re.compile(r"\b(right|rt\.?|right-sided)\b", re.I)
NON_EXCLUSIVE_SIDE_RE = re.compile(
    r"\b(particularly|especially|predominantly|predominant|more\s+prominent|"
    r"most\s+prominent|greater\s+on|worse\s+on|mainly|primarily)\b",
    re.I,
)


@dataclass
class AnatomyDecision:
    target: str
    lobe_names: List[str]
    reason: str


def bbox_from_mask(mask):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    h0, w0, d0 = coords.min(axis=0)
    h1, w1, d1 = coords.max(axis=0) + 1
    return [int(h0), int(h1), int(w0), int(w1), int(d0), int(d1)]


def margin_mm_to_vox(margin_mm, zooms):
    import math

    return [int(math.ceil(float(m) / float(z))) for m, z in zip(margin_mm, zooms)]


def expand_bbox(bbox, shape, margin_vox):
    if bbox is None:
        return None
    h0, h1, w0, w1, d0, d1 = bbox
    mh, mw, md = margin_vox
    H, W, D = shape
    return [
        max(0, h0 - mh),
        min(H, h1 + mh),
        max(0, w0 - mw),
        min(W, w1 + mw),
        max(0, d0 - md),
        min(D, d1 + md),
    ]


def _has(text: str, pattern: str) -> bool:
    return bool(re.search(pattern, text, re.I))


def parse_anatomy(prompt: str, laterality: Optional[str] = None) -> AnatomyDecision:
    text = (prompt or "").lower()
    if laterality == "unknown" and NON_EXCLUSIVE_SIDE_RE.search(text):
        return AnatomyDecision("both_lungs", SIDE_TARGETS["both_lungs"], "non_exclusive_side_fail_open")

    bilateral = bool(BILATERAL_RE.search(text)) or laterality == "bilateral"
    left = bool(LEFT_RE.search(text)) or laterality == "left"
    right = bool(RIGHT_RE.search(text)) or laterality == "right"

    mentions_upper = _has(text, r"\bupper\s+lobes?\b|\bapic(?:al|es)?\b|\bapicoposterior\b")
    mentions_middle = _has(text, r"\bmiddle\s+lobes?\b|\bmedial\s+segment\b")
    mentions_lower = _has(text, r"\blower\s+lobes?\b|\bbasal\b|\bposterobasal\b|\blaterobasal\b|\bsuperior\s+segment\b")
    mentions_lingula = _has(text, r"\blingula(?:r)?\b")

    if bilateral:
        lobes = []
        if mentions_upper:
            lobes += ["left_upper_lobe", "right_upper_lobe"]
        if mentions_middle:
            lobes += ["right_middle_lobe"]
        if mentions_lower:
            lobes += ["left_lower_lobe", "right_lower_lobe"]
        if lobes:
            return AnatomyDecision("bilateral_lobar", sorted(set(lobes)), "bilateral_lobe_phrase")
        return AnatomyDecision("both_lungs", SIDE_TARGETS["both_lungs"], "bilateral_or_diffuse")

    if mentions_lingula:
        return AnatomyDecision("left_upper_lobe", ["left_upper_lobe"], "lingula_maps_to_left_upper_lobe")

    if right:
        if mentions_middle:
            return AnatomyDecision("right_middle_lobe", ["right_middle_lobe"], "right_middle_lobe_phrase")
        if mentions_upper:
            return AnatomyDecision("right_upper_lobe", ["right_upper_lobe"], "right_upper_lobe_phrase")
        if mentions_lower:
            return AnatomyDecision("right_lower_lobe", ["right_lower_lobe"], "right_lower_lobe_phrase")
        return AnatomyDecision("right_lung", SIDE_TARGETS["right_lung"], "right_side_only")

    if left:
        if mentions_upper:
            return AnatomyDecision("left_upper_lobe", ["left_upper_lobe"], "left_upper_lobe_phrase")
        if mentions_lower:
            return AnatomyDecision("left_lower_lobe", ["left_lower_lobe"], "left_lower_lobe_phrase")
        return AnatomyDecision("left_lung", SIDE_TARGETS["left_lung"], "left_side_only")

    return AnatomyDecision("both_lungs", SIDE_TARGETS["both_lungs"], "unknown_location_fail_open")


class TotalSegLobeCache:
    def __init__(self, task: str = "total", fast: bool = True, device: str = "gpu"):
        try:
            from totalsegmentator.python_api import totalsegmentator
        except Exception as exc:
            raise RuntimeError(f"TotalSegmentator unavailable: {exc!r}") from exc
        self.totalsegmentator = totalsegmentator
        self.task = task
        self.fast = fast
        self.device = device
        self.cache: Dict[str, np.ndarray] = {}

    def get_seg(self, image_path: Path, shape) -> np.ndarray:
        key = str(image_path)
        if key in self.cache:
            return self.cache[key]
        seg_img = self.totalsegmentator(
            str(image_path),
            output=None,
            task=self.task,
            fast=self.fast,
            ml=True,
            device=self.device,
            quiet=True,
            nr_thr_resamp=1,
            nr_thr_saving=2,
        )
        seg = np.asanyarray(seg_img.dataobj).astype(np.uint8, copy=False)
        if tuple(seg.shape) != tuple(shape):
            raise RuntimeError(f"TotalSegmentator shape mismatch: seg={seg.shape}, target={shape}")
        self.cache[key] = seg
        return seg


def anatomy_to_bbox(
    decision: AnatomyDecision,
    seg: np.ndarray,
    shape,
    zooms,
    margin_mm,
):
    labels = [LOBE_LABELS[name] for name in decision.lobe_names]
    mask = np.isin(seg, labels)
    bbox = bbox_from_mask(mask)
    if bbox is None:
        return None
    return expand_bbox(bbox, shape, margin_mm_to_vox(margin_mm, zooms))


def anatomy_record(prompt: str, laterality: str, final_tightness: str):
    decision = parse_anatomy(prompt, laterality)
    return {
        "anatomy_target": decision.target,
        "anatomy_lobes": decision.lobe_names,
        "anatomy_reason": decision.reason,
        "anatomy_group": f"anatomy:{decision.target}:{final_tightness}",
    }
