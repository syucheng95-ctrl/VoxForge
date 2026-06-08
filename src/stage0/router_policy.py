import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


CATEGORY_LABELS = [
    "1a",
    "1b",
    "1c",
    "1d",
    "1e",
    "1f",
    "2a",
    "2b",
    "2c",
    "2d",
    "2e",
    "2f",
    "2g",
    "2h",
]

CATEGORY_NAMES = {
    "1a": "bronchial_wall_thickening",
    "1b": "bronchiectasis",
    "1c": "emphysema",
    "1d": "interlobular_septal_thickening",
    "1e": "micronodule",
    "1f": "other_non_focal",
    "2a": "linear_scar",
    "2b": "atelectasis_consolidation",
    "2c": "ground_glass_opacity",
    "2d": "nodule_mass",
    "2e": "pleural_effusion",
    "2f": "honeycombing",
    "2g": "pneumothorax",
    "2h": "other_focal",
}

TIGHTNESS_LABELS = ["conservative", "moderate", "aggressive"]

CATEGORY_TO_TIGHTNESS = {
    "1a": "conservative",
    "1b": "conservative",
    "1c": "conservative",
    "1d": "conservative",
    "1e": "conservative",
    "1f": "conservative",
    "2e": "conservative",
    "2f": "conservative",
    "2g": "conservative",
    "2a": "moderate",
    "2b": "moderate",
    "2c": "moderate",
    "2h": "moderate",
    "2d": "aggressive",
}

CATEGORY_TO_ID = {label: idx for idx, label in enumerate(CATEGORY_LABELS)}
ID_TO_CATEGORY = {idx: label for label, idx in CATEGORY_TO_ID.items()}
TIGHTNESS_TO_ID = {label: idx for idx, label in enumerate(TIGHTNESS_LABELS)}
ID_TO_TIGHTNESS = {idx: label for label, idx in TIGHTNESS_TO_ID.items()}

LEFT_RE = re.compile(r"\b(left|lt\.?|left-sided|leftside)\b", re.I)
RIGHT_RE = re.compile(r"\b(right|rt\.?|right-sided|rightside)\b", re.I)
BOTH_RE = re.compile(
    r"\b(bilateral|bilaterally|both|diffuse|multifocal|scattered|throughout)\b",
    re.I,
)
NON_EXCLUSIVE_SIDE_RE = re.compile(
    r"\b(particularly|especially|predominantly|predominant|more\s+prominent|"
    r"most\s+prominent|greater\s+on|worse\s+on|mainly|primarily)\b",
    re.I,
)
CONSERVATIVE_RISK_RE = re.compile(
    r"(pleural\s+effusion|pneumothorax|emphysem|hyperaerat|hyperinflat|"
    r"bronchiect|bronchial\s+wall|peribronchial|septal\s+thicken|honeycomb|"
    r"interlobular\s+septal|centrilobular|centriacinar|centracinar|tree[-\s]in[-\s]bud|"
    r"calcific|calcified|\bnodules\b|"
    r"\bdiffuse\b|\bscattered\b|\bmultifocal\b)",
    re.I,
)
MODERATE_CAP_RE = re.compile(
    r"(atelectasis|ground[-\s]glass|\bggo\b|consolidation|increased\s+density|"
    r"around\s+the\s+mass|adjacent\s+to\s+the\s+mass)",
    re.I,
)


@dataclass
class RouterDecision:
    raw_pred_category: str
    pred_category: str
    category_confidence: float
    category_tightness: str
    category_postprocess_reason: Optional[str]
    pred_tightness: str
    tightness_confidence: float
    laterality: str
    final_tightness: str
    final_policy: str
    fail_open_reason: Optional[str]


def category_to_tightness(category: str) -> str:
    return CATEGORY_TO_TIGHTNESS.get(category, "conservative")


def tightness_rank(tightness: str) -> int:
    return TIGHTNESS_TO_ID.get(tightness, 0)


def more_conservative(tightness: str, steps: int = 1) -> str:
    rank = max(0, tightness_rank(tightness) - steps)
    return ID_TO_TIGHTNESS[rank]


def min_tightness(a: str, b: str) -> str:
    return a if tightness_rank(a) <= tightness_rank(b) else b


def extract_laterality(prompt: str) -> str:
    text = prompt or ""
    has_left = bool(LEFT_RE.search(text))
    has_right = bool(RIGHT_RE.search(text))
    has_both = bool(BOTH_RE.search(text))
    if has_both or (has_left and has_right):
        return "bilateral"
    if (has_left or has_right) and NON_EXCLUSIVE_SIDE_RE.search(text):
        return "unknown"
    if has_left:
        return "left"
    if has_right:
        return "right"
    return "unknown"


def build_crop_policy(final_tightness: str, laterality: str) -> str:
    if laterality not in {"left", "right"}:
        return "both_conservative"
    if final_tightness == "conservative":
        return "both_conservative"
    return f"{laterality}_{final_tightness}"


def has_conservative_risk_term(prompt: str) -> bool:
    return bool(CONSERVATIVE_RISK_RE.search(prompt or ""))


def has_moderate_cap_term(prompt: str) -> bool:
    return bool(MODERATE_CAP_RE.search(prompt or ""))


def postprocess_category(
    raw_category: str,
    prompt: str,
    category_confidence: float,
    confidence_threshold: float = 0.80,
) -> Tuple[str, Optional[str]]:
    if category_confidence >= confidence_threshold:
        return raw_category, None

    text = prompt or ""
    lower = text.lower()

    high_precision_rules = [
        (r"\bpleural\s+effusion\b|\beffusion\b", "2e", "rule_pleural_effusion"),
        (r"\bpneumothorax\b", "2g", "rule_pneumothorax"),
        (r"\bbronchiectasis\b|\bbronchiectatic\b", "1b", "rule_bronchiectasis"),
        (r"\bemphysema\b|\bemphysematous\b", "1c", "rule_emphysema"),
        (r"\bhyperaeration\b|\bhyperinflation\b", "1c", "rule_hyperaeration"),
        (r"\bhoneycomb(?:ing)?\b", "2f", "rule_honeycombing"),
        (r"bronchial\s+wall\s+thicken|peribronchial\s+thicken", "1a", "rule_bronchial_wall_thickening"),
        (r"interlobular\s+septal\s+thicken|\bseptal\s+thicken", "1d", "rule_septal_thickening"),
        (r"tree[-\s]in[-\s]bud", "1e", "rule_tree_in_bud"),
        (r"centrilobular|centriacinar|centracinar", "1e", "rule_centrilobular_micronodule"),
    ]
    for pattern, category, reason in high_precision_rules:
        if re.search(pattern, lower, re.I):
            return category, reason

    return raw_category, None


def argmax_with_conf(probs: Sequence[float], labels: Sequence[str]) -> Tuple[str, float]:
    best_i = max(range(len(probs)), key=lambda i: float(probs[i]))
    return labels[best_i], float(probs[best_i])


def apply_fail_open(
    category_probs: Sequence[float],
    tightness_probs: Sequence[float],
    prompt: str,
    category_threshold: float = 0.60,
    tightness_threshold: float = 0.60,
) -> RouterDecision:
    raw_pred_category, category_conf = argmax_with_conf(category_probs, CATEGORY_LABELS)
    pred_category = raw_pred_category
    postprocess_reason = None
    pred_tightness, tightness_conf = argmax_with_conf(tightness_probs, TIGHTNESS_LABELS)
    category_tightness = category_to_tightness(pred_category)

    final_tightness = category_tightness
    reasons: List[str] = []

    if category_conf < category_threshold:
        final_tightness = more_conservative(final_tightness)
        reasons.append(f"low_category_confidence:{category_conf:.3f}")

    if tightness_conf >= tightness_threshold:
        safer = min_tightness(final_tightness, pred_tightness)
        if safer != final_tightness:
            reasons.append(
                f"tightness_head_more_conservative:{pred_tightness}:{tightness_conf:.3f}"
            )
            final_tightness = safer
    elif tightness_rank(pred_tightness) < tightness_rank(final_tightness):
        final_tightness = pred_tightness
        reasons.append(f"low_tightness_confidence_but_conservative:{tightness_conf:.3f}")

    laterality = extract_laterality(prompt)
    if laterality not in {"left", "right"} and final_tightness != "conservative":
        final_tightness = "conservative"
        reasons.append(f"non_unilateral_laterality:{laterality}")

    if has_conservative_risk_term(prompt) and final_tightness != "conservative":
        final_tightness = "conservative"
        reasons.append("conservative_risk_term")

    if has_moderate_cap_term(prompt) and final_tightness == "aggressive":
        final_tightness = "moderate"
        reasons.append("moderate_cap_term")

    final_policy = build_crop_policy(final_tightness, laterality)

    return RouterDecision(
        raw_pred_category=raw_pred_category,
        pred_category=pred_category,
        category_confidence=category_conf,
        category_tightness=category_tightness,
        category_postprocess_reason=postprocess_reason,
        pred_tightness=pred_tightness,
        tightness_confidence=tightness_conf,
        laterality=laterality,
        final_tightness=final_tightness,
        final_policy=final_policy,
        fail_open_reason=";".join(reasons) if reasons else None,
    )


def labels_for_category(category: str) -> Tuple[int, int]:
    category_id = CATEGORY_TO_ID[category]
    tightness_id = TIGHTNESS_TO_ID[category_to_tightness(category)]
    return category_id, tightness_id


def policy_maps() -> Dict[str, Dict]:
    return {
        "category_to_id": CATEGORY_TO_ID,
        "id_to_category": ID_TO_CATEGORY,
        "tightness_to_id": TIGHTNESS_TO_ID,
        "id_to_tightness": ID_TO_TIGHTNESS,
        "category_to_tightness": CATEGORY_TO_TIGHTNESS,
        "category_names": CATEGORY_NAMES,
    }
