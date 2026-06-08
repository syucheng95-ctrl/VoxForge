CATEGORY_LABELS = [
    "1a", "1b", "1c", "1d", "1e", "1f",
    "2a", "2b", "2c", "2d", "2e", "2f", "2g", "2h",
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

# Category → expert routing
CATEGORY_TO_EXPERT = {
    # Nodule detector: micronodule + nodule/mass
    "1e": "nodule_detector",
    "2d": "nodule_detector",

    # HU expert: focal/density-based categories
    "1c": "hu",
    "2b": "hu", "2c": "hu", "2e": "hu", "2g": "hu",
}
# Everything not in CATEGORY_TO_EXPERT → "diffuse"
# (1a/1b/1d/1f/2a/2f/2h — structural/diffuse/unclear categories)
# full_ct_voxtell removed from Stage0.5 routing; handled as post-Stage1 fallback

# HU windows per category
HU_THRESHOLDS = {
    # Emphysema / pneumothorax / honeycombing — air (< -900 HU)
    "1c": {"lower": None,  "upper": -950},
    "2g": {"lower": None,  "upper": -900},
    "2f": {"lower": None,  "upper": -900},
    # GGO — partial air/fluid (−750 to −300 HU)
    "2c": {"lower": -750,  "upper": -300},
    # Solid / soft tissue — nodules, scar, consolidation (> −100 HU)
    "2b": {"lower": -100,  "upper": None},
    "2a": {"lower": -100,  "upper": None},
    # Nodule / micronodule — wide catch (ISMI solid nodule > −450 HU)
    "2d": {"lower": -450,  "upper": None},
    "1e": {"lower": -450,  "upper": None},
    # Pleural effusion — water density (−20 to 40 HU)
    "2e": {"lower": -20,   "upper": 40},
}

# Connected-component connectivity (6 = face-neighbors)
CC_STRUCTURE = 6

# Morphology: min component voxels per category
MIN_COMPONENT_VOXELS = {
    "default": 125,
    "1c": 200,
    "1e": 8,
    "2b": 120,
    "2c": 100,
    "2d": 4,
    "2e": 200,
    "2g": 200,
}
MAX_ELONGATION = 5.0

# Proposal fusion
EXPAND_MARGIN_MM = [10, 10, 10]
UNION_IOU_THRESHOLD = 0.5

# Expert priority (lower = earlier in pipeline)
EXPERT_PRIORITY = {"hu": 0, "nodule_detector": 1, "diffuse": 2}

NODULE_DETECTOR_SCORE_THRESHOLDS = {
    "1e": 0.001,
    "2d": 0.001,
}
NODULE_DETECTOR_MAX_DETECTIONS = 80
