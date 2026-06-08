from .config import CATEGORY_TO_EXPERT


def route_category(category: str) -> str:
    return CATEGORY_TO_EXPERT.get(category, "diffuse")
