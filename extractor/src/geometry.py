from __future__ import annotations

import math
from typing import Optional, Tuple

from .detector_qaihub import Detection


def box_center(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def point_inside(point: Tuple[float, float], box: Tuple[int, int, int, int]) -> bool:
    px, py = point
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


def choose_anchor(mobile: Detection, anchors: list[Detection]) -> Optional[Detection]:
    if not anchors:
        return None

    mx, my = box_center(mobile.bbox)
    best: Optional[Detection] = None
    best_score = float("inf")

    for anchor in anchors:
        ax, ay = box_center(anchor.bbox)
        dist = math.hypot(mx - ax, my - ay)

        # Prefer anchor that contains the mobile center.
        if point_inside((mx, my), anchor.bbox):
            dist *= 0.25

        if dist < best_score:
            best_score = dist
            best = anchor

    return best


def relation_to_anchor(
    mobile_box: Tuple[int, int, int, int],
    anchor_box: Tuple[int, int, int, int],
) -> str:
    mx, my = box_center(mobile_box)
    ax1, ay1, ax2, ay2 = anchor_box

    if point_inside((mx, my), anchor_box):
        return "on_top_of"

    if mx < ax1:
        return "left_of"
    if mx > ax2:
        return "right_of"
    if my < ay1:
        return "above"
    if my > ay2:
        return "below"
    return "near"
