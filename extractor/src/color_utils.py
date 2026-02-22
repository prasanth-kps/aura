from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def classify_bgr_image_color(image_bgr: np.ndarray) -> str | None:
    if image_bgr.size == 0:
        return None

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h_mean = float(np.mean(hsv[:, :, 0]))  # 0..179
    s_mean = float(np.mean(hsv[:, :, 1]))  # 0..255
    v_mean = float(np.mean(hsv[:, :, 2]))  # 0..255

    # Brightness/saturation-based neutral colors first.
    if v_mean < 55:
        return "black"
    if s_mean < 35:
        if v_mean > 205:
            return "white"
        return "gray"

    # Chromatic colors.
    if 10 <= h_mean < 25 and v_mean < 160:
        return "brown"
    if h_mean < 10 or h_mean >= 170:
        return "red"
    if h_mean < 20:
        return "orange"
    if h_mean < 35:
        return "yellow"
    if h_mean < 85:
        return "green"
    if h_mean < 130:
        return "blue"
    if h_mean < 165:
        return "purple"
    return "red"


def detect_color_from_image_path(image_path: str | Path) -> str | None:
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    return classify_bgr_image_color(image)
