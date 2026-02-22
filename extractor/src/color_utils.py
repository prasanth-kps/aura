from __future__ import annotations

import colorsys
from pathlib import Path

import numpy as np
try:
    import cv2
except Exception:
    cv2 = None


def classify_bgr_image_color(image_bgr: np.ndarray) -> str | None:
    if image_bgr.size == 0:
        return None

    if cv2 is not None:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        h_mean = float(np.mean(hsv[:, :, 0]))  # 0..179
        s_mean = float(np.mean(hsv[:, :, 1]))  # 0..255
        v_mean = float(np.mean(hsv[:, :, 2]))  # 0..255
    else:
        mean_bgr = np.mean(image_bgr.reshape(-1, 3), axis=0)
        b = float(mean_bgr[0]) / 255.0
        g = float(mean_bgr[1]) / 255.0
        r = float(mean_bgr[2]) / 255.0
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        h_mean = h * 179.0
        s_mean = s * 255.0
        v_mean = v * 255.0

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
    if cv2 is not None:
        image = cv2.imread(str(image_path))
    else:
        try:
            from PIL import Image

            rgb = np.array(Image.open(str(image_path)).convert("RGB"))
            image = rgb[:, :, ::-1]
        except Exception:
            image = None
    if image is None:
        return None
    return classify_bgr_image_color(image)
