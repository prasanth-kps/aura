from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", text.strip().lower())


def prepare_output_paths(out_root: Path, video_id: str) -> tuple[Path, Path]:
    crops_dir = out_root / "crops" / video_id
    memory_dir = out_root / "memory"
    memory_path = memory_dir / f"{video_id}.events.jsonl"

    crops_dir.mkdir(parents=True, exist_ok=True)
    memory_dir.mkdir(parents=True, exist_ok=True)
    memory_path.touch(exist_ok=True)
    return crops_dir, memory_path


def write_thumbnail(
    frame_bgr: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    out_path: Path,
    thumb_size: int = 50,
    padding: int = 8,
    jpeg_quality: int = 75,
) -> bool:
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    if x2 <= x1 or y2 <= y1:
        return False

    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return False

    thumb = cv2.resize(crop, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA)
    ok = cv2.imwrite(str(out_path), thumb, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    return bool(ok)


def append_event(memory_path: Path, event: dict[str, Any]) -> None:
    with memory_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=True) + "\n")
