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


def resolve_frame_memory_path(out_root: Path, video_id: str) -> Path:
    memory_dir = out_root / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    frame_memory_path = memory_dir / f"{video_id}.frames.jsonl"
    frame_memory_path.touch(exist_ok=True)
    return frame_memory_path


def resolve_frames_dir(out_root: Path, video_id: str) -> Path:
    frames_dir = out_root / "frames" / video_id
    frames_dir.mkdir(parents=True, exist_ok=True)
    return frames_dir


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


def append_frame_record(frame_memory_path: Path, frame_record: dict[str, Any]) -> None:
    with frame_memory_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(frame_record, ensure_ascii=True) + "\n")


def write_frame_image(
    frame_bgr: np.ndarray,
    out_path: Path,
    max_width: int = 960,
    jpeg_quality: int = 70,
) -> bool:
    h, w = frame_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return False

    target = frame_bgr
    if max_width > 0 and w > max_width:
        ratio = max_width / float(w)
        new_h = max(1, int(round(h * ratio)))
        target = cv2.resize(frame_bgr, (max_width, new_h), interpolation=cv2.INTER_AREA)

    ok = cv2.imwrite(str(out_path), target, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    return bool(ok)
