from __future__ import annotations

import gc
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2

from .anchors import ANCHOR_CLASSES
from .detector_qaihub import QualcommYoloDetector
from .geometry import choose_anchor, relation_to_anchor
from .storage import append_event, prepare_output_paths, slugify, write_thumbnail


@dataclass
class IngestOptions:
    video_path: Path
    out_root: Path = Path("data")
    prefer_model: str = "yolov11"  # yolov11 -> fallback yolov8
    conf_threshold: float = 0.45
    iou_threshold: float = 0.60
    thumb_size: int = 50
    crop_padding: int = 8
    jpeg_quality: int = 75
    anchor_classes: set[str] = field(default_factory=lambda: set(ANCHOR_CLASSES))
    include_labels: Optional[set[str]] = None
    exclude_labels: set[str] = field(default_factory=set)
    reset_output: bool = False


def run_video_ingestion(opts: IngestOptions) -> dict[str, int]:
    if not opts.video_path.exists():
        raise FileNotFoundError(f"Video not found: {opts.video_path}")

    video_id = opts.video_path.stem
    crops_dir, memory_path = prepare_output_paths(opts.out_root, video_id)

    if opts.reset_output:
        # Clean prior outputs for deterministic reruns.
        if memory_path.exists():
            memory_path.unlink()
        for crop_file in crops_dir.glob("*.jpg"):
            try:
                crop_file.unlink()
            except OSError:
                pass
        # Recreate empty log file after cleanup.
        memory_path.touch(exist_ok=True)

    detector = QualcommYoloDetector(
        prefer_model=opts.prefer_model,
        conf_threshold=opts.conf_threshold,
        iou_threshold=opts.iou_threshold,
    )

    cap = cv2.VideoCapture(str(opts.video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {opts.video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    last_second = -1
    frame_idx = 0
    processed_seconds = 0
    event_count = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if pos_ms > 0:
                second = int(pos_ms // 1000)
            else:
                second = int(frame_idx / fps)
            frame_idx += 1

            # Exactly 1 frame per second.
            if second == last_second:
                continue
            last_second = second
            processed_seconds += 1

            detections = detector.predict(frame)

            if opts.include_labels:
                detections = [d for d in detections if d.label in opts.include_labels]
            if opts.exclude_labels:
                detections = [d for d in detections if d.label not in opts.exclude_labels]

            anchors = [d for d in detections if d.label in opts.anchor_classes]
            mobile_items = [d for d in detections if d.label not in opts.anchor_classes]

            for local_idx, mobile in enumerate(mobile_items):
                anchor = choose_anchor(mobile, anchors)

                if anchor is None:
                    relation = "in_scene"
                    anchor_label = None
                    context = "in_scene"
                else:
                    relation = relation_to_anchor(mobile.bbox, anchor.bbox)
                    anchor_label = anchor.label
                    context = f"{relation} {anchor.label}"

                thumb_name = f"s{second:06d}_{local_idx:03d}_{slugify(mobile.label)}.jpg"
                thumb_path = crops_dir / thumb_name

                saved = write_thumbnail(
                    frame_bgr=frame,
                    bbox_xyxy=mobile.bbox,
                    out_path=thumb_path,
                    thumb_size=opts.thumb_size,
                    padding=opts.crop_padding,
                    jpeg_quality=opts.jpeg_quality,
                )
                if not saved:
                    continue

                event = {
                    "video_id": video_id,
                    "video_second": second,
                    "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
                    "detector_model": detector.model_name,
                    "label": mobile.label,
                    "confidence": round(mobile.confidence, 4),
                    "bbox_xyxy": list(mobile.bbox),
                    "anchor_label": anchor_label,
                    "relation": relation,
                    "context": context,
                    "thumbnail_path": str(thumb_path),
                }
                append_event(memory_path, event)
                event_count += 1

            del frame
            if processed_seconds % 30 == 0:
                gc.collect()

    finally:
        cap.release()

    return {
        "processed_seconds": processed_seconds,
        "events_written": event_count,
    }
