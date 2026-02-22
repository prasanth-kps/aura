from __future__ import annotations

import gc
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

import cv2

from .anchors import ANCHOR_CLASSES
from .color_utils import detect_color_from_image_path
from .detector_qaihub import QualcommYoloDetector
from .geometry import box_iou, center_distance, choose_anchor, relation_to_anchor
from .local_rag import DEFAULT_CAPTION_MODEL, caption_image
from .storage import (
    append_event,
    append_frame_record,
    prepare_output_paths,
    resolve_frame_memory_path,
    resolve_frames_dir,
    slugify,
    write_frame_image,
    write_thumbnail,
)


@dataclass
class IngestOptions:
    video_path: Path
    out_root: Path = Path("data")
    prefer_model: str = "yolov11"  # yolov11 -> fallback yolov8
    conf_threshold: float = 0.25
    iou_threshold: float = 0.60
    sample_fps: float = 3.0
    detector_input_size: int = 960
    thumb_size: int = 50
    crop_padding: int = 8
    jpeg_quality: int = 75
    anchor_classes: set[str] = field(default_factory=lambda: set(ANCHOR_CLASSES))
    include_labels: Optional[set[str]] = None
    exclude_labels: set[str] = field(default_factory=set)
    reset_output: bool = False
    track_max_gap_seconds: int = 3
    track_iou_threshold: float = 0.20
    track_center_dist_ratio: float = 0.12
    runtime: str = "auto"  # auto -> qnn if available, else cpu
    save_full_frames: bool = True
    frame_max_width: int = 960
    frame_jpeg_quality: int = 70
    caption_frames: bool = True
    caption_model: str = DEFAULT_CAPTION_MODEL
    caption_every_n_frames: int = 1


@dataclass
class TrackState:
    instance_id: str
    label: str
    bbox: tuple[int, int, int, int]
    last_second: int
    hit_count: int = 1


def _build_ingestion_run_id(video_id: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{slugify(video_id)}-{ts}-{uuid4().hex[:8]}"


def _new_instance_id(label: str, counters: dict[str, int], run_id: str) -> str:
    idx = counters.get(label, 0) + 1
    counters[label] = idx
    return f"{run_id}:{slugify(label)}_{idx:03d}"


def _prune_stale_tracks(
    tracks: dict[str, list[TrackState]],
    second: int,
    max_gap_seconds: int,
) -> None:
    for label in list(tracks.keys()):
        alive = [
            track
            for track in tracks[label]
            if (second - track.last_second) <= max_gap_seconds
        ]
        if alive:
            tracks[label] = alive
        else:
            del tracks[label]


def _match_to_existing_track(
    mobile,
    tracks_for_label: list[TrackState],
    reserved_track_ids: set[str],
    second: int,
    frame_diag: float,
    opts: IngestOptions,
) -> Optional[TrackState]:
    best_track: Optional[TrackState] = None
    best_score = float("-inf")

    for track in tracks_for_label:
        if track.instance_id in reserved_track_ids:
            continue
        if (second - track.last_second) > opts.track_max_gap_seconds:
            continue

        iou = box_iou(mobile.bbox, track.bbox)
        dist_ratio = center_distance(mobile.bbox, track.bbox) / max(1.0, frame_diag)

        if iou < opts.track_iou_threshold and dist_ratio > opts.track_center_dist_ratio:
            continue

        # Higher IoU + lower distance should win.
        score = iou - dist_ratio
        if score > best_score:
            best_score = score
            best_track = track

    return best_track


def _assign_instance_ids(
    mobile_items: list,
    tracks: dict[str, list[TrackState]],
    counters: dict[str, int],
    run_id: str,
    second: int,
    frame_diag: float,
    opts: IngestOptions,
) -> list[tuple]:
    assignments: list[tuple] = []
    reserved_track_ids: set[str] = set()

    # Greedy confidence-first assignment for deterministic tracking.
    mobile_sorted = sorted(mobile_items, key=lambda d: d.confidence, reverse=True)

    for mobile in mobile_sorted:
        label_tracks = tracks.setdefault(mobile.label, [])
        matched = _match_to_existing_track(
            mobile=mobile,
            tracks_for_label=label_tracks,
            reserved_track_ids=reserved_track_ids,
            second=second,
            frame_diag=frame_diag,
            opts=opts,
        )

        if matched is None:
            instance_id = _new_instance_id(mobile.label, counters, run_id=run_id)
            matched = TrackState(
                instance_id=instance_id,
                label=mobile.label,
                bbox=mobile.bbox,
                last_second=second,
                hit_count=1,
            )
            label_tracks.append(matched)
        else:
            matched.bbox = mobile.bbox
            matched.last_second = second
            matched.hit_count += 1

        reserved_track_ids.add(matched.instance_id)
        assignments.append((mobile, matched.instance_id))

    _prune_stale_tracks(tracks, second, opts.track_max_gap_seconds)
    return assignments


def run_video_ingestion(opts: IngestOptions) -> dict[str, int]:
    if not opts.video_path.exists():
        raise FileNotFoundError(f"Video not found: {opts.video_path}")

    video_id = opts.video_path.stem
    crops_dir, memory_path = prepare_output_paths(opts.out_root, video_id)
    frames_dir = resolve_frames_dir(opts.out_root, video_id)
    frame_memory_path = resolve_frame_memory_path(opts.out_root, video_id)

    if opts.reset_output:
        # Clean prior outputs for deterministic reruns.
        if memory_path.exists():
            memory_path.unlink()
        for crop_file in crops_dir.glob("*.jpg"):
            try:
                crop_file.unlink()
            except OSError:
                pass
        for frame_file in frames_dir.glob("*.jpg"):
            try:
                frame_file.unlink()
            except OSError:
                pass
        if frame_memory_path.exists():
            frame_memory_path.unlink()
        # Recreate empty log file after cleanup.
        memory_path.touch(exist_ok=True)
        frame_memory_path.touch(exist_ok=True)

    detector = QualcommYoloDetector(
        prefer_model=opts.prefer_model,
        conf_threshold=opts.conf_threshold,
        iou_threshold=opts.iou_threshold,
        runtime=opts.runtime,
        input_size=opts.detector_input_size,
    )

    cap = cv2.VideoCapture(str(opts.video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {opts.video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    if opts.sample_fps <= 0:
        raise ValueError("sample_fps must be > 0")

    sample_interval_s = 1.0 / float(opts.sample_fps)
    next_sample_time_s = 0.0
    frame_idx = 0
    processed_seconds = 0
    event_count = 0
    frame_count = 0
    caption_count = 0
    tracks: dict[str, list[TrackState]] = {}
    instance_counters: dict[str, int] = {}
    ingestion_run_id = _build_ingestion_run_id(video_id)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if pos_ms > 0:
                frame_time_s = pos_ms / 1000.0
            else:
                frame_time_s = float(frame_idx) / float(fps)
            frame_idx += 1

            # Sample at requested FPS (default 3 FPS).
            if frame_time_s + 1e-6 < next_sample_time_s:
                continue
            # If decoding jumps forward, catch up sampling schedule.
            while next_sample_time_s <= frame_time_s:
                next_sample_time_s += sample_interval_s

            second = int(frame_time_s)
            processed_seconds += 1

            detections = detector.predict(frame)

            # Persist a compact full-frame memory record for open-ended retrieval.
            if opts.save_full_frames:
                frame_name = f"f{frame_count:06d}_s{second:06d}.jpg"
                frame_path = frames_dir / frame_name
                saved_frame = write_frame_image(
                    frame_bgr=frame,
                    out_path=frame_path,
                    max_width=opts.frame_max_width,
                    jpeg_quality=opts.frame_jpeg_quality,
                )
                if saved_frame:
                    top_labels = sorted(
                        {
                            str(d.label).strip().lower()
                            for d in detections
                            if str(d.label).strip()
                        }
                    )
                    anchor_labels = sorted(
                        {
                            str(d.label).strip().lower()
                            for d in detections
                            if str(d.label).strip().lower() in opts.anchor_classes
                        }
                    )
                    summary_parts: list[str] = []
                    if top_labels:
                        summary_parts.append(f"objects: {', '.join(top_labels[:10])}")
                    if anchor_labels:
                        summary_parts.append(f"anchors: {', '.join(anchor_labels[:6])}")
                    summary_text = "; ".join(summary_parts) if summary_parts else "objects: none"
                    caption_text: str | None = None
                    if opts.caption_frames and opts.caption_every_n_frames > 0:
                        if (frame_count % max(1, opts.caption_every_n_frames)) == 0:
                            caption_text = caption_image(
                                image_path=str(frame_path),
                                model_name=opts.caption_model,
                            )
                            if caption_text:
                                caption_count += 1
                    frame_record = {
                        "video_id": video_id,
                        "ingestion_run_id": ingestion_run_id,
                        "frame_id": f"{video_id}:{frame_count:06d}",
                        "frame_index": frame_count,
                        "video_second": second,
                        "video_time_s": round(frame_time_s, 3),
                        "detector_model": detector.model_name,
                        "detector_backend": detector.backend,
                        "labels": top_labels,
                        "summary_text": summary_text,
                        "caption_text": caption_text,
                        "frame_path": str(frame_path),
                    }
                    append_frame_record(frame_memory_path, frame_record)
            frame_count += 1

            if opts.include_labels:
                detections = [d for d in detections if d.label in opts.include_labels]
            if opts.exclude_labels:
                detections = [d for d in detections if d.label not in opts.exclude_labels]

            anchors = [d for d in detections if d.label in opts.anchor_classes]
            mobile_items = [d for d in detections if d.label not in opts.anchor_classes]
            frame_h, frame_w = frame.shape[:2]
            frame_diag = (frame_h**2 + frame_w**2) ** 0.5
            mobile_with_instance = _assign_instance_ids(
                mobile_items=mobile_items,
                tracks=tracks,
                counters=instance_counters,
                run_id=ingestion_run_id,
                second=second,
                frame_diag=frame_diag,
                opts=opts,
            )

            for local_idx, (mobile, instance_id) in enumerate(mobile_with_instance):
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
                detected_color = detect_color_from_image_path(thumb_path)

                event = {
                    "video_id": video_id,
                    "ingestion_run_id": ingestion_run_id,
                    "frame_id": f"{video_id}:{frame_count - 1:06d}",
                    "video_second": second,
                    "video_time_s": round(frame_time_s, 3),
                    "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
                    "detector_model": detector.model_name,
                    "instance_id": instance_id,
                    "label": mobile.label,
                    "confidence": round(mobile.confidence, 4),
                    "bbox_xyxy": list(mobile.bbox),
                    "anchor_label": anchor_label,
                    "relation": relation,
                    "context": context,
                    "detected_color": detected_color,
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
        "sample_fps": opts.sample_fps,
        "detector_input_size": opts.detector_input_size,
        "events_written": event_count,
        "frames_written": frame_count if opts.save_full_frames else 0,
        "captions_written": caption_count,
        "detector_backend": detector.backend,
        "detector_model": detector.model_name,
    }
