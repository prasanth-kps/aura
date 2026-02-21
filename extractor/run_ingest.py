from __future__ import annotations

import argparse
from pathlib import Path

from src.anchors import ANCHOR_CLASSES
from src.ingest_video import IngestOptions, run_video_ingestion


def parse_csv_labels(raw: str | None) -> set[str] | None:
    if raw is None:
        return None
    values = {x.strip() for x in raw.split(",") if x.strip()}
    return values if values else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Video -> spatial memory backend (Qualcomm model first)."
    )
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--out", default="data", help="Output root directory")
    parser.add_argument("--prefer-model", choices=["yolov11", "yolov8"], default="yolov11")
    parser.add_argument("--conf", type=float, default=0.45, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.60, help="NMS IoU threshold")
    parser.add_argument("--thumb-size", type=int, default=50, help="Thumbnail size in pixels")
    parser.add_argument("--crop-padding", type=int, default=8, help="Crop padding around bbox")
    parser.add_argument("--jpeg-quality", type=int, default=75, help="JPEG quality for thumbnails")
    parser.add_argument(
        "--anchors",
        default=",".join(sorted(ANCHOR_CLASSES)),
        help="Comma-separated anchor labels",
    )
    parser.add_argument(
        "--include-labels",
        default=None,
        help="Only keep these labels (comma-separated)",
    )
    parser.add_argument(
        "--exclude-labels",
        default=None,
        help="Drop these labels (comma-separated)",
    )
    parser.add_argument(
        "--reset-output",
        action="store_true",
        help="Delete existing memory/crops for this video before ingestion",
    )
    parser.add_argument(
        "--track-max-gap",
        type=int,
        default=3,
        help="Max seconds a track can be unmatched before expiring",
    )
    parser.add_argument(
        "--track-iou-threshold",
        type=float,
        default=0.20,
        help="Minimum IoU to match detection to existing track",
    )
    parser.add_argument(
        "--track-center-dist-ratio",
        type=float,
        default=0.12,
        help="Max center-distance/diagonal ratio to match existing track",
    )

    args = parser.parse_args()

    anchor_labels = parse_csv_labels(args.anchors) or set(ANCHOR_CLASSES)
    include_labels = parse_csv_labels(args.include_labels)
    exclude_labels = parse_csv_labels(args.exclude_labels) or set()

    options = IngestOptions(
        video_path=Path(args.video),
        out_root=Path(args.out),
        prefer_model=args.prefer_model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        thumb_size=args.thumb_size,
        crop_padding=args.crop_padding,
        jpeg_quality=args.jpeg_quality,
        anchor_classes=anchor_labels,
        include_labels=include_labels,
        exclude_labels=exclude_labels,
        reset_output=args.reset_output,
        track_max_gap_seconds=args.track_max_gap,
        track_iou_threshold=args.track_iou_threshold,
        track_center_dist_ratio=args.track_center_dist_ratio,
    )

    summary = run_video_ingestion(options)
    print("Ingestion complete.")
    print(f"Processed seconds: {summary['processed_seconds']}")
    print(f"Events written: {summary['events_written']}")
    print(
        "Memory log: "
        f"{(Path(args.out) / 'memory' / (Path(args.video).stem + '.events.jsonl'))}"
    )


if __name__ == "__main__":
    main()
