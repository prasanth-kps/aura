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
    parser.add_argument(
        "--runtime",
        choices=["auto", "qnn", "cpu"],
        default="auto",
        help="Inference runtime: auto (prefer QNN), qnn (require NPU), or cpu",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.60, help="NMS IoU threshold")
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=3.0,
        help="Sampling rate from video stream (frames per second)",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=960,
        help="Detector input size (e.g., 640 or 960)",
    )
    parser.add_argument(
        "--save-full-frames",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Persist compressed full-frame snapshots for open-ended memory queries",
    )
    parser.add_argument(
        "--frame-max-width",
        type=int,
        default=960,
        help="Max width for persisted full-frame snapshots",
    )
    parser.add_argument(
        "--frame-jpeg-quality",
        type=int,
        default=70,
        help="JPEG quality for persisted full-frame snapshots",
    )
    parser.add_argument(
        "--caption-frames",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate local VLM captions for saved full frames (offline after model download)",
    )
    parser.add_argument(
        "--caption-model",
        default="nlpconnect/vit-gpt2-image-captioning",
        help="Local caption model name (transformers image-to-text pipeline)",
    )
    parser.add_argument(
        "--caption-every-n-frames",
        type=int,
        default=1,
        help="Run captioning every N sampled frames (1 = every frame)",
    )
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
        sample_fps=args.sample_fps,
        detector_input_size=args.input_size,
        thumb_size=args.thumb_size,
        crop_padding=args.crop_padding,
        jpeg_quality=args.jpeg_quality,
        save_full_frames=args.save_full_frames,
        frame_max_width=args.frame_max_width,
        frame_jpeg_quality=args.frame_jpeg_quality,
        caption_frames=args.caption_frames,
        caption_model=args.caption_model,
        caption_every_n_frames=max(1, args.caption_every_n_frames),
        anchor_classes=anchor_labels,
        include_labels=include_labels,
        exclude_labels=exclude_labels,
        reset_output=args.reset_output,
        track_max_gap_seconds=args.track_max_gap,
        track_iou_threshold=args.track_iou_threshold,
        track_center_dist_ratio=args.track_center_dist_ratio,
        runtime=args.runtime,
    )

    summary = run_video_ingestion(options)
    print("Ingestion complete.")
    print(f"Processed samples: {summary['processed_seconds']} @ {summary.get('sample_fps')} FPS")
    print(f"Events written: {summary['events_written']}")
    print(f"Frames written: {summary.get('frames_written', 0)}")
    print(f"Captions written: {summary.get('captions_written', 0)}")
    print(f"Detector input size: {summary.get('detector_input_size')}")
    print(f"Detector model/runtime: {summary.get('detector_model')} / {summary.get('detector_backend')}")
    print(
        "Memory log: "
        f"{(Path(args.out) / 'memory' / (Path(args.video).stem + '.events.jsonl'))}"
    )
    print(
        "Frame memory log: "
        f"{(Path(args.out) / 'memory' / (Path(args.video).stem + '.frames.jsonl'))}"
    )


if __name__ == "__main__":
    main()
