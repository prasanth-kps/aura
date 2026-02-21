from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.search import (
    available_labels,
    latest,
    load_events,
    resolve_memory_path,
    timeline,
    where_is,
)


def _resolve_input_memory_path(args: argparse.Namespace) -> Path:
    if args.memory:
        path = Path(args.memory)
    else:
        if not args.video:
            raise ValueError("Pass either --memory or --video (with --out).")
        path = resolve_memory_path(Path(args.out), Path(args.video))
    return path


def _print_event(event: dict) -> None:
    print(f"label          : {event.get('label')}")
    print(f"video_second   : {event.get('video_second')}")
    print(f"context        : {event.get('context')}")
    print(f"confidence     : {event.get('confidence')}")
    print(f"thumbnail_path : {event.get('thumbnail_path')}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Query memory events (where_is/latest/timeline)."
    )
    parser.add_argument(
        "--memory",
        default=None,
        help="Direct path to .events.jsonl file",
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Video path used during ingestion (used with --out to resolve memory file)",
    )
    parser.add_argument(
        "--out",
        default="data",
        help="Output root used during ingestion",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print structured JSON output",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    p_where = subparsers.add_parser("where_is", help="Find most recent location context")
    p_where.add_argument("--label", required=True, help="Label or query (e.g., phone)")

    p_latest = subparsers.add_parser("latest", help="Return latest matching event")
    p_latest.add_argument("--label", required=True, help="Label or query (e.g., bottle)")

    p_timeline = subparsers.add_parser("timeline", help="Return recent matching events")
    p_timeline.add_argument("--label", required=True, help="Label or query")
    p_timeline.add_argument("--limit", type=int, default=5, help="Max entries to return")

    subparsers.add_parser("labels", help="List detected labels in memory log")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    memory_path = _resolve_input_memory_path(args)
    events = load_events(memory_path)

    if args.command == "where_is":
        result = where_is(events, args.label)
        if args.json:
            print(json.dumps(result, indent=2))
            return
        print(result["answer"])
        event = result.get("event")
        if event:
            _print_event(event)
        return

    if args.command == "latest":
        event = latest(events, args.label)
        if args.json:
            print(json.dumps(event, indent=2))
            return
        if event is None:
            print(f"No events found for '{args.label}'.")
            return
        _print_event(event)
        return

    if args.command == "timeline":
        rows = timeline(events, args.label, limit=args.limit)
        if args.json:
            print(json.dumps(rows, indent=2))
            return
        if not rows:
            print(f"No events found for '{args.label}'.")
            return
        for idx, event in enumerate(rows, start=1):
            print(f"[{idx}] sec={event.get('video_second')} context={event.get('context')}")
            print(f"    thumb={event.get('thumbnail_path')}")
        return

    if args.command == "labels":
        labels = available_labels(events)
        if args.json:
            print(json.dumps(labels, indent=2))
            return
        if not labels:
            print("No labels found.")
            return
        for label in labels:
            print(label)
        return

    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
