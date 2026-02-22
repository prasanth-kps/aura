from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.search import (
    ask_memory,
    available_instances,
    available_labels,
    describe_color,
    is_color_query,
    latest,
    load_events,
    load_frame_records,
    resolve_memory_path,
    resolve_frames_path,
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
    print(f"instance_id    : {event.get('instance_id')}")
    print(f"label          : {event.get('label')}")
    print(f"detected_color : {event.get('detected_color')}")
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
    p_where.add_argument(
        "--instance-id",
        default=None,
        help="Optional instance_id to target a specific tracked object",
    )

    p_latest = subparsers.add_parser("latest", help="Return latest matching event")
    p_latest.add_argument("--label", required=True, help="Label or query (e.g., bottle)")
    p_latest.add_argument(
        "--instance-id",
        default=None,
        help="Optional instance_id to target a specific tracked object",
    )

    p_timeline = subparsers.add_parser("timeline", help="Return recent matching events")
    p_timeline.add_argument("--label", required=True, help="Label or query")
    p_timeline.add_argument("--limit", type=int, default=5, help="Max entries to return")
    p_timeline.add_argument(
        "--instance-id",
        default=None,
        help="Optional instance_id to target a specific tracked object",
    )
    p_ask = subparsers.add_parser("ask", help="Open-ended memory question over stored evidence")
    p_ask.add_argument("--question", required=True, help="Natural question")
    p_ask.add_argument("--limit", type=int, default=5, help="Max evidence rows to return")
    p_ask.add_argument("--llm", action="store_true", help="Enable LLM synthesis over retrieved evidence")
    p_ask.add_argument("--llm-model", default=None, help="Optional LLM model name")
    p_ask.add_argument("--llm-base-url", default=None, help="Optional OpenAI-compatible base URL")
    p_ask.add_argument("--llm-api-key", default=None, help="Optional API key (overrides env)")

    subparsers.add_parser("labels", help="List detected labels in memory log")
    p_instances = subparsers.add_parser("instances", help="List instance IDs")
    p_instances.add_argument(
        "--label",
        default=None,
        help="Optional label/query to filter instance IDs",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    memory_path = _resolve_input_memory_path(args)
    events = load_events(memory_path)

    if args.command == "where_is":
        if is_color_query(args.label):
            result = describe_color(events, args.label, instance_id=args.instance_id)
        else:
            result = where_is(events, args.label, instance_id=args.instance_id)
        if args.json:
            print(json.dumps(result, indent=2))
            return
        print(result["answer"])
        event = result.get("event")
        if event:
            _print_event(event)
        return

    if args.command == "latest":
        event = latest(events, args.label, instance_id=args.instance_id)
        if args.json:
            print(json.dumps(event, indent=2))
            return
        if event is None:
            print(f"No events found for '{args.label}'.")
            return
        _print_event(event)
        return

    if args.command == "timeline":
        rows = timeline(
            events,
            args.label,
            limit=args.limit,
            instance_id=args.instance_id,
        )
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

    if args.command == "ask":
        frame_memory_path = (
            resolve_frames_path(Path(args.out), Path(args.video))
            if args.video
            else Path(str(memory_path).replace(".events.jsonl", ".frames.jsonl"))
        )
        frame_records = load_frame_records(frame_memory_path)
        result = ask_memory(
            events,
            frame_records,
            args.question,
            limit=args.limit,
            use_llm=args.llm,
            llm_model=args.llm_model,
            llm_base_url=args.llm_base_url,
            llm_api_key=args.llm_api_key,
        )
        if args.json:
            print(json.dumps(result, indent=2))
            return
        print(result["answer"])
        citations_md = str(result.get("citations_markdown", "")).strip()
        if citations_md and "### Evidence" not in str(result.get("answer", "")):
            print(citations_md)
        for idx, ev in enumerate(result.get("evidence", []), start=1):
            print(
                f"[{idx}] sec={ev.get('video_second')} label={ev.get('label')} "
                f"context={ev.get('context')} color={ev.get('detected_color')}"
            )
            if ev.get("frame_path"):
                print(f"    frame={ev.get('frame_path')}")
            if ev.get("thumbnail_path"):
                print(f"    thumb={ev.get('thumbnail_path')}")
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

    if args.command == "instances":
        ids = available_instances(events, label_or_query=args.label)
        if args.json:
            print(json.dumps(ids, indent=2))
            return
        if not ids:
            print("No instance IDs found.")
            return
        for instance_id in ids:
            print(instance_id)
        return

    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
