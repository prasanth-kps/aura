from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


DEFAULT_LABEL_ALIASES = {
    "phone": "cell phone",
    "mobile": "cell phone",
    "smartphone": "cell phone",
    "table": "dining table",
    "desk": "dining table",
    "sofa": "couch",
    "fridge": "refrigerator",
    "monitor": "tv",
}


def resolve_memory_path(out_root: Path, video_path: Path) -> Path:
    return out_root / "memory" / f"{video_path.stem}.events.jsonl"


def load_events(memory_path: Path) -> list[dict[str, Any]]:
    if not memory_path.exists():
        raise FileNotFoundError(f"Memory file not found: {memory_path}")

    events: list[dict[str, Any]] = []
    with memory_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                events.append(item)
    return events


def _normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def canonicalize_label(
    raw_label: str,
    aliases: dict[str, str] | None = None,
) -> str:
    aliases_map = aliases or DEFAULT_LABEL_ALIASES
    cleaned = _normalize_spaces(raw_label)
    return aliases_map.get(cleaned, cleaned)


def normalize_query_label(
    query: str,
    aliases: dict[str, str] | None = None,
) -> str:
    cleaned = _normalize_spaces(query)
    cleaned = re.sub(r"^[^a-z0-9]*", "", cleaned)
    cleaned = re.sub(r"[?.!]+$", "", cleaned)

    # Handle natural query forms.
    cleaned = re.sub(r"^where\s+is\s+(my|the)\s+", "", cleaned)
    cleaned = re.sub(r"^where\s+are\s+(my|the)\s+", "", cleaned)
    cleaned = re.sub(r"^find\s+(my|the)\s+", "", cleaned)
    cleaned = re.sub(r"^show\s+me\s+(my|the)\s+", "", cleaned)
    cleaned = re.sub(r"^latest\s+(my|the)\s+", "", cleaned)
    cleaned = cleaned.strip()

    return canonicalize_label(cleaned, aliases=aliases)


def _event_second(event: dict[str, Any]) -> int:
    value = event.get("video_second", -1)
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _sorted_recent(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        events,
        key=lambda ev: (_event_second(ev), str(ev.get("ingested_at_utc", ""))),
        reverse=True,
    )


def filter_by_label(
    events: list[dict[str, Any]],
    label_or_query: str,
    aliases: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    target = normalize_query_label(label_or_query, aliases=aliases)
    out: list[dict[str, Any]] = []
    for event in events:
        label = canonicalize_label(str(event.get("label", "")), aliases=aliases)
        if label == target:
            out.append(event)
    return out


def latest(
    events: list[dict[str, Any]],
    label_or_query: str,
    aliases: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    matches = filter_by_label(events, label_or_query, aliases=aliases)
    if not matches:
        return None
    return _sorted_recent(matches)[0]


def timeline(
    events: list[dict[str, Any]],
    label_or_query: str,
    limit: int = 5,
    aliases: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    matches = filter_by_label(events, label_or_query, aliases=aliases)
    return _sorted_recent(matches)[:limit]


def where_is(
    events: list[dict[str, Any]],
    label_or_query: str,
    aliases: dict[str, str] | None = None,
) -> dict[str, Any]:
    target = normalize_query_label(label_or_query, aliases=aliases)
    latest_event = latest(events, target, aliases=aliases)

    if latest_event is None:
        return {
            "found": False,
            "query": label_or_query,
            "canonical_label": target,
            "answer": f"I couldn't find '{target}' in memory yet.",
            "event": None,
        }

    second = _event_second(latest_event)
    relation = str(latest_event.get("relation", "")).strip()
    anchor_label = latest_event.get("anchor_label")
    context = str(latest_event.get("context", "in_scene"))

    if anchor_label and relation:
        answer = (
            f"Last seen at second {second}: {target} is {relation} {anchor_label}."
        )
    else:
        answer = f"Last seen at second {second}: {target} is {context}."

    return {
        "found": True,
        "query": label_or_query,
        "canonical_label": target,
        "answer": answer,
        "event": latest_event,
    }


def available_labels(
    events: list[dict[str, Any]],
    aliases: dict[str, str] | None = None,
) -> list[str]:
    labels = {
        canonicalize_label(str(ev.get("label", "")), aliases=aliases)
        for ev in events
        if ev.get("label")
    }
    return sorted(labels)
