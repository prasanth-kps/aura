from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np


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

COLOR_TOKEN_ALIASES = {
    "black": "black",
    "dark": "black",
    "white": "white",
    "gray": "gray",
    "grey": "gray",
    "silver": "gray",
    "red": "red",
    "maroon": "red",
    "orange": "orange",
    "yellow": "yellow",
    "green": "green",
    "blue": "blue",
    "navy": "blue",
    "purple": "purple",
    "violet": "purple",
    "brown": "brown",
}

QUERY_STOPWORDS = {
    "i",
    "me",
    "did",
    "do",
    "does",
    "where",
    "is",
    "are",
    "was",
    "were",
    "see",
    "seen",
    "last",
    "latest",
    "my",
    "the",
    "a",
    "an",
    "this",
    "that",
    "please",
    "object",
    "item",
}

LOW_CONFIDENCE_THRESHOLD = 0.55
RECENT_SECONDS_WINDOW = 10


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


def _normalize_instance_id(instance_id: str) -> str:
    return _normalize_spaces(instance_id)


def canonicalize_label(
    raw_label: str,
    aliases: dict[str, str] | None = None,
) -> str:
    aliases_map = aliases or DEFAULT_LABEL_ALIASES
    cleaned = _normalize_spaces(raw_label)
    return aliases_map.get(cleaned, cleaned)


def parse_query_intent(
    query: str,
    aliases: dict[str, str] | None = None,
) -> tuple[str, str | None, list[str]]:
    cleaned = _normalize_spaces(query)
    cleaned = re.sub(r"^[^a-z0-9]*", "", cleaned)
    cleaned = re.sub(r"[?.!]+$", "", cleaned)

    # Handle natural query forms.
    cleaned = re.sub(r"^where\s+is\s+(my|the)\s+", "", cleaned)
    cleaned = re.sub(r"^where\s+are\s+(my|the)\s+", "", cleaned)
    cleaned = re.sub(r"^where\s+was\s+(my|the)\s+", "", cleaned)
    cleaned = re.sub(r"^where\s+were\s+(my|the)\s+", "", cleaned)
    cleaned = re.sub(r"^where\s+did\s+i\s+last\s+see\s+(my|the)\s+", "", cleaned)
    cleaned = re.sub(r"^where\s+did\s+i\s+see\s+(my|the)\s+", "", cleaned)
    cleaned = re.sub(r"^where\s+did\s+we\s+last\s+see\s+(my|the)\s+", "", cleaned)
    cleaned = re.sub(r"^where\s+did\s+we\s+see\s+(my|the)\s+", "", cleaned)
    cleaned = re.sub(r"^find\s+(my|the)\s+", "", cleaned)
    cleaned = re.sub(r"^show\s+me\s+(my|the)\s+", "", cleaned)
    cleaned = re.sub(r"^latest\s+(my|the)\s+", "", cleaned)
    cleaned = re.sub(r"^last\s+seen\s+(my|the)\s+", "", cleaned)
    cleaned = cleaned.strip()

    tokens = [t for t in cleaned.split(" ") if t]
    object_tokens: list[str] = []
    target_color: str | None = None

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in QUERY_STOPWORDS:
            i += 1
            continue

        # Handle two-token forms like "dark blue".
        if tok in {"dark", "light"} and i + 1 < len(tokens):
            nxt = tokens[i + 1]
            if nxt in COLOR_TOKEN_ALIASES and target_color is None:
                target_color = COLOR_TOKEN_ALIASES[nxt]
                i += 2
                continue

        mapped_color = COLOR_TOKEN_ALIASES.get(tok)
        if mapped_color and target_color is None:
            target_color = mapped_color
            i += 1
            continue

        object_tokens.append(tok)
        i += 1

    object_phrase = " ".join(object_tokens).strip()
    if not object_phrase:
        object_phrase = cleaned

    canonical = canonicalize_label(object_phrase, aliases=aliases)
    return canonical, target_color, object_tokens


def normalize_query_label(
    query: str,
    aliases: dict[str, str] | None = None,
) -> str:
    label, _, _ = parse_query_intent(query, aliases=aliases)
    return label


def _event_second(event: dict[str, Any]) -> int:
    value = event.get("video_second", -1)
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _event_confidence(event: dict[str, Any]) -> float:
    value = event.get("confidence", 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _sorted_recent(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        events,
        key=lambda ev: (_event_second(ev), str(ev.get("ingested_at_utc", ""))),
        reverse=True,
    )


def _classify_bgr_image_color(image_bgr: np.ndarray) -> str | None:
    if image_bgr.size == 0:
        return None

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h_mean = float(np.mean(hsv[:, :, 0]))  # 0..179
    s_mean = float(np.mean(hsv[:, :, 1]))  # 0..255
    v_mean = float(np.mean(hsv[:, :, 2]))  # 0..255

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


def _event_thumbnail_color(
    event: dict[str, Any],
    color_cache: dict[str, str | None] | None = None,
) -> str | None:
    thumb = str(event.get("thumbnail_path", "")).strip()
    if not thumb:
        return None

    cache = color_cache if color_cache is not None else {}
    if thumb in cache:
        return cache[thumb]

    image = cv2.imread(thumb)
    if image is None:
        cache[thumb] = None
        return None

    color = _classify_bgr_image_color(image)
    cache[thumb] = color
    return color


def _infer_label_from_events(
    events: list[dict[str, Any]],
    object_tokens: list[str],
    aliases: dict[str, str] | None = None,
) -> str | None:
    if not object_tokens:
        return None

    token_set = set(object_tokens)
    labels = available_labels(events, aliases=aliases)
    best_label: str | None = None
    best_score = 0

    for label in labels:
        label_tokens = set(label.split())
        score = len(label_tokens.intersection(token_set))
        if score > best_score:
            best_score = score
            best_label = label

    if best_score <= 0:
        return None
    return best_label


def _group_events_by_instance(events: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for idx, event in enumerate(events):
        instance_id = str(event.get("instance_id", "")).strip()
        if not instance_id:
            # Fallback for old logs with no instance_id.
            instance_id = f"__event_{idx:06d}"
        groups.setdefault(instance_id, []).append(event)
    return groups


def _resolve_query_events(
    events: list[dict[str, Any]],
    label_or_query: str,
    instance_id: str | None = None,
    aliases: dict[str, str] | None = None,
    color_cache: dict[str, str | None] | None = None,
) -> tuple[str, str | None, str | None, list[dict[str, Any]]]:
    target_label, target_color, object_tokens = parse_query_intent(
        label_or_query, aliases=aliases
    )
    target_instance = _normalize_instance_id(instance_id) if instance_id else None

    # Base label-filter.
    base = []
    for event in events:
        label = canonicalize_label(str(event.get("label", "")), aliases=aliases)
        if label == target_label:
            base.append(event)

    # Fallback label inference for phrases like "water bottle" -> "bottle".
    if not base:
        inferred = _infer_label_from_events(events, object_tokens, aliases=aliases)
        if inferred:
            target_label = inferred
            base = []
            for event in events:
                label = canonicalize_label(str(event.get("label", "")), aliases=aliases)
                if label == target_label:
                    base.append(event)

    if not base:
        return target_label, target_color, target_instance, []

    if target_instance:
        scoped = []
        for event in base:
            event_instance = _normalize_instance_id(str(event.get("instance_id", "")))
            if event_instance == target_instance:
                scoped.append(event)
        base = scoped
        if not base:
            return target_label, target_color, target_instance, []

    # No color query => regular recent behavior.
    if not target_color:
        return target_label, None, target_instance, _sorted_recent(base)

    # If instance is explicit, filter by color within that instance, fallback to full instance events.
    if target_instance:
        colored = [
            ev
            for ev in base
            if _event_thumbnail_color(ev, color_cache=color_cache) == target_color
        ]
        if colored:
            return target_label, target_color, target_instance, _sorted_recent(colored)
        return target_label, target_color, target_instance, _sorted_recent(base)

    # Auto-select best instance by color match ratio.
    groups = _group_events_by_instance(base)
    best_instance: str | None = None
    best_score: tuple[float, int, int] = (-1.0, -1, -1)

    for instance_key, rows in groups.items():
        hits = 0
        for row in rows:
            if _event_thumbnail_color(row, color_cache=color_cache) == target_color:
                hits += 1
        ratio = hits / max(1, len(rows))
        recency = max(_event_second(r) for r in rows) if rows else -1
        score = (ratio, hits, recency)
        if score > best_score:
            best_score = score
            best_instance = instance_key

    if not best_instance:
        return target_label, target_color, None, _sorted_recent(base)

    selected_rows = groups[best_instance]
    selected_colored = [
        ev
        for ev in selected_rows
        if _event_thumbnail_color(ev, color_cache=color_cache) == target_color
    ]
    final_rows = selected_colored if selected_colored else selected_rows
    return target_label, target_color, best_instance, _sorted_recent(final_rows)


def filter_by_label(
    events: list[dict[str, Any]],
    label_or_query: str,
    instance_id: str | None = None,
    target_color: str | None = None,
    apply_color_from_query: bool = True,
    color_cache: dict[str, str | None] | None = None,
    aliases: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    target, query_color, _ = parse_query_intent(label_or_query, aliases=aliases)
    target_instance = _normalize_instance_id(instance_id) if instance_id else None
    desired_color = target_color or (query_color if apply_color_from_query else None)
    out: list[dict[str, Any]] = []
    for event in events:
        label = canonicalize_label(str(event.get("label", "")), aliases=aliases)
        if label != target:
            continue
        if target_instance:
            event_instance = _normalize_instance_id(str(event.get("instance_id", "")))
            if event_instance != target_instance:
                continue
        if desired_color:
            event_color = _event_thumbnail_color(event, color_cache=color_cache)
            if event_color != desired_color:
                continue
        out.append(event)
    return out


def filter_by_instance_id(
    events: list[dict[str, Any]],
    instance_id: str,
) -> list[dict[str, Any]]:
    target_instance = _normalize_instance_id(instance_id)
    out: list[dict[str, Any]] = []
    for event in events:
        event_instance = _normalize_instance_id(str(event.get("instance_id", "")))
        if event_instance == target_instance:
            out.append(event)
    return out


def latest(
    events: list[dict[str, Any]],
    label_or_query: str,
    instance_id: str | None = None,
    aliases: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    _, _, _, matches = _resolve_query_events(
        events=events,
        label_or_query=label_or_query,
        instance_id=instance_id,
        aliases=aliases,
    )
    if not matches:
        return None
    return matches[0]


def timeline(
    events: list[dict[str, Any]],
    label_or_query: str,
    limit: int = 5,
    instance_id: str | None = None,
    aliases: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    _, _, _, matches = _resolve_query_events(
        events=events,
        label_or_query=label_or_query,
        instance_id=instance_id,
        aliases=aliases,
    )
    return matches[:limit]


def where_is(
    events: list[dict[str, Any]],
    label_or_query: str,
    instance_id: str | None = None,
    aliases: dict[str, str] | None = None,
) -> dict[str, Any]:
    color_cache: dict[str, str | None] = {}
    target_label, target_color, resolved_instance, matches = _resolve_query_events(
        events=events,
        label_or_query=label_or_query,
        instance_id=instance_id,
        color_cache=color_cache,
        aliases=aliases,
    )
    latest_event = matches[0] if matches else None

    if latest_event is None:
        return {
            "found": False,
            "query": label_or_query,
            "canonical_label": target_label,
            "target_color": target_color,
            "resolved_instance_id": resolved_instance,
            "answer": "I haven't seen it yet.",
            "event": None,
        }

    second = _event_second(latest_event)
    event_instance = str(latest_event.get("instance_id", "")).strip() or None
    relation = str(latest_event.get("relation", "")).strip()
    anchor_label = latest_event.get("anchor_label")
    context = str(latest_event.get("context", "in_scene"))

    if anchor_label and relation:
        answer = f"Last seen at second {second}: {target_label} is {relation} {anchor_label}."
    else:
        answer = f"Last seen at second {second}: {target_label} is {context}."

    event_color = _event_thumbnail_color(latest_event, color_cache=color_cache)
    if target_color and event_color != target_color:
        answer = (
            f"I couldn't find a recent {target_color} {target_label}, "
            f"but the latest {target_label} is at second {second}: {context}."
        )
        if event_color:
            answer = f"{answer} (detected color: {event_color})"
        if event_instance:
            answer = f"{answer} (instance_id: {event_instance})"
        return {
            "found": True,
            "query": label_or_query,
            "canonical_label": target_label,
            "target_color": target_color,
            "resolved_instance_id": resolved_instance or event_instance,
            "answer": answer,
            "event": latest_event,
        }

    latest_global_second = max((_event_second(ev) for ev in events), default=-1)
    is_recent = (
        second >= 0
        and latest_global_second >= 0
        and (latest_global_second - second) <= RECENT_SECONDS_WINDOW
    )
    confidence = _event_confidence(latest_event)
    if is_recent and confidence < LOW_CONFIDENCE_THRESHOLD:
        answer = f"Most likely near {context} (low confidence: {confidence:.2f})."
        if event_instance:
            answer = f"{answer} (instance_id: {event_instance})"
        return {
            "found": True,
            "query": label_or_query,
            "canonical_label": target_label,
            "target_color": target_color,
            "resolved_instance_id": resolved_instance or event_instance,
            "answer": answer,
            "event": latest_event,
        }

    if target_color:
        answer = f"{answer} (matched color: {target_color})"

    if event_instance:
        answer = f"{answer} (instance_id: {event_instance})"

    return {
        "found": True,
        "query": label_or_query,
        "canonical_label": target_label,
        "target_color": target_color,
        "resolved_instance_id": resolved_instance or event_instance,
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


def available_instances(
    events: list[dict[str, Any]],
    label_or_query: str | None = None,
    aliases: dict[str, str] | None = None,
) -> list[str]:
    if label_or_query:
        filtered = filter_by_label(events, label_or_query, aliases=aliases)
    else:
        filtered = events

    instances = {
        str(ev.get("instance_id", "")).strip()
        for ev in filtered
        if str(ev.get("instance_id", "")).strip()
    }
    return sorted(instances)
