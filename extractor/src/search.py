from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import cv2

from .color_utils import classify_bgr_image_color
from .llm_reasoner import synthesize_answer
from .local_rag import semantic_similarity_scores, visual_qa_answers, visual_qa_model_available


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

LABEL_TOKEN_ALIASES = {
    "people": "person",
    "persons": "person",
    "human": "person",
    "humans": "person",
    "man": "person",
    "men": "person",
    "woman": "person",
    "women": "person",
    "child": "person",
    "children": "person",
    "kids": "person",
}

QUERY_STOPWORDS = {
    "how",
    "many",
    "number",
    "of",
    "what",
    "which",
    "there",
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
COUNT_MIN_INSTANCE_HITS = 5
COUNT_MIN_INSTANCE_MAX_CONFIDENCE = 0.50
STRICT_OPEN_MIN_SEMANTIC_SCORE = 0.22
STRICT_OPEN_MIN_LEXICAL_SCORE = 1.0


def is_color_query(text: str) -> bool:
    cleaned = _normalize_spaces(text)
    return bool(
        re.search(r"\b(what|which)\s+color\s+(is|was|are|were)\b", cleaned)
        or re.search(r"\bcolor\s+of\b", cleaned)
    )


def resolve_memory_path(out_root: Path, video_path: Path) -> Path:
    return out_root / "memory" / f"{video_path.stem}.events.jsonl"


def resolve_frames_path(out_root: Path, video_path: Path) -> Path:
    return out_root / "memory" / f"{video_path.stem}.frames.jsonl"


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


def load_frame_records(frame_memory_path: Path) -> list[dict[str, Any]]:
    if not frame_memory_path.exists():
        return []

    rows: list[dict[str, Any]] = []
    with frame_memory_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows


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


def _event_thumbnail_color(
    event: dict[str, Any],
    color_cache: dict[str, str | None] | None = None,
) -> str | None:
    event_color = str(event.get("detected_color", "")).strip().lower()
    if event_color:
        return COLOR_TOKEN_ALIASES.get(event_color, event_color)

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

    color = classify_bgr_image_color(image)
    cache[thumb] = color
    return color


def _tokenize_query(text: str) -> set[str]:
    cleaned = _normalize_spaces(text)
    tokens = [t for t in re.split(r"[^a-z0-9]+", cleaned) if t]
    return {
        t
        for t in tokens
        if t not in QUERY_STOPWORDS and len(t) >= 2
    }


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
    elif event_color:
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


def describe_color(
    events: list[dict[str, Any]],
    label_or_query: str,
    instance_id: str | None = None,
    aliases: dict[str, str] | None = None,
) -> dict[str, Any]:
    target_label, _, resolved_instance, matches = _resolve_query_events(
        events=events,
        label_or_query=label_or_query,
        instance_id=instance_id,
        aliases=aliases,
    )
    latest_event = matches[0] if matches else None

    if latest_event is None:
        return {
            "found": False,
            "query": label_or_query,
            "canonical_label": target_label,
            "target_color": None,
            "resolved_instance_id": resolved_instance,
            "answer": "I haven't seen it yet.",
            "event": None,
        }

    event_color = _event_thumbnail_color(latest_event, color_cache={})
    second = _event_second(latest_event)
    confidence = _event_confidence(latest_event)
    event_instance = str(latest_event.get("instance_id", "")).strip() or None

    if event_color:
        answer = f"The {target_label} appears {event_color} at second {second}."
    else:
        answer = f"I can see the {target_label} at second {second}, but I couldn't determine its color."

    answer = f"{answer} (confidence: {confidence:.2f})"
    if event_instance:
        answer = f"{answer} (instance_id: {event_instance})"

    return {
        "found": True,
        "query": label_or_query,
        "canonical_label": target_label,
        "target_color": event_color,
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


def _extract_second_hint(question: str) -> int | None:
    m = re.search(r"\b(?:at|around|near)\s+second\s+(\d{1,5})\b", _normalize_spaces(question))
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _detect_question_intent(question: str) -> str:
    q = _normalize_spaces(question)
    if re.search(r"\b(how\s+many|number\s+of)\b", q) and re.search(r"\b(can\s+sit|seating|seats?)\b", q):
        return "capacity"
    if re.search(r"\bhow\s+many\b|\bcount\b|\bnumber\s+of\b", q):
        return "count"
    if re.search(r"\bwhat\s+objects?\b|\bwhich\s+objects?\b|\blist\b", q):
        return "list_objects"
    if re.search(r"\bwhat\s+color\b|\bcolor\s+of\b", q):
        return "color"
    if re.search(r"\bfirst\s+see\b|\bfirst\s+seen\b", q):
        return "when_first_seen"
    if re.search(r"\blast\s+see\b|\blast\s+seen\b|\blatest\b", q):
        return "when_last_seen"
    if re.search(r"\bchanged\b|\bdifferent\b|\bbefore\b|\bafter\b", q):
        return "change"
    if re.search(r"\bnear\b|\bnext\s+to\b|\baround\b", q):
        return "near"
    if re.search(r"\bwhere\b", q):
        return "where_is"
    return "open"


def _extract_primary_label(
    question: str,
    events: list[dict[str, Any]],
    aliases: dict[str, str] | None = None,
) -> str | None:
    canonical, _, tokens = parse_query_intent(question, aliases=aliases)
    labels = set(available_labels(events, aliases=aliases))

    candidate_labels: list[str] = []
    if canonical and canonical not in QUERY_STOPWORDS:
        candidate_labels.append(canonical)
        # Handle simple plurals: oranges -> orange, pizzas -> pizza.
        if canonical.endswith("es") and len(canonical) > 3:
            candidate_labels.append(canonical[:-2])
        if canonical.endswith("s") and len(canonical) > 2:
            candidate_labels.append(canonical[:-1])
    for cand in candidate_labels:
        if cand in labels:
            return cand

    normalized_tokens: list[str] = []
    for tok in tokens:
        mapped = LABEL_TOKEN_ALIASES.get(tok, tok)
        normalized_tokens.append(mapped)
        if mapped.endswith("es") and len(mapped) > 3:
            normalized_tokens.append(mapped[:-2])
        if mapped.endswith("s") and len(mapped) > 2:
            normalized_tokens.append(mapped[:-1])

    inferred = _infer_label_from_events(events, normalized_tokens, aliases=aliases)
    if inferred:
        return inferred

    # Token-level fallback for mixed phrases like "people can sit in room".
    for tok in normalized_tokens:
        if tok in labels:
            return tok
    return None


def _stable_instance_count(
    events: list[dict[str, Any]],
    label: str,
    aliases: dict[str, str] | None = None,
) -> tuple[int, int]:
    per_instance_hits: dict[str, int] = {}
    per_instance_max_conf: dict[str, float] = {}
    matching_events = 0
    for ev in events:
        ev_label = canonicalize_label(str(ev.get("label", "")), aliases=aliases)
        if ev_label != label:
            continue
        matching_events += 1
        iid = str(ev.get("instance_id", "")).strip()
        if not iid:
            continue
        per_instance_hits[iid] = per_instance_hits.get(iid, 0) + 1
        conf = _event_confidence(ev)
        prev = per_instance_max_conf.get(iid, 0.0)
        if conf > prev:
            per_instance_max_conf[iid] = conf

    stable_ids = [
        iid
        for iid, hits in per_instance_hits.items()
        if hits >= COUNT_MIN_INSTANCE_HITS
        and per_instance_max_conf.get(iid, 0.0) >= COUNT_MIN_INSTANCE_MAX_CONFIDENCE
    ]
    if stable_ids:
        return len(stable_ids), matching_events

    # Fallback for short clips or sparse detections.
    raw_instance_count = len(per_instance_hits)
    if raw_instance_count > 0:
        return raw_instance_count, matching_events
    return matching_events, matching_events


def _confidence_band(score: float) -> str:
    if score >= 0.8:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def _event_evidence_row(
    event: dict[str, Any],
    aliases: dict[str, str] | None = None,
) -> dict[str, Any]:
    label = canonicalize_label(str(event.get("label", "")), aliases=aliases)
    return {
        "kind": "event",
        "video_second": _event_second(event),
        "video_time_s": event.get("video_time_s"),
        "label": label,
        "detected_color": event.get("detected_color"),
        "context": str(event.get("context", "")),
        "relation": str(event.get("relation", "")),
        "anchor_label": str(event.get("anchor_label", "")),
        "confidence": event.get("confidence"),
        "thumbnail_path": event.get("thumbnail_path"),
        "frame_path": None,
        "frame_id": str(event.get("frame_id", "")).strip(),
        "instance_id": event.get("instance_id"),
    }


def _frame_evidence_row(frame_record: dict[str, Any]) -> dict[str, Any]:
    labels = frame_record.get("labels")
    label_text = ", ".join(labels) if isinstance(labels, list) else ""
    summary_text = str(frame_record.get("summary_text", "")).strip()
    raw_caption = frame_record.get("caption_text")
    caption_text = ""
    if isinstance(raw_caption, str):
        caption_text = raw_caption.strip()
    merged_context = summary_text
    if caption_text:
        merged_context = f"{summary_text} Caption: {caption_text}".strip()
    return {
        "kind": "frame",
        "video_second": int(frame_record.get("video_second", -1)),
        "video_time_s": frame_record.get("video_time_s"),
        "label": label_text,
        "detected_color": None,
        "context": merged_context,
        "caption_text": caption_text,
        "relation": "",
        "anchor_label": "",
        "confidence": None,
        "thumbnail_path": None,
        "frame_path": frame_record.get("frame_path"),
        "frame_id": str(frame_record.get("frame_id", "")).strip(),
        "instance_id": None,
    }


def _score_evidence(
    row: dict[str, Any],
    q_tokens: set[str],
    second_hint: int | None,
    primary_label: str | None,
) -> tuple[float, int]:
    blob = " ".join(
        [
            str(row.get("label", "")),
            str(row.get("detected_color", "")),
            str(row.get("context", "")),
            str(row.get("caption_text", "")),
            str(row.get("relation", "")),
            str(row.get("anchor_label", "")),
        ]
    )
    tokens = _tokenize_query(blob)
    overlap = len(tokens.intersection(q_tokens)) if q_tokens else 0
    score = float(overlap)
    if primary_label and primary_label in str(row.get("label", "")):
        score += 2.5
    if second_hint is not None:
        sec = int(row.get("video_second", -1))
        if sec >= 0:
            dist = abs(sec - second_hint)
            score += max(0.0, 2.0 - (dist / 5.0))
    return score, int(row.get("video_second", -1))


def _frame_summary_from_events(
    frame_id: str,
    events_by_frame: dict[str, list[dict[str, Any]]],
) -> str:
    rows = events_by_frame.get(frame_id, [])
    if not rows:
        return ""
    labels = sorted({str(ev.get("label", "")).strip() for ev in rows if str(ev.get("label", "")).strip()})
    contexts = sorted({str(ev.get("context", "")).strip() for ev in rows if str(ev.get("context", "")).strip()})
    label_text = ", ".join(labels[:8]) if labels else "objects"
    context_text = ", ".join(contexts[:5]) if contexts else "in_scene"
    return f"Visible: {label_text}. Contexts: {context_text}."


def _build_citations_markdown(rows: list[dict[str, Any]], max_rows: int) -> str:
    lines: list[str] = ["### Evidence"]
    for idx, ev in enumerate(rows[:max_rows], start=1):
        sec = ev.get("video_second")
        label = ev.get("label") or "unknown"
        color = ev.get("detected_color") or "unknown"
        ctx = ev.get("context") or "unknown"
        conf = ev.get("confidence")
        conf_text = "n/a" if conf is None else str(conf)
        frame_path = str(ev.get("frame_path") or "").strip()
        thumb_path = str(ev.get("thumbnail_path") or "").strip()
        parts = [
            f"- [E{idx}] second `{sec}`",
            f"`{label}`",
            f"color `{color}`",
            f"context `{ctx}`",
            f"confidence `{conf_text}`",
        ]
        links: list[str] = []
        if frame_path:
            links.append(f"[frame]({frame_path})")
        if thumb_path:
            links.append(f"[thumbnail]({thumb_path})")
        if links:
            parts.append(" | " + " ".join(links))
        lines.append(", ".join(parts))
    return "\n".join(lines)


def _row_text_for_semantic(row: dict[str, Any]) -> str:
    return " ".join(
        [
            str(row.get("label", "")),
            str(row.get("detected_color", "")),
            str(row.get("context", "")),
            str(row.get("caption_text", "")),
            str(row.get("relation", "")),
            str(row.get("anchor_label", "")),
        ]
    ).strip()


def _aggregate_visual_answers(evidence: list[dict[str, Any]]) -> tuple[str | None, float]:
    answer_scores: dict[str, float] = {}
    for row in evidence:
        ans = str(row.get("visual_answer", "")).strip().lower()
        score = float(row.get("visual_score", 0.0))
        if not ans or score <= 0.0:
            continue
        answer_scores[ans] = answer_scores.get(ans, 0.0) + score
    if not answer_scores:
        return None, 0.0

    # Prefer explicit yes/no aggregation when applicable.
    yes_score = answer_scores.get("yes", 0.0)
    no_score = answer_scores.get("no", 0.0)
    if yes_score > 0.0 or no_score > 0.0:
        return ("yes", yes_score) if yes_score >= no_score else ("no", no_score)

    best = max(answer_scores.items(), key=lambda kv: kv[1])
    return str(best[0]), float(best[1])


def ask_memory(
    events: list[dict[str, Any]],
    frame_records: list[dict[str, Any]],
    question: str,
    limit: int = 5,
    aliases: dict[str, str] | None = None,
    use_llm: bool = False,
    llm_model: str | None = None,
    llm_base_url: str | None = None,
    llm_api_key: str | None = None,
    strict_evidence: bool = False,
) -> dict[str, Any]:
    if limit <= 0:
        limit = 5

    intent = "open_visual"
    q_tokens = _tokenize_query(question)
    if not q_tokens:
        q_tokens = _tokenize_query(normalize_query_label(question, aliases=aliases))
    second_hint = _extract_second_hint(question)
    events_by_frame: dict[str, list[dict[str, Any]]] = {}
    for ev in events:
        fid = str(ev.get("frame_id", "")).strip()
        if fid:
            events_by_frame.setdefault(fid, []).append(ev)

    rows: list[dict[str, Any]] = []
    for frame in frame_records:
        fr = dict(frame)
        if not str(fr.get("summary_text", "")).strip():
            fid = str(fr.get("frame_id", "")).strip()
            fr["summary_text"] = _frame_summary_from_events(fid, events_by_frame)
        row = _frame_evidence_row(fr)
        score = _score_evidence(row, q_tokens, second_hint, primary_label=None)
        row["score"] = score
        rows.append(row)

    if not rows:
        # Fallback for legacy memory logs with no frame records.
        for frame_id, frame_events in events_by_frame.items():
            sec = max((_event_second(ev) for ev in frame_events), default=-1)
            labels = sorted(
                {
                    canonicalize_label(str(ev.get("label", "")), aliases=aliases)
                    for ev in frame_events
                    if str(ev.get("label", "")).strip()
                }
            )
            row = {
                "kind": "frame",
                "video_second": sec,
                "video_time_s": frame_events[0].get("video_time_s") if frame_events else None,
                "label": ", ".join(labels),
                "detected_color": None,
                "context": _frame_summary_from_events(frame_id, events_by_frame),
                "caption_text": "",
                "relation": "",
                "anchor_label": "",
                "confidence": None,
                "thumbnail_path": None,
                "frame_path": frame_events[0].get("thumbnail_path") if frame_events else None,
                "frame_id": frame_id,
                "instance_id": None,
            }
            row["score"] = _score_evidence(row, q_tokens, second_hint, primary_label=None)
            rows.append(row)

    if not rows:
        return {
            "found": False,
            "query": question,
            "intent": intent,
            "answer": "I couldn't find frame memory to reason over yet.",
            "evidence": [],
            "citations_markdown": "",
            "confidence": "low",
            "abstained": True,
            "abstain_reason": "No frame records were available for visual reasoning.",
            "llm": {"enabled": use_llm, "used": False},
        }

    semantic_texts = [_row_text_for_semantic(r) for r in rows]
    sem_scores = semantic_similarity_scores(question, semantic_texts)
    for idx, row in enumerate(rows):
        lex_score = float(row.get("score", (0.0, -1))[0])
        sem_score = float(sem_scores[idx]) if idx < len(sem_scores) else 0.0
        row["lexical_score"] = lex_score
        row["semantic_score"] = sem_score
        row["score"] = (lex_score + (sem_score * 3.0), int(row.get("score", (0.0, -1))[1]))

    candidate_limit = max(limit * 4, 12)
    candidates = sorted(
        rows,
        key=lambda x: (x.get("score", (0.0, -1))[0], x.get("score", (0.0, -1))[1]),
        reverse=True,
    )[:candidate_limit]

    image_paths = [str(r.get("frame_path", "")).strip() for r in candidates]
    vqa_results = visual_qa_answers(question, image_paths)
    visual_model_ready = visual_qa_model_available()
    visual_nonempty = False
    for idx, row in enumerate(candidates):
        vqa = vqa_results[idx] if idx < len(vqa_results) else {"answer": "", "score": 0.0}
        vqa_answer = str(vqa.get("answer", "")).strip()
        vqa_score = float(vqa.get("score", 0.0))
        if vqa_answer or vqa_score > 0:
            visual_nonempty = True
        row["visual_answer"] = vqa_answer
        row["visual_score"] = vqa_score
        base_score = float(row.get("score", (0.0, -1))[0])
        row["score"] = (base_score + (vqa_score * 4.0), int(row.get("score", (0.0, -1))[1]))

    evidence = sorted(
        candidates,
        key=lambda x: (x.get("score", (0.0, -1))[0], x.get("score", (0.0, -1))[1]),
        reverse=True,
    )[:limit]

    if not evidence:
        return {
            "found": False,
            "query": question,
            "intent": intent,
            "answer": "I couldn't find strong evidence for that in memory yet.",
            "evidence": [],
            "citations_markdown": "",
            "confidence": "low",
        }

    top_row = evidence[0]
    top_semantic = float(top_row.get("semantic_score", 0.0))
    top_lexical = float(top_row.get("lexical_score", 0.0))
    top_visual = float(top_row.get("visual_score", 0.0))
    if strict_evidence and (
        top_visual < 0.18
        and top_semantic < STRICT_OPEN_MIN_SEMANTIC_SCORE
        and top_lexical < STRICT_OPEN_MIN_LEXICAL_SCORE
    ):
        return {
            "found": False,
            "query": question,
            "intent": intent,
            "primary_label": None,
            "answer": "I don't have enough grounded visual evidence in this video memory to answer that reliably.",
            "evidence": evidence,
            "citations_markdown": _build_citations_markdown(evidence, max_rows=min(5, limit)),
            "confidence": "low",
            "abstained": True,
            "abstain_reason": (
                f"Weak retrieval signal (visual={top_visual:.3f}, semantic={top_semantic:.3f}, lexical={top_lexical:.3f})."
            ),
            "llm": {"enabled": use_llm, "used": False},
            "visual_reasoner": {
                "model_ready": visual_model_ready,
                "answers_produced": visual_nonempty,
            },
        }

    best_answer, best_answer_score = _aggregate_visual_answers(evidence)
    top_score = float(evidence[0].get("score", (0.0, -1))[0])
    confidence = _confidence_band(min(1.0, top_score / 6.0))
    seconds = [str(ev.get("video_second")) for ev in evidence[:3]]
    seconds_text = ", ".join(seconds)
    if best_answer:
        answer = (
            f"Most likely answer: `{best_answer}` (visual confidence aggregate: {best_answer_score:.2f}). "
            f"Grounded in frames around seconds {seconds_text}."
        )
    else:
        lines = []
        for idx, ev in enumerate(evidence[:3], start=1):
            sec = ev.get("video_second")
            label = ev.get("label")
            ctx = ev.get("context")
            lines.append(f"{idx}) second {sec}: {label} in context '{ctx}'")
        answer = "Best grounded evidence from video memory:\n" + "\n".join(lines)

    citations_markdown = _build_citations_markdown(evidence, max_rows=min(5, limit))
    llm_meta: dict[str, Any] = {"enabled": use_llm, "used": False}
    if use_llm:
        llm_result = synthesize_answer(
            question=question,
            evidence=evidence,
            model=llm_model,
            base_url=llm_base_url,
            api_key=llm_api_key,
        )
        if llm_result.get("ok"):
            answer = str(llm_result.get("answer", answer))
            # Keep judge-facing traceability with explicit evidence citations in all LLM responses.
            if citations_markdown and "### Evidence" not in answer:
                answer = f"{answer}\n\n{citations_markdown}"
            llm_meta = {
                "enabled": True,
                "used": True,
                "model": llm_result.get("model"),
                "base_url": llm_result.get("base_url"),
            }
        else:
            llm_meta = {
                "enabled": True,
                "used": False,
                "error": llm_result.get("error"),
            }

    return {
        "found": True,
        "query": question,
        "intent": intent,
        "primary_label": None,
        "answer": answer,
        "evidence": evidence,
        "citations_markdown": citations_markdown,
        "confidence": confidence,
        "abstained": False,
        "llm": llm_meta,
        "visual_reasoner": {
            "model_ready": visual_model_ready,
            "answers_produced": visual_nonempty,
        },
    }
