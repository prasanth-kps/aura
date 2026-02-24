# extractor — Find-My-Object Lens

Ingests video with a YOLO object detector, tracks every object instance across time, and stores a queryable memory log. Ask it in plain English where you last saw something.

---

## Concept

Every time you record something, Aura watches the video frame-by-frame, detects objects with a YOLOv11 detector, assigns persistent instance IDs as objects move across frames, and writes a timestamped event log. Later, you can query that log — "where's my black backpack?" — and get a grounded answer with a second timestamp and spatial context (e.g., *"next to the chair"*).

---

## Architecture

```
Video file
    │
    ▼
QualcommYoloDetector (YOLOv11 via QAI Hub → fallback YOLOv8 CPU)
    │  Runs at configurable FPS (default 3)
    ▼
Instance Tracker
    │  IoU + centre-distance matching across frames
    │  Each unique object gets a stable instance_id
    ▼
Spatial Contextualiser
    │  Classifies each detection relative to an "anchor" object
    │  (e.g. "on the table", "next to the chair")
    ▼
Memory JSONL  ──→  data/memory/<video>.events.jsonl
Thumbnails    ──→  data/crops/<video>/s<second>_<idx>_<label>.jpg
    │
    ▼
Query Engine
    │  where_is()   → last seen location + timestamp
    │  timeline()   → recent sightings
    │  describe_color() → colour of an object instance
    ▼
Natural language answer
```

---

## Key files

| File | Purpose |
|---|---|
| `run_ingest.py` | CLI — ingest a video into memory |
| `run_query.py` | CLI — query the memory log |
| `src/ingest_video.py` | Frame sampling, detection, tracking, event writing |
| `src/detector_qaihub.py` | YOLOv11 (QNN) + YOLOv8 (CPU) detector wrapper |
| `src/search.py` | Query engine: label/colour/instance resolution, `where_is` |
| `src/storage.py` | JSONL event append, thumbnail writer |
| `src/anchors.py` | Anchor class definitions (furniture, rooms, etc.) |
| `src/geometry.py` | IoU, centre distance, spatial relation helpers |
| `src/color_utils.py` | Dominant colour detection from thumbnail crops |

---

## Ingesting a video

```bash
python extractor/run_ingest.py --video path/to/video.mp4
```

Output:
- `data/memory/<video>.events.jsonl` — event log (one JSON object per detection)
- `data/crops/<video>/` — 50×50 JPEG thumbnails for each detected object

```bash
# Full options
python extractor/run_ingest.py \
    --video path/to/video.mp4 \
    --sample-fps 3.0 \        # frames to sample per second
    --conf-threshold 0.25 \   # YOLO confidence threshold
    --detector-input-size 960 \
    --runtime auto \          # auto / qnn / cpu
    --reset                   # clear prior output for this video
```

---

## Querying memory

```bash
python extractor/run_query.py \
    --video path/to/video.mp4 \
    --query "where is my blue backpack"
```

Natural language queries are parsed into a label + optional colour + optional instance ID. Supported query forms:

```
"where is my laptop"
"where did I last see the red mug"
"find my keys"
"what color is the chair"
"show me the phone"
```

---

## Event log format

Each line in `*.events.jsonl` is a JSON object:

```json
{
  "video_id": "meeting",
  "ingestion_run_id": "meeting-20250210T120000Z-a1b2c3d4",
  "video_second": 42,
  "ingested_at_utc": "2025-02-10T12:00:00+00:00",
  "detector_model": "yolov11",
  "instance_id": "meeting-20250210T120000Z-a1b2c3d4:laptop_001",
  "label": "laptop",
  "confidence": 0.87,
  "bbox_xyxy": [120, 80, 450, 320],
  "anchor_label": "dining table",
  "relation": "on",
  "context": "on dining table",
  "detected_color": "gray",
  "thumbnail_path": "data/crops/meeting/s000042_000_laptop.jpg"
}
```

---

## Instance tracking

Objects are tracked across frames using a greedy IoU + centre-distance algorithm:

- Detections above `--iou-threshold` and within `--track-center-dist-ratio` of an existing track are assigned to that track
- Tracks not seen for `--track-max-gap-seconds` seconds are pruned
- Each unique track gets a stable `instance_id` of the form `<run_id>:<label>_<counter>`

This means repeated sightings of *the same physical object* accumulate under one instance ID, enabling per-instance colour and location queries.

---

## Colour-aware queries

Thumbnails are classified by dominant colour at ingest time. The query engine resolves colour queries in two stages:

1. Filter events by label
2. Score each instance by the ratio of colour-matching thumbnails
3. Return the best-matching instance

This lets queries like *"the blue chair"* correctly distinguish between multiple chairs of different colours in the same video.

---

## Spatial context

Detections are classified relative to "anchor" objects — stable scene elements like tables, chairs, and sofas. The relation is one of: `on`, `near`, `above`, `below`, `beside`, or `in_scene` (no anchor nearby).

This allows answers like:
> *"Last seen at second 84: laptop is on dining table."*

---

## Requirements

```bash
pip install -r extractor/requirements.txt
```

Core deps: `opencv-python`, `qai-hub-models`, `numpy`
