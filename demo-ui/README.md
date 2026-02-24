# demo-ui — Unified Streamlit Console · AURA

The single-page interface that brings all three AURA lenses together.

---

## Overview

`unified_app.py` is a Streamlit application that simulates a smart glasses console. One video upload at the top feeds all three AURA lenses — Face ID, Scene Insight, and Find-My-Object — each available in its own tab.

State is managed via `st.cache_resource` so that background threads (face stream, Whisper transcription) survive Streamlit script reruns without being torn down.

---

## Running

```bash
streamlit run demo-ui/unified_app.py
```

Open `http://localhost:8501` in your browser.

---

## Layout

```
┌─────────────────────────────────────────────────────┐
│  Video Upload (shared across all lenses)            │
├──────────────┬───────────────┬──────────────────────┤
│  Face ID     │ Scene Insight │ Find-My-Object       │
│  Lens        │ Lens          │ Lens                 │
└──────────────┴───────────────┴──────────────────────┘
```

### Face ID Lens tab
- Starts a background thread that reads video frames and runs CavaFace embedding
- Whisper transcription runs concurrently on the audio track
- Detected faces are overlaid with name + confidence in a live frame display
- Transcript-derived name hints populate auto-enrollment suggestions
- Enrollment UI shows for unknown faces that persist across frames

### Scene Insight Lens tab
- Plays the video and offers a **Pause & Inspect** button
- When paused, a drawable canvas appears over the current frame
- User draws a rectangle → CLIP/BLIP inference runs on that region
- Result panel shows: natural language caption, top scene/attribute/action labels, inference time, model used

### Find-My-Object Lens tab
- **Ingest** button runs the YOLO detector across the full video
- Progress and stats (events written, detector backend) shown in real time
- **Query** input accepts natural language — "where is my laptop", "find the red mug"
- Result shows: last seen timestamp, spatial context, detected colour, confidence
- **Memory Browser** tab shows all ingested events as a filterable table

---

## Transcript caching

Whisper transcription is expensive. The app caches transcripts keyed by a SHA-256 hash of the first 512 KB of the video file. Cache is stored in `%TEMP%/aura_whisper_cache/` and survives app restarts.

To force re-transcription, delete the cache directory or the specific `<hash>.json` file.

---

## LLM settings (sidebar)

| Setting | Default | Notes |
|---|---|---|
| Provider | `ollama` | `ollama` or `openai` |
| API Base | `http://localhost:11434` | Ollama default |
| Model | `llama3.2:3b` | Any Ollama model |
| API Key | *(empty)* | Required for OpenAI-compatible providers |

---

## Backend status

The sidebar shows live backend indicators for each AI component:

- **Face accelerator**: `npu-qnn` / `directml` / `cuda` / `cpu`
- **Whisper backend**: `faster-whisper (cpu int8)`
- **Detector backend**: `qnn` / `cpu`
- **Caption / Semantic / VQA**: `onnx/qnn` / `transformers/default` / `unavailable`

---

## Dependencies

All dependencies are covered by the root `requirements.txt`. The UI additionally requires:

```
streamlit>=1.34.0
streamlit-drawable-canvas
psutil>=5.9.0
```

---

## Mind Palace UI (feature branch)

`app.py` on the `feature/mind-palace-memory` branch is a standalone Streamlit interface specifically for the Mind Palace experience — semantic memory retrieval with LLM synthesis. It does not require the Face ID or Scene Insight lenses.

```bash
# On feature/mind-palace-memory branch
streamlit run demo-ui/app.py
```
