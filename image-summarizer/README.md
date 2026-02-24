# image-summarizer — Scene Insight Lens

Region-based scene understanding using CLIP and BLIP. Draw a box over any area of a frame and get a structured description of what's in it.

---

## How it works

The user pauses a video frame in the Streamlit UI and draws a rectangle over the region of interest. That bounding box is passed to the inference engine, which crops the region and runs it through one or more AI models.

Three summarisation modes are available:

| Mode | Model | Output | Speed |
|---|---|---|---|
| `clip` | CLIP image encoder + zero-shot prompts | Scene type, attributes, actions | Fast |
| `blip` | BLIP vision encoder + text decoder | Natural language caption | Moderate |
| `hybrid` | Both combined | Caption + structured metadata | Slowest, best quality |

---

## Architecture

```
PIL Image + BBox (x1, y1, x2, y2)
    │
    ▼
InferenceEngine.summarize()
    │
    ├── CLIP path
    │   ├── Crop + preprocess (224×224, normalised)
    │   ├── ONNX/QNN image encoder
    │   ├── Dot product against precomputed text embeddings
    │   └── Top scenes / attributes / actions
    │
    ├── BLIP path
    │   ├── Crop + preprocess
    │   ├── Vision encoder (ONNX/QNN)
    │   └── Text decoder → natural language caption
    │
    └── Hybrid: BLIP caption + CLIP scene labels combined
```

---

## Key files

| File | Purpose |
|---|---|
| `engine/inference_engine.py` | Main orchestrator — single entry point for the UI |
| `engine/clip_summarizer.py` | Zero-shot scene/attribute/action classification |
| `engine/blip_captioner.py` | Natural language captioning |
| `engine/preprocessing.py` | Image crop + normalisation utilities |
| `models/prompts.json` | CLIP prompt templates (scenes, attributes, actions) |
| `scripts/export_clip.py` | Export CLIP image encoder to ONNX |
| `scripts/export_blip.py` | Export BLIP vision encoder to ONNX |
| `scripts/precompute_text_embeddings.py` | Precompute and cache CLIP text embeddings |
| `main.py` | Standalone desktop UI (PyQt6) |

---

## Usage (programmatic)

```python
from PIL import Image
from engine.inference_engine import InferenceEngine

engine = InferenceEngine(mode="hybrid")  # "clip", "blip", or "hybrid"

image = Image.open("frame.jpg").convert("RGB")
bbox = (100, 80, 420, 350)  # (x1, y1, x2, y2)

result = engine.summarize(image, bbox)
print(result["summary"])       # human-readable description
print(result["model_used"])    # which model(s) ran
print(result["total_ms"])      # total inference time
```

---

## CLIP mode — how zero-shot works

CLIP compares the visual embedding of the cropped region against a library of text embeddings precomputed from `prompts.json`. No fine-tuning is needed — prompts cover:

- **Scenes**: e.g. *"a photograph of an office"*, *"a photograph of a kitchen"*
- **Attributes**: e.g. *"a crowded area"*, *"an empty room"*, *"bright lighting"*
- **Actions**: e.g. *"a person sitting"*, *"a person walking"*, *"a person eating"*

Top matches from each category are combined into a summary string.

To add new scene categories, edit `models/prompts.json` and re-run:

```bash
python image-summarizer/scripts/precompute_text_embeddings.py
```

---

## Model export (first-time setup)

Models need to be exported to ONNX before the first run:

```bash
# Export CLIP image encoder
python image-summarizer/scripts/export_clip.py

# Export BLIP vision encoder
python image-summarizer/scripts/export_blip.py

# Precompute CLIP text embeddings
python image-summarizer/scripts/precompute_text_embeddings.py
```

On Snapdragon hardware with `onnxruntime-qnn` installed, the engine will automatically use the QNN execution provider. CPU fallback is transparent.

---

## Standalone desktop UI

A PyQt6-based desktop application is available for testing outside of Streamlit:

```bash
python image-summarizer/main.py
```

Open an image, draw a selection rectangle, and the inference engine runs on the selected region.

---

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `AURA_PREFER_QNN` | `1` | Set to `0` to force CPU even if QNN is available |
| `AURA_FORCE_OFFLINE` | `0` | Set to `1` to never download models from HuggingFace |
| `HF_HUB_OFFLINE` | `0` | Standard HuggingFace offline flag |

---

## Requirements

```bash
pip install -r image-summarizer/requirements.txt
```

Core deps: `transformers`, `Pillow`, `numpy`, `onnxruntime`, `optimum[onnxruntime]`, `sentence-transformers`, `PyQt6`
