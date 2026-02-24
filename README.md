<div align="center">

# AURA
### Augmented Understanding & Relational Archive

**An on-device AI assistant for smart glasses — built around vision, voice, and memory.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Qualcomm AI Hub](https://img.shields.io/badge/Qualcomm%20AI%20Hub-NPU--first-3253DC?style=flat-square)](https://aihub.qualcomm.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](https://opensource.org/licenses/MIT)

</div>

---

## What is AURA?

AURA (Augmented Understanding & Relational Archive) is a proof-of-concept AI assistant designed for the form factor of smart/AR glasses. It processes video and audio in real time — recognizing faces, understanding scenes, and remembering objects — entirely on-device with no cloud dependency.

The project is built as a **unified Streamlit console** that simulates the experience of wearing smart glasses. A single video input feeds into three parallel AI "lenses", each handling a different perceptual task. Hardware acceleration is handled via Qualcomm AI Hub's QNN execution provider, with automatic fallback to CPU.

---

## The Three Lenses

### Face ID Lens
Identifies people in video frames and matches them against an enrolled face database. When a face can't be matched, the system listens to the audio track — transcribing it via Whisper and feeding the transcript through an LLM to extract likely names. Those names are offered as enrollment suggestions, closing the loop from audio to visual identity.

**Pipeline:** Video frames → Haar cascade detection → CavaFace embedding (NPU) → cosine similarity match → unknown face tracking → Whisper transcription → LLM name extraction → auto-enroll

### Scene Insight Lens
Let the user draw a rectangle over a paused video frame. The selected region is analysed by a CLIP + BLIP inference engine to produce a structured scene summary: scene type, detected attributes, likely actions, and a natural language caption.

**Modes:** CLIP (fast zero-shot), BLIP (natural language captioning), Hybrid (both combined)

### Find-My-Object Lens
Ingests a video using a YOLOv11 detector, tracking each object instance across frames. Every detection is stored as a timestamped event in a JSONL memory log, along with a cropped thumbnail and colour classification. The user can then query the memory in plain English — *"where did I last see my blue backpack?"* — and get a grounded answer with a timestamp and spatial context.

**Query engine:** Natural language → label canonicalisation → colour-aware instance resolution → `where_is()` / `timeline()` / `describe_color()` → structured answer

---

## Feature Branch: Mind Palace

> `feature/mind-palace-memory`

Mind Palace extends the Find-My-Object lens with a full **retrieval-augmented generation (RAG)** layer. Instead of keyword-matching against the memory log, it embeds queries and evidence using a sentence-transformer model, retrieves semantically relevant events, and synthesises a grounded narrative answer via an LLM.

New capabilities on this branch:
- Semantic similarity search over stored events (`local_rag.py`)
- VIT-GPT2 image captioning and ViLT visual Q&A run against retrieved thumbnails
- LLM synthesis with grounded evidence bullets (`llm_reasoner.py`)
- Colour classification refinements for better instance disambiguation

See the [extractor README on that branch](extractor/README.md) for full details.

---

## Project Layout

```
aura/
├── demo-ui/
│   ├── unified_app.py      Streamlit app combining all three lenses
│   └── app.py              Mind Palace standalone UI (feature branch)
│
├── FaceRecon/
│   ├── pipeline.py         Whisper → LLM names → CavaFace pipeline
│   ├── run_recognition.py  CLI entrypoint
│   └── ...
│
├── image-summarizer/
│   ├── engine/
│   │   ├── inference_engine.py   CLIP + BLIP orchestrator
│   │   ├── clip_summarizer.py    Zero-shot scene classification
│   │   └── blip_captioner.py     Natural language captioning
│   └── ...
│
├── extractor/
│   ├── run_ingest.py        Ingest video → memory JSONL
│   ├── run_query.py         CLI query tool
│   └── src/
│       ├── ingest_video.py  YOLO detection + instance tracking
│       ├── search.py        Query engine (where_is, timeline, etc.)
│       ├── detector_qaihub.py   YOLOv11/v8 via Qualcomm AI Hub
│       ├── storage.py       JSONL event log + thumbnail writer
│       ├── color_utils.py   Dominant colour detection (mind-palace)
│       ├── local_rag.py     Semantic RAG over memory (mind-palace)
│       └── llm_reasoner.py  LLM synthesis from evidence (mind-palace)
│
└── requirements.txt
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- `ffmpeg` on PATH (recommended for Whisper compatibility)
- For NPU acceleration: Snapdragon device with `onnxruntime-qnn` installed
- For LLM name extraction: [Ollama](https://ollama.ai/) running locally, or an OpenAI-compatible API key

### Install

```bash
git clone https://github.com/prasanth-kps/aura.git
cd aura
pip install -r requirements.txt
```

### Run the Unified App

```bash
streamlit run demo-ui/unified_app.py
```

Upload a video using the file picker at the top. Each lens tab operates independently on that video.

---

## Module Guides

| Module | README |
|---|---|
| Face ID Lens | [FaceRecon/README.md](FaceRecon/README.md) |
| Scene Insight Lens | [image-summarizer/README.md](image-summarizer/README.md) |
| Find-My-Object Lens | [extractor/README.md](extractor/README.md) |
| Unified UI | [demo-ui/README.md](demo-ui/README.md) |

---

## Hardware Acceleration

AURA is built with NPU-first execution in mind. All inference pipelines attempt to load models via `QNNExecutionProvider` (Qualcomm Neural Processing SDK) and fall back to CPU automatically. No code changes are needed to switch between hardware targets.

| Component | NPU Model | Fallback |
|---|---|---|
| Face Embedding | CavaFace (ONNX/QNN) | PyTorch CPU |
| Speech Recognition | Whisper (faster-whisper int8) | CPU |
| Object Detection | YOLOv11 (QAI Hub) | YOLOv8 CPU |
| Image Captioning | BLIP (ONNX/QNN) | Transformers CPU |
| Semantic Search | MiniLM-L6-v2 | CPU |

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Face Recognition | CavaFace via Qualcomm AI Hub |
| Speech-to-Text | faster-whisper (CPU int8) |
| Object Detection | YOLOv11 / YOLOv8 via QAI Hub |
| Scene Understanding | CLIP + BLIP (Transformers / Optimum) |
| Semantic Search | sentence-transformers/all-MiniLM-L6-v2 |
| LLM Inference | Ollama (local) or OpenAI-compatible API |
| Acceleration | ONNX Runtime + QNN Execution Provider |

---

## Built by

[Prasanth KPS](https://github.com/prasanth-kps) · [@prasanth-kps](https://github.com/prasanth-kps)
