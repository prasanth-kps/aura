# Aura

A VR-glasses-style assistant built in Streamlit to provide help from vision and audio clues.

Product intention:

- Support social awareness (identify people and infer likely names)
- Surface important context (scene summaries and key details)
- Build visual memory over time and answer queries from observed data
- Feel like a real-time assistant for smart/AR glasses experiences

Core lenses:

- Face ID Lens (live face detection + recognition)
- Scene Insight Lens (draw-box image summarization)
- Find-My-Object Lens (object memory ingest + prompt retrieval)

The app is designed for on-device acceleration with NPU-first behavior where available.

## Features

- Single shared video upload for all lenses
- Live face stream with overlays and backend status
- Parallel whisper transcription + name hint extraction
- Auto-assign unknown faces from transcript/LLM hints
- Auto-enroll inferred identities into `FaceRecon/embeddings_db.npz`
- Pause-and-inspect current frame for Scene Insight
- Memory prompt tab for querying stored object memory

## Project Layout

- `demo-ui/unified_app.py` - main Streamlit application
- `FaceRecon/` - face recognition pipeline and embeddings DB
- `image-summarizer/` - CLIP/BLIP summarization engine
- `extractor/` - object ingest/query pipeline

## Requirements

Install Python dependencies:

```bash
py -m pip install -r requirements.txt
```

Optional but recommended for better transcription compatibility:

- `ffmpeg` on PATH

Optional for LLM name extraction:

- Ollama installed and running
- Model pulled: `llama3.2:3b`

## Run

From repo root:

```bash
py -m streamlit run "c:\Users\hackathon user\Desktop\Project\aura\demo-ui\unified_app.py"
```

## LLM Setup (Name Extraction)

Install and start Ollama, then pull model:

```bash
ollama serve
ollama pull llama3.2:3b
```

In app settings use:

- Provider: `ollama`
- API Base: `http://localhost:11434`
- Model: `llama3.2:3b`

## NPU / Backend Notes

- Face model supports `npu`, `auto`, `directml`, `cpu`.
- If QNN is available, backend should show `npu-qnn` in the app.
- Terminal logs with `onnxruntime::qnn` indicate QNN/NPU execution.

## Smart Lens Workflow

1. Upload one shared video at top.
2. Start Smart Lens stream (Face + Whisper run in background).
3. Play audio/video while live inference continues.
4. Use `Pause & Inspect Current Frame`.
5. Draw a rectangle to trigger Scene Insight summary.

## Transcript Cache

Whisper transcripts are cached at:

- `%TEMP%\aura_whisper_cache`

Cache key is based on video content hash. Delete cache file to force re-transcription.

## Troubleshooting

- No transcript:
  - ensure video has audio stream
  - ensure Ollama is running for LLM hints
  - clear `%TEMP%\aura_whisper_cache` for stale cache cases
- Canvas not visible:
  - install `streamlit-drawable-canvas`
- No NPU activity:
  - select `Face Accelerator = npu`
  - confirm backend metric and terminal QNN logs

## Status

This repository currently focuses on live demo usability and hardware-accelerated inference in Streamlit.
