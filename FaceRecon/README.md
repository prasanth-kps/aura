# FaceRecon — Face ID Lens

Real-time face recognition that uses speech to identify the people it can't yet recognise.

---

## How it works

The pipeline runs in four sequential stages:

```
Video file
    │
    ▼
[1] Whisper (faster-whisper, CPU int8)
    Transcribes the audio track to text
    │
    ▼
[2] LLM Name Extraction (Ollama / OpenAI-compatible)
    Parses the transcript for introduced names
    Fallback: regex patterns ("Hi, I'm ...")
    │
    ▼
[3] CavaFace — Face Embedding (Qualcomm AI Hub)
    Haar cascade detection → 112×112 crop → L2-normalised 512-dim vector
    Matched against embeddings_db.npz by cosine similarity
    │
    ▼
[4] Unknown Face Handling
    Persists across N frames → prompts for enrollment
    In "assist" mode: auto-enrolls using the top LLM name suggestion
```

When a face is recognised, its name and confidence score are overlaid on the frame. When a face remains unknown for `--min-frames` consecutive frames, the system uses the names extracted from audio as enrollment hints — so if someone says their name in the video, the face is automatically labelled.

---

## Key files

| File | Purpose |
|---|---|
| `pipeline.py` | Main entry point — full Whisper → LLM → face pipeline |
| `run_recognition.py` | CLI wrapper for compile + run workflow |
| `compile_models.py` | Exports CavaFace to ONNX; compiles for QNN/DirectML |
| `parallel_pipeline.py` | Multithreaded variant with producer/consumer queues |
| `llm_name_recognizer.py` | LLM name extraction (Ollama + OpenAI-compatible) |
| `audio_name_recognizer.py` | Regex/heuristic fallback name extraction |
| `npu_whisper_loader.py` | Loads Whisper with NPU acceleration if available |
| `voice_transcribe_live.py` | Live microphone transcription utility |

---

## Running the pipeline

```bash
# Basic: transcribe + extract names + face recognition
python FaceRecon/pipeline.py --video path/to/video.mp4

# Auto-enroll faces from transcript (assist mode)
python FaceRecon/pipeline.py --video path/to/video.mp4 --mode assist

# Use a different Whisper model
python FaceRecon/pipeline.py --video path/to/video.mp4 --whisper-model small

# Use OpenAI instead of Ollama for name extraction
python FaceRecon/pipeline.py --video path/to/video.mp4 \
    --llm-provider openai \
    --llm-api-key sk-... \
    --llm-model gpt-4o-mini
```

---

## CLI reference

| Flag | Default | Description |
|---|---|---|
| `--video` | required | Path to video file |
| `--mode` | `assist` | `assist` = auto-enroll; `camera` = interactive prompt |
| `--whisper-model` | `base` | Whisper model size: tiny / base / small / medium / large-v2 |
| `--llm-provider` | `ollama` | `ollama` or `openai` |
| `--llm-api-base` | `http://localhost:11434` | API endpoint |
| `--llm-model` | `llama3.2:3b` | Model name |
| `--db` | `embeddings_db.npz` | Path to face embeddings database |
| `--threshold` | `0.50` | Cosine similarity threshold for a match |
| `--face-margin` | `0.25` | Padding ratio around detected face crop |
| `--min-frames` | `5` | Frames an unknown face must persist before enrollment prompt |
| `--accelerator` | `auto` | `auto` / `gpu` / `npu` / `directml` / `cpu` |
| `--save-transcript` | off | Save transcript as `.txt` alongside video |
| `--skip-transcription` | off | Skip Whisper, pass names via `--intro-text` |

---

## Face database

Faces are stored in `embeddings_db.npz` as L2-normalised 512-dimensional centroids. When a name is enrolled multiple times, the system computes the centroid of all embeddings for that identity, keeping the database compact.

```python
# Structure
data["names"]       # np.ndarray of str, shape (N,)
data["embeddings"]  # np.ndarray of float32, shape (N, 512)
```

---

## Acceleration

| Backend | How to activate |
|---|---|
| QNN / NPU | `--accelerator npu` (requires `onnxruntime-qnn`) |
| DirectML | `--accelerator directml` (requires `onnxruntime-directml`) |
| CUDA | `--accelerator gpu` |
| CPU (default) | `--accelerator cpu` |

The pipeline auto-exports the CavaFace model to ONNX on first run if the cached file doesn't exist.

---

## LLM setup (Ollama)

```bash
ollama serve
ollama pull llama3.2:3b
```

Any Ollama model works. The prompt asks for a comma-separated list of full names from the transcript — the model doesn't need to be large.

---

## Requirements

```bash
pip install -r FaceRecon/requirements.txt
```

Core deps: `faster-whisper`, `qai-hub-models[cavaface]`, `torch`, `opencv-python`, `numpy`
