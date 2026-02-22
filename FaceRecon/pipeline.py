"""
Video Intelligence Pipeline
============================
1. Whisper (faster-whisper, CPU int8) transcribes the video audio
2. LLM (Ollama local or OpenAI-compatible) extracts speaker names from transcript
3. Names feed into CavaFace face recognition as enroll suggestions

Usage:
    python pipeline.py --video videoplayback.mp4
    python pipeline.py --video videoplayback.mp4 --mode assist   # auto-enroll faces
    python pipeline.py --video videoplayback.mp4 --llm-provider openai --llm-api-key sk-...
"""

import argparse
import os
import queue
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

# ── Optional imports (fail loudly with install hint) ──────────────────────────
try:
    from faster_whisper import WhisperModel
except ImportError:
    raise SystemExit("Missing: pip install faster-whisper")

try:
    from qai_hub_models.models.cavaface import Model as CavaFaceModel
except ImportError:
    raise SystemExit("Missing: pip install qai-hub-models[cavaface]")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — WHISPER TRANSCRIPTION
# ══════════════════════════════════════════════════════════════════════════════

def load_whisper(model_size: str = "base") -> WhisperModel:
    print(f"[Whisper] Loading {model_size} (CPU int8)...", flush=True)
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    print("[Whisper] Ready.", flush=True)
    return model


def transcribe_video(whisper_model: WhisperModel, video_path: str) -> str:
    """Transcribe audio from a video/audio file. Returns full transcript string."""
    print(f"[Whisper] Transcribing: {video_path}", flush=True)

    segments, info = whisper_model.transcribe(
        video_path,
        beam_size=5,
        language="en",
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        no_repeat_ngram_size=4,
    )

    print(f"[Whisper] Language: {info.language} ({info.language_probability:.2f})", flush=True)

    lines = []
    for seg in segments:
        text = seg.text.strip()
        print(f"  [{seg.start:.1f}s → {seg.end:.1f}s] {text}", flush=True)
        lines.append(text)

    transcript = " ".join(lines)
    print(f"[Whisper] Done. {len(transcript)} chars transcribed.", flush=True)
    return transcript


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — LLM NAME EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

import re


def regex_fallback_extract(text: str) -> list[str]:
    """Simple regex to catch 'I'm [Name]', 'my name is [Name]', etc."""
    patterns = [
        r"(?:I'm|I am|my name is|this is|call me|name's)\s+([A-Z][a-z]+(?: [A-Z][a-z]+)?)",
        r"(?:Hi|Hello|Hey),?\s+(?:I'm|I am)\s+([A-Z][a-z]+(?: [A-Z][a-z]+)?)",
    ]
    names = []
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            name = m.group(1).strip().title()
            if name not in names:
                names.append(name)
    return names


def extract_names_with_llm(
    text: str,
    api_base: str = "http://localhost:11434",
    api_key: str = "",
    model: str = "llama3.2:3b",
    provider: str = "ollama",
) -> list[str]:
    """
    Call a local (Ollama) or remote (OpenAI-compatible) LLM to extract
    person names mentioned in the transcript.
    """
    prompt = (
        "You are a name extractor. Read the following transcript and return ONLY "
        "a comma-separated list of the full names of people who are introduced or "
        "mention their own name. If no names are found, return an empty string. "
        "Do not include any explanation.\n\n"
        f"Transcript:\n{text}\n\nNames:"
    )

    try:
        import urllib.request, json as _json

        if provider == "ollama":
            url = f"{api_base.rstrip('/')}/api/generate"
            payload = _json.dumps({
                "model": model,
                "prompt": prompt,
                "stream": False,
            }).encode()
            req = urllib.request.Request(url, data=payload,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = _json.loads(resp.read())
            raw = data.get("response", "").strip()

        else:  # openai-compatible
            url = f"{api_base.rstrip('/')}/v1/chat/completions"
            payload = _json.dumps({
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 100,
            }).encode()
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            req = urllib.request.Request(url, data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = _json.loads(resp.read())
            raw = data["choices"][0]["message"]["content"].strip()

        # Parse comma-separated names
        names = [n.strip().title() for n in raw.split(",") if n.strip()]
        names = [n for n in names if 2 <= len(n) <= 50 and not any(c.isdigit() for c in n)]
        print(f"[LLM] Extracted names: {names}", flush=True)
        return names

    except Exception as e:
        print(f"[LLM] Failed ({e}), falling back to regex.", flush=True)
        return regex_fallback_extract(text)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — FACE RECOGNITION (CavaFace)  ← your original code, unchanged
# ══════════════════════════════════════════════════════════════════════════════

def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec) + 1e-12
    return vec / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)))


def load_local_model() -> torch.nn.Module:
    model = CavaFaceModel.from_pretrained()
    model.eval()
    return model


def resolve_torch_device(accelerator: str) -> tuple[object, str]:
    if accelerator in {"auto", "gpu"}:
        if torch.cuda.is_available():
            return torch.device("cuda"), "cuda"
        try:
            import torch_directml
            return torch_directml.device(), "directml"
        except Exception:
            pass
    return torch.device("cpu"), "cpu"


def _ensure_cavaface_onnx(model: torch.nn.Module, onnx_path: Path) -> None:
    if onnx_path.exists():
        return
    model_cpu = model.to("cpu").eval()
    dummy = torch.randn(1, 3, 112, 112, dtype=torch.float32)
    traced = torch.jit.trace(model_cpu, dummy)
    torch.onnx.export(traced, dummy, str(onnx_path),
                      input_names=["image"], output_names=["embedding"],
                      opset_version=17, do_constant_folding=True)


def build_qnn_session(model, onnx_path):
    import onnxruntime as ort
    if "QNNExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("QNNExecutionProvider not available.")
    _ensure_cavaface_onnx(model, onnx_path)
    return ort.InferenceSession(str(onnx_path),
                                providers=["QNNExecutionProvider", "CPUExecutionProvider"])


def build_dml_session(model, onnx_path):
    import onnxruntime as ort
    if "DmlExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("DmlExecutionProvider not available. pip install onnxruntime-directml")
    _ensure_cavaface_onnx(model, onnx_path)
    return ort.InferenceSession(str(onnx_path),
                                providers=["DmlExecutionProvider", "CPUExecutionProvider"])


def run_embedding_inference(model, input_tensor, model_device):
    if hasattr(model, "run") and hasattr(model, "get_inputs"):
        name = model.get_inputs()[0].name
        out = model.run(None, {name: input_tensor.astype(np.float32)})[0]
        return l2_normalize(np.asarray(out).reshape(-1).astype(np.float32))
    with torch.no_grad():
        batch = torch.from_numpy(input_tensor).float().to(model_device)
        out = model(batch)
        if isinstance(out, (tuple, list)):
            out = out[0]
    return l2_normalize(out.detach().cpu().numpy().reshape(-1).astype(np.float32))


def load_db(db_path: Path):
    if not db_path.exists():
        return [], np.empty((0, 512), dtype=np.float32)
    data = np.load(db_path, allow_pickle=False)
    return data["names"].tolist(), data["embeddings"].astype(np.float32)


def save_db(db_path: Path, names: list, embeddings: np.ndarray) -> None:
    np.savez(db_path, names=np.asarray(names, dtype=np.str_),
             embeddings=embeddings.astype(np.float32))


def enroll(db_path: Path, name: str, embedding: np.ndarray) -> None:
    names, embeddings = load_db(db_path)
    grouped: dict[str, list] = defaultdict(list)
    for n, e in zip(names, embeddings):
        grouped[n].append(e)
    grouped[name].append(embedding)
    merged_names, merged_embeddings = [], []
    for person, embs in grouped.items():
        centroid = l2_normalize(np.mean(np.vstack(embs), axis=0).astype(np.float32))
        merged_names.append(person)
        merged_embeddings.append(centroid)
    save_db(db_path, merged_names, np.vstack(merged_embeddings))
    print(f"[Face] Enrolled '{name}'. Total: {len(merged_names)}")


def best_match(names, embeddings, query_embedding, threshold, second_best_margin):
    if len(names) == 0:
        return "Unknown", -1.0
    scores = [cosine_similarity(query_embedding, e) for e in embeddings]
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    gap = (sorted(scores, reverse=True)[0] - sorted(scores, reverse=True)[1]
           if len(scores) > 1 else 1.0)
    if best_score < threshold or gap < second_best_margin:
        return "Unknown", best_score
    return names[best_idx], best_score


def expand_to_square_with_margin(frame, x, y, w, h, margin_ratio):
    h_img, w_img = frame.shape[:2]
    cx, cy = x + w / 2.0, y + h / 2.0
    side = max(w, h) * (1.0 + margin_ratio)
    x0 = int(max(0, cx - side / 2.0))
    y0 = int(max(0, cy - side / 2.0))
    x1 = int(min(w_img, cx + side / 2.0))
    y1 = int(min(h_img, cy + side / 2.0))
    return frame[y0:y1, x0:x1]


def face_bgr_to_model_input(face_bgr):
    import cv2
    resized = cv2.resize(face_bgr, (112, 112), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    arr = (rgb.astype(np.float32) / 255.0 - 0.5) / 0.5
    arr = np.transpose(arr, (2, 0, 1))
    return np.ascontiguousarray(arr[np.newaxis, ...])


def _prompt_enroll_name(result_queue, embedding, suggested_names,
                        auto_enroll_from_suggestions=False):
    print("\n[Face] Unknown face persisted across frames.")
    if suggested_names:
        print(f"[Face] Name suggestions from transcript: {', '.join(suggested_names)}")
    if auto_enroll_from_suggestions and suggested_names:
        entered = suggested_names[0]
        print(f"[Face] Assist mode auto-enroll: {entered}")
        result_queue.put((entered, embedding))
        return
    if not sys.stdin.isatty():
        entered = suggested_names[0] if suggested_names else ""
        result_queue.put((entered, embedding))
        return
    try:
        entered = input("Enter name to enroll (blank to skip): ").strip()
    except EOFError:
        entered = suggested_names[0] if suggested_names else ""
    result_queue.put((entered, embedding))


def camera_mode(db_path, face_model, model_device, threshold, second_best_margin,
                face_margin, camera_index, video_path, min_frames_to_prompt,
                suggested_names, auto_enroll_from_suggestions=False):
    try:
        import cv2
    except ImportError:
        raise RuntimeError("pip install opencv-python")

    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if detector.empty():
        raise RuntimeError("Failed to load Haar cascade.")

    source = video_path if video_path else camera_index
    cap = (cv2.VideoCapture(source) if video_path
           else cv2.VideoCapture(source, cv2.CAP_DSHOW))
    if not cap.isOpened():
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open: {source}")

    names, embeddings = load_db(db_path)
    last_prompt_time = 0.0
    prompt_cooldown = 3.0
    prompt_results: queue.Queue = queue.Queue()
    prompt_thread = None
    unknown_anchor = None
    unknown_streak = 0
    unknown_buffer = []
    consecutive_fails = 0

    print("[Face] Starting. Press 'q' to quit.", flush=True)
    try:
        while True:
            while not prompt_results.empty():
                entered, emb = prompt_results.get_nowait()
                if entered:
                    enroll(db_path, entered, emb)
                    names, embeddings = load_db(db_path)
                else:
                    print("[Face] Skipped enrollment.")
                last_prompt_time = time.time()
                prompt_thread = None

            ok, frame = cap.read()
            if not ok:
                if video_path:
                    break
                consecutive_fails += 1
                if consecutive_fails >= 30:
                    raise RuntimeError("Camera read failed repeatedly.")
                time.sleep(0.05)
                continue
            consecutive_fails = 0

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))
            frame_unknown_emb = None

            for (x, y, w, h) in faces:
                crop = expand_to_square_with_margin(frame, x, y, w, h, face_margin)
                if crop.size == 0:
                    continue
                inp = face_bgr_to_model_input(crop)
                emb = run_embedding_inference(face_model, inp, model_device)
                label, score = best_match(names, embeddings, emb,
                                          threshold, second_best_margin)
                is_unknown = label == "Unknown" or score < threshold
                y0, y1 = max(0, y), min(frame.shape[0], y + h)
                x0, x1 = max(0, x), min(frame.shape[1], x + w)
                if is_unknown:
                    label, color = "Unknown", (0, 0, 255)
                    if frame_unknown_emb is None:
                        frame_unknown_emb = emb
                else:
                    color = (0, 255, 0)
                cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
                cv2.putText(frame, f"{label} ({score:.2f})" if score >= 0 else label,
                            (x0, max(20, y0 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color, 2, cv2.LINE_AA)

            if frame_unknown_emb is None:
                unknown_anchor, unknown_streak, unknown_buffer = None, 0, []
            else:
                if unknown_anchor is None:
                    unknown_anchor = frame_unknown_emb
                    unknown_streak = 1
                    unknown_buffer = [frame_unknown_emb]
                else:
                    sim = cosine_similarity(frame_unknown_emb, unknown_anchor)
                    if sim >= 0.75:
                        unknown_streak += 1
                        unknown_anchor = l2_normalize(
                            0.7 * unknown_anchor + 0.3 * frame_unknown_emb)
                        unknown_buffer.append(frame_unknown_emb)
                    else:
                        unknown_anchor = frame_unknown_emb
                        unknown_streak = 1
                        unknown_buffer = [frame_unknown_emb]

                now = time.time()
                can_prompt = (unknown_streak >= min_frames_to_prompt
                              and now - last_prompt_time >= prompt_cooldown
                              and (prompt_thread is None or not prompt_thread.is_alive()))
                if can_prompt:
                    candidate = l2_normalize(
                        np.mean(np.vstack(unknown_buffer), axis=0).astype(np.float32))
                    prompt_thread = threading.Thread(
                        target=_prompt_enroll_name,
                        args=(prompt_results, candidate.copy(),
                              suggested_names, auto_enroll_from_suggestions),
                        daemon=True,
                    )
                    prompt_thread.start()
                    last_prompt_time = now
                    unknown_streak = 0
                    unknown_anchor = None
                    unknown_buffer.clear()

            cv2.imshow("Video Intelligence Pipeline", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — PIPELINE ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Video Intelligence Pipeline: Whisper → LLM names → Face Recognition")

    # Video input
    parser.add_argument("--video", required=True,
                        help="Path to video file (used for both transcription and face recognition).")
    parser.add_argument("--skip-transcription", action="store_true",
                        help="Skip Whisper transcription (use --intro-text directly).")
    parser.add_argument("--intro-text", default="",
                        help="Manually provide transcript text instead of transcribing.")
    parser.add_argument("--whisper-model", default="base",
                        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                        help="Whisper model size.")
    parser.add_argument("--save-transcript", action="store_true",
                        help="Save transcript to a .txt file alongside the video.")

    # LLM
    parser.add_argument("--llm-provider", choices=["ollama", "openai"], default="ollama")
    parser.add_argument("--llm-api-base", default="http://localhost:11434")
    parser.add_argument("--llm-api-key", default="")
    parser.add_argument("--llm-model", default="llama3.2:3b")

    # Face recognition
    parser.add_argument("--db", default="embeddings_db.npz")
    parser.add_argument("--mode", choices=["camera", "assist"], default="assist",
                        help="assist = auto-enroll with top LLM suggestion.")
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--second-best-margin", type=float, default=0.06)
    parser.add_argument("--face-margin", type=float, default=0.25)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--min-frames", type=int, default=5)
    parser.add_argument("--accelerator",
                        choices=["auto", "gpu", "npu", "directml", "cpu"],
                        default="auto")

    args = parser.parse_args()

    # ── Step 1: Transcribe ────────────────────────────────────────────────────
    transcript = args.intro_text
    if not args.skip_transcription and not transcript:
        whisper_model = load_whisper(args.whisper_model)
        transcript = transcribe_video(whisper_model, args.video)
        if args.save_transcript:
            txt_path = Path(args.video).with_suffix(".txt")
            txt_path.write_text(transcript, encoding="utf-8")
            print(f"[Whisper] Transcript saved to {txt_path}")
    else:
        print("[Whisper] Skipping transcription.", flush=True)

    # ── Step 2: Extract names ─────────────────────────────────────────────────
    suggested_names: list[str] = []
    if transcript:
        print(f"\n[LLM] Extracting names from transcript...", flush=True)
        suggested_names = extract_names_with_llm(
            text=transcript,
            api_base=args.llm_api_base,
            api_key=args.llm_api_key,
            model=args.llm_model,
            provider=args.llm_provider,
        )
        if suggested_names:
            print(f"[LLM] Name suggestions: {', '.join(suggested_names)}", flush=True)
        else:
            print("[LLM] No names found in transcript.", flush=True)
    else:
        print("[LLM] No transcript available, skipping name extraction.", flush=True)

    # ── Step 3: Load face model ───────────────────────────────────────────────
    print("\n[Face] Loading CavaFace model...", flush=True)
    face_model = load_local_model()
    model_device: object = torch.device("cpu")
    backend = "cpu"

    if args.accelerator == "npu":
        try:
            onnx_cache = Path(os.getenv("LOCALAPPDATA", ".")) / "cavaface_qnn.onnx"
            face_model = build_qnn_session(face_model, onnx_cache)
            backend = "npu-qnn"
            model_device = "onnxruntime"
        except Exception as e:
            print(f"[Face] NPU unavailable ({e}), falling back to CPU.")
    elif args.accelerator == "directml":
        try:
            onnx_cache = Path(os.getenv("LOCALAPPDATA", ".")) / "cavaface_dml.onnx"
            face_model = build_dml_session(face_model, onnx_cache)
            backend = "directml"
            model_device = "onnxruntime"
        except Exception as e:
            print(f"[Face] DirectML unavailable ({e}), falling back to CPU.")
    else:
        model_device, backend = resolve_torch_device(args.accelerator)
        face_model = face_model.to(model_device)

    print(f"[Face] Backend: {backend}", flush=True)

    # ── Step 4: Run face recognition with transcript-derived name suggestions ─
    print(f"\n[Pipeline] Starting face recognition on: {args.video}", flush=True)
    print(f"[Pipeline] Mode: {args.mode} | Suggestions: {suggested_names or 'none'}", flush=True)

    camera_mode(
        db_path=Path(args.db),
        face_model=face_model,
        model_device=model_device,
        threshold=args.threshold,
        second_best_margin=args.second_best_margin,
        face_margin=args.face_margin,
        camera_index=args.camera_index,
        video_path=args.video,
        min_frames_to_prompt=args.min_frames,
        suggested_names=suggested_names,
        auto_enroll_from_suggestions=(args.mode == "assist"),
    )


if __name__ == "__main__":
    main()