"""
Parallel Video Intelligence Pipeline
======================================
Whisper transcription and face recognition run simultaneously in separate threads.
LLM name extraction feeds into face recognition as names are discovered mid-stream.
"""

import argparse
import os
import queue
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

try:
    from faster_whisper import WhisperModel
except ImportError:
    raise SystemExit("pip install faster-whisper")

try:
    from qai_hub_models.models.cavaface import Model as CavaFaceModel
except ImportError:
    raise SystemExit("pip install qai-hub-models[cavaface]")

import re, urllib.request, json as _json


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED STATE — thread-safe name queue
# ══════════════════════════════════════════════════════════════════════════════

class SharedState:
    """Thread-safe shared state between Whisper and Face threads."""
    def __init__(self):
        self.name_queue: queue.Queue[str] = queue.Queue()   # new names from LLM → face thread
        self.suggested_names: list[str] = []                # accumulated suggestions
        self.transcript_done = threading.Event()
        self.transcript = ""
        self.lock = threading.Lock()

    def add_names(self, names: list[str]):
        with self.lock:
            for name in names:
                if name not in self.suggested_names:
                    self.suggested_names.append(name)
                    self.name_queue.put(name)
                    print(f"[LLM→Face] New name available: {name}", flush=True)

    def get_all_names(self) -> list[str]:
        with self.lock:
            return list(self.suggested_names)


# ══════════════════════════════════════════════════════════════════════════════
#  THREAD 1 — WHISPER + LLM
# ══════════════════════════════════════════════════════════════════════════════

def whisper_llm_thread(video_path: str, whisper_model_size: str,
                       llm_provider: str, llm_api_base: str,
                       llm_api_key: str, llm_model: str,
                       state: SharedState, save_transcript: bool):
    try:
        print("[Whisper] Starting transcription...", flush=True)
        model = WhisperModel(whisper_model_size, device="cpu", compute_type="int8")

        segments, info = model.transcribe(
            video_path,
            beam_size=5,
            language="en",
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            no_repeat_ngram_size=4,
        )
        print(f"[Whisper] Language: {info.language} ({info.language_probability:.2f})", flush=True)

        full_text = []
        segment_buffer = []
        FLUSH_EVERY = 5  # extract names every N segments for low latency

        for seg in segments:
            text = seg.text.strip()
            print(f"[Whisper]  [{seg.start:.1f}s→{seg.end:.1f}s] {text}", flush=True)
            full_text.append(text)
            segment_buffer.append(text)

            # ── Stream names to face thread every N segments ──────────────────
            if len(segment_buffer) >= FLUSH_EVERY:
                chunk = " ".join(segment_buffer)
                names = extract_names_with_llm(chunk, llm_api_base, llm_api_key,
                                               llm_model, llm_provider)
                if names:
                    state.add_names(names)
                segment_buffer.clear()

        # Final flush for remaining segments
        if segment_buffer:
            chunk = " ".join(segment_buffer)
            names = extract_names_with_llm(chunk, llm_api_base, llm_api_key,
                                           llm_model, llm_provider)
            if names:
                state.add_names(names)

        state.transcript = " ".join(full_text)

        if save_transcript:
            txt_path = Path(video_path).with_suffix(".txt")
            txt_path.write_text(state.transcript, encoding="utf-8")
            print(f"[Whisper] Transcript saved: {txt_path}", flush=True)

        print(f"[Whisper] Done. {len(state.transcript)} chars.", flush=True)

    except Exception as e:
        print(f"[Whisper] ERROR: {e}", flush=True)
        import traceback; traceback.print_exc()
    finally:
        state.transcript_done.set()


# ══════════════════════════════════════════════════════════════════════════════
#  LLM NAME EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def regex_fallback_extract(text: str) -> list[str]:
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


def extract_names_with_llm(text, api_base, api_key, model, provider) -> list[str]:
    prompt = (
        "Extract ONLY person names that are introduced or self-identified in this text. "
        "Return a comma-separated list of names, or empty string if none. No explanation.\n\n"
        f"Text: {text}\n\nNames:"
    )
    try:
        if provider == "ollama":
            url = f"{api_base.rstrip('/')}/api/generate"
            payload = _json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
            req = urllib.request.Request(url, data=payload,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = _json.loads(resp.read()).get("response", "").strip()
        else:
            url = f"{api_base.rstrip('/')}/v1/chat/completions"
            payload = _json.dumps({
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 60,
            }).encode()
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            req = urllib.request.Request(url, data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = _json.loads(resp.read())["choices"][0]["message"]["content"].strip()

        names = [n.strip().title() for n in raw.split(",") if n.strip()]
        return [n for n in names if 2 <= len(n) <= 50 and not any(c.isdigit() for c in n)]
    except Exception as e:
        print(f"[LLM] Failed ({e}), trying regex.", flush=True)
        return regex_fallback_extract(text)


# ══════════════════════════════════════════════════════════════════════════════
#  THREAD 2 — FACE RECOGNITION
# ══════════════════════════════════════════════════════════════════════════════

def l2_normalize(vec):
    return vec / (np.linalg.norm(vec) + 1e-12)

def cosine_similarity(a, b):
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)))

def load_db(db_path):
    if not db_path.exists():
        return [], np.empty((0, 512), dtype=np.float32)
    data = np.load(db_path, allow_pickle=False)
    return data["names"].tolist(), data["embeddings"].astype(np.float32)

def save_db(db_path, names, embeddings):
    np.savez(db_path, names=np.asarray(names, dtype=np.str_),
             embeddings=embeddings.astype(np.float32))

def enroll(db_path, name, embedding):
    names, embeddings = load_db(db_path)
    grouped = defaultdict(list)
    for n, e in zip(names, embeddings):
        grouped[n].append(e)
    grouped[name].append(embedding)
    merged_names, merged_embs = [], []
    for person, embs in grouped.items():
        merged_names.append(person)
        merged_embs.append(l2_normalize(np.mean(np.vstack(embs), axis=0)))
    save_db(db_path, merged_names, np.vstack(merged_embs))
    print(f"[Face] Enrolled '{name}'. Total: {len(merged_names)}", flush=True)

def best_match(names, embeddings, query, threshold, margin):
    if not names:
        return "Unknown", -1.0
    scores = [cosine_similarity(query, e) for e in embeddings]
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    gap = sorted(scores, reverse=True)[0] - (sorted(scores, reverse=True)[1] if len(scores) > 1 else 0)
    if best_score < threshold or gap < margin:
        return "Unknown", best_score
    return names[best_idx], best_score

def face_bgr_to_input(face_bgr):
    import cv2
    resized = cv2.resize(face_bgr, (112, 112))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    arr = (rgb.astype(np.float32) / 255.0 - 0.5) / 0.5
    return np.ascontiguousarray(np.transpose(arr, (2, 0, 1))[np.newaxis])

def run_embedding(model, inp, device):
    if hasattr(model, "run"):
        name = model.get_inputs()[0].name
        out = model.run(None, {name: inp.astype(np.float32)})[0]
        return l2_normalize(np.asarray(out).reshape(-1).astype(np.float32))
    with torch.no_grad():
        out = model(torch.from_numpy(inp).float().to(device))
        if isinstance(out, (tuple, list)): out = out[0]
    return l2_normalize(out.detach().cpu().numpy().reshape(-1))


def face_recognition_thread(video_path: str, db_path: Path, face_model,
                             model_device, threshold: float, margin: float,
                             face_margin: float, min_frames: int,
                             auto_enroll: bool, state: SharedState):
    try:
        import cv2
    except ImportError:
        raise RuntimeError("pip install opencv-python")

    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    names, embeddings = load_db(db_path)
    enroll_queue: queue.Queue = queue.Queue()
    enroll_results: queue.Queue = queue.Queue()
    unknown_anchor = None
    unknown_streak = 0
    unknown_buffer = []
    last_prompt = 0.0

    print("[Face] Starting face recognition...", flush=True)

    def prompt_worker(emb, suggestions):
        print(f"\n[Face] Unknown face — suggestions: {suggestions or 'none'}", flush=True)
        if auto_enroll and suggestions:
            enroll_results.put((suggestions[0], emb))
            return
        if not sys.stdin.isatty():
            enroll_results.put((suggestions[0] if suggestions else "", emb))
            return
        try:
            entered = input("Enroll as (blank to skip): ").strip()
        except EOFError:
            entered = suggestions[0] if suggestions else ""
        enroll_results.put((entered, emb))

    try:
        while True:
            # ── Drain new names from Whisper thread ───────────────────────────
            while not state.name_queue.empty():
                new_name = state.name_queue.get_nowait()
                print(f"[Face] Received new name suggestion: {new_name}", flush=True)

            # ── Drain enroll results ──────────────────────────────────────────
            while not enroll_results.empty():
                entered, emb = enroll_results.get_nowait()
                if entered:
                    enroll(db_path, entered, emb)
                    names, embeddings = load_db(db_path)

            ok, frame = cap.read()
            if not ok:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))
            frame_unknown_emb = None

            for (x, y, w, h) in faces:
                h_img, w_img = frame.shape[:2]
                cx, cy = x + w / 2, y + h / 2
                side = max(w, h) * (1 + face_margin)
                x0 = int(max(0, cx - side / 2)); y0 = int(max(0, cy - side / 2))
                x1 = int(min(w_img, cx + side / 2)); y1 = int(min(h_img, cy + side / 2))
                crop = frame[y0:y1, x0:x1]
                if crop.size == 0:
                    continue

                inp = face_bgr_to_input(crop)
                emb = run_embedding(face_model, inp, model_device)
                label, score = best_match(names, embeddings, emb, threshold, margin)
                is_unknown = label == "Unknown"
                color = (0, 0, 255) if is_unknown else (0, 255, 0)
                if is_unknown and frame_unknown_emb is None:
                    frame_unknown_emb = emb

                cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
                cv2.putText(frame, f"{label} ({score:.2f})",
                            (x0, max(20, y0 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color, 2, cv2.LINE_AA)

            # ── Unknown face streak tracking ──────────────────────────────────
            if frame_unknown_emb is None:
                unknown_anchor, unknown_streak, unknown_buffer = None, 0, []
            else:
                if unknown_anchor is None:
                    unknown_anchor, unknown_streak = frame_unknown_emb, 1
                    unknown_buffer = [frame_unknown_emb]
                elif cosine_similarity(frame_unknown_emb, unknown_anchor) >= 0.75:
                    unknown_streak += 1
                    unknown_anchor = l2_normalize(0.7 * unknown_anchor + 0.3 * frame_unknown_emb)
                    unknown_buffer.append(frame_unknown_emb)
                else:
                    unknown_anchor, unknown_streak = frame_unknown_emb, 1
                    unknown_buffer = [frame_unknown_emb]

                now = time.time()
                if (unknown_streak >= min_frames and now - last_prompt >= 3.0):
                    candidate = l2_normalize(np.mean(np.vstack(unknown_buffer), axis=0))
                    current_suggestions = state.get_all_names()  # ← live from Whisper
                    threading.Thread(target=prompt_worker,
                                     args=(candidate.copy(), current_suggestions),
                                     daemon=True).start()
                    last_prompt = now
                    unknown_streak, unknown_anchor, unknown_buffer = 0, None, []

            # ── Show transcript status overlay ────────────────────────────────
            status = "Transcribing..." if not state.transcript_done.is_set() else "Transcript ready"
            cv2.putText(frame, f"[Whisper] {status}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1, cv2.LINE_AA)
            known_count = len(load_db(db_path)[0])
            cv2.putText(frame, f"Known faces: {known_count}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1, cv2.LINE_AA)

            cv2.imshow("Parallel Pipeline", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[Face] Done.", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN — launch both threads simultaneously
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Parallel Whisper + Face Recognition")
    parser.add_argument("--video", required=True)
    parser.add_argument("--whisper-model", default="base",
                        choices=["tiny", "base", "small", "medium"])
    parser.add_argument("--save-transcript", action="store_true")
    parser.add_argument("--llm-provider", choices=["ollama", "openai"], default="ollama")
    parser.add_argument("--llm-api-base", default="http://localhost:11434")
    parser.add_argument("--llm-api-key", default="")
    parser.add_argument("--llm-model", default="llama3.2:3b")
    parser.add_argument("--db", default="embeddings_db.npz")
    parser.add_argument("--mode", choices=["camera", "assist"], default="assist")
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--margin", type=float, default=0.06)
    parser.add_argument("--face-margin", type=float, default=0.25)
    parser.add_argument("--min-frames", type=int, default=5)
    parser.add_argument("--accelerator",
                        choices=["auto", "cpu", "directml", "npu"], default="auto")
    args = parser.parse_args()

    # ── Load face model ───────────────────────────────────────────────────────
    print("[Face] Loading CavaFace...", flush=True)
    face_model = CavaFaceModel.from_pretrained().eval()
    model_device: object = torch.device("cpu")
    backend = "cpu"

    if args.accelerator == "directml":
        try:
            import onnxruntime as ort
            from pathlib import Path as _P
            onnx_path = _P(os.getenv("LOCALAPPDATA", ".")) / "cavaface_dml.onnx"
            if not onnx_path.exists():
                dummy = torch.randn(1, 3, 112, 112)
                torch.onnx.export(torch.jit.trace(face_model, dummy), dummy,
                                  str(onnx_path), input_names=["image"],
                                  output_names=["embedding"], opset_version=17)
            face_model = ort.InferenceSession(str(onnx_path),
                                              providers=["DmlExecutionProvider",
                                                         "CPUExecutionProvider"])
            backend = "directml"
            model_device = "onnxruntime"
        except Exception as e:
            print(f"[Face] DirectML failed ({e}), using CPU.")
    else:
        model_device, backend = (torch.device("cpu"), "cpu")
        face_model = face_model.to(model_device)

    print(f"[Face] Backend: {backend}", flush=True)

    # ── Shared state ──────────────────────────────────────────────────────────
    state = SharedState()

    # ── Launch threads ────────────────────────────────────────────────────────
    t_whisper = threading.Thread(
        target=whisper_llm_thread,
        args=(args.video, args.whisper_model, args.llm_provider,
              args.llm_api_base, args.llm_api_key, args.llm_model,
              state, args.save_transcript),
        daemon=True,
        name="Whisper-LLM",
    )

    t_face = threading.Thread(
        target=face_recognition_thread,
        args=(args.video, Path(args.db), face_model, model_device,
              args.threshold, args.margin, args.face_margin,
              args.min_frames, args.mode == "assist", state),
        name="FaceRecognition",
    )

    print("\n[Pipeline] Launching Whisper + Face Recognition in parallel...", flush=True)
    t_whisper.start()
    t_face.start()

    t_face.join()       # wait for video to finish
    t_whisper.join()    # wait for transcript to finish

    print("\n[Pipeline] Complete!", flush=True)
    if state.transcript:
        print("\n─── FINAL TRANSCRIPT ─────────────────────────────")
        print(state.transcript)
        print("──────────────────────────────────────────────────")
    if state.suggested_names:
        print(f"Names found: {', '.join(state.suggested_names)}")


if __name__ == "__main__":
    main()