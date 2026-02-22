"""
Parallel Video Intelligence Pipeline (multiprocessing)
=======================================================
- Face recognition starts FIRST (DirectML/NPU/CPU)
- Whisper starts AFTER first video frame is shown (real-life simulation)
- LLM extracts names from transcript chunks and streams them to face process
- Names feed into face recognition as enroll suggestions live

Usage:
    python multiprocess_pipeline.py --video videoplayback.mp4
    python multiprocess_pipeline.py --video videoplayback.mp4 --accelerator directml
    python multiprocess_pipeline.py --video videoplayback.mp4 --mode camera
"""

import argparse
import multiprocessing as mp
import os
import sys
import time
import re
import json
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
#  PROCESS 1 — WHISPER + LLM
# ══════════════════════════════════════════════════════════════════════════════

def whisper_process(video_path: str, model_size: str,
                    llm_provider: str, llm_api_base: str,
                    llm_api_key: str, llm_model: str,
                    name_queue: mp.Queue,
                    transcript_queue: mp.Queue,
                    save_transcript: bool):
    try:
        from faster_whisper import WhisperModel

        print("[Whisper] Loading model...", flush=True)
        model = WhisperModel(model_size, device="cpu", compute_type="int8")

        print("[Whisper] Starting transcription...", flush=True)
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
        buffer = []
        FLUSH_EVERY = 5  # run LLM every N segments for low-latency name streaming

        for seg in segments:
            text = seg.text.strip()
            print(f"[Whisper] [{seg.start:.1f}s→{seg.end:.1f}s] {text}", flush=True)
            full_text.append(text)
            buffer.append(text)

            if len(buffer) >= FLUSH_EVERY:
                names = _extract_names(" ".join(buffer), llm_provider,
                                       llm_api_base, llm_api_key, llm_model)
                for n in names:
                    print(f"[Whisper→Face] Sending name: {n}", flush=True)
                    name_queue.put(n)
                buffer.clear()

        # Final flush
        if buffer:
            names = _extract_names(" ".join(buffer), llm_provider,
                                   llm_api_base, llm_api_key, llm_model)
            for n in names:
                print(f"[Whisper→Face] Sending name: {n}", flush=True)
                name_queue.put(n)

        transcript = " ".join(full_text)
        transcript_queue.put(transcript)
        name_queue.put(None)  # sentinel — tells face process whisper is done

        if save_transcript:
            txt_path = Path(video_path).with_suffix(".txt")
            txt_path.write_text(transcript, encoding="utf-8")
            print(f"[Whisper] Transcript saved: {txt_path}", flush=True)

        print("[Whisper] Done.", flush=True)

    except Exception as e:
        import traceback
        print(f"[Whisper] ERROR: {e}", flush=True)
        traceback.print_exc()
        name_queue.put(None)
        transcript_queue.put("")


def _extract_names(text: str, provider: str, api_base: str,
                   api_key: str, model: str) -> list:
    prompt = (
        "Extract ONLY person names that are self-introduced in this text. "
        "Return comma-separated names or empty string. No explanation.\n\n"
        f"Text: {text}\n\nNames:"
    )
    try:
        if provider == "ollama":
            url = f"{api_base.rstrip('/')}/api/generate"
            payload = json.dumps({
                "model": model, "prompt": prompt, "stream": False
            }).encode()
            req = urllib.request.Request(
                url, data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=15) as r:
                raw = json.loads(r.read()).get("response", "").strip()
        else:
            url = f"{api_base.rstrip('/')}/v1/chat/completions"
            payload = json.dumps({
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 60,
            }).encode()
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            req = urllib.request.Request(url, data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as r:
                raw = json.loads(r.read())["choices"][0]["message"]["content"].strip()

        return [n.strip().title() for n in raw.split(",")
                if n.strip() and 2 <= len(n.strip()) <= 50
                and not any(c.isdigit() for c in n)]
    except Exception as e:
        print(f"[LLM] Failed ({e}), using regex.", flush=True)
        return _regex_names(text)


def _regex_names(text: str) -> list:
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


# ══════════════════════════════════════════════════════════════════════════════
#  PROCESS 2 — FACE RECOGNITION
# ══════════════════════════════════════════════════════════════════════════════

def face_process(video_path: str, db_path: str,
                 threshold: float, margin: float, face_margin: float,
                 min_frames: int, auto_enroll: bool,
                 accelerator: str,
                 name_queue: mp.Queue,
                 video_started: mp.Event):          # ← signals main when video opens
    try:
        import cv2
        import torch
        from qai_hub_models.models.cavaface import Model as CavaFaceModel

        # ── Build face model ──────────────────────────────────────────────────
        face_model, model_device, backend = _build_face_model(accelerator, CavaFaceModel)
        print(f"[Face] Backend: {backend}", flush=True)

        db = Path(db_path)
        names, embeddings = _load_db(db)
        suggested = []
        unknown_anchor = None
        unknown_streak = 0
        unknown_buffer = []
        last_prompt = 0.0
        whisper_done = False

        detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if detector.empty():
            raise RuntimeError("Failed to load Haar cascade detector.")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        # ── Read and show FIRST frame, then signal Whisper to start ──────────
        ok, frame = cap.read()
        if ok:
            cv2.imshow("Face Recognition", frame)
            cv2.waitKey(1)
            print("[Face] First frame displayed. Signaling Whisper to start.", flush=True)
            video_started.set()
        else:
            print("[Face] Could not read first frame.", flush=True)
            video_started.set()  # signal anyway so main doesn't hang

        print("[Face] Processing video...", flush=True)

        # ── Main video loop ───────────────────────────────────────────────────
        while True:
            # Drain incoming names from Whisper
            while True:
                try:
                    item = name_queue.get_nowait()
                    if item is None:
                        whisper_done = True
                        print("[Face] Whisper transcription complete.", flush=True)
                    elif item not in suggested:
                        suggested.append(item)
                        print(f"[Face] New name suggestion: {item}", flush=True)
                except Exception:
                    break

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
                x0 = int(max(0, cx - side / 2))
                y0 = int(max(0, cy - side / 2))
                x1 = int(min(w_img, cx + side / 2))
                y1 = int(min(h_img, cy + side / 2))
                crop = frame[y0:y1, x0:x1]
                if crop.size == 0:
                    continue

                inp = _face_to_input(crop)
                emb = _run_embedding(face_model, inp, model_device)
                label, score = _best_match(names, embeddings, emb, threshold, margin)
                is_unknown = label == "Unknown"
                color = (0, 0, 255) if is_unknown else (0, 255, 0)
                if is_unknown and frame_unknown_emb is None:
                    frame_unknown_emb = emb

                cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
                cv2.putText(
                    frame,
                    f"{label} ({score:.2f})" if score >= 0 else label,
                    (x0, max(20, y0 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
                )

            # ── Unknown face streak tracking ──────────────────────────────────
            if frame_unknown_emb is None:
                unknown_anchor, unknown_streak, unknown_buffer = None, 0, []
            else:
                if unknown_anchor is None:
                    unknown_anchor = frame_unknown_emb
                    unknown_streak = 1
                    unknown_buffer = [frame_unknown_emb]
                elif _cosine(frame_unknown_emb, unknown_anchor) >= 0.75:
                    unknown_streak += 1
                    unknown_anchor = _l2(0.7 * unknown_anchor + 0.3 * frame_unknown_emb)
                    unknown_buffer.append(frame_unknown_emb)
                else:
                    unknown_anchor = frame_unknown_emb
                    unknown_streak = 1
                    unknown_buffer = [frame_unknown_emb]

                now = time.time()
                if unknown_streak >= min_frames and now - last_prompt >= 3.0:
                    candidate = _l2(np.mean(np.vstack(unknown_buffer), axis=0))
                    entered = _prompt_or_auto(candidate, suggested, auto_enroll)
                    if entered:
                        _enroll(db, entered, candidate)
                        names, embeddings = _load_db(db)
                    last_prompt = now
                    unknown_streak, unknown_anchor, unknown_buffer = 0, None, []

            # ── HUD overlay ───────────────────────────────────────────────────
            w_status = "✓ Done" if whisper_done else "⟳ Transcribing..."
            cv2.putText(frame, f"Whisper: {w_status}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Backend: {backend}",
                        (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame,
                        f"Suggestions: {', '.join(suggested) if suggested else 'waiting...'}",
                        (10, 71), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Known faces: {len(names)}",
                        (10, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as e:
        import traceback
        print(f"[Face] ERROR: {e}", flush=True)
        traceback.print_exc()
        video_started.set()  # unblock main even on error
    finally:
        try:
            cap.release()
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("[Face] Done.", flush=True)


# ── Face model helpers ─────────────────────────────────────────────────────────

def _build_face_model(accelerator: str, CavaFaceModel):
    import torch
    import onnxruntime as ort

    base_model = CavaFaceModel.from_pretrained().eval()
    onnx_path = Path(os.getenv("LOCALAPPDATA", ".")) / f"cavaface_{accelerator}.onnx"

    if not onnx_path.exists():
        print(f"[Face] Exporting ONNX model to {onnx_path}...", flush=True)
        dummy = torch.randn(1, 3, 112, 112)
        traced = torch.jit.trace(base_model.to("cpu"), dummy)
        torch.onnx.export(
            traced, dummy, str(onnx_path),
            input_names=["image"], output_names=["embedding"],
            opset_version=17, do_constant_folding=True,
        )
        print("[Face] ONNX export done.", flush=True)

    providers_map = {
        "directml": ["DmlExecutionProvider",  "CPUExecutionProvider"],
        "npu":      ["QNNExecutionProvider",   "CPUExecutionProvider"],
        "cpu":      ["CPUExecutionProvider"],
    }
    available = ort.get_available_providers()
    priority = (["npu", "directml", "cpu"] if accelerator == "auto"
                else [accelerator])

    for acc in priority:
        provider_list = providers_map.get(acc, ["CPUExecutionProvider"])
        primary = provider_list[0]
        if primary in available or primary == "CPUExecutionProvider":
            try:
                sess = ort.InferenceSession(str(onnx_path), providers=provider_list)
                actual = sess.get_providers()[0]
                print(f"[Face] ONNX session using: {actual}", flush=True)
                return sess, "onnxruntime", acc
            except Exception as e:
                print(f"[Face] {acc} failed: {e}", flush=True)

    # Pure CPU torch fallback
    device = torch.device("cpu")
    model = CavaFaceModel.from_pretrained().eval().to(device)
    return model, device, "cpu-torch"


def _face_to_input(face_bgr: np.ndarray) -> np.ndarray:
    import cv2
    resized = cv2.resize(face_bgr, (112, 112), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    arr = (rgb.astype(np.float32) / 255.0 - 0.5) / 0.5
    return np.ascontiguousarray(np.transpose(arr, (2, 0, 1))[np.newaxis])


def _run_embedding(model, inp: np.ndarray, device) -> np.ndarray:
    if hasattr(model, "run") and hasattr(model, "get_inputs"):
        name = model.get_inputs()[0].name
        out = model.run(None, {name: inp.astype(np.float32)})[0]
        return _l2(np.asarray(out).reshape(-1).astype(np.float32))
    import torch
    with torch.no_grad():
        out = model(torch.from_numpy(inp).float().to(device))
        if isinstance(out, (tuple, list)):
            out = out[0]
    return _l2(out.detach().cpu().numpy().reshape(-1))


def _l2(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) /
                 ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)))


def _load_db(db_path: Path):
    if not db_path.exists():
        return [], np.empty((0, 512), dtype=np.float32)
    data = np.load(db_path, allow_pickle=False)
    return data["names"].tolist(), data["embeddings"].astype(np.float32)


def _save_db(db_path: Path, names: list, embeddings: np.ndarray):
    np.savez(db_path,
             names=np.asarray(names, dtype=np.str_),
             embeddings=embeddings.astype(np.float32))


def _enroll(db_path: Path, name: str, embedding: np.ndarray):
    names, embeddings = _load_db(db_path)
    grouped = defaultdict(list)
    for n, e in zip(names, embeddings):
        grouped[n].append(e)
    grouped[name].append(embedding)
    merged_names, merged_embs = [], []
    for person, embs in grouped.items():
        centroid = _l2(np.mean(np.vstack(embs), axis=0).astype(np.float32))
        merged_names.append(person)
        merged_embs.append(centroid)
    _save_db(db_path, merged_names, np.vstack(merged_embs))
    print(f"[Face] Enrolled '{name}'. Total identities: {len(merged_names)}", flush=True)


def _best_match(names: list, embeddings: np.ndarray,
                query: np.ndarray, threshold: float, margin: float):
    if not names:
        return "Unknown", -1.0
    scores = [_cosine(query, e) for e in embeddings]
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    sorted_s = sorted(scores, reverse=True)
    gap = sorted_s[0] - (sorted_s[1] if len(sorted_s) > 1 else 0.0)
    if best_score < threshold or gap < margin:
        return "Unknown", best_score
    return names[best_idx], best_score


def _prompt_or_auto(embedding: np.ndarray, suggestions: list,
                    auto_enroll: bool) -> str:
    print(f"\n[Face] Unknown face detected.", flush=True)
    print(f"[Face] Current suggestions: {suggestions or 'none'}", flush=True)
    if auto_enroll and suggestions:
        print(f"[Face] Auto-enrolling as: {suggestions[0]}", flush=True)
        return suggestions[0]
    if not sys.stdin.isatty():
        return suggestions[0] if suggestions else ""
    try:
        return input("Enroll as (blank to skip): ").strip()
    except EOFError:
        return suggestions[0] if suggestions else ""


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Parallel Whisper + Face Recognition (multiprocessing)")
    parser.add_argument("--video", required=True,
                        help="Path to video file.")
    parser.add_argument("--whisper-model", default="base",
                        choices=["tiny", "base", "small", "medium"],
                        help="Whisper model size (default: base).")
    parser.add_argument("--save-transcript", action="store_true",
                        help="Save transcript to .txt file.")
    parser.add_argument("--llm-provider", choices=["ollama", "openai"],
                        default="ollama")
    parser.add_argument("--llm-api-base", default="http://localhost:11434")
    parser.add_argument("--llm-api-key",  default="")
    parser.add_argument("--llm-model",    default="llama3.2:3b")
    parser.add_argument("--db",           default="embeddings_db.npz")
    parser.add_argument("--mode",         choices=["camera", "assist"],
                        default="assist",
                        help="assist = auto-enroll with top LLM suggestion.")
    parser.add_argument("--threshold",    type=float, default=0.50)
    parser.add_argument("--margin",       type=float, default=0.06)
    parser.add_argument("--face-margin",  type=float, default=0.25)
    parser.add_argument("--min-frames",   type=int,   default=5)
    parser.add_argument("--accelerator",
                        choices=["auto", "directml", "npu", "cpu"],
                        default="auto",
                        help="auto tries: NPU → DirectML (Adreno GPU) → CPU.")
    args = parser.parse_args()

    # IPC queues
    name_queue       = mp.Queue()   # Whisper → Face: name strings + None sentinel
    transcript_queue = mp.Queue()   # Whisper → Main: final transcript string
    video_started    = mp.Event()   # Face → Main: signals first frame is shown

    print("=" * 55, flush=True)
    print("  Parallel Video Intelligence Pipeline", flush=True)
    print("=" * 55, flush=True)
    print(f"  Video    : {args.video}", flush=True)
    print(f"  Whisper  : {args.whisper_model} (CPU int8)", flush=True)
    print(f"  Face     : {args.accelerator}", flush=True)
    print(f"  LLM      : {args.llm_model} via {args.llm_provider}", flush=True)
    print(f"  Mode     : {args.mode}", flush=True)
    print("=" * 55, flush=True)

    # ── Launch face process FIRST ─────────────────────────────────────────────
    print("\n[Pipeline] Starting face recognition...", flush=True)
    p_face = mp.Process(
        target=face_process,
        args=(
            args.video,
            args.db,
            args.threshold,
            args.margin,
            args.face_margin,
            args.min_frames,
            args.mode == "assist",
            args.accelerator,
            name_queue,
            video_started,           # ← face sets this after first frame
        ),
        name="Face",
    )
    p_face.start()

    # ── Wait for first video frame to appear ─────────────────────────────────
    print("[Pipeline] Waiting for video to open...", flush=True)
    got_signal = video_started.wait(timeout=15)
    if not got_signal:
        print("[Pipeline] WARNING: video_started timed out, launching Whisper anyway.", flush=True)

    # ── Launch Whisper AFTER video has started ────────────────────────────────
    print("[Pipeline] Video started! Launching Whisper now...", flush=True)
    p_whisper = mp.Process(
        target=whisper_process,
        args=(
            args.video,
            args.whisper_model,
            args.llm_provider,
            args.llm_api_base,
            args.llm_api_key,
            args.llm_model,
            name_queue,
            transcript_queue,
            args.save_transcript,
        ),
        name="Whisper",
    )
    p_whisper.start()

    # ── Wait for both to finish ───────────────────────────────────────────────
    p_face.join()
    p_whisper.join()

    # ── Print final transcript ────────────────────────────────────────────────
    print("\n[Pipeline] Complete!", flush=True)
    if not transcript_queue.empty():
        transcript = transcript_queue.get()
        if transcript:
            print("\n─── FINAL TRANSCRIPT ─────────────────────────────")
            print(transcript)
            print("──────────────────────────────────────────────────")
        else:
            print("[Pipeline] No transcript produced.")


if __name__ == "__main__":
    mp.freeze_support()   # required on Windows
    main()