"""
Parallel Video Intelligence Pipeline (multiprocessing)
=======================================================
3 processes launch in order, then run in parallel:

  t=0.0s  Audio   starts  → plays video sound via ffplay
  t=0.0s  Face    starts  → opens video, loads CavaFace model
  t=~0.1s Face    signals → first frame shown on screen
  t=~0.1s Whisper starts  → begins transcribing audio
  t=~5s   Whisper sends   → LLM-extracted names stream to Face in real time

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
#  PROCESS 1 — AUDIO PLAYBACK
# ══════════════════════════════════════════════════════════════════════════════

def audio_process(video_path: str, audio_started: mp.Event):
    import subprocess

    # Search for ffplay in common locations
    candidates = [
        "./ffplay.exe",
        "./ffmpeg/bin/ffplay.exe",
        r".\ffmpeg\bin\ffplay.exe",
        "ffplay.exe",
        "ffplay",
    ]
    ffplay_path = None
    for c in candidates:
        try:
            r = subprocess.run([c, "-version"],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if r.returncode == 0:
                ffplay_path = c
                break
        except Exception:
            continue

    if ffplay_path is None:
        print("[Audio] ffplay not found — skipping audio. "
              "Put ffplay.exe next to this script or add to PATH.", flush=True)
        audio_started.set()
        return

    print(f"[Audio] ffplay found: {ffplay_path}", flush=True)
    print(f"[Audio] Playing: {video_path}", flush=True)

    cmd = [
        ffplay_path,
        "-nodisp",          # no ffplay video window (we use cv2)
        "-autoexit",        # quit when playback ends
        "-loglevel", "quiet",
        video_path,
    ]
    try:
        proc = subprocess.Popen(cmd)
        audio_started.set()          # unblock main → face can start
        proc.wait()
        print("[Audio] Playback finished.", flush=True)
    except Exception as e:
        print(f"[Audio] ERROR: {e}", flush=True)
        audio_started.set()


# ══════════════════════════════════════════════════════════════════════════════
#  PROCESS 2 — FACE RECOGNITION
# ══════════════════════════════════════════════════════════════════════════════

def face_process(video_path: str, db_path: str,
                 threshold: float, margin: float, face_margin: float,
                 min_frames: int, auto_enroll: bool,
                 accelerator: str,
                 name_queue: mp.Queue,
                 video_started: mp.Event):
    try:
        import cv2
        import torch
        from qai_hub_models.models.cavaface import Model as CavaFaceModel

        face_model, model_device, backend = _build_face_model(
            accelerator, CavaFaceModel)
        print(f"[Face] Backend: {backend}", flush=True)

        db = Path(db_path)
        names, embeddings = _load_db(db)
        suggested        = []
        unknown_anchor   = None
        unknown_streak   = 0
        unknown_buffer   = []
        last_prompt      = 0.0
        whisper_done     = False

        detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if detector.empty():
            raise RuntimeError("Failed to load Haar cascade detector.")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        # Read first frame → show it → THEN signal Whisper to start
        ok, frame = cap.read()
        if ok:
            cv2.imshow("Face Recognition", frame)
            cv2.waitKey(1)
            print("[Face] First frame shown — signaling Whisper.", flush=True)
        else:
            print("[Face] Warning: could not read first frame.", flush=True)
        video_started.set()

        print("[Face] Processing...", flush=True)

        while True:
            # ── Drain names sent by Whisper ───────────────────────────────────
            while True:
                try:
                    item = name_queue.get_nowait()
                    if item is None:
                        whisper_done = True
                        print("[Face] Whisper finished.", flush=True)
                    elif item not in suggested:
                        suggested.append(item)
                        print(f"[Face] New suggestion: {item}", flush=True)
                except Exception:
                    break

            ok, frame = cap.read()
            if not ok:
                break

            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))
            frame_unknown_emb = None

            for (x, y, w, h) in faces:
                h_img, w_img = frame.shape[:2]
                cx   = x + w / 2
                cy   = y + h / 2
                side = max(w, h) * (1 + face_margin)
                x0   = int(max(0,     cx - side / 2))
                y0   = int(max(0,     cy - side / 2))
                x1   = int(min(w_img, cx + side / 2))
                y1   = int(min(h_img, cy + side / 2))
                crop = frame[y0:y1, x0:x1]
                if crop.size == 0:
                    continue

                inp   = _face_to_input(crop)
                emb   = _run_embedding(face_model, inp, model_device)
                label, score = _best_match(names, embeddings, emb,
                                           threshold, margin)
                is_unknown = (label == "Unknown")
                color      = (0, 0, 255) if is_unknown else (0, 255, 0)

                if is_unknown and frame_unknown_emb is None:
                    frame_unknown_emb = emb

                cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
                cv2.putText(
                    frame,
                    f"{label} ({score:.2f})" if score >= 0 else label,
                    (x0, max(20, y0 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
                )

            # ── Unknown face streak ───────────────────────────────────────────
            if frame_unknown_emb is None:
                unknown_anchor = None
                unknown_streak = 0
                unknown_buffer = []
            else:
                if unknown_anchor is None:
                    unknown_anchor = frame_unknown_emb
                    unknown_streak = 1
                    unknown_buffer = [frame_unknown_emb]
                elif _cosine(frame_unknown_emb, unknown_anchor) >= 0.75:
                    unknown_streak += 1
                    unknown_anchor  = _l2(
                        0.7 * unknown_anchor + 0.3 * frame_unknown_emb)
                    unknown_buffer.append(frame_unknown_emb)
                else:
                    unknown_anchor = frame_unknown_emb
                    unknown_streak = 1
                    unknown_buffer = [frame_unknown_emb]

                now = time.time()
                if unknown_streak >= min_frames and now - last_prompt >= 3.0:
                    candidate = _l2(
                        np.mean(np.vstack(unknown_buffer), axis=0))
                    entered = _prompt_or_auto(candidate, suggested, auto_enroll)
                    if entered:
                        _enroll(db, entered, candidate)
                        names, embeddings = _load_db(db)
                    last_prompt    = now
                    unknown_streak = 0
                    unknown_anchor = None
                    unknown_buffer = []

            # ── HUD ───────────────────────────────────────────────────────────
            w_status = "Done ✓" if whisper_done else "Transcribing..."
            cv2.putText(frame, f"[Whisper] {w_status}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"[Face] backend: {backend}",
                        (10, 48), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(
                frame,
                "Suggestions: " + (", ".join(suggested) if suggested
                                   else "waiting..."),
                (10, 71), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Known faces: {len(names)}",
                        (10, 94), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as e:
        import traceback
        print(f"[Face] ERROR: {e}", flush=True)
        traceback.print_exc()
        video_started.set()
    finally:
        try:
            cap.release()
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("[Face] Done.", flush=True)


# ── helpers ────────────────────────────────────────────────────────────────────

def _build_face_model(accelerator: str, CavaFaceModel):
    import torch
    import onnxruntime as ort

    base  = CavaFaceModel.from_pretrained().eval()
    opath = Path(os.getenv("LOCALAPPDATA", ".")) / \
        f"cavaface_{accelerator}.onnx"

    if not opath.exists():
        print(f"[Face] Exporting ONNX → {opath} ...", flush=True)
        dummy  = torch.randn(1, 3, 112, 112)
        traced = torch.jit.trace(base.to("cpu"), dummy)
        torch.onnx.export(
            traced, dummy, str(opath),
            input_names=["image"], output_names=["embedding"],
            opset_version=17, do_constant_folding=True,
        )
        print("[Face] Export done.", flush=True)

    pmap      = {
        "directml": ["DmlExecutionProvider",  "CPUExecutionProvider"],
        "npu":      ["QNNExecutionProvider",   "CPUExecutionProvider"],
        "cpu":      ["CPUExecutionProvider"],
    }
    available = ort.get_available_providers()
    order     = (["npu", "directml", "cpu"]
                 if accelerator == "auto" else [accelerator])

    for acc in order:
        plist   = pmap.get(acc, ["CPUExecutionProvider"])
        primary = plist[0]
        if primary in available or primary == "CPUExecutionProvider":
            try:
                sess   = ort.InferenceSession(str(opath), providers=plist)
                actual = sess.get_providers()[0]
                print(f"[Face] ONNX provider: {actual}", flush=True)
                return sess, "onnxruntime", acc
            except Exception as e:
                print(f"[Face] {acc} failed: {e}", flush=True)

    import torch
    dev   = torch.device("cpu")
    model = CavaFaceModel.from_pretrained().eval().to(dev)
    return model, dev, "cpu-torch"


def _face_to_input(face_bgr: np.ndarray) -> np.ndarray:
    import cv2
    r   = cv2.resize(face_bgr, (112, 112), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(r, cv2.COLOR_BGR2RGB)
    arr = (rgb.astype(np.float32) / 255.0 - 0.5) / 0.5
    return np.ascontiguousarray(np.transpose(arr, (2, 0, 1))[np.newaxis])


def _run_embedding(model, inp: np.ndarray, device) -> np.ndarray:
    if hasattr(model, "run") and hasattr(model, "get_inputs"):
        name = model.get_inputs()[0].name
        out  = model.run(None, {name: inp.astype(np.float32)})[0]
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
    mn, me = [], []
    for person, embs in grouped.items():
        mn.append(person)
        me.append(_l2(np.mean(np.vstack(embs), axis=0).astype(np.float32)))
    _save_db(db_path, mn, np.vstack(me))
    print(f"[Face] Enrolled '{name}'. Total: {len(mn)}", flush=True)


def _best_match(names, embeddings, query, threshold, margin):
    if not names:
        return "Unknown", -1.0
    scores    = [_cosine(query, e) for e in embeddings]
    best_idx  = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    ss        = sorted(scores, reverse=True)
    gap       = ss[0] - (ss[1] if len(ss) > 1 else 0.0)
    if best_score < threshold or gap < margin:
        return "Unknown", best_score
    return names[best_idx], best_score


def _prompt_or_auto(embedding, suggestions, auto_enroll) -> str:
    print(f"\n[Face] Unknown face — suggestions: {suggestions or 'none'}",
          flush=True)
    if auto_enroll and suggestions:
        print(f"[Face] Auto-enrolling: {suggestions[0]}", flush=True)
        return suggestions[0]
    if not sys.stdin.isatty():
        return suggestions[0] if suggestions else ""
    try:
        return input("Enroll as (blank to skip): ").strip()
    except EOFError:
        return suggestions[0] if suggestions else ""


# ══════════════════════════════════════════════════════════════════════════════
#  PROCESS 3 — WHISPER + LLM
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

        print("[Whisper] Transcribing...", flush=True)
        segments, info = model.transcribe(
            video_path,
            beam_size=5,
            language="en",
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            no_repeat_ngram_size=4,
        )
        print(f"[Whisper] Language: {info.language} "
              f"({info.language_probability:.2f})", flush=True)

        full_text   = []
        buffer      = []
        FLUSH_EVERY = 5

        for seg in segments:
            text = seg.text.strip()
            print(f"[Whisper] [{seg.start:.1f}s→{seg.end:.1f}s] {text}",
                  flush=True)
            full_text.append(text)
            buffer.append(text)

            if len(buffer) >= FLUSH_EVERY:
                names = _extract_names(" ".join(buffer), llm_provider,
                                       llm_api_base, llm_api_key, llm_model)
                for n in names:
                    print(f"[Whisper→Face] {n}", flush=True)
                    name_queue.put(n)
                buffer.clear()

        if buffer:
            names = _extract_names(" ".join(buffer), llm_provider,
                                   llm_api_base, llm_api_key, llm_model)
            for n in names:
                print(f"[Whisper→Face] {n}", flush=True)
                name_queue.put(n)

        transcript = " ".join(full_text)
        transcript_queue.put(transcript)
        name_queue.put(None)          # sentinel

        if save_transcript:
            p = Path(video_path).with_suffix(".txt")
            p.write_text(transcript, encoding="utf-8")
            print(f"[Whisper] Saved: {p}", flush=True)

        print("[Whisper] Done.", flush=True)

    except Exception as e:
        import traceback
        print(f"[Whisper] ERROR: {e}", flush=True)
        traceback.print_exc()
        name_queue.put(None)
        transcript_queue.put("")


def _extract_names(text, provider, api_base, api_key, model) -> list:
    prompt = (
        "Extract ONLY person names that are self-introduced in this text. "
        "Return comma-separated names or empty string. No explanation.\n\n"
        f"Text: {text}\n\nNames:"
    )
    try:
        if provider == "ollama":
            url     = f"{api_base.rstrip('/')}/api/generate"
            payload = json.dumps({
                "model": model, "prompt": prompt, "stream": False
            }).encode()
            req = urllib.request.Request(
                url, data=payload,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=15) as r:
                raw = json.loads(r.read()).get("response", "").strip()
        else:
            url     = f"{api_base.rstrip('/')}/v1/chat/completions"
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
                raw = json.loads(r.read())["choices"][0]["message"][
                    "content"].strip()

        return [n.strip().title() for n in raw.split(",")
                if n.strip() and 2 <= len(n.strip()) <= 50
                and not any(c.isdigit() for c in n)]
    except Exception as e:
        print(f"[LLM] Failed ({e}), regex fallback.", flush=True)
        return _regex_names(text)


def _regex_names(text: str) -> list:
    patterns = [
        r"(?:I'm|I am|my name is|this is|call me|name's)"
        r"\s+([A-Z][a-z]+(?: [A-Z][a-z]+)?)",
        r"(?:Hi|Hello|Hey),?\s+(?:I'm|I am)"
        r"\s+([A-Z][a-z]+(?: [A-Z][a-z]+)?)",
    ]
    names = []
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            name = m.group(1).strip().title()
            if name not in names:
                names.append(name)
    return names


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Parallel Audio + Face + Whisper Pipeline")
    parser.add_argument("--video",           required=True)
    parser.add_argument("--whisper-model",   default="base",
                        choices=["tiny", "base", "small", "medium"])
    parser.add_argument("--save-transcript", action="store_true")
    parser.add_argument("--llm-provider",    choices=["ollama", "openai"],
                        default="ollama")
    parser.add_argument("--llm-api-base",    default="http://localhost:11434")
    parser.add_argument("--llm-api-key",     default="")
    parser.add_argument("--llm-model",       default="llama3.2:3b")
    parser.add_argument("--db",              default="embeddings_db.npz")
    parser.add_argument("--mode",            choices=["camera", "assist"],
                        default="assist")
    parser.add_argument("--threshold",       type=float, default=0.50)
    parser.add_argument("--margin",          type=float, default=0.06)
    parser.add_argument("--face-margin",     type=float, default=0.25)
    parser.add_argument("--min-frames",      type=int,   default=5)
    parser.add_argument("--accelerator",
                        choices=["auto", "directml", "npu", "cpu"],
                        default="auto",
                        help="auto tries: NPU → DirectML (Adreno) → CPU")
    args = parser.parse_args()

    # IPC
    name_queue       = mp.Queue()
    transcript_queue = mp.Queue()
    audio_started    = mp.Event()   # Audio  → Main
    video_started    = mp.Event()   # Face   → Main

    print("=" * 55, flush=True)
    print("  Parallel Video Intelligence Pipeline", flush=True)
    print("=" * 55, flush=True)
    print(f"  Video    : {args.video}",                       flush=True)
    print(f"  Whisper  : {args.whisper_model} (CPU int8)",    flush=True)
    print(f"  Face     : {args.accelerator}",                 flush=True)
    print(f"  LLM      : {args.llm_model} ({args.llm_provider})", flush=True)
    print(f"  Mode     : {args.mode}",                        flush=True)
    print("=" * 55, flush=True)

    # ── 1. Audio starts immediately ───────────────────────────────────────────
    print("\n[Pipeline] Step 1 — starting audio...", flush=True)
    p_audio = mp.Process(
        target=audio_process,
        args=(args.video, audio_started),
        name="Audio",
    )
    p_audio.start()
    audio_started.wait(timeout=5)
    print("[Pipeline] Audio playing.", flush=True)

    # ── 2. Face starts alongside audio ───────────────────────────────────────
    print("[Pipeline] Step 2 — starting face recognition...", flush=True)
    p_face = mp.Process(
        target=face_process,
        args=(
            args.video, args.db,
            args.threshold, args.margin, args.face_margin,
            args.min_frames, args.mode == "assist",
            args.accelerator,
            name_queue,
            video_started,
        ),
        name="Face",
    )
    p_face.start()

    # ── 3. Whisper starts AFTER first video frame is shown ───────────────────
    print("[Pipeline] Waiting for first video frame...", flush=True)
    if not video_started.wait(timeout=15):
        print("[Pipeline] WARNING: timed out — launching Whisper anyway.",
              flush=True)

    print("[Pipeline] Step 3 — starting Whisper now...", flush=True)
    p_whisper = mp.Process(
        target=whisper_process,
        args=(
            args.video, args.whisper_model,
            args.llm_provider, args.llm_api_base,
            args.llm_api_key, args.llm_model,
            name_queue, transcript_queue, args.save_transcript,
        ),
        name="Whisper",
    )
    p_whisper.start()

    # ── Wait for all ──────────────────────────────────────────────────────────
    p_face.join()
    p_audio.join()
    p_whisper.join()

    print("\n[Pipeline] All processes complete.", flush=True)
    if not transcript_queue.empty():
        transcript = transcript_queue.get()
        if transcript:
            print("\n─── FINAL TRANSCRIPT ─────────────────────────────")
            print(transcript)
            print("──────────────────────────────────────────────────")


if __name__ == "__main__":
    mp.freeze_support()   # required on Windows
    main()