"""
live_subtitles.py
=================
Real-time subtitles using faster-whisper (small, int8 quantized).
Falls back to QAI Hub Whisper if faster-whisper is unavailable.

Works on ARM64 Windows, x64 Windows, Linux, macOS.

Requirements
------------
    py -m pip install faster-whisper opencv-python sounddevice numpy --user

Usage
-----
    # Transcribe a video file (no mic/camera needed)
    py live_subtitles.py --input-file videoplayback.mp4

    # Live mic + camera overlay window
    py live_subtitles.py

    # Terminal-only (no camera)
    py live_subtitles.py --no-window

    # Save to SRT file
    py live_subtitles.py --input-file video.mp4 --srt-out out.srt

    # Tune chunk size
    py live_subtitles.py --chunk-sec 4 --overlap-sec 0.5
"""

from __future__ import annotations

import argparse
import queue
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_RATE        = 16_000
MAX_SUBTITLE_LINES = 3
SUBTITLE_HISTORY   = 6
FONT_SCALE         = 0.65
FONT_THICKNESS     = 2


# ---------------------------------------------------------------------------
# Model loading  (faster-whisper first — most ARM64-friendly)
# ---------------------------------------------------------------------------

def load_whisper_model(accelerator: str = "auto"):
    """
    Priority:
      1. faster-whisper small int8   (ARM64 + x64, no GPU needed)
      2. faster-whisper small float32 (if int8 CTranslate2 build is missing)
      3. QAI Hub whisper_small_en    (Snapdragon NPU path)
      4. QAI Hub whisper_small       (multilingual NPU path)
    """
    try:
        from faster_whisper import WhisperModel as FWModel
    except ImportError:
        print("[WARN] faster-whisper not installed. Run: py -m pip install faster-whisper --user")
        FWModel = None

    if FWModel is not None:
        for compute_type in ("int8", "float32"):
            try:
                print(f"[INFO] Loading faster-whisper/small ({compute_type}) ...")
                m = FWModel("small", device="cpu", compute_type=compute_type)
                # Smoke-test: transcribe 0.5s of silence to confirm model works
                silence = np.zeros(SAMPLE_RATE // 2, dtype=np.float32)
                list(m.transcribe(silence, beam_size=1)[0])
                print(f"[INFO] faster-whisper ready (compute_type={compute_type})")
                return m, "faster_whisper"
            except Exception as e:
                print(f"[WARN] faster-whisper {compute_type} failed: {e}")

    # QAI Hub fallback
    for mod_name in (
        "qai_hub_models.models.whisper_small_en",
        "qai_hub_models.models.whisper_small",
    ):
        try:
            import importlib
            mod = importlib.import_module(mod_name)
            print(f"[INFO] Loading {mod_name} ...")
            m = mod.Model.from_pretrained()
            m.eval()
            print("[INFO] QAI Hub Whisper model ready.")
            return m, "qai_hub"
        except Exception as e:
            print(f"[WARN] {mod_name}: {e}")

    raise RuntimeError(
        "\n[ERROR] No ASR backend available.\n"
        "Fix:  py -m pip install faster-whisper --user\n"
    )


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def transcribe_chunk(model, model_type: str, audio_np: np.ndarray, language: str | None) -> str:
    if len(audio_np) < SAMPLE_RATE * 0.3:
        return ""
    if model_type == "faster_whisper":
        return _transcribe_fw(model, audio_np, language)
    return _transcribe_qai(model, audio_np)


def _transcribe_fw(model, audio_np: np.ndarray, language: str | None) -> str:
    try:
        segments_gen, _info = model.transcribe(
            audio_np,
            language=language or "en",   # default en skips language detection (~0.5s saved)
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 200},
            beam_size=1,               # greedy decode — 2-3x faster, minimal accuracy loss
            without_timestamps=True,
            condition_on_previous_text=False,  # prevents hallucination on silence
        )
        # MUST fully consume the lazy generator to get results
        parts = [seg.text.strip() for seg in segments_gen if seg.text.strip()]
        return " ".join(parts)
    except Exception as e:
        print(f"[WARN] Transcription error: {e}")
        return ""


def _transcribe_qai(model, audio_np: np.ndarray) -> str:
    import torch
    try:
        if hasattr(model, "transcribe"):
            r = model.transcribe(audio_np)
            return (r if isinstance(r, str) else r.get("text", "")).strip()
    except Exception:
        pass
    try:
        mel = _compute_log_mel(audio_np)
        mel_t = torch.from_numpy(mel).float()
        with torch.no_grad():
            out = model(mel_t)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if isinstance(out, str):
            return out.strip()
        if hasattr(out, "numpy"):
            return _decode_tokens(out.squeeze().long().tolist())
    except Exception as e:
        print(f"[WARN] QAI transcription error: {e}")
    return ""


def _compute_log_mel(audio: np.ndarray, n_mels: int = 80, n_fft: int = 400,
                     hop_length: int = 160, target_frames: int = 3000) -> np.ndarray:
    try:
        import torch
        import torchaudio.transforms as T
        wf = torch.from_numpy(audio).float().unsqueeze(0)
        mel = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=n_fft,
            hop_length=hop_length, n_mels=n_mels, power=2.0,
        )(wf)
        lm = torch.clamp(mel, min=1e-10).log10()
        lm = torch.maximum(lm, lm.max() - 8.0)
        lm = (lm + 4.0) / 4.0
        f = lm.shape[-1]
        lm = torch.nn.functional.pad(lm, (0, target_frames - f)) if f < target_frames else lm[..., :target_frames]
        return lm.numpy()
    except Exception:
        return np.zeros((1, n_mels, target_frames), dtype=np.float32)


def _decode_tokens(tokens: list[int]) -> str:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        clean = [t for t in tokens if 0 <= t < enc.n_vocab and t not in {50256, 50257}]
        return enc.decode(clean).strip()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Audio extraction from file  (ffmpeg -> PyAV fallback)
# ---------------------------------------------------------------------------

def extract_audio_from_file(file_path: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Extract mono float32 audio from any video/audio file."""
    tmp = Path(tempfile.mktemp(suffix=".wav"))
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(file_path),
             "-vn", "-ac", "1", "-ar", str(sample_rate), str(tmp)],
            check=True, capture_output=True,
        )
        import wave
        with wave.open(str(tmp), "rb") as wf:
            raw = wf.readframes(wf.getnframes())
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        print(f"[INFO] Audio extracted via ffmpeg: {len(audio)/sample_rate:.1f}s")
        return audio
    except FileNotFoundError:
        print("[WARN] ffmpeg not found in PATH — trying PyAV...")
    except subprocess.CalledProcessError as e:
        print(f"[WARN] ffmpeg failed: {e.stderr.decode()[:200]}")
    finally:
        tmp.unlink(missing_ok=True)

    # PyAV fallback
    try:
        import av
        container = av.open(str(file_path))
        if not container.streams.audio:
            raise RuntimeError("No audio stream found.")
        stream = container.streams.audio[0]
        src_rate = int(stream.sample_rate) if stream.sample_rate else 48000
        chunks: list[np.ndarray] = []
        for frame in container.decode(audio=0):
            arr = frame.to_ndarray()
            if arr.ndim == 2:
                arr = arr.mean(axis=0)
            arr = arr.astype(np.float32)
            if np.issubdtype(arr.dtype, np.integer):
                arr /= 32768.0
            chunks.append(arr)
        container.close()
        audio = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
        if src_rate != sample_rate:
            from scipy.signal import resample_poly
            audio = resample_poly(audio, sample_rate, src_rate).astype(np.float32)
        print(f"[INFO] Audio extracted via PyAV: {len(audio)/sample_rate:.1f}s")
        return audio
    except ImportError:
        pass
    except Exception as e:
        print(f"[WARN] PyAV failed: {e}")

    raise RuntimeError(
        "Cannot extract audio. Install ffmpeg (add to PATH) or:\n"
        "  py -m pip install av --user"
    )


# ---------------------------------------------------------------------------
# Microphone stream
# ---------------------------------------------------------------------------

class MicrophoneStream:
    def __init__(self, sample_rate: int = SAMPLE_RATE, chunk_frames: int = 1024):
        self.sample_rate = sample_rate
        self.chunk_frames = chunk_frames
        self._q: queue.Queue[np.ndarray] = queue.Queue()
        self._stream = None

    def start(self):
        try:
            import sounddevice as sd
        except ImportError:
            raise RuntimeError("sounddevice required: py -m pip install sounddevice --user")
        self._stream = sd.InputStream(
            samplerate=self.sample_rate, channels=1,
            dtype="float32", blocksize=self.chunk_frames,
            callback=lambda d, f, t, s: self._q.put(d[:, 0].copy()),
        )
        self._stream.start()
        print(f"[INFO] Microphone stream started ({self.sample_rate} Hz)")

    def read_seconds(self, seconds: float) -> np.ndarray:
        needed = int(self.sample_rate * seconds)
        buf, collected = [], 0
        while collected < needed:
            chunk = self._q.get(timeout=10)
            buf.append(chunk)
            collected += len(chunk)
        audio = np.concatenate(buf)
        return audio[:needed]

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()


# ---------------------------------------------------------------------------
# SRT writer
# ---------------------------------------------------------------------------

class SRTWriter:
    def __init__(self, path: Path):
        self._path = path
        self._idx = 1
        self._fh = open(path, "w", encoding="utf-8")

    def write(self, text: str, start: float, end: float):
        if not text.strip():
            return
        self._fh.write(f"{self._idx}\n{_srt_ts(start)} --> {_srt_ts(end)}\n{text.strip()}\n\n")
        self._fh.flush()
        self._idx += 1

    def close(self):
        self._fh.close()
        print(f"[INFO] SRT saved → {self._path}")


def _srt_ts(s: float) -> str:
    h, r = divmod(int(s), 3600)
    m, sec = divmod(r, 60)
    ms = int((s - int(s)) * 1000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


# ---------------------------------------------------------------------------
# Overlay renderer
# ---------------------------------------------------------------------------

def draw_subtitles(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    import cv2
    if not lines:
        return frame
    overlay = frame.copy()
    h, w = frame.shape[:2]
    line_h, pad = 32, 10
    total_h = line_h * len(lines) + pad * 2
    cv2.rectangle(overlay, (0, h - total_h - 10), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)
    y = h - total_h - 10 + pad + line_h - 8
    for line in lines:
        cv2.putText(frame, line, (12, y + 2), cv2.FONT_HERSHEY_SIMPLEX,
                    FONT_SCALE, (0, 0, 0), FONT_THICKNESS + 1, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    FONT_SCALE, (255, 255, 255), FONT_THICKNESS, cv2.LINE_AA)
        y += line_h
    return frame


def wrap_text(text: str, max_chars: int = 78) -> list[str]:
    words, lines, current = text.split(), [], ""
    for w in words:
        if len(current) + len(w) + 1 <= max_chars:
            current = f"{current} {w}".strip()
        else:
            if current:
                lines.append(current)
            current = w
    if current:
        lines.append(current)
    return lines or [""]


# ---------------------------------------------------------------------------
# File transcription mode
# ---------------------------------------------------------------------------

def run_file_mode(model, model_type: str, input_file: Path,
                  chunk_sec: float, language: str | None, srt_writer: SRTWriter | None):
    print(f"[INFO] Transcribing: {input_file}")
    audio = extract_audio_from_file(input_file)
    total = len(audio)
    if total == 0:
        print("[ERROR] No audio data extracted.")
        return

    chunk_samples = int(chunk_sec * SAMPLE_RATE)
    total_chunks = (total + chunk_samples - 1) // chunk_samples
    print(f"[INFO] {total/SAMPLE_RATE:.1f}s audio => {total_chunks} chunks of {chunk_sec}s each")
    print("-" * 60)

    # Build chunk list with timestamps
    chunks = []
    t = 0.0
    for offset in range(0, total, chunk_samples):
        c = audio[offset: offset + chunk_samples]
        if len(c) > 0:
            chunks.append((t, t + len(c) / SAMPLE_RATE, c))
            t += len(c) / SAMPLE_RATE

    # Parallel transcription — CTranslate2 releases the GIL so threads help
    from concurrent.futures import ThreadPoolExecutor, as_completed
    n_workers = min(4, len(chunks))
    print(f"[INFO] Processing {len(chunks)} chunks with {n_workers} parallel workers")

    results: dict[int, tuple[float, float, str]] = {}

    def _transcribe_indexed(args):
        idx, (start_s, end_s, chunk) = args
        return idx, start_s, end_s, transcribe_chunk(model, model_type, chunk, language)

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_transcribe_indexed, (i, c)): i for i, c in enumerate(chunks)}
        done = 0
        for fut in as_completed(futs):
            idx, s, e, text = fut.result()
            results[idx] = (s, e, text)
            done += 1
            sys.stdout.write(f"\r  [{done}/{len(chunks)}] chunks done...")
            sys.stdout.flush()

    sys.stdout.write("\r" + " " * 50 + "\r")
    for idx in sorted(results):
        s, e, text = results[idx]
        if text:
            print(f"[{_srt_ts(s)}]  {text}")
            if srt_writer:
                srt_writer.write(text, s, e)

    print("\n" + "-" * 60)
    print("[INFO] Done.")


# ---------------------------------------------------------------------------
# Live microphone mode
# ---------------------------------------------------------------------------

def run_live_mic(model, model_type: str, chunk_sec: float, overlap_sec: float,
                 language: str | None, no_window: bool, camera_index: int,
                 srt_writer: SRTWriter | None):
    mic = MicrophoneStream()
    mic.start()

    cap = None
    if not no_window:
        try:
            import cv2
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                print("[WARN] Camera unavailable — subtitle-only window.")
                cap = None
        except ImportError:
            print("[WARN] opencv-python not installed — terminal mode.")
            no_window = True

    subtitle_history: Deque[str] = deque(maxlen=SUBTITLE_HISTORY)
    current_subtitle: list[str] = []
    result_q: queue.Queue[str] = queue.Queue()
    overlap_buf = np.array([], dtype=np.float32)
    session_start = chunk_start = time.time()

    def worker():
        nonlocal overlap_buf
        while True:
            audio = mic.read_seconds(chunk_sec)
            if len(overlap_buf) > 0:
                audio = np.concatenate([overlap_buf, audio])
            overlap_samples = int(overlap_sec * SAMPLE_RATE)
            overlap_buf = audio[-overlap_samples:] if overlap_samples > 0 else np.array([], dtype=np.float32)
            text = transcribe_chunk(model, model_type, audio, language)
            if text:
                result_q.put(text)

    threading.Thread(target=worker, daemon=True).start()
    print("[INFO] Live subtitles running. Press Ctrl+C or 'q' to stop.")
    print("-" * 60)

    try:
        while True:
            while not result_q.empty():
                text = result_q.get_nowait()
                now = time.time()
                if srt_writer:
                    srt_writer.write(text, chunk_start - session_start, now - session_start)
                chunk_start = now
                for line in wrap_text(text):
                    subtitle_history.append(line)
                current_subtitle = list(subtitle_history)[-MAX_SUBTITLE_LINES:]
                print(f"[SUBTITLE] {text}")

            if not no_window:
                import cv2
                if cap and cap.isOpened():
                    ok, frame = cap.read()
                    if not ok:
                        frame = np.zeros((480, 854, 3), dtype=np.uint8)
                else:
                    frame = np.zeros((200, 854, 3), dtype=np.uint8)
                frame = draw_subtitles(frame, current_subtitle)
                cv2.putText(frame, f"LIVE SUBTITLES  {time.strftime('%H:%M:%S')}  [q=quit]",
                            (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1, cv2.LINE_AA)
                cv2.imshow("Live Subtitles - Whisper Small", frame)
                if cv2.waitKey(30) & 0xFF in (ord("q"), 27):
                    break
            else:
                time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[INFO] Stopping...")
    finally:
        mic.stop()
        if cap:
            cap.release()
        if not no_window:
            try:
                import cv2; cv2.destroyAllWindows()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Live subtitles — faster-whisper small (ARM64 + x64 compatible).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input-file",   default="", help="Video/audio file to transcribe.")
    parser.add_argument("--no-window",    action="store_true", help="Terminal-only output.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--chunk-sec",    type=float, default=3.0,
                        help="Seconds per transcription chunk (default 3).")
    parser.add_argument("--overlap-sec",  type=float, default=0.5,
                        help="Overlap between chunks (default 0.5).")
    parser.add_argument("--language",     default="", help="Language hint e.g. 'en' (default: auto).")
    parser.add_argument("--srt-out",      default="", help="Save subtitles to .srt file.")
    parser.add_argument("--accelerator",  choices=["auto", "gpu", "cpu"], default="auto")
    args = parser.parse_args()

    model, model_type = load_whisper_model(args.accelerator)

    srt_writer = SRTWriter(Path(args.srt_out)) if args.srt_out else None
    try:
        if args.input_file:
            run_file_mode(model, model_type, Path(args.input_file),
                          args.chunk_sec, args.language or None, srt_writer)
        else:
            run_live_mic(model, model_type, args.chunk_sec, args.overlap_sec,
                         args.language or None, args.no_window, args.camera_index, srt_writer)
    finally:
        if srt_writer:
            srt_writer.close()


if __name__ == "__main__":
    main()