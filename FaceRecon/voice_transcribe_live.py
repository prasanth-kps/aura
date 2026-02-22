"""
Live voice recording and transcription.

Records from the microphone in short chunks, transcribes with faster-whisper,
and displays the transcript in real time. No video required.

Usage:
  py voice_transcribe_live.py
  py voice_transcribe_live.py --chunk-sec 2 --model tiny --language en
"""

import argparse
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from audio_capture_utils import record_microphone_wav, temp_wav_path


def _record_mic(duration_sec: float, sample_rate: int = 16000) -> Path | None:
    """Record from microphone. Returns WAV path or None on failure."""
    try:
        out = temp_wav_path("voice_live_")
        record_microphone_wav(
            output_wav_path=out,
            duration_sec=duration_sec,
            sample_rate=sample_rate,
        )
        return out
    except Exception:
        return None


def _transcribe(audio_path: Path, model: str, language: str | None) -> str:
    """Transcribe WAV file with faster-whisper. Returns text."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper required. Install: py -m pip install faster-whisper"
        ) from exc

    whisper = WhisperModel(model_size_or_path=model, device="auto", compute_type="int8")
    segments, _ = whisper.transcribe(
        str(audio_path),
        language=language or None,
        vad_filter=True,
    )
    return " ".join(s.text.strip() for s in segments if s.text.strip()).strip()


def _wrap_text(text: str, max_chars_per_line: int = 50) -> list[str]:
    """Wrap text into lines for display."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for w in words:
        if len(current) + len(w) + 1 <= max_chars_per_line:
            current = f"{current} {w}".strip() if current else w
        else:
            if current:
                lines.append(current)
            current = w
    if current:
        lines.append(current)
    return lines


def _draw_transcript_frame(
    frame: np.ndarray,
    transcript: str,
    status: str = "Listening...",
    font_scale: float = 0.7,
) -> None:
    """Draw transcript and status on a dark frame."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    line_height = 28

    # Status at top
    cv2.putText(
        frame, status, (20, 35),
        font, 0.6, (100, 255, 100), thickness, cv2.LINE_AA,
    )

    # Transcript lines (scroll from bottom)
    lines = _wrap_text(transcript or "(speak into microphone)", 55)
    y = h - 30
    for line in reversed(lines[-8:]):
        cv2.putText(
            frame, line, (20, y),
            font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
        )
        y -= line_height


def run_live(
    chunk_sec: float,
    model: str,
    language: str | None,
    window_width: int,
    window_height: int,
) -> None:
    """Record voice and transcribe live. Display in a window."""
    transcript_lines: list[str] = []
    transcript_lock = threading.Lock()
    current_status = ["Listening..."]
    status_lock = threading.Lock()

    def _worker() -> None:
        while True:
            with status_lock:
                current_status[0] = "Recording..."
            wav = _record_mic(chunk_sec)
            with status_lock:
                current_status[0] = "Transcribing..."
            if wav is None:
                with status_lock:
                    current_status[0] = "Mic error - retrying..."
                time.sleep(1)
                continue
            try:
                text = _transcribe(wav, model, language)
                if text:
                    with transcript_lock:
                        transcript_lines.append(text)
            except Exception as e:
                with status_lock:
                    current_status[0] = f"Error: {e}"
            finally:
                wav.unlink(missing_ok=True)
            with status_lock:
                current_status[0] = "Listening..."

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

    cv2.namedWindow("Live Transcription", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Live Transcription", window_width, window_height)

    print("Speak into the microphone. Press 'q' to quit.")
    print("Transcription appears in the window as you speak.\n")

    while True:
        with transcript_lock:
            full = " ".join(transcript_lines)
        with status_lock:
            status = current_status[0]

        frame = np.zeros((window_height, window_width, 3), dtype=np.uint8)
        frame[:] = (30, 30, 30)
        _draw_transcript_frame(frame, full, status)

        cv2.imshow("Live Transcription", frame)
        if (cv2.waitKey(100) & 0xFF) == ord("q"):
            break

    cv2.destroyAllWindows()
    if transcript_lines:
        print("\n--- Full transcript ---")
        print(" ".join(transcript_lines))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record voice and transcribe live with faster-whisper.",
    )
    parser.add_argument(
        "--chunk-sec",
        type=float,
        default=2.0,
        help="Seconds to record per chunk (shorter = more responsive, less accurate).",
    )
    parser.add_argument(
        "--model",
        default="tiny",
        help="faster-whisper model: tiny, base, small, medium, large-v3.",
    )
    parser.add_argument(
        "--language",
        default="",
        help="Language hint, e.g. 'en' for English.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=720,
        help="Window width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=300,
        help="Window height.",
    )
    args = parser.parse_args()

    lang = args.language.strip() or None

    try:
        run_live(
            args.chunk_sec,
            args.model,
            lang,
            args.width,
            args.height,
        )
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
