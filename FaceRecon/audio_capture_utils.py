import os
import subprocess
import tempfile
from pathlib import Path


def extract_wav_from_video(
    video_path: Path,
    output_wav_path: Path,
    sample_rate: int = 16000,
) -> Path:
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(output_wav_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg not found. Install ffmpeg and add it to PATH.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(f"ffmpeg failed to extract audio: {stderr}") from exc
    return output_wav_path


def record_microphone_wav(
    output_wav_path: Path,
    duration_sec: float = 8.0,
    sample_rate: int = 16000,
) -> Path:
    if duration_sec <= 0:
        raise ValueError("duration_sec must be > 0")

    # Prefer ffmpeg capture on Windows to avoid local PortAudio issues.
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "dshow",
        "-i",
        "audio=default",
        "-t",
        f"{duration_sec:.3f}",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(output_wav_path),
    ]
    try:
        print(f"Recording microphone for {duration_sec:.1f}s...")
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return output_wav_path
    except Exception:
        pass

    # Fallback to sounddevice if ffmpeg capture is unavailable.
    try:
        import numpy as np
        import sounddevice as sd
        import wave

        frames = int(sample_rate * duration_sec)
        print(f"Recording microphone for {duration_sec:.1f}s...")
        audio = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="float32")
        sd.wait()
        audio_int16 = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)

        with wave.open(str(output_wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())
        return output_wav_path
    except Exception as exc:
        raise RuntimeError(
            "Microphone recording failed. Install ffmpeg (recommended) or run: py -m pip install sounddevice numpy"
        ) from exc


def temp_wav_path(prefix: str = "intro_audio_") -> Path:
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".wav")
    os.close(fd)
    Path(path).unlink(missing_ok=True)
    return Path(path)
