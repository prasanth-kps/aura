import json
import mimetypes
import uuid
from pathlib import Path
from typing import Tuple
import urllib.error
import urllib.request

from llm_name_recognizer import extract_names_with_llm, regex_fallback_extract


def _build_multipart_body(
    audio_path: Path,
    model: str,
    language: str | None = None,
    prompt: str | None = None,
) -> Tuple[bytes, str]:
    boundary = f"----cursor-{uuid.uuid4().hex}"
    mime_type = mimetypes.guess_type(str(audio_path))[0] or "application/octet-stream"

    parts: list[bytes] = []

    def add_text_field(name: str, value: str) -> None:
        parts.append(f"--{boundary}\r\n".encode("utf-8"))
        parts.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
        parts.append(value.encode("utf-8"))
        parts.append(b"\r\n")

    def add_file_field(name: str, file_path: Path) -> None:
        filename = file_path.name
        parts.append(f"--{boundary}\r\n".encode("utf-8"))
        parts.append(
            (
                f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                f"Content-Type: {mime_type}\r\n\r\n"
            ).encode("utf-8")
        )
        parts.append(file_path.read_bytes())
        parts.append(b"\r\n")

    add_text_field("model", model)
    if language:
        add_text_field("language", language)
    if prompt:
        add_text_field("prompt", prompt)
    add_file_field("file", audio_path)

    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    body = b"".join(parts)
    content_type = f"multipart/form-data; boundary={boundary}"
    return body, content_type


def transcribe_audio_openai_compatible(
    audio_path: Path,
    api_base: str,
    api_key: str,
    model: str = "gpt-4o-mini-transcribe",
    language: str | None = None,
    prompt: str | None = None,
    timeout_sec: int = 90,
) -> str:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not api_key:
        raise ValueError("API key is required for audio transcription.")

    body, content_type = _build_multipart_body(audio_path, model=model, language=language, prompt=prompt)
    req = urllib.request.Request(
        url=f"{api_base.rstrip('/')}/audio/transcriptions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": content_type,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return str(payload.get("text", "")).strip()


def transcribe_audio_local(
    audio_path: Path,
    model: str = "small",
    language: str | None = None,
) -> str:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    try:
        from faster_whisper import WhisperModel
    except Exception as exc:
        raise RuntimeError(
            "Local ASR dependencies missing. Install with: py -m pip install faster-whisper"
        ) from exc

    whisper_model = WhisperModel(model_size_or_path=model, device="auto", compute_type="int8")
    segments, _ = whisper_model.transcribe(
        str(audio_path),
        language=language or None,
        vad_filter=True,
    )
    transcript = " ".join(seg.text.strip() for seg in segments).strip()
    return transcript


def transcribe_audio_huggingface(
    audio_path: Path,
    model: str = "openai/whisper-small",
) -> str:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # WavLM base-plus is a speech representation model (not direct ASR).
    # Keep this explicit so users don't silently get poor/invalid behavior.
    if "wavlm-base-plus" in model.lower():
        raise RuntimeError(
            "microsoft/wavlm-base-plus is not a speech-to-text model. "
            "Use a Hugging Face ASR model (e.g. openai/whisper-small or facebook/wav2vec2-base-960h)."
        )

    try:
        from transformers import pipeline
    except Exception as exc:
        raise RuntimeError("Hugging Face ASR requires transformers. Install with: py -m pip install transformers") from exc

    asr = pipeline("automatic-speech-recognition", model=model)
    result = asr(str(audio_path))
    if isinstance(result, dict):
        return str(result.get("text", "")).strip()
    return str(result).strip()


def extract_names_from_audio(
    audio_path: Path,
    llm_model: str,
    llm_provider: str = "openai",
    llm_api_base: str = "",
    llm_api_key: str = "",
    transcribe_model: str = "small",
    asr_provider: str = "local",
    asr_api_base: str = "https://api.openai.com/v1",
    asr_api_key: str = "",
    language: str | None = None,
) -> tuple[list[str], str]:
    if asr_provider == "local":
        transcript = transcribe_audio_local(
            audio_path=audio_path,
            model=transcribe_model,
            language=language,
        )
    elif asr_provider == "huggingface":
        transcript = transcribe_audio_huggingface(
            audio_path=audio_path,
            model=transcribe_model,
        )
    else:
        transcript = transcribe_audio_openai_compatible(
            audio_path=audio_path,
            api_base=asr_api_base,
            api_key=asr_api_key,
            model=transcribe_model,
            language=language,
        )
    if not transcript:
        return [], ""

    try:
        names = extract_names_with_llm(
            text=transcript,
            api_base=llm_api_base or asr_api_base,
            api_key=llm_api_key,
            model=llm_model,
            provider=llm_provider,
        )
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, KeyError):
        names = regex_fallback_extract(transcript)
    return names, transcript
