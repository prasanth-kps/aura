"""
Aura Smart Specs Console
========================
A wearable-style Streamlit interface for smart glasses capabilities:
1. Face ID Lens - face recognition + speech-derived name hints
2. Scene Insight Lens - region-based image understanding
3. Find-My-Object Lens - object memory and natural language retrieval

Usage:
    streamlit run demo-ui/unified_app.py
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os

# Disable Windows Media Foundation backend BEFORE any cv2 import.
# MSMF causes "can't grab frame" crashes on many MP4 files; FFMPEG is more compatible.
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")

import queue as _std_queue
import re as _re
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st
import numpy as np

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "FaceRecon"))
sys.path.insert(0, str(PROJECT_ROOT / "image-summarizer"))

# ──────────────────────────────────────────────────────────────────────────────
# STREAMING STATE (rerun-stable)
# Keep state in cache_resource so Streamlit script reruns don't recreate it.
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def _get_face_stream_state() -> dict[str, Any]:
    return {
        "streams": {},              # dict[str, dict[str, Any]]
        "lock": threading.Lock(),   # protects streams map
    }


_FACE_STATE = _get_face_stream_state()
_FACE_STREAMS: dict[str, dict[str, Any]] = _FACE_STATE["streams"]
_FACE_STREAMS_LOCK: threading.Lock = _FACE_STATE["lock"]

# ──────────────────────────────────────────────────────────────────────────────
# TRANSCRIPT DISK CACHE
# Keyed by SHA-256 of the first 512 KB of the video so the same file is never
# transcribed twice — even across full app restarts.
# ──────────────────────────────────────────────────────────────────────────────
_TRANSCRIPT_CACHE_DIR = Path(tempfile.gettempdir()) / "aura_whisper_cache"
_TRANSCRIPT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_TRANSCRIPT_MEM: dict[str, str] = {}   # in-memory layer (survives reruns)


def _video_cache_key(video_path: str) -> str:
    """Fast content hash: SHA-256 of up to 512 KB of the file."""
    try:
        h = hashlib.sha256()
        with open(video_path, "rb") as f:
            h.update(f.read(512 * 1024))
        return h.hexdigest()
    except Exception:
        return hashlib.sha256(video_path.encode()).hexdigest()


def _load_transcript_cache(video_path: str) -> str | None:
    """Return cached transcript string if it exists, else None."""
    key = _video_cache_key(video_path)
    if key in _TRANSCRIPT_MEM:
        return _TRANSCRIPT_MEM[key]
    cache_file = _TRANSCRIPT_CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            transcript = data.get("transcript", "")
            _TRANSCRIPT_MEM[key] = transcript
            return transcript
        except Exception:
            pass
    return None


def _save_transcript_cache(video_path: str, transcript: str) -> None:
    """Persist transcript to both in-memory dict and disk."""
    key = _video_cache_key(video_path)
    _TRANSCRIPT_MEM[key] = transcript
    cache_file = _TRANSCRIPT_CACHE_DIR / f"{key}.json"
    try:
        cache_file.write_text(
            json.dumps({"transcript": transcript}, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        pass


# ============================================================
# STYLING
# ============================================================

def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .main {
            background: #000000 !important;
        }
        .stApp {
            background: #000000 !important;
            color: #f0f4ff;
        }
        .hero-card {
            border: 1px solid rgba(122, 162, 255, 0.35);
            border-radius: 16px;
            padding: 18px 20px;
            margin-bottom: 14px;
            background: rgba(15, 22, 43, 0.75);
            backdrop-filter: blur(2px);
        }
        .hero-title {
            font-size: 1.4rem;
            font-weight: 700;
            margin-bottom: 6px;
            background: linear-gradient(90deg, #7aa2ff 0%, #c084fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .hero-sub {
            color: #d8e2ff;
            margin-bottom: 0;
        }
        .feature-card {
            border: 1px solid rgba(122, 162, 255, 0.25);
            border-radius: 12px;
            padding: 16px;
            margin: 8px 0;
            background: rgba(15, 22, 43, 0.6);
        }
        .feature-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: #d8b4fe;
        }
        .muted {
            color: #d4deff;
            font-size: 0.92rem;
        }
        .success-box {
            background: rgba(34, 197, 94, 0.15);
            border: 1px solid rgba(34, 197, 94, 0.4);
            border-radius: 8px;
            padding: 12px;
            margin: 8px 0;
        }
        .warning-box {
            background: rgba(234, 179, 8, 0.15);
            border: 1px solid rgba(234, 179, 8, 0.4);
            border-radius: 8px;
            padding: 12px;
            margin: 8px 0;
        }
        /* Image Summarizer Tab - Dark theme with accent glow */
        .image-summarizer-header {
            position: relative;
            margin-bottom: 20px;
        }
        .glow-orb {
            position: absolute;
            top: -40px;
            left: 50%;
            transform: translateX(-50%);
            width: 300px;
            height: 150px;
            background: radial-gradient(ellipse at center,
                        rgba(139, 92, 246, 0.4) 0%,
                        rgba(59, 130, 246, 0.2) 30%,
                        transparent 70%);
            filter: blur(30px);
            pointer-events: none;
            z-index: 0;
        }
        .image-card {
            position: relative;
            z-index: 1;
            background: rgba(15, 22, 43, 0.6) !important;
            border: 1px solid rgba(139, 92, 246, 0.4) !important;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3),
                        0 0 60px rgba(139, 92, 246, 0.15),
                        0 0 100px rgba(59, 130, 246, 0.08) !important;
        }
        .image-card .feature-title {
            font-size: 1.3rem !important;
            background: linear-gradient(90deg, #7c3aed 0%, #3b82f6 50%, #8b5cf6 100%) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
        }
        .image-card .muted {
            color: #d4deff !important;
        }
        .glow-text {
            text-shadow: 0 0 20px rgba(139, 92, 246, 0.6),
                         0 0 40px rgba(139, 92, 246, 0.4),
                         0 0 60px rgba(59, 130, 246, 0.2);
        }
        /* Summary result styling - dark theme */
        .summary-result {
            background: rgba(15, 22, 43, 0.75);
            border: 1px solid rgba(139, 92, 246, 0.3);
            border-radius: 12px;
            padding: 20px;
            margin: 16px 0;
            box-shadow: 0 4px 24px rgba(0, 0, 0, 0.2),
                        0 0 40px rgba(139, 92, 246, 0.1);
            color: #f0f4ff;
        }
        .summary-result h3 {
            color: #c4b5fd;
            margin-bottom: 12px;
        }
        .summary-result p {
            color: #f0f4ff !important;
        }
        /* Detailed Analysis styling - black background */
        .detailed-analysis {
            background: linear-gradient(145deg, #0a0a0f 0%, #121218 100%);
            border: 1px solid rgba(139, 92, 246, 0.3);
            border-radius: 12px;
            padding: 20px;
            margin: 8px 0;
            color: #f0f4ff;
        }
        .detailed-analysis h4 {
            color: #c4b5fd;
            margin-bottom: 12px;
            font-size: 1.1rem;
        }
        .detailed-analysis .analysis-section {
            margin-bottom: 16px;
        }
        .detailed-analysis .section-title {
            color: #d8b4fe;
            font-weight: 600;
            margin-bottom: 8px;
        }
        .detailed-analysis .item {
            color: #e8eaed;
            padding: 4px 0;
            border-bottom: 1px solid rgba(139, 92, 246, 0.1);
        }
        .detailed-analysis .score {
            color: #93c5fd;
            font-family: monospace;
        }
        /* ========== GLOBAL TEXT - BRIGHT WHITE ========== */
        * {
            color: #ffffff;
        }

        /* All paragraphs and spans */
        p, span, div, label, li, td, th {
            color: #ffffff !important;
        }

        /* Headings - bright white */
        h1, h2, h3, h4, h5, h6 {
            color: #ffffff !important;
        }

        /* ========== STREAMLIT WIDGET LABELS ========== */
        .stRadio > label,
        .stSelectbox > label,
        .stTextInput > label,
        .stNumberInput > label,
        .stSlider > label,
        .stFileUploader > label,
        .stCheckbox > label,
        .stMultiSelect > label,
        .stTextArea > label,
        [data-testid="stWidgetLabel"] {
            color: #ffffff !important;
            font-weight: 500 !important;
        }

        /* Radio button options */
        .stRadio > div[role="radiogroup"] > label,
        .stRadio [data-testid="stMarkdownContainer"] p {
            color: #ffffff !important;
        }

        /* Selectbox text and background */
        .stSelectbox > div > div,
        .stSelectbox [data-baseweb="select"] span,
        [data-baseweb="select"] > div {
            color: #ffffff !important;
            background-color: #000000 !important;
        }

        /* Selectbox container */
        .stSelectbox [data-baseweb="select"],
        .stSelectbox [data-baseweb="select"] > div {
            background-color: #000000 !important;
            border-color: rgba(122, 162, 255, 0.4) !important;
        }

        /* Dropdown menu container */
        [data-baseweb="popover"],
        [data-baseweb="popover"] > div,
        [data-baseweb="menu"],
        [data-baseweb="select"] [role="listbox"],
        ul[role="listbox"],
        div[data-baseweb="popover"] {
            background-color: #000000 !important;
            background: #000000 !important;
            border: 1px solid rgba(122, 162, 255, 0.4) !important;
        }

        /* Dropdown list container */
        [data-baseweb="list"],
        [data-baseweb="menu"] > div,
        [role="listbox"],
        [role="listbox"] > div {
            background-color: #000000 !important;
            background: #000000 !important;
        }

        /* Dropdown options */
        [data-baseweb="menu"] li,
        [data-baseweb="select"] [role="option"],
        ul[role="listbox"] li,
        [role="option"],
        li[role="option"] {
            background-color: #000000 !important;
            background: #000000 !important;
            color: #ffffff !important;
        }

        /* Dropdown option hover */
        [data-baseweb="menu"] li:hover,
        [data-baseweb="select"] [role="option"]:hover,
        ul[role="listbox"] li:hover,
        [role="option"]:hover,
        li[role="option"]:hover {
            background-color: #1a1a2e !important;
            background: #1a1a2e !important;
        }

        /* Selected option highlight */
        [data-baseweb="menu"] li[aria-selected="true"],
        [data-baseweb="select"] [role="option"][aria-selected="true"],
        [role="option"][aria-selected="true"] {
            background-color: rgba(122, 162, 255, 0.2) !important;
            background: rgba(122, 162, 255, 0.2) !important;
        }

        /* Streamlit specific dropdown styling */
        .stSelectbox div[data-baseweb="popover"] > div {
            background-color: #000000 !important;
        }

        .stSelectbox ul {
            background-color: #000000 !important;
            background: #000000 !important;
        }

        .stSelectbox ul li {
            background-color: #000000 !important;
            background: #000000 !important;
            color: #ffffff !important;
        }

        .stSelectbox ul li:hover {
            background-color: #1a1a2e !important;
            background: #1a1a2e !important;
        }

        /* Text inputs */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stTextArea textarea,
        input, textarea {
            color: #ffffff !important;
            background-color: rgba(30, 30, 50, 0.8) !important;
        }

        /* Placeholder text */
        input::placeholder, textarea::placeholder {
            color: #a0a0b0 !important;
        }

        /* ========== BUTTONS ========== */
        .stButton > button,
        .stDownloadButton > button,
        button {
            color: #ffffff !important;
        }

        /* Primary button */
        .stButton > button[kind="primary"] {
            background: linear-gradient(90deg, #7c3aed, #3b82f6) !important;
            color: #ffffff !important;
            font-weight: 600 !important;
        }

        /* ========== SLIDERS ========== */
        .stSlider > div > div > div > div,
        .stSlider [data-testid="stTickBarMin"],
        .stSlider [data-testid="stTickBarMax"],
        .stSlider span {
            color: #ffffff !important;
        }

        /* ========== FILE UPLOADER ========== */
        .stFileUploader > div > div,
        .stFileUploader label,
        .stFileUploader span,
        [data-testid="stFileUploader"] * {
            color: #ffffff !important;
        }

        /* Drag and drop box */
        .stFileUploader [data-testid="stFileUploaderDropzone"] {
            color: #ffffff !important;
            background-color: #000000 !important;
            border-color: rgba(122, 162, 255, 0.4) !important;
        }

        /* File uploader dropzone inner elements */
        .stFileUploader [data-testid="stFileUploaderDropzone"] > div {
            background-color: #000000 !important;
        }

        .stFileUploader section {
            background-color: #000000 !important;
        }

        .stFileUploader [data-testid="stFileUploaderDropzoneInput"] {
            background-color: #000000 !important;
        }

        /* Browse files button */
        .stFileUploader button,
        .stFileUploader [data-testid="stFileUploaderDropzone"] button,
        .stFileUploader [data-testid="baseButton-secondary"] {
            background-color: #000000 !important;
            color: #ffffff !important;
            border: 1px solid rgba(122, 162, 255, 0.4) !important;
        }

        .stFileUploader button:hover {
            background-color: #1a1a2e !important;
            border-color: rgba(122, 162, 255, 0.6) !important;
        }

        /* ========== EXPANDERS ========== */
        .streamlit-expanderHeader,
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] span {
            color: #ffffff !important;
        }

        /* ========== TABS ========== */
        .stTabs [data-baseweb="tab"],
        .stTabs [data-baseweb="tab-list"] button {
            color: #ffffff !important;
        }

        /* Active tab */
        .stTabs [aria-selected="true"] {
            color: #a78bfa !important;
            border-bottom-color: #a78bfa !important;
        }

        /* ========== CHECKBOXES ========== */
        .stCheckbox > label > span,
        .stCheckbox label {
            color: #ffffff !important;
        }

        /* ========== METRICS ========== */
        [data-testid="stMetricLabel"],
        [data-testid="stMetricLabel"] p {
            color: #c4b5fd !important;
        }

        [data-testid="stMetricValue"],
        [data-testid="stMetricValue"] div {
            color: #ffffff !important;
            font-weight: 600 !important;
        }

        /* ========== CAPTIONS & INFO ========== */
        .stCaption, small, .caption {
            color: #c4b5fd !important;
        }

        /* Info, success, warning, error boxes */
        .stAlert, [data-testid="stAlert"] {
            color: #ffffff !important;
        }

        .stAlert p, [data-testid="stAlert"] p {
            color: #ffffff !important;
        }

        /* ========== MARKDOWN TEXT ========== */
        [data-testid="stMarkdownContainer"],
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stMarkdownContainer"] span {
            color: #ffffff !important;
        }

        /* ========== SIDEBAR ========== */
        [data-testid="stSidebar"] {
            background: #000000 !important;
        }

        [data-testid="stSidebar"] > div:first-child {
            background: #000000 !important;
        }

        [data-testid="stSidebar"],
        [data-testid="stSidebar"] * {
            color: #ffffff !important;
        }

        [data-testid="stSidebar"] .stMarkdown p {
            color: #e0e0f0 !important;
        }

        /* Sidebar header */
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #ffffff !important;
        }

        /* ========== CODE BLOCKS ========== */
        code, pre, .stCode {
            color: #e0e0f0 !important;
            background-color: rgba(20, 20, 40, 0.9) !important;
        }

        /* ========== TEXT AREA ========== */
        .stTextArea label,
        .stTextArea textarea {
            color: #ffffff !important;
        }

        /* ========== DIVIDERS ========== */
        hr, .stDivider {
            border-color: rgba(122, 162, 255, 0.3) !important;
        }

        /* ========== LINKS ========== */
        a {
            color: #93c5fd !important;
        }

        a:hover {
            color: #bfdbfe !important;
        }

        /* ========== SPINNERS ========== */
        .stSpinner > div > div {
            color: #ffffff !important;
        }

        /* ========== WRITE OUTPUT ========== */
        .stWrite, .element-container {
            color: #ffffff !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FACE RECOGNITION MODULE
# ============================================================


@dataclass
class CommandResult:
    ok: bool
    command: list[str]
    stdout: str
    stderr: str
    return_code: int
    elapsed_seconds: float
    error: str | None = None


def run_command(cmd: list[str], cwd: Path, timeout_seconds: int = 300) -> CommandResult:
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        elapsed = max(0.0, time.perf_counter() - start)
        ok = proc.returncode == 0
        error = None if ok else f"Command failed with exit code {proc.returncode}"
        return CommandResult(
            ok=ok,
            command=cmd,
            stdout=proc.stdout,
            stderr=proc.stderr,
            return_code=proc.returncode,
            elapsed_seconds=elapsed,
            error=error,
        )
    except subprocess.TimeoutExpired:
        elapsed = max(0.0, time.perf_counter() - start)
        return CommandResult(
            ok=False,
            command=cmd,
            stdout="",
            stderr="",
            return_code=-9,
            elapsed_seconds=elapsed,
            error=f"Command timed out after {timeout_seconds}s",
        )


@st.cache_data
def detect_hardware_capabilities() -> dict[str, Any]:
    info = {
        "cuda": False,
        "qnn": False,
        "directml": False,
        "onnx_providers": [],
    }
    try:
        import torch
        info["cuda"] = bool(torch.cuda.is_available())
    except Exception:
        pass

    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        info["onnx_providers"] = providers
        info["qnn"] = "QNNExecutionProvider" in providers
        info["directml"] = "DmlExecutionProvider" in providers
    except Exception:
        pass
    return info

@st.cache_resource
def load_face_model(accelerator: str = "auto"):
    """Load CavaFace model for face recognition."""
    try:
        import torch
        from pipeline import build_dml_session, build_qnn_session, load_local_model, resolve_torch_device

        model = load_local_model()
        backend = "cpu"
        model_device: object = torch.device("cpu")

        if accelerator == "npu":
            onnx_cache = Path(os.getenv("LOCALAPPDATA", ".")) / "cavaface_qnn.onnx"
            session = build_qnn_session(model, onnx_cache)
            return session, "onnxruntime", "npu-qnn"

        if accelerator == "directml":
            onnx_cache = Path(os.getenv("LOCALAPPDATA", ".")) / "cavaface_dml.onnx"
            session = build_dml_session(model, onnx_cache)
            return session, "onnxruntime", "directml"

        if accelerator == "auto":
            try:
                onnx_cache = Path(os.getenv("LOCALAPPDATA", ".")) / "cavaface_qnn.onnx"
                session = build_qnn_session(model, onnx_cache)
                return session, "onnxruntime", "npu-qnn"
            except Exception:
                pass

        model_device, backend = resolve_torch_device(accelerator)
        model = model.to(model_device)
        return model, model_device, backend
    except Exception as e:
        st.error(f"Failed to load face model: {e}")
        return None, None, None


@st.cache_resource
def load_whisper_model(model_size: str = "base"):
    """Load Whisper model for transcription."""
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        return model
    except Exception as e:
        st.error(f"Failed to load Whisper: {e}")
        return None


def transcribe_video(whisper_model, video_path: str) -> str:
    """Transcribe audio from video."""
    try:
        segments, info = whisper_model.transcribe(
            video_path,
            beam_size=5,
            language="en",
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        lines = [seg.text.strip() for seg in segments]
        return " ".join(lines)
    except Exception as e:
        return f"Transcription error: {e}"


def extract_names_from_transcript(text: str, llm_provider: str = "ollama",
                                   api_base: str = "http://localhost:11434",
                                   model: str = "llama3.2:3b") -> list[str]:
    """Extract names from transcript using LLM or regex fallback."""
    import re
    import urllib.request

    # Regex fallback
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

    # Try LLM if available
    if llm_provider == "ollama":
        try:
            prompt = (
                "You are a name extractor. Read the following transcript and return ONLY "
                "a comma-separated list of full names of people who introduce themselves. "
                f"Transcript:\n{text}\n\nNames:"
            )
            payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
            req = urllib.request.Request(
                f"{api_base.rstrip('/')}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
            raw = data.get("response", "").strip()
            llm_names = [n.strip().title() for n in raw.split(",") if n.strip()]
            llm_names = [n for n in llm_names if 2 <= len(n) <= 50]
            for n in llm_names:
                if n not in names:
                    names.append(n)
        except Exception:
            pass  # Use regex fallback

    return names


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec) + 1e-12
    return vec / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)))


def load_face_db(db_path: Path):
    """Load face embeddings database."""
    if not db_path.exists():
        return [], np.empty((0, 512), dtype=np.float32)
    data = np.load(db_path, allow_pickle=False)
    return data["names"].tolist(), data["embeddings"].astype(np.float32)


def save_face_db(db_path: Path, names: list, embeddings: np.ndarray):
    """Save face embeddings database."""
    np.savez(db_path, names=np.asarray(names, dtype=np.str_), embeddings=embeddings.astype(np.float32))


def enroll_face_identity(db_path: Path, name: str, embedding: np.ndarray) -> tuple[bool, str]:
    """Merge-enroll one identity embedding into DB (centroid per person)."""
    clean = name.strip().title()
    if not clean:
        return False, "Name is empty."
    if embedding.size == 0:
        return False, "Embedding is empty."

    emb = l2_normalize(np.asarray(embedding, dtype=np.float32).reshape(-1))
    names, embeddings = load_face_db(db_path)
    grouped: dict[str, list[np.ndarray]] = {}

    if len(names) > 0 and embeddings.size > 0:
        for n, e in zip(names, embeddings):
            grouped.setdefault(str(n), []).append(np.asarray(e, dtype=np.float32))

    grouped.setdefault(clean, []).append(emb)

    merged_names: list[str] = []
    merged_embeddings: list[np.ndarray] = []
    for person, embs in grouped.items():
        centroid = l2_normalize(np.mean(np.vstack(embs), axis=0).astype(np.float32))
        merged_names.append(person)
        merged_embeddings.append(centroid)

    save_face_db(db_path, merged_names, np.vstack(merged_embeddings))
    return True, f"Saved '{clean}' to face DB."


def process_face_video_inline(
    video_path: str,
    db_path: str,
    accelerator: str,
    threshold: float,
    second_best_margin: float = 0.06,
    face_margin: float = 0.25,
    max_frames: int = 0,
) -> tuple[str | None, dict[str, Any]]:
    """Run face recognition and return annotated MP4 for Streamlit playback."""
    try:
        import cv2
        from pipeline import (
            best_match as pipeline_best_match,
            expand_to_square_with_margin,
            face_bgr_to_model_input,
            run_embedding_inference,
        )
    except Exception as e:
        return None, {"ok": False, "error": f"Face inline dependencies unavailable: {e}"}

    face_model, model_device, backend = load_face_model(accelerator)
    if face_model is None:
        return None, {"ok": False, "error": "Face model failed to load."}

    names, embeddings = load_face_db(Path(db_path))
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if detector.empty():
        return None, {"ok": False, "error": "Failed to load Haar cascade detector."}

    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        return None, {"ok": False, "error": f"Cannot open video: {video_path}"}

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1 or fps > 120:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        return None, {"ok": False, "error": "Invalid video dimensions."}

    out_path = str(Path(tempfile.mkdtemp()) / "face_inline_output.mp4")
    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )

    processed = 0
    known_hits: dict[str, int] = {}
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))

        for (x, y, w, h) in faces:
            crop = expand_to_square_with_margin(frame, x, y, w, h, face_margin)
            if crop.size == 0:
                continue
            inp = face_bgr_to_model_input(crop)
            emb = run_embedding_inference(face_model, inp, model_device)
            label, score = pipeline_best_match(names, embeddings, emb, threshold, second_best_margin)
            is_unknown = label == "Unknown" or score < threshold
            color = (0, 0, 255) if is_unknown else (0, 255, 0)
            if not is_unknown:
                known_hits[label] = known_hits.get(label, 0) + 1

            y0, y1 = max(0, y), min(frame.shape[0], y + h)
            x0, x1 = max(0, x), min(frame.shape[1], x + w)
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
            cv2.putText(
                frame,
                f"{label} ({score:.2f})" if score >= 0 else label,
                (x0, max(20, y0 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

        writer.write(frame)
        processed += 1
        if max_frames > 0 and processed >= max_frames:
            break

    cap.release()
    writer.release()

    return out_path, {
        "ok": True,
        "backend": backend,
        "frames": processed,
        "fps": fps,
        "known_hits": known_hits,
    }


def mux_audio_from_source(video_no_audio: str, source_video: str) -> str:
    """Attach source audio track to rendered video using ffmpeg when available."""
    out_with_audio = str(Path(tempfile.mkdtemp()) / "face_inline_with_audio.mp4")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_no_audio,
        "-i",
        source_video,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        out_with_audio,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if proc.returncode == 0 and Path(out_with_audio).exists():
            return out_with_audio
    except Exception:
        pass
    return video_no_audio


def is_valid_video_file(path: str) -> bool:
    p = Path(path)
    return p.exists() and p.is_file() and p.stat().st_size > 2048


def transcode_to_mp4_if_needed(input_path: str) -> str:
    """Best-effort conversion to browser-friendly MP4/H264."""
    in_path = Path(input_path)
    if in_path.suffix.lower() == ".mp4":
        return str(in_path)
    out_path = str(Path(tempfile.mkdtemp()) / f"{in_path.stem}_web.mp4")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(in_path),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        out_path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode == 0 and is_valid_video_file(out_path):
            return out_path
    except Exception:
        pass
    return str(in_path)


def _extract_audio_wav_for_whisper(video_path: str) -> str | None:
    """
    Extract a stable mono/16k WAV for Whisper when container demux fails.
    Returns WAV path on success, else None.
    """
    out_path = str(Path(tempfile.mkdtemp()) / "whisper_audio.wav")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        out_path,
    ]
    try:
        res = run_command(cmd, PROJECT_ROOT, timeout_seconds=180)
        if res.ok and Path(out_path).exists() and Path(out_path).stat().st_size > 1024:
            return out_path
    except Exception:
        pass
    return None


def _video_has_audio_stream(video_path: str) -> bool:
    """Best-effort check for at least one audio stream in container."""
    try:
        import av  # type: ignore
        with av.open(video_path) as container:
            return len(container.streams.audio) > 0
    except Exception:
        return True  # unknown; let Whisper attempt decode


def _decode_audio_with_av_for_whisper(video_path: str) -> np.ndarray | None:
    """
    Decode audio with PyAV and return mono 16k float32 samples in [-1, 1].
    This avoids depending on system ffmpeg binary for fallback.
    """
    try:
        import av  # type: ignore
    except Exception:
        return None

    try:
        container = av.open(video_path)
    except Exception:
        return None

    try:
        if len(container.streams.audio) == 0:
            return None
        audio_stream = container.streams.audio[0]
        resampler = av.audio.resampler.AudioResampler(
            format="s16",
            layout="mono",
            rate=16000,
        )
        chunks: list[np.ndarray] = []
        for packet in container.demux(audio_stream):
            for frame in packet.decode():
                out = resampler.resample(frame)
                frames = out if isinstance(out, list) else [out]
                for rf in frames:
                    arr = rf.to_ndarray()
                    if arr is None:
                        continue
                    arr = np.asarray(arr)
                    if arr.ndim > 1:
                        arr = arr.reshape(-1)
                    chunks.append(arr.astype(np.float32) / 32768.0)
        if not chunks:
            return None
        audio = np.concatenate(chunks).astype(np.float32)
        if audio.size == 0:
            return None
        return audio
    except Exception:
        return None
    finally:
        try:
            container.close()
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# REAL-TIME STREAMING WORKERS  (follows multiprocess_pipeline.py architecture)
# ──────────────────────────────────────────────────────────────────────────────

def _regex_names_quick(text: str) -> list[str]:
    """Regex name extraction – fast fallback used inside streaming threads."""
    patterns = [
        r"(?:I'm|I am|my name is|this is|call me|name's)\s+([A-Z][a-z]+(?: [A-Z][a-z]+)?)",
        r"(?:Hi|Hello|Hey),?\s+(?:I'm|I am)\s+([A-Z][a-z]+(?: [A-Z][a-z]+)?)",
    ]
    found: list[str] = []
    for pat in patterns:
        for m in _re.finditer(pat, text, _re.IGNORECASE):
            n = m.group(1).strip().title()
            if n not in found:
                found.append(n)
    return found


def _name_hints_from_text(text: str) -> list[str]:
    """
    Build name hints from both regex and optional LLM extraction.
    Keeps streaming robust by tolerating LLM/network failures.
    """
    hints = _regex_names_quick(text)
    try:
        for n in extract_names_from_transcript(text):
            if n not in hints:
                hints.append(n)
    except Exception:
        pass
    return hints


def _debug_names(stage: str, hints: list[str], extra: str = "") -> None:
    """Compact debug logger for transcript/name-hint flow."""
    hint_txt = ", ".join(hints) if hints else "<none>"
    suffix = f" | {extra}" if extra else ""
    print(f"[NamesDebug] {stage}: {hint_txt}{suffix}", flush=True)


def _whisper_stream_thread(
    stream_key: str,
    video_path: str,
    whisper_model: Any,        # pre-loaded by main thread — NO re-initialisation here
    name_q: "_std_queue.Queue[str | None]",
) -> None:
    """
    Whisper sub-thread (mirrors whisper_process in multiprocess_pipeline.py).
    Starts AFTER face thread signals first frame is ready.
    Puts name strings to name_q; puts None sentinel when done.
    Receives a pre-loaded WhisperModel so it starts transcribing immediately.
    """
    try:
        if whisper_model is None:
            print("[Whisper] No model available — skipping transcription.", flush=True)
            name_q.put(None)
            return

        # ── Check disk/memory cache first — never transcribe the same video twice ──
        cached = _load_transcript_cache(video_path)
        if cached:
            print("[Whisper] Cache hit — skipping transcription.", flush=True)
            with _FACE_STREAMS_LOCK:
                if stream_key in _FACE_STREAMS:
                    _FACE_STREAMS[stream_key]["transcript"] = cached
            cached_hints = _name_hints_from_text(cached)
            _debug_names("Whisper cache hints", cached_hints, f"stream={stream_key[:18]}")
            for n in cached_hints:
                name_q.put(n)
            name_q.put(None)
            return

        if not _video_has_audio_stream(video_path):
            print("[Whisper] No audio stream in video — skipping transcription.", flush=True)
            with _FACE_STREAMS_LOCK:
                if stream_key in _FACE_STREAMS:
                    _FACE_STREAMS[stream_key]["transcript"] = ""
            name_q.put(None)
            return

        print("[Whisper] Starting transcription...", flush=True)
        try:
            segments, info = whisper_model.transcribe(
                video_path,
                beam_size=5,
                language="en",
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
                no_repeat_ngram_size=4,
            )
        except Exception as direct_err:
            print(f"[Whisper] Direct video decode failed: {direct_err}", flush=True)
            audio_arr = _decode_audio_with_av_for_whisper(video_path)
            if audio_arr is not None:
                print("[Whisper] Retrying transcription from PyAV-decoded audio...", flush=True)
                segments, info = whisper_model.transcribe(
                    audio_arr,
                    beam_size=5,
                    language="en",
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500),
                    no_repeat_ngram_size=4,
                )
                # Success via PyAV fallback
                print(f"[Whisper] Language: {info.language} ({info.language_probability:.2f})", flush=True)
            else:
                wav_path = _extract_audio_wav_for_whisper(video_path)
                if not wav_path:
                    print("[Whisper] Audio fallback unavailable (PyAV+ffmpeg failed). Skipping transcription.", flush=True)
                    with _FACE_STREAMS_LOCK:
                        if stream_key in _FACE_STREAMS:
                            _FACE_STREAMS[stream_key]["transcript"] = ""
                    name_q.put(None)
                    return
                print("[Whisper] Retrying transcription from extracted WAV...", flush=True)
                segments, info = whisper_model.transcribe(
                    wav_path,
                    beam_size=5,
                    language="en",
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500),
                    no_repeat_ngram_size=4,
                )
        print(f"[Whisper] Language: {info.language} ({info.language_probability:.2f})", flush=True)

        full_text: list[str] = []
        buffer: list[str] = []
        FLUSH_EVERY = 5  # same batch size as multiprocess_pipeline.py

        for seg in segments:
            text = seg.text.strip()
            print(f"[Whisper] [{seg.start:.1f}s->{seg.end:.1f}s] {text}", flush=True)
            full_text.append(text)
            buffer.append(text)
            if len(buffer) >= FLUSH_EVERY:
                chunk_hints = _regex_names_quick(" ".join(buffer))
                _debug_names("Whisper chunk hints", chunk_hints, f"stream={stream_key[:18]}")
                for n in chunk_hints:
                    print(f"[Whisper->Face] Sending name: {n}", flush=True)
                    name_q.put(n)
                buffer.clear()

        if buffer:
            tail_hints = _regex_names_quick(" ".join(buffer))
            _debug_names("Whisper tail hints", tail_hints, f"stream={stream_key[:18]}")
            for n in tail_hints:
                name_q.put(n)

        transcript = " ".join(full_text)

        # Final pass with regex + LLM over full transcript for better hints.
        final_hints = _name_hints_from_text(transcript)
        _debug_names("Whisper final hints", final_hints, f"stream={stream_key[:18]}")
        for n in final_hints:
            name_q.put(n)

        # Save to disk cache so this video is never transcribed again
        _save_transcript_cache(video_path, transcript)

        with _FACE_STREAMS_LOCK:
            if stream_key in _FACE_STREAMS:
                _FACE_STREAMS[stream_key]["transcript"] = transcript

        name_q.put(None)  # sentinel: whisper is done
        print("[Whisper] Done.", flush=True)

    except Exception as e:
        import traceback
        print(f"[Whisper] ERROR: {e}", flush=True)
        traceback.print_exc()
        name_q.put(None)


def _face_stream_thread(
    stream_key: str,
    video_path: str,
    face_model: Any,          # pre-loaded by main thread — no init delay
    model_device: Any,
    backend: str,
    whisper_model: Any,       # pre-loaded WhisperModel — no re-init in thread
    threshold: float,
    face_margin: float,
    face_sys_path: str,
) -> None:
    """
    Face recognition streaming thread (mirrors face_process in multiprocess_pipeline.py).

    The face model is pre-loaded by the MAIN Streamlit thread (via @st.cache_resource)
    and passed in — so this thread never triggers model initialisation loops.

    Sequence:
      1. Import cv2 + pipeline helpers (fast, no model download)
      2. Open video, read first frame, signal Whisper to start
      3. For every subsequent frame:
         - Drain name_q for live suggestions
         - Haar-detect faces
         - Run embedding + best_match
         - Draw bounding box + label + HUD overlay
         - Encode JPEG and put into frame_q
         - Pace to real FPS with time.perf_counter()
      4. Put None sentinel into frame_q when video ends
    """
    if face_sys_path not in sys.path:
        sys.path.insert(0, face_sys_path)

    try:
        import cv2  # type: ignore
        from pipeline import (  # type: ignore
            best_match as _best_match,
            expand_to_square_with_margin as _expand,
            face_bgr_to_model_input as _to_input,
            run_embedding_inference as _run_emb,
        )
    except Exception as e:
        with _FACE_STREAMS_LOCK:
            if stream_key in _FACE_STREAMS:
                _FACE_STREAMS[stream_key]["status"] = f"error: import failed - {e}"
        return

    # ── Load face DB ──────────────────────────────────────────────────────────
    db_path = Path(face_sys_path) / "embeddings_db.npz"
    names, embeddings = load_face_db(db_path)

    # ── Haar cascade detector ─────────────────────────────────────────────────
    try:
        import cv2
        detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if detector.empty():
            raise RuntimeError("Empty Haar cascade")
    except Exception as e:
        with _FACE_STREAMS_LOCK:
            if stream_key in _FACE_STREAMS:
                _FACE_STREAMS[stream_key]["status"] = f"error: detector - {e}"
        return

    # ── Open video (force FFMPEG backend — avoids Windows MSMF grab errors) ───
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        with _FACE_STREAMS_LOCK:
            if stream_key in _FACE_STREAMS:
                _FACE_STREAMS[stream_key]["status"] = "error: cannot open video"
        return

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    if fps <= 1 or fps > 120:
        fps = 25.0
    target_interval = 1.0 / fps

    # Retrieve shared queues
    with _FACE_STREAMS_LOCK:
        info = _FACE_STREAMS.get(stream_key, {})
        frame_q: "_std_queue.Queue[bytes | None]" = info.get("frame_q")  # type: ignore
        name_q: "_std_queue.Queue[str | None]" = info.get("name_q")     # type: ignore

    if frame_q is None or name_q is None:
        cap.release()
        return

    # ── READ FIRST FRAME then signal Whisper to start ─────────────────────────
    # (exact same pattern as multiprocess_pipeline.py: face shows frame THEN whisper starts)
    ok, frame = cap.read()
    if ok:
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        try:
            frame_q.put_nowait(buf.tobytes())
        except _std_queue.Full:
            pass

        print("[FaceStream] First frame ready. Launching Whisper...", flush=True)
        whisper_t = threading.Thread(
            target=_whisper_stream_thread,
            args=(stream_key, video_path, whisper_model, name_q),  # pass pre-loaded model
            daemon=True,
        )
        whisper_t.start()
    else:
        name_q.put(None)  # signal anyway so pipeline doesn't hang

    with _FACE_STREAMS_LOCK:
        if stream_key in _FACE_STREAMS:
            _FACE_STREAMS[stream_key]["status"] = "streaming"

    # ── MAIN FRAME LOOP ───────────────────────────────────────────────────────
    suggested_names: list[str] = []
    whisper_done = False
    frames_sent = 0

    try:
        while True:
            with _FACE_STREAMS_LOCK:
                if _FACE_STREAMS.get(stream_key, {}).get("stop"):
                    break

            frame_t0 = time.perf_counter()

            # Drain incoming names from Whisper (non-blocking, per multiprocess_pipeline.py)
            while True:
                try:
                    item = name_q.get_nowait()
                    if item is None:
                        whisper_done = True
                        print("[FaceStream] Whisper complete.", flush=True)
                    elif item not in suggested_names:
                        suggested_names.append(item)
                        with _FACE_STREAMS_LOCK:
                            if stream_key in _FACE_STREAMS:
                                _FACE_STREAMS[stream_key]["names"] = list(suggested_names)
                        print(f"[FaceStream] New name hint: {item}", flush=True)
                except _std_queue.Empty:
                    break

            ok, frame = cap.read()
            if not ok:
                break

            # ── Detect + annotate ──────────────────────────────────────────────
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))

            with _FACE_STREAMS_LOCK:
                override_name = str(_FACE_STREAMS.get(stream_key, {}).get("unknown_override_name", "")).strip()

            needs_unknown_prompt = False

            for (x, y, w, h) in faces:
                crop = _expand(frame, x, y, w, h, face_margin)
                if crop.size == 0:
                    continue
                emb = None
                try:
                    inp = _to_input(crop)
                    emb = _run_emb(face_model, inp, model_device)
                    label, score = _best_match(names, embeddings, emb, threshold, 0.06)
                except Exception as fe:
                    label, score = "Unknown", 0.0
                    print(f"[FaceStream] inference error: {fe}", flush=True)

                is_unknown = label == "Unknown" or score < threshold
                color = (0, 0, 255) if is_unknown else (0, 255, 0)
                label_to_draw = label
                if is_unknown:
                    if override_name:
                        label_to_draw = f"{override_name} (?)"
                        color = (0, 165, 255)  # orange: provisional name from prompt/hints
                    else:
                        # Fully automatic mode: if transcript/LLM already has a name hint,
                        # assign and enroll immediately without user interaction.
                        auto_name = (suggested_names[-1] if suggested_names else "").strip()
                        if auto_name and emb is not None:
                            _debug_names(
                                "Face auto-assign candidate",
                                [auto_name],
                                f"stream={stream_key[:18]} score={score:.3f}",
                            )
                            label_to_draw = f"{auto_name} (?)"
                            color = (0, 165, 255)
                            try:
                                with _FACE_STREAMS_LOCK:
                                    st_info = _FACE_STREAMS.get(stream_key, {})
                                    auto_done = set(st_info.get("auto_enrolled_names", []))
                                if auto_name not in auto_done:
                                    ok, msg = enroll_face_identity(db_path, auto_name, emb)
                                    print(f"[FaceStream] Auto-enroll '{auto_name}': {msg}", flush=True)
                                    if ok:
                                        # Refresh in-thread DB view so matching can improve immediately.
                                        names, embeddings = load_face_db(db_path)
                                        with _FACE_STREAMS_LOCK:
                                            if stream_key in _FACE_STREAMS:
                                                _FACE_STREAMS[stream_key]["unknown_override_name"] = auto_name
                                                _FACE_STREAMS[stream_key]["unknown_suggested_name"] = auto_name
                                                prev = set(_FACE_STREAMS[stream_key].get("auto_enrolled_names", []))
                                                prev.add(auto_name)
                                                _FACE_STREAMS[stream_key]["auto_enrolled_names"] = list(prev)
                            except Exception as ae:
                                print(f"[FaceStream] Auto-enroll error: {ae}", flush=True)
                            needs_unknown_prompt = False
                        else:
                            if frames_sent % 60 == 0:
                                _debug_names(
                                    "Face unknown without hint",
                                    suggested_names,
                                    f"stream={stream_key[:18]}",
                                )
                            needs_unknown_prompt = True
                            with _FACE_STREAMS_LOCK:
                                if stream_key in _FACE_STREAMS:
                                    if suggested_names:
                                        # Always track latest hint (not just first-ever one)
                                        _FACE_STREAMS[stream_key]["unknown_suggested_name"] = suggested_names[-1]
                                    _FACE_STREAMS[stream_key]["unknown_embedding"] = emb.astype(np.float32).tolist()

                h_img, w_img = frame.shape[:2]
                cx, cy = x + w / 2, y + h / 2
                side = max(w, h) * (1 + face_margin)
                x0 = int(max(0, cx - side / 2))
                y0 = int(max(0, cy - side / 2))
                x1 = int(min(w_img, cx + side / 2))
                y1 = int(min(h_img, cy + side / 2))
                cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
                cv2.putText(
                    frame,
                    f"{label_to_draw} ({score:.2f})" if (not is_unknown and score >= 0) else label_to_draw,
                    (x0, max(20, y0 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA,
                )

            # ── HUD overlay (same style as multiprocess_pipeline.py) ──────────
            w_status = "Done" if whisper_done else "Transcribing..."
            cv2.putText(frame, f"Whisper: {w_status}",
                        (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Backend: {backend}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
            names_txt = ", ".join(suggested_names) if suggested_names else "detecting..."
            cv2.putText(frame, f"Name hints: {names_txt}",
                        (10, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1, cv2.LINE_AA)

            with _FACE_STREAMS_LOCK:
                if stream_key in _FACE_STREAMS:
                    _FACE_STREAMS[stream_key]["unknown_detected"] = bool(needs_unknown_prompt)
                    if not needs_unknown_prompt:
                        _FACE_STREAMS[stream_key]["unknown_embedding"] = None

            # ── Encode + enqueue (drop if full to stay real-time) ───────────────
            # Keep OpenCV frame in BGR for imencode; converting to RGB here
            # causes red/blue channel swap in the rendered JPEG.
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
            try:
                frame_q.put_nowait(buf.tobytes())
                frames_sent += 1
                # Publish frame count so UI can show progress
                if frames_sent % 30 == 0:
                    with _FACE_STREAMS_LOCK:
                        if stream_key in _FACE_STREAMS:
                            _FACE_STREAMS[stream_key]["frames_sent"] = frames_sent
            except _std_queue.Full:
                pass  # drop frame — maintain real-time pace

            # ── Real-time FPS pacing (identical to multiprocess_pipeline.py) ────
            elapsed = time.perf_counter() - frame_t0
            delay = target_interval - elapsed
            if delay > 0:
                time.sleep(delay)

    except Exception as loop_err:
        import traceback
        print(f"[FaceStream] LOOP ERROR: {loop_err}", flush=True)
        traceback.print_exc()
        with _FACE_STREAMS_LOCK:
            if stream_key in _FACE_STREAMS:
                _FACE_STREAMS[stream_key]["status"] = f"error: {loop_err}"
        frame_q.put(None)
        cap.release()
        return

    cap.release()

    with _FACE_STREAMS_LOCK:
        if stream_key in _FACE_STREAMS:
            _FACE_STREAMS[stream_key]["status"] = "done"
            _FACE_STREAMS[stream_key]["names"] = list(suggested_names)
            _FACE_STREAMS[stream_key]["frames_sent"] = frames_sent

    frame_q.put(None)  # sentinel: stream ended
    print(f"[FaceStream] Done. {frames_sent} frames sent.", flush=True)


def _start_face_stream(
    stream_key: str,
    video_path: str,
    accelerator: str,
    threshold: float,
    face_margin: float,
    whisper_size: str,
) -> None:
    """
    Start the face+whisper streaming thread.

    Guards:
    - No-op if status is 'starting' or 'streaming' (already live).
    - No-op if status is 'done' or starts with 'error' (finished or failed).
      The user must click Reset to force a restart.
    - Face model is loaded HERE in the main Streamlit thread via @st.cache_resource,
      so the background thread never triggers model-init loops.
    """
    # ── ATOMIC guard + reservation ──────────────────────────────────────────
    # Both the guard check AND the initial "starting" reservation happen inside
    # a single lock acquisition.  Without this, two concurrent Streamlit reruns
    # (at ~25 fps) both pass the guard before either writes "starting", spawning
    # duplicate threads that cause the endless restart loop.
    frame_q: _std_queue.Queue = _std_queue.Queue(maxsize=90)
    name_q: _std_queue.Queue = _std_queue.Queue()
    with _FACE_STREAMS_LOCK:
        existing = _FACE_STREAMS.get(stream_key)
        if existing:
            st_val = existing.get("status", "")
            if st_val in ("starting", "streaming", "done") or st_val.startswith("error"):
                return
        # Reserve the slot immediately — any concurrent call will now hit the guard
        _FACE_STREAMS[stream_key] = {
            "status": "starting",
            "frame_q": frame_q,
            "name_q": name_q,
            "names": [],
            "transcript": "",
            "backend": "loading...",
            "stop": False,
            "frames_sent": 0,
            "unknown_detected": False,
            "unknown_suggested_name": "",
            "unknown_override_name": "",
            "unknown_embedding": None,
            "auto_enrolled_names": [],
        }

    # ── Load BOTH models in the main thread (cached via @st.cache_resource) ───
    face_model, model_device, backend = load_face_model(accelerator)
    if face_model is None:
        with _FACE_STREAMS_LOCK:
            _FACE_STREAMS[stream_key]["status"] = "error: face model unavailable"
            _FACE_STREAMS[stream_key]["backend"] = "unavailable"
        return

    whisper_model = load_whisper_model(whisper_size)  # also cached, may be None

    with _FACE_STREAMS_LOCK:
        _FACE_STREAMS[stream_key]["backend"] = backend or "cpu"

    t = threading.Thread(
        target=_face_stream_thread,
        args=(
            stream_key, video_path,
            face_model, model_device, backend or "cpu",
            whisper_model,            # pre-loaded — no WhisperModel() in thread
            threshold, face_margin,
            str(PROJECT_ROOT / "FaceRecon"),
        ),
        daemon=True,
    )
    t.start()


# keep legacy function name so old callers still compile
@st.cache_resource
def get_smart_lens_runtime() -> dict[str, Any]:
    return {"tasks": {}, "lock": threading.Lock()}


def _smart_lens_worker(runtime: dict[str, Any], run_key: str, video_path: str, accelerator: str, whisper_size: str) -> None:
    try:
        out_video, meta = process_face_video_inline(
            video_path=video_path,
            db_path=str(PROJECT_ROOT / "FaceRecon" / "embeddings_db.npz"),
            accelerator=accelerator,
            threshold=0.5,
            second_best_margin=0.06,
            face_margin=0.25,
            max_frames=0,
        )
        if out_video and meta.get("ok"):
            out_video = mux_audio_from_source(out_video, video_path)
            if not is_valid_video_file(out_video):
                out_video = ""
        else:
            out_video = ""

        transcript = ""
        names: list[str] = []
        try:
            whisper = load_whisper_model(whisper_size)
            transcript = transcribe_video(whisper, video_path) if whisper else ""
            names = extract_names_from_transcript(transcript) if transcript else []
        except Exception:
            pass

        with runtime["lock"]:
            runtime["tasks"][run_key] = {
                "status": "done",
                "face_meta": meta,
                "annotated_video": out_video,
                "transcript": transcript,
                "names": names,
            }
    except Exception as e:
        with runtime["lock"]:
            runtime["tasks"][run_key] = {
                "status": "error",
                "error": str(e),
                "face_meta": {},
                "annotated_video": "",
                "transcript": "",
                "names": [],
            }


def ensure_smart_lens_task(runtime: dict[str, Any], run_key: str, video_path: str, accelerator: str, whisper_size: str) -> dict[str, Any]:
    with runtime["lock"]:
        existing = runtime["tasks"].get(run_key)
        if existing:
            return dict(existing)
        runtime["tasks"][run_key] = {
            "status": "running",
            "face_meta": {},
            "annotated_video": "",
            "transcript": "",
            "names": [],
        }
    t = threading.Thread(
        target=_smart_lens_worker,
        args=(runtime, run_key, video_path, accelerator, whisper_size),
        daemon=True,
    )
    t.start()
    return {"status": "running", "face_meta": {}, "annotated_video": "", "transcript": "", "names": []}


def extract_frame_from_video(video_path: str, second: float):
    try:
        import cv2
        from PIL import Image as PILImage

        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, second) * 1000.0)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return None
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return PILImage.fromarray(frame_rgb)
    except Exception:
        return None


def _ensure_drawable_canvas_streamlit_compat() -> None:
    """
    Compatibility shim for streamlit-drawable-canvas on newer Streamlit versions.
    streamlit-drawable-canvas expects streamlit.elements.image.image_to_url with
    an old signature. Streamlit>=1.54 moved/changed this API.
    """
    try:
        import streamlit.elements.image as st_image_mod
        if hasattr(st_image_mod, "image_to_url"):
            return

        from streamlit.elements.lib.image_utils import image_to_url as _new_image_to_url
        from streamlit.elements.lib.layout_utils import LayoutConfig

        def _compat_image_to_url(image, width, clamp, channels, output_format, image_id):
            layout = LayoutConfig(width=width)
            return _new_image_to_url(
                image=image,
                layout_config=layout,
                clamp=clamp,
                channels=channels,
                output_format=output_format,
                image_id=image_id,
            )

        st_image_mod.image_to_url = _compat_image_to_url  # type: ignore[attr-defined]
    except Exception:
        # If patching fails, canvas will raise its own error message in UI.
        pass


def stream_face_video_live(
    video_path: str,
    db_path: str,
    accelerator: str,
    threshold: float,
    frame_skip: int,
    loop_count: int,
    render_width: int = 960,
) -> dict[str, Any]:
    """Stream annotated face-recognition frames in-app."""
    try:
        import cv2
        from pipeline import (
            best_match as pipeline_best_match,
            expand_to_square_with_margin,
            face_bgr_to_model_input,
            run_embedding_inference,
        )
    except Exception as e:
        return {"ok": False, "error": f"Streaming dependencies unavailable: {e}"}

    face_model, model_device, backend = load_face_model(accelerator)
    if face_model is None:
        return {"ok": False, "error": "Face model failed to load."}

    names, embeddings = load_face_db(Path(db_path))
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if detector.empty():
        return {"ok": False, "error": "Failed to load Haar cascade detector."}

    container = st.empty()
    stat = st.empty()
    known_hits: dict[str, int] = {}
    total_rendered = 0
    loops_done = 0

    while loops_done < max(1, loop_count):
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            return {"ok": False, "error": f"Cannot open video: {video_path}"}
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 1.0 or fps > 120.0:
            fps = 25.0
        target_interval = (1.0 / fps) * max(1, int(frame_skip))

        frame_idx = 0
        while True:
            frame_t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if frame_skip > 1 and (frame_idx % frame_skip) != 0:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))
            for (x, y, w, h) in faces:
                crop = expand_to_square_with_margin(frame, x, y, w, h, 0.25)
                if crop.size == 0:
                    continue
                inp = face_bgr_to_model_input(crop)
                emb = run_embedding_inference(face_model, inp, model_device)
                label, score = pipeline_best_match(names, embeddings, emb, threshold, 0.06)
                is_unknown = label == "Unknown" or score < threshold
                color = (0, 0, 255) if is_unknown else (0, 255, 0)
                if not is_unknown:
                    known_hits[label] = known_hits.get(label, 0) + 1

                y0, y1 = max(0, y), min(frame.shape[0], y + h)
                x0, x1 = max(0, x), min(frame.shape[1], x + w)
                cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
                cv2.putText(
                    frame,
                    f"{label} ({score:.2f})" if score >= 0 else label,
                    (x0, max(20, y0 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            container.image(frame_rgb, channels="RGB", width="stretch")
            stat.caption(f"Backend: {backend} | loop {loops_done + 1}/{loop_count} | frame {frame_idx}")
            total_rendered += 1
            # Keep playback near real-time speed instead of rushing frames.
            elapsed = time.perf_counter() - frame_t0
            delay = target_interval - elapsed
            if delay > 0:
                time.sleep(delay)

        cap.release()
        loops_done += 1

    return {"ok": True, "backend": backend, "frames": total_rendered, "known_hits": known_hits}


def get_video_duration_seconds(video_path: str) -> float:
    try:
        import cv2

        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            return 0.0
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        cap.release()
        if fps <= 0 or frames <= 0:
            return 0.0
        return max(0.0, frames / fps)
    except Exception:
        return 0.0


def smart_lens_play_panel():
    """
    Real-time Smart Lens panel.

    Architecture (mirrors multiprocess_pipeline.py):
      - st.video(original)   → browser plays audio natively at real speed
      - _face_stream_thread  → face detect + HUD on every frame, real-time paced
      - _whisper_stream_thread → starts after first frame; names stream to face thread
      - st.empty().image()   → shows latest annotated frame; auto-refreshes via st.rerun()
      - Pause mode           → drag-box on paused frame → Scene Insight summary
    """
    st.markdown("""
    <div class="feature-card">
        <div class="feature-title">Smart Lens Play</div>
        <p class="muted">
            Upload a video to start. Audio plays instantly. Face ID + Whisper run in
            parallel in the background — face boxes appear live on the stream below.
            Toggle <b>Pause mode</b> to draw a region and get a Scene Insight summary.
        </p>
    </div>
    """, unsafe_allow_html=True)

    video_path = st.session_state.get("shared_video_path", "")
    if not video_path or not Path(video_path).exists():
        st.warning("Upload one shared video at the top to begin.")
        return

    # ── Settings row ─────────────────────────────────────────────────────────
    hw = detect_hardware_capabilities()
    accel_options = (["npu", "auto", "directml", "cpu"] if hw.get("qnn")
                     else ["auto", "directml", "cpu"] if hw.get("directml")
                     else ["auto", "cpu"])
    c1, c2, c3, c4 = st.columns(4)
    accelerator  = c1.selectbox("Face Accelerator", accel_options, index=0, key="smart_face_accel")
    whisper_size = c2.selectbox("Whisper Size", ["tiny", "base", "small", "medium"], index=0, key="smart_whisper")
    summary_mode = c3.selectbox("Scene Mode", ["clip", "blip", "hybrid"], index=0, key="smart_scene_mode")
    use_npu      = c4.checkbox("NPU for Scene", value=True, key="smart_scene_npu")

    # Use a content-derived key so the same video always maps to one stream,
    # even if temp paths/session values vary across reruns.
    stream_key = f"video:{_video_cache_key(video_path)}"

    # ── Read current stream state BEFORE starting (snapshot under lock) ───────
    with _FACE_STREAMS_LOCK:
        snap = dict(_FACE_STREAMS.get(stream_key, {}))
    status = snap.get("status", "")

    # ── Reset button (shows when stream is done or errored) ───────────────────
    if status == "done" or status.startswith("error"):
        col_rst, _ = st.columns([1, 3])
        if col_rst.button("Reset Stream", key="smart_reset_stream"):
            with _FACE_STREAMS_LOCK:
                _FACE_STREAMS.pop(stream_key, None)
            st.rerun()
        if status.startswith("error"):
            st.error(f"Stream error: {status}")
        else:
            st.success("Stream finished. Click Reset to replay.")

    # ── Kick off background stream (no-op if running/done/error) ─────────────
    _start_face_stream(stream_key, video_path, accelerator, 0.50, 0.25, whisper_size)

    # Re-read state after potential start
    with _FACE_STREAMS_LOCK:
        snap = dict(_FACE_STREAMS.get(stream_key, {}))

    status     = snap.get("status", "starting")
    backend    = snap.get("backend", "loading...")
    names      = snap.get("names", [])
    transcript = snap.get("transcript", "")
    frame_q: "_std_queue.Queue | None" = snap.get("frame_q")
    unknown_detected = bool(snap.get("unknown_detected", False))
    unknown_suggested_name = str(snap.get("unknown_suggested_name", "")).strip()
    unknown_override_name = str(snap.get("unknown_override_name", "")).strip()
    unknown_embedding = snap.get("unknown_embedding")

    # ── Read pause-mode flag BEFORE any st.rerun() call ──────────────────────
    is_paused = st.session_state.get("smart_pause_mode", False)

    frames_sent = snap.get("frames_sent", 0)

    # ── STATUS BAR ────────────────────────────────────────────────────────────
    status_badge = {
        "starting":  "🔄 Initialising...",
        "streaming": "🟢 Live",
        "done":      "✅ Done",
    }.get(status, f"⚠️ {status[:60]}")

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Stream", status_badge)
    sc2.metric("Backend", backend)
    sc3.metric("Frames", frames_sent)
    sc4.metric("Name Hints", len(names))

    if names:
        st.info("Transcript name hints: " + ", ".join(names))

    # ── UNKNOWN FACE AUTO-ASSIGN FROM HINTS ───────────────────────────────────
    # If transcript/LLM already suggested a name (e.g., "Kate"), auto-use it and
    # persist to DB with no manual interaction.
    if unknown_detected:
        # Prefer latest live hint from transcript/LLM.
        auto_name = ((names[-1] if names else unknown_suggested_name) or "").strip()
        if auto_name and unknown_embedding is not None:
            try:
                db_path = PROJECT_ROOT / "FaceRecon" / "embeddings_db.npz"
                emb_arr = np.asarray(unknown_embedding, dtype=np.float32)
                ok, msg = enroll_face_identity(db_path, auto_name, emb_arr)
                if ok:
                    with _FACE_STREAMS_LOCK:
                        if stream_key in _FACE_STREAMS:
                            _FACE_STREAMS[stream_key]["unknown_override_name"] = auto_name
                            _FACE_STREAMS[stream_key]["unknown_suggested_name"] = auto_name
                            _FACE_STREAMS[stream_key]["unknown_embedding"] = None
                    st.info(f"Auto-assigned unknown face as '{auto_name}' and saved to DB.")
                else:
                    st.warning(msg)
            except Exception as e:
                st.error(f"Failed to auto-save unknown identity: {e}")
            st.rerun()
        elif not auto_name:
            st.warning("Unknown face detected. Waiting for transcript/LLM name hint to auto-assign.")

    # ── LIVE ANNOTATED STREAM (PRIMARY — full width, face boxes visible) ──────
    st.markdown("#### Live Face ID Stream  *(face boxes drawn in real-time)*")
    frame_placeholder = st.empty()
    latest: bytes | None = None

    had_new_frame = False
    if frame_q is not None:
        # Drain to most recent frame (skip stale buffered frames)
        drained = 0
        while drained < 10:
            try:
                item = frame_q.get_nowait()
                if item is None:
                    break  # stream ended sentinel
                latest = item
                drained += 1
            except _std_queue.Empty:
                break

        if latest is not None:
            had_new_frame = True
            st.session_state["smart_latest_frame_jpg"] = latest
            # Avoid repainting identical frames; repaint churn is the main
            # source of visible flicker during rapid Streamlit reruns.
            frame_sig = hashlib.sha1(latest).hexdigest()
            if st.session_state.get("smart_last_frame_sig") != frame_sig:
                st.session_state["smart_last_frame_sig"] = frame_sig
                # Render via data URI to avoid Streamlit media-store churn
                # ("MediaFileHandler: Missing file ...jpg") during rapid reruns.
                frame_b64 = base64.b64encode(latest).decode("ascii")
                frame_placeholder.markdown(
                    f'<img src="data:image/jpeg;base64,{frame_b64}" style="width:100%; border-radius:8px;" />',
                    unsafe_allow_html=True,
                )
        elif status == "starting":
            frame_placeholder.info("🔄 Loading face model and opening video…  (first frame will appear shortly)")
        elif status == "streaming":
            # Keep last rendered frame visible instead of constantly replacing it
            # with info text, which causes additional flicker.
            if not st.session_state.get("smart_last_frame_sig"):
                frame_placeholder.info("🎞️ Waiting for next frame from background thread…")
        elif status == "done":
            frame_placeholder.success("✅ Stream complete — all frames processed.")
    else:
        frame_placeholder.info("Stream not yet started.")

    # ── LIVE CONTROLS (pause during active inference) ─────────────────────────
    ctl1, ctl2 = st.columns([1, 1])
    if not is_paused:
        if ctl1.button("Pause & Inspect Current Frame", key="smart_pause_now"):
            frozen = latest or st.session_state.get("smart_latest_frame_jpg")
            if frozen:
                st.session_state["smart_frozen_frame_jpg"] = frozen
                st.session_state["smart_pause_mode"] = True
                st.rerun()
            else:
                st.warning("No frame available yet. Wait for live stream frame then pause.")
    else:
        if ctl1.button("Resume Live Stream", key="smart_resume_live"):
            st.session_state["smart_pause_mode"] = False
            st.rerun()
    if st.session_state.get("smart_frozen_frame_jpg"):
        ctl2.caption("Frozen frame ready for Scene Insight below.")

    # ── AUTO-REFRESH LOOP (only while live and not in pause mode) ─────────────
    if status in ("starting", "streaming") and not is_paused:
        # Adaptive UI refresh: fast enough for live feel, slower when no new frame.
        refresh_s = 0.08 if (status == "streaming" and had_new_frame) else 0.14
        time.sleep(refresh_s)
        st.rerun()

    # ── AUDIO: original video in collapsible section ──────────────────────────
    with st.expander("Audio / Original Video", expanded=False):
        st.caption("Original video with sound. Face boxes appear in the stream above.")
        st.video(video_path)

    # ── TRANSCRIPT ────────────────────────────────────────────────────────────
    if transcript or status == "done":
        with st.expander("Transcript + Name Hints", expanded=False):
            st.text_area("Transcript", transcript, height=140, key="smart_transcript_view")
            if names:
                st.caption("Names: " + ", ".join(names))

    # ── PAUSE MODE ────────────────────────────────────────────────────────────
    pause_mode = st.toggle("Pause mode for Scene Insight", value=False, key="smart_pause_mode")
    if pause_mode:
        st.markdown("#### Draw a box on the paused frame → auto Scene Insight")
        frame_image = None
        frozen = st.session_state.get("smart_frozen_frame_jpg")
        if frozen:
            try:
                from PIL import Image as PILImage
                frame_image = PILImage.open(io.BytesIO(frozen)).convert("RGB")
                st.caption("Using frozen live frame captured at pause.")
            except Exception:
                frame_image = None

        if frame_image is None:
            duration = get_video_duration_seconds(video_path)
            if duration <= 0:
                st.warning("Could not detect video duration.")
                return
            frame_sec = st.slider(
                "Pause at (seconds)", 0.0, float(duration), 0.0, step=0.2, key="smart_pause_second"
            )
            frame_image = extract_frame_from_video(video_path, frame_sec)
            if frame_image is None:
                st.warning("Could not load frame at that timestamp.")
                return

        bbox = None
        try:
            _ensure_drawable_canvas_streamlit_compat()
            from streamlit_drawable_canvas import st_canvas  # type: ignore

            fw, fh = frame_image.size
            # Cap canvas height to keep it on-screen
            max_h = 520
            if fh > max_h:
                scale = max_h / fh
                fw, fh = int(fw * scale), max_h
                frame_image = frame_image.resize((fw, fh))

            canvas = st_canvas(
                fill_color="rgba(124, 58, 237, 0.12)",
                stroke_width=2,
                stroke_color="#8b5cf6",
                background_image=frame_image,
                update_streamlit=True,
                height=fh,
                width=fw,
                drawing_mode="rect",
                key=f"smart_canvas_{'frozen' if frozen else int(st.session_state.get('smart_pause_second', 0.0) * 10)}",
            )
            if canvas.json_data and canvas.json_data.get("objects"):
                obj = canvas.json_data["objects"][-1]
                x1 = int(max(0, obj.get("left", 0)))
                y1 = int(max(0, obj.get("top", 0)))
                x2 = int(min(fw, x1 + obj.get("width", 0)))
                y2 = int(min(fh, y1 + obj.get("height", 0)))
                if x2 > x1 + 5 and y2 > y1 + 5:
                    bbox = (x1, y1, x2, y2)
                    st.caption(f"Region  x1={x1}  y1={y1}  x2={x2}  y2={y2}")
        except ImportError:
            st.error("streamlit-drawable-canvas not installed. Run: pip install streamlit-drawable-canvas")
        except Exception as exc:
            st.error(f"Canvas error: {exc}")

        if bbox is None:
            st.info("Draw a rectangle on the frame above to trigger Scene Insight.")
        else:
            with st.spinner("Running Scene Insight on selected region..."):
                engine = load_summarizer_engine(summary_mode, use_npu)
                if engine and engine.is_ready:
                    result = engine.summarize(frame_image, bbox)
                    st.markdown(f"**Summary:** {result.get('summary', '-')}")
                    st.caption(
                        f"Model: {result.get('model_used', '-')} | "
                        f"Time: {result.get('total_ms', 0)} ms"
                    )
                else:
                    st.error("Scene Insight model is not ready.")


def warmup_all_models(face_accelerator: str, whisper_size: str, use_npu: bool) -> dict[str, Any]:
    """Warm up all major models so first interaction is faster."""
    status: dict[str, Any] = {}
    face_model, _, face_backend = load_face_model(face_accelerator)
    status["face_ready"] = face_model is not None
    status["face_backend"] = face_backend or "unavailable"

    whisper = load_whisper_model(whisper_size)
    status["whisper_ready"] = whisper is not None

    clip_engine = load_summarizer_engine("clip", use_npu)
    status["clip_ready"] = bool(clip_engine and clip_engine.is_ready)

    blip_engine = load_summarizer_engine("blip", use_npu)
    status["blip_ready"] = bool(blip_engine and blip_engine.is_ready)

    status["all_ready"] = bool(
        status["face_ready"] and status["whisper_ready"] and status["clip_ready"] and status["blip_ready"]
    )
    return status


def preload_face_npu_on_startup(hw: dict[str, Any]) -> dict[str, Any]:
    """Preload Face ID model as soon as app starts, preferring NPU."""
    preferred = "npu" if hw.get("qnn") else "auto"
    model, _, backend = load_face_model(preferred)
    return {
        "requested": preferred,
        "ok": model is not None,
        "backend": backend or "unavailable",
    }


def face_recognition_panel():
    """Face ID lens interface panel."""
    st.markdown("""
    <div class="feature-card">
        <div class="feature-title">Face ID Lens</div>
        <p class="muted">
            Identify known people, suggest names from spoken intros, and keep recognition on-device.
            Hardware path prioritizes NPU/QNN or GPU when available.
        </p>
    </div>
    """, unsafe_allow_html=True)

    hw = detect_hardware_capabilities()
    face_accelerators = ["auto", "cpu"]
    if hw.get("cuda"):
        face_accelerators.insert(1, "gpu")
    if hw.get("qnn"):
        face_accelerators.insert(1, "npu")
    if hw.get("directml"):
        face_accelerators.insert(1, "directml")

    with st.expander("Model Settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            default_face_idx = face_accelerators.index("npu") if "npu" in face_accelerators else 0
            accelerator = st.selectbox(
                "Face Model Accelerator",
                face_accelerators,
                index=default_face_idx,
                key="face_accelerator"
            )
        with col2:
            whisper_size = st.selectbox(
                "Whisper Model Size",
                ["tiny", "base", "small", "medium"],
                index=1,
                key="whisper_size"
            )

    st.subheader("Lens Feed")
    video_path = st.session_state.get("shared_video_path", "")
    if video_path and Path(video_path).exists():
        st.caption(f"Using shared video: {video_path}")
        st.video(video_path)
    else:
        st.warning("No shared video loaded. Use the global uploader at top of the app.")

    col1, col2 = st.columns(2)
    with col1:
        db_path = st.text_input(
            "Embeddings Database",
            value=str(PROJECT_ROOT / "FaceRecon" / "embeddings_db.npz"),
            key="face_db_path"
        )
    with col2:
        threshold = st.slider("Match Threshold", 0.3, 0.8, 0.5, 0.05, key="face_threshold")

    with st.expander("LLM Settings (for name extraction)", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            llm_provider = st.selectbox("LLM Provider", ["ollama", "openai"], key="face_llm_provider")
            llm_api_base = st.text_input("API Base", "http://localhost:11434", key="face_llm_base")
        with col2:
            llm_model = st.text_input("Model", "llama3.2:3b", key="face_llm_model")

    st.subheader("Lens Actions")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Transcribe Video", type="primary", use_container_width=True, disabled=not video_path):
            with st.spinner("Loading Whisper model..."):
                whisper = load_whisper_model(whisper_size)

            if whisper:
                with st.spinner("Transcribing..."):
                    transcript = transcribe_video(whisper, video_path)
                    st.session_state.face_transcript = transcript

                st.success("Transcription complete!")
                st.text_area("Transcript", transcript, height=150)

                names = extract_names_from_transcript(
                    transcript, llm_provider, llm_api_base, llm_model
                )
                if names:
                    st.info(f"Detected names: {', '.join(names)}")
                    st.session_state.face_suggested_names = names

    with col2:
        frame_skip = st.slider("Frame Skip (performance)", 1, 6, 2, 1, key="face_stream_skip")
        loop_count = st.slider("Streaming Loops", 1, 10, 1, 1, key="face_stream_loops")
        if st.button("Start Face ID Lens", type="primary", use_container_width=True, disabled=not video_path):
            with st.spinner("Streaming face recognition in-app..."):
                meta = stream_face_video_live(
                    video_path=video_path,
                    db_path=db_path,
                    accelerator=accelerator,
                    threshold=threshold,
                    frame_skip=int(frame_skip),
                    loop_count=int(loop_count),
                )

            if meta.get("ok"):
                st.success(
                    f"Face ID stream complete ({meta.get('frames', 0)} frames, backend={meta.get('backend')})"
                )
                known_hits = meta.get("known_hits", {})
                if known_hits:
                    st.write("**Recognized identities:**")
                    for name, hits in sorted(known_hits.items(), key=lambda x: x[1], reverse=True):
                        st.write(f"- {name}: {hits} hits")
            else:
                st.error(meta.get("error", "Inline face processing failed"))

    with col3:
        if st.button("View Database", type="primary", use_container_width=True):
            db = Path(db_path)
            if db.exists():
                names, embeddings = load_face_db(db)
                if names:
                    st.write(f"**{len(names)} identities enrolled:**")
                    for i, name in enumerate(names):
                        st.write(f"- {name}")
                else:
                    st.info("Database is empty")
            else:
                st.warning("Database file not found")

    # Show transcript if available
    if st.session_state.get("face_transcript"):
        with st.expander("Last Transcript", expanded=False):
            st.text_area("", st.session_state.face_transcript, height=200, key="face_transcript_display")


# ============================================================
# IMAGE SUMMARIZER MODULE
# ============================================================

@st.cache_resource
def load_summarizer_engine(mode: str = "clip", use_npu: bool = True):
    """Load the image summarization engine."""
    try:
        from engine.inference_engine import InferenceEngine
        engine = InferenceEngine(mode=mode, use_npu=use_npu)
        return engine
    except Exception as e:
        st.error(f"Failed to load summarizer engine: {e}")
        return None


def image_summarizer_panel():
    """Scene insight lens interface panel."""
    # Styled header with glow effect
    st.markdown("""
    <div class="image-summarizer-header">
        <div class="glow-orb"></div>
        <div class="feature-card image-card">
            <div class="feature-title glow-text">Scene Insight Lens</div>
            <p class="muted">
                Understand what you are looking at through a selected region.
                Uses CLIP (fast) or BLIP (detailed) for on-device scene insight.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Settings
    col1, col2, col3 = st.columns(3)
    with col1:
        mode = st.selectbox(
            "Summarization Mode",
            ["clip", "blip", "hybrid"],
            index=0,
            help="CLIP: Fast zero-shot classification | BLIP: Natural language | Hybrid: Both",
            key="summarizer_mode"
        )
    with col2:
        use_npu = st.checkbox("Prefer NPU (QNN) Acceleration", value=True, key="summarizer_npu")
    with col3:
        st.write("")  # Spacer

    # Image input
    st.subheader("Frame Input")
    image_source = st.radio(
        "Source",
        ["Upload Image", "Image Path", "Shared Video Frame"],
        horizontal=True,
        key="summarizer_image_source"
    )

    image = None
    image_path = None

    if image_source == "Upload Image":
        uploaded = st.file_uploader(
            "Upload an image",
            type=["png", "jpg", "jpeg", "bmp", "webp"],
            key="summarizer_upload"
        )
        if uploaded:
            from PIL import Image as PILImage
            image = PILImage.open(uploaded).convert("RGB")
            image_path = uploaded.name
    elif image_source == "Image Path":
        path_input = st.text_input(
            "Image file path",
            value=st.session_state.get("summarizer_image_path", ""),
            placeholder="/path/to/image.jpg",
            key="summarizer_path_input"
        )
        if path_input and Path(path_input).exists():
            from PIL import Image as PILImage
            image = PILImage.open(path_input).convert("RGB")
            image_path = path_input
            st.session_state.summarizer_image_path = path_input
        elif path_input:
            st.warning("Image file not found")
    else:
        shared_video = st.session_state.get("shared_video_path", "")
        if not shared_video or not Path(shared_video).exists():
            st.warning("No shared video loaded. Use the global uploader at top of the app.")
        else:
            frame_sec = st.slider("Frame Timestamp (seconds)", 0.0, 300.0, 0.0, 0.5, key="summarizer_frame_second")
            frame_image = extract_frame_from_video(shared_video, float(frame_sec))
            if frame_image is not None:
                image = frame_image.convert("RGB")
                image_path = f"{shared_video} @ {frame_sec:.1f}s"
            else:
                st.warning("Could not read frame from shared video at this timestamp.")

    if image:
        # Display image
        col1, col2 = st.columns([2, 1])

        with col1:
            st.image(image, caption=f"Loaded: {image_path}", use_container_width=True)

            # Region selection
            st.subheader("Select Region")
            img_width, img_height = image.size

            c1, c2 = st.columns(2)
            with c1:
                x1 = st.slider("X1 (left)", 0, img_width - 10, 0, key="region_x1")
                y1 = st.slider("Y1 (top)", 0, img_height - 10, 0, key="region_y1")
            with c2:
                x2 = st.slider("X2 (right)", 10, img_width, img_width, key="region_x2")
                y2 = st.slider("Y2 (bottom)", 10, img_height, img_height, key="region_y2")

            # Ensure valid bbox
            x1, x2 = min(x1, x2 - 10), max(x1 + 10, x2)
            y1, y2 = min(y1, y2 - 10), max(y1 + 10, y2)

        with col2:
            # Show selected region preview
            cropped = image.crop((x1, y1, x2, y2))
            st.image(cropped, caption="Selected Region", use_container_width=True)
            st.caption(f"Region: ({x1}, {y1}) to ({x2}, {y2})")

        # Summarize button
        if st.button("Analyze View Region", type="primary", use_container_width=True):
            with st.spinner(f"Loading {mode.upper()} model..."):
                engine = load_summarizer_engine(mode, use_npu)

            if engine and engine.is_ready:
                with st.spinner("Generating summary..."):
                    bbox = (x1, y1, x2, y2)
                    result = engine.summarize(image, bbox)

                # Display results with dark theme styling
                st.markdown("""
                <div class="summary-result">
                    <h3>Summary</h3>
                    <p style="font-size: 1.1rem; line-height: 1.6; color: #f0f4ff;">
                """ + result["summary"] + """
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Metrics
                c1, c2, c3 = st.columns(3)
                c1.metric("Model Used", result["model_used"])
                c2.metric("Inference Time", f"{result['total_ms']:.1f} ms")
                c3.metric("Active Models", ", ".join(engine.get_active_models()))

                # Details with black background
                if result.get("details"):
                    details = result["details"]

                    # Build HTML for detailed analysis
                    html_content = '<div class="detailed-analysis"><h4>Detailed Analysis</h4>'

                    if "scenes" in details:
                        html_content += '<div class="analysis-section"><div class="section-title">Top Scenes</div>'
                        for label, score in details["scenes"][:5]:
                            html_content += f'<div class="item">{label}: <span class="score">{score:.3f}</span></div>'
                        html_content += '</div>'

                    if "attributes" in details:
                        html_content += '<div class="analysis-section"><div class="section-title">Attributes</div>'
                        for label, score in details["attributes"][:5]:
                            html_content += f'<div class="item">{label}: <span class="score">{score:.3f}</span></div>'
                        html_content += '</div>'

                    if "actions" in details:
                        html_content += '<div class="analysis-section"><div class="section-title">Actions</div>'
                        for label, score in details["actions"][:3]:
                            html_content += f'<div class="item">{label}: <span class="score">{score:.3f}</span></div>'
                        html_content += '</div>'

                    html_content += '</div>'
                    st.markdown(html_content, unsafe_allow_html=True)
            else:
                st.error("Failed to initialize summarization engine")
    else:
        st.info("Upload or select an image to begin")


# ============================================================
# OBJECT DETECTION MODULE (EXISTING EXTRACTOR)
# ============================================================

def object_memory_ingest_panel():
    """Find-My-Object memory ingestion panel."""
    st.markdown("""
    <div class="feature-card">
        <div class="feature-title">Find-My-Object Memory (Store Only)</div>
        <p class="muted">
            Scan the video feed and store object memory. This panel only ingests and saves memory.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.info("This uses the existing extractor pipeline. Configure settings below.")

    video_path = st.session_state.get("shared_video_path", "")
    if video_path and Path(video_path).exists():
        st.caption(f"Using shared video: {video_path}")
    else:
        st.warning("No shared video loaded. Use the global uploader at top of the app.")

    col1, col2 = st.columns(2)
    with col1:
        out_dir = st.text_input(
            "Output Directory",
            value=str(PROJECT_ROOT / "demo_data"),
            key="extractor_out_dir"
        )
    with col2:
        runtime = st.selectbox("Runtime", ["auto", "qnn", "cpu"], key="extractor_runtime")

    with st.expander("Detection Settings"):
        c1, c2, c3, c4 = st.columns(4)
        conf = c1.slider("Confidence", 0.1, 0.95, 0.25, key="extractor_conf")
        iou = c2.slider("NMS IoU", 0.1, 0.95, 0.60, key="extractor_iou")
        sample_fps = c3.slider("Sample FPS", 1.0, 4.0, 3.0, key="extractor_fps")
        input_size = c4.selectbox("Input Size", [640, 960], index=1, key="extractor_input_size")

    if st.button("Store Object Memory", type="primary", use_container_width=True, disabled=not video_path):
        if not Path(video_path).exists():
            st.error("Video file not found")
        else:
            st.session_state.extractor_video_path = video_path
            st.session_state.extractor_out_dir_state = out_dir

            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "extractor" / "run_ingest.py"),
                "--video", video_path,
                "--out", out_dir,
                "--runtime", runtime,
                "--conf", str(conf),
                "--iou", str(iou),
                "--sample-fps", str(sample_fps),
                "--input-size", str(input_size),
                "--reset-output",
            ]

            with st.spinner("Building object memory..."):
                result = run_command(cmd, cwd=PROJECT_ROOT / "extractor", timeout_seconds=600)

            if result.ok:
                st.success(f"Memory stored in {result.elapsed_seconds:.1f}s")
                st.text_area("Ingestion Output", result.stdout, height=200)
            else:
                st.error(result.error or "Memory ingestion failed")
                if result.stderr.strip():
                    st.text_area("stderr", result.stderr, height=180)


def object_memory_prompt_panel():
    """Prompt-only panel for stored object memory."""
    st.markdown("""
    <div class="feature-card">
        <div class="feature-title">Memory Prompts</div>
        <p class="muted">
            Ask natural prompts against already stored object memory.
            This panel does not ingest video; it only queries existing memory.
        </p>
    </div>
    """, unsafe_allow_html=True)

    video_path = st.text_input(
        "Stored Video Path",
        value=st.session_state.get("extractor_video_path", ""),
        placeholder="/path/to/video.mp4",
        key="extractor_video_query_input",
    )
    out_dir = st.text_input(
        "Stored Output Directory",
        value=st.session_state.get("extractor_out_dir_state", str(PROJECT_ROOT / "demo_data")),
        key="extractor_out_dir_query",
    )
    query = st.text_input(
        "Prompt",
        value="where is my black bottle",
        key="extractor_query_prompt",
    )

    if st.button("Run Memory Prompt", type="primary", use_container_width=True, disabled=not video_path):
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "extractor" / "run_query.py"),
            "--video", video_path,
            "--out", out_dir,
            "--json", "where_is",
            "--label", query,
        ]

        with st.spinner("Searching stored memory..."):
            result = run_command(cmd, cwd=PROJECT_ROOT / "extractor", timeout_seconds=120)

        if result.ok:
            try:
                payload = json.loads(result.stdout)
                if payload.get("found"):
                    st.success(payload.get("answer", "Found!"))

                    event = payload.get("event", {})
                    if event:
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Label", event.get("label", "-"))
                        c2.metric("Second", event.get("video_second", "-"))
                        c3.metric("Color", event.get("detected_color", "-"))

                        thumb = event.get("thumbnail_path")
                        if thumb and Path(thumb).exists():
                            st.image(thumb, caption="Detected Object", width=200)
                else:
                    st.warning(payload.get("answer", "Not found"))
            except json.JSONDecodeError:
                st.text(result.stdout)
        else:
            st.error(result.error or "Prompt query failed")
            if result.stderr.strip():
                st.text_area("stderr", result.stderr, height=180)


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    st.set_page_config(
        page_title="Aura Smart Specs Console",
        page_icon=":sparkles:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_styles()

    hw = detect_hardware_capabilities()

    if "startup_face_preload" not in st.session_state:
        with st.spinner("Preloading Face ID model (NPU preferred)..."):
            st.session_state.startup_face_preload = preload_face_npu_on_startup(hw)

    # Header
    st.markdown("""
    <div class="hero-card">
        <div class="hero-title">Aura Smart Specs Console</div>
        <p class="hero-sub">
            A smart-glasses style assistant for identity, scene understanding, and object recall.
            Powered by on-device inference with NPU/GPU acceleration when available.
        </p>
    </div>
    """, unsafe_allow_html=True)
    preload = st.session_state.get("startup_face_preload", {})
    if preload:
        state_text = "ready" if preload.get("ok") else "failed"
        st.caption(
            f"Startup Face preload: {state_text} | requested={preload.get('requested')} | backend={preload.get('backend')}"
        )

    with st.sidebar:
        st.header("Smart Specs Control")
        st.markdown("""
        **Lens Pack:**
        - **Face ID Lens**: CavaFace + Whisper
        - **Scene Insight Lens**: CLIP + BLIP
        - **Find-My-Object Lens**: YOLOv11 + Memory

        **Smart Specs Flow:**
        1. Capture a feed/frame
        2. Infer on edge accelerators
        3. Return a concise assistive answer
        """)

        st.divider()
        st.header("System Status")
        st.metric("CUDA (GPU)", "Yes" if hw.get("cuda") else "No")
        st.metric("QNN (NPU)", "Yes" if hw.get("qnn") else "No")
        st.metric("DirectML", "Yes" if hw.get("directml") else "No")
        providers = hw.get("onnx_providers", [])
        if providers:
            st.caption(f"ONNX providers: {', '.join(providers[:4])}")

        st.divider()
        st.caption("Acceleration policy: NPU/QNN -> GPU -> CPU fallback")
        st.caption("Qualcomm AI Hub × Edge AI Hackathon")

        st.divider()
        st.markdown("### Model Readiness")
        preload_face_accel = st.selectbox(
            "Preload Face Accelerator",
            ["auto", "npu", "gpu", "directml", "cpu"],
            index=1,
            key="preload_face_accel",
        )
        preload_whisper = st.selectbox(
            "Preload Whisper",
            ["tiny", "base", "small", "medium"],
            index=1,
            key="preload_whisper_size",
        )
        preload_npu = st.checkbox("Prefer NPU for summarizers", value=True, key="preload_npu")
        if st.button("Load All Models", use_container_width=True):
            with st.spinner("Warming up all models..."):
                st.session_state.model_warm_status = warmup_all_models(
                    preload_face_accel, preload_whisper, preload_npu
                )
        if st.session_state.get("model_warm_status"):
            ms = st.session_state.model_warm_status
            st.metric("Face", "Ready" if ms.get("face_ready") else "Not ready")
            st.metric("Whisper", "Ready" if ms.get("whisper_ready") else "Not ready")
            st.metric("CLIP", "Ready" if ms.get("clip_ready") else "Not ready")
            st.metric("BLIP", "Ready" if ms.get("blip_ready") else "Not ready")
            st.caption(f"Face backend: {ms.get('face_backend', '-')}")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Face ID Lens", "Ready")
    k2.metric("Scene Insight", "Ready")
    k3.metric("Object Recall", "Ready")
    k4.metric("Best Accelerator", "NPU" if hw.get("qnn") else ("GPU" if hw.get("cuda") else "CPU"))

    st.markdown("### Shared Video Source (All Lenses)")
    shared_upload = st.file_uploader(
        "Upload one video at start for Face ID, Scene Insight, and Object Memory",
        type=["mp4", "avi", "mov", "mkv"],
        key="shared_video_upload",
    )
    if shared_upload:
        # Guard: only write the temp file when the upload actually changes.
        # st.file_uploader returns the same object on every st.rerun(), so
        # without this check video_path gets a new temp path every 40 ms,
        # which changes the stream_key and restarts the face+whisper stream.
        upload_id = f"{shared_upload.name}::{shared_upload.size}"
        if st.session_state.get("_shared_upload_id") != upload_id:
            temp_dir = tempfile.mkdtemp()
            shared_path = os.path.join(temp_dir, shared_upload.name)
            with open(shared_path, "wb") as f:
                f.write(shared_upload.read())
            normalized_path = transcode_to_mp4_if_needed(shared_path)
            st.session_state.shared_video_path = normalized_path
            st.session_state.face_video_path = normalized_path
            st.session_state.extractor_video_path = normalized_path
            st.session_state["_shared_upload_id"] = upload_id

    if st.session_state.get("shared_video_path") and Path(st.session_state.shared_video_path).exists():
        st.caption(f"Active shared video: {st.session_state.shared_video_path}")

    st.markdown("### Smart Specs Experience")
    st.caption("Tab 1 runs Face ID + Scene Insight + Object Memory storage. Tab 2 is prompt-only over stored memory.")

    tab_live, tab_prompts = st.tabs(["Live Lenses + Store Memory", "Memory Prompts"])

    with tab_live:
        smart_lens_play_panel()
        st.divider()
        object_memory_ingest_panel()

    with tab_prompts:
        object_memory_prompt_panel()


if __name__ == "__main__":
    main()
