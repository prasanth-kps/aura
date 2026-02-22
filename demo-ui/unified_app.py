"""
Unified Aura Demo Interface
============================
A single Streamlit interface combining:
1. Face Recognition (FaceRecon) - Video face recognition with Whisper transcription
2. Image Summarizer - Image region summarization using CLIP/BLIP
3. Object Detection (Extractor) - Video object detection and memory

Usage:
    streamlit run demo-ui/unified_app.py
"""

from __future__ import annotations

import json
import os
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

@st.cache_resource
def load_face_model(accelerator: str = "auto"):
    """Load CavaFace model for face recognition."""
    try:
        import torch
        from qai_hub_models.models.cavaface import Model as CavaFaceModel

        model = CavaFaceModel.from_pretrained()
        model.eval()

        # Resolve device
        if accelerator in {"auto", "gpu"}:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                model = model.to(device)
                return model, device, "cuda"

        device = torch.device("cpu")
        model = model.to(device)
        return model, device, "cpu"
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


def face_recognition_tab():
    """Face Recognition interface tab."""
    st.markdown("""
    <div class="feature-card">
        <div class="feature-title">Face Recognition</div>
        <p class="muted">
            Recognize faces in video with Whisper transcription for name extraction.
            Uses CavaFace embeddings for on-device face matching.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Model loading
    with st.expander("Model Settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            accelerator = st.selectbox(
                "Face Model Accelerator",
                ["auto", "cpu", "gpu"],
                index=0,
                key="face_accelerator"
            )
        with col2:
            whisper_size = st.selectbox(
                "Whisper Model Size",
                ["tiny", "base", "small", "medium"],
                index=1,
                key="whisper_size"
            )

    # Video input
    st.subheader("Video Input")
    video_source = st.radio(
        "Source",
        ["Upload Video", "Video Path", "Webcam"],
        horizontal=True,
        key="face_video_source"
    )

    video_path = None

    if video_source == "Upload Video":
        uploaded = st.file_uploader("Upload video file", type=["mp4", "avi", "mov", "mkv"], key="face_video_upload")
        if uploaded:
            # Save to temp file
            temp_dir = tempfile.mkdtemp()
            video_path = os.path.join(temp_dir, uploaded.name)
            with open(video_path, "wb") as f:
                f.write(uploaded.read())
            st.success(f"Video loaded: {uploaded.name}")

    elif video_source == "Video Path":
        video_path_input = st.text_input(
            "Video file path",
            value=st.session_state.get("face_video_path", ""),
            placeholder="/path/to/video.mp4",
            key="face_video_path_input"
        )
        if video_path_input and Path(video_path_input).exists():
            video_path = video_path_input
            st.session_state.face_video_path = video_path
        elif video_path_input:
            st.warning("Video file not found")

    else:  # Webcam
        st.info("Webcam mode will open a live preview window")

    # Database settings
    col1, col2 = st.columns(2)
    with col1:
        db_path = st.text_input(
            "Embeddings Database",
            value=str(PROJECT_ROOT / "FaceRecon" / "embeddings_db.npz"),
            key="face_db_path"
        )
    with col2:
        threshold = st.slider("Match Threshold", 0.3, 0.8, 0.5, 0.05, key="face_threshold")

    # LLM settings for name extraction
    with st.expander("LLM Settings (for name extraction)", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            llm_provider = st.selectbox("LLM Provider", ["ollama", "openai"], key="face_llm_provider")
            llm_api_base = st.text_input("API Base", "http://localhost:11434", key="face_llm_base")
        with col2:
            llm_model = st.text_input("Model", "llama3.2:3b", key="face_llm_model")

    # Actions
    st.subheader("Actions")
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

                # Extract names
                names = extract_names_from_transcript(
                    transcript, llm_provider, llm_api_base, llm_model
                )
                if names:
                    st.info(f"Detected names: {', '.join(names)}")
                    st.session_state.face_suggested_names = names

    with col2:
        if st.button("Run Face Recognition", type="primary", use_container_width=True, disabled=not video_path):
            # Run the CLI pipeline via subprocess
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "FaceRecon" / "pipeline.py"),
                "--video", video_path,
                "--db", db_path,
                "--mode", "camera",
                "--threshold", str(threshold),
                "--accelerator", accelerator,
            ]

            if st.session_state.get("face_transcript"):
                cmd.extend(["--intro-text", st.session_state.face_transcript[:1000]])

            st.info("Opening face recognition window... Press 'q' to quit.")
            st.code(" ".join(cmd))

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                st.success("Face recognition completed!")
            else:
                st.error(f"Error: {result.stderr}")

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


def image_summarizer_tab():
    """Image Summarizer interface tab."""
    # Styled header with glow effect
    st.markdown("""
    <div class="image-summarizer-header">
        <div class="glow-orb"></div>
        <div class="feature-card image-card">
            <div class="feature-title glow-text">Image Summarizer</div>
            <p class="muted">
                Summarize regions of images using CLIP (fast zero-shot) or BLIP (detailed captioning).
                Select a region and get AI-powered descriptions.
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
        use_npu = st.checkbox("Use NPU Acceleration", value=True, key="summarizer_npu")
    with col3:
        st.write("")  # Spacer

    # Image input
    st.subheader("Image Input")
    image_source = st.radio(
        "Source",
        ["Upload Image", "Image Path"],
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
    else:
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
        if st.button("Summarize Region", type="primary", use_container_width=True):
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

def object_detection_tab():
    """Object Detection interface tab - wrapper for existing extractor."""
    st.markdown("""
    <div class="feature-card">
        <div class="feature-title">Object Detection & Memory</div>
        <p class="muted">
            Process videos to detect and track objects. Query the memory to find objects
            by description (e.g., "where is my black bottle").
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Import from existing app
    st.info("This uses the existing extractor pipeline. Configure settings below.")

    # Video input
    video_path = st.text_input(
        "Video Path",
        value=st.session_state.get("extractor_video_path", ""),
        placeholder="/path/to/video.mp4",
        key="extractor_video_input"
    )

    col1, col2 = st.columns(2)
    with col1:
        out_dir = st.text_input(
            "Output Directory",
            value=str(PROJECT_ROOT / "demo_data"),
            key="extractor_out_dir"
        )
    with col2:
        runtime = st.selectbox("Runtime", ["auto", "qnn", "cpu"], key="extractor_runtime")

    # Detection settings
    with st.expander("Detection Settings"):
        c1, c2, c3, c4 = st.columns(4)
        conf = c1.slider("Confidence", 0.1, 0.95, 0.25, key="extractor_conf")
        iou = c2.slider("NMS IoU", 0.1, 0.95, 0.60, key="extractor_iou")
        sample_fps = c3.slider("Sample FPS", 1.0, 4.0, 3.0, key="extractor_fps")
        input_size = c4.selectbox("Input Size", [640, 960], index=1, key="extractor_input_size")

    # Actions
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Run Ingestion", type="primary", use_container_width=True, disabled=not video_path):
            if not Path(video_path).exists():
                st.error("Video file not found")
            else:
                st.session_state.extractor_video_path = video_path

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

                with st.spinner("Processing video..."):
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

                if result.returncode == 0:
                    st.success("Ingestion complete!")
                    st.text_area("Output", result.stdout, height=200)
                else:
                    st.error(f"Failed: {result.stderr}")

    with col2:
        query = st.text_input("Query", "where is my black bottle", key="extractor_query")
        if st.button("Search Memory", use_container_width=True, disabled=not video_path):
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "extractor" / "run_query.py"),
                "--video", video_path,
                "--out", out_dir,
                "--json", "where_is",
                "--label", query,
            ]

            with st.spinner("Searching..."):
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
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
                st.error(f"Query failed: {result.stderr}")


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    st.set_page_config(
        page_title="Aura - Unified AI Demo",
        page_icon=":sparkles:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_styles()

    # Header
    st.markdown("""
    <div class="hero-card">
        <div class="hero-title">Aura - Unified AI Demo</div>
        <p class="hero-sub">
            An integrated interface for face recognition, image summarization, and object detection.
            Powered by Qualcomm AI Hub models for on-device inference.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("About Aura")
        st.markdown("""
        **Components:**
        - **Face Recognition**: CavaFace + Whisper
        - **Image Summarizer**: CLIP + BLIP
        - **Object Detection**: YOLOv11 + Memory

        **Features:**
        - On-device inference (NPU/GPU/CPU)
        - Natural language queries
        - Real-time video processing
        """)

        st.divider()

        # System info
        st.header("System Status")

        try:
            import torch
            cuda_available = torch.cuda.is_available()
            st.metric("CUDA Available", "Yes" if cuda_available else "No")
        except ImportError:
            st.metric("PyTorch", "Not installed")

        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            qnn = "QNNExecutionProvider" in providers
            st.metric("QNN (NPU)", "Yes" if qnn else "No")
            st.caption(f"Providers: {', '.join(providers[:3])}")
        except ImportError:
            st.metric("ONNX Runtime", "Not installed")

        st.divider()
        st.caption("Qualcomm AI Hub × Edge AI Hackathon")

    # Main tabs
    tabs = st.tabs([
        "Face Recognition",
        "Image Summarizer",
        "Object Detection",
    ])

    with tabs[0]:
        face_recognition_tab()

    with tabs[1]:
        image_summarizer_tab()

    with tabs[2]:
        object_detection_tab()


if __name__ == "__main__":
    main()
