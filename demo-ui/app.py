from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTOR_DIR = PROJECT_ROOT / "extractor"
DEFAULT_OUT_DIR = PROJECT_ROOT / "demo_data"


@dataclass
class CommandResult:
    ok: bool
    command: list[str]
    stdout: str
    stderr: str
    return_code: int
    payload: Any | None = None
    error: str | None = None


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .main {
            background: linear-gradient(180deg, #0b1020 0%, #12192f 100%);
        }
        .stApp {
            color: #e8ecff;
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
            font-size: 1.15rem;
            font-weight: 700;
            margin-bottom: 6px;
        }
        .hero-sub {
            color: #bac8ff;
            margin-bottom: 0;
        }
        .event-card {
            border: 1px solid rgba(122, 162, 255, 0.25);
            border-radius: 12px;
            padding: 12px;
            margin: 8px 0;
            background: rgba(15, 22, 43, 0.6);
        }
        .event-title {
            font-size: 0.98rem;
            font-weight: 600;
        }
        .muted {
            color: #b8c2e8;
            font-size: 0.92rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _safe_json_parse(raw: str) -> Any | None:
    raw = raw.strip()
    if not raw:
        return None

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Some dependencies print logs before JSON; salvage from first JSON token.
    first_obj = raw.find("{")
    first_arr = raw.find("[")
    candidates = [x for x in [first_obj, first_arr] if x >= 0]
    if not candidates:
        return None
    start = min(candidates)
    snippet = raw[start:]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return None


def _run_command(
    cmd: list[str],
    cwd: Path,
    expect_json: bool = False,
) -> CommandResult:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )
    payload = _safe_json_parse(proc.stdout) if expect_json else None
    ok = proc.returncode == 0 and (payload is not None if expect_json else True)

    error = None
    if proc.returncode != 0:
        error = f"Command failed with exit code {proc.returncode}."
    elif expect_json and payload is None:
        error = "Command succeeded but output was not valid JSON."

    return CommandResult(
        ok=ok,
        command=cmd,
        stdout=proc.stdout,
        stderr=proc.stderr,
        return_code=proc.returncode,
        payload=payload,
        error=error,
    )


def _resolve_memory_path(out_dir: Path, video_path: str) -> Path:
    return out_dir / "memory" / f"{Path(video_path).stem}.events.jsonl"


def _event_label(event: dict[str, Any]) -> str:
    label = str(event.get("label", "unknown"))
    instance_id = str(event.get("instance_id", "")).strip()
    if instance_id:
        return f"{label} ({instance_id})"
    return label


def _show_event(event: dict[str, Any]) -> None:
    thumb = str(event.get("thumbnail_path", "")).strip()
    second = event.get("video_second", "?")
    context = str(event.get("context", "in_scene"))
    confidence = event.get("confidence", "?")

    st.markdown(
        f"""
        <div class="event-card">
          <div class="event-title">{_event_label(event)}</div>
          <div class="muted">second={second} • context={context} • confidence={confidence}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if thumb:
        thumb_path = Path(thumb)
        if thumb_path.exists():
            st.image(str(thumb_path), use_container_width=False, width=170)
        else:
            st.caption(f"Thumbnail missing: {thumb}")


def _build_ingest_command(
    python_exec: str,
    extractor_dir: Path,
    video_path: str,
    out_dir: Path,
    prefer_model: str,
    conf: float,
    iou: float,
    thumb_size: int,
    crop_padding: int,
    jpeg_quality: int,
    reset_output: bool,
    track_max_gap: int,
    track_iou_threshold: float,
    track_center_dist_ratio: float,
) -> list[str]:
    cmd = [
        python_exec,
        str(extractor_dir / "run_ingest.py"),
        "--video",
        video_path,
        "--out",
        str(out_dir),
        "--prefer-model",
        prefer_model,
        "--conf",
        str(conf),
        "--iou",
        str(iou),
        "--thumb-size",
        str(thumb_size),
        "--crop-padding",
        str(crop_padding),
        "--jpeg-quality",
        str(jpeg_quality),
        "--track-max-gap",
        str(track_max_gap),
        "--track-iou-threshold",
        str(track_iou_threshold),
        "--track-center-dist-ratio",
        str(track_center_dist_ratio),
    ]
    if reset_output:
        cmd.append("--reset-output")
    return cmd


def _build_query_command(
    python_exec: str,
    extractor_dir: Path,
    out_dir: Path,
    video_path: str,
    command: str,
    label_or_query: str | None = None,
    instance_id: str | None = None,
    limit: int | None = None,
) -> list[str]:
    cmd = [
        python_exec,
        str(extractor_dir / "run_query.py"),
        "--video",
        video_path,
        "--out",
        str(out_dir),
        "--json",
        command,
    ]
    if label_or_query is not None:
        cmd.extend(["--label", label_or_query])
    if instance_id:
        cmd.extend(["--instance-id", instance_id])
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    return cmd


def _validate_paths(python_exec: str, extractor_dir: Path) -> list[str]:
    problems: list[str] = []
    if not python_exec.strip():
        problems.append("Python executable is empty.")
    if not extractor_dir.exists():
        problems.append(f"Extractor directory does not exist: {extractor_dir}")
    if not (extractor_dir / "run_ingest.py").exists():
        problems.append("Missing extractor/run_ingest.py")
    if not (extractor_dir / "run_query.py").exists():
        problems.append("Missing extractor/run_query.py")
    return problems


def main() -> None:
    st.set_page_config(
        page_title="Aura Vision Demo Console",
        page_icon=":camera_with_flash:",
        layout="wide",
    )
    _inject_styles()

    st.markdown(
        """
        <div class="hero-card">
          <div class="hero-title">Aura Vision Demo Console</div>
          <p class="hero-sub">
            A presentation-ready UI wrapper for your existing <code>extractor</code> pipeline.
            No extractor code changes required.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "last_video_path" not in st.session_state:
        st.session_state.last_video_path = ""
    if "last_out_dir" not in st.session_state:
        st.session_state.last_out_dir = str(DEFAULT_OUT_DIR)
    if "last_answer" not in st.session_state:
        st.session_state.last_answer = ""
    if "last_event" not in st.session_state:
        st.session_state.last_event = None
    if "last_cmd" not in st.session_state:
        st.session_state.last_cmd = []

    with st.sidebar:
        st.header("Runtime Settings")
        python_exec = st.text_input("Python executable", value=sys.executable)
        extractor_dir_str = st.text_input(
            "Extractor directory",
            value=str(DEFAULT_EXTRACTOR_DIR),
        )
        out_dir_str = st.text_input(
            "Output directory",
            value=st.session_state.last_out_dir,
        )
        st.caption("This app only calls extractor scripts via subprocess.")

        extractor_dir = Path(extractor_dir_str).expanduser().resolve()
        out_dir = Path(out_dir_str).expanduser()
        st.session_state.last_out_dir = str(out_dir)

        if st.button("Validate setup", use_container_width=True):
            problems = _validate_paths(python_exec, extractor_dir)
            if problems:
                for p in problems:
                    st.error(p)
            else:
                st.success("Setup looks good.")

        st.divider()
        st.markdown("### Demo Tips")
        st.write("- Use a fixed test video for repeatable stage demos.")
        st.write("- Keep `--reset-output` enabled for clean reruns.")
        st.write("- Use natural queries like `where is my black bottle`.")

    tabs = st.tabs(["Ingest", "Ask", "Timeline", "Explore"])

    with tabs[0]:
        st.subheader("1) Ingest Video")
        video_path = st.text_input(
            "Video path",
            value=st.session_state.last_video_path,
            placeholder="/absolute/path/to/video.mp4",
        )

        c1, c2, c3, c4 = st.columns(4)
        prefer_model = c1.selectbox("Model", ["yolov11", "yolov8"], index=0)
        conf = c2.slider("Confidence", min_value=0.1, max_value=0.95, value=0.45, step=0.05)
        iou = c3.slider("NMS IoU", min_value=0.1, max_value=0.95, value=0.60, step=0.05)
        reset_output = c4.checkbox("Reset output", value=True)

        c5, c6, c7 = st.columns(3)
        thumb_size = c5.number_input("Thumb size", min_value=20, max_value=256, value=50, step=2)
        crop_padding = c6.number_input("Crop padding", min_value=0, max_value=64, value=8, step=1)
        jpeg_quality = c7.number_input("JPEG quality", min_value=40, max_value=100, value=75, step=1)

        c8, c9, c10 = st.columns(3)
        track_max_gap = c8.number_input("Track max gap (sec)", min_value=1, max_value=15, value=3, step=1)
        track_iou_threshold = c9.slider(
            "Track IoU threshold", min_value=0.05, max_value=0.9, value=0.20, step=0.05
        )
        track_center_dist_ratio = c10.slider(
            "Track center-dist ratio", min_value=0.02, max_value=0.8, value=0.12, step=0.02
        )

        if st.button("Run ingestion", type="primary", use_container_width=True):
            st.session_state.last_video_path = video_path.strip()
            problems = _validate_paths(python_exec, extractor_dir)
            if problems:
                for p in problems:
                    st.error(p)
            elif not video_path.strip():
                st.error("Provide a video path.")
            elif not Path(video_path).expanduser().exists():
                st.error(f"Video not found: {video_path}")
            else:
                cmd = _build_ingest_command(
                    python_exec=python_exec,
                    extractor_dir=extractor_dir,
                    video_path=video_path,
                    out_dir=out_dir,
                    prefer_model=prefer_model,
                    conf=conf,
                    iou=iou,
                    thumb_size=int(thumb_size),
                    crop_padding=int(crop_padding),
                    jpeg_quality=int(jpeg_quality),
                    reset_output=reset_output,
                    track_max_gap=int(track_max_gap),
                    track_iou_threshold=track_iou_threshold,
                    track_center_dist_ratio=track_center_dist_ratio,
                )
                with st.spinner("Running ingestion..."):
                    result = _run_command(cmd, cwd=extractor_dir)

                st.session_state.last_cmd = cmd
                if result.ok:
                    st.success("Ingestion completed.")
                    mem_path = _resolve_memory_path(out_dir, video_path)
                    st.info(f"Memory log: {mem_path}")
                else:
                    st.error(result.error or "Ingestion failed.")

                with st.expander("Command output", expanded=not result.ok):
                    st.code(" ".join(result.command))
                    if result.stdout.strip():
                        st.text_area("stdout", result.stdout, height=180)
                    if result.stderr.strip():
                        st.text_area("stderr", result.stderr, height=180)

    with tabs[1]:
        st.subheader("2) Ask Natural Query")
        query = st.text_input(
            "Question",
            value="where is my black bottle",
            help="Example: where is my black bottle",
        )
        instance_id = st.text_input(
            "Optional instance_id",
            value="",
            placeholder="bottle_001 (optional)",
        )

        if st.button("Find now", type="primary", use_container_width=True):
            problems = _validate_paths(python_exec, extractor_dir)
            if problems:
                for p in problems:
                    st.error(p)
            elif not st.session_state.last_video_path:
                st.error("Run ingestion first or set a video path in the Ingest tab.")
            else:
                cmd = _build_query_command(
                    python_exec=python_exec,
                    extractor_dir=extractor_dir,
                    out_dir=out_dir,
                    video_path=st.session_state.last_video_path,
                    command="where_is",
                    label_or_query=query,
                    instance_id=instance_id.strip() or None,
                )
                with st.spinner("Querying memory..."):
                    result = _run_command(cmd, cwd=extractor_dir, expect_json=True)

                st.session_state.last_cmd = cmd
                if not result.ok:
                    st.error(result.error or "Query failed.")
                    with st.expander("Command output", expanded=True):
                        st.code(" ".join(result.command))
                        if result.stdout.strip():
                            st.text_area("stdout", result.stdout, height=160)
                        if result.stderr.strip():
                            st.text_area("stderr", result.stderr, height=160)
                else:
                    payload = result.payload if isinstance(result.payload, dict) else {}
                    st.session_state.last_answer = str(payload.get("answer", ""))
                    st.session_state.last_event = payload.get("event")

                    if payload.get("found"):
                        st.success(st.session_state.last_answer)
                    else:
                        st.warning(st.session_state.last_answer or "No match found.")

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Canonical label", str(payload.get("canonical_label", "-")))
                    c2.metric("Target color", str(payload.get("target_color", "-")))
                    c3.metric("Resolved instance", str(payload.get("resolved_instance_id", "-")))

                    event = payload.get("event")
                    if isinstance(event, dict):
                        st.markdown("#### Matched Event")
                        _show_event(event)

    with tabs[2]:
        st.subheader("3) Timeline")
        timeline_query = st.text_input(
            "Timeline query",
            value="black bottle",
            help="Use label or natural phrase.",
        )
        timeline_limit = st.slider("Entries", min_value=1, max_value=25, value=8, step=1)
        timeline_instance = st.text_input("Optional instance_id filter", value="")

        if st.button("Load timeline", use_container_width=True):
            problems = _validate_paths(python_exec, extractor_dir)
            if problems:
                for p in problems:
                    st.error(p)
            elif not st.session_state.last_video_path:
                st.error("Set video path and ingest first.")
            else:
                cmd = _build_query_command(
                    python_exec=python_exec,
                    extractor_dir=extractor_dir,
                    out_dir=out_dir,
                    video_path=st.session_state.last_video_path,
                    command="timeline",
                    label_or_query=timeline_query,
                    instance_id=timeline_instance.strip() or None,
                    limit=int(timeline_limit),
                )
                with st.spinner("Loading timeline..."):
                    result = _run_command(cmd, cwd=extractor_dir, expect_json=True)

                st.session_state.last_cmd = cmd
                if not result.ok:
                    st.error(result.error or "Timeline query failed.")
                else:
                    rows = result.payload if isinstance(result.payload, list) else []
                    st.write(f"Found {len(rows)} timeline events.")
                    for idx, row in enumerate(rows, start=1):
                        st.markdown(f"##### #{idx}")
                        _show_event(row)

    with tabs[3]:
        st.subheader("4) Explore Labels and Instances")
        c1, c2 = st.columns(2)

        with c1:
            if st.button("List labels", use_container_width=True):
                if not st.session_state.last_video_path:
                    st.error("Set video path and ingest first.")
                else:
                    cmd = _build_query_command(
                        python_exec=python_exec,
                        extractor_dir=extractor_dir,
                        out_dir=out_dir,
                        video_path=st.session_state.last_video_path,
                        command="labels",
                    )
                    result = _run_command(cmd, cwd=extractor_dir, expect_json=True)
                    if result.ok and isinstance(result.payload, list):
                        st.dataframe({"labels": result.payload}, use_container_width=True, height=300)
                    else:
                        st.error(result.error or "Could not load labels.")

        with c2:
            instance_query = st.text_input(
                "Instance filter query",
                value="bottle",
                help="Label or phrase, e.g. bottle or black bottle.",
            )
            if st.button("List instances", use_container_width=True):
                if not st.session_state.last_video_path:
                    st.error("Set video path and ingest first.")
                else:
                    cmd = _build_query_command(
                        python_exec=python_exec,
                        extractor_dir=extractor_dir,
                        out_dir=out_dir,
                        video_path=st.session_state.last_video_path,
                        command="instances",
                        label_or_query=instance_query,
                    )
                    result = _run_command(cmd, cwd=extractor_dir, expect_json=True)
                    if result.ok and isinstance(result.payload, list):
                        st.dataframe(
                            {"instance_id": result.payload},
                            use_container_width=True,
                            height=300,
                        )
                    else:
                        st.error(result.error or "Could not load instances.")

        if st.session_state.last_cmd:
            with st.expander("Last command"):
                st.code(" ".join(st.session_state.last_cmd))


if __name__ == "__main__":
    main()
