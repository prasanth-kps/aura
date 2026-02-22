from __future__ import annotations

import json
import re
import subprocess
import sys
import time
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
    elapsed_seconds: float
    timed_out: bool = False
    telemetry: dict[str, list[float]] | None = None
    payload: Any | None = None
    error: str | None = None


def _parse_ingest_summary(stdout: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if line.startswith("Processed seconds:"):
            try:
                out["processed_seconds"] = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.startswith("Processed samples:"):
            payload = line.split(":", 1)[1].strip()
            # format: "<count> @ <fps> FPS"
            count_part = payload.split("@", 1)[0].strip()
            try:
                out["processed_seconds"] = int(count_part)
            except ValueError:
                pass
            fps_match = re.search(r"@\s*([0-9.]+)\s*FPS", payload, re.IGNORECASE)
            if fps_match:
                try:
                    out["sample_fps"] = float(fps_match.group(1))
                except ValueError:
                    pass
        elif line.startswith("Events written:"):
            try:
                out["events_written"] = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.startswith("Frames written:"):
            try:
                out["frames_written"] = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.startswith("Captions written:"):
            try:
                out["captions_written"] = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.startswith("Detector input size:"):
            try:
                out["detector_input_size"] = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.startswith("Detector model/runtime:"):
            payload = line.split(":", 1)[1].strip()
            if "/" in payload:
                model, backend = [x.strip() for x in payload.split("/", 1)]
                out["detector_model"] = model
                out["detector_backend"] = backend
            else:
                out["detector_model_runtime"] = payload
    return out


def _ort_provider_info(python_exec: str, extractor_dir: Path) -> dict[str, Any]:
    script = (
        "import json; "
        "import onnxruntime as ort; "
        "providers = ort.get_available_providers(); "
        "print(json.dumps({'providers': providers, 'qnn_available': 'QNNExecutionProvider' in providers}))"
    )
    cmd = [python_exec, "-c", script]
    res = _run_command(cmd, cwd=extractor_dir, expect_json=True)
    if res.ok and isinstance(res.payload, dict):
        return dict(res.payload)
    return {"providers": [], "qnn_available": False, "error": res.error or res.stderr.strip()}


def _windows_npu_stats() -> dict[str, Any]:
    """
    Best-effort NPU utilization probe on Windows via perf counters.
    Returns:
      {'available': bool, 'avg_util_percent': float | None, 'max_util_percent': float | None, 'source': str}
    """
    ps_script = (
        "$ls = Get-Counter -ListSet *NPU* -ErrorAction SilentlyContinue; "
        "if (-not $ls) { "
        "  Write-Output '{\"available\":false,\"source\":\"no_npu_counter_set\"}'; exit 0 "
        "} "
        "$paths = @(); foreach ($set in $ls) { $paths += $set.Counter } "
        "$util = $paths | Where-Object { $_ -match 'Utilization Percentage' }; "
        "if (-not $util) { "
        "  Write-Output '{\"available\":false,\"source\":\"no_utilization_counter\"}'; exit 0 "
        "} "
        "$sample = Get-Counter -Counter $util -SampleInterval 0.1 -MaxSamples 1 -ErrorAction SilentlyContinue; "
        "if (-not $sample) { "
        "  Write-Output '{\"available\":false,\"source\":\"counter_sample_failed\"}'; exit 0 "
        "} "
        "$vals = $sample.CounterSamples | ForEach-Object { [double]$_.CookedValue }; "
        "if (-not $vals -or $vals.Count -eq 0) { "
        "  Write-Output '{\"available\":false,\"source\":\"empty_samples\"}'; exit 0 "
        "} "
        "$avg = ($vals | Measure-Object -Average).Average; "
        "$max = ($vals | Measure-Object -Maximum).Maximum; "
        "$obj = @{available=$true; avg_util_percent=$avg; max_util_percent=$max; source='windows_perf_counter'}; "
        "$obj | ConvertTo-Json -Compress"
    )
    cmd = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        ps_script,
    ]
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    except OSError as exc:
        return {"available": False, "source": f"powershell_unavailable: {exc}"}

    payload = _safe_json_parse(proc.stdout)
    if isinstance(payload, dict):
        return payload
    return {"available": False, "source": "json_parse_failed"}


def _inject_styles() -> None:
    # Revert to near-default Streamlit appearance to avoid theme conflicts.
    st.markdown(
        """
        <style>
        .event-card {
            margin: 10px 0;
        }
        .event-title {
            font-weight: 600;
        }
        .muted {
            opacity: 0.8;
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
    sample_telemetry: bool = False,
    sample_accelerator: bool = False,
    timeout_seconds: float | None = None,
) -> CommandResult:
    def _kill_process_tree(proc: subprocess.Popen[str]) -> None:
        if proc.poll() is not None:
            return
        try:
            if sys.platform.startswith("win"):
                subprocess.run(
                    ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                    text=True,
                    capture_output=True,
                    check=False,
                )
            else:
                proc.terminate()
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _sample_cpu_percent() -> float | None:
        try:
            import psutil  # type: ignore

            return float(psutil.cpu_percent(interval=None))
        except Exception:
            return None

    def _sample_compute_engine_percent_windows() -> float | None:
        if not sys.platform.startswith("win"):
            return None
        ps_script = (
            "$s = Get-Counter '\\GPU Engine(*engtype_Compute*)\\Utilization Percentage' "
            "-SampleInterval 0.05 -MaxSamples 1 -ErrorAction SilentlyContinue; "
            "if (-not $s) { '' } else { "
            "$vals = $s.CounterSamples | ForEach-Object { [double]$_.CookedValue }; "
            "if (-not $vals -or $vals.Count -eq 0) { '' } "
            "else { ($vals | Measure-Object -Average).Average } }"
        )
        try:
            proc = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    ps_script,
                ],
                text=True,
                capture_output=True,
                check=False,
            )
        except OSError:
            return None
        raw = proc.stdout.strip()
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    t0 = time.perf_counter()
    cpu_series: list[float] = []
    accel_series: list[float] = []
    elapsed_series: list[float] = []
    timed_out = False

    if not sample_telemetry:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout_seconds)
            return_code = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_process_tree(proc)
            stdout, stderr = proc.communicate()
            return_code = -9
        except KeyboardInterrupt:
            _kill_process_tree(proc)
            raise
        elapsed = max(0.0, time.perf_counter() - t0)
    else:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        last_accel_sample = 0.0
        try:
            while proc.poll() is None:
                now = time.perf_counter()
                elapsed_now = max(0.0, now - t0)
                if timeout_seconds and elapsed_now > timeout_seconds:
                    timed_out = True
                    _kill_process_tree(proc)
                    break
                cpu = _sample_cpu_percent()
                if cpu is not None:
                    cpu_series.append(cpu)
                    elapsed_series.append(elapsed_now)
                # Accelerator counter sampling is expensive on some systems; keep it sparse and optional.
                if sample_accelerator and (now - last_accel_sample) >= 3.0:
                    accel = _sample_compute_engine_percent_windows()
                    if accel is not None:
                        accel_series.append(accel)
                    last_accel_sample = now
                time.sleep(0.4)
        except KeyboardInterrupt:
            _kill_process_tree(proc)
            raise

        stdout, stderr = proc.communicate()
        return_code = proc.returncode if not timed_out else -9
        elapsed = max(0.0, time.perf_counter() - t0)

    payload = _safe_json_parse(stdout) if expect_json else None
    ok = return_code == 0 and (payload is not None if expect_json else True)

    error = None
    if timed_out:
        timeout_label = f"{timeout_seconds:.0f}s" if timeout_seconds else "configured timeout"
        error = f"Command timed out after {timeout_label} and was terminated."
    elif return_code != 0:
        error = f"Command failed with exit code {return_code}."
    elif expect_json and payload is None:
        error = "Command succeeded but output was not valid JSON."

    return CommandResult(
        ok=ok,
        command=cmd,
        stdout=stdout,
        stderr=stderr,
        return_code=return_code,
        elapsed_seconds=elapsed,
        timed_out=timed_out,
        telemetry={
            "elapsed_seconds": elapsed_series,
            "cpu_percent": cpu_series,
            "accel_compute_percent": accel_series,
        },
        payload=payload,
        error=error,
    )


def _extract_qnn_stage_timings(raw_text: str) -> list[tuple[str, float]]:
    """
    Parse QNN stage timing lines like:
      Completed stage: Graph Optimization (2736677 us)
    """
    rows: list[tuple[str, float]] = []
    pattern = re.compile(r"Completed stage:\s*(.+?)\s*\((\d+)\s*us\)", re.IGNORECASE)
    for line in raw_text.splitlines():
        m = pattern.search(line)
        if not m:
            continue
        stage = m.group(1).strip()
        micros = float(m.group(2))
        millis = micros / 1000.0
        rows.append((stage, millis))
    return rows


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
    detected_color = str(event.get("detected_color", "")).strip() or "unknown"

    st.markdown(
        f"""
        <div class="event-card">
          <div class="event-title">{_event_label(event)}</div>
          <div class="muted">second={second} • context={context} • color={detected_color} • confidence={confidence}</div>
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
    runtime: str,
    conf: float,
    iou: float,
    sample_fps: float,
    input_size: int,
    save_full_frames: bool,
    frame_max_width: int,
    frame_jpeg_quality: int,
    caption_frames: bool,
    caption_model: str,
    caption_every_n_frames: int,
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
        "--runtime",
        runtime,
        "--conf",
        str(conf),
        "--iou",
        str(iou),
        "--sample-fps",
        str(sample_fps),
        "--input-size",
        str(input_size),
        "--frame-max-width",
        str(frame_max_width),
        "--frame-jpeg-quality",
        str(frame_jpeg_quality),
        "--caption-model",
        caption_model,
        "--caption-every-n-frames",
        str(max(1, int(caption_every_n_frames))),
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
    if save_full_frames:
        cmd.append("--save-full-frames")
    else:
        cmd.append("--no-save-full-frames")
    if caption_frames:
        cmd.append("--caption-frames")
    else:
        cmd.append("--no-caption-frames")
    return cmd


def _build_query_command(
    python_exec: str,
    extractor_dir: Path,
    out_dir: Path,
    video_path: str,
    command: str,
    label_or_query: str | None = None,
    question: str | None = None,
    use_llm: bool = False,
    llm_model: str | None = None,
    llm_base_url: str | None = None,
    llm_api_key: str | None = None,
    strict_evidence: bool = False,
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
    if question is not None:
        cmd.extend(["--question", question])
    if use_llm:
        cmd.append("--llm")
    if llm_model:
        cmd.extend(["--llm-model", llm_model])
    if llm_base_url:
        cmd.extend(["--llm-base-url", llm_base_url])
    if llm_api_key:
        cmd.extend(["--llm-api-key", llm_api_key])
    if strict_evidence:
        cmd.append("--strict-evidence")
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
        page_title="Aura Demo Console",
        page_icon=":camera_with_flash:",
        layout="wide",
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
    if "last_ingest_summary" not in st.session_state:
        st.session_state.last_ingest_summary = {}
    if "ingest_timeout_seconds" not in st.session_state:
        st.session_state.ingest_timeout_seconds = 180

    with st.sidebar:
        st.header("Session")
        python_exec = sys.executable
        extractor_dir_str = str(DEFAULT_EXTRACTOR_DIR)
        out_dir_str = st.session_state.last_out_dir

        extractor_dir = Path(extractor_dir_str).expanduser().resolve()
        out_dir = Path(out_dir_str).expanduser()
        st.session_state.last_out_dir = str(out_dir)

        ingest_summary = st.session_state.last_ingest_summary
        if isinstance(ingest_summary, dict) and ingest_summary:
            model = ingest_summary.get("detector_model", "-")
            backend = ingest_summary.get("detector_backend", "-")
            st.metric("Last ingest backend", f"{model}/{backend}")

        st.divider()
        st.number_input(
            "Ingest timeout (sec)",
            min_value=30,
            max_value=3600,
            step=10,
            key="ingest_timeout_seconds",
            help="Ingest process is force-terminated after this duration to avoid hangs.",
        )
        with st.expander("Advanced paths", expanded=False):
            python_exec = st.text_input("Python executable", value=python_exec)
            extractor_dir_str = st.text_input("Extractor directory", value=extractor_dir_str)
            out_dir_str = st.text_input("Output directory", value=out_dir_str)
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

    tabs = st.tabs(["Ingest", "Mind Palace"])

    with tabs[0]:
        st.subheader("Ingest Video")
        video_path = st.text_input(
            "Video path",
            value=st.session_state.last_video_path,
            placeholder="/absolute/path/to/video.mp4",
        )

        # Defaults (all controls exposed under Advanced).
        prefer_model = "yolov11"
        runtime = "qnn"
        sample_fps = 3.0
        reset_output = True
        save_full_frames = True
        conf = 0.25
        input_size = 960
        caption_frames = True
        strict_evidence_default = True
        iou = 0.60
        thumb_size = 50
        crop_padding = 8
        jpeg_quality = 75
        track_max_gap = 3
        track_iou_threshold = 0.20
        track_center_dist_ratio = 0.12
        frame_max_width = 960
        frame_jpeg_quality = 70
        caption_every_n_frames = 1
        caption_model = "nlpconnect/vit-gpt2-image-captioning"

        with st.expander("Advanced ingest options", expanded=False):
            c1, c2, c3 = st.columns(3)
            prefer_model = c1.selectbox("Model", ["yolov11", "yolov8"], index=0)
            c2.caption("Runtime: QNN (NPU enforced)")
            sample_fps = c3.slider("Sample FPS", min_value=1.0, max_value=4.0, value=3.0, step=0.5)

            c4, c5 = st.columns(2)
            reset_output = c4.checkbox("Reset output before ingest", value=True)
            save_full_frames = c5.checkbox("Save full frames", value=True)

            c6, c7 = st.columns(2)
            conf = c6.slider("Confidence", min_value=0.1, max_value=0.95, value=0.25, step=0.05)
            input_size = c7.selectbox("Input size", [640, 960], index=1)
            caption_frames = st.checkbox(
                "Caption frames locally (Video RAG)",
                value=True,
                help="Uses local transformers image-caption model. First run downloads weights.",
            )
            strict_evidence_default = st.checkbox("Strict evidence mode default", value=True)

        if st.button("Ingest Video", type="primary", use_container_width=True):
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
                provider_info = _ort_provider_info(python_exec, extractor_dir)
                if not bool(provider_info.get("qnn_available")):
                    st.error(
                        "NPU/QNN is required for this demo mode, but QNNExecutionProvider is unavailable."
                    )
                    providers = provider_info.get("providers", [])
                    if providers:
                        st.caption(f"Available providers: {', '.join(providers)}")
                    return
                cmd = _build_ingest_command(
                    python_exec=python_exec,
                    extractor_dir=extractor_dir,
                    video_path=video_path,
                    out_dir=out_dir,
                    prefer_model=prefer_model,
                    runtime=runtime,
                    conf=conf,
                    iou=iou,
                    sample_fps=sample_fps,
                    input_size=int(input_size),
                    save_full_frames=save_full_frames,
                    frame_max_width=int(frame_max_width),
                    frame_jpeg_quality=int(frame_jpeg_quality),
                    caption_frames=bool(caption_frames),
                    caption_model=caption_model.strip() or "nlpconnect/vit-gpt2-image-captioning",
                    caption_every_n_frames=int(caption_every_n_frames),
                    thumb_size=int(thumb_size),
                    crop_padding=int(crop_padding),
                    jpeg_quality=int(jpeg_quality),
                    reset_output=reset_output,
                    track_max_gap=int(track_max_gap),
                    track_iou_threshold=track_iou_threshold,
                    track_center_dist_ratio=track_center_dist_ratio,
                )
                with st.spinner("Running ingestion..."):
                    result = _run_command(
                        cmd,
                        cwd=extractor_dir,
                        sample_telemetry=False,
                        sample_accelerator=False,
                        timeout_seconds=float(st.session_state.ingest_timeout_seconds),
                    )

                st.session_state.last_cmd = cmd
                if result.ok:
                    st.session_state.last_ingest_summary = _parse_ingest_summary(result.stdout)
                    st.success("Ingestion completed.")
                    mem_path = _resolve_memory_path(out_dir, video_path)
                    st.info(f"Memory log: {mem_path}")
                    summary = st.session_state.last_ingest_summary
                    model = summary.get("detector_model")
                    backend = summary.get("detector_backend")
                    detector_input_size = summary.get("detector_input_size")
                    sample_fps_used = summary.get("sample_fps")
                    if model and backend:
                        st.caption(
                            f"Detector runtime used: {model}/{backend}"
                            + (
                                f" | input={detector_input_size} | sample_fps={sample_fps_used}"
                                if detector_input_size or sample_fps_used
                                else ""
                            )
                        )

                    s1, s2, s3, s4 = st.columns(4)
                    processed_seconds = summary.get("processed_seconds")
                    events_written = summary.get("events_written")
                    captions_written = summary.get("captions_written")
                    s1.metric("Backend", f"{model}/{backend}" if model and backend else "-")
                    s2.metric(
                        "Samples",
                        str(processed_seconds) if processed_seconds is not None else "-",
                    )
                    s3.metric(
                        "Events",
                        str(events_written) if events_written is not None else "-",
                    )
                    s4.metric("Time (s)", f"{result.elapsed_seconds:.2f}")
                    st.caption(f"Captions written: {captions_written if captions_written is not None else '-'}")

                    if isinstance(events_written, int) and result.elapsed_seconds > 0:
                        st.caption(
                            f"Throughput: {events_written / result.elapsed_seconds:.2f} events/sec "
                            f"(wall-clock)."
                        )

                else:
                    st.error(result.error or "Ingestion failed.")

                with st.expander("Command output (advanced)", expanded=not result.ok):
                    st.code(" ".join(result.command))
                    show_raw = st.checkbox(
                        "Show raw stdout/stderr",
                        value=not result.ok,
                        key=f"show_raw_ingest_{hash(' '.join(result.command))}",
                    )
                    if show_raw:
                        if result.stdout.strip():
                            st.text_area("stdout", result.stdout, height=180)
                        if result.stderr.strip():
                            st.text_area("stderr", result.stderr, height=180)

    with tabs[1]:
        st.subheader("Mind Palace")
        st.caption(
            "Ask open-ended questions over frame memory and evidence."
        )
        mp_question = st.text_input(
            "Question (open-ended)",
            value="What happened near the dining table?",
            help="Examples: 'What did I place near the table?', 'What objects were visible around second 10?'",
        )
        # Defaults (all controls exposed under Advanced).
        mp_limit = 5
        strict_evidence = bool(strict_evidence_default)
        llm_on = False
        llm_model = ""
        llm_base_url = ""
        llm_api_key = ""

        with st.expander("Advanced mind palace options", expanded=False):
            mp_limit = st.slider("Evidence limit", min_value=1, max_value=15, value=5, step=1)
            strict_evidence = st.checkbox(
                "Strict evidence mode (abstain on weak retrieval)",
                value=bool(strict_evidence_default),
                help="Prevents nearest-match hallucinations for unsupported open-ended questions.",
            )
            llm_on = st.checkbox("Use LLM synthesis", value=False)
            c1, c2 = st.columns(2)
            llm_model = c1.text_input("LLM model (optional)", value="", placeholder="gpt-4o-mini")
            llm_base_url = c2.text_input(
                "LLM base URL (optional)",
                value="",
                placeholder="https://api.openai.com/v1",
            )
            llm_api_key = st.text_input(
                "LLM API key (optional)",
                value="",
                type="password",
                help="If empty, extractor uses AURA_LLM_API_KEY env var.",
            )

        if st.button("Ask", type="primary", use_container_width=True):
            problems = _validate_paths(python_exec, extractor_dir)
            if problems:
                for p in problems:
                    st.error(p)
            elif not st.session_state.last_video_path:
                st.error("Run ingestion first or set a video path in the Ingest tab.")
            elif not mp_question.strip():
                st.error("Enter a question.")
            else:
                cmd = _build_query_command(
                    python_exec=python_exec,
                    extractor_dir=extractor_dir,
                    out_dir=out_dir,
                    video_path=st.session_state.last_video_path,
                    command="ask",
                    question=mp_question.strip(),
                    use_llm=llm_on,
                    llm_model=llm_model.strip() or None,
                    llm_base_url=llm_base_url.strip() or None,
                    llm_api_key=llm_api_key.strip() or None,
                    strict_evidence=bool(strict_evidence),
                    limit=int(mp_limit),
                )
                with st.spinner("Searching memory evidence..."):
                    result = _run_command(cmd, cwd=extractor_dir, expect_json=True)

                st.session_state.last_cmd = cmd
                if not result.ok:
                    st.error(result.error or "Mind palace query failed.")
                else:
                    payload = result.payload if isinstance(result.payload, dict) else {}
                    answer_text = str(payload.get("answer", ""))
                    i1, i2, i3 = st.columns(3)
                    i1.metric("Intent", str(payload.get("intent", "-")))
                    i2.metric("Confidence", str(payload.get("confidence", "-")))
                    i3.metric("Primary label", str(payload.get("primary_label", "-")))
                    if answer_text:
                        st.markdown("#### Answer")
                        if bool(payload.get("abstained")):
                            st.warning(answer_text)
                        else:
                            st.markdown(answer_text)
                    if bool(payload.get("abstained")) and payload.get("abstain_reason"):
                        st.caption(f"Abstain reason: {payload.get('abstain_reason')}")
                    llm_meta = payload.get("llm", {})
                    if isinstance(llm_meta, dict) and llm_meta.get("enabled"):
                        if llm_meta.get("used"):
                            st.success(f"LLM answer generated ({llm_meta.get('model')}).")
                        else:
                            st.warning(
                                "LLM was enabled but fallback heuristic answer was used: "
                                + str(llm_meta.get("error", "unknown reason"))
                            )
                    visual_meta = payload.get("visual_reasoner", {})
                    if isinstance(visual_meta, dict):
                        model_ready = bool(visual_meta.get("model_ready"))
                        produced = bool(visual_meta.get("answers_produced"))
                        if not model_ready:
                            st.caption(
                                "Local visual QA model not available yet. Install/download model weights before offline demo."
                            )
                        elif not produced:
                            st.caption(
                                "Visual QA model is loaded but produced low-confidence answers for this question."
                            )
                    rows = payload.get("evidence", [])
                    citations_md = str(payload.get("citations_markdown", "")).strip()
                    if citations_md and "### Evidence" not in answer_text:
                        st.markdown(citations_md)
                    if isinstance(rows, list) and rows:
                        st.markdown("#### Evidence")
                        for idx, row in enumerate(rows, start=1):
                            st.markdown(
                                f"**#{idx}** sec={row.get('video_second')} "
                                f"label={row.get('label')} color={row.get('detected_color')} "
                                f"context={row.get('context')}"
                            )
                            frame_path = str(row.get("frame_path", "")).strip()
                            thumb_path = str(row.get("thumbnail_path", "")).strip()
                            c1, c2 = st.columns(2)
                            if frame_path and Path(frame_path).exists():
                                c1.image(frame_path, caption="frame", use_container_width=True)
                            if thumb_path and Path(thumb_path).exists():
                                c2.image(thumb_path, caption="thumbnail", use_container_width=False, width=180)


if __name__ == "__main__":
    main()
