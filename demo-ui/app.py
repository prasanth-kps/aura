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
    if "last_ingest_summary" not in st.session_state:
        st.session_state.last_ingest_summary = {}
    if "telemetry" not in st.session_state:
        st.session_state.telemetry = {}
    if "ingest_telemetry_enabled" not in st.session_state:
        st.session_state.ingest_telemetry_enabled = False
    if "accel_telemetry_enabled" not in st.session_state:
        st.session_state.accel_telemetry_enabled = False
    if "ingest_timeout_seconds" not in st.session_state:
        st.session_state.ingest_timeout_seconds = 180

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
        st.markdown("### NPU Telemetry")
        if st.button("Refresh telemetry", use_container_width=True):
            provider_info = _ort_provider_info(python_exec, extractor_dir)
            npu_stats = _windows_npu_stats() if sys.platform.startswith("win") else {
                "available": False,
                "source": "non_windows_platform",
            }
            st.session_state.telemetry = {
                "provider_info": provider_info,
                "npu_stats": npu_stats,
            }

        telemetry = st.session_state.telemetry
        provider_info = telemetry.get("provider_info", {}) if isinstance(telemetry, dict) else {}
        providers = provider_info.get("providers", [])
        qnn_available = bool(provider_info.get("qnn_available"))
        st.metric("QNN provider available", "Yes" if qnn_available else "No")
        if providers:
            st.caption(f"Providers: {', '.join(providers)}")

        npu_stats = telemetry.get("npu_stats", {}) if isinstance(telemetry, dict) else {}
        if npu_stats.get("available"):
            avg_util = npu_stats.get("avg_util_percent")
            max_util = npu_stats.get("max_util_percent")
            st.metric("NPU avg util (%)", f"{float(avg_util):.1f}" if avg_util is not None else "-")
            st.metric("NPU peak util (%)", f"{float(max_util):.1f}" if max_util is not None else "-")
        else:
            src = str(npu_stats.get("source", "not_refreshed"))
            st.caption(f"NPU utilization: unavailable ({src})")

        ingest_summary = st.session_state.last_ingest_summary
        if isinstance(ingest_summary, dict) and ingest_summary:
            model = ingest_summary.get("detector_model", "-")
            backend = ingest_summary.get("detector_backend", "-")
            st.metric("Last ingest backend", f"{model}/{backend}")

        st.divider()
        st.markdown("### Demo Tips")
        st.write("- Use a fixed test video for repeatable stage demos.")
        st.write("- Keep `--reset-output` enabled for clean reruns.")
        st.write("- Use natural queries like `where is my black bottle`.")
        st.checkbox(
            "Enable live utilization graphs during ingest",
            value=st.session_state.ingest_telemetry_enabled,
            key="ingest_telemetry_enabled",
            help="Can add overhead; keep OFF for fastest ingestion.",
        )
        st.checkbox(
            "Include accelerator counter sampling (slower)",
            value=st.session_state.accel_telemetry_enabled,
            key="accel_telemetry_enabled",
            help="Windows performance counters can be expensive on some machines.",
        )
        st.number_input(
            "Ingest timeout (sec)",
            min_value=30,
            max_value=3600,
            value=int(st.session_state.ingest_timeout_seconds),
            step=10,
            key="ingest_timeout_seconds",
            help="Ingest process is force-terminated after this duration to avoid hangs.",
        )

    tabs = st.tabs(["Ingest", "Ask", "Timeline", "Explore", "Mind Palace"])

    with tabs[0]:
        st.subheader("1) Ingest Video")
        video_path = st.text_input(
            "Video path",
            value=st.session_state.last_video_path,
            placeholder="/absolute/path/to/video.mp4",
        )

        c1, c2, c3, c4, c5 = st.columns(5)
        prefer_model = c1.selectbox("Model", ["yolov11", "yolov8"], index=0)
        runtime = c2.selectbox("Runtime", ["auto", "qnn", "cpu"], index=0)
        conf = c3.slider("Confidence", min_value=0.1, max_value=0.95, value=0.25, step=0.05)
        iou = c4.slider("NMS IoU", min_value=0.1, max_value=0.95, value=0.60, step=0.05)
        reset_output = c5.checkbox("Reset output", value=True)

        c6, c7, c8 = st.columns(3)
        sample_fps = c6.slider("Sample FPS", min_value=1.0, max_value=4.0, value=3.0, step=0.5)
        input_size = c7.selectbox("Input size", [640, 960], index=1)
        thumb_size = c8.number_input("Thumb size", min_value=20, max_value=256, value=50, step=2)

        c9, c10, c11, c12 = st.columns(4)
        crop_padding = c9.number_input("Crop padding", min_value=0, max_value=64, value=8, step=1)
        jpeg_quality = c10.number_input("JPEG quality", min_value=40, max_value=100, value=75, step=1)
        track_max_gap = c11.number_input("Track max gap (sec)", min_value=1, max_value=15, value=3, step=1)
        track_iou_threshold = c12.slider(
            "Track IoU threshold", min_value=0.05, max_value=0.9, value=0.20, step=0.05
        )
        track_center_dist_ratio = st.slider(
            "Track center-dist ratio", min_value=0.02, max_value=0.8, value=0.12, step=0.02
        )
        c13, c14, c15 = st.columns(3)
        save_full_frames = c13.checkbox("Save full frames (mind palace)", value=True)
        frame_max_width = c14.selectbox("Frame max width", [640, 960, 1280], index=1)
        frame_jpeg_quality = c15.number_input(
            "Frame JPEG quality", min_value=40, max_value=100, value=70, step=1
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
                    runtime=runtime,
                    conf=conf,
                    iou=iou,
                    sample_fps=sample_fps,
                    input_size=int(input_size),
                    save_full_frames=save_full_frames,
                    frame_max_width=int(frame_max_width),
                    frame_jpeg_quality=int(frame_jpeg_quality),
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
                        sample_telemetry=bool(st.session_state.ingest_telemetry_enabled),
                        sample_accelerator=bool(st.session_state.accel_telemetry_enabled),
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

                    # Demo-focused performance summary.
                    s1, s2, s3, s4, s5 = st.columns(5)
                    processed_seconds = summary.get("processed_seconds")
                    events_written = summary.get("events_written")
                    frames_written = summary.get("frames_written")
                    s1.metric("Runtime backend", f"{model}/{backend}" if model and backend else "-")
                    s2.metric(
                        "Processed samples",
                        str(processed_seconds) if processed_seconds is not None else "-",
                    )
                    s3.metric(
                        "Events written",
                        str(events_written) if events_written is not None else "-",
                    )
                    s4.metric(
                        "Frames written",
                        str(frames_written) if frames_written is not None else "-",
                    )
                    s5.metric("Ingest wall time (s)", f"{result.elapsed_seconds:.2f}")

                    if isinstance(events_written, int) and result.elapsed_seconds > 0:
                        st.caption(
                            f"Throughput: {events_written / result.elapsed_seconds:.2f} events/sec "
                            f"(wall-clock)."
                        )

                    telemetry = result.telemetry if isinstance(result.telemetry, dict) else {}
                    cpu_values = telemetry.get("cpu_percent", [])
                    accel_values = telemetry.get("accel_compute_percent", [])
                    if isinstance(cpu_values, list) and cpu_values:
                        st.markdown("#### Utilization Graphs")
                        st.line_chart({"CPU %": cpu_values})
                    if isinstance(accel_values, list) and accel_values:
                        st.line_chart({"Accelerator Compute % (NPU/GPU engine proxy)": accel_values})
                        st.caption(
                            "Accelerator graph uses Windows 'GPU Engine ... engtype_Compute' counters. "
                            "On some Snapdragon builds this reflects NPU/QNN activity; on others only "
                            "a proxy is available."
                        )
                    elif runtime in {"qnn", "auto"}:
                        st.caption(
                            "Accelerator utilization counter is unavailable on this machine build. "
                            "Use backend=qnn + QNN stage timings as primary NPU proof."
                        )

                    if backend == "qnn":
                        qnn_timings = _extract_qnn_stage_timings(result.stdout)
                        if qnn_timings:
                            st.markdown("#### QNN Stage Timings")
                            for stage, millis in qnn_timings[:8]:
                                st.write(f"- {stage}: {millis:.2f} ms")
                        else:
                            st.caption("QNN active; detailed stage timings were not found in stdout.")
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

    with tabs[4]:
        st.subheader("5) Mind Palace Q&A")
        st.caption(
            "Ask open-ended questions over full-frame memory + object events. "
            "This retrieves evidence rows and summarizes the best matches."
        )
        mp_question = st.text_input(
            "Question (open-ended)",
            value="What happened near the dining table?",
            help="Examples: 'What did I place near the table?', 'What objects were visible around second 10?'",
        )
        mp_limit = st.slider("Evidence limit", min_value=1, max_value=15, value=5, step=1)
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

        if st.button("Ask memory", type="primary", use_container_width=True):
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
                    if answer_text:
                        st.markdown("#### Answer")
                        st.markdown(answer_text)
                    llm_meta = payload.get("llm", {})
                    if isinstance(llm_meta, dict) and llm_meta.get("enabled"):
                        if llm_meta.get("used"):
                            st.success(f"LLM answer generated ({llm_meta.get('model')}).")
                        else:
                            st.warning(
                                "LLM was enabled but fallback heuristic answer was used: "
                                + str(llm_meta.get("error", "unknown reason"))
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
