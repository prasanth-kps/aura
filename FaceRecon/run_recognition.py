import argparse
from collections import defaultdict
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from qai_hub_models.models.cavaface import Model as CavaFaceModel

from llm_name_recognizer import extract_names_with_llm, regex_fallback_extract


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec) + 1e-12
    return vec / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)))


def load_local_model() -> torch.nn.Module:
    model = CavaFaceModel.from_pretrained()
    model.eval()
    return model


def resolve_torch_device(accelerator: str) -> tuple[object, str]:
    if accelerator in {"auto", "gpu"}:
        if torch.cuda.is_available():
            return torch.device("cuda"), "cuda"
        try:
            import torch_directml  # type: ignore

            return torch_directml.device(), "directml"
        except Exception:
            pass
    return torch.device("cpu"), "cpu"


def _ensure_cavaface_onnx(model: torch.nn.Module, onnx_path: Path) -> None:
    """Export CavaFace to ONNX if file does not exist."""
    if onnx_path.exists():
        return
    model_cpu = model.to("cpu").eval()
    dummy = torch.randn(1, 3, 112, 112, dtype=torch.float32)
    traced = torch.jit.trace(model_cpu, dummy)
    torch.onnx.export(
        traced,
        dummy,
        str(onnx_path),
        input_names=["image"],
        output_names=["embedding"],
        opset_version=17,
        do_constant_folding=True,
    )


def build_qnn_session(model: torch.nn.Module, onnx_path: Path) -> object:
    try:
        import onnxruntime as ort
    except Exception as exc:
        raise RuntimeError("onnxruntime-qnn is required for NPU mode. Install with: py -m pip install --user onnxruntime-qnn") from exc

    providers = ort.get_available_providers()
    if "QNNExecutionProvider" not in providers:
        raise RuntimeError("QNNExecutionProvider not available on this machine.")

    _ensure_cavaface_onnx(model, onnx_path)
    session = ort.InferenceSession(str(onnx_path), providers=["QNNExecutionProvider", "CPUExecutionProvider"])
    return session


def build_dml_session(model: torch.nn.Module, onnx_path: Path) -> object:
    """Build ONNX session with DirectML for Qualcomm Adreno / AMD / Intel GPUs."""
    try:
        import onnxruntime as ort
    except Exception as exc:
        raise RuntimeError("onnxruntime required. Install: py -m pip install onnxruntime-directml") from exc

    providers = ort.get_available_providers()
    if "DmlExecutionProvider" not in providers:
        raise RuntimeError(
            "DmlExecutionProvider not available. Install: py -m pip install onnxruntime-directml"
        ) from None

    _ensure_cavaface_onnx(model, onnx_path)
    session = ort.InferenceSession(str(onnx_path), providers=["DmlExecutionProvider", "CPUExecutionProvider"])
    return session


def make_input_tensor(input_npy: str | None) -> np.ndarray:
    if input_npy:
        arr = np.load(input_npy).astype(np.float32)
        if arr.shape != (1, 3, 112, 112):
            raise ValueError("Input tensor must have shape (1, 3, 112, 112).")
        return np.ascontiguousarray(arr)
    return np.ascontiguousarray(np.random.randn(1, 3, 112, 112).astype(np.float32))


def run_embedding_inference(model: torch.nn.Module, input_tensor: np.ndarray, model_device: object) -> np.ndarray:
    if hasattr(model, "run") and callable(getattr(model, "run")) and hasattr(model, "get_inputs"):
        input_name = model.get_inputs()[0].name
        ort_out = model.run(None, {input_name: np.ascontiguousarray(input_tensor).astype(np.float32)})[0]
        embedding = np.asarray(ort_out).reshape(-1).astype(np.float32)
        return l2_normalize(embedding)

    with torch.no_grad():
        batch = torch.from_numpy(np.ascontiguousarray(input_tensor)).float().contiguous().to(model_device)
        output = model(batch)
        if isinstance(output, (tuple, list)):
            output = output[0]
    embedding = np.asarray(output.detach().cpu().numpy()).reshape(-1).astype(np.float32)
    return l2_normalize(embedding)


def load_db(db_path: Path) -> tuple[list[str], np.ndarray]:
    if not db_path.exists():
        return [], np.empty((0, 512), dtype=np.float32)
    data = np.load(db_path, allow_pickle=False)
    names = data["names"].tolist()
    embs = data["embeddings"].astype(np.float32)
    return names, embs


def save_db(db_path: Path, names: list[str], embeddings: np.ndarray) -> None:
    np.savez(db_path, names=np.asarray(names, dtype=np.str_), embeddings=embeddings.astype(np.float32))


def enroll(db_path: Path, name: str, embedding: np.ndarray) -> None:
    names, embeddings = load_db(db_path)
    if len(names) == 0:
        save_db(db_path, [name], embedding.reshape(1, -1))
        print(f"Enrolled '{name}'. Total identities: 1")
        return

    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for n, e in zip(names, embeddings):
        grouped[n].append(e)
    grouped[name].append(embedding)

    merged_names: list[str] = []
    merged_embeddings: list[np.ndarray] = []
    for person, embs in grouped.items():
        centroid = l2_normalize(np.mean(np.vstack(embs), axis=0).astype(np.float32))
        merged_names.append(person)
        merged_embeddings.append(centroid)

    save_db(db_path, merged_names, np.vstack(merged_embeddings))
    print(f"Enrolled/updated '{name}'. Total identities: {len(merged_names)}")


def search(db_path: Path, query_embedding: np.ndarray) -> None:
    names, embeddings = load_db(db_path)
    if len(names) == 0:
        print("Database is empty. Enroll at least one identity first.")
        return

    scores = [cosine_similarity(query_embedding, emb) for emb in embeddings]
    best_idx = int(np.argmax(scores))
    print(f"Best match: {names[best_idx]} (cosine={scores[best_idx]:.4f})")

    print("Top matches:")
    ranked = sorted(zip(names, scores), key=lambda x: x[1], reverse=True)[:5]
    for name, score in ranked:
        print(f"  - {name}: {score:.4f}")


def best_match(
    names: list[str],
    embeddings: np.ndarray,
    query_embedding: np.ndarray,
    threshold: float,
    second_best_margin: float,
) -> Tuple[str, float]:
    if len(names) == 0:
        return "Unknown", -1.0
    scores = [cosine_similarity(query_embedding, emb) for emb in embeddings]
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    if len(scores) > 1:
        sorted_scores = sorted(scores, reverse=True)
        score_gap = sorted_scores[0] - sorted_scores[1]
    else:
        score_gap = 1.0
    if best_score < threshold or score_gap < second_best_margin:
        return "Unknown", best_score
    return names[best_idx], best_score


def expand_to_square_with_margin(
    frame: np.ndarray, x: int, y: int, w: int, h: int, margin_ratio: float
) -> np.ndarray:
    h_img, w_img = frame.shape[:2]
    cx = x + w / 2.0
    cy = y + h / 2.0
    side = max(w, h) * (1.0 + margin_ratio)

    x0 = int(max(0, cx - side / 2.0))
    y0 = int(max(0, cy - side / 2.0))
    x1 = int(min(w_img, cx + side / 2.0))
    y1 = int(min(h_img, cy + side / 2.0))

    crop = frame[y0:y1, x0:x1]
    return crop


def face_bgr_to_model_input(face_bgr: np.ndarray) -> np.ndarray:
    import cv2

    resized = cv2.resize(face_bgr, (112, 112), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # ArcFace-style input normalization improves embedding stability.
    arr = (rgb.astype(np.float32) / 255.0 - 0.5) / 0.5
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    return np.ascontiguousarray(arr[np.newaxis, ...])


def _prompt_enroll_name(
    result_queue: "queue.Queue[tuple[str, np.ndarray]]",
    embedding: np.ndarray,
    suggested_names: list[str],
    auto_enroll_from_suggestions: bool = False,
) -> None:
    merged_suggestions = list(suggested_names)

    print("\nUnknown face persisted across frames.")
    if merged_suggestions:
        print(f"Name suggestions: {', '.join(merged_suggestions)}")

    # In assist mode, auto-enroll with the top suggestion when available.
    if auto_enroll_from_suggestions and merged_suggestions:
        entered = merged_suggestions[0]
        print(f"Assist mode auto-enroll: {entered}")
        result_queue.put((entered, embedding))
        return

    # In non-interactive terminals, stdin is unavailable.
    # Fall back to first suggestion (if any) instead of crashing.
    if not sys.stdin.isatty():
        entered = merged_suggestions[0] if merged_suggestions else ""
        if entered:
            print(f"Auto-enrolling from suggestion: {entered}")
        else:
            print("Skipping enrollment (no suggestions).")
        result_queue.put((entered, embedding))
        return

    try:
        entered = input("Enter name to enroll (blank to skip): ").strip()
    except EOFError:
        entered = merged_suggestions[0] if merged_suggestions else ""
        if entered:
            print(f"EOF on input; auto-enrolling from suggestion: {entered}")
        else:
            print("EOF on input; skipping enrollment.")
    result_queue.put((entered, embedding))


def camera_mode(
    db_path: Path,
    model: torch.nn.Module,
    model_device: object,
    threshold: float,
    second_best_margin: float,
    face_margin: float,
    camera_index: int,
    video_path: str | None,
    min_frames_to_prompt: int,
    suggested_names: list[str],
    auto_enroll_from_suggestions: bool = False,
) -> None:
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("OpenCV is required for camera mode. Install with: pip install opencv-python") from exc

    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if detector.empty():
        raise RuntimeError("Failed to load OpenCV Haar cascade detector.")

    source = video_path if video_path else camera_index
    if video_path:
        cap = cv2.VideoCapture(source)
    else:
        # DSHOW is often more reliable than MSMF on Windows webcams.
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        if video_path:
            raise RuntimeError(f"Could not open video file: {video_path}")
        raise RuntimeError(f"Could not open camera index {camera_index}.")

    names, embeddings = load_db(db_path)
    last_prompt_time = 0.0
    prompt_cooldown_seconds = 3.0
    prompt_results: "queue.Queue[tuple[str, np.ndarray]]" = queue.Queue()
    prompt_thread: threading.Thread | None = None
    unknown_anchor_embedding: np.ndarray | None = None
    unknown_streak = 0
    unknown_buffer: list[np.ndarray] = []
    consecutive_read_failures = 0

    print("Video mode started. Press 'q' to quit." if video_path else "Camera mode started. Press 'q' to quit.")
    try:
        while True:
            while not prompt_results.empty():
                entered, candidate_embedding = prompt_results.get_nowait()
                if entered:
                    enroll(db_path, entered, candidate_embedding)
                    names, embeddings = load_db(db_path)
                else:
                    print("Skipped enrollment for unknown face.")
                last_prompt_time = time.time()
                prompt_thread = None

            ok, frame = cap.read()
            if not ok:
                if video_path:
                    break
                consecutive_read_failures += 1
                if consecutive_read_failures >= 30:
                    raise RuntimeError(
                        "Camera opened but failed to read frames repeatedly. "
                        "Check camera permissions/in-use state or try --camera-index 1."
                    )
                time.sleep(0.05)
                continue
            consecutive_read_failures = 0

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
            frame_unknown_embedding: np.ndarray | None = None

            for (x, y, w, h) in faces:
                face_crop = expand_to_square_with_margin(frame, x, y, w, h, face_margin)
                if face_crop.size == 0:
                    continue

                input_tensor = face_bgr_to_model_input(face_crop)
                embedding = run_embedding_inference(model, input_tensor, model_device)
                label, score = best_match(names, embeddings, embedding, threshold, second_best_margin)
                is_unknown = (label == "Unknown") or (score < threshold)

                y0, y1 = max(0, y), min(frame.shape[0], y + h)
                x0, x1 = max(0, x), min(frame.shape[1], x + w)

                if is_unknown:
                    label = "Unknown"
                    color = (0, 0, 255)
                    if frame_unknown_embedding is None:
                        frame_unknown_embedding = embedding
                else:
                    color = (0, 255, 0)

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

            if frame_unknown_embedding is None:
                unknown_anchor_embedding = None
                unknown_streak = 0
                unknown_buffer.clear()
            else:
                if unknown_anchor_embedding is None:
                    unknown_anchor_embedding = frame_unknown_embedding
                    unknown_streak = 1
                    unknown_buffer = [frame_unknown_embedding]
                else:
                    sim_anchor = cosine_similarity(frame_unknown_embedding, unknown_anchor_embedding)
                    if sim_anchor >= 0.75:
                        unknown_streak += 1
                        unknown_anchor_embedding = l2_normalize(
                            0.7 * unknown_anchor_embedding + 0.3 * frame_unknown_embedding
                        )
                        unknown_buffer.append(frame_unknown_embedding)
                    else:
                        unknown_anchor_embedding = frame_unknown_embedding
                        unknown_streak = 1
                        unknown_buffer = [frame_unknown_embedding]

                now = time.time()
                can_prompt = (
                    unknown_streak >= min_frames_to_prompt
                    and (now - last_prompt_time >= prompt_cooldown_seconds)
                    and (prompt_thread is None or not prompt_thread.is_alive())
                )
                if can_prompt:
                    candidate_embedding = l2_normalize(np.mean(np.vstack(unknown_buffer), axis=0).astype(np.float32))
                    prompt_thread = threading.Thread(
                        target=_prompt_enroll_name,
                        args=(
                            prompt_results,
                            candidate_embedding.copy(),
                            suggested_names,
                            auto_enroll_from_suggestions,
                        ),
                        daemon=True,
                    )
                    prompt_thread.start()
                    # Prevent back-to-back prompts before this result is consumed.
                    last_prompt_time = now
                    unknown_streak = 0
                    unknown_anchor_embedding = None
                    unknown_buffer.clear()

            cv2.imshow("Offline Face Recognition", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fully offline local enrollment/search with face embeddings.")
    parser.add_argument(
        "--metadata",
        default="compiled_models.json",
        help="Ignored in offline mode (kept for CLI compatibility).",
    )
    parser.add_argument(
        "--db",
        default="embeddings_db.npz",
        help="Path to local embedding database (.npz).",
    )
    parser.add_argument(
        "--mode",
        choices=["enroll", "search", "demo", "camera", "assist"],
        default="assist",
        help="Operation mode.",
    )
    parser.add_argument("--name", help="Identity name (required for --mode enroll).")
    parser.add_argument(
        "--input-npy",
        help="Optional .npy input tensor of shape (1, 3, 112, 112). If omitted, random tensor is used.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Cosine threshold below which a face is treated as unknown.",
    )
    parser.add_argument(
        "--second-best-margin",
        type=float,
        default=0.06,
        help="Minimum gap between top-1 and top-2 cosine score to accept a match.",
    )
    parser.add_argument(
        "--face-margin",
        type=float,
        default=0.25,
        help="Extra margin ratio around detected face before embedding extraction.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="OpenCV camera index for --mode camera.",
    )
    parser.add_argument(
        "--video-path",
        help="Optional video file path. If set, --mode camera reads frames from this video instead of webcam.",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=5,
        help="How many consecutive unknown-face frames before asking for a name.",
    )
    parser.add_argument(
        "--intro-text",
        default="",
        help="Optional text where people introduce themselves; used for LLM name suggestions in camera mode.",
    )
    parser.add_argument(
        "--llm-provider",
        choices=["openai", "ollama"],
        default="ollama",
        help="LLM backend provider. Use 'ollama' for local LLM.",
    )
    parser.add_argument(
        "--llm-api-base",
        default="http://localhost:11434",
        help="Base URL for LLM/API calls. For ollama use http://localhost:11434.",
    )
    parser.add_argument(
        "--llm-api-key",
        default="",
        help="API key for LLM name extraction. If empty, regex fallback is used.",
    )
    parser.add_argument(
        "--llm-model",
        default="llama3.2:3b",
        help="Model name for OpenAI-compatible LLM name extraction.",
    )
    parser.add_argument(
        "--accelerator",
        choices=["auto", "gpu", "npu", "directml", "cpu"],
        default="auto",
        help="Face embedding backend: auto, gpu (CUDA/DirectML), npu (Qualcomm), directml (Adreno/AMD/Intel), cpu.",
    )
    args = parser.parse_args()

    if args.mode == "enroll" and not args.name:
        raise ValueError("--name is required in enroll mode.")

    _ = args.metadata  # Kept only for backward-compatible CLI.
    model = load_local_model()
    model_device: object = torch.device("cpu")
    model_backend = "cpu"
    if args.accelerator == "npu":
        try:
            onnx_cache = Path(os.getenv("LOCALAPPDATA", str(Path.cwd()))) / "cavaface_qnn.onnx"
            model = build_qnn_session(model, onnx_cache)
            model_backend = "npu-qnn"
            model_device = "onnxruntime"
        except Exception as exc:
            print(f"NPU backend unavailable, falling back to CPU: {exc}")
            model_device, model_backend = torch.device("cpu"), "cpu"
            model = load_local_model().to(model_device)
    elif args.accelerator == "directml":
        try:
            onnx_cache = Path(os.getenv("LOCALAPPDATA", str(Path.cwd()))) / "cavaface_dml.onnx"
            model = build_dml_session(model, onnx_cache)
            model_backend = "directml"
            model_device = "onnxruntime"
        except Exception as exc:
            print(f"DirectML (Adreno GPU) unavailable, falling back to CPU: {exc}")
            model_device, model_backend = torch.device("cpu"), "cpu"
            model = load_local_model().to(model_device)
    else:
        model_device, model_backend = resolve_torch_device(args.accelerator)
        model = model.to(model_device)
    print(f"Face embedding backend: {model_backend}")
    db_path = Path(args.db)
    suggested_names: list[str] = []
    if args.intro_text:
        if args.llm_provider == "ollama":
            suggested_names = extract_names_with_llm(
                text=args.intro_text,
                api_base=args.llm_api_base,
                api_key="",
                model=args.llm_model,
                provider="ollama",
            )
        elif args.llm_api_key:
            suggested_names = extract_names_with_llm(
                text=args.intro_text,
                api_base=args.llm_api_base,
                api_key=args.llm_api_key,
                model=args.llm_model,
                provider="openai",
            )
        else:
            suggested_names = regex_fallback_extract(args.intro_text)
        if suggested_names:
            print(f"Loaded suggested names from intro text: {', '.join(suggested_names)}")
        else:
            print("No name suggestions detected from intro text.")

    if args.mode in {"camera", "assist"}:
        camera_mode(
            db_path,
            model,
            model_device,
            args.threshold,
            args.second_best_margin,
            args.face_margin,
            args.camera_index,
            args.video_path,
            args.min_frames,
            suggested_names,
            args.mode == "assist",
        )
    elif args.mode == "enroll":
        input_tensor = make_input_tensor(args.input_npy)
        embedding = run_embedding_inference(model, input_tensor, model_device)
        enroll(db_path, args.name, embedding)
    elif args.mode == "search":
        input_tensor = make_input_tensor(args.input_npy)
        embedding = run_embedding_inference(model, input_tensor, model_device)
        search(db_path, embedding)
    else:
        input_tensor = make_input_tensor(args.input_npy)
        embedding = run_embedding_inference(model, input_tensor, model_device)
        print("Demo mode: generated one embedding vector.")
        print(f"Embedding dimension: {embedding.shape[0]}")
        print(f"First 8 values: {embedding[:8]}")


if __name__ == "__main__":
    main()
