from __future__ import annotations

import os
import socket
from functools import lru_cache
from typing import Any

import numpy as np


DEFAULT_CAPTION_MODEL = "nlpconnect/vit-gpt2-image-captioning"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_VQA_MODEL = "dandelin/vilt-b32-finetuned-vqa"
QNN_PROVIDER = "QNNExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"
_RUNTIME_BACKENDS: dict[str, str] = {
    "semantic": "uninitialized",
    "caption": "uninitialized",
    "vqa": "uninitialized",
}


def _normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def _set_runtime_backend(name: str, backend: str) -> None:
    if name in _RUNTIME_BACKENDS:
        _RUNTIME_BACKENDS[name] = backend


def get_runtime_backends() -> dict[str, str]:
    return dict(_RUNTIME_BACKENDS)


def _prefer_qnn() -> bool:
    raw = str(os.getenv("AURA_PREFER_QNN", "1")).strip().lower()
    return raw not in {"0", "false", "off", "no"}


@lru_cache(maxsize=1)
def _qnn_provider_available() -> bool:
    try:
        import onnxruntime as ort  # type: ignore
    except Exception:
        return False
    try:
        return QNN_PROVIDER in set(ort.get_available_providers())
    except Exception:
        return False


def _ort_provider_attempts() -> list[dict[str, Any]]:
    if _prefer_qnn() and _qnn_provider_available():
        return [
            {"provider": QNN_PROVIDER},
            {"providers": [QNN_PROVIDER, CPU_PROVIDER]},
            {"provider": CPU_PROVIDER},
            {"providers": [CPU_PROVIDER]},
            {},
        ]
    return [
        {"provider": CPU_PROVIDER},
        {"providers": [CPU_PROVIDER]},
        {},
    ]


def _backend_label_from_provider_kwargs(provider_kwargs: dict[str, Any]) -> str:
    provider = provider_kwargs.get("provider")
    providers = provider_kwargs.get("providers")
    if provider == QNN_PROVIDER:
        return "onnx/qnn"
    if isinstance(providers, list) and QNN_PROVIDER in providers:
        return "onnx/qnn"
    if provider == CPU_PROVIDER:
        return "onnx/cpu"
    if isinstance(providers, list) and providers and all(p == CPU_PROVIDER for p in providers):
        return "onnx/cpu"
    return "onnx/default"


def _env_flag(name: str, default: str = "0") -> bool:
    raw = str(os.getenv(name, default)).strip().lower()
    return raw in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def _internet_reachable(timeout_s: float = 0.45) -> bool:
    # Fast connectivity probe to avoid long HF retries when offline.
    targets = [("huggingface.co", 443), ("pypi.org", 443), ("github.com", 443)]
    for host, port in targets:
        try:
            with socket.create_connection((host, port), timeout=timeout_s):
                return True
        except OSError:
            continue
    return False


def _local_only_attempts() -> list[bool]:
    """
    Return model-loading attempt order:
    - [True] when offline is forced or internet appears unreachable.
    - [True, False] when online: prefer cached local artifacts first.
    """
    forced_offline = (
        _env_flag("AURA_FORCE_OFFLINE", "0")
        or _env_flag("HF_HUB_OFFLINE", "0")
        or _env_flag("TRANSFORMERS_OFFLINE", "0")
    )
    if forced_offline:
        return [True]
    return [True, False] if _internet_reachable() else [True]


@lru_cache(maxsize=1)
def _get_embedder(model_name: str = DEFAULT_EMBED_MODEL):
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception:
        _set_runtime_backend("semantic", "unavailable")
        return None

    for local_only in _local_only_attempts():
        kwargs: dict[str, Any] = {"local_files_only": True} if local_only else {}
        try:
            embedder = SentenceTransformer(model_name, **kwargs)
            _set_runtime_backend("semantic", "torch/default")
            return embedder
        except Exception:
            continue

    _set_runtime_backend("semantic", "unavailable")
    return None


def semantic_model_available(model_name: str = DEFAULT_EMBED_MODEL) -> bool:
    return _get_embedder(model_name) is not None


def semantic_similarity_scores(
    query: str,
    texts: list[str],
    model_name: str = DEFAULT_EMBED_MODEL,
) -> list[float]:
    embedder = _get_embedder(model_name)
    if embedder is None or not texts:
        return [0.0 for _ in texts]

    clean_texts = [_normalize_text(t) for t in texts]
    try:
        q_vec = embedder.encode(
            [_normalize_text(query)],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]
        t_vecs = embedder.encode(
            clean_texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
    except Exception:
        return [0.0 for _ in texts]

    try:
        scores = np.dot(t_vecs, q_vec)
    except Exception:
        return [0.0 for _ in texts]
    return [float(x) for x in scores.tolist()]


def _build_ort_pipeline(task: str, model_name: str) -> tuple[Any | None, str | None]:
    try:
        from transformers import AutoProcessor, pipeline  # type: ignore
    except Exception:
        return None, None

    if task == "image-to-text":
        try:
            from optimum.onnxruntime import ORTModelForVision2Seq  # type: ignore
        except Exception:
            return None, None
        ort_model_class = ORTModelForVision2Seq
    elif task == "visual-question-answering":
        try:
            from optimum.onnxruntime import ORTModelForVisualQuestionAnswering  # type: ignore
        except Exception:
            return None, None
        ort_model_class = ORTModelForVisualQuestionAnswering
    else:
        return None, None

    for local_only in _local_only_attempts():
        processor = None
        hf_kwargs: dict[str, Any] = {"local_files_only": True} if local_only else {}
        try:
            processor = AutoProcessor.from_pretrained(model_name, **hf_kwargs)
        except Exception:
            processor = None

        for provider_kwargs in _ort_provider_attempts():
            try:
                model = ort_model_class.from_pretrained(
                    model_name,
                    export=True,
                    **provider_kwargs,
                    **hf_kwargs,
                )
                pipe_kwargs: dict[str, Any] = {"task": task, "model": model}
                if processor is not None:
                    tokenizer = getattr(processor, "tokenizer", None)
                    image_processor = getattr(processor, "image_processor", None)
                    feature_extractor = getattr(processor, "feature_extractor", None)
                    if tokenizer is not None:
                        pipe_kwargs["tokenizer"] = tokenizer
                    if image_processor is not None:
                        pipe_kwargs["image_processor"] = image_processor
                    elif feature_extractor is not None:
                        pipe_kwargs["feature_extractor"] = feature_extractor
                return pipeline(**pipe_kwargs), _backend_label_from_provider_kwargs(provider_kwargs)
            except Exception:
                continue
    return None, None


@lru_cache(maxsize=1)
def _get_caption_pipeline(model_name: str = DEFAULT_CAPTION_MODEL):
    try:
        from transformers import pipeline  # type: ignore
    except Exception:
        _set_runtime_backend("caption", "unavailable")
        return None
    for local_only in _local_only_attempts():
        hf_kwargs: dict[str, Any] = {"local_files_only": True} if local_only else {}
        try:
            pipe = pipeline("image-to-text", model=model_name, **hf_kwargs)
            _set_runtime_backend("caption", "transformers/default")
            return pipe
        except Exception:
            continue
    _set_runtime_backend("caption", "unavailable")
    return None


def caption_image(
    image_path: str,
    model_name: str = DEFAULT_CAPTION_MODEL,
    max_new_tokens: int = 32,
) -> str | None:
    pipe = _get_caption_pipeline(model_name)
    if pipe is None:
        return None

    try:
        out: Any = pipe(image_path, max_new_tokens=max_new_tokens)
    except Exception:
        return None

    if not out:
        return None
    first = out[0] if isinstance(out, list) else out
    if not isinstance(first, dict):
        return None
    text = str(first.get("generated_text", "")).strip()
    return _normalize_text(text) or None


@lru_cache(maxsize=1)
def _get_vqa_pipeline(model_name: str = DEFAULT_VQA_MODEL):
    try:
        from transformers import pipeline  # type: ignore
    except Exception:
        _set_runtime_backend("vqa", "unavailable")
        return None
    for local_only in _local_only_attempts():
        hf_kwargs: dict[str, Any] = {"local_files_only": True} if local_only else {}
        try:
            pipe = pipeline("visual-question-answering", model=model_name, **hf_kwargs)
            _set_runtime_backend("vqa", "transformers/default")
            return pipe
        except Exception:
            continue
    _set_runtime_backend("vqa", "unavailable")
    return None


def visual_qa_model_available(model_name: str = DEFAULT_VQA_MODEL) -> bool:
    return _get_vqa_pipeline(model_name) is not None


def visual_qa_answers(
    question: str,
    image_paths: list[str],
    model_name: str = DEFAULT_VQA_MODEL,
) -> list[dict[str, Any]]:
    pipe = _get_vqa_pipeline(model_name)
    if pipe is None:
        return [{"answer": "", "score": 0.0} for _ in image_paths]

    out: list[dict[str, Any]] = []
    for image_path in image_paths:
        try:
            result: Any = pipe(image=image_path, question=question)
        except Exception:
            out.append({"answer": "", "score": 0.0})
            continue

        entry = result[0] if isinstance(result, list) and result else result
        if not isinstance(entry, dict):
            out.append({"answer": "", "score": 0.0})
            continue
        answer = _normalize_text(str(entry.get("answer", "")))
        try:
            score = float(entry.get("score", 0.0))
        except Exception:
            score = 0.0
        out.append({"answer": answer, "score": score})
    return out
