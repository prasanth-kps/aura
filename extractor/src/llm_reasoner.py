from __future__ import annotations

import json
import os
from typing import Any
from urllib import error, request


def _default_base_url() -> str:
    return os.getenv("AURA_LLM_BASE_URL", "https://api.openai.com/v1")


def _default_model() -> str:
    return os.getenv("AURA_LLM_MODEL", "gpt-4o-mini")


def _default_api_key() -> str | None:
    return os.getenv("AURA_LLM_API_KEY")


def _build_messages(question: str, evidence: list[dict[str, Any]]) -> list[dict[str, str]]:
    lines: list[str] = []
    for idx, ev in enumerate(evidence, start=1):
        lines.append(
            f"{idx}) second={ev.get('video_second')}, label={ev.get('label')}, "
            f"color={ev.get('detected_color')}, context={ev.get('context')}, "
            f"confidence={ev.get('confidence')}, frame_path={ev.get('frame_path')}, "
            f"thumbnail_path={ev.get('thumbnail_path')}"
        )
    evidence_block = "\n".join(lines) if lines else "(no evidence)"

    system_prompt = (
        "You are a grounded memory assistant. Answer strictly using the provided evidence rows. "
        "If evidence is insufficient, say so explicitly. "
        "Keep answer concise (2-4 sentences) and include at least one timestamp/second reference."
    )
    user_prompt = (
        f"Question: {question}\n\n"
        f"Evidence rows:\n{evidence_block}\n\n"
        "Provide a direct answer, then list 1-3 supporting evidence bullets."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def synthesize_answer(
    question: str,
    evidence: list[dict[str, Any]],
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout_seconds: float = 20.0,
) -> dict[str, Any]:
    chosen_model = model or _default_model()
    chosen_base_url = (base_url or _default_base_url()).rstrip("/")
    chosen_api_key = api_key or _default_api_key()

    if not chosen_api_key:
        return {
            "ok": False,
            "error": "Missing API key. Set AURA_LLM_API_KEY or provide one in the UI/CLI.",
        }

    payload = {
        "model": chosen_model,
        "messages": _build_messages(question, evidence),
        "temperature": 0.2,
    }
    data = json.dumps(payload).encode("utf-8")

    req = request.Request(
        url=f"{chosen_base_url}/chat/completions",
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {chosen_api_key}",
        },
    )

    try:
        with request.urlopen(req, timeout=timeout_seconds) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {"ok": False, "error": f"HTTP {exc.code}: {body[:400]}"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"LLM request failed: {exc}"}

    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        return {"ok": False, "error": "LLM returned non-JSON response."}

    choices = obj.get("choices") or []
    if not choices:
        return {"ok": False, "error": "LLM response contained no choices."}
    message = choices[0].get("message") or {}
    content = str(message.get("content", "")).strip()
    if not content:
        return {"ok": False, "error": "LLM response content was empty."}

    return {
        "ok": True,
        "answer": content,
        "model": chosen_model,
        "base_url": chosen_base_url,
    }
