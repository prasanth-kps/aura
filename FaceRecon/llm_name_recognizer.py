import argparse
import json
import os
import re
import urllib.error
import urllib.request
from typing import List


SYSTEM_PROMPT = (
    "You extract only person names from user text. "
    "Return strict JSON with this shape: {\"names\": [\"Name 1\", \"Name 2\"]}. "
    "If no names exist, return {\"names\": []}. "
    "Do not include extra keys or explanations."
)


def _normalize_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z\-\s']", "", name).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if not cleaned:
        return ""
    return " ".join(part.capitalize() for part in cleaned.split(" "))


def regex_fallback_extract(text: str) -> List[str]:
    patterns = [
        r"\bmy name is\s+([A-Za-z][A-Za-z\-\s']{0,40})",
        r"\bi am\s+([A-Za-z][A-Za-z\-\s']{0,40})",
        r"\bi'm\s+([A-Za-z][A-Za-z\-\s']{0,40})",
        r"\bthis is\s+([A-Za-z][A-Za-z\-\s']{0,40})",
    ]
    found: List[str] = []
    for pattern in patterns:
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            # Truncate at likely phrase boundaries
            name = re.split(r"\b(and|from|working|here|nice|hello|hi)\b", match, maxsplit=1, flags=re.IGNORECASE)[0]
            normalized = _normalize_name(name)
            if normalized and normalized not in found:
                found.append(normalized)
    return found


def _chat_openai_compatible(
    api_base: str,
    api_key: str,
    model: str,
    user_text: str,
    timeout_sec: int = 45,
) -> str:
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
    }
    req = urllib.request.Request(
        url=f"{api_base.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def _chat_ollama(
    ollama_url: str,
    model: str,
    user_text: str,
    timeout_sec: int = 45,
) -> str:
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        "format": "json",
        "options": {"temperature": 0},
    }
    req = urllib.request.Request(
        url=f"{ollama_url.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["message"]["content"]


def _parse_llm_json(raw: str) -> List[str]:
    raw = raw.strip()
    # Handle accidental markdown fences
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    obj = json.loads(raw)
    names = obj.get("names", [])
    cleaned: List[str] = []
    for n in names:
        normalized = _normalize_name(str(n))
        if normalized and normalized not in cleaned:
            cleaned.append(normalized)
    return cleaned


def extract_names_with_llm(
    text: str,
    api_base: str,
    api_key: str,
    model: str,
    provider: str = "openai",
) -> List[str]:
    try:
        if provider == "ollama":
            raw = _chat_ollama(ollama_url=api_base, model=model, user_text=text)
        else:
            raw = _chat_openai_compatible(api_base=api_base, api_key=api_key, model=model, user_text=text)
        names = _parse_llm_json(raw)
        if names:
            return names
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, ValueError, json.JSONDecodeError):
        pass
    return regex_fallback_extract(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recognize person names from text using an LLM.")
    parser.add_argument("--text", required=True, help="Input text where people introduce themselves.")
    parser.add_argument(
        "--provider",
        choices=["openai", "ollama"],
        default=os.getenv("LLM_PROVIDER", "openai"),
        help="LLM backend provider.",
    )
    parser.add_argument(
        "--api-base",
        default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        help="API base URL. For ollama use http://localhost:11434.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY", ""),
        help="API key for OpenAI-compatible endpoint. If missing, regex fallback is used.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        help="Model name. For ollama use a local model like llama3.2:3b.",
    )
    args = parser.parse_args()

    if args.provider == "ollama":
        names = extract_names_with_llm(
            text=args.text,
            api_base=args.api_base,
            api_key="",
            model=args.model,
            provider="ollama",
        )
    elif args.api_key:
        names = extract_names_with_llm(
            text=args.text,
            api_base=args.api_base,
            api_key=args.api_key,
            model=args.model,
            provider="openai",
        )
    else:
        names = regex_fallback_extract(args.text)

    print(json.dumps({"names": names}, ensure_ascii=True))


if __name__ == "__main__":
    main()
