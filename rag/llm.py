"""Клиент к любому OpenAI-совместимому API (Ollama, vLLM, облачные провайдеры).

LLM_BASE_URL   — например http://localhost:11434/v1 для Ollama
LLM_MODEL      — модель для ответов (по умолчанию qwen2.5:7b)
EMBED_MODEL    — модель эмбеддингов (например bge-m3); без неё поиск гибридный без векторов
LLM_API_KEY    — если нужен

Без LLM_BASE_URL ничего не вызывается: поиск, экстрактивные ответы и карточки
работают на собственных алгоритмах.
"""
import json
import os
import urllib.request


def enabled() -> bool:
    return bool(os.getenv("LLM_BASE_URL"))


def embeddings_enabled() -> bool:
    return enabled() and bool(os.getenv("EMBED_MODEL"))


def _post(path: str, body: dict, timeout=120) -> dict:
    req = urllib.request.Request(
        os.environ["LLM_BASE_URL"].rstrip("/") + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {os.getenv('LLM_API_KEY', 'none')}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def chat(prompt: str, temperature: float = 0.1) -> str:
    body = {"model": os.getenv("LLM_MODEL", "qwen2.5:7b"), "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}]}
    return _post("/chat/completions", body)["choices"][0]["message"]["content"]


def embed(texts: list[str]) -> list[list[float]]:
    out = []
    for i in range(0, len(texts), 64):
        r = _post("/embeddings", {"model": os.environ["EMBED_MODEL"], "input": texts[i:i + 64]})
        out += [d["embedding"] for d in sorted(r["data"], key=lambda d: d["index"])]
    return out
