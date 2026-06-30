"""Генерация ответа через любой OpenAI-совместимый API (Ollama, vLLM, облако).

Если переменная LLM_BASE_URL не задана, LLM не вызывается и пользователь
получает найденные фрагменты конспекта. Поиск полезен и сам по себе.
"""
import json
import os
import urllib.request

from .chunk import Chunk

PROMPT = """Ты помогаешь студенту готовиться к экзамену.
Отвечай ТОЛЬКО по фрагментам конспекта ниже. Если ответа в них нет — так и скажи.
После каждого утверждения ставь номер фрагмента в квадратных скобках.

{context}

Вопрос: {question}"""


def build_prompt(question: str, chunks: list[Chunk]) -> str:
    context = "\n\n".join(f"[{i + 1}] ({c.section})\n{c.text}" for i, c in enumerate(chunks))
    return PROMPT.format(context=context, question=question)


def ask(question: str, chunks: list[Chunk]) -> str | None:
    base = os.getenv("LLM_BASE_URL")  # например http://localhost:11434/v1 для Ollama
    if not base:
        return None
    body = {
        "model": os.getenv("LLM_MODEL", "qwen2.5:7b"),
        "messages": [{"role": "user", "content": build_prompt(question, chunks)}],
        "temperature": 0.1,
    }
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {os.getenv('LLM_API_KEY', 'none')}"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["choices"][0]["message"]["content"]
