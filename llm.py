"""Thin wrapper around any OpenAI-compatible chat API (Groq, OpenAI, Ollama)."""
from openai import OpenAI

import config


def is_configured() -> bool:
    return bool(config.LLM_BASE_URL and config.LLM_MODEL and config.LLM_API_KEY)


def chat(system: str, user: str) -> str:
    client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY, timeout=60)
    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    content = response.choices[0].message.content
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("The language model returned an empty answer.")
    return content.strip()
