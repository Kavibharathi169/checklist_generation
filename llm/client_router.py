import logging
import os
from typing import Iterator

from llm.groq_client import call_groq
from llm.picoclaw_client import call_picoclaw

logger = logging.getLogger(__name__)


def _provider() -> str:
    return os.getenv("LLM_PROVIDER", "groq").strip().lower()


def call_llm(
    system_prompt: str,
    user_prompt: str,
    model: str = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> str:
    provider = _provider()
    if provider == "picoclaw":
        return call_picoclaw(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    return call_groq(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def stream_llm_fallback(response_text: str, chunk_size: int = 40) -> Iterator[str]:
    """
    Yield fixed-size chunks to simulate streaming for non-streaming providers.
    """
    for i in range(0, len(response_text), chunk_size):
        yield response_text[i : i + chunk_size]
