import logging
import os
from typing import Iterator

logger = logging.getLogger(__name__)


def _build_messages(system_prompt: str, user_prompt: str) -> str:
    return (
        "You are an assistant in a governance compliance application.\n\n"
        f"System instructions:\n{system_prompt}\n\n"
        f"User request:\n{user_prompt}"
    )


def _get_picoclaw_client():
    api_key = os.getenv("PICOCLAW_API_KEY", "").strip() or None
    from picoclaw import PicoClaw  # type: ignore

    client = None
    errors: list[str] = []

    try:
        client = PicoClaw.remote()
    except Exception as exc:
        errors.append(f"remote() failed: {exc}")

    if client is None and api_key:
        try:
            client = PicoClaw.remote(api_key=api_key)
        except Exception as exc:
            errors.append(f"remote(api_key=...) failed: {exc}")

    if client is None:
        raise RuntimeError("Unable to initialize PicoClaw client. " + " | ".join(errors))
    return client


def call_picoclaw(
    system_prompt: str,
    user_prompt: str,
    model: str = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> str:
    """
    Call PicoClaw in either local mode (no key) or remote mode (key).
    Returns plain text response.
    """
    prompt = _build_messages(system_prompt, user_prompt)
    try:
        client = _get_picoclaw_client()

        # Different PicoClaw versions may expose either ask(...) or run(...).
        if hasattr(client, "ask"):
            response = client.ask(
                prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            response = client.run(prompt)

        return str(response)
    except TypeError:
        # Fallback for minimal ask(...) signatures on older wrappers.
        try:
            client = _get_picoclaw_client()
            response = client.ask(prompt)
            return str(response)
        except Exception as exc:
            logger.error("PicoClaw call failed: %s", exc)
            raise
    except Exception as exc:
        logger.error("PicoClaw call failed: %s", exc)
        raise


def stream_picoclaw(
    system_prompt: str,
    user_prompt: str,
    model: str = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> Iterator[str]:
    """
    Try true streaming if PicoClaw client supports it.
    Falls back to a single response chunk otherwise.
    """
    prompt = _build_messages(system_prompt, user_prompt)
    client = _get_picoclaw_client()

    if hasattr(client, "stream"):
        try:
            for chunk in client.stream(prompt, model=model, temperature=temperature, max_tokens=max_tokens):
                yield str(chunk)
            return
        except Exception as exc:
            logger.warning("PicoClaw stream() failed, falling back to ask(): %s", exc)

    if hasattr(client, "ask"):
        try:
            response = client.ask(
                prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            if hasattr(response, "__iter__") and not isinstance(response, (str, bytes)):
                for chunk in response:
                    yield str(chunk)
                return
        except TypeError:
            pass
        except Exception as exc:
            logger.warning("PicoClaw ask(stream=True) failed, falling back to non-stream: %s", exc)

    yield call_picoclaw(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
