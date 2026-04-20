import logging
import os

from llm.client_router import call_llm

logger = logging.getLogger(__name__)


def _agentic_enabled() -> bool:
    """
    Enable agentic workflow only when explicitly requested.
    """
    provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()
    enabled = os.getenv("PICOCLAW_AGENTIC_ENABLED", "false").strip().lower() == "true"
    return provider == "picoclaw" and enabled


def generate_with_agentic_loop(
    system_prompt: str,
    user_prompt: str,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    expects_json_array: bool = False,
) -> str:
    """
    Run an agentic loop for PicoClaw (plan -> execute -> verify -> optional revise).
    Falls back to a single standard generation when not enabled.
    """
    if not _agentic_enabled():
        return call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    logger.info("PicoClaw agentic loop enabled for response generation.")

    planner_prompt = (
        "Create a short reasoning plan (3-6 bullets) to answer the request using only the "
        "provided context. Do not produce the final answer yet."
        f"\n\nSYSTEM INSTRUCTIONS:\n{system_prompt}\n\nUSER REQUEST:\n{user_prompt}"
    )
    plan = call_llm(
        system_prompt="You are a precise planning assistant.",
        user_prompt=planner_prompt,
        model=model,
        temperature=0.0,
        max_tokens=700,
    )

    output_constraints = (
        "Return ONLY a raw JSON array. Start with [ and end with ]."
        if expects_json_array
        else "Return the final user-facing answer directly."
    )

    execute_prompt = (
        f"Use this plan:\n{plan}\n\n"
        f"Now produce the final output.\n{output_constraints}\n\n"
        f"SYSTEM INSTRUCTIONS:\n{system_prompt}\n\nUSER REQUEST:\n{user_prompt}"
    )
    draft = call_llm(
        system_prompt=system_prompt,
        user_prompt=execute_prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    verifier_prompt = (
        "Review the candidate answer for instruction-following, factual grounding in context, "
        "and output format compliance. Reply with exactly one of:\n"
        "1) APPROVED\n"
        "2) REVISE: <single concise fix instruction>\n\n"
        f"FORMAT REQUIREMENT: {output_constraints}\n\n"
        f"ORIGINAL USER REQUEST:\n{user_prompt}\n\n"
        f"CANDIDATE ANSWER:\n{draft}"
    )
    verdict = call_llm(
        system_prompt="You are a strict quality reviewer.",
        user_prompt=verifier_prompt,
        model=model,
        temperature=0.0,
        max_tokens=200,
    )

    if verdict.strip().upper().startswith("APPROVED"):
        return draft

    feedback = verdict.strip()
    revise_prompt = (
        f"Revise the answer using this reviewer feedback:\n{feedback}\n\n"
        f"Preserve correctness and obey output requirement: {output_constraints}\n\n"
        f"USER REQUEST:\n{user_prompt}\n\n"
        f"CURRENT ANSWER:\n{draft}"
    )
    revised = call_llm(
        system_prompt=system_prompt,
        user_prompt=revise_prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return revised
