import json
import logging
import os
import re
from dotenv import load_dotenv
from classification.rule_classifier import get_valid_domains

from llm.agentic_framework import generate_with_agentic_loop
from llm.picoclaw_client import stream_picoclaw
from llm.prompt_templates import (
    CHECKLIST_SYSTEM_PROMPT,
    CHECKLIST_USER_PROMPT
)

load_dotenv()

logger = logging.getLogger(__name__)

# â”€â”€ Allowed domains â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€      
VALID_DOMAINS = get_valid_domains()

SAFE_FALLBACK_ANSWER = "I cannot verify this from the provided documents."
PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}
VALID_ACTION_TYPES = {
    "policy",
    "process",
    "technical",
    "training",
    "monitoring",
    "reporting",
    "governance",
    "documentation",
}
GENERIC_EVIDENCE = {
    "documentation or log review.",
    "documentation or log review",
    "documentation",
    "n/a",
}


def _extract_context_chunk_ids(context: str) -> set[str]:
    return set(re.findall(r"chunk_id:\s*([^\n\r]+)", context))


def _extract_answer_citations(answer: str) -> set[str]:
    return set(re.findall(r"\[chunk_id:([^\]]+)\]", answer))


def _is_answer_grounded(answer: str, context: str) -> bool:
    context_ids = _extract_context_chunk_ids(context)
    if not context_ids:
        return False
    cited_ids = {c.strip() for c in _extract_answer_citations(answer)}
    if not cited_ids:
        return False
    return len(cited_ids.intersection(context_ids)) > 0


def _parse_json_response(raw: str) -> list[dict]:
    """
    Parse JSON array from LLM response.
    Handles cases where LLM adds extra text around JSON.
    """
    # Try direct parse first
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            for key in ("items", "checklist", "data"):
                if isinstance(parsed.get(key), list):
                    return parsed[key]
    except json.JSONDecodeError:
        pass

    # Try to extract JSON array from response
    try:
        start = raw.find("[")
        end   = raw.rfind("]") + 1
        if start != -1 and end > start:
            parsed = json.loads(raw[start:end])
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                for key in ("items", "checklist", "data"):
                    if isinstance(parsed.get(key), list):
                        return parsed[key]
    except json.JSONDecodeError:
        pass

    logger.warning("Could not parse JSON from LLM response")
    return []


def _normalize_priority(priority: str) -> str:
    p = str(priority or "").strip().lower()
    if p in ("critical", "urgent", "high"):
        return "High"
    if p in ("moderate", "medium"):
        return "Medium"
    if p in ("low", "minor"):
        return "Low"
    return "Medium"


def _normalize_action_type(action_type: str) -> str:
    a = str(action_type or "").strip().lower()
    if not a:
        return "Process"
    aliases = {
        "tech": "technical",
        "security": "technical",
        "procedure": "process",
        "ops": "process",
        "audit": "monitoring",
        "recordkeeping": "documentation",
        "records": "documentation",
    }
    a = aliases.get(a, a)
    if a not in VALID_ACTION_TYPES:
        return "Process"
    return a.capitalize()


def _clean_requirement_text(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    text = re.sub(r"^[\-\*\d\.\)\s]+", "", text)
    return text


def _best_evidence(item: dict) -> str:
    evidence = str(item.get("evidence_required", "")).strip()
    if not evidence or evidence.lower() in GENERIC_EVIDENCE:
        section = str(item.get("source_section", "")).strip()
        if section and section not in ("—", "-"):
            return f"Documented proof from '{section}' and related logs/records."
        return "Documented proof and related logs/records."
    return evidence


def _validate_items(items: list[dict]) -> list[dict]:
    """
    Validate and clean checklist items.
    Ensures all required fields exist.
    """
    valid = []
    for item in items:
        if not isinstance(item, dict):
            continue

        text = _clean_requirement_text(item.get("item", ""))
        if not text:
            continue

        domain = item.get("domain", "").strip().lower()
        if domain not in VALID_DOMAINS:
            domain = "audit_compliance"

        valid.append({
            "item"               : text,
            "domain"             : domain,
            "source_section"     : item.get("source_section", "—"),
            "priority"           : _normalize_priority(item.get("priority", "Medium")),
            "action_type"        : _normalize_action_type(item.get("action_type", "Process")),
            "evidence_required"  : _best_evidence(item),
            "source_url"         : "",
            "chunk_id"           : str(item.get("chunk_id", "")),
            "compliance_framework": ""
        })

    return valid


def _enforce_grounded_checklist(items: list[dict], results: list[dict]) -> list[dict]:
    """
    Keep only items that trace to retrieved chunks.
    """
    valid_chunk_ids = set()
    section_titles = []
    for r in results:
        cid = r.get("chunk_id") or r.get("metadata", {}).get("chunk_id", "")
        if cid:
            valid_chunk_ids.add(str(cid))
        section = (r.get("metadata", {}) or {}).get("section_title", "")
        if section:
            section_titles.append(section.lower())

    grounded = []
    for item in items:
        cid = str(item.get("chunk_id", "")).strip()
        source_section = str(item.get("source_section", "")).lower()
        has_valid_chunk = cid in valid_chunk_ids if cid else False
        has_valid_section = any(source_section and source_section in s for s in section_titles)
        if has_valid_chunk or has_valid_section:
            grounded.append(item)
    return grounded


def _enrich_with_metadata(
    items  : list[dict],
    results: list[dict]
) -> list[dict]:
    # Build lookup: chunk_id -> metadata
    chunk_map = {}
    for r in results:
        cid = r.get("chunk_id") or r.get("metadata", {}).get("chunk_id", "")
        meta = r.get("metadata", {})
        if cid:
            chunk_map[cid] = meta

    for item in items:
        cid = item.get("chunk_id", "")
        meta = chunk_map.get(cid)

        if meta:
            item["source_url"]           = meta.get("source_url","")
            item["compliance_framework"] = meta.get("compliance_framework","")
        else:
            source_section = item.get("source_section", "")
            for rcid, rmeta in chunk_map.items():
                sec = rmeta.get("section_title", "")
                if sec and (source_section.lower() in sec.lower() or sec.lower() in source_section.lower()):
                    item["source_url"]           = rmeta.get("source_url","")
                    item["chunk_id"]             = rcid
                    item["compliance_framework"] = rmeta.get("compliance_framework","")
                    break

    return items


def _dedupe_and_rank_checklist(items: list[dict], results: list[dict]) -> list[dict]:
    chunk_score = {}
    for r in results:
        cid = str(r.get("chunk_id") or r.get("metadata", {}).get("chunk_id", "")).strip()
        if cid:
            chunk_score[cid] = float(r.get("hybrid_score", r.get("score", 0.0)) or 0.0)

    dedup = {}
    for item in items:
        norm_item = re.sub(r"\W+", " ", item.get("item", "").lower()).strip()
        key = (norm_item, str(item.get("chunk_id", "")).strip())
        if not norm_item:
            continue

        cur_score = chunk_score.get(str(item.get("chunk_id", "")).strip(), 0.0)
        item["retrieval_score"] = cur_score
        prev = dedup.get(key)
        if prev is None or cur_score > prev.get("retrieval_score", 0.0):
            dedup[key] = item

    ranked = list(dedup.values())
    ranked.sort(
        key=lambda x: (
            PRIORITY_ORDER.get(str(x.get("priority", "")).lower(), 1),
            -float(x.get("retrieval_score", 0.0)),
            x.get("domain", ""),
        )
    )

    max_items = max(10, int(os.getenv("CHECKLIST_MAX_ITEMS", "120")))
    return ranked[:max_items]


def generate_checklist(
    context_string: str,
    results       : list[dict]
) -> list[dict]:
    """
    Generate a structured governance checklist
    from retrieved chunks using Groq API.
    """
    if not context_string or not context_string.strip():
        logger.warning(
            "generate_checklist called with empty context"
        )
        return []

    logger.info("Starting checklist generation via Groq API")

    user_prompt = CHECKLIST_USER_PROMPT.format(
        context=context_string
    )

    # ACCURACY FIX: Lowered temperature to 0.0 (from 0.3) to force purely deterministic, factual outputs.
    # High temperatures allow the model to get "creative", which leads to hallucinations in legal/compliance checks.
    logger.info("Attempt 1: Generating checklist with 0.0 temperature for maximum accuracy...")
    try:
        raw = generate_with_agentic_loop(
            system_prompt = CHECKLIST_SYSTEM_PROMPT,
            user_prompt   = user_prompt,
            temperature   = 0.0,  # Zero temperature for deterministic facts
            max_tokens    = 3000,  # Increased token limit so it doesn't arbitrarily cut off long checklists
            expects_json_array = True,
        )
    except Exception as e:
        logger.error(f"Groq API call failed: {e}")
        return []

    items = _parse_json_response(raw)

    if not items:
        logger.warning("Attempt 1 failed. Retrying with stricter prompt...")
        strict_prompt = (
            user_prompt
            + "\n\nCRITICAL: Return ONLY a raw JSON array. "
            "Start with [ and end with ]. "
            "No markdown. No explanation. "
            "No text before or after the array."
        )
        try:
            raw = generate_with_agentic_loop(
                system_prompt = CHECKLIST_SYSTEM_PROMPT,
                user_prompt   = strict_prompt,
                temperature   = 0.0,
                max_tokens    = 3000,
                expects_json_array = True,
            )
            items = _parse_json_response(raw)
        except Exception as e:
            logger.error(f"Retry failed: {e}")
            return []

    if not items:
        return []

    items = _validate_items(items)
    items = _enrich_with_metadata(items, results)
    items = _enforce_grounded_checklist(items, results)
    items = _dedupe_and_rank_checklist(items, results)

    return items


def generate_answer(
    query  : str,
    context: str,
    chat_history: list[dict] | None = None,
) -> str:
    from llm.prompt_templates import (
        ANSWER_SYSTEM_PROMPT,
        ANSWER_USER_PROMPT
    )

    if not query or not context:
        return "Insufficient context to answer this question."

    history_text = ""
    if chat_history:
        recent = chat_history[-8:]
        lines = []
        for turn in recent:
            role = str(turn.get("role", "user")).lower()
            content = str(turn.get("content", "")).strip()
            if content:
                lines.append(f"{role}: {content}")
        if lines:
            history_text = "\n\nConversation history:\n" + "\n".join(lines)

    user_prompt = ANSWER_USER_PROMPT.format(
        query   = query,
        context = context
    ) + history_text

    try:
        # ACCURACY FIX for Q&A Generation
        answer = generate_with_agentic_loop(
            system_prompt = ANSWER_SYSTEM_PROMPT,
            user_prompt   = user_prompt,
            temperature   = 0.0, # Lock to zero to avoid Q&A hallucination
            max_tokens    = 1500
        )
        if not _is_answer_grounded(answer, context):
            logger.warning("Answer failed grounding checks; returning safe fallback.")
            return SAFE_FALLBACK_ANSWER
        return answer
    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        return f"Could not generate answer: {e}"

def stream_answer(
    query  : str,
    context: str,
    chat_history: list[dict] | None = None,
):
    from llm.prompt_templates import (
        ANSWER_SYSTEM_PROMPT,
        ANSWER_USER_PROMPT
    )
    provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()

    if not query or not context:
        yield "Insufficient context to answer this question."
        return

    history_text = ""
    if chat_history:
        recent = chat_history[-8:]
        lines = []
        for turn in recent:
            role = str(turn.get("role", "user")).lower()
            content = str(turn.get("content", "")).strip()
            if content:
                lines.append(f"{role}: {content}")
        if lines:
            history_text = "\n\nConversation history:\n" + "\n".join(lines)

    user_prompt = ANSWER_USER_PROMPT.format(
        query   = query,
        context = context
    ) + history_text
    
    model = os.getenv("GROQ_GENERATOR_MODEL", "llama-3.3-70b-versatile")

    try:
        if provider == "groq":
            from llm.groq_client import get_groq_client

            client = get_groq_client()
            response = client.chat.completions.create(
                model       = model,
                messages    = [
                    {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature = 0.0,
                max_tokens  = 1500,
                stream      = True
            )
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
            return

        # PicoClaw path: prefer native streaming if available.
        for chunk in stream_picoclaw(
            system_prompt=ANSWER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model=model,
            temperature=0.0,
            max_tokens=1500,
        ):
            yield chunk
    except Exception as e:
        logger.error(f"Answer stream failed: {e}")
        yield f"Could not generate answer stream: {e}"
