import hashlib
import logging
import json
import os
import re
import numpy as np
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chunking.token_utils import count_tokens, tiktoken_length

load_dotenv()

logger = logging.getLogger(__name__)
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output")

MAX_TOKENS = int(os.getenv("CHUNK_MAX_TOKENS", "512"))
OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))
if OVERLAP >= MAX_TOKENS:
    OVERLAP = max(0, MAX_TOKENS // 8)
SEPARATORS = ["\n\n", "\n", ".", "!", "?", " ", ""]


def _get_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size       = MAX_TOKENS,
        chunk_overlap    = OVERLAP,
        length_function  = tiktoken_length,
        separators       = SEPARATORS,
    )

def _cosine_sim(u, v) -> float:
    # SentenceTransformer vectors are normalized in our embedder, so dot ~= cosine.
    try:
        return float(u @ v)
    except Exception:
        # Fallback for list types
        s = 0.0
        for a, b in zip(u, v):
            s += float(a) * float(b)
        return s

def _split_into_units(text: str) -> list[str]:
    """
    Split into semantic-ish units without losing wording.
    We prefer paragraphs; if a paragraph is huge, keep it as-is and let token limits enforce boundaries.
    """
    if not text or not text.strip():
        return []
    parts = [p.strip() for p in text.split("\n\n") if p and p.strip()]
    return parts if parts else [text.strip()]

_HEADING_RE = re.compile(r"^\s*((?:\d+\.)+\d+|\d+)(?:\s*[:.)-]|\s+)(.{0,120})\s*$")

def _extract_section_heading(text: str) -> str:
    """
    Best-effort heading extraction.
    Looks at the first few non-empty lines for patterns like:
    - "3.1 Access Control"
    - "4.2: Logging"
    - "12) Incident Response"
    """
    if not text or not text.strip():
        return ""
    lines = [ln.strip() for ln in text.splitlines() if ln and ln.strip()]
    for ln in lines[:6]:
        m = _HEADING_RE.match(ln)
        if m:
            prefix = (m.group(1) or "").strip()
            title = (m.group(2) or "").strip()
            if title:
                return f"{prefix} {title}".strip()
            return prefix
    return ""

def _detect_is_table(text: str) -> int:
    if not text or not text.strip():
        return 0
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 3:
        return 0
    pipe_lines = sum(1 for ln in lines if ln.count("|") >= 2)
    tab_lines = sum(1 for ln in lines if "\t" in ln)
    # Heuristic: repeated column separators across several lines
    if pipe_lines >= 3 or tab_lines >= 3:
        return 1
    return 0

_ENTITY_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("GDPR", re.compile(r"\bGDPR\b", re.IGNORECASE)),
    ("DPA", re.compile(r"\bdata protection (act|authority)\b", re.IGNORECASE)),
    ("ISO 27001", re.compile(r"\bISO\s*27001\b", re.IGNORECASE)),
    ("SOC 2", re.compile(r"\bSOC\s*2\b", re.IGNORECASE)),
    ("SOX", re.compile(r"\bSOX\b|\bSarbanes[-\s]?Oxley\b", re.IGNORECASE)),
    ("NIST", re.compile(r"\bNIST\b", re.IGNORECASE)),
    ("PCI DSS", re.compile(r"\bPCI\s*DSS\b", re.IGNORECASE)),
    ("HIPAA", re.compile(r"\bHIPAA\b", re.IGNORECASE)),
    ("Article", re.compile(r"\bArticle\s+\d{1,3}\b", re.IGNORECASE)),
]

def _extract_entities(text: str) -> str:
    if not text or not text.strip():
        return ""
    found: list[str] = []
    for label, pat in _ENTITY_PATTERNS:
        if pat.search(text):
            found.append(label)
    # stable, deduped
    out = sorted(set(found))
    return ", ".join(out)

def _save_chunks(chunks: list[dict]) -> None:
    if os.getenv("SAVE_CHUNK_JSONL", "false").lower() not in ("1", "true", "yes"):
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "chunked.jsonl")
    try:
        with open(output_path, "a", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(chunks)} chunks to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save chunks: {e}")

def _build_chunk(text: str, base_block: dict, chunk_index: int, source_format: str) -> dict:
    chunk = dict(base_block)
    base_id = base_block.get("chunk_id", "")
    unique_id = hashlib.sha256(f"{base_id}_{chunk_index}_{text[:32]}".encode("utf-8")).hexdigest()[:16]
    chunk["chunk_id"]      = unique_id
    chunk["text"]          = text
    chunk["char_count"]    = len(text)
    chunk["token_count"]   = count_tokens(text)
    chunk["chunk_index"]   = chunk_index
    chunk["source_format"] = source_format
    # Enrichment for retrieval + citations
    chunk["page_number"]   = base_block.get("page_number")
    chunk["section_heading"] = (
        base_block.get("section_heading")
        or _extract_section_heading(text)
        or base_block.get("section_title", "")
    )
    chunk["is_table"]      = _detect_is_table(text)
    chunk["named_entities"] = _extract_entities(text)
    return chunk

def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on '.', '!', '?' followed by whitespace."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def _semantic_chunking(blocks: list[dict], source_format: str) -> list[dict]:
    """
    Merge small ingestion blocks up to MAX_TOKENS, then split with overlap.
    Uses the same token counter as the splitter so boundaries stay consistent.
    Joins blocks with newlines only (no extra spaces) to preserve source wording.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_TOKENS,
        chunk_overlap=OVERLAP,
        length_function=tiktoken_length,
        separators=SEPARATORS,
    )
    all_chunks = []
    chunk_index = 0

    buffer_text = ""
    buffer_block = blocks[0] if blocks else {}

    for block in blocks:
        text = block.get("text", "").strip()
        if not text:
            continue

        combined = f"{buffer_text}\n\n{text}" if buffer_text else text

        if count_tokens(combined) > MAX_TOKENS:
            if buffer_text.strip():
                try:
                    sub_texts = splitter.split_text(buffer_text)
                except Exception:
                    sub_texts = [buffer_text]
                
                for sub in sub_texts:
                    if sub.strip():
                        all_chunks.append(_build_chunk(sub, buffer_block, chunk_index, source_format))
                        chunk_index += 1
            buffer_text = text
            buffer_block = block
        else:
            buffer_text = combined

    # Flush the remaining semantic buffer safely
    if buffer_text.strip():
        try:
            sub_texts = splitter.split_text(buffer_text)
        except Exception:
            sub_texts = [buffer_text]
            
        for sub in sub_texts:
            if sub.strip():
                all_chunks.append(_build_chunk(sub, buffer_block, chunk_index, source_format))
                chunk_index += 1

        return all_chunks

def process_blocks(blocks: list[dict]) -> list[dict]:
    if not blocks:
        return []

    # Detect source format for metadata only
    first_url = blocks[0].get("source_url", "").lower()
    if first_url.startswith("http"):
        source_format = "web"
    elif first_url.endswith(".docx") or first_url.endswith(".doc"):
        source_format = "word"
    elif first_url.endswith(".xlsx") or first_url.endswith(".xls") or first_url.endswith(".csv"):
        source_format = "spreadsheet"
    elif first_url.endswith(".pdf"):
        source_format = "pdf"
    elif first_url.endswith(".png") or first_url.endswith(".jpg") or first_url.endswith(".jpeg"):
        source_format = "image"
    else:
        source_format = "txt"

    # Unified semantic chunking strategy for all documents
    chunks = _semantic_chunking(blocks, source_format)

    _save_chunks(chunks)
    return chunks