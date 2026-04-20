import hashlib
import logging
import json
import os
import re
import numpy as np
from dotenv import load_dotenv
from chunking.token_utils import count_tokens

load_dotenv()

logger = logging.getLogger(__name__)
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output")

MAX_TOKENS = int(os.getenv("CHUNK_MAX_TOKENS", "512"))
# Cosine similarity drop threshold to detect a semantic boundary (0–1; lower = more splits)
SEMANTIC_THRESHOLD = float(os.getenv("SEMANTIC_THRESHOLD", "0.3"))



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
    return chunk

def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on '.', '!', '?' followed by whitespace."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def _semantic_chunking(blocks: list[dict], source_format: str) -> list[dict]:
    """
    True semantic chunking: embed every sentence, detect topic-shift breakpoints
    via cosine-similarity drops, then group sentences into token-bounded chunks.
    """
    from embedding.embedder import get_model

    # 1. Collect all sentences with their originating block
    sentences, sent_blocks = [], []
    for block in blocks:
        text = block.get("text", "").strip()
        if not text:
            continue
        for s in _split_sentences(text):
            sentences.append(s)
            sent_blocks.append(block)

    if not sentences:
        return []

    # 2. Embed all sentences (reuse the already-loaded model)
    model = get_model()
    vecs = model.encode(
        ["passage: " + s for s in sentences],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )  # shape: (N, dim)

    # 3. Compute cosine similarity between consecutive sentences
    #    (vectors are already L2-normalised, so dot product == cosine sim)
    similarities = [
        float(np.dot(vecs[i], vecs[i + 1])) for i in range(len(vecs) - 1)
    ]

    # 4. Group sentences into chunks: start a new chunk when similarity drops
    #    below threshold OR the current group would exceed MAX_TOKENS.
    all_chunks, chunk_index = [], 0
    group, group_block = [sentences[0]], sent_blocks[0]

    def _flush(group, group_block):
        nonlocal chunk_index
        text = " ".join(group)
        all_chunks.append(_build_chunk(text, group_block, chunk_index, source_format))
        chunk_index += 1

    for i, sim in enumerate(similarities):
        next_sent = sentences[i + 1]
        candidate = " ".join(group + [next_sent])
        is_boundary = sim < (1.0 - SEMANTIC_THRESHOLD)
        over_budget = count_tokens(candidate) > MAX_TOKENS

        if is_boundary or over_budget:
            _flush(group, group_block)
            group = [next_sent]
            group_block = sent_blocks[i + 1]
        else:
            group.append(next_sent)

    if group:
        _flush(group, group_block)

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