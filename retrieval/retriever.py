import logging
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Search Weight Tuning Variables
# ALPHA controls the balance between Vector (Semantic) and BM25 (Keyword) Search.
# ALPHA = 0.7 means 70% Vector and 30% Keyword importance.
# Higher Alpha = better for conceptual questions ("What are the duties?")
# Lower Alpha = better for exact matches ("Section 3.2.1", "ISO 27001")
ALPHA = max(0.0, min(1.0, float(os.getenv("SEARCH_WEIGHT_ALPHA", "0.7"))))
RRF_K = int(os.getenv("SEARCH_RRF_K", "60"))

# Cross-encoder re-ranking disabled — requires a separate model download
# and blocks first query. Hybrid BM25+vector scoring is sufficient.
_reranker = False

def _get_reranker():
    return None


def normalize_scores(results: list[dict], score_key: str = "score", reverse: bool = False) -> list[dict]:
    """Min-Max normalization for scores to combine dense and sparse results."""
    if not results:
        return []
    
    scores = [r.get(score_key, 0) for r in results]
    min_score, max_score = min(scores), max(scores)
    
    for r in results:
        raw_score = r.get(score_key, 0)
        if max_score > min_score:
            r['normalized_score'] = (raw_score - min_score) / (max_score - min_score)
        else:
            r['normalized_score'] = 1.0 # If all scores are the same
            
        # Invert if it's a distance metric (like L2 in vector search)
        if reverse:
            r['normalized_score'] = 1.0 - r['normalized_score']
            
    return results


def _rrf_fuse(
    dense_results: list[dict],
    sparse_results: list[dict],
    alpha: float,
    k: int,
) -> list[dict]:
    """
    Reciprocal Rank Fusion with dense/sparse weighting.
    This avoids brittle score calibration between BM25 and vector scores.
    """
    merged: dict[str, dict] = {}

    for rank, doc in enumerate(dense_results, start=1):
        cid = doc.get("chunk_id") or doc.get("metadata", {}).get("chunk_id", "")
        if not cid:
            continue
        if cid not in merged:
            merged[cid] = {
                "doc": doc,
                "dense_rrf": 0.0,
                "sparse_rrf": 0.0,
            }
        merged[cid]["dense_rrf"] += 1.0 / (k + rank)

    for rank, doc in enumerate(sparse_results, start=1):
        cid = doc.get("chunk_id") or doc.get("metadata", {}).get("chunk_id", "")
        if not cid:
            continue
        if cid not in merged:
            merged[cid] = {
                "doc": doc,
                "dense_rrf": 0.0,
                "sparse_rrf": 0.0,
            }
        merged[cid]["sparse_rrf"] += 1.0 / (k + rank)

    fused = []
    for data in merged.values():
        doc = data["doc"]
        doc["hybrid_score"] = (alpha * data["dense_rrf"]) + ((1.0 - alpha) * data["sparse_rrf"])
        fused.append(doc)

    fused.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
    return fused

def retrieve(
    query         : str,
    content_domain: str = None,
    document_type : str = None,
    source_format : str = None,
    top_k         : int = 5,
    user_id       : str = "anonymous"
) -> list[dict]:
    if not query or not query.strip():
        logger.warning("retrieve called with empty query")
        return []

    from embedding.embedder import embed_query
    from vectorstore.chroma_store import search
    from retrieval.bm25_store import search_bm25

    # Intercept generic checklist prompt and transform it to semantic keywords
    if "Strictly format as JSON" in query:
        search_query = "compliance requirement rule policy obligation procedure control standard"
    else:
        search_query = query

    # Step 2: Build metadata filters
    filters = {}
    if content_domain and content_domain != "all":
        filters["content_domain"] = content_domain
    if document_type and document_type != "all":
        filters["document_type"] = document_type
    if source_format and source_format != "all":
        filters["source_format"] = source_format
    if user_id:
        filters["user_id"] = user_id

    # We fetch more chunks to ensure good overlap for hybrid scoring
    fetch_k = max(top_k * 3, top_k + 5)

    # Step 3: Search ChromaDB (Vector/Dense)
    semantic_results = []
    query_vector = []
    logger.info(f"Embedding query: '{search_query[:60]}'")
    try:
        query_vector = embed_query(search_query)
    except Exception as e:
        logger.error(f"Query embedding failed, continuing with BM25 only: {e}")

    try:
        if query_vector:
            semantic_results = search(query_vector=query_vector, filters=filters, top_k=fetch_k)
        semantic_results = normalize_scores(semantic_results, score_key="score", reverse=False)
    except Exception as e:
        logger.error(f"Vector search failed: {e}")

    # Step 4: Search BM25 (Keyword/Sparse)
    keyword_results = search_bm25(search_query, top_k=fetch_k, user_id=user_id) or []
    # BM25 returns score where higher is better
    keyword_results = normalize_scores(keyword_results, score_key="score", reverse=False)

    # Step 5: Robust hybrid fusion (Weighted RRF)
    unique_results = _rrf_fuse(semantic_results, keyword_results, ALPHA, RRF_K)
    if not unique_results:
        return []

    # Take the best candidates for Re-Ranking
    candidates = unique_results[:top_k * 2]

    # Step 6: Cross-Encoder Re-Ranking (Now optimally cached and scaled)
    reranker = _get_reranker()
    if reranker:
        try:
            pairs = [[search_query, doc.get("text", "")] for doc in candidates]        
            scores = reranker.predict(pairs)

            for doc, score in zip(candidates, scores):
                doc["score"] = float(score)

            candidates.sort(key=lambda x: x["score"], reverse=True)
            results = candidates[:top_k]
        except Exception as e:
            logger.warning(f"Re-ranking failed, falling back to hybrid weights: {e}")
            results = candidates[:top_k]
    else:
        results = candidates[:top_k]

    # Clean up sorting keys
    for r in results:
        r.pop("normalized_score", None)
        r.pop("distance", None)

    logger.info(f"Retrieved {len(results)} results using ALPHA={ALPHA}")  
    return results


def build_context_string(results: list[dict]) -> str:
    if not results:
        return ""

    context_parts = []
    for i, r in enumerate(results, 1):
        meta    = r.get("metadata", {})
        text    = r.get("text", "")
        # score could be hybrid_score or cross-encoder score
        score   = r.get("score", r.get("hybrid_score", 0))
        section = meta.get("section_title", "â€”")
        source  = meta.get("source_url", "â€”")
        domain  = meta.get("content_domain", "â€”")
        cid     = r.get("chunk_id") or meta.get("chunk_id", "")

        context_parts.append(
            f"[Chunk {i}]\n"
            f"chunk_id: {cid}\n"
            f"Section : {section}\n"
            f"Source  : {source}\n"
            f"Domain  : {domain}\n"
            f"Score   : {score:.4f}\n"
            f"Text    : {text}\n"
        )

    return "\n".join(context_parts)


def retrieve_and_format(
    query         : str,
    content_domain: str = None,
    document_type : str = None,
    top_k         : int = 5,
    user_id       : str = "anonymous"
) -> tuple[list[dict], str]:
    results = retrieve(
        query          = query,
        content_domain = content_domain,
        document_type  = document_type,
        top_k          = top_k,
        user_id        = user_id
    )
    context = build_context_string(results)
    return results, context
