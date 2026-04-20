import gc
import logging
import os
import hashlib
import pickle

from workers.celery_app import celery_app
from workers.job_status import set_status

logger = logging.getLogger(__name__)

def _get_file_hash(file_path: str, filename: str) -> str:
    """Generate SHA256 file hash to skip re-embedding identical files."""
    if file_path.startswith("http://") or file_path.startswith("https://"):
        return hashlib.sha256(f"URL:{file_path}".encode("utf-8")).hexdigest()
    
    if os.path.exists(file_path):
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    return hashlib.sha256(f"UNKNOWN:{filename}".format().encode("utf-8")).hexdigest()


@celery_app.task(name="workers.tasks.run_ingestion_pipeline")
def run_ingestion_pipeline(file_path: str, filename: str, job_id: str, user_id: str = "anonymous"):
    file_hash = _get_file_hash(file_path, filename)
    cache_dir = os.path.join(os.getenv("OUTPUT_DIR", "output"), "_global_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{file_hash}.pkl")

    try:
        from embedding.embedder import embed_chunks
        from retrieval.bm25_store import build_bm25_index
        from vectorstore.chroma_store import upsert_chunks
        import copy

        if os.path.exists(cache_file):
            set_status(job_id, {"status": "processing", "progress": 50, "message": "Using cached document embeddings..."})
            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)
            classified = copy.deepcopy(cached_data["chunks"])
            vectors = cached_data["vectors"]
        else:
            set_status(job_id, {"status": "processing", "progress": 20, "message": "Extracting text..."})
            from ingestion.router import route_file

            blocks = route_file(file_path, "upload", "Unknown")
            if not blocks:
                raise RuntimeError("No content extracted from source.")

            set_status(job_id, {"status": "processing", "progress": 45, "message": "Chunking text..."})
            from chunking.chunker import process_blocks

            chunks = process_blocks(blocks)
            del blocks
            gc.collect()

            set_status(job_id, {"status": "processing", "progress": 70, "message": "Classifying chunks..."})
            from classification.rule_classifier import classify_chunks

            classified = classify_chunks(chunks)
            del chunks
            gc.collect()

            set_status(job_id, {"status": "processing", "progress": 85, "message": "Embedding text..."})
            vectors = embed_chunks(classified)

            with open(cache_file, "wb") as f:
                pickle.dump({"chunks": classified, "vectors": vectors}, f)

        # Make chunks unique to user
        for c in classified:
            c["user_id"] = user_id
            if not c["chunk_id"].startswith(f"{user_id}_"):
                c["chunk_id"] = f"{user_id}_{c['chunk_id']}"

        set_status(job_id, {"status": "processing", "progress": 90, "message": "Indexing data..."})
        upsert_chunks(classified, vectors)
        build_bm25_index(classified, user_id)

        total = len(classified)
        del vectors
        del classified
        gc.collect()

        set_status(job_id, {"status": "completed", "progress": 100, "message": f"Successfully ingested {total} chunks"})
    except Exception as exc:
        logger.exception("Ingestion pipeline failed for job %s", job_id)
        set_status(job_id, {"status": "error", "progress": 0, "message": str(exc)})
        raise
    finally:
        # Uploaded local files are temporary and can be cleaned after processing.
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                logger.warning("Could not remove temporary file: %s", file_path)
