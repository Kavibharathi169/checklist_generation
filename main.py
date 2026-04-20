from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
import os
import uuid
from typing import Optional
from celery.result import AsyncResult

from workers.celery_app import celery_app
from workers.job_status import get_status as get_job_status, set_status
from workers.tasks import run_ingestion_pipeline

# Setup directories
os.makedirs("./output", exist_ok=True)

app = FastAPI(title="Governance AI API", version="2.0")

# CORS for frontend decoupling
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import threading

def _prewarm_model():
    try:
        print("Pre-warming embedding model (background thread)...")
        from embedding.embedder import get_model
        get_model()
        print("Embedding model ready.")
    except Exception as e:
        print(f"Warning: Model pre-warm failed: {e}")

# Startup: Init db immediately; warm model in background so server starts fast
@app.on_event("startup")
async def startup_event():
    print("Initializing Database...")
    from storage.db import init_db
    init_db()
    # Warm model in background thread — server accepts requests immediately
    t = threading.Thread(target=_prewarm_model, daemon=True)
    t.start()
    print("Server ready. Model warming up in background...")

class ChatRequest(BaseModel):
    query: str
    domain: Optional[str] = "all"
    user_id: Optional[str] = "anonymous"
    chat_history: Optional[list[dict]] = None

@app.post("/api/upload")
async def upload_file(file: Optional[UploadFile] = File(None), link: Optional[str] = Form(None), user_id: str = Form("anonymous")):
    job_id = str(uuid.uuid4())
    os.makedirs(f"output/{user_id}", exist_ok=True)
    
    if link:
        # Process as a cloud link / webpage
        file_path = link
        filename = link
    elif file:
        # Process as a physical file upload
        file_path = os.path.join(f"output/{user_id}", f"{job_id}_{file.filename}")  
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        filename = file.filename
    else:
        raise HTTPException(status_code=400, detail="Must provide either a file or a link")

    set_status(job_id, {"status": "queued", "progress": 0, "message": "Job queued"})
    run_ingestion_pipeline.apply_async(args=[file_path, filename, job_id, user_id], task_id=job_id)
    return {"job_id": job_id, "message": "Upload successful, queued for processing"}

@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    status_payload = get_job_status(job_id)
    if status_payload:
        return status_payload

    task_result = AsyncResult(job_id, app=celery_app)
    if task_result.state == "PENDING":
        raise HTTPException(status_code=404, detail="Job not found")
    if task_result.state in {"RECEIVED", "STARTED"}:
        return {"status": "processing", "progress": 10, "message": "Worker started processing task"}
    if task_result.state == "SUCCESS":
        return {"status": "completed", "progress": 100, "message": "Task completed"}
    if task_result.state == "FAILURE":
        return {"status": "error", "progress": 0, "message": str(task_result.result)}
    return {"status": "queued", "progress": 0, "message": f"Task state: {task_result.state}"}

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    from retrieval.retriever import retrieve_and_format
    from llm.generator import generate_answer, generate_checklist

    query_lower = request.query.lower()
    is_checklist = "checklist" in query_lower or "map " in query_lower
    
    # EFFICIENCY FIX: Non-blocking threadpool offloaded execution
    # Fetch top 50 chunks for full checklist generation instead of 5
    fetch_amount = 50 if is_checklist else 5
    results, context_string = await run_in_threadpool(
        retrieve_and_format, request.query, request.domain, top_k=fetch_amount, user_id=request.user_id
    )

    if not results or not context_string:
        return {"response": "No relevant context found in documents. Please ensure valid governance documents are uploaded and indexed."}

    if is_checklist:
        checklist_items = await run_in_threadpool(generate_checklist, context_string, results)
        if not checklist_items:
            return {"response": "Failed to generate structured checklist from the context.", "raw_data": None}

        # Format the JSON items into a beautiful markdown string for Streamlit UI
        from storage.db import save_checklist
        save_checklist(request.query, checklist_items, request.user_id)

        md_lines = ["### Compliance Extraction Request\n"]
        for it in checklist_items:
            md_lines.append(f"- **Domain:** {it.get('domain', 'general').capitalize()}\n  **Requirement:** {it.get('item', '')}")
            if it.get("source_section") and it.get("source_section") not in ["—", "â€”"]:  
                md_lines.append(f"  *(Source Section: {it['source_section']})*")

        answer_string = "\n".join(md_lines)
        return {"response": answer_string, "raw_data": checklist_items}
    else:
        # Otherwise, use standard Chat Q&A
        answer_string = await run_in_threadpool(generate_answer, request.query, context_string, request.chat_history)
        return {"response": answer_string, "raw_data": None}

@app.post("/api/chat/stream")
async def chat_endpoint_stream(request: ChatRequest):
    from retrieval.retriever import retrieve_and_format
    from llm.generator import stream_answer
    from fastapi.responses import StreamingResponse

    # EFFICIENCY FIX: Non-blocking threadpool offloaded execution
    results, context_string = await run_in_threadpool(
        retrieve_and_format, request.query, request.domain, top_k=5, user_id=request.user_id
    )

    if not results or not context_string:
        async def mock_stream():
            yield "No relevant context found in documents. Please ensure valid governance documents are uploaded and indexed."
        return StreamingResponse(mock_stream(), media_type="text/plain")

    # Use standard Chat Q&A via stream
    generator = stream_answer(request.query, context_string, request.chat_history)
    
    async def generate():
        for chunk in generator:
            yield chunk

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/")
async def root():
    return {
        "service": "GovCheck API",
        "status": "ok",
        "ui": "Streamlit-only mode. Start UI with: streamlit run app.py",
    }

