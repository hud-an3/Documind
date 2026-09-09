
from __future__ import annotations

import asyncio
import hashlib
import shutil
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, List, Optional

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Header,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, HttpUrl

from core.config import cfg
from core.vector_store import VectorStoreManager
from ingestion.document_loader import DocumentLoader
from agents.rag_agent import DocuMindAgent


# ── App lifecycle ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup / shutdown lifecycle.
    Initialise shared singletons here so they're ready before any request.
    In a multi-worker setup (gunicorn -w 4), each worker gets its own copy
    — use Redis-backed memory and a shared vector store server for true
    multi-worker support.
    """
    print("[DocuMind] Starting up…")
    app.state.vsm = VectorStoreManager()
    app.state.loader = DocumentLoader()
    app.state.agent = DocuMindAgent(app.state.vsm)
    print(f"[DocuMind] Vector store: {cfg.vector_store} | Vision: {cfg.vision_provider}")
    yield
    print("[DocuMind] Shutting down.")


app = FastAPI(
    title="DocuMind API",
    description=(
        "Intelligent multi-source document Q&A. "
        "Supports PDF, DOCX, TXT, images (via AWS Rekognition / GCP Vision), and URLs. "
        "Powered by LangChain + LlamaIndex + FAISS / ChromaDB."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.allowed_origins.split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    job_id: str
    filename: str
    chunks_created: int
    avg_chunk_words: int
    vector_store: str
    embedding_model: str
    vision_provider: str
    duration_seconds: float
    message: str


class URLIngestRequest(BaseModel):
    url: str = Field(..., description="Public URL to crawl and ingest")
    session_id: Optional[str] = None


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = Field(None, description="Pass same ID across turns for memory")
    source_filter: Optional[str] = Field(None, description="Restrict search to one document filename")
    top_k: Optional[int] = Field(None, ge=1, le=20)


class SourceCitation(BaseModel):
    source: str
    page: str
    excerpt: str
    ingestion_method: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceCitation]
    session_id: str
    duration_seconds: float


class DocumentInfo(BaseModel):
    source: str
    chunks: int


class StatsResponse(BaseModel):
    backend: str
    total_chunks: int
    collection: Optional[str] = None
    embedding_model: str
    top_k: int


# ── Helper ─────────────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx", ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def _file_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    return {
        "status": "ok",
        "vector_store": cfg.vector_store,
        "vision_provider": cfg.vision_provider,
        "llm_model": cfg.llm_model,
    }


@app.get("/stats", response_model=StatsResponse, tags=["System"])
def stats():
    """Vector store health — useful for monitoring dashboards."""
    return app.state.vsm.stats()


@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_file(file: UploadFile = File(...)):
    """
    Upload a document or image.

    - PDF / DOCX / TXT / MD → parsed by LlamaIndex
    - JPG / PNG / BMP / TIFF → OCR via AWS Rekognition or GCP Vision API
    
    The file is chunked, embedded, and stored in the vector store.
    All subsequent queries will search across this document.
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            400,
            f"Unsupported file type '{suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    content = await file.read()

    if len(content) > cfg.max_upload_bytes:
        raise HTTPException(
            413,
            f"File too large ({len(content) // (1024*1024)}MB). Max: {cfg.max_upload_mb}MB.",
        )

    t0 = time.time()
    job_id = str(uuid.uuid4())[:8]

    # Save to temp file (required for LlamaIndex and boto3 which expect file paths)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        loader: DocumentLoader = app.state.loader
        vsm: VectorStoreManager = app.state.vsm

        nodes = loader.load_and_chunk(tmp_path)
        if not nodes:
            raise HTTPException(422, "No content could be extracted from this file.")

        # Tag every chunk with the real filename (not the temp path)
        for node in nodes:
            node.metadata["source"] = file.filename
            node.metadata["job_id"] = job_id

        vsm.ingest(nodes, extra_metadata={"job_id": job_id})

        # Invalidate cached chain so new docs are included in next query
        app.state.agent._chains.clear()

        report = loader.ingest_report(nodes, file.filename)
        return IngestResponse(
            job_id=job_id,
            filename=file.filename,
            chunks_created=report["chunks_created"],
            avg_chunk_words=report["avg_chunk_words"],
            vector_store=report["vector_store"],
            embedding_model=report["embedding_model"],
            vision_provider=report["vision_provider"],
            duration_seconds=round(time.time() - t0, 2),
            message=f"'{file.filename}' successfully ingested.",
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/ingest/url", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_url(body: URLIngestRequest):
    """
    Crawl a public URL and ingest its text content.
    Useful for ingesting documentation pages, news articles, or web reports.
    """
    t0 = time.time()
    job_id = str(uuid.uuid4())[:8]

    loader: DocumentLoader = app.state.loader
    vsm: VectorStoreManager = app.state.vsm

    try:
        docs = loader.load_url(body.url)
    except Exception as e:
        raise HTTPException(422, f"Failed to fetch URL: {e}")

    nodes = loader.chunk_documents(docs)
    if not nodes:
        raise HTTPException(422, "No text content found at this URL.")

    for node in nodes:
        node.metadata["job_id"] = job_id

    vsm.ingest(nodes, extra_metadata={"job_id": job_id})
    app.state.agent._chains.clear()

    source_name = docs[0].metadata.get("source", body.url)
    report = loader.ingest_report(nodes, source_name)

    return IngestResponse(
        job_id=job_id,
        filename=source_name,
        chunks_created=report["chunks_created"],
        avg_chunk_words=report["avg_chunk_words"],
        vector_store=report["vector_store"],
        embedding_model=report["embedding_model"],
        vision_provider=report["vision_provider"],
        duration_seconds=round(time.time() - t0, 2),
        message=f"URL '{body.url}' successfully ingested.",
    )


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query(body: QueryRequest):
    """
    Ask a question across all ingested documents.
    Pass the same session_id across multiple turns to maintain conversation memory.
    Use source_filter to restrict the search to a specific document.
    """
    t0 = time.time()
    session_id = body.session_id or str(uuid.uuid4())
    agent: DocuMindAgent = app.state.agent

    try:
        result = agent.query(
            question=body.question,
            session_id=session_id,
            source_filter=body.source_filter,
        )
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Agent error: {e}")

    return QueryResponse(
        answer=result["answer"],
        sources=[SourceCitation(**s) for s in result["sources"]],
        session_id=session_id,
        duration_seconds=round(time.time() - t0, 2),
    )


@app.get("/query/stream", tags=["Query"])
async def query_stream(
    question: str = Query(..., min_length=1),
    session_id: Optional[str] = Query(None),
    source_filter: Optional[str] = Query(None),
):
    """
    Streaming Q&A via Server-Sent Events (SSE).
    Tokens are pushed to the client as the LLM generates them.

    Client usage (JavaScript):
        const es = new EventSource(`/query/stream?question=What+are+the+terms&session_id=abc`);
        es.onmessage = (e) => process.stdout.write(e.data);
        es.addEventListener('done', () => es.close());
    """
    session_id = session_id or str(uuid.uuid4())
    agent: DocuMindAgent = app.state.agent

    async def event_generator() -> AsyncIterator[str]:
        try:
            # query_stream is a synchronous generator; run in thread pool
            loop = asyncio.get_event_loop()
            gen = agent.query_stream(
                question=question,
                session_id=session_id,
                source_filter=source_filter,
            )
            for token in gen:
                # SSE format: "data: <payload>\n\n"
                yield f"data: {token}\n\n"
                await asyncio.sleep(0)  # yield control to event loop
            yield "event: done\ndata: [DONE]\n\n"
        except RuntimeError as e:
            yield f"event: error\ndata: {e}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering for SSE
            "X-Session-Id": session_id,
        },
    )


@app.get("/sources", response_model=List[DocumentInfo], tags=["Management"])
def list_sources():
    """List all ingested documents with chunk counts."""
    return app.state.vsm.list_sources()


@app.delete("/document/{filename}", tags=["Management"])
def delete_document(filename: str):
    """
    Remove a document and all its chunks from the vector store.
    For FAISS this rebuilds the index; for ChromaDB it's a targeted delete.
    """
    deleted = app.state.vsm.delete_document(filename)
    if not deleted:
        raise HTTPException(404, f"Document '{filename}' not found in vector store.")
    app.state.agent._chains.clear()
    return {"message": f"Document '{filename}' deleted.", "filename": filename}


@app.delete("/session/{session_id}", tags=["Management"])
def reset_session(session_id: str):
    """Clear conversation memory for a session (start fresh while keeping documents)."""
    cleared = app.state.agent.reset_session(session_id)
    if not cleared:
        raise HTTPException(404, f"Session '{session_id}' not found.")
    return {"message": f"Session '{session_id}' cleared.", "session_id": session_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=cfg.api_host,
        port=cfg.api_port,
        reload=cfg.debug,
        log_level=cfg.log_level.lower(),
    )
