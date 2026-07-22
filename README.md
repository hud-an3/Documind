# 🧠 DocuMind — Multi-Source Intelligent Document Agent

> Chat with your documents. PDFs, scanned invoices, images, even web pages — ask anything, get cited answers.

DocuMind is a production-ready RAG (Retrieval-Augmented Generation) system that lets you upload documents and query them conversationally. It demonstrates a complete AI engineering stack across 4 build phases: core RAG pipeline, dual vector stores, vision AI ingestion, and a production API + UI.

---

## Build Phases

| Phase | What it adds |
|-------|--------------|
| **1 — Core RAG** | PDF ingestion (LlamaIndex), chunking, FAISS vector store, LangChain conversational agent with memory and source citations |
| **2 — Dual Vector Store** | ChromaDB support with metadata filtering, additive FAISS merging, document deletion, source inventory, SageMaker embedding option |
| **3 — Vision AI** | AWS Rekognition + GCP Vision OCR for scanned images/invoices, URL crawling, a Mock vision provider for credential-free local dev |
| **4 — Production API** | Streaming responses (SSE), per-session memory, source-scoped queries, file size limits, full test suite, Docker deployment |

## Architecture

```
Documents (PDF / images / URLs)
       │
       ▼
 Ingestion Layer
   ├── LlamaIndex (PDF/DOCX/TXT parsing + chunking)
   ├── AWS Rekognition (image OCR + label detection)
   ├── GCP Vision API (dense document OCR)
   └── Web crawler (URL → clean text)
       │
       ▼
 Embedding + Vector Store
   ├── OpenAI / AWS SageMaker (embedding models)
   ├── FAISS (fast local index, additive merging)
   └── ChromaDB (persistent, metadata-filterable)
       │
       ▼
 LangChain Agent
   ├── ConversationalRetrievalChain (per-session memory)
   ├── Source citation (document + page references)
   ├── Source-scoped retrieval (query one doc only)
   └── Token streaming (SSE)
       │
       ▼
 FastAPI backend  ←→  Streamlit UI
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Document parsing | LlamaIndex |
| Agent orchestration | LangChain 0.2.x |
| Vector store (local) | FAISS |
| Vector store (persistent) | ChromaDB |
| Embedding model | OpenAI `text-embedding-3-small` / AWS SageMaker |
| Vision AI | AWS Rekognition + GCP Vision API |
| LLM | GPT-4o-mini (swappable) |
| Backend | FastAPI (streaming via SSE) |
| Frontend | Streamlit |
| Deployment | Docker Compose |
| Testing | pytest (16 tests, no cloud credentials required) |

> **Note on LangChain version:** pinned to `0.2.x`. LangChain 1.x removed the legacy `langchain.chains` / `langchain.memory` API used here in favor of LangGraph-style agents. If you want to port this to LangChain 1.x, the `ConversationalRetrievalChain` in `agents/rag_agent.py` is the piece to rewrite.

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/hud-an3/documind
cd documind
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Add your OPENAI_API_KEY (minimum required to run)

# 3. Start the API
uvicorn api.main:app --reload

# 4. Start the UI (new terminal)
streamlit run frontend/app.py
```

Open http://localhost:8501, upload a PDF, and start chatting.

### Run without any cloud credentials

Set `VISION_PROVIDER=mock` in `.env` to test the image ingestion path without AWS/GCP accounts. Useful for local dev and for anyone cloning this repo to review your code.

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

16 tests cover config loading, vision loaders (including the mock provider), document chunking, vector store operations, and the edge cases that were caught during manual testing (empty index handling for `/stats`, `/sources`, and document deletion).

## API Reference

Auto-generated docs at http://localhost:8000/docs

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/ingest` | Upload a document or image for indexing |
| `POST` | `/ingest/url` | Crawl and ingest a web page |
| `POST` | `/query` | Ask a question, get a cited answer |
| `GET` | `/query/stream` | Same as above, streamed via Server-Sent Events |
| `GET` | `/sources` | List all ingested documents with chunk counts |
| `DELETE` | `/document/{filename}` | Remove a document and its chunks |
| `DELETE` | `/session/{session_id}` | Clear conversation memory |
| `GET` | `/stats` | Vector store health info |
| `GET` | `/health` | Health check |

### Example

```bash
# Ingest a PDF
curl -X POST http://localhost:8000/ingest \
  -F "file=@contract.pdf"

# Query it (session_id keeps conversation memory across turns)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the payment terms?", "session_id": "demo-1"}'

# Stream a response
curl -N "http://localhost:8000/query/stream?question=Summarise+section+3&session_id=demo-1"

# Restrict search to one document
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the penalty clause?", "source_filter": "contract.pdf"}'
```

Response:
```json
{
  "answer": "Payment is due within 30 days of invoice date, with a 2% late fee per month.",
  "sources": [
    {
      "source": "contract.pdf",
      "page": "4",
      "excerpt": "…payment shall be made within thirty (30) calendar days…",
      "ingestion_method": "llamaindex"
    }
  ],
  "session_id": "demo-1",
  "duration_seconds": 1.84
}
```

## Configuration

| Variable | Default | Description |
|----------|---------|--------------|
| `OPENAI_API_KEY` | — | Required |
| `VECTOR_STORE` | `faiss` | `faiss` or `chroma` |
| `EMBEDDING_PROVIDER` | `openai` | `openai` or `sagemaker` |
| `VISION_PROVIDER` | `auto` | `auto` \| `rekognition` \| `gcp` \| `mock` |
| `MAX_CHUNK_SIZE` | `512` | Token size per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `TOP_K_RETRIEVAL` | `5` | Chunks retrieved per query |
| `MAX_UPLOAD_MB` | `50` | Max upload file size |
| `AWS_*` | — | For Rekognition + SageMaker |
| `GOOGLE_APPLICATION_CREDENTIALS` | — | For GCP Vision |

## Docker Deployment

```bash
cp .env.example .env   # fill in your keys
docker-compose up --build
```

API: http://localhost:8000 · UI: http://localhost:8501

## Use Cases

- **Legal**: Query contracts for specific clauses
- **Finance**: Parse invoices and financial statements (via vision AI)
- **HR**: Search policy documents and employee handbooks
- **Research**: Chat with academic papers, or ingest a docs site URL
- **Real estate**: Extract terms from scanned property documents

## Extending the Agent

Add a new tool to the LangChain agent in `agents/rag_agent.py`:

```python
from langchain.tools import Tool

calculator_tool = Tool(
    name="calculator",
    func=lambda x: eval(x),
    description="Useful for arithmetic questions about numbers in documents"
)
# Add to agent executor tools list
```

## License

MIT

