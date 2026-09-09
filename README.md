# 🧠 DocuMind 

> Chat with the agent about your documents, PDFs, scanned invoices, images and even web pages. Ask anything and get answers with the sources cited.

DocuMind is a RAG (Retrieval-Augmented Generation) system that lets you upload documents and ask questions about them.

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

<img width="841" height="1380" alt="image" src="https://github.com/user-attachments/assets/db7626b4-f336-4c46-a6e5-4a273b85cac2" />



## Tech Stack

| Layer | Technology |
|-------|-----------|
| Document parsing | LlamaIndex |
| Agent orchestration | LangChain 0.2.x |
| Vector store (local) | FAISS |
| Vector store (persistent) | ChromaDB |
| Embedding model | OpenAI `text-embedding-3-small` / AWS SageMaker |
| Vision AI | AWS Rekognition + GCP Vision API |
| LLM | Gemini (you can choose any llm model of your choice :) ) |
| Backend | FastAPI (streaming via SSE) |
| Frontend | Streamlit |
| Deployment | Docker Compose |
| Testing | pytest (16 tests, no cloud credentials required) |

## API Reference

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


## Configuration

| Variable | Default | Description |
|----------|---------|--------------|
| `GEMINI_API_KEY` | — | Required |
| `VECTOR_STORE` | `faiss` | `faiss` or `chroma` |
| `EMBEDDING_PROVIDER` | `openai` |
| `VISION_PROVIDER` | `auto` | `auto` \| `rekognition` \| `gcp` \| `mock` |
| `MAX_CHUNK_SIZE` | `512` | Token size per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `TOP_K_RETRIEVAL` | `5` | Chunks retrieved per query |
| `MAX_UPLOAD_MB` | `50` | Max upload file size |


