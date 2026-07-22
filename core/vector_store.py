"""
core/vector_store.py  — Phase 2 upgrade

FAISS vs ChromaDB — when to use which:

  FAISS
    - Pure in-memory (loaded from disk on startup, saved after writes)
    - Approximate nearest neighbour via IVF / HNSW — extremely fast at scale
    - No built-in metadata filtering — you filter after retrieval in Python
    - Best for: single-user apps, prototypes, corpora under ~5M vectors

  ChromaDB
    - SQLite-backed persistence — survives restarts, no re-embedding on boot
    - Native metadata filtering ($where clauses) — filter by source, date, tag
    - Supports multi-tenancy via collections — one collection per user/client
    - REST server mode — multiple services can share one Chroma instance
    - Best for: production SaaS, multi-user, need document-level filtering

Phase 2 additions:
  - ChromaDB: proper collection-per-user support + metadata $where filtering
  - FAISS: merge strategy (add_documents to existing index, not rebuild)
  - SageMaker embedding provider path
  - delete_document() — lets users remove a specific file's chunks
  - list_sources() — inventory of ingested documents
  - stats() — index health info exposed to the API
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain_community.vectorstores import FAISS, Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document as LCDocument

from core.config import cfg


# ── Embedding provider factory ────────────────────────────────────────────────

def get_embeddings():
    """
    Returns the configured embedding model.

    openai  → OpenAI text-embedding-3-small (default, cheap, 1536-dim)
    sagemaker → Your own SageMaker endpoint (full data privacy, bring your model)

    SageMaker setup:
      1. Deploy a HuggingFace embedding model (e.g. BAAI/bge-large-en-v1.5)
         via SageMaker JumpStart
      2. Set EMBEDDING_PROVIDER=sagemaker and SAGEMAKER_ENDPOINT_NAME=<name>
      3. The SagemakerEndpointEmbeddings class handles batching automatically

    Why this matters for clients: data-sensitive industries (legal, healthcare,
    finance) often cannot send documents to OpenAI. SageMaker keeps everything
    inside their AWS account. This is a key selling point.
    """
    if cfg.embedding_provider == "sagemaker":
        # Lazy import — boto3 may not be installed in all environments
        try:
            from langchain_community.embeddings import SagemakerEndpointEmbeddings
            from langchain_community.embeddings.sagemaker_endpoint import EmbeddingsContentHandler
            import json

            class BGEContentHandler(EmbeddingsContentHandler):
                """
                Content handler for HuggingFace BGE models deployed on SageMaker.
                SageMaker endpoints expect a specific JSON schema — this translates
                LangChain's generic embedding interface to that schema.
                """
                content_type = "application/json"
                accepts = "application/json"

                def transform_input(self, inputs: list[str], model_kwargs: dict) -> bytes:
                    return json.dumps({"inputs": inputs, **model_kwargs}).encode("utf-8")

                def transform_output(self, output) -> list[list[float]]:
                    response_json = json.loads(output.read().decode("utf-8"))
                    return response_json["vectors"]

            return SagemakerEndpointEmbeddings(
                endpoint_name=cfg.sagemaker_endpoint,
                region_name=cfg.aws_region,
                content_handler=BGEContentHandler(),
            )
        except ImportError:
            print("[Embeddings] boto3 not installed, falling back to OpenAI.")

    # Default: OpenAI
    return GoogleGenerativeAIEmbeddings(
        model=cfg.embedding_model, google_api_key=cfg.google_api_key,)


# ── Node conversion ───────────────────────────────────────────────────────────

def nodes_to_lc_docs(nodes, extra_metadata: Optional[Dict] = None) -> List[LCDocument]:
    """
    Convert LlamaIndex nodes → LangChain Documents.
    extra_metadata is merged in — used to tag chunks with user_id,
    session_id, upload_timestamp, ingestion_method, etc.
    """
    docs = []
    for i, node in enumerate(nodes):
        meta = {
            **node.metadata,
            "node_id": node.node_id,
            "chunk_index": i,
            **(extra_metadata or {}),
        }
        docs.append(LCDocument(page_content=node.get_content(), metadata=meta))
    return docs


# ── FAISS helpers ─────────────────────────────────────────────────────────────

def _faiss_load(path: str, embeddings) -> Optional[FAISS]:
    if (Path(path) / "index.faiss").exists():
        return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    return None


def _faiss_save(store: FAISS, path: str):
    Path(path).mkdir(parents=True, exist_ok=True)
    store.save_local(path)


# ── VectorStoreManager ────────────────────────────────────────────────────────

class VectorStoreManager:
    """
    Unified interface over FAISS and ChromaDB.

    Key design decisions (explain these in interviews):

    1. Lazy loading — the store is only loaded from disk on first use,
       so app startup is fast even with large indexes.

    2. Additive ingestion — calling ingest() twice merges docs into the
       existing index rather than rebuilding it from scratch.

    3. Source-level deletion — ChromaDB supports $where metadata filters,
       so we can delete all chunks for a specific filename. FAISS doesn't
       support deletion natively, so we rebuild the index minus those docs.

    4. Multi-collection (Chroma) — each collection_name is isolated.
       In a SaaS product you'd pass collection_name=user_id so each
       customer only searches their own documents.
    """

    def __init__(self, collection_name: Optional[str] = None):
        self.embeddings = get_embeddings()
        self.collection_name = collection_name or cfg.chroma_collection
        self._store = None
        self._source_registry: Dict[str, int] = {}  # source → chunk count

    # ── Ingest ────────────────────────────────────────────────────────────

    def ingest(self, nodes, extra_metadata: Optional[Dict] = None) -> int:
        """
        Embed and store nodes. Returns number of chunks ingested.
        Thread-safe for sequential calls; for concurrent writes use a queue.
        """
        lc_docs = nodes_to_lc_docs(nodes, extra_metadata)
        if not lc_docs:
            return 0

        if cfg.vector_store == "faiss":
            self._ingest_faiss(lc_docs)
        elif cfg.vector_store == "chroma":
            self._ingest_chroma(lc_docs)
        else:
            raise ValueError(f"Unknown VECTOR_STORE: '{cfg.vector_store}'. Use 'faiss' or 'chroma'.")

        # Update in-memory source registry
        for doc in lc_docs:
            src = doc.metadata.get("source", "unknown")
            self._source_registry[src] = self._source_registry.get(src, 0) + 1

        print(f"[VectorStore] +{len(lc_docs)} chunks → {cfg.vector_store} (collection: {self.collection_name})")
        return len(lc_docs)

    def _ingest_faiss(self, lc_docs: List[LCDocument]):
        existing = _faiss_load(cfg.faiss_index_path, self.embeddings)
        if existing:
            existing.add_documents(lc_docs)   # merge into existing index
            self._store = existing
        else:
            self._store = FAISS.from_documents(lc_docs, self.embeddings)
        _faiss_save(self._store, cfg.faiss_index_path)

    def _ingest_chroma(self, lc_docs: List[LCDocument]):
        if self._store is None:
            self._store = Chroma(
                persist_directory=cfg.chroma_persist_dir,
                embedding_function=self.embeddings,
                collection_name=self.collection_name,
            )
        self._store.add_documents(lc_docs)

    # ── Retrieval ─────────────────────────────────────────────────────────

    def get_retriever(self, k: Optional[int] = None, filter_source: Optional[str] = None):
        """
        Returns a LangChain retriever.

        filter_source (ChromaDB only) — restrict retrieval to a single document.
        Useful when a user asks "only look in contract.pdf for this answer".
        FAISS doesn't support this natively; you'd post-filter the results.
        """
        store = self._get_or_load()
        search_kwargs: Dict = {"k": k or cfg.top_k}

        if filter_source and cfg.vector_store == "chroma":
            # ChromaDB $where filter — native metadata filtering
            search_kwargs["filter"] = {"source": {"$eq": filter_source}}

        return store.as_retriever(search_type="similarity", search_kwargs=search_kwargs)

    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
        filter_source: Optional[str] = None,
    ) -> List[Tuple[LCDocument, float]]:
        """Returns (document, similarity_score) pairs."""
        store = self._get_or_load()
        kwargs: Dict = {"k": k or cfg.top_k}
        if filter_source and cfg.vector_store == "chroma":
            kwargs["filter"] = {"source": {"$eq": filter_source}}
        return store.similarity_search_with_score(query, **kwargs)

    # ── Management ────────────────────────────────────────────────────────

    def delete_document(self, source_filename: str) -> bool:
        """
        Remove all chunks associated with a specific source file.
        Returns False if no index exists yet, or if the filename isn't found.

        ChromaDB: uses native $where delete — efficient O(n_matching)
        FAISS: no native delete — rebuilds index from remaining docs (expensive)
               For large indexes, consider switching to Chroma in production.
        """
        try:
            store = self._get_or_load()
        except RuntimeError:
            return False  # nothing has been ingested yet

        if cfg.vector_store == "chroma":
            store.delete(where={"source": {"$eq": source_filename}})
            self._source_registry.pop(source_filename, None)
            print(f"[VectorStore] Deleted chunks for '{source_filename}' from ChromaDB.")
            return True

        elif cfg.vector_store == "faiss":
            # FAISS rebuild strategy
            all_docs = list(store.docstore._dict.values())
            remaining = [d for d in all_docs if d.metadata.get("source") != source_filename]
            if len(remaining) == len(all_docs):
                return False  # document wasn't found
            if remaining:
                self._store = FAISS.from_documents(remaining, self.embeddings)
                _faiss_save(self._store, cfg.faiss_index_path)
            else:
                # All docs deleted — remove index files
                import shutil
                shutil.rmtree(cfg.faiss_index_path, ignore_errors=True)
                self._store = None
            self._source_registry.pop(source_filename, None)
            print(f"[VectorStore] Rebuilt FAISS index after deleting '{source_filename}'.")
            return True

        return False

    def list_sources(self) -> List[Dict]:
        """Returns a list of ingested document sources with chunk counts."""
        try:
            store = self._get_or_load()
        except RuntimeError:
            return []  # no documents ingested yet

        if cfg.vector_store == "chroma":
            results = store.get(include=["metadatas"])
            sources: Dict[str, int] = {}
            for meta in results.get("metadatas", []):
                src = meta.get("source", "unknown")
                sources[src] = sources.get(src, 0) + 1
            return [{"source": k, "chunks": v} for k, v in sorted(sources.items())]

        elif cfg.vector_store == "faiss":
            sources: Dict[str, int] = {}
            for doc in store.docstore._dict.values():
                src = doc.metadata.get("source", "unknown")
                sources[src] = sources.get(src, 0) + 1
            return [{"source": k, "chunks": v} for k, v in sorted(sources.items())]

        return []

    def stats(self) -> Dict:
        """Index health — exposed via GET /stats endpoint."""
        try:
            store = self._get_or_load()
            if cfg.vector_store == "chroma":
                count = store._collection.count()
            else:
                count = len(store.docstore._dict)
            return {
                "backend": cfg.vector_store,
                "total_chunks": count,
                "collection": self.collection_name,
                "embedding_model": cfg.embedding_model,
                "top_k": cfg.top_k,
            }
        except Exception:
            return {
                "backend": cfg.vector_store,
                "total_chunks": 0,
                "collection": self.collection_name,
                "embedding_model": cfg.embedding_model,
                "top_k": cfg.top_k,
            }

    # ── Internal ──────────────────────────────────────────────────────────

    def _get_or_load(self):
        if self._store is not None:
            return self._store
        if cfg.vector_store == "faiss":
            self._store = _faiss_load(cfg.faiss_index_path, self.embeddings)
            if not self._store:
                raise RuntimeError(
                    "No FAISS index found. Upload a document first via POST /ingest."
                )
        elif cfg.vector_store == "chroma":
            self._store = Chroma(
                persist_directory=cfg.chroma_persist_dir,
                embedding_function=self.embeddings,
                collection_name=self.collection_name,
            )
        return self._store
