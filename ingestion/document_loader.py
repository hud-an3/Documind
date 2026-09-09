
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

from core.config import cfg
from ingestion.vision_loader import VisionLoaderFactory


# File types handled by LlamaIndex's built-in readers
DOCUMENT_EXTENSIONS = {".pdf", ".txt", ".md", ".docx", ".html", ".csv"}


class DocumentLoader:
 

    def __init__(self):
        self.splitter = SentenceSplitter(
            chunk_size=cfg.max_chunk_size,
            chunk_overlap=cfg.chunk_overlap,
        )
        self._vision_loader = None  # lazy-init — only created when needed

    @property
    def vision_loader(self):
        if self._vision_loader is None:
            self._vision_loader = VisionLoaderFactory.get()
        return self._vision_loader

    # ── File loading ──────────────────────────────────────────────────────

    def load_file(self, file_path: str) -> List[Document]:
        """
        Load a single file. Automatically routes to the correct loader
        based on extension.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Not found: {file_path}")

        ext = path.suffix.lower()

        if ext in cfg.image_extensions:
            return self._load_image(str(path))
        elif ext in DOCUMENT_EXTENSIONS:
            return self._load_document(str(path))
        else:
            raise ValueError(
                f"Unsupported file type: '{ext}'. "
                f"Documents: {DOCUMENT_EXTENSIONS} | Images: {cfg.image_extensions}"
            )

    def _load_document(self, file_path: str) -> List[Document]:
        """Load text-based documents via LlamaIndex."""
        path = Path(file_path)
        reader = SimpleDirectoryReader(input_files=[str(path)])
        docs = reader.load_data()
        for doc in docs:
            doc.metadata.update({
                "source": path.name,
                "file_path": str(path),
                "file_type": "document",
                "file_extension": path.suffix.lower(),
                "ingestion_method": "llamaindex",
                "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "file_hash": self._hash_file(str(path)),
            })
        print(f"[Loader] Loaded {len(docs)} page(s) from {path.name}")
        return docs

    def _load_image(self, file_path: str) -> List[Document]:
        """Load image via vision AI (Rekognition / GCP / Mock)."""
        doc = self.vision_loader.load_image_as_document(
            file_path,
            extra_metadata={
                "file_type": "image",
                "file_extension": Path(file_path).suffix.lower(),
                "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "file_hash": self._hash_file(file_path),
            },
        )
        return [doc]

    def load_directory(
        self,
        dir_path: str,
        recursive: bool = True,
    ) -> List[Document]:
        """
        Load all supported files from a directory.
        Mixes PDFs and images — each is routed to the right loader.
        """
        path = Path(dir_path)
        all_docs: List[Document] = []
        skipped = 0

        for file in sorted(path.rglob("*") if recursive else path.glob("*")):
            if not file.is_file():
                continue
            ext = file.suffix.lower()
            if ext in DOCUMENT_EXTENSIONS or ext in cfg.image_extensions:
                try:
                    docs = self.load_file(str(file))
                    all_docs.extend(docs)
                except Exception as e:
                    print(f"[Loader] Skipping {file.name}: {e}")
                    skipped += 1
            else:
                skipped += 1

        print(f"[Loader] Directory scan: {len(all_docs)} docs loaded, {skipped} skipped.")
        return all_docs

    def load_url(self, url: str) -> List[Document]:
        """
        Crawl a URL, extract main text content, return as a Document.
        """
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("Run: pip install requests beautifulsoup4")

        print(f"[Loader] Fetching URL: {url}")
        resp = requests.get(url, timeout=15, headers={"User-Agent": "DocuMind/1.0"})
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        # Remove boilerplate
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        title = soup.find("title")
        title_text = title.get_text(strip=True) if title else url

        doc = Document(
            text=text,
            metadata={
                "source": title_text,
                "url": url,
                "file_type": "url",
                "ingestion_method": "web_crawl",
                "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
        )
        print(f"[Loader] URL ingested: '{title_text}' ({len(text.split())} words)")
        return [doc]

    # ── Chunking ──────────────────────────────────────────────────────────

    def chunk_documents(self, documents: List[Document]) -> List[TextNode]:
        """
        Split Documents into TextNodes (chunks) using SentenceSplitter.
        """
        nodes = self.splitter.get_nodes_from_documents(documents)
        print(
            f"[Loader] {len(documents)} doc(s) → {len(nodes)} chunks "
            f"(size={cfg.max_chunk_size}, overlap={cfg.chunk_overlap})"
        )
        return nodes

    def load_and_chunk(self, file_path: str) -> List[TextNode]:
        """Convenience: load + chunk in one call."""
        docs = self.load_file(file_path)
        return self.chunk_documents(docs)

    def ingest_report(self, nodes: List[TextNode], filename: str) -> Dict:
        """Build a summary dict for the API /ingest response."""
        word_counts = []
        for n in nodes:
            word_counts.append(len(n.get_content().split()))
        return {
            "filename": filename,
            "chunks_created": len(nodes),
            "avg_chunk_words": round(sum(word_counts) / len(word_counts)) if word_counts else 0,
            "vector_store": cfg.vector_store,
            "embedding_model": cfg.embedding_model,
            "vision_provider": cfg.vision_provider,
        }

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _hash_file(path: str) -> str:
        """MD5 fingerprint — detect duplicate uploads."""
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
