"""
tests/test_pipeline.py

End-to-end tests for the DocuMind pipeline.
These run without cloud credentials by using MockVisionLoader
and a temp FAISS index — so any developer can clone and run them.

Run with: pytest tests/ -v
"""
import os
import tempfile
import shutil
import pytest

# Point to mock providers before importing anything else
os.environ.setdefault("OPENAI_API_KEY", "sk-test-key")
os.environ.setdefault("VECTOR_STORE", "faiss")
os.environ.setdefault("VISION_PROVIDER", "mock")


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def tmp_index(tmp_path_factory):
    """Temp directory for FAISS index — cleaned up after tests."""
    d = tmp_path_factory.mktemp("faiss_index")
    os.environ["FAISS_INDEX_PATH"] = str(d)
    return d


@pytest.fixture
def sample_txt(tmp_path):
    """Create a minimal text document for testing."""
    p = tmp_path / "sample.txt"
    p.write_text(
        "PAYMENT TERMS\n\n"
        "All invoices are payable within 30 days of the invoice date.\n"
        "Late payments incur a 2% monthly fee.\n"
        "Accepted methods: bank transfer, credit card, PayPal.\n\n"
        "TERMINATION\n\n"
        "Either party may terminate this agreement with 14 days written notice.\n"
    )
    return str(p)


@pytest.fixture
def sample_image(tmp_path):
    """Create a minimal PNG for testing the vision pipeline."""
    p = tmp_path / "invoice.png"
    # Minimal 1x1 white PNG
    png_bytes = bytes([
        0x89,0x50,0x4E,0x47,0x0D,0x0A,0x1A,0x0A,
        0x00,0x00,0x00,0x0D,0x49,0x48,0x44,0x52,
        0x00,0x00,0x00,0x01,0x00,0x00,0x00,0x01,
        0x08,0x02,0x00,0x00,0x00,0x90,0x77,0x53,
        0xDE,0x00,0x00,0x00,0x0C,0x49,0x44,0x41,
        0x54,0x08,0xD7,0x63,0xF8,0xFF,0xFF,0x3F,
        0x00,0x05,0xFE,0x02,0xFE,0xDC,0xCC,0x59,
        0xE7,0x00,0x00,0x00,0x00,0x49,0x45,0x4E,
        0x44,0xAE,0x42,0x60,0x82,
    ])
    p.write_bytes(png_bytes)
    return str(p)


# ── Config tests ──────────────────────────────────────────────────────────────

def test_config_loads():
    from core.config import cfg
    assert cfg.vector_store == "faiss"
    assert cfg.vision_provider == "mock"
    assert cfg.max_chunk_size > 0
    assert cfg.chunk_overlap < cfg.max_chunk_size


def test_config_properties():
    from core.config import cfg
    assert isinstance(cfg.has_aws, bool)
    assert isinstance(cfg.has_gcp, bool)
    assert cfg.max_upload_bytes == cfg.max_upload_mb * 1024 * 1024


# ── Vision loader tests ────────────────────────────────────────────────────────

def test_mock_vision_loader(sample_image):
    from ingestion.vision_loader import MockVisionLoader
    loader = MockVisionLoader()
    assert loader.provider_name() == "mock_vision"
    text = loader.extract_text(sample_image)
    assert len(text) > 10


def test_mock_vision_produces_document(sample_image):
    from ingestion.vision_loader import MockVisionLoader
    loader = MockVisionLoader()
    doc = loader.load_image_as_document(sample_image)
    assert doc.text
    assert doc.metadata["source"] == "invoice.png"
    assert doc.metadata["ingestion_method"] == "mock_vision"
    assert doc.metadata["file_type"] == "image"


def test_vision_factory_returns_mock():
    from ingestion.vision_loader import VisionLoaderFactory, MockVisionLoader
    loader = VisionLoaderFactory.get("mock")
    assert isinstance(loader, MockVisionLoader)


# ── Document loader tests ──────────────────────────────────────────────────────

def test_load_text_file(sample_txt):
    from ingestion.document_loader import DocumentLoader
    loader = DocumentLoader()
    docs = loader.load_file(sample_txt)
    assert len(docs) >= 1
    assert "PAYMENT TERMS" in docs[0].text or any("PAYMENT" in d.text for d in docs)


def test_load_image_file(sample_image):
    from ingestion.document_loader import DocumentLoader
    loader = DocumentLoader()
    docs = loader.load_file(sample_image)
    assert len(docs) == 1
    assert docs[0].metadata["file_type"] == "image"


def test_chunking(sample_txt):
    from ingestion.document_loader import DocumentLoader
    loader = DocumentLoader()
    docs = loader.load_file(sample_txt)
    nodes = loader.chunk_documents(docs)
    assert len(nodes) >= 1
    for node in nodes:
        assert len(node.get_content()) > 0


def test_unsupported_extension(tmp_path):
    from ingestion.document_loader import DocumentLoader
    bad_file = tmp_path / "data.xyz"
    bad_file.write_text("hello")
    loader = DocumentLoader()
    with pytest.raises(ValueError, match="Unsupported"):
        loader.load_file(str(bad_file))


def test_missing_file():
    from ingestion.document_loader import DocumentLoader
    loader = DocumentLoader()
    with pytest.raises(FileNotFoundError):
        loader.load_file("/nonexistent/path/file.pdf")


def test_ingest_report(sample_txt):
    from ingestion.document_loader import DocumentLoader
    loader = DocumentLoader()
    nodes = loader.load_and_chunk(sample_txt)
    report = loader.ingest_report(nodes, "sample.txt")
    assert report["chunks_created"] == len(nodes)
    assert report["avg_chunk_words"] > 0
    assert report["vector_store"] in ("faiss", "chroma")


# ── Vector store tests (FAISS, no embedding calls — mocked) ──────────────────

def test_vector_store_stats_empty(tmp_index):
    from core.vector_store import VectorStoreManager
    vsm = VectorStoreManager()
    stats = vsm.stats()
    assert stats["backend"] == "faiss"
    # Should not crash even when empty


def test_nodes_to_lc_docs():
    from core.vector_store import nodes_to_lc_docs
    from llama_index.core.schema import TextNode
    nodes = [
        TextNode(text="Hello world", metadata={"source": "test.txt"}),
        TextNode(text="Second chunk", metadata={"source": "test.txt"}),
    ]
    docs = nodes_to_lc_docs(nodes, extra_metadata={"test": True})
    assert len(docs) == 2
    assert docs[0].page_content == "Hello world"
    assert docs[0].metadata["test"] is True
    assert docs[1].metadata["chunk_index"] == 1


def test_list_sources_empty_index_does_not_raise(tmp_index):
    """
    Regression test: list_sources() used to raise RuntimeError when no
    documents had been ingested yet, causing a 500 on GET /sources.
    It must return an empty list instead.
    """
    from core.vector_store import VectorStoreManager
    vsm = VectorStoreManager()
    result = vsm.list_sources()
    assert result == []


def test_delete_document_empty_index_returns_false(tmp_index):
    """
    Regression test: delete_document() used to raise RuntimeError when no
    index existed yet, causing a 500 on DELETE /document/{name} instead
    of a clean 404. It must return False instead.
    """
    from core.vector_store import VectorStoreManager
    vsm = VectorStoreManager()
    result = vsm.delete_document("nonexistent.pdf")
    assert result is False


def test_stats_empty_index_has_all_required_fields(tmp_index):
    """
    Regression test: stats()'s fallback dict was missing 'embedding_model'
    and 'top_k', which broke FastAPI response validation on GET /stats
    when the index was empty.
    """
    from core.vector_store import VectorStoreManager
    vsm = VectorStoreManager()
    result = vsm.stats()
    assert "embedding_model" in result
    assert "top_k" in result
    assert "backend" in result
    assert "total_chunks" in result
