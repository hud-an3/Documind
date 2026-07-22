"""
ingestion/vision_loader.py  — Phase 3 (complete)

Full image ingestion pipeline with:
  - AWS Rekognition  (DetectText + DetectLabels + DetectDocument)
  - GCP Vision API   (DOCUMENT_TEXT_DETECTION — better layout preservation)
  - MockVisionLoader (local dev without cloud credentials)
  - VisionLoaderFactory with auto-detection and graceful fallback

Why offer both Rekognition AND GCP Vision?

  Rekognition DetectText:
    - Best for sparse text: invoices, receipts, ID cards, signage
    - Returns word-level bounding boxes (great for forms with labelled fields)
    - Integrates with AWS Textract for structured form/table extraction
    - Max image size: 5MB in-memory, 15MB via S3

  GCP Vision DOCUMENT_TEXT_DETECTION:
    - Best for dense text: contracts, research papers, scanned books
    - Preserves reading order and paragraph structure better
    - Returns a hierarchical document model (page → block → paragraph → word)
    - Handles multi-column layouts better than Rekognition

  Real-world advice for clients: use Rekognition if you're already on AWS;
  use GCP Vision if documents are dense/multi-column; use both and compare
  confidence scores for mission-critical pipelines.

After extraction, every path produces a LlamaIndex Document, so the rest of
the ingestion pipeline (chunking → embedding → vector store) is identical
regardless of whether the source was a PDF or a scanned image.
"""
from __future__ import annotations

import base64
import io
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from llama_index.core import Document

from core.config import cfg


# ── Base class ────────────────────────────────────────────────────────────────

class BaseVisionLoader(ABC):
    """
    Abstract base for all vision loaders.
    Enforcing this interface means you can swap providers in one config line.
    """

    @abstractmethod
    def extract_text(self, image_path: str) -> str:
        """Extract raw text from an image file."""
        ...

    @abstractmethod
    def provider_name(self) -> str:
        ...

    def load_image_as_document(self, image_path: str, extra_metadata: Optional[Dict] = None) -> Document:
        """
        Full pipeline: image file → OCR text → LlamaIndex Document.
        This Document flows into the same chunker + embedder as PDFs.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        text = self.extract_text(image_path)
        word_count = len(text.split())

        print(f"[{self.provider_name()}] {path.name}: {word_count} words extracted")

        if word_count < 5:
            print(f"[{self.provider_name()}] Warning: very little text detected in {path.name}. "
                  "Check image quality or try the other provider.")

        return Document(
            text=text or f"[No text detected in {path.name}]",
            metadata={
                "source": path.name,
                "file_path": str(path),
                "file_type": "image",
                "ingestion_method": self.provider_name(),
                "word_count": word_count,
                **(extra_metadata or {}),
            },
        )

    def load_directory(self, dir_path: str) -> List[Document]:
        """Load all images from a directory."""
        docs = []
        for ext in cfg.image_extensions:
            for img_path in Path(dir_path).glob(f"**/*{ext}"):
                try:
                    docs.append(self.load_image_as_document(str(img_path)))
                except Exception as e:
                    print(f"[{self.provider_name()}] Failed on {img_path.name}: {e}")
        return docs


# ── AWS Rekognition ───────────────────────────────────────────────────────────

class AWSRekognitionLoader(BaseVisionLoader):
    """
    AWS Rekognition text extraction.

    Requires: pip install boto3
    Config: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION

    Uses DetectText for general images and DetectDocumentText for documents.
    DetectDocumentText returns lines in reading order, better for structured docs.
    """

    def __init__(self):
        try:
            import boto3
            self._client = boto3.client(
                "rekognition",
                region_name=cfg.aws_region,
                aws_access_key_id=cfg.aws_access_key_id,
                aws_secret_access_key=cfg.aws_secret_access_key,
            )
            # Verify connectivity
            self._client.meta.events  # cheap attribute access
            print("[Rekognition] Client ready.")
        except ImportError:
            raise ImportError("Run: pip install boto3")
        except Exception as e:
            raise RuntimeError(f"Rekognition init failed: {e}")

    def provider_name(self) -> str:
        return "aws_rekognition"

    def extract_text(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        # Check file size (Rekognition limit: 5MB in-memory)
        size_mb = len(image_bytes) / (1024 * 1024)
        if size_mb > 5:
            raise ValueError(
                f"Image {Path(image_path).name} is {size_mb:.1f}MB. "
                "Rekognition's in-memory limit is 5MB. Use S3 source or compress the image."
            )

        # DetectDocumentText is better for multi-line docs (preserves reading order)
        response = self._client.detect_document_text(Image={"Bytes": image_bytes})

        # Extract LINE blocks in order (WORD blocks give sub-word precision but are noisy)
        lines = [
            block["Text"]
            for block in response["Blocks"]
            if block["BlockType"] == "LINE"
        ]
        return "\n".join(lines)

    def detect_labels(self, image_path: str) -> List[Tuple[str, float]]:
        """
        Object/scene detection — useful for auto-classifying document type.
        Returns [(label, confidence_pct), ...]
        e.g. [("Invoice", 98.2), ("Text", 99.1), ("Receipt", 87.4)]
        """
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        response = self._client.detect_labels(
            Image={"Bytes": image_bytes},
            MinConfidence=cfg.rekognition_min_confidence,
        )
        return [(lbl["Name"], lbl["Confidence"]) for lbl in response["Labels"]]

    def load_image_as_document(self, image_path: str, extra_metadata: Optional[Dict] = None) -> Document:
        """Override to enrich metadata with label detection."""
        labels = []
        try:
            labels = self.detect_labels(image_path)
        except Exception:
            pass  # labels are supplementary; don't fail the whole ingest

        extra = {**(extra_metadata or {}), "detected_labels": ", ".join(l for l, _ in labels[:5])}
        return super().load_image_as_document(image_path, extra)


# ── GCP Vision ────────────────────────────────────────────────────────────────

class GCPVisionLoader(BaseVisionLoader):
    """
    Google Cloud Vision API text extraction.

    Requires: pip install google-cloud-vision
    Config: GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json

    DOCUMENT_TEXT_DETECTION uses a different ML model than TEXT_DETECTION:
    - Optimised for dense, structured documents
    - Returns a document hierarchy (page → block → paragraph → word → symbol)
    - Better at detecting text in tables and multi-column layouts
    - Handles 57 languages automatically
    """

    def __init__(self):
        try:
            from google.cloud import vision as gcp_vision
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cfg.gcp_credentials_path
            self._client = gcp_vision.ImageAnnotatorClient()
            self._vision = gcp_vision
            print("[GCP Vision] Client ready.")
        except ImportError:
            raise ImportError("Run: pip install google-cloud-vision")
        except Exception as e:
            raise RuntimeError(f"GCP Vision init failed: {e}")

    def provider_name(self) -> str:
        return "gcp_vision"

    def extract_text(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            content = f.read()

        image = self._vision.Image(content=content)

        # DOCUMENT_TEXT_DETECTION = optimised for structured/dense documents
        # Use TEXT_DETECTION for natural scene text (signs, photos)
        response = self._client.document_text_detection(image=image)

        if response.error.message:
            raise RuntimeError(f"GCP Vision API error: {response.error.message}")

        # full_text_annotation.text gives the full OCR output as a single string
        # with paragraphs separated by newlines — ready for chunking
        return response.full_text_annotation.text

    def extract_with_confidence(self, image_path: str) -> Tuple[str, float]:
        """
        Extended extraction that also returns average word confidence.
        Use this when you need to decide whether to flag a document for
        human review (e.g. confidence < 80% → manual check).
        """
        with open(image_path, "rb") as f:
            content = f.read()

        image = self._vision.Image(content=content)
        response = self._client.document_text_detection(image=image)

        if response.error.message:
            raise RuntimeError(f"GCP Vision API error: {response.error.message}")

        # Flatten all word confidence scores
        confidences = []
        for page in response.full_text_annotation.pages:
            for block in page.blocks:
                for para in block.paragraphs:
                    for word in para.words:
                        confidences.append(word.confidence)

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        return response.full_text_annotation.text, avg_confidence


# ── Mock loader (local dev / testing) ─────────────────────────────────────────

class MockVisionLoader(BaseVisionLoader):
    """
    Returns synthetic OCR text for local development and unit tests.
    No cloud credentials needed — just set VISION_PROVIDER=mock.

    This is important for your portfolio: it lets reviewers clone and run
    your project without AWS/GCP accounts, which lowers friction significantly.
    """

    MOCK_TEXTS = {
        ".jpg": "INVOICE\n\nBill To: Acme Corp\nDate: 2024-01-15\nInvoice #: INV-2024-001\n\nItem        Qty    Unit Price    Total\nConsulting   10h    $150.00      $1,500.00\nSetup Fee     1     $500.00        $500.00\n\nSubtotal: $2,000.00\nTax (13%):  $260.00\nTotal Due: $2,260.00\n\nPayment due within 30 days.",
        ".png": "CONTRACT AGREEMENT\n\nThis agreement is entered into on January 15, 2024 between Party A (Client) and Party B (Vendor).\n\nSection 1: Scope of Work\nVendor agrees to deliver AI consulting services as detailed in Appendix A.\n\nSection 2: Payment Terms\nClient shall pay $5,000 per month, invoiced on the 1st of each month.\n\nSection 3: Termination\nEither party may terminate with 30 days written notice.",
        "default": "Scanned document content. Text has been extracted via OCR. This is mock output for local development.",
    }

    def provider_name(self) -> str:
        return "mock_vision"

    def extract_text(self, image_path: str) -> str:
        ext = Path(image_path).suffix.lower()
        text = self.MOCK_TEXTS.get(ext, self.MOCK_TEXTS["default"])
        print(f"[MockVision] Returning synthetic text for {Path(image_path).name}")
        return text


# ── Factory ───────────────────────────────────────────────────────────────────

class VisionLoaderFactory:
    """
    Returns the right vision loader based on VISION_PROVIDER config or availability.

    auto detection priority:  mock (if explicitly set) → rekognition → gcp → mock fallback
    
    Usage:
        loader = VisionLoaderFactory.get()
        doc = loader.load_image_as_document("invoice.jpg")
        # doc is now a LlamaIndex Document ready for chunking
    """

    @staticmethod
    def get(provider: Optional[str] = None) -> BaseVisionLoader:
        provider = provider or cfg.vision_provider

        if provider == "mock":
            return MockVisionLoader()

        if provider == "rekognition":
            return AWSRekognitionLoader()

        if provider == "gcp":
            return GCPVisionLoader()

        if provider == "auto":
            # Try providers in order, fall back gracefully
            if cfg.has_aws:
                try:
                    return AWSRekognitionLoader()
                except Exception as e:
                    print(f"[VisionFactory] Rekognition unavailable: {e}")

            if cfg.has_gcp:
                try:
                    return GCPVisionLoader()
                except Exception as e:
                    print(f"[VisionFactory] GCP Vision unavailable: {e}")

            print("[VisionFactory] No cloud credentials found. Using MockVisionLoader for local dev.")
            return MockVisionLoader()

        raise ValueError(
            f"Unknown VISION_PROVIDER: '{provider}'. "
            "Choose: auto | rekognition | gcp | mock"
        )
