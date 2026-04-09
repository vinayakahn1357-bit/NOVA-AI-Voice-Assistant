"""
services/pdf_service.py — PDF Upload & Processing Service for NOVA (Phase 11)
Extracts text from PDFs with page tracking, chunks with overlap,
supports 50MB files, fallback parsing, and processing status callbacks.
"""

import hashlib
import io
import re
from typing import Callable

from config import (
    PDF_MAX_FILE_SIZE, PDF_CHUNK_SIZE, PDF_CHUNK_OVERLAP, IS_VERCEL,
)
from utils.logger import get_logger

log = get_logger("pdf")

# ─── Constants ────────────────────────────────────────────────────────────────
_LIMIT_MB = round(PDF_MAX_FILE_SIZE / (1024 * 1024), 1)

# In-memory extraction cache: sha256 → extracted result
_extraction_cache: dict[str, dict] = {}


class PDFService:
    """
    PDF text extraction, page-aware chunking, and summarization.
    Phase 11: 50MB support, fallback parser, page citations, processing status.
    """

    # ── Validation ────────────────────────────────────────────────────────

    @staticmethod
    def validate(file_bytes: bytes, filename: str = "") -> str | None:
        """
        Validate a PDF upload.
        Returns error message string, or None if valid.
        Includes actionable suggestions when file exceeds limit.
        """
        if not filename.lower().endswith(".pdf"):
            return "Only .pdf files are supported."

        size = len(file_bytes)
        if size > PDF_MAX_FILE_SIZE:
            size_mb = round(size / (1024 * 1024), 1)
            env = "serverless (Vercel)" if IS_VERCEL else "local"
            return (
                f"File too large ({size_mb}MB). Maximum is {_LIMIT_MB}MB ({env}).\n\n"
                f"💡 Suggestions:\n"
                f"• Compress the PDF using tools like smallpdf.com or ilovepdf.com\n"
                f"• Split the PDF into smaller parts (e.g., by chapter)\n"
                f"• Remove embedded images if text-only content is needed"
            )

        if size < 10:
            return "File is empty or corrupted. Please re-upload the file."

        return None

    # ── File Hashing ──────────────────────────────────────────────────────

    @staticmethod
    def file_hash(file_bytes: bytes) -> str:
        """Generate SHA256 hash of file for caching/deduplication."""
        return hashlib.sha256(file_bytes).hexdigest()

    # ── Text Extraction (with fallback) ───────────────────────────────────

    @staticmethod
    def extract_text(
        file_bytes: bytes,
        filename: str = "",
        on_status: Callable[[str], None] | None = None,
    ) -> dict:
        """
        Extract text from a PDF with page-level tracking.

        Args:
            file_bytes: Raw PDF file bytes.
            filename: Original filename for logging.
            on_status: Optional callback for processing status updates.

        Returns:
            {
                "text": str,              # Full cleaned text
                "pages": [                # Per-page text
                    {"page": 1, "text": "..."},
                    ...
                ],
                "total_pages": int,
                "extracted_pages": int,
                "doc_hash": str,
                "filename": str,
            }

        Raises:
            ValueError: If extraction fails completely.
        """
        doc_hash = PDFService.file_hash(file_bytes)

        # Check cache
        if doc_hash in _extraction_cache:
            log.info("PDF '%s': using cached extraction (hash=%s…)", filename, doc_hash[:12])
            if on_status:
                on_status("Using cached extraction...")
            return _extraction_cache[doc_hash]

        if on_status:
            on_status("Extracting text...")

        # Try primary parser (pdfplumber)
        try:
            result = PDFService._extract_with_pdfplumber(file_bytes, filename, on_status)
        except Exception as primary_exc:
            log.warning("pdfplumber failed: %s — trying PyPDF2 fallback", primary_exc)
            if on_status:
                on_status("Primary parser failed. Trying fallback parser...")

            # Try fallback parser (PyPDF2)
            try:
                result = PDFService._extract_with_pypdf2(file_bytes, filename, on_status)
            except Exception as fallback_exc:
                log.error("Both parsers failed: pdfplumber=%s, PyPDF2=%s",
                          primary_exc, fallback_exc)
                raise ValueError(
                    f"Failed to read PDF with both parsers. "
                    f"The file may be encrypted, image-based, or corrupted.\n\n"
                    f"💡 Try:\n"
                    f"• Re-saving the PDF from the original application\n"
                    f"• Using an OCR tool for image-based PDFs\n"
                    f"• Checking if the PDF is password-protected"
                ) from fallback_exc

        # Cache the result
        _extraction_cache[doc_hash] = result
        return result

    @staticmethod
    def _extract_with_pdfplumber(
        file_bytes: bytes, filename: str,
        on_status: Callable[[str], None] | None = None,
    ) -> dict:
        """Extract text using pdfplumber (primary parser)."""
        import pdfplumber

        pages_data = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            total_pages = len(pdf.pages)
            log.info("PDF '%s': %d pages (pdfplumber)", filename or "upload", total_pages)

            for i, page in enumerate(pdf.pages):
                if on_status and i % 10 == 0:
                    on_status(f"Extracting page {i + 1}/{total_pages}...")

                text = page.extract_text()
                if text and text.strip():
                    pages_data.append({
                        "page": i + 1,
                        "text": text.strip(),
                    })

        if not pages_data:
            raise ValueError("No text extracted — PDF may be image-based or empty.")

        full_text = "\n\n".join(p["text"] for p in pages_data)
        cleaned = _clean_text(full_text)

        log.info("PDF '%s': extracted %d chars from %d/%d pages (pdfplumber)",
                 filename, len(cleaned), len(pages_data), total_pages)

        return {
            "text": cleaned,
            "pages": pages_data,
            "total_pages": total_pages,
            "extracted_pages": len(pages_data),
            "doc_hash": PDFService.file_hash(file_bytes),
            "filename": filename or "uploaded.pdf",
        }

    @staticmethod
    def _extract_with_pypdf2(
        file_bytes: bytes, filename: str,
        on_status: Callable[[str], None] | None = None,
    ) -> dict:
        """Extract text using PyPDF2 (fallback parser)."""
        from PyPDF2 import PdfReader

        reader = PdfReader(io.BytesIO(file_bytes))
        total_pages = len(reader.pages)
        log.info("PDF '%s': %d pages (PyPDF2 fallback)", filename or "upload", total_pages)

        pages_data = []
        for i, page in enumerate(reader.pages):
            if on_status and i % 10 == 0:
                on_status(f"Extracting page {i + 1}/{total_pages} (fallback)...")

            text = page.extract_text()
            if text and text.strip():
                pages_data.append({
                    "page": i + 1,
                    "text": text.strip(),
                })

        if not pages_data:
            raise ValueError("No text extracted by fallback parser.")

        full_text = "\n\n".join(p["text"] for p in pages_data)
        cleaned = _clean_text(full_text)

        log.info("PDF '%s': extracted %d chars from %d/%d pages (PyPDF2)",
                 filename, len(cleaned), len(pages_data), total_pages)

        return {
            "text": cleaned,
            "pages": pages_data,
            "total_pages": total_pages,
            "extracted_pages": len(pages_data),
            "doc_hash": PDFService.file_hash(file_bytes),
            "filename": filename or "uploaded.pdf",
        }

    # ── Page-Aware Chunking with Overlap ──────────────────────────────────

    @staticmethod
    def chunk_text(
        pages: list[dict],
        chunk_size: int = PDF_CHUNK_SIZE,
        overlap: int = PDF_CHUNK_OVERLAP,
        on_status: Callable[[str], None] | None = None,
    ) -> list[dict]:
        """
        Split page-tracked text into overlapping chunks with metadata.

        Args:
            pages: List of {"page": int, "text": str} from extract_text.
            chunk_size: Max characters per chunk.
            overlap: Character overlap between consecutive chunks.
            on_status: Optional callback for processing status.

        Returns:
            List of {
                "text": str,
                "page": int,          # Source page number
                "chunk_index": int,   # Global chunk index
                "start_page": int,    # First page in chunk
                "end_page": int,      # Last page in chunk
            }
        """
        if not pages:
            return []

        if on_status:
            on_status("Chunking document...")

        # Merge all text with page boundaries tracked
        chunks = []
        current_text = ""
        current_start_page = pages[0]["page"]
        current_end_page = pages[0]["page"]
        chunk_index = 0

        for page_data in pages:
            page_num = page_data["page"]
            page_text = page_data["text"]

            # Split page text into sentences for cleaner breaks
            sentences = re.split(r'(?<=[.!?])\s+', page_text)

            for sentence in sentences:
                if len(current_text) + len(sentence) + 1 > chunk_size:
                    if current_text.strip():
                        chunks.append({
                            "text": current_text.strip(),
                            "page": current_start_page,
                            "chunk_index": chunk_index,
                            "start_page": current_start_page,
                            "end_page": current_end_page,
                        })
                        chunk_index += 1

                        # Overlap: keep the tail of the current chunk
                        if overlap > 0 and len(current_text) > overlap:
                            current_text = current_text[-overlap:]
                            current_start_page = current_end_page
                        else:
                            current_text = ""
                            current_start_page = page_num

                    # Handle sentence longer than chunk_size
                    if len(sentence) > chunk_size:
                        words = sentence.split()
                        for word in words:
                            if len(current_text) + len(word) + 1 > chunk_size:
                                if current_text.strip():
                                    chunks.append({
                                        "text": current_text.strip(),
                                        "page": current_start_page,
                                        "chunk_index": chunk_index,
                                        "start_page": current_start_page,
                                        "end_page": page_num,
                                    })
                                    chunk_index += 1
                                current_text = word
                                current_start_page = page_num
                            else:
                                current_text = f"{current_text} {word}" if current_text else word
                    else:
                        current_text = f"{current_text} {sentence}" if current_text else sentence

                else:
                    current_text = f"{current_text} {sentence}" if current_text else sentence

                current_end_page = page_num

        # Final chunk
        if current_text.strip():
            chunks.append({
                "text": current_text.strip(),
                "page": current_start_page,
                "chunk_index": chunk_index,
                "start_page": current_start_page,
                "end_page": current_end_page,
            })

        log.info("Chunked: %d pages → %d chunks (size=%d, overlap=%d)",
                 len(pages), len(chunks), chunk_size, overlap)

        if on_status:
            on_status(f"Document chunked into {len(chunks)} sections.")

        return chunks

    # ── Summarization ─────────────────────────────────────────────────────

    @staticmethod
    def summarize_for_context(
        text: str, ai_service, filename: str = "",
        max_chunks: int = 10,
    ) -> str:
        """
        Create a condensed context from PDF text for LLM injection.
        Summarizes chunks individually, then combines summaries.
        """
        # Simple sentence-split for summarization (not page-aware)
        raw_chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current = ""
        for s in sentences:
            if len(current) + len(s) + 1 > 2000:
                if current:
                    raw_chunks.append(current.strip())
                current = s
            else:
                current = f"{current} {s}" if current else s
        if current.strip():
            raw_chunks.append(current.strip())

        if not raw_chunks:
            return ""

        # Short text: use directly
        if len(text) <= 3000:
            return f"[Document: {filename or 'uploaded PDF'}]\n{text}"

        # Summarize
        chunks_to_summarize = raw_chunks[:max_chunks]
        summaries = []

        for i, chunk in enumerate(chunks_to_summarize):
            try:
                prompt = (
                    f"Summarize the following document excerpt concisely. "
                    f"Keep key facts, numbers, and important details:\n\n{chunk}"
                )
                messages = [{"role": "user", "content": prompt}]
                response, model, provider, meta = ai_service.generate(messages, prompt)
                summaries.append(response.strip())
            except Exception as exc:
                log.warning("Chunk %d summarization failed: %s", i, exc)
                summaries.append(chunk[:500] + "...")

        combined = "\n\n".join(summaries)
        header = f"[Document: {filename or 'uploaded PDF'}]"
        if len(raw_chunks) > max_chunks:
            header += f" (showing {max_chunks} of {len(raw_chunks)} sections)"

        return f"{header}\n{combined}"

    # ── Utility ───────────────────────────────────────────────────────────

    @staticmethod
    def is_supported(filename: str) -> bool:
        """Check if a file is a supported PDF."""
        return filename.lower().endswith(".pdf")

    @staticmethod
    def clear_cache() -> int:
        """Clear the extraction cache. Returns count of entries removed."""
        count = len(_extraction_cache)
        _extraction_cache.clear()
        log.info("Extraction cache cleared (%d entries)", count)
        return count


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """Normalize whitespace and clean extracted text."""
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    lines = [line.strip() for line in text.split('\n')]
    return '\n'.join(lines).strip()
