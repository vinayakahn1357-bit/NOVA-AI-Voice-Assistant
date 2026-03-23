"""
services/pdf_service.py — PDF Upload & Q&A Service for NOVA
Extracts text from PDFs, chunks it, and summarizes for LLM context injection.
"""

import re
from utils.logger import get_logger

log = get_logger("pdf")

# ─── Constants ────────────────────────────────────────────────────────────────
MAX_FILE_SIZE = 5 * 1024 * 1024   # 5 MB
MAX_CHUNK_LEN = 2000              # characters per chunk
MAX_CHUNKS_FOR_SUMMARY = 10      # max chunks to summarize (prevents token overload)


class PDFService:
    """
    PDF text extraction, chunking, and summarization.
    Uses pdfplumber for reliable text extraction.
    """

    @staticmethod
    def validate(file_bytes: bytes, filename: str = "") -> str | None:
        """
        Validate a PDF upload.
        Returns error message string, or None if valid.
        """
        if not filename.lower().endswith(".pdf"):
            return "Only .pdf files are supported."
        if len(file_bytes) > MAX_FILE_SIZE:
            size_mb = round(len(file_bytes) / (1024 * 1024), 1)
            return f"File too large ({size_mb}MB). Maximum is 5MB."
        if len(file_bytes) < 10:
            return "File is empty or corrupted."
        return None

    @staticmethod
    def extract_text(file_bytes: bytes, filename: str = "") -> str:
        """
        Extract all text from a PDF file.

        Args:
            file_bytes: Raw PDF file bytes.
            filename: Original filename (for logging).

        Returns:
            Cleaned extracted text string.

        Raises:
            ValueError: If extraction fails or PDF has no text.
        """
        import io
        try:
            import pdfplumber
        except ImportError:
            raise ValueError(
                "pdfplumber is not installed. Run: pip install pdfplumber"
            )

        try:
            pages_text = []
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                total_pages = len(pdf.pages)
                log.info("PDF '%s': %d pages", filename or "upload", total_pages)

                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        pages_text.append(text.strip())

            if not pages_text:
                raise ValueError(
                    "Could not extract text from this PDF. "
                    "It may be image-based or encrypted."
                )

            raw = "\n\n".join(pages_text)
            cleaned = _clean_text(raw)
            log.info("PDF '%s': extracted %d characters from %d/%d pages",
                     filename or "upload", len(cleaned), len(pages_text), total_pages)
            return cleaned

        except ValueError:
            raise
        except Exception as exc:
            log.warning("PDF extraction error: %s", exc, exc_info=True)
            raise ValueError(f"Failed to read PDF: {str(exc)}")

    @staticmethod
    def chunk_text(text: str, max_length: int = MAX_CHUNK_LEN) -> list[str]:
        """
        Split text into chunks, preserving sentence boundaries.

        Args:
            text: Full extracted text.
            max_length: Max characters per chunk.

        Returns:
            List of text chunks.
        """
        if not text:
            return []
        if len(text) <= max_length:
            return [text]

        # Split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current = ""

        for sentence in sentences:
            if len(current) + len(sentence) + 1 > max_length:
                if current:
                    chunks.append(current.strip())
                # If a single sentence exceeds max_length, split by words
                if len(sentence) > max_length:
                    words = sentence.split()
                    current = ""
                    for word in words:
                        if len(current) + len(word) + 1 > max_length:
                            if current:
                                chunks.append(current.strip())
                            current = word
                        else:
                            current = f"{current} {word}" if current else word
                else:
                    current = sentence
            else:
                current = f"{current} {sentence}" if current else sentence

        if current.strip():
            chunks.append(current.strip())

        log.info("Chunked text: %d chars → %d chunks", len(text), len(chunks))
        return chunks

    @staticmethod
    def summarize_for_context(text: str, ai_service, filename: str = "") -> str:
        """
        Create a condensed context from PDF text for LLM injection.
        Summarizes chunks individually, then combines summaries.

        Args:
            text: Full extracted text.
            ai_service: AIService instance for summarization.
            filename: Original filename for context.

        Returns:
            Summarized context string ready for prompt injection.
        """
        chunks = PDFService.chunk_text(text)

        if not chunks:
            return ""

        # If text is small enough, use it directly
        if len(text) <= 3000:
            log.info("PDF context: short enough, using directly (%d chars)", len(text))
            return f"[Document: {filename or 'uploaded PDF'}]\n{text}"

        # Summarize up to MAX_CHUNKS_FOR_SUMMARY chunks
        chunks_to_summarize = chunks[:MAX_CHUNKS_FOR_SUMMARY]
        summaries = []

        for i, chunk in enumerate(chunks_to_summarize):
            try:
                prompt = (
                    f"Summarize the following document excerpt concisely. "
                    f"Keep key facts, numbers, and important details:\n\n{chunk}"
                )
                messages = [{"role": "user", "content": prompt}]
                response, model, provider, meta = ai_service.generate(
                    messages, prompt
                )
                summaries.append(response.strip())
                log.info("Summarized chunk %d/%d (%d chars → %d chars)",
                         i + 1, len(chunks_to_summarize),
                         len(chunk), len(response))
            except Exception as exc:
                log.warning("Chunk %d summarization failed: %s", i, exc)
                # Fallback: use first 500 chars of chunk
                summaries.append(chunk[:500] + "...")

        combined = "\n\n".join(summaries)

        header = f"[Document: {filename or 'uploaded PDF'}]"
        if len(chunks) > MAX_CHUNKS_FOR_SUMMARY:
            header += f" (showing {MAX_CHUNKS_FOR_SUMMARY} of {len(chunks)} sections)"

        context = f"{header}\n{combined}"
        log.info("PDF context ready: %d chars (%d summaries)", len(context), len(summaries))
        return context

    @staticmethod
    def is_supported(filename: str) -> bool:
        """Check if a file is a supported PDF."""
        return filename.lower().endswith(".pdf")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """Normalize whitespace and clean extracted text."""
    # Collapse multiple spaces/tabs
    text = re.sub(r'[ \t]+', ' ', text)
    # Collapse 3+ newlines into 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.split('\n')]
    return '\n'.join(lines).strip()
