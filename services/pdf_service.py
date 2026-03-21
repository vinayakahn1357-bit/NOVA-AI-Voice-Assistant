"""
services/pdf_service.py — Placeholder PDF Service for NOVA
Integration-ready interface for future PDF text extraction.
"""

from utils.logger import get_logger

log = get_logger("pdf")


class PDFService:
    """
    Placeholder for PDF text extraction.
    Future implementation: use PyMuPDF, pdfplumber, or cloud OCR.
    """

    @staticmethod
    def extract_text(file_bytes: bytes, filename: str = "") -> str:
        """
        Extract text content from a PDF file.

        Args:
            file_bytes: Raw PDF file bytes.
            filename: Original filename for logging.

        Returns:
            Extracted text string.

        Raises:
            NotImplementedError: PDF processing is not yet implemented.
        """
        raise NotImplementedError(
            "PDF processing is not yet available. "
            "This feature is planned for a future release."
        )

    @staticmethod
    def is_supported(filename: str) -> bool:
        """Check if a file is a supported PDF."""
        return filename.lower().endswith(".pdf")
