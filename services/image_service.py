"""
services/image_service.py — Placeholder Image Service for NOVA
Integration-ready interface for future image analysis/vision models.
"""

from utils.logger import get_logger

log = get_logger("image")

_SUPPORTED_FORMATS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")


class ImageService:
    """
    Placeholder for image analysis via vision models.
    Future implementation: use GPT-4V, LLaVA, or Gemini Vision.
    """

    @staticmethod
    def analyze_image(file_bytes: bytes, filename: str = "",
                      prompt: str = "Describe this image.") -> str:
        """
        Analyse an image and return a text description.

        Args:
            file_bytes: Raw image file bytes.
            filename: Original filename.
            prompt: Instruction for the vision model.

        Returns:
            Text description of the image.

        Raises:
            NotImplementedError: Image analysis is not yet implemented.
        """
        raise NotImplementedError(
            "Image analysis is not yet available. "
            "This feature is planned for a future release."
        )

    @staticmethod
    def is_supported(filename: str) -> bool:
        """Check if a file is a supported image format."""
        return filename.lower().endswith(_SUPPORTED_FORMATS)
