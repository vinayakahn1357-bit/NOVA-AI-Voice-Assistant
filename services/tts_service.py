"""
services/tts_service.py — Text-to-Speech Service for NOVA
Uses Microsoft Edge TTS for neural speech synthesis.
"""

import asyncio
import io

import edge_tts

from config import DEFAULT_TTS_VOICE
from utils.logger import get_logger

log = get_logger("tts")


class TTSService:
    """Neural text-to-speech via Microsoft Edge TTS."""

    @staticmethod
    def synthesize(text: str, voice: str | None = None, rate: str = "+0%",
                   pitch: str = "+0Hz") -> bytes:
        """
        Generate speech audio from text.
        Returns: MP3 audio bytes.
        Raises: Exception if synthesis fails.
        """
        voice = voice or DEFAULT_TTS_VOICE

        if not text or not text.strip():
            raise ValueError("No text provided for TTS")

        async def _generate():
            communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
            audio_buf = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_buf.write(chunk["data"])
            audio_buf.seek(0)
            return audio_buf

        loop = asyncio.new_event_loop()
        try:
            audio_buf = loop.run_until_complete(_generate())
        finally:
            loop.close()

        audio_bytes = audio_buf.read()
        if not audio_bytes:
            raise RuntimeError("TTS generated no audio")

        log.info("Synthesized %d bytes of audio (voice=%s)", len(audio_bytes), voice)
        return audio_bytes

    @staticmethod
    def list_voices() -> list:
        """List available English Edge TTS voices."""
        async def _list():
            return await edge_tts.list_voices()

        loop = asyncio.new_event_loop()
        try:
            voices = loop.run_until_complete(_list())
        finally:
            loop.close()

        en_voices = [
            {
                "name": v["Name"],
                "short": v["ShortName"],
                "lang": v["Locale"],
                "gender": v["Gender"],
            }
            for v in voices if v["Locale"].startswith("en")
        ]
        return en_voices
