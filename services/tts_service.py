"""
services/tts_service.py — Text-to-Speech Service for NOVA (Phase 13)
Uses Microsoft Edge TTS for neural speech synthesis.
Phase 13: Personality-aware voice hints for TTS synchronization.
"""

import asyncio
import io
import re

import edge_tts

from config import DEFAULT_TTS_VOICE
from utils.logger import get_logger

log = get_logger("tts")


# ─── Text Preprocessing for Natural Speech ────────────────────────────────────
# Clean text so the neural voices produce clear, realistic output.

def _preprocess_for_speech(text: str) -> str:
    """
    Transform raw text into clean, speech-optimised input.
    Handles URLs, code fragments, special chars, abbreviations,
    and punctuation patterns that cause robotic artefacts.
    """
    if not text:
        return text

    t = text

    # 1. Remove URLs — they sound terrible when read aloud
    t = re.sub(r'https?://\S+', 'a link', t)

    # 2. Remove code blocks and inline code markers
    t = re.sub(r'```[\s\S]*?```', ' code block ', t)
    t = re.sub(r'`([^`]+)`', r'\1', t)

    # 3. Strip markdown formatting
    t = re.sub(r'\*\*(.+?)\*\*', r'\1', t)  # bold
    t = re.sub(r'\*(.+?)\*', r'\1', t)       # italic
    t = re.sub(r'^#{1,6}\s+', '', t, flags=re.MULTILINE)  # headings
    t = re.sub(r'^[-•*]\s+', '', t, flags=re.MULTILINE)   # bullet points
    t = re.sub(r'^\d+\.\s+', '', t, flags=re.MULTILINE)   # numbered lists

    # 4. Remove HTML tags
    t = re.sub(r'<[^>]+>', '', t)

    # 5. Normalise common abbreviations for clearer pronunciation
    abbreviations = {
        r'\bAPI\b': 'A P I',
        r'\bURL\b': 'U R L',
        r'\bHTML\b': 'H T M L',
        r'\bCSS\b': 'C S S',
        r'\bJSON\b': 'Jason',
        r'\bSQL\b': 'sequel',
        r'\bUI\b': 'U I',
        r'\bAI\b': 'A I',
        r'\bML\b': 'M L',
        r'\bJS\b': 'JavaScript',
        r'\bTS\b': 'TypeScript',
        r'\bOS\b': 'O S',
        r'\bCPU\b': 'C P U',
        r'\bGPU\b': 'G P U',
        r'\bRAM\b': 'ram',
        r'\bSSH\b': 'S S H',
        r'\bHTTP\b': 'H T T P',
        r'\betc\b': 'etcetera',
        r'\bvs\b': 'versus',
        r'\be\.g\.': 'for example',
        r'\bi\.e\.': 'that is',
    }
    for pattern, replacement in abbreviations.items():
        t = re.sub(pattern, replacement, t, flags=re.IGNORECASE)

    # 6. Normalise unicode quotes and dashes
    t = t.replace('\u2018', "'").replace('\u2019', "'")
    t = t.replace('\u201c', '"').replace('\u201d', '"')
    t = t.replace('\u2014', ' — ').replace('\u2013', ' – ')
    t = t.replace('\u2026', '...')

    # 7. Remove emoji and special symbols that cause garbled speech
    t = re.sub(
        r'[\U0001F600-\U0001F64F'   # emoticons
        r'\U0001F300-\U0001F5FF'     # misc symbols & pictographs
        r'\U0001F680-\U0001F6FF'     # transport & map symbols
        r'\U0001F1E0-\U0001F1FF'     # flags
        r'\U00002702-\U000027B0'     # dingbats
        r'\U0001FA00-\U0001FA6F'     # chess symbols
        r'\U0001FA70-\U0001FAFF'     # symbols supplement
        r'\U00002600-\U000026FF]+',  # misc symbols
        '', t)

    # 8. Collapse excessive punctuation (e.g. "!!!" → "!", "..." stays as pause)
    t = re.sub(r'!{2,}', '!', t)
    t = re.sub(r'\?{2,}', '?', t)
    t = re.sub(r'\.{4,}', '...', t)

    # 9. Remove parenthetical asides that break speech flow if very short
    t = re.sub(r'\(\s*\)', '', t)  # empty parens

    # 10. Collapse whitespace and trim
    t = re.sub(r'\s+', ' ', t).strip()

    # 11. Ensure text ends with sentence-ending punctuation for natural cadence
    if t and t[-1] not in '.!?':
        t += '.'

    return t


# ─── Phase 13: Voice Hint Mapping ─────────────────────────────────────────────
# Map personality voice hints to Edge TTS rate/pitch parameters.

_PACE_MAP = {
    "slow":   "-15%",
    "normal": "+0%",
    "fast":   "+12%",
}

_PITCH_MAP = {
    # Maps warmth+emphasis to pitch adjustments
    ("warm", "dramatic"): "+2Hz",
    ("warm", "moderate"): "+1Hz",
    ("warm", "subtle"): "+0Hz",
    ("cool", "dramatic"): "+1Hz",
    ("cool", "moderate"): "+0Hz",
    ("cool", "subtle"): "-1Hz",
    ("balanced", "dramatic"): "+1Hz",
    ("balanced", "moderate"): "+0Hz",
    ("balanced", "subtle"): "+0Hz",
}


def _apply_voice_hints(voice_hints: dict) -> tuple:
    """Convert personality voice hints to TTS rate/pitch parameters."""
    pace = voice_hints.get("pace", "normal")
    warmth = voice_hints.get("warmth", "balanced")
    emphasis = voice_hints.get("emphasis", "moderate")

    rate = _PACE_MAP.get(pace, "+0%")
    pitch = _PITCH_MAP.get((warmth, emphasis), "+0Hz")

    return rate, pitch


class TTSService:
    """Neural text-to-speech via Microsoft Edge TTS. Phase 13: Personality-aware."""

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

        # Preprocess text for cleaner, more natural speech
        clean_text = _preprocess_for_speech(text)
        if not clean_text or len(clean_text) < 2:
            raise ValueError("Text too short after preprocessing")

        log.debug("TTS preprocessing: %d → %d chars", len(text), len(clean_text))

        async def _generate():
            communicate = edge_tts.Communicate(clean_text, voice, rate=rate, pitch=pitch)
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

    @staticmethod
    def synthesize_with_personality(text: str, voice: str | None = None,
                                    personality: str = "default") -> bytes:
        """
        Phase 13: Personality-aware TTS synthesis.
        Automatically applies voice hints based on the active personality.

        Returns: MP3 audio bytes.
        """
        try:
            from services.personality_service import get_voice_hints
            hints = get_voice_hints(personality)
            rate, pitch = _apply_voice_hints(hints)
            log.info("TTS personality sync: personality=%s rate=%s pitch=%s",
                     personality, rate, pitch)
        except Exception as exc:
            log.warning("Failed to apply voice hints for '%s': %s", personality, exc)
            rate, pitch = "+0%", "+0Hz"

        return TTSService.synthesize(text, voice=voice, rate=rate, pitch=pitch)

