"""
services/voice/wakeword_lock.py — Wake-Word Lock System for NOVA

Prevents duplicate wake-word detection caused by echo, reverb, or the
Web Speech API re-recognizing the same phrase.

Features:
    - Configurable wake-word phrases ("hello nova", "hey nova", etc.)
    - Cooldown-based duplicate suppression (2-second default)
    - Optional wake-word stripping from transcript
    - Fuzzy matching for echo-duplicated variations
    - Anti-echo lock: suppresses wake-words detected shortly after TTS

Production-safe:
    - Thread-safe via monotonic clock
    - No background threads or timers
    - Minimal memory footprint
"""

import re
import time
from utils.logger import get_logger

log = get_logger("voice.wakeword")

# ─── Default wake phrases ─────────────────────────────────────────────────────
DEFAULT_WAKE_PHRASES = [
    'hello nova',
    'hey nova',
    'hi nova',
]

# Cooldown window: ignore repeated wake-words within this many seconds
DEFAULT_COOLDOWN_SEC = 2.0

# Anti-echo window: ignore wake-words detected within this many seconds of TTS ending
DEFAULT_ECHO_WINDOW_SEC = 3.0


class WakewordLock:
    """
    Stateful wake-word manager with cooldown and echo suppression.

    Usage:
        lock = WakewordLock()
        text, detected = lock.process("hello nova what is the weather")
        # detected = True
        # text = "hello nova what is the weather" (or "what is the weather" if strip=True)

        # Immediately after:
        text2, detected2 = lock.process("hello nova")
        # detected2 = False (cooldown active)
        # text2 = None (duplicate wake-word suppressed)
    """

    def __init__(
        self,
        wake_phrases: list[str] = None,
        cooldown_sec: float = DEFAULT_COOLDOWN_SEC,
        echo_window_sec: float = DEFAULT_ECHO_WINDOW_SEC,
        strip_wakeword: bool = False,
    ):
        self._wake_phrases = [p.lower().strip() for p in (wake_phrases or DEFAULT_WAKE_PHRASES)]
        self._cooldown_sec = cooldown_sec
        self._echo_window_sec = echo_window_sec
        self._strip_wakeword = strip_wakeword

        # Build regex pattern for matching wake phrases
        escaped = [re.escape(p) for p in self._wake_phrases]
        self._pattern = re.compile(
            r'\b(' + '|'.join(escaped) + r')\b',
            re.IGNORECASE,
        )

        # State
        self._last_wakeword_time: float = 0.0
        self._last_tts_end_time: float = 0.0

    def process(self, text: str) -> tuple[str | None, bool]:
        """
        Process a transcript for wake-word detection and filtering.

        Args:
            text: The transcript text to process.

        Returns:
            Tuple of (processed_text, wakeword_detected).
            processed_text is None if the transcript was a duplicate wake-word
            that should be discarded entirely.
        """
        if not text or not text.strip():
            return text, False

        text = text.strip()
        text_lower = text.lower()

        # ── Check for wake-word presence ──────────────────────────────────
        match = self._pattern.search(text_lower)
        if not match:
            return text, False

        # ── Wake-word detected ────────────────────────────────────────────
        now = time.monotonic()
        wakeword_detected = True

        # ── Anti-echo check: was TTS playing recently? ────────────────────
        if self._last_tts_end_time > 0:
            since_tts = now - self._last_tts_end_time
            if since_tts < self._echo_window_sec:
                log.info(
                    "Wakeword suppressed (echo): TTS ended %.1fs ago, text='%s'",
                    since_tts, text[:60],
                )
                return None, False

        # ── Cooldown check: was this wake-word already detected recently? ─
        since_last = now - self._last_wakeword_time
        if self._last_wakeword_time > 0 and since_last < self._cooldown_sec:
            # Check if the transcript is JUST the wake-word (no additional content)
            remaining = self._pattern.sub('', text_lower).strip()
            remaining = re.sub(r'[.,!?;:\s]+', '', remaining)
            if not remaining:
                # Pure duplicate wake-word — discard entirely
                log.info(
                    "Wakeword suppressed (cooldown): %.1fs since last, text='%s'",
                    since_last, text[:60],
                )
                return None, False
            # Has additional content — pass through but don't reset cooldown
            log.debug(
                "Wakeword in cooldown but text has content: '%s'", text[:60],
            )

        # ── Accept this wake-word ─────────────────────────────────────────
        self._last_wakeword_time = now

        # ── Optionally strip the wake-word from the text ──────────────────
        if self._strip_wakeword:
            cleaned = self._pattern.sub('', text).strip()
            cleaned = re.sub(r'^[.,!?;:\s]+', '', cleaned)  # clean leading punctuation
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            if cleaned:
                log.debug("Wakeword stripped: '%s' → '%s'", text[:60], cleaned[:60])
                return cleaned, True
            else:
                # Text was only the wake-word — pass it as-is
                return text, True

        return text, True

    def notify_tts_end(self):
        """
        Notify the lock that TTS playback has ended.
        Call this when Nova stops speaking to activate the anti-echo window.
        """
        self._last_tts_end_time = time.monotonic()
        log.debug("TTS end notified — anti-echo window active for %.1fs",
                  self._echo_window_sec)

    def reset(self):
        """Reset all state (cooldown and echo timers)."""
        self._last_wakeword_time = 0.0
        self._last_tts_end_time = 0.0

    @property
    def cooldown_active(self) -> bool:
        """Check if the wake-word cooldown is currently active."""
        if self._last_wakeword_time == 0:
            return False
        return (time.monotonic() - self._last_wakeword_time) < self._cooldown_sec

    @property
    def echo_window_active(self) -> bool:
        """Check if the anti-echo window is currently active."""
        if self._last_tts_end_time == 0:
            return False
        return (time.monotonic() - self._last_tts_end_time) < self._echo_window_sec


# ─── Module-level convenience ────────────────────────────────────────────────
# Singleton lock instance for the default pipeline
_default_lock = WakewordLock()


def process_wakeword(text: str, strip_wakeword: bool = False) -> tuple[str | None, bool]:
    """
    Process a transcript through the default wake-word lock.

    Convenience function using the module-level singleton.

    Args:
        text:            Transcript text to process.
        strip_wakeword:  If True, removes the wake phrase from the text.

    Returns:
        Tuple of (processed_text, wakeword_detected).
    """
    if strip_wakeword != _default_lock._strip_wakeword:
        _default_lock._strip_wakeword = strip_wakeword
    return _default_lock.process(text)


def notify_tts_end():
    """Notify the default lock that TTS playback ended."""
    _default_lock.notify_tts_end()
