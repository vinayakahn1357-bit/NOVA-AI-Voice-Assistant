"""
utils/response_sanitizer.py — Security Sanitizer for NOVA AI Responses
Last line of defense before responses reach the user.
Scrubs sensitive keywords, API key patterns, and system prompt fragments.
"""

import re
from utils.logger import get_logger

log = get_logger("sanitizer")


# ─── Sensitive Keyword Patterns ───────────────────────────────────────────────
# Each pattern is compiled once at import time for performance.

_SENSITIVE_PATTERNS = [
    # Environment variable names
    r"(?i)\bFLASK_SECRET_KEY\b",
    r"(?i)\bFLASK_SECRET\b",
    r"(?i)\bGROQ_API_KEY\b",
    r"(?i)\bOLLAMA_API_KEY\b",
    r"(?i)\bGOOGLE_CLIENT_SECRET\b",
    r"(?i)\bGOOGLE_CLIENT_ID\b",
    r"(?i)\bNOVA_PROVIDER\b",
    r"(?i)\bNOVA_LIVE_MODE\b",
    r"(?i)\bOLLAMA_CLOUD_URL\b",
    r"(?i)\bOLLAMA_DEFAULT_MODEL\b",
    r"(?i)\bNOVA_TEMPERATURE\b",
    r"(?i)\bNOVA_MAX_TOKENS\b",
    r"(?i)\bADMIN_EMAILS\b",
    r"(?i)\bSECRET_KEY\b",
    r"(?i)\bAPI_KEY\b",

    # API key patterns (Groq keys start with gsk_)
    r"\bgsk_[A-Za-z0-9]{10,}\b",

    # Internal code / config references
    r"(?i)\bDEFAULT_SYSTEM_PROMPT\b",
    r"(?i)\bsystem_prompt\b",
    r"(?i)\bNOVA_SETTINGS\b",
    r"(?i)\bbuild_provider_config\b",
    r"(?i)\bNovaValidationError\b",
    r"(?i)\bNovaProviderError\b",
    r"(?i)\bNovaAuthError\b",

    # System prompt fragments that confirm internal instructions
    r"(?i)\bcore behaviours?\b",
    r"(?i)\bnever mention ollama.{0,15}groq.{0,15}llama\b",
    r"(?i)\byou are nova .{0,30}senior.level ai\b",
    r"(?i)\bprompt_augment\b",
    r"(?i)\bagent.?mode\b",

    # Framework / infrastructure references
    r"(?i)\bvercel\.json\b",
    r"(?i)\bnova_memory\.db\b",
    r"(?i)\bnova_users\.json\b",
]

_COMPILED_SENSITIVE = [re.compile(p) for p in _SENSITIVE_PATTERNS]

# Known system prompt keywords (partial match — these appear in refusal responses)
_SYSTEM_PROMPT_KEYWORDS = [
    "anticipate follow-ups",
    "offer next steps",
    "be an assistant, not a chatbot",
    "remember everything",
    "show, don't tell",
    "never hallucinate facts",
    "never mention ollama",
    "underlying model",
]

# Safe refusal templates for when injection is detected
_SAFE_REFUSALS = [
    "I'm not able to share internal system details. How can I help you with something else?",
    "That information is confidential. Let me know how else I can assist you!",
    "I can't disclose system internals, but I'm happy to help with your actual question.",
]


class ResponseSanitizer:
    """
    Security sanitizer that runs as the last pipeline stage.
    Scrubs sensitive information from AI responses before they reach the user.
    """

    def sanitize(self, text: str, was_injection: bool = False) -> str:
        """
        Sanitize an AI response by removing sensitive keywords.

        Args:
            text: AI response text to sanitize.
            was_injection: if True, the input was flagged as a prompt injection attempt.

        Returns:
            Sanitized response text.
        """
        if not text or not text.strip():
            return text

        # Preserve code blocks — don't scrub inside user-requested code
        code_blocks = []

        def _save_code(match):
            code_blocks.append(match.group(0))
            return f"__CODE_BLOCK_{len(code_blocks) - 1}__"

        processed = re.sub(r'```[\s\S]*?```', _save_code, text)

        # Stage 1: Replace sensitive keyword matches with [redacted]
        redaction_count = 0
        for pattern in _COMPILED_SENSITIVE:
            new_text = pattern.sub("[redacted]", processed)
            if new_text != processed:
                redaction_count += 1
            processed = new_text

        # Stage 2: Check for system prompt keyword leaks
        lower = processed.lower()
        leaked_keywords = [kw for kw in _SYSTEM_PROMPT_KEYWORDS if kw in lower]
        if leaked_keywords:
            log.warning(
                "System prompt keywords leaked in response: %s",
                leaked_keywords,
            )
            # Remove the sentences containing leaked keywords
            for kw in leaked_keywords:
                # Find and remove sentences containing the keyword
                pattern = re.compile(
                    r'[^.!?\n]*\b' + re.escape(kw) + r'\b[^.!?\n]*[.!?\n]?',
                    re.IGNORECASE,
                )
                processed = pattern.sub("", processed)
            redaction_count += len(leaked_keywords)

        # Stage 3: If this was a prompt injection attempt, ensure clean refusal
        if was_injection and redaction_count > 2:
            log.warning("Heavy redaction on injection attempt, using safe refusal")
            return _SAFE_REFUSALS[0]

        # Restore code blocks
        for i, block in enumerate(code_blocks):
            processed = processed.replace(f"__CODE_BLOCK_{i}__", block)

        # Clean up any double spaces or empty lines from redactions
        processed = re.sub(r'  +', ' ', processed)
        processed = re.sub(r'\n{3,}', '\n\n', processed)

        if redaction_count > 0:
            log.info("Sanitized response: %d redactions applied", redaction_count)

        return processed.strip()

    @staticmethod
    def check_for_leaks(text: str) -> dict:
        """
        Analyze a response for potential information leaks.
        Returns a report dict for debug metadata.
        """
        issues = []
        for pattern in _COMPILED_SENSITIVE:
            if pattern.search(text):
                issues.append(pattern.pattern[:40])

        lower = text.lower()
        for kw in _SYSTEM_PROMPT_KEYWORDS:
            if kw in lower:
                issues.append(f"system_prompt_keyword:{kw[:20]}")

        return {
            "has_leaks": len(issues) > 0,
            "leak_count": len(issues),
            "leak_patterns": issues[:5],  # limit for debug output
        }
