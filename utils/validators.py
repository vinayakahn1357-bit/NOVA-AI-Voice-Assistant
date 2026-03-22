"""
utils/validators.py — Input Validation & Sanitisation for NOVA
"""

import re
from utils.errors import NovaValidationError

# ─── Constants ────────────────────────────────────────────────────────────────
MAX_MESSAGE_LENGTH = 10_000  # characters
MIN_PASSWORD_LENGTH = 6
MAX_EMAIL_LENGTH = 254

# Prompt injection detection patterns (comprehensive)
_INJECTION_PATTERNS = [
    # Classic instruction override
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(all\s+)?above",
    r"disregard\s+(all\s+)?previous",
    r"forget\s+(all\s+)?(previous|your|the)\s+(instructions|rules|guidelines)",
    r"override\s+(your\s+)?(instructions|rules|system|prompt)",
    # Role manipulation
    r"you\s+are\s+now\s+(a|an)\s+",
    r"pretend\s+(you are|to be|you're)\s+",
    r"act\s+as\s+(a|an|if)\s+",
    r"roleplay\s+as\s+",
    r"simulate\s+(being|a|an)\s+",
    # New instruction injection
    r"new\s+instructions?\s*:",
    r"system\s*:\s*",
    r"<\s*system\s*>",
    r"\[INST\]",
    r"\[/INST\]",
    # DAN / jailbreak patterns
    r"(?i)\bDAN\b.*(?:mode|prompt|jailbreak)",
    r"(?i)do\s+anything\s+now",
    r"(?i)jailbreak",
    r"(?i)developer\s+mode",
    r"(?i)unlocked\s+mode",
    r"(?i)no\s+restrictions?\s+mode",
    # Prompt extraction
    r"(?i)(?:show|reveal|display|print|output|repeat|echo)\s+(?:your|the|system)\s+(?:prompt|instructions|rules)",
    r"(?i)what\s+(?:are|is)\s+your\s+(?:system\s+)?(?:prompt|instructions|rules)",
    r"(?i)(?:give|tell|share)\s+me\s+your\s+(?:system\s+)?(?:prompt|instructions|rules)",
    # API / structured injection
    r'"\s*role\s*"\s*:\s*"\s*system\s*"',
    r"<\|im_start\|>",
    r"<\|im_end\|>",
]
_COMPILED_INJECTIONS = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]


def validate_chat_input(message: str) -> str:
    """
    Validate and sanitise a chat message.
    Returns the cleaned message or raises NovaValidationError.
    """
    if not message or not message.strip():
        raise NovaValidationError("No message provided.", code="EMPTY_MESSAGE")

    message = message.strip()

    if len(message) > MAX_MESSAGE_LENGTH:
        raise NovaValidationError(
            f"Message too long ({len(message)} chars, max {MAX_MESSAGE_LENGTH}).",
            code="MESSAGE_TOO_LONG",
        )

    return message


def sanitize_prompt(text: str) -> str:
    """
    Light sanitisation to reduce prompt injection risk.
    Does NOT block the message — just strips dangerous patterns.
    """
    cleaned = text
    for pattern in _COMPILED_INJECTIONS:
        cleaned = pattern.sub("[filtered]", cleaned)
    return cleaned


def check_prompt_injection(text: str) -> bool:
    """Return True if the text contains suspicious prompt injection patterns."""
    lower = text.lower()
    for pattern in _COMPILED_INJECTIONS:
        if pattern.search(lower):
            return True
    return False


def validate_email(email: str) -> str:
    """Validate and normalise an email address."""
    if not email or not email.strip():
        raise NovaValidationError("Email is required.")

    email = email.strip().lower()

    if len(email) > MAX_EMAIL_LENGTH:
        raise NovaValidationError("Email address is too long.")

    if "@" not in email or "." not in email.split("@")[-1]:
        raise NovaValidationError("Please enter a valid email address.")

    return email


def validate_password(password: str) -> str:
    """Validate a password."""
    if not password:
        raise NovaValidationError("Password is required.")

    if len(password) < MIN_PASSWORD_LENGTH:
        raise NovaValidationError(
            f"Password must be at least {MIN_PASSWORD_LENGTH} characters."
        )

    return password


def validate_name(name: str) -> str:
    """Validate a display name."""
    if not name or not name.strip():
        raise NovaValidationError("Name is required.")
    return name.strip()
