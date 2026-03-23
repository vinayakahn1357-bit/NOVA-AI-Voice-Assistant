"""
utils/structured_logger.py — Structured Logging for NOVA (Phase 6)
JSON structured logger for production, text logger for local dev.
Controlled by LOG_FORMAT env var: "json" or "text".
Drop-in replacement for get_logger() — same interface, better output.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone

from flask import g, has_request_context


class StructuredFormatter(logging.Formatter):
    """
    JSON log formatter that includes request context.
    Adds request_id, user_id, timestamp, and latency when available.
    """

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add request context if available
        if has_request_context():
            entry["request_id"] = getattr(g, "request_id", None)
            entry["user_id"] = getattr(g, "user_id", None)
            latency = getattr(g, "latency_ms", None)
            if latency:
                entry["latency_ms"] = latency

        # Add exception info
        if record.exc_info and record.exc_info[0]:
            entry["exception"] = self.formatException(record.exc_info)

        # Remove None values
        entry = {k: v for k, v in entry.items() if v is not None}

        return json.dumps(entry, default=str)


class TextFormatter(logging.Formatter):
    """
    Human-readable text formatter matching existing logger behavior.
    Adds request_id, user_id when available.
    """

    def format(self, record: logging.LogRecord) -> str:
        # Base format
        ts = datetime.now().strftime("%H:%M:%S")
        base = f"[{ts}] {record.levelname:7s} {record.name}: {record.getMessage()}"

        # Add context if available
        ctx_parts = []
        if has_request_context():
            rid = getattr(g, "request_id", None)
            uid = getattr(g, "user_id", None)
            if rid:
                ctx_parts.append(f"req={rid[:8]}")
            if uid:
                ctx_parts.append(f"user={uid}")

        if ctx_parts:
            base += f" [{', '.join(ctx_parts)}]"

        # Add exception info
        if record.exc_info and record.exc_info[0]:
            base += "\n" + self.formatException(record.exc_info)

        return base


def get_structured_logger(name: str, level: int = None) -> logging.Logger:
    """
    Create a structured logger.
    Uses JSON format for production, text format for local.

    Drop-in replacement for get_logger():
        from utils.structured_logger import get_structured_logger as get_logger
    """
    from config import LOG_FORMAT

    logger = logging.getLogger(f"nova.{name}")

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    log_format = (LOG_FORMAT or "text").lower()

    handler = logging.StreamHandler()
    if log_format == "json":
        handler.setFormatter(StructuredFormatter())
    else:
        handler.setFormatter(TextFormatter())

    logger.addHandler(handler)
    logger.setLevel(level or logging.INFO)
    logger.propagate = False

    return logger
