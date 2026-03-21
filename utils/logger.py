"""
utils/logger.py — Structured Logging for NOVA
Replaces all print() calls with proper Python logging.
"""

import logging
import os
import sys


def get_logger(name: str = "NOVA") -> logging.Logger:
    """
    Return a named logger with a consistent format.
    Usage:  from utils.logger import get_logger
            log = get_logger(__name__)
            log.info("Server started")
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        level_str = os.getenv("NOVA_LOG_LEVEL", "INFO").upper()
        level = getattr(logging, level_str, logging.INFO)
        logger.setLevel(level)

        fmt = logging.Formatter(
            "[NOVA.%(name)s] %(levelname)s  %(message)s",
            datefmt="%H:%M:%S",
        )

        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(fmt)
        logger.addHandler(console)

        # Prevent duplicate logs when Flask also adds handlers
        logger.propagate = False

    return logger
