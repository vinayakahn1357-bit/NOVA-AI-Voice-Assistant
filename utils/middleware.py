"""
utils/middleware.py — Request Lifecycle Middleware for NOVA
Handles request ID generation, timing, and response headers.
"""

import time
import uuid

from flask import request, g
from utils.logger import get_logger

log = get_logger("middleware")


def register_middleware(app):
    """Register request lifecycle middleware on the Flask app."""

    @app.before_request
    def before_request():
        """Generate request ID and start timing."""
        g.request_id = request.headers.get("X-Request-Id", uuid.uuid4().hex[:12])
        g.request_start = time.time()

    @app.after_request
    def after_request(response):
        """Attach timing and tracing headers to every response."""
        # Request ID for tracing
        request_id = getattr(g, "request_id", "unknown")
        response.headers["X-Request-Id"] = request_id

        # Latency header
        start = getattr(g, "request_start", None)
        if start:
            elapsed_ms = int((time.time() - start) * 1000)
            response.headers["X-Nova-Latency"] = str(elapsed_ms)

        # Model header (set by controller if available)
        model = getattr(g, "nova_model", None)
        if model:
            response.headers["X-Nova-Model"] = model

        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"

        return response
