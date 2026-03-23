"""
utils/middleware.py — Request Middleware for NOVA (Phase 6)
Adds request tracing, security headers, and user context.
Phase 6: adds user_id tracking and provider headers.
"""

import time
import uuid

from flask import g, request
from utils.logger import get_logger

log = get_logger("middleware")


def setup_middleware(app):
    """Register before/after request hooks on the Flask app."""

    @app.before_request
    def _before():
        g.request_id = request.headers.get("X-Request-Id", uuid.uuid4().hex[:12])
        g.start_time = time.time()

        # Phase 6: set user context from session/JWT for logging
        from flask import session as flask_session
        g.user_id = getattr(g, "user_id", None) or flask_session.get("user_id")
        g.user_email = getattr(g, "user_email", None) or flask_session.get("user_email")

    @app.after_request
    def _after(response):
        elapsed = (time.time() - g.get("start_time", time.time())) * 1000
        g.latency_ms = int(elapsed)

        response.headers["X-Request-Id"] = g.get("request_id", "")
        response.headers["X-Nova-Latency"] = f"{int(elapsed)}ms"

        # Phase 6: add model and provider headers
        nova_model = g.get("nova_model")
        if nova_model:
            response.headers["X-Nova-Model"] = nova_model

        nova_provider = g.get("nova_provider")
        if nova_provider:
            response.headers["X-Nova-Provider"] = nova_provider

        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"

        return response


# Alias for backward compatibility (app.py imports register_middleware)
register_middleware = setup_middleware
