"""
utils/security.py — Security Utilities for NOVA
Password hashing, rate limiting, CORS configuration.
"""

import hashlib
import hmac
import time
import uuid
from collections import defaultdict
from functools import wraps
from threading import Lock

from flask import request, session, redirect, jsonify
from utils.errors import NovaRateLimitError, NovaAuthError


# ─── Password Hashing (PBKDF2-HMAC-SHA256) ────────────────────────────────────

def hash_password(password: str, salt: str = None) -> tuple:
    """Return (hashed_hex, salt) using PBKDF2-HMAC-SHA256 with 260k iterations."""
    if salt is None:
        salt = uuid.uuid4().hex
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 260_000)
    return dk.hex(), salt


def verify_password(password: str, stored_hash: str, salt: str) -> bool:
    """Verify a password against a stored hash. Uses constant-time comparison."""
    candidate, _ = hash_password(password, salt)
    return hmac.compare_digest(candidate, stored_hash)


# ─── Login Required Decorator ──────────────────────────────────────────────────

def login_required(f):
    """Decorator: redirects to /login if the user is not authenticated."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("user_id"):
            return redirect("/login?next=" + request.path)
        return f(*args, **kwargs)
    return decorated


# ─── Simple In-Memory Rate Limiter ────────────────────────────────────────────

class RateLimiter:
    """
    Simple per-key rate limiter using a sliding window.
    Not suitable for multi-process deployments — use Redis for that.
    """

    def __init__(self, max_requests: int = 30, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self._hits = defaultdict(list)
        self._lock = Lock()

    def is_allowed(self, key: str) -> bool:
        """Return True if the key is within rate limits."""
        now = time.time()
        cutoff = now - self.window

        with self._lock:
            # Prune old timestamps
            self._hits[key] = [t for t in self._hits[key] if t > cutoff]

            if len(self._hits[key]) >= self.max_requests:
                return False

            self._hits[key].append(now)
            return True

    def check_or_raise(self, key: str):
        """Raise NovaRateLimitError if rate limit exceeded."""
        if not self.is_allowed(key):
            raise NovaRateLimitError(
                "Too many requests. Please slow down.",
                code="RATE_LIMIT",
            )


# Global rate limiter for chat endpoints (30 requests per minute per session)
chat_rate_limiter = RateLimiter(max_requests=30, window_seconds=60)

# Stricter limiter for auth endpoints (10 per minute)
auth_rate_limiter = RateLimiter(max_requests=10, window_seconds=60)


# ─── CORS Helper ──────────────────────────────────────────────────────────────

def configure_cors(app):
    """Register CORS after_request handler on the Flask app."""

    @app.after_request
    def add_cors(response):
        origin = request.headers.get("Origin", "")
        if origin:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
        else:
            response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Session-Id"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return response

    @app.route("/", defaults={"path": ""}, methods=["OPTIONS"])
    @app.route("/<path:path>", methods=["OPTIONS"])
    def handle_options(path):
        return "", 204
