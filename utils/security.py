"""
utils/security.py — Security Utilities for NOVA (Phase 6)
Password hashing, JWT authentication, rate limiting, CORS configuration.
Phase 6: JWT support with strict priority — JWT → Session → Unauthenticated.
"""

import os
import hashlib
import hmac
import time
import uuid
from collections import defaultdict
from functools import wraps
from threading import Lock

from flask import request, session, redirect, jsonify, g
from utils.errors import NovaRateLimitError, NovaAuthError
from utils.logger import get_logger

log = get_logger("security")


# ─── Password Hashing (PBKDF2-HMAC-SHA256) ────────────────────────────────────

def hash_password(password: str, salt: str | None = None) -> tuple:
    """Return (hashed_hex, salt) using PBKDF2-HMAC-SHA256 with 260k iterations."""
    if salt is None:
        salt = uuid.uuid4().hex
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 260_000)
    return dk.hex(), salt


def verify_password(password: str, stored_hash: str, salt: str) -> bool:
    """Verify a password against a stored hash. Uses constant-time comparison."""
    candidate, _ = hash_password(password, salt)
    return hmac.compare_digest(candidate, stored_hash)


# ─── JWT Authentication (Phase 6) ─────────────────────────────────────────────

_jwt_secret = None
_jwt_available = False


def _init_jwt():
    """Lazy-init JWT settings."""
    global _jwt_secret, _jwt_available
    from config import JWT_SECRET_KEY
    _jwt_secret = JWT_SECRET_KEY
    _jwt_available = bool(_jwt_secret)
    if _jwt_available:
        log.info("JWT authentication: enabled")


def create_jwt(user_id: str, email: str, role: str = "user") -> str | None:
    """
    Create a JWT token for a user.
    Returns the token string, or None if JWT is not configured.
    """
    global _jwt_secret, _jwt_available
    if _jwt_secret is None:
        _init_jwt()

    if not _jwt_available:
        return None

    try:
        import jwt
        from config import JWT_EXPIRY_HOURS

        payload = {
            "user_id": user_id,
            "email": email,
            "role": role,
            "iat": int(time.time()),
            "exp": int(time.time()) + (JWT_EXPIRY_HOURS * 3600),
        }
        return jwt.encode(payload, _jwt_secret or "", algorithm="HS256")
    except ImportError:
        log.warning("JWT: PyJWT not installed")
        return None
    except Exception as e:
        log.warning("JWT creation failed: %s", e)
        return None


def verify_jwt(token: str) -> dict | None:
    """
    Verify and decode a JWT token.
    Returns the decoded payload dict, or None if invalid.
    """
    global _jwt_secret, _jwt_available
    if _jwt_secret is None:
        _init_jwt()

    if not _jwt_available or not token:
        return None

    try:
        import jwt
        payload = jwt.decode(token, _jwt_secret or "", algorithms=["HS256"])
        return payload
    except ImportError:
        return None
    except Exception:
        return None


def get_current_user() -> dict | None:
    """
    Get the current authenticated user from any auth method.
    Returns: {"user_id": str, "email": str, "role": str} or None if unauthenticated.

    Priority:
    1. JWT token (Authorization: Bearer <token>) → highest priority
    2. Flask session → fallback
    3. Unauthenticated → returns None
    """
    # 1. Check JWT (if present, it ALWAYS wins)
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
        payload = verify_jwt(token)
        if payload:
            return {
                "user_id": payload.get("user_id"),
                "email": payload.get("email"),
                "role": payload.get("role", "user"),
                "auth_method": "jwt",
            }
        # Invalid JWT → don't fallback to session (strict priority)
        return None

    # 2. Flask session
    user_id = session.get("user_id")
    if user_id:
        return {
            "user_id": user_id,
            "email": session.get("user_email", ""),
            "role": get_user_role(),
            "auth_method": "session",
        }

    # 3. Unauthenticated
    return None


# --- Login Required Decorator ---

def login_required(f):
    """
    Decorator: verifies authentication via JWT or session.
    Priority: JWT → Session → 401/redirect.
    Sets g.user_id, g.user_email, g.user_role, g.auth_method for downstream use.
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        user = get_current_user()
        if user:
            g.user_id = user["user_id"]
            g.user_email = user["email"]
            g.user_role = user["role"]
            g.auth_method = user.get("auth_method", "session")
            return f(*args, **kwargs)

        # Not authenticated
        if request.is_json or request.headers.get("Authorization"):
            return jsonify({"error": "Authentication required"}), 401
        return redirect("/login?next=" + request.path)

    return decorated


# --- Role-Based Access Control ---

def is_admin(user_email=None):
    """Check if the current session user (or given email) is an admin."""
    from config import ADMIN_EMAILS
    email = user_email or session.get("user_email", "")
    if not email:
        return False
    return email.strip().lower() in ADMIN_EMAILS


def get_user_role():
    """Return 'admin' or 'user' based on current session."""
    return "admin" if is_admin() else "user"


def admin_required(f):
    """Decorator: returns 403 if user is not an admin."""
    @wraps(f)
    def decorated(*args, **kwargs):
        user = get_current_user()
        if not user:
            return jsonify({"error": "Authentication required"}), 401
        if user.get("role") != "admin" and not is_admin(user.get("email")):
            return jsonify({"error": "Admin access required"}), 403

        g.user_id = user["user_id"]
        g.user_email = user["email"]
        g.user_role = user["role"]
        return f(*args, **kwargs)
    return decorated


# ─── Rate Limiter (In-Memory + Optional Redis) ───────────────────────────────

class RateLimiter:
    """
    Per-key rate limiter using sliding window.
    Phase 6: supports optional Redis backend for distributed rate limiting.
    Falls back to in-memory when Redis is unavailable.
    """

    def __init__(self, max_requests: int = 30, window_seconds: int = 60,
                 redis_service=None):
        self.max_requests = max_requests
        self.window = window_seconds
        self._redis = redis_service
        self._use_redis = redis_service is not None and redis_service.is_available

        # In-memory fallback
        self._hits = defaultdict(list)
        self._lock = Lock()

    def update_redis(self, redis_service):
        """Update the Redis backend (called after Redis is initialized)."""
        self._redis = redis_service
        self._use_redis = redis_service is not None and redis_service.is_available

    def is_allowed(self, key: str) -> bool:
        """Return True if the key is within rate limits."""
        # Try Redis first
        if self._use_redis and self._redis:
            redis_key = f"nova:ratelimit:{key}"
            return self._redis.check_rate_limit(
                redis_key, self.max_requests, self.window
            )

        # In-memory fallback
        now = time.time()
        cutoff = now - self.window

        with self._lock:
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

# Configurable allowlist — override via ALLOWED_ORIGINS env var
# Default: common local dev origins
_ALLOWED_ORIGINS = None


def _get_allowed_origins():
    """Lazy-load allowed origins from config."""
    global _ALLOWED_ORIGINS
    if _ALLOWED_ORIGINS is None:
        import os
        raw = os.getenv("ALLOWED_ORIGINS", "")
        if raw:
            _ALLOWED_ORIGINS = {o.strip().rstrip("/") for o in raw.split(",") if o.strip()}
        else:
            # Sensible defaults for local development
            _ALLOWED_ORIGINS = {
                "http://localhost:5000",
                "http://localhost:3000",
                "http://127.0.0.1:5000",
                "http://127.0.0.1:3000",
            }
            # Auto-add Vercel deployment URLs if available
            vercel_url = os.getenv("VERCEL_URL")
            if vercel_url:
                _ALLOWED_ORIGINS.add(f"https://{vercel_url}")
            vercel_project = os.getenv("VERCEL_PROJECT_PRODUCTION_URL")
            if vercel_project:
                _ALLOWED_ORIGINS.add(f"https://{vercel_project}")
    return _ALLOWED_ORIGINS or set()


def configure_cors(app):
    """Register CORS after_request handler on the Flask app."""

    @app.after_request
    def add_cors(response):
        origin = request.headers.get("Origin", "")
        allowed = _get_allowed_origins()

        if origin and origin.rstrip("/") in allowed:
            # Known origin — allow with credentials
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
        elif not origin:
            # Same-origin request (no Origin header) — allow
            pass
        # else: unknown origin → NO CORS headers → browser blocks the request

        response.headers["Access-Control-Allow-Headers"] = (
            "Content-Type, X-Session-Id, Authorization"
        )
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Vary"] = "Origin"  # Required for correct caching
        return response

    @app.route("/", defaults={"path": ""}, methods=["OPTIONS"])
    @app.route("/<path:path>", methods=["OPTIONS"])
    def handle_options(path):
        return "", 204
