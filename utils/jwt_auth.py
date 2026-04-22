"""
utils/jwt_auth.py — Dedicated JWT Authentication Module for NOVA
Provides stateless token-based authentication using HS256 signing.

This module wraps the core JWT functions from utils/security and adds:
- Configurable expiry (default 7 days)
- Token refresh support
- Structured token payloads
- Extraction helpers for middleware/route use

All functions are safe to call even if PyJWT is not installed or
JWT_SECRET_KEY is not configured — they return None gracefully.
"""

import time

from utils.logger import get_logger

log = get_logger("jwt_auth")

# ── Lazy-loaded config ─────────────────────────────────────────────────────────

_jwt_secret = None
_jwt_expiry_hours = None
_jwt_ready = None  # None = not checked yet


def _ensure_init():
    """Lazy-init JWT settings from config on first use."""
    global _jwt_secret, _jwt_expiry_hours, _jwt_ready
    if _jwt_ready is not None:
        return

    try:
        from config import JWT_SECRET_KEY, JWT_EXPIRY_HOURS
        _jwt_secret = JWT_SECRET_KEY
        _jwt_expiry_hours = JWT_EXPIRY_HOURS or 168  # Default 7 days
        _jwt_ready = bool(_jwt_secret)
        if _jwt_ready:
            log.info("JWT auth module: enabled (expiry=%dh)", _jwt_expiry_hours)
        else:
            log.info("JWT auth module: disabled (no JWT_SECRET_KEY)")
    except Exception as e:
        _jwt_ready = False
        log.warning("JWT auth module: init failed (%s)", e)


# ── Token Generation ──────────────────────────────────────────────────────────

def generate_token(user_id: str, email: str, role: str = "user") -> str | None:
    """
    Generate a JWT token for a user.

    Args:
        user_id: Unique user identifier
        email: User's email address
        role: User role ('user' or 'admin')

    Returns:
        JWT token string, or None if JWT is not configured.
    """
    _ensure_init()
    if not _jwt_ready:
        return None

    try:
        import jwt

        now = int(time.time())
        payload = {
            "user_id": user_id,
            "email": email,
            "role": role,
            "iat": now,
            "exp": now + ((_jwt_expiry_hours or 168) * 3600),
        }
        token = jwt.encode(payload, _jwt_secret or "", algorithm="HS256")
        log.info("JWT generated for user=%s role=%s", email, role)
        return token

    except ImportError:
        log.warning("PyJWT not installed — cannot generate tokens")
        return None
    except Exception as e:
        log.warning("JWT generation failed: %s", e)
        return None


def generate_refresh_token(user_id: str, email: str) -> str | None:
    """
    Generate a long-lived refresh token (30 days).
    Used to obtain new access tokens without re-authentication.
    """
    _ensure_init()
    if not _jwt_ready:
        return None

    try:
        import jwt

        now = int(time.time())
        payload = {
            "user_id": user_id,
            "email": email,
            "type": "refresh",
            "iat": now,
            "exp": now + (30 * 24 * 3600),  # 30 days
        }
        return jwt.encode(payload, _jwt_secret or "", algorithm="HS256")

    except (ImportError, Exception):
        return None


# ── Token Verification ────────────────────────────────────────────────────────

def verify_token(token: str) -> dict | None:
    """
    Verify and decode a JWT token.

    Args:
        token: The JWT token string to verify.

    Returns:
        Decoded payload dict with keys: user_id, email, role, iat, exp
        Returns None if token is invalid, expired, or JWT is not configured.
    """
    _ensure_init()
    if not _jwt_ready or not token:
        return None

    try:
        import jwt
        payload = jwt.decode(token, _jwt_secret or "", algorithms=["HS256"])
        return payload

    except ImportError:
        return None
    except Exception:
        # Covers: ExpiredSignatureError, InvalidTokenError, DecodeError
        return None


def verify_refresh_token(token: str) -> dict | None:
    """
    Verify a refresh token and ensure it's the correct type.
    Returns the payload if valid, None otherwise.
    """
    payload = verify_token(token)
    if payload and payload.get("type") == "refresh":
        return payload
    return None


# ── Extraction Helpers ────────────────────────────────────────────────────────

def extract_user_from_token() -> dict | None:
    """
    Extract user info from the JWT in the current Flask request's
    Authorization header.

    Returns:
        {"user_id": str, "email": str, "role": str} or None

    Usage in routes/middleware:
        user = extract_user_from_token()
        if user:
            g.user_id = user["user_id"]
    """
    try:
        from flask import request
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return None

        token = auth_header[7:].strip()
        payload = verify_token(token)
        if not payload:
            return None

        return {
            "user_id": payload.get("user_id"),
            "email": payload.get("email", ""),
            "role": payload.get("role", "user"),
            "auth_method": "jwt",
        }

    except Exception:
        return None


def is_jwt_enabled() -> bool:
    """Check if JWT authentication is configured and available."""
    _ensure_init()
    return bool(_jwt_ready)
