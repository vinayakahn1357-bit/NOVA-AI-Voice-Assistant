"""
controllers/auth_controller.py — Authentication Logic for NOVA
Handles register, login, logout, OAuth, and user management.
Phase 6+: Returns JWT tokens on login/register for stateless auth.
"""

import json
import os
import time
import uuid

from flask import session, request, redirect

from config import (
    GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET,
    USERS_FILE, USERS_FILE_BUNDLED, IS_VERCEL,
)
from utils.logger import get_logger
from utils.errors import NovaValidationError, NovaAuthError
from utils.validators import validate_email, validate_password, validate_name
from utils.security import hash_password, verify_password, is_admin, get_user_role
from utils.jwt_auth import generate_token, is_jwt_enabled

log = get_logger("auth")


# ─── User Store (JSON flat-file) ──────────────────────────────────────────────

def _load_users() -> dict:
    """Load the users JSON file."""
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    # On Vercel cold-start, seed from bundled file
    if IS_VERCEL and os.path.exists(USERS_FILE_BUNDLED):
        try:
            import shutil
            shutil.copy2(USERS_FILE_BUNDLED, USERS_FILE)
            with open(USERS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_users(users: dict):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


# ─── Auth Handlers ────────────────────────────────────────────────────────────

def handle_register(data: dict) -> dict:  # type: ignore[return]
    """
    Register a new user.
    Returns: {"ok": True, "user": {...}, "token": str|None}
    Raises: NovaValidationError, NovaAuthError
    """
    name = validate_name(data.get("name", ""))
    email = validate_email(data.get("email", ""))
    password = validate_password(data.get("password", ""))

    users = _load_users()
    if email in users:
        raise NovaAuthError("An account with this email already exists.", status_code=409)

    hashed, salt = hash_password(password)
    user_id = str(uuid.uuid4())
    role = "admin" if is_admin(email) else "user"
    users[email] = {
        "id":       user_id,
        "name":     name,
        "email":    email,
        "hash":     hashed,
        "salt":     salt,
        "provider": "email",
        "created":  int(time.time()),
    }
    _save_users(users)

    # Session (backward compat for browser-based auth)
    session.permanent = True
    session["user_id"] = user_id
    session["user_email"] = email
    session["user_name"] = name

    # JWT token (stateless auth for API clients)
    token = generate_token(user_id, email, role)

    log.info("New user registered: %s (jwt=%s)", email, "yes" if token else "no")
    result: dict = {"ok": True, "user": {"name": name, "email": email}}
    if token:
        result["token"] = token
    return result


def handle_login(data: dict) -> dict:  # type: ignore[return]
    """
    Log in a user.
    Returns: {"ok": True, "user": {...}, "token": str|None}
    Raises: NovaAuthError
    """
    email = validate_email(data.get("email", ""))
    password = data.get("password", "")
    if not password:
        raise NovaValidationError("Email and password are required.")

    users = _load_users()
    user = users.get(email)
    if not user:
        raise NovaAuthError("Incorrect email or password.")

    if user.get("provider") == "google" and not user.get("hash"):
        raise NovaAuthError(
            "This account uses Google sign-in. Please click 'Continue with Google' instead."
        )

    if not verify_password(password, user["hash"], user["salt"]):
        raise NovaAuthError("Incorrect email or password.")

    # Session (backward compat for browser-based auth)
    session.permanent = True
    session["user_id"] = user["id"]
    session["user_email"] = email
    session["user_name"] = user["name"]

    # JWT token (stateless auth for API clients)
    role = "admin" if is_admin(email) else "user"
    token = generate_token(user["id"], email, role)

    log.info("User logged in: %s (jwt=%s)", email, "yes" if token else "no")
    result: dict = {"ok": True, "user": {"name": user["name"], "email": email}}
    if token:
        result["token"] = token
    return result


def handle_logout():
    """Log out the current user."""
    session.clear()
    return {"ok": True}


def handle_me() -> dict | None:
    """
    Return the current authenticated user, or None.
    Supports both JWT and session authentication.
    """
    from utils.security import get_current_user

    user = get_current_user()
    if not user:
        return None

    email = user.get("email", "")
    user_id = user.get("user_id", "")
    name = session.get("user_name", email.split("@")[0] if email else "")

    return {
        "id":          user_id,
        "name":        name,
        "email":       email,
        "is_admin":    is_admin(email),
        "role":        user.get("role", "user"),
        "auth_method": user.get("auth_method", "session"),
    }


# ─── Google OAuth ──────────────────────────────────────────────────────────────

def get_base_url() -> str:
    """Build the canonical base URL for OAuth redirect URIs."""
    host = request.host or ""
    is_local = host.startswith("localhost") or host.startswith("127.0.0.1")

    if is_local:
        return request.host_url.rstrip("/")

    base = os.getenv("NOVA_BASE_URL", "").rstrip("/")
    if base:
        return base

    proto = request.headers.get("X-Forwarded-Proto", "").split(",")[0].strip()
    fwd_host = request.headers.get("X-Forwarded-Host", "").split(",")[0].strip()
    if not fwd_host:
        fwd_host = host

    if proto in ("https", "http"):
        return f"{proto}://{fwd_host}"

    return request.host_url.rstrip("/")


def handle_google_auth():
    """Redirect to Google's OAuth consent screen."""
    if not GOOGLE_CLIENT_ID:
        return redirect("/login?error=google_not_configured")

    state = uuid.uuid4().hex
    session["oauth_state"] = state

    base = get_base_url()
    redirect_uri = base + "/auth/google/callback"
    log.info("Google OAuth → redirect_uri: %s", redirect_uri)

    from urllib.parse import urlencode
    params = {
        "client_id":     GOOGLE_CLIENT_ID,
        "redirect_uri":  redirect_uri,
        "response_type": "code",
        "scope":         "openid email profile",
        "access_type":   "offline",
        "prompt":        "select_account",
        "state":         state,
    }
    return redirect("https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(params))


def handle_google_callback():
    """Google OAuth callback — exchanges code for tokens and logs the user in."""
    import requests as req

    error = request.args.get("error")
    if error:
        log.warning("Google OAuth error from Google: %s", error)
        return redirect("/login?error=google_cancelled")

    code = request.args.get("code")
    state = request.args.get("state", "")

    if not code:
        return redirect("/login?error=google_failed")

    expected_state = session.pop("oauth_state", None)
    if not expected_state or state != expected_state:
        log.warning("Google OAuth CSRF state mismatch!")
        return redirect("/login?error=google_failed")

    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        return redirect("/login?error=google_not_configured")

    base = get_base_url()
    redirect_uri = base + "/auth/google/callback"
    log.info("Google callback → redirect_uri: %s", redirect_uri)

    # Exchange code for access token
    token_res = req.post(
        "https://oauth2.googleapis.com/token",
        data={
            "code":          code,
            "client_id":     GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uri":  redirect_uri,
            "grant_type":    "authorization_code",
        },
        timeout=15,
    )
    if not token_res.ok:
        log.error("Google token exchange FAILED: %d — %s", token_res.status_code, token_res.text[:300])
        return redirect("/login?error=google_token_failed")

    token_data = token_res.json()
    access_token = token_data.get("access_token", "")
    if not access_token:
        log.error("Google token response missing access_token: %s", token_data)
        return redirect("/login?error=google_token_failed")

    # Fetch user profile
    userinfo_res = req.get(
        "https://www.googleapis.com/oauth2/v3/userinfo",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=10,
    )
    if not userinfo_res.ok:
        log.error("Google userinfo FAILED: %d — %s", userinfo_res.status_code, userinfo_res.text[:200])
        return redirect("/login?error=google_userinfo_failed")

    info = userinfo_res.json()
    email = info.get("email", "").strip().lower()
    name = info.get("name", "").strip() or email.split("@")[0]

    if not email:
        log.error("Google userinfo returned no email address!")
        return redirect("/login?error=google_userinfo_failed")

    # Upsert user
    users = _load_users()
    if email not in users:
        users[email] = {
            "id":       str(uuid.uuid4()),
            "name":     name,
            "email":    email,
            "hash":     "",
            "salt":     "",
            "provider": "google",
            "created":  int(time.time()),
        }
        _save_users(users)
        log.info("New Google user created: %s", email)
    else:
        if name and users[email].get("name") != name:
            users[email]["name"] = name
            _save_users(users)

    # Session (for browser redirect flow)
    session.permanent = True
    session["user_id"] = users[email]["id"]
    session["user_email"] = email
    session["user_name"] = users[email]["name"]

    # Generate JWT for potential API use (stored in cookie for SPA pickup)
    role = "admin" if is_admin(email) else "user"
    token = generate_token(users[email]["id"], email, role)
    if token:
        session["jwt_token"] = token

    log.info("Google login success: %s (jwt=%s)", email, "yes" if token else "no")
    return redirect("/app")
