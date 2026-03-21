"""
routes/auth.py — Auth Routes Blueprint for NOVA
"""

from flask import Blueprint, request, jsonify, send_from_directory, redirect, session

from config import FRONTEND_DIR
from controllers.auth_controller import (
    handle_register, handle_login, handle_logout, handle_me,
    handle_google_auth, handle_google_callback,
)
from utils.security import login_required, auth_rate_limiter
from utils.logger import get_logger

log = get_logger("routes.auth")

auth_bp = Blueprint("auth", __name__)


@auth_bp.route("/auth/register", methods=["POST"])
def register():
    auth_rate_limiter.check_or_raise(request.remote_addr or "unknown")
    data = request.get_json() or {}
    result = handle_register(data)
    return jsonify(result)


@auth_bp.route("/auth/login", methods=["POST"])
def login():
    auth_rate_limiter.check_or_raise(request.remote_addr or "unknown")
    data = request.get_json() or {}
    result = handle_login(data)
    return jsonify(result)


@auth_bp.route("/auth/logout", methods=["POST"])
def logout():
    result = handle_logout()
    return jsonify(result)


@auth_bp.route("/auth/me", methods=["GET"])
def me():
    user = handle_me()
    if not user:
        return jsonify({"ok": False, "user": None}), 401
    return jsonify({"ok": True, "user": user})


@auth_bp.route("/auth/google")
def google_auth():
    return handle_google_auth()


@auth_bp.route("/auth/google/callback")
def google_callback():
    return handle_google_callback()


# ─── Page Serving ─────────────────────────────────────────────────────────────

@auth_bp.route("/")
def serve_landing():
    """Marketing / landing page."""
    return send_from_directory(FRONTEND_DIR, "landing.html")


@auth_bp.route("/login")
def serve_login():
    """Sign in / sign up page."""
    if session.get("user_id"):
        return redirect("/app")
    return send_from_directory(FRONTEND_DIR, "login.html")


@auth_bp.route("/app")
@login_required
def serve_app():
    """Main NOVA AI assistant application (requires auth)."""
    return send_from_directory(FRONTEND_DIR, "index.html")
