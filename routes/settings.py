"""
routes/settings.py - Settings Routes with RBAC for NOVA
Admin users get full config access; normal users get filtered, safe access.
Includes debug logging for every request.
"""

from flask import Blueprint, request, jsonify, session

from controllers.settings_controller import (
    get_current_settings, update_settings, list_models,
)
from utils.security import get_user_role, is_admin, login_required
from utils.logger import get_logger

log = get_logger("routes.settings")

settings_bp = Blueprint("settings", __name__)


@settings_bp.route("/settings", methods=["GET", "POST"])
@login_required
def settings():
    email = session.get("user_email", "anonymous")
    role = get_user_role()

    log.info("Settings %s: email=%s role=%s admin=%s",
             request.method, email, role, is_admin())

    if request.method == "GET":
        return jsonify(get_current_settings(role=role))

    data = request.get_json() or {}
    result = update_settings(data, role=role)
    return jsonify(result)


@settings_bp.route("/models")
@login_required
def models():
    return jsonify(list_models())
