"""
routes/settings.py - Settings Routes with RBAC for NOVA
Admin users get full config access; normal users get filtered, safe access.
"""

from flask import Blueprint, request, jsonify

from controllers.settings_controller import (
    get_current_settings, update_settings, list_models,
)
from utils.security import get_user_role
from utils.logger import get_logger

log = get_logger("routes.settings")

settings_bp = Blueprint("settings", __name__)


@settings_bp.route("/settings", methods=["GET", "POST"])
def settings():
    role = get_user_role()

    if request.method == "GET":
        return jsonify(get_current_settings(role=role))

    data = request.get_json() or {}
    result = update_settings(data, role=role)
    return jsonify(result)


@settings_bp.route("/models")
def models():
    return jsonify(list_models())
