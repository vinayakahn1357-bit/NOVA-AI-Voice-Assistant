"""
routes/settings.py — Settings Routes Blueprint for NOVA
"""

from flask import Blueprint, request, jsonify

from controllers.settings_controller import (
    get_current_settings, update_settings, list_models,
)
from utils.logger import get_logger

log = get_logger("routes.settings")

settings_bp = Blueprint("settings", __name__)


@settings_bp.route("/settings", methods=["GET", "POST"])
def settings():
    if request.method == "GET":
        return jsonify(get_current_settings())
    data = request.get_json() or {}
    result = update_settings(data)
    return jsonify(result)


@settings_bp.route("/models")
def models():
    return jsonify(list_models())
