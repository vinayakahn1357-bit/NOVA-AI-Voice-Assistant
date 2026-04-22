"""
routes/memory.py — Memory Routes Blueprint for NOVA
"""

from flask import Blueprint, request, jsonify, session

from config import get_settings, build_provider_config
from utils.logger import get_logger

log = get_logger("routes.memory")

memory_bp = Blueprint("memory", __name__)

# Injected services
_memory_service = None
_session_service = None


def init_app(memory_service, session_service):
    """Inject dependencies."""
    global _memory_service, _session_service
    _memory_service = memory_service
    _session_service = session_service


@memory_bp.route("/memory", methods=["GET"])
def get_memory():
    """Return all of Nova's learned memory as JSON."""
    assert _memory_service is not None
    return jsonify(_memory_service.get_stats())


@memory_bp.route("/memory/reset", methods=["POST"])
def reset_memory():
    """Wipe all learned memory."""
    assert _memory_service is not None
    _memory_service.reset()
    return jsonify({"status": "ok", "message": "Nova's memory has been reset."})


@memory_bp.route("/memory/summary", methods=["POST"])
def generate_summary():
    """Trigger a daily summary for a specific session."""
    data = request.get_json() or {}
    session_id = data.get("session_id") or request.headers.get("X-Session-Id", "default")
    user_id = session.get("user_id", "default")
    assert _session_service is not None
    history = _session_service.get_history(session_id, user_id=user_id)
    if history:
        settings = get_settings()
        active_model = (
            settings["nvidia_model"] if settings.get("provider") == "nvidia"
            else settings["groq_model"]
        )
        assert _memory_service is not None
        _memory_service.generate_daily_summary(
            history, active_model, build_provider_config()
        )
        return jsonify({"status": "ok", "message": "Daily summary generation started."})
    return jsonify({"status": "skipped", "message": "No conversation history for this session."})
