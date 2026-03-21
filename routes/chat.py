"""
routes/chat.py — Chat Routes Blueprint for NOVA
"""

from flask import Blueprint, request, jsonify

from utils.logger import get_logger
from utils.security import chat_rate_limiter

log = get_logger("routes.chat")

chat_bp = Blueprint("chat", __name__)

# The ChatController instance is injected via init_app()
_chat_controller = None


def init_app(chat_controller):
    """Inject the ChatController instance."""
    global _chat_controller
    _chat_controller = chat_controller


@chat_bp.route("/chat", methods=["POST"])
def chat():
    try:
        session_id = (
            (request.get_json() or {}).get("session_id")
            or request.headers.get("X-Session-Id", "default")
        )
        chat_rate_limiter.check_or_raise(session_id)

        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided", "code": "NO_DATA"}), 400

        result = _chat_controller.handle_chat(data, session_id)
        return jsonify(result)

    except Exception as e:
        # NovaError subclasses are handled by the global error handler
        # This catches unexpected errors
        import traceback
        log.error("Chat error: %s\n%s", e, traceback.format_exc())
        return jsonify({"error": str(e), "code": "UNKNOWN"}), 500


@chat_bp.route("/chat/stream", methods=["POST"])
def chat_stream():
    try:
        session_id = (
            (request.get_json() or {}).get("session_id")
            or request.headers.get("X-Session-Id", "default")
        )
        chat_rate_limiter.check_or_raise(session_id)

        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        return _chat_controller.handle_chat_stream(data, session_id)

    except Exception as e:
        import traceback
        log.error("Chat stream error: %s\n%s", e, traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@chat_bp.route("/reset", methods=["POST"])
def reset_conversation():
    data = request.get_json() or {}
    session_id = data.get("session_id") or request.headers.get("X-Session-Id")
    result = _chat_controller.handle_reset(session_id)
    return jsonify(result)
