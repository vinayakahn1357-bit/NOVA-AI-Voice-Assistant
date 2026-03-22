"""
routes/chat.py — Chat Routes Blueprint for NOVA
"""

from flask import Blueprint, request, jsonify

from utils.logger import get_logger
from utils.security import chat_rate_limiter
from utils.errors import NovaValidationError, NovaRateLimitError, NovaProviderError

log = get_logger("routes.chat")

chat_bp = Blueprint("chat", __name__)

# Injected services
_chat_controller = None
_cache_service = None


def init_app(chat_controller, cache_service=None):
    """Inject the ChatController and CacheService instances."""
    global _chat_controller, _cache_service
    _chat_controller = chat_controller
    _cache_service = cache_service


@chat_bp.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json() or {}
        session_id = data.get("session_id") or request.headers.get("X-Session-Id", "default")
        chat_rate_limiter.check_or_raise(session_id)

        if not data:
            return jsonify({"error": "No data provided", "code": "NO_DATA"}), 400

        result = _chat_controller.handle_chat(data, session_id)
        return jsonify(result)

    except NovaValidationError as e:
        return jsonify({"error": e.message, "code": e.code}), 400
    except NovaRateLimitError as e:
        return jsonify({"error": e.message, "code": e.code}), 429
    except NovaProviderError as e:
        log.error("Provider error: %s", e.message)
        return jsonify({"error": "AI service temporarily unavailable.", "code": e.code}), 503
    except Exception as e:
        import traceback
        log.error("Chat error: %s\n%s", e, traceback.format_exc())
        return jsonify({"error": "An unexpected error occurred.", "code": "INTERNAL_ERROR"}), 500


@chat_bp.route("/chat/stream", methods=["POST"])
def chat_stream():
    try:
        data = request.get_json() or {}
        session_id = data.get("session_id") or request.headers.get("X-Session-Id", "default")
        chat_rate_limiter.check_or_raise(session_id)

        if not data:
            return jsonify({"error": "No data provided"}), 400

        return _chat_controller.handle_chat_stream(data, session_id)

    except NovaValidationError as e:
        return jsonify({"error": e.message, "code": e.code}), 400
    except NovaRateLimitError as e:
        return jsonify({"error": e.message, "code": e.code}), 429
    except NovaProviderError as e:
        log.error("Provider error (stream): %s", e.message)
        return jsonify({"error": "AI service temporarily unavailable.", "code": e.code}), 503
    except Exception as e:
        import traceback
        log.error("Chat stream error: %s\n%s", e, traceback.format_exc())
        return jsonify({"error": "An unexpected error occurred.", "code": "INTERNAL_ERROR"}), 500


@chat_bp.route("/reset", methods=["POST"])
def reset_conversation():
    data = request.get_json() or {}
    session_id = data.get("session_id") or request.headers.get("X-Session-Id")
    result = _chat_controller.handle_reset(session_id)
    return jsonify(result)


@chat_bp.route("/chat/cache/clear", methods=["POST"])
def clear_cache():
    """Clear the response cache."""
    if _cache_service:
        _cache_service.clear()
        return jsonify({"status": "ok", "message": "Cache cleared."})
    return jsonify({"status": "ok", "message": "No cache configured."})


@chat_bp.route("/chat/cache/stats", methods=["GET"])
def cache_stats():
    """Return cache statistics."""
    if _cache_service:
        return jsonify(_cache_service.stats())
    return jsonify({"entries": 0, "message": "No cache configured."})
