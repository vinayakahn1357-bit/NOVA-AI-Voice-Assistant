"""
routes/chat.py — Chat Routes Blueprint for NOVA
Supports both JSON and multipart/form-data (PDF upload).
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
_pdf_service = None
_ai_service = None


def init_app(chat_controller, cache_service=None, pdf_service=None, ai_service=None):
    """Inject the ChatController, CacheService, PDFService, and AIService instances."""
    global _chat_controller, _cache_service, _pdf_service, _ai_service
    _chat_controller = chat_controller
    _cache_service = cache_service
    _pdf_service = pdf_service
    _ai_service = ai_service


def _parse_request_data():
    """
    Parse request data from either JSON or multipart/form-data.
    Returns (data_dict, file_bytes_or_None, filename_or_None).
    """
    content_type = request.content_type or ""

    if "multipart/form-data" in content_type:
        # FormData — message in form field, file in 'file' field
        message = request.form.get("message", "").strip()
        session_id = (request.form.get("session_id")
                      or request.headers.get("X-Session-Id", "default"))
        data = {"message": message, "session_id": session_id}

        file = request.files.get("file")
        if file and file.filename:
            file_bytes = file.read()
            return data, file_bytes, file.filename
        return data, None, None

    else:
        # Standard JSON
        data = request.get_json() or {}
        return data, None, None


def _process_pdf_context(data: dict, file_bytes: bytes, filename: str) -> dict:
    """
    If a PDF file is attached, extract text, summarize, and inject into message.
    Returns the modified data dict.
    """
    if not _pdf_service or not file_bytes:
        return data

    # Validate
    error = _pdf_service.validate(file_bytes, filename)
    if error:
        raise NovaValidationError(error, code="PDF_INVALID")

    # Extract text
    text = _pdf_service.extract_text(file_bytes, filename)

    # Summarize for context
    if _ai_service:
        context = _pdf_service.summarize_for_context(text, _ai_service, filename)
    else:
        # Fallback: use first 3000 chars
        context = f"[Document: {filename}]\n{text[:3000]}"

    # Inject context into the user's message
    user_message = data.get("message", "").strip()
    if not user_message:
        user_message = "Summarize this document."

    data["message"] = (
        f"The user uploaded a PDF document. Use the following document content "
        f"to answer their question.\n\n{context}\n\n"
        f"User's question: {user_message}"
    )
    log.info("PDF context injected: %s (%d chars context)", filename, len(context))
    return data


@chat_bp.route("/chat", methods=["POST"])
def chat():
    try:
        data, file_bytes, filename = _parse_request_data()
        session_id = data.get("session_id") or request.headers.get("X-Session-Id", "default")
        chat_rate_limiter.check_or_raise(session_id)

        if not data.get("message") and not file_bytes:
            return jsonify({"error": "No data provided", "code": "NO_DATA"}), 400

        # Process PDF if attached
        if file_bytes and filename:
            data = _process_pdf_context(data, file_bytes, filename)

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
        data, file_bytes, filename = _parse_request_data()
        session_id = data.get("session_id") or request.headers.get("X-Session-Id", "default")
        chat_rate_limiter.check_or_raise(session_id)

        if not data.get("message") and not file_bytes:
            return jsonify({"error": "No data provided"}), 400

        # Process PDF if attached
        if file_bytes and filename:
            data = _process_pdf_context(data, file_bytes, filename)

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
