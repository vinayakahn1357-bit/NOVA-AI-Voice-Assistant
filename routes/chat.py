"""
routes/chat.py — Chat Routes Blueprint for NOVA (Phase 11)
Supports JSON and multipart/form-data (PDF upload).
Phase 8: Persistent document context for multi-turn PDF Q&A.
Phase 9: Per-session personality API endpoints.
Phase 11: Multi-document management, dedicated upload, exam mode, processing status.
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
_document_store = None
_personality_store = None
_retriever = None
_smart_responder = None


def init_app(chat_controller, cache_service=None, pdf_service=None,
             ai_service=None, document_store=None, personality_store=None,
             retriever=None, smart_responder=None):
    """Inject all required service instances."""
    global _chat_controller, _cache_service, _pdf_service, _ai_service
    global _document_store, _personality_store, _retriever, _smart_responder
    _chat_controller = chat_controller
    _cache_service = cache_service
    _pdf_service = pdf_service
    _ai_service = ai_service
    _document_store = document_store
    _personality_store = personality_store
    _retriever = retriever
    _smart_responder = smart_responder


def _parse_request_data():
    """
    Parse request data from either JSON or multipart/form-data.
    Returns (data_dict, file_bytes_or_None, filename_or_None).
    """
    content_type = request.content_type or ""

    if "multipart/form-data" in content_type:
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
        data = request.get_json() or {}
        return data, None, None


def _process_pdf_upload(file_bytes: bytes, filename: str, session_id: str) -> dict:
    """
    Process a PDF file: validate, extract, chunk, index, and store.
    Returns processing result dict with status info.
    """
    if not _pdf_service:
        raise NovaValidationError("PDF service not configured.", code="PDF_UNAVAILABLE")

    statuses = []

    def on_status(msg):
        statuses.append(msg)

    # Validate
    error = _pdf_service.validate(file_bytes, filename)
    if error:
        raise NovaValidationError(error, code="PDF_INVALID")

    on_status("Extracting text...")

    # Extract text (with page tracking)
    extraction = _pdf_service.extract_text(file_bytes, filename, on_status=on_status)

    # Chunk with page awareness
    on_status("Chunking document...")
    chunks = _pdf_service.chunk_text(extraction["pages"], on_status=on_status)

    # Index for retrieval (immediate — no lazy delay)
    doc_hash = extraction["doc_hash"]
    if _retriever:
        on_status("Indexing document for search...")
        _retriever.index_chunks(doc_hash, chunks)

    # Summarize for context (used as fallback / overview)
    on_status("Generating summary...")
    if _ai_service:
        summary = _pdf_service.summarize_for_context(
            extraction["text"], _ai_service, filename
        )
    else:
        summary = f"[Document: {filename}]\n{extraction['text'][:3000]}"

    # Store in multi-document context
    doc_id = None
    if _document_store:
        doc_id = _document_store.add_document(
            session_id, filename, summary, chunks, doc_hash=doc_hash
        )
        on_status("Document ready!")

    log.info("PDF processed: '%s' → %d pages, %d chunks, indexed=%s, doc_id=%s",
             filename, extraction["total_pages"], len(chunks),
             "yes" if _retriever else "no",
             doc_id[:12] if doc_id else "none")

    return {
        "doc_id": doc_id,
        "filename": filename,
        "total_pages": extraction["total_pages"],
        "extracted_pages": extraction["extracted_pages"],
        "chunk_count": len(chunks),
        "indexed": _retriever is not None,
        "statuses": statuses,
    }


def _process_pdf_context(data: dict, file_bytes: bytes, filename: str,
                         session_id: str) -> dict:
    """
    If a PDF file is attached to a chat message, process it and inject context.
    Returns the modified data dict.
    """
    result = _process_pdf_upload(file_bytes, filename, session_id)

    # Inject document context into the user's message
    user_message = data.get("message", "").strip()
    if not user_message:
        user_message = "Summarize this document."

    # Get the active document's summary for initial context
    doc = _document_store.get(session_id) if _document_store else None
    context = doc["summary"] if doc else f"[Document: {filename}]"

    data["message"] = (
        f"The user uploaded a PDF document. Use the following document content "
        f"to answer their question.\n\n{context}\n\n"
        f"User's question: {user_message}"
    )
    data["_pdf_result"] = result
    log.info("PDF context injected: %s (%d chars context)", filename, len(context))
    return data


# ── Chat Endpoint ─────────────────────────────────────────────────────────────

@chat_bp.route("/chat", methods=["POST"])
def chat():
    try:
        data, file_bytes, filename = _parse_request_data()
        session_id = data.get("session_id") or request.headers.get("X-Session-Id", "default")
        chat_rate_limiter.check_or_raise(session_id)

        if not data.get("message") and not file_bytes:
            return jsonify({"error": "No data provided", "code": "NO_DATA"}), 400

        # Process PDF if attached
        pdf_result = None
        if file_bytes and filename:
            data = _process_pdf_context(data, file_bytes, filename, session_id)
            pdf_result = data.pop("_pdf_result", None)

        result = _chat_controller.handle_chat(data, session_id)

        # Include PDF processing info if applicable
        if pdf_result:
            result["pdf"] = pdf_result

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
            data = _process_pdf_context(data, file_bytes, filename, session_id)

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


# ── Phase 11: Dedicated Document Upload ───────────────────────────────────────

@chat_bp.route("/document/upload", methods=["POST"])
def document_upload():
    """
    Dedicated PDF upload endpoint (separate from chat).
    Returns processing status and document info without generating a chat response.
    """
    try:
        session_id = (request.form.get("session_id")
                      or request.headers.get("X-Session-Id", "default"))

        file = request.files.get("file")
        if not file or not file.filename:
            return jsonify({"error": "No file provided.", "code": "NO_FILE"}), 400

        file_bytes = file.read()
        filename = file.filename

        result = _process_pdf_upload(file_bytes, filename, session_id)

        return jsonify({
            "status": "ok",
            "message": f"Document '{filename}' processed successfully.",
            **result,
        })

    except NovaValidationError as e:
        return jsonify({"error": e.message, "code": e.code}), 400
    except Exception as e:
        import traceback
        log.error("Document upload error: %s\n%s", e, traceback.format_exc())
        return jsonify({"error": "Failed to process document.", "code": "UPLOAD_ERROR"}), 500


# ── Phase 11: Document Management Endpoints ───────────────────────────────────

@chat_bp.route("/document/status", methods=["GET"])
def document_status():
    """Return the active document status (multi-doc aware)."""
    session_id = request.headers.get("X-Session-Id", "default")
    if _document_store:
        return jsonify(_document_store.get_status(session_id))
    return jsonify({
        "has_document": False, "filename": None,
        "document_count": 0, "documents": [], "active_document_id": None,
    })


@chat_bp.route("/document/list", methods=["GET"])
def document_list():
    """List all active documents in the session."""
    session_id = request.headers.get("X-Session-Id", "default")
    if _document_store:
        docs = _document_store.list_documents(session_id)
        return jsonify({"documents": docs, "count": len(docs)})
    return jsonify({"documents": [], "count": 0})


@chat_bp.route("/document/switch", methods=["POST"])
def document_switch():
    """Switch the active document by doc_id or filename."""
    data = request.get_json() or {}
    session_id = (data.get("session_id")
                  or request.headers.get("X-Session-Id", "default"))

    doc_id = data.get("doc_id")
    filename = data.get("filename")

    if not _document_store:
        return jsonify({"error": "Document store not configured."}), 503

    if doc_id:
        success = _document_store.set_active_document(session_id, doc_id)
    elif filename:
        success = _document_store.switch_by_filename(session_id, filename)
    else:
        return jsonify({"error": "Provide 'doc_id' or 'filename'."}), 400

    if success:
        status = _document_store.get_status(session_id)
        return jsonify({"status": "ok", "message": f"Switched to '{status['filename']}'.", **status})
    return jsonify({"error": "Document not found in this session."}), 404


@chat_bp.route("/document/remove", methods=["POST"])
def document_remove():
    """Remove a specific document by doc_id or filename."""
    data = request.get_json() or {}
    session_id = (data.get("session_id")
                  or request.headers.get("X-Session-Id", "default"))

    doc_id = data.get("doc_id")
    filename = data.get("filename")

    if not _document_store:
        return jsonify({"error": "Document store not configured."}), 503

    removed = _document_store.remove_document(session_id, doc_id=doc_id, filename=filename)
    if removed:
        # Also clean up retriever index
        if _retriever and doc_id:
            _retriever.remove(doc_id)
        return jsonify({"status": "ok", "message": "Document removed."})
    return jsonify({"error": "Document not found."}), 404


@chat_bp.route("/document/clear", methods=["POST"])
def document_clear():
    """Clear all document contexts for the current session."""
    data = request.get_json() or {}
    session_id = (data.get("session_id")
                  or request.headers.get("X-Session-Id", "default"))
    if _document_store:
        # Get doc hashes before clearing (for retriever cleanup)
        all_docs = _document_store.get_all(session_id)
        removed = _document_store.clear(session_id)

        # Clean up retriever indexes
        if _retriever and all_docs:
            for doc in all_docs:
                doc_hash = doc.get("doc_hash")
                if doc_hash:
                    _retriever.remove(doc_hash)

        if removed:
            return jsonify({"status": "ok", "message": "All document contexts cleared."})
        return jsonify({"status": "ok", "message": "No active documents."})
    return jsonify({"status": "ok", "message": "Document store not configured."})


# ── Phase 11: Exam Mode Endpoint ──────────────────────────────────────────────

@chat_bp.route("/settings/exam-mode", methods=["GET"])
def get_exam_mode():
    """Get current exam mode state."""
    session_id = request.headers.get("X-Session-Id", "default")
    if _smart_responder:
        return jsonify(_smart_responder.get_exam_mode(session_id))
    return jsonify({"enabled": False, "query_count": 0, "auto_detected": False, "manual": False})


@chat_bp.route("/settings/exam-mode", methods=["POST"])
def set_exam_mode():
    """Manually toggle exam mode."""
    data = request.get_json() or {}
    session_id = (data.get("session_id")
                  or request.headers.get("X-Session-Id", "default"))
    enabled = data.get("enabled", False)

    if not _smart_responder:
        return jsonify({"error": "Smart responder not configured."}), 503

    state = _smart_responder.set_exam_mode(session_id, bool(enabled))
    return jsonify({"status": "ok", **state})


# ── Cache Endpoints ───────────────────────────────────────────────────────────

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


# ── Phase 9: Personality Endpoints ────────────────────────────────────────────

@chat_bp.route("/settings/personality", methods=["GET"])
def get_personality():
    """Return the current personality and available options."""
    session_id = request.headers.get("X-Session-Id", "default")
    if _personality_store:
        from services.personality_service import PersonalityStore
        return jsonify({
            "current": _personality_store.get(session_id),
            "info": _personality_store.get_info(session_id),
            "available": _personality_store.list_all(),
        })
    return jsonify({"current": "default", "available": {}})


@chat_bp.route("/settings/personality", methods=["POST"])
def set_personality():
    """Set the personality for the current session."""
    data = request.get_json() or {}
    personality = data.get("personality", "").strip().lower()
    session_id = (data.get("session_id")
                  or request.headers.get("X-Session-Id", "default"))

    if not personality:
        return jsonify({"error": "Missing 'personality' field."}), 400

    if not _personality_store:
        return jsonify({"error": "Personality system not configured."}), 503

    from services.personality_service import VALID_PERSONALITIES
    if personality not in VALID_PERSONALITIES:
        return jsonify({
            "error": f"Invalid personality '{personality}'.",
            "valid": sorted(VALID_PERSONALITIES),
        }), 400

    _personality_store.set(session_id, personality)
    return jsonify({
        "status": "ok",
        "personality": personality,
        "info": _personality_store.get_info(session_id),
    })
