"""
controllers/chat_controller.py — Chat Request Processing for NOVA (Phase 11)
Orchestrates: validate → pipeline/agent → save → return.
Phase 6: user_id multi-user isolation.
Phase 7: detects complex tasks → routes to AgentRunner with step streaming.
Phase 8: persistent document context injection for multi-turn PDF Q&A.
Phase 9: per-session personality system for response style adaptation.
Phase 10: ML-based automatic personality prediction.
Phase 11: RAG-based retrieval, smart follow-ups, exam mode, page citations.
"""

import json
import re
import time

from flask import Response, stream_with_context, g, session

from config import (
    get_settings, build_provider_config,
    ENABLE_DOCUMENT_EMBEDDINGS,
)
from utils.logger import get_logger
from utils.validators import validate_chat_input
from utils.errors import NovaValidationError

log = get_logger("chat")

# ── Document clear triggers (natural language) ────────────────────────────────
_DOC_CLEAR_PATTERNS = [
    r"\bclear\s+document\b",
    r"\bremove\s+(?:the\s+)?pdf\b",
    r"\bforget\s+(?:the\s+)?document\b",
    r"\bnew\s+document\b",
    r"\bremove\s+(?:the\s+)?document\b",
    r"\bclear\s+(?:the\s+)?pdf\b",
]
_DOC_CLEAR_RE = re.compile("|".join(_DOC_CLEAR_PATTERNS), re.IGNORECASE)

# ── Document switch triggers ─────────────────────────────────────────────────
_DOC_SWITCH_PATTERNS = [
    r"\bswitch\s+(?:to\s+)?(?:document|pdf|file)\s+(.+)",
    r"\buse\s+(?:document|pdf|file)\s+(.+)",
    r"\bopen\s+(.+\.pdf)\b",
]
_DOC_SWITCH_RE = re.compile("|".join(_DOC_SWITCH_PATTERNS), re.IGNORECASE)


def _get_user_id() -> str:
    """
    Extract user_id from the current request context.
    Priority: g.user_id (set by JWT middleware) → session → "default"
    """
    uid = getattr(g, "user_id", None)
    if uid:
        return uid
    uid = session.get("user_id")
    if uid:
        return uid
    return "default"


class ChatController:
    """
    Handles chat request processing and orchestration.
    Phase 11: RAG retrieval, smart follow-ups, exam mode, multi-doc.
    """

    def __init__(self, ai_service, session_service, memory_service, command_service,
                 agent_engine=None, response_pipeline=None, agent_runner=None,
                 document_store=None, personality_store=None, personality_model=None,
                 retriever=None, smart_responder=None):
        self._ai = ai_service
        self._session = session_service
        self._memory = memory_service
        self._commands = command_service
        self._agent = agent_engine
        self._pipeline = response_pipeline
        self._agent_runner = agent_runner            # Phase 7
        self._document_store = document_store        # Phase 8/11
        self._personality_store = personality_store   # Phase 9
        self._personality_model = personality_model   # Phase 10
        self._retriever = retriever                  # Phase 11: RAG retriever
        self._smart_responder = smart_responder      # Phase 11: Smart response builder

    # ── Document Context Helpers ──────────────────────────────────────────

    def _is_doc_clear_request(self, message: str) -> bool:
        """Check if user wants to clear the active document context."""
        return bool(_DOC_CLEAR_RE.search(message))

    def _is_doc_switch_request(self, message: str) -> str | None:
        """Check if user wants to switch active document. Returns filename or None."""
        match = _DOC_SWITCH_RE.search(message)
        if match:
            # Get the first non-None group
            for group in match.groups():
                if group:
                    return group.strip().strip("'\"")
        return None

    def _handle_doc_clear(self, session_id: str) -> dict:
        """Clear document context and return a confirmation response."""
        if self._document_store and self._document_store.has_document(session_id):
            docs = self._document_store.list_documents(session_id)
            count = len(docs)
            filenames = ", ".join(d["filename"] for d in docs)

            # Also clean up retriever indexes
            if self._retriever:
                all_docs = self._document_store.get_all(session_id)
                for doc in all_docs:
                    doc_hash = doc.get("doc_hash")
                    if doc_hash:
                        self._retriever.remove(doc_hash)

            self._document_store.clear(session_id)

            # Clear exam state
            if self._smart_responder:
                self._smart_responder.clear_exam_state(session_id)

            return {
                "reply": (
                    f"📄 All document contexts cleared. Removed **{count}** document(s): {filenames}.\n\n"
                    f"You can upload new PDFs or continue chatting normally."
                ),
                "session_id": session_id,
                "meta": {
                    "model": "nova-commands",
                    "provider": "internal",
                    "mode": "document_clear",
                },
            }
        return {
            "reply": "No active documents to clear. You can upload a PDF anytime using the attach button.",
            "session_id": session_id,
            "meta": {
                "model": "nova-commands",
                "provider": "internal",
                "mode": "document_clear",
            },
        }

    def _handle_doc_switch(self, session_id: str, filename: str) -> dict:
        """Switch active document and return confirmation."""
        if not self._document_store:
            return {
                "reply": "Document management is not available.",
                "session_id": session_id,
                "meta": {"model": "nova-commands", "provider": "internal", "mode": "document_switch"},
            }

        if self._document_store.switch_by_filename(session_id, filename):
            doc = self._document_store.get_active_document(session_id)
            return {
                "reply": (
                    f"📄 Switched to **{doc['filename']}**. "
                    f"All follow-up questions will now reference this document."
                ),
                "session_id": session_id,
                "meta": {
                    "model": "nova-commands",
                    "provider": "internal",
                    "mode": "document_switch",
                    "active_document": doc["filename"],
                },
            }

        # Document not found — show available docs
        docs = self._document_store.list_documents(session_id)
        if docs:
            doc_list = "\n".join(
                f"  • {'🟢' if d['is_active'] else '⚪'} {d['filename']}"
                for d in docs
            )
            return {
                "reply": (
                    f"Document '{filename}' not found. Available documents:\n{doc_list}\n\n"
                    f"Say **switch to [filename]** to change."
                ),
                "session_id": session_id,
                "meta": {"model": "nova-commands", "provider": "internal", "mode": "document_switch"},
            }

        return {
            "reply": "No documents are currently loaded. Upload a PDF first.",
            "session_id": session_id,
            "meta": {"model": "nova-commands", "provider": "internal", "mode": "document_switch"},
        }

    def _inject_document_context(self, message: str, session_id: str) -> str:
        """
        Phase 11: RAG-based context injection.
        Uses retriever to find relevant chunks instead of injecting full summary.
        Falls back to summary if retriever is unavailable.
        """
        if not self._document_store:
            return message

        doc = self._document_store.get(session_id)
        if not doc:
            return message

        # Skip injection if message already contains document context
        if "The user uploaded a PDF document" in message:
            return message

        filename = doc["filename"]
        doc_hash = doc.get("doc_hash", "")

        # Phase 11: RAG retrieval
        if (self._retriever
                and ENABLE_DOCUMENT_EMBEDDINGS
                and doc_hash
                and self._retriever.is_indexed(doc_hash)):
            retrieved = self._retriever.retrieve(doc_hash, message)
            if retrieved:
                from services.smart_responder import SmartResponder
                context = SmartResponder.build_retrieval_context(retrieved, filename)

                augmented = (
                    f"You are answering based on a previously uploaded document: {filename}.\n\n"
                    f"Relevant document sections (retrieved by relevance):\n{context}\n\n"
                    f"When citing information, mention the source page number.\n\n"
                    f"User's question: {message}"
                )
                log.info("RAG context injected: %d chunks from '%s' for session %s",
                         len(retrieved), filename, session_id)
                return augmented

        # Fallback: summary-based injection
        context = (
            f"You are answering based on a previously uploaded document: "
            f"{filename}.\n\n"
            f"Document context:\n{doc['summary']}\n\n"
            f"User's question: {message}"
        )
        log.info("Summary context injected from store for session %s ('%s')",
                 session_id, filename)
        return context

    def _get_retrieved_chunks(self, message: str, session_id: str) -> list[dict]:
        """Get retrieved chunks for the current query (for citation formatting)."""
        if not self._retriever or not self._document_store:
            return []

        doc = self._document_store.get(session_id)
        if not doc:
            return []

        doc_hash = doc.get("doc_hash", "")
        if not doc_hash or not self._retriever.is_indexed(doc_hash):
            return []

        return self._retriever.retrieve(doc_hash, message)

    # ── Chat Handlers ─────────────────────────────────────────────────────

    def handle_chat(self, data: dict, session_id: str) -> dict:
        """
        Handle a non-streaming chat request via the response pipeline.
        Phase 11: RAG retrieval, smart follow-ups, exam mode.
        """
        t0 = time.time()
        user_message = validate_chat_input(data.get("message", ""))
        user_id = _get_user_id()

        # ── Document clear detection ─────────────────────────────────
        if self._is_doc_clear_request(user_message):
            return self._handle_doc_clear(session_id)

        # ── Document switch detection ────────────────────────────────
        switch_target = self._is_doc_switch_request(user_message)
        if switch_target:
            return self._handle_doc_switch(session_id, switch_target)

        # ── Phase 11: Exam mode detection ────────────────────────────
        is_exam = False
        if self._smart_responder:
            is_exam = self._smart_responder.detect_exam_intent(session_id, user_message)

        # ── Command detection ────────────────────────────────────────
        intent = self._commands.detect_intent(user_message)
        if intent["type"] == "command":
            result = self._commands.execute(
                intent["command"], intent["args"], session_id
            )
            return {
                "reply": result["response"],
                "session_id": session_id,
                "meta": {
                    "model": "nova-commands",
                    "provider": "internal",
                    "mode": "command",
                    "latency_ms": int((time.time() - t0) * 1000),
                    "command": intent["command"],
                    "action": result.get("action"),
                },
            }

        # ── Agent detection (document-aware) ─────────────────────────
        has_doc = bool(self._document_store
                       and self._document_store.has_document(session_id))
        if self._agent_runner:
            if self._should_use_document_agent(user_message, has_doc):
                return self._handle_agent_chat(
                    user_message, session_id, user_id, t0, with_document=True
                )
            elif self._should_use_agent(user_message):
                return self._handle_agent_chat(
                    user_message, session_id, user_id, t0, with_document=False
                )

        # ── Phase 11: RAG context injection ──────────────────────────
        augmented_message = self._inject_document_context(user_message, session_id)

        # ── Phase 11: Exam mode prompt enhancement ───────────────────
        if is_exam and self._smart_responder:
            augmented_message = self._smart_responder.enhance_prompt_for_exam(augmented_message)

        # ── Personality (user-selected > ML > default) ───────────────
        personality, ml_meta = self._get_personality(session_id, user_message)

        # ── Standard pipeline ────────────────────────────────────────
        self._session.append_message(session_id, "user", user_message, user_id=user_id)
        history = self._session.get_history(session_id, user_id=user_id)

        if self._pipeline:
            pipeline_result = self._pipeline.execute(
                history, augmented_message, personality=personality
            )
            ai_response = pipeline_result["reply"]
            active_model = pipeline_result["model"]
            provider = pipeline_result["provider"]
            meta = pipeline_result["meta"]
        else:
            ai_response, active_model, provider, ai_meta = self._ai.generate(
                history, augmented_message, personality=personality
            )
            meta = {
                "model": active_model,
                "provider": provider,
                "mode": "normal",
                "latency_ms": int((time.time() - t0) * 1000),
            }

        # Phase 9: Include personality in meta
        if personality != "default":
            meta["personality"] = personality

        # Phase 10: Include ML prediction meta
        if ml_meta:
            meta.update(ml_meta)

        # ── Phase 11: Smart response formatting ──────────────────────
        retrieved_chunks = self._get_retrieved_chunks(user_message, session_id) if has_doc else []
        doc_filename = ""
        if has_doc and self._document_store:
            active_doc = self._document_store.get(session_id)
            if active_doc:
                doc_filename = active_doc.get("filename", "")
                meta["active_document"] = doc_filename

        if self._smart_responder:
            smart_result = self._smart_responder.format_response(
                ai_response, user_message, session_id,
                retrieved_chunks=retrieved_chunks,
                has_document=has_doc,
                doc_filename=doc_filename,
            )
            ai_response = smart_result["reply"]
            meta["suggestions"] = smart_result["suggestions"]
            meta["exam_mode"] = smart_result["exam_mode"]
            if smart_result["citations"]:
                meta["citations"] = smart_result["citations"]
        elif has_doc:
            # Fallback: at least include document info
            active_doc = self._document_store.get(session_id) if self._document_store else None
            if active_doc:
                meta["active_document"] = active_doc["filename"]

        self._session.append_message(session_id, "nova", ai_response, user_id=user_id)

        self._memory.record_turn(user_id=user_id)
        self._memory.extract_and_store(
            user_message, ai_response, active_model, build_provider_config(),
            user_id=user_id,
        )

        elapsed_ms = int((time.time() - t0) * 1000)
        meta["latency_ms"] = elapsed_ms
        meta["model"] = active_model
        meta["provider"] = provider

        log.info("Chat complete: %dms provider=%s model=%s user=%s",
                 elapsed_ms, provider, active_model, user_id)

        try:
            g.nova_model = active_model
        except RuntimeError:
            pass

        return {
            "reply": ai_response,
            "session_id": session_id,
            "meta": meta,
        }

    def _handle_agent_chat(self, user_message: str, session_id: str,
                            user_id: str, t0: float,
                            with_document: bool = False) -> dict:
        """
        Handle a chat request via AgentRunner (Phase 7).
        Phase 8: Passes document context when available.
        Falls back to normal pipeline on failure.
        """
        doc_context = None
        if with_document and self._document_store:
            doc_context = self._document_store.get(session_id)

        mode_label = "document-agent" if doc_context else "agent"
        log.info("%s mode activated: user=%s task='%.60s'",
                 mode_label.title(), user_id, user_message)

        self._session.append_message(session_id, "user", user_message, user_id=user_id)
        history = self._session.get_history(session_id, user_id=user_id)

        try:
            assert self._agent_runner is not None
            result = self._agent_runner.run(
                task=user_message,
                history=history,
                user_id=user_id,
                document_context=doc_context,
            )

            ai_response = result.final_answer
            self._session.append_message(session_id, "nova", ai_response, user_id=user_id)

            self._memory.record_turn(user_id=user_id)
            if hasattr(self._memory, "store_agent_run"):
                self._memory.store_agent_run(user_id, user_message, result)

            elapsed_ms = int((time.time() - t0) * 1000)

            meta = {
                "mode": mode_label,
                "model": "nova-agent",
                "provider": "internal",
                "latency_ms": elapsed_ms,
                "steps_taken": result.steps_taken,
                "tools_used": result.tools_used,
            }
            if doc_context:
                meta["active_document"] = doc_context.get("filename")

            # Phase 11: Add suggestions for agent results too
            if self._smart_responder:
                has_doc = (self._document_store
                           and self._document_store.has_document(session_id))
                suggestions = self._smart_responder.generate_suggestions(
                    user_message, has_doc,
                    self._smart_responder.is_exam_mode(session_id),
                    doc_context.get("filename", "") if doc_context else "",
                )
                meta["suggestions"] = suggestions

            return {
                "reply": ai_response,
                "session_id": session_id,
                "meta": meta,
            }

        except Exception as exc:
            log.warning("Agent failed, falling back to pipeline: %s", exc)
            augmented_message = self._inject_document_context(user_message, session_id)
            if self._pipeline:
                pipeline_result = self._pipeline.execute(history, augmented_message)
                ai_response = pipeline_result["reply"]
                self._session.append_message(session_id, "nova", ai_response, user_id=user_id)
                return {
                    "reply": ai_response,
                    "session_id": session_id,
                    "meta": {
                        **pipeline_result["meta"],
                        "mode": "fallback",
                        "agent_error": str(exc),
                    },
                }
            raise

    def handle_chat_stream(self, data: dict, session_id: str) -> Response:
        """
        Handle a streaming (SSE) chat request.
        Phase 11: RAG context, exam mode, smart follow-ups.
        """
        t0 = time.time()
        user_message = validate_chat_input(data.get("message", ""))
        user_id = _get_user_id()

        # ── Document clear detection ─────────────────────────────────
        if self._is_doc_clear_request(user_message):
            result = self._handle_doc_clear(session_id)

            def _doc_clear_gen():
                yield f'data: {json.dumps({"token": result["reply"]})}\n\n'
                yield f'data: {json.dumps({"done": True, "session_id": session_id, "model": "nova-commands"})}\n\n'

            return Response(
                stream_with_context(_doc_clear_gen()),  # type: ignore[arg-type]
                mimetype="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "X-Response-Time": str(int((time.time() - t0) * 1000)),
                },
            )

        # ── Document switch detection ────────────────────────────────
        switch_target = self._is_doc_switch_request(user_message)
        if switch_target:
            result = self._handle_doc_switch(session_id, switch_target)

            def _switch_gen():
                yield f'data: {json.dumps({"token": result["reply"]})}\n\n'
                yield f'data: {json.dumps({"done": True, "session_id": session_id, "model": "nova-commands"})}\n\n'

            return Response(
                stream_with_context(_switch_gen()),  # type: ignore[arg-type]
                mimetype="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "X-Response-Time": str(int((time.time() - t0) * 1000)),
                },
            )

        # ── Phase 11: Exam mode detection ────────────────────────────
        is_exam = False
        if self._smart_responder:
            is_exam = self._smart_responder.detect_exam_intent(session_id, user_message)

        # ── Command detection ────────────────────────────────────────
        intent = self._commands.detect_intent(user_message)
        if intent["type"] == "command":
            result = self._commands.execute(
                intent["command"], intent["args"], session_id
            )

            def _cmd_gen():
                yield f'data: {json.dumps({"token": result["response"]})}\n\n'
                yield f'data: {json.dumps({"done": True, "session_id": session_id, "model": "nova-commands"})}\n\n'

            return Response(
                stream_with_context(_cmd_gen()),  # type: ignore[arg-type]
                mimetype="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "X-Response-Time": str(int((time.time() - t0) * 1000)),
                },
            )

        # ── Agent streaming (document-aware) ─────────────────────────
        has_doc = bool(self._document_store
                       and self._document_store.has_document(session_id))
        if self._agent_runner:
            if self._should_use_document_agent(user_message, has_doc):
                return self._handle_agent_stream(
                    user_message, session_id, user_id, t0, with_document=True
                )
            elif self._should_use_agent(user_message):
                return self._handle_agent_stream(
                    user_message, session_id, user_id, t0, with_document=False
                )

        # ── Phase 11: RAG context injection ──────────────────────────
        augmented_message = self._inject_document_context(user_message, session_id)

        # ── Phase 11: Exam mode prompt enhancement ───────────────────
        if is_exam and self._smart_responder:
            augmented_message = self._smart_responder.enhance_prompt_for_exam(augmented_message)

        # ── Personality (user-selected > ML > default) ───────────────
        personality, ml_meta = self._get_personality(session_id, user_message)

        # ── Normal streaming conversation ────────────────────────────
        self._session.append_message(session_id, "user", user_message, user_id=user_id)
        history = self._session.get_history(session_id, user_id=user_id)

        # Pre-compute items for post-streaming
        retrieved_chunks = self._get_retrieved_chunks(user_message, session_id) if has_doc else []
        doc_filename = ""
        if has_doc and self._document_store:
            active_doc = self._document_store.get(session_id)
            if active_doc:
                doc_filename = active_doc.get("filename", "")

        smart_responder = self._smart_responder

        def generate():
            full_reply = []
            _s = get_settings()
            provider = _s.get("provider", "groq")

            # ── Mode dispatch ─────────────────────────────────────────
            # fast      → Groq only  (provider == 'groq', router may still pick nvidia for complex)
            # balanced  → Hybrid     (Groq-first, NVIDIA escalation when needed)
            # powerful  → NVIDIA     (provider == 'nvidia', forced)
            use_hybrid = (provider == "balanced")

            active_model = (
                _s["nvidia_model"] if provider == "nvidia"
                else _s["groq_model"]
            )

            if use_hybrid:
                stream_gen = self._ai.generate_stream_hybrid(
                    history, augmented_message, personality=personality
                )
            else:
                stream_gen = self._ai.generate_stream(
                    history, augmented_message, personality=personality
                )

            for chunk in stream_gen:
                yield chunk
                try:
                    if chunk.startswith("data: "):
                        payload = json.loads(chunk[6:].strip())
                        if payload.get("token"):
                            full_reply.append(payload["token"])
                        if payload.get("model"):
                            active_model = payload["model"]
                except (json.JSONDecodeError, KeyError):
                    pass

            complete_reply = "".join(full_reply).strip()

            # Phase 11: Append citations and suggestions via final SSE event
            if complete_reply:
                extra_meta = {}

                if smart_responder:
                    smart_result = smart_responder.format_response(
                        complete_reply, user_message, session_id,
                        retrieved_chunks=retrieved_chunks,
                        has_document=has_doc,
                        doc_filename=doc_filename,
                    )
                    # Send citation suffix as a final token if applicable
                    citations_text = smart_responder.format_citations(retrieved_chunks)
                    if citations_text:
                        yield f'data: {json.dumps({"token": citations_text})}\n\n'
                        complete_reply += citations_text

                    extra_meta["suggestions"] = smart_result["suggestions"]
                    extra_meta["exam_mode"] = smart_result["exam_mode"]
                    if smart_result["citations"]:
                        extra_meta["citations"] = smart_result["citations"]

                if has_doc and doc_filename:
                    extra_meta["active_document"] = doc_filename

                # Emit suggestions via a separate event
                if extra_meta:
                    yield f'data: {json.dumps({"type": "meta", **extra_meta})}\n\n'

                self._session.append_message(session_id, "nova", complete_reply, user_id=user_id)
                self._memory.record_turn(user_id=user_id)
                self._memory.extract_and_store(
                    user_message, complete_reply, active_model, build_provider_config(),
                    user_id=user_id,
                )

        return Response(
            stream_with_context(generate()),  # type: ignore[arg-type]
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Response-Time": str(int((time.time() - t0) * 1000)),
            },
        )

    def _handle_agent_stream(self, user_message: str, session_id: str,
                              user_id: str, t0: float,
                              with_document: bool = False) -> Response:
        """
        Stream agent reasoning steps via SSE (Phase 7).
        Phase 8: Passes document context, emits ANALYZE_DOC events.
        """
        doc_context = None
        if with_document and self._document_store:
            doc_context = self._document_store.get(session_id)

        mode_label = "document-agent" if doc_context else "agent"
        log.info("%s stream activated: user=%s task='%.60s'",
                 mode_label.title(), user_id, user_message)

        self._session.append_message(session_id, "user", user_message, user_id=user_id)
        history = self._session.get_history(session_id, user_id=user_id)

        def generate():
            final_answer = ""

            try:
                assert self._agent_runner is not None
                for event in self._agent_runner.run_stream(
                    task=user_message, history=history, user_id=user_id,
                    document_context=doc_context,
                ):
                    yield f'data: {json.dumps(event)}\n\n'

                    if event.get("type") == "done":
                        final_answer = event.get("final_answer", "")

            except Exception as exc:
                log.warning("Agent stream failed: %s", exc)
                yield f'data: {json.dumps({"type": "step", "phase": "ERROR", "content": f"Agent error: {exc}", "step": -1})}\n\n'

                augmented = self._inject_document_context(user_message, session_id)
                try:
                    if self._pipeline:
                        result = self._pipeline.execute(history, augmented)
                        final_answer = result["reply"]
                    else:
                        final_answer, _, _, _ = self._ai.generate(history, augmented)
                except Exception:
                    final_answer = "I encountered an error during analysis. Please try again."

                yield f'data: {json.dumps({"type": "done", "final_answer": final_answer, "steps_taken": 0, "tools_used": [], "mode": "fallback"})}\n\n'

            if final_answer:
                self._session.append_message(
                    session_id, "nova", final_answer, user_id=user_id
                )
                self._memory.record_turn(user_id=user_id)

        return Response(
            stream_with_context(generate()),  # type: ignore[arg-type]
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Response-Time": str(int((time.time() - t0) * 1000)),
            },
        )

    def handle_reset(self, session_id: str, generate_summary: bool = True) -> dict:
        """Reset a conversation session."""
        user_id = _get_user_id()

        if session_id:
            if generate_summary:
                history = self._session.get_history(session_id, user_id=user_id)
                if history:
                    settings = get_settings()
                    active_model = (
                        settings["nvidia_model"] if settings.get("provider") == "nvidia"
                        else settings["groq_model"]
                    )
                    self._memory.generate_daily_summary(
                        history, active_model, build_provider_config(),
                        user_id=user_id,
                    )
            self._session.clear_session(session_id, user_id=user_id)

            # Clean up retriever indexes for this session's documents
            if self._retriever and self._document_store:
                all_docs = self._document_store.get_all(session_id)
                for doc in all_docs:
                    doc_hash = doc.get("doc_hash")
                    if doc_hash:
                        self._retriever.remove(doc_hash)

            # Phase 8/11: Clear document context
            if self._document_store:
                self._document_store.clear(session_id)
            # Phase 9: Reset personality
            if self._personality_store:
                self._personality_store.clear(session_id)
            # Phase 11: Clear exam state
            if self._smart_responder:
                self._smart_responder.clear_exam_state(session_id)
        else:
            self._session.clear_all_sessions(user_id=user_id)
            if self._document_store:
                self._document_store.clear_all()
            if self._personality_store:
                self._personality_store.clear_all()
            if self._retriever:
                self._retriever.clear()
            if self._smart_responder:
                self._smart_responder.clear_all_exam_states()

        return {"status": "Conversation reset."}

    # ── Agent Detection ───────────────────────────────────────────────────

    @staticmethod
    def _should_use_agent(message: str) -> bool:
        """Detect if a message warrants autonomous agent processing."""
        from services.agent_runner import should_use_agent
        return should_use_agent(message)

    @staticmethod
    def _should_use_document_agent(message: str, has_document: bool) -> bool:
        """Phase 8: Detect if a message warrants document-aware agent processing."""
        from services.agent_runner import should_use_document_agent
        return should_use_document_agent(message, has_document)

    # ── Personality Helper ─────────────────────────────────────────────────

    def _get_personality(self, session_id: str, user_message: str = "") -> tuple[str, dict]:
        """
        Phase 9+10: Get personality with hybrid priority.
        Priority: User-selected > ML prediction > default
        """
        ml_meta = {}

        user_selected = "default"
        if self._personality_store:
            user_selected = self._personality_store.get(session_id)

        if user_selected != "default":
            return (user_selected, ml_meta)

        if (self._personality_model
                and self._personality_model.is_ready
                and user_message):
            try:
                from config import ENABLE_PERSONALITY_ML, PERSONALITY_ML_CONFIDENCE
                if ENABLE_PERSONALITY_ML:
                    predicted, confidence = self._personality_model.predict(user_message)
                    ml_meta = {
                        "ml_personality": predicted,
                        "ml_confidence": round(confidence, 3),
                    }
                    if predicted != "default" and confidence >= PERSONALITY_ML_CONFIDENCE:
                        log.info(
                            "ML personality: %s (%.2f) for session %s",
                            predicted, confidence, session_id,
                        )
                        return (predicted, ml_meta)
            except Exception as exc:
                log.warning("ML personality prediction failed: %s", exc)

        return ("default", ml_meta)
