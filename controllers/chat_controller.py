"""
controllers/chat_controller.py — Chat Request Processing for NOVA (Phase 8)
Orchestrates: validate → pipeline/agent → save → return.
Phase 6: user_id multi-user isolation.
Phase 7: detects complex tasks → routes to AgentRunner with step streaming.
Phase 8: persistent document context injection for multi-turn PDF Q&A.
"""

import json
import re
import time

from flask import Response, stream_with_context, g, session

from config import get_settings, build_provider_config
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
    Phase 7: accepts optional agent_runner for autonomous task execution.
    Phase 8: injects persistent document context for multi-turn PDF Q&A.
    """

    def __init__(self, ai_service, session_service, memory_service, command_service,
                 agent_engine=None, response_pipeline=None, agent_runner=None,
                 document_store=None):
        self._ai = ai_service
        self._session = session_service
        self._memory = memory_service
        self._commands = command_service
        self._agent = agent_engine
        self._pipeline = response_pipeline
        self._agent_runner = agent_runner      # Phase 7: autonomous agent
        self._document_store = document_store  # Phase 8: persistent PDF context

    # ── Document Context Helpers ──────────────────────────────────────────

    def _is_doc_clear_request(self, message: str) -> bool:
        """Check if user wants to clear the active document context."""
        return bool(_DOC_CLEAR_RE.search(message))

    def _handle_doc_clear(self, session_id: str) -> dict:
        """Clear document context and return a confirmation response."""
        if self._document_store and self._document_store.has_document(session_id):
            doc = self._document_store.get(session_id)
            filename = doc["filename"] if doc else "document"
            self._document_store.clear(session_id)
            return {
                "reply": f"📄 Document context cleared. I've removed **{filename}** from this session. "
                         f"You can upload a new PDF or continue chatting normally.",
                "session_id": session_id,
                "meta": {
                    "model": "nova-commands",
                    "provider": "internal",
                    "mode": "document_clear",
                },
            }
        return {
            "reply": "No active document to clear. You can upload a PDF anytime using the attach button.",
            "session_id": session_id,
            "meta": {
                "model": "nova-commands",
                "provider": "internal",
                "mode": "document_clear",
            },
        }

    def _inject_document_context(self, message: str, session_id: str) -> str:
        """
        If the session has an active document and the message doesn't already
        contain document context, prepend the document summary.
        """
        if not self._document_store:
            return message

        doc = self._document_store.get(session_id)
        if not doc:
            return message

        # Skip injection if message already contains document context
        # (e.g., from a fresh PDF upload in this same request)
        if "The user uploaded a PDF document" in message:
            return message

        context = (
            f"You are answering based on a previously uploaded document: "
            f"{doc['filename']}.\n\n"
            f"Document context:\n{doc['summary']}\n\n"
            f"User's question: {message}"
        )
        log.info("Document context injected from store for session %s ('%s')",
                 session_id, doc["filename"])
        return context

    # ── Chat Handlers ─────────────────────────────────────────────────────

    def handle_chat(self, data: dict, session_id: str) -> dict:
        """
        Handle a non-streaming chat request via the response pipeline.
        Phase 7: routes complex tasks to AgentRunner when available.
        Phase 8: injects persistent document context automatically.
        """
        t0 = time.time()
        user_message = validate_chat_input(data.get("message", ""))
        user_id = _get_user_id()

        # ── Phase 8: Document clear detection ─────────────────────────────
        if self._is_doc_clear_request(user_message):
            return self._handle_doc_clear(session_id)

        # ── Command detection ──────────────────────────────────────────
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

        # ── Phase 7: Agent detection ───────────────────────────────────
        if self._agent_runner and self._should_use_agent(user_message):
            return self._handle_agent_chat(user_message, session_id, user_id, t0)

        # ── Phase 8: Inject document context ──────────────────────────────
        augmented_message = self._inject_document_context(user_message, session_id)

        # ── Standard pipeline ──────────────────────────────────────────
        self._session.append_message(session_id, "user", user_message, user_id=user_id)
        history = self._session.get_history(session_id, user_id=user_id)

        if self._pipeline:
            pipeline_result = self._pipeline.execute(history, augmented_message)
            ai_response = pipeline_result["reply"]
            active_model = pipeline_result["model"]
            provider = pipeline_result["provider"]
            meta = pipeline_result["meta"]
        else:
            ai_response, active_model, provider, ai_meta = self._ai.generate(
                history, augmented_message
            )
            meta = {
                "model": active_model,
                "provider": provider,
                "mode": "normal",
                "latency_ms": int((time.time() - t0) * 1000),
            }

        # Add document indicator to meta if active
        if self._document_store and self._document_store.has_document(session_id):
            doc = self._document_store.get(session_id)
            meta["active_document"] = doc["filename"] if doc else None

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
                            user_id: str, t0: float) -> dict:
        """Handle a chat request via AgentRunner (Phase 7)."""
        log.info("Agent mode activated: user=%s task='%.60s'", user_id, user_message)

        self._session.append_message(session_id, "user", user_message, user_id=user_id)
        history = self._session.get_history(session_id, user_id=user_id)

        # Run the autonomous agent
        result = self._agent_runner.run(
            task=user_message,
            history=history,
            user_id=user_id,
        )

        ai_response = result.final_answer
        self._session.append_message(session_id, "nova", ai_response, user_id=user_id)

        # Store agent run in memory
        self._memory.record_turn(user_id=user_id)
        if hasattr(self._memory, "store_agent_run"):
            self._memory.store_agent_run(user_id, user_message, result)

        elapsed_ms = int((time.time() - t0) * 1000)

        return {
            "reply": ai_response,
            "session_id": session_id,
            "meta": {
                "mode": "agent",
                "model": "nova-agent",
                "provider": "internal",
                "latency_ms": elapsed_ms,
                "steps_taken": result.steps_taken,
                "tools_used": result.tools_used,
            },
        }

    def handle_chat_stream(self, data: dict, session_id: str) -> Response:
        """
        Handle a streaming (SSE) chat request.
        Phase 7: streams agent steps when agent mode detected.
        Phase 8: injects persistent document context automatically.
        """
        t0 = time.time()
        user_message = validate_chat_input(data.get("message", ""))
        user_id = _get_user_id()

        # ── Phase 8: Document clear detection ─────────────────────────────
        if self._is_doc_clear_request(user_message):
            result = self._handle_doc_clear(session_id)

            def _doc_clear_gen():
                yield f'data: {json.dumps({"token": result["reply"]})}\n\n'
                yield f'data: {json.dumps({"done": True, "session_id": session_id, "model": "nova-commands"})}\n\n'

            return Response(
                stream_with_context(_doc_clear_gen()),
                mimetype="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "X-Response-Time": str(int((time.time() - t0) * 1000)),
                },
            )

        # ── Command detection ──────────────────────────────────────────
        intent = self._commands.detect_intent(user_message)
        if intent["type"] == "command":
            result = self._commands.execute(
                intent["command"], intent["args"], session_id
            )

            def _cmd_gen():
                yield f'data: {json.dumps({"token": result["response"]})}\n\n'
                yield f'data: {json.dumps({"done": True, "session_id": session_id, "model": "nova-commands"})}\n\n'

            return Response(
                stream_with_context(_cmd_gen()),
                mimetype="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "X-Response-Time": str(int((time.time() - t0) * 1000)),
                },
            )

        # ── Phase 7: Agent streaming ───────────────────────────────────
        if self._agent_runner and self._should_use_agent(user_message):
            return self._handle_agent_stream(user_message, session_id, user_id, t0)

        # ── Phase 8: Inject document context ──────────────────────────────
        augmented_message = self._inject_document_context(user_message, session_id)

        # ── Normal streaming conversation ──────────────────────────────
        self._session.append_message(session_id, "user", user_message, user_id=user_id)
        history = self._session.get_history(session_id, user_id=user_id)

        def generate():
            full_reply = []
            active_model = get_settings()["model"]

            for chunk in self._ai.generate_stream(history, augmented_message):
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
            if complete_reply:
                self._session.append_message(session_id, "nova", complete_reply, user_id=user_id)
                self._memory.record_turn(user_id=user_id)
                self._memory.extract_and_store(
                    user_message, complete_reply, active_model, build_provider_config(),
                    user_id=user_id,
                )

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Response-Time": str(int((time.time() - t0) * 1000)),
            },
        )

    def _handle_agent_stream(self, user_message: str, session_id: str,
                              user_id: str, t0: float) -> Response:
        """
        Stream agent reasoning steps via SSE (Phase 7).

        Events:
            data: {"type": "step", "phase": "THINK", "content": "...", "step": 1}
            data: {"type": "step", "phase": "ACT", "content": "...", "step": 1}
            data: {"type": "step", "phase": "OBSERVE", "content": "...", "step": 1}
            data: {"type": "done", "final_answer": "...", "steps_taken": 3}
        """
        log.info("Agent stream activated: user=%s task='%.60s'", user_id, user_message)

        self._session.append_message(session_id, "user", user_message, user_id=user_id)
        history = self._session.get_history(session_id, user_id=user_id)

        def generate():
            final_answer = ""

            for event in self._agent_runner.run_stream(
                task=user_message, history=history, user_id=user_id
            ):
                yield f'data: {json.dumps(event)}\n\n'

                if event.get("type") == "done":
                    final_answer = event.get("final_answer", "")

            # Save the final answer
            if final_answer:
                self._session.append_message(
                    session_id, "nova", final_answer, user_id=user_id
                )
                self._memory.record_turn(user_id=user_id)

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Response-Time": str(int((time.time() - t0) * 1000)),
            },
        )

    def handle_reset(self, session_id: str, generate_summary: bool = True) -> dict:
        """Reset a conversation session, optionally generating a daily summary first."""
        user_id = _get_user_id()

        if session_id:
            if generate_summary:
                history = self._session.get_history(session_id, user_id=user_id)
                if history:
                    settings = get_settings()
                    self._memory.generate_daily_summary(
                        history, settings["model"], build_provider_config(),
                        user_id=user_id,
                    )
            self._session.clear_session(session_id, user_id=user_id)

            # Phase 8: Also clear document context on reset
            if self._document_store:
                self._document_store.clear(session_id)
        else:
            self._session.clear_all_sessions(user_id=user_id)
            # Phase 8: Clear all document contexts
            if self._document_store:
                self._document_store.clear_all()

        return {"status": "Conversation reset."}

    # ── Agent Detection ───────────────────────────────────────────────────

    @staticmethod
    def _should_use_agent(message: str) -> bool:
        """Detect if a message warrants autonomous agent processing."""
        from services.agent_runner import should_use_agent
        return should_use_agent(message)
