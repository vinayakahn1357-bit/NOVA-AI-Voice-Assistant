"""
controllers/chat_controller.py — Chat Request Processing for NOVA
Orchestrates: validate → pipeline (analyze → agent → route → generate → format → sanitize) → save → return.
Uses ResponsePipeline for clean, modular response flow.
"""

import json
import time

from flask import Response, stream_with_context, g

from config import get_settings, build_provider_config
from utils.logger import get_logger
from utils.validators import validate_chat_input
from utils.errors import NovaValidationError

log = get_logger("chat")


class ChatController:
    """Handles chat request processing and orchestration."""

    def __init__(self, ai_service, session_service, memory_service, command_service,
                 agent_engine=None, response_pipeline=None):
        self._ai = ai_service
        self._session = session_service
        self._memory = memory_service
        self._commands = command_service
        self._agent = agent_engine
        self._pipeline = response_pipeline

    def handle_chat(self, data: dict, session_id: str) -> dict:
        """
        Handle a non-streaming chat request via the response pipeline.
        Returns consistent response format with meta block.
        """
        t0 = time.time()
        user_message = validate_chat_input(data.get("message", ""))

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

        # ── Execute response pipeline ──────────────────────────────────
        self._session.append_message(session_id, "user", user_message)
        history = self._session.get_history(session_id)

        if self._pipeline:
            # Use the modular pipeline
            pipeline_result = self._pipeline.execute(history, user_message)

            ai_response = pipeline_result["reply"]
            active_model = pipeline_result["model"]
            provider = pipeline_result["provider"]
            meta = pipeline_result["meta"]
        else:
            # Fallback: direct AI call (no pipeline)
            ai_response, active_model, provider, ai_meta = self._ai.generate(
                history, user_message
            )
            meta = {
                "model": active_model,
                "provider": provider,
                "mode": "normal",
                "latency_ms": int((time.time() - t0) * 1000),
            }

        # Persist Nova's reply
        self._session.append_message(session_id, "nova", ai_response)

        # Background learning (non-blocking)
        self._memory.record_turn()
        self._memory.extract_and_store(
            user_message, ai_response, active_model, build_provider_config()
        )

        elapsed_ms = int((time.time() - t0) * 1000)
        meta["latency_ms"] = elapsed_ms
        meta["model"] = active_model
        meta["provider"] = provider

        log.info("Chat complete: %dms provider=%s model=%s", elapsed_ms, provider, active_model)

        # Set model on g for response headers
        try:
            g.nova_model = active_model
        except RuntimeError:
            pass

        return {
            "reply": ai_response,
            "session_id": session_id,
            "meta": meta,
        }

    def handle_chat_stream(self, data: dict, session_id: str) -> Response:
        """
        Handle a streaming (SSE) chat request.
        Returns a Flask Response with text/event-stream.
        """
        t0 = time.time()
        user_message = validate_chat_input(data.get("message", ""))

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

        # ── Normal streaming conversation ──────────────────────────────
        self._session.append_message(session_id, "user", user_message)
        history = self._session.get_history(session_id)

        def generate():
            full_reply = []
            active_model = get_settings()["model"]

            for chunk in self._ai.generate_stream(history, user_message):
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

            # Save complete reply and trigger learning
            complete_reply = "".join(full_reply).strip()
            if complete_reply:
                self._session.append_message(session_id, "nova", complete_reply)
                self._memory.record_turn()
                self._memory.extract_and_store(
                    user_message, complete_reply, active_model, build_provider_config()
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

    def handle_reset(self, session_id: str, generate_summary: bool = True) -> dict:
        """Reset a conversation session, optionally generating a daily summary first."""
        if session_id:
            if generate_summary:
                history = self._session.get_history(session_id)
                if history:
                    settings = get_settings()
                    self._memory.generate_daily_summary(
                        history, settings["model"], build_provider_config()
                    )
            self._session.clear_session(session_id)
        else:
            self._session.clear_all_sessions()

        return {"status": "Conversation reset."}
