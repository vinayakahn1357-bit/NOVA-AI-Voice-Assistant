"""
controllers/chat_controller.py — Chat Request Processing for NOVA (Phase 7)
Orchestrates: validate → pipeline/agent → save → return.
Phase 6: user_id multi-user isolation.
Phase 7: detects complex tasks → routes to AgentRunner with step streaming.
"""

import json
import time

from flask import Response, stream_with_context, g, session

from config import get_settings, build_provider_config
from utils.logger import get_logger
from utils.validators import validate_chat_input
from utils.errors import NovaValidationError

log = get_logger("chat")


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
    """

    def __init__(self, ai_service, session_service, memory_service, command_service,
                 agent_engine=None, response_pipeline=None, agent_runner=None):
        self._ai = ai_service
        self._session = session_service
        self._memory = memory_service
        self._commands = command_service
        self._agent = agent_engine
        self._pipeline = response_pipeline
        self._agent_runner = agent_runner      # Phase 7: autonomous agent

    def handle_chat(self, data: dict, session_id: str) -> dict:
        """
        Handle a non-streaming chat request via the response pipeline.
        Phase 7: routes complex tasks to AgentRunner when available.
        """
        t0 = time.time()
        user_message = validate_chat_input(data.get("message", ""))
        user_id = _get_user_id()

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

        # ── Standard pipeline ──────────────────────────────────────────
        self._session.append_message(session_id, "user", user_message, user_id=user_id)
        history = self._session.get_history(session_id, user_id=user_id)

        if self._pipeline:
            pipeline_result = self._pipeline.execute(history, user_message)
            ai_response = pipeline_result["reply"]
            active_model = pipeline_result["model"]
            provider = pipeline_result["provider"]
            meta = pipeline_result["meta"]
        else:
            ai_response, active_model, provider, ai_meta = self._ai.generate(
                history, user_message
            )
            meta = {
                "model": active_model,
                "provider": provider,
                "mode": "normal",
                "latency_ms": int((time.time() - t0) * 1000),
            }

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
        """
        t0 = time.time()
        user_message = validate_chat_input(data.get("message", ""))
        user_id = _get_user_id()

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

        # ── Normal streaming conversation ──────────────────────────────
        self._session.append_message(session_id, "user", user_message, user_id=user_id)
        history = self._session.get_history(session_id, user_id=user_id)

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
        else:
            self._session.clear_all_sessions(user_id=user_id)

        return {"status": "Conversation reset."}

    # ── Agent Detection ───────────────────────────────────────────────────

    @staticmethod
    def _should_use_agent(message: str) -> bool:
        """Detect if a message warrants autonomous agent processing."""
        from services.agent_runner import should_use_agent
        return should_use_agent(message)
