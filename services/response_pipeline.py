"""
services/response_pipeline.py — Modular Response Pipeline for NOVA
Orchestrates the entire response flow as a clean, sequential pipeline:

    Analyze → Agent → Route → Generate → Format → Sanitize → Package

Each stage is pluggable and independently testable.
"""

import time
from flask import g
from utils.logger import get_logger

log = get_logger("pipeline")


class ResponsePipeline:
    """
    Central orchestrator for NOVA's response flow.
    Replaces the ad-hoc logic scattered across ChatController and AIService.

    All stages are optional — the pipeline gracefully skips missing components.
    """

    def __init__(self, ai_service, query_analyzer=None, agent_engine=None,
                 response_formatter=None, response_sanitizer=None,
                 cache_service=None):
        self._ai = ai_service
        self._analyzer = query_analyzer
        self._agent = agent_engine
        self._formatter = response_formatter
        self._sanitizer = response_sanitizer
        self._cache = cache_service

    def execute(self, history: list, user_message: str) -> dict:
        """
        Execute the full response pipeline.

        Args:
            history: conversation history (list of {role, content} dicts)
            user_message: the current user message (already validated)

        Returns: {
            "reply": str,
            "model": str,
            "provider": str,
            "meta": {
                "mode": str,
                "query_type": str,
                "complexity": int,
                "latency_ms": int,
                "cached": bool,
                "agent_action": str,
                "agent_confidence": float,
            }
        }
        """
        t0 = time.time()

        # ── Stage 1: Analyze ──────────────────────────────────────────────
        qa = {}
        if self._analyzer:
            qa = self._analyzer.analyze(user_message)

        query_type = qa.get("query_type", "conversation")
        complexity = qa.get("complexity", 5)

        # ── Stage 2: Agent Intelligence ───────────────────────────────────
        agent_result = None
        agent_mode = "normal"
        prompt_augment = ""

        if self._agent:
            agent_result = self._agent.process(user_message, qa)
            agent_mode = agent_result.get("agent_mode", "normal")

            # Tool mode: return immediately without LLM
            if agent_result.get("skip_llm") and agent_result.get("tool_result"):
                tool_response = self._agent.format_tool_response(
                    agent_result["tool_result"]
                )
                elapsed_ms = int((time.time() - t0) * 1000)

                return {
                    "reply": tool_response,
                    "model": "nova-agent",
                    "provider": "internal",
                    "meta": {
                        "mode": "tool",
                        "query_type": query_type,
                        "complexity": 1,
                        "latency_ms": elapsed_ms,
                        "cached": False,
                        "agent_action": agent_result.get("action", ""),
                        "agent_confidence": agent_result.get("confidence", 1.0),
                        "tool": agent_result["tool_result"].get("tool", ""),
                    },
                }

            prompt_augment = agent_result.get("prompt_augment", "")

        # ── Stage 3: Generate (with prompt augmentation) ──────────────────
        ai_response, active_model, provider, ai_meta = self._ai.generate(
            history, user_message, prompt_augment=prompt_augment
        )

        # Set model on Flask g for middleware headers
        try:
            g.nova_model = active_model
        except RuntimeError:
            pass  # Outside request context (e.g., tests)

        # ── Stage 4: Format (query-type-aware) ────────────────────────────
        if self._formatter:
            ai_response = self._formatter.format(ai_response, query_type)

        # ── Stage 5: Sanitize (security) ──────────────────────────────────
        was_injection = False
        if self._sanitizer:
            from utils.validators import check_prompt_injection
            was_injection = check_prompt_injection(user_message)
            ai_response = self._sanitizer.sanitize(ai_response, was_injection)

        # ── Stage 6: Package ──────────────────────────────────────────────
        elapsed_ms = int((time.time() - t0) * 1000)

        meta = {
            "mode": agent_mode,
            "query_type": query_type,
            "complexity": complexity,
            "latency_ms": elapsed_ms,
            "cached": ai_meta.get("cached", False),
            "provider": provider,
            "fallback_used": ai_meta.get("fallback_used", False),
        }

        # Add agent metadata if non-normal mode
        if agent_result and agent_mode != "normal":
            meta["agent_action"] = agent_result.get("action", "")
            meta["agent_confidence"] = agent_result.get("confidence", 0)

        # Add routing metadata
        if ai_meta.get("route_reason"):
            meta["route_reason"] = ai_meta["route_reason"]

        log.info(
            "Pipeline complete: %dms mode=%s type=%s provider=%s model=%s",
            elapsed_ms, agent_mode, query_type, provider, active_model,
        )

        return {
            "reply": ai_response,
            "model": active_model,
            "provider": provider,
            "meta": meta,
        }
