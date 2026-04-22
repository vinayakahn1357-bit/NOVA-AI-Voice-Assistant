"""
services/response_pipeline.py — Modular Response Pipeline for NOVA (Phase 12)
Orchestrates the entire response flow as a clean, sequential pipeline:

    Analyze → Agent → Route → Generate → Format → Enforce → Sanitize → Package

Phase 12 additions:
  - Stage 4.5: Personality enforcement (forbidden phrases, structure patching)
  - Stage 4.6: Personality strength scoring + conditional regeneration
  - Regenerates ONCE if score < threshold, then falls back to default personality
  - All enforcement actions are logged for debugging

Each stage is pluggable and independently testable.
"""

import time
from flask import g
from utils.logger import get_logger
from services.personality_enforcer import (
    PersonalityEnforcer,
    SCORE_PASS_THRESHOLD,
    SCORE_REGEN_THRESHOLD,
)

log = get_logger("pipeline")

# Singleton enforcer (stateless, safe to share)
_enforcer = PersonalityEnforcer()


class ResponsePipeline:
    """
    Central orchestrator for NOVA's response flow.
    All stages are optional — the pipeline gracefully skips missing components.

    Phase 12: Personality enforcement and strength-scored regeneration.
    """

    def __init__(self, ai_service, query_analyzer=None, agent_engine=None,
                 response_formatter=None, response_sanitizer=None,
                 cache_service=None, personality_enforcer=None):
        self._ai = ai_service
        self._analyzer = query_analyzer
        self._agent = agent_engine
        self._formatter = response_formatter
        self._sanitizer = response_sanitizer
        self._cache = cache_service
        # Phase 12: use injected enforcer or module-level singleton
        self._enforcer = personality_enforcer or _enforcer

    def execute(self, history: list, user_message: str,
                personality: str = "default") -> dict:
        """
        Execute the full response pipeline.

        Phase 12 flow:
            Analyze → Agent → Generate → Format → Enforce → Score → [Regen] → Sanitize → Package

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
                "personality": str,
                "personality_score": float,
                "personality_regenerated": bool,
                "personality_fallback": bool,
                ...
            }
        }
        """
        t0 = time.time()

        # Validate personality key — fallback to default safely
        try:
            from services.personality_service import VALID_PERSONALITIES
            if personality not in VALID_PERSONALITIES:
                log.warning(
                    "Pipeline received unknown personality '%s' — falling back to 'default'",
                    personality,
                )
                personality = "default"
        except ImportError:
            personality = "default"

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
                        "personality": personality,
                    },
                }

            prompt_augment = agent_result.get("prompt_augment", "")

        # ── Stage 3: Generate ─────────────────────────────────────────────
        ai_response, active_model, provider, ai_meta = self._ai.generate(
            history, user_message, prompt_augment=prompt_augment,
            personality=personality,
        )

        # Set model on Flask g for middleware headers
        try:
            g.nova_model = active_model
        except RuntimeError:
            pass

        # ── Stage 4: Format (query-type-aware) ────────────────────────────
        if self._formatter:
            ai_response = self._formatter.format(ai_response, query_type)

        # ── Stage 4.5: Personality Enforcement ────────────────────────────
        personality_score = 1.0
        personality_regenerated = False
        personality_fallback = False

        try:
            # Apply enforcement (forbidden phrases, structure patching, safety)
            ai_response = self._enforcer.enforce(ai_response, personality, query_type)

            # Score the enforced response
            personality_score, score_breakdown = self._enforcer.score(ai_response, personality)

            log.info(
                "Personality score: personality=%s score=%.3f breakdown=%s",
                personality, personality_score, score_breakdown,
            )

            # ── Stage 4.6: Conditional Regeneration ───────────────────────
            if personality_score < SCORE_PASS_THRESHOLD and personality != "default":
                log.info(
                    "Score %.3f < threshold %.3f for '%s' — regenerating with strict prompt",
                    personality_score, SCORE_PASS_THRESHOLD, personality,
                )
                ai_response, active_model, provider, ai_meta = self._regenerate_with_strict_prompt(
                    history, user_message, personality, ai_response, prompt_augment
                )
                personality_regenerated = True

                # Re-enforce and re-score the regenerated response
                ai_response = self._enforcer.enforce(ai_response, personality, query_type)
                personality_score, _ = self._enforcer.score(ai_response, personality)

                # If still below hard floor → fall back to default personality
                if personality_score < SCORE_REGEN_THRESHOLD:
                    log.warning(
                        "Regen score %.3f still below floor %.3f — falling back to 'default'",
                        personality_score, SCORE_REGEN_THRESHOLD,
                    )
                    # Use the already-generated response but enforce with default rules
                    ai_response = self._enforcer.enforce(ai_response, "default", query_type)
                    personality_fallback = True

        except Exception as exc:
            # Enforcement must NEVER crash the pipeline
            log.warning("Personality enforcement failed (non-fatal): %s", exc)

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
            # Phase 12: Personality metadata
            "personality": personality,
            "personality_score": round(personality_score, 3),
            "personality_regenerated": personality_regenerated,
            "personality_fallback": personality_fallback,
        }

        # Add agent metadata if non-normal mode
        if agent_result and agent_mode != "normal":
            meta["agent_action"] = agent_result.get("action", "")
            meta["agent_confidence"] = agent_result.get("confidence", 0)

        # Add routing metadata
        if ai_meta.get("route_reason"):
            meta["route_reason"] = ai_meta["route_reason"]

        log.info(
            "Pipeline complete: %dms mode=%s type=%s provider=%s model=%s "
            "personality=%s score=%.3f regen=%s fallback=%s",
            elapsed_ms, agent_mode, query_type, provider, active_model,
            personality, personality_score, personality_regenerated, personality_fallback,
        )

        return {
            "reply": ai_response,
            "model": active_model,
            "provider": provider,
            "meta": meta,
        }

    def _regenerate_with_strict_prompt(
        self, history: list, user_message: str, personality: str,
        failed_response: str, prompt_augment: str = ""
    ) -> tuple:
        """
        Regenerate using a reinforced prompt when the first response
        fails personality scoring. Called at most ONCE per request.

        Returns: (ai_response, active_model, provider, ai_meta)
        """
        try:
            # Build a stricter prompt with failure context
            strict_messages = self._ai._prompt.build_strict_regen_messages(
                history, personality, failed_response
            )

            # Inject agent prompt augmentation if any
            if prompt_augment:
                strict_messages.insert(-1, {
                    "role": "system",
                    "content": prompt_augment,
                })

            # Resolve provider for the regen call
            from config import get_settings
            settings = get_settings()
            provider = self._ai._resolve_provider(settings["provider"])

            # Get personality temperature
            personality_temp = self._ai._resolve_temperature(personality)

            # Direct generation call (bypass routing for determinism)
            import time as _time
            t_regen = _time.time()

            if provider == "nvidia" and self._ai._nvidia_configured():
                regen_text, active_model = self._ai._generate_nvidia(
                    strict_messages, temperature=personality_temp
                )
            elif self._ai._groq_configured():
                regen_text, active_model = self._ai._generate_groq(
                    strict_messages, temperature=personality_temp
                )
            else:
                log.warning("No provider available for regeneration — using original")
                return failed_response, "unknown", "internal", {}

            regen_latency = round(_time.time() - t_regen, 2)
            log.info(
                "Regeneration complete: personality=%s latency=%.2fs len=%d",
                personality, regen_latency, len(regen_text),
            )

            return regen_text, active_model, provider, {
                "cached": False,
                "fallback_used": False,
                "regen_latency": regen_latency,
            }

        except Exception as exc:
            log.warning("Regeneration failed (%s) — using original response", exc)
            return failed_response, "unknown", "internal", {}
