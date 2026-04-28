"""
services/response_pipeline.py — Modular Response Pipeline for NOVA (Phase 13)
Orchestrates the entire response flow as a clean, sequential pipeline:

    Analyze → Agent → Route → Generate → Format → Enforce → Quality → Sanitize → Package

Phase 12 additions:
  - Stage 4.5: Personality enforcement (forbidden phrases, structure patching)
  - Stage 4.6: Personality strength scoring + conditional regeneration
  - Regenerates ONCE if score < threshold, then falls back to default personality

Phase 13 additions:
  - Stage 4.7: Response quality enforcement (forbidden phrase cleanup, substance
    validation, filler detection, quality scoring)
  - Quality-based regeneration: max 1 retry with stricter prompt if quality < 0.4
  - Safe fallback: if regen also fails, delivers best-effort response

Each stage is pluggable and independently testable.
"""

import re
import time
from flask import g
from utils.logger import get_logger
from services.personality_enforcer import (
    PersonalityEnforcer,
    SCORE_PASS_THRESHOLD,
    SCORE_REGEN_THRESHOLD,
)
from services.response_quality import ResponseQualityEnforcer
from config import QUALITY_SCORE_THRESHOLD, QUALITY_REGEN_MAX

log = get_logger("pipeline")

# Singleton enforcers (stateless, safe to share)
_enforcer = PersonalityEnforcer()
_quality_enforcer = ResponseQualityEnforcer()

# ── Session Context Store ────────────────────────────────────────────────────
# Stores the last realtime search context per user/session so follow-up
# queries like "which team?" can reference the previous search results.
# Key: user_id or session_id string. Value: {context, query, timestamp}.
_session_context: dict = {}
_SESSION_CTX_TTL = 600  # 10 minutes

# ── Placeholder Response Detector ───────────────────────────────────────────
# Catches LLM responses that contain unfilled template placeholders.
# These must NEVER reach the user.
_PLACEHOLDER_RE = re.compile(
    r"\[(?:Team \d|Score|Venue|Winner|Match|Player|Date|Result|TBD|N/A|Unknown)\]",
    re.IGNORECASE,
)


class ResponsePipeline:
    """
    Central orchestrator for NOVA's response flow.
    All stages are optional — the pipeline gracefully skips missing components.

    Phase 12: Personality enforcement and strength-scored regeneration.
    """

    def __init__(self, ai_service, query_analyzer=None, agent_engine=None,
                 response_formatter=None, response_sanitizer=None,
                 cache_service=None, personality_enforcer=None,
                 quality_enforcer=None, realtime_service=None):
        self._ai = ai_service
        self._analyzer = query_analyzer
        self._agent = agent_engine
        self._formatter = response_formatter
        self._sanitizer = response_sanitizer
        self._cache = cache_service
        # Phase 12: use injected enforcer or module-level singleton
        self._enforcer = personality_enforcer or _enforcer
        # Phase 13: response quality enforcement
        self._quality = quality_enforcer or _quality_enforcer
        # Phase 13: real-time search
        self._realtime = realtime_service

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

        # ── Stage 1.5: Real-Time Search ─────────────────────────────────
        realtime_used = False
        search_results = []
        realtime_detected = False
        # CRITICAL: Save original message BEFORE search injection
        # The agent engine must receive the ORIGINAL query, not the augmented one,
        # otherwise calculator/tool detection will match numeric scores (e.g. 3/41).
        original_message = user_message

        # Derive a session key for context persistence
        _user_id = getattr(g, "user_id", None) or "anon"
        _session_key = str(_user_id)
        _now = time.time()

        # ── Stage 1.4: Follow-Up Context Injection ───────────────────────
        # If this is a SHORT follow-up query (< 6 words) and we have a recent
        # session context, inject it so the LLM can answer "which team?" etc.
        _stored = _session_context.get(_session_key, {})
        _is_followup = (
            len(user_message.split()) < 6
            and _stored
            and (_now - _stored.get("timestamp", 0)) < _SESSION_CTX_TTL
            and not any(w in user_message.lower()
                        for w in ["hello", "hi", "thanks", "okay", "ok", "bye"])
        )

        if _is_followup and not self._realtime.detect_realtime_intent(user_message) if self._realtime else False:
            _ctx = _stored["context"]
            _prev_q = _stored["query"]
            log.info("Follow-up detected: injecting session context from '%s'", _prev_q[:60])
            followup_directive = (
                f"[FOLLOW-UP CONTEXT: The user's previous question was: '{_prev_q}'. "
                f"The following search results were retrieved for that question. "
                f"Use them to answer this follow-up question.\n\n{_ctx}]"
            )
            user_message = f"{user_message}\n\n{followup_directive}"
            query_type = "realtime"
            realtime_used = True
            log.info("Session context injected for follow-up (%d chars)", len(_ctx))

        if self._realtime:
            realtime_detected = self._realtime.detect_realtime_intent(user_message)
            log.info("Realtime intent check: detected=%s for: '%s'",
                     realtime_detected, user_message[:80])

        if realtime_detected and self._realtime:
            log.info("Executing Tavily search for: '%s'", user_message[:80])
            search_results = self._realtime.search(user_message)
            log.info("Tavily returned %d results", len(search_results))

            if search_results:
                search_context = self._realtime.build_search_context(
                    search_results, user_message
                )
                log.info("Search context built (%d chars), injecting into prompt",
                         len(search_context))

                # === CRITICAL: Override directive ===
                realtime_override = (
                    "\n\n[REALTIME OVERRIDE: You HAVE access to real-time information "
                    "through web search. The search results below are FRESH and CURRENT. "
                    "Use them to answer the user's question accurately. "
                    "Do NOT say 'I don't have access to real-time information' or "
                    "'my knowledge cutoff' or 'I cannot browse the internet'. "
                    "You MUST use the ACTUAL data from the search results. "
                    "Do NOT use placeholder values like [Team 1], [Score], [Venue]. "
                    "If a specific value is not in the results, say 'not found in search' — "
                    "NEVER invent placeholder brackets. "
                    "Any numbers in the results are factual data (scores, statistics, prices), "
                    "NOT math expressions. Do NOT calculate them.]\n\n"
                )
                user_message = f"{user_message}\n\n{realtime_override}{search_context}"
                realtime_used = True
                query_type = "realtime"
                log.info("Real-time search: %d results injected with override directive",
                         len(search_results))

                # Store search context in session for follow-up queries
                _session_context[_session_key] = {
                    "context": search_context,
                    "query": original_message,
                    "timestamp": _now,
                }
                # Prune expired entries
                expired_keys = [k for k, v in _session_context.items()
                                if _now - v.get("timestamp", 0) > _SESSION_CTX_TTL]
                for k in expired_keys:
                    del _session_context[k]

                log.debug("Final augmented prompt (first 600 chars): %s", user_message[:600])
            else:
                log.warning("Tavily search returned 0 results for: '%s'", user_message[:80])
                fallback_msg = (
                    "\n\n[REALTIME SEARCH NOTE: A web search was performed but returned "
                    "no results at this time. Please provide your best answer based on "
                    "general knowledge. Do NOT use placeholder values like [Team 1] or "
                    "[Score]. If you cannot provide the specific result, clearly say: "
                    "'I could not fetch the latest match result from search at this time.']"
                )
                user_message = f"{user_message}{fallback_msg}"
                query_type = "realtime"

        # ── Stage 2: Agent Intelligence ───────────────────────────────────
        agent_result = None
        agent_mode = "normal"
        prompt_augment = ""

        if self._agent:
            # CRITICAL: Pass ORIGINAL message to agent, NOT the search-augmented one.
            # This prevents the calculator tool from matching cricket scores (3/41, 183-176).
            agent_result = self._agent.process(original_message, qa)
            agent_mode = agent_result.get("agent_mode", "normal")

            # Tool mode: return immediately without LLM
            # BUT: If realtime search is active, NEVER skip LLM — the search results
            # must be processed by the LLM, not returned as a tool result.
            if (agent_result.get("skip_llm")
                    and agent_result.get("tool_result")
                    and not realtime_detected):
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

        # ── Stage 3.5: Placeholder Guard ─────────────────────────────────
        # If the LLM generated template placeholders like [Team 1] or [Score],
        # the response is unusable. Replace it with a clear honest message.
        if _PLACEHOLDER_RE.search(ai_response):
            log.warning("LLM generated placeholder values — intercepting response")
            log.warning("Placeholder response (first 200 chars): %s", ai_response[:200])
            if realtime_used:
                ai_response = (
                    "I could not fetch the specific match details from search results at this time. "
                    "Please try asking again in a moment, or check a sports site like "
                    "ESPNcricinfo or the official IPL website for the latest results."
                )
            else:
                ai_response = (
                    "I wasn't able to retrieve the specific details for that. "
                    "Could you provide more context or try rephrasing?"
                )

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

        # ── Stage 4.7: Response Quality Enforcement ─────────────────────
        quality_score = 1.0
        quality_issues = []
        quality_regenerated = False

        try:
            ai_response, quality_score, quality_issues = self._quality.enforce(
                ai_response, query_type, complexity
            )

            # Quality-based regeneration (max 1 retry)
            if (quality_score < QUALITY_SCORE_THRESHOLD
                    and query_type not in ("greeting", "simple_qa")
                    and not personality_regenerated):  # avoid double-regen
                log.info(
                    "Quality score %.3f < threshold %.3f — regenerating (max %d)",
                    quality_score, QUALITY_SCORE_THRESHOLD, QUALITY_REGEN_MAX,
                )
                # Regenerate with a quality-focused prompt hint
                quality_hint = (
                    "\n\n[QUALITY DIRECTIVE: Your previous response was too brief or "
                    "generic. Provide a thorough, expert-level answer with concrete "
                    "details. Do NOT use filler phrases. Lead with the key insight.]"
                )
                regen_response, regen_model, regen_provider, regen_meta = self._ai.generate(
                    history, user_message,
                    prompt_augment=prompt_augment + quality_hint,
                    personality=personality,
                )
                # Re-enforce quality on the regenerated response
                regen_cleaned, regen_score, _ = self._quality.enforce(
                    regen_response, query_type, complexity
                )
                # Keep the better response (safe fallback)
                if regen_score > quality_score:
                    ai_response = regen_cleaned
                    quality_score = regen_score
                    active_model = regen_model
                    provider = regen_provider
                    quality_regenerated = True
                    log.info("Quality regen succeeded: new score=%.3f", regen_score)
                else:
                    log.info("Quality regen did not improve (%.3f vs %.3f) — keeping original",
                             regen_score, quality_score)

        except Exception as exc:
            log.warning("Quality enforcement failed (non-fatal): %s", exc)

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
            # Phase 13: Quality metadata
            "quality_score": round(quality_score, 3),
            "quality_regenerated": quality_regenerated,
            # Phase 13: Real-time search metadata
            "realtime_search": realtime_used,
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
