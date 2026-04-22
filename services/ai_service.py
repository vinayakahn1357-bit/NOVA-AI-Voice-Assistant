"""
services/ai_service.py - LLM Provider Dispatch for NOVA V2
Handles all communication with AI providers: Groq (fast default) and NVIDIA (advanced reasoning).
Includes intelligent routing, performance tracking, retry logic, response caching,
adaptive intelligence, latency logging, and per-personality temperature routing (Phase 12).
"""

import json
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import (
    get_settings, NOVA_ENV, NOVA_LIVE_MODE,
    GROQ_API_URL, NVIDIA_API_URL, API_TIMEOUT,
)
from utils.logger import get_logger
from utils.errors import NovaProviderError
from utils.retry_handler import with_retry

log = get_logger("ai_service")


class AIService:
    """
    Encapsulates all LLM provider communication with:
    - Intelligent model routing via ModelRouter (Groq + NVIDIA)
    - Performance tracking & circuit breaker via PerformanceTracker
    - Retry logic on transient failures
    - Response caching for repeated queries
    - Adaptive intelligence (query-aware param tuning)
    - Response formatting and cleanup
    - Debug metadata on every response
    """

    def __init__(self, prompt_builder, hybrid_evaluator=None, cache_service=None,
                 query_analyzer=None, response_formatter=None,
                 _unused_validator=None, model_router=None,
                 performance_tracker=None):
        self._prompt = prompt_builder
        self._hybrid = hybrid_evaluator  # Kept for backward compat, unused in V2
        self._cache = cache_service
        self._analyzer = query_analyzer
        self._formatter = response_formatter
        self._router = model_router
        self._tracker = performance_tracker

        # ── Shared HTTP session with connection pooling ────────────────────
        # Eliminates TCP+TLS handshake per call (~100-200ms savings)
        self._http = requests.Session()
        _adapter = HTTPAdapter(
            pool_connections=10,   # number of hosts to keep connections for
            pool_maxsize=20,       # max connections per host
            max_retries=Retry(  # type: ignore[arg-type]
                total=0,           # retries handled by our own retry_handler
                raise_on_status=False,
            ),
        )
        self._http.mount("https://", _adapter)  # type: ignore[arg-type]
        self._http.mount("http://", _adapter)  # type: ignore[arg-type]
        log.info("HTTP session pool initialized (pool_connections=10, pool_maxsize=20)")

    # --- Provider Checks ---

    @staticmethod
    def _groq_configured():
        return bool(str(get_settings().get("groq_api_key", "")).strip())

    @staticmethod
    def _nvidia_configured():
        return bool(str(get_settings().get("nvidia_api_key", "")).strip())


    @staticmethod
    def _resolve_temperature(personality: str = "default") -> float:
        """
        Phase 12: Return the per-personality temperature.
        Falls back to global settings temperature for 'default' or unknown keys.
        Per-personality temperatures are fixed and override the global setting.
        """
        try:
            from services.personality_service import get_personality_temperature, VALID_PERSONALITIES
            if personality and personality in VALID_PERSONALITIES:
                temp = get_personality_temperature(personality)
                log.debug("Temperature override: personality=%s temp=%.2f", personality, temp)
                return temp
        except ImportError:
            pass
        # Fallback to global setting
        return float(get_settings().get("temperature", 0.7))

    @with_retry(label="groq")
    def _call_groq(self, messages, stream=False, model_override="", temperature=None):
        """Call Groq API with retry logic. Accepts optional temperature override."""
        settings = get_settings()
        groq_model = model_override or settings["groq_model"] or "llama-3.3-70b-versatile"
        effective_temp = temperature if temperature is not None else settings["temperature"]
        payload = {
            "model": groq_model,
            "messages": messages,
            "temperature": effective_temp,
            "max_tokens": settings["num_predict"],
            "top_p": settings["top_p"],
            "stream": stream,
        }
        headers = {
            "Authorization": "Bearer " + settings["groq_api_key"],
            "Content-Type": "application/json",
        }
        return self._http.post(GROQ_API_URL, json=payload, headers=headers,
                               stream=stream, timeout=API_TIMEOUT)

    @with_retry(label="nvidia")
    def _call_nvidia(self, messages, stream=False, model_override="", temperature=None):
        """Call NVIDIA API (OpenAI-compatible format) with retry logic. Accepts optional temperature override."""
        settings = get_settings()
        nvidia_model = model_override or settings["nvidia_model"] or "nvidia/llama-3.3-70b-instruct"
        effective_temp = temperature if temperature is not None else settings["temperature"]
        payload = {
            "model": nvidia_model,
            "messages": messages,
            "temperature": effective_temp,
            "max_tokens": settings["num_predict"],
            "top_p": settings["top_p"],
            "stream": stream,
        }
        headers = {
            "Authorization": "Bearer " + settings["nvidia_api_key"],
            "Content-Type": "application/json",
        }
        return self._http.post(NVIDIA_API_URL, json=payload, headers=headers,
                               stream=stream, timeout=API_TIMEOUT)

    # --- Resolve Provider ---

    def _resolve_provider(self, provider):
        """Resolve the effective provider. V2: groq, nvidia, or balanced."""
        import os

        # Fallback: if provider is empty/None, read from env or default to groq
        if not provider:
            provider = os.getenv("NOVA_PROVIDER", "groq")
            log.warning("No provider set, defaulting to '%s'", provider)

        # 'balanced' is managed externally by hybrid_service — treat as groq for internal resolution
        if provider == "balanced":
            return "groq" if self._groq_configured() else "nvidia"
        if provider == "nvidia" and not self._nvidia_configured():
            if self._groq_configured():
                log.info("NVIDIA not configured; falling back to Groq")
                return "groq"
            log.warning("No AI provider API keys configured — defaulting to 'groq' (calls will fail until keys are set)")
            return "groq"
        if provider == "groq" and not self._groq_configured():
            if self._nvidia_configured():
                log.info("Groq not configured; falling back to NVIDIA")
                return "nvidia"
            log.warning("No AI provider API keys configured — defaulting to 'groq' (calls will fail until keys are set)")
            return "groq"
        # Clamp unknown providers to groq
        if provider not in ("groq", "nvidia"):
            log.warning("Unknown provider '%s', defaulting to 'groq'", provider)
            return "groq" if self._groq_configured() else "nvidia"
        log.info("Provider resolved: %s", provider)
        return provider

    def _get_failover(self, failed_provider):
        """Get an alternative provider when one fails. V2: nvidia→groq only."""
        if failed_provider == "nvidia" and self._groq_configured():
            if not self._tracker or self._tracker.is_available("groq"):
                return "groq"
        # Groq is last resort — no further fallback
        return None

    # --- Full (Non-Streaming) Generation ---

    def generate(self, history, user_message, prompt_augment="", personality="default"):
        """
        Generate a full AI response with intelligent routing, performance tracking,
        adaptive intelligence, caching, and timing.
        Args:
            history: conversation history
            user_message: current user message
            prompt_augment: optional agent-mode prompt instructions
            personality: personality mode key for prompt styling
        Returns: (ai_response, active_model, provider, metadata)
        """
        t0 = time.time()
        settings = get_settings()
        base_provider = self._resolve_provider(settings["provider"])
        fallback_used = False

        # Adaptive Intelligence: analyse query
        qa = self._analyzer.analyze(user_message) if self._analyzer else {}

        # Intelligent Routing (V2: select_provider)
        route_decision = {}
        if self._router:
            provider = self._router.select_provider(user_message, qa)
            provider = self._resolve_provider(provider)
            route_decision = {"provider": provider, "reason": "v2_router"}
            log.info("Provider selected: %s (query_type=%s, complexity=%d)",
                     provider, qa.get("query_type", "?"), qa.get("complexity", 0))
        else:
            provider = base_provider

        # Build chat messages (OpenAI format — used by both Groq and NVIDIA)
        chat_messages = self._prompt.build_chat_messages(history, personality)

        # Inject agent-mode prompt augmentation
        if prompt_augment:
            chat_messages.insert(-1, {
                "role": "system",
                "content": prompt_augment,
            })

        # Cache check
        if self._cache:
            cached = self._cache.get(provider, settings.get("groq_model", ""), chat_messages)
            if cached:
                log.info("Cache HIT (%.3fs)", time.time() - t0)
                return cached, settings.get("groq_model", ""), provider, {"cached": True}

        ai_response = ""
        active_model = settings["groq_model"]
        call_t0 = time.time()

        # Phase 12: Resolve per-personality temperature
        personality_temp = self._resolve_temperature(personality)
        log.debug("generate: personality=%s temperature=%.2f", personality, personality_temp)

        try:
            if provider == "nvidia" and self._nvidia_configured():
                ai_response, active_model = self._generate_nvidia(chat_messages, qa, temperature=personality_temp)
            elif provider == "groq" and self._groq_configured():
                ai_response, active_model = self._generate_groq(chat_messages, qa, temperature=personality_temp)
            else:
                log.warning("No AI provider keys configured for '%s' — returning fallback response", provider)

            # Record success in tracker
            if self._tracker:
                self._tracker.record_success(provider, time.time() - call_t0)

        except NovaProviderError as e:
            call_latency = time.time() - call_t0
            log.error("Provider %s failed (%.2fs): %s", provider, call_latency, e)

            # Record failure in tracker
            if self._tracker:
                self._tracker.record_failure(provider, call_latency)

            # Auto-failover to alternative provider
            alt_provider = self._get_failover(provider)
            if alt_provider:
                fallback_used = True
                log.warning("FALLBACK: %s → %s (reason: %s)", provider, alt_provider, e)
                try:
                    call_t0 = time.time()
                    if alt_provider == "groq":
                        ai_response, active_model = self._generate_groq(
                            chat_messages, qa, temperature=personality_temp)
                    else:
                        ai_response, active_model = self._generate_nvidia(
                            chat_messages, qa, temperature=personality_temp)
                    provider = alt_provider
                    if self._tracker:
                        self._tracker.record_success(alt_provider, time.time() - call_t0)
                except Exception as e2:
                    log.error("Failover to %s also failed: %s", alt_provider, e2)
                    if self._tracker:
                        self._tracker.record_failure(alt_provider, time.time() - call_t0)
            else:
                log.error("No failover available for %s", provider)

        if not ai_response:
            ai_response = "I'm not sure how to respond to that. Could you rephrase?"

        # Response formatting
        if self._formatter:
            ai_response = self._formatter.format(
                ai_response, qa.get("query_type", "conversation")
            )

        elapsed = round(time.time() - t0, 2)
        log.info("Response: provider=%s model=%s type=%s time=%.2fs len=%d fallback=%s",
                 provider, active_model, qa.get('query_type', '?'), elapsed,
                 len(ai_response), fallback_used)

        # Cache store
        if self._cache:
            self._cache.put(provider, active_model, chat_messages, ai_response)

        # Build debug metadata
        metadata = {
            "selected_model": active_model,
            "provider": provider,
            "query_type": qa.get("query_type", "unknown"),
            "complexity": qa.get("complexity", 0),
            "latency": elapsed,
            "fallback_used": fallback_used,
            "route_reason": route_decision.get("reason", ""),
        }

        return ai_response, active_model, provider, metadata

    # --- Provider-Specific Generation ---

    def _generate_groq(self, messages, qa=None, temperature=None):
        """Generate via Groq with adaptive params. Returns (text, model)."""
        settings = get_settings()
        r = self._call_groq(messages, model_override=settings["groq_model"], temperature=temperature)
        assert r is not None
        if r.status_code != 200:
            try:
                err = r.json().get("error", {}).get("message",
                      "Groq API error (%d)" % r.status_code)
            except Exception:
                err = "Groq API error (%d): %s" % (r.status_code, r.text[:300])
            raise NovaProviderError(err, code="GROQ_ERROR")
        try:
            text = r.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            raise NovaProviderError("Groq response parse error: %s" % exc, code="GROQ_PARSE")
        return text, settings["groq_model"]

    def _generate_nvidia(self, messages, qa=None, temperature=None):
        """Generate via NVIDIA with adaptive params. Returns (text, model)."""
        settings = get_settings()
        r = self._call_nvidia(messages, model_override=settings["nvidia_model"], temperature=temperature)
        assert r is not None
        if r.status_code != 200:
            try:
                err = r.json().get("error", {}).get("message",
                      "NVIDIA API error (%d)" % r.status_code)
            except Exception:
                err = "NVIDIA API error (%d): %s" % (r.status_code, r.text[:300])
            raise NovaProviderError(err, code="NVIDIA_ERROR")
        try:
            text = r.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            raise NovaProviderError("NVIDIA response parse error: %s" % exc, code="NVIDIA_PARSE")
        return text, settings["nvidia_model"]


    # --- Streaming Generation ---

    def generate_stream(self, history, user_message, personality="default"):
        """
        Generator yielding SSE-formatted tokens.
        V2: Routes to Groq or NVIDIA streaming based on select_provider().
        Phase 12: Applies per-personality temperature to streaming calls.
        """
        settings = get_settings()
        chat_messages = self._prompt.build_chat_messages(history, personality)

        # Analyse query for routing
        qa = self._analyzer.analyze(user_message) if self._analyzer else {}

        # Phase 12: Resolve personality temperature for streaming
        personality_temp = self._resolve_temperature(personality)
        log.debug("generate_stream: personality=%s temperature=%.2f", personality, personality_temp)

        # Route to provider
        if self._router:
            provider = self._router.select_provider(user_message, qa)
            provider = self._resolve_provider(provider)
        else:
            raw_provider = settings["provider"]
            provider = self._resolve_provider(raw_provider)

        full_reply = []
        active_model = settings["groq_model"]

        use_nvidia = provider == "nvidia" and self._nvidia_configured()
        use_groq = provider == "groq" and self._groq_configured()

        if use_nvidia:
            active_model = settings["nvidia_model"]

        log.info("Stream: provider=%s model=%s", provider, active_model)

        try:
            if use_nvidia:
                yield from self._stream_nvidia(chat_messages, full_reply, temperature=personality_temp)
            elif use_groq:
                yield from self._stream_groq(chat_messages, full_reply, temperature=personality_temp)
            else:
                yield 'data: %s\n\n' % json.dumps({"error": "No AI provider available."})
                return

        except requests.exceptions.ConnectionError:
            yield 'data: %s\n\n' % json.dumps({"error": "Cannot reach AI engine."})
            return
        except requests.exceptions.Timeout:
            yield 'data: %s\n\n' % json.dumps({"error": "AI engine timed out."})
            return
        except NovaProviderError as exc:
            yield 'data: %s\n\n' % json.dumps({"error": str(exc)})
            return
        except Exception as e:
            yield 'data: %s\n\n' % json.dumps({"error": str(e)})
            return

        complete_reply = "".join(full_reply).strip()
        if not complete_reply:
            yield 'data: %s\n\n' % json.dumps({"error": "AI returned an empty response."})

        yield 'data: %s\n\n' % json.dumps({"done": True, "session_id": "", "model": active_model})

    def generate_stream_hybrid(self, history, user_message, personality="default"):
        """
        Balanced-mode streaming: Groq-first with intelligent NVIDIA escalation.
        Delegates to services/hybrid_service.py — zero duplicate logic here.
        """
        from services.hybrid_service import hybrid_generate_stream
        settings = get_settings()

        full_reply = []
        yield from hybrid_generate_stream(
            self, history, user_message, full_reply, personality=personality
        )

        complete_reply = "".join(full_reply).strip()
        if not complete_reply:
            yield 'data: %s\n\n' % json.dumps({"error": "AI returned an empty response."})
            return

        # Emit the standard 'done' event so the frontend treats this identically
        active_model = settings.get("groq_model", "llama-3.3-70b-versatile")
        yield 'data: %s\n\n' % json.dumps({
            "done": True, "session_id": "", "model": active_model, "mode": "hybrid",
        })

    # --- Streaming Helpers ---

    def _stream_groq(self, messages, full_reply, temperature=None):
        """Stream from Groq API. Accepts optional temperature override."""
        settings = get_settings()
        log.info("Groq stream: model=%s temp=%s", settings["groq_model"], temperature)
        with self._call_groq(messages, stream=True, temperature=temperature) as r:  # type: ignore[union-attr]
            if r.status_code != 200:
                log.error("Groq stream ERROR: %d", r.status_code)
                yield 'data: %s\n\n' % json.dumps({"error": "Groq API error (%d)" % r.status_code})
                return
            yield from self._parse_groq_stream(r, full_reply)

    def _stream_nvidia(self, messages, full_reply, temperature=None):
        """Stream from NVIDIA API (OpenAI-compatible SSE format). Accepts optional temperature override."""
        settings = get_settings()
        nvidia_model = settings["nvidia_model"]
        log.info("NVIDIA stream: model=%s temp=%s", nvidia_model, temperature)
        try:
            with self._call_nvidia(messages, stream=True, temperature=temperature) as r:  # type: ignore[union-attr]
                if r.status_code != 200:
                    log.error("NVIDIA stream ERROR: %d", r.status_code)
                    # Fallback to Groq
                    if self._groq_configured():
                        log.warning("NVIDIA stream failed -> Groq fallback")
                        yield from self._stream_groq(messages, full_reply, temperature=temperature)
                        return
                    yield 'data: %s\n\n' % json.dumps({"error": "NVIDIA API error (%d)" % r.status_code})
                    return
                yield from self._parse_groq_stream(r, full_reply)  # Same SSE format
        except Exception as exc:
            log.warning("NVIDIA stream exception: %s, falling back to Groq", exc)
            if self._groq_configured():
                full_reply.clear()
                yield from self._stream_groq(messages, full_reply, temperature=temperature)
            else:
                yield 'data: %s\n\n' % json.dumps({"error": "NVIDIA stream failed: %s" % str(exc)})

    @staticmethod
    def _parse_groq_stream(response, full_reply):
        """Parse Groq's OpenAI-format SSE stream and yield tokens."""
        for line in response.iter_lines():
            if not line:
                continue
            line_str = line.decode("utf-8", errors="ignore")
            if not line_str.startswith("data: "):
                continue
            data_str = line_str[6:]
            if data_str.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                token = delta.get("content", "")
                if token:
                    full_reply.append(token)
                    yield 'data: %s\n\n' % json.dumps({"token": token})
            except (json.JSONDecodeError, IndexError, KeyError):
                continue
