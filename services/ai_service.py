"""
services/ai_service.py - LLM Provider Dispatch for NOVA (v4)
Handles all communication with AI providers: Ollama Cloud, Groq, and True Parallel Hybrid.
Includes intelligent routing, performance tracking, retry logic, response caching,
adaptive intelligence, model validation, and latency logging.
"""

import json
import time
import requests

from config import (
    get_settings, NOVA_ENV, NOVA_LIVE_MODE, OLLAMA_URL,
    GROQ_API_URL, API_TIMEOUT, LOCAL_TIMEOUT,
)
from utils.logger import get_logger
from utils.errors import NovaProviderError
from utils.retry_handler import with_retry

log = get_logger("ai_service")


class AIService:
    """
    Encapsulates all LLM provider communication with:
    - Intelligent model routing via ModelRouter
    - Performance tracking & circuit breaker via PerformanceTracker
    - Retry logic on transient failures
    - Response caching for repeated queries
    - True parallel hybrid execution via HybridEvaluator
    - Adaptive intelligence (query-aware param tuning)
    - Model validation via OllamaValidator
    - Response formatting and cleanup
    - Debug metadata on every response
    """

    def __init__(self, prompt_builder, hybrid_evaluator=None, cache_service=None,
                 query_analyzer=None, response_formatter=None,
                 ollama_validator=None, model_router=None,
                 performance_tracker=None):
        self._prompt = prompt_builder
        self._hybrid = hybrid_evaluator
        self._cache = cache_service
        self._analyzer = query_analyzer
        self._formatter = response_formatter
        self._ollama_validator = ollama_validator
        self._router = model_router
        self._tracker = performance_tracker

    # --- Provider Checks ---

    @staticmethod
    def _ollama_cloud_configured():
        return bool(get_settings().get("ollama_api_key", "").strip())

    @staticmethod
    def _groq_configured():
        return bool(get_settings().get("groq_api_key", "").strip())

    # --- Low-Level Provider Calls (with retry) ---

    @staticmethod
    @with_retry(label="ollama_local")
    def _call_ollama_local(full_prompt, stream=False):
        """Call local Ollama. Only allowed in local ENV."""
        if NOVA_ENV != "local":
            raise ConnectionError("Ollama local is disabled in production")
        import psutil
        settings = get_settings()
        payload = {
            "model": settings["model"],
            "prompt": full_prompt,
            "stream": stream,
            "options": {
                "temperature":    settings["temperature"],
                "top_p":          settings["top_p"],
                "num_predict":    settings["num_predict"],
                "num_ctx":        2048,
                "num_thread":     psutil.cpu_count(logical=False) or 2,
                "repeat_penalty": 1.1,
            }
        }
        return requests.post(OLLAMA_URL, json=payload, stream=stream, timeout=LOCAL_TIMEOUT)

    @with_retry(label="ollama_cloud")
    def _call_ollama_cloud(self, full_prompt, stream=False, model_override=""):
        """
        Call Ollama Cloud endpoint with:
        - Model validation via OllamaValidator
        - Retry logic on transient failures
        - Full logging of selected vs resolved model
        """
        settings = get_settings()
        cloud_url = settings["ollama_cloud_url"] or "https://api.ollama.com/api/generate"

        # Production guard: never hit localhost
        if NOVA_LIVE_MODE and ("localhost" in cloud_url or "127.0.0.1" in cloud_url):
            raise ConnectionError("Ollama Cloud URL points to localhost in production!")

        # Determine requested model
        requested_model = model_override or (
            settings["hybrid_ollama_model"]
            if settings["provider"] == "hybrid"
            else settings["model"]
        ) or "gemma3:12b"

        # Model Validation & Resolution
        model = requested_model
        if self._ollama_validator:
            resolution = self._ollama_validator.resolve_model(requested_model)
            model = resolution["resolved"]
            if model != requested_model:
                log.warning(
                    "Ollama model resolved: '%s' -> '%s' (%s)",
                    requested_model, model, resolution["match_type"],
                )
            else:
                log.info("Ollama model validated: '%s' (match=%s)",
                         model, resolution["match_type"])
        else:
            log.info("Ollama call: model='%s' (no validator)", model)

        payload = {
            "model": model,
            "prompt": full_prompt,
            "stream": stream,
            "options": {
                "temperature": settings["temperature"],
                "top_p":       settings["top_p"],
                "num_predict": settings["num_predict"],
            }
        }
        headers = {"Content-Type": "application/json"}
        if settings["ollama_api_key"]:
            headers["Authorization"] = "Bearer " + settings["ollama_api_key"]

        r = requests.post(cloud_url, json=payload, headers=headers,
                          stream=stream, timeout=API_TIMEOUT)

        # Handle model-not-found in 200-with-error-body
        if not stream and r.status_code == 200:
            try:
                body = r.json()
                err_msg = body.get("error", "")
                if err_msg and "not found" in err_msg.lower():
                    log.error("Ollama model-not-found in body: %s (model='%s')", err_msg, model)
                    if self._ollama_validator:
                        available = self._ollama_validator.discover_models(force_refresh=True)
                        if available:
                            fallback_model = available[0]
                            log.warning("Retrying with fallback model: '%s'", fallback_model)
                            payload["model"] = fallback_model
                            r = requests.post(cloud_url, json=payload, headers=headers,
                                              stream=stream, timeout=API_TIMEOUT)
            except (ValueError, KeyError):
                pass

        return r

    @staticmethod
    @with_retry(label="groq")
    def _call_groq(messages, stream=False, model_override=""):
        """Call Groq API with retry logic."""
        settings = get_settings()
        groq_model = model_override or (
            settings["hybrid_groq_model"]
            if settings["provider"] == "hybrid"
            else settings["groq_model"]
        ) or "llama-3.3-70b-versatile"
        payload = {
            "model": groq_model,
            "messages": messages,
            "temperature": settings["temperature"],
            "max_tokens": settings["num_predict"],
            "top_p": settings["top_p"],
            "stream": stream,
        }
        headers = {
            "Authorization": "Bearer " + settings["groq_api_key"],
            "Content-Type": "application/json",
        }
        return requests.post(GROQ_API_URL, json=payload, headers=headers,
                             stream=stream, timeout=API_TIMEOUT)

    # --- Resolve Provider ---

    def _resolve_provider(self, provider):
        """Resolve the effective provider, handling fallbacks."""
        if provider == "ollama_cloud" and not self._ollama_cloud_configured():
            if self._groq_configured():
                log.info("ollama_cloud not configured; falling back to Groq")
                return "groq"
            elif not NOVA_LIVE_MODE:
                return "ollama"
            else:
                raise NovaProviderError("No cloud AI provider configured.")
        return provider

    def _get_failover(self, failed_provider):
        """Get an alternative provider when one fails."""
        if "ollama" in failed_provider and self._groq_configured():
            if not self._tracker or self._tracker.is_available("groq"):
                return "groq"
        if "groq" in failed_provider and self._ollama_cloud_configured():
            if not self._tracker or self._tracker.is_available("ollama"):
                return "ollama_cloud"
        return None

    # --- Full (Non-Streaming) Generation ---

    def generate(self, history, user_message, prompt_augment=""):
        """
        Generate a full AI response with intelligent routing, performance tracking,
        adaptive intelligence, caching, and timing.
        Args:
            history: conversation history
            user_message: current user message
            prompt_augment: optional agent-mode prompt instructions
        Returns: (ai_response, active_model, provider, metadata)
        """
        t0 = time.time()
        settings = get_settings()
        base_provider = self._resolve_provider(settings["provider"])
        fallback_used = False

        # Adaptive Intelligence: analyse query
        qa = self._analyzer.analyze(user_message) if self._analyzer else {}

        # Intelligent Routing
        route_decision = {}
        if self._router:
            route_decision = self._router.route(qa, base_provider)
            provider = route_decision.get("provider", base_provider)
            if route_decision.get("adjusted"):
                fallback_used = True
                log.info("Router adjusted: %s -> %s (%s)",
                         base_provider, provider, route_decision.get("reason"))
        else:
            provider = base_provider

        # Build prompts (identical for both providers)
        full_prompt = self._prompt.build_ollama_prompt(history)
        groq_messages = self._prompt.build_chat_messages(history)

        # Inject agent-mode prompt augmentation
        if prompt_augment:
            full_prompt = full_prompt.replace("Nova:", prompt_augment + "\nNova:")
            # For Groq: inject as system message before last user message
            groq_messages.insert(-1, {
                "role": "system",
                "content": prompt_augment,
            })

        # Cache check
        if self._cache and provider != "hybrid":
            cached = self._cache.get(provider, settings.get("model", ""), groq_messages)
            if cached:
                log.info("Cache HIT (%.3fs)", time.time() - t0)
                return cached, settings.get("model", ""), provider, {"cached": True}

        ai_response = ""
        active_model = settings["model"]
        call_t0 = time.time()

        try:
            if provider == "hybrid" and self._hybrid:
                ai_response, active_model, provider = self._generate_hybrid_parallel(
                    user_message, full_prompt, groq_messages, qa
                )
            elif provider == "groq" and self._groq_configured():
                ai_response, active_model = self._generate_groq(groq_messages, qa)
            elif provider in ("ollama_cloud", "ollama") and self._ollama_cloud_configured():
                ai_response, active_model = self._generate_ollama_cloud(full_prompt, qa)
            else:
                ai_response, active_model = self._generate_local_with_fallback(
                    full_prompt, groq_messages
                )

            # Record success in tracker
            if self._tracker and provider != "hybrid":
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
                log.warning("Failover: %s -> %s", provider, alt_provider)
                try:
                    call_t0 = time.time()
                    if alt_provider == "groq":
                        ai_response, active_model = self._generate_groq(
                            groq_messages, qa)
                    else:
                        ai_response, active_model = self._generate_ollama_cloud(
                            full_prompt, qa)
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
        if self._cache and provider != "hybrid":
            self._cache.put(provider, active_model, groq_messages, ai_response)

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

    def _generate_groq(self, messages, qa=None):
        """Generate via Groq with adaptive params. Returns (text, model)."""
        settings = get_settings()
        r = self._call_groq(messages, model_override=settings["groq_model"])
        if r.status_code != 200:
            err = r.json().get("error", {}).get("message", "Groq API error (%d)" % r.status_code)
            raise NovaProviderError(err, code="GROQ_ERROR")
        text = r.json()["choices"][0]["message"]["content"].strip()
        return text, settings["groq_model"]

    def _generate_ollama_cloud(self, full_prompt, qa=None):
        """
        Generate via Ollama Cloud with error recovery.
        Handles 404 model-not-found by retrying with a validated model.
        Returns (text, model).
        """
        settings = get_settings()
        r = self._call_ollama_cloud(full_prompt)

        # Handle 404 / 400 model-not-found with retry
        if r.status_code in (404, 400):
            error_text = r.text[:300]
            log.error("Ollama Cloud %d: %s", r.status_code, error_text)

            # Try to get correct model via validator and retry
            if self._ollama_validator:
                available = self._ollama_validator.discover_models(force_refresh=True)
                if available:
                    fallback_model = available[0]
                    log.warning(
                        "Retrying with discovered model: '%s' (was: '%s')",
                        fallback_model, settings.get("model", "?")
                    )
                    r = self._call_ollama_cloud(full_prompt, model_override=fallback_model)
                    if r.status_code == 200:
                        body = r.json()
                        if not body.get("error"):
                            text = body.get("response", "").strip()
                            return text, fallback_model

            raise NovaProviderError(
                "Ollama Cloud error (%d): %s" % (r.status_code, error_text),
                code="OLLAMA_CLOUD_ERROR",
            )

        if r.status_code != 200:
            raise NovaProviderError(
                "Ollama Cloud error (%d): %s" % (r.status_code, r.text[:200]),
                code="OLLAMA_CLOUD_ERROR",
            )

        # Check for error in response body
        try:
            body = r.json()
            if "error" in body and body["error"]:
                err = body["error"]
                log.error("Ollama returned error in body: %s", err)
                raise NovaProviderError("Ollama: %s" % err, code="OLLAMA_CLOUD_ERROR")
            text = body.get("response", "").strip()
        except (ValueError, KeyError) as e:
            raise NovaProviderError("Invalid Ollama response: %s" % e,
                                    code="OLLAMA_CLOUD_ERROR")

        return text, settings["model"]

    # --- True Parallel Hybrid ---

    def _generate_hybrid_parallel(self, user_message, full_prompt, groq_messages,
                                   qa=None):
        """
        Execute both providers in parallel via HybridEvaluator.
        Returns: (ai_response, active_model, provider_source)
        """
        settings = get_settings()
        qa = qa or {}

        ollama_ok = self._ollama_cloud_configured()
        groq_ok = self._groq_configured()

        if not ollama_ok and not groq_ok:
            if NOVA_LIVE_MODE:
                raise NovaProviderError("No cloud AI provider configured for hybrid mode.")
            # Local fallback
            return self._generate_local_with_fallback(full_prompt, groq_messages) + ("local",)

        # Single-provider fast path
        if not ollama_ok:
            text, model = self._generate_groq(groq_messages, qa)
            return text, model, "groq"
        if not groq_ok:
            text, model = self._generate_ollama_cloud(full_prompt, qa)
            return text, model, "ollama_cloud"

        # Both configured: true parallel execution
        def _ollama_fn():
            r = self._call_ollama_cloud(full_prompt)
            if r.status_code != 200:
                raise NovaProviderError("Ollama: %d" % r.status_code)
            body = r.json()
            if body.get("error"):
                raise NovaProviderError("Ollama: %s" % body["error"])
            return body.get("response", "").strip()

        def _groq_fn():
            r = self._call_groq(groq_messages)
            if r.status_code != 200:
                raise NovaProviderError("Groq: %d" % r.status_code)
            return r.json()["choices"][0]["message"]["content"].strip()

        result = self._hybrid.evaluate_parallel(_ollama_fn, _groq_fn, user_message, qa)

        # Record hybrid metrics
        if self._tracker:
            if result.get("ollama_time"):
                if result.get("source") != "none" and result.get("ollama_score", 0) > 0:
                    self._tracker.record_success("ollama", result["ollama_time"])
                elif result.get("ollama_time") > 0 and result.get("ollama_score", 0) == 0:
                    self._tracker.record_failure("ollama", result["ollama_time"])
            if result.get("groq_time"):
                if result.get("source") != "none" and result.get("groq_score", 0) > 0:
                    self._tracker.record_success("groq", result["groq_time"])
                elif result.get("groq_time") > 0 and result.get("groq_score", 0) == 0:
                    self._tracker.record_failure("groq", result["groq_time"])

        if not result["response"]:
            raise NovaProviderError("Both AI providers failed in hybrid mode.")

        # Map source to model name
        source = result["source"]
        if source in ("groq", "merged"):
            model = settings["hybrid_groq_model"] or settings["groq_model"]
        else:
            model = settings["hybrid_ollama_model"] or settings["model"]

        if source == "merged":
            model = settings["hybrid_ollama_model"] + "+" + settings["hybrid_groq_model"]

        return result["response"], model, "hybrid(%s)" % source

    def _generate_local_with_fallback(self, full_prompt, groq_messages):
        """Try local Ollama with Groq fallback."""
        settings = get_settings()
        ai_response = ""
        active_model = settings["model"]

        try:
            r = self._call_ollama_local(full_prompt)
            if r.status_code == 200:
                ai_response = r.json().get("response", "").strip()
            else:
                log.warning("Ollama local error (%d), trying Groq fallback", r.status_code)
        except Exception as exc:
            log.warning("Ollama local failed: %s", exc)

        if not ai_response and self._groq_configured():
            log.info("Fallback to Groq")
            try:
                text, model = self._generate_groq(groq_messages)
                return text, model
            except Exception as exc2:
                log.warning("Groq fallback also failed: %s", exc2)

        if not ai_response:
            raise NovaProviderError("AI engine error - all providers failed")

        return ai_response, active_model

    # --- Streaming Generation ---

    def generate_stream(self, history, user_message):
        """
        Generator yielding SSE-formatted tokens.
        Supports streaming for single-provider modes.
        For hybrid mode: falls back to non-streaming parallel then streams result.
        """
        settings = get_settings()
        provider = self._resolve_provider(settings["provider"])
        full_prompt = self._prompt.build_ollama_prompt(history)
        groq_messages = self._prompt.build_chat_messages(history)

        full_reply = []
        active_model = settings["model"]

        use_groq = provider == "groq" and self._groq_configured()
        use_ollama_cloud = provider == "ollama_cloud"
        use_hybrid = provider == "hybrid"

        if use_groq:
            active_model = settings["groq_model"]

        try:
            if use_hybrid:
                yield from self._stream_hybrid_parallel(
                    user_message, full_prompt, groq_messages, full_reply
                )
                if full_reply and full_reply[-1].startswith("__MODEL__"):
                    active_model = full_reply.pop()[9:]

            elif use_groq:
                yield from self._stream_groq(groq_messages, full_reply)

            elif use_ollama_cloud:
                yield from self._stream_ollama_cloud(
                    full_prompt, groq_messages, full_reply
                )
                if full_reply and full_reply[-1].startswith("__MODEL__"):
                    active_model = full_reply.pop()[9:]

            else:
                if NOVA_ENV != "local":
                    yield 'data: %s\n\n' % json.dumps({"error": "No AI provider available."})
                    return
                yield from self._stream_ollama_local(full_prompt, full_reply)

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

    # --- Streaming Helpers ---

    def _stream_hybrid_parallel(self, user_message, full_prompt, groq_messages, full_reply):
        """
        Hybrid streaming: execute both providers in parallel (non-streaming),
        pick/merge the best, then stream the result token-by-token.
        """
        if not self._hybrid:
            yield from self._stream_groq(groq_messages, full_reply)
            return

        settings = get_settings()
        ollama_ok = self._ollama_cloud_configured()
        groq_ok = self._groq_configured()

        if not ollama_ok and not groq_ok:
            yield 'data: %s\n\n' % json.dumps({"error": "No AI provider configured."})
            return

        if not ollama_ok:
            yield from self._stream_groq(groq_messages, full_reply)
            return
        if not groq_ok:
            yield from self._stream_ollama_cloud(full_prompt, groq_messages, full_reply)
            return

        # Both available: parallel non-streaming execution
        def _ollama_fn():
            r = self._call_ollama_cloud(full_prompt)
            if r.status_code != 200:
                raise NovaProviderError("Ollama: %d" % r.status_code)
            body = r.json()
            if body.get("error"):
                raise NovaProviderError("Ollama: %s" % body["error"])
            return body.get("response", "").strip()

        def _groq_fn():
            r = self._call_groq(groq_messages)
            if r.status_code != 200:
                raise NovaProviderError("Groq: %d" % r.status_code)
            return r.json()["choices"][0]["message"]["content"].strip()

        result = self._hybrid.evaluate_parallel(_ollama_fn, _groq_fn, user_message)

        if not result["response"]:
            yield 'data: %s\n\n' % json.dumps({"error": "Both providers failed in hybrid."})
            return

        # Stream the winning response word-by-word for smooth UX
        response = result["response"]
        words = response.split(" ")
        chunk_size = 3

        for i in range(0, len(words), chunk_size):
            token = " ".join(words[i:i + chunk_size])
            if i > 0:
                token = " " + token
            full_reply.append(token)
            yield 'data: %s\n\n' % json.dumps({"token": token})

        # Append model tag
        source = result["source"]
        if source in ("groq", "merged"):
            model = settings["hybrid_groq_model"] or settings["groq_model"]
        else:
            model = settings["hybrid_ollama_model"] or settings["model"]
        if source == "merged":
            model = settings["hybrid_ollama_model"] + "+" + settings["hybrid_groq_model"]
        full_reply.append("__MODEL__" + model)

    def _stream_groq(self, messages, full_reply):
        """Stream from Groq API."""
        settings = get_settings()
        log.info("Groq stream: model=%s", settings["groq_model"])
        with self._call_groq(messages, stream=True) as r:
            if r.status_code != 200:
                log.error("Groq stream ERROR: %d", r.status_code)
                yield 'data: %s\n\n' % json.dumps({"error": "Groq API error (%d)" % r.status_code})
                return
            yield from self._parse_groq_stream(r, full_reply)

    def _stream_ollama_local(self, full_prompt, full_reply):
        """Stream from local Ollama."""
        with self._call_ollama_local(full_prompt, stream=True) as r:
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    token = chunk.get("response", "")
                    if token:
                        full_reply.append(token)
                        yield 'data: %s\n\n' % json.dumps({"token": token})
                    if chunk.get("done"):
                        break
                except json.JSONDecodeError:
                    continue

    def _stream_ollama_cloud(self, full_prompt, groq_messages, full_reply):
        """Stream from Ollama Cloud with Groq fallback."""
        ollama_cloud_ok = False
        try:
            with self._call_ollama_cloud(full_prompt, stream=True) as r:
                if r.status_code == 200:
                    for line in r.iter_lines():
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                            if chunk.get("error"):
                                log.warning("Ollama Cloud error: %s", chunk["error"])
                                break
                            token = chunk.get("response", "")
                            if token:
                                full_reply.append(token)
                                yield 'data: %s\n\n' % json.dumps({"token": token})
                            if chunk.get("done"):
                                ollama_cloud_ok = True
                                break
                        except json.JSONDecodeError:
                            continue
                else:
                    log.warning("Ollama Cloud error: %d", r.status_code)
        except Exception as exc:
            log.warning("Ollama Cloud stream exception: %s", exc)

        if ollama_cloud_ok and full_reply:
            return

        # Fallback to Groq
        if self._groq_configured():
            log.info("Ollama Cloud failed -> Groq stream fallback")
            full_reply.clear()
            try:
                with self._call_groq(groq_messages, stream=True) as r:
                    if r.status_code == 200:
                        yield from self._parse_groq_stream(r, full_reply)
                        settings = get_settings()
                        full_reply.append("__MODEL__" + settings["groq_model"])
                    else:
                        log.warning("Groq fallback stream error: %d", r.status_code)
            except Exception as exc2:
                log.warning("Groq fallback stream exception: %s", exc2)
        elif not full_reply:
            yield 'data: %s\n\n' % json.dumps({"error": "Ollama Cloud failed, no Groq fallback."})

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
