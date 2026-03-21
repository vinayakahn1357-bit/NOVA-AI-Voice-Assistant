"""
services/ai_service.py — LLM Provider Dispatch for NOVA (v2)
Handles all communication with AI providers: Ollama Cloud, Groq, and True Parallel Hybrid.
Includes retry logic, response caching, and latency logging.
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
    - Retry logic on transient failures
    - Response caching for repeated queries
    - True parallel hybrid execution via HybridEvaluator
    - Latency logging on every call
    """

    def __init__(self, prompt_builder, hybrid_evaluator=None, cache_service=None):
        self._prompt = prompt_builder
        self._hybrid = hybrid_evaluator
        self._cache = cache_service

    # ─── Provider Checks ──────────────────────────────────────────────────

    @staticmethod
    def _ollama_cloud_configured() -> bool:
        return bool(get_settings().get("ollama_api_key", "").strip())

    @staticmethod
    def _groq_configured() -> bool:
        return bool(get_settings().get("groq_api_key", "").strip())

    # ─── Low-Level Provider Calls (with retry) ────────────────────────────

    @staticmethod
    @with_retry(label="ollama_local")
    def _call_ollama_local(full_prompt: str, stream: bool = False):
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

    @staticmethod
    @with_retry(label="ollama_cloud")
    def _call_ollama_cloud(full_prompt: str, stream: bool = False,
                           model_override: str = ""):
        """Call Ollama Cloud endpoint with retry logic."""
        settings = get_settings()
        cloud_url = settings["ollama_cloud_url"] or "https://api.ollama.com/api/generate"

        # Production guard: never hit localhost
        if NOVA_LIVE_MODE and ("localhost" in cloud_url or "127.0.0.1" in cloud_url):
            raise ConnectionError("Ollama Cloud URL points to localhost in production!")

        model = model_override or (
            settings["hybrid_ollama_model"]
            if settings["provider"] == "hybrid"
            else settings["model"]
        ) or "gemma3:12b"

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
            headers["Authorization"] = f"Bearer {settings['ollama_api_key']}"
        return requests.post(cloud_url, json=payload, headers=headers,
                             stream=stream, timeout=API_TIMEOUT)

    @staticmethod
    @with_retry(label="groq")
    def _call_groq(messages: list, stream: bool = False, model_override: str = ""):
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
            "Authorization": f"Bearer {settings['groq_api_key']}",
            "Content-Type": "application/json",
        }
        return requests.post(GROQ_API_URL, json=payload, headers=headers,
                             stream=stream, timeout=API_TIMEOUT)

    # ─── Resolve Provider ─────────────────────────────────────────────────

    def _resolve_provider(self, provider: str) -> str:
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

    # ─── Full (Non-Streaming) Generation ──────────────────────────────────

    def generate(self, history: list, user_message: str) -> tuple:
        """
        Generate a full AI response with caching and timing.
        Returns: (ai_response: str, active_model: str, provider: str)
        """
        t0 = time.time()
        settings = get_settings()
        provider = self._resolve_provider(settings["provider"])

        # Build prompts (identical for both providers)
        full_prompt = self._prompt.build_ollama_prompt(history)
        groq_messages = self._prompt.build_chat_messages(history)

        # ── Cache check ───────────────────────────────────────────────
        if self._cache and provider != "hybrid":
            cached = self._cache.get(provider, settings.get("model", ""), groq_messages)
            if cached:
                log.info("Cache HIT (%.3fs)", time.time() - t0)
                return cached, settings.get("model", ""), provider

        ai_response = ""
        active_model = settings["model"]

        if provider == "hybrid" and self._hybrid:
            ai_response, active_model, provider = self._generate_hybrid_parallel(
                user_message, full_prompt, groq_messages
            )
        elif provider == "groq" and self._groq_configured():
            ai_response, active_model = self._generate_groq(groq_messages)
        elif provider == "ollama_cloud" and self._ollama_cloud_configured():
            ai_response, active_model = self._generate_ollama_cloud(full_prompt)
        else:
            ai_response, active_model = self._generate_local_with_fallback(
                full_prompt, groq_messages
            )

        if not ai_response:
            ai_response = "I'm not sure how to respond to that. Could you rephrase?"

        elapsed = round(time.time() - t0, 2)
        log.info("Generated response: provider=%s model=%s time=%.2fs len=%d",
                 provider, active_model, elapsed, len(ai_response))

        # ── Cache store ───────────────────────────────────────────────
        if self._cache and provider != "hybrid":
            self._cache.put(provider, active_model, groq_messages, ai_response)

        return ai_response, active_model, provider

    # ─── Provider-Specific Generation ─────────────────────────────────────

    def _generate_groq(self, messages: list) -> tuple:
        """Generate via Groq. Returns (text, model)."""
        settings = get_settings()
        r = self._call_groq(messages, model_override=settings["groq_model"])
        if r.status_code != 200:
            err = r.json().get("error", {}).get("message", f"Groq API error ({r.status_code})")
            raise NovaProviderError(err, code="GROQ_ERROR")
        text = r.json()["choices"][0]["message"]["content"].strip()
        return text, settings["groq_model"]

    def _generate_ollama_cloud(self, full_prompt: str) -> tuple:
        """Generate via Ollama Cloud. Returns (text, model)."""
        settings = get_settings()
        r = self._call_ollama_cloud(full_prompt)
        if r.status_code != 200:
            raise NovaProviderError(
                f"Ollama Cloud error ({r.status_code}): {r.text[:200]}",
                code="OLLAMA_CLOUD_ERROR",
            )
        text = r.json().get("response", "").strip()
        return text, settings["model"]

    # ─── True Parallel Hybrid ─────────────────────────────────────────────

    def _generate_hybrid_parallel(self, user_message, full_prompt, groq_messages) -> tuple:
        """
        Execute both providers in parallel via HybridEvaluator.
        Returns: (ai_response, active_model, provider_source)
        """
        settings = get_settings()

        ollama_ok = self._ollama_cloud_configured()
        groq_ok = self._groq_configured()

        if not ollama_ok and not groq_ok:
            if NOVA_LIVE_MODE:
                raise NovaProviderError("No cloud AI provider configured for hybrid mode.")
            # Local fallback
            return self._generate_local_with_fallback(full_prompt, groq_messages) + ("ollama",)

        # If only one provider is configured, skip parallel execution
        if not ollama_ok:
            text, model = self._generate_groq(groq_messages)
            return text, model, "groq"
        if not groq_ok:
            text, model = self._generate_ollama_cloud(full_prompt)
            return text, model, "ollama_cloud"

        # ── Both configured: true parallel execution ──────────────────
        def _ollama_fn():
            r = self._call_ollama_cloud(full_prompt)
            if r.status_code != 200:
                raise NovaProviderError(f"Ollama Cloud: {r.status_code}")
            return r.json().get("response", "").strip()

        def _groq_fn():
            r = self._call_groq(groq_messages)
            if r.status_code != 200:
                raise NovaProviderError(f"Groq: {r.status_code}")
            return r.json()["choices"][0]["message"]["content"].strip()

        result = self._hybrid.evaluate_parallel(_ollama_fn, _groq_fn, user_message)

        if not result["response"]:
            raise NovaProviderError("Both AI providers failed in hybrid mode.")

        # Map source to model name
        source = result["source"]
        if source in ("groq", "merged"):
            model = settings["hybrid_groq_model"] or settings["groq_model"]
        else:
            model = settings["hybrid_ollama_model"] or settings["model"]

        if source == "merged":
            model = f"{settings['hybrid_ollama_model']}+{settings['hybrid_groq_model']}"

        return result["response"], model, f"hybrid({source})"

    def _generate_local_with_fallback(self, full_prompt, groq_messages) -> tuple:
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
            raise NovaProviderError("AI engine error — all providers failed")

        return ai_response, active_model

    # ─── Streaming Generation ─────────────────────────────────────────────

    def generate_stream(self, history: list, user_message: str):
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
                # Hybrid: parallel execution, then stream the winning result
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
                    yield f'data: {json.dumps({"error": "No AI provider available."})}\n\n'
                    return
                yield from self._stream_ollama_local(full_prompt, full_reply)

        except requests.exceptions.ConnectionError:
            yield f'data: {json.dumps({"error": "Cannot reach AI engine."})}\n\n'
            return
        except requests.exceptions.Timeout:
            yield f'data: {json.dumps({"error": "AI engine timed out."})}\n\n'
            return
        except NovaProviderError as exc:
            yield f'data: {json.dumps({"error": str(exc)})}\n\n'
            return
        except Exception as e:
            yield f'data: {json.dumps({"error": str(e)})}\n\n'
            return

        complete_reply = "".join(full_reply).strip()
        if not complete_reply:
            yield f'data: {json.dumps({"error": "AI returned an empty response."})}\n\n'

        yield f'data: {json.dumps({"done": True, "session_id": "", "model": active_model})}\n\n'

    # ─── Streaming Helpers ────────────────────────────────────────────────

    def _stream_hybrid_parallel(self, user_message, full_prompt, groq_messages, full_reply):
        """
        Hybrid streaming: execute both providers in parallel (non-streaming),
        pick/merge the best, then stream the result token-by-token.
        This gives UX of streaming while leveraging parallel evaluation.
        """
        if not self._hybrid:
            # No evaluator — fall back to Groq stream
            yield from self._stream_groq(groq_messages, full_reply)
            return

        settings = get_settings()
        ollama_ok = self._ollama_cloud_configured()
        groq_ok = self._groq_configured()

        if not ollama_ok and not groq_ok:
            yield f'data: {json.dumps({"error": "No AI provider configured."})}\n\n'
            return

        # If only one provider available, stream directly from it
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
                raise NovaProviderError(f"Ollama: {r.status_code}")
            return r.json().get("response", "").strip()

        def _groq_fn():
            r = self._call_groq(groq_messages)
            if r.status_code != 200:
                raise NovaProviderError(f"Groq: {r.status_code}")
            return r.json()["choices"][0]["message"]["content"].strip()

        result = self._hybrid.evaluate_parallel(_ollama_fn, _groq_fn, user_message)

        if not result["response"]:
            yield f'data: {json.dumps({"error": "Both providers failed in hybrid."})}\n\n'
            return

        # Stream the winning response word-by-word for smooth UX
        response = result["response"]
        words = response.split(" ")
        chunk_size = 3  # Stream 3 words at a time for natural flow

        for i in range(0, len(words), chunk_size):
            token = " ".join(words[i:i + chunk_size])
            if i > 0:
                token = " " + token
            full_reply.append(token)
            yield f'data: {json.dumps({"token": token})}\n\n'

        # Append model tag
        source = result["source"]
        if source in ("groq", "merged"):
            model = settings["hybrid_groq_model"] or settings["groq_model"]
        else:
            model = settings["hybrid_ollama_model"] or settings["model"]
        if source == "merged":
            model = f"{settings['hybrid_ollama_model']}+{settings['hybrid_groq_model']}"
        full_reply.append(f"__MODEL__{model}")

    def _stream_groq(self, messages, full_reply):
        """Stream from Groq API."""
        settings = get_settings()
        log.info("Groq stream: model=%s", settings["groq_model"])
        with self._call_groq(messages, stream=True) as r:
            if r.status_code != 200:
                log.error("Groq stream ERROR: %d", r.status_code)
                yield f'data: {json.dumps({"error": f"Groq API error ({r.status_code})"})}\n\n'
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
                        yield f'data: {json.dumps({"token": token})}\n\n'
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
                                yield f'data: {json.dumps({"token": token})}\n\n'
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
            log.info("Ollama Cloud failed → Groq stream fallback")
            full_reply.clear()
            try:
                with self._call_groq(groq_messages, stream=True) as r:
                    if r.status_code == 200:
                        yield from self._parse_groq_stream(r, full_reply)
                        settings = get_settings()
                        full_reply.append(f"__MODEL__{settings['groq_model']}")
                    else:
                        log.warning("Groq fallback stream error: %d", r.status_code)
            except Exception as exc2:
                log.warning("Groq fallback stream exception: %s", exc2)
        elif not full_reply:
            yield f'data: {json.dumps({"error": "Ollama Cloud failed, no Groq fallback."})}\n\n'

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
                    yield f'data: {json.dumps({"token": token})}\n\n'
            except (json.JSONDecodeError, IndexError, KeyError):
                continue
