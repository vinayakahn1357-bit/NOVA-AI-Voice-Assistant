"""
services/ai_service.py — LLM Provider Dispatch for NOVA
Handles all communication with AI providers: Ollama Local, Ollama Cloud, Groq, Hybrid.
"""

import json
import requests

from config import (
    get_settings, NOVA_ENV, NOVA_LIVE_MODE, OLLAMA_URL,
    GROQ_API_URL, API_TIMEOUT, LOCAL_TIMEOUT,
)
from utils.logger import get_logger
from utils.errors import NovaProviderError

log = get_logger("ai_service")

# ─── Hybrid Query Classifier ──────────────────────────────────────────────────
_COMPLEX_KEYWORDS = (
    "def ", "function ", "class ", "import ", "```", "code", "debug", "error",
    "bug", "implement", "algorithm", "write a ", "create a ", "build a ",
    "program", "script", "api", "database", "sql", "regex",
    "math", "calculate", "solve", "equation", "prove", "integral",
    "derivative", "matrix", "statistics", "probability",
    "analyze", "analyse", "summarize", "summarise", "essay", "detailed",
    "comprehensive", "explain why", "compare", "difference between",
    "pros and cons", "advantages", "disadvantages", "research",
    "translate", "rewrite", "refactor",
)
_COMPLEX_WORD_THRESHOLD = 60
_SIMPLE_WORD_THRESHOLD = 30


class AIService:
    """Encapsulates all LLM provider communication and routing logic."""

    def __init__(self, prompt_builder):
        self._prompt_builder = prompt_builder

    # ─── Provider Checks ──────────────────────────────────────────────────

    @staticmethod
    def _ollama_cloud_configured() -> bool:
        return bool(get_settings().get("ollama_api_key", "").strip())

    @staticmethod
    def _groq_configured() -> bool:
        return bool(get_settings().get("groq_api_key", "").strip())

    # ─── Low-Level Provider Calls ─────────────────────────────────────────

    @staticmethod
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
    def _call_ollama_cloud(full_prompt: str, stream: bool = False):
        """Call Ollama Cloud endpoint."""
        settings = get_settings()
        cloud_url = settings["ollama_cloud_url"] or "https://api.ollama.com/api/generate"
        model = (
            settings["hybrid_ollama_model"]
            if settings["provider"] == "hybrid"
            else settings["model"]
        ) or "mistral"
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
    def _call_groq(messages: list, stream: bool = False, model_override: str = ""):
        """Call Groq API."""
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

    # ─── Hybrid Classification ────────────────────────────────────────────

    @staticmethod
    def _classify_query(message: str) -> str:
        """Returns 'groq' for complex queries, 'ollama_cloud' for simple ones."""
        lower = message.lower()
        word_count = len(message.split())
        if word_count > _COMPLEX_WORD_THRESHOLD:
            return "groq"
        if any(kw in lower for kw in _COMPLEX_KEYWORDS):
            return "groq"
        return "ollama_cloud"

    def _hybrid_pick_sub(self, message: str) -> str:
        """Pick sub-provider for hybrid mode."""
        ollama_ok = self._ollama_cloud_configured()
        groq_ok = self._groq_configured()
        if not ollama_ok and not groq_ok:
            return "none"
        if not ollama_ok:
            return "groq"
        if not groq_ok:
            return "ollama_cloud"
        return self._classify_query(message)

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
        Generate a full AI response.
        Returns: (ai_response: str, active_model: str, provider: str)
        Raises NovaProviderError if all providers fail.
        """
        settings = get_settings()
        provider = self._resolve_provider(settings["provider"])
        full_prompt = self._prompt_builder.build_ollama_prompt(history)
        groq_messages = self._prompt_builder.build_chat_messages(history)

        ai_response = ""
        active_model = settings["model"]

        if provider == "hybrid":
            ai_response, active_model = self._generate_hybrid(
                user_message, full_prompt, history, groq_messages
            )
        elif provider == "groq" and self._groq_configured():
            r = self._call_groq(groq_messages, model_override=settings["groq_model"])
            if r.status_code != 200:
                err_detail = r.json().get("error", {}).get("message", "Groq API error")
                raise NovaProviderError(err_detail, code="GROQ_ERROR")
            ai_response = r.json()["choices"][0]["message"]["content"].strip()
            active_model = settings["groq_model"]
        elif provider == "ollama_cloud" and self._ollama_cloud_configured():
            r = self._call_ollama_cloud(full_prompt)
            if r.status_code != 200:
                raise NovaProviderError(f"Ollama Cloud error: {r.text[:200]}", code="OLLAMA_CLOUD_ERROR")
            ai_response = r.json().get("response", "").strip()
            active_model = settings["model"]
        else:
            ai_response, active_model = self._generate_local_with_fallback(
                full_prompt, history, groq_messages
            )

        if not ai_response:
            ai_response = "I'm not sure how to respond to that. Could you rephrase?"

        return ai_response, active_model, provider

    def _generate_hybrid(self, user_message, full_prompt, history, groq_messages) -> tuple:
        """Handle hybrid provider routing."""
        settings = get_settings()
        sub = self._hybrid_pick_sub(user_message)
        log.info("Hybrid → %s (%d words)", sub, len(user_message.split()))

        if sub == "none":
            if not NOVA_LIVE_MODE:
                sub = "_ollama_local"
            else:
                raise NovaProviderError(
                    "No cloud AI provider configured. Set a Groq API key or Ollama Cloud URL."
                )

        ai_response = ""
        actual_sub = sub

        if sub == "groq":
            try:
                r = self._call_groq(groq_messages)
                if r.status_code == 200:
                    ai_response = r.json()["choices"][0]["message"]["content"].strip()
                else:
                    log.warning("Hybrid Groq failed (%d), trying Ollama Cloud fallback", r.status_code)
            except Exception as exc:
                log.warning("Hybrid Groq exception: %s, trying Ollama Cloud fallback", exc)
            if not ai_response and self._ollama_cloud_configured():
                r = self._call_ollama_cloud(full_prompt)
                if r.status_code == 200:
                    ai_response = r.json().get("response", "").strip()
                    actual_sub = "ollama_cloud"

        elif sub == "ollama_cloud":
            try:
                r = self._call_ollama_cloud(full_prompt)
                if r.status_code == 200:
                    ai_response = r.json().get("response", "").strip()
                else:
                    log.warning("Hybrid Ollama Cloud failed (%d), trying Groq fallback", r.status_code)
            except Exception as exc:
                log.warning("Hybrid Ollama Cloud exception: %s, trying Groq fallback", exc)
            if not ai_response and self._groq_configured():
                r = self._call_groq(groq_messages)
                if r.status_code == 200:
                    ai_response = r.json()["choices"][0]["message"]["content"].strip()
                    actual_sub = "groq"

        else:
            try:
                r = self._call_ollama_local(full_prompt)
                if r.status_code == 200:
                    ai_response = r.json().get("response", "").strip()
            except Exception as exc:
                log.warning("Hybrid local Ollama failed: %s", exc)
            if not ai_response and self._groq_configured():
                log.info("Fallback to Groq (local Ollama failed)")
                try:
                    r = self._call_groq(groq_messages)
                    if r.status_code == 200:
                        ai_response = r.json()["choices"][0]["message"]["content"].strip()
                        actual_sub = "groq"
                except Exception as exc2:
                    log.warning("Groq fallback also failed: %s", exc2)

        active_model = self._resolve_hybrid_model(actual_sub)
        return ai_response, active_model

    def _generate_local_with_fallback(self, full_prompt, history, groq_messages) -> tuple:
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
                r = self._call_groq(groq_messages)
                if r.status_code == 200:
                    ai_response = r.json()["choices"][0]["message"]["content"].strip()
                    active_model = settings["groq_model"]
            except Exception as exc2:
                log.warning("Groq fallback also failed: %s", exc2)

        if not ai_response:
            raise NovaProviderError("AI engine error — all providers failed")

        return ai_response, active_model

    @staticmethod
    def _resolve_hybrid_model(sub: str) -> str:
        settings = get_settings()
        if sub == "groq":
            return settings["hybrid_groq_model"] or settings["groq_model"]
        elif sub == "ollama_cloud":
            return settings["hybrid_ollama_model"] or settings["model"]
        return settings["model"]

    # ─── Streaming Generation ─────────────────────────────────────────────

    def generate_stream(self, history: list, user_message: str):
        """
        Generator that yields SSE-formatted tokens.
        Yields: 'data: {"token": "..."}\n\n' for each token
                'data: {"done": true, ...}\n\n' when complete
                'data: {"error": "..."}\n\n' on failure
        Returns the full reply text.
        """
        settings = get_settings()
        provider = self._resolve_provider(settings["provider"])
        full_prompt = self._prompt_builder.build_ollama_prompt(history)
        groq_messages = self._prompt_builder.build_chat_messages(history)

        full_reply = []
        active_model = settings["model"]

        use_groq = provider == "groq" and self._groq_configured()
        use_ollama_cloud = provider == "ollama_cloud"
        use_hybrid = provider == "hybrid"

        if use_groq:
            active_model = settings["groq_model"]

        try:
            if use_hybrid:
                yield from self._stream_hybrid(
                    user_message, full_prompt, history, groq_messages, full_reply
                )
                active_model = full_reply[-1] if full_reply and full_reply[-1].startswith("__MODEL__") else settings["model"]
                # Extract model tag if present
                if full_reply and full_reply[-1].startswith("__MODEL__"):
                    active_model = full_reply.pop()[9:]

            elif use_groq:
                yield from self._stream_groq(groq_messages, full_reply)

            elif use_ollama_cloud:
                yield from self._stream_ollama_cloud(
                    full_prompt, history, groq_messages, full_reply
                )
                if full_reply and full_reply[-1].startswith("__MODEL__"):
                    active_model = full_reply.pop()[9:]

            else:
                if NOVA_ENV != "local":
                    yield f'data: {json.dumps({"error": "No AI provider available. Configure Groq or Ollama Cloud."})}\n\n'
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
            yield f'data: {json.dumps({"error": "AI returned an empty response. Please try again."})}\n\n'

        yield f'data: {json.dumps({"done": True, "session_id": "", "model": active_model})}\n\n'

    # ─── Streaming Helpers ────────────────────────────────────────────────

    def _stream_groq(self, messages, full_reply):
        """Stream from Groq API."""
        settings = get_settings()
        log.info("Groq stream: model=%s", settings["groq_model"])
        with self._call_groq(messages, stream=True) as r:
            if r.status_code != 200:
                err_body = r.text[:300]
                log.error("Groq stream ERROR: %s", err_body)
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

    def _stream_ollama_cloud(self, full_prompt, history, groq_messages, full_reply):
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
                                log.warning("Ollama Cloud model error: %s", chunk["error"])
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
            yield f'data: {json.dumps({"error": "Ollama Cloud failed and no Groq fallback configured."})}\n\n'

    def _stream_hybrid(self, user_message, full_prompt, history, groq_messages, full_reply):
        """Stream in hybrid mode with cross-fallback."""
        settings = get_settings()
        sub = self._hybrid_pick_sub(user_message)
        log.info("Hybrid stream → %s (%d words)", sub, len(user_message.split()))

        if sub == "none":
            if not NOVA_LIVE_MODE:
                sub = "_ollama_local"
            else:
                yield f'data: {json.dumps({"error": "No cloud AI provider configured."})}\n\n'
                return

        used_provider = None

        if sub == "groq":
            try:
                with self._call_groq(groq_messages, stream=True) as r:
                    if r.status_code == 200:
                        yield from self._parse_groq_stream(r, full_reply)
                        used_provider = "groq"
            except Exception as exc:
                log.warning("Hybrid stream Groq failed: %s, trying Ollama Cloud", exc)
                full_reply.clear()

        elif sub == "ollama_cloud":
            try:
                with self._call_ollama_cloud(full_prompt, stream=True) as r:
                    if r.status_code == 200:
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
                        used_provider = "ollama_cloud"
                    else:
                        log.warning("Hybrid Ollama Cloud error %d, trying Groq", r.status_code)
            except Exception as exc:
                log.warning("Hybrid stream Ollama Cloud failed: %s, trying Groq", exc)
                full_reply.clear()

        # Fallback if primary failed
        if not used_provider:
            if sub == "groq" and self._ollama_cloud_configured():
                try:
                    with self._call_ollama_cloud(full_prompt, stream=True) as r:
                        if r.status_code == 200:
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
                            used_provider = "ollama_cloud"
                except Exception as exc:
                    log.warning("Hybrid fallback Ollama Cloud failed: %s", exc)

            elif sub == "ollama_cloud" and self._groq_configured():
                try:
                    with self._call_groq(groq_messages, stream=True) as r:
                        if r.status_code == 200:
                            yield from self._parse_groq_stream(r, full_reply)
                            used_provider = "groq"
                        else:
                            log.warning("Hybrid fallback Groq error: %d", r.status_code)
                except Exception as exc:
                    log.warning("Hybrid fallback Groq exception: %s", exc)

            elif NOVA_ENV == "local":
                try:
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
                        used_provider = "_ollama_local"
                except Exception as exc:
                    log.warning("Hybrid fallback local Ollama exception: %s", exc)

        active_model = self._resolve_hybrid_model(used_provider or sub)
        full_reply.append(f"__MODEL__{active_model}")

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
