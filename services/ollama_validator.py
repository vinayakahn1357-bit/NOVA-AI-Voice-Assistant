"""
services/ollama_validator.py — Ollama Model Discovery & Validation for NOVA
Queries /api/tags to discover available models, validates requested model,
and provides intelligent fallback with full transparency logging.
"""

import re
import time
import requests
from threading import Lock

from config import get_settings, NOVA_ENV, NOVA_LIVE_MODE, API_TIMEOUT
from utils.logger import get_logger

log = get_logger("ollama_validator")

# Cache discovered models for 60s to avoid hitting /api/tags on every request
_MODEL_CACHE = {"models": [], "timestamp": 0, "base_url": ""}
_CACHE_TTL = 60
_CACHE_LOCK = Lock()


class OllamaValidator:
    """
    Validates and resolves Ollama model names against the actual server.
    Provides:
    - Model discovery via /api/tags
    - Fuzzy model matching (e.g. "mistral" -> "mistral:latest")
    - Transparent fallback with full logging
    - Cached model list (60s TTL)
    """

    @staticmethod
    def _get_base_url() -> str:
        """
        Derive the base URL (without /api/generate) from the configured URL.
        Handles both /api/generate and /api/chat endpoints.
        """
        settings = get_settings()
        cloud_url = settings.get("ollama_cloud_url", "") or ""

        # Strip /api/generate or /api/chat suffix to get base URL
        base = cloud_url.rstrip("/")
        for suffix in ("/api/generate", "/api/chat", "/api"):
            if base.endswith(suffix):
                base = base[:-len(suffix)]
                break

        # For local mode
        if not base and NOVA_ENV == "local":
            base = "http://localhost:11434"

        return base.rstrip("/")

    @staticmethod
    def _get_auth_headers() -> dict:
        """Build authentication headers."""
        settings = get_settings()
        headers = {"Content-Type": "application/json"}
        key = settings.get("ollama_api_key", "")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def discover_models(self, force_refresh: bool = False) -> list:
        """
        Query the Ollama server for available models.
        Returns a list of model name strings (e.g. ["gemma3:12b", "mistral:latest"]).
        Results are cached for 60 seconds.
        """
        base_url = self._get_base_url()
        if not base_url:
            log.warning("No Ollama base URL configured")
            return []

        # Check cache
        with _CACHE_LOCK:
            now = time.time()
            if (not force_refresh
                    and _MODEL_CACHE["models"]
                    and _MODEL_CACHE["base_url"] == base_url
                    and now - _MODEL_CACHE["timestamp"] < _CACHE_TTL):
                return list(_MODEL_CACHE["models"])

        # Query /api/tags
        tags_url = f"{base_url}/api/tags"
        log.info("Discovering models: GET %s", tags_url)

        try:
            r = requests.get(
                tags_url,
                headers=self._get_auth_headers(),
                timeout=min(API_TIMEOUT, 15),
            )

            if r.status_code != 200:
                log.error("Model discovery failed: %d %s", r.status_code, r.text[:200])
                return []

            data = r.json()
            models_raw = data.get("models", [])
            model_names = []

            for m in models_raw:
                name = m.get("name") or m.get("model", "")
                if name:
                    model_names.append(name)

            log.info("Discovered %d models: %s", len(model_names), model_names[:10])

            # Update cache
            with _CACHE_LOCK:
                _MODEL_CACHE["models"] = model_names
                _MODEL_CACHE["timestamp"] = time.time()
                _MODEL_CACHE["base_url"] = base_url

            return model_names

        except requests.exceptions.ConnectionError as e:
            log.error("Cannot reach Ollama server at %s: %s", tags_url, e)
            return []
        except requests.exceptions.Timeout:
            log.error("Ollama server timed out: %s", tags_url)
            return []
        except Exception as e:
            log.error("Model discovery error: %s", e)
            return []

    def resolve_model(self, requested_model: str) -> dict:
        """
        Validate and resolve a model name against available models.

        Returns: {
            "requested": str,    # what was asked for
            "resolved": str,     # what will actually be used
            "is_valid": bool,    # whether the model exists
            "match_type": str,   # "exact"|"base_name"|"fallback"|"not_found"
            "available": list,   # all available models
            "message": str,      # human-readable explanation
        }
        """
        available = self.discover_models()

        result = {
            "requested": requested_model,
            "resolved": requested_model,
            "is_valid": False,
            "match_type": "not_found",
            "available": available,
            "message": "",
        }

        if not available:
            result["message"] = "Could not discover models (server unreachable or no models)"
            log.warning("Model validation: no models available from server")
            return result

        # 1. Exact match
        if requested_model in available:
            result["is_valid"] = True
            result["match_type"] = "exact"
            result["message"] = f"Model '{requested_model}' found (exact match)"
            log.info("Model resolved: '%s' -> EXACT match", requested_model)
            return result

        # 2. Case-insensitive exact match
        lower_map = {m.lower(): m for m in available}
        if requested_model.lower() in lower_map:
            resolved = lower_map[requested_model.lower()]
            result["resolved"] = resolved
            result["is_valid"] = True
            result["match_type"] = "exact"
            result["message"] = f"Model '{requested_model}' -> '{resolved}' (case-insensitive)"
            log.info("Model resolved: '%s' -> '%s' (case-insensitive)", requested_model, resolved)
            return result

        # 3. Base name match (e.g. "mistral" matches "mistral:latest" or "mistral:7b")
        base_requested = requested_model.split(":")[0].lower()
        base_matches = [m for m in available if m.lower().startswith(base_requested + ":")]

        if base_matches:
            # Prefer :latest tag, otherwise first match
            best = next((m for m in base_matches if ":latest" in m), base_matches[0])
            result["resolved"] = best
            result["is_valid"] = True
            result["match_type"] = "base_name"
            result["message"] = (
                f"Model '{requested_model}' not found exactly, "
                f"resolved to '{best}' (base name match from {base_matches})"
            )
            log.warning(
                "Model resolved: '%s' -> '%s' (base name match, options: %s)",
                requested_model, best, base_matches,
            )
            return result

        # 4. Fallback to first available model
        fallback = available[0]
        result["resolved"] = fallback
        result["is_valid"] = True
        result["match_type"] = "fallback"
        result["message"] = (
            f"Model '{requested_model}' NOT FOUND. "
            f"Falling back to first available: '{fallback}'. "
            f"Available models: {available}"
        )
        log.warning(
            "Model '%s' NOT FOUND on server! Falling back to '%s'. Available: %s",
            requested_model, fallback, available,
        )
        return result

    def test_connection(self) -> dict:
        """
        Full diagnostic test of Ollama connectivity and model availability.
        Returns structured debug information.
        """
        settings = get_settings()
        base_url = self._get_base_url()
        configured_model = settings.get("model", "")

        result = {
            "status": "failed",
            "base_url": base_url,
            "configured_model": configured_model,
            "resolved_model": "",
            "is_valid": False,
            "available_models": [],
            "api_key_set": bool(settings.get("ollama_api_key", "")),
            "provider": settings.get("provider", ""),
            "error": None,
        }

        if not base_url:
            result["error"] = "No Ollama base URL configured"
            return result

        # Test connectivity
        try:
            r = requests.get(
                f"{base_url}/api/tags",
                headers=self._get_auth_headers(),
                timeout=10,
            )

            if r.status_code == 401:
                result["error"] = "Authentication failed (401). Check OLLAMA_API_KEY."
                return result

            if r.status_code != 200:
                result["error"] = f"Server returned {r.status_code}: {r.text[:200]}"
                return result

        except requests.exceptions.ConnectionError:
            result["error"] = f"Cannot connect to {base_url}"
            return result
        except requests.exceptions.Timeout:
            result["error"] = f"Connection timed out to {base_url}"
            return result

        # Discover models
        available = self.discover_models(force_refresh=True)
        result["available_models"] = available

        if not available:
            result["error"] = "Connected but no models available on server"
            return result

        # Validate configured model
        resolution = self.resolve_model(configured_model)
        result["resolved_model"] = resolution["resolved"]
        result["is_valid"] = resolution["is_valid"]
        result["match_type"] = resolution["match_type"]
        result["resolution_message"] = resolution["message"]
        result["status"] = "connected"

        return result

    def clear_cache(self):
        """Force clear the model cache."""
        with _CACHE_LOCK:
            _MODEL_CACHE["models"] = []
            _MODEL_CACHE["timestamp"] = 0
            _MODEL_CACHE["base_url"] = ""
        log.info("Model cache cleared")
