"""
controllers/settings_controller.py - Settings Management for NOVA
Handles GET/POST settings and model listing with strict RBAC enforcement.
Admin users see full config (keys masked); normal users see ONLY safe fields.
API keys are NEVER sent to the frontend — only has_*_key booleans.
"""

import requests

from config import get_settings, NOVA_LIVE_MODE, NOVA_ENV, DEFAULT_SYSTEM_PROMPT
from utils.logger import get_logger

log = get_logger("settings")


# --- Sensitive fields that only admins can view/edit ---
_ADMIN_ONLY_VIEW = {
    "ollama_cloud_url", "provider",
}

_ADMIN_ONLY_EDIT = {
    "groq_api_key", "ollama_api_key",
    "ollama_cloud_url", "provider",
}


def get_current_settings(role="user"):
    """
    Return settings filtered by role.
    Admin: full config with presence flags for keys (never actual keys)
    User: safe subset only
    """
    settings = get_settings()

    log.info("Settings GET: role=%s", role)

    if role == "admin":
        return {
            # Model config
            "model":                settings["model"],
            "temperature":          settings["temperature"],
            "top_p":                settings["top_p"],
            "num_predict":          settings["num_predict"],
            "system_prompt":        settings["system_prompt"],
            # Provider config (admin only)
            "provider":             settings["provider"],
            "ollama_cloud_url":     settings["ollama_cloud_url"],
            # API key presence (NEVER actual values)
            "has_groq_key":         bool(settings["groq_api_key"]),
            "has_ollama_key":       bool(settings["ollama_api_key"]),
            # Model names
            "groq_model":           settings["groq_model"],
            "hybrid_ollama_model":  settings["hybrid_ollama_model"],
            "hybrid_groq_model":    settings["hybrid_groq_model"],
            # System
            "live_mode":            NOVA_LIVE_MODE,
            "role":                 "admin",
        }

    # Normal user: NO API keys, NO URLs, NO provider internals
    return {
        "model":                settings["model"],
        "temperature":          settings["temperature"],
        "top_p":                settings["top_p"],
        "num_predict":          settings["num_predict"],
        "system_prompt":        settings["system_prompt"],
        "groq_model":           settings["groq_model"],
        "hybrid_ollama_model":  settings["hybrid_ollama_model"],
        "hybrid_groq_model":    settings["hybrid_groq_model"],
        "live_mode":            NOVA_LIVE_MODE,
        "role":                 "user",
    }


def update_settings(data, role="user"):
    """
    Update settings from a request payload with strict RBAC enforcement.
    Admin: can update everything
    User: can only update non-sensitive fields (model, temperature, etc.)
    """
    settings = get_settings()
    blocked_fields = []

    log.info("Settings POST: role=%s fields=%s", role, list(data.keys()))

    # --- Fields anyone can edit ---
    if "model" in data:
        settings["model"] = str(data["model"])
    if "temperature" in data:
        settings["temperature"] = max(0.0, min(2.0, float(data["temperature"])))
    if "top_p" in data:
        settings["top_p"] = max(0.0, min(1.0, float(data["top_p"])))
    if "num_predict" in data:
        settings["num_predict"] = max(64, min(4096, int(data["num_predict"])))
    if "system_prompt" in data:
        sp = data["system_prompt"].strip()
        settings["system_prompt"] = sp if sp else DEFAULT_SYSTEM_PROMPT
    if "groq_model" in data:
        settings["groq_model"] = str(data["groq_model"])
    if "hybrid_ollama_model" in data:
        settings["hybrid_ollama_model"] = str(data["hybrid_ollama_model"])
    if "hybrid_groq_model" in data:
        settings["hybrid_groq_model"] = str(data["hybrid_groq_model"])

    # --- Admin-only fields ---

    if "provider" in data:
        if role == "admin":
            requested = data["provider"]
            allowed = (
                ("ollama_cloud", "groq", "hybrid")
                if NOVA_LIVE_MODE
                else ("ollama", "ollama_cloud", "groq", "hybrid")
            )
            if requested in allowed:
                settings["provider"] = requested
            elif NOVA_LIVE_MODE and requested == "ollama":
                log.info("Rejected provider 'ollama' in live mode")
        else:
            blocked_fields.append("provider")
            log.warning("RBAC: non-admin tried to change provider")

    if "groq_api_key" in data:
        if role == "admin":
            key = str(data["groq_api_key"]).strip()
            if key and not key.startswith("****"):
                settings["groq_api_key"] = key
                log.info("Admin updated groq_api_key")
        else:
            blocked_fields.append("groq_api_key")
            log.warning("RBAC: non-admin tried to change groq_api_key")

    if "ollama_api_key" in data:
        if role == "admin":
            key = str(data["ollama_api_key"]).strip()
            if key and not key.startswith("****"):
                settings["ollama_api_key"] = key
                log.info("Admin updated ollama_api_key")
        else:
            blocked_fields.append("ollama_api_key")
            log.warning("RBAC: non-admin tried to change ollama_api_key")

    if "ollama_cloud_url" in data:
        if role == "admin":
            url = str(data["ollama_cloud_url"]).strip()
            if url:
                settings["ollama_cloud_url"] = url
                log.info("Admin updated ollama_cloud_url")
        else:
            blocked_fields.append("ollama_cloud_url")
            log.warning("RBAC: non-admin tried to change ollama_cloud_url")

    log.info("Settings updated: provider=%s model=%s role=%s blocked=%s",
             settings["provider"], settings["model"], role, blocked_fields or "none")

    result = {
        "status": "ok",
        "role": role,
        "settings": {
            "provider":             settings["provider"],
            "model":                settings["model"],
            "groq_model":           settings["groq_model"],
            "hybrid_ollama_model":  settings["hybrid_ollama_model"],
            "hybrid_groq_model":    settings["hybrid_groq_model"],
        }
    }

    if blocked_fields:
        result["blocked_fields"] = blocked_fields
        result["message"] = "Admin access required for: " + ", ".join(blocked_fields)

    return result


def list_models():
    """List available models for the current provider."""
    settings = get_settings()
    provider = settings["provider"]

    if provider == "groq":
        groq_models = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "llama3-70b-8192",
            "llama3-8b-8192",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ]
        return {"models": groq_models, "provider": "groq"}

    if provider == "ollama_cloud":
        try:
            tags_url = settings["ollama_cloud_url"].replace("/api/generate", "/api/tags")
            cloud_headers = {"Authorization": "Bearer " + settings["ollama_api_key"]}
            r = requests.get(tags_url, headers=cloud_headers, timeout=8)
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                if models:
                    return {"models": models, "provider": "ollama_cloud"}
        except Exception:
            pass
        return {
            "models": ["mistral", "llama3", "gemma2", "phi3", "deepseek-r1"],
            "provider": "ollama_cloud",
        }

    # Local Ollama
    if NOVA_ENV == "local":
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=5)
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                return {"models": models, "provider": "ollama"}
        except Exception:
            pass

    return {
        "models": ["mistral", "llama3", "gemma2", "phi3", "codellama"],
        "provider": "ollama",
    }
