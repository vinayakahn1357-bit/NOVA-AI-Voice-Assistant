"""
controllers/settings_controller.py - Settings Management for NOVA
Handles GET/POST settings and model listing with RBAC enforcement.
Admin users see full config; normal users see filtered, safe config.
"""

import requests

from config import get_settings, NOVA_LIVE_MODE, NOVA_ENV, DEFAULT_SYSTEM_PROMPT
from utils.logger import get_logger

log = get_logger("settings")


def _mask(key):
    """Mask an API key, showing only the last 4 chars."""
    return ("****" + key[-4:]) if key else ""


# --- Fields visible to each role ---

# Sensitive fields that only admins can see/edit
_ADMIN_ONLY_FIELDS = {
    "groq_api_key", "ollama_api_key",
    "ollama_cloud_url",
}

# Fields that normal users can update
_USER_EDITABLE_FIELDS = {
    "model", "temperature", "top_p", "num_predict",
    "system_prompt", "groq_model",
    "hybrid_ollama_model", "hybrid_groq_model",
}


def get_current_settings(role="user"):
    """
    Return settings filtered by role.
    Admin: full config with masked keys
    User: safe subset (no API keys, no URLs)
    """
    settings = get_settings()

    if role == "admin":
        return {
            "model":                settings["model"],
            "temperature":          settings["temperature"],
            "top_p":                settings["top_p"],
            "num_predict":          settings["num_predict"],
            "system_prompt":        settings["system_prompt"],
            "provider":             settings["provider"],
            "groq_api_key":         _mask(settings["groq_api_key"]),
            "groq_model":           settings["groq_model"],
            "ollama_api_key":       _mask(settings["ollama_api_key"]),
            "ollama_cloud_url":     settings["ollama_cloud_url"],
            "hybrid_ollama_model":  settings["hybrid_ollama_model"],
            "hybrid_groq_model":    settings["hybrid_groq_model"],
            "live_mode":            NOVA_LIVE_MODE,
            "role":                 "admin",
        }

    # Normal user - no sensitive data
    return {
        "model":                settings["model"],
        "temperature":          settings["temperature"],
        "top_p":                settings["top_p"],
        "num_predict":          settings["num_predict"],
        "system_prompt":        settings["system_prompt"],
        "provider":             settings["provider"],
        "groq_model":           settings["groq_model"],
        "hybrid_ollama_model":  settings["hybrid_ollama_model"],
        "hybrid_groq_model":    settings["hybrid_groq_model"],
        "live_mode":            NOVA_LIVE_MODE,
        "role":                 "user",
    }


def update_settings(data, role="user"):
    """
    Update settings from a request payload with RBAC enforcement.
    Admin: can update everything
    User: can only update non-sensitive fields
    """
    settings = get_settings()
    blocked_fields = []

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

    # Provider switching
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
                log.info("Rejected provider 'ollama' - live mode is active")
        else:
            blocked_fields.append("provider")

    # API keys - admin only
    if "groq_api_key" in data:
        if role == "admin":
            key = str(data["groq_api_key"]).strip()
            if key and not key.startswith("****"):
                settings["groq_api_key"] = key
        else:
            blocked_fields.append("groq_api_key")

    if "ollama_api_key" in data:
        if role == "admin":
            key = str(data["ollama_api_key"]).strip()
            if key and not key.startswith("****"):
                settings["ollama_api_key"] = key
        else:
            blocked_fields.append("ollama_api_key")

    # URLs - admin only
    if "ollama_cloud_url" in data:
        if role == "admin":
            url = str(data["ollama_cloud_url"]).strip()
            if url:
                settings["ollama_cloud_url"] = url
        else:
            blocked_fields.append("ollama_cloud_url")

    # Model names - anyone can change
    if "groq_model" in data:
        settings["groq_model"] = str(data["groq_model"])
    if "hybrid_ollama_model" in data:
        settings["hybrid_ollama_model"] = str(data["hybrid_ollama_model"])
    if "hybrid_groq_model" in data:
        settings["hybrid_groq_model"] = str(data["hybrid_groq_model"])

    log.info("Settings updated: provider=%s, model=%s (role=%s)",
             settings["provider"], settings["model"], role)

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
        result["message"] = "Some fields require admin access: " + ", ".join(blocked_fields)
        log.warning("Non-admin tried to update admin-only fields: %s", blocked_fields)

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
