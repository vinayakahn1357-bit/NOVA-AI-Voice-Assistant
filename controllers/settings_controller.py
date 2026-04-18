"""
controllers/settings_controller.py - Settings Management for NOVA V2
Handles GET/POST settings and model listing with strict RBAC enforcement.
Admin users see full config (keys masked); normal users see ONLY safe fields.
API keys are NEVER sent to the frontend — only has_*_key booleans.
"""

import requests
from requests.adapters import HTTPAdapter

from config import get_settings, NOVA_LIVE_MODE, NOVA_ENV, DEFAULT_SYSTEM_PROMPT, save_settings_to_disk
from utils.logger import get_logger

log = get_logger("settings")


# --- Sensitive fields that only admins can view/edit ---
_ADMIN_ONLY_VIEW = {
    "provider",
}

_ADMIN_ONLY_EDIT = {
    "groq_api_key", "nvidia_api_key",
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
            "temperature":          settings["temperature"],
            "top_p":                settings["top_p"],
            "num_predict":          settings["num_predict"],
            "system_prompt":        settings["system_prompt"],
            # Provider config (admin only)
            "provider":             settings["provider"],
            # API key presence (NEVER actual values)
            "has_groq_key":         bool(settings["groq_api_key"]),
            "has_nvidia_key":       bool(settings["nvidia_api_key"]),
            # Model names
            "groq_model":           settings["groq_model"],
            "nvidia_model":         settings["nvidia_model"],
            # System
            "live_mode":            NOVA_LIVE_MODE,
            "role":                 "admin",
        }

    # Normal user: NO API keys, NO URLs, NO provider internals
    return {
        "temperature":          settings["temperature"],
        "top_p":                settings["top_p"],
        "num_predict":          settings["num_predict"],
        "system_prompt":        settings["system_prompt"],
        "groq_model":           settings["groq_model"],
        "nvidia_model":         settings["nvidia_model"],
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
    if "temperature" in data:
        settings["temperature"] = max(0.0, min(2.0, float(data["temperature"])))
    if "top_p" in data:
        settings["top_p"] = max(0.0, min(1.0, float(data["top_p"])))
    if "num_predict" in data:
        settings["num_predict"] = max(64, min(4096, int(data["num_predict"])))
    if "system_prompt" in data:
        sp = str(data["system_prompt"]).strip()[:2000]  # max 2000 chars
        settings["system_prompt"] = sp if sp else DEFAULT_SYSTEM_PROMPT
    if "groq_model" in data:
        settings["groq_model"] = str(data["groq_model"])
    if "nvidia_model" in data:
        settings["nvidia_model"] = str(data["nvidia_model"])

    # --- Admin-only fields ---

    if "provider" in data:
        requested = str(data["provider"])
        allowed = ("groq", "nvidia", "balanced")
        if requested in allowed:
            settings["provider"] = requested
        else:
            log.info("Rejected invalid provider '%s'", requested)

    if "groq_api_key" in data:
        if role == "admin":
            key = str(data["groq_api_key"]).strip()
            if key and not key.startswith("****"):
                settings["groq_api_key"] = key
                log.info("Admin updated groq_api_key")
        else:
            blocked_fields.append("groq_api_key")
            log.warning("RBAC: non-admin tried to change groq_api_key")

    if "nvidia_api_key" in data:
        if role == "admin":
            key = str(data["nvidia_api_key"]).strip()
            if key and not key.startswith("****"):
                settings["nvidia_api_key"] = key
                log.info("Admin updated nvidia_api_key")
        else:
            blocked_fields.append("nvidia_api_key")
            log.warning("RBAC: non-admin tried to change nvidia_api_key")

    log.info("Settings updated: provider=%s role=%s blocked=%s",
             settings["provider"], role, blocked_fields or "none")

    # Persist non-sensitive settings to disk
    save_settings_to_disk()

    result = {
        "status": "ok",
        "role": role,
        "settings": {
            "provider":             settings["provider"],
            "groq_model":           settings["groq_model"],
            "nvidia_model":         settings["nvidia_model"],
        }
    }

    if blocked_fields:
        result["blocked_fields"] = blocked_fields
        result["message"] = "Admin access required for: " + ", ".join(blocked_fields)

    return result


# Pooled HTTP session for model listing (Fix #11)
_model_session = requests.Session()
_model_session.mount("http://", HTTPAdapter(pool_connections=2, pool_maxsize=4))
_model_session.mount("https://", HTTPAdapter(pool_connections=2, pool_maxsize=4))


def list_models():
    """List available models for the current provider."""
    settings = get_settings()
    provider = settings["provider"]

    groq_models = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
    ]

    nvidia_models = [
        # Verified working on this account
        "meta/llama-3.1-70b-instruct",
        "mistralai/mixtral-8x22b-instruct-v0.1",
        "google/gemma-3-27b-it",
    ]

    if provider == "nvidia":
        return {"models": nvidia_models, "provider": "nvidia"}

    if provider == "groq":
        return {"models": groq_models, "provider": "groq"}

    # Default: return both
    return {
        "models": groq_models,
        "groq_models": groq_models,
        "nvidia_models": nvidia_models,
        "provider": provider,
    }
