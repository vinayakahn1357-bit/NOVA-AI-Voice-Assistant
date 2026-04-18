"""
config.py — Centralised NOVA Configuration
All environment variables and settings are managed here.
"""

import os
import json
from dotenv import load_dotenv

load_dotenv()

# ─── Phase 6: Scalable Architecture Config ────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "")          # Empty = SQLite fallback
REDIS_URL = os.getenv("REDIS_URL", "")                # Empty = in-memory fallback
USE_REDIS = bool(REDIS_URL)                            # Convenience flag for feature gating
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "")       # Empty = JWT disabled
JWT_EXPIRY_HOURS = int(os.getenv("JWT_EXPIRY_HOURS", "168"))  # 7 days default
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", "")       # Empty = no encryption (plaintext)
LOG_FORMAT = os.getenv("LOG_FORMAT", "text")            # "json" for production, "text" for local
TASK_QUEUE_WORKERS = int(os.getenv("TASK_QUEUE_WORKERS", "0"))  # 0 = auto-detect

# ─── Environment Detection ────────────────────────────────────────────────────
_live_flag = os.getenv("NOVA_LIVE_MODE", "false").lower() in ("true", "1", "yes")
_vercel_flag = bool(os.getenv("VERCEL") or os.getenv("VERCEL_ENV"))
IS_VERCEL = _vercel_flag
IS_RAILWAY = bool(os.getenv("RAILWAY_ENVIRONMENT"))

NOVA_ENV = os.getenv("ENV", "production" if (_live_flag or _vercel_flag) else "local")
NOVA_LIVE_MODE = (NOVA_ENV == "production")

# ─── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "Frontend")

# ─── Flask ─────────────────────────────────────────────────────────────────────
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "")
if not FLASK_SECRET_KEY:
    import secrets
    FLASK_SECRET_KEY = secrets.token_hex(32)
    import logging
    logging.getLogger("config").warning("FLASK_SECRET_KEY not set — using auto-generated key (sessions won't persist across restarts)")
FLASK_PORT = int(os.getenv("FLASK_PORT", 5000))
FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
SESSION_LIFETIME_SECONDS = 60 * 60 * 24 * 30  # 30 days

# --- Admin Access Control ---
# Comma-separated list of admin emails
ADMIN_EMAILS = [
    e.strip().lower()
    for e in os.getenv("ADMIN_EMAILS", "vinayakahn1357@gmail.com").split(",")
    if e.strip()
]

# ─── Provider API URLs ─────────────────────────────────────────────────────────
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
NVIDIA_API_URL = os.getenv(
    "NVIDIA_BASE_URL",
    "https://integrate.api.nvidia.com/v1/chat/completions",
)

# ─── Default Provider ─────────────────────────────────────────────────────────
_default_provider = os.getenv("NOVA_PROVIDER", "groq")
if _default_provider not in ("groq", "nvidia", "balanced"):
    _default_provider = "groq"

# ─── Default System Prompt (Assistant-Level Personality) ───────────────────────
DEFAULT_SYSTEM_PROMPT = (
    "You are Nova — a senior-level AI assistant with deep expertise across programming, "
    "science, mathematics, writing, and general knowledge. You are warm, confident, and "
    "proactive, like a brilliant colleague who anticipates needs.\n\n"
    "## Core Behaviours\n"
    "- **Be an assistant, not a chatbot.** Anticipate follow-ups, offer next steps, "
    "and provide actionable advice without being asked.\n"
    "- **Remember everything.** Use memory context to personalise: greet by name, "
    "reference past topics, build on previous conversations.\n"
    "- **Be concise but thorough.** Answer fully in the fewest words possible. "
    "No filler phrases, no restating the question.\n"
    "- **Show, don't tell.** For code, give working examples with ```language blocks. "
    "For math, show the steps. For writing, give the draft.\n"
    "- **Be honest.** If unsure, say so. Never hallucinate facts.\n"
    "- **Natural tone.** Speak like a smart human friend — confident, slightly playful, "
    "never robotic. Use lists/headers only when they genuinely help clarity.\n\n"
    "## Identity\n"
    "- You are Nova. Never mention NVIDIA, Groq, LLaMA, or any underlying model.\n"
    "- You were created to be the most helpful AI assistant possible.\n"
    "- Your knowledge is broad and deep. Your responses should reflect expertise.\n"
)

# ─── Nova Settings (mutable at runtime via /settings) ─────────────────────────
NOVA_SETTINGS = {
    "temperature":          float(os.getenv("NOVA_TEMPERATURE", "0.75")),
    "top_p":                float(os.getenv("NOVA_TOP_P", "0.9")),
    "num_predict":          int(os.getenv("NOVA_MAX_TOKENS", "1024")),
    "system_prompt":        DEFAULT_SYSTEM_PROMPT,
    "provider":             _default_provider,
    "groq_api_key":         os.getenv("GROQ_API_KEY", ""),
    "groq_model":           os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
    "nvidia_api_key":       os.getenv("NVIDIA_API_KEY", ""),
    "nvidia_model":         os.getenv("NVIDIA_MODEL", "meta/llama-3.1-70b-instruct"),
}

# ─── Valid model lists (to reject decommissioned models on load) ───────────────
_VALID_GROQ_MODELS = {
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
}
_VALID_NVIDIA_MODELS = {
    "meta/llama-3.1-70b-instruct",
    "mistralai/mixtral-8x22b-instruct-v0.1",
    "google/gemma-3-27b-it",
}
_FALLBACK_GROQ_MODEL   = "llama-3.3-70b-versatile"
_FALLBACK_NVIDIA_MODEL = "meta/llama-3.1-70b-instruct"

# ─── Session History ───────────────────────────────────────────────────────────
MAX_HISTORY = int(os.getenv("NOVA_MAX_HISTORY", "30"))

# ─── TTS ───────────────────────────────────────────────────────────────────────
DEFAULT_TTS_VOICE = os.getenv("TTS_DEFAULT_VOICE", "en-IN-NeerjaExpressiveNeural")

# ─── Google OAuth ──────────────────────────────────────────────────────────────
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")

# ─── Users File ────────────────────────────────────────────────────────────────
_USERS_FILE_BUNDLED = os.path.join(BASE_DIR, "nova_users.json")
USERS_FILE = os.path.join("/tmp", "nova_users.json") if IS_VERCEL else _USERS_FILE_BUNDLED
USERS_FILE_BUNDLED = _USERS_FILE_BUNDLED

# ─── Timeouts & Retry ─────────────────────────────────────────────────────────
API_TIMEOUT = int(os.getenv("NOVA_API_TIMEOUT", "30"))      # per-provider timeout
HYBRID_TIMEOUT = int(os.getenv("NOVA_HYBRID_TIMEOUT", "20")) # per-model in hybrid parallel
LOCAL_TIMEOUT = 120
MAX_RETRY = int(os.getenv("NOVA_MAX_RETRY", "2"))            # retries on transient errors
RETRY_BACKOFF = float(os.getenv("NOVA_RETRY_BACKOFF", "0.5")) # initial backoff seconds

# ─── Cache ─────────────────────────────────────────────────────────────────────
CACHE_TTL = int(os.getenv("NOVA_CACHE_TTL", "300"))          # 5 min default
CACHE_MAX_ENTRIES = int(os.getenv("NOVA_CACHE_MAX", "100"))

# ─── Hybrid Evaluator ─────────────────────────────────────────────────────────
HYBRID_MERGE_THRESHOLD = float(os.getenv("NOVA_HYBRID_MERGE_THRESHOLD", "0.15"))

# ─── Phase 10: ML Personality Prediction ──────────────────────────────────────
ENABLE_PERSONALITY_ML = os.getenv("ENABLE_PERSONALITY_ML", "true").lower() in ("true", "1", "yes")
PERSONALITY_ML_CONFIDENCE = float(os.getenv("PERSONALITY_ML_CONFIDENCE", "0.6"))

# ─── Phase 11: PDF Intelligence & Document-Aware Assistant ─────────────────────
PDF_MAX_FILE_SIZE_LOCAL = int(os.getenv("PDF_MAX_FILE_SIZE", str(50 * 1024 * 1024)))  # 50MB local
PDF_MAX_FILE_SIZE_VERCEL = 10 * 1024 * 1024                                            # 10MB serverless
PDF_MAX_FILE_SIZE = PDF_MAX_FILE_SIZE_VERCEL if IS_VERCEL else PDF_MAX_FILE_SIZE_LOCAL
PDF_MAX_DOCUMENTS_PER_SESSION = int(os.getenv("PDF_MAX_DOCUMENTS", "3"))
PDF_CHUNK_SIZE = int(os.getenv("PDF_CHUNK_SIZE", "1500"))          # chars per chunk
PDF_CHUNK_OVERLAP = int(os.getenv("PDF_CHUNK_OVERLAP", "200"))     # overlap between chunks
PDF_TOP_K_CHUNKS = int(os.getenv("PDF_TOP_K_CHUNKS", "5"))        # chunks retrieved per query
ENABLE_DOCUMENT_EMBEDDINGS = os.getenv("ENABLE_DOCUMENT_EMBEDDINGS", "true").lower() in ("true", "1", "yes")
ENABLE_EXAM_MODE = os.getenv("ENABLE_EXAM_MODE", "true").lower() in ("true", "1", "yes")


# ─── Settings Persistence ──────────────────────────────────────────────────────
_SETTINGS_FILE = os.path.join("/tmp" if IS_VERCEL else BASE_DIR, "nova_settings.json")

# Safe fields to persist (NEVER persist API keys to disk)
_PERSISTABLE_FIELDS = {
    "temperature", "top_p", "num_predict", "system_prompt",
    "provider", "groq_model", "nvidia_model",
}

def _load_persisted_settings():
    """Load settings from disk on startup (non-sensitive fields only)."""
    try:
        if os.path.exists(_SETTINGS_FILE):
            with open(_SETTINGS_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            for key, value in saved.items():
                if key in _PERSISTABLE_FIELDS and key in NOVA_SETTINGS:
                    NOVA_SETTINGS[key] = value
            # Validate provider — reject empty/invalid values
            _valid_providers = ("groq", "nvidia", "balanced")
            if not NOVA_SETTINGS.get("provider") or NOVA_SETTINGS["provider"] not in _valid_providers:
                import logging
                logging.getLogger("config").warning(
                    "Invalid persisted provider '%s' — resetting to 'groq'",
                    NOVA_SETTINGS.get("provider", "<empty>"),
                )
                NOVA_SETTINGS["provider"] = "groq"
            # Validate groq model — reject decommissioned ones
            if NOVA_SETTINGS["groq_model"] not in _VALID_GROQ_MODELS:
                import logging
                logging.getLogger("config").warning(
                    "Decommissioned groq_model '%s' detected — resetting to '%s'",
                    NOVA_SETTINGS["groq_model"], _FALLBACK_GROQ_MODEL,
                )
                NOVA_SETTINGS["groq_model"] = _FALLBACK_GROQ_MODEL
            # Validate nvidia model — reject decommissioned ones
            if NOVA_SETTINGS["nvidia_model"] not in _VALID_NVIDIA_MODELS:
                import logging
                logging.getLogger("config").warning(
                    "Decommissioned nvidia_model '%s' detected — resetting to '%s'",
                    NOVA_SETTINGS["nvidia_model"], _FALLBACK_NVIDIA_MODEL,
                )
                NOVA_SETTINGS["nvidia_model"] = _FALLBACK_NVIDIA_MODEL
    except Exception:
        pass  # File corrupt or inaccessible — use env defaults

def save_settings_to_disk():
    """Persist non-sensitive settings to disk."""
    try:
        data = {k: NOVA_SETTINGS[k] for k in _PERSISTABLE_FIELDS if k in NOVA_SETTINGS}
        with open(_SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass  # /tmp might be read-only in some envs

# Load persisted settings on import
_load_persisted_settings()


def get_settings():
    """Return the current mutable settings dict."""
    return NOVA_SETTINGS


def build_provider_config() -> dict:
    """Snapshot current settings for background memory tasks."""
    return {
        "provider":         NOVA_SETTINGS["provider"],
        "groq_api_key":     NOVA_SETTINGS["groq_api_key"],
        "groq_model":       NOVA_SETTINGS["groq_model"],
        "nvidia_api_key":   NOVA_SETTINGS["nvidia_api_key"],
        "nvidia_model":     NOVA_SETTINGS["nvidia_model"],
    }
