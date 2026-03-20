from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context, session, redirect, url_for
import requests
import psutil
import json
import asyncio
import io
import os
import subprocess
import hashlib
import uuid
import hmac
from functools import wraps
import time
import edge_tts
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from nova_memory import NovaMemory

# ─── CPU / GPU Detection ──────────────────────────────────────────────────────
CPU_CORES_LOGICAL  = os.cpu_count() or 4
CPU_CORES_PHYSICAL = psutil.cpu_count(logical=False) or 2

def _detect_gpu():
    """Return GPU info dict from nvidia-smi, or None if no NVIDIA GPU found."""
    try:
        r = subprocess.run(
            ['nvidia-smi',
             '--query-gpu=name,memory.total,memory.free,utilization.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=3
        )
        if r.returncode == 0 and r.stdout.strip():
            parts = [p.strip() for p in r.stdout.strip().split(',')]
            return {
                'name': parts[0],
                'vram_total_mb': int(parts[1]),
                'vram_free_mb':  int(parts[2]),
                'utilization_pct': int(parts[3]),
            }
    except Exception:
        pass
    return None

GPU_INFO = _detect_gpu()   # None if no CUDA GPU
if GPU_INFO:
    print(f"[NOVA] GPU detected: {GPU_INFO['name']}  "
          f"VRAM {GPU_INFO['vram_total_mb']}MB")
else:
    print(f"[NOVA] No CUDA GPU detected — running CPU-only backend")
print(f"[NOVA] CPU: {CPU_CORES_PHYSICAL} physical / {CPU_CORES_LOGICAL} logical cores")

# ─── Background Thread Pool ───────────────────────────────────────────────────
# Used for CPU-bound background tasks (memory extraction, summarisation).
# Use 2× physical cores so we can overlap I/O waits without saturating CPU.
_BG_WORKERS = max(4, CPU_CORES_PHYSICAL * 2)
bg_pool = ThreadPoolExecutor(max_workers=_BG_WORKERS,
                             thread_name_prefix='nova-bg')
print(f"[NOVA] Background thread pool: {_BG_WORKERS} workers")

# ─── Load .env ────────────────────────────────────────────────────────────────
load_dotenv()

# ─── Determine absolute path for Frontend ─────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "Frontend")

app = Flask(
    __name__,
    static_folder=os.path.join(FRONTEND_DIR, "static"),
    static_url_path="/static"
)
app.json.sort_keys = False
app.secret_key = os.getenv("FLASK_SECRET_KEY", "nova-secret-change-me-in-env")
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["PERMANENT_SESSION_LIFETIME"] = 60 * 60 * 24 * 30  # 30 days

# On HTTPS deployments (Vercel, etc.), cookies MUST be Secure
if os.getenv("NOVA_LIVE_MODE", "false").lower() in ("true", "1", "yes"):
    app.config["SESSION_COOKIE_SECURE"] = True

# ─── User Database (JSON flat-file) ───────────────────────────────────────────
# Vercel serverless has a read-only project dir; use /tmp for writes.
_is_vercel = bool(os.getenv("VERCEL") or os.getenv("VERCEL_ENV"))
_USERS_FILE_BUNDLED = os.path.join(BASE_DIR, "nova_users.json")   # ships with deploy
_USERS_FILE = os.path.join("/tmp", "nova_users.json") if _is_vercel else _USERS_FILE_BUNDLED


def _load_users() -> dict:
    # Try writable location first
    if os.path.exists(_USERS_FILE):
        try:
            with open(_USERS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    # On Vercel cold-start, seed from the bundled file in the project dir
    if _is_vercel and os.path.exists(_USERS_FILE_BUNDLED):
        try:
            import shutil
            shutil.copy2(_USERS_FILE_BUNDLED, _USERS_FILE)
            with open(_USERS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def _save_users(users: dict):
    with open(_USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)

def _hash_password(password: str, salt: str = None) -> tuple:
    """Return (hashed, salt) using PBKDF2-HMAC-SHA256."""
    if salt is None:
        salt = uuid.uuid4().hex
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 260_000)
    return dk.hex(), salt

def _verify_password(password: str, stored_hash: str, salt: str) -> bool:
    candidate, _ = _hash_password(password, salt)
    return hmac.compare_digest(candidate, stored_hash)

def login_required(f):
    """Decorator: redirects to /login if the user is not authenticated."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("user_id"):
            return redirect("/login?next=" + request.path)
        return f(*args, **kwargs)
    return decorated

print("[NOVA] Auth system ready.")

# ─── CORS ─────────────────────────────────────────────────────────────────────
@app.after_request
def add_cors(response):
    origin = request.headers.get("Origin", "")
    if origin:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
    else:
        response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Session-Id"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

@app.route("/", defaults={"path": ""}, methods=["OPTIONS"])
@app.route("/<path:path>", methods=["OPTIONS"])
def handle_options(path):
    return "", 204

# ─── Nova Memory Engine ───────────────────────────────────────────────────────
memory = NovaMemory()

# ─── Session History Table (SQLite, shared nova_memory.db connection) ─────────
_db = memory._conn    # re-use the WAL connection from NovaMemory
print("[NOVA] Session history table ready (SQLite).")

# ─── Nova Default Personality ─────────────────────────────────────────────────
DEFAULT_SYSTEM_PROMPT = (
    "You are Nova, a highly intelligent, warm, and conversational AI assistant. "
    "You are like a brilliant friend who explains things clearly, concisely, and naturally — "
    "similar to how ChatGPT and Gemini converse. You always remember what was said earlier in the "
    "conversation and build on it naturally. "
    "Key behaviours:\n"
    "- Respond in clear, flowing natural language (no robotic lists unless explicitly asked).\n"
    "- Keep replies concise but complete. Avoid over-explaining.\n"
    "- If the user greets you, greet back warmly and ask how you can help.\n"
    "- If asked to help with a task (coding, math, writing, research), do it fully.\n"
    "- For code, always use properly fenced code blocks with the language name (e.g. ```python).\n"
    "- Never say you are an Ollama model or reveal underlying technology. You are Nova.\n"
    "- Speak in a confident, friendly, slightly futuristic tone.\n"
    "- When you don't know something, say so honestly rather than guessing.\n"
    "- USE YOUR MEMORY to personalise responses — address the user by name if known, "
    "reference their interests, and adapt to their preferences.\n"
)

# ─── Environment Detection ────────────────────────────────────────────────────
# ENV='production' on Vercel/cloud, 'local' for dev.  Backward-compat with NOVA_LIVE_MODE.
_live_flag = os.getenv("NOVA_LIVE_MODE", "false").lower() in ("true", "1", "yes")
_vercel_flag = bool(os.getenv("VERCEL") or os.getenv("VERCEL_ENV"))
NOVA_ENV = os.getenv("ENV", "production" if (_live_flag or _vercel_flag) else "local")
NOVA_LIVE_MODE = (NOVA_ENV == "production")   # backward compat for existing code

# Ollama URL — NEVER localhost in production
if NOVA_ENV == "local":
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
else:
    OLLAMA_URL = os.getenv("OLLAMA_API_URL",
                           os.getenv("OLLAMA_CLOUD_URL", "https://api.ollama.com/api/generate"))

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# ─── Startup Logging ──────────────────────────────────────────────────────────
print(f"[NOVA] ENV: {NOVA_ENV}")
print(f"[NOVA] Using Ollama: {OLLAMA_URL}")
if NOVA_ENV == "production":
    print("[NOVA] Production mode — Ollama Local DISABLED, only cloud providers active.")

# ─── Global Settings (can be changed via /settings) ──────────────────────────
_default_provider = os.getenv("NOVA_PROVIDER", "ollama")
# In live mode, default provider must be a cloud provider
if NOVA_LIVE_MODE and _default_provider == "ollama":
    _default_provider = "groq" if os.getenv("GROQ_API_KEY") else "ollama_cloud"
nova_settings = {
    "model":                os.getenv("OLLAMA_DEFAULT_MODEL", "mistral"),
    "temperature":          float(os.getenv("NOVA_TEMPERATURE", "0.75")),
    "top_p":                float(os.getenv("NOVA_TOP_P", "0.9")),
    "num_predict":          int(os.getenv("NOVA_MAX_TOKENS", "1024")),
    "system_prompt":        DEFAULT_SYSTEM_PROMPT,
    "provider":             _default_provider,  # "ollama"|"ollama_cloud"|"groq"|"hybrid"
    "groq_api_key":         os.getenv("GROQ_API_KEY", ""),
    "groq_model":           os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
    # ── Ollama Cloud (remote Ollama endpoint with API key) ──────────────────
    "ollama_api_key":       os.getenv("OLLAMA_API_KEY", ""),
    "ollama_cloud_url":     os.getenv("OLLAMA_CLOUD_URL", "https://api.ollama.com/api/generate"),
    # ── Hybrid mode — per-sub-provider model choices ────────────────────────
    "hybrid_ollama_model":  os.getenv("NOVA_HYBRID_OLLAMA_MODEL", "mistral"),
    "hybrid_groq_model":    os.getenv("NOVA_HYBRID_GROQ_MODEL", "llama-3.3-70b-versatile"),
}

def _build_provider_config() -> dict:
    """Snapshot current settings for background memory tasks."""
    return {
        "provider":         nova_settings["provider"],
        "groq_api_key":     nova_settings["groq_api_key"],
        "groq_model":       nova_settings["groq_model"],
        "ollama_api_key":   nova_settings["ollama_api_key"],
        "ollama_cloud_url": nova_settings["ollama_cloud_url"],
    }

# ─── Ollama Cloud Sanity Check ────────────────────────────────────────────────
def _ollama_cloud_configured() -> bool:
    """
    Return True when an Ollama API key is present (non-empty).
    We intentionally do NOT block api.ollama.com — that is the real Ollama
    Cloud endpoint and users may have valid keys for it.
    """
    return bool(nova_settings.get("ollama_api_key", "").strip())


def _groq_configured() -> bool:
    """Return True when a Groq API key is present."""
    return bool(nova_settings.get("groq_api_key", "").strip())


def _hybrid_pick_sub(message: str) -> str:
    """
    Decide which sub-provider to use for a hybrid request.
    Classification:
      - 'groq'        → Groq Cloud for complex / long messages
      - 'ollama_cloud' → Ollama Cloud for simple / short messages
      - 'groq_only'   → Ollama Cloud not configured, always use Groq
    Falls back gracefully when a provider is not available.
    """
    ollama_ok = _ollama_cloud_configured()
    groq_ok   = _groq_configured()

    if not ollama_ok and not groq_ok:
        return "none"  # nothing configured

    if not ollama_ok:
        return "groq"  # only Groq available — route everything there

    if not groq_ok:
        return "ollama_cloud"  # only Ollama Cloud available

    # Both configured — route by complexity
    sub = _classify_query(message)
    return sub  # 'groq' or 'ollama_cloud'

# ─── Per-Session Conversation History (SQLite) ────────────────────────────────
MAX_HISTORY = int(os.getenv("NOVA_MAX_HISTORY", "30"))


def get_history(session_id: str) -> list:
    """Return the conversation history list for the given session from SQLite."""
    rows = _db.execute(
        "SELECT role, content FROM sessions "
        "WHERE session_id = ? ORDER BY turn ASC",
        (session_id,)
    ).fetchall()
    return [{"role": r[0], "content": r[1]} for r in rows]


def _next_turn(session_id: str) -> int:
    """Return the next turn index for a session."""
    row = _db.execute(
        "SELECT COALESCE(MAX(turn), -1) FROM sessions WHERE session_id = ?",
        (session_id,)
    ).fetchone()
    return (row[0] + 1) if row else 0


def append_message(session_id: str, role: str, content: str):
    """Append a single message to the session history in SQLite."""
    turn = _next_turn(session_id)
    _db.execute(
        "INSERT INTO sessions (session_id, turn, role, content) VALUES (?, ?, ?, ?)",
        (session_id, turn, role, content)
    )
    # Trim to MAX_HISTORY — keep only the most recent turns
    _db.execute(
        "DELETE FROM sessions WHERE session_id = ? AND turn <= "
        "(SELECT MAX(turn) - ? FROM sessions WHERE session_id = ?)",
        (session_id, MAX_HISTORY, session_id)
    )
    _db.commit()


def build_prompt(history: list) -> str:
    """Build the full prompt string from history for Ollama, injecting memory."""
    base_prompt = nova_settings["system_prompt"].strip()
    memory_ctx = memory.get_memory_context()
    lines = [base_prompt + memory_ctx]
    for msg in history:
        role = "User" if msg["role"] == "user" else "Nova"
        lines.append(f"{role}: {msg['content']}")
    lines.append("Nova:")
    return "\n".join(lines)


def build_groq_messages(history: list) -> list:
    """Build OpenAI-format messages list for Groq API."""
    memory_ctx = memory.get_memory_context()
    system_msg = nova_settings["system_prompt"].strip() + memory_ctx
    messages = [{"role": "system", "content": system_msg}]
    for msg in history:
        role = "user" if msg["role"] == "user" else "assistant"
        messages.append({"role": role, "content": msg["content"]})
    return messages


# ─── Hybrid Query Classifier ──────────────────────────────────────────────────
_COMPLEX_KEYWORDS = (
    # Code & technical
    "def ", "function ", "class ", "import ", "```", "code", "debug", "error",
    "bug", "implement", "algorithm", "write a ", "create a ", "build a ",
    "program", "script", "api", "database", "sql", "regex",
    # Math & reasoning
    "math", "calculate", "solve", "equation", "prove", "integral",
    "derivative", "matrix", "statistics", "probability",
    # Analysis & writing
    "analyze", "analyse", "summarize", "summarise", "essay", "detailed",
    "comprehensive", "explain why", "compare", "difference between",
    "pros and cons", "advantages", "disadvantages", "research",
    "translate", "rewrite", "refactor",
)
_COMPLEX_WORD_THRESHOLD  = 60   # > 60 words → use Groq
_SIMPLE_WORD_THRESHOLD   = 30   # ≤ 30 words AND no keywords → use Ollama Cloud

def _classify_query(message: str) -> str:
    """
    Returns 'groq' for complex queries, 'ollama_cloud' for simple ones.
    Used only when provider == 'hybrid'.
    """
    lower = message.lower()
    word_count = len(message.split())

    # Always use Groq for long messages
    if word_count > _COMPLEX_WORD_THRESHOLD:
        return "groq"

    # Check for complex keywords regardless of length
    if any(kw in lower for kw in _COMPLEX_KEYWORDS):
        return "groq"

    # Short, conversational → Ollama Cloud (fast & cheap)
    return "ollama_cloud"


def _call_ollama_cloud(full_prompt: str, stream: bool = False):
    """Helper: call the Ollama Cloud endpoint. Returns requests.Response."""
    cloud_url = nova_settings["ollama_cloud_url"] or "https://api.ollama.com/api/generate"
    model = (
        nova_settings["hybrid_ollama_model"]
        if nova_settings["provider"] == "hybrid"
        else nova_settings["model"]
    ) or "mistral"  # safe fallback if setting is empty
    payload = {
        "model": model,
        "prompt": full_prompt,
        "stream": stream,
        "options": {
            "temperature": nova_settings["temperature"],
            "top_p":       nova_settings["top_p"],
            "num_predict": nova_settings["num_predict"],
        }
    }
    # Include Authorization header only if an API key is set
    headers = {"Content-Type": "application/json"}
    if nova_settings["ollama_api_key"]:
        headers["Authorization"] = f"Bearer {nova_settings['ollama_api_key']}"
    _timeout = 8 if NOVA_ENV == "production" else 30
    return requests.post(cloud_url, json=payload, headers=headers,
                         stream=stream, timeout=_timeout)


def _call_groq(history: list, stream: bool = False, model_override: str = ""):
    """Helper: call the Groq API. Returns requests.Response."""
    groq_model = model_override or (
        nova_settings["hybrid_groq_model"]
        if nova_settings["provider"] == "hybrid"
        else nova_settings["groq_model"]
    ) or "llama-3.3-70b-versatile"  # safe fallback if setting is empty
    groq_messages = build_groq_messages(history)
    payload = {
        "model": groq_model,
        "messages": groq_messages,
        "temperature": nova_settings["temperature"],
        "max_tokens": nova_settings["num_predict"],
        "top_p": nova_settings["top_p"],
        "stream": stream,
    }
    headers = {
        "Authorization": f"Bearer {nova_settings['groq_api_key']}",
        "Content-Type": "application/json",
    }
    _timeout = 8 if NOVA_ENV == "production" else 30
    return requests.post(GROQ_API_URL, json=payload, headers=headers,
                         stream=stream, timeout=_timeout)


# ─── Local Ollama Helper (only for ENV=local) ─────────────────────────────────
def _call_ollama_local(full_prompt: str, stream: bool = False):
    """Call local Ollama. Only allowed in local ENV. Returns requests.Response."""
    if NOVA_ENV != "local":
        raise ConnectionError("[NOVA] Ollama local is disabled in production")
    payload = {
        "model": nova_settings["model"],
        "prompt": full_prompt,
        "stream": stream,
        "options": {
            "temperature":    nova_settings["temperature"],
            "top_p":          nova_settings["top_p"],
            "num_predict":    nova_settings["num_predict"],
            "num_ctx":        2048,
            "num_thread":     CPU_CORES_PHYSICAL,
            "repeat_penalty": 1.1,
        }
    }
    return requests.post(OLLAMA_URL, json=payload, stream=stream, timeout=120)


def _safe_memory(fn, *args):
    """Submit a memory task to the background pool. Never crashes."""
    try:
        bg_pool.submit(fn, *args)
    except Exception as exc:
        print(f"[NOVA] Memory task failed (ignored): {exc}")



# ─── Auth Routes ─────────────────────────────────────────────────────────────

@app.route("/auth/register", methods=["POST"])
def auth_register():
    data = request.get_json() or {}
    name     = str(data.get("name", "")).strip()
    email    = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", ""))

    if not name or not email or not password:
        return jsonify({"ok": False, "error": "All fields are required."}), 400
    if len(password) < 6:
        return jsonify({"ok": False, "error": "Password must be at least 6 characters."}), 400
    if "@" not in email or "." not in email.split("@")[-1]:
        return jsonify({"ok": False, "error": "Please enter a valid email address."}), 400

    users = _load_users()
    if email in users:
        return jsonify({"ok": False, "error": "An account with this email already exists."}), 409

    hashed, salt = _hash_password(password)
    users[email] = {
        "id":       str(uuid.uuid4()),
        "name":     name,
        "email":    email,
        "hash":     hashed,
        "salt":     salt,
        "provider": "email",
        "created":  int(time.time()),
    }
    _save_users(users)

    session.permanent = True
    session["user_id"]    = users[email]["id"]
    session["user_email"] = email
    session["user_name"]  = name

    return jsonify({"ok": True, "user": {"name": name, "email": email}})


@app.route("/auth/login", methods=["POST"])
def auth_login():
    data = request.get_json() or {}
    email    = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", ""))

    if not email or not password:
        return jsonify({"ok": False, "error": "Email and password are required."}), 400

    users = _load_users()
    user  = users.get(email)
    if not user:
        return jsonify({"ok": False, "error": "Incorrect email or password."}), 401
    # Google-only accounts can't use password login
    if user.get("provider") == "google" and not user.get("hash"):
        return jsonify({"ok": False, "error": "This account uses Google sign-in. Please click 'Continue with Google' instead."}), 401
    if not _verify_password(password, user["hash"], user["salt"]):
        return jsonify({"ok": False, "error": "Incorrect email or password."}), 401

    session.permanent = True
    session["user_id"]    = user["id"]
    session["user_email"] = email
    session["user_name"]  = user["name"]

    return jsonify({"ok": True, "user": {"name": user["name"], "email": email}})


@app.route("/auth/logout", methods=["POST"])
def auth_logout():
    session.clear()
    return jsonify({"ok": True})


@app.route("/auth/me", methods=["GET"])
def auth_me():
    if not session.get("user_id"):
        return jsonify({"ok": False, "user": None}), 401
    return jsonify({
        "ok": True,
        "user": {
            "id":    session.get("user_id"),
            "name":  session.get("user_name"),
            "email": session.get("user_email"),
        }
    })


def _get_base_url() -> str:
    """
    Build the canonical base URL for OAuth redirect URIs.

    On localhost: always returns http://localhost:<port> (ignores NOVA_BASE_URL)
    On production: uses NOVA_BASE_URL or X-Forwarded-Proto headers
    """
    # Detect localhost from the actual request — most reliable check
    host = request.host or ""  # e.g. "localhost:5000" or "nova-ai...vercel.app"
    is_local = host.startswith("localhost") or host.startswith("127.0.0.1")

    if is_local:
        # Always use the real local URL for OAuth on localhost
        return request.host_url.rstrip("/")

    # Production: explicit override from env var
    base = os.getenv("NOVA_BASE_URL", "").rstrip("/")
    if base:
        return base

    # Detect scheme from Vercel / proxy headers
    proto = request.headers.get("X-Forwarded-Proto", "").split(",")[0].strip()
    fwd_host = request.headers.get("X-Forwarded-Host", "").split(",")[0].strip()
    if not fwd_host:
        fwd_host = host

    if proto in ("https", "http"):
        return f"{proto}://{fwd_host}"

    # Last resort
    return request.host_url.rstrip("/")


@app.route("/auth/google")
def auth_google():
    """Google OAuth 2.0 — redirect to Google's consent screen."""
    google_client_id = os.getenv("GOOGLE_CLIENT_ID", "")
    if not google_client_id:
        return redirect("/login?error=google_not_configured")

    # CSRF protection: store a random state token in the session
    state = uuid.uuid4().hex
    session["oauth_state"] = state

    base         = _get_base_url()
    redirect_uri = base + "/auth/google/callback"
    print(f"[NOVA] Google OAuth → redirect_uri: {redirect_uri}")

    from urllib.parse import urlencode
    params = {
        "client_id":     google_client_id,
        "redirect_uri":  redirect_uri,
        "response_type": "code",
        "scope":         "openid email profile",
        "access_type":   "offline",
        "prompt":        "select_account",
        "state":         state,
    }
    return redirect("https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(params))


@app.route("/auth/google/callback")
def auth_google_callback():
    """Google OAuth callback — exchanges code for tokens and logs the user in."""
    # Check for OAuth errors from Google
    error = request.args.get("error")
    if error:
        print(f"[NOVA] Google OAuth error from Google: {error}")
        return redirect("/login?error=google_cancelled")

    code  = request.args.get("code")
    state = request.args.get("state", "")

    if not code:
        return redirect("/login?error=google_failed")

    # CSRF check — state must match what we stored before redirecting
    expected_state = session.pop("oauth_state", None)
    if not expected_state or state != expected_state:
        print("[NOVA] Google OAuth CSRF state mismatch!")
        return redirect("/login?error=google_failed")

    google_client_id     = os.getenv("GOOGLE_CLIENT_ID", "")
    google_client_secret = os.getenv("GOOGLE_CLIENT_SECRET", "")
    if not google_client_id or not google_client_secret:
        return redirect("/login?error=google_not_configured")

    # Build the SAME redirect_uri used in /auth/google (must match exactly)
    base         = _get_base_url()
    redirect_uri = base + "/auth/google/callback"
    print(f"[NOVA] Google callback → redirect_uri: {redirect_uri}")

    # ── Step 1: Exchange authorisation code for access token ─────────────────
    token_res = requests.post(
        "https://oauth2.googleapis.com/token",
        data={
            "code":          code,
            "client_id":     google_client_id,
            "client_secret": google_client_secret,
            "redirect_uri":  redirect_uri,
            "grant_type":    "authorization_code",
        },
        timeout=15,
    )
    if not token_res.ok:
        print(f"[NOVA] Google token exchange FAILED: {token_res.status_code} — {token_res.text[:300]}")
        return redirect("/login?error=google_token_failed")

    token_data   = token_res.json()
    access_token = token_data.get("access_token", "")
    if not access_token:
        print(f"[NOVA] Google token response missing access_token: {token_data}")
        return redirect("/login?error=google_token_failed")

    # ── Step 2: Fetch user profile from Google ───────────────────────────────
    userinfo_res = requests.get(
        "https://www.googleapis.com/oauth2/v3/userinfo",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=10,
    )
    if not userinfo_res.ok:
        print(f"[NOVA] Google userinfo FAILED: {userinfo_res.status_code} — {userinfo_res.text[:200]}")
        return redirect("/login?error=google_userinfo_failed")

    info  = userinfo_res.json()
    email = info.get("email", "").strip().lower()
    name  = info.get("name", "").strip() or email.split("@")[0]

    if not email:
        print("[NOVA] Google userinfo returned no email address!")
        return redirect("/login?error=google_userinfo_failed")

    # ── Step 3: Upsert user in our store ─────────────────────────────────────
    users = _load_users()
    if email not in users:
        users[email] = {
            "id":       str(uuid.uuid4()),
            "name":     name,
            "email":    email,
            "hash":     "",
            "salt":     "",
            "provider": "google",
            "created":  int(time.time()),
        }
        _save_users(users)
        print(f"[NOVA] New Google user created: {email}")
    else:
        # Update name in case it changed
        if name and users[email].get("name") != name:
            users[email]["name"] = name
            _save_users(users)

    # ── Step 4: Create Flask session ─────────────────────────────────────────
    session.permanent = True
    session["user_id"]    = users[email]["id"]
    session["user_email"] = email
    session["user_name"]  = users[email]["name"]
    print(f"[NOVA] Google login success: {email}")
    return redirect("/app")


# ─── Serve Frontend ───────────────────────────────────────────────────────────
@app.route("/")
def serve_landing():
    """Marketing / landing page — shown to unauthenticated visitors."""
    return send_from_directory(FRONTEND_DIR, "landing.html")


@app.route("/login")
def serve_login():
    """Sign in / sign up page."""
    # If already logged in, go straight to the app
    if session.get("user_id"):
        return redirect("/app")
    return send_from_directory(FRONTEND_DIR, "login.html")


@app.route("/app")
@login_required
def serve_index():
    """Main NOVA AI assistant application (requires auth)."""
    return send_from_directory(FRONTEND_DIR, "index.html")


# ─── Chat Route (standard, full response) ─────────────────────────────────────
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        if not data or "message" not in data:
            return jsonify({"error": "No message provided", "code": "NO_MESSAGE"}), 400

        user_message = data.get("message", "").strip()
        if not user_message:
            return jsonify({"error": "Empty message", "code": "EMPTY_MESSAGE"}), 400

        session_id = data.get("session_id") or request.headers.get("X-Session-Id", "default")

        # Persist user turn to SQLite then read updated history
        append_message(session_id, "user", user_message)
        history = get_history(session_id)

        full_prompt = build_prompt(history)

        provider = nova_settings["provider"]

        # ─── Provider dispatch ────────────────────────────────────────────
        # Guard: for ollama_cloud-only mode, fall back if not configured
        if provider == "ollama_cloud" and not _ollama_cloud_configured():
            if _groq_configured():
                print("[NOVA] ollama_cloud → not configured; falling back to Groq")
                provider = "groq"
            elif not NOVA_LIVE_MODE:
                provider = "ollama"
            else:
                return jsonify({"error": "No cloud AI provider configured.", "code": "NO_CLOUD_PROVIDER"}), 503

        if provider == "hybrid":
            # ── Hybrid: smart routing by complexity + availability ───────
            sub = _hybrid_pick_sub(user_message)
            print(f"[NOVA] Hybrid → {sub} ({len(user_message.split())} words)")

            if sub == "none":
                if not NOVA_LIVE_MODE:
                    # last resort: local Ollama
                    sub = "_ollama_local"
                else:
                    return jsonify({"error": "No cloud AI provider configured. Set a Groq API key or a valid Ollama Cloud URL.", "code": "NO_CLOUD_PROVIDER"}), 503

            ai_response = ""
            actual_sub  = sub  # track which sub-provider ended up answering

            if sub == "groq":
                try:
                    r = _call_groq(history)
                    if r.status_code == 200:
                        ai_response = r.json()["choices"][0]["message"]["content"].strip()
                    else:
                        print(f"[NOVA] Hybrid Groq failed ({r.status_code}), trying Ollama Cloud fallback")
                except Exception as exc:
                    print(f"[NOVA] Hybrid Groq exception: {exc}, trying Ollama Cloud fallback")
                # Fallback to Ollama Cloud if Groq failed
                if not ai_response and _ollama_cloud_configured():
                    r = _call_ollama_cloud(full_prompt)
                    if r.status_code == 200:
                        ai_response = r.json().get("response", "").strip()
                        actual_sub  = "ollama_cloud"

            elif sub == "ollama_cloud":
                try:
                    r = _call_ollama_cloud(full_prompt)
                    if r.status_code == 200:
                        ai_response = r.json().get("response", "").strip()
                    else:
                        print(f"[NOVA] Hybrid Ollama Cloud failed ({r.status_code}), trying Groq fallback")
                except Exception as exc:
                    print(f"[NOVA] Hybrid Ollama Cloud exception: {exc}, trying Groq fallback")
                # Fallback to Groq if Ollama Cloud failed
                if not ai_response and _groq_configured():
                    r = _call_groq(history)
                    if r.status_code == 200:
                        ai_response = r.json()["choices"][0]["message"]["content"].strip()
                        actual_sub  = "groq"

            else:  # _ollama_local
                try:
                    r = _call_ollama_local(full_prompt)
                    if r.status_code == 200:
                        ai_response = r.json().get("response", "").strip()
                except Exception as exc:
                    print(f"[NOVA] Hybrid local Ollama failed: {exc}")
                # Fallback to Groq if local Ollama failed
                if not ai_response and _groq_configured():
                    print("[NOVA] Fallback to Groq (local Ollama failed)")
                    try:
                        r = _call_groq(history)
                        if r.status_code == 200:
                            ai_response = r.json()["choices"][0]["message"]["content"].strip()
                            actual_sub = "groq"
                    except Exception as exc2:
                        print(f"[NOVA] Groq fallback also failed: {exc2}")

            if actual_sub == "groq":
                active_model = nova_settings["hybrid_groq_model"] or nova_settings["groq_model"]
            elif actual_sub == "ollama_cloud":
                active_model = nova_settings["hybrid_ollama_model"] or nova_settings["model"]
            else:
                active_model = nova_settings["model"]

        elif provider == "groq" and _groq_configured():
            # Groq Cloud only
            r = _call_groq(history, model_override=nova_settings["groq_model"])
            if r.status_code != 200:
                err_detail = r.json().get("error", {}).get("message", "Groq API error")
                return jsonify({"error": err_detail, "code": "GROQ_ERROR"}), 500
            ai_response = r.json()["choices"][0]["message"]["content"].strip()
            active_model = nova_settings["groq_model"]

        elif provider == "ollama_cloud" and _ollama_cloud_configured():
            # Ollama Cloud only
            r = _call_ollama_cloud(full_prompt)
            if r.status_code != 200:
                return jsonify({"error": f"Ollama Cloud error: {r.text[:200]}", "code": "OLLAMA_CLOUD_ERROR"}), 500
            ai_response = r.json().get("response", "").strip()
            active_model = nova_settings["model"]

        else:
            # Ollama Local API (default / last resort)
            ai_response = ""
            try:
                r = _call_ollama_local(full_prompt)
                if r.status_code == 200:
                    ai_response = r.json().get("response", "").strip()
                else:
                    print(f"[NOVA] Ollama local error ({r.status_code}), trying Groq fallback")
            except Exception as exc:
                print(f"[NOVA] Ollama local failed: {exc}")
            # Auto-fallback to Groq
            if not ai_response and _groq_configured():
                print("[NOVA] Fallback to Groq")
                try:
                    r = _call_groq(history)
                    if r.status_code == 200:
                        ai_response = r.json()["choices"][0]["message"]["content"].strip()
                        active_model = nova_settings["groq_model"]
                except Exception as exc2:
                    print(f"[NOVA] Groq fallback also failed: {exc2}")
            if not ai_response:
                return jsonify({"error": "AI engine error — all providers failed", "code": "ALL_PROVIDERS_FAILED"}), 503
            active_model = active_model if ai_response else nova_settings["model"]

        if not ai_response:
            ai_response = "I'm not sure how to respond to that. Could you rephrase?"

        # Persist Nova turn to SQLite
        append_message(session_id, "nova", ai_response)

        # ── Learning: offload to background thread pool (non-blocking) ───────
        _safe_memory(memory.record_conversation)
        _safe_memory(memory.extract_and_store, user_message, ai_response, active_model, _build_provider_config())

        return jsonify({
            "reply": ai_response,
            "session_id": session_id,
            "model": active_model,
            "provider": provider,
        })

    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Cannot reach AI engine. Is it running?", "code": "CONNECTION_ERROR"}), 503
    except requests.exceptions.Timeout:
        return jsonify({"error": "The AI engine timed out. Please try again.", "code": "TIMEOUT"}), 504
    except Exception as e:
        return jsonify({"error": str(e), "code": "UNKNOWN"}), 500


# ─── Chat Stream Route (SSE — token by token) ─────────────────────────────────
@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    try:
        data = request.get_json()
        if not data or "message" not in data:
            return jsonify({"error": "No message provided"}), 400

        user_message = data.get("message", "").strip()
        if not user_message:
            return jsonify({"error": "Empty message"}), 400

        session_id = data.get("session_id") or request.headers.get("X-Session-Id", "default")

        # Persist user turn to SQLite then read updated history
        append_message(session_id, "user", user_message)
        history = get_history(session_id)

        full_prompt = build_prompt(history)

        provider = nova_settings["provider"]
        # Guard: for ollama_cloud-only mode, fall back if not configured
        if provider == "ollama_cloud" and not _ollama_cloud_configured():
            if _groq_configured():
                print("[NOVA] Stream ollama_cloud → not configured; falling back to Groq")
                provider = "groq"
            elif not NOVA_LIVE_MODE:
                provider = "ollama"
            else:
                def _err_gen():
                    yield f"data: {json.dumps({'error': 'No cloud AI provider configured.'})}\n\n"
                return Response(stream_with_context(_err_gen()), mimetype="text/event-stream",
                                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

        use_groq        = provider == "groq" and _groq_configured()
        use_ollama_cloud = provider == "ollama_cloud"
        use_hybrid      = provider == "hybrid"

        # active_model for non-hybrid paths (hybrid will be set per-sub-provider inside generate())
        if use_groq:
            active_model = nova_settings["groq_model"]
        else:
            active_model = nova_settings["model"]

        def generate():
            nonlocal active_model
            full_reply = []
            try:
                if use_hybrid:
                    # ─── Hybrid streaming: smart route + fallback ──────────────
                    sub = _hybrid_pick_sub(user_message)
                    print(f"[NOVA] Hybrid stream → {sub} ({len(user_message.split())} words)")

                    if sub == "none":
                        if not NOVA_LIVE_MODE:
                            sub = "_ollama_local"
                        else:
                            yield f"data: {json.dumps({'error': 'No cloud AI provider configured.'})}\n\n"
                            return

                    used_provider = None  # track which sub actually streamed tokens

                    # ── Try primary sub-provider ─────────────────────────────
                    if sub == "groq":
                        try:
                            with _call_groq(history, stream=True) as r:
                                for line in r.iter_lines():
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
                                            yield f"data: {json.dumps({'token': token})}\n\n"
                                    except (json.JSONDecodeError, IndexError, KeyError):
                                        continue
                            used_provider = "groq"
                        except Exception as exc:
                            print(f"[NOVA] Hybrid stream Groq failed: {exc}, trying Ollama Cloud")
                            full_reply = []

                    elif sub == "ollama_cloud":
                        try:
                            with _call_ollama_cloud(full_prompt, stream=True) as r:
                                if r.status_code == 200:
                                    for line in r.iter_lines():
                                        if not line:
                                            continue
                                        try:
                                            chunk = json.loads(line)
                                            if chunk.get("error"):
                                                print(f"[NOVA] Hybrid Ollama Cloud model error: {chunk['error']}")
                                                break
                                            token = chunk.get("response", "")
                                            if token:
                                                full_reply.append(token)
                                                yield f"data: {json.dumps({'token': token})}\n\n"
                                            if chunk.get("done"):
                                                break
                                        except json.JSONDecodeError:
                                            continue
                                    used_provider = "ollama_cloud"
                                else:
                                    print(f"[NOVA] Hybrid Ollama Cloud error {r.status_code}, trying Groq")
                        except Exception as exc:
                            print(f"[NOVA] Hybrid stream Ollama Cloud failed: {exc}, trying Groq")
                            full_reply = []

                    # ── Fallback: if primary failed, try the other provider ──
                    if not used_provider:
                        if sub == "groq" and _ollama_cloud_configured():
                            # Groq failed → try Ollama Cloud
                            try:
                                with _call_ollama_cloud(full_prompt, stream=True) as r:
                                    if r.status_code == 200:
                                        for line in r.iter_lines():
                                            if not line:
                                                continue
                                            try:
                                                chunk = json.loads(line)
                                                token = chunk.get("response", "")
                                                if token:
                                                    full_reply.append(token)
                                                    yield f"data: {json.dumps({'token': token})}\n\n"
                                                if chunk.get("done"):
                                                    break
                                            except json.JSONDecodeError:
                                                continue
                                        used_provider = "ollama_cloud"
                            except Exception as exc:
                                print(f"[NOVA] Hybrid fallback Ollama Cloud failed: {exc}")
                        elif sub == "ollama_cloud" and _groq_configured():
                            # Ollama Cloud failed/absent → try Groq
                            print(f"[NOVA] Hybrid fallback: Groq configured={_groq_configured()}, attempting Groq stream")
                            try:
                                with _call_groq(history, stream=True) as r:
                                    print(f"[NOVA] Hybrid fallback Groq response status: {r.status_code}")
                                    if r.status_code != 200:
                                        print(f"[NOVA] Hybrid fallback Groq error body: {r.text[:300]}")
                                    else:
                                        for line in r.iter_lines():
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
                                                    yield f"data: {json.dumps({'token': token})}\n\n"
                                            except (json.JSONDecodeError, IndexError, KeyError):
                                                continue
                                        used_provider = "groq"
                                        print(f"[NOVA] Hybrid fallback Groq streamed {len(full_reply)} tokens")
                            except Exception as exc:
                                print(f"[NOVA] Hybrid fallback Groq EXCEPTION: {exc}")
                        elif NOVA_ENV == "local":
                            # Last resort: Ollama local (only in local ENV)
                            try:
                                with _call_ollama_local(full_prompt, stream=True) as r:
                                    for line in r.iter_lines():
                                        if not line:
                                            continue
                                        try:
                                            chunk = json.loads(line)
                                            token = chunk.get("response", "")
                                            if token:
                                                full_reply.append(token)
                                                yield f"data: {json.dumps({'token': token})}\n\n"
                                            if chunk.get("done"):
                                                break
                                        except json.JSONDecodeError:
                                            continue
                                used_provider = "_ollama_local"
                            except Exception as exc:
                                print(f"[NOVA] Hybrid fallback local Ollama EXCEPTION: {exc}")

                    # ── Set active_model based on which sub actually answered ─
                    if used_provider == "groq":
                        active_model = nova_settings["hybrid_groq_model"] or nova_settings["groq_model"]
                    elif used_provider == "ollama_cloud":
                        active_model = nova_settings["hybrid_ollama_model"] or nova_settings["model"]
                    else:
                        active_model = nova_settings["model"]

                    complete_reply = "".join(full_reply).strip()
                    if complete_reply:
                        append_message(session_id, "nova", complete_reply)
                        _safe_memory(memory.record_conversation)
                        _safe_memory(memory.extract_and_store, user_message, complete_reply, active_model, _build_provider_config())
                    yield f"data: {json.dumps({'done': True, 'session_id': session_id, 'model': active_model})}\n\n"

                elif use_groq:
                    # ─── Groq Cloud streaming (OpenAI SSE format) ────────
                    groq_messages = build_groq_messages(history)
                    groq_payload = {
                        "model": nova_settings["groq_model"],
                        "messages": groq_messages,
                        "temperature": nova_settings["temperature"],
                        "max_tokens": nova_settings["num_predict"],
                        "top_p": nova_settings["top_p"],
                        "stream": True,
                    }
                    groq_headers = {
                        "Authorization": f"Bearer {nova_settings['groq_api_key']}",
                        "Content-Type": "application/json",
                    }
                    _groq_timeout = 8 if NOVA_ENV == "production" else 30
                    with requests.post(GROQ_API_URL, json=groq_payload,
                                       headers=groq_headers, stream=True, timeout=_groq_timeout) as r:
                        for line in r.iter_lines():
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
                                    yield f"data: {json.dumps({'token': token})}\n\n"
                            except (json.JSONDecodeError, IndexError, KeyError):
                                continue
                    complete_reply = "".join(full_reply).strip()
                    if complete_reply:
                        append_message(session_id, "nova", complete_reply)
                        _safe_memory(memory.record_conversation)
                        _safe_memory(memory.extract_and_store,
                                       user_message, complete_reply, active_model, _build_provider_config())
                    yield f"data: {json.dumps({'done': True, 'session_id': session_id})}\n\n"

                elif use_ollama_cloud:
                    # ─── Ollama Cloud streaming ──────────────────────────
                    with _call_ollama_cloud(full_prompt, stream=True) as r:
                        for line in r.iter_lines():
                            if not line:
                                continue
                            try:
                                chunk = json.loads(line)
                                token = chunk.get("response", "")
                                if token:
                                    full_reply.append(token)
                                    yield f"data: {json.dumps({'token': token})}\n\n"
                                if chunk.get("done"):
                                    complete_reply = "".join(full_reply).strip()
                                    append_message(session_id, "nova", complete_reply)
                                    _safe_memory(memory.record_conversation)
                                    _safe_memory(memory.extract_and_store,
                                                   user_message, complete_reply, active_model, _build_provider_config())
                                    yield f"data: {json.dumps({'done': True, 'session_id': session_id})}\n\n"
                                    break
                            except json.JSONDecodeError:
                                continue


                else:
                    # ─── Ollama Local streaming (only in local ENV) ────────
                    if NOVA_ENV != "local":
                        _no_provider_err = json.dumps({"error": "No AI provider available. Configure Groq or Ollama Cloud."})
                        yield f"data: {_no_provider_err}\n\n"
                        return
                    with _call_ollama_local(full_prompt, stream=True) as r:
                        for line in r.iter_lines():
                            if not line:
                                continue
                            try:
                                chunk = json.loads(line)
                                token = chunk.get("response", "")
                                if token:
                                    full_reply.append(token)
                                    yield f"data: {json.dumps({'token': token})}\n\n"
                                if chunk.get("done"):
                                    complete_reply = "".join(full_reply).strip()
                                    append_message(session_id, "nova", complete_reply)
                                    _safe_memory(memory.record_conversation)
                                    _safe_memory(memory.extract_and_store,
                                                   user_message, complete_reply, active_model, _build_provider_config())
                                    yield f"data: {json.dumps({'done': True, 'session_id': session_id})}\n\n"
                                    break
                            except json.JSONDecodeError:
                                continue
            except requests.exceptions.ConnectionError:
                yield f"data: {json.dumps({'error': 'Cannot reach AI engine.'})}\n\n"
            except requests.exceptions.Timeout:
                yield f"data: {json.dumps({'error': 'AI engine timed out.'})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── Memory Routes ────────────────────────────────────────────────────────────

@app.route("/memory", methods=["GET"])
def get_memory():
    """Return all of Nova's learned memory as JSON."""
    return jsonify(memory.get_stats())


@app.route("/memory/reset", methods=["POST"])
def reset_memory():
    """Wipe all learned memory."""
    memory.reset()
    return jsonify({"status": "ok", "message": "Nova's memory has been reset."})


@app.route("/memory/summary", methods=["POST"])
def generate_summary():
    """Trigger a daily summary for a specific session."""
    data = request.get_json() or {}
    session_id = data.get("session_id") or request.headers.get("X-Session-Id", "default")
    history = get_history(session_id)
    if history:
        memory.generate_daily_summary(history, nova_settings["model"], _build_provider_config())
        return jsonify({"status": "ok", "message": "Daily summary generation started."})
    return jsonify({"status": "skipped", "message": "No conversation history for this session."})


# ─── Settings Route ────────────────────────────────────────────────────────────
@app.route("/settings", methods=["GET", "POST"])
def settings():
    global nova_settings
    if request.method == "GET":
        # Mask API keys — show last 4 chars only
        def _mask(key):
            return ("****" + key[-4:]) if key else ""
        return jsonify({
            "model":                nova_settings["model"],
            "temperature":          nova_settings["temperature"],
            "top_p":                nova_settings["top_p"],
            "num_predict":          nova_settings["num_predict"],
            "system_prompt":        nova_settings["system_prompt"],
            "provider":             nova_settings["provider"],
            "groq_api_key":         _mask(nova_settings["groq_api_key"]),
            "groq_model":           nova_settings["groq_model"],
            "ollama_api_key":       _mask(nova_settings["ollama_api_key"]),
            "ollama_cloud_url":     nova_settings["ollama_cloud_url"],
            "hybrid_ollama_model":  nova_settings["hybrid_ollama_model"],
            "hybrid_groq_model":    nova_settings["hybrid_groq_model"],
            # ── Live Mode flag — tells UI to hide Ollama Local option ───────
            "live_mode":            NOVA_LIVE_MODE,
        })

    data = request.get_json() or {}
    if "model" in data:
        nova_settings["model"] = str(data["model"])
    if "temperature" in data:
        nova_settings["temperature"] = max(0.0, min(2.0, float(data["temperature"])))
    if "top_p" in data:
        nova_settings["top_p"] = max(0.0, min(1.0, float(data["top_p"])))
    if "num_predict" in data:
        nova_settings["num_predict"] = max(64, min(4096, int(data["num_predict"])))
    if "system_prompt" in data:
        sp = data["system_prompt"].strip()
        nova_settings["system_prompt"] = sp if sp else DEFAULT_SYSTEM_PROMPT
    if "provider" in data:
        requested = data["provider"]
        allowed = ("ollama_cloud", "groq", "hybrid") if NOVA_LIVE_MODE else ("ollama", "ollama_cloud", "groq", "hybrid")
        if requested in allowed:
            nova_settings["provider"] = requested
        elif NOVA_LIVE_MODE and requested == "ollama":
            print(f"[NOVA] Rejected provider 'ollama' — live mode is active")
    if "groq_api_key" in data:
        key = str(data["groq_api_key"]).strip()
        if key and not key.startswith("****"):
            nova_settings["groq_api_key"] = key
    if "groq_model" in data:
        nova_settings["groq_model"] = str(data["groq_model"])
    if "ollama_api_key" in data:
        key = str(data["ollama_api_key"]).strip()
        if key and not key.startswith("****"):
            nova_settings["ollama_api_key"] = key
    if "ollama_cloud_url" in data:
        url = str(data["ollama_cloud_url"]).strip()
        if url:
            nova_settings["ollama_cloud_url"] = url
    if "hybrid_ollama_model" in data:
        nova_settings["hybrid_ollama_model"] = str(data["hybrid_ollama_model"])
    if "hybrid_groq_model" in data:
        nova_settings["hybrid_groq_model"] = str(data["hybrid_groq_model"])

    return jsonify({"status": "ok", "settings": {
        "provider":             nova_settings["provider"],
        "model":                nova_settings["model"],
        "groq_model":           nova_settings["groq_model"],
        "ollama_cloud_url":     nova_settings["ollama_cloud_url"],
        "hybrid_ollama_model":  nova_settings["hybrid_ollama_model"],
        "hybrid_groq_model":    nova_settings["hybrid_groq_model"],
    }})


# ─── Reset Conversation (by session) ─────────────────────────────────────────
@app.route("/reset", methods=["POST"])
def reset_conversation():
    data = request.get_json() or {}
    session_id = data.get("session_id") or request.headers.get("X-Session-Id")
    if session_id:
        # Generate daily summary before resetting
        history = get_history(session_id)
        if history:
            _safe_memory(memory.generate_daily_summary, history, nova_settings["model"], _build_provider_config())
        _db.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        _db.commit()
    else:
        # Reset all sessions
        _db.execute("DELETE FROM sessions")
        _db.commit()
    return jsonify({"status": "Conversation reset."})


# ─── List Available Models ─────────────────────────────────────────────────
@app.route("/models")
def list_models():
    provider = nova_settings["provider"]
    if provider == "groq":
        groq_models = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "llama3-70b-8192",
            "llama3-8b-8192",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ]
        return jsonify({"models": groq_models, "provider": "groq"})
    if provider == "ollama_cloud":
        # Try to fetch model list from the cloud endpoint if it supports /api/tags
        try:
            tags_url = nova_settings["ollama_cloud_url"].replace("/api/generate", "/api/tags")
            cloud_headers = {"Authorization": f"Bearer {nova_settings['ollama_api_key']}"}
            r = requests.get(tags_url, headers=cloud_headers, timeout=8)
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                if models:
                    return jsonify({"models": models, "provider": "ollama_cloud"})
        except Exception:
            pass
        # Fallback to common open models if endpoint doesn't expose /api/tags
        return jsonify({"models": ["mistral", "llama3", "gemma2", "phi3", "deepseek-r1"], "provider": "ollama_cloud"})
    # Local Ollama (only in local ENV)
    if NOVA_ENV == "local":
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=5)
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                return jsonify({"models": models, "provider": "ollama"})
        except Exception:
            pass
    return jsonify({"models": ["mistral", "llama3", "gemma2", "phi3", "codellama"], "provider": "ollama"})


# ─── System Monitoring ────────────────────────────────────────────────────────
@app.route("/system")
def system_status():
    mem = psutil.virtual_memory()
    payload = {
        "cpu": psutil.cpu_percent(interval=0),
        "cpu_cores_logical":  CPU_CORES_LOGICAL,
        "cpu_cores_physical": CPU_CORES_PHYSICAL,
        "cpu_threads_per_core": CPU_CORES_LOGICAL // max(1, CPU_CORES_PHYSICAL),
        "memory": mem.percent,
        "memory_used_gb": round(mem.used / (1024 ** 3), 2),
        "memory_total_gb": round(mem.total / (1024 ** 3), 2),
        "tasks": len(psutil.pids()),
        "bg_pool_workers": _BG_WORKERS,
        "gpu": GPU_INFO,
    }
    return jsonify(payload)


# ─── Neural TTS Route (Microsoft Edge TTS) ────────────────────────────────────
# Default voice — loaded from .env
DEFAULT_TTS_VOICE = os.getenv("TTS_DEFAULT_VOICE", "en-IN-NeerjaExpressiveNeural")

@app.route("/tts", methods=["POST"])
def tts():
    """Generate realistic neural speech via Microsoft Edge TTS."""
    try:
        data = request.get_json() or {}
        text = data.get("text", "").strip()
        voice = data.get("voice", DEFAULT_TTS_VOICE)
        rate  = data.get("rate",  "+0%")    # e.g. '+10%' to speed up
        pitch = data.get("pitch", "+0Hz")   # e.g. '+5Hz' for higher pitch

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # edge-tts is async — run in a fresh event loop
        async def _generate():
            communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
            audio_buf = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_buf.write(chunk["data"])
            audio_buf.seek(0)
            return audio_buf
        # edge-tts is async — run in a dedicated event loop
        loop = asyncio.new_event_loop()
        try:
            audio_buf = loop.run_until_complete(_generate())
        finally:
            loop.close()
        audio_bytes = audio_buf.read()

        if not audio_bytes:
            return jsonify({"error": "TTS generated no audio"}), 500

        return Response(
            audio_bytes,
            mimetype="audio/mpeg",
            headers={
                "Content-Length": str(len(audio_bytes)),
                "Cache-Control": "no-cache",
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/tts/voices", methods=["GET"])
def tts_voices():
    """List available Edge TTS English voices."""
    async def _list():
        return await edge_tts.list_voices()
    try:
        loop = asyncio.new_event_loop()
        try:
            voices = loop.run_until_complete(_list())
        finally:
            loop.close()
        en_voices = [
            {"name": v["Name"], "short": v["ShortName"], "lang": v["Locale"],
             "gender": v["Gender"]}
            for v in voices if v["Locale"].startswith("en")
        ]
        return jsonify({"voices": en_voices})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port  = int(os.getenv("FLASK_PORT", 5000))
    host  = os.getenv("FLASK_HOST", "0.0.0.0")
    # Scale Waitress threads to match logical CPU cores (min 4, max 32)
    wsgi_threads = max(4, min(32, CPU_CORES_LOGICAL))

    try:
        from waitress import serve
        print(f"")
        print(f"  ███╗   ██╗ ██████╗ ██╗   ██╗ █████╗ ")
        print(f"  ████╗  ██║██╔═══██╗██║   ██║██╔══██╗")
        print(f"  ██╔██╗ ██║██║   ██║██║   ██║███████║")
        print(f"  ██║╚██╗██║██║   ██║╚██╗ ██╔╝██╔══██║")
        print(f"  ██║ ╚████║╚██████╔╝ ╚████╔╝ ██║  ██║")
        print(f"  ╚═╝  ╚═══╝ ╚═════╝   ╚═══╝  ╚═╝  ╚═╝")
        print(f"")
        print(f"  [NOVA] Waitress WSGI Server  (production mode)")
        print(f"  [NOVA] Listening on   http://{host}:{port}")
        print(f"  [NOVA] Open           http://localhost:{port}")
        print(f"  [NOVA] WSGI threads   {wsgi_threads}  (= logical CPU cores)")
        print(f"  [NOVA] BG pool        {_BG_WORKERS} workers  (memory/learning tasks)")
        print(f"  [NOVA] GPU            {'Yes — ' + GPU_INFO['name'] if GPU_INFO else 'None (CPU-only)'}")
        print(f"  [NOVA] Channel timeout  120s")
        print(f"")
        serve(
            app,
            host=host,
            port=port,
            threads=wsgi_threads,    # auto-scaled to CPU core count
            channel_timeout=120,     # keep-alive for SSE streams
            cleanup_interval=30,
            ident="NOVA/Waitress",
        )
    except ImportError:
        print(f"[NOVA] Waitress not installed — falling back to Flask dev server")
        print(f"[NOVA] Install with:  pip install waitress")
        app.run(host=host, port=port, debug=False,
                threaded=True)   # Flask uses its own thread pool when threaded=True

