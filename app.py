from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
import requests
import psutil
import json
import asyncio
import io
import os
import subprocess
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

# ─── CORS (allow all origins for local dev) ──────────────────────────────────
@app.after_request
def add_cors(response):
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

OLLAMA_URL  = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
# MAX_HISTORY is defined below with the SQLite session helper
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# ─── Live Mode — when True, Ollama Local is completely disabled ───────────────
# Set NOVA_LIVE_MODE=true in .env when deploying to a cloud/live server.
# In this mode, only Ollama Cloud and Groq Cloud are available as providers.
NOVA_LIVE_MODE = os.getenv("NOVA_LIVE_MODE", "false").lower() in ("true", "1", "yes")
if NOVA_LIVE_MODE:
    print("[NOVA] Live Mode ENABLED — Ollama Local is disabled. Only cloud providers are active.")

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

# ─── Ollama Cloud Sanity Check ────────────────────────────────────────────────
_FAKE_OLLAMA_DOMAINS = ("api.ollama.com",)

def _ollama_cloud_configured() -> bool:
    """
    Return True only when we have a non-empty Ollama API key AND the cloud URL
    does NOT point to a known fake/non-existent public endpoint.
    """
    key = nova_settings.get("ollama_api_key", "")
    url = nova_settings.get("ollama_cloud_url", "")
    if not key:
        return False
    if any(domain in url for domain in _FAKE_OLLAMA_DOMAINS):
        return False
    return True

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
    return requests.post(cloud_url, json=payload, headers=headers,
                         stream=stream, timeout=120)


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
    return requests.post(GROQ_API_URL, json=payload, headers=headers,
                         stream=stream, timeout=60)


# ─── Serve Frontend ───────────────────────────────────────────────────────────
@app.route("/")
def serve_index():
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
        # Guard: if hybrid/ollama_cloud is selected but the Ollama Cloud URL
        # is the default placeholder (api.ollama.com) which is not a real API,
        # automatically redirect to Groq or Ollama local.
        if provider in ("hybrid", "ollama_cloud") and not _ollama_cloud_configured():
            if nova_settings["groq_api_key"]:
                print(f"[NOVA] {provider} → Ollama Cloud not configured; falling back to Groq")
                provider = "groq"
            elif not NOVA_LIVE_MODE:
                provider = "ollama"  # last resort: try local (only when NOT in live mode)
            else:
                return jsonify({"error": "No cloud AI provider configured. Set a Groq API key or a valid Ollama Cloud URL.", "code": "NO_CLOUD_PROVIDER"}), 503

        if provider == "hybrid":
            # Hybrid: route by complexity
            sub = _classify_query(user_message)
            print(f"[NOVA] Hybrid → {sub} ({len(user_message.split())} words)")
            ai_response = ""
            if sub == "groq" and nova_settings["groq_api_key"]:
                try:
                    r = _call_groq(history)
                    if r.status_code == 200:
                        ai_response = r.json()["choices"][0]["message"]["content"].strip()
                    else:
                        print(f"[NOVA] Hybrid Groq failed ({r.status_code}), falling back to Ollama Cloud")
                except Exception as exc:
                    print(f"[NOVA] Hybrid Groq exception: {exc}, falling back to Ollama Cloud")
            # Fallback to Ollama Cloud (also used for simple queries)
            if not ai_response:
                r = _call_ollama_cloud(full_prompt)
                if r.status_code != 200:
                    return jsonify({"error": "Ollama Cloud error in hybrid mode", "code": "HYBRID_ERROR"}), 500
                ai_response = r.json().get("response", "").strip()

        elif provider == "groq" and nova_settings["groq_api_key"]:
            # Groq Cloud only
            r = _call_groq(history, model_override=nova_settings["groq_model"])
            if r.status_code != 200:
                err_detail = r.json().get("error", {}).get("message", "Groq API error")
                return jsonify({"error": err_detail, "code": "GROQ_ERROR"}), 500
            ai_response = r.json()["choices"][0]["message"]["content"].strip()

        elif provider == "ollama_cloud" and nova_settings["ollama_api_key"]:
            # Ollama Cloud only
            r = _call_ollama_cloud(full_prompt)
            if r.status_code != 200:
                return jsonify({"error": f"Ollama Cloud error: {r.text[:200]}", "code": "OLLAMA_CLOUD_ERROR"}), 500
            ai_response = r.json().get("response", "").strip()

        else:
            # Ollama Local API (default fallback)
            payload = {
                "model": nova_settings["model"],
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature":    nova_settings["temperature"],
                    "top_p":          nova_settings["top_p"],
                    "num_predict":    nova_settings["num_predict"],
                    "num_ctx":        2048,
                    "num_thread":     CPU_CORES_PHYSICAL,
                    "repeat_penalty": 1.1,
                }
            }
            response = requests.post(OLLAMA_URL, json=payload, timeout=120)
            if response.status_code != 200:
                return jsonify({"error": "AI engine error", "code": "OLLAMA_ERROR"}), 500
            ai_response = response.json().get("response", "").strip()

        if not ai_response:
            ai_response = "I'm not sure how to respond to that. Could you rephrase?"

        # Persist Nova turn to SQLite
        append_message(session_id, "nova", ai_response)

        # ── Learning: offload to background thread pool (non-blocking) ───────
        bg_pool.submit(memory.record_conversation)
        active_model = nova_settings["groq_model"] if provider == "groq" else nova_settings["model"]
        bg_pool.submit(memory.extract_and_store, user_message, ai_response, active_model)

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
        # Guard: redirect hybrid/ollama_cloud to groq if Ollama Cloud is misconfigured
        if provider in ("hybrid", "ollama_cloud") and not _ollama_cloud_configured():
            if nova_settings["groq_api_key"]:
                print(f"[NOVA] Stream {provider} → Ollama Cloud not configured; falling back to Groq")
                provider = "groq"
            elif not NOVA_LIVE_MODE:
                provider = "ollama"  # last resort: try local (only when NOT in live mode)
            else:
                def _err_gen():
                    yield f"data: {json.dumps({'error': 'No cloud AI provider configured. Set a Groq API key or a valid Ollama Cloud URL.'})}\n\n"
                return Response(stream_with_context(_err_gen()), mimetype="text/event-stream",
                                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        use_groq = provider == "groq" and nova_settings["groq_api_key"]
        use_ollama_cloud = provider == "ollama_cloud"  # key check is in _call_ollama_cloud
        use_hybrid = provider == "hybrid"              # key check is in _call_ollama_cloud
        # Determine the label for the active model
        if use_groq:
            active_model = nova_settings["groq_model"]
        elif use_hybrid:
            active_model = nova_settings["hybrid_ollama_model"] or nova_settings["model"] or "mistral"
        else:
            active_model = nova_settings["model"]

        def generate():
            full_reply = []
            try:
                if use_hybrid:
                    # ─── Hybrid streaming: classify → route → stream ──────────────
                    sub = _classify_query(user_message)
                    print(f"[NOVA] Hybrid stream → {sub}")
                    used_cloud = False
                    if sub == "groq" and nova_settings["groq_api_key"]:
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
                            used_cloud = True
                        except Exception as exc:
                            print(f"[NOVA] Hybrid stream Groq failed: {exc}, falling back to Ollama Cloud")
                            full_reply = []
                    if not used_cloud:
                        # Simple query OR Groq fallback → always Ollama Cloud
                        with _call_ollama_cloud(full_prompt, stream=True) as r:
                            if r.status_code != 200:
                                # Emit readable error so the bubble isn't blank
                                err_body = r.text[:300]
                                print(f"[NOVA] Hybrid Ollama Cloud error {r.status_code}: {err_body}")
                                err_msg = f"Ollama Cloud error ({r.status_code}): {err_body}"
                                yield f"data: {json.dumps({'token': err_msg})}\n\n"
                                full_reply.append(err_msg)
                            else:
                                for line in r.iter_lines():
                                    if not line:
                                        continue
                                    try:
                                        chunk = json.loads(line)
                                        # Check if Ollama returned an error JSON
                                        if chunk.get("error"):
                                            err_msg = f"Ollama Cloud model error: {chunk['error']}"
                                            print(f"[NOVA] {err_msg}")
                                            yield f"data: {json.dumps({'token': err_msg})}\n\n"
                                            full_reply.append(err_msg)
                                            break
                                        token = chunk.get("response", "")
                                        if token:
                                            full_reply.append(token)
                                            yield f"data: {json.dumps({'token': token})}\n\n"
                                        if chunk.get("done"):
                                            break
                                    except json.JSONDecodeError:
                                        continue
                    complete_reply = "".join(full_reply).strip()
                    if complete_reply:
                        append_message(session_id, "nova", complete_reply)
                        bg_pool.submit(memory.record_conversation)
                        bg_pool.submit(memory.extract_and_store,
                                       user_message, complete_reply, active_model)
                    yield f"data: {json.dumps({'done': True, 'session_id': session_id})}\n\n"

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
                    with requests.post(GROQ_API_URL, json=groq_payload,
                                       headers=groq_headers, stream=True, timeout=60) as r:
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
                        bg_pool.submit(memory.record_conversation)
                        bg_pool.submit(memory.extract_and_store,
                                       user_message, complete_reply, active_model)
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
                                    bg_pool.submit(memory.record_conversation)
                                    bg_pool.submit(memory.extract_and_store,
                                                   user_message, complete_reply, active_model)
                                    yield f"data: {json.dumps({'done': True, 'session_id': session_id})}\n\n"
                                    break
                            except json.JSONDecodeError:
                                continue

                else:
                    # ─── Ollama Local streaming ──────────────────────────
                    payload = {
                        "model": nova_settings["model"],
                        "prompt": full_prompt,
                        "stream": True,
                        "options": {
                            "temperature":    nova_settings["temperature"],
                            "top_p":          nova_settings["top_p"],
                            "num_predict":    nova_settings["num_predict"],
                            "num_ctx":        2048,
                            "num_thread":     CPU_CORES_PHYSICAL,
                            "repeat_penalty": 1.1,
                        }
                    }
                    with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=120) as r:
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
                                    bg_pool.submit(memory.record_conversation)
                                    bg_pool.submit(memory.extract_and_store,
                                                   user_message, complete_reply, active_model)
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
        memory.generate_daily_summary(history, nova_settings["model"])
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
            bg_pool.submit(memory.generate_daily_summary, history, nova_settings["model"])
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
    # Local Ollama
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
        "cpu": psutil.cpu_percent(interval=0.5),
        "cpu_cores_logical":  CPU_CORES_LOGICAL,
        "cpu_cores_physical": CPU_CORES_PHYSICAL,
        "cpu_threads_per_core": CPU_CORES_LOGICAL // max(1, CPU_CORES_PHYSICAL),
        "memory": mem.percent,
        "memory_used_gb": round(mem.used / (1024 ** 3), 2),
        "memory_total_gb": round(mem.total / (1024 ** 3), 2),
        "tasks": len(psutil.pids()),
        "bg_pool_workers": _BG_WORKERS,
        "gpu": None,
    }
    # Refresh GPU stats each request (fast call)
    gpu = _detect_gpu()
    if gpu:
        payload["gpu"] = gpu
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

