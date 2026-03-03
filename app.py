from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
import requests
import psutil
import json
import uuid
import time
import asyncio
import io
import os
import edge_tts
from dotenv import load_dotenv
from nova_memory import NovaMemory

# ─── Load .env ────────────────────────────────────────────────────────────────
load_dotenv()

app = Flask(
    __name__,
    static_folder="Frontend/static",
    static_url_path="/static"
)
app.config["JSON_SORT_KEYS"] = False

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
print(f"[NOVA] Memory loaded — {len(memory._state['facts'])} facts, "
      f"{len(memory._state['interests'])} interests known.")

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
MAX_HISTORY = int(os.getenv("NOVA_MAX_HISTORY", "30"))

# ─── Global Settings (can be changed via /settings) ──────────────────────────
nova_settings = {
    "model":       os.getenv("OLLAMA_DEFAULT_MODEL", "mistral"),
    "temperature": float(os.getenv("NOVA_TEMPERATURE", "0.75")),
    "top_p":       float(os.getenv("NOVA_TOP_P", "0.9")),
    "num_predict": int(os.getenv("NOVA_MAX_TOKENS", "1024")),
    "system_prompt": DEFAULT_SYSTEM_PROMPT,
}

# ─── Per-Session Conversation History ─────────────────────────────────────────
session_histories = {}


def get_history(session_id: str) -> list:
    if session_id not in session_histories:
        session_histories[session_id] = []
    return session_histories[session_id]


def build_prompt(history: list) -> str:
    """Build the full prompt string from history for Ollama, injecting memory."""
    base_prompt = nova_settings["system_prompt"].strip()
    # Inject accumulated memory context right after the system prompt
    memory_ctx = memory.get_memory_context()
    lines = [base_prompt + memory_ctx]
    for msg in history:
        role = "User" if msg["role"] == "user" else "Nova"
        lines.append(f"{role}: {msg['content']}")
    lines.append("Nova:")
    return "\n".join(lines)


# ─── Serve Frontend ───────────────────────────────────────────────────────────
@app.route("/")
def serve_index():
    return send_from_directory("Frontend", "index.html")


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
        history = get_history(session_id)

        # Add user turn
        history.append({"role": "user", "content": user_message})
        while len(history) > MAX_HISTORY:
            history.pop(0)

        full_prompt = build_prompt(history)

        payload = {
            "model": nova_settings["model"],
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": nova_settings["temperature"],
                "top_p": nova_settings["top_p"],
                "num_predict": nova_settings["num_predict"],
            }
        }

        response = requests.post(OLLAMA_URL, json=payload, timeout=120)

        if response.status_code != 200:
            return jsonify({"error": "AI engine error", "code": "OLLAMA_ERROR"}), 500

        ai_response = response.json().get("response", "").strip()
        if not ai_response:
            ai_response = "I'm not sure how to respond to that. Could you rephrase?"

        # Add Nova turn
        history.append({"role": "nova", "content": ai_response})

        # ── Learning: record turn + extract facts in background ──────────────
        memory.record_conversation()
        memory.extract_and_store(user_message, ai_response, nova_settings["model"])

        return jsonify({
            "reply": ai_response,
            "session_id": session_id,
            "model": nova_settings["model"],
        })

    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Cannot reach AI engine. Is Ollama running?", "code": "CONNECTION_ERROR"}), 503
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
        history = get_history(session_id)

        history.append({"role": "user", "content": user_message})
        while len(history) > MAX_HISTORY:
            history.pop(0)

        full_prompt = build_prompt(history)

        payload = {
            "model": nova_settings["model"],
            "prompt": full_prompt,
            "stream": True,
            "options": {
                "temperature": nova_settings["temperature"],
                "top_p": nova_settings["top_p"],
                "num_predict": nova_settings["num_predict"],
            }
        }

        def generate():
            full_reply = []
            try:
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
                                history.append({"role": "nova", "content": complete_reply})
                                # ── Learning ───────────────────────────────
                                memory.record_conversation()
                                memory.extract_and_store(
                                    user_message, complete_reply, nova_settings["model"]
                                )
                                yield f"data: {json.dumps({'done': True, 'session_id': session_id})}\n\n"
                                break
                        except json.JSONDecodeError:
                            continue
            except requests.exceptions.ConnectionError:
                yield f"data: {json.dumps({'error': 'Cannot reach AI engine. Is Ollama running?'})}\n\n"
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
        return jsonify({
            "model": nova_settings["model"],
            "temperature": nova_settings["temperature"],
            "top_p": nova_settings["top_p"],
            "num_predict": nova_settings["num_predict"],
            "system_prompt": nova_settings["system_prompt"],
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
    if "system_prompt" in data and data["system_prompt"].strip():
        nova_settings["system_prompt"] = data["system_prompt"].strip()

    return jsonify({"status": "ok", "settings": nova_settings})


# ─── Reset Conversation (by session) ─────────────────────────────────────────
@app.route("/reset", methods=["POST"])
def reset_conversation():
    data = request.get_json() or {}
    session_id = data.get("session_id") or request.headers.get("X-Session-Id")
    if session_id and session_id in session_histories:
        # Generate daily summary before resetting
        history = session_histories.get(session_id, [])
        if history:
            memory.generate_daily_summary(history, nova_settings["model"])
        session_histories[session_id] = []
    elif not session_id:
        session_histories.clear()
    return jsonify({"status": "Conversation reset."})


# ─── List Available Ollama Models ─────────────────────────────────────────────
@app.route("/models")
def list_models():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            return jsonify({"models": models})
    except Exception:
        pass
    return jsonify({"models": ["mistral", "llama3", "gemma2", "phi3", "codellama"]})


# ─── System Monitoring ────────────────────────────────────────────────────────
@app.route("/system")
def system_status():
    mem = psutil.virtual_memory()
    return jsonify({
        "cpu": psutil.cpu_percent(interval=0.5),
        "memory": mem.percent,
        "memory_used_gb": round(mem.used / (1024 ** 3), 2),
        "memory_total_gb": round(mem.total / (1024 ** 3), 2),
        "tasks": len(psutil.pids()),
    })


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

        audio_buf = asyncio.run(_generate())
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
        voices = asyncio.run(_list())
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
    debug = os.getenv("FLASK_DEBUG", "true").lower() == "true"
    print(f"NOVA AI Server starting on http://localhost:{port}")
    app.run(port=port, debug=debug, threaded=True)
