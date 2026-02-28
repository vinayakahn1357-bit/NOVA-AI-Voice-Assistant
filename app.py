from flask import Flask, request, jsonify, send_from_directory
import requests
import psutil

app = Flask(
    __name__,
    static_folder="Frontend/static",
    static_url_path="/static"
)

app.config["JSON_SORT_KEYS"] = False
conversation_history = [
    "You are Nova, an intelligent, futuristic AI assistant. Speak naturally, confidently, and conversationally. Avoid robotic responses."
]
OLLAMA_URL = "http://localhost:11434/api/generate"

# Conversation history storage
conversation_history = []

# ------------------------
# Serve Frontend
# ------------------------
@app.route("/")
def serve_index():
    return send_from_directory("Frontend", "index.html")


# ------------------------
# Chat Route
# ------------------------
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()

        if not data or "message" not in data:
            return jsonify({"error": "No message provided"}), 400

        user_message = data.get("message", "")

        # Add user message to memory
        conversation_history.append(f"User: {user_message}")

        # Keep last 10 exchanges only
        if len(conversation_history) > 10:
            conversation_history.pop(0)

        # Create full conversation prompt
        full_prompt = "\n".join(conversation_history) + "\nNova:"

        payload = {
            "model": "mistral",
            "prompt": full_prompt,
            "stream": False
        }
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)

        if response.status_code != 200:
            return jsonify({"error": "Ollama server error"}), 500

        ai_response = response.json().get("response", "No response from model.")

        conversation_history.append(f"Nova: {ai_response}")

        return jsonify({"reply": ai_response})

    except requests.exceptions.RequestException:
        return jsonify({"error": "Unable to connect to AI engine"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------------
# System Monitoring Route
# ------------------------
@app.route("/system")
def system_status():
    mem = psutil.virtual_memory()
    return jsonify({
        "cpu": psutil.cpu_percent(interval=0.5),
        "memory": mem.percent,
        "memory_used_gb": mem.used / (1024 ** 3),
        "memory_total_gb": mem.total / (1024 ** 3),
        "tasks": len(psutil.pids())
    })
# ------------------------
# Run App
# ------------------------
if __name__ == "__main__":
    app.run(port=5000, debug=True)