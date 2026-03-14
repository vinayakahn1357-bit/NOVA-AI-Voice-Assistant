# 🌌 NOVA — AI Voice Assistant

> **A locally-powered, conversational AI assistant with neural Text-to-Speech, persistent memory, and a sleek futuristic UI.**

NOVA is a self-hosted AI assistant built on top of [Ollama](https://ollama.com/) LLMs (like Mistral, LLaMA 3, Gemma 2) with a Flask backend and a beautiful browser-based frontend. It supports real-time streaming responses, Microsoft Edge neural TTS voices, per-session conversation history, and a long-term memory engine that learns about you over time.

---

## ✨ Features

- 🧠 **Persistent Memory** — Nova remembers facts, interests, and context across conversations
- 🗣️ **Neural Text-to-Speech** — Powered by Microsoft Edge TTS (`edge-tts`) with realistic voices (Indian English, anime, and more)
- ⚡ **Streaming Responses** — Token-by-token streaming via Server-Sent Events (SSE)
- 🔁 **Multi-Session Support** — Independent conversation histories per session
- 🎛️ **Configurable Settings** — Change model, temperature, top-p, max tokens, and system prompt at runtime via API
- 📊 **System Monitoring** — Live CPU/RAM stats endpoint
- 🌐 **Responsive Frontend** — Glassmorphism UI with animations, voice playback, and volume control
- 🔒 **Fully Local** — Your data never leaves your machine (Ollama runs offline)

---

## 🖥️ Tech Stack

| Layer     | Technology                            |
|-----------|---------------------------------------|
| Backend   | Python 3.11+, Flask 3.x               |
| AI Engine | [Ollama](https://ollama.com/) (local LLM runner) |
| TTS       | `edge-tts` (Microsoft Edge Neural TTS)|
| Monitoring| `psutil`                              |
| Frontend  | Vanilla HTML, CSS, JavaScript         |

---

## 📁 Project Structure

```
NOVAbackend/
├── app.py              # Flask application — API routes & backend logic
├── nova_memory.py      # Long-term memory engine (facts, interests, summaries)
├── nova_memory.json    # Persisted memory store (auto-generated)
├── requirements.txt    # Python dependencies
└── Frontend/
    ├── index.html      # Main UI
    └── static/
        ├── script.js   # Frontend logic (chat, TTS, streaming)
        └── Style.css   # Futuristic glassmorphism styling
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.11+**
- **[Ollama](https://ollama.com/)** installed and running locally
- At least one Ollama model pulled (e.g., `mistral`)

```bash
# Install Ollama (Windows/macOS/Linux)
# Visit https://ollama.com/download

# Pull the default model
ollama pull mistral
```

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/NOVAbackend.git
cd NOVAbackend

# 2. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Running NOVA

```bash
# Make sure Ollama is running first
ollama serve

# In a separate terminal, start the Flask server
python app.py
```

Open your browser at **[http://localhost:5000](http://localhost:5000)** 🎉

---

## 🔌 API Reference

| Method | Endpoint          | Description                                  |
|--------|-------------------|----------------------------------------------|
| `POST` | `/chat`           | Send a message, get a full response           |
| `POST` | `/chat/stream`    | Send a message, get a token-by-token SSE stream |
| `GET`  | `/memory`         | View Nova's current learned memory            |
| `POST` | `/memory/reset`   | Clear all learned memory                      |
| `POST` | `/memory/summary` | Trigger a daily summary for a session         |
| `GET`  | `/settings`       | Get current model settings                    |
| `POST` | `/settings`       | Update model, temperature, top-p, etc.        |
| `POST` | `/reset`          | Reset conversation history for a session      |
| `GET`  | `/models`         | List available Ollama models                  |
| `GET`  | `/system`         | Get live CPU & RAM usage stats                |
| `POST` | `/tts`            | Generate neural speech audio (MP3)            |
| `GET`  | `/tts/voices`     | List available Edge TTS English voices        |

### Example: Chat Request

```json
POST /chat
{
  "message": "Hello Nova, how are you?",
  "session_id": "user-123"
}
```

### Example: TTS Request

```json
POST /tts
{
  "text": "Hello! I am Nova, your AI assistant.",
  "voice": "en-IN-NeerjaExpressiveNeural",
  "rate": "+0%",
  "pitch": "+0Hz"
}
```

---

## 🎛️ Supported Ollama Models

NOVA works with any model available in Ollama. The default is **Mistral**. You can switch models via the `/settings` endpoint or the UI.

Recommended models:
- `mistral` — Fast, well-rounded (default)
- `llama3` — Strong reasoning
- `gemma2` — Lightweight and capable
- `phi3` — Very fast on low-end hardware
- `codellama` — Code-focused tasks

---

## 🧠 Memory Engine

NOVA's memory engine (`nova_memory.py`) automatically:
- Extracts **facts** about the user (name, location, preferences)
- Tracks **interests** and **topics** discussed
- Generates **daily conversation summaries**
- Injects relevant context into every prompt for personalised responses

Memory is stored locally in `nova_memory.json` and persists across server restarts.

---

## ⚙️ Configuration

You can update Nova's behaviour at runtime by posting to `/settings`:

```json
POST /settings
{
  "model": "llama3",
  "temperature": 0.7,
  "top_p": 0.9,
  "num_predict": 1024,
  "system_prompt": "You are Nova, a friendly AI assistant..."
}
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- [Ollama](https://ollama.com/) for making local LLMs accessible
- [edge-tts](https://github.com/rany2/edge-tts) for Microsoft Edge neural TTS
- [Flask](https://flask.palletsprojects.com/) for the lightweight Python web framework

---

<p align="center">Made with ❤️ by Vinayaka H N</p>
