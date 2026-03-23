# 🌌 NOVA — AI Voice Assistant

[![Live Demo](https://img.shields.io/badge/Live%20Demo-novaarcai.com-blue?style=for-the-badge&logo=vercel)](https://novaarcai.com)
[![Deployed on Vercel](https://img.shields.io/badge/Deployed%20on-Vercel-black?style=flat&logo=vercel)](https://novaarcai.com)

> **An intelligent AI assistant with multi-provider LLM support, agent intelligence, persistent memory, neural TTS, and a premium futuristic UI.**

🔗 **Live at [novaarcai.com](https://novaarcai.com)**

NOVA is a production-ready AI assistant with a Python/Flask backend and a browser-based frontend, **deployed on Vercel**. It supports **multiple AI providers** (Ollama Local, Ollama Cloud, Groq), features a **persistent SQLite memory engine**, an **agent intelligence system** with built-in tools, a **6-stage response pipeline**, and runs on **Vercel serverless** in production or **Waitress WSGI** locally.

---

## ✨ Features

### 🤖 AI & Intelligence
- **Multi-Provider AI** — Ollama Local, Ollama Cloud, and Groq API with automatic failover
- **Hybrid Mode** — True parallel execution of two providers with intelligent response scoring & merging
- **Agent Engine** — Decision, Opinion, Planning, and Tool modes that augment LLM responses
- **Built-in Tools** — Calculator, system info, date/time, unit conversion (no LLM needed)
- **Query Analyzer** — Classifies 13 query types with adaptive temperature/token tuning
- **Intelligent Routing** — Selects optimal provider based on query type, latency, failure rate, and cost

### 🧠 Memory & Learning
- **Persistent Memory** — SQLite-backed (WAL mode) storage for facts, interests, and preferences
- **LLM-Powered Extraction** — Automatically learns about you from every conversation
- **Daily Summaries** — Auto-generates conversation digests
- **Memory Context Injection** — Personalises every response with what NOVA knows about you

### 🔒 Security & Auth
- **Google OAuth 2.0** — One-click Google sign-in
- **Email/Password Auth** — PBKDF2-HMAC-SHA256 hashing (260k iterations)
- **RBAC** — Admin/user role-based access control
- **Rate Limiting** — 30 req/min (chat), 10 req/min (auth)
- **Prompt Injection Detection** — Input validation and response sanitization
- **Production Hardening** — Localhost blocking, secure cookies, CSRF protection

### 🎨 Frontend & UX
- **Streaming Responses** — Token-by-token SSE for real-time output
- **Neural TTS** — Microsoft Edge TTS with expressive voices
- **3D Orb UI** — Three.js-powered interactive orb with glassmorphism design
- **Responsive Design** — Works across desktop and mobile
- **Dark Mode** — Premium futuristic aesthetic

### ⚡ Performance & Reliability
- **Response Caching** — In-memory LRU cache (5 min TTL, 100 entries)
- **Circuit Breaker** — Auto-disables failing providers
- **Retry Logic** — Exponential backoff on transient failures
- **Performance Tracking** — Per-provider latency and failure rate monitoring
- **Background Processing** — Thread pool for non-blocking memory extraction

---

## 🖥️ Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.10+ |
| **Framework** | Flask ≥3.0.0 |
| **WSGI Server** | Waitress ≥2.1.0 |
| **Database** | SQLite (WAL mode) |
| **AI Providers** | Ollama (local/cloud), Groq (LLaMA 3.3 70B) |
| **Auth** | Flask sessions, Google OAuth 2.0, PBKDF2-SHA256 |
| **TTS** | Edge-TTS (Microsoft Neural voices) |
| **Frontend** | HTML/CSS/JS, Three.js |
| **Deployment** | Vercel (serverless) + Local |

---

## 📁 Project Structure

```
NOVAbackend/
├── app.py                     # Application factory & dependency wiring
├── config.py                  # Centralized configuration (env-aware)
├── nova_memory.py             # SQLite persistent memory engine
├── requirements.txt           # Python dependencies
├── vercel.json                # Vercel deployment config
│
├── controllers/               # Request orchestration
│   ├── auth_controller.py     #   Register, login, OAuth, user management
│   ├── chat_controller.py     #   Chat pipeline orchestration & streaming
│   └── settings_controller.py #   Runtime settings management
│
├── routes/                    # Flask Blueprint endpoints
│   ├── auth.py                #   /auth/* routes
│   ├── chat.py                #   /chat, /chat/stream routes
│   ├── memory.py              #   /memory/* routes
│   ├── settings.py            #   /settings routes
│   └── system.py              #   /system/status, /health routes
│
├── services/                  # Core business logic (17 services)
│   ├── ai_service.py          #   LLM provider dispatch (Ollama/Groq/Hybrid)
│   ├── agent_engine.py        #   AI agent decision engine
│   ├── hybrid_evaluator.py    #   Parallel scoring & response merging
│   ├── model_router.py        #   Intelligent provider selection
│   ├── query_analyzer.py      #   Query classification & tuning
│   ├── response_pipeline.py   #   6-stage response orchestrator
│   ├── tool_executor.py       #   Built-in tool system
│   ├── ollama_validator.py    #   Model discovery & validation
│   ├── performance_tracker.py #   Latency/failure stats & circuit breaker
│   ├── prompt_builder.py      #   System prompt construction
│   ├── cache_service.py       #   In-memory response cache
│   ├── command_service.py     #   Slash-command detection
│   ├── session_service.py     #   Session history management
│   ├── memory_service.py      #   Memory abstraction layer
│   ├── tts_service.py         #   Text-to-Speech via Edge-TTS
│   ├── image_service.py       #   Image handling
│   └── pdf_service.py         #   PDF handling
│
├── utils/                     # Cross-cutting utilities
│   ├── errors.py              #   Custom exception hierarchy
│   ├── logger.py              #   Logging configuration
│   ├── middleware.py           #   Request ID & latency headers
│   ├── response_formatter.py  #   Query-type-aware formatting
│   ├── response_sanitizer.py  #   Security sanitization
│   ├── retry_handler.py       #   Retry with exponential backoff
│   ├── security.py            #   Auth, CORS, rate limiting, RBAC
│   └── validators.py          #   Input validation
│
└── Frontend/                  # Browser-based UI
    ├── index.html             #   Main AI assistant interface
    ├── landing.html           #   Landing/marketing page
    ├── login.html             #   Auth page (email + Google)
    └── static/
        ├── script.js          #   App logic, SSE streaming, orb
        ├── Style.css          #   Design system & animations
        └── nova_logo.png      #   Brand logo
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- **[Ollama](https://ollama.com/)** (for local mode) or a **Groq API key** (for cloud mode)

### Installation

```bash
# Clone the repository
git clone https://github.com/vinayakahn1357-bit/NOVA-AI-Voice-Assistant.git
cd NOVA-AI-Voice-Assistant

# Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment config
cp .env.example .env
# Edit .env with your API keys and settings
```

### Environment Variables

```env
# Required for cloud mode
GROQ_API_KEY=your_groq_api_key

# Optional - Ollama Cloud
OLLAMA_API_KEY=your_ollama_api_key
OLLAMA_CLOUD_URL=https://your-ollama-cloud.com/api/generate

# Optional - Google OAuth
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret

# Optional - Configuration
NOVA_PROVIDER=groq            # ollama | groq | ollama_cloud | hybrid
FLASK_SECRET_KEY=your_secret
NOVA_TEMPERATURE=0.75
NOVA_MAX_TOKENS=1024
```

### Running NOVA

```bash
# Local mode with Ollama
ollama serve                  # Start Ollama in a separate terminal
ollama pull mistral           # Pull a model
python app.py                 # Start NOVA

# Cloud mode (Groq)
# Set GROQ_API_KEY in .env, then:
python app.py
```

Open **[http://localhost:5000](http://localhost:5000)** 🎉

```
  ███╗   ██╗ ██████╗ ██╗   ██╗ █████╗
  ████╗  ██║██╔═══██╗██║   ██║██╔══██╗
  ██╔██╗ ██║██║   ██║██║   ██║███████║
  ██║╚██╗██║██║   ██║╚██╗ ██╔╝██╔══██║
  ██║ ╚████║╚██████╔╝ ╚████╔╝ ██║  ██║
  ╚═╝  ╚═══╝ ╚═════╝   ╚═══╝  ╚═╝  ╚═╝
```

---

## 🔌 API Reference

### Chat

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/chat` | Send a message, get a full JSON response with metadata |
| `POST` | `/chat/stream` | Send a message, get SSE token-by-token stream |

### Auth

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/auth/register` | Register with email/password |
| `POST` | `/auth/login` | Login with email/password |
| `POST` | `/auth/logout` | Logout current session |
| `GET` | `/auth/me` | Get current authenticated user |
| `GET` | `/auth/google` | Initiate Google OAuth flow |
| `GET` | `/auth/google/callback` | Google OAuth callback |

### Memory

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/memory` | View NOVA's learned memory |
| `POST` | `/memory/reset` | Clear all learned memory |
| `POST` | `/memory/summary` | Trigger daily summary generation |

### Settings & System

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/settings` | Get current model settings |
| `POST` | `/settings` | Update model, provider, temperature, etc. |
| `GET` | `/system/status` | System health, CPU, RAM, GPU stats |
| `GET` | `/health` | Health check endpoint |

### Example: Chat Request

```json
POST /chat
{
  "message": "Compare Python vs JavaScript for backend development",
  "session_id": "user-123"
}
```

Response:
```json
{
  "reply": "Here's my analysis...",
  "session_id": "user-123",
  "meta": {
    "model": "llama-3.3-70b-versatile",
    "provider": "groq",
    "mode": "decision",
    "query_type": "reasoning",
    "complexity": 8,
    "latency_ms": 1240,
    "agent_action": "analyze_and_recommend",
    "agent_confidence": 0.85
  }
}
```

---

## 🧠 Response Pipeline

Every request flows through a **6-stage modular pipeline**:

```
Analyze → Agent → Route → Generate → Format → Sanitize → Package
```

| Stage | Service | What It Does |
|---|---|---|
| **1. Analyze** | `QueryAnalyzer` | Classify query type (13 types), estimate complexity |
| **2. Agent** | `AgentEngine` | Decide mode: normal, decision, opinion, planning, or tool |
| **3. Route** | `ModelRouter` | Select optimal provider based on 4 factors |
| **4. Generate** | `AIService` | Call LLM(s), handle failover and hybrid execution |
| **5. Format** | `ResponseFormatter` | Query-type-aware response formatting |
| **6. Sanitize** | `ResponseSanitizer` | Strip model leaks, PII, injection responses |

### Agent Modes

| Mode | Trigger | Behavior |
|---|---|---|
| **Normal** | Default | Standard LLM response |
| **Decision** | "which is better", "should I use" | Structured pros/cons + recommendation |
| **Opinion** | "what do you think" | Confident, opinionated advice |
| **Planning** | "build a", "how to create" | Step-by-step actionable plan |
| **Tool** | "calculate 5+3", "what time is it" | Execute built-in tool, skip LLM |

### Hybrid Evaluator Scoring

When both providers run in parallel, responses are scored on 5 criteria:

| Criterion | Weight |
|---|---|
| Relevance | 30% |
| Completeness | 25% |
| Clarity | 20% |
| Tone | 15% |
| Depth | 10% |

---

## 🧠 Memory Engine

NOVA's memory system (`nova_memory.py`) uses **SQLite in WAL mode** for fast, thread-safe, concurrent access:

| Table | Schema | Purpose |
|---|---|---|
| `facts` | id, fact (UNIQUE), created_at | Personal facts about the user (cap: 100) |
| `interests` | topic (PK), count | Topics with frequency tracking |
| `preferences` | key (PK), value | User preferences |
| `daily_summaries` | day (PK), summary | Daily conversation digests (last 30 days) |
| `sessions` | session_id, turn, role, content | Conversation history |
| `meta` | key (PK), value | Stats: total conversations, days active |

After each turn, the LLM extracts facts/interests/preferences in the background and merges them into the database.

---

## 🛡️ Security

- **PBKDF2-HMAC-SHA256** password hashing (260,000 iterations)
- **Constant-time** password comparison
- **Rate limiting** — 30 req/min chat, 10 req/min auth
- **CSRF protection** on OAuth flows
- **HTTP-only, SameSite** session cookies (Secure in production)
- **Prompt injection detection** and response sanitization
- **RBAC** — Admin email whitelist
- **Localhost blocking** in production mode

---

## 🌐 Deployment

### Vercel (Production) — Live at [novaarcai.com](https://nova-ai-voice-assistant-fawn.vercel.app/)

NOVA is **currently deployed and running** on Vercel with a custom domain.

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel --prod
```

**Production configuration** (`vercel.json`):
- Runtime: `@vercel/python` (serverless)
- Max duration: 60 seconds
- Memory: 1 GB
- SSE streaming: enabled (`X-Accel-Buffering: no`)
- Static assets: 1-year immutable cache
- Custom domain: `novaarcai.com`

**Production environment**:
- Ollama Local is **disabled** (no localhost access)
- Provider auto-falls back to **Groq** cloud
- User files stored in `/tmp` (Vercel writable directory)
- Secure cookies enforced

### Local (Development)

Runs on **Waitress WSGI** server with auto-tuned thread count based on CPU cores:

```bash
python app.py
# Listens on http://0.0.0.0:5000
```

---

## 🎛️ Supported AI Providers

| Provider | Models | Best For |
|---|---|---|
| **Groq** | LLaMA 3.3 70B, Mixtral | Complex queries, coding, reasoning |
| **Ollama Cloud** | Mistral, Gemma 3, LLaMA | General queries, balanced performance |
| **Ollama Local** | Any Ollama model | Offline, privacy-focused usage |
| **Hybrid** | Both Groq + Ollama | Best quality (parallel scoring) |

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
- [Groq](https://groq.com/) for ultra-fast cloud inference
- [edge-tts](https://github.com/rany2/edge-tts) for Microsoft Edge neural TTS
- [Flask](https://flask.palletsprojects.com/) for the lightweight Python web framework
- [Three.js](https://threejs.org/) for the 3D orb visualization

---

<p align="center">Made with ❤️ by Vinayaka H N</p>
