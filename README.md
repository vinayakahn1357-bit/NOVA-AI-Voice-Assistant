# 🌌 NOVA — AI Voice Assistant

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Vercel-blue?style=for-the-badge&logo=vercel)](https://nova-ai-ashen-beta.vercel.app/)
[![Deployed on Vercel](https://img.shields.io/badge/Deployed%20on-Vercel-black?style=flat&logo=vercel)](https://nova-ai-ashen-beta.vercel.app/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0+-green?style=flat&logo=flask)](https://flask.palletsprojects.com/)

> **A production-grade AI assistant with dual-provider intelligence, JWT authentication, Redis infrastructure, persistent memory, neural TTS, PDF analysis, and a premium futuristic UI.**

🔗 **Live at [nova-ai-ashen-beta.vercel.app](https://nova-ai-ashen-beta.vercel.app/)**

NOVA is a full-stack AI assistant with a Python/Flask backend and a browser-based frontend. It supports **dual AI providers** (Groq + NVIDIA), features **JWT + session hybrid authentication**, **Redis-backed caching and rate limiting**, a **persistent SQLite memory engine**, an **agent intelligence system** with workflow orchestration, **PDF document analysis**, and a **6-stage response pipeline**. Deployed on **Vercel** in production and runs on **Waitress WSGI** locally.

---

## ✨ Features

### 🤖 AI & Intelligence
- **Dual-Provider AI** — Groq Cloud and NVIDIA API with automatic failover
- **Hybrid Mode** — True parallel execution of two providers with intelligent response scoring & merging
- **Agent Runner** — Decision, Opinion, Planning, and Tool modes that augment LLM responses
- **Workflow Engine** — Multi-step task orchestration with state management
- **Built-in Tools** — Calculator, system info, date/time, unit conversion (no LLM needed)
- **Query Analyzer** — Classifies 13 query types with adaptive temperature/token tuning
- **Intelligent Routing** — Selects optimal provider based on query type, latency, failure rate, and cost
- **Personality Engine** — ML-driven personality adaptation (5 preset voice personas)

### 📄 Document Intelligence
- **PDF Processing** — Upload and analyze PDFs up to 50MB with streaming extraction
- **TF-IDF Document Retriever** — Semantic chunk retrieval for document-aware responses
- **Smart Responder** — Structured, exam-oriented outputs with page-level citations
- **Multi-Document Sessions** — Active document control with per-session context

### 🧠 Memory & Learning
- **Persistent Memory** — SQLite-backed (WAL mode) storage for facts, interests, and preferences
- **LLM-Powered Extraction** — Automatically learns about you from every conversation
- **Daily Summaries** — Auto-generates conversation digests
- **Memory Context Injection** — Personalizes every response with what NOVA knows about you

### 🔒 Security & Auth
- **JWT Authentication** — Stateless HS256-signed tokens (7-day expiry) for API clients
- **Session + JWT Hybrid** — Backward-compatible: browsers use sessions, APIs use Bearer tokens
- **Token Refresh** — `/auth/refresh` endpoint for seamless token renewal
- **Google OAuth 2.0** — One-click Google sign-in with JWT bridge
- **Email/Password Auth** — PBKDF2-HMAC-SHA256 hashing (260k iterations)
- **RBAC** — Admin/user role-based access control
- **Redis Rate Limiting** — Distributed rate limiting with in-memory fallback
- **Prompt Injection Detection** — Input validation and response sanitization
- **Production Hardening** — Secure cookies, CSRF protection, strict JWT priority

### 🎨 Frontend & UX
- **Streaming Responses** — Token-by-token SSE for real-time output
- **Neural TTS** — Microsoft Edge TTS with 5 expressive voice personas (Nova, Anya, Mitsuri, Jarvis, Sage)
- **Voice Chat View** — Dedicated voice interface with Jarvis-style animated AI core
- **Markdown Rendering** — Rich formatted replies with syntax highlighting in both chat and voice views
- **3D Orb UI** — Three.js-powered interactive orb with glassmorphism design
- **Responsive Design** — Works across desktop and mobile
- **Dark Mode** — Premium futuristic sci-fi aesthetic throughout

### ⚡ Performance & Reliability
- **Redis Caching** — Distributed response cache with in-memory LRU fallback
- **Redis Rate Limiting** — Distributed sliding window with in-memory fallback
- **Circuit Breaker** — Auto-disables failing providers
- **Retry Logic** — Exponential backoff on transient failures
- **Performance Tracking** — Per-provider latency and failure rate monitoring
- **Background Processing** — Thread pool for non-blocking memory extraction
- **Task Queue** — Background task processing with priority scheduling

---

## 🖥️ Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.10+ |
| **Framework** | Flask ≥3.0.0 |
| **WSGI Server** | Waitress ≥2.1.0 |
| **Database** | SQLite (WAL mode) |
| **Cache / Rate Limit** | Redis (with in-memory fallback) |
| **Authentication** | JWT (HS256) + Flask sessions + Google OAuth 2.0 |
| **AI Providers** | Groq (LLaMA 3.3 70B), NVIDIA API |
| **TTS** | Edge-TTS (Microsoft Neural voices) |
| **Document Analysis** | TF-IDF retriever, PDF streaming extraction |
| **Frontend** | HTML/CSS/JS, Three.js |
| **Deployment** | Railway (production) + Vercel (static) + Local |

---

## 📁 Project Structure

```
NOVAbackend/
├── app.py                        # Application factory & dependency wiring
├── config.py                     # Centralized configuration (env-aware)
├── nova_memory.py                # SQLite persistent memory engine
├── database.py                   # Database connection management
├── requirements.txt              # Python dependencies
├── Procfile                      # Railway deployment entrypoint
├── nixpacks.toml                 # Railway build configuration
├── vercel.json                   # Vercel deployment config
│
├── controllers/                  # Request orchestration
│   ├── auth_controller.py        #   Register, login, OAuth, JWT issuance
│   ├── chat_controller.py        #   Chat pipeline orchestration & streaming
│   └── settings_controller.py    #   Runtime settings management
│
├── routes/                       # Flask Blueprint endpoints
│   ├── auth.py                   #   /auth/* routes (login, register, refresh, token)
│   ├── chat.py                   #   /chat, /chat/stream, /document/* routes
│   ├── memory.py                 #   /memory/* routes
│   ├── settings.py               #   /settings routes
│   └── system.py                 #   /system/status, /health routes
│
├── services/                     # Core business logic (29 services)
│   ├── ai_service.py             #   LLM provider dispatch (Groq/NVIDIA/Hybrid)
│   ├── agent_engine.py           #   AI agent decision engine
│   ├── agent_runner.py           #   Multi-step agent execution orchestrator
│   ├── workflow_engine.py        #   Multi-step task workflow orchestration
│   ├── hybrid_evaluator.py       #   Parallel scoring & response merging
│   ├── hybrid_service.py         #   Hybrid provider execution
│   ├── model_router.py           #   Intelligent provider selection
│   ├── query_analyzer.py         #   Query classification & tuning
│   ├── response_pipeline.py      #   6-stage response orchestrator
│   ├── smart_responder.py        #   Structured response generation
│   ├── tool_executor.py          #   Built-in tool system
│   ├── plugin_manager.py         #   Plugin system for extensibility
│   ├── task_queue.py             #   Background task queue
│   ├── ollama_validator.py       #   Model discovery & validation
│   ├── performance_tracker.py    #   Latency/failure stats & circuit breaker
│   ├── prompt_builder.py         #   System prompt construction
│   ├── personality_service.py    #   ML-driven personality adaptation
│   ├── cache_service.py          #   Redis + in-memory response cache
│   ├── redis_service.py          #   Redis connection & operations
│   ├── command_service.py        #   Slash-command detection
│   ├── session_service.py        #   Session history management
│   ├── memory_service.py         #   Memory abstraction layer
│   ├── db_memory_service.py      #   Database-backed memory operations
│   ├── document_context.py       #   Active document session management
│   ├── document_retriever.py     #   TF-IDF semantic document retrieval
│   ├── pdf_service.py            #   PDF extraction (up to 50MB streaming)
│   ├── tts_service.py            #   Text-to-Speech via Edge-TTS
│   └── image_service.py          #   Image handling
│
├── utils/                        # Cross-cutting utilities
│   ├── security.py               #   Auth decorators, CORS, rate limiting, RBAC
│   ├── jwt_auth.py               #   JWT token generation, verification, refresh
│   ├── crypto.py                 #   Cryptographic utilities
│   ├── middleware.py             #   Request ID & latency headers
│   ├── errors.py                 #   Custom exception hierarchy
│   ├── logger.py                 #   Logging configuration
│   ├── structured_logger.py      #   JSON structured logging
│   ├── async_helpers.py          #   Async/threading utilities
│   ├── response_formatter.py     #   Query-type-aware formatting
│   ├── response_sanitizer.py     #   Security sanitization
│   ├── retry_handler.py          #   Retry with exponential backoff
│   └── validators.py             #   Input validation
│
├── ml/                           # Machine learning models
│   └── personality_model.pkl     #   Trained personality classifier
│
├── tests/                        # Test suite
│   ├── test_retriever.py         #   Document retriever tests
│   └── test_calculator.py        #   Calculator tool tests
│
└── Frontend/                     # Browser-based UI
    ├── index.html                #   Main AI assistant interface
    ├── landing.html              #   Landing/marketing page
    ├── login.html                #   Auth page (email + Google OAuth)
    └── static/
        ├── script.js             #   App logic, SSE, voice, 3D orb
        ├── Style.css             #   Design system & animations (8900+ lines)
        ├── nova_logo.png         #   Brand logo
        └── modules/
            ├── markdown.js       #   Markdown rendering engine
            ├── chatbubbles.js    #   Chat message bubble components
            └── toast.js          #   Toast notification system
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- **[Groq API Key](https://console.groq.com/)** or **[NVIDIA API Key](https://build.nvidia.com/)**
- **Redis** (optional — falls back to in-memory)

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
# ─── AI Providers (at least one required) ───
GROQ_API_KEY=your_groq_api_key
NVIDIA_API_KEY=your_nvidia_api_key

# ─── Authentication ───
FLASK_SECRET_KEY=your_secret_key
JWT_SECRET_KEY=your_jwt_secret            # Required for stateless API auth
JWT_EXPIRY_HOURS=168                       # Token lifetime (default: 7 days)

# ─── Redis (optional — graceful fallback to in-memory) ───
REDIS_URL=redis://default:password@host:port

# ─── Google OAuth (optional) ───
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret

# ─── Configuration ───
NOVA_PROVIDER=groq                         # groq | nvidia | hybrid
NOVA_TEMPERATURE=0.75
NOVA_MAX_TOKENS=1024
```

### Running NOVA

```bash
# Start the server
python app.py

# With Groq (cloud) — just set GROQ_API_KEY in .env
# With NVIDIA — just set NVIDIA_API_KEY in .env
# With Hybrid — set both keys, NOVA_PROVIDER=hybrid
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

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `POST` | `/chat` | JWT / Session | Send a message, get a full JSON response |
| `POST` | `/chat/stream` | JWT / Session | Send a message, get SSE token stream |

### Auth

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `POST` | `/auth/register` | None | Register with email/password → returns JWT |
| `POST` | `/auth/login` | None | Login with email/password → returns JWT |
| `POST` | `/auth/logout` | Session | Logout current session |
| `GET` | `/auth/me` | JWT / Session | Get current authenticated user |
| `POST` | `/auth/refresh` | JWT | Exchange valid JWT for a fresh one |
| `GET` | `/auth/token` | Session | Get JWT for current session user |
| `GET` | `/auth/google` | None | Initiate Google OAuth flow |
| `GET` | `/auth/google/callback` | None | Google OAuth callback |

### Document Intelligence

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `POST` | `/chat` | JWT / Session | Upload PDF via multipart form data |
| `GET` | `/document/status` | JWT / Session | Check active document for session |
| `POST` | `/document/clear` | JWT / Session | Clear active document context |

### Memory

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `GET` | `/memory` | JWT / Session | View NOVA's learned memory |
| `POST` | `/memory/reset` | JWT / Session | Clear all learned memory |
| `POST` | `/memory/summary` | JWT / Session | Trigger daily summary generation |

### Settings & System

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `GET` | `/settings` | JWT / Session | Get current model settings |
| `POST` | `/settings` | JWT / Session | Update model, provider, temperature |
| `GET` | `/system/status` | None | System health, CPU, RAM stats |
| `GET` | `/health` | None | Health check endpoint |

### Authentication Examples

```bash
# Register → get JWT
curl -X POST https://novaarcai.com/auth/register \
  -H "Content-Type: application/json" \
  -d '{"name": "User", "email": "user@example.com", "password": "secret"}'
# → {"ok": true, "user": {...}, "token": "eyJ..."}

# Use JWT for chat
curl -X POST https://novaarcai.com/chat \
  -H "Authorization: Bearer eyJ..." \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello Nova", "session_id": "abc123"}'

# Refresh token
curl -X POST https://novaarcai.com/auth/refresh \
  -H "Authorization: Bearer eyJ..."
# → {"ok": true, "token": "eyJ...<fresh>"}
```

### Chat Response Example

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
| **2. Agent** | `AgentRunner` | Decide mode: normal, decision, opinion, planning, or tool |
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

### Authentication
- **JWT (HS256)** — Stateless tokens with 7-day expiry for API clients
- **Flask Sessions** — Cookie-based sessions for browser users
- **Hybrid Auth** — `login_required` decorator supports both; strict JWT priority (invalid token = 401, no session fallback)
- **Token Refresh** — Exchange valid JWT for fresh token at `/auth/refresh`
- **Session-to-JWT Bridge** — Get JWT from session at `/auth/token` (for Google OAuth → API flow)

### Data Protection
- **PBKDF2-HMAC-SHA256** password hashing (260,000 iterations)
- **Constant-time** password comparison
- **HTTP-only, SameSite** session cookies (Secure in production)
- **CSRF protection** on OAuth flows

### Rate Limiting & Infrastructure
- **Redis rate limiting** — Distributed sliding window (with in-memory fallback)
- **30 req/min** (chat), **10 req/min** (auth)
- **Prompt injection detection** and response sanitization
- **RBAC** — Admin email whitelist
- **`USE_REDIS` flag** — Feature-gate Redis-dependent capabilities

---

## 🌐 Deployment

### Vercel (Production) — Live at [nova-ai-ashen-beta.vercel.app](https://nova-ai-ashen-beta.vercel.app/)

NOVA is **currently deployed and running** on Vercel.

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel --prod
```

**Production configuration** (`vercel.json`):
- Runtime: `@vercel/python` (serverless)
- Max duration: 60 seconds
- SSE streaming: enabled (`X-Accel-Buffering: no`)
- Static assets: 1-year immutable cache
- JWT: Set `JWT_SECRET_KEY` in Vercel environment

**Production environment**:
- Provider auto-selects **Groq** or **NVIDIA** based on available keys
- User files stored in `/tmp` (Vercel writable directory)
- Secure cookies enforced, JWT for API clients
- SQLite WAL mode for concurrent read/write

### Vercel (Static/Serverless)

```bash
npm i -g vercel
vercel --prod
```

- Runtime: `@vercel/python` (serverless, 60s max duration)
- SSE streaming: enabled (`X-Accel-Buffering: no`)
- Static assets: 1-year immutable cache
- Note: Vercel uses non-streaming `/chat` fallback due to SSE buffering

### Local (Development)

Runs on **Waitress WSGI** server with auto-tuned thread count:

```bash
python app.py
# Listens on http://0.0.0.0:5000
```

---

## 🎛️ Supported AI Providers

| Provider | Models | Best For |
|---|---|---|
| **Groq** | LLaMA 3.3 70B, Mixtral | Complex queries, coding, reasoning |
| **NVIDIA** | Meta LLaMA, Mistral Large | Advanced reasoning, document analysis |
| **Hybrid** | Groq + NVIDIA in parallel | Best quality (parallel scoring + merging) |

---

## 🎙️ Voice Personalities

| Persona | Emoji | Style | Rate | Pitch |
|---|---|---|---|---|
| **NOVA** | 🤖 | Clear, professional, precise | 1.0 | 1.0 |
| **Anya** | 🎀 | Playful, curious, child-like energy | 0.82 | 1.45 |
| **Mitsuri** | 🌸 | Warm, gentle, bubbly | 0.91 | 1.22 |
| **Jarvis** | 🎩 | Authoritative, measured, butler-like | 0.88 | 0.85 |
| **Sage** | 🧘 | Calm, deliberate, grounding | 0.78 | 0.92 |

Voice presets are selectable in the Settings panel. All voices use Microsoft Edge Neural TTS.

---

## 📄 Recent Upgrades

### v3.0 — JWT Auth + Redis Infrastructure (April 2026)
- ✅ **JWT Authentication** — Stateless HS256 tokens for API clients
- ✅ **Redis Integration** — Distributed caching, rate limiting, session storage
- ✅ **Token Refresh** — `/auth/refresh` endpoint
- ✅ **Session-to-JWT Bridge** — `/auth/token` endpoint
- ✅ **Hybrid Auth Middleware** — `g.auth_method` for downstream audit
- ✅ **`USE_REDIS` flag** — Feature-gating for Redis-dependent features
- ✅ **Voice Chat Scrolling** — Scrollable reply bubbles with markdown rendering
- ✅ **Backward Compatible** — Session auth continues to work for browser users

### v2.5 — PDF Intelligence (April 2026)
- ✅ **PDF Processing** — 50MB streaming extraction
- ✅ **TF-IDF Document Retriever** — Semantic chunk retrieval
- ✅ **Smart Responder** — Structured outputs with citations
- ✅ **Multi-Document Sessions** — Active document management

### v2.0 — Dual Provider + Agent System
- ✅ **NVIDIA API** integration alongside Groq
- ✅ **Agent Runner** with workflow orchestration
- ✅ **Personality Engine** with 5 voice personas
- ✅ **Voice Chat View** — Dedicated Jarvis-style interface

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

- [Groq](https://groq.com/) for ultra-fast cloud inference
- [NVIDIA](https://build.nvidia.com/) for advanced AI model hosting
- [edge-tts](https://github.com/rany2/edge-tts) for Microsoft Edge neural TTS
- [Flask](https://flask.palletsprojects.com/) for the lightweight Python web framework
- [Three.js](https://threejs.org/) for the 3D orb visualization
- [Redis](https://redis.io/) for distributed caching infrastructure
- [PyJWT](https://pyjwt.readthedocs.io/) for JSON Web Token implementation

---

<p align="center">Made with ❤️ by Vinayaka H N</p>
