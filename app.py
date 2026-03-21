"""
app.py — NOVA AI Assistant — Application Factory (v2)
Thin entry point wiring all modules, including hybrid evaluator and cache.
"""

import os
import subprocess
import psutil
from concurrent.futures import ThreadPoolExecutor

from flask import Flask

from config import (
    BASE_DIR, FRONTEND_DIR, FLASK_SECRET_KEY, SESSION_LIFETIME_SECONDS,
    NOVA_ENV, NOVA_LIVE_MODE, OLLAMA_URL,
)
from utils.logger import get_logger
from utils.errors import register_error_handlers
from utils.security import configure_cors

log = get_logger("app")

# ─── CPU / GPU Detection ──────────────────────────────────────────────────────
CPU_CORES_LOGICAL  = os.cpu_count() or 4
CPU_CORES_PHYSICAL = psutil.cpu_count(logical=False) or 2


def _detect_gpu():
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


GPU_INFO = _detect_gpu()

# ─── Background Thread Pool ───────────────────────────────────────────────────
_BG_WORKERS = max(4, CPU_CORES_PHYSICAL * 2)
bg_pool = ThreadPoolExecutor(max_workers=_BG_WORKERS, thread_name_prefix='nova-bg')

# ─── Initialise Core Services ─────────────────────────────────────────────────
from nova_memory import NovaMemory

memory_engine = NovaMemory()

from services.memory_service import MemoryService
from services.session_service import SessionService
from services.prompt_builder import PromptBuilder
from services.ai_service import AIService
from services.command_service import CommandService
from services.hybrid_evaluator import HybridEvaluator
from services.cache_service import CacheService
from services.query_analyzer import QueryAnalyzer
from services.ollama_validator import OllamaValidator
from services.performance_tracker import PerformanceTracker
from services.model_router import ModelRouter
from services.agent_engine import AgentEngine
from services.tool_executor import ToolExecutor
from utils.response_formatter import ResponseFormatter
from controllers.chat_controller import ChatController

# Wire services together
memory_service      = MemoryService(memory_engine, bg_pool)
session_service     = SessionService(memory_engine._conn)
prompt_builder      = PromptBuilder(memory_service)
hybrid_evaluator    = HybridEvaluator()
cache_service       = CacheService()
query_analyzer      = QueryAnalyzer()
response_formatter  = ResponseFormatter()
ollama_validator    = OllamaValidator()
performance_tracker = PerformanceTracker()
model_router        = ModelRouter(performance_tracker)
tool_executor       = ToolExecutor()
agent_engine        = AgentEngine(tool_executor)
ai_service          = AIService(
    prompt_builder, hybrid_evaluator, cache_service,
    query_analyzer, response_formatter, ollama_validator,
    model_router, performance_tracker,
)
command_service     = CommandService(session_service, memory_service)
chat_controller     = ChatController(
    ai_service, session_service, memory_service, command_service, agent_engine
)

# ─── Create Flask App ─────────────────────────────────────────────────────────
app = Flask(
    __name__,
    static_folder=os.path.join(FRONTEND_DIR, "static"),
    static_url_path="/static",
)
app.json.sort_keys = False
app.secret_key = FLASK_SECRET_KEY
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["PERMANENT_SESSION_LIFETIME"] = SESSION_LIFETIME_SECONDS

if NOVA_LIVE_MODE:
    app.config["SESSION_COOKIE_SECURE"] = True

# ─── Register Error Handlers & CORS ───────────────────────────────────────────
register_error_handlers(app)
configure_cors(app)

# ─── Register Blueprints ──────────────────────────────────────────────────────
from routes.auth import auth_bp
from routes.chat import chat_bp
from routes.memory import memory_bp
from routes.settings import settings_bp
from routes.system import system_bp

# Inject dependencies
import routes.chat as chat_routes
chat_routes.init_app(chat_controller, cache_service)

import routes.memory as memory_routes
memory_routes.init_app(memory_service, session_service)

import routes.system as system_routes
system_routes.init_app(GPU_INFO, CPU_CORES_LOGICAL, CPU_CORES_PHYSICAL, _BG_WORKERS,
                       ollama_validator)

app.register_blueprint(auth_bp)
app.register_blueprint(chat_bp)
app.register_blueprint(memory_bp)
app.register_blueprint(settings_bp)
app.register_blueprint(system_bp)

# ─── Startup Logging ──────────────────────────────────────────────────────────
if GPU_INFO:
    log.info("GPU: %s  VRAM %dMB", GPU_INFO['name'], GPU_INFO['vram_total_mb'])
else:
    log.info("No CUDA GPU — CPU-only backend")
log.info("CPU: %d physical / %d logical cores", CPU_CORES_PHYSICAL, CPU_CORES_LOGICAL)
log.info("BG pool: %d workers", _BG_WORKERS)
log.info("ENV: %s | Provider: %s", NOVA_ENV, ai_service._resolve_provider(
    __import__('config').get_settings()['provider']
))
log.info("Hybrid evaluator: ACTIVE | Cache: ACTIVE (TTL=%ds)",
         __import__('config').CACHE_TTL)
log.info("Retry: max=%d backoff=%.1fs",
         __import__('config').MAX_RETRY, __import__('config').RETRY_BACKOFF)
if NOVA_ENV == "production":
    log.info("Production mode - Ollama Local DISABLED, localhost BLOCKED")
log.info("Adaptive intelligence: ACTIVE (query_analyzer + response_formatter)")
log.info("All blueprints registered. NOVA v3 ready.")

# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", 5000))
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    wsgi_threads = max(4, min(32, CPU_CORES_LOGICAL))

    try:
        from waitress import serve
        print()
        print("  ███╗   ██╗ ██████╗ ██╗   ██╗ █████╗ ")
        print("  ████╗  ██║██╔═══██╗██║   ██║██╔══██╗")
        print("  ██╔██╗ ██║██║   ██║██║   ██║███████║")
        print("  ██║╚██╗██║██║   ██║╚██╗ ██╔╝██╔══██║")
        print("  ██║ ╚████║╚██████╔╝ ╚████╔╝ ██║  ██║")
        print("  ╚═╝  ╚═══╝ ╚═════╝   ╚═══╝  ╚═╝  ╚═╝")
        print()
        print(f"  [NOVA v2] Waitress WSGI Server")
        print(f"  [NOVA v2] http://{host}:{port}")
        print(f"  [NOVA v2] Open → http://localhost:{port}")
        print(f"  [NOVA v2] WSGI threads: {wsgi_threads}")
        print(f"  [NOVA v2] BG workers: {_BG_WORKERS}")
        print(f"  [NOVA v2] Hybrid evaluator: ACTIVE (parallel)")
        print(f"  [NOVA v2] Response cache: ACTIVE")
        print(f"  [NOVA v2] GPU: {'Yes — ' + GPU_INFO['name'] if GPU_INFO else 'None (CPU-only)'}")
        print()
        serve(
            app,
            host=host,
            port=port,
            threads=wsgi_threads,
            channel_timeout=120,
            cleanup_interval=30,
            ident="NOVA/Waitress",
        )
    except ImportError:
        log.warning("Waitress not installed — falling back to Flask dev server")
        app.run(host=host, port=port, debug=False, threaded=True)
