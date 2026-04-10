"""
app.py — NOVA AI Assistant — Application Factory (Phase 7)
Thin entry point wiring all modules.
Phase 6: PostgreSQL, Redis, TaskQueue, Plugin System, JWT, Structured Logging.
Phase 7: Autonomous AgentRunner, WorkflowEngine, Unified Capabilities.
"""

import os
import subprocess
import psutil
from concurrent.futures import ThreadPoolExecutor

from flask import Flask

from config import (
    BASE_DIR, FRONTEND_DIR, FLASK_SECRET_KEY, SESSION_LIFETIME_SECONDS,
    NOVA_ENV, NOVA_LIVE_MODE, OLLAMA_URL, DATABASE_URL, REDIS_URL, IS_VERCEL,
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

# ─── Phase 6: Database Initialization ─────────────────────────────────────────
from database import init_db, is_postgres

try:
    init_db()
    _db_type = "PostgreSQL" if is_postgres() else "SQLite"
    log.info("Phase 6 — Database: %s", _db_type)
except Exception as e:
    log.warning("Phase 6 — Database init failed: %s (falling back to legacy)", e)
    _db_type = "SQLite (fallback)"

# ─── Phase 6: Redis Initialization ────────────────────────────────────────────
from services.redis_service import RedisService

redis_service = RedisService(redis_url=REDIS_URL)
log.info("Phase 6 — Redis: %s", "connected" if redis_service.is_available else "disabled")

# ─── Phase 6: Task Queue ──────────────────────────────────────────────────────
from services.task_queue import TaskQueue

task_queue = TaskQueue()
log.info("Phase 6 — TaskQueue: %d workers", task_queue._max_workers)

# ─── Phase 6: Plugin Manager ─────────────────────────────────────────────────
from services.plugin_manager import PluginManager

_plugins_dir = os.path.join(BASE_DIR, "plugins")
plugin_manager = PluginManager(plugins_dir=_plugins_dir if os.path.isdir(_plugins_dir) else None)

# ─── Phase 6: DbMemoryService (optional PostgreSQL backend) ──────────────────
db_memory = None
if is_postgres():
    try:
        from services.db_memory_service import DbMemoryService
        db_memory = DbMemoryService()
        log.info("Phase 6 — DbMemoryService: active (PostgreSQL)")
    except Exception as e:
        log.warning("Phase 6 — DbMemoryService init failed: %s", e)

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
from utils.response_sanitizer import ResponseSanitizer
from services.response_pipeline import ResponsePipeline
from controllers.chat_controller import ChatController

# Wire services together (Phase 6: inject Redis, DbMemory, PluginManager)
memory_service      = MemoryService(memory_engine, bg_pool, db_memory=db_memory)
session_service     = SessionService(memory_engine._conn)
prompt_builder      = PromptBuilder(memory_service)
hybrid_evaluator    = HybridEvaluator()
cache_service       = CacheService(redis_service=redis_service)
query_analyzer      = QueryAnalyzer()
response_formatter  = ResponseFormatter()
response_sanitizer  = ResponseSanitizer()
ollama_validator    = OllamaValidator()
performance_tracker = PerformanceTracker()
model_router        = ModelRouter(performance_tracker)
tool_executor       = ToolExecutor(plugin_manager=plugin_manager)
agent_engine        = AgentEngine(tool_executor)
ai_service          = AIService(
    prompt_builder, hybrid_evaluator, cache_service,
    query_analyzer, response_formatter, ollama_validator,
    model_router, performance_tracker,
)
command_service     = CommandService(session_service, memory_service)
response_pipeline   = ResponsePipeline(
    ai_service, query_analyzer, agent_engine,
    response_formatter, response_sanitizer, cache_service,
)

# ─── Phase 7: Autonomous Agent & Workflow Engine ──────────────────────────────
from services.agent_runner import AgentRunner
from services.workflow_engine import WorkflowEngine

agent_runner     = AgentRunner(ai_service, tool_executor, memory_service)
workflow_engine  = WorkflowEngine(ai_service, tool_executor)

# Reduce agent steps on Vercel to avoid 60s timeout
if IS_VERCEL:
    import services.agent_runner as _ar
    _ar.DEFAULT_MAX_STEPS = 3
    log.info("Phase 7 — AgentRunner: ACTIVE (max_steps=3, Vercel-limited)")
else:
    log.info("Phase 7 — AgentRunner: ACTIVE (max_steps=%d)", 7)
log.info("Phase 7 — WorkflowEngine: ACTIVE")

# ─── Phase 8→11: Document Context Store & Retrieval ─────────────────────────────
from services.document_context import DocumentContextStore
from services.document_retriever import create_retriever
from services.smart_responder import SmartResponder

document_store = DocumentContextStore()
document_retriever = create_retriever()
smart_responder = SmartResponder()
log.info("Phase 11 — DocumentContextStore: ACTIVE (multi-doc, max=%d)",
         __import__('config').PDF_MAX_DOCUMENTS_PER_SESSION)
log.info("Phase 11 — DocumentRetriever: ACTIVE (backend=%s)",
         document_retriever.stats().get('backend', 'unknown') if hasattr(document_retriever, 'stats') else 'tfidf')
log.info("Phase 11 — SmartResponder: ACTIVE (exam_mode=%s)",
         'enabled' if __import__('config').ENABLE_EXAM_MODE else 'disabled')

# ─── Phase 9: Personality Store ───────────────────────────────────────────────
from services.personality_service import PersonalityStore

personality_store = PersonalityStore()
log.info("Phase 9 — PersonalityStore: ACTIVE")

# ─── Phase 10: ML Personality Model ──────────────────────────────────────────
from config import ENABLE_PERSONALITY_ML

personality_model = None
if ENABLE_PERSONALITY_ML:
    try:
        from ml.personality_model import PersonalityModel
        _ml_model_path = os.path.join(BASE_DIR, "ml", "personality_model.pkl")
        personality_model = PersonalityModel()
        if personality_model.load(_ml_model_path):
            log.info("Phase 10 — PersonalityModel: ACTIVE (loaded from %s)", _ml_model_path)
        else:
            log.warning("Phase 10 — PersonalityModel: pkl not found, run 'python -m ml.train_personality'")
            personality_model = None
    except Exception as exc:
        log.warning("Phase 10 — PersonalityModel: DISABLED (%s)", exc)
        personality_model = None
else:
    log.info("Phase 10 — PersonalityModel: DISABLED (ENABLE_PERSONALITY_ML=false)")

chat_controller     = ChatController(
    ai_service, session_service, memory_service, command_service,
    agent_engine, response_pipeline,
    agent_runner=agent_runner,
    document_store=document_store,
    personality_store=personality_store,
    personality_model=personality_model,
    retriever=document_retriever,
    smart_responder=smart_responder,
)

# ─── Phase 6: Update Rate Limiters with Redis ─────────────────────────────────
from utils.security import chat_rate_limiter, auth_rate_limiter

if redis_service.is_available:
    chat_rate_limiter.update_redis(redis_service)
    auth_rate_limiter.update_redis(redis_service)
    log.info("Phase 6 — Rate limiters: Redis-backed")

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
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024   # 50MB — prevents memory exhaustion DoS

if NOVA_LIVE_MODE:
    app.config["SESSION_COOKIE_SECURE"] = True

# ─── Register Error Handlers, CORS & Middleware ──────────────────────────────
register_error_handlers(app)
configure_cors(app)

from utils.middleware import register_middleware
register_middleware(app)


# ─── Favicon Route ────────────────────────────────────────────────────────────
@app.route('/favicon.ico')
def favicon():
    """Serve favicon to prevent 405 errors."""
    favicon_path = os.path.join(FRONTEND_DIR, "static", "nova_logo.png")
    if os.path.exists(favicon_path):
        return app.send_static_file("nova_logo.png")
    return '', 204

# ─── Register Blueprints ──────────────────────────────────────────────────────
from routes.auth import auth_bp
from routes.chat import chat_bp
from routes.memory import memory_bp
from routes.settings import settings_bp
from routes.system import system_bp

# Inject dependencies
from services.pdf_service import PDFService
pdf_service = PDFService()

import routes.chat as chat_routes
chat_routes.init_app(chat_controller, cache_service,
                     pdf_service=pdf_service, ai_service=ai_service,
                     document_store=document_store,
                     personality_store=personality_store,
                     retriever=document_retriever,
                     smart_responder=smart_responder)

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
log.info("Hybrid evaluator: ACTIVE | Cache: ACTIVE (TTL=%ds, backend=%s)",
         __import__('config').CACHE_TTL,
         "Redis" if redis_service.is_available else "in-memory")
log.info("Retry: max=%d backoff=%.1fs",
         __import__('config').MAX_RETRY, __import__('config').RETRY_BACKOFF)
if NOVA_ENV == "production":
    log.info("Production mode - Ollama Local DISABLED, localhost BLOCKED")
log.info("Adaptive intelligence: ACTIVE (query_analyzer + response_formatter)")
log.info("Response pipeline: ACTIVE (analyze -> agent -> route -> generate -> format -> sanitize)")
log.info("Security sanitizer: ACTIVE")
log.info("Request middleware: ACTIVE (X-Request-Id + X-Nova-Latency)")

# Phase 6+7 summary
log.info("═══════════════════════════════════════════════════════")
log.info("Phase 6 Scalable Architecture:")
log.info("  Database:     %s", _db_type)
log.info("  Redis:        %s", "connected" if redis_service.is_available else "disabled (in-memory fallback)")
log.info("  Task Queue:   %d workers", task_queue._max_workers)
log.info("  Plugins:      %d loaded", len(plugin_manager.list_plugins()))
log.info("  Cache:        %s", "Redis" if cache_service._use_redis else "in-memory LRU")
log.info("  JWT Auth:     %s", "enabled" if os.getenv("JWT_SECRET_KEY") else "disabled (session-only)")
log.info("  Encryption:   %s", "enabled" if os.getenv("ENCRYPTION_KEY") else "disabled (plaintext)")
log.info("  Log Format:   %s", os.getenv("LOG_FORMAT", "text"))
log.info("Phase 7 AI OS:")
log.info("  AgentRunner:  ACTIVE (max_steps=7, tools=%d)", len(tool_executor.get_tools()))
log.info("  Workflows:    ACTIVE")
log.info("  Capabilities: %d (tools + plugins)", len(tool_executor.list_capabilities()))
log.info("Phase 11 PDF Intelligence:")
log.info("  Max PDF Size: %dMB (%s)", __import__('config').PDF_MAX_FILE_SIZE // (1024*1024),
         'Vercel' if IS_VERCEL else 'local')
log.info("  Multi-Doc:    max %d per session", __import__('config').PDF_MAX_DOCUMENTS_PER_SESSION)
log.info("  Retriever:    TF-IDF (scikit-learn)")
log.info("  Exam Mode:    %s", 'enabled' if __import__('config').ENABLE_EXAM_MODE else 'disabled')
log.info("  Smart Resp:   ACTIVE (citations + suggestions)")
log.info("═══════════════════════════════════════════════════════")
log.info("All blueprints registered. NOVA v5 (Phase 11) ready.")

# ─── Vercel Serverless Handler ────────────────────────────────────────────────
handler = app

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
        print("  ██║╚██╗██║██║   ██║╚██  ██╔╝██╔══██║")
        print("  ██║ ╚████║╚██████╔╝  ████╔╝ ██║  ██║")
        print("  ╚═╝  ╚═══╝ ╚═════╝   ╚═══╝  ╚═╝  ╚═╝")
        print()
        print(f"  [NOVA v5] Waitress WSGI Server — Phase 7 Active")
        print(f"  [NOVA v5] http://{host}:{port}")
        print(f"  [NOVA v5] Open -> http://localhost:{port}")
        print(f"  [NOVA v5] WSGI threads: {wsgi_threads}")
        print(f"  [NOVA v5] BG workers: {_BG_WORKERS}")
        print(f"  [NOVA v5] Database: {_db_type}")
        print(f"  [NOVA v5] Redis: {'connected' if redis_service.is_available else 'disabled'}")
        print(f"  [NOVA v5] AgentRunner: ACTIVE")
        print(f"  [NOVA v5] WorkflowEngine: ACTIVE")
        print(f"  [NOVA v5] Capabilities: {len(tool_executor.list_capabilities())}")
        print(f"  [NOVA v5] Plugins: {len(plugin_manager.list_plugins())} loaded")
        print(f"  [NOVA v5] GPU: {'Yes -- ' + GPU_INFO['name'] if GPU_INFO else 'None (CPU-only)'}")
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
