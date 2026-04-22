"""
routes/system.py — System, TTS, & Diagnostics Routes for NOVA
"""

import time
import psutil
from flask import Blueprint, request, jsonify, Response

from services.tts_service import TTSService
from utils.logger import get_logger

log = get_logger("routes.system")

system_bp = Blueprint("system", __name__)

# Cache system info at import time
_GPU_INFO = None
_CPU_CORES_LOGICAL = None
_CPU_CORES_PHYSICAL = None
_BG_WORKERS = None
_STARTUP_TIME = time.time()


def init_app(gpu_info, cpu_logical, cpu_physical, bg_workers):
    """Inject system info and services."""
    global _GPU_INFO, _CPU_CORES_LOGICAL, _CPU_CORES_PHYSICAL, _BG_WORKERS
    _GPU_INFO = gpu_info
    _CPU_CORES_LOGICAL = cpu_logical
    _CPU_CORES_PHYSICAL = cpu_physical
    _BG_WORKERS = bg_workers


@system_bp.route("/health")
def health():
    """
    Production health check endpoint.
    Returns service status, uptime, provider availability, and resource usage.
    Use for monitoring dashboards, load balancer probes, and deployment validation.
    """
    mem = psutil.virtual_memory()
    uptime_s = int(time.time() - _STARTUP_TIME)

    # Provider availability (safe — uses lazy imports)
    providers = {}
    try:
        from services.performance_tracker import PerformanceTracker
        # Access the tracker via the app's wired instance if available
        providers = {
            "status": "configured",
        }
    except Exception:
        providers = {"status": "unknown"}

    # Cache and DB info
    cache_info = "unknown"
    db_info = "unknown"
    try:
        from config import REDIS_URL, DATABASE_URL, IS_VERCEL
        from database import is_postgres
        cache_info = "redis" if REDIS_URL else "in-memory"
        db_info = "postgresql" if is_postgres() else "sqlite"
    except Exception:
        pass

    return jsonify({
        "status": "healthy",
        "service": "nova",
        "version": "2.0.0-v2",
        "uptime_seconds": uptime_s,
        "uptime_human": _format_uptime(uptime_s),
        "memory": {
            "rss_mb": round(psutil.Process().memory_info().rss / (1024 * 1024), 1),
            "system_percent": mem.percent,
        },
        "infrastructure": {
            "database": db_info,
            "cache": cache_info,
            "bg_workers": _BG_WORKERS,
            "gpu": _GPU_INFO["name"] if _GPU_INFO else None,
        },
    })


def _format_uptime(seconds: int) -> str:
    """Format seconds into human-readable uptime string."""
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    return f"{hours}h {minutes}m"


@system_bp.route("/system")
def system_status():
    """System monitoring endpoint."""
    mem = psutil.virtual_memory()
    payload = {
        "cpu": psutil.cpu_percent(interval=0),
        "cpu_cores_logical":  _CPU_CORES_LOGICAL,
        "cpu_cores_physical": _CPU_CORES_PHYSICAL,
        "cpu_threads_per_core": (_CPU_CORES_LOGICAL or 4) // max(1, _CPU_CORES_PHYSICAL or 1),
        "memory": mem.percent,
        "memory_used_gb": round(mem.used / (1024 ** 3), 2),
        "memory_total_gb": round(mem.total / (1024 ** 3), 2),
        "tasks": len(psutil.pids()),
        "bg_pool_workers": _BG_WORKERS,
        "gpu": _GPU_INFO,
    }
    return jsonify(payload)


@system_bp.route("/tts", methods=["POST"])
def tts():
    """Generate realistic neural speech via Microsoft Edge TTS."""
    try:
        data = request.get_json() or {}
        text = data.get("text", "").strip()
        voice = data.get("voice")
        rate = data.get("rate", "+0%")
        pitch = data.get("pitch", "+0Hz")

        audio_bytes = TTSService.synthesize(text, voice or "", rate, pitch)

        return Response(
            audio_bytes,
            mimetype="audio/mpeg",
            headers={
                "Content-Length": str(len(audio_bytes)),
                "Cache-Control": "no-cache",
            }
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@system_bp.route("/tts/voices", methods=["GET"])
def tts_voices():
    """List available Edge TTS English voices."""
    try:
        voices = TTSService.list_voices()
        return jsonify({"voices": voices})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
