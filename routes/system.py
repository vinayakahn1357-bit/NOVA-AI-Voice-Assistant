"""
routes/system.py — System, TTS, & Diagnostics Routes for NOVA
"""

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
_OLLAMA_VALIDATOR = None


def init_app(gpu_info, cpu_logical, cpu_physical, bg_workers, ollama_validator=None):
    """Inject system info and services."""
    global _GPU_INFO, _CPU_CORES_LOGICAL, _CPU_CORES_PHYSICAL, _BG_WORKERS, _OLLAMA_VALIDATOR
    _GPU_INFO = gpu_info
    _CPU_CORES_LOGICAL = cpu_logical
    _CPU_CORES_PHYSICAL = cpu_physical
    _BG_WORKERS = bg_workers
    _OLLAMA_VALIDATOR = ollama_validator


@system_bp.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "service": "nova"})


@system_bp.route("/system")
def system_status():
    """System monitoring endpoint."""
    mem = psutil.virtual_memory()
    payload = {
        "cpu": psutil.cpu_percent(interval=0),
        "cpu_cores_logical":  _CPU_CORES_LOGICAL,
        "cpu_cores_physical": _CPU_CORES_PHYSICAL,
        "cpu_threads_per_core": _CPU_CORES_LOGICAL // max(1, _CPU_CORES_PHYSICAL),
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

        audio_bytes = TTSService.synthesize(text, voice, rate, pitch)

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


@system_bp.route("/api/test-ollama", methods=["GET"])
def test_ollama():
    """
    Full diagnostic test of Ollama connectivity, model availability,
    and model validation. Returns structured debug info.
    """
    if not _OLLAMA_VALIDATOR:
        return jsonify({
            "status": "failed",
            "error": "OllamaValidator not configured on this server",
        }), 500

    try:
        result = _OLLAMA_VALIDATOR.test_connection()
        status_code = 200 if result["status"] == "connected" else 503
        return jsonify(result), status_code
    except Exception as e:
        log.error("test-ollama error: %s", e)
        return jsonify({
            "status": "failed",
            "error": str(e),
        }), 500
