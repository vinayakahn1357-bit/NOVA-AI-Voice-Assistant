"""
services/hybrid_service.py — Intelligent Escalation Hybrid for NOVA V2

Strategy: Groq-first. NVIDIA only when truly needed.
Maximum 2 API calls per query. 80-90% of queries stay on Groq.

Uses existing AIService._stream_groq / _stream_nvidia / _generate_groq /
_generate_nvidia — zero duplication of provider logic.

Flow:
    1. Pre-check complexity         (free, rule-based)
    2. Call Groq                    (always, fast)
    3. Evaluate response quality    (free, rule-based)
    4. Escalate to NVIDIA?          (only if needed)
    5. Return best response
"""

import json
from utils.logger import get_logger

log = get_logger("hybrid")

# ─── Complexity Keywords (pre-check) ──────────────────────────────────────────
_COMPLEX_KEYWORDS = frozenset({
    "analyze", "analyse", "compare", "explain deeply", "optimize",
    "code", "why", "implement", "architecture", "design", "debug",
    "refactor", "algorithm", "step by step", "in detail", "calculate",
    "derive", "proof", "evaluate", "comprehensive",
})

# ─── Low-confidence Markers (post-response check) ────────────────────────────
_UNCERTAINTY_PHRASES = (
    "not sure", "i think", "maybe", "i believe", "i'm not certain",
    "i cannot", "i don't know", "unclear", "it depends", "i'm unsure",
)


def is_complex_query(query: str) -> bool:
    """
    Cheap rule-based pre-check: is this query likely to need NVIDIA?
    Returns True if query is long OR contains complexity keywords.
    """
    q = query.strip().lower()
    if len(q) > 120:
        return True
    return any(kw in q for kw in _COMPLEX_KEYWORDS)


def is_low_confidence(response: str, query: str) -> bool:
    """
    Cheap rule-based check: is the Groq response unsatisfying?
    Returns True if response is too short, uncertain, or mismatched.
    """
    r = response.strip()
    if len(r) < 50:
        return True
    if len(query.strip()) > 80 and len(r) < 120:
        return True
    r_lower = r.lower()
    return any(phrase in r_lower for phrase in _UNCERTAINTY_PHRASES)


def hybrid_generate_stream(ai_service, history: list, user_message: str,
                           full_reply: list,
                           personality: str = "default"):
    """
    Streaming hybrid generation.

    Uses ai_service._stream_groq / _stream_nvidia internally.
    No duplication of HTTP/SSE parsing logic.

    Strategy:
    - Always stream Groq tokens first (feels fast)
    - Collect full response
    - If escalation needed: tell frontend to replace, then stream NVIDIA
    - If NVIDIA fails: keep Groq response (already in full_reply)

    Yields standard SSE strings.
    """
    settings_import = None
    try:
        from config import get_settings
        settings_import = get_settings
    except Exception:
        pass

    pre_complex = is_complex_query(user_message)
    chat_messages = ai_service._prompt.build_chat_messages(history, personality)

    # ── Phase A: Stream Groq ──────────────────────────────────────────────
    groq_tokens: list[str] = []
    groq_full_reply: list[str] = []  # separate buffer for groq
    groq_ok = False

    try:
        if not ai_service._groq_configured():
            raise RuntimeError("Groq not configured")

        for chunk in ai_service._stream_groq(chat_messages, groq_full_reply):
            # Forward token chunks to client immediately
            if chunk.startswith("data: "):
                try:
                    payload = json.loads(chunk[6:].strip())
                    if payload.get("token"):
                        groq_tokens.append(payload["token"])
                    # Don't forward 'done' event yet — we may escalate
                    if payload.get("done") or payload.get("error"):
                        continue
                except (json.JSONDecodeError, KeyError):
                    pass
            yield chunk  # forward raw SSE line (includes error lines)
        groq_ok = True

    except Exception as exc:
        log.warning("Hybrid stream: Groq failed (%s)", exc)

    groq_response = "".join(groq_tokens) or "".join(groq_full_reply)

    # ── Phase B: Escalation Check ─────────────────────────────────────────
    if groq_ok:
        escalate = pre_complex or is_low_confidence(groq_response, user_message)
    else:
        escalate = True  # Groq failed → must try NVIDIA

    if not escalate:
        # Happy path: Groq was good enough
        full_reply.extend(groq_tokens or groq_full_reply)
        log.info("Hybrid: groq_only | complex=%s | len=%d", pre_complex, len(groq_response))
        yield 'data: %s\n\n' % json.dumps({
            "type": "hybrid_meta",
            "used_provider": "groq",
            "escalated": False,
        })
        return

    # ── Phase C: Escalate to NVIDIA ──────────────────────────────────────
    log.info("Hybrid: escalating → nvidia | complex=%s | low_conf=%s",
             pre_complex, is_low_confidence(groq_response, user_message))

    # Tell the frontend to clear the Groq partial and expect new tokens
    yield 'data: %s\n\n' % json.dumps({
        "type": "hybrid_escalating",
        "reason": "complex" if pre_complex else "low_confidence",
    })

    nvidia_tokens: list[str] = []
    nvidia_full_reply: list[str] = []
    nvidia_ok = False

    try:
        if not ai_service._nvidia_configured():
            raise RuntimeError("NVIDIA not configured")

        for chunk in ai_service._stream_nvidia(chat_messages, nvidia_full_reply):
            if chunk.startswith("data: "):
                try:
                    payload = json.loads(chunk[6:].strip())
                    if payload.get("token"):
                        nvidia_tokens.append(payload["token"])
                    if payload.get("done") or payload.get("error"):
                        continue
                except (json.JSONDecodeError, KeyError):
                    pass
            yield chunk  # forward NVIDIA tokens to client
        nvidia_ok = True

    except Exception as exc:
        log.warning("Hybrid stream: NVIDIA escalation failed (%s), using Groq response", exc)

    if nvidia_ok and (nvidia_tokens or nvidia_full_reply):
        full_reply.extend(nvidia_tokens or nvidia_full_reply)
        log.info("Hybrid: used_nvidia | tokens=%d", len(full_reply))
        yield 'data: %s\n\n' % json.dumps({
            "type": "hybrid_meta",
            "used_provider": "nvidia",
            "escalated": True,
        })
    else:
        # NVIDIA failed — use Groq response that was already streamed
        full_reply.extend(groq_tokens or groq_full_reply)
        log.warning("Hybrid: nvidia_failed → groq_fallback")
        yield 'data: %s\n\n' % json.dumps({
            "type": "hybrid_meta",
            "used_provider": "groq",
            "escalated": False,
            "nvidia_failed": True,
        })
