"""
services/token_estimator.py — Token Estimator for NOVA (Phase 14)

Fast, lightweight token estimation without requiring tiktoken.
Uses character-based heuristics calibrated against GPT/Llama tokenizers.

Purpose:
    - Pre-flight token budget checks before LLM calls
    - Prompt size guardrails (prevent context overflow)
    - Token usage logging for cost tracking
    - Dynamic max_tokens calculation

Accuracy: within 10-15% of actual token count (sufficient for budgeting).
"""

import re
from utils.logger import get_logger

log = get_logger("token_estimator")

# ── Calibration Constants ─────────────────────────────────────────────────────
# Average chars-per-token ratios by content type
_CHARS_PER_TOKEN_EN = 4.0       # English prose
_CHARS_PER_TOKEN_CODE = 3.5     # Code/technical content
_CHARS_PER_TOKEN_MIXED = 3.8    # Mixed content

# Model context windows (in tokens)
MODEL_CONTEXT_LIMITS = {
    # Groq models
    "llama-3.3-70b-versatile": 131072,
    "llama-3.1-8b-instant": 131072,
    "llama3-70b-8192": 8192,
    "llama3-8b-8192": 8192,
    "mixtral-8x7b-32768": 32768,
    "gemma2-9b-it": 8192,
    # NVIDIA models
    "meta/llama-3.3-70b-instruct": 131072,
    "nvidia/llama-3.1-nemotron-70b-instruct": 131072,
    "deepseek-ai/deepseek-r1": 131072,
    "google/gemma-2-27b-it": 8192,
}

_DEFAULT_CONTEXT_LIMIT = 8192

# Safety margins
_RESPONSE_TOKEN_RESERVE = 2048  # reserve tokens for the response
_SAFETY_MARGIN = 0.90           # use at most 90% of context window


class TokenEstimator:
    """
    Fast token estimation for prompt budgeting and cost tracking.
    """

    def estimate(self, text: str) -> int:
        """
        Estimate token count for a string.
        Uses content-aware character ratio.
        """
        if not text:
            return 0

        length = len(text)

        # Detect content type for better accuracy
        code_indicators = len(re.findall(
            r"[{}\[\]();=<>]|def |class |import |function |const |var |let ",
            text[:500]
        ))

        if code_indicators > 5:
            ratio = _CHARS_PER_TOKEN_CODE
        elif code_indicators > 2:
            ratio = _CHARS_PER_TOKEN_MIXED
        else:
            ratio = _CHARS_PER_TOKEN_EN

        return max(1, int(length / ratio))

    def estimate_messages(self, messages: list[dict]) -> int:
        """
        Estimate token count for a list of chat messages.
        Accounts for message formatting overhead (~4 tokens per message).
        """
        total = 0
        for msg in messages:
            total += 4  # per-message overhead (role, content markers)
            content = msg.get("content", "")
            total += self.estimate(content)
        total += 2  # conversation framing
        return total

    def get_context_limit(self, model: str) -> int:
        """Get the context window limit for a model."""
        return MODEL_CONTEXT_LIMITS.get(model, _DEFAULT_CONTEXT_LIMIT)

    def get_available_tokens(self, model: str, messages: list[dict]) -> int:
        """
        Calculate available tokens for response generation.

        Args:
            model: Model name
            messages: Current conversation messages

        Returns:
            Available tokens for response (after prompt and safety margins)
        """
        limit = self.get_context_limit(model)
        usable = int(limit * _SAFETY_MARGIN)
        prompt_tokens = self.estimate_messages(messages)
        available = usable - prompt_tokens - _RESPONSE_TOKEN_RESERVE

        if available < 256:
            log.warning(
                "Token budget LOW: model=%s limit=%d prompt=%d available=%d",
                model, limit, prompt_tokens, available,
            )

        return max(256, available)

    def check_budget(self, model: str, messages: list[dict],
                     additional_context: str = "") -> dict:
        """
        Pre-flight token budget check before LLM call.

        Returns dict with:
            - within_budget: bool
            - prompt_tokens: estimated prompt tokens
            - available_tokens: tokens available for response
            - context_limit: model's full context window
            - utilization_pct: how much of context is used
        """
        limit = self.get_context_limit(model)
        prompt_tokens = self.estimate_messages(messages)

        if additional_context:
            prompt_tokens += self.estimate(additional_context)

        usable = int(limit * _SAFETY_MARGIN)
        available = usable - prompt_tokens - _RESPONSE_TOKEN_RESERVE
        utilization = prompt_tokens / limit * 100

        result = {
            "within_budget": available >= 256,
            "prompt_tokens": prompt_tokens,
            "available_tokens": max(0, available),
            "context_limit": limit,
            "utilization_pct": round(utilization, 1),
            "response_reserve": _RESPONSE_TOKEN_RESERVE,
        }

        if not result["within_budget"]:
            log.warning(
                "Token budget EXCEEDED: model=%s prompt=%d limit=%d",
                model, prompt_tokens, limit,
            )

        return result


# Module-level singleton
token_estimator = TokenEstimator()
