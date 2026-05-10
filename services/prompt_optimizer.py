"""
services/prompt_optimizer.py — Prompt Optimizer for NOVA (Phase 14)

Optimizes prompts before LLM submission to reduce token usage:
    - Trims excessive conversation history
    - Compresses system messages
    - Removes duplicate instructions
    - Enforces max prompt size
    - Adapts to model context limits

Works with TokenEstimator for budget-aware optimization.
"""

import re
from utils.logger import get_logger

log = get_logger("prompt_optimizer")

# ── Config ────────────────────────────────────────────────────────────────────
_MAX_HISTORY_MESSAGES = 20     # keep last N messages
_MAX_SYSTEM_CHARS = 3000       # truncate system prompts beyond this
_MIN_MESSAGES_KEEP = 4         # always keep at least last 4 messages
_DUPLICATE_THRESHOLD = 0.85    # similarity threshold for duplicate detection


class PromptOptimizer:
    """
    Optimizes chat message lists for minimal token usage while
    preserving conversation quality.
    """

    def __init__(self, token_estimator=None):
        self._estimator = token_estimator

    def optimize(self, messages: list[dict], model: str = "",
                 max_tokens: int | None = None) -> list[dict]:
        """
        Optimize a message list for token efficiency.

        Args:
            messages: Chat messages list
            model: Model name (for context limit awareness)
            max_tokens: Override max prompt tokens

        Returns:
            Optimized messages list (may be shorter)
        """
        if not messages:
            return messages

        try:
            optimized = list(messages)

            # Step 1: Trim history to max messages
            optimized = self._trim_history(optimized)

            # Step 2: Remove duplicate system instructions
            optimized = self._dedup_system_messages(optimized)

            # Step 3: Compress verbose system messages
            optimized = self._compress_system_messages(optimized)

            # Step 4: Token budget enforcement
            if self._estimator and model:
                optimized = self._enforce_budget(optimized, model, max_tokens)

            original_count = len(messages)
            final_count = len(optimized)
            if original_count != final_count:
                log.info("Prompt optimized: %d -> %d messages",
                         original_count, final_count)

            return optimized

        except Exception as exc:
            log.warning("Prompt optimization error (returning original): %s", exc)
            return messages

    def _trim_history(self, messages: list[dict]) -> list[dict]:
        """Keep system messages + last N conversation messages."""
        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        if len(non_system) > _MAX_HISTORY_MESSAGES:
            trimmed = non_system[-_MAX_HISTORY_MESSAGES:]
            log.debug("History trimmed: %d -> %d messages",
                       len(non_system), len(trimmed))
            return system_msgs + trimmed

        return messages

    def _dedup_system_messages(self, messages: list[dict]) -> list[dict]:
        """Remove duplicate system messages (keeping the last occurrence)."""
        seen_fingerprints: set[str] = set()
        result = []

        # Process in reverse to keep latest duplicates
        for msg in reversed(messages):
            if msg.get("role") == "system":
                content = msg.get("content", "")
                fp = self._fingerprint(content)
                if fp in seen_fingerprints:
                    continue
                seen_fingerprints.add(fp)
            result.append(msg)

        result.reverse()
        return result

    def _compress_system_messages(self, messages: list[dict]) -> list[dict]:
        """Truncate overly long system messages."""
        result = []
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if len(content) > _MAX_SYSTEM_CHARS:
                    msg = {**msg, "content": content[:_MAX_SYSTEM_CHARS] + "..."}
                    log.debug("System message compressed: %d -> %d chars",
                              len(content), _MAX_SYSTEM_CHARS)
            result.append(msg)
        return result

    def _enforce_budget(self, messages: list[dict], model: str,
                        max_tokens: int | None) -> list[dict]:
        """Remove old messages until we're within token budget."""
        if not self._estimator:
            return messages

        budget = self._estimator.check_budget(model, messages)
        if budget["within_budget"]:
            return messages

        # Progressively remove oldest non-system messages
        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        while (len(non_system) > _MIN_MESSAGES_KEEP
               and not self._estimator.check_budget(
                   model, system_msgs + non_system
               )["within_budget"]):
            removed = non_system.pop(0)
            log.debug("Budget enforcement: removed message (role=%s, len=%d)",
                      removed.get("role"), len(removed.get("content", "")))

        log.info("Budget enforced: %d -> %d non-system messages",
                 len(messages) - len(system_msgs), len(non_system))
        return system_msgs + non_system

    @staticmethod
    def _fingerprint(text: str) -> str:
        """Create a normalized fingerprint for dedup."""
        return re.sub(r"\s+", " ", text.lower().strip())[:200]


# Module-level singleton
prompt_optimizer = PromptOptimizer()
