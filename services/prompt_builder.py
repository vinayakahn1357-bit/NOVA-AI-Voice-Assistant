"""
services/prompt_builder.py — Prompt Construction Pipeline for NOVA
Builds structured prompts for both Ollama and OpenAI/Groq formats.
"""

from config import get_settings
from utils.logger import get_logger

log = get_logger("prompt_builder")


class PromptBuilder:
    """Constructs prompts with system instructions, memory context, and history."""

    def __init__(self, memory_service=None):
        self._memory = memory_service

    def _get_memory_context(self) -> str:
        """Safely retrieve memory context string."""
        if not self._memory:
            return ""
        try:
            return self._memory.get_context()
        except Exception as exc:
            log.warning("Memory context failed (ignored): %s", exc)
            return ""

    def build_ollama_prompt(self, history: list) -> str:
        """
        Build a single prompt string for Ollama-style API.
        Format:  System prompt + Memory + User: ... / Nova: ...
        """
        settings = get_settings()
        base_prompt = settings["system_prompt"].strip()
        memory_ctx = self._get_memory_context()

        lines = [base_prompt + memory_ctx]
        for msg in history:
            role = "User" if msg["role"] == "user" else "Nova"
            lines.append(f"{role}: {msg['content']}")
        lines.append("Nova:")
        return "\n".join(lines)

    def build_chat_messages(self, history: list) -> list:
        """
        Build OpenAI/Groq-format messages list.
        Returns: [{"role": "system", "content": ...}, {"role": "user/assistant", ...}, ...]
        """
        settings = get_settings()
        memory_ctx = self._get_memory_context()
        system_msg = settings["system_prompt"].strip() + memory_ctx

        messages = [{"role": "system", "content": system_msg}]
        for msg in history:
            role = "user" if msg["role"] == "user" else "assistant"
            messages.append({"role": role, "content": msg["content"]})
        return messages
