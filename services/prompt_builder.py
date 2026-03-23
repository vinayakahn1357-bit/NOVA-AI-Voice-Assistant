"""
services/prompt_builder.py — Structured Prompt Pipeline for NOVA
Builds identical prompts for all providers: system prompt → personality → memory → context → user input.
"""

from config import get_settings
from utils.logger import get_logger

log = get_logger("prompt_builder")


class PromptBuilder:
    """
    Structured prompt pipeline ensuring identical prompts for all providers.

    Pipeline:
        1. System prompt (assistant base personality)
        2. Personality instruction (teacher/friend/expert/coach)
        3. Memory context (learned facts, interests, preferences)
        4. Conversation history (short-term context / continuity)
        5. User input (current message)
    """

    def __init__(self, memory_service=None):
        self._memory = memory_service

    def _get_memory_context(self) -> str:
        """Safely retrieve memory context string."""
        if not self._memory:
            return ""
        try:
            ctx = self._memory.get_context()
            if ctx and ctx.strip():
                return (
                    "\n\n## Your Memory (use this to personalise)\n"
                    + ctx.strip()
                    + "\n"
                )
            return ""
        except Exception as exc:
            log.warning("Memory context failed (ignored): %s", exc)
            return ""

    def _build_system_block(self, personality: str = "default") -> str:
        """Build the full system block: prompt + personality + memory."""
        settings = get_settings()
        system = settings["system_prompt"].strip()

        # Inject personality instruction
        personality_block = self._get_personality_block(personality)
        if personality_block:
            system += personality_block

        memory = self._get_memory_context()
        return system + memory

    @staticmethod
    def _get_personality_block(personality: str) -> str:
        """Build personality instruction block if not default."""
        if not personality or personality == "default":
            return ""
        try:
            from services.personality_service import PERSONALITIES
            info = PERSONALITIES.get(personality)
            if info:
                return (
                    f"\n\n## Response Personality: {info['name']}\n"
                    f"{info['instruction']}\n"
                )
        except ImportError:
            log.warning("personality_service not available")
        return ""

    # ─── Ollama Format ────────────────────────────────────────────────────

    def build_ollama_prompt(self, history: list, personality: str = "default") -> str:
        """
        Build a single prompt string for Ollama-style API.
        Identical content to chat messages but in Ollama's format.

        Structure:
            [System prompt + Personality + Memory]
            User: message
            Nova: response
            ...
            Nova:
        """
        system_block = self._build_system_block(personality)

        lines = [system_block]
        for msg in history:
            role = "User" if msg["role"] == "user" else "Nova"
            lines.append(f"{role}: {msg['content']}")
        lines.append("Nova:")
        return "\n".join(lines)

    # ─── OpenAI/Groq Chat Format ──────────────────────────────────────────

    def build_chat_messages(self, history: list, personality: str = "default") -> list:
        """
        Build OpenAI/Groq-format messages list.
        Identical system context as Ollama, just different format.

        Returns: [
            {"role": "system", "content": system + personality + memory},
            {"role": "user/assistant", "content": ...},
            ...
        ]
        """
        system_block = self._build_system_block(personality)

        messages = [{"role": "system", "content": system_block}]
        for msg in history:
            role = "user" if msg["role"] == "user" else "assistant"
            messages.append({"role": role, "content": msg["content"]})
        return messages

    # ─── Multimodal Future ────────────────────────────────────────────────

    def build_with_attachments(self, history: list, attachments: list = None,
                               personality: str = "default") -> list:
        """
        Build chat messages with optional multimodal attachments.
        Currently passes through to build_chat_messages.
        When PDF/image services are implemented, this will inject
        extracted text/descriptions into the context.

        Args:
            history: conversation history
            attachments: list of {"type": "pdf"|"image", "content": str}
            personality: personality mode key

        Returns: OpenAI-format messages with attachment context injected.
        """
        messages = self.build_chat_messages(history, personality)

        if attachments:
            attachment_ctx = []
            for att in attachments:
                att_type = att.get("type", "unknown")
                content = att.get("content", "")
                if att_type == "pdf":
                    attachment_ctx.append(
                        f"[Attached PDF content]:\n{content}\n"
                    )
                elif att_type == "image":
                    attachment_ctx.append(
                        f"[Attached image description]:\n{content}\n"
                    )

            if attachment_ctx:
                # Inject attachment context before the last user message
                ctx_msg = {
                    "role": "system",
                    "content": (
                        "The user has attached the following files. "
                        "Use this content to answer their question:\n\n"
                        + "\n".join(attachment_ctx)
                    )
                }
                # Insert before the last message
                messages.insert(-1, ctx_msg)

        return messages

