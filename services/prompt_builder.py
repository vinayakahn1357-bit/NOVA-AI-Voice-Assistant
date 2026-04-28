"""
services/prompt_builder.py — Structured Prompt Pipeline for NOVA (Phase 12)

Builds identical prompts for all providers: system prompt → personality → memory → context → user input.

Phase 12 upgrade: Full personality system prompt injection with enforcement lock.
"""

from config import get_settings
from utils.logger import get_logger

log = get_logger("prompt_builder")


class PromptBuilder:
    """
    Structured prompt pipeline ensuring identical prompts for all providers.

    Pipeline:
        1. System prompt (NOVA base identity)
        2. Personality system prompt (full per-personality instruction block)
        3. Enforcement lock (strict character enforcement text)
        4. Forbidden phrases instruction (what to never say)
        5. Memory context (learned facts, interests, preferences)
        6. Conversation history (short-term context / continuity)
        7. User input (current message)
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
        """Build the full system block: base prompt + personality + memory."""
        settings = get_settings()
        system = settings["system_prompt"].strip()

        # Inject personality block (Phase 12: full system prompt + enforcement)
        personality_block = self._get_personality_block(personality)
        if personality_block:
            system += personality_block

        memory = self._get_memory_context()
        return system + memory

    @staticmethod
    def _get_personality_block(personality: str) -> str:
        """
        Build the full personality instruction block.

        Phase 12: Injects:
        1. Full personality system prompt
        2. Strict enforcement lock
        3. Forbidden phrases list

        Returns empty string for 'default' (base system prompt handles it).
        """
        if not personality or personality == "default":
            # For default, just inject enforcement of global forbidden phrases
            try:
                from services.personality_service import GLOBAL_FORBIDDEN_PHRASES
                forbidden_list = "\n".join(f"  - \"{p}\"" for p in GLOBAL_FORBIDDEN_PHRASES[:10])
                return (
                    "\n\n## Behavioral Rules\n"
                    "You MUST strictly follow this personality. Breaking character is not allowed.\n"
                    f"NEVER use these phrases:\n{forbidden_list}\n"
                )
            except ImportError:
                return ""

        try:
            from services.personality_service import (
                get_personality_config,
                get_enforcement_prompt,
                get_system_prompt,
                get_forbidden_phrases,
                VALID_PERSONALITIES,
            )

            # Safety: fall back to default for unknown keys
            if personality not in VALID_PERSONALITIES:
                log.warning("Unknown personality '%s' in prompt builder — using default", personality)
                return ""

            config = get_personality_config(personality)
            p_system_prompt = get_system_prompt(personality)
            enforcement = get_enforcement_prompt(personality)
            forbidden = get_forbidden_phrases(personality)

            # Build forbidden phrases injection (limit to 12 most important)
            key_forbidden = [p for p in forbidden if p not in [
                "Great question", "I'd be happy to help", "As an AI"
            ]][:12]
            forbidden_list = "\n".join(f'  - "{p}"' for p in key_forbidden)

            block = (
                f"\n\n## Active Personality: {config['name']} {config.get('emoji', '')}\n"
                f"**Tone:** {config.get('tone', 'balanced')} | "
                f"**Structure:** {config.get('structure', 'adaptive')} | "
                f"**Depth:** {config.get('depth', 'medium')}\n\n"
                f"### Personality Instructions\n"
                f"{p_system_prompt}\n"
                f"{enforcement}\n\n"
                f"### Forbidden Phrases (NEVER use these)\n"
                f"{forbidden_list}\n"
            )

            log.debug("Personality block built for '%s' (%d chars)", personality, len(block))
            return block

        except ImportError:
            log.warning("personality_service not available — skipping personality block")
            return ""
        except Exception as exc:
            log.warning("Personality block build failed (%s) — skipping", exc)
            return ""

    # ─── Plain Text Format (legacy) ───────────────────────────────────────

    def build_plain_prompt(self, history: list, personality: str = "default") -> str:
        """
        Build a single prompt string for plain-text API format.
        Both Groq and NVIDIA use chat messages format instead.
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
        Build OpenAI/Groq-format messages list with full personality injection.

        Returns: [
            {"role": "system", "content": base + personality + enforcement + memory},
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

    # ─── Multimodal / Attachments ─────────────────────────────────────────

    def build_with_attachments(self, history: list, attachments: list | None = None,
                               personality: str = "default") -> list:
        """
        Build chat messages with optional multimodal attachments.
        Personality injection is preserved.
        """
        messages = self.build_chat_messages(history, personality)

        if attachments:
            attachment_ctx = []
            for att in attachments:
                att_type = att.get("type", "unknown")
                content = att.get("content", "")
                if att_type == "pdf":
                    attachment_ctx.append(f"[Attached PDF content]:\n{content}\n")
                elif att_type == "image":
                    attachment_ctx.append(f"[Attached image description]:\n{content}\n")

            if attachment_ctx:
                ctx_msg = {
                    "role": "system",
                    "content": (
                        "The user has attached the following files. "
                        "Use this content to answer their question:\n\n"
                        + "\n".join(attachment_ctx)
                    )
                }
                messages.insert(-1, ctx_msg)

        return messages

    # ─── Strict Regeneration Prompt ───────────────────────────────────────

    def build_strict_regen_messages(self, history: list, personality: str,
                                    failed_response: str) -> list:
        """
        Build a reinforced prompt for regeneration when the first response
        fails personality scoring. Adds extra enforcement context.

        Called by ResponsePipeline when score < threshold.
        """
        messages = self.build_chat_messages(history, personality)

        try:
            from services.personality_service import get_personality_config
            config = get_personality_config(personality)
            required_format = config.get("required_format", "")
            tone = config.get("tone", "")
            depth = config.get("depth", "")
        except ImportError:
            return messages

        regen_instruction = (
            f"\n\n## REGENERATION — PERSONALITY ENFORCEMENT FAILED\n"
            f"Your previous response did not match the required personality '{personality}'.\n"
            f"You MUST correct this in your next response.\n\n"
            f"REQUIRED tone: {tone}\n"
            f"REQUIRED structure: {required_format or 'adaptive'}\n"
            f"REQUIRED depth: {depth}\n\n"
            f"Previous (non-compliant) response:\n"
            f"\"\"\"\n{failed_response[:300]}\n\"\"\"\n\n"
            f"Generate a NEW response that STRICTLY follows the {personality} personality. "
            f"Do not repeat the above response."
        )

        # Inject regen instruction as a system message before the last user message
        messages.insert(-1, {"role": "system", "content": regen_instruction})
        log.info("Regen prompt built for personality '%s'", personality)
        return messages

    # ─── Premium PDF Analysis Prompt ──────────────────────────────────────

    def build_pdf_analysis_prompt(self, user_question: str, filename: str,
                                  retrieved_chunks: list[dict] | None = None,
                                  summary: str = "") -> str:
        """
        Build a premium PDF analysis prompt that instructs the LLM to
        interpret, synthesize, and structure document content.

        Used by chat_controller when document context is detected.
        Replaces raw text injection with structured analysis instructions.
        """
        # Build context from retrieved chunks or summary
        if retrieved_chunks:
            from services.smart_responder import SmartResponder
            doc_context = SmartResponder.build_retrieval_context(retrieved_chunks, filename)
        elif summary:
            doc_context = f"[Document Summary: {filename}]\n{summary}"
        else:
            return user_question  # No context available

        prompt = (
            f"You are answering based on a previously uploaded document: **{filename}**.\n\n"
            f"## Document Analysis Instructions\n"
            f"- **Interpret and synthesize** — never just quote raw text. Extract meaning.\n"
            f"- **Structure your response** as: Key Insight → Supporting Details → Implications\n"
            f"- **Cite page numbers** when referencing specific content (e.g., 'As stated on Page 3...')\n"
            f"- **Identify patterns** — connect concepts across different sections\n"
            f"- **Explain jargon** — define technical terms inline when relevant\n"
            f"- **Preserve critical data** — keep statistics, dates, and figures exact\n"
            f"- **Be comprehensive** but concise — executive summary quality\n\n"
            f"## Relevant Document Sections\n{doc_context}\n\n"
            f"## User's Question\n{user_question}"
        )

        return prompt

