"""
services/command_service.py — Intent Detection & Command Execution for NOVA
Detects /commands in chat messages and returns structured results.
"""

import re
from utils.logger import get_logger

log = get_logger("commands")


class CommandService:
    """
    Extensible command system for NOVA.
    Detects slash commands in user messages and returns structured output.
    """

    # Command prefix — messages starting with this are treated as commands
    PREFIX = "/"

    # Built-in command definitions
    _COMMANDS = {
        "reset":   {"description": "Reset the current conversation", "aliases": ["clear"]},
        "memory":  {"description": "Show what Nova remembers about you", "aliases": ["remember"]},
        "export":  {"description": "Export conversation as Markdown", "aliases": ["save"]},
        "help":    {"description": "Show available commands", "aliases": ["commands", "?"]},
        "status":  {"description": "Show system status", "aliases": ["sys", "info"]},
        "forget":  {"description": "Reset Nova's memory", "aliases": ["amnesia"]},
    }

    def __init__(self, session_service=None, memory_service=None):
        self._session = session_service
        self._memory = memory_service
        # Build alias lookup
        self._alias_map = {}
        for cmd, info in self._COMMANDS.items():
            self._alias_map[cmd] = cmd
            for alias in info.get("aliases", []):
                self._alias_map[alias] = cmd

    def is_command(self, message: str) -> bool:
        """Check if a message is a command (starts with /)."""
        return message.strip().startswith(self.PREFIX)

    def detect_intent(self, message: str) -> dict:
        """
        Detect the intent of a message.
        Returns: {"type": "command"|"conversation", "command": str|None, "args": str}
        """
        stripped = message.strip()
        if not stripped.startswith(self.PREFIX):
            return {"type": "conversation", "command": None, "args": stripped}

        # Parse command and arguments
        parts = stripped[1:].split(maxsplit=1)
        cmd_name = parts[0].lower() if parts else ""
        args = parts[1] if len(parts) > 1 else ""

        # Resolve alias
        resolved = self._alias_map.get(cmd_name)

        if resolved:
            return {"type": "command", "command": resolved, "args": args}

        # Unknown command — treat as conversation
        return {"type": "conversation", "command": None, "args": stripped}

    def execute(self, command: str, args: str, session_id: str = None) -> dict:
        """
        Execute a command and return structured output.
        Returns: {"response": str, "action": str|None, "data": dict|None}
        """
        handler = getattr(self, f"_cmd_{command}", None)
        if handler:
            return handler(args, session_id)
        return {
            "response": f"Unknown command: /{command}. Type /help for available commands.",
            "action": None,
            "data": None,
        }

    # ─── Built-in Command Handlers ────────────────────────────────────────

    def _cmd_help(self, args, session_id):
        lines = ["**Available Commands:**\n"]
        for cmd, info in sorted(self._COMMANDS.items()):
            aliases = ", ".join(f"/{a}" for a in info.get("aliases", []))
            alias_str = f" (aliases: {aliases})" if aliases else ""
            lines.append(f"• `/{cmd}` — {info['description']}{alias_str}")
        return {"response": "\n".join(lines), "action": None, "data": None}

    def _cmd_reset(self, args, session_id):
        if self._session and session_id:
            # Pass user_id for session isolation
            from flask import g, session as flask_session
            user_id = getattr(g, "user_id", None) or flask_session.get("user_id", "default")
            self._session.clear_session(session_id, user_id=user_id)
        return {
            "response": "Conversation has been reset. Let's start fresh! 🔄",
            "action": "reset_conversation",
            "data": None,
        }

    def _cmd_memory(self, args, session_id):
        if self._memory:
            stats = self._memory.get_stats()
            lines = [
                "**Nova's Memory:**\n",
                f"• **Facts known:** {stats.get('facts_count', 0)}",
                f"• **Interests tracked:** {stats.get('interests_count', 0)}",
                f"• **Top interests:** {', '.join(stats.get('top_interests', [])) or 'None yet'}",
                f"• **Conversations:** {stats.get('total_conversations', 0)}",
                f"• **Days active:** {stats.get('days_active', 0)}",
                f"• **First seen:** {stats.get('first_seen', 'Unknown')}",
            ]
            return {"response": "\n".join(lines), "action": None, "data": stats}
        return {"response": "Memory system is not available.", "action": None, "data": None}

    def _cmd_forget(self, args, session_id):
        if self._memory:
            self._memory.reset()
        return {
            "response": "All of Nova's memory has been reset. I won't remember past conversations. 🧹",
            "action": "memory_reset",
            "data": None,
        }

    def _cmd_export(self, args, session_id):
        return {
            "response": "Use the export button in the chat interface to download your conversation as Markdown.",
            "action": "export",
            "data": None,
        }

    def _cmd_status(self, args, session_id):
        return {
            "response": "Checking system status... See the dashboard for detailed system information.",
            "action": "show_status",
            "data": None,
        }
