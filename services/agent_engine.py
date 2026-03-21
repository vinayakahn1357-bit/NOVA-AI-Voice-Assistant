"""
services/agent_engine.py - AI Agent Decision Engine for NOVA (Phase 5)
Intercepts between QueryAnalyzer and LLM to decide HOW to respond:
- Normal: standard LLM response
- Decision: structured pros/cons analysis with recommendation
- Opinion: confident, opinionated suggestion
- Planning: multi-step task breakdown
- Tool: execute a built-in tool and return result

The agent does NOT replace the LLM — it augments the prompt with agent-mode
instructions so the LLM produces structured, intelligent output.
"""

import re
from utils.logger import get_logger

log = get_logger("agent")


# --- Agent Mode Detection Patterns ---

_DECISION_PATTERNS = [
    r"which (?:is|one|should|would be) (?:better|best|faster|more)",
    r"should i (?:use|learn|pick|choose|go with|switch to|buy|start)",
    r"(?:compare|vs\.?|versus)\s+\w+",
    r"(?:better|best)\s+(?:option|choice|approach|way|method)",
    r"what should i (?:do|use|choose|pick|learn|try)",
    r"recommend\s+(?:a|an|me|the)",
    r"(?:pros and cons|advantages and disadvantages|trade-?offs?)",
    r"is it (?:worth|better|good idea) to",
    r"(?:which|what)\s+(?:framework|language|tool|library|database)",
]

_OPINION_PATTERNS = [
    r"what do you (?:think|feel|believe|prefer|recommend|suggest)",
    r"(?:your|give me your|share your)\s+(?:opinion|thoughts?|view|take|suggestion|recommendation)",
    r"(?:suggest|advise|tell) me (?:what|which|something)",
    r"(?:in your (?:opinion|view|experience))",
    r"do you (?:think|believe|prefer|like|recommend)",
    r"what (?:would|do) you (?:suggest|recommend|prefer|choose)",
    r"give (?:your|me|an) (?:honest|personal)?\s*(?:opinion|thought|suggestion|advice)",
]

_PLANNING_PATTERNS = [
    r"(?:build|create|make|design|implement|develop|set up|setup)\s+(?:a|an|the|my)",
    r"(?:how (?:do i|to|can i|should i))\s+(?:build|create|make|design|implement|set up)",
    r"(?:steps? (?:to|for)|plan (?:to|for)|roadmap)",
    r"(?:guide|tutorial|walkthrough)\s+(?:for|to|on)",
    r"(?:project|app|website|system|api|backend|frontend|application)\s+(?:from scratch|from zero|step by step)",
]

_TOOL_KEYWORDS = {
    "calculator": ["calculate", "compute", "evaluate", "what is \\d"],
    "system_info": ["system info", "cpu usage", "memory usage", "ram", "system status"],
    "datetime": ["what time", "current time", "what date", "today's date", "what day is"],
    "unit_convert": ["convert", "km to", "miles to", "kg to", "celsius to", "fahrenheit to"],
}


class AgentEngine:
    """
    AI Agent Decision Engine.

    Decides the agent mode for each query and augments the prompt
    with agent-mode instructions so the LLM produces structured output.

    Does NOT replace existing AI pipeline — sits between QueryAnalyzer
    and PromptBuilder to inject intelligence.
    """

    def __init__(self, tool_executor=None):
        self._tools = tool_executor
        self._compiled_decision = [re.compile(p, re.IGNORECASE) for p in _DECISION_PATTERNS]
        self._compiled_opinion = [re.compile(p, re.IGNORECASE) for p in _OPINION_PATTERNS]
        self._compiled_planning = [re.compile(p, re.IGNORECASE) for p in _PLANNING_PATTERNS]

    def process(self, message, query_analysis=None):
        """
        Main agent decision point.

        Args:
            message: user's raw message
            query_analysis: dict from QueryAnalyzer

        Returns: {
            "agent_mode": str,      # "normal"|"decision"|"opinion"|"planning"|"tool"
            "action": str,          # what the agent decided to do
            "confidence": float,    # 0.0-1.0
            "prompt_augment": str,  # extra instructions to prepend to prompt
            "tool_result": dict,    # if mode=="tool", the tool output
            "skip_llm": bool,       # if True, return tool_result directly
        }
        """
        qa = query_analysis or {}

        # 1. Check for tool calls first (fastest)
        if self._tools:
            tool_name, tool_arg = self._tools.detect_tool(message)
            if tool_name:
                result = self._tools.execute(tool_name, tool_arg)
                if result.get("success"):
                    log.info("Agent: TOOL mode -> %s", tool_name)
                    return {
                        "agent_mode": "tool",
                        "action": "execute_%s" % tool_name,
                        "confidence": 1.0,
                        "prompt_augment": "",
                        "tool_result": result,
                        "skip_llm": True,
                    }

        # 2. Detect agent mode from message patterns
        mode = self._detect_mode(message)

        if mode == "decision":
            return self._build_decision(message, qa)
        elif mode == "opinion":
            return self._build_opinion(message, qa)
        elif mode == "planning":
            return self._build_planning(message, qa)

        # 3. Default: normal mode
        return {
            "agent_mode": "normal",
            "action": "standard_response",
            "confidence": 0.5,
            "prompt_augment": "",
            "tool_result": None,
            "skip_llm": False,
        }

    def _detect_mode(self, message):
        """Detect agent mode from message patterns."""
        lower = message.lower()

        # Decision mode: comparison, recommendation, choice
        for pat in self._compiled_decision:
            if pat.search(lower):
                return "decision"

        # Opinion mode: asking for personal take
        for pat in self._compiled_opinion:
            if pat.search(lower):
                return "opinion"

        # Planning mode: building/creating something
        for pat in self._compiled_planning:
            if pat.search(lower):
                return "planning"

        return "normal"

    def _build_decision(self, message, qa):
        """Build decision mode prompt augmentation."""
        augment = (
            "\n\n[AGENT MODE: DECISION]\n"
            "The user is asking you to make a decision or comparison. "
            "You MUST respond with:\n"
            "1. Brief analysis of each option\n"
            "2. Clear pros and cons for each\n"
            "3. Your definitive recommendation (pick ONE)\n"
            "4. Reasoning for your choice\n"
            "5. Confidence level (how sure you are)\n\n"
            "Be decisive. Do NOT say 'it depends' without following up with a clear recommendation. "
            "Give your honest expert opinion. "
            "Structure your response clearly with headers or bullet points.\n"
        )

        log.info("Agent: DECISION mode (query: %.60s...)", message)
        return {
            "agent_mode": "decision",
            "action": "analyze_and_recommend",
            "confidence": 0.85,
            "prompt_augment": augment,
            "tool_result": None,
            "skip_llm": False,
        }

    def _build_opinion(self, message, qa):
        """Build opinion mode prompt augmentation."""
        augment = (
            "\n\n[AGENT MODE: OPINION]\n"
            "The user is asking for YOUR opinion, suggestion, or personal recommendation. "
            "You MUST:\n"
            "1. Give a clear, confident opinion (not a neutral overview)\n"
            "2. State your recommendation directly\n"
            "3. Back it up with reasoning\n"
            "4. Be honest and specific\n\n"
            "Do NOT give wishy-washy 'it depends on your needs' answers. "
            "Take a clear stance. Be the expert friend who gives real advice. "
            "Start with your recommendation, then explain why.\n"
        )

        log.info("Agent: OPINION mode (query: %.60s...)", message)
        return {
            "agent_mode": "opinion",
            "action": "give_opinion",
            "confidence": 0.8,
            "prompt_augment": augment,
            "tool_result": None,
            "skip_llm": False,
        }

    def _build_planning(self, message, qa):
        """Build planning mode prompt augmentation."""
        augment = (
            "\n\n[AGENT MODE: PLANNING]\n"
            "The user wants to build/create something or needs a step-by-step plan. "
            "You MUST:\n"
            "1. Break the task into clear, numbered steps\n"
            "2. Each step should be actionable and specific\n"
            "3. Include code snippets where relevant\n"
            "4. Mention tools, libraries, or technologies needed\n"
            "5. Estimate effort or complexity for each step\n"
            "6. Provide a recommended order of execution\n\n"
            "Think like a senior architect planning a project. "
            "Be practical and specific, not vague. "
            "Include potential pitfalls and how to avoid them.\n"
        )

        log.info("Agent: PLANNING mode (query: %.60s...)", message)
        return {
            "agent_mode": "planning",
            "action": "create_plan",
            "confidence": 0.85,
            "prompt_augment": augment,
            "tool_result": None,
            "skip_llm": False,
        }

    def format_tool_response(self, tool_result):
        """Format a tool result into a human-friendly response string."""
        if not tool_result:
            return "Tool execution failed."

        tool_name = tool_result.get("tool", "unknown")
        result = tool_result.get("result", tool_result.get("formatted", ""))

        if tool_name == "calculator":
            expr = tool_result.get("expression", "")
            val = tool_result.get("result", "?")
            return "**%s = %s**" % (expr, val)

        if tool_name == "datetime":
            if isinstance(result, dict):
                return "**%s, %s** (%s)" % (
                    result.get("day", ""), result.get("date", ""),
                    result.get("time", ""),
                )
            return str(result)

        if tool_name == "system_info":
            if isinstance(result, dict):
                lines = [
                    "**System Status:**",
                    "- OS: %s" % result.get("os", "?"),
                    "- CPU: %s%%" % result.get("cpu_percent", "?"),
                    "- Memory: %s%% (%sGB / %sGB)" % (
                        result.get("memory_percent", "?"),
                        result.get("memory_used_gb", "?"),
                        result.get("memory_total_gb", "?"),
                    ),
                    "- Python: %s" % result.get("python_version", "?"),
                ]
                return "\n".join(lines)
            return str(result)

        if tool_name == "unit_convert":
            return "**%s**" % tool_result.get("formatted", str(result))

        # Generic fallback
        if isinstance(result, dict):
            return "\n".join("- %s: %s" % (k, v) for k, v in result.items())
        return str(result)
