"""
services/agent_runner.py — Autonomous Agent System for NOVA (Phase 7)
Multi-step reasoning loop: Think → Plan → Act → Observe → Repeat

Uses existing AIService for LLM calls and ToolExecutor for tool/plugin execution.
Does NOT modify any existing services — wraps them in an autonomous loop.
"""

import time
import json
import re
from dataclasses import dataclass, field, asdict

from utils.logger import get_logger

log = get_logger("agent_runner")

# ─── Constants ────────────────────────────────────────────────────────────────
DEFAULT_MAX_STEPS = 7
HARD_MAX_STEPS = 15


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class StepRecord:
    """Record of a single agent reasoning step."""
    step: int
    phase: str          # THINK, PLAN, ACT, OBSERVE
    content: str        # What happened in this phase
    action_type: str = ""   # "tool", "plugin", "ai", "none"
    action_name: str = ""   # Name of tool/plugin called
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AgentResult:
    """Result of an autonomous agent run."""
    success: bool
    final_answer: str
    steps: list = field(default_factory=list)    # List of StepRecord dicts
    steps_taken: int = 0
    tools_used: list = field(default_factory=list)
    total_time_ms: int = 0
    task: str = ""
    error: str = None

    def to_dict(self, include_steps: bool = False) -> dict:
        d = {
            "success": self.success,
            "final_answer": self.final_answer,
            "steps_taken": self.steps_taken,
            "tools_used": self.tools_used,
            "total_time_ms": self.total_time_ms,
            "task": self.task,
        }
        if include_steps:
            d["steps"] = self.steps
        if self.error:
            d["error"] = self.error
        return d


# ─── Agent Prompts ────────────────────────────────────────────────────────────

_AGENT_SYSTEM_PROMPT = """You are NOVA's autonomous agent. You solve complex tasks step-by-step.

## Capabilities
You have access to these tools/capabilities:
{capabilities}

## Instructions
For each step, respond in EXACTLY this JSON format:
```json
{{
    "thought": "What I'm thinking about the task and current state",
    "plan": "My next action and why",
    "action": {{
        "type": "tool" | "ai" | "done",
        "name": "tool_name (if type=tool)",
        "argument": "argument for the tool (if type=tool)",
        "query": "question to think about (if type=ai)"
    }},
    "is_done": false
}}
```

When you have enough information to give a final answer, set "is_done" to true and put your complete answer in "thought".

## Rules
- Break complex tasks into small, concrete steps
- Use tools when they can help (calculator, system_info, datetime, etc.)
- Each step should make meaningful progress
- Do NOT repeat the same action
- If stuck after 3 attempts, synthesize what you have and finish
- Always provide a final answer, even if partial
"""

_STEP_PROMPT = """## Current Task
{task}

## Work Done So Far
{history}

## Available Capabilities
{capabilities}

What is your next step? Respond in the JSON format specified.
Remember: set "is_done": true when you have enough to give a final answer."""


# ─── Agent Runner ─────────────────────────────────────────────────────────────

class AgentRunner:
    """
    Autonomous multi-step reasoning agent.

    Orchestrates a Think→Plan→Act→Observe loop using:
    - AIService for reasoning (LLM calls)
    - ToolExecutor for tool/plugin execution
    - MemoryService for context (optional)

    Does NOT modify any underlying service — purely a wrapper.
    """

    def __init__(self, ai_service, tool_executor, memory_service=None):
        self._ai = ai_service
        self._tools = tool_executor
        self._memory = memory_service

    def run(self, task: str, history: list = None, user_id: str = "default",
            max_steps: int = DEFAULT_MAX_STEPS, debug: bool = False) -> AgentResult:
        """
        Execute an autonomous agent run for a given task.

        Args:
            task: The user's task description
            history: Conversation history (list of {role, content} dicts)
            user_id: User identifier for memory scoping
            max_steps: Maximum reasoning steps (capped at HARD_MAX_STEPS)
            debug: If True, include step details in result

        Returns:
            AgentResult with final_answer and metadata
        """
        t0 = time.time()
        max_steps = min(max_steps, HARD_MAX_STEPS)
        steps: list[StepRecord] = []
        tools_used = set()
        work_history = []  # Track what the agent has done

        # Get available capabilities
        capabilities = self._format_capabilities()

        # Get memory context if available
        memory_ctx = ""
        if self._memory:
            try:
                memory_ctx = self._memory.get_context(user_id=user_id)
            except Exception:
                pass

        log.info("[AgentRunner] Starting: task='%.80s' max_steps=%d user=%s",
                 task, max_steps, user_id)

        for step_num in range(1, max_steps + 1):
            step_t0 = time.time()

            try:
                # ── THINK + PLAN: Ask LLM for next action ─────────────────
                history_text = self._format_work_history(work_history)

                step_prompt = _STEP_PROMPT.format(
                    task=task,
                    history=history_text or "(No work done yet)",
                    capabilities=capabilities,
                )

                # Build messages for AIService
                messages = []
                if history:
                    messages.extend(history[-4:])  # Keep last 4 for context
                if memory_ctx:
                    messages.append({"role": "system", "content": f"[Memory] {memory_ctx}"})
                messages.append({"role": "user", "content": step_prompt})

                # Call LLM via existing AIService
                response, model, provider, meta = self._ai.generate(
                    messages, step_prompt,
                    prompt_augment=_AGENT_SYSTEM_PROMPT.format(capabilities=capabilities),
                )

                # ── Parse agent response ──────────────────────────────────
                parsed = self._parse_agent_response(response)

                thought = parsed.get("thought", response[:200])
                plan = parsed.get("plan", "")
                action = parsed.get("action", {})
                is_done = parsed.get("is_done", False)

                # Record THINK step
                steps.append(StepRecord(
                    step=step_num, phase="THINK",
                    content=thought,
                    timestamp=time.time(),
                ))

                log.info("[AgentRunner] Step %d THINK: %.100s", step_num, thought)

                if is_done:
                    # Agent says it's done — use thought as final answer
                    steps.append(StepRecord(
                        step=step_num, phase="DONE",
                        content="Agent completed the task",
                        timestamp=time.time(),
                    ))
                    log.info("[AgentRunner] Done at step %d", step_num)

                    elapsed = int((time.time() - t0) * 1000)
                    return AgentResult(
                        success=True,
                        final_answer=thought,
                        steps=[s.to_dict() for s in steps] if debug else [],
                        steps_taken=step_num,
                        tools_used=list(tools_used),
                        total_time_ms=elapsed,
                        task=task,
                    )

                # ── ACT: Execute the planned action ───────────────────────
                action_type = action.get("type", "ai")
                action_name = action.get("name", "")
                observation = ""

                if action_type == "tool" and action_name:
                    # Execute tool/plugin via ToolExecutor
                    tool_arg = action.get("argument", "")
                    tool_result = self._tools.execute(action_name, tool_arg)

                    observation = self._format_tool_observation(tool_result)
                    tools_used.add(action_name)

                    steps.append(StepRecord(
                        step=step_num, phase="ACT",
                        content=f"Called tool '{action_name}' with: {tool_arg}",
                        action_type="tool", action_name=action_name,
                        timestamp=time.time(),
                    ))
                    log.info("[AgentRunner] Step %d ACT: tool=%s", step_num, action_name)

                elif action_type == "ai":
                    # Use LLM for sub-reasoning
                    sub_query = action.get("query", plan or thought)
                    sub_messages = [{"role": "user", "content": sub_query}]

                    sub_response, _, _, _ = self._ai.generate(sub_messages, sub_query)
                    observation = sub_response

                    steps.append(StepRecord(
                        step=step_num, phase="ACT",
                        content=f"AI sub-query: {sub_query[:100]}",
                        action_type="ai",
                        timestamp=time.time(),
                    ))
                    log.info("[AgentRunner] Step %d ACT: ai sub-query", step_num)

                else:
                    observation = "No action taken."
                    steps.append(StepRecord(
                        step=step_num, phase="ACT",
                        content="No action (continuing to think)",
                        action_type="none",
                        timestamp=time.time(),
                    ))

                # ── OBSERVE: Record observation ───────────────────────────
                steps.append(StepRecord(
                    step=step_num, phase="OBSERVE",
                    content=observation[:500],  # Limit observation size
                    timestamp=time.time(),
                ))

                work_history.append({
                    "step": step_num,
                    "thought": thought[:200],
                    "action": f"{action_type}:{action_name}" if action_name else action_type,
                    "observation": observation[:300],
                })

                log.info("[AgentRunner] Step %d OBSERVE: %.100s", step_num, observation)

            except Exception as exc:
                log.warning("[AgentRunner] Step %d error: %s", step_num, exc, exc_info=True)
                steps.append(StepRecord(
                    step=step_num, phase="ERROR",
                    content=f"Error: {str(exc)}",
                    timestamp=time.time(),
                ))
                work_history.append({
                    "step": step_num,
                    "thought": "Error occurred",
                    "action": "error",
                    "observation": str(exc)[:200],
                })

        # ── Max steps reached — synthesize final answer ───────────────────
        log.info("[AgentRunner] Max steps (%d) reached. Synthesizing answer.", max_steps)

        final_answer = self._synthesize_final(task, work_history)
        elapsed = int((time.time() - t0) * 1000)

        return AgentResult(
            success=True,
            final_answer=final_answer,
            steps=[s.to_dict() for s in steps] if debug else [],
            steps_taken=max_steps,
            tools_used=list(tools_used),
            total_time_ms=elapsed,
            task=task,
        )

    # ── Streaming Interface ───────────────────────────────────────────────────

    def run_stream(self, task: str, history: list = None, user_id: str = "default",
                   max_steps: int = DEFAULT_MAX_STEPS):
        """
        Generator that yields step events for real-time streaming.

        Yields dicts with:
            {"type": "step", "phase": "THINK"|"ACT"|"OBSERVE", "content": ..., "step": N}
            {"type": "done", "final_answer": ..., "steps_taken": N, "tools_used": [...]}
        """
        t0 = time.time()
        max_steps = min(max_steps, HARD_MAX_STEPS)
        tools_used = set()
        work_history = []
        capabilities = self._format_capabilities()

        memory_ctx = ""
        if self._memory:
            try:
                memory_ctx = self._memory.get_context(user_id=user_id)
            except Exception:
                pass

        for step_num in range(1, max_steps + 1):
            try:
                history_text = self._format_work_history(work_history)

                step_prompt = _STEP_PROMPT.format(
                    task=task,
                    history=history_text or "(No work done yet)",
                    capabilities=capabilities,
                )

                messages = []
                if history:
                    messages.extend(history[-4:])
                if memory_ctx:
                    messages.append({"role": "system", "content": f"[Memory] {memory_ctx}"})
                messages.append({"role": "user", "content": step_prompt})

                response, model, provider, meta = self._ai.generate(
                    messages, step_prompt,
                    prompt_augment=_AGENT_SYSTEM_PROMPT.format(capabilities=capabilities),
                )

                parsed = self._parse_agent_response(response)
                thought = parsed.get("thought", response[:200])
                action = parsed.get("action", {})
                is_done = parsed.get("is_done", False)

                # Yield THINK
                yield {"type": "step", "phase": "THINK", "content": thought, "step": step_num}

                if is_done:
                    elapsed = int((time.time() - t0) * 1000)
                    yield {
                        "type": "done",
                        "final_answer": thought,
                        "steps_taken": step_num,
                        "tools_used": list(tools_used),
                        "total_time_ms": elapsed,
                    }
                    return

                # ACT
                action_type = action.get("type", "ai")
                action_name = action.get("name", "")
                observation = ""

                if action_type == "tool" and action_name:
                    tool_arg = action.get("argument", "")
                    yield {"type": "step", "phase": "ACT",
                           "content": f"Calling {action_name}({tool_arg})", "step": step_num}

                    tool_result = self._tools.execute(action_name, tool_arg)
                    observation = self._format_tool_observation(tool_result)
                    tools_used.add(action_name)

                elif action_type == "ai":
                    sub_query = action.get("query", thought)
                    yield {"type": "step", "phase": "ACT",
                           "content": f"Reasoning: {sub_query[:100]}", "step": step_num}

                    sub_response, _, _, _ = self._ai.generate(
                        [{"role": "user", "content": sub_query}], sub_query
                    )
                    observation = sub_response
                else:
                    yield {"type": "step", "phase": "ACT",
                           "content": "Continuing to think...", "step": step_num}

                # OBSERVE
                yield {"type": "step", "phase": "OBSERVE",
                       "content": observation[:500], "step": step_num}

                work_history.append({
                    "step": step_num,
                    "thought": thought[:200],
                    "action": f"{action_type}:{action_name}" if action_name else action_type,
                    "observation": observation[:300],
                })

            except Exception as exc:
                log.warning("[AgentRunner] Stream step %d error: %s", step_num, exc)
                yield {"type": "step", "phase": "ERROR",
                       "content": str(exc), "step": step_num}
                work_history.append({
                    "step": step_num, "thought": "Error",
                    "action": "error", "observation": str(exc)[:200],
                })

        # Max steps — synthesize
        final = self._synthesize_final(task, work_history)
        elapsed = int((time.time() - t0) * 1000)
        yield {
            "type": "done",
            "final_answer": final,
            "steps_taken": max_steps,
            "tools_used": list(tools_used),
            "total_time_ms": elapsed,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _format_capabilities(self) -> str:
        """Format available tools/plugins as a string for prompts."""
        caps = self._tools.get_tools()
        if not caps:
            return "(No tools available)"
        lines = [f"- **{name}**: {desc}" for name, desc in caps.items()]
        return "\n".join(lines)

    def _format_work_history(self, work_history: list) -> str:
        """Format past steps for the agent's context window."""
        if not work_history:
            return ""
        lines = []
        for step in work_history:
            lines.append(f"Step {step['step']}:")
            lines.append(f"  Thought: {step['thought']}")
            lines.append(f"  Action: {step['action']}")
            lines.append(f"  Result: {step['observation'][:150]}")
        return "\n".join(lines)

    def _parse_agent_response(self, response: str) -> dict:
        """Parse the agent's JSON response, with fallback for non-JSON."""
        # Try to extract JSON from the response
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try raw JSON
        try:
            return json.loads(response)
        except (json.JSONDecodeError, TypeError):
            pass

        # Try to find JSON object in text
        brace_match = re.search(r'\{.*\}', response, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group())
            except json.JSONDecodeError:
                pass

        # Fallback: treat entire response as thought, mark as done
        return {
            "thought": response,
            "plan": "",
            "action": {"type": "ai"},
            "is_done": len(response) > 100,  # Long response likely = final answer
        }

    def _format_tool_observation(self, tool_result: dict) -> str:
        """Format a tool result into a text observation."""
        if not tool_result:
            return "Tool returned no result."

        if tool_result.get("error"):
            return f"Tool error: {tool_result['error']}"

        result = tool_result.get("result", tool_result.get("formatted", ""))
        formatted = tool_result.get("formatted", "")

        if formatted:
            return f"Tool result: {formatted}"
        if isinstance(result, dict):
            return "Tool result:\n" + "\n".join(
                f"  {k}: {v}" for k, v in result.items()
            )
        return f"Tool result: {result}"

    def _synthesize_final(self, task: str, work_history: list) -> str:
        """Use LLM to synthesize a final answer from work history."""
        if not work_history:
            return "I wasn't able to complete this task. Please try rephrasing your request."

        history_text = self._format_work_history(work_history)
        synthesis_prompt = (
            f"You worked on this task: {task}\n\n"
            f"Here's the work you did:\n{history_text}\n\n"
            f"Now synthesize all findings into a clear, complete final answer. "
            f"Be concise but thorough."
        )

        try:
            messages = [{"role": "user", "content": synthesis_prompt}]
            response, _, _, _ = self._ai.generate(messages, synthesis_prompt)
            return response
        except Exception as exc:
            log.warning("[AgentRunner] Synthesis failed: %s", exc)
            # Fallback: concatenate observations
            observations = [s["observation"] for s in work_history if s.get("observation")]
            return "\n\n".join(observations) if observations else (
                "I worked on your task but couldn't synthesize a complete answer. "
                "Please try again with a more specific request."
            )


# ─── Agent Task Detection ────────────────────────────────────────────────────

_AGENT_TRIGGER_PATTERNS = [
    r"(?:research|investigate|analyze deeply|deep dive)\s+",
    r"(?:figure out|work out|solve step by step)\s+",
    r"(?:build a plan|create a strategy|develop a plan)\s+",
    r"(?:compare|evaluate|assess)\s+.+\s+(?:and|vs|versus)\s+",
    r"agent mode",
    r"(?:multi-step|step.by.step)\s+(?:analysis|breakdown|investigation)",
    r"(?:thoroughly|comprehensively|extensively)\s+(?:research|analyze|examine|review)",
]

_COMPILED_AGENT_TRIGGERS = [re.compile(p, re.IGNORECASE) for p in _AGENT_TRIGGER_PATTERNS]


def should_use_agent(message: str) -> bool:
    """
    Detect if a message should trigger autonomous agent processing.
    Returns True for complex, multi-step tasks.
    """
    lower = message.lower().strip()

    # Explicit trigger
    if "agent mode" in lower:
        return True

    # Pattern matching
    for pat in _COMPILED_AGENT_TRIGGERS:
        if pat.search(lower):
            return True

    return False
