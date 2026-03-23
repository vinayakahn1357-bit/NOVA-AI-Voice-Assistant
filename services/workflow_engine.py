"""
services/workflow_engine.py — JSON Workflow Execution Engine for NOVA (Phase 7)
Define and execute multi-step workflows as JSON.
Supports AI, tool, and plugin step types with output chaining.
"""

import time
import re
from dataclasses import dataclass, field

from utils.logger import get_logger

log = get_logger("workflow")


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class StepResult:
    """Result of a single workflow step."""
    step_index: int
    step_type: str          # "ai", "tool", "plugin"
    name: str               # step name or tool/plugin name
    success: bool
    output: str
    error: str = None
    time_ms: int = 0


@dataclass
class WorkflowResult:
    """Result of a complete workflow execution."""
    success: bool
    workflow_name: str
    outputs: list = field(default_factory=list)     # List of StepResult dicts
    final_output: str = ""
    steps_completed: int = 0
    total_steps: int = 0
    errors: list = field(default_factory=list)
    total_time_ms: int = 0

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "workflow_name": self.workflow_name,
            "final_output": self.final_output,
            "steps_completed": self.steps_completed,
            "total_steps": self.total_steps,
            "errors": self.errors,
            "total_time_ms": self.total_time_ms,
            "outputs": [
                {
                    "step": o.step_index,
                    "type": o.step_type,
                    "name": o.name,
                    "success": o.success,
                    "output": o.output[:200],
                    "error": o.error,
                }
                for o in self.outputs
            ],
        }


# ─── Template Variable Resolution ────────────────────────────────────────────

def _resolve_template(text: str, variables: dict) -> str:
    """
    Replace {variable_name} placeholders in text.
    Supports: {input}, {prev_output}, {step_N_output}
    """
    if not text or not isinstance(text, str):
        return text or ""

    for key, value in variables.items():
        text = text.replace(f"{{{key}}}", str(value))

    return text


# ─── Workflow Engine ──────────────────────────────────────────────────────────

class WorkflowEngine:
    """
    Execute JSON-defined multi-step workflows.

    Workflow format:
    {
        "name": "workflow_name",
        "description": "What this workflow does",
        "steps": [
            {"type": "ai", "prompt": "Do something with {input}"},
            {"type": "tool", "name": "calculator", "argument": "5 * 3"},
            {"type": "plugin", "name": "web_search", "input": {"query": "{prev_output}"}}
        ]
    }

    Step types:
    - "ai": Call AIService with a prompt (template variables resolved)
    - "tool": Call a built-in tool
    - "plugin": Call a registered plugin

    Template variables:
    - {input}: The initial input to the workflow
    - {prev_output}: Output of the previous step
    - {step_N_output}: Output of step N (0-indexed)
    """

    def __init__(self, ai_service, tool_executor):
        self._ai = ai_service
        self._tools = tool_executor

    def execute(self, workflow_def: dict, initial_input: str = "",
                user_id: str = "default", abort_on_error: bool = False) -> WorkflowResult:
        """
        Execute a workflow definition.

        Args:
            workflow_def: JSON workflow definition dict
            initial_input: Initial input value (available as {input})
            user_id: User identifier for scoping
            abort_on_error: If True, stop on first step failure

        Returns:
            WorkflowResult with per-step outputs and final result
        """
        t0 = time.time()

        name = workflow_def.get("name", "unnamed")
        steps = workflow_def.get("steps", [])
        total_steps = len(steps)

        if not steps:
            return WorkflowResult(
                success=False, workflow_name=name,
                final_output="Workflow has no steps.",
                total_steps=0, errors=["Empty workflow"],
            )

        log.info("[Workflow] Starting '%s': %d steps, input='%.60s'",
                 name, total_steps, initial_input)

        # Template variable context
        variables = {
            "input": initial_input,
            "prev_output": initial_input,
        }

        step_results: list[StepResult] = []
        errors: list[str] = []

        for i, step_def in enumerate(steps):
            step_t0 = time.time()
            step_type = step_def.get("type", "ai")
            step_name = step_def.get("name", step_def.get("prompt", f"step_{i}")[:30])

            try:
                if step_type == "ai":
                    result = self._execute_ai_step(step_def, variables)
                elif step_type == "tool":
                    result = self._execute_tool_step(step_def, variables)
                elif step_type == "plugin":
                    result = self._execute_plugin_step(step_def, variables)
                else:
                    result = StepResult(
                        step_index=i, step_type=step_type, name=step_name,
                        success=False, output="",
                        error=f"Unknown step type: {step_type}",
                    )

                result.step_index = i
                result.time_ms = int((time.time() - step_t0) * 1000)

                # Update variables with this step's output
                if result.success:
                    variables["prev_output"] = result.output
                    variables[f"step_{i}_output"] = result.output
                else:
                    errors.append(f"Step {i} ({step_type}:{step_name}): {result.error}")
                    if abort_on_error:
                        step_results.append(result)
                        log.warning("[Workflow] '%s' aborted at step %d: %s",
                                    name, i, result.error)
                        break

                step_results.append(result)
                log.info("[Workflow] Step %d/%d %s (success=%s, %dms)",
                         i + 1, total_steps, step_type, result.success, result.time_ms)

            except Exception as exc:
                error_msg = f"Step {i} exception: {str(exc)}"
                errors.append(error_msg)
                step_results.append(StepResult(
                    step_index=i, step_type=step_type, name=step_name,
                    success=False, output="", error=error_msg,
                    time_ms=int((time.time() - step_t0) * 1000),
                ))
                log.warning("[Workflow] Step %d exception: %s", i, exc, exc_info=True)

                if abort_on_error:
                    break

        elapsed = int((time.time() - t0) * 1000)
        completed = sum(1 for r in step_results if r.success)
        final_output = variables.get("prev_output", "")

        log.info("[Workflow] '%s' completed: %d/%d steps, %dms",
                 name, completed, total_steps, elapsed)

        return WorkflowResult(
            success=completed == total_steps,
            workflow_name=name,
            outputs=step_results,
            final_output=final_output,
            steps_completed=completed,
            total_steps=total_steps,
            errors=errors,
            total_time_ms=elapsed,
        )

    def validate(self, workflow_def: dict) -> list[str]:
        """
        Validate a workflow definition without executing it.
        Returns list of error messages (empty = valid).
        """
        errors = []

        if not isinstance(workflow_def, dict):
            return ["Workflow must be a dict"]

        if "steps" not in workflow_def:
            errors.append("Missing 'steps' key")
            return errors

        steps = workflow_def["steps"]
        if not isinstance(steps, list):
            errors.append("'steps' must be a list")
            return errors

        valid_types = {"ai", "tool", "plugin"}
        for i, step in enumerate(steps):
            if not isinstance(step, dict):
                errors.append(f"Step {i}: must be a dict")
                continue

            step_type = step.get("type")
            if step_type not in valid_types:
                errors.append(f"Step {i}: invalid type '{step_type}' (must be: {valid_types})")

            if step_type == "ai" and not step.get("prompt"):
                errors.append(f"Step {i}: AI step missing 'prompt'")

            if step_type == "tool" and not step.get("name"):
                errors.append(f"Step {i}: tool step missing 'name'")

            if step_type == "plugin" and not step.get("name"):
                errors.append(f"Step {i}: plugin step missing 'name'")

        return errors

    # ── Step Executors ────────────────────────────────────────────────────────

    def _execute_ai_step(self, step_def: dict, variables: dict) -> StepResult:
        """Execute an AI (LLM) step."""
        prompt = _resolve_template(step_def.get("prompt", ""), variables)

        if not prompt:
            return StepResult(
                step_index=0, step_type="ai", name="ai",
                success=False, output="", error="Empty prompt",
            )

        messages = [{"role": "user", "content": prompt}]
        response, model, provider, meta = self._ai.generate(messages, prompt)

        return StepResult(
            step_index=0, step_type="ai", name="ai",
            success=True, output=response,
        )

    def _execute_tool_step(self, step_def: dict, variables: dict) -> StepResult:
        """Execute a tool step."""
        tool_name = step_def.get("name", "")
        argument = _resolve_template(step_def.get("argument", ""), variables)

        result = self._tools.execute(tool_name, argument)

        if result.get("success"):
            output = result.get("formatted", str(result.get("result", "")))
            return StepResult(
                step_index=0, step_type="tool", name=tool_name,
                success=True, output=output,
            )
        else:
            return StepResult(
                step_index=0, step_type="tool", name=tool_name,
                success=False, output="", error=result.get("error", "Tool failed"),
            )

    def _execute_plugin_step(self, step_def: dict, variables: dict) -> StepResult:
        """Execute a plugin step."""
        plugin_name = step_def.get("name", "")
        raw_input = step_def.get("input", {})

        # Resolve template variables in plugin input
        if isinstance(raw_input, dict):
            resolved_input = {
                k: _resolve_template(str(v), variables)
                for k, v in raw_input.items()
            }
        elif isinstance(raw_input, str):
            resolved_input = {"argument": _resolve_template(raw_input, variables)}
        else:
            resolved_input = {"argument": str(raw_input)}

        result = self._tools.execute(plugin_name, resolved_input)

        if result.get("success"):
            output = str(result.get("result", ""))
            return StepResult(
                step_index=0, step_type="plugin", name=plugin_name,
                success=True, output=output,
            )
        else:
            return StepResult(
                step_index=0, step_type="plugin", name=plugin_name,
                success=False, output="", error=result.get("error", "Plugin failed"),
            )
