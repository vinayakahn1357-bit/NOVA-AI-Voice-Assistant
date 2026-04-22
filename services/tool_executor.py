"""
services/tool_executor.py - Built-in Tool System for NOVA Agent (Phase 7)
Provides lightweight tools the agent can call without external APIs.
Phase 7: Unified capability registry (tools + plugins as one abstraction).
"""

import re
import math
import platform
import datetime

import psutil

from utils.logger import get_logger

log = get_logger("tools")


# --- Tool Registry ---

_TOOLS = {}


def tool(name, description):
    """Decorator to register a tool."""
    def wrapper(fn):
        _TOOLS[name] = {"fn": fn, "description": description, "name": name}
        return fn
    return wrapper


# --- Built-in Tools ---

@tool("calculator", "Evaluate mathematical expressions safely")
def tool_calculator(expression):
    """
    Evaluate a math expression safely using AST parsing.
    Supports: +, -, *, /, **, sqrt, sin, cos, tan, log, pi, e, abs, round
    NO eval() — uses ast.parse() with a safe node walker.
    """
    import ast
    import operator
    from typing import Any

    # Allowed binary operators
    _OPS: dict[type, Any] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    # Allowed function names → callables
    _SAFE_FUNCS = {
        "abs": abs, "round": round, "min": min, "max": max,
        "int": int, "float": float, "pow": pow,
        "sqrt": math.sqrt,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "asin": math.asin, "acos": math.acos, "atan": math.atan,
        "log": math.log, "log10": math.log10, "log2": math.log2,
        "ceil": math.ceil, "floor": math.floor,
        "factorial": math.factorial,
    }

    # Allowed constants
    _SAFE_NAMES = {
        "pi": math.pi,
        "e": math.e,
        "tau": math.tau,
        "inf": math.inf,
    }

    def _safe_eval_node(node):
        """Recursively evaluate an AST node — only safe math operations."""
        # Numbers: 3, 4.5, etc.
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value

        # Named constants: pi, e
        if isinstance(node, ast.Name) and node.id in _SAFE_NAMES:
            return _SAFE_NAMES[node.id]

        # Unary operators: -5, +3
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_safe_eval_node(node.operand))

        # Binary operators: 3 + 4, 2 ** 8
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            left = _safe_eval_node(node.left)
            right = _safe_eval_node(node.right)
            # Safety: prevent huge exponents
            if isinstance(node.op, ast.Pow) and isinstance(right, (int, float)) and abs(right) > 1000:
                raise ValueError("Exponent too large (max 1000)")
            return _OPS[type(node.op)](left, right)

        # Function calls: sqrt(16), sin(3.14)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _SAFE_FUNCS:
                args = [_safe_eval_node(arg) for arg in node.args]
                return _SAFE_FUNCS[node.func.id](*args)
            raise ValueError("Unknown function: %s" % getattr(node.func, 'id', '?'))

        raise ValueError("Unsupported expression element: %s" % type(node).__name__)

    try:
        expr = expression.strip()
        if not expr:
            return {"error": "Empty expression"}

        # Replace common notations
        expr = expr.replace("^", "**").replace("%", "/100")

        # Parse into AST and evaluate safely
        tree = ast.parse(expr, mode="eval")
        result = _safe_eval_node(tree.body)

        return {
            "result": result,
            "expression": expression.strip(),
            "formatted": str(result),
        }
    except ZeroDivisionError:
        return {"error": "Division by zero"}
    except (ValueError, TypeError) as e:
        return {"error": "Cannot evaluate: %s" % str(e)}
    except SyntaxError:
        return {"error": "Invalid math expression"}


@tool("system_info", "Get current system information")
def tool_system_info(_=""):
    """Return system stats."""
    mem = psutil.virtual_memory()
    return {
        "result": {
            "os": platform.system() + " " + platform.release(),
            "cpu_percent": psutil.cpu_percent(interval=0),
            "memory_percent": mem.percent,
            "memory_used_gb": round(mem.used / (1024 ** 3), 2),
            "memory_total_gb": round(mem.total / (1024 ** 3), 2),
            "python_version": platform.python_version(),
            "timestamp": datetime.datetime.now().isoformat(),
        }
    }


@tool("datetime", "Get current date and time information")
def tool_datetime(_=""):
    """Return current date/time."""
    now = datetime.datetime.now()
    return {
        "result": {
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "day": now.strftime("%A"),
            "iso": now.isoformat(),
            "timestamp": int(now.timestamp()),
        }
    }


@tool("unit_convert", "Convert between common units")
def tool_unit_convert(query):
    """Simple unit conversion."""
    q = query.lower().strip()

    conversions = {
        ("km", "miles"): 0.621371,
        ("miles", "km"): 1.60934,
        ("kg", "lbs"): 2.20462,
        ("lbs", "kg"): 0.453592,
        ("celsius", "fahrenheit"): lambda c: c * 9/5 + 32,
        ("fahrenheit", "celsius"): lambda f: (f - 32) * 5/9,
        ("meters", "feet"): 3.28084,
        ("feet", "meters"): 0.3048,
        ("liters", "gallons"): 0.264172,
        ("gallons", "liters"): 3.78541,
    }

    # Try to parse "X unit to unit"
    match = re.match(r"([\d.]+)\s*(\w+)\s+(?:to|in)\s+(\w+)", q)
    if not match:
        return {"error": "Format: '<number> <from_unit> to <to_unit>'"}

    value = float(match.group(1))
    from_unit = match.group(2).lower()
    to_unit = match.group(3).lower()

    key = (from_unit, to_unit)
    if key not in conversions:
        available = ["%s -> %s" % (a, b) for a, b in conversions.keys()]
        return {"error": "Unsupported conversion. Available: " + ", ".join(available)}

    factor = conversions[key]
    if callable(factor):
        result = factor(value)
    else:
        result = value * factor

    return {
        "result": round(result, 4),
        "formatted": "%.4g %s = %.4g %s" % (value, from_unit, result, to_unit),
    }


# --- Tool Executor ---

class ToolExecutor:
    """
    Execute registered tools and plugins by name.
    Phase 6: plugin_manager support.
    Phase 7: unified capability registry (tools + plugins as "capabilities").
    """

    def __init__(self, plugin_manager=None):
        self._plugin_manager = plugin_manager

    def get_tools(self):
        """Return list of available tools (built-in + plugins)."""
        tools = {
            name: info["description"]
            for name, info in _TOOLS.items()
        }
        if self._plugin_manager:
            for p in self._plugin_manager.list_plugins():
                tools[p["name"]] = p["description"]
        return tools

    def execute(self, tool_name, argument=""):
        """
        Execute a tool by name.
        Priority: built-in tools → plugins
        Returns: {"tool": name, "success": bool, "result": ..., "error": ...}
        """
        # 1. Check built-in tools first
        if tool_name in _TOOLS:
            return self._execute_builtin(tool_name, argument)

        # 2. Check plugins
        if self._plugin_manager and self._plugin_manager.has_plugin(tool_name):
            return self._execute_plugin(tool_name, argument)

        return {
            "tool": tool_name,
            "success": False,
            "error": "Unknown tool: %s" % tool_name,
        }

    def _execute_builtin(self, tool_name, argument):
        """Execute a built-in tool."""
        try:
            output = _TOOLS[tool_name]["fn"](argument)
            success = "error" not in output

            log.info("Tool executed: %s (success=%s)", tool_name, success)

            return {
                "tool": tool_name,
                "success": success,
                **output,
            }
        except Exception as e:
            log.error("Tool %s failed: %s", tool_name, e)
            return {
                "tool": tool_name,
                "success": False,
                "error": str(e),
            }

    def _execute_plugin(self, tool_name, argument):
        """Execute a plugin tool with response validation."""
        if self._plugin_manager is None:
            return {"tool": tool_name, "success": False, "result": None,
                    "error": "Plugin manager not available"}
        input_data = {"argument": argument} if isinstance(argument, str) else argument

        response = self._plugin_manager.execute(tool_name, input_data)

        # Validate response shape
        required_keys = {"success", "result", "error"}
        if not isinstance(response, dict) or not required_keys.issubset(response.keys()):
            log.warning("Plugin '%s' returned invalid response: %s", tool_name, response)
            return {
                "tool": tool_name,
                "success": False,
                "result": None,
                "error": "Plugin returned invalid response",
            }

        response["tool"] = tool_name
        log.info("Plugin executed: %s (success=%s)", tool_name, response.get("success"))
        return response

    # ── Phase 7: Unified Capability Registry ──────────────────────────────

    def list_capabilities(self) -> list[dict]:
        """
        Return a unified list of all capabilities (tools + plugins).
        Each entry has: name, type ("tool"|"plugin"), description,
        and optionally category/input_schema/output_schema for plugins.
        """
        caps = []

        # Built-in tools
        for name, info in _TOOLS.items():
            caps.append({
                "name": name,
                "type": "tool",
                "description": info["description"],
            })

        # Plugins
        if self._plugin_manager:
            for p in self._plugin_manager.list_plugins():
                entry = {
                    "name": p["name"],
                    "type": "plugin",
                    "description": p.get("description", ""),
                }
                if p.get("category"):
                    entry["category"] = p["category"]
                if p.get("input_schema"):
                    entry["input_schema"] = p["input_schema"]
                if p.get("output_schema"):
                    entry["output_schema"] = p["output_schema"]
                caps.append(entry)

        return caps

    def has_capability(self, name: str) -> bool:
        """Check if a capability (tool or plugin) exists by name."""
        if name in _TOOLS:
            return True
        if self._plugin_manager and self._plugin_manager.has_plugin(name):
            return True
        return False

    def execute_capability(self, name: str, input_data=None) -> dict:
        """
        Execute any capability (tool or plugin) by name.
        Unified interface over execute().
        """
        argument = input_data if input_data is not None else ""
        return self.execute(name, argument)

    @staticmethod
    def detect_tool(message):
        """
        Detect if a message should trigger a tool call.
        Returns: (tool_name, argument) or (None, None)
        """
        lower = message.lower().strip()

        # Calculator: detect math expressions
        math_patterns = [
            r"(?:calculate|compute|what is|what's|evaluate)\s+(.+)",
            r"(\d+[\s]*[+\-*/^%][\s]*\d+[\d\s+\-*/^%.()]*)",
        ]
        for pat in math_patterns:
            m = re.search(pat, lower)
            if m:
                expr = m.group(1).strip().rstrip("?. ")
                # Verify it looks like a math expression
                if re.search(r'\d+\s*[+\-*/^%]\s*\d+', expr):
                    return "calculator", expr

        # Unit conversion
        if re.search(r'\d+\s*\w+\s+(?:to|in)\s+\w+', lower):
            for unit_pair in [("km", "miles"), ("kg", "lbs"), ("celsius", "fahrenheit"),
                              ("meters", "feet"), ("liters", "gallons"),
                              ("miles", "km"), ("lbs", "kg"), ("fahrenheit", "celsius"),
                              ("feet", "meters"), ("gallons", "liters")]:
                if unit_pair[0] in lower and unit_pair[1] in lower:
                    return "unit_convert", lower

        # System info
        if any(kw in lower for kw in ["system info", "system status", "cpu usage",
                                       "memory usage", "ram usage"]):
            return "system_info", ""

        # Date/time
        if any(kw in lower for kw in ["what time", "current time", "what date",
                                       "today's date", "what day"]):
            return "datetime", ""

        return None, None
