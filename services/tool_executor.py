"""
services/tool_executor.py - Built-in Tool System for NOVA Agent
Provides lightweight tools the agent can call without external APIs.
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
    Evaluate a math expression safely.
    Supports: +, -, *, /, **, sqrt, sin, cos, tan, log, pi, e, abs, round
    """
    try:
        # Clean the expression
        expr = expression.strip()
        # Remove any non-math characters for safety
        allowed = set("0123456789+-*/().,%^ sincotaqrlgbdephu")
        if not all(c in allowed or c.isspace() for c in expr.lower()):
            return {"error": "Invalid characters in expression"}

        # Safe math namespace
        safe_ns = {
            "abs": abs, "round": round, "min": min, "max": max,
            "sqrt": math.sqrt, "pow": pow,
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "log": math.log, "log10": math.log10, "log2": math.log2,
            "pi": math.pi, "e": math.e,
            "ceil": math.ceil, "floor": math.floor,
            "factorial": math.factorial,
        }

        # Replace common notations
        expr = expr.replace("^", "**").replace("%", "/100")

        result = eval(expr, {"__builtins__": {}}, safe_ns)
        return {
            "result": result,
            "expression": expression.strip(),
            "formatted": str(result),
        }
    except ZeroDivisionError:
        return {"error": "Division by zero"}
    except Exception as e:
        return {"error": "Cannot evaluate: %s" % str(e)}


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
    """Execute registered tools by name."""

    @staticmethod
    def get_tools():
        """Return list of available tools."""
        return {
            name: info["description"]
            for name, info in _TOOLS.items()
        }

    @staticmethod
    def execute(tool_name, argument=""):
        """
        Execute a tool by name.
        Returns: {"tool": name, "success": bool, "result": ..., "error": ...}
        """
        if tool_name not in _TOOLS:
            return {
                "tool": tool_name,
                "success": False,
                "error": "Unknown tool: %s" % tool_name,
            }

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
