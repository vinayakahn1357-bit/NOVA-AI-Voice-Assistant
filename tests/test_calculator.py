"""
tests/test_calculator.py — Security tests for the AST-based calculator tool.
Validates that the eval()-replacement blocks all injection vectors.
"""

import math
import pytest
from services.tool_executor import tool_calculator


class TestCalculatorBasicMath:
    """Verify basic arithmetic works correctly."""

    def test_addition(self):
        assert tool_calculator("2 + 3")["result"] == 5

    def test_subtraction(self):
        assert tool_calculator("10 - 4")["result"] == 6

    def test_multiplication(self):
        assert tool_calculator("10 * 5")["result"] == 50

    def test_division(self):
        assert tool_calculator("20 / 4")["result"] == 5.0

    def test_power_caret(self):
        assert tool_calculator("2 ^ 10")["result"] == 1024

    def test_power_double_star(self):
        assert tool_calculator("2 ** 3")["result"] == 8

    def test_modulo(self):
        r = tool_calculator("17 % 5")
        # Modulo may or may not be supported by the AST parser
        if "result" in r:
            assert r["result"] == 2
        else:
            assert "error" in r  # acceptable: not all ops are whitelisted

    def test_negative_number(self):
        assert tool_calculator("-5 + 3")["result"] == -2

    def test_nested_parentheses(self):
        assert tool_calculator("(2 + 3) * (4 - 1)")["result"] == 15


class TestCalculatorFunctions:
    """Verify whitelisted math functions work."""

    def test_sqrt(self):
        assert tool_calculator("sqrt(16)")["result"] == 4.0

    def test_sin_zero(self):
        assert tool_calculator("sin(0)")["result"] == 0.0

    def test_cos_zero(self):
        assert tool_calculator("cos(0)")["result"] == 1.0

    def test_abs_negative(self):
        assert tool_calculator("abs(-42)")["result"] == 42

    def test_log(self):
        r = tool_calculator("log(100)")
        assert abs(r["result"] - math.log(100)) < 0.001


class TestCalculatorConstants:
    """Verify math constants are accessible."""

    def test_pi(self):
        r = tool_calculator("pi")
        assert abs(r["result"] - math.pi) < 0.001

    def test_e(self):
        r = tool_calculator("e")
        assert abs(r["result"] - math.e) < 0.001


class TestCalculatorErrors:
    """Verify proper error handling."""

    def test_division_by_zero(self):
        r = tool_calculator("1 / 0")
        assert "error" in r
        assert "Division by zero" in r["error"]

    def test_empty_expression(self):
        r = tool_calculator("")
        assert "error" in r

    def test_whitespace_only(self):
        r = tool_calculator("   ")
        assert "error" in r


class TestCalculatorSecurity:
    """CRITICAL: Verify all injection vectors are blocked."""

    def test_import_blocked(self):
        r = tool_calculator('__import__("os").system("echo pwned")')
        assert "error" in r, "SECURITY BREACH: __import__ was not blocked"

    def test_eval_blocked(self):
        r = tool_calculator('eval("1+1")')
        assert "error" in r, "SECURITY BREACH: eval() was not blocked"

    def test_exec_blocked(self):
        r = tool_calculator('exec("print(1)")')
        assert "error" in r, "SECURITY BREACH: exec() was not blocked"

    def test_attribute_access_blocked(self):
        r = tool_calculator('"hello".__class__')
        assert "error" in r, "SECURITY BREACH: attribute access was not blocked"

    def test_exponent_bomb_blocked(self):
        r = tool_calculator("2 ** 9999")
        assert "error" in r, "SECURITY BREACH: exponent bomb was not blocked"

    def test_large_caret_exponent_blocked(self):
        r = tool_calculator("10 ^ 5000")
        assert "error" in r, "SECURITY BREACH: caret exponent bomb was not blocked"

    def test_lambda_blocked(self):
        r = tool_calculator("(lambda: 1)()")
        assert "error" in r, "SECURITY BREACH: lambda was not blocked"

    def test_list_comprehension_blocked(self):
        r = tool_calculator("[x for x in range(10)]")
        assert "error" in r, "SECURITY BREACH: list comprehension was not blocked"

    def test_global_access_blocked(self):
        r = tool_calculator("globals()")
        assert "error" in r, "SECURITY BREACH: globals() was not blocked"
