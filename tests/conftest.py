"""
tests/conftest.py — Shared pytest fixtures for NOVA test suite.
"""

import sys
import os
import pytest

# Ensure the project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
