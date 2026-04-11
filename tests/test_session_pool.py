"""
tests/test_session_pool.py — Tests for HTTP session pooling in AIService.
Verifies that AIService uses a shared requests.Session with connection pooling.
"""

import pytest
import requests
from requests.adapters import HTTPAdapter


class TestHTTPSessionPool:
    """Verify AIService creates and uses a shared HTTP session."""

    def test_ai_service_has_http_session(self):
        """AIService should have a _http attribute that is a requests.Session."""
        from services.prompt_builder import PromptBuilder
        from services.memory_service import MemoryService
        from services.ai_service import AIService

        # Minimal construction — only prompt_builder is required
        # Create a mock-like MemoryService to satisfy PromptBuilder
        ms = MemoryService.__new__(MemoryService)
        ms._memory = type("M", (), {"get_memory_context": lambda self: ""})()
        ms._use_db = False
        ms._db_memory = None

        pb = PromptBuilder.__new__(PromptBuilder)
        pb._memory = ms

        ai = AIService(pb)
        assert hasattr(ai, "_http"), "AIService should have _http attribute"
        assert isinstance(ai._http, requests.Session), \
            "_http should be a requests.Session"

    def test_session_has_adapters(self):
        """Session should have HTTPAdapter mounted for both http and https."""
        from services.ai_service import AIService
        from services.prompt_builder import PromptBuilder

        pb = PromptBuilder.__new__(PromptBuilder)
        pb._memory = type("M", (), {"get_memory_context": lambda self, **kw: ""})()

        ai = AIService(pb)

        # Check that adapters are mounted
        https_adapter = ai._http.get_adapter("https://example.com")
        http_adapter = ai._http.get_adapter("http://example.com")

        assert isinstance(https_adapter, HTTPAdapter)
        assert isinstance(http_adapter, HTTPAdapter)

    def test_provider_methods_are_not_static(self):
        """_call_groq and _call_ollama_local should be instance methods (not static)."""
        from services.ai_service import AIService

        # These should NOT have __func__ that's a staticmethod
        assert not isinstance(
            AIService.__dict__.get("_call_groq"), staticmethod
        ), "_call_groq should be an instance method, not static"

        assert not isinstance(
            AIService.__dict__.get("_call_ollama_local"), staticmethod
        ), "_call_ollama_local should be an instance method, not static"
