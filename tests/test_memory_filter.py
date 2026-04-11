"""
tests/test_memory_filter.py — Tests for the memory extraction skip-guard.
Verifies trivial messages are skipped and real messages pass through.
"""

import pytest
from services.memory_service import MemoryService


@pytest.fixture
def filter_instance():
    """Create a bare MemoryService to test the filter method without DB."""
    ms = MemoryService.__new__(MemoryService)
    return ms


class TestTrivialMessagesSkipped:
    """Verify trivial messages are correctly identified and skipped."""

    @pytest.mark.parametrize("msg", [
        "hi", "hey", "hello", "yo", "sup",
        "ok", "okay", "sure", "yes", "no", "yep", "nope",
        "thanks", "thx", "ty",
        "bye", "goodbye",
        "hmm", "wow", "lol", "haha", "nice", "cool",
        "got it", "alright", "fine",
    ])
    def test_skip_words(self, filter_instance, msg):
        assert filter_instance._should_skip_extraction(msg) is True, \
            f"'{msg}' should be skipped but wasn't"

    @pytest.mark.parametrize("msg", [
        "what?", "why?", "how?", "yes!", "ok.",
    ])
    def test_skip_words_with_punctuation(self, filter_instance, msg):
        assert filter_instance._should_skip_extraction(msg) is True, \
            f"'{msg}' should be skipped but wasn't"

    @pytest.mark.parametrize("msg", [
        "a", "no", "go", "x", "test",
    ])
    def test_single_word_short(self, filter_instance, msg):
        assert filter_instance._should_skip_extraction(msg) is True, \
            f"Single word '{msg}' should be skipped"

    def test_short_message_under_15_chars(self, filter_instance):
        assert filter_instance._should_skip_extraction("hi there") is True

    def test_empty_string(self, filter_instance):
        assert filter_instance._should_skip_extraction("") is True

    def test_whitespace_only(self, filter_instance):
        assert filter_instance._should_skip_extraction("   ") is True


class TestRealMessagesPassThrough:
    """Verify real, fact-containing messages are NOT skipped."""

    @pytest.mark.parametrize("msg", [
        "My name is John and I work at Google",
        "I prefer Python over JavaScript for backend development",
        "I am studying machine learning at MIT",
        "Can you help me understand quantum computing?",
        "I live in Bangalore and I'm interested in AI",
        "My favorite programming language is Rust",
        "I'm working on a project about climate change",
    ])
    def test_real_messages_not_skipped(self, filter_instance, msg):
        assert filter_instance._should_skip_extraction(msg) is False, \
            f"Real message '{msg[:40]}...' was wrongly skipped"
