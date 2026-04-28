"""Phase 13 Integration Test Suite — validates all premium upgrade modules."""
import sys
sys.path.insert(0, '.')

from services.response_quality import ResponseQualityEnforcer
from services.realtime_service import RealtimeSearchService
from services.query_analyzer import QueryAnalyzer
from services.agent_engine import AgentEngine
from utils.response_formatter import ResponseFormatter
from services.personality_service import get_voice_hints, VOICE_HINTS
from services.tts_service import _apply_voice_hints

passed = 0
failed = 0

def check(name, condition):
    global passed, failed
    if condition:
        print(f"  [PASS] {name}")
        passed += 1
    else:
        print(f"  [FAIL] {name}")
        failed += 1

print("=== TEST 1: Quality Enforcer ===")
qe = ResponseQualityEnforcer()

t1, s1, i1 = qe.enforce(
    "Certainly! I would be happy to help you. Python is great. Hope this helps!",
    "explanation", 5
)
check("Strips generic opener", "stripped_generic_opener" in i1)
check("Strips generic closer", "stripped_generic_closer" in i1)
check("Cleaned text removes 'Certainly'", "Certainly" not in t1)

t2, s2, i2 = qe.enforce(
    "Python uses duck typing, which means the type of an object is determined by its behavior "
    "rather than its class hierarchy. This enables polymorphism without inheritance, making code "
    "more flexible and reducing coupling between components. A practical example: any object with "
    "a __len__ method works with len(), regardless of its class.",
    "explanation", 5
)
check("Expert response scores high", s2 > 0.6)
check("No issues on expert response", len(i2) == 0)

t3, s3, i3 = qe.enforce(
    "As an AI language model, I don't have personal opinions. But I think Python is nice.",
    "conversation", 3
)
check("Strips AI self-reference", "removed" in str(i3))

t4, s4, i4 = qe.enforce("", "conversation", 3)
check("Empty response returns 0.0", s4 == 0.0)
print()

print("=== TEST 2: Realtime Intent Detection ===")
rs = RealtimeSearchService()
check("Service is available", rs.is_available)
check("Detects 'latest news about AI'", rs.detect_realtime_intent("latest news about AI"))
check("Detects 'Bitcoin price today'", rs.detect_realtime_intent("Bitcoin price today"))
check("Detects 'current weather in London'", rs.detect_realtime_intent("current weather in London"))
check("Rejects 'what is recursion'", not rs.detect_realtime_intent("what is recursion"))
check("Rejects 'history of Rome'", not rs.detect_realtime_intent("history of Rome"))
check("Rejects 'explain the theory of relativity'", not rs.detect_realtime_intent("explain the theory of relativity"))
check("Context builder works", len(RealtimeSearchService.build_search_context(
    [{"title": "Test", "snippet": "Result", "url": "https://example.com", "date": "", "score": 0.9}],
    "test query"
)) > 50)
print()

print("=== TEST 3: Query Classification ===")
qa = QueryAnalyzer()
r = qa.analyze("hello!")
check("Greeting detected", r["query_type"] == "greeting")
r = qa.analyze("latest stock market news today")
check("Realtime detected", r["query_type"] == "realtime")
r = qa.analyze("Write a comprehensive detailed report analyzing AI")
check("Report detected", r["query_type"] == "report")
r = qa.analyze("def fibonacci(n): implement this function")
check("Coding detected", r["query_type"] == "coding")
r = qa.analyze("solve the integral of x^2 dx using calculus formula")
check("Math detected", r["query_type"] == "math")
check("Token budget 2048 for coding", QueryAnalyzer().analyze("implement a REST API server")["optimal_tokens"] == 2048)
print()

print("=== TEST 4: Agent Reasoning Mode ===")
ae = AgentEngine()
r1 = ae.process("Why does Python use indentation instead of braces?")
check("Why-does triggers reasoning", r1["agent_mode"] == "reasoning")
r2 = ae.process("What would happen if we removed garbage collection?")
check("What-would-happen triggers reasoning", r2["agent_mode"] == "reasoning")
r3 = ae.process("Hello!")
check("Greeting stays normal", r3["agent_mode"] == "normal")
r4 = ae.process("Should I use React or Vue for my project?")
check("Comparison triggers decision", r4["agent_mode"] == "decision")
r5 = ae.process("Build me a REST API from scratch")
check("Build triggers planning", r5["agent_mode"] == "planning")
check("Reasoning has prompt augment", "STRUCTURED REASONING" in r1.get("prompt_augment", ""))
print()

print("=== TEST 5: Voice Hints ===")
check("10 personalities have voice hints", len(VOICE_HINTS) == 10)
hints = get_voice_hints("coach")
check("Coach is fast pace", hints["pace"] == "fast")
check("Coach is warm", hints["warmth"] == "warm")
hints = get_voice_hints("calm")
check("Calm is slow pace", hints["pace"] == "slow")
rate, pitch = _apply_voice_hints({"pace": "slow", "warmth": "warm", "emphasis": "moderate"})
check("Slow pace maps to -15%", rate == "-15%")
check("Warm+moderate maps to +1Hz", pitch == "+1Hz")
check("Unknown personality falls back to default", get_voice_hints("nonexistent") == VOICE_HINTS["default"])
print()

print("=== TEST 6: Response Formatter ===")
rf = ResponseFormatter()
# Test _format_coding directly (format() preserves code blocks before formatting)
bt = chr(96) * 3
py_code = f"{bt}\ndef hello():\n    print('hi')\n{bt}"
result = rf._format_coding(py_code)
check("Auto-detects Python in code blocks", "python" in result)

js_code = f"{bt}\nconst x = () => {{\n  return 42;\n}}\n{bt}"
js_result = rf._format_coding(js_code)
check("Auto-detects JavaScript in code blocks", "javascript" in js_result)

# Test new format types exist
check("PDF formatter exists", hasattr(rf, '_format_pdf_analysis'))
check("Realtime formatter exists", hasattr(rf, '_format_realtime'))
check("Report formatter exists", hasattr(rf, '_format_report'))
print()

print("=== TEST 7: Model Router ===")
from services.model_router import ModelRouter
mr = ModelRouter()
check("Realtime routes to Groq", mr.select_provider(
    "What are the latest trending news about artificial intelligence today?",
    {"query_type": "realtime", "complexity": 3}) == "groq")
check("Greeting routes to Groq", mr.select_provider(
    "Good morning! How are you doing today my friend?",
    {"query_type": "greeting", "complexity": 1}) == "groq")
check("PDF analysis routes to NVIDIA", mr.select_provider(
    "Please analyze this document content deeply and provide a comprehensive summary of the findings",
    {"query_type": "pdf_analysis", "complexity": 7}) == "nvidia")
check("Report routes to NVIDIA", mr.select_provider(
    "Write a comprehensive detailed analysis report covering all the major trends in technology",
    {"query_type": "report", "complexity": 8}) == "nvidia")
print()

print("=== TEST 8: Premium Config ===")
from config import (QUALITY_SCORE_THRESHOLD, QUALITY_REGEN_MAX,
                     REALTIME_SEARCH_API_KEY, ENABLE_REALTIME_SEARCH,
                     NOVA_SETTINGS, DEFAULT_SYSTEM_PROMPT)
check("Quality threshold is 0.4", QUALITY_SCORE_THRESHOLD == 0.4)
check("Regen max is 1", QUALITY_REGEN_MAX == 1)
check("Tavily key is set", len(REALTIME_SEARCH_API_KEY) > 10)
check("Realtime search enabled", ENABLE_REALTIME_SEARCH)
check("Token budget is 2048", NOVA_SETTINGS["num_predict"] == 2048)
check("System prompt has forbidden patterns", "Forbidden Patterns" in DEFAULT_SYSTEM_PROMPT)
check("System prompt has depth calibration", "Depth calibration" in DEFAULT_SYSTEM_PROMPT)
check("System prompt has document mode", "Document-Aware" in DEFAULT_SYSTEM_PROMPT)
print()

print("=" * 50)
print(f"RESULTS: {passed} passed, {failed} failed, {passed + failed} total")
if failed == 0:
    print("ALL TESTS PASSED")
else:
    print(f"WARNING: {failed} test(s) failed")
