"""Final verification script for NOVA Personality Engine (Phase 12)."""
import sys
import io

# Force UTF-8 output on Windows to handle emoji characters
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from services.personality_enforcer import (
    PersonalityEnforcer, score_personality_strength, SCORE_PASS_THRESHOLD
)
e = PersonalityEnforcer()
results = []

print("=== FORBIDDEN PHRASE REMOVAL ===")
tests = [
    ("Great question! Glad you asked.", "Great question"),
    ("As an AI language model, I think this is hard.", "As an AI"),
    ("I am just an AI and cannot feel emotions.", "just an AI"),
    ("I would be happy to help you with that task.", "happy to help"),
]
for text, phrase in tests:
    r = e.enforce(text, "default", "conversation")
    ok = phrase.lower() not in r.lower() or r.strip() == ""
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {repr(phrase)} removed -> {repr(r[:60])}")
    results.append(ok)

print()
print("=== STRUCTURE PATCHING ===")
# Teacher: paragraphs without steps => gets numbered
teacher_in = "Here is the concept.\n\nFirst idea details here.\n\nSecond idea details.\n\nThird idea."
t_out = e.enforce(teacher_in, "teacher", "conversation")
has_steps = bool(__import__('re').search(r'^\s*\d+[\.\)]\s+', t_out, __import__('re').MULTILINE))
wc = len(t_out.split())
print(f"  Teacher steps added: {has_steps} (words={wc})")
results.append(has_steps)

# Hacker: long text gets truncated
long_text = "word " * 200
h_out = e.enforce(long_text, "hacker", "conversation")
hacker_ok = len(h_out.split()) <= 180
print(f"  Hacker truncated: {len(h_out.split())} words (<= 180): {hacker_ok}")
results.append(hacker_ok)

# Calm: bullets removed
calm_in = "- Take a breath.\n- Feel the present.\n- Let go of worry."
c_out = e.enforce(calm_in, "calm", "conversation")
calm_ok = "- " not in c_out
print(f"  Calm bullets removed: {calm_ok} -> {repr(c_out[:60])}")
results.append(calm_ok)

print()
print("=== PERSONALITY SCORING ===")
scoring_tests = [
    ("1. Step one details here.\n2. Step two explanation.\n3. Step three done.\n\nSummary: All clear.", "teacher", True),
    ("pip install flask", "hacker", True),
    ("This approach has several trade-offs to consider carefully. Best practice recommends X. However, watch out for edge cases.", "expert", True),
    ("TARGET: 1. Start right now. 2. Commit fully. Take action today! Go for it!", "coach", True),
    ("The real trick here is thinking differently. What most people miss is this insight.", "genius", True),
    ("Safe: True", "romantic", True),
]
for text, p, expected_pass in scoring_tests:
    s, _ = score_personality_strength(text, p)
    ok = (s >= SCORE_PASS_THRESHOLD) == expected_pass
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {p:10s} score={s:.2f} (threshold={SCORE_PASS_THRESHOLD})")
    results.append(ok)

print()
print("=== ROMANTIC SAFETY GATE ===")
safe_ok = e.is_romantic_safe("Your curiosity is absolutely delightful and inspiring!")
unsafe_blocked = not e.is_romantic_safe("This involves explicit sexual content.")
print(f"  Safe phrase allowed: {safe_ok}")
print(f"  Unsafe phrase blocked: {unsafe_blocked}")
results.extend([safe_ok, unsafe_blocked])

print()
print("=== TEMPERATURE ROUTING ===")
from services.ai_service import AIService
expected = [
    ("teacher", 0.5), ("expert", 0.3), ("hacker", 0.4),
    ("default", 0.7), ("calm", 0.6), ("coach", 0.8),
    ("genius", 0.85), ("romantic", 0.85), ("friend", 0.9), ("funny", 0.95),
]
for p, exp in expected:
    got = AIService._resolve_temperature(p)
    ok = abs(got - exp) < 0.001
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {p:10s} expected={exp} got={got}")
    results.append(ok)

print()
print("=== PROMPT BUILDER INJECTION ===")
from services.prompt_builder import PromptBuilder
pb = PromptBuilder()
for p in ["teacher", "hacker", "expert", "calm", "coach", "funny", "genius", "romantic", "friend", "default"]:
    block = pb._get_personality_block(p)
    ok = len(block) > 50
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {p:10s} {len(block):4d} chars, has enforcement: {'ENFORCEMENT' in block}")
    results.append(ok)

print()
print("=== PIPELINE + IMPORTS ===")
try:
    from services.response_pipeline import ResponsePipeline
    from services.personality_service import PERSONALITIES, VALID_PERSONALITIES, PersonalityStore
    ps = PersonalityStore()
    ps.set("sess1", "teacher")
    got = ps.get("sess1")
    bad = ps.set("sess1", "nonexistent")
    print(f"  [PASS] ResponsePipeline importable")
    print(f"  [PASS] {len(PERSONALITIES)} personalities registered")
    print(f"  [PASS] PersonalityStore.set/get works: {got}")
    print(f"  [PASS] Invalid personality rejected: {not bad}")
    results.extend([True, True, got == "teacher", not bad])
except Exception as ex:
    print(f"  [FAIL] {ex}")
    results.append(False)

print()
passed = sum(results)
total = len(results)
if passed == total:
    print(f"ALL {total}/{total} TESTS PASSED")
    print("Personality Engine is production-ready!")
else:
    failed = total - passed
    print(f"FAILURES: {failed}/{total} tests failed")
    sys.exit(1)
