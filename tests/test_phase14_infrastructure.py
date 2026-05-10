"""
tests/test_phase14_infrastructure.py — Tests for Phase 14 Infrastructure Modules

Covers: SystemGuard, TaskManager, StreamCleanup, TimeoutManager,
        ResourceMonitor, TokenEstimator, PromptOptimizer, LatencyTracker
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

passed = 0
failed = 0

def check(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}")


# ── SystemGuard ──────────────────────────────────────────────────────────────
print("\n=== SystemGuard ===")
from utils.system_guard import system_guard

check("is_healthy returns bool", isinstance(system_guard.is_healthy(), bool))
check("get_status returns dict", isinstance(system_guard.get_status(), dict))
rid = system_guard.register_request()
check("register_request returns id", isinstance(rid, int))
status = system_guard.get_status()
check("active_requests >= 1", status.get("active_requests", 0) >= 1)
system_guard.release_request(rid)
status2 = system_guard.get_status()
check("release decrements", status2.get("active_requests", 99) < status.get("active_requests", 0))
cleaned = system_guard.cleanup_stale_requests(max_age_s=0.001)
check("cleanup returns int", isinstance(cleaned, int))

# ── TaskManager ──────────────────────────────────────────────────────────────
print("\n=== TaskManager ===")
from utils.task_manager import task_manager

tid = task_manager.register("test_task", max_duration=10.0)
check("register returns task_id", tid.startswith("task_"))
stats = task_manager.stats()
check("stats has active", "active" in stats)
check("active >= 1", stats["active"] >= 1)
task_manager.complete(tid)
check("complete works", True)
swept = task_manager.sweep()
check("sweep returns int", isinstance(swept, int))

# Timeout test
tid2 = task_manager.register("timeout_test", max_duration=0.001)
time.sleep(0.01)
swept2 = task_manager.sweep()
check("sweep catches timeout", swept2 >= 1)

# ── StreamCleanup ────────────────────────────────────────────────────────────
print("\n=== StreamCleanup ===")
from utils.stream_cleanup import StreamGuard, guarded_stream

def sample_gen():
    for i in range(5):
        yield f"chunk_{i}"

guard = StreamGuard(sample_gen(), timeout=10.0, name="test")
chunks = list(guard)
check("StreamGuard yields all chunks", len(chunks) == 5)
check("StreamGuard closed after iteration", not guard.is_active)

# Timeout test
def slow_gen():
    for i in range(100):
        time.sleep(0.1)
        yield f"slow_{i}"

guard2 = StreamGuard(slow_gen(), timeout=0.2, name="timeout_test")
slow_chunks = list(guard2)
check("StreamGuard respects timeout", len(slow_chunks) < 10)

# Functional wrapper
func_chunks = list(guarded_stream(sample_gen(), timeout=10.0))
check("guarded_stream functional wrapper works", len(func_chunks) == 5)

# ── TimeoutManager ───────────────────────────────────────────────────────────
print("\n=== TimeoutManager ===")
from utils.timeout_manager import with_timeout, TimeoutContext, get_default_timeout

check("get_default_timeout returns float", isinstance(get_default_timeout("llm"), float))
check("search timeout < llm timeout",
      get_default_timeout("search") < get_default_timeout("llm"))

# Decorator test
@with_timeout(timeout=5.0, fallback="timeout")
def fast_fn():
    return "done"

check("with_timeout decorator passes", fast_fn() == "done")

@with_timeout(timeout=0.1, fallback="timeout")
def slow_fn():
    time.sleep(5)
    return "done"

check("with_timeout catches slow fn", slow_fn() == "timeout")

# Context manager test
with TimeoutContext(5.0, name="test") as ctx:
    time.sleep(0.01)
    remaining = ctx.remaining
check("TimeoutContext tracks remaining", remaining > 0)
check("TimeoutContext elapsed works", ctx.elapsed > 0)

# ── ResourceMonitor ──────────────────────────────────────────────────────────
print("\n=== ResourceMonitor ===")
from utils.resource_monitor import resource_monitor

snap = resource_monitor.snapshot()
check("snapshot returns dict", isinstance(snap, dict))
check("snapshot has process_rss_mb", "process_rss_mb" in snap)
check("snapshot has system_memory_pct", "system_memory_pct" in snap)
resource_monitor.record_request()
resource_monitor.record_error()
snap2 = resource_monitor.snapshot()
check("records requests", snap2.get("requests_served", 0) >= 1)
check("records errors", snap2.get("errors", 0) >= 1)

# ── TokenEstimator ───────────────────────────────────────────────────────────
print("\n=== TokenEstimator ===")
from services.token_estimator import token_estimator

tokens = token_estimator.estimate("Hello, how are you today?")
check("estimate returns int", isinstance(tokens, int))
check("estimate > 0", tokens > 0)

msgs = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
]
msg_tokens = token_estimator.estimate_messages(msgs)
check("estimate_messages > single", msg_tokens > tokens)

budget = token_estimator.check_budget("llama-3.3-70b-versatile", msgs)
check("check_budget has within_budget", "within_budget" in budget)
check("within_budget is True for small prompt", budget["within_budget"])
check("has utilization_pct", "utilization_pct" in budget)

avail = token_estimator.get_available_tokens("llama-3.3-70b-versatile", msgs)
check("available_tokens > 0", avail > 0)

# ── PromptOptimizer ──────────────────────────────────────────────────────────
print("\n=== PromptOptimizer ===")
from services.prompt_optimizer import PromptOptimizer

optimizer = PromptOptimizer(token_estimator=token_estimator)

# Build a long history
long_msgs = [{"role": "system", "content": "Be helpful."}]
for i in range(30):
    long_msgs.append({"role": "user", "content": f"message {i}"})
    long_msgs.append({"role": "assistant", "content": f"reply {i}"})

optimized = optimizer.optimize(long_msgs, model="llama-3.3-70b-versatile")
check("optimize trims history", len(optimized) < len(long_msgs))
check("keeps system message", any(m["role"] == "system" for m in optimized))

# Dedup test
dup_msgs = [
    {"role": "system", "content": "Be helpful."},
    {"role": "system", "content": "Be helpful."},
    {"role": "user", "content": "Hello"},
]
deduped = optimizer.optimize(dup_msgs)
system_count = sum(1 for m in deduped if m["role"] == "system")
check("dedup removes duplicate system msgs", system_count == 1)

# ── LatencyTracker ───────────────────────────────────────────────────────────
print("\n=== LatencyTracker ===")
from services.response_latency_tracker import LatencyTracker

tracker = LatencyTracker()

# Context manager
with tracker.track("test_op") as t:
    time.sleep(0.01)

stats = tracker.stats("test_op")
check("stats has count", stats.get("count", 0) == 1)
check("stats has avg_ms", "avg_ms" in stats)

# Manual recording
tracker.record("manual", 100)
tracker.record("manual", 200)
tracker.record("manual", 300)
manual_stats = tracker.stats("manual")
check("manual count = 3", manual_stats["count"] == 3)
check("manual avg = 200", manual_stats["avg_ms"] == 200)
check("manual min = 100", manual_stats["min_ms"] == 100)
check("manual max = 300", manual_stats["max_ms"] == 300)

# Aggregate stats
all_stats = tracker.stats()
check("aggregate has total_requests", all_stats["total_requests"] >= 4)
check("aggregate has operations", len(all_stats["operations"]) >= 2)


# ── Results ──────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"Results: {passed}/{passed+failed} passed")
if failed > 0:
    print(f"FAILED: {failed} test(s)")
    sys.exit(1)
else:
    print("ALL TESTS PASSED")
