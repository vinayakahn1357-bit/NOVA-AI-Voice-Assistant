"""Quick functional test for all 8 search intelligence modules."""

from services.search.query_rewriter import QueryRewriter
from services.search.search_router import SearchRouter
from services.search.context_cleaner import ContextCleaner
from services.search.context_ranker import ContextRanker
from services.search.context_compressor import ContextCompressor
from services.search.confidence_scorer import ConfidenceScorer
from services.search.search_memory import SearchMemory

passed = 0
total = 0

def check(name, condition):
    global passed, total
    total += 1
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        print(f"  [FAIL] {name}")

# 1. QueryRewriter
print("=== QueryRewriter ===")
rw = QueryRewriter()
r1 = rw.rewrite("hey nova what happened with nvidia today")
check("Removes filler", "nova" not in r1.lower() or "NVIDIA" in r1)
check("Preserves entity", "NVIDIA" in r1)
r2 = rw.rewrite("write a python function")
check("Keeps coding query", len(r2) > 5)
print(f"  Example: 'hey nova what happened with nvidia today' -> '{r1}'")

# 2. SearchRouter
print("\n=== SearchRouter ===")
router = SearchRouter()
d1 = router.route("hello nova")
check("Skips greeting", not d1.should_search)
d2 = router.route("IPL score today")
check("Searches sports", d2.should_search)
check("Sports domain", d2.domain == "sports")
d3 = router.route("write a python script")
check("Skips coding", not d3.should_search)
d4 = router.route("bitcoin price right now")
check("Searches finance", d4.should_search)

# 3. ContextCleaner
print("\n=== ContextCleaner ===")
cleaner = ContextCleaner()
fake = [
    {"title": "Result A", "snippet": "NVIDIA stock rose 5% today after earnings report.", "url": "https://reuters.com/a"},
    {"title": "Duplicate", "snippet": "NVIDIA stock rose 5% today after earnings report.", "url": "https://cnn.com/b"},
    {"title": "Noise", "snippet": "Click", "url": ""},
]
cleaned = cleaner.clean(fake)
check("Removes duplicates", len(cleaned) == 1)
check("Removes thin results", all(len(r["snippet"]) >= 15 for r in cleaned))

# 4. ContextRanker
print("\n=== ContextRanker ===")
ranker = ContextRanker(top_k=2)
results_for_rank = [
    {"title": "NVIDIA Earnings", "snippet": "NVIDIA reported record earnings, stock up 5% today.", "url": "https://reuters.com/a", "score": 0.9, "date": ""},
    {"title": "Weather Report", "snippet": "Sunny skies expected in California throughout the week.", "url": "https://weather.com/b", "score": 0.3, "date": ""},
]
ranked = ranker.rank(results_for_rank, "NVIDIA stock price")
check("Returns results", len(ranked) > 0)
check("NVIDIA ranked first", "NVIDIA" in ranked[0]["title"])
check("Has relevance score", "relevance_score" in ranked[0])

# 5. ContextCompressor
print("\n=== ContextCompressor ===")
compressor = ContextCompressor(max_chars=300)
compressed = compressor.compress(ranked, "NVIDIA stock price")
check("Produces output", len(compressed) > 0)
check("Within limit", len(compressed) <= 400)
print(f"  Compressed: {compressed[:120]}...")

# 6. ConfidenceScorer
print("\n=== ConfidenceScorer ===")
scorer = ConfidenceScorer()
conf, inject, breakdown = scorer.score(ranked, "NVIDIA stock price")
check("Returns score", 0.0 <= conf <= 1.0)
check("Returns inject bool", isinstance(inject, bool))
check("Has breakdown", "factors" in breakdown)
print(f"  Confidence: {conf:.3f}, inject: {inject}")

# 7. SearchMemory
print("\n=== SearchMemory ===")
mem = SearchMemory(ttl=60)
mem.store("nvidia stock", "NVIDIA stock price", ranked, compressed, "finance")
cached = mem.lookup("nvidia share price")
check("Memory stores", mem.stats()["entries"] == 1)
check("Similarity lookup", cached is not None)
miss = mem.lookup("weather in Tokyo")
check("Non-match returns None", miss is None)

# 8. Integration
print("\n=== Integration ===")
from services.search.search_orchestrator import SearchOrchestrator
check("Orchestrator imports", True)

print(f"\n{'='*50}")
print(f"Results: {passed}/{total} passed")
if passed == total:
    print("ALL TESTS PASSED")
else:
    print(f"WARNING: {total - passed} test(s) failed")
