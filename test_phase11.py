"""Phase 11 integration test script."""
import sys
sys.path.insert(0, ".")

from services.pdf_service import PDFService
from services.document_retriever import create_retriever, TFIDFRetriever
from services.smart_responder import SmartResponder
from services.document_context import DocumentContextStore

# 1. Test retriever
print("1. Testing DocumentRetriever...")
retriever = create_retriever()
assert isinstance(retriever, TFIDFRetriever)
chunks = [
    {"text": "Machine learning is a subset of artificial intelligence", "page": 1, "chunk_index": 0, "start_page": 1, "end_page": 1},
    {"text": "Deep learning uses neural networks with multiple layers", "page": 2, "chunk_index": 1, "start_page": 2, "end_page": 2},
    {"text": "Python is widely used for data science and ML", "page": 3, "chunk_index": 2, "start_page": 3, "end_page": 3},
    {"text": "Natural language processing handles text and speech data", "page": 4, "chunk_index": 3, "start_page": 4, "end_page": 4},
]
retriever.index_chunks("test123", chunks)
assert retriever.is_indexed("test123")
results = retriever.retrieve("test123", "what is machine learning?", top_k=2)
assert len(results) > 0
assert results[0]["score"] > 0
print(f"   OK: {len(results)} chunks retrieved, top score={results[0]['score']:.3f}")
stats = retriever.stats()
print(f"   Stats: {stats}")

# 2. Test document store (multi-doc)
print("\n2. Testing DocumentContextStore (multi-doc)...")
store = DocumentContextStore()
doc_id = store.add_document("sess1", "test.pdf", "Summary text", chunks, doc_hash="hash1")
assert store.has_document("sess1")
assert store.get_active_document_id("sess1") == doc_id
print(f"   Added doc1: id={doc_id[:12]}")

doc_id2 = store.add_document("sess1", "test2.pdf", "Summary 2", [], doc_hash="hash2")
docs = store.list_documents("sess1")
assert len(docs) == 2
assert docs[0]["is_active"]
print(f"   Added doc2: id={doc_id2[:12]}, total={len(docs)}")

# Test switching
store.set_active_document("sess1", doc_id)
active = store.get_active_document("sess1")
assert active["filename"] == "test.pdf"
print(f"   Switched active to: {active['filename']}")

# Test status
status = store.get_status("sess1")
assert status["document_count"] == 2
assert status["has_document"] is True
print(f"   Status: {status['document_count']} docs, active={status['filename']}")

# Test switch by filename
store.switch_by_filename("sess1", "test2")
active2 = store.get_active_document("sess1")
assert active2["filename"] == "test2.pdf"
print(f"   Switched by filename to: {active2['filename']}")

# Test max documents (limit=3, add a 4th should evict oldest)
store.add_document("sess1", "test3.pdf", "Summary 3", [], doc_hash="hash3")
store.add_document("sess1", "test4.pdf", "Summary 4", [], doc_hash="hash4")
docs_after = store.list_documents("sess1")
assert len(docs_after) == 3  # max 3
filenames = [d["filename"] for d in docs_after]
assert "test.pdf" not in filenames  # oldest evicted
print(f"   Eviction: 4 added, {len(docs_after)} stored, oldest evicted")

# 3. Test smart responder
print("\n3. Testing SmartResponder...")
sr = SmartResponder()

# Exam mode auto-detection (queries must match _EXAM_PATTERNS)
sr.detect_exam_intent("sess1", "define machine learning")
sr.detect_exam_intent("sess1", "what is deep learning")
assert not sr.is_exam_mode("sess1")  # only 2 queries, not yet
sr.detect_exam_intent("sess1", "explain the concept of neural networks")
assert sr.is_exam_mode("sess1")  # auto-activated after 3
print("   Auto exam mode: OK (activated after 3 queries)")

# Manual toggle
sr.set_exam_mode("sess2", True)
assert sr.is_exam_mode("sess2")
sr.set_exam_mode("sess2", False)
assert not sr.is_exam_mode("sess2")
print("   Manual exam toggle: OK")

# Format response
result = sr.format_response(
    "ML is a type of AI...", "what is ML?", "sess1",
    retrieved_chunks=results, has_document=True, doc_filename="test.pdf",
)
assert "suggestions" in result
assert len(result["suggestions"]) > 0
assert result["exam_mode"] is True
assert len(result["citations"]) > 0
print(f"   Format: exam={result['exam_mode']}, suggestions={len(result['suggestions'])}, citations={result['citations']}")

# Citations
citations = sr.format_citations(results)
assert "Page" in citations
print(f"   Citations present: {'Page' in citations}")

# Follow-up suggestions
suggestions = sr.generate_suggestions("define osmosis", has_document=True, is_exam=True, doc_filename="bio.pdf")
assert len(suggestions) > 0
print(f"   Suggestions: {suggestions}")

# 4. Backward compatibility
print("\n4. Testing backward compatibility...")
store2 = DocumentContextStore()
store2.set("sess_legacy", "legacy.pdf", "Old summary", ["chunk1", "chunk2"])
assert store2.has_document("sess_legacy")
doc = store2.get("sess_legacy")
assert doc["filename"] == "legacy.pdf"
print("   Legacy set/get: OK")

# 5. PDF Service validation
print("\n5. Testing PDFService...")
err = PDFService.validate(b"x" * 100, "test.pdf")
assert err is None
print("   Valid file: OK")

err = PDFService.validate(b"x" * 100, "test.txt")
assert err is not None
assert "Only .pdf" in err
print("   Wrong extension: OK")

err = PDFService.validate(b"x" * (60 * 1024 * 1024), "big.pdf")
assert err is not None
assert "too large" in err.lower()
assert "Suggestions" in err
print("   Oversized file: OK (includes suggestions)")

err = PDFService.validate(b"", "empty.pdf")
assert err is not None
print("   Empty file: OK")

# File hash
h = PDFService.file_hash(b"test content")
assert len(h) == 64
print(f"   File hash: {h[:16]}...")

print("\n" + "=" * 50)
print("ALL PHASE 11 TESTS PASSED")
print("=" * 50)
