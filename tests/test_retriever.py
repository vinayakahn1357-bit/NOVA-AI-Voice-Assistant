"""
tests/test_retriever.py — Tests for the document retriever system.
Verifies TF-IDF retrieval and factory auto-selection logic.
"""

import pytest
from services.document_retriever import (
    TFIDFRetriever, EmbeddingRetriever, create_retriever, BaseRetriever,
)


@pytest.fixture
def tfidf():
    """Fresh TF-IDF retriever instance."""
    return TFIDFRetriever()


@pytest.fixture
def sample_chunks():
    """Sample document chunks for testing."""
    return [
        {"text": "Machine learning is a subset of artificial intelligence",
         "page": 1, "chunk_index": 0},
        {"text": "Deep learning uses neural networks with many layers",
         "page": 1, "chunk_index": 1},
        {"text": "Python is the most popular language for data science",
         "page": 2, "chunk_index": 2},
        {"text": "Natural language processing handles text data",
         "page": 3, "chunk_index": 3},
        {"text": "Computer vision deals with image recognition tasks",
         "page": 4, "chunk_index": 4},
    ]


class TestTFIDFRetriever:
    """Test TF-IDF retriever core functionality."""

    def test_index_and_retrieve(self, tfidf, sample_chunks):
        tfidf.index_chunks("doc1", sample_chunks)
        results = tfidf.retrieve("doc1", "machine learning AI")
        assert len(results) > 0
        assert results[0]["score"] > 0

    def test_is_indexed(self, tfidf, sample_chunks):
        assert tfidf.is_indexed("doc1") is False
        tfidf.index_chunks("doc1", sample_chunks)
        assert tfidf.is_indexed("doc1") is True

    def test_remove(self, tfidf, sample_chunks):
        tfidf.index_chunks("doc1", sample_chunks)
        assert tfidf.remove("doc1") is True
        assert tfidf.is_indexed("doc1") is False

    def test_clear(self, tfidf, sample_chunks):
        tfidf.index_chunks("doc1", sample_chunks)
        tfidf.index_chunks("doc2", sample_chunks)
        count = tfidf.clear()
        assert count == 2
        assert tfidf.is_indexed("doc1") is False

    def test_retrieve_nonexistent_doc(self, tfidf):
        results = tfidf.retrieve("nonexistent", "query")
        assert results == []

    def test_empty_chunks_skipped(self, tfidf):
        tfidf.index_chunks("empty_doc", [])
        assert tfidf.is_indexed("empty_doc") is False

    def test_relevance_ranking(self, tfidf, sample_chunks):
        tfidf.index_chunks("doc1", sample_chunks)
        results = tfidf.retrieve("doc1", "neural networks deep learning")
        # The deep learning chunk should rank highest
        assert results[0]["chunk_index"] == 1

    def test_stats(self, tfidf, sample_chunks):
        tfidf.index_chunks("doc1", sample_chunks)
        stats = tfidf.stats()
        assert stats["indexed_documents"] == 1
        assert stats["total_chunks"] == 5
        assert stats["backend"] == "tfidf"


class TestRetrieverFactory:
    """Test the create_retriever factory function."""

    def test_factory_returns_base_retriever(self):
        r = create_retriever()
        assert isinstance(r, BaseRetriever)

    def test_factory_returns_working_retriever(self):
        r = create_retriever()
        # Should support all base methods
        assert hasattr(r, "index_chunks")
        assert hasattr(r, "retrieve")
        assert hasattr(r, "is_indexed")
        assert hasattr(r, "remove")
        assert hasattr(r, "clear")

    def test_embedding_retriever_exists(self):
        """Verify EmbeddingRetriever class exists and inherits correctly."""
        assert issubclass(EmbeddingRetriever, BaseRetriever)


class TestEmbeddingRetrieverAvailability:
    """Test embedding retriever (only if sentence-transformers is installed)."""

    @pytest.fixture
    def has_sentence_transformers(self):
        try:
            import sentence_transformers
            return True
        except ImportError:
            return False

    def test_factory_selects_embeddings_when_available(self, has_sentence_transformers):
        r = create_retriever()
        if has_sentence_transformers:
            assert isinstance(r, EmbeddingRetriever), \
                "Factory should select EmbeddingRetriever when sentence-transformers is installed"
        else:
            assert isinstance(r, TFIDFRetriever), \
                "Factory should fall back to TFIDFRetriever"
