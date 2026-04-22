"""
services/document_retriever.py — Modular Document Retrieval for NOVA (Phase 11)
Abstract retriever interface with TF-IDF implementation.
Designed for easy swapping to sentence-transformers or OpenAI embeddings later.
"""

import threading
from abc import ABC, abstractmethod

from config import PDF_TOP_K_CHUNKS, ENABLE_DOCUMENT_EMBEDDINGS
from utils.logger import get_logger

log = get_logger("retriever")


# ─── Abstract Base Retriever ──────────────────────────────────────────────────

class BaseRetriever(ABC):
    """
    Abstract interface for document chunk retrieval.
    Implementations must support indexing and querying.

    To add a new retriever (e.g., sentence-transformers, OpenAI):
        1. Subclass BaseRetriever
        2. Implement index_chunks() and retrieve()
        3. Swap in app.py: retriever = NewRetriever()
    """

    @abstractmethod
    def index_chunks(self, doc_id: str, chunks: list[dict]) -> None:
        """
        Index document chunks for later retrieval.

        Args:
            doc_id: Unique document identifier (e.g., doc_hash).
            chunks: List of {"text": str, "page": int, "chunk_index": int, ...}
        """
        ...

    @abstractmethod
    def retrieve(self, doc_id: str, query: str, top_k: int = PDF_TOP_K_CHUNKS) -> list[dict]:
        """
        Retrieve the most relevant chunks for a query.

        Args:
            doc_id: Document identifier to search within.
            query: User's question.
            top_k: Number of top chunks to return.

        Returns:
            List of {"text": str, "page": int, "score": float, "chunk_index": int}
            sorted by relevance (highest score first).
        """
        ...

    @abstractmethod
    def is_indexed(self, doc_id: str) -> bool:
        """Check if a document has been indexed."""
        ...

    @abstractmethod
    def remove(self, doc_id: str) -> bool:
        """Remove a document's index. Returns True if removed."""
        ...

    @abstractmethod
    def clear(self) -> int:
        """Clear all indexes. Returns count removed."""
        ...


# ─── TF-IDF Retriever ────────────────────────────────────────────────────────

class TFIDFRetriever(BaseRetriever):
    """
    TF-IDF-based chunk retriever using scikit-learn.
    Lightweight, fast, no external API calls.
    Good for keyword and topic-based matching.
    """

    def __init__(self):
        self._indexes: dict[str, dict] = {}  # doc_id → {matrix, vectorizer, chunks}
        self._lock = threading.Lock()
        log.info("TFIDFRetriever initialized (scikit-learn backend)")

    def index_chunks(self, doc_id: str, chunks: list[dict]) -> None:
        """Build TF-IDF matrix from document chunks."""
        if not chunks:
            log.warning("No chunks to index for doc %s", doc_id[:12])
            return

        from sklearn.feature_extraction.text import TfidfVectorizer

        texts = [c["text"] for c in chunks]

        # Build vectorizer with document-appropriate settings
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 2),     # unigrams + bigrams for better matching
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,      # apply log normalization
        )

        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
        except ValueError as e:
            log.warning("TF-IDF indexing failed for doc %s: %s", doc_id[:12], e)
            return

        with self._lock:
            self._indexes[doc_id] = {
                "matrix": tfidf_matrix,
                "vectorizer": vectorizer,
                "chunks": chunks,
            }

        log.info("Indexed doc %s: %d chunks, %d features",
                 doc_id[:12], len(chunks),
                 len(vectorizer.get_feature_names_out()))

    def retrieve(self, doc_id: str, query: str, top_k: int = PDF_TOP_K_CHUNKS) -> list[dict]:
        """Find top-k most relevant chunks using cosine similarity."""
        with self._lock:
            index = self._indexes.get(doc_id)

        if not index:
            log.warning("No index for doc %s — returning empty", doc_id[:12])
            return []

        from sklearn.metrics.pairwise import cosine_similarity

        vectorizer = index["vectorizer"]
        matrix = index["matrix"]
        chunks = index["chunks"]

        try:
            query_vec = vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, matrix).flatten()
        except Exception as e:
            log.warning("Retrieval failed for doc %s: %s", doc_id[:12], e)
            return chunks[:top_k]  # fallback: return first chunks

        # Get top-k indices sorted by score
        top_indices = similarities.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score < 0.001:  # skip near-zero matches
                continue

            chunk = chunks[idx].copy()
            chunk["score"] = round(score, 4)
            results.append(chunk)

        log.info("Retrieved %d/%d chunks for doc %s (top score=%.3f)",
                 len(results), len(chunks), doc_id[:12],
                 results[0]["score"] if results else 0)

        return results

    def is_indexed(self, doc_id: str) -> bool:
        """Check if a document has been indexed."""
        with self._lock:
            return doc_id in self._indexes

    def remove(self, doc_id: str) -> bool:
        """Remove a document's index."""
        with self._lock:
            if doc_id in self._indexes:
                del self._indexes[doc_id]
                log.info("Removed index for doc %s", doc_id[:12])
                return True
            return False

    def clear(self) -> int:
        """Clear all indexes."""
        with self._lock:
            count = len(self._indexes)
            self._indexes.clear()
            log.info("All document indexes cleared (%d removed)", count)
            return count

    def stats(self) -> dict:
        """Return retriever statistics."""
        with self._lock:
            return {
                "indexed_documents": len(self._indexes),
                "total_chunks": sum(
                    len(idx["chunks"]) for idx in self._indexes.values()
                ),
                "backend": "tfidf",
            }


# ─── Embedding Retriever (Semantic Search) ────────────────────────────────────

class EmbeddingRetriever(BaseRetriever):
    """
    Semantic chunk retriever using sentence-transformers.
    Uses all-MiniLM-L6-v2 (22MB, 384-dim, CPU-optimized).
    Dramatically better at understanding meaning vs keyword-matching.

    Lazy model loading: the model is only loaded on first index/retrieve call,
    so importing this class has zero startup cost.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self):
        self._model = None         # lazy-loaded
        self._indexes: dict[str, dict] = {}  # doc_id → {embeddings, chunks}
        self._lock = threading.Lock()
        log.info("EmbeddingRetriever initialized (lazy model: %s)", self.MODEL_NAME)

    def _get_model(self):
        """Lazy-load the sentence-transformers model on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped,import-not-found]
            log.info("Loading embedding model '%s'...", self.MODEL_NAME)
            self._model = SentenceTransformer(self.MODEL_NAME)
            log.info("Embedding model loaded (dim=%d)", self._model.get_sentence_embedding_dimension())
        return self._model

    def index_chunks(self, doc_id: str, chunks: list[dict]) -> None:
        """Encode document chunks into dense embeddings."""
        if not chunks:
            log.warning("No chunks to index for doc %s", doc_id[:12])
            return

        model = self._get_model()
        texts = [c["text"] for c in chunks]

        # Batch encode (CPU-optimized)
        embeddings = model.encode(texts, show_progress_bar=False, batch_size=32)

        with self._lock:
            self._indexes[doc_id] = {
                "embeddings": embeddings,
                "chunks": chunks,
            }

        log.info("Indexed doc %s: %d chunks (embedding dim=%d)",
                 doc_id[:12], len(chunks), embeddings.shape[1])

    def retrieve(self, doc_id: str, query: str, top_k: int = PDF_TOP_K_CHUNKS) -> list[dict]:
        """Find top-k most relevant chunks using cosine similarity on embeddings."""
        with self._lock:
            index = self._indexes.get(doc_id)

        if not index:
            log.warning("No index for doc %s — returning empty", doc_id[:12])
            return []

        import numpy as np

        model = self._get_model()
        query_embedding = model.encode([query], show_progress_bar=False)

        embeddings = index["embeddings"]
        chunks = index["chunks"]

        # Cosine similarity (embeddings are already normalized by default)
        similarities = np.dot(embeddings, query_embedding.T).flatten()

        # Get top-k indices sorted by score
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score < 0.05:  # skip very low matches
                continue
            chunk = chunks[idx].copy()
            chunk["score"] = round(score, 4)
            results.append(chunk)

        log.info("Retrieved %d/%d chunks for doc %s (top score=%.3f, backend=embedding)",
                 len(results), len(chunks), doc_id[:12],
                 results[0]["score"] if results else 0)

        return results

    def is_indexed(self, doc_id: str) -> bool:
        with self._lock:
            return doc_id in self._indexes

    def remove(self, doc_id: str) -> bool:
        with self._lock:
            if doc_id in self._indexes:
                del self._indexes[doc_id]
                log.info("Removed embedding index for doc %s", doc_id[:12])
                return True
            return False

    def clear(self) -> int:
        with self._lock:
            count = len(self._indexes)
            self._indexes.clear()
            log.info("All embedding indexes cleared (%d removed)", count)
            return count

    def stats(self) -> dict:
        with self._lock:
            return {
                "indexed_documents": len(self._indexes),
                "total_chunks": sum(
                    len(idx["chunks"]) for idx in self._indexes.values()
                ),
                "backend": "sentence-transformers",
                "model": self.MODEL_NAME,
            }


# ─── Factory ──────────────────────────────────────────────────────────────────

def create_retriever() -> BaseRetriever:
    """
    Factory function to create the appropriate retriever.
    Selection logic:
        - RETRIEVER_BACKEND=tfidf      → always TF-IDF
        - RETRIEVER_BACKEND=embeddings  → always embeddings (errors if not installed)
        - RETRIEVER_BACKEND=auto (default) → embeddings if available, else TF-IDF
    """
    import os
    backend = os.getenv("RETRIEVER_BACKEND", "auto").lower().strip()

    if not ENABLE_DOCUMENT_EMBEDDINGS:
        log.info("Document embeddings disabled — using TFIDFRetriever")
        return TFIDFRetriever()

    if backend == "tfidf":
        log.info("RETRIEVER_BACKEND=tfidf — using TFIDFRetriever")
        return TFIDFRetriever()

    if backend == "embeddings":
        return EmbeddingRetriever()

    # auto mode: try embeddings, fallback to TF-IDF
    try:
        import sentence_transformers  # type: ignore[import-untyped,import-not-found]  # noqa: F401
        log.info("sentence-transformers available — using EmbeddingRetriever")
        return EmbeddingRetriever()
    except ImportError:
        log.info("sentence-transformers not installed — falling back to TFIDFRetriever")
        return TFIDFRetriever()
