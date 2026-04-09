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


# ─── Factory ──────────────────────────────────────────────────────────────────

def create_retriever() -> BaseRetriever:
    """
    Factory function to create the appropriate retriever.
    Currently returns TFIDFRetriever.
    Future: switch based on config or available dependencies.
    """
    if not ENABLE_DOCUMENT_EMBEDDINGS:
        log.info("Document embeddings disabled — retriever will return raw chunks")
        # Return TFIDFRetriever anyway; it just won't be called if disabled
        return TFIDFRetriever()

    # Future extension point:
    # if config.RETRIEVER_BACKEND == "sentence_transformers":
    #     return SentenceTransformerRetriever()
    # elif config.RETRIEVER_BACKEND == "openai":
    #     return OpenAIEmbeddingRetriever()

    return TFIDFRetriever()
