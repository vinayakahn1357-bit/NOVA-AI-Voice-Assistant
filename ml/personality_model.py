"""
ml/personality_model.py — Lightweight ML Personality Predictor for NOVA
Uses TF-IDF + LogisticRegression to predict user personality preference from text.
Safe: returns ("default", 0.0) on any error; never blocks the chat pipeline.
"""

import os
import pickle
from utils.logger import get_logger

log = get_logger("ml.personality")


class PersonalityModel:
    """
    Lightweight text classifier for predicting user personality preference.

    Uses scikit-learn's TfidfVectorizer + LogisticRegression pipeline.
    Designed for fast inference (<1ms) with minimal memory footprint.
    """

    VALID_LABELS = {"default", "teacher", "friend", "expert", "coach"}

    def __init__(self):
        self._pipeline = None
        self._is_loaded = False

    @property
    def is_ready(self) -> bool:
        return self._is_loaded and self._pipeline is not None

    def train(self, X: list[str], y: list[str]):
        """
        Train the personality classifier.

        Args:
            X: list of text samples
            y: list of personality labels
        """
        try:
            from sklearn.pipeline import Pipeline
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression

            self._pipeline = Pipeline([
                ("tfidf", TfidfVectorizer(
                    max_features=500,
                    ngram_range=(1, 2),
                    stop_words="english",
                    lowercase=True,
                    sublinear_tf=True,
                )),
                ("clf", LogisticRegression(
                    max_iter=1000,
                    C=1.0,
                    solver="lbfgs",
                    class_weight="balanced",
                )),
            ])

            self._pipeline.fit(X, y)
            self._is_loaded = True
            log.info("Model trained: %d samples, %d classes",
                     len(X), len(set(y)))

        except Exception as exc:
            log.error("Training failed: %s", exc)
            self._pipeline = None
            self._is_loaded = False
            raise

    def predict(self, text: str) -> tuple[str, float]:
        """
        Predict the personality for a given text.

        Args:
            text: user message text

        Returns:
            (personality_label, confidence) — e.g. ("teacher", 0.82)
            Falls back to ("default", 0.0) on any error.
        """
        if not self.is_ready or not text or not text.strip():
            return ("default", 0.0)

        try:
            label = self._pipeline.predict([text])[0]
            probabilities = self._pipeline.predict_proba([text])[0]
            confidence = float(max(probabilities))

            if label not in self.VALID_LABELS:
                return ("default", 0.0)

            return (label, confidence)

        except Exception as exc:
            log.warning("Prediction failed (fallback to default): %s", exc)
            return ("default", 0.0)

    def predict_top_n(self, text: str, n: int = 3) -> list[tuple[str, float]]:
        """
        Return top-N predictions with confidence scores.

        Returns:
            List of (label, confidence) sorted by confidence descending.
        """
        if not self.is_ready or not text or not text.strip():
            return [("default", 0.0)]

        try:
            probabilities = self._pipeline.predict_proba([text])[0]
            classes = self._pipeline.classes_
            scored = sorted(
                zip(classes, probabilities),
                key=lambda x: x[1],
                reverse=True,
            )
            return [(label, float(conf)) for label, conf in scored[:n]]

        except Exception as exc:
            log.warning("Top-N prediction failed: %s", exc)
            return [("default", 0.0)]

    def save(self, path: str):
        """Save the trained model to disk."""
        if not self.is_ready:
            raise RuntimeError("No trained model to save")

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._pipeline, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info("Model saved: %s", path)

    def load(self, path: str) -> bool:
        """
        Load a pre-trained model from disk.

        Returns:
            True if loaded successfully, False otherwise.
        """
        if not os.path.exists(path):
            log.warning("Model file not found: %s", path)
            return False

        try:
            with open(path, "rb") as f:
                self._pipeline = pickle.load(f)
            self._is_loaded = True
            log.info("Model loaded: %s", path)
            return True

        except Exception as exc:
            log.error("Failed to load model: %s", exc)
            self._pipeline = None
            self._is_loaded = False
            return False
