"""
services/query_analyzer.py — Adaptive Intelligence Layer for NOVA
Detects query type, complexity, and suggests optimal model parameters.
Does NOT override user-selected mode — enhances results within it.
"""

import re
from utils.logger import get_logger

log = get_logger("query_analyzer")

# ─── Query Type Classifications ───────────────────────────────────────────────
QUERY_TYPES = {
    "greeting":       {"complexity": 1, "optimal_tokens": 128,  "temperature": 0.8},
    "simple_qa":      {"complexity": 2, "optimal_tokens": 256,  "temperature": 0.7},
    "explanation":    {"complexity": 4, "optimal_tokens": 512,  "temperature": 0.7},
    "coding":         {"complexity": 7, "optimal_tokens": 1024, "temperature": 0.4},
    "reasoning":      {"complexity": 8, "optimal_tokens": 1024, "temperature": 0.5},
    "creative":       {"complexity": 5, "optimal_tokens": 768,  "temperature": 0.9},
    "math":           {"complexity": 7, "optimal_tokens": 512,  "temperature": 0.3},
    "conversation":   {"complexity": 2, "optimal_tokens": 256,  "temperature": 0.75},
    "complex_task":   {"complexity": 9, "optimal_tokens": 1536, "temperature": 0.5},
}

# ─── Pattern Matchers ─────────────────────────────────────────────────────────
_GREETING_PATTERNS = re.compile(
    r'^\s*(hi|hello|hey|good\s*(morning|evening|afternoon|night)|'
    r'sup|yo|what\'s up|howdy|greetings)\s*[!?.,]*\s*$',
    re.IGNORECASE
)

_CODING_KEYWORDS = frozenset({
    "code", "function", "implement", "debug", "error", "bug", "script",
    "program", "algorithm", "api", "database", "sql", "regex", "class",
    "method", "variable", "loop", "array", "list", "dict", "json",
    "html", "css", "javascript", "python", "java", "react", "node",
    "import", "export", "module", "package", "library", "framework",
    "compile", "runtime", "syntax", "refactor", "deploy", "docker",
    "git", "github", "terminal", "command", "server", "endpoint",
})

_MATH_KEYWORDS = frozenset({
    "calculate", "compute", "solve", "equation", "integral", "derivative",
    "matrix", "probability", "statistics", "average", "median", "sum",
    "formula", "theorem", "proof", "algebra", "geometry", "calculus",
    "trigonometry", "logarithm", "factorial", "percentage",
})

_REASONING_KEYWORDS = frozenset({
    "analyze", "analyse", "compare", "contrast", "evaluate", "assess",
    "argue", "debate", "critique", "pros and cons", "advantages",
    "disadvantages", "implications", "consequences", "why does",
    "how does", "explain why", "what if", "difference between",
    "trade-off", "tradeoff", "strategy", "approach",
})

_CREATIVE_KEYWORDS = frozenset({
    "write", "story", "poem", "song", "creative", "imagine", "design",
    "invent", "brainstorm", "idea", "concept", "fiction", "narrative",
    "essay", "letter", "email draft", "slogan", "tagline", "name",
})

_EXPLANATION_KEYWORDS = frozenset({
    "explain", "what is", "what are", "how to", "define", "describe",
    "meaning of", "tell me about", "overview", "summary", "summarize",
    "introduction", "guide", "tutorial", "walkthrough",
})


class QueryAnalyzer:
    """
    Analyses user queries to determine type, complexity, and optimal parameters.
    This enhances AI responses without overriding the user's chosen mode.
    """

    def analyze(self, message: str) -> dict:
        """
        Analyse a query and return classification with optimal parameters.

        Returns: {
            "query_type": str,
            "complexity": int (1-10),
            "optimal_tokens": int,
            "optimal_temperature": float,
            "word_count": int,
            "has_code": bool,
            "prefers_groq": bool,    # hint for hybrid mode
            "prefers_ollama": bool,  # hint for hybrid mode
        }
        """
        if not message or not message.strip():
            return self._default_result()

        query_type = self._classify(message)
        config = QUERY_TYPES.get(query_type, QUERY_TYPES["conversation"])
        word_count = len(message.split())
        has_code = "```" in message or bool(re.search(r'(def |function |class |import |var |const |let )', message))

        # Adjust complexity based on message length
        length_boost = min(3, word_count // 50)
        complexity = min(10, config["complexity"] + length_boost)

        # Determine model preference hints for hybrid mode
        prefers_groq = complexity >= 6 or query_type in ("coding", "reasoning", "complex_task", "math")
        prefers_ollama = complexity <= 3 or query_type in ("greeting", "simple_qa", "conversation")

        result = {
            "query_type": query_type,
            "complexity": complexity,
            "optimal_tokens": config["optimal_tokens"],
            "optimal_temperature": config["temperature"],
            "word_count": word_count,
            "has_code": has_code,
            "prefers_groq": prefers_groq,
            "prefers_ollama": prefers_ollama,
        }

        log.info(
            "Query analysis: type=%s complexity=%d tokens=%d words=%d groq=%s ollama=%s",
            query_type, complexity, config["optimal_tokens"], word_count,
            prefers_groq, prefers_ollama,
        )

        return result

    def _classify(self, message: str) -> str:
        """Classify query type based on content patterns."""
        text = message.strip()
        lower = text.lower()
        words = set(lower.split())

        # Greeting check (exact match patterns)
        if _GREETING_PATTERNS.match(text):
            return "greeting"

        # Check for code in the message
        if "```" in text:
            return "coding"

        # Check for math-specific phrases first (higher priority)
        math_phrases = ["integral of", "derivative of", "solve for", "calculate the",
                        "compute the", "equation", "x^", "dx", "dy", "dz",
                        "sin(", "cos(", "tan(", "log(", "ln("]
        math_phrase_hits = sum(1 for p in math_phrases if p in lower)
        if math_phrase_hits >= 2:
            return "math"

        # Calculate keyword overlap scores
        coding_score = len(words & _CODING_KEYWORDS)
        math_score = len(words & _MATH_KEYWORDS)
        reasoning_score = len(words & _REASONING_KEYWORDS)
        creative_score = len(words & _CREATIVE_KEYWORDS)
        explanation_score = len(words & _EXPLANATION_KEYWORDS)

        # Pick the highest scoring category
        scores = {
            "coding": coding_score * 2,
            "math": math_score * 3,           # strongest boost for math
            "reasoning": reasoning_score * 1.8,
            "creative": creative_score * 1.5,
            "explanation": explanation_score * 1.5,
        }

        best = max(scores, key=scores.get)
        if scores[best] >= 2:
            return best

        # Length-based classification
        word_count = len(message.split())
        if word_count > 80:
            return "complex_task"
        if word_count < 8:
            return "simple_qa"

        return "conversation"

    @staticmethod
    def _default_result() -> dict:
        return {
            "query_type": "conversation",
            "complexity": 2,
            "optimal_tokens": 256,
            "optimal_temperature": 0.75,
            "word_count": 0,
            "has_code": False,
            "prefers_groq": False,
            "prefers_ollama": False,
        }
