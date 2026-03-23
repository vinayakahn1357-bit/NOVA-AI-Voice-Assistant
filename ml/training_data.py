"""
ml/training_data.py — Labeled Training Data for Personality Classifier
~50 examples across 5 personality categories.
"""

TRAINING_DATA = [
    # ── Teacher (structured, step-by-step, conceptual) ────────────────────
    ("explain this step by step", "teacher"),
    ("can you teach me how this works", "teacher"),
    ("break this down for me", "teacher"),
    ("I don't understand, explain simply", "teacher"),
    ("give me a tutorial on this topic", "teacher"),
    ("explain like I'm a beginner", "teacher"),
    ("walk me through this concept", "teacher"),
    ("how does this work in simple terms", "teacher"),
    ("can you explain the basics of", "teacher"),
    ("help me understand this better", "teacher"),
    ("what are the fundamentals of", "teacher"),
    ("teach me about this from scratch", "teacher"),

    # ── Friend (casual, fun, conversational) ──────────────────────────────
    ("bro tell me simply", "friend"),
    ("hey what's up with this thing", "friend"),
    ("lol can you help me out", "friend"),
    ("dude I'm so confused", "friend"),
    ("yo explain this real quick", "friend"),
    ("haha that's crazy, tell me more", "friend"),
    ("just chat with me about this", "friend"),
    ("chill explanation please", "friend"),
    ("keep it casual and simple", "friend"),
    ("talk to me like a friend would", "friend"),

    # ── Expert (deep, technical, precise) ─────────────────────────────────
    ("give deep technical details", "expert"),
    ("provide a comprehensive analysis", "expert"),
    ("explain the underlying architecture", "expert"),
    ("what are the edge cases and caveats", "expert"),
    ("give me the technical specification", "expert"),
    ("explain the algorithmic complexity", "expert"),
    ("what's the implementation detail here", "expert"),
    ("provide a thorough technical review", "expert"),
    ("analyze this with full depth", "expert"),
    ("give me the advanced explanation", "expert"),
    ("discuss the internal mechanisms", "expert"),
    ("what are the performance implications", "expert"),

    # ── Coach (motivational, actionable, goal-oriented) ───────────────────
    ("motivate me to study", "coach"),
    ("help me stay on track with my goals", "coach"),
    ("I need encouragement to keep going", "coach"),
    ("give me actionable steps to improve", "coach"),
    ("push me to be better", "coach"),
    ("help me build a plan to succeed", "coach"),
    ("I'm feeling stuck, guide me forward", "coach"),
    ("what should I do next to grow", "coach"),
    ("inspire me to take action", "coach"),
    ("help me overcome this challenge", "coach"),

    # ── Default (general, neutral) ────────────────────────────────────────
    ("what is the weather today", "default"),
    ("tell me a fact", "default"),
    ("what time is it", "default"),
    ("summarize this for me", "default"),
    ("who is the president", "default"),
    ("translate this to Spanish", "default"),
    ("what's the capital of France", "default"),
    ("calculate 25 times 47", "default"),
    ("write a poem about nature", "default"),
    ("search for Python documentation", "default"),
]


def get_training_data() -> tuple[list[str], list[str]]:
    """Return (texts, labels) from training data."""
    texts = [item[0] for item in TRAINING_DATA]
    labels = [item[1] for item in TRAINING_DATA]
    return texts, labels
