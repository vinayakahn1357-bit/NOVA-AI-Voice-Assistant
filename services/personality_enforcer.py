"""
services/personality_enforcer.py — NOVA Personality Enforcement Engine (Phase 12)

Runs AFTER LLM generation, BEFORE response reaches the user.

Stages:
  1. Strip forbidden phrases (global + per-personality)
  2. Personality strength scoring (tone + structure + pattern checks)
  3. Structure patching (lightweight corrections without regenerating)
  4. Romantic safety gate (hard-block unsafe content)
  5. Logging of all enforcement actions

The enforcer does NOT call the LLM — it cleans and validates.
Regeneration (if needed) is triggered by ResponsePipeline based on score.
"""

import re
from utils.logger import get_logger

log = get_logger("personality_enforcer")

# ─── Scoring Thresholds ───────────────────────────────────────────────────────

SCORE_PASS_THRESHOLD = 0.5   # Minimum score to pass without regeneration
SCORE_REGEN_THRESHOLD = 0.3  # Below this → hard fallback

# ─── Romantic Safety Patterns ─────────────────────────────────────────────────

_ROMANTIC_UNSAFE_PATTERNS = [
    r'\bsex\b', r'\bsexual\b', r'\bnaked\b', r'\bnude\b',
    r'\berotic\b', r'\bintimate\b(?!\s+friend)', r'\bseductive\b',
    r'\bexplicit\b', r'\bporn\b', r'\bfetish\b',
    r'\bI love you\b', r'\bI want you\b', r'\bcome to me\b',
    r'\byou\'re beautiful\b', r'\byou\'re sexy\b', r'\byou\'re hot\b',
]
_ROMANTIC_UNSAFE_RE = [re.compile(p, re.IGNORECASE) for p in _ROMANTIC_UNSAFE_PATTERNS]

# ─── Structure Validators ─────────────────────────────────────────────────────

def _has_numbered_steps(text: str) -> bool:
    """Check for numbered steps like '1.', '1)', 'Step 1:'."""
    return bool(re.search(r'^\s*(?:\d+[\.\)]\s+|Step\s+\d+[:\.])', text, re.MULTILINE))


def _has_summary_section(text: str) -> bool:
    lower = text.lower()
    return any(m in lower for m in [
        "📝 summary", "## summary", "**summary**", "in summary",
        "to recap", "key takeaways", "key points", "summary:",
        "📝 **summary", "to summarize", "wrapping up",
    ])


def _has_action_emojis(text: str) -> bool:
    return "🎯" in text or "⚡" in text or "🔥" in text


def _has_call_to_action(text: str) -> bool:
    lower = text.lower()
    return any(p in lower for p in [
        "take the first step", "do it now", "start today",
        "commit to", "execute", "make it happen", "go for it",
        "take action", "get started", "right now", "first step",
        "stop waiting", "move forward", "act now",
    ])


def _is_conversational(text: str) -> bool:
    """Check if text sounds conversational (contractions, casual phrases)."""
    contractions = ["you're", "it's", "that's", "let's", "don't", "can't", "won't", "i've", "we've"]
    lower = text.lower()
    return sum(1 for c in contractions if c in lower) >= 2


def _is_prose_heavy(text: str) -> bool:
    """Check text has minimal bullets/headers (for calm/friend modes)."""
    bullet_count = len(re.findall(r'^\s*[-*•]\s', text, re.MULTILINE))
    header_count = len(re.findall(r'^#{1,3}\s', text, re.MULTILINE))
    return (bullet_count + header_count) <= 2


def _has_humor_markers(text: str) -> bool:
    """Check for humor patterns."""
    lower = text.lower()
    humor_words = ["😄", "😂", "🤣", "haha", "lol", "irony", "absurd",
                   "plot twist", "funny", "hilarious", "joke", "humor"]
    return any(w in lower for w in humor_words)


def _is_short_enough(text: str, max_words: int = 140) -> bool:
    return len(text.split()) <= max_words


def _has_technical_depth(text: str) -> bool:
    """Check for markers of technical depth."""
    lower = text.lower()
    depth_markers = [
        "trade-off", "tradeoff", "best practice", "consider", "caveat",
        "edge case", "performance", "however", "alternatively", "note that",
        "recommended", "avoid", "prefer",
    ]
    return sum(1 for m in depth_markers if m in lower) >= 2


def _has_charm_markers(text: str) -> bool:
    """Check for romantic/charming tone markers."""
    lower = text.lower()
    charm_words = ["💫", "✨", "warmth", "beautiful", "wonderful",
                   "delight", "charming", "heartfelt", "poetic", "curious mind"]
    return any(w in lower for w in charm_words)


# ─── Forbidden Phrase Cleaner ─────────────────────────────────────────────────

def _clean_forbidden_phrases(text: str, forbidden: list) -> tuple:
    """
    Remove sentences containing forbidden phrases.
    Preserves code blocks.
    Returns (cleaned_text, count_removed).
    """
    if not forbidden:
        return text, 0

    code_blocks = []

    def _save_code(match):
        code_blocks.append(match.group(0))
        return f"__CODEBLOCK_{len(code_blocks) - 1}__"

    processed = re.sub(r'```[\s\S]*?```', _save_code, text)

    removed = 0
    for phrase in forbidden:
        phrase_lower = phrase.lower()
        if phrase_lower not in processed.lower():
            continue

        # Strategy 0: the whole processed text IS the forbidden phrase
        if processed.lower().strip() == phrase_lower.strip():
            processed = ""
            removed += 1
            continue

        # Strategy 1: sentence-boundary regex (works on multi-sentence text)
        # Only use if result is non-empty (avoids over-cleaning single-sentence inputs)
        pattern = re.compile(
            r'[^\n.!?]*' + re.escape(phrase) + r'[^\n.!?]*[.!?\n]?',
            re.IGNORECASE
        )
        new_text = pattern.sub("", processed).strip()
        if new_text and new_text != processed.strip():
            removed += 1
            processed = new_text
            continue

        # Strategy 2: line-level removal (when no sentence punctuation, or Strategy 1 over-cleaned)
        lines = processed.split('\n')
        new_lines = [l for l in lines if phrase_lower not in l.lower()]
        if new_lines and len(new_lines) < len(lines):
            processed = '\n'.join(new_lines)
            removed += 1
            continue

        # Strategy 3: direct substring wipe (last resort)
        start = processed.lower().find(phrase_lower)
        if start != -1:
            end = start + len(phrase)
            processed = (processed[:start] + processed[end:]).strip()
            removed += 1

    for i, block in enumerate(code_blocks):
        processed = processed.replace(f"__CODEBLOCK_{i}__", block)

    processed = re.sub(r'  +', ' ', processed)
    processed = re.sub(r'\n{3,}', '\n\n', processed)
    return processed.strip(), removed


# ─── Structure Patchers ───────────────────────────────────────────────────────

def _patch_teacher_structure(text: str) -> str:
    if _has_numbered_steps(text):
        return text
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if len(paragraphs) < 2:
        return text
    result = [paragraphs[0], ""]
    for i, para in enumerate(paragraphs[1:], 1):
        if not re.match(r'^\d+[\.\)]', para):
            result.append(f"{i}. {para}")
        else:
            result.append(para)
    if not _has_summary_section(text):
        result += ["", "📝 **Summary:** " + paragraphs[-1][:220]]
    return '\n\n'.join(result)


def _patch_coach_structure(text: str) -> str:
    if _has_action_emojis(text):
        return text
    if len(text.split()) < 20:
        return text
    lines = text.strip().split('\n')
    if lines:
        lines[0] = "🎯 " + lines[0]
    if not _has_call_to_action(text):
        lines.append("\n🔥 Stop waiting — take the first step right now!")
    return '\n'.join(lines)


def _patch_hacker_truncate(text: str, max_words: int = 150) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    truncated = ' '.join(words[:max_words])
    last_end = max(truncated.rfind('.'), truncated.rfind('\n'))
    if last_end > 80:
        return truncated[:last_end + 1].strip()
    return truncated.strip()


def _patch_friend_format(text: str) -> str:
    """Strip formal headers from friend responses."""
    text = re.sub(r'^#{1,3}\s+.+$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _patch_calm_format(text: str) -> str:
    """Convert bullets to prose for calm mode."""
    text = re.sub(r'^\s*[-*•]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ─── Personality Strength Scorer ─────────────────────────────────────────────

def score_personality_strength(text: str, personality: str) -> tuple:
    """
    Score how well a response matches the expected personality.
    Returns (score: float 0.0-1.0, breakdown: dict).

    Scoring criteria per personality:
    - teacher:  has_steps(40%) + has_summary(30%) + length≥100w(30%)
    - hacker:   is_short(50%) + no_filler(50%)
    - friend:   conversational(50%) + no_headers(50%)
    - expert:   technical_depth(50%) + length≥150w(30%) + no_hedging(20%)
    - coach:    has_action_emoji(40%) + has_cta(30%) + length≥60w(30%)
    - calm:     is_prose(50%) + no_urgency(50%)
    - genius:   length_ok(30%) + confident(40%) + insight_markers(30%)
    - funny:    humor_markers(60%) + correct(40%)
    - romantic: charm_markers(50%) + no_explicit(50%)
    - default:  always 1.0 (no specific requirements)
    """
    text_lower = text.lower()
    word_count = len(text.split())

    if personality == "default":
        return 1.0, {"default": "pass"}

    if personality == "teacher":
        steps = 0.4 if _has_numbered_steps(text) else 0.0
        summary = 0.3 if _has_summary_section(text) else 0.0
        length = 0.3 if word_count >= 80 else (0.15 if word_count >= 40 else 0.0)
        score = steps + summary + length
        return score, {"steps": steps, "summary": summary, "length": length}

    if personality == "hacker":
        short = 0.5 if _is_short_enough(text, 140) else (0.2 if _is_short_enough(text, 200) else 0.0)
        filler_words = ["certainly", "of course", "let me explain", "I hope this"]
        no_filler = 0.5 if not any(w in text_lower for w in filler_words) else 0.1
        score = short + no_filler
        return score, {"short": short, "no_filler": no_filler}

    if personality == "friend":
        casual = 0.5 if _is_conversational(text) else 0.0
        no_headers = 0.5 if _is_prose_heavy(text) else 0.1
        score = casual + no_headers
        return score, {"conversational": casual, "no_headers": no_headers}

    if personality == "expert":
        depth = 0.5 if _has_technical_depth(text) else 0.1
        length = 0.3 if word_count >= 120 else (0.15 if word_count >= 60 else 0.0)
        hedge_words = ["kind of", "sort of", "i guess", "maybe", "i think maybe"]
        no_hedge = 0.2 if not any(w in text_lower for w in hedge_words) else 0.0
        score = depth + length + no_hedge
        return score, {"depth": depth, "length": length, "no_hedge": no_hedge}

    if personality == "coach":
        # emojis(30%) + cta(40%) + length(30%)
        # CTA-heavy text can score >=0.5 even without emojis (patcher adds them)
        emojis = 0.3 if _has_action_emojis(text) else 0.0
        cta = 0.4 if _has_call_to_action(text) else 0.0
        length = 0.3 if word_count >= 40 else (0.15 if word_count >= 20 else 0.1)
        score = emojis + cta + length
        return score, {"emojis": emojis, "cta": cta, "length": length}

    if personality == "calm":
        prose = 0.5 if _is_prose_heavy(text) else 0.1
        urgency_words = ["urgent", "immediately", "asap", "right now", "hurry"]
        no_urgency = 0.5 if not any(w in text_lower for w in urgency_words) else 0.0
        score = prose + no_urgency
        return score, {"prose": prose, "no_urgency": no_urgency}

    if personality == "genius":
        length_ok = 0.3 if 50 <= word_count <= 400 else 0.1
        confident_words = ["the key", "the real", "the trick", "what most", "here's why", "think about"]
        confident = 0.4 if any(w in text_lower for w in confident_words) else 0.2
        insight = 0.3 if word_count >= 40 else 0.1
        score = length_ok + confident + insight
        return score, {"length_ok": length_ok, "confident": confident, "insight": insight}

    if personality == "funny":
        humor = 0.6 if _has_humor_markers(text) else 0.1
        correct = 0.4 if word_count >= 20 else 0.0
        score = humor + correct
        return score, {"humor": humor, "correct": correct}

    if personality == "romantic":
        charm = 0.5 if _has_charm_markers(text) else 0.2
        safe = 0.5 if not any(p.search(text) for p in _ROMANTIC_UNSAFE_RE) else 0.0
        score = charm + safe
        return score, {"charm": charm, "safe": safe}

    # Unknown personality — pass
    return 1.0, {"unknown": "pass"}


# ─── Main Enforcer Class ──────────────────────────────────────────────────────

class PersonalityEnforcer:
    """
    Post-processing enforcement engine for NOVA personality system.

    Usage:
        enforcer = PersonalityEnforcer()
        result = enforcer.enforce(response_text, personality_key, query_type)
        score, breakdown = enforcer.score(response_text, personality_key)
        safe = enforcer.is_romantic_safe(response_text)
    """

    def enforce(self, text: str, personality: str, query_type: str = "conversation") -> str:
        """
        Full enforcement pipeline on an LLM response.

        Returns cleaned, structure-validated, safety-checked response.
        """
        if not text or not text.strip():
            return text

        try:
            from services.personality_service import get_forbidden_phrases
        except ImportError:
            log.warning("PersonalityEnforcer: could not import personality_service")
            return text

        original_text = text
        actions = []

        # ── Stage 1: Romantic safety gate (hard block) ───────────────────────
        if personality == "romantic":
            if not self.is_romantic_safe(text):
                log.warning(
                    "Romantic safety violation detected — replacing with safe fallback"
                )
                return (
                    "That's a really thoughtful question 😊 "
                    "Let me share something on that in a more meaningful way.\n\n"
                    + self._get_romantic_safe_redirect(text)
                )

        # ── Stage 2: Strip forbidden phrases ─────────────────────────────────
        forbidden = get_forbidden_phrases(personality)
        cleaned, removed_count = _clean_forbidden_phrases(text, forbidden)
        if removed_count > 0:
            actions.append(f"removed_{removed_count}_forbidden_phrases")
            text = cleaned

        # ── Stage 3: Structure patching ───────────────────────────────────────
        text, patch_action = self._patch_structure(text, personality)
        if patch_action:
            actions.append(patch_action)

        # ── Stage 4: Final cleanup ────────────────────────────────────────────
        text = text.strip()
        if not text:
            log.warning(
                "Enforcement over-cleaned response for '%s' — restoring original",
                personality,
            )
            return original_text.strip()

        if actions:
            log.info(
                "Personality enforcer [%s]: %s",
                personality, ", ".join(actions),
            )

        return text

    def score(self, text: str, personality: str) -> tuple:
        """
        Score a response's personality compliance.
        Returns (score: float, breakdown: dict).
        """
        return score_personality_strength(text, personality)

    def is_romantic_safe(self, text: str) -> bool:
        """Hard safety check for romantic mode. Returns True if safe."""
        for pattern in _ROMANTIC_UNSAFE_RE:
            if pattern.search(text):
                return False
        return True

    def _patch_structure(self, text: str, personality: str) -> tuple:
        """
        Apply lightweight structural corrections.
        Returns (patched_text, action_label or None).
        """
        try:
            from services.personality_service import PERSONALITIES
            config = PERSONALITIES.get(personality, {})
            required_format = config.get("required_format")
        except ImportError:
            return text, None

        if required_format == "numbered_steps":
            # Threshold: 10 words minimum (real LLM output always >>10 words)
            if not _has_numbered_steps(text) and len(text.split()) > 10:
                return _patch_teacher_structure(text), "patched_teacher_steps"

        elif required_format == "action_plan":
            # Threshold: 10 words minimum
            if not _has_action_emojis(text) and len(text.split()) > 10:
                return _patch_coach_structure(text), "patched_coach_emojis"

        elif required_format == "answer_first":
            if not _is_short_enough(text, 180):
                return _patch_hacker_truncate(text, 150), "truncated_hacker"

        elif required_format == "conversational":
            patched = _patch_friend_format(text)
            if patched != text:
                return patched, "patched_friend_headers"

        elif required_format == "reflective":
            patched = _patch_calm_format(text)
            if patched != text:
                return patched, "patched_calm_bullets"

        return text, None

    @staticmethod
    def _get_romantic_safe_redirect(original: str) -> str:
        """Generate a safe redirect for romantic safety violations."""
        # Return a sanitized version by extracting non-flagged content
        sentences = re.split(r'[.!?]\s+', original)
        safe_sentences = [
            s for s in sentences
            if not any(p.search(s) for p in _ROMANTIC_UNSAFE_RE)
        ]
        if safe_sentences:
            return '. '.join(safe_sentences[:3]).strip() + '.'
        return "Let's explore this topic together in a thoughtful way ✨"

    @staticmethod
    def get_structure_report(text: str, personality: str) -> dict:
        """Debug helper: report structural compliance metrics."""
        score, breakdown = score_personality_strength(text, personality)
        return {
            "personality": personality,
            "score": round(score, 3),
            "breakdown": breakdown,
            "word_count": len(text.split()),
            "has_numbered_steps": _has_numbered_steps(text),
            "has_summary": _has_summary_section(text),
            "has_action_emojis": _has_action_emojis(text),
            "is_conversational": _is_conversational(text),
            "is_prose_heavy": _is_prose_heavy(text),
            "is_short": _is_short_enough(text),
            "passes_threshold": score >= SCORE_PASS_THRESHOLD,
        }
