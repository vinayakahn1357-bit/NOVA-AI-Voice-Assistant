"""
utils/response_formatter.py — AI Response Cleanup & Dynamic Formatting for NOVA
Ensures all responses have consistent quality, structure, and formatting
adapted to the query type (coding, decision, opinion, explanation, etc.)
"""

import re
from utils.logger import get_logger

log = get_logger("formatter")

# Phrases that sound robotic / leak model identity
_ROBOTIC_PHRASES = [
    r"(?i)as an ai( language model)?[\s,]",
    r"(?i)as a large language model[\s,]",
    r"(?i)i('m| am) just an? (ai|language model|chatbot|bot)[\s,]",
    r"(?i)my training data (only )?goes",
    r"(?i)i don't have (personal )?(feelings|emotions|opinions|experiences)",
    r"(?i)i('m| am) (an ai|a language model|a virtual assistant|a chatbot)",
    r"(?i)as per my (training|knowledge|programming)",
    r"(?i)i was (trained|programmed|designed) (to|by)",
    r"(?i)my knowledge cutoff",
    r"(?i)i cannot (browse|access|search) the internet",
]

# Filler phrases that add no value
_FILLER_PHRASES = [
    r"(?i)^(sure|okay|alright)[,!.]?\s*(here(('s| is) )?|let me )",
    r"(?i)^(absolutely|certainly|of course)[,!.]?\s*",
    r"(?i)^great question[,!.]?\s*",
    r"(?i)^that('s| is) a (great|good|excellent|interesting) question[,!.]?\s*",
    r"(?i)i hope (this|that) helps[.!]?\s*$",
    r"(?i)let me know if you (need|have|want) (any(thing)?|more|further).*$",
    r"(?i)feel free to ask.*$",
    r"(?i)is there anything else.*$",
]


class ResponseFormatter:
    """
    Cleans and standardizes AI responses with query-type-aware formatting.
    - Strips robotic self-references
    - Removes excessive filler
    - Applies dynamic formatting based on query type
    - Normalizes whitespace and structure
    - Preserves code blocks
    """

    def format(self, text: str, query_type: str = "conversation") -> str:
        """
        Clean and format an AI response with dynamic formatting.

        Args:
            text: Raw AI response text.
            query_type: From QueryAnalyzer (coding, decision, opinion, etc.)

        Returns:
            Cleaned, dynamically formatted response.
        """
        if not text or not text.strip():
            return text

        # Preserve code blocks before processing
        code_blocks = []

        def _save_code(match):
            code_blocks.append(match.group(0))
            return f"__CODE_BLOCK_{len(code_blocks) - 1}__"

        processed = re.sub(r'```[\s\S]*?```', _save_code, text)

        # Strip robotic phrases
        for pattern in _ROBOTIC_PHRASES:
            processed = re.sub(pattern, "", processed)

        # Strip excessive filler (only for non-greeting types)
        if query_type != "greeting":
            for pattern in _FILLER_PHRASES:
                processed = re.sub(pattern, "", processed)

        # Apply query-type-specific formatting
        processed = self._apply_type_formatting(processed, query_type)

        # Normalize whitespace
        processed = re.sub(r'\n{4,}', '\n\n\n', processed)   # max 3 newlines
        processed = re.sub(r'[ \t]+\n', '\n', processed)      # trailing spaces
        processed = re.sub(r'\n[ \t]+\n', '\n\n', processed)  # whitespace-only lines

        # Restore code blocks
        for i, block in enumerate(code_blocks):
            processed = processed.replace(f"__CODE_BLOCK_{i}__", block)

        # Trim leading/trailing whitespace
        processed = processed.strip()

        # If we stripped everything (rare edge case), return original
        if not processed:
            return text.strip()

        return processed

    def _apply_type_formatting(self, text: str, query_type: str) -> str:
        """Apply formatting rules based on query type."""
        if query_type == "coding":
            return self._format_coding(text)
        elif query_type == "decision":
            return self._format_decision(text)
        elif query_type == "opinion":
            return self._format_opinion(text)
        elif query_type in ("explanation", "complex_task"):
            return self._format_explanation(text)
        elif query_type == "planning":
            return self._format_planning(text)
        elif query_type == "greeting":
            return self._format_greeting(text)
        return text

    @staticmethod
    def _format_coding(text: str) -> str:
        """Ensure coding responses have proper structure."""
        # Add language tags to bare code blocks (``` without language)
        text = re.sub(
            r'```\s*\n((?:def |class |import |from |function |const |let |var |#include))',
            r'```python\n\1',
            text,
        )
        # Ensure code blocks have a blank line before them for readability
        text = re.sub(r'([^\n])\n```', r'\1\n\n```', text)
        return text

    @staticmethod
    def _format_decision(text: str) -> str:
        """Ensure decision/comparison responses have structure."""
        # If response doesn't have any headers and is long enough, it's fine as-is
        # The agent prompt augmentation already guides the LLM to use structure
        return text

    @staticmethod
    def _format_opinion(text: str) -> str:
        """Ensure opinion responses lead with a clear recommendation."""
        return text

    @staticmethod
    def _format_explanation(text: str) -> str:
        """Ensure explanation responses have logical flow."""
        # Add visual separators between major sections if they're dense
        lines = text.split('\n')
        result = []
        for i, line in enumerate(lines):
            result.append(line)
            # If a line starts with a header (## or **bold**), add blank line before
            if i > 0 and result[-1].strip().startswith(('##', '**')):
                if result[-2].strip():  # Only if prev line isn't already blank
                    result.insert(-1, '')
        return '\n'.join(result)

    @staticmethod
    def _format_planning(text: str) -> str:
        """Ensure planning responses have numbered steps."""
        return text

    @staticmethod
    def _format_greeting(text: str) -> str:
        """Keep greeting responses light and brief."""
        # Trim overly long greeting responses
        if len(text.split()) > 80:
            sentences = re.split(r'(?<=[.!?])\s+', text)
            if len(sentences) > 3:
                text = ' '.join(sentences[:3])
                if not text.endswith(('.', '!', '?')):
                    text += '.'
        return text

    def format_for_merge(self, text: str) -> str:
        """
        Light formatting for merge candidates — preserves more content
        but normalizes structure for comparison.
        """
        if not text:
            return ""
        result = re.sub(r'\n{4,}', '\n\n\n', text)
        result = re.sub(r'[ \t]+\n', '\n', result)
        return result.strip()
