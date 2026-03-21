"""
utils/response_formatter.py — AI Response Cleanup & Standardization for NOVA
Ensures all responses have consistent assistant-style tone and formatting.
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
    Cleans and standardizes AI responses for consistent assistant quality.
    - Strips robotic self-references
    - Removes excessive filler
    - Normalizes whitespace and formatting
    - Preserves code blocks
    """

    def format(self, text: str, query_type: str = "conversation") -> str:
        """
        Clean and format an AI response.

        Args:
            text: Raw AI response text.
            query_type: From QueryAnalyzer (affects formatting rules).

        Returns:
            Cleaned, standardized response.
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

        # Normalize whitespace
        processed = re.sub(r'\n{4,}', '\n\n\n', processed)  # max 3 newlines
        processed = re.sub(r'[ \t]+\n', '\n', processed)     # trailing spaces
        processed = re.sub(r'\n[ \t]+\n', '\n\n', processed) # whitespace-only lines

        # Restore code blocks
        for i, block in enumerate(code_blocks):
            processed = processed.replace(f"__CODE_BLOCK_{i}__", block)

        # Trim leading/trailing whitespace
        processed = processed.strip()

        # If we stripped everything (rare edge case), return original
        if not processed:
            return text.strip()

        return processed

    def format_for_merge(self, text: str) -> str:
        """
        Light formatting for merge candidates — preserves more content
        but normalizes structure for comparison.
        """
        if not text:
            return ""

        # Just normalize whitespace and trim
        result = re.sub(r'\n{4,}', '\n\n\n', text)
        result = re.sub(r'[ \t]+\n', '\n', result)
        return result.strip()
