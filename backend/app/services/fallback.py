"""
Heuristic fallback for claim extraction.

Used when the LLM is unavailable or returns an error.
Splits text into sentence-level claims using simple rules.
"""

import re
from typing import List


def extract_claims_heuristic(text: str) -> List[str]:
    """
    Extract candidate factual claims from text using heuristics.
    
    Strategy:
    1. Split into sentences.
    2. Filter out very short or question-like sentences.
    3. Return remaining sentences as candidate claims.
    """
    # Split on sentence-ending punctuation
    sentences = re.split(r'(?<=[.!])\s+', text.strip())

    claims: List[str] = []
    for sentence in sentences:
        sentence = sentence.strip()

        # Skip empty or very short fragments
        if len(sentence) < 15:
            continue

        # Skip questions
        if sentence.endswith("?"):
            continue

        # Skip obvious opinion markers (very basic filter)
        opinion_markers = [
            "i think", "i believe", "i feel", "in my opinion",
            "it seems", "maybe", "perhaps", "probably",
        ]
        lower = sentence.lower()
        if any(lower.startswith(marker) for marker in opinion_markers):
            continue

        claims.append(sentence)

    # If nothing survived filtering, return the whole text as one claim
    if not claims and len(text.strip()) >= 15:
        claims = [text.strip()]

    return claims
