"""
Heuristic fallback for claim extraction.

Used when the LLM is unavailable or returns an error.
Splits text into atomic-ish claims using regex-based rules:
  1. Split into sentences.
  2. Split compound sentences on conjunctions.
  3. Filter out opinions, questions, and short fragments.
  4. Deduplicate.
"""

import re
from typing import List


# Conjunctions that typically join separate factual claims
_COMPOUND_SPLITTERS = re.compile(
    r'\b(?:and|but|while|whereas|moreover|also|furthermore|however|additionally)\b',
    re.IGNORECASE,
)

# Opinion markers — sentences starting with these are likely opinions
_OPINION_MARKERS = [
    "i think", "i believe", "i feel", "in my opinion",
    "it seems", "maybe", "perhaps", "probably",
    "i guess", "i suppose", "i reckon", "personally",
    "in my view", "from my perspective", "i would say",
]

# Meta-statement markers — not factual claims
_META_MARKERS = [
    "let me", "here is", "here are", "the text",
    "this text", "the following", "note that", "as mentioned",
    "for example", "in other words", "to summarize",
]

MIN_CLAIM_LENGTH = 15


def extract_claims_heuristic(text: str) -> List[str]:
    """
    Extract candidate factual claims from text using heuristics.

    Strategy:
    1. Split into sentences (on . ! and newlines).
    2. For each sentence, try to split on conjunctions if compound.
    3. Filter out questions, opinions, meta-statements, and short fragments.
    4. Deduplicate (case-insensitive).
    """
    # Step 1: Split into sentences
    raw_sentences = _split_sentences(text)

    # Step 2: Split compound sentences
    fragments: List[str] = []
    for sentence in raw_sentences:
        parts = _split_compound(sentence)
        fragments.extend(parts)

    # Step 3: Filter and clean
    claims: List[str] = []
    seen: set[str] = set()

    for fragment in fragments:
        fragment = fragment.strip()

        # Skip too short
        if len(fragment) < MIN_CLAIM_LENGTH:
            continue

        # Skip questions
        if fragment.endswith("?"):
            continue

        lower = fragment.lower()

        # Skip opinions
        if any(lower.startswith(marker) for marker in _OPINION_MARKERS):
            continue

        # Skip meta-statements
        if any(lower.startswith(marker) for marker in _META_MARKERS):
            continue

        # Deduplicate
        normalized = lower.rstrip(".")
        if normalized in seen:
            continue
        seen.add(normalized)

        # Ensure ends with period for consistency
        if not fragment.endswith((".","!","?")):
            fragment = fragment + "."

        claims.append(fragment)

    # If nothing survived filtering, return the whole text as one claim
    if not claims and len(text.strip()) >= MIN_CLAIM_LENGTH:
        claims = [text.strip()]

    return claims


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences using punctuation and newlines."""
    # First split on newlines (paragraphs)
    paragraphs = text.strip().split("\n")

    sentences: List[str] = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        # Split on sentence-ending punctuation followed by space or end
        parts = re.split(r'(?<=[.!])\s+', para)
        sentences.extend(p.strip() for p in parts if p.strip())

    return sentences


def _split_compound(sentence: str) -> List[str]:
    """
    Split a compound sentence on conjunctions if the resulting parts
    are long enough to be standalone claims.
    """
    parts = _COMPOUND_SPLITTERS.split(sentence)

    # If we didn't actually split, return original
    if len(parts) <= 1:
        return [sentence]

    # Only keep parts that are substantial enough
    result: List[str] = []
    for part in parts:
        part = part.strip().rstrip(",;:")
        if len(part) >= MIN_CLAIM_LENGTH:
            result.append(part)

    # If splitting didn't produce useful parts, return original
    return result if result else [sentence]
