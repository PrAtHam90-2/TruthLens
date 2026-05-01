"""
Semantic retrieval for TruthLens using TF-IDF vectorization.

Provides fuzzy, paraphrase-aware search against the curated evidence store.
Uses scikit-learn's TfidfVectorizer with character + word n-grams to catch
near-miss matches that keyword/alias matching would miss.

The index is built lazily on first use and cached for the process lifetime.
"""

from dataclasses import dataclass
from typing import List, Optional
import logging

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.services.evidence_store import EVIDENCE_STORE, SourceEntry
from app.core.config import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SemanticMatch:
    """A corpus entry matched via semantic (TF-IDF) similarity."""
    entry: SourceEntry
    similarity_score: float
    keyword_hits: int = 0
    keyword_ratio: float = 0.0
    is_ambiguous: bool = False      # True if top matches span different topics with close scores
    is_low_confidence: bool = False  # True if top score is below the confidence floor


# ---------------------------------------------------------------------------
# Semantic index
# ---------------------------------------------------------------------------

class SemanticIndex:
    """
    TF-IDF index over the curated evidence store.

    Each corpus entry is represented as a combined document:
        "{fact} {keywords} {aliases} {topic} {synonyms}"

    Synonyms are **topic-scoped** — stored per-entry to avoid cross-topic
    semantic leakage (e.g., "radiation" matching both 5G and medical topics).

    Two vectorizers are used and their scores averaged:
    1. Word n-grams (1,2) — catches phrase-level overlap
    2. Character n-grams (3,5) — catches substring/morphological overlap
       (e.g., "immunization" ↔ "vaccination")
    """

    # Score gap required between top-2 matches of different topics.
    # If the gap is smaller, the match is considered ambiguous.
    _AMBIGUITY_GAP = 0.04
    _LOW_CONFIDENCE_FLOOR = 0.15  # Below this, matches are noisy — skip ambiguity, flag low-confidence

    def __init__(self, entries: Optional[List[SourceEntry]] = None):
        self._entries = entries or list(EVIDENCE_STORE)
        self._documents: List[str] = []
        self._word_vectorizer: Optional[TfidfVectorizer] = None
        self._char_vectorizer: Optional[TfidfVectorizer] = None
        self._word_matrix = None
        self._char_matrix = None
        self._build_index()

    def _entry_to_document(self, entry: SourceEntry) -> str:
        """Convert a corpus entry to a searchable document string.

        Uses topic-scoped synonyms from the entry itself, NOT a global map.
        """
        parts = [
            entry.fact,
            " ".join(entry.keywords),
            " ".join(entry.aliases),
            entry.topic,
            " ".join(entry.synonyms),   # topic-scoped synonyms
        ]
        return " ".join(parts).lower()

    def _build_index(self):
        """Build TF-IDF matrices from all corpus entries."""
        self._documents = [self._entry_to_document(e) for e in self._entries]

        if not self._documents:
            logger.warning("SemanticIndex: no documents to index")
            return

        # Word n-grams (1,2) — phrase-level matching
        self._word_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            stop_words="english",
            max_features=5000,
        )
        self._word_matrix = self._word_vectorizer.fit_transform(self._documents)

        # Character n-grams (3,5) — morphological/substring matching
        self._char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=5000,
        )
        self._char_matrix = self._char_vectorizer.fit_transform(self._documents)

        logger.info(
            f"SemanticIndex built: {len(self._entries)} entries, "
            f"word_features={self._word_matrix.shape[1]}, "
            f"char_features={self._char_matrix.shape[1]}"
        )

    def _detect_ambiguity(self, results: List[SemanticMatch]) -> bool:
        """Check if top-2 matches span different topics with close scores.

        Returns True if the match is ambiguous and should be downgraded.
        """
        if len(results) < 2:
            return False

        top = results[0]
        runner_up = results[1]

        # Same topic → not ambiguous
        if top.entry.topic == runner_up.entry.topic:
            return False

        # Different topics — check score gap
        gap = top.similarity_score - runner_up.similarity_score
        if gap < self._AMBIGUITY_GAP:
            logger.info(
                f"Ambiguous semantic match: '{top.entry.topic}' ({top.similarity_score}) "
                f"vs '{runner_up.entry.topic}' ({runner_up.similarity_score}), "
                f"gap={gap:.4f} < {self._AMBIGUITY_GAP}"
            )
            return True

        return False

    def search(
        self,
        claim: str,
        top_k: int = 3,
        min_similarity: float = 0.15,
    ) -> List[SemanticMatch]:
        """
        Find corpus entries semantically similar to the claim.

        Args:
            claim: The claim text to search for.
            top_k: Maximum number of results to return.
            min_similarity: Minimum cosine similarity threshold (0.0–1.0).

        Returns:
            List of SemanticMatch objects, ranked by similarity (descending).
            The top match's ``is_ambiguous`` flag is set if the top-2 results
            span different topics with close scores.
        """
        if self._word_matrix is None or self._char_matrix is None:
            return []

        claim_lower = claim.lower()

        # Vectorize the claim
        word_vec = self._word_vectorizer.transform([claim_lower])
        char_vec = self._char_vectorizer.transform([claim_lower])

        # Compute cosine similarity for both representations
        word_sims = cosine_similarity(word_vec, self._word_matrix).flatten()
        char_sims = cosine_similarity(char_vec, self._char_matrix).flatten()

        # Combined score: weighted average (word n-grams weighted higher)
        combined_sims = (word_sims * 0.6) + (char_sims * 0.4)

        # Count keyword hits for grounding compatibility
        results: List[SemanticMatch] = []
        for idx in range(len(self._entries)):
            score = float(combined_sims[idx])
            if score >= min_similarity:
                entry = self._entries[idx]

                # Count actual keyword overlap for grounding
                keyword_hits = sum(1 for kw in entry.keywords if kw in claim_lower)
                total_kw = len(entry.keywords)
                keyword_ratio = keyword_hits / total_kw if total_kw > 0 else 0.0

                results.append(SemanticMatch(
                    entry=entry,
                    similarity_score=round(score, 4),
                    keyword_hits=keyword_hits,
                    keyword_ratio=keyword_ratio,
                ))

        # Sort by similarity (descending)
        results.sort(key=lambda m: m.similarity_score, reverse=True)
        results = results[:top_k]

        if results:
            top_score = results[0].similarity_score
            if top_score < self._LOW_CONFIDENCE_FLOOR:
                # Score is too low for confident matching — flag, don't check ambiguity
                results[0].is_low_confidence = True
                logger.info(
                    f"Low-confidence semantic match: '{results[0].entry.topic}' "
                    f"(score={top_score} < floor={self._LOW_CONFIDENCE_FLOOR})"
                )
            else:
                # Only check ambiguity for matches above the confidence floor
                if self._detect_ambiguity(results):
                    results[0].is_ambiguous = True

        return results

    @property
    def entry_count(self) -> int:
        """Number of entries in the index."""
        return len(self._entries)


# ---------------------------------------------------------------------------
# Module-level singleton (built lazily)
# ---------------------------------------------------------------------------

_semantic_index: Optional[SemanticIndex] = None


def get_semantic_index() -> SemanticIndex:
    """Return the cached semantic index, building it on first call."""
    global _semantic_index
    if _semantic_index is None:
        _semantic_index = SemanticIndex()
    return _semantic_index


def _reset_semantic_index():
    """Reset the singleton index (used in tests)."""
    global _semantic_index
    _semantic_index = None


def semantic_search(
    claim: str,
    top_k: int = 3,
    min_similarity: Optional[float] = None,
) -> List[SemanticMatch]:
    """
    Search the corpus using TF-IDF semantic similarity.

    Public API function. Uses the configured min_similarity from settings
    unless overridden.

    Args:
        claim: The claim text to search for.
        top_k: Maximum results.
        min_similarity: Override the config threshold.

    Returns:
        List of SemanticMatch objects, ranked by similarity.
    """
    settings = get_settings()
    if min_similarity is None:
        min_similarity = settings.semantic_min_similarity

    index = get_semantic_index()
    return index.search(claim, top_k=top_k, min_similarity=min_similarity)
