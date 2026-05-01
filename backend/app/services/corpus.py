"""
Compatibility wrapper for the evidence retrieval system.

This module preserves the original corpus.py API surface while
delegating to the new evidence_store module. Existing imports
like `from app.services.corpus import retrieve_evidence` continue
to work unchanged.
"""

from typing import Optional, List
from dataclasses import dataclass

from app.services.evidence_store import (
    retrieve_evidence as _retrieve,
    EvidenceResult,
    RankedEvidence,
    SourceEntry,
    Source,
    EVIDENCE_STORE,
)


# ---------------------------------------------------------------------------
# Legacy data classes (preserved for backward compatibility)
# ---------------------------------------------------------------------------

@dataclass
class CorpusEntry:
    """Legacy data class — wraps a SourceEntry for backward compatibility."""
    topic: str
    keywords: List[str]
    fact: str
    source: str
    source_count: int = 1
    evidence_strength: float = 0.7


@dataclass
class EvidenceMatch:
    """Legacy result class — wraps a RankedEvidence for backward compatibility."""
    entry: CorpusEntry
    keyword_score: int
    keyword_ratio: float


def _source_entry_to_corpus_entry(se: SourceEntry) -> CorpusEntry:
    """Convert a new-style SourceEntry to a legacy CorpusEntry."""
    return CorpusEntry(
        topic=se.topic,
        keywords=se.keywords,
        fact=se.fact,
        source=se.source_names,
        source_count=se.source_count,
        evidence_strength=se.evidence_strength,
    )


def _ranked_to_legacy(ranked: RankedEvidence) -> EvidenceMatch:
    """Convert a RankedEvidence to a legacy EvidenceMatch."""
    return EvidenceMatch(
        entry=_source_entry_to_corpus_entry(ranked.entry),
        keyword_score=ranked.keyword_hits,
        keyword_ratio=ranked.keyword_ratio,
    )


# ---------------------------------------------------------------------------
# Public API (backward-compatible)
# ---------------------------------------------------------------------------

def retrieve_evidence(claim_text: str) -> Optional[EvidenceMatch]:
    """
    Search for evidence matching a claim.

    This is the backward-compatible wrapper. It calls the new
    evidence_store.retrieve_evidence() and converts the result
    to the legacy EvidenceMatch format.
    """
    result: EvidenceResult = _retrieve(claim_text)

    if result.has_evidence and result.best is not None:
        return _ranked_to_legacy(result.best)
    return None


# Legacy corpus list (for any code that iterates TRUSTED_CORPUS directly)
TRUSTED_CORPUS: List[CorpusEntry] = [
    _source_entry_to_corpus_entry(se) for se in EVIDENCE_STORE
]
