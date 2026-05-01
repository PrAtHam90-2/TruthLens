"""
Evidence fusion and ranking.

Collects evidence from multiple retrieval sources (corpus, semantic, dynamic),
deduplicates, ranks by trust × relevance, detects agreement/conflict between
sources, and produces a FusionResult with a confidence adjustment.

Key design decisions
--------------------
- **source_diversity** (unique source *types*) drives confidence scaling, not
  raw source_count — prevents duplicate inflation.
- Corpus + semantic matches of the **same underlying fact** (same topic) are
  treated as a single source to avoid fake multi-source agreement.
- Confidence adjustments are conservative: +0.05–0.10 for agreement,
  −0.10–0.15 for conflict.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from app.models.schemas import EvidenceSource
from app.services.evidence_store import RankedEvidence, SourceEntry
from app.services.semantic_index import SemanticMatch
from app.services.dynamic_retrieval import DynamicEvidence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Trust levels per source type
# ---------------------------------------------------------------------------
_TRUST_LEVELS = {
    EvidenceSource.CORPUS: 0.95,
    EvidenceSource.SEMANTIC: 0.80,
    EvidenceSource.DYNAMIC: 0.65,
    EvidenceSource.WEAK_CORPUS: 0.45,
    EvidenceSource.LLM_ONLY: 0.25,
    EvidenceSource.NONE: 0.0,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EvidenceItem:
    """One piece of evidence from any source."""
    text: str
    source_name: str
    source_url: str
    source_type: EvidenceSource
    relevance_score: float          # 0.0–1.0 (normalised)
    trust_level: float              # 0.0–1.0 (from source tier)
    topic: str = ""                 # For deduplication (corpus/semantic topic)
    role: str = "neutral"           # "supporting", "conflicting", "neutral"


@dataclass
class FusionResult:
    """Output of the evidence fusion process."""
    items: List[EvidenceItem]              # Deduplicated, ranked evidence
    source_count: int = 0                  # Total unique sources
    source_diversity: int = 0              # Number of distinct source *types*
    agreement_signal: str = "insufficient" # "supporting" | "conflicting" | "mixed" | "insufficient"
    confidence_adjustment: float = 0.0     # −0.15 to +0.10 modifier
    fusion_summary: str = ""               # Human-readable explanation


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def collect_evidence(
    corpus_match: Optional[RankedEvidence],
    corpus_source: EvidenceSource,
    semantic_matches: Optional[List[SemanticMatch]],
    dynamic_evidence: Optional[DynamicEvidence],
) -> List[EvidenceItem]:
    """Gather EvidenceItems from all retrieval sources."""
    items: List[EvidenceItem] = []

    # Corpus match
    if corpus_match and corpus_source in (EvidenceSource.CORPUS, EvidenceSource.WEAK_CORPUS):
        entry = corpus_match.entry
        items.append(EvidenceItem(
            text=entry.fact,
            source_name=entry.source_names,
            source_url=entry.best_url,
            source_type=corpus_source,
            relevance_score=min(corpus_match.relevance_score / 5.0, 1.0),  # normalise
            trust_level=_TRUST_LEVELS[corpus_source],
            topic=entry.topic,
        ))

    # Semantic matches (up to 3)
    if semantic_matches:
        for sm in semantic_matches[:3]:
            if sm.is_low_confidence:
                continue
            items.append(EvidenceItem(
                text=sm.entry.fact,
                source_name=sm.entry.source_names,
                source_url=sm.entry.best_url,
                source_type=EvidenceSource.SEMANTIC,
                relevance_score=min(sm.similarity_score / 0.5, 1.0),  # normalise
                trust_level=_TRUST_LEVELS[EvidenceSource.SEMANTIC],
                topic=sm.entry.topic,
            ))

    # Dynamic evidence (Wikipedia)
    if dynamic_evidence and dynamic_evidence.relevance_score >= 0.20:
        items.append(EvidenceItem(
            text=dynamic_evidence.snippet,
            source_name=dynamic_evidence.source_name,
            source_url=dynamic_evidence.url,
            source_type=EvidenceSource.DYNAMIC,
            relevance_score=min(dynamic_evidence.relevance_score, 1.0),
            trust_level=_TRUST_LEVELS[EvidenceSource.DYNAMIC],
            topic=dynamic_evidence.title or "",
        ))

    return items


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _word_set(text: str) -> set:
    """Lower-case word set for overlap comparison."""
    return set(text.lower().split())


def deduplicate(items: List[EvidenceItem]) -> List[EvidenceItem]:
    """Remove near-duplicate evidence.

    Rules:
    - Corpus + semantic matches with the SAME topic → keep corpus only
      (they reference the same underlying fact; treating them as two
      sources would inflate agreement).
    - Two items with >80% word overlap → keep the higher-trust item.
    """
    if len(items) <= 1:
        return items

    # Phase 1: merge corpus + semantic with same topic
    seen_topics: dict[str, int] = {}   # topic → index of best item
    deduped: List[EvidenceItem] = []

    for item in items:
        key = item.topic.strip().lower() if item.topic else None

        if key and item.source_type in (EvidenceSource.CORPUS, EvidenceSource.SEMANTIC,
                                         EvidenceSource.WEAK_CORPUS):
            if key in seen_topics:
                existing_idx = seen_topics[key]
                existing = deduped[existing_idx]
                # Keep the higher-trust item
                if item.trust_level > existing.trust_level:
                    deduped[existing_idx] = item
                continue
            seen_topics[key] = len(deduped)

        deduped.append(item)

    # Phase 2: word-overlap deduplication (for dynamic vs corpus overlaps)
    final: List[EvidenceItem] = []
    for item in deduped:
        ws = _word_set(item.text)
        is_dup = False
        for existing in final:
            ews = _word_set(existing.text)
            if not ws or not ews:
                continue
            overlap = len(ws & ews) / min(len(ws), len(ews))
            if overlap > 0.80:
                # Keep higher trust
                if item.trust_level > existing.trust_level:
                    final[final.index(existing)] = item
                is_dup = True
                break
        if not is_dup:
            final.append(item)

    return final


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def rank_evidence(items: List[EvidenceItem]) -> List[EvidenceItem]:
    """Sort by composite score: trust_level * 0.5 + relevance_score * 0.5."""
    return sorted(
        items,
        key=lambda it: (it.trust_level * 0.5 + it.relevance_score * 0.5),
        reverse=True,
    )


# ---------------------------------------------------------------------------
# Agreement detection
# ---------------------------------------------------------------------------

def detect_agreement(items: List[EvidenceItem]) -> tuple[str, float]:
    """Determine whether evidence items agree, conflict, or are mixed.

    Uses a simple heuristic: items from the same corpus topic are
    "agreeing" (they reference the same fact).  Items from different
    topics or different source types are compared by whether they
    share thematic overlap (>40% word overlap → agreeing).

    Returns (signal, confidence_adjustment).
    """
    if len(items) == 0:
        return "insufficient", 0.0

    if len(items) == 1:
        return "insufficient", 0.0

    # Count unique source *types* (the user's diversity metric)
    source_types = {it.source_type for it in items}
    diversity = len(source_types)

    # Check pairwise agreement using word overlap
    agree_pairs = 0
    conflict_pairs = 0
    total_pairs = 0

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            total_pairs += 1
            ws_i = _word_set(items[i].text)
            ws_j = _word_set(items[j].text)
            if not ws_i or not ws_j:
                continue
            overlap = len(ws_i & ws_j) / min(len(ws_i), len(ws_j))
            if overlap > 0.30:
                agree_pairs += 1
            elif overlap < 0.10:
                conflict_pairs += 1

    if total_pairs == 0:
        return "insufficient", 0.0

    agree_ratio = agree_pairs / total_pairs
    conflict_ratio = conflict_pairs / total_pairs

    # Scale confidence adjustment by diversity (not raw count)
    diversity_factor = min(diversity / 3.0, 1.0)  # caps at 3 source types

    if agree_ratio >= 0.6 and conflict_ratio < 0.2:
        adjustment = 0.05 + (0.05 * diversity_factor)  # +0.05 to +0.10
        return "supporting", round(adjustment, 3)

    if conflict_ratio >= 0.5:
        adjustment = -0.10 - (0.05 * diversity_factor)  # -0.10 to -0.15
        return "conflicting", round(adjustment, 3)

    if agree_ratio > 0 and conflict_ratio > 0:
        adjustment = -0.03 - (0.02 * diversity_factor)  # -0.03 to -0.05
        return "mixed", round(adjustment, 3)

    return "insufficient", 0.0


# ---------------------------------------------------------------------------
# Role assignment
# ---------------------------------------------------------------------------

def _assign_roles(items: List[EvidenceItem]) -> None:
    """Assign roles to evidence items based on agreement with the top item."""
    if not items:
        return

    top_ws = _word_set(items[0].text)
    items[0].role = "supporting"  # top item is the reference

    for item in items[1:]:
        ws = _word_set(item.text)
        if not ws or not top_ws:
            item.role = "neutral"
            continue
        overlap = len(ws & top_ws) / min(len(ws), len(top_ws))
        if overlap > 0.30:
            item.role = "supporting"
        elif overlap < 0.10:
            item.role = "conflicting"
        else:
            item.role = "neutral"


# ---------------------------------------------------------------------------
# Main fusion function
# ---------------------------------------------------------------------------

def fuse_evidence(
    corpus_match: Optional[RankedEvidence],
    corpus_source: EvidenceSource,
    semantic_matches: Optional[List[SemanticMatch]],
    dynamic_evidence: Optional[DynamicEvidence],
) -> FusionResult:
    """Orchestrate evidence fusion: collect → dedup → rank → detect → result."""

    # 1. Collect
    raw_items = collect_evidence(corpus_match, corpus_source, semantic_matches, dynamic_evidence)

    if not raw_items:
        return FusionResult(
            items=[],
            source_count=0,
            source_diversity=0,
            agreement_signal="insufficient",
            confidence_adjustment=0.0,
            fusion_summary="No evidence collected from any source.",
        )

    # 2. Deduplicate
    deduped = deduplicate(raw_items)

    # 3. Rank
    ranked = rank_evidence(deduped)

    # 4. Assign roles
    _assign_roles(ranked)

    # 5. Detect agreement
    signal, adjustment = detect_agreement(ranked)

    # 6. Compute counts
    source_types = {it.source_type for it in ranked}
    unique_sources = set()
    for it in ranked:
        unique_sources.add(f"{it.source_type.value}:{it.source_name}")
    source_count = len(unique_sources)
    source_diversity = len(source_types)

    # 7. Build summary
    summary_parts = [f"{source_count} source(s) from {source_diversity} type(s)."]
    if signal == "supporting":
        summary_parts.append("Multiple sources agree — confidence boosted.")
    elif signal == "conflicting":
        summary_parts.append("Sources conflict — confidence reduced.")
    elif signal == "mixed":
        summary_parts.append("Mixed signals from sources — confidence slightly reduced.")
    else:
        summary_parts.append("Insufficient evidence for cross-source comparison.")

    return FusionResult(
        items=ranked[:5],  # cap at 5
        source_count=source_count,
        source_diversity=source_diversity,
        agreement_signal=signal,
        confidence_adjustment=adjustment,
        fusion_summary=" ".join(summary_parts),
    )
