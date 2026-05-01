"""
Tests for evidence grounding — ensuring that verdicts are never
issued without proportional trusted evidence backing.

These tests validate the core safety invariant:
  "No trusted evidence → no Supported/Contradicted verdict."
"""

import pytest
from app.services.analyzer import (
    _classify_evidence_source,
    _apply_evidence_grounding,
    _build_evidence_display,
)
from app.models.schemas import EvidenceSource, ClaimResult, ClaimStatus, ClaimType
from app.services.evidence_store import (
    retrieve_evidence,
    SourceEntry,
    Source,
    RankedEvidence,
)


# ===================================================================
# Evidence source classification
# ===================================================================

class TestClassifyEvidenceSource:
    """Test that evidence sources are correctly classified as corpus/weak/llm_only."""

    def test_no_match_returns_llm_only(self):
        result = _classify_evidence_source(None)
        assert result == EvidenceSource.LLM_ONLY

    def test_strong_match_returns_corpus(self):
        entry = SourceEntry(
            topic="test",
            category="science",
            keywords=["earth", "flat"],
            aliases=[],
            fact="The Earth is round.",
            sources=[Source("NASA", "https://nasa.gov", 0.95)],
            evidence_strength=0.90,
        )
        match = RankedEvidence(
            entry=entry,
            relevance_score=3.0,
            keyword_hits=2,
            alias_hits=0,
            keyword_ratio=0.5,  # >= 0.25 threshold
        )
        result = _classify_evidence_source(match)
        assert result == EvidenceSource.CORPUS

    def test_weak_match_low_strength(self):
        entry = SourceEntry(
            topic="test",
            category="science",
            keywords=["earth", "flat", "round", "globe"],
            aliases=[],
            fact="Something about earth.",
            sources=[Source("Test", "", 0.5)],
            evidence_strength=0.40,  # Below 0.7 threshold
        )
        match = RankedEvidence(
            entry=entry,
            relevance_score=1.0,
            keyword_hits=1,
            alias_hits=0,
            keyword_ratio=0.5,
        )
        result = _classify_evidence_source(match)
        assert result == EvidenceSource.WEAK_CORPUS

    def test_weak_match_low_keyword_ratio(self):
        entry = SourceEntry(
            topic="test",
            category="science",
            keywords=["a", "b", "c", "d", "e", "f", "g", "h"],
            aliases=[],
            fact="Some fact.",
            sources=[Source("Test", "", 0.9)],
            evidence_strength=0.90,  # Strong evidence...
        )
        match = RankedEvidence(
            entry=entry,
            relevance_score=1.0,
            keyword_hits=1,
            alias_hits=0,
            keyword_ratio=0.12,  # ...but terrible keyword match (< 0.25)
        )
        result = _classify_evidence_source(match)
        assert result == EvidenceSource.WEAK_CORPUS


# ===================================================================
# Evidence grounding rules
# ===================================================================

class TestEvidenceGrounding:
    """Test that ungrounded verdicts are properly downgraded."""

    # --- CORPUS: LLM verdict trusted ---

    def test_corpus_supported_stays_supported(self):
        status, conf, note = _apply_evidence_grounding(
            "Supported", 0.85, EvidenceSource.CORPUS, ""
        )
        assert status == "Supported"
        assert conf == 0.85

    def test_corpus_contradicted_stays_contradicted(self):
        status, conf, note = _apply_evidence_grounding(
            "Contradicted", 0.82, EvidenceSource.CORPUS, ""
        )
        assert status == "Contradicted"
        assert conf == 0.82

    # --- WEAK_CORPUS: Supported downgraded ---

    def test_weak_corpus_supported_downgraded_to_unknown(self):
        status, conf, note = _apply_evidence_grounding(
            "Supported", 0.85, EvidenceSource.WEAK_CORPUS, ""
        )
        assert status == "Unknown", f"Expected Unknown, got {status}"
        assert conf <= 0.55
        assert "Downgraded" in note

    def test_weak_corpus_contradicted_stays_but_capped(self):
        status, conf, note = _apply_evidence_grounding(
            "Contradicted", 0.90, EvidenceSource.WEAK_CORPUS, ""
        )
        assert status == "Contradicted"  # Keeps contradicted (evidence exists, just weak)
        assert conf <= 0.60

    def test_weak_corpus_unknown_stays_unknown(self):
        status, conf, note = _apply_evidence_grounding(
            "Unknown", 0.50, EvidenceSource.WEAK_CORPUS, ""
        )
        assert status == "Unknown"

    # --- LLM_ONLY: Everything definitive is downgraded ---

    def test_llm_only_supported_downgraded_to_unknown(self):
        """THE CORE BUG: LLM says Supported but there's no evidence → must be Unknown."""
        status, conf, note = _apply_evidence_grounding(
            "Supported", 0.88, EvidenceSource.LLM_ONLY, ""
        )
        assert status == "Unknown", f"CRITICAL: LLM-only Supported was NOT downgraded!"
        assert conf <= 0.50
        assert "Downgraded" in note
        assert "not sufficient" in note.lower()

    def test_llm_only_contradicted_downgraded_to_unknown(self):
        status, conf, note = _apply_evidence_grounding(
            "Contradicted", 0.82, EvidenceSource.LLM_ONLY, ""
        )
        assert status == "Unknown", f"CRITICAL: LLM-only Contradicted was NOT downgraded!"
        assert conf <= 0.50
        assert "Downgraded" in note

    def test_llm_only_unknown_stays_unknown(self):
        status, conf, note = _apply_evidence_grounding(
            "Unknown", 0.55, EvidenceSource.LLM_ONLY, ""
        )
        assert status == "Unknown"
        assert conf <= 0.50

    def test_llm_only_mixed_stays_but_capped(self):
        status, conf, note = _apply_evidence_grounding(
            "Mixed", 0.70, EvidenceSource.LLM_ONLY, ""
        )
        assert status == "Mixed"  # Mixed is already uncertain, keep it
        assert conf <= 0.50


# ===================================================================
# Evidence display labeling
# ===================================================================

class TestEvidenceDisplay:
    """Test that evidence strings clearly label their provenance."""

    def test_corpus_evidence_shows_source(self):
        display = _build_evidence_display(
            EvidenceSource.CORPUS,
            "The Earth is round.",
            "NASA, ESA",
            "https://nasa.gov",
            "LLM explanation",
        )
        assert "NASA" in display
        assert "https://nasa.gov" in display
        assert "[No trusted evidence]" not in display

    def test_weak_corpus_labeled_partial(self):
        display = _build_evidence_display(
            EvidenceSource.WEAK_CORPUS,
            "Some fact.",
            "Test Source",
            "",
            "LLM explanation",
        )
        assert "[Partial match]" in display

    def test_llm_only_labeled_no_trusted_evidence(self):
        display = _build_evidence_display(
            EvidenceSource.LLM_ONLY,
            None,
            None,
            "",
            "Based on general knowledge, this seems true.",
        )
        assert "[No trusted evidence]" in display
        assert "general knowledge" in display

    def test_none_labeled_no_trusted_evidence(self):
        display = _build_evidence_display(
            EvidenceSource.NONE,
            None,
            None,
            "",
            "Unable to assess.",
        )
        assert "[No trusted evidence]" in display


# ===================================================================
# End-to-end grounding scenarios
# ===================================================================

class TestGroundingScenarios:
    """Integration-style tests for realistic claim scenarios."""

    def test_known_claim_gets_corpus_evidence(self):
        """A well-known misinformation claim should get corpus evidence."""
        result = retrieve_evidence("The Earth is flat.")
        assert result.has_evidence
        source = _classify_evidence_source(result.best)
        assert source == EvidenceSource.CORPUS

    def test_unknown_claim_gets_no_corpus(self):
        """An obscure claim should NOT match any corpus entry."""
        result = retrieve_evidence("The population of Tuvalu doubled in 2024.")
        source = _classify_evidence_source(result.best)
        assert source == EvidenceSource.LLM_ONLY

    def test_unknown_claim_cannot_be_supported(self):
        """Even if LLM says Supported, no corpus = downgrade to Unknown."""
        result = retrieve_evidence("The population of Tuvalu doubled in 2024.")
        source = _classify_evidence_source(result.best)
        status, conf, note = _apply_evidence_grounding(
            "Supported", 0.88, source, ""
        )
        assert status == "Unknown"
        assert conf <= 0.50

    def test_known_claim_can_be_contradicted(self):
        """A known false claim with strong corpus evidence CAN be Contradicted."""
        result = retrieve_evidence("The Earth is flat.")
        source = _classify_evidence_source(result.best)
        status, conf, note = _apply_evidence_grounding(
            "Contradicted", 0.85, source, ""
        )
        assert status == "Contradicted"
        assert conf == 0.85

    def test_claim_result_has_evidence_source_field(self):
        """Verify the schema includes evidence_source."""
        result = ClaimResult(
            claim="Test claim.",
            status=ClaimStatus.UNKNOWN,
            evidence="No evidence.",
            evidence_source=EvidenceSource.LLM_ONLY,
            confidence=0.45,
        )
        data = result.model_dump()
        assert "evidence_source" in data
        assert data["evidence_source"] == "llm_only"

    def test_claim_result_defaults_to_none(self):
        """Evidence source should default to 'none' if not specified."""
        result = ClaimResult(
            claim="Test.",
            status=ClaimStatus.UNKNOWN,
            evidence="Nothing.",
            confidence=0.30,
        )
        assert result.evidence_source == EvidenceSource.NONE
