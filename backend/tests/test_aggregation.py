"""
Tests for verdict aggregation logic.

Verifies that the overall verdict correctly reflects the WEAKEST claim,
not the strongest, per the conservative aggregation rules.
"""

import pytest
from app.services.analyzer import _compute_overall_verdict
from app.models.schemas import (
    ClaimResult,
    ClaimStatus,
    ClaimType,
    EvidenceSource,
)


def _make_claim(
    status: ClaimStatus,
    confidence: float = 0.70,
    claim_type: ClaimType = ClaimType.FACTUAL,
    evidence_source: EvidenceSource = EvidenceSource.CORPUS,
) -> ClaimResult:
    """Helper to create a ClaimResult with the given status."""
    return ClaimResult(
        claim=f"Test claim ({status.value})",
        claim_type=claim_type,
        status=status,
        evidence="Test evidence.",
        evidence_source=evidence_source,
        confidence=confidence,
    )


# ===================================================================
# Single claim verdicts
# ===================================================================

class TestSingleClaimVerdict:
    """With a single factual claim, the verdict should match the claim."""

    def test_single_supported(self):
        verdict, conf, _ = _compute_overall_verdict([
            _make_claim(ClaimStatus.SUPPORTED),
        ])
        assert verdict == ClaimStatus.SUPPORTED

    def test_single_contradicted(self):
        verdict, _, _ = _compute_overall_verdict([
            _make_claim(ClaimStatus.CONTRADICTED),
        ])
        assert verdict == ClaimStatus.CONTRADICTED

    def test_single_unknown(self):
        verdict, _, _ = _compute_overall_verdict([
            _make_claim(ClaimStatus.UNKNOWN, evidence_source=EvidenceSource.NONE),
        ])
        assert verdict == ClaimStatus.UNKNOWN

    def test_single_mixed(self):
        verdict, _, _ = _compute_overall_verdict([
            _make_claim(ClaimStatus.MIXED),
        ])
        assert verdict == ClaimStatus.MIXED


# ===================================================================
# Multi-claim: Supported + Unknown → MIXED (not Supported!)
# ===================================================================

class TestSupportedPlusUnknown:
    """The core bug fix: Supported + Unknown must NOT produce Supported."""

    def test_supported_plus_unknown_is_mixed(self):
        verdict, _, explanation = _compute_overall_verdict([
            _make_claim(ClaimStatus.SUPPORTED),
            _make_claim(ClaimStatus.UNKNOWN, evidence_source=EvidenceSource.LLM_ONLY),
        ])
        assert verdict == ClaimStatus.MIXED
        assert "1 of 2" in explanation

    def test_supported_plus_unknown_never_supported(self):
        """Even with 3 Supported and 1 Unknown, verdict must not be Supported."""
        verdict, _, _ = _compute_overall_verdict([
            _make_claim(ClaimStatus.SUPPORTED),
            _make_claim(ClaimStatus.SUPPORTED),
            _make_claim(ClaimStatus.SUPPORTED),
            _make_claim(ClaimStatus.UNKNOWN, evidence_source=EvidenceSource.LLM_ONLY),
        ])
        assert verdict == ClaimStatus.MIXED
        assert verdict != ClaimStatus.SUPPORTED

    def test_supported_plus_unknown_confidence_penalized(self):
        """Confidence should be lower when some claims lack evidence."""
        _, conf_all_supported, _ = _compute_overall_verdict([
            _make_claim(ClaimStatus.SUPPORTED, confidence=0.85),
            _make_claim(ClaimStatus.SUPPORTED, confidence=0.80),
        ])
        _, conf_mixed, _ = _compute_overall_verdict([
            _make_claim(ClaimStatus.SUPPORTED, confidence=0.85),
            _make_claim(ClaimStatus.UNKNOWN, confidence=0.45),
        ])
        assert conf_mixed < conf_all_supported


# ===================================================================
# Multi-claim: Contradicted scenarios
# ===================================================================

class TestContradictedScenarios:
    """Any Contradicted claim should influence the overall verdict."""

    def test_all_contradicted(self):
        verdict, _, _ = _compute_overall_verdict([
            _make_claim(ClaimStatus.CONTRADICTED),
            _make_claim(ClaimStatus.CONTRADICTED),
        ])
        assert verdict == ClaimStatus.CONTRADICTED

    def test_supported_plus_contradicted_is_mixed(self):
        verdict, _, _ = _compute_overall_verdict([
            _make_claim(ClaimStatus.SUPPORTED),
            _make_claim(ClaimStatus.CONTRADICTED),
        ])
        assert verdict == ClaimStatus.MIXED

    def test_contradicted_plus_unknown(self):
        verdict, _, _ = _compute_overall_verdict([
            _make_claim(ClaimStatus.CONTRADICTED),
            _make_claim(ClaimStatus.UNKNOWN),
        ])
        assert verdict == ClaimStatus.CONTRADICTED

    def test_supported_contradicted_unknown(self):
        verdict, _, _ = _compute_overall_verdict([
            _make_claim(ClaimStatus.SUPPORTED),
            _make_claim(ClaimStatus.CONTRADICTED),
            _make_claim(ClaimStatus.UNKNOWN),
        ])
        assert verdict == ClaimStatus.MIXED


# ===================================================================
# All Supported → only case where SUPPORTED is returned
# ===================================================================

class TestAllSupported:
    """Supported verdict requires ALL factual claims to be Supported."""

    def test_all_supported(self):
        verdict, _, _ = _compute_overall_verdict([
            _make_claim(ClaimStatus.SUPPORTED),
            _make_claim(ClaimStatus.SUPPORTED),
            _make_claim(ClaimStatus.SUPPORTED),
        ])
        assert verdict == ClaimStatus.SUPPORTED

    def test_all_supported_explanation(self):
        _, _, explanation = _compute_overall_verdict([
            _make_claim(ClaimStatus.SUPPORTED),
        ])
        assert "supported" in explanation.lower()


# ===================================================================
# All Unknown
# ===================================================================

class TestAllUnknown:
    """All Unknown claims → overall Unknown."""

    def test_all_unknown(self):
        verdict, _, _ = _compute_overall_verdict([
            _make_claim(ClaimStatus.UNKNOWN),
            _make_claim(ClaimStatus.UNKNOWN),
        ])
        assert verdict == ClaimStatus.UNKNOWN


# ===================================================================
# Non-factual claims excluded from verdict
# ===================================================================

class TestNonFactualExclusion:
    """Opinion/Unverifiable claims should not affect the verdict."""

    def test_opinion_excluded(self):
        verdict, _, explanation = _compute_overall_verdict([
            _make_claim(ClaimStatus.SUPPORTED),
            _make_claim(
                ClaimStatus.UNVERIFIABLE,
                claim_type=ClaimType.OPINION,
                evidence_source=EvidenceSource.NONE,
            ),
        ])
        # Only 1 factual claim (Supported), so overall = Supported
        assert verdict == ClaimStatus.SUPPORTED
        assert "opinion/unverifiable" in explanation.lower()

    def test_all_opinion_is_unverifiable(self):
        verdict, _, _ = _compute_overall_verdict([
            _make_claim(
                ClaimStatus.UNVERIFIABLE,
                claim_type=ClaimType.OPINION,
                evidence_source=EvidenceSource.NONE,
                confidence=0.55,
            ),
        ])
        assert verdict == ClaimStatus.UNVERIFIABLE


# ===================================================================
# Empty claims
# ===================================================================

class TestEmptyClaims:
    """No claims at all → Unknown."""

    def test_no_claims(self):
        verdict, conf, _ = _compute_overall_verdict([])
        assert verdict == ClaimStatus.UNKNOWN
        assert conf == 0.0


# ===================================================================
# Supported + Unverifiable (factual claim typed as unverifiable)
# ===================================================================

class TestSupportedPlusUnverifiable:
    """Supported + factual-Unverifiable → MIXED."""

    def test_supported_plus_factual_unverifiable_is_mixed(self):
        verdict, _, _ = _compute_overall_verdict([
            _make_claim(ClaimStatus.SUPPORTED),
            _make_claim(ClaimStatus.UNVERIFIABLE),  # Still ClaimType.FACTUAL
        ])
        assert verdict == ClaimStatus.MIXED


# ===================================================================
# Explanation text quality
# ===================================================================

class TestExplanationQuality:
    """Explanations should be descriptive and accurate."""

    def test_mixed_explanation_shows_counts(self):
        _, _, explanation = _compute_overall_verdict([
            _make_claim(ClaimStatus.SUPPORTED),
            _make_claim(ClaimStatus.UNKNOWN),
        ])
        assert "1 of 2" in explanation

    def test_contradicted_explanation_shows_count(self):
        _, _, explanation = _compute_overall_verdict([
            _make_claim(ClaimStatus.CONTRADICTED),
            _make_claim(ClaimStatus.UNKNOWN),
        ])
        assert "1 of 2" in explanation or "contradicted" in explanation.lower()

    def test_all_supported_explanation_mentions_all(self):
        _, _, explanation = _compute_overall_verdict([
            _make_claim(ClaimStatus.SUPPORTED),
            _make_claim(ClaimStatus.SUPPORTED),
        ])
        assert "all" in explanation.lower()
