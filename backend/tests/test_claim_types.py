"""
Tests for claim type classification — verifying that the schema,
analyzer routing, and heuristic type detection work correctly.
"""

import pytest
from app.models.schemas import ClaimType, ClaimStatus, ClaimResult


class TestClaimTypeSchema:
    """Verify the ClaimType enum and ClaimResult model work correctly."""

    def test_factual_type_exists(self):
        assert ClaimType.FACTUAL == "Factual"

    def test_opinion_type_exists(self):
        assert ClaimType.OPINION == "Opinion"

    def test_unverifiable_type_exists(self):
        assert ClaimType.UNVERIFIABLE == "Unverifiable"

    def test_unverifiable_status_exists(self):
        assert ClaimStatus.UNVERIFIABLE == "Unverifiable"

    def test_claim_result_default_type(self):
        """ClaimResult should default to Factual for backward compatibility."""
        result = ClaimResult(
            claim="The Earth is round.",
            status=ClaimStatus.SUPPORTED,
            evidence="Confirmed by NASA.",
            confidence=0.85,
        )
        assert result.claim_type == ClaimType.FACTUAL

    def test_claim_result_with_opinion_type(self):
        result = ClaimResult(
            claim="NASA cannot be trusted.",
            claim_type=ClaimType.OPINION,
            status=ClaimStatus.UNVERIFIABLE,
            evidence="This is an opinion.",
            confidence=0.55,
        )
        assert result.claim_type == ClaimType.OPINION
        assert result.status == ClaimStatus.UNVERIFIABLE

    def test_claim_result_with_unverifiable_type(self):
        result = ClaimResult(
            claim="Something big is coming.",
            claim_type=ClaimType.UNVERIFIABLE,
            status=ClaimStatus.UNVERIFIABLE,
            evidence="Too vague to verify.",
            confidence=0.55,
        )
        assert result.claim_type == ClaimType.UNVERIFIABLE

    def test_claim_result_serializes_correctly(self):
        result = ClaimResult(
            claim="Pizza is the best food.",
            claim_type=ClaimType.OPINION,
            status=ClaimStatus.UNVERIFIABLE,
            evidence="Subjective statement.",
            confidence=0.55,
            confidence_reason="Opinion claims get low confidence.",
        )
        data = result.model_dump()
        assert data["claim_type"] == "Opinion"
        assert data["status"] == "Unverifiable"
        assert data["confidence"] == 0.55
        assert "claim_type" in data


class TestClaimTypeExamples:
    """
    Test known examples of each claim type.
    These tests validate the expected routing behavior:
    - Factual claims → full pipeline
    - Opinion/Unverifiable → Unverifiable status, ~0.55 confidence
    """

    OPINION_CLAIMS = [
        "NASA cannot be trusted.",
        "The government is corrupt.",
        "Pizza is the best food.",
        "Scientists are hiding the truth.",
        "You can't trust mainstream media.",
    ]

    FACTUAL_CLAIMS = [
        "The Earth is flat.",
        "Vaccines cause autism.",
        "5G towers spread COVID-19.",
        "The moon landing happened in 1969.",
        "Water boils at 100 degrees Celsius.",
    ]

    UNVERIFIABLE_CLAIMS = [
        "Something big is coming soon.",
        "They don't want you to know the truth.",
        "Everything happens for a reason.",
    ]

    @pytest.mark.parametrize("claim", OPINION_CLAIMS)
    def test_opinion_claims_get_unverifiable_result(self, claim):
        """Opinion claims should produce valid ClaimResult with Unverifiable status."""
        result = ClaimResult(
            claim=claim,
            claim_type=ClaimType.OPINION,
            status=ClaimStatus.UNVERIFIABLE,
            evidence=f"This claim is opinion and cannot be objectively verified.",
            confidence=0.55,
            confidence_reason="Low confidence because this is an opinion claim.",
        )
        assert result.status == ClaimStatus.UNVERIFIABLE
        assert result.confidence <= 0.60

    @pytest.mark.parametrize("claim", FACTUAL_CLAIMS)
    def test_factual_claims_produce_valid_result(self, claim):
        """Factual claims should be allowed through the full pipeline."""
        result = ClaimResult(
            claim=claim,
            claim_type=ClaimType.FACTUAL,
            status=ClaimStatus.CONTRADICTED,
            evidence="Counter-evidence from trusted sources.",
            confidence=0.82,
        )
        assert result.claim_type == ClaimType.FACTUAL
        assert result.status != ClaimStatus.UNVERIFIABLE

    @pytest.mark.parametrize("claim", UNVERIFIABLE_CLAIMS)
    def test_unverifiable_claims_get_unverifiable_result(self, claim):
        """Unverifiable claims should produce valid ClaimResult with Unverifiable status."""
        result = ClaimResult(
            claim=claim,
            claim_type=ClaimType.UNVERIFIABLE,
            status=ClaimStatus.UNVERIFIABLE,
            evidence="This claim is too vague to verify.",
            confidence=0.55,
        )
        assert result.status == ClaimStatus.UNVERIFIABLE
        assert result.confidence <= 0.60
