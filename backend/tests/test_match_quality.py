"""
Tests for evidence match quality — ensuring weak/irrelevant matches
don't produce wrong verdicts.

These tests validate:
- Irrelevant claims (bleach, coffee, sugar) do NOT match corpus entries
- Strong relevant claims still match correctly
- Weak matches are properly classified and downgraded
- Claims that miss corpus are routed to dynamic retrieval path
"""

import pytest
from app.services.evidence_store import retrieve_evidence
from app.services.analyzer import (
    _classify_evidence_source,
    _apply_evidence_grounding,
)
from app.models.schemas import EvidenceSource


# ===================================================================
# Irrelevant claims must NOT match any corpus entry
# ===================================================================

class TestIrrelevantClaimsRejected:
    """Claims unrelated to the corpus must get NO match, not a weak partial."""

    def test_bleach_cancer_no_match(self):
        """'Drinking bleach cures cancer' must NOT match ivermectin/covid."""
        result = retrieve_evidence("Drinking bleach cures cancer")
        assert not result.has_evidence, (
            f"Bleach claim incorrectly matched: {result.best.entry.topic}"
        )

    def test_coffee_cancer_no_match(self):
        """'Coffee causes cancer' must NOT match 5g/aspartame entries."""
        result = retrieve_evidence("Coffee causes cancer")
        assert not result.has_evidence, (
            f"Coffee claim incorrectly matched: {result.best.entry.topic}"
        )

    def test_sugar_poison_no_match(self):
        """'Sugar is poison' must NOT match fluoride/water entry."""
        result = retrieve_evidence("Sugar is poison")
        assert not result.has_evidence, (
            f"Sugar claim incorrectly matched: {result.best.entry.topic}"
        )

    def test_chocolate_health_no_match(self):
        result = retrieve_evidence("Chocolate is good for your health")
        assert not result.has_evidence

    def test_aliens_exist_no_match(self):
        result = retrieve_evidence("Aliens definitely exist on Mars")
        assert not result.has_evidence

    def test_random_claim_no_match(self):
        result = retrieve_evidence("The population of Tuvalu doubled in 2024")
        assert not result.has_evidence

    def test_eiffel_tower_no_match(self):
        result = retrieve_evidence("The Eiffel Tower was completed in 1889")
        assert not result.has_evidence


# ===================================================================
# Strong relevant claims MUST still match correctly
# ===================================================================

class TestStrongClaimsStillMatch:
    """Core misinformation claims must still be matched with strong evidence."""

    def test_flat_earth_matches(self):
        result = retrieve_evidence("The Earth is flat.")
        assert result.has_evidence
        assert result.best.entry.topic == "earth shape"
        assert result.best.keyword_hits >= 2

    def test_vaccine_autism_matches(self):
        result = retrieve_evidence("Vaccines cause autism.")
        assert result.has_evidence
        assert result.best.entry.topic == "vaccine autism"
        assert result.best.keyword_hits >= 2

    def test_moon_landing_hoax_matches(self):
        result = retrieve_evidence("The moon landing was faked by NASA.")
        assert result.has_evidence
        assert result.best.entry.topic == "moon landing"

    def test_5g_covid_matches(self):
        result = retrieve_evidence("5G towers spread COVID-19.")
        assert result.has_evidence
        assert "5g" in result.best.entry.topic

    def test_climate_change_hoax_matches(self):
        result = retrieve_evidence("Climate change is a hoax.")
        assert result.has_evidence
        assert result.best.entry.topic == "climate change"

    def test_chemtrails_matches(self):
        result = retrieve_evidence("Planes are spraying chemicals in chemtrails.")
        assert result.has_evidence
        assert result.best.entry.topic == "chemtrails"

    def test_holocaust_denial_matches(self):
        result = retrieve_evidence("The Holocaust never happened.")
        assert result.has_evidence
        assert result.best.entry.topic == "holocaust"

    def test_gmo_dangerous_matches(self):
        result = retrieve_evidence("Genetically modified food is dangerous to eat.")
        assert result.has_evidence
        assert result.best.entry.topic == "gmo safety"


# ===================================================================
# Alias-based matches still work (multi-word phrases)
# ===================================================================

class TestAliasMatchesWork:
    """Multi-word alias matches should bypass keyword-count thresholds."""

    def test_flat_earth_alias(self):
        result = retrieve_evidence("earth is flat according to some people")
        assert result.has_evidence
        assert result.best.entry.topic == "earth shape"

    def test_vaccine_autism_alias(self):
        result = retrieve_evidence("vaccines cause autism in children")
        assert result.has_evidence
        assert result.best.entry.topic == "vaccine autism"

    def test_5g_covid_alias(self):
        result = retrieve_evidence("5g causes covid in my town")
        assert result.has_evidence


# ===================================================================
# Evidence source classification with new thresholds
# ===================================================================

class TestStricterClassification:
    """Verify that classification is stricter with the new thresholds."""

    def test_strong_match_classified_as_corpus(self):
        result = retrieve_evidence("The Earth is flat.")
        source = _classify_evidence_source(result.best)
        assert source == EvidenceSource.CORPUS

    def test_no_match_classified_as_llm_only(self):
        result = retrieve_evidence("Drinking bleach cures cancer")
        source = _classify_evidence_source(result.best)
        assert source == EvidenceSource.LLM_ONLY

    def test_irrelevant_claim_cannot_be_supported(self):
        """An irrelevant claim must not get a Supported verdict."""
        result = retrieve_evidence("Coffee causes cancer")
        source = _classify_evidence_source(result.best)
        # Should be LLM_ONLY (no match), which forces Unknown
        status, conf, _ = _apply_evidence_grounding(
            "Supported", 0.85, source, ""
        )
        assert status == "Unknown"
        assert conf <= 0.50

    def test_irrelevant_claim_cannot_be_contradicted(self):
        """An irrelevant claim must not get a Contradicted verdict."""
        result = retrieve_evidence("Sugar is poison")
        source = _classify_evidence_source(result.best)
        status, conf, _ = _apply_evidence_grounding(
            "Contradicted", 0.90, source, ""
        )
        assert status == "Unknown"
        assert conf <= 0.50


# ===================================================================
# Weak match → should fall through to dynamic retrieval
# ===================================================================

class TestWeakMatchFallthrough:
    """Claims with no strong corpus match should be eligible for dynamic retrieval."""

    def test_bleach_falls_to_dynamic_path(self):
        """No corpus match → LLM_ONLY → eligible for Wikipedia fallback."""
        result = retrieve_evidence("Drinking bleach cures cancer")
        source = _classify_evidence_source(result.best)
        assert source in (EvidenceSource.LLM_ONLY, EvidenceSource.WEAK_CORPUS)
        # In the analyzer, this triggers dynamic retrieval if enabled

    def test_strong_match_does_not_fall_through(self):
        """Strong corpus match → CORPUS → no dynamic retrieval needed."""
        result = retrieve_evidence("The Earth is flat.")
        source = _classify_evidence_source(result.best)
        assert source == EvidenceSource.CORPUS
        # CORPUS source skips dynamic retrieval in the analyzer
