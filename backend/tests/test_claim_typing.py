"""
Tests for refined claim type classification.

Tests verify that:
- Conspiracy/misinformation claims are classified as FACTUAL (not Opinion)
- Pure opinions remain OPINION
- Vague/speculative claims remain UNVERIFIABLE
- "I think X" hedging is handled correctly
- Borderline cases are resolved correctly
"""

import pytest
from app.services.analyzer import _heuristic_classify_type
from app.models.schemas import ClaimType


# ===================================================================
# Conspiracy claims → must be FACTUAL
# ===================================================================

class TestConspiracyClaimsAreFactual:
    """Conspiracy theories make testable assertions → must be FACTUAL."""

    def test_5g_spreads_covid(self):
        result = _heuristic_classify_type("5G spreads COVID-19.")
        assert result is not None
        assert result["claim_type"] == ClaimType.FACTUAL

    def test_moon_landing_faked(self):
        result = _heuristic_classify_type("The moon landing was faked.")
        assert result is not None
        assert result["claim_type"] == ClaimType.FACTUAL

    def test_vaccines_contain_microchips(self):
        result = _heuristic_classify_type("Vaccines contain microchips.")
        assert result is not None
        assert result["claim_type"] == ClaimType.FACTUAL

    def test_vaccines_cause_autism(self):
        result = _heuristic_classify_type("Vaccines cause autism.")
        assert result is not None
        assert result["claim_type"] == ClaimType.FACTUAL

    def test_earth_is_flat(self):
        result = _heuristic_classify_type("The Earth is flat.")
        assert result is not None
        assert result["claim_type"] == ClaimType.FACTUAL

    def test_chemtrails_are_real(self):
        result = _heuristic_classify_type("Chemtrails are real.")
        assert result is not None
        assert result["claim_type"] == ClaimType.FACTUAL

    def test_holocaust_never_happened(self):
        result = _heuristic_classify_type("The Holocaust never happened.")
        assert result is not None
        assert result["claim_type"] == ClaimType.FACTUAL

    def test_climate_change_is_a_hoax(self):
        result = _heuristic_classify_type("Climate change is a hoax.")
        assert result is not None
        assert result["claim_type"] == ClaimType.FACTUAL

    def test_ivermectin_cures_covid(self):
        result = _heuristic_classify_type("Ivermectin cures COVID-19.")
        assert result is not None
        assert result["claim_type"] == ClaimType.FACTUAL

    def test_fluoride_kills_brain_cells(self):
        result = _heuristic_classify_type("Fluoride kills brain cells.")
        assert result is not None
        assert result["claim_type"] == ClaimType.FACTUAL

    def test_911_was_staged(self):
        result = _heuristic_classify_type("9/11 was staged by the government.")
        assert result is not None
        assert result["claim_type"] == ClaimType.FACTUAL


# ===================================================================
# Scientific/factual claims → must be FACTUAL
# ===================================================================

class TestScientificClaimsAreFactual:
    """Legitimate scientific claims → FACTUAL."""

    def test_water_boils_at_100(self):
        result = _heuristic_classify_type("Water boils at 100 degrees Celsius.")
        assert result is not None
        assert result["claim_type"] == ClaimType.FACTUAL

    def test_earth_orbits_sun(self):
        result = _heuristic_classify_type("The Earth orbits the sun.")
        assert result is not None
        assert result["claim_type"] == ClaimType.FACTUAL

    def test_gmo_food_is_safe(self):
        result = _heuristic_classify_type("Genetically modified food is safe.")
        assert result is not None
        assert result["claim_type"] == ClaimType.FACTUAL

    def test_aspirin_prevents_heart_attacks(self):
        result = _heuristic_classify_type("Aspirin prevents heart attacks.")
        assert result is not None
        assert result["claim_type"] == ClaimType.FACTUAL


# ===================================================================
# Pure opinions → must be OPINION
# ===================================================================

class TestPureOpinionsStayOpinion:
    """Subjective value judgments → OPINION."""

    def test_nasa_cannot_be_trusted(self):
        result = _heuristic_classify_type("NASA cannot be trusted.")
        assert result is not None
        assert result["claim_type"] == ClaimType.OPINION

    def test_pizza_is_the_best(self):
        result = _heuristic_classify_type("Pizza is the best food.")
        assert result is not None
        assert result["claim_type"] == ClaimType.OPINION

    def test_government_is_corrupt(self):
        result = _heuristic_classify_type("The government is corrupt.")
        assert result is not None
        assert result["claim_type"] == ClaimType.OPINION

    def test_scientists_are_dishonest(self):
        result = _heuristic_classify_type("Scientists are dishonest.")
        assert result is not None
        assert result["claim_type"] == ClaimType.OPINION

    def test_vaccines_should_be_banned(self):
        result = _heuristic_classify_type("Vaccines should be banned.")
        assert result is not None
        assert result["claim_type"] == ClaimType.OPINION

    def test_modern_art_is_disgusting(self):
        result = _heuristic_classify_type("Modern art is disgusting.")
        assert result is not None
        assert result["claim_type"] == ClaimType.OPINION


# ===================================================================
# Unverifiable → must be UNVERIFIABLE
# ===================================================================

class TestUnverifiableStaysUnverifiable:
    """Vague, speculative, unfalsifiable → UNVERIFIABLE."""

    def test_something_big_is_coming(self):
        result = _heuristic_classify_type("Something big is coming soon.")
        assert result is not None
        assert result["claim_type"] == ClaimType.UNVERIFIABLE

    def test_truth_will_come_out(self):
        result = _heuristic_classify_type("The truth will come out eventually.")
        assert result is not None
        assert result["claim_type"] == ClaimType.UNVERIFIABLE

    def test_everything_happens_for_a_reason(self):
        result = _heuristic_classify_type("Everything happens for a reason.")
        assert result is not None
        assert result["claim_type"] == ClaimType.UNVERIFIABLE

    def test_they_dont_want_you_to_know(self):
        result = _heuristic_classify_type("They don't want you to know the truth.")
        assert result is not None
        assert result["claim_type"] == ClaimType.UNVERIFIABLE

    def test_mark_my_words(self):
        result = _heuristic_classify_type("Mark my words, this will change everything.")
        assert result is not None
        assert result["claim_type"] == ClaimType.UNVERIFIABLE


# ===================================================================
# "I think X" hedging — depends on core claim
# ===================================================================

class TestHedgingHandling:
    """'I think' prefix: factual core → FACTUAL, opinion core → OPINION."""

    def test_i_think_moon_landing_fake(self):
        """'I think the moon landing was faked' → FACTUAL (core is testable)."""
        result = _heuristic_classify_type("I think the moon landing was faked.")
        assert result is not None
        assert result["claim_type"] == ClaimType.FACTUAL

    def test_i_think_vaccines_cause_autism(self):
        """'I think vaccines cause autism' → FACTUAL (core is testable)."""
        result = _heuristic_classify_type("I think vaccines cause autism.")
        assert result is not None
        assert result["claim_type"] == ClaimType.FACTUAL

    def test_i_think_vaccines_are_bad(self):
        """'I think vaccines are bad' → OPINION (core is subjective)."""
        result = _heuristic_classify_type("I think vaccines are bad.")
        assert result is not None
        assert result["claim_type"] == ClaimType.OPINION

    def test_i_believe_earth_is_flat(self):
        """'I believe the earth is flat' → FACTUAL (core is testable)."""
        result = _heuristic_classify_type("I believe the earth is flat.")
        assert result is not None
        assert result["claim_type"] == ClaimType.FACTUAL

    def test_in_my_opinion_pizza_is_good(self):
        """'In my opinion pizza is good' → OPINION (core is subjective)."""
        result = _heuristic_classify_type("In my opinion, pizza is good.")
        assert result is not None
        assert result["claim_type"] == ClaimType.OPINION

    def test_personally_5g_is_dangerous(self):
        """'Personally, 5G is dangerous' → FACTUAL (core is testable safety claim)."""
        result = _heuristic_classify_type("Personally, 5G is dangerous.")
        assert result is not None
        assert result["claim_type"] == ClaimType.FACTUAL


# ===================================================================
# Ambiguous cases → returns None (defers to LLM)
# ===================================================================

class TestAmbiguousDeferToLLM:
    """Claims without clear patterns should return None for LLM classification."""

    def test_ambiguous_claim_returns_none(self):
        result = _heuristic_classify_type("The economy is struggling right now.")
        assert result is None

    def test_simple_statement_returns_none(self):
        result = _heuristic_classify_type("Many people disagree with this policy.")
        assert result is None
