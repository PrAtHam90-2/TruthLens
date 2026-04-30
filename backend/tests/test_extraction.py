"""
Tests for claim extraction — both the heuristic fallback and the
post-processing pipeline.
"""

import pytest
from app.services.fallback import extract_claims_heuristic
from app.services.llm_client import _post_process_claims


# ===================================================================
# Heuristic fallback tests
# ===================================================================

class TestHeuristicExtraction:
    """Tests for the regex/rule-based fallback extractor."""

    def test_single_claim(self):
        text = "The Earth is an oblate spheroid."
        claims = extract_claims_heuristic(text)
        assert len(claims) == 1
        assert "Earth" in claims[0]

    def test_two_sentences(self):
        text = "The Earth is flat. Vaccines cause autism."
        claims = extract_claims_heuristic(text)
        assert len(claims) == 2

    def test_compound_sentence_splits(self):
        text = "The Earth is flat and vaccines cause autism."
        claims = extract_claims_heuristic(text)
        assert len(claims) >= 2, f"Expected 2+ claims but got {claims}"

    def test_filters_questions(self):
        text = "Is the Earth flat? The moon landing was faked."
        claims = extract_claims_heuristic(text)
        # Should only keep the statement, not the question
        assert len(claims) == 1
        assert "moon" in claims[0].lower()

    def test_filters_opinions(self):
        text = "I think pizza is great. The Earth revolves around the Sun."
        claims = extract_claims_heuristic(text)
        assert len(claims) == 1
        assert "Sun" in claims[0]

    def test_filters_short_fragments(self):
        text = "Yes. No. The Earth is an oblate spheroid confirmed by NASA."
        claims = extract_claims_heuristic(text)
        assert len(claims) == 1  # Only the long sentence

    def test_deduplicates(self):
        text = "The Earth is flat. The earth is flat."
        claims = extract_claims_heuristic(text)
        assert len(claims) == 1

    def test_multiline_text(self):
        text = """Climate change is real.
Vaccines are safe and effective.
The moon landing happened in 1969."""
        claims = extract_claims_heuristic(text)
        assert len(claims) >= 3

    def test_mixed_opinions_and_facts(self):
        text = "I believe the sky is blue. Water boils at 100 degrees Celsius. In my opinion, summer is the best season."
        claims = extract_claims_heuristic(text)
        # Should keep only the factual statement
        assert len(claims) == 1
        assert "boils" in claims[0].lower()

    def test_compound_with_but(self):
        text = "5G is safe for humans but some people claim it causes cancer."
        claims = extract_claims_heuristic(text)
        assert len(claims) >= 2

    def test_empty_input_returns_empty(self):
        claims = extract_claims_heuristic("")
        assert claims == []

    def test_very_short_input(self):
        claims = extract_claims_heuristic("Short.")
        assert claims == []

    def test_long_paragraph(self):
        text = (
            "The Earth is flat and the moon landing was faked. "
            "Vaccines contain microchips that track people. "
            "5G towers are spreading COVID-19 and climate change is a hoax invented by China."
        )
        claims = extract_claims_heuristic(text)
        assert len(claims) >= 4, f"Expected 4+ claims from compound paragraph, got {claims}"


# ===================================================================
# Post-processing tests (_post_process_claims)
# ===================================================================

class TestPostProcessClaims:
    """Tests for the LLM output post-processing."""

    def test_removes_duplicates(self):
        claims = [
            "The Earth is flat.",
            "The earth is flat.",
            "the earth is flat",
        ]
        result = _post_process_claims(claims)
        assert len(result) == 1

    def test_removes_questions(self):
        claims = [
            "The Earth is flat.",
            "Is the Earth really flat?",
        ]
        result = _post_process_claims(claims)
        assert len(result) == 1
        assert "?" not in result[0]

    def test_removes_short_strings(self):
        claims = ["Yes.", "No.", "The Earth is an oblate spheroid."]
        result = _post_process_claims(claims)
        assert len(result) == 1
        assert "spheroid" in result[0]

    def test_removes_meta_statements(self):
        claims = [
            "Here is the analysis of the claims.",
            "The Earth is flat.",
            "The following claims were extracted.",
        ]
        result = _post_process_claims(claims)
        assert len(result) == 1
        assert "flat" in result[0]

    def test_preserves_good_claims(self):
        claims = [
            "The Earth is flat.",
            "Vaccines cause autism.",
            "5G causes cancer.",
        ]
        result = _post_process_claims(claims)
        assert len(result) == 3

    def test_caps_at_15(self):
        claims = [f"Claim number {i} is a factual statement about science." for i in range(20)]
        result = _post_process_claims(claims)
        assert len(result) == 15

    def test_strips_whitespace(self):
        claims = ["  The Earth is flat.  ", "\tVaccines cause autism.\n"]
        result = _post_process_claims(claims)
        assert len(result) == 2
        assert result[0] == "The Earth is flat."
        assert result[1] == "Vaccines cause autism."

    def test_empty_list(self):
        assert _post_process_claims([]) == []
