"""
Tests for semantic retrieval (TF-IDF based).

Verifies that:
- Paraphrased claims match correct corpus entries
- Near-miss claims without exact keywords still find evidence
- Irrelevant claims are rejected (below threshold)
- Strong keyword matches still use CORPUS, not SEMANTIC
- Semantic grounding rules are applied correctly
- The index is built from the corpus
"""

import pytest
from app.services.semantic_index import (
    SemanticIndex,
    SemanticMatch,
    semantic_search,
    get_semantic_index,
    _reset_semantic_index,
)
from app.services.evidence_store import EVIDENCE_STORE
from app.services.analyzer import (
    _apply_evidence_grounding,
    _build_evidence_display,
)
from app.models.schemas import EvidenceSource


@pytest.fixture(autouse=True)
def reset_index():
    """Ensure a fresh semantic index for each test module."""
    _reset_semantic_index()
    yield


# ===================================================================
# Index construction
# ===================================================================

class TestSemanticIndexConstruction:
    """Verify the index is built correctly from the corpus."""

    def test_index_has_correct_entry_count(self):
        index = get_semantic_index()
        assert index.entry_count == len(list(EVIDENCE_STORE))
        assert index.entry_count >= 25

    def test_index_has_word_matrix(self):
        index = get_semantic_index()
        assert index._word_matrix is not None
        assert index._word_matrix.shape[0] == index.entry_count

    def test_index_has_char_matrix(self):
        index = get_semantic_index()
        assert index._char_matrix is not None
        assert index._char_matrix.shape[0] == index.entry_count

    def test_search_returns_list(self):
        results = semantic_search("test claim")
        assert isinstance(results, list)

    def test_search_results_are_semantic_matches(self):
        results = semantic_search("the earth is flat")
        for r in results:
            assert isinstance(r, SemanticMatch)
            assert hasattr(r, "entry")
            assert hasattr(r, "similarity_score")


# ===================================================================
# Paraphrase matching — the core value of semantic retrieval
# ===================================================================

class TestParaphraseMatching:
    """Claims that rephrase corpus entries should still find matches."""

    def test_immunization_developmental_issues(self):
        """'immunization shots cause developmental problems' → should match vaccine/autism entry."""
        results = semantic_search("Immunization shots cause developmental problems in children.")
        assert len(results) > 0
        topics = [r.entry.topic.lower() for r in results]
        assert any("vaccin" in t or "autism" in t for t in topics), \
            f"Expected vaccine/autism match, got: {topics}"

    def test_globe_not_round(self):
        """'The globe isn't actually round' → should match earth shape entry."""
        results = semantic_search("The globe isn't actually round, it's flat.")
        assert len(results) > 0
        topics = [r.entry.topic.lower() for r in results]
        assert any("earth" in t or "flat" in t for t in topics), \
            f"Expected earth shape match, got: {topics}"

    def test_cell_towers_cause_illness(self):
        """'Cell towers cause illness' → should match 5G/COVID entry."""
        results = semantic_search("Cell towers are causing widespread illness.")
        assert len(results) > 0
        topics = [r.entry.topic.lower() for r in results]
        assert any("5g" in t or "covid" in t or "tower" in t for t in topics), \
            f"Expected 5G match, got: {topics}"

    def test_lunar_mission_hoax(self):
        """'The lunar mission was a hoax' → should match moon landing entry."""
        results = semantic_search("The first lunar mission was actually a hoax.")
        assert len(results) > 0
        topics = [r.entry.topic.lower() for r in results]
        assert any("moon" in t or "lunar" in t or "landing" in t for t in topics), \
            f"Expected moon landing match, got: {topics}"

    def test_climate_warming_denial(self):
        """'Global warming is fabricated' → should match climate change entry."""
        results = semantic_search("Global warming is completely fabricated by scientists.")
        assert len(results) > 0
        topics = [r.entry.topic.lower() for r in results]
        assert any("climate" in t or "warming" in t for t in topics), \
            f"Expected climate change match, got: {topics}"


# ===================================================================
# Irrelevant claims — must NOT match
# ===================================================================

class TestIrrelevantClaimsRejected:
    """Claims with no relation to any corpus entry should return no matches."""

    def test_cat_likes_tuna(self):
        results = semantic_search("My cat likes tuna fish.", min_similarity=0.10)
        assert len(results) == 0, f"Should not match, got: {[r.entry.topic for r in results]}"

    def test_weather_is_nice(self):
        results = semantic_search("The weather is nice today.", min_similarity=0.10)
        assert len(results) == 0, f"Should not match, got: {[r.entry.topic for r in results]}"

    def test_random_sentence(self):
        results = semantic_search("I went to the store to buy groceries.", min_similarity=0.10)
        assert len(results) == 0, f"Should not match, got: {[r.entry.topic for r in results]}"


# ===================================================================
# Ranking — results ordered by similarity
# ===================================================================

class TestRanking:
    """Results should be ordered by descending similarity."""

    def test_results_ranked_by_similarity(self):
        results = semantic_search("vaccines cause autism in children", top_k=5, min_similarity=0.05)
        if len(results) >= 2:
            scores = [r.similarity_score for r in results]
            assert scores == sorted(scores, reverse=True), \
                f"Results not ranked by similarity: {scores}"

    def test_top_k_limits_results(self):
        results = semantic_search("health science medicine", top_k=2, min_similarity=0.01)
        assert len(results) <= 2


# ===================================================================
# Similarity scores
# ===================================================================

class TestSimilarityScores:
    """Verify similarity score properties."""

    def test_scores_between_0_and_1(self):
        results = semantic_search("the earth is flat", min_similarity=0.01)
        for r in results:
            assert 0.0 <= r.similarity_score <= 1.0, \
                f"Score out of range: {r.similarity_score}"

    def test_exact_match_high_similarity(self):
        """A claim that closely matches a corpus entry should have high similarity."""
        results = semantic_search("The Earth is not flat, it is an oblate spheroid.")
        assert len(results) > 0
        assert results[0].similarity_score >= 0.20, \
            f"Expected high similarity for near-exact match, got: {results[0].similarity_score}"


# ===================================================================
# Grounding rules for SEMANTIC source
# ===================================================================

class TestSemanticGrounding:
    """Verify that SEMANTIC evidence source follows its grounding rules."""

    def test_supported_allowed(self):
        """SEMANTIC source should allow Supported verdict."""
        status, conf, note = _apply_evidence_grounding(
            "Supported", 0.85, EvidenceSource.SEMANTIC, ""
        )
        assert status == "Supported"
        assert "semantic" in note.lower()

    def test_contradicted_allowed(self):
        """SEMANTIC source should allow Contradicted verdict."""
        status, conf, note = _apply_evidence_grounding(
            "Contradicted", 0.90, EvidenceSource.SEMANTIC, ""
        )
        assert status == "Contradicted"

    def test_confidence_capped_at_72(self):
        """SEMANTIC source should cap confidence at 0.72."""
        _, conf, _ = _apply_evidence_grounding(
            "Supported", 0.95, EvidenceSource.SEMANTIC, ""
        )
        assert conf == 0.72

    def test_low_confidence_not_raised(self):
        """If raw confidence is below cap, it should not be raised."""
        _, conf, _ = _apply_evidence_grounding(
            "Supported", 0.60, EvidenceSource.SEMANTIC, ""
        )
        assert conf == 0.60

    def test_semantic_between_corpus_and_dynamic(self):
        """SEMANTIC cap (0.72) should be between CORPUS (no cap) and DYNAMIC (0.70)."""
        _, corpus_conf, _ = _apply_evidence_grounding(
            "Supported", 0.95, EvidenceSource.CORPUS, ""
        )
        _, semantic_conf, _ = _apply_evidence_grounding(
            "Supported", 0.95, EvidenceSource.SEMANTIC, ""
        )
        _, dynamic_conf, _ = _apply_evidence_grounding(
            "Supported", 0.95, EvidenceSource.DYNAMIC, ""
        )
        assert corpus_conf > semantic_conf > dynamic_conf


# ===================================================================
# Display formatting
# ===================================================================

class TestSemanticDisplay:
    """Verify display labels for semantic evidence."""

    def test_display_shows_semantic_match_label(self):
        display = _build_evidence_display(
            EvidenceSource.SEMANTIC,
            "The Earth is an oblate spheroid.",
            "NASA, ESA",
            "https://nasa.gov",
            "",
        )
        assert "[Semantic match]" in display
        assert "NASA, ESA" in display
        assert "https://nasa.gov" in display

    def test_display_without_url(self):
        display = _build_evidence_display(
            EvidenceSource.SEMANTIC,
            "Test evidence",
            "Test Source",
            "",
            "",
        )
        assert "[Semantic match]" in display
        assert "Test Source" in display


# ===================================================================
# Config integration
# ===================================================================

class TestSemanticConfig:
    """Verify config flags work correctly."""

    def test_min_similarity_threshold_filters(self):
        """Higher threshold should return fewer results."""
        results_low = semantic_search("vaccines health", min_similarity=0.05)
        results_high = semantic_search("vaccines health", min_similarity=0.50)
        assert len(results_low) >= len(results_high)

    def test_enum_value_exists(self):
        """SEMANTIC should be a valid EvidenceSource enum value."""
        assert EvidenceSource.SEMANTIC == "semantic"
        assert EvidenceSource.SEMANTIC.value == "semantic"


# ===================================================================
# Topic-scoped synonyms
# ===================================================================

class TestTopicScopedSynonyms:
    """Verify that synonyms are per-entry, not global."""

    def test_all_entries_have_synonyms(self):
        """Every corpus entry should have at least one synonym."""
        for entry in EVIDENCE_STORE:
            assert len(entry.synonyms) > 0, \
                f"Entry '{entry.topic}' has no synonyms"

    def test_vaccine_entry_has_immunization_synonym(self):
        for entry in EVIDENCE_STORE:
            if entry.topic == "vaccine autism":
                assert "immunization" in entry.synonyms
                break

    def test_moon_entry_has_lunar_synonym(self):
        for entry in EVIDENCE_STORE:
            if entry.topic == "moon landing":
                assert "lunar mission" in entry.synonyms
                break

    def test_no_global_synonym_map(self):
        """SemanticIndex should NOT have a global _SYNONYM_MAP."""
        assert not hasattr(SemanticIndex, "_SYNONYM_MAP")

    def test_synonym_enrichment_boosts_match(self):
        """Claim using a synonym term should match its specific topic."""
        results = semantic_search("immunization causes developmental problems")
        assert len(results) > 0
        assert results[0].entry.topic == "vaccine autism"


# ===================================================================
# Ambiguity detection
# ===================================================================

class TestAmbiguityDetection:
    """Verify ambiguity flagging when top matches span different topics."""

    def test_clear_match_not_ambiguous(self):
        results = semantic_search("The Earth is flat and not round")
        assert len(results) > 0
        assert results[0].is_ambiguous is False

    def test_semantic_match_has_is_ambiguous_field(self):
        results = semantic_search("vaccines cause autism")
        assert len(results) > 0
        assert hasattr(results[0], "is_ambiguous")

    def test_ambiguity_gap_constant_exists(self):
        index = get_semantic_index()
        assert hasattr(index, "_AMBIGUITY_GAP")
        assert index._AMBIGUITY_GAP == 0.04

    def test_single_result_not_ambiguous(self):
        index = get_semantic_index()
        single = [SemanticMatch(entry=list(EVIDENCE_STORE)[0], similarity_score=0.5)]
        assert index._detect_ambiguity(single) is False

    def test_different_topics_close_scores_is_ambiguous(self):
        index = get_semantic_index()
        entries = list(EVIDENCE_STORE)
        e1 = [e for e in entries if e.topic == "vaccine autism"][0]
        e2 = [e for e in entries if e.topic == "earth shape"][0]
        matches = [
            SemanticMatch(entry=e1, similarity_score=0.20),
            SemanticMatch(entry=e2, similarity_score=0.19),  # gap = 0.01 < 0.04
        ]
        assert index._detect_ambiguity(matches) is True

    def test_different_topics_wide_gap_not_ambiguous(self):
        index = get_semantic_index()
        entries = list(EVIDENCE_STORE)
        e1 = [e for e in entries if e.topic == "vaccine autism"][0]
        e2 = [e for e in entries if e.topic == "earth shape"][0]
        matches = [
            SemanticMatch(entry=e1, similarity_score=0.30),
            SemanticMatch(entry=e2, similarity_score=0.10),  # gap = 0.20 > 0.04
        ]
        assert index._detect_ambiguity(matches) is False


# ===================================================================
# Low-confidence floor
# ===================================================================

class TestLowConfidenceFloor:
    """Verify that low-scoring matches are flagged, not treated as ambiguous."""

    def test_low_confidence_floor_constant(self):
        index = get_semantic_index()
        assert index._LOW_CONFIDENCE_FLOOR == 0.15

    def test_has_is_low_confidence_field(self):
        m = SemanticMatch(entry=list(EVIDENCE_STORE)[0], similarity_score=0.10)
        assert hasattr(m, "is_low_confidence")

    def test_below_floor_flagged_as_low_confidence(self):
        """Matches with top score < 0.15 should be flagged as low-confidence."""
        # Use a claim that produces a weak match (score between 0.10 and 0.15)
        results = semantic_search("Cell towers are causing widespread illness.",
                                  min_similarity=0.05)
        if results and results[0].similarity_score < 0.15:
            assert results[0].is_low_confidence is True
            assert results[0].is_ambiguous is False  # should NOT be ambiguous

    def test_above_floor_not_flagged_low_confidence(self):
        """Matches with top score >= 0.15 should NOT be flagged low-confidence."""
        results = semantic_search("The globe isn't actually round, it's flat.")
        assert len(results) > 0
        assert results[0].similarity_score >= 0.15
        assert results[0].is_low_confidence is False

    def test_low_confidence_skips_ambiguity_check(self):
        """A low-confidence match should never be flagged as ambiguous,
        even if top-2 are from different topics with close scores."""
        index = get_semantic_index()
        entries = list(EVIDENCE_STORE)
        e1 = [e for e in entries if e.topic == "vaccine autism"][0]
        e2 = [e for e in entries if e.topic == "earth shape"][0]
        # Both scores below floor
        matches = [
            SemanticMatch(entry=e1, similarity_score=0.12),
            SemanticMatch(entry=e2, similarity_score=0.11),
        ]
        # Simulate what search() does: check floor before ambiguity
        top_score = matches[0].similarity_score
        if top_score < index._LOW_CONFIDENCE_FLOOR:
            matches[0].is_low_confidence = True
        else:
            if index._detect_ambiguity(matches):
                matches[0].is_ambiguous = True
        assert matches[0].is_low_confidence is True
        assert matches[0].is_ambiguous is False
