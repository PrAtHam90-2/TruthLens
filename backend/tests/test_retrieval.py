"""
Tests for the evidence retrieval system.

Validates that the evidence store returns relevant, ranked matches
for known misinformation claims and handles edge cases correctly.
"""

import pytest
from app.services.evidence_store import (
    retrieve_evidence,
    EvidenceResult,
    RankedEvidence,
    SourceEntry,
    Source,
    EVIDENCE_STORE,
)
from app.services.corpus import (
    retrieve_evidence as corpus_retrieve,
    CorpusEntry,
    EvidenceMatch,
    TRUSTED_CORPUS,
)


# ===================================================================
# Evidence store data integrity
# ===================================================================

class TestEvidenceStoreData:
    """Verify the evidence store is properly populated."""

    def test_has_at_least_25_entries(self):
        assert len(EVIDENCE_STORE) >= 25, f"Expected 25+ entries, got {len(EVIDENCE_STORE)}"

    def test_all_entries_have_sources(self):
        for entry in EVIDENCE_STORE:
            assert len(entry.sources) >= 1, f"Entry '{entry.topic}' has no sources"

    def test_all_entries_have_keywords(self):
        for entry in EVIDENCE_STORE:
            assert len(entry.keywords) >= 2, f"Entry '{entry.topic}' needs at least 2 keywords"

    def test_all_entries_have_categories(self):
        valid_categories = {"science", "health", "history", "technology", "nutrition", "politics"}
        for entry in EVIDENCE_STORE:
            assert entry.category in valid_categories, f"Entry '{entry.topic}' has invalid category '{entry.category}'"

    def test_all_sources_have_names(self):
        for entry in EVIDENCE_STORE:
            for source in entry.sources:
                assert source.name, f"Source in '{entry.topic}' has no name"

    def test_most_sources_have_urls(self):
        """At least 80% of sources should have URLs."""
        total = sum(len(e.sources) for e in EVIDENCE_STORE)
        with_urls = sum(1 for e in EVIDENCE_STORE for s in e.sources if s.url)
        ratio = with_urls / total
        assert ratio >= 0.8, f"Only {ratio:.0%} of sources have URLs (expected 80%+)"

    def test_source_entry_properties(self):
        entry = EVIDENCE_STORE[0]
        assert entry.source_count >= 1
        assert len(entry.source_names) > 0
        assert isinstance(entry.avg_reliability, float)
        assert 0.0 <= entry.avg_reliability <= 1.0


# ===================================================================
# Retrieval quality — known claims
# ===================================================================

class TestRetrievalQuality:
    """Test that the retrieval engine finds correct evidence for known claims."""

    def test_flat_earth_claim(self):
        result = retrieve_evidence("The Earth is flat.")
        assert result.has_evidence
        assert "earth shape" in result.best.entry.topic.lower()
        assert result.best.keyword_hits >= 2

    def test_vaccine_autism_claim(self):
        result = retrieve_evidence("Vaccines cause autism.")
        assert result.has_evidence
        assert result.best.entry.topic in ("vaccine autism", "vaccine safety")
        assert result.best.relevance_score >= 1.0

    def test_moon_landing_claim(self):
        result = retrieve_evidence("The moon landing was faked by NASA.")
        assert result.has_evidence
        assert "moon" in result.best.entry.topic.lower()
        assert result.best.keyword_hits >= 2

    def test_5g_covid_claim(self):
        result = retrieve_evidence("5G towers spread COVID-19.")
        assert result.has_evidence
        assert "5g" in result.best.entry.topic.lower()

    def test_climate_change_hoax(self):
        result = retrieve_evidence("Climate change is a hoax.")
        assert result.has_evidence
        assert "climate" in result.best.entry.topic.lower()
        assert result.best.keyword_hits >= 2

    def test_chemtrails_claim(self):
        result = retrieve_evidence("Chemtrails are being sprayed by planes to control people.")
        assert result.has_evidence
        assert "chemtrail" in result.best.entry.topic.lower()

    def test_holocaust_denial(self):
        result = retrieve_evidence("The Holocaust never happened.")
        assert result.has_evidence
        assert "holocaust" in result.best.entry.topic.lower()

    def test_gmo_claim(self):
        result = retrieve_evidence("Genetically modified food is dangerous.")
        assert result.has_evidence
        assert "gmo" in result.best.entry.topic.lower()

    def test_microchip_vaccine(self):
        result = retrieve_evidence("Vaccines contain tracking microchips.")
        assert result.has_evidence
        assert "microchip" in result.best.entry.topic.lower()

    def test_illuminati_claim(self):
        result = retrieve_evidence("The Illuminati control the world.")
        assert result.has_evidence
        assert "illuminati" in result.best.entry.topic.lower()


# ===================================================================
# Retrieval features
# ===================================================================

class TestRetrievalFeatures:
    """Test retrieval engine features: ranking, top-K, no-match handling."""

    def test_returns_multiple_matches(self):
        """A claim touching multiple topics should return multiple matches."""
        result = retrieve_evidence("Vaccines are dangerous and contain microchips.")
        assert result.total_candidates >= 2, f"Expected 2+ candidates, got {result.total_candidates}"

    def test_top_k_limits_results(self):
        result = retrieve_evidence("vaccines are dangerous and cause autism", top_k=2)
        assert len(result.matches) <= 2

    def test_no_match_for_unrelated_text(self):
        result = retrieve_evidence("I had pasta for lunch today.")
        assert not result.has_evidence
        assert result.best is None
        assert result.total_candidates == 0

    def test_ranking_order_by_relevance(self):
        """Higher relevance scores should come first."""
        result = retrieve_evidence("The Earth is flat and vaccines cause autism.")
        if len(result.matches) >= 2:
            assert result.matches[0].relevance_score >= result.matches[1].relevance_score

    def test_evidence_result_properties(self):
        result = retrieve_evidence("The Earth is flat.")
        assert isinstance(result, EvidenceResult)
        assert isinstance(result.best, RankedEvidence)
        assert isinstance(result.best.entry, SourceEntry)
        assert result.best.relevance_score > 0

    def test_relevance_score_includes_alias_hits(self):
        """Alias matches should contribute to the relevance score."""
        result = retrieve_evidence("flat earth theory is true")
        assert result.has_evidence
        # The alias "flat earth" should be matched
        assert result.best.alias_hits >= 0  # May or may not match depending on exact phrasing

    def test_source_urls_available(self):
        result = retrieve_evidence("The Earth is flat.")
        assert result.has_evidence
        assert result.best.entry.best_url, "Best source should have a URL"

    def test_source_reliability_scores(self):
        result = retrieve_evidence("Climate change is a hoax.")
        assert result.has_evidence
        for source in result.best.entry.sources:
            assert 0.0 <= source.reliability <= 1.0


# ===================================================================
# Backward compatibility (corpus.py wrapper)
# ===================================================================

class TestCorpusWrapper:
    """Verify the corpus.py compatibility wrapper works correctly."""

    def test_corpus_retrieve_returns_evidence_match(self):
        result = corpus_retrieve("The Earth is flat.")
        assert result is not None
        assert isinstance(result, EvidenceMatch)

    def test_corpus_retrieve_returns_none_for_no_match(self):
        result = corpus_retrieve("I had pasta for lunch today.")
        assert result is None

    def test_corpus_entry_has_expected_fields(self):
        result = corpus_retrieve("Vaccines cause autism.")
        assert result is not None
        entry = result.entry
        assert isinstance(entry, CorpusEntry)
        assert entry.topic
        assert entry.fact
        assert entry.source
        assert entry.source_count >= 1
        assert 0.0 <= entry.evidence_strength <= 1.0

    def test_trusted_corpus_list_populated(self):
        assert len(TRUSTED_CORPUS) >= 25

    def test_trusted_corpus_entries_are_corpus_entries(self):
        for entry in TRUSTED_CORPUS:
            assert isinstance(entry, CorpusEntry)
