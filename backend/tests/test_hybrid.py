"""
Tests for hybrid retrieval: static corpus + dynamic Wikipedia fallback.

Tests cover:
- Dynamic retrieval module (WikipediaRetriever) with mocked HTTP
- Hybrid flow scenarios (static-only, dynamic fallback, both-miss)
- Dynamic evidence grounding rules
- Config flag (enable_dynamic_retrieval)
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from dataclasses import dataclass

from app.services.dynamic_retrieval import (
    WikipediaRetriever,
    DynamicEvidence,
    _extract_search_terms,
    _score_relevance,
    _truncate_snippet,
    retrieve_dynamic_evidence,
)
from app.services.analyzer import (
    _classify_evidence_source,
    _apply_evidence_grounding,
    _build_evidence_display,
)
from app.models.schemas import EvidenceSource, ClaimResult, ClaimStatus


# ===================================================================
# Search term extraction
# ===================================================================

class TestSearchTermExtraction:
    """Test that search terms are extracted correctly from claims."""

    def test_removes_stop_words(self):
        terms = _extract_search_terms("The Earth is flat and round")
        assert "the" not in terms
        assert "is" not in terms
        assert "and" not in terms
        assert "earth" in terms
        assert "flat" in terms

    def test_removes_short_tokens(self):
        terms = _extract_search_terms("I am a big fan of x y z things")
        assert "x" not in terms  # single character
        assert "a" not in terms  # single character
        assert "big" in terms    # meaningful word kept

    def test_preserves_meaningful_words(self):
        terms = _extract_search_terms("Vaccines cause autism in children")
        assert "vaccines" in terms
        assert "cause" in terms
        assert "autism" in terms
        assert "children" in terms

    def test_handles_empty_string(self):
        terms = _extract_search_terms("")
        assert terms == []

    def test_handles_only_stop_words(self):
        terms = _extract_search_terms("the is a an")
        assert terms == []


# ===================================================================
# Relevance scoring
# ===================================================================

class TestRelevanceScoring:
    """Test relevance scoring of Wikipedia extracts against search terms."""

    def test_perfect_match(self):
        score = _score_relevance(
            ["earth", "flat", "round"],
            "The Earth is round, not flat. It is an oblate spheroid."
        )
        assert score == 1.0

    def test_partial_match(self):
        score = _score_relevance(
            ["earth", "flat", "ocean", "deep"],
            "The Earth is an oblate spheroid."
        )
        assert 0.0 < score < 1.0

    def test_no_match(self):
        score = _score_relevance(
            ["quantum", "entanglement"],
            "Pizza is a popular food worldwide."
        )
        assert score == 0.0

    def test_empty_terms(self):
        score = _score_relevance([], "Some text here")
        assert score == 0.0

    def test_empty_extract(self):
        score = _score_relevance(["earth"], "")
        assert score == 0.0


# ===================================================================
# Snippet truncation
# ===================================================================

class TestTruncateSnippet:
    """Test text truncation logic."""

    def test_short_text_unchanged(self):
        text = "Short text."
        assert _truncate_snippet(text, max_len=500) == text

    def test_long_text_truncated(self):
        text = "A" * 600
        result = _truncate_snippet(text, max_len=500)
        assert len(result) <= 501  # +1 for ellipsis

    def test_truncates_at_sentence_boundary(self):
        text = "First sentence here. Second sentence is quite a bit longer and goes well beyond the character limit we set. Third sentence follows."
        result = _truncate_snippet(text, max_len=80)
        # Should break at a sentence boundary or add ellipsis
        assert result.endswith(".") or result.endswith("…")


# ===================================================================
# WikipediaRetriever with mocked HTTP
# ===================================================================

class TestWikipediaRetriever:
    """Test the Wikipedia retriever with mocked API responses."""

    @pytest.fixture
    def retriever(self):
        return WikipediaRetriever()

    @pytest.mark.asyncio
    async def test_successful_search(self, retriever):
        """Mock a successful Wikipedia API search + extract."""
        search_response = {
            "query": {
                "search": [
                    {"title": "Speed of light"}
                ]
            }
        }
        extract_response = {
            "query": {
                "pages": {
                    "12345": {
                        "title": "Speed of light",
                        "extract": "The speed of light in vacuum is 299,792,458 metres per second. This is a universal physical constant important in many areas of physics."
                    }
                }
            }
        }

        mock_responses = [
            MagicMock(status_code=200, json=lambda: search_response, raise_for_status=lambda: None),
            MagicMock(status_code=200, json=lambda: extract_response, raise_for_status=lambda: None),
        ]

        with patch("app.services.dynamic_retrieval.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(side_effect=mock_responses)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            result = await retriever._do_search(
                "The speed of light is 300,000 km/s"
            )

        assert result is not None
        assert isinstance(result, DynamicEvidence)
        assert result.source_name == "Wikipedia"
        assert "speed" in result.snippet.lower() or "light" in result.snippet.lower()
        assert result.url.startswith("https://en.wikipedia.org/wiki/")
        assert result.relevance_score > 0

    @pytest.mark.asyncio
    async def test_no_results_returns_none(self, retriever):
        """Wikipedia returns empty search results."""
        search_response = {"query": {"search": []}}

        with patch("app.services.dynamic_retrieval.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(
                return_value=MagicMock(
                    status_code=200,
                    json=lambda: search_response,
                    raise_for_status=lambda: None,
                )
            )
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            result = await retriever._do_search("xylplthon causes znorbification")

        assert result is None

    @pytest.mark.asyncio
    async def test_timeout_returns_none(self, retriever):
        """Network timeout should be handled gracefully."""
        import httpx

        with patch("app.services.dynamic_retrieval.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            result = await retriever._do_search("Speed of light")

        assert result is None

    @pytest.mark.asyncio
    async def test_caching_works(self, retriever):
        """Second call with same claim should use cache."""
        evidence = DynamicEvidence(
            snippet="Test snippet",
            title="Test",
            url="https://en.wikipedia.org/wiki/Test",
            source_name="Wikipedia",
            relevance_score=0.5,
        )
        retriever._cache["test claim"] = evidence

        result = await retriever.search("Test claim")  # Case-insensitive
        assert result is evidence  # Same object from cache

    @pytest.mark.asyncio
    async def test_low_relevance_filtered(self, retriever):
        """Wikipedia result with low relevance score should be filtered out."""
        search_response = {
            "query": {"search": [{"title": "Pizza"}]}
        }
        extract_response = {
            "query": {
                "pages": {
                    "999": {
                        "title": "Pizza",
                        "extract": "Pizza is a dish of Italian origin consisting of a usually round, flat base of leavened wheat-based dough topped with tomato sauce, cheese, and various other ingredients."
                    }
                }
            }
        }

        mock_responses = [
            MagicMock(status_code=200, json=lambda: search_response, raise_for_status=lambda: None),
            MagicMock(status_code=200, json=lambda: extract_response, raise_for_status=lambda: None),
        ]

        with patch("app.services.dynamic_retrieval.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(side_effect=mock_responses)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            # Claim about quantum physics → pizza extract = low relevance
            result = await retriever._do_search("Quantum entanglement is faster than light")

        assert result is None  # Filtered by low relevance


# ===================================================================
# Dynamic evidence grounding rules
# ===================================================================

class TestDynamicGrounding:
    """Test that dynamic evidence is grounded with appropriate trust level."""

    def test_dynamic_supported_allowed(self):
        """Dynamic evidence CAN produce Supported (unlike weak_corpus)."""
        status, conf, note = _apply_evidence_grounding(
            "Supported", 0.85, EvidenceSource.DYNAMIC, ""
        )
        assert status == "Supported"  # Allowed!
        assert conf <= 0.70  # But capped

    def test_dynamic_contradicted_allowed(self):
        status, conf, note = _apply_evidence_grounding(
            "Contradicted", 0.88, EvidenceSource.DYNAMIC, ""
        )
        assert status == "Contradicted"  # Allowed
        assert conf <= 0.70  # Capped

    def test_dynamic_confidence_never_exceeds_cap(self):
        """Even with very high raw confidence, dynamic should be capped."""
        status, conf, note = _apply_evidence_grounding(
            "Supported", 0.99, EvidenceSource.DYNAMIC, ""
        )
        assert conf <= 0.70

    def test_dynamic_unknown_stays_unknown(self):
        status, conf, note = _apply_evidence_grounding(
            "Unknown", 0.50, EvidenceSource.DYNAMIC, ""
        )
        assert status == "Unknown"
        assert conf <= 0.70

    def test_dynamic_note_mentions_wikipedia(self):
        _, _, note = _apply_evidence_grounding(
            "Supported", 0.80, EvidenceSource.DYNAMIC, ""
        )
        assert "Wikipedia" in note or "dynamic" in note.lower()


# ===================================================================
# Dynamic evidence display
# ===================================================================

class TestDynamicDisplay:
    """Test display strings for dynamic evidence."""

    def test_dynamic_display_labels_wikipedia(self):
        display = _build_evidence_display(
            EvidenceSource.DYNAMIC,
            "The speed of light is 299,792,458 m/s.",
            "Wikipedia",
            "https://en.wikipedia.org/wiki/Speed_of_light",
            "",
        )
        assert "[Wikipedia]" in display
        assert "299,792,458" in display
        assert "https://en.wikipedia.org" in display

    def test_dynamic_display_without_url(self):
        display = _build_evidence_display(
            EvidenceSource.DYNAMIC,
            "Some evidence text.",
            "Wikipedia",
            "",
            "",
        )
        assert "[Wikipedia]" in display
        assert "Some evidence text." in display


# ===================================================================
# EvidenceSource enum
# ===================================================================

class TestEvidenceSourceEnum:
    """Test that the DYNAMIC value exists in the enum."""

    def test_dynamic_value_exists(self):
        assert EvidenceSource.DYNAMIC.value == "dynamic"

    def test_claim_result_with_dynamic_source(self):
        result = ClaimResult(
            claim="Test.",
            status=ClaimStatus.SUPPORTED,
            evidence="Wikipedia says so.",
            evidence_source=EvidenceSource.DYNAMIC,
            confidence=0.65,
        )
        data = result.model_dump()
        assert data["evidence_source"] == "dynamic"


# ===================================================================
# Hybrid flow scenarios (integration-style)
# ===================================================================

class TestHybridFlowScenarios:
    """Test the overall hybrid retrieval decision logic."""

    def test_corpus_hit_skips_dynamic(self):
        """When corpus has a strong match, dynamic should not be needed."""
        from app.services.evidence_store import retrieve_evidence
        result = retrieve_evidence("The Earth is flat.")
        source = _classify_evidence_source(result.best)
        # Strong corpus hit → no dynamic needed
        assert source == EvidenceSource.CORPUS

    def test_corpus_miss_triggers_dynamic_eligibility(self):
        """When corpus misses, the evidence source should be LLM_ONLY → eligible for dynamic."""
        from app.services.evidence_store import retrieve_evidence
        result = retrieve_evidence("The Eiffel Tower was completed in 1889.")
        source = _classify_evidence_source(result.best)
        # No corpus match → LLM_ONLY → should trigger dynamic fallback
        assert source in (EvidenceSource.LLM_ONLY, EvidenceSource.WEAK_CORPUS)

    def test_dynamic_evidence_dataclass(self):
        """DynamicEvidence dataclass should have all required fields."""
        evidence = DynamicEvidence(
            snippet="Test snippet",
            title="Test Title",
            url="https://example.com",
            source_name="Wikipedia",
            relevance_score=0.75,
        )
        assert evidence.snippet == "Test snippet"
        assert evidence.source_name == "Wikipedia"
        assert evidence.relevance_score == 0.75
