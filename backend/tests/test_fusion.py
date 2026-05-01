"""
Tests for multi-evidence fusion and ranking.

Verifies that:
- Evidence is collected from all source types
- Deduplication merges same-topic corpus+semantic into one source
- Ranking uses trust × relevance weighting
- Agreement detection correctly identifies supporting/conflicting/mixed
- Confidence adjustments are bounded and scale by source_diversity
- End-to-end fusion produces correct FusionResults
"""

import pytest
from app.services.evidence_fusion import (
    EvidenceItem,
    FusionResult,
    collect_evidence,
    deduplicate,
    rank_evidence,
    detect_agreement,
    fuse_evidence,
    _word_set,
)
from app.services.evidence_store import (
    SourceEntry, Source, RankedEvidence, EVIDENCE_STORE,
)
from app.services.semantic_index import SemanticMatch
from app.services.dynamic_retrieval import DynamicEvidence
from app.models.schemas import EvidenceSource


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entry(topic="test_topic", fact="Test fact about the world."):
    return SourceEntry(
        topic=topic,
        category="science",
        keywords=["test"],
        aliases=["test alias"],
        fact=fact,
        sources=[Source(name="TestSource", url="https://test.com")],
        synonyms=["test_syn"],
        evidence_strength=0.9,
    )

def _make_ranked(topic="test_topic", fact="Test fact about the world.",
                 relevance=3.0, keyword_hits=3, keyword_ratio=0.5):
    entry = _make_entry(topic=topic, fact=fact)
    return RankedEvidence(
        entry=entry,
        relevance_score=relevance,
        keyword_hits=keyword_hits,
        alias_hits=1,
        keyword_ratio=keyword_ratio,
    )

def _make_semantic(topic="test_topic", fact="Test fact about the world.",
                   similarity=0.3):
    entry = _make_entry(topic=topic, fact=fact)
    return SemanticMatch(
        entry=entry,
        similarity_score=similarity,
    )

def _make_dynamic(snippet="Dynamic evidence snippet.", title="Test Article",
                  relevance=0.5):
    return DynamicEvidence(
        snippet=snippet,
        title=title,
        url="https://en.wikipedia.org/wiki/Test",
        source_name="Wikipedia",
        relevance_score=relevance,
    )


# ===================================================================
# Collection
# ===================================================================

class TestCollection:
    """Verify evidence is gathered from each source type."""

    def test_collect_corpus_only(self):
        items = collect_evidence(
            corpus_match=_make_ranked(),
            corpus_source=EvidenceSource.CORPUS,
            semantic_matches=None,
            dynamic_evidence=None,
        )
        assert len(items) == 1
        assert items[0].source_type == EvidenceSource.CORPUS

    def test_collect_semantic_only(self):
        items = collect_evidence(
            corpus_match=None,
            corpus_source=EvidenceSource.LLM_ONLY,
            semantic_matches=[_make_semantic()],
            dynamic_evidence=None,
        )
        assert len(items) == 1
        assert items[0].source_type == EvidenceSource.SEMANTIC

    def test_collect_dynamic_only(self):
        items = collect_evidence(
            corpus_match=None,
            corpus_source=EvidenceSource.LLM_ONLY,
            semantic_matches=None,
            dynamic_evidence=_make_dynamic(),
        )
        assert len(items) == 1
        assert items[0].source_type == EvidenceSource.DYNAMIC

    def test_collect_all_sources(self):
        items = collect_evidence(
            corpus_match=_make_ranked(),
            corpus_source=EvidenceSource.CORPUS,
            semantic_matches=[_make_semantic(topic="other_topic", fact="Different fact.")],
            dynamic_evidence=_make_dynamic(),
        )
        assert len(items) == 3
        types = {it.source_type for it in items}
        assert types == {EvidenceSource.CORPUS, EvidenceSource.SEMANTIC, EvidenceSource.DYNAMIC}

    def test_collect_skips_low_relevance_dynamic(self):
        items = collect_evidence(
            corpus_match=None,
            corpus_source=EvidenceSource.LLM_ONLY,
            semantic_matches=None,
            dynamic_evidence=_make_dynamic(relevance=0.10),  # below 0.20 threshold
        )
        assert len(items) == 0

    def test_collect_skips_low_confidence_semantic(self):
        sm = _make_semantic(similarity=0.08)
        sm.is_low_confidence = True
        items = collect_evidence(
            corpus_match=None,
            corpus_source=EvidenceSource.LLM_ONLY,
            semantic_matches=[sm],
            dynamic_evidence=None,
        )
        assert len(items) == 0


# ===================================================================
# Deduplication
# ===================================================================

class TestDeduplication:
    """Verify same-topic corpus+semantic are treated as one source."""

    def test_same_topic_corpus_and_semantic_merged(self):
        """Corpus + semantic with same topic → keep only corpus (higher trust)."""
        items = [
            EvidenceItem(text="The Earth is round.", source_name="NASA",
                         source_url="", source_type=EvidenceSource.CORPUS,
                         relevance_score=0.9, trust_level=0.95, topic="earth shape"),
            EvidenceItem(text="The Earth is round.", source_name="NASA",
                         source_url="", source_type=EvidenceSource.SEMANTIC,
                         relevance_score=0.7, trust_level=0.80, topic="earth shape"),
        ]
        deduped = deduplicate(items)
        assert len(deduped) == 1
        assert deduped[0].source_type == EvidenceSource.CORPUS

    def test_different_topics_kept(self):
        """Different topics from same source type → both kept."""
        items = [
            EvidenceItem(text="Vaccines are safe.", source_name="CDC",
                         source_url="", source_type=EvidenceSource.CORPUS,
                         relevance_score=0.9, trust_level=0.95, topic="vaccine safety"),
            EvidenceItem(text="The Earth is round.", source_name="NASA",
                         source_url="", source_type=EvidenceSource.CORPUS,
                         relevance_score=0.8, trust_level=0.95, topic="earth shape"),
        ]
        deduped = deduplicate(items)
        assert len(deduped) == 2

    def test_word_overlap_dedup(self):
        """Two items with >80% word overlap → keep higher trust."""
        items = [
            EvidenceItem(text="The quick brown fox jumps over the lazy dog daily",
                         source_name="Src1", source_url="",
                         source_type=EvidenceSource.DYNAMIC,
                         relevance_score=0.5, trust_level=0.65, topic="foxes"),
            EvidenceItem(text="The quick brown fox jumps over the lazy dog nightly",
                         source_name="Src2", source_url="",
                         source_type=EvidenceSource.CORPUS,
                         relevance_score=0.8, trust_level=0.95, topic="canines"),
        ]
        deduped = deduplicate(items)
        assert len(deduped) == 1
        assert deduped[0].trust_level == 0.95  # kept higher trust

    def test_single_item_unchanged(self):
        items = [
            EvidenceItem(text="Single evidence.", source_name="Src",
                         source_url="", source_type=EvidenceSource.CORPUS,
                         relevance_score=0.9, trust_level=0.95, topic="test"),
        ]
        deduped = deduplicate(items)
        assert len(deduped) == 1

    def test_empty_items(self):
        assert deduplicate([]) == []


# ===================================================================
# Ranking
# ===================================================================

class TestRanking:
    """Verify trust × relevance weighted ranking."""

    def test_higher_trust_ranks_first(self):
        items = [
            EvidenceItem(text="Low trust", source_name="Src",
                         source_url="", source_type=EvidenceSource.DYNAMIC,
                         relevance_score=0.5, trust_level=0.3, topic="a"),
            EvidenceItem(text="High trust", source_name="Src",
                         source_url="", source_type=EvidenceSource.CORPUS,
                         relevance_score=0.5, trust_level=0.95, topic="b"),
        ]
        ranked = rank_evidence(items)
        assert ranked[0].trust_level == 0.95

    def test_ranking_combines_trust_and_relevance(self):
        items = [
            EvidenceItem(text="Low relevance, high trust", source_name="Src",
                         source_url="", source_type=EvidenceSource.CORPUS,
                         relevance_score=0.1, trust_level=0.95, topic="a"),
            EvidenceItem(text="High relevance, medium trust", source_name="Src",
                         source_url="", source_type=EvidenceSource.SEMANTIC,
                         relevance_score=0.9, trust_level=0.80, topic="b"),
        ]
        ranked = rank_evidence(items)
        # 0.95*0.5 + 0.1*0.5 = 0.525  vs  0.80*0.5 + 0.9*0.5 = 0.85
        assert ranked[0].relevance_score == 0.9

    def test_ranking_preserves_all_items(self):
        items = [
            EvidenceItem(text=f"Item {i}", source_name="Src",
                         source_url="", source_type=EvidenceSource.CORPUS,
                         relevance_score=0.5, trust_level=0.5, topic=f"t{i}")
            for i in range(5)
        ]
        ranked = rank_evidence(items)
        assert len(ranked) == 5


# ===================================================================
# Agreement detection
# ===================================================================

class TestAgreementDetection:
    """Verify supporting/conflicting/mixed/insufficient signals."""

    def test_no_evidence_insufficient(self):
        signal, adj = detect_agreement([])
        assert signal == "insufficient"
        assert adj == 0.0

    def test_single_evidence_insufficient(self):
        items = [
            EvidenceItem(text="Only one source.", source_name="Src",
                         source_url="", source_type=EvidenceSource.CORPUS,
                         relevance_score=0.9, trust_level=0.95, topic="t"),
        ]
        signal, adj = detect_agreement(items)
        assert signal == "insufficient"
        assert adj == 0.0

    def test_agreeing_sources_supporting(self):
        """Two items with high word overlap → supporting."""
        items = [
            EvidenceItem(text="Vaccines are safe and effective according to research",
                         source_name="CDC", source_url="",
                         source_type=EvidenceSource.CORPUS,
                         relevance_score=0.9, trust_level=0.95, topic="vax"),
            EvidenceItem(text="Vaccines are safe and effective based on research evidence",
                         source_name="Wikipedia", source_url="",
                         source_type=EvidenceSource.DYNAMIC,
                         relevance_score=0.7, trust_level=0.65, topic="vax_wiki"),
        ]
        signal, adj = detect_agreement(items)
        assert signal == "supporting"
        assert adj > 0

    def test_conflicting_sources(self):
        """Two items with very low word overlap → conflicting."""
        items = [
            EvidenceItem(text="The planet Mercury orbits closest to the sun in our solar system",
                         source_name="NASA", source_url="",
                         source_type=EvidenceSource.CORPUS,
                         relevance_score=0.9, trust_level=0.95, topic="mercury"),
            EvidenceItem(text="Bananas provide potassium and dietary fiber for nutrition",
                         source_name="Wikipedia", source_url="",
                         source_type=EvidenceSource.DYNAMIC,
                         relevance_score=0.5, trust_level=0.65, topic="bananas"),
        ]
        signal, adj = detect_agreement(items)
        assert signal in ("conflicting", "insufficient")
        assert adj <= 0

    def test_diversity_scales_adjustment(self):
        """More diverse source types → larger adjustment magnitude."""
        # 1 type
        items_1 = [
            EvidenceItem(text="Fact one about science and research",
                         source_name="S1", source_url="",
                         source_type=EvidenceSource.CORPUS,
                         relevance_score=0.9, trust_level=0.95, topic="a"),
            EvidenceItem(text="Fact one about science and research confirmed",
                         source_name="S2", source_url="",
                         source_type=EvidenceSource.CORPUS,
                         relevance_score=0.8, trust_level=0.95, topic="b"),
        ]
        # 2 types
        items_2 = [
            EvidenceItem(text="Fact one about science and research",
                         source_name="S1", source_url="",
                         source_type=EvidenceSource.CORPUS,
                         relevance_score=0.9, trust_level=0.95, topic="a"),
            EvidenceItem(text="Fact one about science and research confirmed",
                         source_name="S2", source_url="",
                         source_type=EvidenceSource.DYNAMIC,
                         relevance_score=0.8, trust_level=0.65, topic="b"),
        ]
        _, adj_1 = detect_agreement(items_1)
        _, adj_2 = detect_agreement(items_2)
        # 2-type diversity should give >= 1-type adjustment
        assert adj_2 >= adj_1


# ===================================================================
# Confidence adjustment bounds
# ===================================================================

class TestConfidenceAdjustment:
    """Verify adjustment is bounded between -0.15 and +0.10."""

    def test_supporting_adjustment_positive(self):
        items = [
            EvidenceItem(text="Vaccines are safe and effective in studies",
                         source_name="CDC", source_url="",
                         source_type=EvidenceSource.CORPUS,
                         relevance_score=0.9, trust_level=0.95, topic="a"),
            EvidenceItem(text="Vaccines are safe and effective for protection",
                         source_name="WHO", source_url="",
                         source_type=EvidenceSource.DYNAMIC,
                         relevance_score=0.8, trust_level=0.65, topic="b"),
        ]
        _, adj = detect_agreement(items)
        assert 0.0 < adj <= 0.10

    def test_conflicting_adjustment_negative(self):
        items = [
            EvidenceItem(text="The planet Mercury orbits closest to the sun in our solar system",
                         source_name="NASA", source_url="",
                         source_type=EvidenceSource.CORPUS,
                         relevance_score=0.9, trust_level=0.95, topic="a"),
            EvidenceItem(text="Bananas provide potassium and dietary fiber for nutrition",
                         source_name="Wikipedia", source_url="",
                         source_type=EvidenceSource.DYNAMIC,
                         relevance_score=0.5, trust_level=0.65, topic="b"),
        ]
        _, adj = detect_agreement(items)
        assert adj <= 0

    def test_adjustment_bounded_positive(self):
        """Max positive adjustment should not exceed 0.10."""
        items = [
            EvidenceItem(text="Climate change is real and caused by humans burning fossil fuels",
                         source_name="S1", source_url="",
                         source_type=EvidenceSource.CORPUS,
                         relevance_score=0.9, trust_level=0.95, topic="a"),
            EvidenceItem(text="Climate change is real and caused by humans burning fossil fuels globally",
                         source_name="S2", source_url="",
                         source_type=EvidenceSource.SEMANTIC,
                         relevance_score=0.8, trust_level=0.80, topic="b"),
            EvidenceItem(text="Climate change is real and caused by humans burning fossil fuels confirmed",
                         source_name="S3", source_url="",
                         source_type=EvidenceSource.DYNAMIC,
                         relevance_score=0.7, trust_level=0.65, topic="c"),
        ]
        _, adj = detect_agreement(items)
        assert adj <= 0.10

    def test_adjustment_bounded_negative(self):
        """Max negative adjustment should not exceed -0.15."""
        items = [
            EvidenceItem(text="The sun rises in the east direction every morning", source_name="S1",
                         source_url="", source_type=EvidenceSource.CORPUS,
                         relevance_score=0.9, trust_level=0.95, topic="a"),
            EvidenceItem(text="Bananas grow on trees in tropical regions worldwide", source_name="S2",
                         source_url="", source_type=EvidenceSource.DYNAMIC,
                         relevance_score=0.5, trust_level=0.65, topic="b"),
            EvidenceItem(text="Quantum computing uses qubits for parallel processing tasks", source_name="S3",
                         source_url="", source_type=EvidenceSource.SEMANTIC,
                         relevance_score=0.6, trust_level=0.80, topic="c"),
        ]
        _, adj = detect_agreement(items)
        assert adj >= -0.15


# ===================================================================
# End-to-end fusion
# ===================================================================

class TestFuseEvidence:
    """Verify the full fusion pipeline."""

    def test_no_evidence_returns_empty(self):
        result = fuse_evidence(
            corpus_match=None,
            corpus_source=EvidenceSource.LLM_ONLY,
            semantic_matches=None,
            dynamic_evidence=None,
        )
        assert result.items == []
        assert result.source_count == 0
        assert result.source_diversity == 0
        assert result.agreement_signal == "insufficient"

    def test_single_corpus_match(self):
        result = fuse_evidence(
            corpus_match=_make_ranked(),
            corpus_source=EvidenceSource.CORPUS,
            semantic_matches=None,
            dynamic_evidence=None,
        )
        assert len(result.items) == 1
        assert result.source_count == 1
        assert result.source_diversity == 1
        assert result.agreement_signal == "insufficient"

    def test_corpus_plus_semantic_same_topic_deduped(self):
        """Corpus + semantic with same topic → 1 item after dedup."""
        corpus = _make_ranked(topic="vaccine autism", fact="Vaccines do not cause autism.")
        semantic = [_make_semantic(topic="vaccine autism", fact="Vaccines do not cause autism.")]
        result = fuse_evidence(
            corpus_match=corpus,
            corpus_source=EvidenceSource.CORPUS,
            semantic_matches=semantic,
            dynamic_evidence=None,
        )
        # Same topic → merged into one
        assert len(result.items) == 1
        assert result.source_diversity == 1

    def test_corpus_plus_dynamic_different(self):
        """Corpus + dynamic with different content → 2 items."""
        result = fuse_evidence(
            corpus_match=_make_ranked(topic="earth shape", fact="The Earth is round."),
            corpus_source=EvidenceSource.CORPUS,
            semantic_matches=None,
            dynamic_evidence=_make_dynamic(snippet="Entirely different text about bananas."),
        )
        assert len(result.items) == 2
        assert result.source_diversity == 2

    def test_fusion_items_capped_at_5(self):
        """Fusion should return at most 5 items."""
        semantics = [
            _make_semantic(topic=f"topic_{i}", fact=f"Fact number {i} about the world.")
            for i in range(10)
        ]
        result = fuse_evidence(
            corpus_match=None,
            corpus_source=EvidenceSource.LLM_ONLY,
            semantic_matches=semantics,
            dynamic_evidence=None,
        )
        assert len(result.items) <= 5

    def test_fusion_result_has_summary(self):
        result = fuse_evidence(
            corpus_match=_make_ranked(),
            corpus_source=EvidenceSource.CORPUS,
            semantic_matches=None,
            dynamic_evidence=None,
        )
        assert len(result.fusion_summary) > 0

    def test_grounding_rules_preserved(self):
        """FusionResult items keep their source_type, so grounding is unaffected."""
        result = fuse_evidence(
            corpus_match=_make_ranked(),
            corpus_source=EvidenceSource.CORPUS,
            semantic_matches=None,
            dynamic_evidence=None,
        )
        assert result.items[0].source_type == EvidenceSource.CORPUS


# ===================================================================
# Schema backward compatibility
# ===================================================================

class TestSchemaCompat:
    """Verify new ClaimResult fields have safe defaults."""

    def test_claim_result_defaults(self):
        from app.models.schemas import ClaimResult, ClaimStatus, ClaimType
        cr = ClaimResult(
            claim="Test",
            status=ClaimStatus.UNKNOWN,
            evidence="No evidence",
            confidence=0.5,
        )
        assert cr.evidence_items == []
        assert cr.source_count == 0

    def test_evidence_item_response_creation(self):
        from app.models.schemas import EvidenceItemResponse
        item = EvidenceItemResponse(
            text="Test evidence",
            source_name="CDC",
            source_type="corpus",
            role="supporting",
        )
        assert item.text == "Test evidence"
        assert item.source_url == ""
