"""
Core analysis orchestrator.

Pipeline:
1. Extract atomic claims from input text (LLM, with heuristic fallback).
2. Classify each claim as Factual / Opinion / Unverifiable.
3. For FACTUAL claims: retrieve evidence (hybrid), classify, calibrate confidence.
4. For OPINION/UNVERIFIABLE claims: skip verification, assign Unverifiable status.
5. **Evidence grounding**: prevent Supported/Contradicted without trusted evidence.
6. Compute an overall verdict and confidence.

Hybrid retrieval:
  Static corpus → Semantic (TF-IDF) → Dynamic (Wikipedia) fallback → LLM-only

Evidence grounding rules:
- CORPUS (strong):   LLM verdict trusted, no cap.
- SEMANTIC:          Supported/Contradicted allowed, confidence capped at 0.72.
- DYNAMIC:           Supported/Contradicted allowed, confidence capped at 0.70.
- WEAK_CORPUS:       Supported → Unknown. Confidence capped at 0.60.
- LLM_ONLY:          Always Unknown. Confidence capped at 0.50.
"""

import logging
from typing import List

from app.models.schemas import (
    AnalyzeResponse, ClaimResult, ClaimStatus, ClaimType, EvidenceSource,
    EvidenceItemResponse,
)
from app.services.llm_client import GroqLLMClient
from app.services.evidence_store import retrieve_evidence, EvidenceResult
from app.services.dynamic_retrieval import retrieve_dynamic_evidence, DynamicEvidence
from app.services.semantic_index import semantic_search, SemanticMatch
from app.services.fallback import extract_claims_heuristic
from app.services.confidence import (
    calibrate,
    calibrate_overall,
    ConfidenceFactors,
)
from app.core.config import get_settings
from app.services.evidence_fusion import fuse_evidence, FusionResult

logger = logging.getLogger(__name__)

# Singleton LLM client (initialised once at module load)
_llm_client = GroqLLMClient()

# Thresholds for "strong" vs "weak" corpus match
_STRONG_EVIDENCE_STRENGTH = 0.7
_STRONG_KEYWORD_RATIO = 0.28
_STRONG_MIN_KEYWORD_HITS = 2
_STRONG_MIN_RELEVANCE_SCORE = 1.5
_NO_CORPUS_CONFIDENCE_CAP = 0.50
_WEAK_CORPUS_CONFIDENCE_CAP = 0.60
_DYNAMIC_CONFIDENCE_CAP = 0.70
_SEMANTIC_CONFIDENCE_CAP = 0.72


async def analyze_text(text: str) -> AnalyzeResponse:
    """
    Full analysis pipeline for the given input text.
    Returns an AnalyzeResponse with verdict, claims, and explanation.
    """
    # ------------------------------------------------------------------
    # Step 1: Extract claims (LLM → heuristic fallback)
    # ------------------------------------------------------------------
    extraction_method = "LLM"
    try:
        claims = await _llm_client.extract_claims(text)
        if not claims:
            raise ValueError("LLM returned empty claims list")
    except Exception as e:
        logger.warning(f"LLM claim extraction failed ({e}), using heuristic fallback.")
        claims = extract_claims_heuristic(text)
        extraction_method = "heuristic"

    # ------------------------------------------------------------------
    # Step 2–5: Classify type → Route → Verify → Ground
    # ------------------------------------------------------------------
    claim_results: List[ClaimResult] = []

    for claim_text in claims:
        # --- Step 2: Classify claim type (Factual / Opinion / Unverifiable) ---
        # Try heuristic first (fast, reliable for obvious cases), then LLM
        claim_type = ClaimType.FACTUAL  # default
        type_reason = ""

        heuristic_result = _heuristic_classify_type(claim_text)
        if heuristic_result:
            claim_type = heuristic_result["claim_type"]
            type_reason = heuristic_result["reason"]
        else:
            try:
                type_result = await _llm_client.classify_claim_type(claim_text)
                claim_type = ClaimType(type_result["claim_type"])
                type_reason = type_result.get("reason", "")
            except Exception as e:
                logger.warning(f"Claim type classification failed for '{claim_text[:50]}…': {e}")
                claim_type = ClaimType.FACTUAL

        # --- Step 3: Route based on claim type ---
        if claim_type in (ClaimType.OPINION, ClaimType.UNVERIFIABLE):
            claim_results.append(
                ClaimResult(
                    claim=claim_text,
                    claim_type=claim_type,
                    status=ClaimStatus.UNVERIFIABLE,
                    evidence=f"This claim is {claim_type.value.lower()} and cannot be objectively verified. {type_reason}",
                    evidence_source=EvidenceSource.NONE,
                    confidence=0.55,
                    confidence_reason=f"Low confidence assigned because this is a {claim_type.value.lower()} claim, not a verifiable factual statement.",
                )
            )
            continue

        # --- Step 4: Full verification pipeline for FACTUAL claims ---
        # Step 4a: Static retrieval (corpus)
        evidence_result: EvidenceResult = retrieve_evidence(claim_text)
        has_evidence = evidence_result.has_evidence
        best_match = evidence_result.best

        source_entry = best_match.entry if best_match else None
        evidence_text = source_entry.fact if source_entry else None
        source_names = source_entry.source_names if source_entry else None
        source_url = source_entry.best_url if source_entry else ""

        evidence_source = _classify_evidence_source(best_match)

        # Step 4a²: Semantic retrieval (TF-IDF) if keyword match is weak/missing
        settings = get_settings()
        semantic_matches_all = []  # Keep all matches for fusion
        if evidence_source in (EvidenceSource.LLM_ONLY, EvidenceSource.WEAK_CORPUS):
            if settings.enable_semantic_retrieval:
                try:
                    semantic_matches_all = semantic_search(
                        claim_text,
                        min_similarity=settings.semantic_min_similarity,
                    )
                    if semantic_matches_all:
                        sem_best = semantic_matches_all[0]
                        if sem_best.is_low_confidence:
                            # Score too low for confident matching — skip, let fallback handle
                            logger.info(
                                f"Low-confidence semantic match for '{claim_text[:50]}…': "
                                f"score={sem_best.similarity_score}, skipping to fallback"
                            )
                        elif sem_best.is_ambiguous:
                            # Ambiguous match → downgrade to WEAK_CORPUS
                            source_entry = sem_best.entry
                            evidence_text = source_entry.fact
                            source_names = source_entry.source_names
                            source_url = source_entry.best_url
                            evidence_source = EvidenceSource.WEAK_CORPUS
                            has_evidence = True
                            best_match = None
                            logger.info(
                                f"Ambiguous semantic match for '{claim_text[:50]}…': "
                                f"downgraded to WEAK_CORPUS"
                            )
                        else:
                            # Upgrade to SEMANTIC source
                            source_entry = sem_best.entry
                            evidence_text = source_entry.fact
                            source_names = source_entry.source_names
                            source_url = source_entry.best_url
                            evidence_source = EvidenceSource.SEMANTIC
                            has_evidence = True
                            best_match = None  # Clear keyword match (semantic overrides)
                            logger.info(
                                f"Semantic match for '{claim_text[:50]}…': "
                                f"{source_entry.topic} (similarity={sem_best.similarity_score})"
                            )
                except Exception as e:
                    logger.warning(f"Semantic retrieval failed for '{claim_text[:50]}…': {e}")

        # Step 4b: Dynamic fallback (Wikipedia) if still weak/missing
        dynamic_evidence: DynamicEvidence | None = None
        if evidence_source in (EvidenceSource.LLM_ONLY, EvidenceSource.WEAK_CORPUS):
            if settings.enable_dynamic_retrieval:
                try:
                    dynamic_evidence = await retrieve_dynamic_evidence(claim_text)
                except Exception as e:
                    logger.warning(f"Dynamic retrieval failed for '{claim_text[:50]}…': {e}")

                if dynamic_evidence and dynamic_evidence.relevance_score >= 0.20:
                    # Dynamic evidence is good enough — upgrade source
                    evidence_text = dynamic_evidence.snippet
                    source_names = dynamic_evidence.source_name
                    source_url = dynamic_evidence.url
                    evidence_source = EvidenceSource.DYNAMIC
                    has_evidence = True
                    logger.info(
                        f"Dynamic evidence used for '{claim_text[:50]}…': "
                        f"{dynamic_evidence.title} (relevance={dynamic_evidence.relevance_score})"
                    )

        # Step 4c: EVIDENCE FUSION — combine all sources
        fusion = fuse_evidence(
            corpus_match=evidence_result.best,
            corpus_source=_classify_evidence_source(evidence_result.best),
            semantic_matches=semantic_matches_all or None,
            dynamic_evidence=dynamic_evidence,
        )
        fusion_items = [
            EvidenceItemResponse(
                text=it.text[:200],
                source_name=it.source_name,
                source_url=it.source_url,
                source_type=it.source_type.value,
                role=it.role,
            )
            for it in fusion.items
        ]

        # Classify via LLM (with fallback)
        try:
            classification = await _llm_client.classify_claim(
                claim=claim_text,
                evidence=evidence_text,
            )
            status_str = classification["status"]
            raw_confidence = classification["confidence"]
            explanation = classification["explanation"]
            llm_confidence_reason = classification.get("confidence_reason", "")

            # Build evidence display string
            evidence_display = _build_evidence_display(
                evidence_source, evidence_text, source_names, source_url, explanation,
            )

            # ============================================================
            # Step 5: EVIDENCE GROUNDING — enforce trust boundaries
            # ============================================================
            status_str, raw_confidence, grounding_note = _apply_evidence_grounding(
                status_str, raw_confidence, evidence_source, llm_confidence_reason,
            )

            # Determine if LLM and corpus agree
            llm_corpus_agree = True
            if has_evidence and source_entry:
                if status_str in ("Unknown", "Mixed") and source_entry.evidence_strength > 0.8:
                    llm_corpus_agree = False

            # --- Calibrate confidence ---
            factors = ConfidenceFactors(
                llm_raw_confidence=raw_confidence,
                has_corpus_evidence=has_evidence,
                corpus_evidence_strength=source_entry.evidence_strength if source_entry else 0.0,
                corpus_source_count=source_entry.source_count if source_entry else 0,
                keyword_match_ratio=best_match.keyword_ratio if best_match else 0.0,
                claim_word_count=len(claim_text.split()),
                llm_and_corpus_agree=llm_corpus_agree,
            )
            calibrated = calibrate(factors)
            final_confidence = calibrated.score

            # Apply confidence caps for ungrounded verdicts
            if evidence_source == EvidenceSource.LLM_ONLY:
                final_confidence = min(final_confidence, _NO_CORPUS_CONFIDENCE_CAP)
            elif evidence_source == EvidenceSource.DYNAMIC:
                final_confidence = min(final_confidence, _DYNAMIC_CONFIDENCE_CAP)
            elif evidence_source == EvidenceSource.WEAK_CORPUS:
                final_confidence = min(final_confidence, _WEAK_CORPUS_CONFIDENCE_CAP)

            # Apply fusion confidence adjustment (scaled by source_diversity)
            if fusion.confidence_adjustment != 0.0:
                final_confidence = max(0.0, min(1.0,
                    final_confidence + fusion.confidence_adjustment
                ))

            # Build confidence reason
            confidence_parts = []
            if llm_confidence_reason:
                confidence_parts.append(llm_confidence_reason)
            if grounding_note:
                confidence_parts.append(grounding_note)
            confidence_parts.append(f"[Calibration: {calibrated.reason}]")
            if fusion.agreement_signal != "insufficient":
                confidence_parts.append(f"[Fusion: {fusion.fusion_summary}]")
            confidence_reason = " ".join(confidence_parts)

        except Exception as e:
            logger.warning(f"LLM classification failed for claim '{claim_text[:50]}…': {e}")
            # Fallback classification when LLM is unavailable
            if source_entry:
                status_str = "Mixed"
                final_confidence = 0.45
                evidence_display = f"{evidence_text} (Source: {source_names})"
                confidence_reason = "Classified via heuristic fallback — LLM unavailable."
            else:
                status_str = "Unknown"
                final_confidence = 0.30
                evidence_display = "No matching evidence found in evidence store."
                evidence_source = EvidenceSource.NONE
                confidence_reason = "Could not verify — no evidence match and LLM unavailable."

        claim_results.append(
            ClaimResult(
                claim=claim_text,
                claim_type=ClaimType.FACTUAL,
                status=ClaimStatus(status_str),
                evidence=evidence_display,
                evidence_source=evidence_source,
                confidence=round(final_confidence, 2),
                confidence_reason=confidence_reason,
                evidence_items=fusion_items,
                source_count=fusion.source_count,
            )
        )

    # ------------------------------------------------------------------
    # Step 6: Compute overall verdict
    # ------------------------------------------------------------------
    verdict, overall_confidence, explanation = _compute_overall_verdict(claim_results)

    # Build uncertainty note
    uncertainty_parts = []
    if extraction_method == "heuristic":
        uncertainty_parts.append("Claims were extracted using heuristics (LLM unavailable).")

    non_factual = sum(1 for c in claim_results if c.claim_type != ClaimType.FACTUAL)
    if non_factual > 0:
        uncertainty_parts.append(
            f"{non_factual} claim(s) were classified as opinion/unverifiable and skipped from fact-checking."
        )

    # Note ungrounded claims
    llm_only = sum(1 for c in claim_results if c.evidence_source == EvidenceSource.LLM_ONLY)
    if llm_only > 0:
        uncertainty_parts.append(
            f"{llm_only} claim(s) had no trusted evidence and were assessed using LLM reasoning only."
        )

    dynamic_count = sum(1 for c in claim_results if c.evidence_source == EvidenceSource.DYNAMIC)
    if dynamic_count > 0:
        uncertainty_parts.append(
            f"{dynamic_count} claim(s) used dynamically retrieved evidence (Wikipedia)."
        )

    uncertainty_parts.append(
        "Analysis is based on a limited trusted corpus and LLM reasoning. "
        "Results should be treated as indicative, not definitive."
    )
    uncertainty_note = " ".join(uncertainty_parts)

    return AnalyzeResponse(
        verdict=verdict,
        confidence_score=round(overall_confidence, 2),
        uncertainty_note=uncertainty_note,
        explanation=explanation,
        claims=claim_results,
    )


# ---------------------------------------------------------------------------
# Heuristic claim-type pre-classifier
# ---------------------------------------------------------------------------

# Patterns that strongly indicate FACTUAL claims (testable assertions)
_FACTUAL_VERB_PATTERNS = [
    "causes", "cause", "caused",
    "contains", "contain", "contained",
    "spreads", "spread",
    "prevents", "prevent",
    "cures", "cure", "cured",
    "kills", "kill", "killed",
    "originated", "originates",
    "invented", "created",
    "was faked", "is faked", "was fake", "is fake", "is a hoax", "was a hoax",
    "was staged", "is staged",
    "is flat", "is round", "is spherical",
    "never happened", "didn't happen", "did not happen",
    "does not exist", "doesn't exist", "don't exist",
    "is real", "are real", "is not real", "are not real",
    "is safe", "is unsafe", "is dangerous",
    "is effective", "is ineffective",
    "is made of", "is composed of",
    "was discovered", "was proven",
    "boils at", "melts at", "freezes at",
    "orbits", "revolves around",
]

# Patterns that strongly indicate OPINION claims (value judgments)
_OPINION_PATTERNS = [
    "cannot be trusted", "can't be trusted",
    "is the best", "is the worst",
    "is corrupt", "are corrupt",
    "is dishonest", "are dishonest",
    "should", "shouldn't",
    "is better than", "is worse than",
    "is disgusting", "is amazing", "is terrible", "is wonderful",
    "is overrated", "is underrated",
]

# Prefixes that indicate opinion (when the core claim is also subjective)
_OPINION_PREFIXES = [
    "i think", "i believe", "i feel", "in my opinion",
    "personally", "in my view",
]

# Patterns that indicate UNVERIFIABLE claims
_UNVERIFIABLE_PATTERNS = [
    "something big is coming",
    "the truth will come out",
    "everything happens for a reason",
    "they don't want you to know",
    "they don't want us to know",
    "time will tell",
    "mark my words",
    "wait and see",
    "you'll see",
]


def _heuristic_classify_type(claim: str) -> dict | None:
    """
    Fast rule-based claim type classification for obvious cases.

    Returns {"claim_type": ClaimType, "reason": str} or None if uncertain.
    When None, the caller should fall back to LLM classification.

    Key principle: A claim is FACTUAL if it makes a testable assertion about
    reality, even if the assertion is false or a conspiracy theory.
    """
    lower = claim.lower().strip().rstrip(".")

    # --- Check UNVERIFIABLE first (vague/prophetic) ---
    for pattern in _UNVERIFIABLE_PATTERNS:
        if pattern in lower:
            return {
                "claim_type": ClaimType.UNVERIFIABLE,
                "reason": "This claim is too vague or speculative to verify with evidence.",
            }

    # --- Check for FACTUAL verb patterns (strongest signal) ---
    for pattern in _FACTUAL_VERB_PATTERNS:
        if pattern in lower:
            return {
                "claim_type": ClaimType.FACTUAL,
                "reason": f"Contains a testable assertion ('{pattern}') about the physical world.",
            }

    # --- Check for pure OPINION patterns ---
    for pattern in _OPINION_PATTERNS:
        if pattern in lower:
            return {
                "claim_type": ClaimType.OPINION,
                "reason": "This expresses a subjective value judgment, not a testable fact.",
            }

    # --- Handle "I think X" hedging ---
    for prefix in _OPINION_PREFIXES:
        if lower.startswith(prefix):
            # Strip the prefix and check if the core claim is factual
            core = lower[len(prefix):].strip().lstrip(",").strip()
            for pattern in _FACTUAL_VERB_PATTERNS:
                if pattern in core:
                    return {
                        "claim_type": ClaimType.FACTUAL,
                        "reason": "The hedging prefix does not change the testable nature of the core claim.",
                    }
            # If core isn't clearly factual, it's likely opinion
            return {
                "claim_type": ClaimType.OPINION,
                "reason": "Personal belief/opinion marker with a subjective core claim.",
            }

    # --- Uncertain: defer to LLM ---
    return None


# ---------------------------------------------------------------------------
# Evidence grounding helpers
# ---------------------------------------------------------------------------

def _classify_evidence_source(best_match) -> EvidenceSource:
    """
    Determine the quality tier of the evidence match.

    A match is STRONG (corpus) only if ALL of:
    - evidence_strength >= 0.7
    - keyword_ratio >= 0.28
    - keyword_hits >= 2
    - relevance_score >= 1.5

    Otherwise it's WEAK_CORPUS, which cannot produce Supported/Contradicted.
    If there's no match at all, it's LLM_ONLY.
    """
    if best_match is None:
        return EvidenceSource.LLM_ONLY

    entry = best_match.entry
    is_strong = (
        entry.evidence_strength >= _STRONG_EVIDENCE_STRENGTH
        and best_match.keyword_ratio >= _STRONG_KEYWORD_RATIO
        and best_match.keyword_hits >= _STRONG_MIN_KEYWORD_HITS
        and best_match.relevance_score >= _STRONG_MIN_RELEVANCE_SCORE
    )

    if is_strong:
        return EvidenceSource.CORPUS
    else:
        return EvidenceSource.WEAK_CORPUS


def _apply_evidence_grounding(
    status: str,
    confidence: float,
    evidence_source: EvidenceSource,
    llm_reason: str,
) -> tuple[str, float, str]:
    """
    Enforce evidence grounding rules. Returns (new_status, new_confidence, grounding_note).

    Rules:
    - CORPUS:      LLM verdict trusted as-is.
    - DYNAMIC:     Supported/Contradicted allowed, confidence capped at 0.70.
    - WEAK_CORPUS: Supported → Unknown, Contradicted stays but confidence capped.
    - LLM_ONLY:    Supported → Unknown, Contradicted → Unknown.
    """
    grounding_note = ""

    if evidence_source == EvidenceSource.CORPUS:
        grounding_note = "Verdict grounded in trusted corpus evidence."
        return status, confidence, grounding_note

    if evidence_source == EvidenceSource.SEMANTIC:
        # Semantic match from corpus — trusted text, fuzzy match
        grounding_note = (
            "Verdict grounded in corpus evidence via semantic matching. "
            "Confidence capped at 72% for fuzzy matches."
        )
        return status, min(confidence, _SEMANTIC_CONFIDENCE_CAP), grounding_note

    if evidence_source == EvidenceSource.DYNAMIC:
        # Dynamic evidence (Wikipedia) — moderate trust
        grounding_note = (
            "Verdict supported by dynamically retrieved evidence (Wikipedia). "
            "Confidence capped at 70% for non-corpus sources."
        )
        return status, min(confidence, _DYNAMIC_CONFIDENCE_CAP), grounding_note

    if evidence_source == EvidenceSource.WEAK_CORPUS:
        if status == "Supported":
            grounding_note = (
                "[Downgraded from Supported → Unknown] "
                "Corpus match was weak — insufficient to confirm the claim."
            )
            return "Unknown", min(confidence, 0.55), grounding_note
        elif status == "Contradicted":
            grounding_note = (
                "[Evidence weak] Corpus match is partial — "
                "contradicting evidence exists but match quality is low."
            )
            return status, min(confidence, 0.60), grounding_note
        else:
            grounding_note = "Weak corpus match — confidence reduced."
            return status, min(confidence, 0.60), grounding_note

    # LLM_ONLY — no trusted evidence at all
    if status in ("Supported", "Contradicted"):
        original = status
        grounding_note = (
            f"[Downgraded from {original} → Unknown] "
            "No trusted evidence found. LLM reasoning alone is not sufficient "
            "to verify or refute this claim."
        )
        return "Unknown", min(confidence, 0.45), grounding_note
    else:
        grounding_note = "No trusted evidence available — assessed by LLM reasoning only."
        return status, min(confidence, 0.50), grounding_note


def _build_evidence_display(
    evidence_source: EvidenceSource,
    evidence_text: str | None,
    source_names: str | None,
    source_url: str,
    llm_explanation: str,
) -> str:
    """Build a display string that clearly labels evidence provenance."""
    if evidence_source == EvidenceSource.CORPUS:
        display = f"{evidence_text} (Source: {source_names})"
        if source_url:
            display += f" [{source_url}]"
        return display

    if evidence_source == EvidenceSource.SEMANTIC:
        display = f"[Semantic match] {evidence_text} (Source: {source_names})"
        if source_url:
            display += f" [{source_url}]"
        return display

    if evidence_source == EvidenceSource.DYNAMIC:
        display = f"[Wikipedia] {evidence_text}"
        if source_url:
            display += f" [{source_url}]"
        return display

    if evidence_source == EvidenceSource.WEAK_CORPUS:
        display = f"[Partial match] {evidence_text} (Source: {source_names})"
        if source_url:
            display += f" [{source_url}]"
        return display

    # LLM_ONLY or NONE
    return f"[No trusted evidence] {llm_explanation}"


# ---------------------------------------------------------------------------
# Overall verdict computation
# ---------------------------------------------------------------------------

def _compute_overall_verdict(
    claims: List[ClaimResult],
) -> tuple[ClaimStatus, float, str]:
    """
    Derive an overall verdict from individual claim results.

    Aggregation rules (conservative — weakest claim drives the verdict):

    1. No claims at all → UNKNOWN
    2. All claims opinion/unverifiable (no factual) → UNVERIFIABLE
    3. Any factual claim is Contradicted → overall CONTRADICTED
    4. All factual claims Supported → overall SUPPORTED
    5. All factual claims Unknown → overall UNKNOWN
    6. Mixture of Supported + Unknown/Mixed → overall MIXED
    7. Mixture of Contradicted + Unknown → overall CONTRADICTED
    8. Any other mixture → overall MIXED

    Key rule: SUPPORTED is only returned when ALL factual claims are Supported.
    A single Unknown or ungrounded claim prevents a positive verdict.
    """
    if not claims:
        return (
            ClaimStatus.UNKNOWN,
            0.0,
            "No verifiable claims were found in the text.",
        )

    # Only consider factual claims for the overall verdict
    factual_claims = [c for c in claims if c.claim_type == ClaimType.FACTUAL]

    # If ALL claims are opinion/unverifiable, return Unverifiable verdict
    if not factual_claims:
        avg_conf = sum(c.confidence for c in claims) / len(claims)
        return (
            ClaimStatus.UNVERIFIABLE,
            round(avg_conf, 2),
            "All extracted claims are opinions or unverifiable statements. No factual claims to verify.",
        )

    status_counts = {s: 0 for s in ClaimStatus}
    for c in factual_claims:
        status_counts[c.status] += 1

    n = len(factual_claims)
    supported = status_counts[ClaimStatus.SUPPORTED]
    contradicted = status_counts[ClaimStatus.CONTRADICTED]
    unknown = status_counts[ClaimStatus.UNKNOWN]
    mixed = status_counts[ClaimStatus.MIXED]
    unverifiable = status_counts[ClaimStatus.UNVERIFIABLE]

    # Use calibrated overall confidence (factual claims only)
    overall_confidence = calibrate_overall(
        claim_scores=[c.confidence for c in factual_claims],
        claim_statuses=[c.status.value for c in factual_claims],
    )

    # ------------------------------------------------------------------
    # Decision logic (conservative: weakest claim drives the verdict)
    # ------------------------------------------------------------------

    # Rule 3: Any contradiction → overall Contradicted
    if contradicted > 0:
        if supported > 0:
            verdict = ClaimStatus.MIXED
            explanation = (
                f"The text contains {supported} supported and {contradicted} contradicted "
                f"factual claim(s) out of {n}. The presence of contradicted claims makes "
                f"the overall assessment mixed."
            )
        else:
            verdict = ClaimStatus.CONTRADICTED
            explanation = (
                f"{contradicted} of {n} factual claim(s) are contradicted by trusted evidence."
            )

    # Rule 4: ALL claims Supported → only case where Supported is returned
    elif supported == n:
        verdict = ClaimStatus.SUPPORTED
        explanation = "All factual claims in the text are supported by trusted evidence."

    # Rule 5: ALL claims Unknown → overall Unknown
    elif unknown == n:
        verdict = ClaimStatus.UNKNOWN
        explanation = "None of the factual claims could be verified against the trusted corpus."

    # Rule 6: Supported + Unknown/Mixed/Unverifiable → MIXED (not Supported!)
    elif supported > 0 and (unknown > 0 or mixed > 0 or unverifiable > 0):
        unverified = unknown + mixed + unverifiable
        verdict = ClaimStatus.MIXED
        explanation = (
            f"{supported} of {n} factual claim(s) are supported, but {unverified} "
            f"could not be fully verified. Overall verdict is mixed because not all "
            f"claims have trusted evidence."
        )
        # Penalize confidence when some claims lack evidence
        penalty = (unverified / n) * 0.15
        overall_confidence = max(0.30, overall_confidence - penalty)

    # Rule 8: Everything else → MIXED
    else:
        verdict = ClaimStatus.MIXED
        explanation = "The analysis produced mixed results across the factual claims."

    # Append non-factual note if applicable
    non_factual = len(claims) - len(factual_claims)
    if non_factual > 0:
        explanation += f" ({non_factual} opinion/unverifiable claim(s) were excluded from the verdict.)"

    return verdict, round(overall_confidence, 2), explanation

