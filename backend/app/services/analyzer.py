"""
Core analysis orchestrator.

Pipeline:
1. Extract atomic claims from input text (LLM, with heuristic fallback).
2. Classify each claim as Factual / Opinion / Unverifiable.
3. For FACTUAL claims: retrieve evidence, classify, calibrate confidence.
4. For OPINION/UNVERIFIABLE claims: skip verification, assign Unverifiable status.
5. Compute an overall verdict and confidence.
"""

import logging
from typing import List

from app.models.schemas import AnalyzeResponse, ClaimResult, ClaimStatus, ClaimType
from app.services.llm_client import GroqLLMClient
from app.services.corpus import retrieve_evidence
from app.services.fallback import extract_claims_heuristic
from app.services.confidence import (
    calibrate,
    calibrate_overall,
    ConfidenceFactors,
)

logger = logging.getLogger(__name__)

# Singleton LLM client (initialised once at module load)
_llm_client = GroqLLMClient()


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
    # Step 2–4: Classify type → Route → Verify factual claims
    # ------------------------------------------------------------------
    claim_results: List[ClaimResult] = []

    for claim_text in claims:
        # --- Step 2: Classify claim type (Factual / Opinion / Unverifiable) ---
        claim_type = ClaimType.FACTUAL  # default
        type_reason = ""
        try:
            type_result = await _llm_client.classify_claim_type(claim_text)
            claim_type = ClaimType(type_result["claim_type"])
            type_reason = type_result.get("reason", "")
        except Exception as e:
            logger.warning(f"Claim type classification failed for '{claim_text[:50]}…': {e}")
            # Default to Factual so it goes through verification
            claim_type = ClaimType.FACTUAL

        # --- Step 3: Route based on claim type ---
        if claim_type in (ClaimType.OPINION, ClaimType.UNVERIFIABLE):
            # Skip evidence retrieval and fact-checking for non-factual claims
            claim_results.append(
                ClaimResult(
                    claim=claim_text,
                    claim_type=claim_type,
                    status=ClaimStatus.UNVERIFIABLE,
                    evidence=f"This claim is {claim_type.value.lower()} and cannot be objectively verified. {type_reason}",
                    confidence=0.55,
                    confidence_reason=f"Low confidence assigned because this is a {claim_type.value.lower()} claim, not a verifiable factual statement.",
                )
            )
            continue

        # --- Step 4: Full verification pipeline for FACTUAL claims ---
        # Retrieve evidence from corpus
        evidence_match = retrieve_evidence(claim_text)
        has_corpus = evidence_match is not None

        corpus_entry = evidence_match.entry if evidence_match else None
        evidence_text = corpus_entry.fact if corpus_entry else None
        source = corpus_entry.source if corpus_entry else None

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
            if evidence_text:
                evidence_display = f"{evidence_text} (Source: {source})"
            else:
                evidence_display = explanation

            # Determine if LLM and corpus agree
            llm_corpus_agree = True  # default
            if has_corpus:
                if status_str in ("Unknown", "Mixed") and corpus_entry.evidence_strength > 0.8:
                    llm_corpus_agree = False

            # --- Calibrate confidence ---
            factors = ConfidenceFactors(
                llm_raw_confidence=raw_confidence,
                has_corpus_evidence=has_corpus,
                corpus_evidence_strength=corpus_entry.evidence_strength if corpus_entry else 0.0,
                corpus_source_count=corpus_entry.source_count if corpus_entry else 0,
                keyword_match_ratio=evidence_match.keyword_ratio if evidence_match else 0.0,
                claim_word_count=len(claim_text.split()),
                llm_and_corpus_agree=llm_corpus_agree,
            )
            calibrated = calibrate(factors)
            final_confidence = calibrated.score
            confidence_reason = f"{llm_confidence_reason} [Calibration: {calibrated.reason}]" if llm_confidence_reason else calibrated.reason

        except Exception as e:
            logger.warning(f"LLM classification failed for claim '{claim_text[:50]}…': {e}")
            # Fallback classification when LLM is unavailable
            if corpus_entry:
                status_str = "Mixed"
                final_confidence = 0.45
                evidence_display = f"{evidence_text} (Source: {source})"
                confidence_reason = "Classified via heuristic fallback — LLM unavailable. Low confidence due to lack of LLM reasoning."
            else:
                status_str = "Unknown"
                final_confidence = 0.30
                evidence_display = "No matching evidence found in trusted corpus."
                confidence_reason = "Could not verify — no corpus match and LLM unavailable."

        claim_results.append(
            ClaimResult(
                claim=claim_text,
                claim_type=ClaimType.FACTUAL,
                status=ClaimStatus(status_str),
                evidence=evidence_display,
                confidence=round(final_confidence, 2),
                confidence_reason=confidence_reason,
            )
        )

    # ------------------------------------------------------------------
    # Step 5: Compute overall verdict
    # ------------------------------------------------------------------
    verdict, overall_confidence, explanation = _compute_overall_verdict(claim_results)

    # Build uncertainty note
    uncertainty_parts = []
    if extraction_method == "heuristic":
        uncertainty_parts.append("Claims were extracted using heuristics (LLM unavailable).")

    # Note if any claims were non-factual
    non_factual = sum(1 for c in claim_results if c.claim_type != ClaimType.FACTUAL)
    if non_factual > 0:
        uncertainty_parts.append(
            f"{non_factual} claim(s) were classified as opinion/unverifiable and skipped from fact-checking."
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


def _compute_overall_verdict(
    claims: List[ClaimResult],
) -> tuple[ClaimStatus, float, str]:
    """Derive an overall verdict from individual claim results."""
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

    # Use calibrated overall confidence (factual claims only)
    overall_confidence = calibrate_overall(
        claim_scores=[c.confidence for c in factual_claims],
        claim_statuses=[c.status.value for c in factual_claims],
    )

    # Decision logic
    if status_counts[ClaimStatus.CONTRADICTED] == n:
        verdict = ClaimStatus.CONTRADICTED
        explanation = "All factual claims in the text are contradicted by trusted evidence."
    elif status_counts[ClaimStatus.SUPPORTED] == n:
        verdict = ClaimStatus.SUPPORTED
        explanation = "All factual claims in the text are supported by trusted evidence."
    elif status_counts[ClaimStatus.UNKNOWN] == n:
        verdict = ClaimStatus.UNKNOWN
        explanation = "None of the factual claims could be verified against the trusted corpus."
    elif status_counts[ClaimStatus.CONTRADICTED] > 0 and status_counts[ClaimStatus.SUPPORTED] > 0:
        verdict = ClaimStatus.MIXED
        explanation = (
            f"The text contains a mix of supported ({status_counts[ClaimStatus.SUPPORTED]}) "
            f"and contradicted ({status_counts[ClaimStatus.CONTRADICTED]}) factual claims."
        )
    elif status_counts[ClaimStatus.CONTRADICTED] > 0:
        verdict = ClaimStatus.CONTRADICTED
        explanation = (
            f"{status_counts[ClaimStatus.CONTRADICTED]} of {n} factual claims are contradicted by evidence."
        )
    elif status_counts[ClaimStatus.SUPPORTED] > 0:
        verdict = ClaimStatus.SUPPORTED
        explanation = (
            f"{status_counts[ClaimStatus.SUPPORTED]} of {n} factual claims are supported by evidence."
        )
    else:
        verdict = ClaimStatus.MIXED
        explanation = "The analysis produced mixed results across the factual claims."

    # Append non-factual note if applicable
    non_factual = len(claims) - len(factual_claims)
    if non_factual > 0:
        explanation += f" ({non_factual} opinion/unverifiable claim(s) were excluded from the verdict.)"

    return verdict, overall_confidence, explanation
