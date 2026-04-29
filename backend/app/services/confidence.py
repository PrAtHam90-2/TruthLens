"""
Confidence calibration engine for TruthLens.

Transforms raw LLM confidence into a calibrated score based on
multiple factors: evidence strength, source count, keyword match
quality, and claim clarity.

Design goals:
- Typical output range: 0.50 – 0.88
- Avoid extremes (>0.92 or <0.30) unless very warranted
- Always produce a human-readable reason for the assigned confidence
"""

import math
import logging
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceFactors:
    """All inputs that influence the final confidence score."""
    llm_raw_confidence: float        # Raw confidence from the LLM (0.0–1.0)
    has_corpus_evidence: bool        # Whether the corpus had a match
    corpus_evidence_strength: float  # Strength of the corpus entry (0.0–1.0)
    corpus_source_count: int         # How many sources back the corpus entry
    keyword_match_ratio: float       # How well the claim matched corpus keywords (0.0–1.0)
    claim_word_count: int            # Number of words in the claim (proxy for specificity)
    llm_and_corpus_agree: bool       # Whether LLM verdict aligns with corpus direction


@dataclass
class CalibratedConfidence:
    """Output of the calibration engine."""
    score: float           # Final calibrated confidence (0.0–1.0)
    reason: str            # Human-readable explanation of why this confidence was assigned


def calibrate(factors: ConfidenceFactors) -> CalibratedConfidence:
    """
    Compute a calibrated confidence score from multiple quality signals.
    
    The algorithm:
    1. Dampen the raw LLM confidence (pull toward 0.6 center)
    2. Apply bonuses/penalties for evidence quality
    3. Clamp to a realistic range
    4. Generate a human-readable reason
    """
    reasons = []

    # --- Step 1: Dampen raw LLM confidence ---
    # LLMs tend to be overconfident. Apply a sigmoid-like dampening
    # that compresses extreme values toward the center.
    raw = factors.llm_raw_confidence
    dampened = _dampen(raw, center=0.62, strength=0.45)
    reasons.append(f"Base LLM confidence: {raw:.0%} → dampened to {dampened:.0%}")

    score = dampened

    # --- Step 2: Evidence quality adjustments ---
    if factors.has_corpus_evidence:
        # Bonus for having corpus evidence
        evidence_bonus = factors.corpus_evidence_strength * 0.12
        score += evidence_bonus
        reasons.append(f"Corpus evidence found (strength: {factors.corpus_evidence_strength:.0%}, +{evidence_bonus:.0%})")

        # Bonus for multiple independent sources
        if factors.corpus_source_count >= 4:
            source_bonus = 0.06
            score += source_bonus
            reasons.append(f"Strong source backing ({factors.corpus_source_count} sources, +{source_bonus:.0%})")
        elif factors.corpus_source_count >= 2:
            source_bonus = 0.03
            score += source_bonus
            reasons.append(f"Moderate source backing ({factors.corpus_source_count} sources, +{source_bonus:.0%})")
        else:
            reasons.append(f"Limited source backing ({factors.corpus_source_count} source)")

        # Bonus for good keyword match
        if factors.keyword_match_ratio >= 0.5:
            match_bonus = 0.04
            score += match_bonus
            reasons.append(f"Strong keyword match ({factors.keyword_match_ratio:.0%}, +{match_bonus:.0%})")
        elif factors.keyword_match_ratio >= 0.25:
            match_bonus = 0.02
            score += match_bonus
            reasons.append(f"Partial keyword match ({factors.keyword_match_ratio:.0%}, +{match_bonus:.0%})")

        # Bonus for LLM + corpus agreement
        if factors.llm_and_corpus_agree:
            agree_bonus = 0.05
            score += agree_bonus
            reasons.append(f"LLM and corpus evidence agree (+{agree_bonus:.0%})")
        else:
            disagree_penalty = -0.08
            score += disagree_penalty
            reasons.append(f"LLM and corpus evidence disagree ({disagree_penalty:.0%})")

    else:
        # No corpus evidence — significant penalty
        no_evidence_penalty = -0.15
        score += no_evidence_penalty
        reasons.append(f"No corpus evidence found ({no_evidence_penalty:.0%})")

    # --- Step 3: Claim clarity adjustment ---
    if factors.claim_word_count < 5:
        clarity_penalty = -0.05
        score += clarity_penalty
        reasons.append(f"Very short claim — may lack context ({clarity_penalty:.0%})")
    elif factors.claim_word_count > 30:
        clarity_penalty = -0.03
        score += clarity_penalty
        reasons.append(f"Long compound claim — harder to verify precisely ({clarity_penalty:.0%})")

    # --- Step 4: Clamp to realistic range ---
    # Hard floor at 0.30, soft ceiling at 0.92
    score = max(0.30, min(0.92, score))

    # Round to 2 decimals
    score = round(score, 2)

    reason = "; ".join(reasons)
    return CalibratedConfidence(score=score, reason=reason)


def _dampen(raw: float, center: float = 0.62, strength: float = 0.45) -> float:
    """
    Pull a raw confidence value toward a center point.
    
    strength controls how aggressively:
      0.0 = no dampening (return raw)
      1.0 = always return center
    """
    return center + (raw - center) * (1 - strength)


def calibrate_overall(
    claim_scores: list[float],
    claim_statuses: list[str],
) -> float:
    """
    Compute an overall confidence score from individual claim scores.
    
    Uses a weighted approach that penalizes disagreement between claims
    and avoids naive averaging that hides uncertainty.
    """
    if not claim_scores:
        return 0.0

    n = len(claim_scores)
    avg = sum(claim_scores) / n

    # Penalty for mixed verdicts (disagreement between claims)
    unique_statuses = set(claim_statuses)
    if len(unique_statuses) > 1 and "Unknown" not in unique_statuses:
        # Mix of Supported/Contradicted = less overall confidence
        disagreement_penalty = 0.08 * (len(unique_statuses) - 1)
        avg -= disagreement_penalty

    # Penalty for having any Unknown claims
    unknown_count = claim_statuses.count("Unknown")
    if unknown_count > 0:
        unknown_ratio = unknown_count / n
        avg -= 0.10 * unknown_ratio

    return round(max(0.30, min(0.90, avg)), 2)
