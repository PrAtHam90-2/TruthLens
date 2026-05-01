"""
Pydantic schemas for API request and response validation.
"""

from pydantic import BaseModel, Field
from enum import Enum
from typing import List, Optional


class ClaimStatus(str, Enum):
    """Possible verdicts for an individual claim."""
    SUPPORTED = "Supported"
    CONTRADICTED = "Contradicted"
    MIXED = "Mixed"
    UNKNOWN = "Unknown"
    UNVERIFIABLE = "Unverifiable"


class ClaimType(str, Enum):
    """Classification of what kind of claim was extracted."""
    FACTUAL = "Factual"
    OPINION = "Opinion"
    UNVERIFIABLE = "Unverifiable"


class EvidenceSource(str, Enum):
    """Where the evidence for a verdict came from."""
    CORPUS = "corpus"           # Strong trusted evidence from the evidence store
    SEMANTIC = "semantic"       # Semantic (TF-IDF) match from the corpus — fuzzy but trusted text
    DYNAMIC = "dynamic"         # Dynamic retrieval (e.g. Wikipedia) — moderate trust
    WEAK_CORPUS = "weak_corpus" # Corpus match, but low relevance or evidence strength
    LLM_ONLY = "llm_only"      # LLM reasoning without trusted evidence (not proof)
    NONE = "none"               # No evidence at all


class AnalyzeRequest(BaseModel):
    """Request body for the /analyze endpoint."""
    text: str = Field(
        ...,
        min_length=10,
        max_length=5000,
        description="The text to analyze for misinformation.",
        examples=["The Earth is flat and the moon landing was faked."],
    )


class ClaimResult(BaseModel):
    """Analysis result for a single extracted claim."""
    claim: str = Field(..., description="The atomic factual claim extracted from the text.")
    claim_type: ClaimType = Field(
        default=ClaimType.FACTUAL,
        description="Whether the claim is factual, opinion-based, or unverifiable.",
    )
    status: ClaimStatus = Field(..., description="Classification of the claim.")
    evidence: str = Field(..., description="Supporting or refuting evidence snippet.")
    evidence_source: EvidenceSource = Field(
        default=EvidenceSource.NONE,
        description="Where the evidence came from (corpus, weak_corpus, llm_only, none).",
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Calibrated confidence score for this classification."
    )
    confidence_reason: str = Field(
        default="", description="Human-readable explanation of why this confidence level was assigned."
    )
    evidence_items: List['EvidenceItemResponse'] = Field(
        default_factory=list, description="Individual evidence items from multi-source fusion."
    )
    source_count: int = Field(
        default=0, description="Number of unique evidence sources used."
    )


class EvidenceItemResponse(BaseModel):
    """A single evidence item in the fusion result."""
    text: str = Field(..., description="The evidence snippet.")
    source_name: str = Field(..., description="Name of the source (e.g. CDC, Wikipedia).")
    source_url: str = Field(default="", description="URL of the source.")
    source_type: str = Field(..., description="Evidence source type (corpus, semantic, dynamic, etc.).")
    role: str = Field(default="supporting", description="Role of this evidence: supporting, conflicting, or neutral.")


class AnalyzeResponse(BaseModel):
    """Full response for the /analyze endpoint."""
    verdict: ClaimStatus = Field(..., description="Overall verdict for the input text.")
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Overall confidence score."
    )
    uncertainty_note: str = Field(
        ..., description="Transparency note about analysis limitations."
    )
    explanation: str = Field(
        ..., description="Human-readable explanation of the overall verdict."
    )
    claims: List[ClaimResult] = Field(
        ..., description="List of individual claim analyses."
    )
