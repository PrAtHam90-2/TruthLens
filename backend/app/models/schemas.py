"""
Pydantic schemas for API request and response validation.
"""

from pydantic import BaseModel, Field
from enum import Enum
from typing import List


class ClaimStatus(str, Enum):
    """Possible verdicts for an individual claim."""
    SUPPORTED = "Supported"
    CONTRADICTED = "Contradicted"
    MIXED = "Mixed"
    UNKNOWN = "Unknown"


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
    status: ClaimStatus = Field(..., description="Classification of the claim.")
    evidence: str = Field(..., description="Supporting or refuting evidence snippet.")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Calibrated confidence score for this classification."
    )
    confidence_reason: str = Field(
        default="", description="Human-readable explanation of why this confidence level was assigned."
    )


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
