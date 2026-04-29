"""
API route definitions for TruthLens.
"""

from fastapi import APIRouter, HTTPException
import logging

from app.models.schemas import AnalyzeRequest, AnalyzeResponse
from app.services.analyzer import analyze_text

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["analysis"])


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Analyze a piece of text for misinformation.

    Extracts factual claims, retrieves evidence from a trusted corpus,
    and classifies each claim as Supported / Contradicted / Mixed / Unknown.
    """
    try:
        result = await analyze_text(request.text)
        return result
    except Exception as e:
        logger.exception("Analysis failed")
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@router.get("/health")
async def health():
    """Simple health-check endpoint."""
    return {"status": "ok", "service": "TruthLens API"}
