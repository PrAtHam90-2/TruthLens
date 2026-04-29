"""
LLM client abstraction and Groq (Llama 3 70B) implementation.

The interface is intentionally simple so it can be swapped for
OpenAI, Gemini, Anthropic, or any other provider with minimal changes.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import json
import logging
from groq import Groq

from app.core.config import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------
class BaseLLMClient(ABC):
    """Abstract base class for LLM-backed claim extraction and classification."""

    @abstractmethod
    async def extract_claims(self, text: str) -> List[str]:
        """Extract atomic factual claims from the input text."""
        ...

    @abstractmethod
    async def classify_claim(self, claim: str, evidence: Optional[str] = None) -> dict:
        """
        Classify a claim given evidence.
        Returns: {"status": "Supported|Contradicted|Mixed|Unknown",
                  "confidence": 0.0-1.0,
                  "explanation": "..."}
        """
        ...


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
EXTRACT_SYSTEM = "You are a fact-checking assistant that extracts atomic factual claims from text."

EXTRACT_PROMPT = """Extract atomic factual claims from the following text.

Rules:
- Each claim must be a single, self-contained factual statement.
- Do NOT include opinions, questions, or subjective statements.
- Return ONLY a JSON array of strings. No markdown, no explanation, no preamble.

Text:
\"\"\"
{text}
\"\"\"

Example output:
["The Earth is flat.", "Vaccines cause autism."]
"""

CLASSIFY_SYSTEM = "You are a careful, conservative fact-checking assistant. You classify claims against evidence and always explain your reasoning. You avoid overconfidence."

CLASSIFY_PROMPT = """Given a CLAIM and EVIDENCE from a trusted source, classify the claim.

CLAIM: {claim}

EVIDENCE: {evidence}

IMPORTANT CONFIDENCE GUIDELINES:
- 0.50-0.65: Weak or indirect evidence, partial match
- 0.65-0.80: Moderate evidence, mostly clear
- 0.80-0.90: Strong, direct evidence from multiple sources
- Above 0.90: ONLY for universally established scientific facts with overwhelming consensus
- Never output 0.95+ unless the evidence is absolutely incontrovertible

Respond with ONLY a JSON object (no markdown, no explanation, no preamble) with these fields:
- "status": one of "Supported", "Contradicted", "Mixed", or "Unknown"
- "confidence": a float between 0.5 and 0.9 (see guidelines above)
- "explanation": a brief 1-2 sentence explanation of your verdict
- "confidence_reason": a brief sentence explaining WHY you assigned that specific confidence level

Example:
{{"status": "Contradicted", "confidence": 0.82, "explanation": "The claim directly contradicts established scientific evidence.", "confidence_reason": "High confidence because multiple major scientific organizations confirm the counter-evidence, but leaving room for nuance."}}
"""

CLASSIFY_NO_EVIDENCE_PROMPT = """Given a CLAIM but NO matching evidence in our trusted corpus, assess the claim using your general knowledge.

CLAIM: {claim}

IMPORTANT CONFIDENCE GUIDELINES:
- Without corpus evidence, your confidence should be LOWER (typically 0.40-0.65)
- 0.40-0.50: Uncertain, relying only on general knowledge
- 0.50-0.65: Fairly confident from general knowledge, but no corpus backing
- Above 0.65: ONLY if this is a widely known, unambiguous fact
- If genuinely unsure, use "Unknown" with confidence around 0.40

Respond with ONLY a JSON object (no markdown, no explanation, no preamble) with these fields:
- "status": one of "Supported", "Contradicted", "Mixed", or "Unknown"
- "confidence": a float between 0.35 and 0.70 (see guidelines above)
- "explanation": a brief 1-2 sentence explanation
- "confidence_reason": a brief sentence explaining WHY you assigned that specific confidence level

Be conservative: if you're unsure, use "Unknown" with low confidence.
"""


# ---------------------------------------------------------------------------
# Groq implementation (Llama 3 70B)
# ---------------------------------------------------------------------------
class GroqLLMClient(BaseLLMClient):
    """Groq-backed LLM client using Llama 3 70B."""

    def __init__(self):
        settings = get_settings()
        if not settings.groq_api_key:
            logger.warning("GROQ_API_KEY not set — LLM calls will fail, falling back to heuristics.")
            self._client = None
            return

        self._client = Groq(api_key=settings.groq_api_key)
        self._model = settings.groq_model
        logger.info(f"Groq LLM client initialized with model: {self._model}")

    def _is_available(self) -> bool:
        return self._client is not None

    def _clean_json_response(self, text: str) -> str:
        """Strip markdown fences and whitespace from LLM output."""
        text = text.strip()
        if text.startswith("```"):
            # Remove opening fence (```json or ```)
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    def _chat(self, system: str, user: str) -> str:
        """Send a chat completion request to Groq and return the response text."""
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
            max_tokens=1024,
        )
        return response.choices[0].message.content

    async def extract_claims(self, text: str) -> List[str]:
        if not self._is_available():
            raise RuntimeError("Groq client not configured")

        prompt = EXTRACT_PROMPT.format(text=text)
        raw_response = self._chat(EXTRACT_SYSTEM, prompt)
        raw = self._clean_json_response(raw_response)

        try:
            claims = json.loads(raw)
            if isinstance(claims, list) and all(isinstance(c, str) for c in claims):
                return claims
            raise ValueError("Unexpected JSON structure")
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse LLM claim extraction: {e}\nRaw: {raw}")
            raise RuntimeError(f"LLM returned unparseable response: {raw}") from e

    async def classify_claim(self, claim: str, evidence: Optional[str] = None) -> dict:
        if not self._is_available():
            raise RuntimeError("Groq client not configured")

        if evidence:
            prompt = CLASSIFY_PROMPT.format(claim=claim, evidence=evidence)
        else:
            prompt = CLASSIFY_NO_EVIDENCE_PROMPT.format(claim=claim)

        raw_response = self._chat(CLASSIFY_SYSTEM, prompt)
        raw = self._clean_json_response(raw_response)

        try:
            result = json.loads(raw)
            status = result.get("status", "Unknown")
            confidence = float(result.get("confidence", 0.5))
            explanation = result.get("explanation", "No explanation provided.")
            confidence_reason = result.get("confidence_reason", "")

            valid_statuses = {"Supported", "Contradicted", "Mixed", "Unknown"}
            if status not in valid_statuses:
                status = "Unknown"

            confidence = max(0.0, min(1.0, confidence))

            return {
                "status": status,
                "confidence": confidence,
                "explanation": explanation,
                "confidence_reason": confidence_reason,
            }
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse LLM classification: {e}\nRaw: {raw}")
            raise RuntimeError(f"LLM returned unparseable response: {raw}") from e
