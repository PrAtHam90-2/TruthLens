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
    async def classify_claim_type(self, claim: str) -> dict:
        """
        Determine whether a claim is factual, opinion, or unverifiable.
        Returns: {"claim_type": "Factual|Opinion|Unverifiable",
                  "reason": "..."}
        """
        ...

    @abstractmethod
    async def classify_claim(self, claim: str, evidence: Optional[str] = None) -> dict:
        """
        Classify a claim given evidence.
        Returns: {"status": "Supported|Contradicted|Mixed|Unknown",
                  "confidence": 0.0-1.0,
                  "explanation": "...",
                  "confidence_reason": "..."}
        """
        ...


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
EXTRACT_SYSTEM = """You are an expert fact-checking analyst. Your job is to decompose text into atomic, independently verifiable factual claims.

You are meticulous: you never merge unrelated facts, you always split compound statements, and you cleanly separate facts from opinions. You output valid JSON only."""

EXTRACT_PROMPT = """Extract every atomic factual claim from the text below.

## Rules

1. **One fact per claim.** Each claim must contain exactly ONE independently verifiable factual assertion.
2. **Split compound sentences.** If a sentence contains "and", "or", "but", "while", "also", "moreover", or any conjunction linking separate facts, split them into separate claims.
   - "The Earth is flat and vaccines cause autism" → TWO claims, not one.
3. **Ignore opinions.** Do NOT extract subjective opinions, personal feelings, value judgments, or preferences. Phrases like "I think", "I believe", "in my opinion", "it seems", "the best", "everyone agrees" are opinion markers — skip them.
4. **Extract implied facts.** If a statement implies a factual claim, extract the implied fact explicitly.
   - "Since the moon landing was faked..." → extract "The moon landing was faked."
5. **Handle sarcasm and rhetoric.** If the sentence is sarcastic or rhetorical but implies a factual claim, extract the *implied* claim being asserted, not the literal words.
   - "Oh sure, because 5G definitely causes cancer" → extract "5G causes cancer." (the implied assertion being mocked)
6. **Rephrase for clarity.** Each claim should be a clear, standalone sentence. Add minimal context if the original is ambiguous.
7. **No duplicates.** If the same fact is stated multiple times in different words, include it only once.
8. **Skip questions.** Do not extract questions.
9. **Skip meta-statements.** Do not extract statements about the act of writing or speaking (e.g., "Let me tell you something").

## Output format

Return ONLY a JSON array of strings. No markdown, no explanation, no preamble, no code fences.

## Text to analyze

\"\"\"
{text}
\"\"\"

## Examples

Input: "The Earth is flat and vaccines cause autism. I think pizza is the best food."
Output: ["The Earth is flat.", "Vaccines cause autism."]

Input: "Climate change is a hoax invented by China, and 5G towers spread COVID-19."
Output: ["Climate change is a hoax.", "Climate change was invented by China.", "5G towers spread COVID-19."]

Input: "Since the moon landing was clearly faked, we can't trust NASA about Mars either."
Output: ["The moon landing was faked.", "NASA cannot be trusted about Mars."]

Input: "Water fluoridation is basically poisoning people, and the government knows it but does nothing."
Output: ["Water fluoridation is poisoning people.", "The government knows water fluoridation is harmful.", "The government is not taking action on water fluoridation."]
"""

CLASSIFY_TYPE_SYSTEM = "You are an expert at distinguishing factual claims from opinions and unverifiable statements. You are precise and conservative."

CLASSIFY_TYPE_PROMPT = """Classify the following claim into one of three types:

- **Factual**: An objective, testable statement about the real world that can be verified or refuted with evidence (e.g., "The Earth is round", "Vaccines contain mercury", "COVID-19 originated in Wuhan").
- **Opinion**: A subjective value judgment, personal belief, preference, or normative statement that cannot be objectively proven true or false (e.g., "NASA cannot be trusted", "The government is corrupt", "Pizza is the best food").
- **Unverifiable**: A claim that is too vague, speculative, or hypothetical to verify with evidence (e.g., "Something big is coming", "They don't want you to know the truth", "Everything happens for a reason").

CLAIM: {claim}

Respond with ONLY a JSON object (no markdown, no code fences):
- "claim_type": one of "Factual", "Opinion", or "Unverifiable"
- "reason": a brief 1-sentence explanation of why you chose this type

Examples:
{{"claim_type": "Factual", "reason": "This is a testable scientific claim about the shape of the Earth."}}
{{"claim_type": "Opinion", "reason": "Trustworthiness is a subjective judgment, not an objective fact."}}
{{"claim_type": "Unverifiable", "reason": "This is too vague to test against any evidence."}}
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
# Post-processing for extracted claims
# ---------------------------------------------------------------------------
def _post_process_claims(claims: List[str]) -> List[str]:
    """
    Clean and deduplicate LLM-extracted claims.

    - Strip whitespace
    - Remove empty / too-short strings
    - Remove questions that slipped through
    - Remove near-duplicate claims (case-insensitive)
    - Cap at 15 claims to prevent runaway extraction
    """
    MIN_LENGTH = 10
    MAX_CLAIMS = 15

    seen: set[str] = set()
    result: List[str] = []

    for claim in claims:
        claim = claim.strip()

        # Skip empty / too short
        if len(claim) < MIN_LENGTH:
            continue

        # Skip questions
        if claim.endswith("?"):
            continue

        # Skip meta-statements
        meta_starts = [
            "let me", "here is", "here are", "the text",
            "this text", "the following", "note that", "as mentioned",
        ]
        if any(claim.lower().startswith(m) for m in meta_starts):
            continue

        # Deduplicate (case-insensitive, strip trailing punctuation for comparison)
        normalized = claim.lower().rstrip(".")
        if normalized in seen:
            continue
        seen.add(normalized)

        result.append(claim)

        if len(result) >= MAX_CLAIMS:
            break

    return result


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
                return _post_process_claims(claims)
            raise ValueError("Unexpected JSON structure")
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse LLM claim extraction: {e}\nRaw: {raw}")
            raise RuntimeError(f"LLM returned unparseable response: {raw}") from e

    async def classify_claim_type(self, claim: str) -> dict:
        if not self._is_available():
            raise RuntimeError("Groq client not configured")

        prompt = CLASSIFY_TYPE_PROMPT.format(claim=claim)
        raw_response = self._chat(CLASSIFY_TYPE_SYSTEM, prompt)
        raw = self._clean_json_response(raw_response)

        try:
            result = json.loads(raw)
            claim_type = result.get("claim_type", "Factual")
            reason = result.get("reason", "")

            valid_types = {"Factual", "Opinion", "Unverifiable"}
            if claim_type not in valid_types:
                claim_type = "Factual"  # Default to factual if unsure

            return {"claim_type": claim_type, "reason": reason}
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse claim type classification: {e}")
            # Default to Factual so the claim still goes through verification
            return {"claim_type": "Factual", "reason": "Classification failed, defaulting to factual."}

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
