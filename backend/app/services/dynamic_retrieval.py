"""
Dynamic evidence retrieval via Wikipedia.

Used as a fallback when the static evidence store has no strong match.
Queries the public MediaWiki API to fetch article summaries relevant to a claim.

Trust model:
- Wikipedia evidence is treated as moderate-trust (between corpus and LLM-only)
- Can produce Supported/Contradicted verdicts, but confidence is capped at 0.70
- Network failures are handled gracefully (returns None, falls back to LLM-only)
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional, List

import httpx

logger = logging.getLogger(__name__)

# MediaWiki API endpoint (English Wikipedia)
_WIKI_API = "https://en.wikipedia.org/w/api.php"
_TIMEOUT = 5.0  # seconds
_MAX_EXTRACT_CHARS = 1200

# Simple stop-words for query term extraction
_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "must", "need", "dare",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "each", "every",
    "both", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "just", "because", "but", "and", "or", "if", "while", "about",
    "that", "this", "these", "those", "it", "its", "they", "them",
    "their", "we", "our", "you", "your", "he", "she", "him", "her",
    "what", "which", "who", "whom", "up", "out", "off", "over",
    "also", "really", "actually", "never", "always", "still",
})


@dataclass
class DynamicEvidence:
    """Evidence retrieved from an external dynamic source."""
    snippet: str            # Relevant text excerpt
    title: str              # Article/page title
    url: str                # URL to the source
    source_name: str        # e.g. "Wikipedia"
    relevance_score: float  # 0.0-1.0 estimated relevance


class WikipediaRetriever:
    """
    Retrieves evidence from Wikipedia via the MediaWiki API.

    Uses the search + extracts pipeline:
    1. Extract key terms from the claim
    2. Search Wikipedia for matching articles
    3. Fetch the introduction extract of the best match
    4. Score relevance based on term overlap with the extract
    """

    def __init__(self):
        self._cache: dict[str, Optional[DynamicEvidence]] = {}

    async def search(self, claim_text: str) -> Optional[DynamicEvidence]:
        """
        Search Wikipedia for evidence relevant to the claim.

        Returns a DynamicEvidence if a relevant article is found, None otherwise.
        Results are cached in-memory per claim text.
        """
        cache_key = claim_text.lower().strip()
        if cache_key in self._cache:
            logger.debug(f"Wikipedia cache hit for: {cache_key[:50]}")
            return self._cache[cache_key]

        result = await self._do_search(claim_text)
        self._cache[cache_key] = result
        return result

    async def _do_search(self, claim_text: str) -> Optional[DynamicEvidence]:
        """Execute the Wikipedia search pipeline."""
        # Step 1: Extract search terms
        terms = _extract_search_terms(claim_text)
        if len(terms) < 1:
            logger.debug(f"No meaningful search terms from: {claim_text[:50]}")
            return None

        search_query = " ".join(terms[:6])  # Limit to 6 terms for API

        try:
            # Step 2: Search for matching articles
            title = await self._search_articles(search_query)
            if not title:
                logger.debug(f"No Wikipedia article found for: {search_query}")
                return None

            # Step 3: Fetch article extract
            extract = await self._fetch_extract(title)
            if not extract or len(extract) < 50:
                logger.debug(f"Wikipedia extract too short for: {title}")
                return None

            # Step 4: Score relevance
            relevance = _score_relevance(terms, extract)
            if relevance < 0.15:
                logger.debug(
                    f"Wikipedia result not relevant enough ({relevance:.2f}): {title}"
                )
                return None

            # Build URL
            url_title = title.replace(" ", "_")
            url = f"https://en.wikipedia.org/wiki/{url_title}"

            return DynamicEvidence(
                snippet=_truncate_snippet(extract, max_len=500),
                title=title,
                url=url,
                source_name="Wikipedia",
                relevance_score=round(relevance, 2),
            )

        except httpx.TimeoutException:
            logger.warning(f"Wikipedia API timeout for: {search_query}")
            return None
        except httpx.HTTPError as e:
            logger.warning(f"Wikipedia API HTTP error: {e}")
            return None
        except Exception as e:
            logger.warning(f"Wikipedia retrieval failed: {e}")
            return None

    async def _search_articles(self, query: str) -> Optional[str]:
        """Search Wikipedia for articles matching the query. Returns the best title."""
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": 3,
            "format": "json",
            "utf8": 1,
        }

        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(_WIKI_API, params=params)
            resp.raise_for_status()
            data = resp.json()

        results = data.get("query", {}).get("search", [])
        if not results:
            return None

        # Return the title of the first (most relevant) result
        return results[0].get("title")

    async def _fetch_extract(self, title: str) -> Optional[str]:
        """Fetch the introduction extract of a Wikipedia article."""
        params = {
            "action": "query",
            "titles": title,
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "exchars": _MAX_EXTRACT_CHARS,
            "format": "json",
            "utf8": 1,
        }

        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(_WIKI_API, params=params)
            resp.raise_for_status()
            data = resp.json()

        pages = data.get("query", {}).get("pages", {})
        for page_id, page in pages.items():
            if page_id == "-1":
                return None
            return page.get("extract", "")

        return None


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _extract_search_terms(claim_text: str) -> List[str]:
    """
    Extract meaningful search terms from a claim by removing stop words
    and short tokens.
    """
    # Remove punctuation except hyphens
    cleaned = re.sub(r"[^\w\s-]", "", claim_text.lower())
    words = cleaned.split()

    # Filter stop words and very short tokens
    terms = [w for w in words if w not in _STOP_WORDS and len(w) >= 2]
    return terms


def _score_relevance(search_terms: List[str], extract: str) -> float:
    """
    Score how relevant a Wikipedia extract is to the search terms.

    Returns 0.0-1.0 based on term overlap:
    - What fraction of search terms appear in the extract?
    """
    if not search_terms or not extract:
        return 0.0

    extract_lower = extract.lower()
    hits = sum(1 for term in search_terms if term in extract_lower)
    return hits / len(search_terms)


def _truncate_snippet(text: str, max_len: int = 500) -> str:
    """Truncate text to max_len, breaking at sentence boundary if possible."""
    if len(text) <= max_len:
        return text

    truncated = text[:max_len]
    # Try to break at the last sentence boundary
    last_period = truncated.rfind(". ")
    if last_period > max_len * 0.5:
        return truncated[: last_period + 1]
    return truncated + "…"


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_retriever = WikipediaRetriever()


async def retrieve_dynamic_evidence(claim_text: str) -> Optional[DynamicEvidence]:
    """Module-level convenience function for dynamic evidence retrieval."""
    return await _retriever.search(claim_text)
