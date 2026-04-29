"""
Trusted fact corpus for evidence retrieval.

This module provides a small, curated set of well-established facts
that the analyzer uses to find supporting or contradicting evidence
for extracted claims.  Easy to extend or replace with a database / 
vector store in future versions.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class CorpusEntry:
    """A single fact in the trusted corpus."""
    topic: str
    keywords: List[str]
    fact: str
    source: str
    source_count: int = 1          # Number of independent sources backing this fact
    evidence_strength: float = 0.7  # 0.0-1.0, how definitive the evidence is


# ---------------------------------------------------------------------------
# Pre-built trusted corpus (MVP)
# ---------------------------------------------------------------------------
TRUSTED_CORPUS: List[CorpusEntry] = [
    # Earth & Space
    CorpusEntry(
        topic="earth shape",
        keywords=["earth", "flat", "round", "sphere", "globe", "spherical"],
        fact="The Earth is an oblate spheroid. This is confirmed by satellite imagery, physics, and centuries of scientific observation.",
        source="NASA, ESA, and scientific consensus",
        source_count=5,
        evidence_strength=0.95,
    ),
    CorpusEntry(
        topic="moon landing",
        keywords=["moon", "landing", "apollo", "nasa", "faked", "hoax"],
        fact="The Apollo moon landings (1969-1972) are well-documented historical events confirmed by multiple independent sources including retroreflectors left on the lunar surface.",
        source="NASA, independent verification by global space agencies",
        source_count=4,
        evidence_strength=0.93,
    ),
    # Health & Vaccines
    CorpusEntry(
        topic="vaccine safety",
        keywords=["vaccine", "vaccines", "safe", "unsafe", "dangerous", "autism"],
        fact="Large-scale studies involving millions of children show no link between vaccines and autism. Vaccines undergo rigorous safety testing before approval.",
        source="WHO, CDC, The Lancet retraction of Wakefield study",
        source_count=4,
        evidence_strength=0.92,
    ),
    CorpusEntry(
        topic="vaccine microchips",
        keywords=["vaccine", "microchip", "microchips", "chip", "tracking", "5g"],
        fact="Vaccines do not contain microchips or tracking devices. They contain biological or chemical components designed to produce an immune response.",
        source="WHO, FDA, peer-reviewed immunology research",
        source_count=3,
        evidence_strength=0.90,
    ),
    CorpusEntry(
        topic="covid origin",
        keywords=["covid", "coronavirus", "sars-cov-2", "origin", "lab", "wuhan", "natural"],
        fact="The origin of SARS-CoV-2 remains under investigation. Both natural spillover and laboratory-related hypotheses are being studied. No definitive conclusion has been reached by the scientific community.",
        source="WHO investigation reports, US intelligence assessments",
        source_count=2,
        evidence_strength=0.50,  # Genuinely uncertain topic
    ),
    # Climate
    CorpusEntry(
        topic="climate change",
        keywords=["climate", "change", "global", "warming", "hoax", "real"],
        fact="Climate change driven by human activity is supported by overwhelming scientific evidence. Over 97% of climate scientists agree that human-caused global warming is occurring.",
        source="IPCC, NASA, NOAA",
        source_count=5,
        evidence_strength=0.94,
    ),
    CorpusEntry(
        topic="carbon dioxide",
        keywords=["co2", "carbon", "dioxide", "emissions", "greenhouse"],
        fact="Carbon dioxide is a greenhouse gas. Increased atmospheric CO2 from fossil fuel combustion is a primary driver of observed global warming.",
        source="IPCC AR6, NOAA Global Monitoring Laboratory",
        source_count=3,
        evidence_strength=0.90,
    ),
    # History & Politics
    CorpusEntry(
        topic="holocaust",
        keywords=["holocaust", "deny", "denial", "never", "happened"],
        fact="The Holocaust is one of the most documented events in history. Approximately six million Jews were systematically murdered by Nazi Germany during World War II.",
        source="United States Holocaust Memorial Museum, Yad Vashem, Nuremberg trial records",
        source_count=5,
        evidence_strength=0.97,
    ),
    CorpusEntry(
        topic="evolution",
        keywords=["evolution", "darwin", "natural", "selection", "species", "created"],
        fact="Biological evolution through natural selection is supported by extensive evidence from genetics, paleontology, comparative anatomy, and direct observation.",
        source="Nature, Science, peer-reviewed biology research",
        source_count=4,
        evidence_strength=0.93,
    ),
    # Technology
    CorpusEntry(
        topic="5g health",
        keywords=["5g", "radiation", "cancer", "health", "dangerous", "harmful"],
        fact="5G networks use non-ionizing radio waves. Extensive research has found no confirmed health risks from exposure within established safety guidelines.",
        source="WHO, ICNIRP, IEEE",
        source_count=3,
        evidence_strength=0.85,
    ),
    # Nutrition
    CorpusEntry(
        topic="water fluoridation",
        keywords=["fluoride", "water", "fluoridation", "poison", "toxic", "mind"],
        fact="Community water fluoridation at recommended levels (0.7 mg/L) is endorsed by major health organizations as safe and effective for preventing tooth decay.",
        source="CDC, ADA, WHO",
        source_count=3,
        evidence_strength=0.82,
    ),
    CorpusEntry(
        topic="gmo safety",
        keywords=["gmo", "genetically", "modified", "food", "safe", "dangerous"],
        fact="The scientific consensus, supported by thousands of studies, is that approved genetically modified foods are safe for human consumption.",
        source="National Academies of Sciences, WHO, FDA",
        source_count=3,
        evidence_strength=0.88,
    ),
]


@dataclass
class EvidenceMatch:
    """Result of an evidence lookup, including match quality metadata."""
    entry: CorpusEntry
    keyword_score: int     # Number of keywords matched
    keyword_ratio: float   # Fraction of keywords matched (0.0-1.0)


def retrieve_evidence(claim_text: str) -> Optional[EvidenceMatch]:
    """
    Search the trusted corpus for the most relevant entry matching a claim.
    
    Uses keyword overlap scoring — simple and deterministic.
    Returns an EvidenceMatch with metadata, or None if no relevant entry is found.
    """
    claim_lower = claim_text.lower()
    best_match: Optional[CorpusEntry] = None
    best_score = 0
    best_total = 0

    for entry in TRUSTED_CORPUS:
        score = sum(1 for kw in entry.keywords if kw in claim_lower)
        if score > best_score:
            best_score = score
            best_match = entry
            best_total = len(entry.keywords)

    # Require at least one keyword match
    if best_score >= 1 and best_match is not None:
        return EvidenceMatch(
            entry=best_match,
            keyword_score=best_score,
            keyword_ratio=best_score / best_total if best_total > 0 else 0.0,
        )
    return None
