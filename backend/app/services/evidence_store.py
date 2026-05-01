"""
Evidence store for TruthLens.

Provides a curated, structured knowledge base of well-established facts
with multiple sources, URLs, and rich metadata for evidence retrieval.

Retrieval features:
- Keyword + alias matching with query expansion
- Multi-match ranking (returns top N candidates)
- Structured source metadata (name, URL, reliability)
- Relevance scoring that considers keyword hits, alias hits, and topic match
"""

from dataclasses import dataclass, field
from typing import List, Optional
import logging
import re

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Source:
    """A single authoritative source backing a fact."""
    name: str
    url: str = ""
    reliability: float = 0.8  # 0.0-1.0 trustworthiness

    def __str__(self) -> str:
        if self.url:
            return f"{self.name} ({self.url})"
        return self.name


@dataclass
class SourceEntry:
    """A fact in the evidence store with structured metadata."""
    topic: str
    category: str                    # science, health, history, technology, nutrition, politics
    keywords: List[str]             # Primary match keywords
    aliases: List[str]              # Synonyms and alternate phrasings
    fact: str                       # The established fact
    sources: List[Source]           # Structured source list
    synonyms: List[str] = field(default_factory=list)  # Topic-scoped synonyms for semantic search
    evidence_strength: float = 0.8  # 0.0-1.0, how definitive the evidence is

    @property
    def source_count(self) -> int:
        return len(self.sources)

    @property
    def source_names(self) -> str:
        return ", ".join(s.name for s in self.sources)

    @property
    def best_url(self) -> str:
        """Return the URL from the most reliable source."""
        if not self.sources:
            return ""
        ranked = sorted(self.sources, key=lambda s: s.reliability, reverse=True)
        return ranked[0].url

    @property
    def avg_reliability(self) -> float:
        if not self.sources:
            return 0.0
        return sum(s.reliability for s in self.sources) / len(self.sources)


@dataclass
class RankedEvidence:
    """A single evidence match with its relevance score."""
    entry: SourceEntry
    relevance_score: float    # Combined score from keyword + alias matching
    keyword_hits: int         # Number of primary keywords matched
    alias_hits: int           # Number of aliases matched
    keyword_ratio: float      # Fraction of keywords matched (0.0-1.0)


@dataclass
class EvidenceResult:
    """Full retrieval result with multiple ranked candidates."""
    matches: List[RankedEvidence]  # All matches, ranked by relevance
    total_candidates: int          # How many entries were scored > 0

    @property
    def best(self) -> Optional[RankedEvidence]:
        return self.matches[0] if self.matches else None

    @property
    def has_evidence(self) -> bool:
        return len(self.matches) > 0


# ---------------------------------------------------------------------------
# Query expansion — simple synonym/alias map
# ---------------------------------------------------------------------------
_QUERY_EXPANSIONS = {
    "flat earth": ["earth flat", "earth is flat", "flat earthers"],
    "moon hoax": ["moon landing faked", "apollo hoax", "never went to moon"],
    "antivax": ["anti-vaccine", "vaccines dangerous", "vaccine injury"],
    "5g covid": ["5g coronavirus", "5g causes covid", "5g pandemic"],
    "gmo": ["genetically modified", "genetic engineering food"],
    "chemtrails": ["chemical trails", "chem trails", "spraying chemicals"],
    "qanon": ["q anon", "the storm", "wwg1wga"],
}


def _expand_query(claim_lower: str) -> List[str]:
    """Generate expanded search terms from a claim."""
    expansions = [claim_lower]
    for trigger, aliases in _QUERY_EXPANSIONS.items():
        if trigger in claim_lower:
            expansions.extend(aliases)
    return expansions


# ---------------------------------------------------------------------------
# Evidence store — 25+ curated entries
# ---------------------------------------------------------------------------
EVIDENCE_STORE: List[SourceEntry] = [
    # ==================== SCIENCE ====================
    SourceEntry(
        topic="earth shape",
        category="science",
        keywords=["earth", "flat", "round", "sphere", "globe", "spherical", "oblate"],
        aliases=["flat earth", "earth is flat", "earth is round", "shape of earth"],
        fact="The Earth is an oblate spheroid. This is confirmed by satellite imagery, physics, GPS systems, and centuries of scientific observation including circumnavigation.",
        sources=[
            Source("NASA", "https://science.nasa.gov/earth/facts/", 0.95),
            Source("ESA", "https://www.esa.int/Science_Exploration/Space_Science", 0.95),
            Source("National Geographic", "https://education.nationalgeographic.org/resource/earth/", 0.90),
        ],
        synonyms=["globe", "planet", "world", "not round", "disc", "spheroid"],
        evidence_strength=0.96,
    ),
    SourceEntry(
        topic="moon landing",
        category="science",
        keywords=["moon", "landing", "apollo", "nasa", "faked", "hoax", "1969"],
        aliases=["moon landing hoax", "apollo conspiracy", "never went to moon"],
        fact="The Apollo moon landings (1969-1972) are well-documented historical events confirmed by retroreflectors left on the lunar surface, independently verified by observatories worldwide, and corroborated by the Soviet space program.",
        sources=[
            Source("NASA History", "https://www.nasa.gov/mission_pages/apollo/index.html", 0.95),
            Source("Smithsonian Air & Space Museum", "https://airandspace.si.edu/explore/stories/apollo", 0.92),
            Source("Lunar Reconnaissance Orbiter (images of landing sites)", "https://www.nasa.gov/mission/lro/", 0.95),
        ],
        synonyms=["lunar mission", "apollo program", "space program", "staged", "fabricated"],
        evidence_strength=0.94,
    ),
    SourceEntry(
        topic="evolution",
        category="science",
        keywords=["evolution", "darwin", "natural", "selection", "species", "created", "creationism"],
        aliases=["theory of evolution", "darwinism", "origin of species", "intelligent design"],
        fact="Biological evolution through natural selection is supported by extensive evidence from genetics, paleontology, comparative anatomy, molecular biology, and direct observation of speciation.",
        sources=[
            Source("Nature", "https://www.nature.com/subjects/evolution", 0.95),
            Source("National Academy of Sciences", "https://www.nationalacademies.org/evolution", 0.95),
            Source("Smithsonian Museum of Natural History", "https://naturalhistory.si.edu/education/teaching-resources/life-science/evolution", 0.90),
        ],
        synonyms=["darwin", "natural selection", "species adaptation", "intelligent design", "creationism"],
        evidence_strength=0.94,
    ),
    SourceEntry(
        topic="age of earth",
        category="science",
        keywords=["earth", "age", "old", "young", "6000", "billion", "years"],
        aliases=["young earth", "earth age", "how old is earth"],
        fact="The Earth is approximately 4.54 billion years old, as determined by radiometric dating of meteorites and Earth rocks, consistent across multiple independent methods.",
        sources=[
            Source("US Geological Survey", "https://www.usgs.gov/special-topics/astrogeology/science/age-earth", 0.95),
            Source("National Geographic", "https://education.nationalgeographic.org/resource/resource-library-age-earth/", 0.90),
        ],
        synonyms=["young earth", "6000 years", "billions of years", "radiometric dating"],
        evidence_strength=0.93,
    ),
    SourceEntry(
        topic="sun heliocentrism",
        category="science",
        keywords=["sun", "earth", "orbit", "revolve", "center", "heliocentric", "geocentric"],
        aliases=["earth goes around sun", "sun revolves around earth"],
        fact="The Earth orbits the Sun (heliocentrism). This has been established since Copernicus and confirmed by stellar parallax, planetary motion observations, and space missions.",
        sources=[
            Source("NASA Solar System Exploration", "https://solarsystem.nasa.gov/solar-system/sun/overview/", 0.95),
            Source("European Space Agency", "https://www.esa.int/", 0.93),
        ],
        synonyms=["geocentric", "copernicus", "orbital", "revolves around"],
        evidence_strength=0.97,
    ),

    # ==================== HEALTH & VACCINES ====================
    SourceEntry(
        topic="vaccine autism",
        category="health",
        keywords=["vaccine", "vaccines", "autism", "mmr", "wakefield"],
        aliases=["vaccines cause autism", "vaccine injury autism", "mmr autism link"],
        fact="Large-scale studies involving millions of children across multiple countries show no link between vaccines and autism. The original 1998 Wakefield study was retracted due to fraud.",
        sources=[
            Source("CDC", "https://www.cdc.gov/vaccinesafety/concerns/autism.html", 0.95),
            Source("WHO", "https://www.who.int/news-room/questions-and-answers/item/vaccines-and-immunization-what-is-vaccination", 0.95),
            Source("The Lancet (retraction notice)", "https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(10)60175-4/fulltext", 0.95),
            Source("Institute of Medicine / National Academies", "https://nap.nationalacademies.org/catalog/13164/adverse-effects-of-vaccines-evidence-and-causality", 0.93),
        ],
        synonyms=["immunization", "shots", "jab", "inoculation", "developmental disorder", "developmental problems", "ASD", "spectrum"],
        evidence_strength=0.95,
    ),
    SourceEntry(
        topic="vaccine safety",
        category="health",
        keywords=["vaccine", "vaccines", "safe", "unsafe", "dangerous", "side", "effects"],
        aliases=["are vaccines safe", "vaccine safety", "vaccine side effects"],
        fact="Vaccines undergo rigorous multi-phase clinical trials and continuous post-market safety monitoring. Serious adverse events are extremely rare compared to the diseases they prevent.",
        sources=[
            Source("FDA", "https://www.fda.gov/vaccines-blood-biologics/safety-availability-biologics/vaccine-safety", 0.95),
            Source("WHO", "https://www.who.int/news-room/feature-stories/detail/how-are-vaccines-developed", 0.95),
            Source("CDC VAERS", "https://vaers.hhs.gov/", 0.90),
        ],
        synonyms=["immunization", "shots", "jab", "adverse reaction", "injection", "inoculation"],
        evidence_strength=0.92,
    ),
    SourceEntry(
        topic="vaccine microchips",
        category="health",
        keywords=["vaccine", "microchip", "microchips", "chip", "tracking", "bill", "gates"],
        aliases=["microchip vaccine", "tracking chip vaccine", "bill gates microchip"],
        fact="Vaccines do not contain microchips or tracking devices. Vaccine ingredients are publicly listed and regulated. No tracking technology small enough to fit through a needle exists.",
        sources=[
            Source("WHO", "https://www.who.int/emergencies/diseases/novel-coronavirus-2019/advice-for-public/myth-busters", 0.95),
            Source("FDA", "https://www.fda.gov/vaccines-blood-biologics/", 0.95),
            Source("AP News Fact Check", "https://apnews.com/hub/ap-fact-check", 0.88),
        ],
        synonyms=["tracking device", "implant", "nanotechnology", "surveillance", "injection"],
        evidence_strength=0.93,
    ),
    SourceEntry(
        topic="covid origin",
        category="health",
        keywords=["covid", "coronavirus", "sars-cov-2", "origin", "lab", "wuhan", "natural", "leak"],
        aliases=["covid lab leak", "covid origin", "wuhan lab", "covid natural origin"],
        fact="The origin of SARS-CoV-2 remains under active scientific investigation. Both natural spillover and laboratory-related hypotheses are being studied. No definitive conclusion has been reached by the scientific community.",
        sources=[
            Source("WHO Investigation Report", "https://www.who.int/publications/i/item/who-convened-global-study-of-origins-of-sars-cov-2-china-part", 0.90),
            Source("US NIH", "https://www.nih.gov/", 0.90),
        ],
        synonyms=["laboratory", "bioweapon", "engineered virus", "gain of function"],
        evidence_strength=0.50,
    ),
    SourceEntry(
        topic="ivermectin covid",
        category="health",
        keywords=["ivermectin", "covid", "treatment", "cure", "horse", "dewormer"],
        aliases=["ivermectin for covid", "ivermectin coronavirus"],
        fact="Clinical trials have not demonstrated that ivermectin is an effective treatment for COVID-19. The FDA, WHO, and EMA advise against its use for COVID-19 outside of clinical trials.",
        sources=[
            Source("FDA", "https://www.fda.gov/consumers/consumer-updates/why-you-should-not-use-ivermectin-treat-or-prevent-covid-19", 0.95),
            Source("WHO", "https://www.who.int/news-room/feature-stories/detail/who-advises-that-ivermectin-only-be-used-to-treat-covid-19-within-clinical-trials", 0.95),
            Source("Cochrane Review", "https://www.cochranelibrary.com/cdsr/doi/10.1002/14651858.CD015017.pub3/full", 0.93),
        ],
        synonyms=["antiparasitic", "dewormer", "medication", "alternative treatment"],
        evidence_strength=0.88,
    ),
    SourceEntry(
        topic="masks effectiveness",
        category="health",
        keywords=["mask", "masks", "face", "covering", "useless", "effective", "covid"],
        aliases=["do masks work", "mask effectiveness", "face masks covid"],
        fact="Multiple studies and meta-analyses show that properly worn face masks reduce the transmission of respiratory viruses including SARS-CoV-2, particularly in indoor settings.",
        sources=[
            Source("CDC", "https://www.cdc.gov/coronavirus/2019-ncov/science/science-briefs/masking-science-sars-cov2.html", 0.93),
            Source("BMJ", "https://www.bmj.com/content/375/bmj-2021-068302", 0.92),
        ],
        synonyms=["face covering", "N95", "surgical mask", "respiratory protection"],
        evidence_strength=0.85,
    ),

    # ==================== CLIMATE ====================
    SourceEntry(
        topic="climate change",
        category="science",
        keywords=["climate", "change", "global", "warming", "hoax", "real", "man-made", "human"],
        aliases=["climate change hoax", "global warming hoax", "climate change is real", "anthropogenic warming"],
        fact="Climate change driven by human activity is supported by overwhelming scientific evidence. Over 97% of actively publishing climate scientists agree that human activities are causing global warming.",
        sources=[
            Source("IPCC AR6", "https://www.ipcc.ch/assessment-report/ar6/", 0.97),
            Source("NASA Climate", "https://climate.nasa.gov/", 0.95),
            Source("NOAA", "https://www.noaa.gov/climate", 0.95),
            Source("Royal Society", "https://royalsociety.org/topics-policy/projects/climate-change-evidence-causes/", 0.93),
        ],
        synonyms=["global warming", "climate crisis", "environmental", "fabricated", "warming planet"],
        evidence_strength=0.96,
    ),
    SourceEntry(
        topic="carbon dioxide",
        category="science",
        keywords=["co2", "carbon", "dioxide", "emissions", "greenhouse", "fossil", "fuel"],
        aliases=["co2 greenhouse gas", "carbon emissions warming"],
        fact="Carbon dioxide is a greenhouse gas. Increased atmospheric CO2 from fossil fuel combustion is the primary driver of observed global warming, rising from ~280 ppm pre-industrial to over 420 ppm today.",
        sources=[
            Source("IPCC AR6", "https://www.ipcc.ch/assessment-report/ar6/", 0.97),
            Source("NOAA Global Monitoring Lab", "https://gml.noaa.gov/ccgg/trends/", 0.95),
            Source("Scripps CO2 Program", "https://keelingcurve.ucsd.edu/", 0.93),
        ],
        synonyms=["greenhouse effect", "emissions", "fossil fuel", "atmospheric"],
        evidence_strength=0.94,
    ),
    SourceEntry(
        topic="sea level rise",
        category="science",
        keywords=["sea", "level", "rise", "rising", "ocean", "ice", "melt", "glacier"],
        aliases=["sea level rising", "ice caps melting", "glaciers melting"],
        fact="Global mean sea level has risen approximately 21-24 cm since 1880, with the rate of rise accelerating. This is driven by thermal expansion and melting of ice sheets and glaciers.",
        sources=[
            Source("NASA Sea Level", "https://climate.nasa.gov/vital-signs/sea-level/", 0.95),
            Source("NOAA Tides & Currents", "https://tidesandcurrents.noaa.gov/sltrends/", 0.93),
        ],
        synonyms=["coastal flooding", "ice caps", "polar ice", "ocean warming"],
        evidence_strength=0.91,
    ),

    # ==================== HISTORY ====================
    SourceEntry(
        topic="holocaust",
        category="history",
        keywords=["holocaust", "deny", "denial", "never", "happened", "6", "million", "nazi"],
        aliases=["holocaust denial", "holocaust hoax", "holocaust didn't happen"],
        fact="The Holocaust is one of the most extensively documented events in history. Approximately six million Jews were systematically murdered by Nazi Germany. Evidence includes Nazi records, survivor testimony, physical evidence, and Allied liberation documentation.",
        sources=[
            Source("US Holocaust Memorial Museum", "https://www.ushmm.org/", 0.97),
            Source("Yad Vashem", "https://www.yadvashem.org/", 0.97),
            Source("Nuremberg Trial Records", "https://www.loc.gov/rr/frd/Military_Law/Nuremberg_Trials.html", 0.95),
            Source("Auschwitz-Birkenau Memorial", "https://www.auschwitz.org/en/", 0.95),
        ],
        synonyms=["genocide", "mass killing", "atrocity", "concentration camp", "extermination"],
        evidence_strength=0.98,
    ),
    SourceEntry(
        topic="9/11",
        category="history",
        keywords=["9/11", "september", "11", "twin", "towers", "inside", "job", "controlled", "demolition"],
        aliases=["9/11 inside job", "9/11 conspiracy", "twin towers demolition"],
        fact="The September 11, 2001 attacks were carried out by 19 al-Qaeda terrorists who hijacked four commercial aircraft. Extensive investigations by the 9/11 Commission, NIST, and FBI confirmed the events. Engineering analyses explain the structural collapse.",
        sources=[
            Source("9/11 Commission Report", "https://www.9-11commission.gov/report/", 0.95),
            Source("NIST Investigation", "https://www.nist.gov/topics/disaster-failure-studies/world-trade-center-disaster-study", 0.95),
            Source("National September 11 Memorial", "https://www.911memorial.org/", 0.90),
        ],
        synonyms=["september 11", "twin towers", "world trade center", "attack", "terrorist"],
        evidence_strength=0.93,
    ),

    # ==================== TECHNOLOGY ====================
    SourceEntry(
        topic="5g health",
        category="technology",
        keywords=["5g", "radiation", "cancer", "health", "dangerous", "harmful", "radio"],
        aliases=["5g causes cancer", "5g dangerous", "5g health risks", "5g radiation"],
        fact="5G networks use non-ionizing radio waves at frequencies that have been extensively studied. No confirmed health risks have been found from exposure within established safety guidelines (ICNIRP).",
        sources=[
            Source("WHO EMF Project", "https://www.who.int/health-topics/electromagnetic-fields", 0.95),
            Source("ICNIRP Guidelines", "https://www.icnirp.org/en/frequencies/radiofrequency/", 0.93),
            Source("IEEE", "https://www.ieee.org/", 0.90),
            Source("FDA", "https://www.fda.gov/radiation-emitting-products/cell-phones/scientific-evidence-cell-phone-safety", 0.93),
        ],
        synonyms=["cell tower", "wireless", "mobile network", "cellular", "radio tower", "electromagnetic"],
        evidence_strength=0.87,
    ),
    SourceEntry(
        topic="5g covid",
        category="technology",
        keywords=["5g", "covid", "coronavirus", "spread", "cause", "pandemic", "virus"],
        aliases=["5g causes covid", "5g spreads coronavirus", "5g pandemic"],
        fact="There is no connection between 5G networks and COVID-19. Viruses cannot travel through radio waves. COVID-19 has spread in many countries without 5G networks.",
        sources=[
            Source("WHO Myth Busters", "https://www.who.int/emergencies/diseases/novel-coronavirus-2019/advice-for-public/myth-busters", 0.95),
            Source("Full Fact", "https://fullfact.org/online/5g-and-covid-19/", 0.85),
        ],
        synonyms=["cell tower", "wireless", "illness", "disease", "sickness", "pandemic", "transmit"],
        evidence_strength=0.92,
    ),

    # ==================== NUTRITION ====================
    SourceEntry(
        topic="water fluoridation",
        category="nutrition",
        keywords=["fluoride", "water", "fluoridation", "poison", "toxic", "mind", "control", "teeth"],
        aliases=["fluoride poison", "water fluoridation safe", "fluoride mind control"],
        fact="Community water fluoridation at recommended levels (0.7 mg/L) is endorsed by major health organizations as safe and effective for preventing tooth decay. Over 75 years of research supports its safety.",
        sources=[
            Source("CDC", "https://www.cdc.gov/fluoridation/", 0.93),
            Source("ADA", "https://www.ada.org/resources/community-initiatives/fluoride-in-water", 0.90),
            Source("WHO", "https://www.who.int/publications/i/item/9789241548649", 0.93),
        ],
        synonyms=["fluoridation", "water treatment", "dental health", "drinking water"],
        evidence_strength=0.84,
    ),
    SourceEntry(
        topic="gmo safety",
        category="nutrition",
        keywords=["gmo", "genetically", "modified", "food", "safe", "dangerous", "organic", "franken"],
        aliases=["gmo dangerous", "genetically modified food safe", "frankenfood"],
        fact="The scientific consensus, supported by thousands of studies and every major scientific body that has examined the evidence, is that approved genetically modified foods are safe for human consumption.",
        sources=[
            Source("National Academies of Sciences", "https://nap.nationalacademies.org/catalog/23395/genetically-engineered-crops-experiences-and-prospects", 0.95),
            Source("WHO", "https://www.who.int/health-topics/food-genetically-modified", 0.95),
            Source("FDA", "https://www.fda.gov/food/agricultural-biotechnology/gmos-your-food", 0.93),
        ],
        synonyms=["genetic engineering", "transgenic", "biotech", "frankenfood", "crop modification"],
        evidence_strength=0.90,
    ),
    SourceEntry(
        topic="msg safety",
        category="nutrition",
        keywords=["msg", "monosodium", "glutamate", "chinese", "restaurant", "syndrome", "headache"],
        aliases=["msg dangerous", "msg headache", "chinese restaurant syndrome"],
        fact="Monosodium glutamate (MSG) is generally recognized as safe by food safety authorities worldwide. Double-blind studies have failed to demonstrate a consistent relationship between MSG consumption and reported symptoms.",
        sources=[
            Source("FDA", "https://www.fda.gov/food/food-additives-petitions/questions-and-answers-monosodium-glutamate-msg", 0.93),
            Source("WHO/FAO Joint Expert Committee", "https://www.who.int/", 0.90),
        ],
        synonyms=["food additive", "flavor enhancer", "umami", "glutamic acid"],
        evidence_strength=0.82,
    ),
    SourceEntry(
        topic="aspartame safety",
        category="nutrition",
        keywords=["aspartame", "artificial", "sweetener", "cancer", "diet", "soda"],
        aliases=["aspartame cancer", "aspartame dangerous", "diet soda cancer"],
        fact="Aspartame has been extensively studied and is approved as safe by over 100 regulatory agencies worldwide at current consumption levels. The IARC classification (Group 2B, 'possibly carcinogenic') noted the evidence was limited and inconclusive.",
        sources=[
            Source("FDA", "https://www.fda.gov/food/food-additives-petitions/aspartame-and-other-sweeteners-food", 0.93),
            Source("EFSA", "https://www.efsa.europa.eu/en/topics/topic/aspartame", 0.93),
            Source("WHO/IARC", "https://www.iarc.who.int/", 0.90),
        ],
        synonyms=["artificial sweetener", "diet drink", "sugar substitute", "saccharin"],
        evidence_strength=0.80,
    ),

    # ==================== CONSPIRACY / MISC ====================
    SourceEntry(
        topic="chemtrails",
        category="science",
        keywords=["chemtrail", "chemtrails", "contrail", "contrails", "spray", "chemical", "sky", "planes"],
        aliases=["chemtrails conspiracy", "chemical spraying planes"],
        fact="'Chemtrails' are actually condensation trails (contrails) — water vapor that freezes when hot jet exhaust meets cold air at high altitude. Atmospheric scientists have found no evidence of deliberate chemical spraying programs.",
        sources=[
            Source("EPA", "https://www.epa.gov/", 0.90),
            Source("Carnegie Science / UC Irvine Study", "https://iopscience.iop.org/article/10.1088/1748-9326/11/8/084011", 0.90),
            Source("Scientific American", "https://www.scientificamerican.com/", 0.85),
        ],
        synonyms=["chemical trails", "aerial spraying", "contrails", "sky lines"],
        evidence_strength=0.88,
    ),
    SourceEntry(
        topic="illuminati",
        category="history",
        keywords=["illuminati", "secret", "society", "new", "world", "order", "nwo", "cabal"],
        aliases=["illuminati control world", "new world order", "secret world government"],
        fact="The historical Illuminati was a Bavarian secret society founded in 1776 and disbanded by 1785. There is no credible evidence of a surviving secret organization controlling world events. Modern 'Illuminati' claims are unfounded conspiracy theories.",
        sources=[
            Source("Britannica", "https://www.britannica.com/topic/Illuminati", 0.88),
            Source("History.com", "https://www.history.com/topics/secret-societies/illuminati", 0.85),
        ],
        synonyms=["secret society", "new world order", "conspiracy", "cabal", "elite"],
        evidence_strength=0.82,
    ),
    SourceEntry(
        topic="birds aren't real",
        category="science",
        keywords=["birds", "real", "drones", "government", "surveillance", "fake"],
        aliases=["birds are drones", "birds aren't real", "government birds"],
        fact="Birds are real, living organisms. The 'Birds Aren't Real' movement was created as a satirical conspiracy theory by Peter McIndoe in 2017 to parody conspiracy thinking.",
        sources=[
            Source("Audubon Society", "https://www.audubon.org/", 0.90),
            Source("New York Times (profile of satire movement)", "https://www.nytimes.com/2021/12/09/technology/birds-arent-real-gen-z-conspiracy.html", 0.85),
        ],
        synonyms=["drones", "surveillance", "government spy", "robotic animals"],
        evidence_strength=0.90,
    ),
]


# ---------------------------------------------------------------------------
# Retrieval engine
# ---------------------------------------------------------------------------

def retrieve_evidence(
    claim_text: str,
    top_k: int = 3,
    min_score: float = 1.5,
    min_keyword_hits: int = 2,
    min_keyword_ratio: float = 0.20,
) -> EvidenceResult:
    """
    Search the evidence store for the most relevant entries matching a claim.

    Uses a multi-signal scoring approach:
    - Primary keyword hits (weighted 1.0 each)
    - Alias phrase hits (weighted 0.8 each)
    - Query expansion matches (weighted 0.5 each)

    Filtering thresholds (a match must pass ALL to be included):
    - min_score >= 1.5 (prevents single-keyword accidental matches)
    - min_keyword_hits >= 2 (at least 2 distinct keywords must match)
    - min_keyword_ratio >= 0.20 (at least 20% of the entry's keywords must match)

    Exception: alias matches can satisfy min_score alone (they are multi-word
    phrases and therefore much more specific than single keywords).

    Returns an EvidenceResult with up to top_k ranked matches.
    """
    claim_lower = claim_text.lower()
    expanded_queries = _expand_query(claim_lower)

    scored: List[RankedEvidence] = []

    for entry in EVIDENCE_STORE:
        keyword_hits = 0
        alias_hits = 0

        # Score primary keywords
        for kw in entry.keywords:
            if kw in claim_lower:
                keyword_hits += 1

        # Score alias matches (multi-word phrases — more specific)
        for alias in entry.aliases:
            if alias in claim_lower:
                alias_hits += 1

        # Score query expansion matches
        expansion_hits = 0
        for expanded in expanded_queries[1:]:
            for kw in entry.keywords:
                if kw in expanded:
                    expansion_hits += 1

        # Combined relevance score
        total_keywords = len(entry.keywords)
        score = (keyword_hits * 1.0) + (alias_hits * 0.8) + (expansion_hits * 0.5)
        keyword_ratio = keyword_hits / total_keywords if total_keywords > 0 else 0.0

        # --- Filtering: must pass score AND quality thresholds ---
        # Alias matches are high-quality (multi-word), so they bypass keyword-count checks
        has_alias = alias_hits > 0
        has_enough_keywords = (
            keyword_hits >= min_keyword_hits and keyword_ratio >= min_keyword_ratio
        )

        if score >= min_score and (has_enough_keywords or has_alias):
            scored.append(
                RankedEvidence(
                    entry=entry,
                    relevance_score=round(score, 2),
                    keyword_hits=keyword_hits,
                    alias_hits=alias_hits,
                    keyword_ratio=keyword_ratio,
                )
            )

    # Rank by relevance score (descending), then by evidence strength as tiebreaker
    scored.sort(key=lambda r: (r.relevance_score, r.entry.evidence_strength), reverse=True)

    # Take top_k
    top_matches = scored[:top_k]
    total_candidates = len(scored)

    return EvidenceResult(
        matches=top_matches,
        total_candidates=total_candidates,
    )

