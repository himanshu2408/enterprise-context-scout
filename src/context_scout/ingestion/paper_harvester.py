"""Automated open-access quantitative research paper harvester across IN, DE, and US markets (M3).

Integrates arXiv Atom XML and OpenAlex REST APIs with exponential backoff and offline mock fixtures.
"""

import gzip
import io
import json
import logging
import random
import re
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zlib
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from typing import Final, Protocol, runtime_checkable

from context_scout.schemas.contracts import MarketRegime

logger = logging.getLogger(__name__)

ARXIV_XML_NS: Final[dict[str, str]] = {"atom": "http://www.w3.org/2005/Atom"}
DEFAULT_USER_AGENT: Final[str] = "EnterpriseContextScout/1.0 (mailto:scout@example.com)"


class HarvesterSource(StrEnum):
    ARXIV = "arxiv"
    OPENALEX = "openalex"
    LOCAL_CACHE = "local_cache"


@dataclass(frozen=True)
class HarvestedPaper:
    """Standardized metadata and content payload for a harvested research paper."""

    doc_id: str  # e.g. "arxiv:2502.08192" or "openalex:W4389201"
    source: HarvesterSource
    title: str
    authors: list[str]
    published_date: date
    market_scope: MarketRegime
    source_uri: str
    abstract: str
    raw_tex_content: str | None = None
    raw_pdf_bytes: bytes | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class MarketHarvestQuery:
    """Configured query parameters targeting a specific equity market regime."""

    market_scope: MarketRegime
    keywords: list[str]
    arxiv_categories: list[str]
    openalex_concepts: list[str]
    start_year: int = 2018
    max_results: int = 50


QUANT_ARXIV_CATEGORIES: Final[list[str]] = [
    "q-fin.TR",  # Trading and Market Microstructure
    "q-fin.PM",  # Portfolio Management
    "q-fin.ST",  # Statistical Finance
    "q-fin.CP",  # Computational Finance
    "q-fin.RM",  # Risk Management
]

POSITIVE_QUANT_CONCEPTS: Final[list[str]] = [
    "momentum",
    "mean reversion",
    "statistical arbitrage",
    "pairs trading",
    "limit order book",
    "order flow",
    "market microstructure",
    "high-frequency trading",
    "market making",
    "optimal execution",
    "Almgren-Chriss",
    "Hawkes process",
    "transaction cost",
    "slippage",
    "bid-ask spread",
    "volatility forecasting",
    "GARCH",
    "factor alpha",
    "portfolio optimization",
    "Sharpe ratio",
    "drawdown",
    "reinforcement learning trading",
    "intraday",
    "tick size",
    "closing auction",
    "derivatives hedging",
]

NEGATIVE_BLOCKING_KEYWORDS: Final[list[str]] = [
    "covid",
    "coronavirus",
    "pandemic",
    "mandatory csr",
    "corporate social responsibility",
    "climate change",
    "natural disaster",
    "green bond",
    "circular economy",
    "brexit",
    "industrial symbiotic",
    "board gender",
]


@dataclass(frozen=True)
class RelevanceVerdict:
    """Outcome of quantitative finance relevance and negative-topic evaluation."""

    is_relevant: bool
    matched_positive: list[str]
    matched_negative: list[str]
    score: float
    rejection_reason: str | None = None


class QuantRelevanceClassifier:
    """Strict relevance classifier matching actionable quant concepts and blocking noise."""

    def __init__(self, min_positive_matches: int = 1) -> None:
        self.min_positive_matches = min_positive_matches

        self._neg_patterns: dict[str, re.Pattern[str]] = {
            "covid": re.compile(r"\b(?:covid(?:-?19)?|sars-cov-2)\b", re.IGNORECASE),
            "coronavirus": re.compile(r"\bcoronavirus(?:es)?\b", re.IGNORECASE),
            "pandemic": re.compile(r"\bpandemics?\b", re.IGNORECASE),
            "mandatory csr": re.compile(r"\bmandatory\s+csr\b", re.IGNORECASE),
            "corporate social responsibility": re.compile(
                r"\bcorporate\s+social\s+responsibility\b|\bcsr\b", re.IGNORECASE
            ),
            "climate change": re.compile(
                r"\bclimate\s+change\b|\bglobal\s+warming\b|\bcarbon\s+emission", re.IGNORECASE
            ),
            "natural disaster": re.compile(r"\bnatural\s+disasters?\b", re.IGNORECASE),
            "green bond": re.compile(r"\bgreen\s+bonds?\b", re.IGNORECASE),
            "circular economy": re.compile(r"\bcircular\s+economy\b", re.IGNORECASE),
            "brexit": re.compile(r"\bbrexit\b", re.IGNORECASE),
            "industrial symbiotic": re.compile(r"\bindustrial\s+symbio\w*", re.IGNORECASE),
            "board gender": re.compile(
                r"\bboard\s+gender\b|\bgender\s+diversity\s+(?:on|in)\s+board|\bfemale\s+directors?",
                re.IGNORECASE,
            ),
        }

        self._pos_patterns: dict[str, re.Pattern[str]] = {
            "momentum": re.compile(r"\bmomentum\b", re.IGNORECASE),
            "mean reversion": re.compile(r"\bmean\s*[- ]\s*rever(?:sion|ting)\b", re.IGNORECASE),
            "statistical arbitrage": re.compile(
                r"\bstat(?:istical)?\s*[- ]\s*arbitrage\b|\bstat\s*arb\b", re.IGNORECASE
            ),
            "pairs trading": re.compile(r"\bpairs?\s*[- ]\s*trading\b", re.IGNORECASE),
            "limit order book": re.compile(r"\blimit\s+order\s+books?\b|\blob\b", re.IGNORECASE),
            "order flow": re.compile(r"\border\s+flows?\b|\bofi\b", re.IGNORECASE),
            "market microstructure": re.compile(r"\b(?:market\s+)?microstructure\b", re.IGNORECASE),
            "high-frequency trading": re.compile(
                r"\bhigh\s*[- ]\s*frequency\s+trading\b|\bhft\b|\balgorithmic\s+trading\b",
                re.IGNORECASE,
            ),
            "market making": re.compile(r"\bmarket\s*[- ]\s*mak(?:ing|er|ers)\b", re.IGNORECASE),
            "optimal execution": re.compile(
                r"\boptimal\s+execution\b|\bexecution\s+algorithms?\b|\btwap\b|\bvwap\b",
                re.IGNORECASE,
            ),
            "Almgren-Chriss": re.compile(r"\balmgren\s*[- ]\s*chriss\b", re.IGNORECASE),
            "Hawkes process": re.compile(r"\bhawkes\s+process(?:es)?\b", re.IGNORECASE),
            "transaction cost": re.compile(
                r"\btransaction\s+costs?\b|\btrading\s+frictions?\b|\bturnover\s+costs?\b|\bsecurities\s+transaction\s+tax\b|\bstt\b",
                re.IGNORECASE,
            ),
            "slippage": re.compile(r"\bslippage\b", re.IGNORECASE),
            "bid-ask spread": re.compile(
                r"\bbid\s*[- ]\s*ask\s+spreads?\b|\bhalf\s*[- ]\s*spreads?\b", re.IGNORECASE
            ),
            "volatility forecasting": re.compile(
                r"\bvolatility\s+forecast(?:ing|s)?\b|\brealized\s+volatility\b|\bimplied\s+volatility\b",
                re.IGNORECASE,
            ),
            "GARCH": re.compile(r"\b(?:e|gjr-)?garch\b|\barch\s+models?\b", re.IGNORECASE),
            "factor alpha": re.compile(
                r"\b(?:factor\s+alpha|alpha\s+factors?|multi-factor\s+model|cross-sectional\s+momentum|fama-french)\b",
                re.IGNORECASE,
            ),
            "portfolio optimization": re.compile(
                r"\bportfolio\s+optimi[zs]ation\b|\bmean\s*[- ]\s*variance\b|\brisk\s+parity\b",
                re.IGNORECASE,
            ),
            "Sharpe ratio": re.compile(
                r"\bsharpe\s+ratio\b|\binformation\s+ratio\b|\bsortino\s+ratio\b", re.IGNORECASE
            ),
            "drawdown": re.compile(r"\b(?:max(?:imum)?\s+)?drawdowns?\b", re.IGNORECASE),
            "reinforcement learning trading": re.compile(
                r"\breinforcement\s+learning(?:\s+\w+){0,3}\s+trading\b|\bq-learning\b",
                re.IGNORECASE,
            ),
            "intraday": re.compile(r"\bintraday\b", re.IGNORECASE),
            "tick size": re.compile(r"\btick\s+sizes?\b", re.IGNORECASE),
            "closing auction": re.compile(
                r"\bclosing\s+auctions?\b|\bclosing\s+cross\b|\bauction\s+imbalance\b",
                re.IGNORECASE,
            ),
            "derivatives hedging": re.compile(
                r"\b(?:derivatives?|delta|gamma|options?)\s+hedging\b", re.IGNORECASE
            ),
        }

    def evaluate(self, title: str, abstract: str = "", content: str = "") -> RelevanceVerdict:
        """Evaluates text against positive concepts and negative blocking keywords."""
        full_text = f"{title}\n{abstract}\n{content}"

        matched_neg: list[str] = []
        for kw, pattern in self._neg_patterns.items():
            if pattern.search(full_text):
                matched_neg.append(kw)

        if matched_neg:
            neg_desc = ", ".join(matched_neg)
            return RelevanceVerdict(
                is_relevant=False,
                matched_positive=[],
                matched_negative=matched_neg,
                score=0.0,
                rejection_reason=f"Rejected: Matched negative blocking keyword(s): {neg_desc}",
            )

        matched_pos: list[str] = []
        for concept, pattern in self._pos_patterns.items():
            if pattern.search(full_text):
                matched_pos.append(concept)

        if len(matched_pos) < self.min_positive_matches:
            return RelevanceVerdict(
                is_relevant=False,
                matched_positive=matched_pos,
                matched_negative=[],
                score=0.0,
                rejection_reason=(
                    f"Rejected: Insufficient positive quant concepts "
                    f"({len(matched_pos)} < {self.min_positive_matches})"
                ),
            )

        score = min(1.0, len(matched_pos) / 5.0)
        return RelevanceVerdict(
            is_relevant=True,
            matched_positive=matched_pos,
            matched_negative=[],
            score=score,
            rejection_reason=None,
        )

    def is_relevant(self, title: str, abstract: str = "", content: str = "") -> bool:
        """Returns True if the text passes positive concepts and negative blocking checks."""
        return self.evaluate(title, abstract, content).is_relevant


MARKET_QUERY_PRESETS: Final[dict[MarketRegime, MarketHarvestQuery]] = {
    MarketRegime.INDIA_NSE_BSE: MarketHarvestQuery(
        market_scope=MarketRegime.INDIA_NSE_BSE,
        keywords=[
            "National Stock Exchange",
            "NSE",
            "BSE",
            "Nifty 50",
            "Bank Nifty",
            "Securities Transaction Tax",
            "STT",
            "SEBI intraday",
            "Indian equity momentum",
        ],
        arxiv_categories=["q-fin.TR", "q-fin.PM", "q-fin.ST", "q-fin.CP", "q-fin.RM"],
        openalex_concepts=["National Stock Exchange of India", "Stock market", "Transaction cost"],
        start_year=2018,
        max_results=50,
    ),
    MarketRegime.GERMANY_XETRA_DAX: MarketHarvestQuery(
        market_scope=MarketRegime.GERMANY_XETRA_DAX,
        keywords=[
            "Xetra",
            "Deutsche Börse",
            "DAX 40",
            "MDAX",
            "Eurex",
            "MiFID II tick size",
            "closing auction liquidity",
            "electronic order book Germany",
        ],
        arxiv_categories=["q-fin.TR", "q-fin.PM", "q-fin.ST", "q-fin.CP", "q-fin.RM"],
        openalex_concepts=["Deutsche Börse", "Financial market", "Order book"],
        start_year=2018,
        max_results=50,
    ),
    MarketRegime.US_NYSE_NASDAQ: MarketHarvestQuery(
        market_scope=MarketRegime.US_NYSE_NASDAQ,
        keywords=[
            "NYSE",
            "NASDAQ",
            "Reg NMS",
            "limit order book microstructure",
            "post-earnings announcement drift",
            "PEAD",
            "ETF statistical arbitrage",
            "market maker inventory",
            "Almgren-Chriss",
        ],
        arxiv_categories=["q-fin.TR", "q-fin.PM", "q-fin.ST", "q-fin.CP", "q-fin.RM"],
        openalex_concepts=["New York Stock Exchange", "NASDAQ", "High-frequency trading"],
        start_year=2018,
        max_results=50,
    ),
}


def build_arxiv_query_url(query: MarketHarvestQuery) -> str:
    """Constructs URL for arXiv Atom XML API."""
    cat_terms = [f"cat:{cat}" for cat in query.arxiv_categories]
    cat_query = " OR ".join(cat_terms)

    kw_terms = [f'all:"{kw}"' if " " in kw else f"all:{kw}" for kw in query.keywords[:5]]
    kw_query = " OR ".join(kw_terms)

    search_query = f"({cat_query}) AND ({kw_query})"
    params = {
        "search_query": search_query,
        "start": "0",
        "max_results": str(query.max_results),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    encoded = urllib.parse.urlencode(params)
    return f"http://export.arxiv.org/api/query?{encoded}"


OPENALEX_CONCEPT_MAP: Final[dict[str, str]] = {
    "Market microstructure": "C51926234",
    "Algorithmic trading": "C78508483",
    "High-frequency trading": "C24683644",
    "Stock market": "C2780299701",
    "Order book": "C2779309563",
    "Market liquidity": "C183582576",
    "Transaction cost": "C98965940",
    "Financial market": "C19244329",
    "Financial economics": "C106159729",
    "National Stock Exchange of India": "C2780299701",
    "Deutsche Börse": "C19244329",
    "New York Stock Exchange": "C2780299701",
    "NASDAQ": "C2780299701",
}


def build_openalex_query_url(
    query: MarketHarvestQuery,
    sort_by: str = "cited_by_count:desc",
    email: str = "scout@example.com",
) -> str:
    """Constructs a valid URL for OpenAlex works REST API with verified concept IDs and filters."""
    resolved_concept_ids: list[str] = []
    for concept in query.openalex_concepts:
        cleaned = concept.strip()
        if re.match(r"^C\d+$", cleaned):
            resolved_concept_ids.append(cleaned)
        elif cleaned in OPENALEX_CONCEPT_MAP:
            resolved_concept_ids.append(OPENALEX_CONCEPT_MAP[cleaned])

    # De-duplicate concept IDs preserving order
    unique_concept_ids = list(dict.fromkeys(resolved_concept_ids))

    filter_terms: list[str] = [
        f"from_publication_date:{query.start_year}-01-01",
        "has_fulltext:true",
        "is_oa:true",
        "type:article|preprint",
        "is_retracted:false",
    ]
    if unique_concept_ids:
        filter_terms.append(f"concepts.id:{'|'.join(unique_concept_ids)}")

    # Format search keywords: quote multi-word terms for exact phrase matching
    kw_terms = [f'"{kw}"' if " " in kw else kw for kw in query.keywords[:4]]
    search_query = " OR ".join(kw_terms)

    params: dict[str, str] = {
        "filter": ",".join(filter_terms),
        "search": search_query,
        "sort": sort_by,
        "per-page": str(min(query.max_results, 200)),
        "mailto": email,
    }
    encoded = urllib.parse.urlencode(params)
    return f"https://api.openalex.org/works?{encoded}"


def resolve_latex_includes(
    root_name: str, tex_files: dict[str, str], visited: set[str] | None = None
) -> str:
    """Recursively inlines \\input, \\include, and \\subfile into root document."""
    if visited is None:
        visited = set()
    if root_name in visited:
        return ""
    visited.add(root_name)

    content = tex_files.get(root_name, "")

    def replace_directive(match: re.Match[str]) -> str:
        target = match.group(1).strip()
        target = target.strip('"\'{} ')
        candidates = [
            target,
            f"{target}.tex",
            target.replace("./", ""),
            f"{target.replace('./', '')}.tex",
            target.split("/")[-1],
            f"{target.split('/')[-1]}.tex",
        ]
        for cand in candidates:
            cand_lower = cand.lower()
            for key in tex_files:
                if (
                    key == cand
                    or key.lower() == cand_lower
                    or key.endswith("/" + cand)
                    or key.lower().endswith("/" + cand_lower)
                ):
                    sub_text = resolve_latex_includes(key, tex_files, visited)
                    sub_match = re.search(
                        r"\\begin\{document\}(.*?)\\end\{document\}", sub_text, re.DOTALL
                    )
                    if sub_match:
                        return sub_match.group(1)
                    return sub_text
        return match.group(0)

    merged = re.sub(r"\\(?:input|include|subfile)\s*\{([^}]+)\}", replace_directive, content)
    return merged


def unpack_latex_archive(raw_bytes: bytes) -> str | None:
    """Extracts and merges multi-file LaTeX sources from an archive or standalone gzip."""
    if raw_bytes.startswith(b"%PDF"):
        return None

    tex_files: dict[str, str] = {}

    # 1. Try opening as tar archive (handles .tar.gz, .tar.bz2, .tar)
    try:
        with tarfile.open(fileobj=io.BytesIO(raw_bytes), mode="r:*") as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.lower().endswith(".tex"):
                    extracted = tar.extractfile(member)
                    if extracted:
                        raw_member = extracted.read()
                        try:
                            tex_files[member.name] = raw_member.decode("utf-8")
                        except UnicodeDecodeError:
                            tex_files[member.name] = raw_member.decode("latin-1", errors="replace")
    except (tarfile.TarError, gzip.BadGzipFile, EOFError, OSError, zlib.error):
        pass

    # 2. If no tex files found via tar, try standalone gzip decompression
    if not tex_files:
        try:
            decompressed = gzip.decompress(raw_bytes)
            # Check if decompressed bytes are themselves a tar file
            try:
                with tarfile.open(fileobj=io.BytesIO(decompressed), mode="r:") as tar:
                    for member in tar.getmembers():
                        if member.isfile() and member.name.lower().endswith(".tex"):
                            extracted = tar.extractfile(member)
                            if extracted:
                                raw_member = extracted.read()
                                try:
                                    tex_files[member.name] = raw_member.decode("utf-8")
                                except UnicodeDecodeError:
                                    tex_files[member.name] = raw_member.decode(
                                        "latin-1", errors="replace"
                                    )
            except (tarfile.TarError, OSError, EOFError):
                # Single standalone .tex file
                try:
                    text = decompressed.decode("utf-8")
                except UnicodeDecodeError:
                    text = decompressed.decode("latin-1", errors="replace")
                if "\\documentclass" in text or "\\begin{document}" in text or "%" in text:
                    return text
        except (gzip.BadGzipFile, OSError, UnicodeDecodeError, EOFError, zlib.error):
            pass

    if not tex_files:
        # Check if raw_bytes was already uncompressed text
        try:
            plain = raw_bytes.decode("utf-8")
            if "\\documentclass" in plain or "\\begin{document}" in plain:
                return plain
        except UnicodeDecodeError:
            pass
        return None

    # 3. Identify root .tex file (scoring out journal guides/templates in subdirectories)
    def _root_score(item: tuple[str, str]) -> tuple[int, int]:
        name, content = item
        lower = name.lower()
        score = 0
        if "\\documentclass" in content and "\\begin{document}" in content:
            score += 500
        elif "\\documentclass" in content or "\\begin{document}" in content:
            score += 200
        if any(bad in lower for bad in ("guide", "template", "sample", "howto", "instructions")):
            score -= 1000
        if "/" not in name:
            score += 100
        if any(k in lower for k in ("main", "ms", "paper", "article", "introduction")):
            score += 50
        return (score, len(content))

    root_candidate = max(tex_files.items(), key=_root_score)[0]

    # 4. Recursively resolve \input, \include, and \subfile
    return resolve_latex_includes(root_candidate, tex_files)


def extract_tex_title_and_abstract(
    tex_content: str, fallback_title: str
) -> tuple[str, str]:
    """Extracts clean academic title and abstract from raw LaTeX manuscript source."""
    clean = re.sub(r"(?<!\\)%.*", "", tex_content)
    title = ""

    # 1. Standard \title{...}, \Title{...}, \icmltitle{...}, \ShortTitle{...}
    title_patterns = [
        r"\\(?:title|Title|icmltitle|ShortTitle|runningtitle)\s*(?:\[[^\]]*\])?\s*\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})+)\}",
        r"\{\\fontsize\{22\}\{27\}\\selectfont\\bfseries\s*([^}]+)\}",
        r"\{\\fontsize\{15pt\}\{15pt\}\\textbf\{([^}]+)\}\}",
        r"\{\\sffamily\\bfseries\\fontsize\{20\}\{22\}\\selectfont\s*([^}]+?)(?:\\par|\})",
        r"\\begin\{center\}\s*\{\\bf\s+([^}]+)\}",
    ]
    for pat in title_patterns:
        m = re.search(pat, clean, re.DOTALL)
        if m:
            raw_t = m.group(1).strip()
            if raw_t == r"\fulltitle":
                continue
            raw_t = re.sub(
                r"\\(?:thanks|footnote|affil|inst|author|orcidlink)\s*\{(?:[^{}]|\{[^{}]*\})*\}",
                " ",
                raw_t,
            )
            raw_t = re.sub(r"\\[a-zA-Z]+\s*", " ", raw_t)
            raw_t = re.sub(r"[\{\}\\\$\n\r\t]+", " ", raw_t)
            raw_t = re.sub(r"\s+", " ", raw_t).strip(" -:;,.!")
            if len(raw_t) > 6 and not raw_t.lower().startswith("quantitative finance"):
                title = raw_t[:220]
                break

    # 2. Extract abstract
    abstract = ""
    abs_m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", clean, re.DOTALL)
    if not abs_m:
        abs_m = re.search(
            r"\\abstract\s*\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})+)\}", clean, re.DOTALL
        )
    if abs_m:
        raw_a = abs_m.group(1)
        raw_a = re.sub(r"\\(?:noindent|vspace|hspace)\s*(?:\{[^}]*\})?", " ", raw_a)
        raw_a = re.sub(r"\s+", " ", raw_a).strip()
        if len(raw_a) > 20:
            abstract = raw_a[:2000]

    if not title and abstract:
        first_sentence = abstract.split(". ")[0].strip()
        if len(first_sentence) > 12:
            title = first_sentence[:140]

    return (title or fallback_title, abstract)


def parse_arxiv_atom_feed(xml_text: str, market_scope: MarketRegime) -> list[HarvestedPaper]:
    """Parses arXiv Atom XML feed into HarvestedPaper records."""
    if not xml_text.strip():
        return []

    root = ET.fromstring(xml_text)
    entries = root.findall("atom:entry", ARXIV_XML_NS)
    papers: list[HarvestedPaper] = []

    for entry in entries:
        raw_id = entry.findtext("atom:id", default="", namespaces=ARXIV_XML_NS).strip()
        doc_id = re.sub(r"^https?://arxiv\.org/abs/", "arxiv:", raw_id)
        # Strip trailing version if any, e.g. arxiv:2502.08192v1 -> arxiv:2502.08192
        doc_id = re.sub(r"v\d+$", "", doc_id)

        raw_title = entry.findtext("atom:title", default="", namespaces=ARXIV_XML_NS)
        title = re.sub(r"\s+", " ", raw_title).strip()

        raw_abstract = entry.findtext("atom:summary", default="", namespaces=ARXIV_XML_NS)
        abstract = re.sub(r"\s+", " ", raw_abstract).strip()

        pub_text = entry.findtext("atom:published", default="", namespaces=ARXIV_XML_NS).strip()
        try:
            pub_date = (
                date.fromisoformat(pub_text[:10]) if len(pub_text) >= 10 else date(2024, 1, 1)
            )
        except ValueError:
            pub_date = date(2024, 1, 1)

        authors: list[str] = []
        for author_elem in entry.findall("atom:author", ARXIV_XML_NS):
            name = author_elem.findtext("atom:name", default="", namespaces=ARXIV_XML_NS).strip()
            if name:
                authors.append(name)

        source_uri = raw_id
        for link in entry.findall("atom:link", ARXIV_XML_NS):
            if link.attrib.get("title") == "pdf":
                source_uri = link.attrib.get("href", source_uri)

        if doc_id and title:
            papers.append(
                HarvestedPaper(
                    doc_id=doc_id,
                    source=HarvesterSource.ARXIV,
                    title=title,
                    authors=authors,
                    published_date=pub_date,
                    market_scope=market_scope,
                    source_uri=source_uri,
                    abstract=abstract,
                )
            )

    return papers


def _reconstruct_openalex_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    if not inverted_index:
        return ""
    pos_word: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        for pos in positions:
            pos_word.append((pos, word))
    pos_word.sort(key=lambda x: x[0])
    return " ".join(w for _, w in pos_word)


def parse_openalex_response(
    data: dict[str, object], market_scope: MarketRegime
) -> list[HarvestedPaper]:
    """Parses OpenAlex JSON response into HarvestedPaper records."""
    results = data.get("results")
    if not isinstance(results, list):
        return []

    papers: list[HarvestedPaper] = []
    for item in results:
        if not isinstance(item, dict):
            continue

        raw_id = str(item.get("id", ""))
        doc_id = re.sub(r"^https?://openalex\.org/", "openalex:", raw_id)

        title = str(item.get("title", "")).strip()
        pub_date_str = str(item.get("publication_date", ""))
        try:
            pub_date = date.fromisoformat(pub_date_str) if pub_date_str else date(2024, 1, 1)
        except ValueError:
            pub_date = date(2024, 1, 1)

        authorships = item.get("authorships")
        authors: list[str] = []
        if isinstance(authorships, list):
            for a in authorships:
                if isinstance(a, dict):
                    author_obj = a.get("author")
                    if isinstance(author_obj, dict):
                        display_name = author_obj.get("display_name")
                        if display_name:
                            authors.append(str(display_name))

        raw_inv = item.get("abstract_inverted_index")
        abstract = ""
        if isinstance(raw_inv, dict):
            typed_inv: dict[str, list[int]] = {
                str(k): [int(p) for p in v]
                for k, v in raw_inv.items()
                if isinstance(v, list)
            }
            abstract = _reconstruct_openalex_abstract(typed_inv)

        doi = str(item.get("doi") or raw_id)
        best_oa_loc = item.get("best_oa_location")
        best_oa_pdf = (
            best_oa_loc.get("pdf_url")
            if isinstance(best_oa_loc, dict) and best_oa_loc.get("pdf_url")
            else None
        )
        open_access = item.get("open_access")
        oa_url = (
            open_access.get("oa_url")
            if isinstance(open_access, dict) and open_access.get("oa_url")
            else None
        )
        if not best_oa_pdf and oa_url:
            best_oa_pdf = str(oa_url)

        locations = item.get("locations")
        oa_locations: list[str] = []
        if isinstance(locations, list):
            for loc in locations:
                if isinstance(loc, dict):
                    if loc.get("pdf_url"):
                        oa_locations.append(str(loc.get("pdf_url")))
                    elif loc.get("landing_page_url"):
                        lp = str(loc.get("landing_page_url"))
                        if "pdf" in lp.lower() or "arxiv.org" in lp.lower():
                            oa_locations.append(lp)
        if oa_url and str(oa_url) not in oa_locations:
            oa_locations.append(str(oa_url))

        metadata: dict[str, object] = {
            "cited_by_count": int(item.get("cited_by_count") or 0),
            "doi": doi,
            "best_oa_pdf_url": best_oa_pdf,
            "oa_url": oa_url,
            "oa_locations": oa_locations,
        }

        if doc_id and title:
            papers.append(
                HarvestedPaper(
                    doc_id=doc_id,
                    source=HarvesterSource.OPENALEX,
                    title=title,
                    authors=authors,
                    published_date=pub_date,
                    market_scope=market_scope,
                    source_uri=doi,
                    abstract=abstract,
                    metadata=metadata,
                )
            )

    return papers


@runtime_checkable
class PaperHarvesterClient(Protocol):
    """Protocol for fetching papers across markets."""

    def fetch_papers(self, query: MarketHarvestQuery) -> list[HarvestedPaper]:
        """Fetches papers matching the market harvest query."""
        ...

    def download_source(self, paper: HarvestedPaper) -> HarvestedPaper:
        """Downloads or resolves the paper's raw LaTeX source or PDF bytes."""
        ...


class HttpPaperHarvesterClient:
    """Production client with arXiv $3.5s \times 2^n$ exponential backoff and polite headers."""

    def __init__(
        self,
        base_backoff_sec: float = 3.5,
        max_retries: int = 3,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout_sec: float = 30.0,
    ) -> None:
        self.base_backoff_sec = base_backoff_sec
        self.max_retries = max_retries
        self.user_agent = user_agent
        self.timeout_sec = timeout_sec
        self._last_arxiv_time: float = 0.0

    def _execute_http_get(self, url: str) -> str:
        if "export.arxiv.org" in url:
            elapsed = time.time() - self._last_arxiv_time
            if elapsed < 3.0:
                time.sleep(3.0 - elapsed)
            self._last_arxiv_time = time.time()

        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                    raw_bytes = resp.read()
                    return str(raw_bytes.decode("utf-8", errors="replace"))
            except urllib.error.HTTPError as err:
                if err.code == 429 and "openalex.org" in url:
                    logger.warning(
                        "OpenAlex rate limit reached (HTTP 429); skipping query for %s", url
                    )
                    return ""
                if attempt == self.max_retries:
                    logger.error("HTTP request failed after %d retries: %s", self.max_retries, err)
                    raise
                backoff = self.base_backoff_sec * (2**attempt) + random.uniform(0.1, 0.5)
                logger.warning(
                    "HTTP attempt %d failed (%s); backing off %.2fs", attempt + 1, err, backoff
                )
                time.sleep(backoff)
            except Exception as err:
                if attempt == self.max_retries:
                    logger.error("HTTP request failed after %d retries: %s", self.max_retries, err)
                    raise
                backoff = self.base_backoff_sec * (2**attempt) + random.uniform(0.1, 0.5)
                logger.warning(
                    "HTTP attempt %d failed (%s); backing off %.2fs", attempt + 1, err, backoff
                )
                time.sleep(backoff)
        return ""

    def fetch_papers(self, query: MarketHarvestQuery) -> list[HarvestedPaper]:
        """Fetches papers from arXiv and OpenAlex for the given market query."""
        papers: list[HarvestedPaper] = []

        # 1. Fetch arXiv feed
        try:
            arxiv_url = build_arxiv_query_url(query)
            arxiv_xml = self._execute_http_get(arxiv_url)
            arxiv_papers = parse_arxiv_atom_feed(arxiv_xml, query.market_scope)
            papers.extend(arxiv_papers)
        except Exception as e:
            logger.warning("arXiv harvesting encountered an error: %s", e)

        # 2. Fetch OpenAlex feed
        try:
            openalex_url = build_openalex_query_url(query)
            openalex_json = self._execute_http_get(openalex_url)
            parsed_json = json.loads(openalex_json)
            if isinstance(parsed_json, dict):
                openalex_papers = parse_openalex_response(parsed_json, query.market_scope)
                papers.extend(openalex_papers)
        except Exception as e:
            logger.warning("OpenAlex harvesting encountered an error: %s", e)

        return papers

    def download_source(self, paper: HarvestedPaper) -> HarvestedPaper:
        """Attempts to retrieve LaTeX source for an arXiv paper, falling back to PDF."""
        if paper.source == HarvesterSource.ARXIV and paper.raw_tex_content is None:
            arxiv_num = paper.doc_id.replace("arxiv:", "")
            src_url = f"https://arxiv.org/e-print/{arxiv_num}"
            raw_tex: str | None = None
            raw_pdf: bytes | None = None

            # 1. Attempt /e-print/ download
            try:
                req = urllib.request.Request(src_url, headers={"User-Agent": self.user_agent})
                with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                    raw_bytes = resp.read()
                    if raw_bytes.startswith(b"%PDF"):
                        raw_pdf = raw_bytes
                    else:
                        raw_tex = unpack_latex_archive(raw_bytes)
                        if not raw_tex and raw_bytes.startswith(b"%PDF"):
                            raw_pdf = raw_bytes
            except Exception as e:
                logger.info(
                    "e-print source unavailable for %s (%s); will attempt PDF fallback",
                    paper.doc_id,
                    e,
                )

            # 2. PDF Fallback if no LaTeX source extracted
            if not raw_tex and not raw_pdf:
                pdf_url = f"https://arxiv.org/pdf/{arxiv_num}.pdf"
                try:
                    pdf_req = urllib.request.Request(
                        pdf_url, headers={"User-Agent": self.user_agent}
                    )
                    with urllib.request.urlopen(pdf_req, timeout=self.timeout_sec) as resp:
                        pdf_bytes = resp.read()
                        if pdf_bytes.startswith(b"%PDF"):
                            raw_pdf = pdf_bytes
                except Exception as e:
                    logger.warning("Failed PDF fallback for %s: %s", paper.doc_id, e)

            return HarvestedPaper(
                doc_id=paper.doc_id,
                source=paper.source,
                title=paper.title,
                authors=paper.authors,
                published_date=paper.published_date,
                market_scope=paper.market_scope,
                source_uri=paper.source_uri,
                abstract=paper.abstract,
                raw_tex_content=raw_tex,
                raw_pdf_bytes=raw_pdf,
                metadata=paper.metadata,
            )

        if (
            paper.source == HarvesterSource.OPENALEX
            and paper.raw_tex_content is None
            and paper.raw_pdf_bytes is None
        ):
            best_oa_pdf_url = paper.metadata.get("best_oa_pdf_url")
            candidate_urls: list[str] = []
            if isinstance(best_oa_pdf_url, str) and best_oa_pdf_url:
                candidate_urls.append(best_oa_pdf_url)
            oa_url = paper.metadata.get("oa_url")
            if isinstance(oa_url, str) and oa_url and oa_url not in candidate_urls:
                candidate_urls.append(oa_url)
            oa_locs = paper.metadata.get("oa_locations")
            if isinstance(oa_locs, list):
                for loc in oa_locs:
                    if isinstance(loc, str) and loc and loc not in candidate_urls:
                        candidate_urls.append(loc)

            raw_tex_oa: str | None = None
            raw_pdf_oa: bytes | None = None
            for url in candidate_urls:
                # If URL routes to arXiv, attempt high-signal source extraction
                arxiv_m = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]+\.[0-9]+(?:v[0-9]+)?)", url)
                if arxiv_m:
                    arxiv_id = arxiv_m.group(1)
                    try:
                        req = urllib.request.Request(
                            f"https://arxiv.org/e-print/{arxiv_id}",
                            headers={"User-Agent": self.user_agent},
                        )
                        with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                            data = resp.read()
                            if data.startswith(b"%PDF"):
                                raw_pdf_oa = data
                                break
                            raw_tex_oa = unpack_latex_archive(data)
                            if raw_tex_oa:
                                break
                    except Exception:
                        pass

                    if not raw_tex_oa and not raw_pdf_oa:
                        try:
                            req = urllib.request.Request(
                                f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                                headers={"User-Agent": self.user_agent},
                            )
                            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                                data = resp.read()
                                if data.startswith(b"%PDF"):
                                    raw_pdf_oa = data
                                    break
                        except Exception:
                            pass

                try:
                    req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
                    with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                        pdf_bytes = resp.read()
                        if pdf_bytes.startswith(b"%PDF"):
                            raw_pdf_oa = pdf_bytes
                            break
                except Exception:
                    continue

            if raw_tex_oa or raw_pdf_oa:
                return HarvestedPaper(
                    doc_id=paper.doc_id,
                    source=paper.source,
                    title=paper.title,
                    authors=paper.authors,
                    published_date=paper.published_date,
                    market_scope=paper.market_scope,
                    source_uri=paper.source_uri,
                    abstract=paper.abstract,
                    raw_tex_content=raw_tex_oa,
                    raw_pdf_bytes=raw_pdf_oa,
                    metadata=paper.metadata,
                )

        return paper


# Deterministic offline test fixtures for IN_NSE, DE_XETRA, and US_EQUITY
FIXTURE_PAPERS: Final[dict[MarketRegime, list[HarvestedPaper]]] = {
    MarketRegime.INDIA_NSE_BSE: [
        HarvestedPaper(
            doc_id="arxiv:2403.01182",
            source=HarvesterSource.ARXIV,
            title=(
                "Momentum and Reversal Dynamics in Indian Equities with "
                "Securities Transaction Tax"
            ),
            authors=["Arjun Sharma", "Priya Venkatesh"],
            published_date=date(2024, 3, 15),
            market_scope=MarketRegime.INDIA_NSE_BSE,
            source_uri="https://arxiv.org/abs/2403.01182",
            abstract=(
                "We study momentum strategies across Nifty 50 and Nifty 500 equities "
                "from 2018 to 2023. While gross annualized returns achieve 22.4% with a "
                "Sharpe ratio of 1.45, incorporating Indian market friction (STT 10 bps "
                "on delivery per leg, GST 18%, and exchange turnover fees) requires a "
                "minimum round-trip friction floor of 15 bps, reducing net Sharpe to 0.92."
            ),
            raw_tex_content=r"""\documentclass{article}
\begin{document}
\section{Introduction}
Momentum phenomena in emerging equity markets exhibit significant turnover penalties.
In the Indian National Stock Exchange (NSE), statutory levies such as
Securities Transaction Tax (STT) fundamentally shift profitability thresholds.

\section{Methodology and Signal Formulation}
Let $R_{i,t}$ denote the simple return of asset $i$ at month $t$.
The cross-sectional momentum signal is formulated as:
\begin{equation}
S_{i,t} = \sum_{k=2}^{12} \log(1 + R_{i,t-k}) - \beta_i R_{m,t}
\end{equation}
where $\beta_i$ is estimated against the Nifty 50 benchmark over a 36-month rolling window.

\subsection{Market Friction and Statutory Taxation}
Trading friction in the cash delivery segment comprises:
\begin{align}
C_{\text{roundtrip}} &= 2 \times \tau_{\text{STT}} + \tau_{\text{brokerage}}
+ \tau_{\text{turnover}} + \tau_{\text{GST}} \\
&\ge 15.0\text{ bps}
\end{align}
Strategies assuming less than 15 bps delivery friction severely overestimate net alpha.

\section{Empirical Results}
Table 1 outlines performance across friction specifications.
\begin{table}[h]
\begin{tabular}{|l|c|c|c|}
\hline
Friction Assumption & Annualized Return & Net Sharpe & Max Drawdown \\
\hline
Zero Cost & 22.4\% & 1.45 & -18.2\% \\
8 bps (Unrealistic) & 18.1\% & 1.18 & -21.4\% \\
15 bps (Realistic STT) & 14.8\% & 0.92 & -24.1\% \\
\hline
\end{tabular}
\end{table}
\end{document}
""",
            metadata={
                "strategy_type": "cross_sectional_momentum",
                "sample_period": "2018-2023",
                "sample_period_years": "2018-2023",
                "assumed_friction_bps": 15.0,
            },
        )
    ],
    MarketRegime.GERMANY_XETRA_DAX: [
        HarvestedPaper(
            doc_id="arxiv:2311.08941",
            source=HarvesterSource.ARXIV,
            title="Closing Auction Dynamics and Tick Size Regimes on Deutsche Börse Xetra",
            authors=["Hans Weber", "Klaus Müller"],
            published_date=date(2023, 11, 20),
            market_scope=MarketRegime.GERMANY_XETRA_DAX,
            source_uri="https://arxiv.org/abs/2311.08941",
            abstract=(
                "Analyzing continuous trading and closing auction liquidity on Xetra "
                "DAX 40 under MiFID II. Effective half-spreads combined with CCP "
                "clearing fees and short borrow costs establish a conservative "
                "10 bps friction floor for German equity rebalancing."
            ),
            raw_tex_content=r"""\documentclass{article}
\begin{document}
\section{Market Microstructure of Xetra}
Trading on Deutsche Börse Xetra operates under MiFID II tick size tables.
Order execution during continuous trading incurs spread and clearing costs.

\section{Spread Estimation}
The effective percentage half-spread is defined as:
\begin{equation}
S_{\text{eff}} = \frac{|P_t - M_t|}{M_t} \times 10^4
\end{equation}
where $M_t = \frac{P_t^a + P_t^b}{2}$ is the prevailing midpoint quote.
Total round-trip rebalancing friction on DAX constituents averages 10 bps.
\end{document}
""",
            metadata={
                "strategy_type": "auction_imbalance_arbitrage",
                "sample_period": "2019-2023",
                "sample_period_years": "2019-2023",
                "assumed_friction_bps": 10.0,
            },
        )
    ],
    MarketRegime.US_NYSE_NASDAQ: [
        HarvestedPaper(
            doc_id="arxiv:2401.05432",
            source=HarvesterSource.ARXIV,
            title=(
                "High-Frequency Limit Order Book Imbalance and Optimal Execution "
                "in US Equities"
            ),
            authors=["David Chen", "Sarah Jenkins"],
            published_date=date(2024, 1, 10),
            market_scope=MarketRegime.US_NYSE_NASDAQ,
            source_uri="https://arxiv.org/abs/2401.05432",
            abstract=(
                "We evaluate queue position and order flow imbalance across NYSE and "
                "NASDAQ listed equities. Using the Almgren-Chriss framework with SEC "
                "Section 31 and FINRA TAF fees, a 5 bps friction floor accurately "
                "bounds institutional trade execution slippage."
            ),
            raw_tex_content=r"""\documentclass{article}
\begin{document}
\section{Order Flow Imbalance Signal}
We compute instantaneous order flow imbalance (OFI) as:
\begin{equation}
\text{OFI}_t = I_{\{P_t^b \ge P_{t-1}^b\}} q_t^b - I_{\{P_t^b \le P_{t-1}^b\}} q_{t-1}^b
- I_{\{P_t^a \le P_{t-1}^a\}} q_t^a + I_{\{P_t^a \ge P_{t-1}^a\}} q_{t-1}^a
\end{equation}
Execution friction in liquid US large caps incurs effective cost:
\begin{equation}
C_{\text{US}} = \text{HalfSpread} + \tau_{\text{SEC}} + \tau_{\text{FINRA}} \ge 5.0\text{ bps}
\end{equation}
\end{document}
""",
            metadata={
                "strategy_type": "order_flow_imbalance",
                "sample_period": "2020-2023",
                "sample_period_years": "2020-2023",
                "assumed_friction_bps": 5.0,
            },
        )
    ],
}



class MockPaperHarvesterClient:
    """Deterministic offline harvester client delivering genuine quant papers across markets."""

    def __init__(
        self, custom_fixtures: dict[MarketRegime, list[HarvestedPaper]] | None = None
    ) -> None:
        self.fixtures = custom_fixtures or FIXTURE_PAPERS

    def fetch_papers(self, query: MarketHarvestQuery) -> list[HarvestedPaper]:
        """Returns deterministic fixture papers matching market scope."""
        papers = self.fixtures.get(query.market_scope, [])
        return list(papers[: query.max_results])

    def download_source(self, paper: HarvestedPaper) -> HarvestedPaper:
        """Returns paper with preloaded LaTeX content."""
        return paper


__all__ = [
    "DEFAULT_USER_AGENT",
    "FIXTURE_PAPERS",
    "MARKET_QUERY_PRESETS",
    "NEGATIVE_BLOCKING_KEYWORDS",
    "OPENALEX_CONCEPT_MAP",
    "POSITIVE_QUANT_CONCEPTS",
    "QUANT_ARXIV_CATEGORIES",
    "HarvestedPaper",
    "HarvesterSource",
    "HttpPaperHarvesterClient",
    "MarketHarvestQuery",
    "MockPaperHarvesterClient",
    "PaperHarvesterClient",
    "QuantRelevanceClassifier",
    "RelevanceVerdict",
    "build_arxiv_query_url",
    "build_openalex_query_url",
    "extract_tex_title_and_abstract",
    "parse_arxiv_atom_feed",
    "parse_openalex_response",
    "resolve_latex_includes",
    "unpack_latex_archive",
]
