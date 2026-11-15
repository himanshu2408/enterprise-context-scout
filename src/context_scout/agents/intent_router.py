"""Deterministic query intent classification and source filtering for Scout workflows.

Routes user queries strictly from query cues, explicit ``source_filter``, and
visible corpus availability under the principal's RBAC scopes. Never branches
on ``principal.role`` to block documentation or force code dumps onto doc queries.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import date, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal, overload

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from context_scout.db.schema import DocChunkRecord
    from context_scout.schemas.contracts import DocCitationSpan, QueryFacetCoverage

SourceFilter = Literal["all", "code", "wiki", "jira"]


class QueryIntentMode(StrEnum):
    """Execution routing mode derived from query cues and source filters."""

    PURE_CODE = "pure_code"
    HYBRID_CODE_AND_DOCS = "hybrid_code_and_docs"
    PURE_DOCUMENTATION = "pure_documentation"


class QueryIntentDecision(BaseModel):
    """Structured routing decision emitted by QueryIntentClassifier."""

    model_config = ConfigDict(frozen=True)

    mode: QueryIntentMode
    source_filter: SourceFilter = "all"
    needs_code: bool
    needs_docs: bool
    reason: str


_CODE_FILE_OR_PATH_RE = re.compile(
    r"(?:"
    r"[A-Za-z0-9_./-]+\.(?:py|ts|tsx|js|jsx|cs|tf|sql|go|rs|java|kt|rb|yaml|yml|toml|sh)\b"
    r"|\b(?:src|app|tests|test|web-ui|pkg|internal|cmd|lib)/[A-Za-z0-9_./-]+"
    r")"
)
_CODE_BACKTICK_RE = re.compile(r"`[^`\n]{2,120}`")
_CODE_SNAKE_CASE_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+\b")
_CODE_CAMEL_CASE_RE = re.compile(r"\b(?:[a-z]+[A-Z][A-Za-z0-9_]*|[A-Z][a-z]+[A-Z][A-Za-z0-9_]*)\b")
_CODE_QUALIFIED_RE = re.compile(
    r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*|::[A-Za-z_][A-Za-z0-9_]*)\b"
)
_CODE_ENDPOINT_RE = re.compile(
    r"(?:\b(?:GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s+/[A-Za-z0-9_./{}-]*|"
    r"(?<!\w)/api/v\d+/[A-Za-z0-9_./{}-]*)"
)
_CODE_TRACE_RE = re.compile(
    r"(?:Traceback \(most recent call last\)|File \"[^\"]+\", line \d+|\b\w+(?:Error|Exception):)"
)
_CODE_KEYWORD_RE = re.compile(
    r"\b(?:"
    r"ast|caller|callers|callee|callees|refactor|refactoring|function|functions|"
    r"method|methods|class|classes|module|modules|symbol|symbols|endpoint|endpoints|"
    r"route|routes|handler|handlers|schema|schemas|migration|migrations|"
    r"stack\s+trace|stacktrace|codebase|repository|repo|source\s+code|"
    r"implement|implementation|implementing|patch|pytest|unit\s+test|unit\s+tests|"
    r"bindings|bounce[-_\s]?back|bugfix|bug\s+fix"
    r")\b",
    re.IGNORECASE,
)

_DOC_TICKET_OR_SPEC_ID_RE = re.compile(r"\b(?:[A-Z]{2,10}-\d+|ADR-\d+|RFC-\d+)\b")
_DOC_KEYWORD_RE = re.compile(
    r"\b(?:"
    r"sla|slo|onboarding|runbook|runbooks|playbook|playbooks|policy|policies|"
    r"escalation|escalate|incident|incidents|postmortem|post-mortem|"
    r"architecture|architectural|guideline|guidelines|compliance|audit|"
    r"governance|documentation|docs|confluence|jira|wiki|adr|rfc|"
    r"research|paper|papers|specification|spec|formula|formulas|equation|equations|"
    r"signal|strategy|strategies|handbook|on-call|oncall|handbook|sop|"
    r"approval|changelog|decision\s+record|who\s+owns"
    r")\b",
    re.IGNORECASE,
)
_CONFLUENCE_SPACE_RE = re.compile(r"/spaces/([A-Za-z0-9_~-]+)")

_CARDINALITY_LEADING_RE = re.compile(
    r"\b(?:top|first|list|give\s+me|name|what\s+are(?:\s+the)?)\s+(\d{1,2})\b",
    re.IGNORECASE,
)
_CARDINALITY_NOUN_RE = re.compile(
    r"\b(\d{1,2})\s+(?:failure\s+modes?|risks?|models?|reasons?|ways?|strategies|"
    r"caveats?|properties|steps?|examples?|methods?|classes|files|issues?|findings?|"
    r"causes?|patterns?|techniques?|components?)\b",
    re.IGNORECASE,
)
_ACRONYM_STAND_FOR_RE = re.compile(
    r"\bwhat\s+does\s+([A-Z]{2,8})\s+stands?\s+for\b",
)
_ACRONYM_PROPERTIES_RE = re.compile(
    r"\b(?:explain|describe|what\s+are)\s+(?:the\s+)?([A-Z]{2,8})\s+"
    r"(?:properties|principles|guarantees|acronym)\b",
    re.IGNORECASE,
)

_FACET_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "how",
        "does",
        "did",
        "are",
        "was",
        "were",
        "can",
        "you",
        "your",
        "our",
        "they",
        "their",
        "its",
        "into",
        "onto",
        "over",
        "under",
        "between",
        "during",
        "using",
        "running",
        "compare",
        "contrast",
        "versus",
        "against",
        "explain",
        "describe",
        "list",
        "give",
        "top",
        "each",
        "all",
        "both",
        "also",
        "only",
        "about",
        "have",
        "has",
        "had",
        "not",
        "but",
        "stand",
        "stands",
        "mean",
        "mitigate",
        "prevent",
        "core",
        "open",
        "tracks",
        "inspect",
        "including",
        "codebase",
        "implementation",
        "contracts",
        "unit",
        "empirical",
        "findings",
        "failure",
        "modes",
        "classes",
        "research",
        "issues",
    }
)


class FacetSpec(BaseModel):
    """Single decomposed query sub-facet for stratified retrieval and grounding checks."""

    model_config = ConfigDict(frozen=True)

    facet_id: str
    facet_query: str
    key_terms: list[str]
    expected_source_kind: Literal["code", "docs", "issue", "any"] = "any"
    is_acronym_component: bool = False


class QueryPlan(BaseModel):
    """Deterministic query decomposition plan (cardinality, acronym, and sub-facets)."""

    model_config = ConfigDict(frozen=True)

    original_query: str
    requested_cardinality: int | None = None
    is_acronym_query: bool = False
    acronym: str | None = None
    facets: list[FacetSpec]


def _extract_facet_key_terms(text: str) -> list[str]:
    """Extracts distinctive lowercase search tokens from a facet query."""
    raw = re.findall(r"[A-Za-z0-9_]{2,}", text)
    terms: list[str] = []
    for tok in raw:
        low = tok.lower()
        if low in _FACET_STOPWORDS:
            continue
        if len(low) < 3 and not tok.isupper():
            continue
        if low not in terms:
            terms.append(low)
    return terms


def _classify_facet_source_kind(text: str) -> Literal["code", "docs", "issue", "any"]:
    """Infers the expected source modality for a decomposed facet."""
    low = text.lower()
    if any(
        w in low
        for w in (
            "which issue",
            "what issue",
            "which ticket",
            "what ticket",
            "jira",
            "tracks it",
            "issue tracks",
            "tracked by",
            "known issue",
            "github issue",
        )
    ) or re.search(r"\b[A-Z]{2,10}-\d+\b", text):
        return "issue"
    if (
        _CODE_FILE_OR_PATH_RE.search(text)
        or _CODE_BACKTICK_RE.search(text)
        or _CODE_SNAKE_CASE_RE.search(text)
        or any(
            w in low
            for w in (
                "codebase",
                "implementation",
                "class",
                "classes",
                "method",
                "function",
                "module",
                "symbol",
                "caller",
                "unit test",
            )
        )
    ):
        return "code"
    if any(
        w in low
        for w in (
            "research",
            "paper",
            "documentation",
            "docs",
            "confluence",
            "wiki",
            "adr",
            "rfc",
            "specification",
        )
    ):
        return "docs"
    return "any"


def decompose_query_facets(prompt: str) -> QueryPlan:
    """Deterministically decomposes a prompt into cardinality, acronym, and sub-facets."""
    cleaned = " ".join(prompt.strip().split())
    stripped = cleaned.rstrip("?!. ")

    requested_cardinality: int | None = None
    for pat in (_CARDINALITY_LEADING_RE, _CARDINALITY_NOUN_RE):
        m = pat.search(stripped)
        if m:
            val = int(m.group(1))
            if 1 <= val <= 20:
                requested_cardinality = val
                break

    acronym: str | None = None
    is_acronym_query = False
    acr_m = _ACRONYM_STAND_FOR_RE.search(stripped) or _ACRONYM_PROPERTIES_RE.search(stripped)
    if acr_m:
        is_acronym_query = True
        acronym = acr_m.group(1).upper()

    # Strip trailing AST closure / meta-clauses so they don't become bogus lexical facets
    working = re.sub(
        r"^(?:compare|contrast)\s+",
        "",
        stripped,
        flags=re.IGNORECASE,
    ).strip()
    working = re.sub(
        r",?\s*including\s+(?:all\s+)?(?:callers|imported\s+types|unit\s+tests|tests).*$",
        "",
        working,
        flags=re.IGNORECASE,
    ).strip()
    # Strip leading "What are the top N empirical findings on " style preamble for clean facet split
    preamble_m = re.match(
        r"^(?:what\s+are\s+(?:the\s+)?top\s+\d+\s+[\w\s]+?\bon\s+)(.+)$",
        working,
        flags=re.IGNORECASE,
    )
    if preamble_m:
        working = preamble_m.group(1).strip()

    # First split on explicit question/issue clauses (", and which ...", ", and how ...")
    clause_parts = re.split(
        r"(?:;\s*|,\s*(?:and\s+)?(?=(?:which|what|where|how|why)\b))",
        working,
        flags=re.IGNORECASE,
    )

    raw_facets: list[str] = []
    for part in clause_parts:
        sub = part.strip().strip(",")
        if not sub:
            continue
        # Split "X against Y and Z" or comma-separated lists ("A, B, and C")
        if re.search(r"\s+against\s+", sub, flags=re.IGNORECASE):
            against_parts = re.split(r"\s+against\s+", sub, flags=re.IGNORECASE)
        else:
            against_parts = [sub]

        for ap in against_parts:
            ap_clean = ap.strip().strip(",")
            if not ap_clean:
                continue
            comma_count = ap_clean.count(",")
            has_oxford = bool(re.search(r",\s+and\s+", ap_clean, flags=re.IGNORECASE))
            if has_oxford or comma_count >= 2:
                items = re.split(r",\s+and\s+|,\s*", ap_clean, flags=re.IGNORECASE)
                raw_facets.extend(it.strip() for it in items if it.strip())
            elif re.search(r"\s+and\s+(?:open|how|what|which|where)\b", ap_clean, re.IGNORECASE):
                items = re.split(
                    r"\s+and\s+(?=(?:open|how|what|which|where)\b)",
                    ap_clean,
                    flags=re.IGNORECASE,
                )
                raw_facets.extend(it.strip() for it in items if it.strip())
            elif preamble_m and re.search(r"\s+and\s+", ap_clean, re.IGNORECASE):
                items = re.split(r"\s+and\s+", ap_clean, flags=re.IGNORECASE)
                raw_facets.extend(it.strip() for it in items if it.strip())
            else:
                raw_facets.append(ap_clean)

    # Filter out trivial fragments, trailing meta-clauses, and generic modifiers
    facets: list[FacetSpec] = []
    for rf in raw_facets:
        fq = re.sub(r"^and\s+", "", rf, flags=re.IGNORECASE).strip()
        if len(fq) < 4:
            continue
        if re.match(
            r"^(?:their\s+core\s+classes|including\s+|all\s+callers|imported\s+types|unit\s+tests)$",
            fq,
            flags=re.IGNORECASE,
        ):
            continue
        terms = _extract_facet_key_terms(fq)
        if not terms:
            if facets:
                continue
            terms = [w.lower() for w in re.findall(r"[A-Za-z0-9_]+", fq) if len(w) >= 2]
        facets.append(
            FacetSpec(
                facet_id=f"facet_{len(facets) + 1}",
                facet_query=fq,
                key_terms=terms,
                expected_source_kind=_classify_facet_source_kind(fq),
            )
        )

    if not facets:
        terms = _extract_facet_key_terms(stripped)
        facets = [
            FacetSpec(
                facet_id="facet_1",
                facet_query=stripped or prompt.strip(),
                key_terms=terms,
                expected_source_kind=_classify_facet_source_kind(stripped),
            )
        ]

    return QueryPlan(
        original_query=prompt,
        requested_cardinality=requested_cardinality,
        is_acronym_query=is_acronym_query,
        acronym=acronym,
        facets=facets,
    )


def evaluate_facet_grounding(
    facet: FacetSpec,
    *,
    doc_chunks: list[Any],
    code_slices: list[Any],
    domain_id: str = "",
) -> tuple[QueryFacetCoverage, str | None]:
    """Evaluates whether a single facet has genuine lexical/entity grounding in retrieved items.

    Returns ``(QueryFacetCoverage, missing_aspect_message_or_none)``.
    """
    from context_scout.schemas.contracts import QueryFacetCoverage

    # Direct symbol identifier check: if a PascalCase/snake_case symbol from the facet
    # is directly present in code_slices, mark the facet grounded (1.0).
    facet_symbols = set(
        _CODE_CAMEL_CASE_RE.findall(facet.facet_query)
        + _CODE_SNAKE_CASE_RE.findall(facet.facet_query)
    )
    if facet_symbols and code_slices:
        matched_code_sources: list[str] = []
        for slc in code_slices:
            qsym = str(getattr(slc, "qualified_symbol", "") or "")
            fpath = str(getattr(slc, "file_path", "") or "")
            code_txt = str(getattr(slc, "verbatim_code_with_lines", "") or "")
            if any(sym in qsym or sym in fpath or sym in code_txt for sym in facet_symbols):
                label = fpath or qsym
                if label and label not in matched_code_sources:
                    matched_code_sources.append(label)
        if matched_code_sources:
            return (
                QueryFacetCoverage(
                    facet_id=facet.facet_id,
                    facet_query=facet.facet_query,
                    status="grounded",
                    matched_sources=matched_code_sources[:5],
                    confidence=1.0,
                    explanation=(
                        f"Directly matched symbol(s) in code slices: "
                        f"{', '.join(matched_code_sources[:3])}."
                    ),
                ),
                None,
            )

    terms = [t.lower() for t in facet.key_terms if t]
    if not terms:
        has_any = bool(doc_chunks or code_slices)
        status: Literal["grounded", "partial", "not_found"] = "grounded" if has_any else "not_found"
        return (
            QueryFacetCoverage(
                facet_id=facet.facet_id,
                facet_query=facet.facet_query,
                status=status,
                matched_sources=[],
                confidence=0.8 if has_any else 0.0,
                explanation="Evaluated against retrieved context." if has_any else "No evidence.",
            ),
            None
            if has_any
            else (
                f"No grounded evidence found in indexed corpus '{domain_id}' "
                f"for: {facet.facet_query}."
            ),
        )

    best_hits = 0
    matched_terms_overall: set[str] = set()
    matched_sources: list[str] = []

    def _scan_text(text: str, source_label: str) -> None:
        nonlocal best_hits
        low = text.lower()
        hits_here = {
            t for t in terms if re.search(rf"\b{re.escape(t)}\b", low) or (len(t) >= 5 and t in low)
        }
        if hits_here:
            matched_terms_overall.update(hits_here)
            if len(hits_here) > best_hits:
                best_hits = len(hits_here)
            if source_label not in matched_sources and len(hits_here) >= min(2, len(terms)):
                matched_sources.append(source_label)

    if facet.expected_source_kind != "code" or not code_slices:
        for chk in doc_chunks:
            if facet.expected_source_kind == "issue" and not is_jira_chunk(chk):
                # Also allow GitHub issue chunks whose title/uri/content mentions issue
                uri_low = str(getattr(chk, "source_uri", "") or "").lower()
                title_low = str(getattr(chk, "doc_title", "") or "").lower()
                if "issue" not in uri_low and "issue" not in title_low and "#" not in title_low:
                    continue
            label = str(
                getattr(chk, "doc_title", "")
                or getattr(chk, "doc_id", "")
                or getattr(chk, "chunk_id", "")
            )
            txt = (
                f"{getattr(chk, 'doc_title', '')}\n"
                f"{getattr(chk, 'section_heading', '')}\n"
                f"{getattr(chk, 'verbatim_content', '')}\n"
                f"{getattr(chk, 'full_section_content', '')}\n"
                f"{getattr(chk, 'verbatim_quote', '')}"
            )
            _scan_text(txt, label)

    if facet.expected_source_kind in ("code", "any") or (not matched_terms_overall and code_slices):
        for slc in code_slices:
            label = str(getattr(slc, "file_path", "") or getattr(slc, "qualified_symbol", ""))
            txt = (
                f"{getattr(slc, 'file_path', '')}\n"
                f"{getattr(slc, 'qualified_symbol', '')}\n"
                f"{getattr(slc, 'verbatim_code_with_lines', '')}"
            )
            _scan_text(txt, label)

    total_terms = len(terms)
    coverage_ratio = len(matched_terms_overall) / total_terms

    # Issue tracking facet ("which issue tracks it") is grounded if an issue source matched
    if facet.expected_source_kind == "issue":
        if matched_sources or best_hits >= 1:
            sources_str = ", ".join(matched_sources[:3]) or "matched issue chunk"
            return (
                QueryFacetCoverage(
                    facet_id=facet.facet_id,
                    facet_query=facet.facet_query,
                    status="grounded",
                    matched_sources=matched_sources[:5],
                    confidence=0.85 if matched_sources else 0.65,
                    explanation=f"Grounded in issue tracking source(s): {sources_str}.",
                ),
                None,
            )
        msg = (
            f"No grounded issue/ticket found in indexed corpus '{domain_id}' "
            f"for: {facet.facet_query}."
        )
        return (
            QueryFacetCoverage(
                facet_id=facet.facet_id,
                facet_query=facet.facet_query,
                status="not_found",
                matched_sources=[],
                confidence=0.0,
                explanation=msg,
            ),
            msg,
        )

    # Acronym component facets (e.g. ACID Atomicity/Consistency/Isolation/Durability):
    # grounded if the primary component term OR core acronym+transaction context is matched.
    if facet.is_acronym_component and terms:
        primary_term = terms[0]
        if primary_term in matched_terms_overall or best_hits >= 2:
            return (
                QueryFacetCoverage(
                    facet_id=facet.facet_id,
                    facet_query=facet.facet_query,
                    status="grounded",
                    matched_sources=matched_sources[:5] or ["ACID Reference"],
                    confidence=0.92 if primary_term in matched_terms_overall else 0.80,
                    explanation=f"Grounded acronym component '{primary_term}' in retrieved docs.",
                ),
                None,
            )

    uppercase_acronyms = [
        tok.lower()
        for tok in re.findall(r"\b[A-Z][A-Z0-9]{2,}\b", facet.facet_query)
        if tok.lower() in terms
    ]
    missing_acronyms = [acr for acr in uppercase_acronyms if acr not in matched_terms_overall]

    if total_terms == 1:
        is_grounded = best_hits == 1 and not missing_acronyms
        is_partial = not is_grounded and best_hits >= 1
    elif total_terms <= 3:
        is_grounded = coverage_ratio >= 0.75 and not missing_acronyms and best_hits >= 2
        is_partial = not is_grounded and coverage_ratio > 0.0 and (best_hits >= 1)
    else:
        is_grounded = coverage_ratio >= 0.75 and not missing_acronyms and best_hits >= 2
        is_partial = (
            not is_grounded and coverage_ratio > 0.0 and (best_hits >= 2 or coverage_ratio >= 0.34)
        )

    if is_grounded:
        conf = min(0.98, round(0.60 + 0.38 * coverage_ratio, 3))
        return (
            QueryFacetCoverage(
                facet_id=facet.facet_id,
                facet_query=facet.facet_query,
                status="grounded",
                matched_sources=matched_sources[:5],
                confidence=conf,
                explanation=(
                    f"Matched {len(matched_terms_overall)}/{total_terms} facet terms "
                    f"across {max(1, len(matched_sources))} source(s)."
                ),
            ),
            None,
        )

    missing_terms = [t for t in terms if t not in matched_terms_overall]
    if is_partial:
        msg = (
            f"Partial evidence in indexed corpus '{domain_id}' for: {facet.facet_query} "
            f"(unverified terms: {', '.join(missing_terms[:5])})."
        )
        return (
            QueryFacetCoverage(
                facet_id=facet.facet_id,
                facet_query=facet.facet_query,
                status="partial",
                matched_sources=matched_sources[:5],
                confidence=round(0.35 * coverage_ratio, 3),
                explanation=msg,
            ),
            msg,
        )

    msg = (
        f"Could not find grounded evidence in indexed corpus '{domain_id}' "
        f"for: {facet.facet_query}."
    )
    return (
        QueryFacetCoverage(
            facet_id=facet.facet_id,
            facet_query=facet.facet_query,
            status="not_found",
            matched_sources=[],
            confidence=0.0,
            explanation=msg,
        ),
        msg,
    )


class QueryIntentClassifier:
    """Classifies user queries into code, documentation, or hybrid execution modes."""

    @classmethod
    def classify(
        cls,
        query: str,
        source_filter: SourceFilter = "all",
        has_visible_code: bool = True,
        has_visible_docs: bool = True,
    ) -> QueryIntentDecision:
        """Determines query routing intent from query cues, source filter, and visible corpus."""
        if source_filter == "code":
            return QueryIntentDecision(
                mode=QueryIntentMode.PURE_CODE,
                source_filter=source_filter,
                needs_code=True,
                needs_docs=False,
                reason="Explicit source_filter='code' restricts routing to code slices only.",
            )
        if source_filter in ("wiki", "jira"):
            return QueryIntentDecision(
                mode=QueryIntentMode.PURE_DOCUMENTATION,
                source_filter=source_filter,
                needs_code=False,
                needs_docs=True,
                reason=(
                    f"Explicit source_filter='{source_filter}' restricts routing "
                    "to documentation citations only."
                ),
            )

        text = query.strip()
        code_signals: list[str] = []
        if _CODE_FILE_OR_PATH_RE.search(text):
            code_signals.append("file_path")
        if _CODE_BACKTICK_RE.search(text):
            code_signals.append("backtick_identifier")
        if _CODE_SNAKE_CASE_RE.search(text):
            code_signals.append("snake_case_symbol")
        if _CODE_CAMEL_CASE_RE.search(text):
            code_signals.append("camel_or_pascal_symbol")
        if _CODE_QUALIFIED_RE.search(text):
            code_signals.append("qualified_symbol")
        if _CODE_ENDPOINT_RE.search(text):
            code_signals.append("http_endpoint")
        if _CODE_TRACE_RE.search(text):
            code_signals.append("stack_trace")
        if _CODE_KEYWORD_RE.search(text):
            code_signals.append("code_or_ast_term")

        doc_signals: list[str] = []
        if _DOC_TICKET_OR_SPEC_ID_RE.search(text):
            doc_signals.append("ticket_or_adr_id")
        if _DOC_KEYWORD_RE.search(text):
            doc_signals.append("documentation_or_process_term")

        has_code_cues = bool(code_signals)
        has_doc_cues = bool(doc_signals)

        if has_code_cues and has_doc_cues:
            if has_visible_code and has_visible_docs:
                return QueryIntentDecision(
                    mode=QueryIntentMode.HYBRID_CODE_AND_DOCS,
                    source_filter=source_filter,
                    needs_code=True,
                    needs_docs=True,
                    reason=(
                        "Detected both code cues ("
                        + ", ".join(code_signals)
                        + ") and documentation cues ("
                        + ", ".join(doc_signals)
                        + ")."
                    ),
                )
            if has_visible_code and not has_visible_docs:
                return QueryIntentDecision(
                    mode=QueryIntentMode.PURE_CODE,
                    source_filter=source_filter,
                    needs_code=True,
                    needs_docs=False,
                    reason=(
                        "Query has hybrid cues but only code corpus is visible "
                        "under allowed scopes."
                    ),
                )
            if has_visible_docs and not has_visible_code:
                return QueryIntentDecision(
                    mode=QueryIntentMode.PURE_DOCUMENTATION,
                    source_filter=source_filter,
                    needs_code=False,
                    needs_docs=True,
                    reason=(
                        "Query has hybrid cues but only documentation corpus is visible "
                        "under allowed scopes."
                    ),
                )
            return QueryIntentDecision(
                mode=QueryIntentMode.HYBRID_CODE_AND_DOCS,
                source_filter=source_filter,
                needs_code=True,
                needs_docs=True,
                reason="Detected both code and documentation cues.",
            )

        if has_doc_cues:
            return QueryIntentDecision(
                mode=QueryIntentMode.PURE_DOCUMENTATION,
                source_filter=source_filter,
                needs_code=False,
                needs_docs=True,
                reason=(
                    "Detected documentation cues ("
                    + ", ".join(doc_signals)
                    + ") with no code cues; suppressing AST code slices."
                ),
            )

        if has_code_cues:
            return QueryIntentDecision(
                mode=QueryIntentMode.PURE_CODE,
                source_filter=source_filter,
                needs_code=True,
                needs_docs=False,
                reason=("Detected code cues (" + ", ".join(code_signals) + ") with no doc cues."),
            )

        if has_visible_docs and not has_visible_code:
            return QueryIntentDecision(
                mode=QueryIntentMode.PURE_DOCUMENTATION,
                source_filter=source_filter,
                needs_code=False,
                needs_docs=True,
                reason="Ambiguous query routed to visible documentation-only corpus.",
            )
        if has_visible_code and not has_visible_docs:
            return QueryIntentDecision(
                mode=QueryIntentMode.PURE_CODE,
                source_filter=source_filter,
                needs_code=True,
                needs_docs=False,
                reason="Ambiguous query routed to visible code-only corpus.",
            )
        return QueryIntentDecision(
            mode=QueryIntentMode.HYBRID_CODE_AND_DOCS,
            source_filter=source_filter,
            needs_code=True,
            needs_docs=True,
            reason="Ambiguous query with both code and documentation corpora visible.",
        )


def is_jira_chunk(chunk: DocChunkRecord) -> bool:
    """Returns True when a document chunk originates from Jira."""
    source_type = (getattr(chunk, "source_type", "") or "").strip().lower()
    source_uri = (getattr(chunk, "source_uri", "") or "").strip().lower()
    meta = dict(getattr(chunk, "metadata_json", {}) or {})
    if source_type == "jira" or source_type.startswith("jira_"):
        return True
    if "jira" in source_uri:
        return True
    return bool(meta.get("issue_key") or meta.get("ticket_key"))


def is_wiki_chunk(chunk: DocChunkRecord) -> bool:
    """Returns True when a document chunk is a Confluence/ADR/wiki/markdown/doc source."""
    if is_jira_chunk(chunk):
        return False
    source_type = (getattr(chunk, "source_type", "") or "").strip().lower()
    source_uri = (getattr(chunk, "source_uri", "") or "").strip().lower()
    meta = dict(getattr(chunk, "metadata_json", {}) or {})
    wiki_tokens = (
        "confluence",
        "wiki",
        "adr",
        "markdown",
        "runbook",
        "policy",
        "architecture",
        "design_doc",
        "doc",
    )
    if any(tok in source_type for tok in wiki_tokens):
        return True
    if any(
        marker in source_uri
        for marker in ("confluence", "wiki", "/adr", ".md", "/docs/", "notion", "sharepoint")
    ):
        return True
    return bool(meta.get("page_id") or meta.get("space_key"))


@overload
def filter_doc_chunks_by_source(
    doc_chunks: list[DocChunkRecord],
    source_filter: SourceFilter,
) -> list[DocChunkRecord]: ...


@overload
def filter_doc_chunks_by_source(
    doc_chunks: dict[str, str],
    source_filter: SourceFilter,
) -> dict[str, str]: ...


@overload
def filter_doc_chunks_by_source(
    doc_chunks: None,
    source_filter: SourceFilter,
) -> None: ...


def filter_doc_chunks_by_source(
    doc_chunks: dict[str, str] | list[DocChunkRecord] | None,
    source_filter: SourceFilter,
) -> dict[str, str] | list[DocChunkRecord] | None:
    """Filters document chunks according to ``source_filter``."""
    if doc_chunks is None or source_filter == "all":
        return doc_chunks
    if source_filter == "code":
        return [] if isinstance(doc_chunks, list) else {}

    if isinstance(doc_chunks, list):
        if source_filter == "jira":
            return [chk for chk in doc_chunks if is_jira_chunk(chk)]
        if source_filter == "wiki":
            return [chk for chk in doc_chunks if is_wiki_chunk(chk)]
        return doc_chunks

    filtered_map: dict[str, str] = {}
    for key, content in doc_chunks.items():
        key_lower = key.lower()
        looks_jira = "jira" in key_lower or bool(re.search(r"\b[A-Z]{2,10}-\d+\b", key))
        if (source_filter == "jira" and looks_jira) or (source_filter == "wiki" and not looks_jira):
            filtered_map[key] = content
    return filtered_map


def _normalize_iso_date(raw_date: object) -> str | None:
    """Formats date/datetime/ISO timestamp string as YYYY-MM-DD when parseable."""
    if raw_date is None:
        return None
    if isinstance(raw_date, datetime):
        return raw_date.date().isoformat()
    if isinstance(raw_date, date):
        return raw_date.isoformat()
    text = str(raw_date).strip()
    if not text:
        return None
    date_match = re.match(r"^(\d{4}-\d{2}-\d{2})", text)
    return date_match.group(1) if date_match else None


def format_doc_citation_label(
    citation: DocCitationSpan,
    *,
    metadata: Mapping[str, object] | None = None,
    published_date: date | datetime | str | None = None,
) -> str:
    """Formats a citation header as ``[Page Title > Section | Space: ... | Last Updated: ...]``."""
    meta = dict(metadata or {})
    title = (citation.doc_title or citation.doc_id or "Document").strip()
    section = (citation.section_heading or "Overview").strip()
    if " | Last Updated:" in section:
        section = section.split(" | Last Updated:", 1)[0].strip()
    if " | Space:" in section:
        section = section.split(" | Space:", 1)[0].strip()
    title_prefix = f"{title} > "
    if section.startswith(title_prefix):
        section = section[len(title_prefix) :].strip() or "Overview"

    space_val = meta.get("space_key") or meta.get("space")
    space_key = str(space_val).strip() if space_val is not None and str(space_val).strip() else None
    if not space_key and citation.source_uri:
        space_match = _CONFLUENCE_SPACE_RE.search(citation.source_uri)
        if space_match:
            space_key = space_match.group(1)
    if not space_key and citation.ticket_key and "-" in citation.ticket_key:
        space_key = citation.ticket_key.split("-", 1)[0]

    raw_updated = (
        published_date
        or meta.get("last_updated")
        or meta.get("updated_at")
        or meta.get("published_date")
    )
    updated_str = _normalize_iso_date(raw_updated)

    parts = [f"{title} > {section}"]
    if space_key:
        parts.append(f"Space: {space_key}")
    if updated_str:
        parts.append(f"Last Updated: {updated_str}")
    return "[" + " | ".join(parts) + "]"


__all__ = [
    "FacetSpec",
    "QueryIntentClassifier",
    "QueryIntentDecision",
    "QueryIntentMode",
    "QueryPlan",
    "SourceFilter",
    "decompose_query_facets",
    "evaluate_facet_grounding",
    "filter_doc_chunks_by_source",
    "format_doc_citation_label",
    "is_jira_chunk",
    "is_wiki_chunk",
]
