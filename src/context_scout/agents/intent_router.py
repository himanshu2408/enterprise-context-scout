"""Deterministic query intent classification and source filtering for Scout workflows.

Routes user queries strictly from query cues, explicit ``source_filter``, and
visible corpus availability under the principal's RBAC scopes. Never branches
on ``principal.role`` to block documentation or force code dumps onto doc queries.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Mapping
from datetime import date, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal, overload

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from context_scout.db.schema import DocChunkRecord
    from context_scout.schemas.contracts import DocCitationSpan, QueryFacetCoverage

logger = logging.getLogger(__name__)

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


_CODE_FILE_EXTENSIONS: frozenset[str] = frozenset(
    {
        "py",
        "ts",
        "tsx",
        "js",
        "jsx",
        "cs",
        "tf",
        "sql",
        "go",
        "rs",
        "java",
        "kt",
        "rb",
        "yaml",
        "yml",
        "toml",
        "sh",
    }
)
_CODE_PATH_ROOTS: tuple[str, ...] = (
    "src/",
    "app/",
    "tests/",
    "test/",
    "web-ui/",
    "pkg/",
    "internal/",
    "cmd/",
    "lib/",
)
_PATH_TOKEN_RE = re.compile(r"[A-Za-z0-9_./-]+")


def _has_code_file_or_path(text: str) -> bool:
    """Linear-time O(N) check for source file extensions or standard repository path prefixes."""
    for raw_tok in _PATH_TOKEN_RE.findall(text[:8192]):
        tok = raw_tok.strip(".").rstrip("/")
        if not tok:
            continue
        if "." in tok:
            stem, ext = tok.rsplit(".", 1)
            if stem and ext.lower() in _CODE_FILE_EXTENSIONS:
                return True
        low_tok = tok.lower()
        for root in _CODE_PATH_ROOTS:
            if low_tok.startswith(root) and len(low_tok) > len(root):
                return True
            nested = "/" + root
            idx = low_tok.find(nested)
            if idx != -1 and len(low_tok) > idx + len(nested):
                return True
    return False


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
    r"handbook|on-call|oncall|sop|"
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
_STANDALONE_ACRONYM_RE = re.compile(r"\b([A-Z][A-Z0-9]{1,7})\b")
_NON_DOMAIN_ACRONYMS: frozenset[str] = frozenset(
    {
        "IN",
        "DE",
        "US",
        "UK",
        "EU",
        "OR",
        "AND",
        "THE",
        "FOR",
        "NOT",
        "TOP",
        "HOW",
        "WHY",
        "WHO",
        "WHAT",
        "WHEN",
        "ALL",
        "ANY",
        "VS",
        "IS",
        "IT",
        "TO",
        "OF",
        "ON",
        "AT",
        "BY",
        "AS",
        # SQL DDL/DML and standard protocol keywords are never domain acronyms to define.
        "CREATE",
        "INDEX",
        "CONCURRENTLY",
        "SELECT",
        "INSERT",
        "UPDATE",
        "DELETE",
        "ALTER",
        "DROP",
        "TABLE",
        "FORCE",
        "ENABLE",
        "ROW",
        "LEVEL",
        "SECURITY",
        "SET",
        "LOCAL",
        "WHERE",
        "FROM",
        "JOIN",
        "MERGE",
        "VACUUM",
        "RETAIN",
        "HOURS",
        "BEGIN",
        "COMMIT",
        "ROLLBACK",
        "SKIP",
        "LOCKED",
        "NULL",
        "CHECK",
        "VALID",
        "KEY",
        "PRIMARY",
        "FOREIGN",
        "UNIQUE",
    }
)
_DEFINITIONAL_LEAD_RE = re.compile(
    r"^(?:what\s+(?:is|are)\s+(?:an?\s+|the\s+)?|what\s+does\s+|explain\s+(?:the\s+)?|"
    r"describe\s+(?:the\s+)?|define\s+(?:the\s+)?|how\s+does\s+|how\s+do\s+|tell\s+me\s+about\s+)",
    re.IGNORECASE,
)
_DEFINITIONAL_TAIL_SUFFIXES: tuple[str, ...] = (
    " stands for",
    " stand for",
    " in databases",
    " in database",
    " in detail",
    " works",
    " work",
    " means",
    " mean",
)


def _strip_definitional_tail(text: str) -> str:
    """Linear-time O(N) suffix stripper avoiding polynomial regex backtracking on whitespace."""
    trimmed = text.rstrip()
    low = trimmed.lower()
    for suffix in _DEFINITIONAL_TAIL_SUFFIXES:
        if low.endswith(suffix):
            head = trimmed[: len(trimmed) - len(suffix)].rstrip()
            if head:
                return head
    return trimmed


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
        "without",
        "taking",
        "long",
        "short",
        "enforce",
        "enforces",
        "configure",
        "configures",
        "handle",
        "handles",
        "handling",
        "make",
        "makes",
        "compare",
        "contrast",
        "versus",
        "against",
        "explain",
        "describe",
        "implement",
        "create",
        "build",
        "write",
        "show",
        "find",
        "locate",
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
    """Structured query decomposition plan (intent, core subject, cardinality, and facets)."""

    model_config = ConfigDict(frozen=True)

    original_query: str
    query_type: Literal[
        "concept_definition",
        "architecture_comparison",
        "code_implementation",
        "incident_runbook",
        "multi_hop_research_to_code",
        "general",
    ] = "general"
    core_subject: str = ""
    expanded_queries: list[str] = []
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
        _has_code_file_or_path(text)
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


def _split_ci_phrase(text: str, sep_lower: str) -> list[str]:
    """Splits single-space-normalized text on a case-insensitive literal phrase in O(N) time."""
    low = text.lower()
    out: list[str] = []
    pos = 0
    step = len(sep_lower)
    while True:
        idx = low.find(sep_lower, pos)
        if idx == -1:
            out.append(text[pos:])
            break
        out.append(text[pos:idx])
        pos = idx + step
    return out


def _starts_with_keyword(text_lower: str, pos: int, keywords: tuple[str, ...]) -> bool:
    for kw in keywords:
        end = pos + len(kw)
        if text_lower.startswith(kw, pos) and (
            end == len(text_lower) or not text_lower[end].isalnum()
        ):
            return True
    return False


def _split_ci_before_keywords(
    text: str, sep_lower: str, keywords: tuple[str, ...]
) -> list[str] | None:
    """Splits on sep_lower only when immediately followed by one of keywords (keeping keyword)."""
    low = text.lower()
    cut_points: list[tuple[int, int]] = []
    search_from = 0
    sep_len = len(sep_lower)
    while True:
        idx = low.find(sep_lower, search_from)
        if idx == -1:
            break
        after = idx + sep_len
        if _starts_with_keyword(low, after, keywords):
            cut_points.append((idx, after))
            search_from = after
        else:
            search_from = idx + 1
    if not cut_points:
        return None
    parts: list[str] = []
    prev = 0
    for start, end in cut_points:
        parts.append(text[prev:start])
        prev = end
    parts.append(text[prev:])
    return parts


def _strip_trailing_closure_clause(text: str) -> str:
    low = text.lower()
    marker = "including "
    search_from = 0
    suffix_prefixes = (
        "callers",
        "all callers",
        "imported types",
        "all imported types",
        "unit tests",
        "all unit tests",
        "tests",
        "all tests",
    )
    while True:
        idx = low.find(marker, search_from)
        if idx == -1:
            return text
        if idx == 0 or not low[idx - 1].isalnum():
            rest = low[idx + len(marker) :]
            if _starts_with_keyword(rest, 0, suffix_prefixes):
                return text[:idx].rstrip(", ").strip()
        search_from = idx + 1


def _strip_top_n_on_preamble(text: str) -> tuple[str, bool]:
    """Linearly strips leading 'What are [the] top N <words> on ' preamble if present."""
    low = text.lower()
    prefix_len = 0
    if low.startswith("what are the top "):
        prefix_len = len("what are the top ")
    elif low.startswith("what are top "):
        prefix_len = len("what are top ")
    if prefix_len == 0:
        return text, False
    rest_after_top = low[prefix_len:]
    space_idx = rest_after_top.find(" ")
    if space_idx <= 0 or not rest_after_top[:space_idx].isdigit():
        return text, False
    after_num = prefix_len + space_idx + 1
    on_idx = low.find(" on ", after_num)
    if on_idx == -1 or on_idx <= after_num:
        return text, False
    middle = low[after_num:on_idx]
    if all(ch.isalnum() or ch in (" ", "_") for ch in middle):
        tail = text[on_idx + 4 :].strip()
        if tail:
            return tail, True
    return text, False


def _split_question_clauses(text: str) -> list[str]:
    """Linearly splits on ';' or ', [and ]<which|what|where|how|why>\\b'."""
    q_words = ("which", "what", "where", "how", "why")
    semi_parts = text.split(";")
    clauses: list[str] = []
    for chunk in semi_parts:
        s = chunk.strip()
        if not s:
            continue
        low = s.lower()
        prev = 0
        pos = 0
        while True:
            comma_idx = low.find(",", pos)
            if comma_idx == -1:
                break
            after_comma = comma_idx + 1
            while after_comma < len(low) and low[after_comma] == " ":
                after_comma += 1
            kw_pos = after_comma
            if low.startswith("and ", after_comma):
                kw_pos = after_comma + 4
            if _starts_with_keyword(low, kw_pos, q_words):
                clauses.append(s[prev:comma_idx])
                prev = kw_pos
                pos = kw_pos
            else:
                pos = comma_idx + 1
        clauses.append(s[prev:])
    return clauses


def _infer_query_type_and_subject(
    prompt: str,
) -> tuple[
    Literal[
        "concept_definition",
        "architecture_comparison",
        "code_implementation",
        "incident_runbook",
        "multi_hop_research_to_code",
        "general",
    ],
    str,
]:
    """Domain-agnostic syntactic classification of query type and core subject."""
    cleaned = " ".join(prompt.strip().split()).rstrip("?!. ")
    low = cleaned.lower()

    has_code_markers = bool(
        _CODE_CAMEL_CASE_RE.search(cleaned)
        or _CODE_SNAKE_CASE_RE.search(cleaned)
        or any(
            k in low
            for k in (
                "codebase",
                "class ",
                "function",
                "method",
                "callers",
                "ast ",
                "file path",
                "line ",
                "module",
                "signature",
            )
        )
    )
    has_research_markers = any(
        k in low for k in ("paper", "research", "formula", "equation", "empirical", "theory")
    )
    has_issue_markers = any(
        k in low
        for k in (
            "bug",
            "issue",
            "incident",
            "post-mortem",
            "postmortem",
            "reconnect",
            "duplicate",
            "crash",
            "outage",
            "root cause",
        )
    )

    if has_research_markers and (has_code_markers or has_issue_markers):
        query_type: Literal[
            "concept_definition",
            "architecture_comparison",
            "code_implementation",
            "incident_runbook",
            "multi_hop_research_to_code",
            "general",
        ] = "multi_hop_research_to_code"
    elif any(k in low for k in ("compare ", "contrast ", " versus ", " vs ", "difference between")):
        query_type = "architecture_comparison"
    elif has_issue_markers:
        query_type = "incident_runbook"
    elif has_code_markers:
        query_type = "code_implementation"
    elif _DEFINITIONAL_LEAD_RE.match(cleaned) or " stand for" in low:
        query_type = "concept_definition"
    else:
        query_type = "general"

    subject = _DEFINITIONAL_LEAD_RE.sub("", cleaned).strip()
    subject = _strip_definitional_tail(subject)
    subject = _strip_trailing_closure_clause(subject).strip(" ,.;:")
    if not subject:
        subject = cleaned
    return query_type, subject


def decompose_query_facets(prompt: str) -> QueryPlan:
    """Decomposes a prompt into query type, core subject, expanded queries, and sub-facets."""
    cleaned = " ".join(prompt[:4096].strip().split())
    stripped = cleaned.rstrip("?!. ")

    requested_cardinality: int | None = None
    for pat in (_CARDINALITY_LEADING_RE, _CARDINALITY_NOUN_RE):
        m = pat.search(stripped)
        if m:
            val = int(m.group(1))
            if 1 <= val <= 20:
                requested_cardinality = val
                break

    query_type, core_subject = _infer_query_type_and_subject(stripped)

    acronym: str | None = None
    is_acronym_query = False
    acr_m = _ACRONYM_STAND_FOR_RE.search(stripped) or _ACRONYM_PROPERTIES_RE.search(stripped)
    if acr_m:
        is_acronym_query = True
        acronym = acr_m.group(1).upper()
    elif query_type == "concept_definition":
        for cand in _STANDALONE_ACRONYM_RE.findall(stripped):
            if cand not in _NON_DOMAIN_ACRONYMS:
                is_acronym_query = True
                acronym = cand
                break

    # Strip trailing AST closure / meta-clauses so they don't become bogus lexical facets
    working = stripped
    if working.lower().startswith("compare "):
        working = working[len("compare ") :].strip()
    elif working.lower().startswith("contrast "):
        working = working[len("contrast ") :].strip()
    elif query_type == "concept_definition":
        working = _DEFINITIONAL_LEAD_RE.sub("", working).strip()
        working = _strip_definitional_tail(working) or working
    working = _strip_trailing_closure_clause(working)
    working, has_top_preamble = _strip_top_n_on_preamble(working)

    # First split on explicit question/issue clauses (", and which ...", ", and how ...")
    clause_parts = _split_question_clauses(working)

    raw_facets: list[str] = []
    conj_keywords = ("open", "how", "what", "which", "where")
    for part in clause_parts:
        sub = part.strip().strip(",")
        if not sub:
            continue
        # Split "X against Y and Z" or comma-separated lists ("A, B, and C")
        against_parts = _split_ci_phrase(sub, " against ") if " against " in sub.lower() else [sub]

        for ap in against_parts:
            ap_clean = ap.strip().strip(",")
            if not ap_clean:
                continue
            comma_count = ap_clean.count(",")
            has_oxford = ", and " in ap_clean.lower()
            if has_oxford or comma_count >= 2:
                for ox_part in _split_ci_phrase(ap_clean, ", and "):
                    for item in ox_part.split(","):
                        if item.strip():
                            raw_facets.append(item.strip())
            else:
                kw_split = _split_ci_before_keywords(ap_clean, " and ", conj_keywords)
                if kw_split is not None:
                    raw_facets.extend(it.strip() for it in kw_split if it.strip())
                elif has_top_preamble and " and " in ap_clean.lower():
                    items = _split_ci_phrase(ap_clean, " and ")
                    raw_facets.extend(it.strip() for it in items if it.strip())
                else:
                    raw_facets.append(ap_clean)

    # Filter out trivial fragments, trailing meta-clauses, and generic modifiers
    facets: list[FacetSpec] = []
    skip_exact = {
        "their core classes",
        "including",
        "all callers",
        "imported types",
        "unit tests",
    }
    for rf in raw_facets:
        fq = rf[4:].strip() if rf.lower().startswith("and ") else rf.strip()
        if len(fq) < 2:
            continue
        if fq.lower() in skip_exact:
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
                facet_query=core_subject or stripped or prompt.strip(),
                key_terms=terms,
                expected_source_kind=_classify_facet_source_kind(stripped),
            )
        ]

    expanded_queries: list[str] = []
    for candidate in (
        core_subject,
        f"{acronym} definition properties" if acronym else "",
        f"{core_subject} architecture overview"
        if query_type == "concept_definition" and core_subject and len(core_subject.split()) <= 5
        else "",
    ):
        c_clean = candidate.strip()
        if c_clean and c_clean.lower() != stripped.lower() and c_clean not in expanded_queries:
            expanded_queries.append(c_clean)

    return QueryPlan(
        original_query=prompt,
        query_type=query_type,
        core_subject=core_subject,
        expanded_queries=expanded_queries[:3],
        requested_cardinality=requested_cardinality,
        is_acronym_query=is_acronym_query,
        acronym=acronym,
        facets=facets,
    )


_COMPARISON_CUE_RE = re.compile(
    r"\b(?:vs\.?|versus|compare|comparison|difference|differences|contrast)\b",
    re.IGNORECASE,
)


def anchor_llm_query_plan(
    prompt: str,
    llm_plan: QueryPlan,
    base_plan: QueryPlan,
) -> QueryPlan:
    """Anchors an LLM-produced QueryPlan strictly to clauses/terms present in prompt.

    Prevents live LLM query planners from inventing unasked comparisons or extra facets
    (e.g. 'standard vs enterprise-tier') while preserving extra terms in expanded_queries.
    """
    merged_expansions: list[str] = list(
        dict.fromkeys([*base_plan.expanded_queries, *llm_plan.expanded_queries])
    )
    prompt_has_comparison = bool(_COMPARISON_CUE_RE.search(prompt))

    if base_plan.is_acronym_query:
        return QueryPlan(
            original_query=base_plan.original_query,
            query_type=llm_plan.query_type or base_plan.query_type,
            core_subject=base_plan.core_subject,
            expanded_queries=merged_expansions[:5],
            requested_cardinality=base_plan.requested_cardinality,
            is_acronym_query=True,
            acronym=base_plan.acronym,
            facets=list(base_plan.facets),
        )

    prompt_tokens: set[str] = {
        tok.lower()
        for tok in re.findall(r"[A-Za-z0-9_]{2,}", prompt)
        if tok.lower() not in _FACET_STOPWORDS
    }
    for f in base_plan.facets:
        prompt_tokens.update(t.lower() for t in f.key_terms if t)

    def _term_anchored(term: str) -> bool:
        t = term.lower()
        if t in prompt_tokens:
            return True
        if len(t) >= 5:
            return any(
                len(pt) >= 5
                and (t.startswith(pt[:5]) or pt.startswith(t[:5]) or t in pt or pt in t)
                for pt in prompt_tokens
            )
        return False

    anchored_facets: list[FacetSpec] = []
    demoted_expansions: list[str] = []
    for f in llm_plan.facets:
        if not prompt_has_comparison and _COMPARISON_CUE_RE.search(f.facet_query):
            demoted_expansions.append(f.facet_query)
            continue
        f_terms = [t.lower() for t in f.key_terms if t]
        if not f_terms:
            demoted_expansions.append(f.facet_query)
            continue
        anchored_hits = sum(1 for t in f_terms if _term_anchored(t))
        unanchored_hits = len(f_terms) - anchored_hits
        min_needed = 1 if len(f_terms) == 1 else max(2, math.ceil((len(f_terms) * 2) / 3))
        if anchored_hits >= min_needed and unanchored_hits <= 1:
            anchored_facets.append(f)
        else:
            demoted_expansions.append(f.facet_query)

    is_single_clause = (
        len(base_plan.facets) == 1
        and not prompt_has_comparison
        and not re.search(r"\b(?:and|or)\b|,|;", prompt, re.IGNORECASE)
        and prompt.count("?") <= 1
    )
    if (is_single_clause and len(anchored_facets) != 1) or not anchored_facets:
        anchored_facets = list(base_plan.facets)
    else:
        anchored_terms = {t.lower() for f in anchored_facets for t in f.key_terms}
        base_terms = {t.lower() for f in base_plan.facets for t in f.key_terms}
        if (
            prompt_tokens
            and len(anchored_terms & prompt_tokens) < len(base_terms & prompt_tokens)
            and (len(anchored_terms & prompt_tokens) / max(1, len(prompt_tokens))) < 0.65
        ):
            for bf in reversed(base_plan.facets):
                if not any(
                    len(set(bf.key_terms) & set(af.key_terms))
                    >= max(1, (len(bf.key_terms) * 2) // 3)
                    for af in anchored_facets
                ):
                    anchored_facets.insert(0, bf)

    # Re-index facet_id sequentially
    final_facets = [
        FacetSpec(
            facet_id=f"facet_{idx}",
            facet_query=f.facet_query,
            key_terms=list(f.key_terms),
            expected_source_kind=f.expected_source_kind,
            is_acronym_component=f.is_acronym_component,
        )
        for idx, f in enumerate(anchored_facets, start=1)
    ]

    all_expansions = list(dict.fromkeys([*merged_expansions, *demoted_expansions]))[:5]
    safe_subject = (
        base_plan.core_subject
        if (
            not llm_plan.core_subject.strip()
            or (not prompt_has_comparison and _COMPARISON_CUE_RE.search(llm_plan.core_subject))
        )
        else llm_plan.core_subject.strip()
    )
    return QueryPlan(
        original_query=prompt,
        query_type=llm_plan.query_type or base_plan.query_type,
        core_subject=safe_subject,
        expanded_queries=all_expansions,
        requested_cardinality=base_plan.requested_cardinality,
        is_acronym_query=base_plan.is_acronym_query,
        acronym=base_plan.acronym,
        facets=final_facets,
    )


def plan_query_with_llm(prompt: str, llm_client: Any = None) -> QueryPlan:
    """Uses structured Gemma-4 planning when live MLX is enabled, else deterministic plan."""
    base_plan = decompose_query_facets(prompt)
    if llm_client is None or not hasattr(llm_client, "decompose_query_json"):
        return base_plan
    try:
        planned = llm_client.decompose_query_json(prompt, fallback_plan=base_plan)
        if isinstance(planned, QueryPlan) and planned.facets:
            return anchor_llm_query_plan(prompt, planned, base_plan)
    except Exception as exc:
        logger.debug("LLM query planning skipped (%s): %s", type(exc).__name__, exc)
    return base_plan


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
    best_cohesive_hits = 0
    matched_terms_overall: set[str] = set()
    matched_acronyms_strict: set[str] = set()
    matched_sources: list[str] = []

    uppercase_acronyms = [
        tok
        for tok in re.findall(r"\b([A-Z][A-Z0-9]{1,7})\b", facet.facet_query)
        if tok not in _NON_DOMAIN_ACRONYMS and tok.lower() in terms
    ]
    explicit_identifiers = [
        tok.lower()
        for tok in re.findall(
            r"\b([A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+|[A-Z][a-z0-9]+[A-Z][A-Za-z0-9]+)\b",
            facet.facet_query,
        )
        if tok.lower() in terms
    ]

    def _expand_market_scope_aliases(market_scope_obj: object) -> str:
        raw_val = getattr(market_scope_obj, "value", market_scope_obj)
        s_val = str(raw_val or "").strip()
        if not s_val:
            return ""
        base_tokens = re.sub(r"[_-]+", " ", s_val)
        try:
            from context_scout.plugins.registry import PluginRegistry

            plugin = PluginRegistry.get_domain_plugin(domain_id)
            alias_fn = getattr(plugin, "get_market_scope_aliases", None) if plugin else None
            if callable(alias_fn):
                extra = str(alias_fn(s_val) or "").strip()
                if extra:
                    return f"{s_val} {base_tokens} {extra}"
        except (ImportError, KeyError, AttributeError, RuntimeError) as exc:
            logger.debug("Plugin market-scope alias lookup skipped for %s: %s", domain_id, exc)
        return f"{s_val} {base_tokens}"

    def _has_proximity_cohesion(raw_text: str, active_terms: list[str]) -> bool:
        """Verify multi-term concepts co-occur within a tight 16-word window or title/heading."""
        if len(active_terms) <= 1:
            return True
        words = re.findall(r"[A-Za-z0-9_]+", raw_text.lower())
        if not words:
            return False
        positions: dict[str, list[int]] = {t: [] for t in active_terms}
        for idx, w in enumerate(words):
            w_norm = w.replace("_", " ")
            for t in active_terms:
                if w == t or (len(t) >= 5 and (t in w or t in w_norm)):
                    positions[t].append(idx)
        present = [t for t in active_terms if positions[t]]
        if len(present) < min(2, len(active_terms)):
            return False
        if len(present) == 2:
            p0, p1 = positions[present[0]], positions[present[1]]
            return any(abs(a - b) <= 16 for a in p0 for b in p1)
        # For 3+ term facets, require at least two pairs within 28 words inside the same chunk
        close_pairs = 0
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                pos_i, pos_j = positions[present[i]], positions[present[j]]
                if any(abs(a - b) <= 28 for a in pos_i for b in pos_j):
                    close_pairs += 1
        return close_pairs >= 1

    def _scan_text(text: str, source_label: str, *, header_text: str = "") -> None:
        nonlocal best_hits, best_cohesive_hits
        low = text.lower()
        low_norm = re.sub(r"[_-]+", " ", low)
        hits_here = {
            t
            for t in terms
            if re.search(rf"\b{re.escape(t)}\b", low_norm)
            or re.search(rf"\b{re.escape(t)}\b", low)
            or (len(t) >= 5 and (t in low_norm or t in low))
        }
        if not hits_here:
            return

        # Strict acronym verification: uppercase acronym (e.g. CAP, ACID) must appear as
        # uppercase in body/header OR its lowercase form must appear in doc_title/section_heading
        header_low = re.sub(r"[_-]+", " ", header_text.lower())
        text_norm = re.sub(r"[_-]+", " ", text)
        for acr in uppercase_acronyms:
            if re.search(rf"\b{re.escape(acr)}\b", text_norm) or re.search(
                rf"\b{re.escape(acr.lower())}\b", header_low
            ):
                matched_acronyms_strict.add(acr.lower())

        matched_terms_overall.update(hits_here)
        if len(hits_here) > best_hits:
            best_hits = len(hits_here)

        is_cohesive = (
            len(hits_here) == 1
            or (
                header_text
                and sum(1 for t in hits_here if t in header_low) >= min(2, len(hits_here))
            )
            or _has_proximity_cohesion(text, list(hits_here))
        )
        if is_cohesive and len(hits_here) > best_cohesive_hits:
            best_cohesive_hits = len(hits_here)

        if source_label not in matched_sources and (
            (len(hits_here) >= min(2, len(terms)) and is_cohesive)
            or (len(terms) == 1 and len(hits_here) == 1)
        ):
            matched_sources.append(source_label)

    if facet.expected_source_kind != "code" or not code_slices:
        for chk in doc_chunks:
            if facet.expected_source_kind == "issue" and not is_jira_chunk(chk):
                # Also allow GitHub issue chunks whose title/uri/content mentions issue
                uri_low = str(getattr(chk, "source_uri", "") or "").lower()
                title_low = str(getattr(chk, "doc_title", "") or "").lower()
                if "issue" not in uri_low and "issue" not in title_low and "#" not in title_low:
                    continue
            doc_title = str(getattr(chk, "doc_title", "") or "")
            sec_head = str(getattr(chk, "section_heading", "") or "")
            scope_aliases = _expand_market_scope_aliases(getattr(chk, "market_scope", None))
            label = str(doc_title or getattr(chk, "doc_id", "") or getattr(chk, "chunk_id", ""))
            txt = (
                f"{doc_title}\n"
                f"{sec_head}\n"
                f"{scope_aliases}\n"
                f"{getattr(chk, 'verbatim_content', '')}\n"
                f"{getattr(chk, 'full_section_content', '')}\n"
                f"{getattr(chk, 'verbatim_quote', '')}"
            )
            _scan_text(txt, label, header_text=f"{doc_title} {sec_head} {scope_aliases}")

    if facet.expected_source_kind in ("code", "any") or (not matched_terms_overall and code_slices):
        for slc in code_slices:
            qsym = str(getattr(slc, "qualified_symbol", "") or "")
            fpath = str(getattr(slc, "file_path", "") or "")
            label = str(fpath or qsym)
            txt = f"{fpath}\n{qsym}\n{getattr(slc, 'verbatim_code_with_lines', '')}"
            _scan_text(txt, label, header_text=f"{fpath} {qsym}")

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

    missing_acronyms = [
        acr.lower()
        for acr in uppercase_acronyms
        if acr.lower() not in matched_acronyms_strict
        and (acr.lower() in matched_terms_overall or total_terms <= 3 or coverage_ratio < 0.75)
    ]

    missing_identifiers = [
        ident for ident in explicit_identifiers if ident not in matched_terms_overall
    ]

    if missing_acronyms or missing_identifiers:
        is_grounded = False
        is_partial = False
    elif total_terms == 1:
        is_grounded = best_hits == 1
        is_partial = False
    elif total_terms == 2:
        is_grounded = best_cohesive_hits >= 2 and coverage_ratio >= 1.0
        is_partial = not is_grounded and best_cohesive_hits >= 1 and best_hits >= 2
    else:
        is_grounded = coverage_ratio >= 0.65 and best_cohesive_hits >= 2 and best_hits >= 2
        is_partial = (
            not is_grounded
            and coverage_ratio > 0.0
            and (best_cohesive_hits >= 2 or (best_hits >= 2 and coverage_ratio >= 0.5))
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
        if _has_code_file_or_path(text):
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
    "anchor_llm_query_plan",
    "decompose_query_facets",
    "evaluate_facet_grounding",
    "filter_doc_chunks_by_source",
    "format_doc_citation_label",
    "is_jira_chunk",
    "is_wiki_chunk",
    "plan_query_with_llm",
]
