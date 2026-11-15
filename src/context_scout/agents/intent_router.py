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
from typing import TYPE_CHECKING, Literal, overload

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from context_scout.db.schema import DocChunkRecord
    from context_scout.schemas.contracts import DocCitationSpan

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
    "QueryIntentClassifier",
    "QueryIntentDecision",
    "QueryIntentMode",
    "SourceFilter",
    "filter_doc_chunks_by_source",
    "format_doc_citation_label",
    "is_jira_chunk",
    "is_wiki_chunk",
]
