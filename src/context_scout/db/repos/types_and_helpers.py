"""RLS-enforced repositories and 3-way Normalized RRF search (M1 Foundation)."""

import logging
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Final
from uuid import UUID

from context_scout.db.schema import (
    ASTEdgeRecord as ASTEdgeRecord,
)
from context_scout.db.schema import (
    ASTNeighborhoodRecord as ASTNeighborhoodRecord,
)
from context_scout.db.schema import (
    ASTSymbolRecord as ASTSymbolRecord,
)
from context_scout.db.schema import (
    CodeFileRecord as CodeFileRecord,
)
from context_scout.db.schema import (
    ConfluencePageRecord as ConfluencePageRecord,
)
from context_scout.db.schema import (
    ConnectorWatermarkRecord as ConnectorWatermarkRecord,
)
from context_scout.db.schema import (
    DocChunkRecord as DocChunkRecord,
)
from context_scout.db.schema import (
    DomainSyncManifestRecord as DomainSyncManifestRecord,
)
from context_scout.db.schema import (
    EffectCategory as EffectCategory,
)
from context_scout.db.schema import (
    EmbeddingSpaceRecord as EmbeddingSpaceRecord,
)
from context_scout.db.schema import (
    JiraIssueRecord as JiraIssueRecord,
)
from context_scout.db.schema import (
    ModelArtifactRecord as ModelArtifactRecord,
)
from context_scout.db.schema import (
    OverlayFileDeltaRecord as OverlayFileDeltaRecord,
)
from context_scout.db.schema import (
    RankedSearchHit as RankedSearchHit,
)
from context_scout.db.schema import (
    RefSnapshotRecord as RefSnapshotRecord,
)
from context_scout.db.schema import (
    RepoCommitRecord as RepoCommitRecord,
)
from context_scout.db.schema import (
    RepositoryRecord as RepositoryRecord,
)
from context_scout.db.schema import (
    ResolvedOverlayFileRecord as ResolvedOverlayFileRecord,
)
from context_scout.db.schema import (
    ScoutSessionRecord as ScoutSessionRecord,
)
from context_scout.db.schema import (
    SideEffectRecord as SideEffectRecord,
)
from context_scout.db.schema import (
    SideEffectSeverity as SideEffectSeverity,
)
from context_scout.retrieval.embedder import (
    normalize_embedding,
    validate_non_zero_embedding,
)
from context_scout.schemas.contracts import (
    DocumentScope,
)

logger = logging.getLogger(__name__)


def _as_dt(val: object) -> datetime:
    if isinstance(val, datetime):
        return val
    return datetime.fromisoformat(str(val))


def _as_opt_dt(val: object) -> datetime | None:
    if val is None:
        return None
    return _as_dt(val)


def _serialize_market_scope(market_scope: DocumentScope | str | None) -> str | None:
    """Serializes DocumentScope | str | None to SQL NULL or text without 'None' strings."""
    if market_scope is None:
        return None
    if isinstance(market_scope, DocumentScope):
        return market_scope.value
    cleaned = str(market_scope).strip()
    if not cleaned or cleaned == "None":
        return None
    return cleaned


def _parse_market_scope(raw_scope: object) -> DocumentScope | str | None:
    """Parses DB market_scope into DocumentScope, preserved str label, or None."""
    if raw_scope is None:
        return None
    val = str(raw_scope).strip()
    if not val or val == "None":
        return None
    try:
        return DocumentScope(val)
    except ValueError:
        return val


def _apply_authority_score(base_rrf: float, authority_weight: float) -> float:
    """Factors authority_weight into RRF score while preserving exact 6-decimal equality at 1.0."""
    if math.isclose(authority_weight, 1.0, rel_tol=0.0, abs_tol=1e-9):
        return base_rrf
    return round(min(1.0, max(0.0, base_rrf * authority_weight)), 6)


@dataclass(frozen=True)
class OwnershipEdgeRecord:
    """Ownership or system membership graph edge view."""

    edge_id: UUID
    edge_kind: str
    source_qualified_name: str
    source_symbol_kind: str
    target_qualified_name: str
    target_symbol_kind: str


@dataclass(frozen=True)
class CalibratedRankedSearchHit(RankedSearchHit):
    """RankedSearchHit enriched with real cosine, BM25, trigram, and calibrated scores."""

    cosine_similarity: float = 0.0
    bm25_score: float = 0.0
    trigram_score: float = 0.0
    raw_rrf_score: float = 0.0
    calibrated_score: float = 0.0
    domain_id: str = ""
    lexical_coverage: float = 0.0
    is_boilerplate: bool = False


_CALIBRATION_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "between",
        "by",
        "compare",
        "define",
        "describe",
        "do",
        "does",
        "empirical",
        "explain",
        "failure",
        "findings",
        "for",
        "from",
        "give",
        "how",
        "in",
        "inspect",
        "into",
        "is",
        "it",
        "its",
        "list",
        "mean",
        "means",
        "mitigate",
        "modes",
        "of",
        "on",
        "or",
        "reasons",
        "stand",
        "stands",
        "that",
        "the",
        "this",
        "to",
        "top",
        "use",
        "used",
        "using",
        "what",
        "when",
        "where",
        "which",
        "why",
        "with",
        "work",
        "works",
    }
)


def _extract_substantive_tokens(query_text: str) -> list[str]:
    raw = [t.lower() for t in re.findall(r"[A-Za-z0-9_]+", query_text)]
    deduped = list(dict.fromkeys(t for t in raw if len(t) >= 2 and t not in _CALIBRATION_STOPWORDS))
    return deduped[:16]


def _is_license_boilerplate(title: str, section: str, content: str, query_text: str) -> bool:
    q_low = query_text.lower()
    if any(k in q_low for k in ("license", "apache", "mit", "copyright", "gpl", "bsd")):
        return False
    combined = f"{title} {section} {content[:900]}".lower()
    if "apache license" in combined and "version 2.0" in combined:
        return True
    if "permission is hereby granted, free of charge" in combined and "mit license" in combined:
        return True
    return bool("gnu general public license" in combined and "terms and conditions" in combined)


@dataclass(frozen=True)
class NormalizedPurgeSelectors:
    """Validated, mutually consistent purge selectors."""

    domain_id: str
    doc_id: str | None = None
    confluence_page_id: str | None = None
    jira_issue_key: str | None = None
    repo_name: str | None = None
    file_path: str | None = None
    commit_sha: str | None = None
    blob_sha: str | None = None


def normalize_purge_selectors(
    domain_id: str,
    *,
    doc_id: str | None = None,
    confluence_page_id: str | None = None,
    jira_issue_key: str | None = None,
    repo_name: str | None = None,
    file_path: str | None = None,
    commit_sha: str | None = None,
    blob_sha: str | None = None,
    has_object_keys: bool = False,
) -> NormalizedPurgeSelectors:
    """Validate and normalize purge selectors fail-closed before touching any store."""
    clean_domain = domain_id.strip()
    if not clean_domain:
        raise ValueError("domain_id is required for artifact purge")

    # Validate file_path first to preserve fail-closed repo_name error semantics.
    if file_path is not None:
        clean_file: str | None = file_path.strip()
        if not clean_file:
            raise ValueError("file_path must not be empty or whitespace when provided")
        if repo_name is None or not repo_name.strip():
            raise ValueError("file_path purge requires repo_name")
    else:
        clean_file = None

    cleaned_opt: dict[str, str | None] = {}
    for field_name, raw_val in (
        ("repo_name", repo_name),
        ("commit_sha", commit_sha),
        ("blob_sha", blob_sha),
        ("doc_id", doc_id),
        ("confluence_page_id", confluence_page_id),
        ("jira_issue_key", jira_issue_key),
    ):
        if raw_val is not None:
            stripped = raw_val.strip()
            if not stripped:
                raise ValueError(f"{field_name} must not be empty or whitespace when provided")
            cleaned_opt[field_name] = stripped
        else:
            cleaned_opt[field_name] = None

    clean_repo = cleaned_opt["repo_name"]
    clean_commit = cleaned_opt["commit_sha"]
    clean_blob = cleaned_opt["blob_sha"]
    clean_doc = cleaned_opt["doc_id"]
    clean_cf = cleaned_opt["confluence_page_id"]
    clean_jira = cleaned_opt["jira_issue_key"]

    doc_selectors = [
        name
        for name, val in (
            ("doc_id", clean_doc),
            ("confluence_page_id", clean_cf),
            ("jira_issue_key", clean_jira),
        )
        if val is not None
    ]
    if doc_selectors and any((clean_repo, clean_file, clean_commit, clean_blob)):
        raise ValueError(
            "Ambiguous purge selectors: document selectors cannot be combined "
            "with repository or code selectors"
        )
    if len(doc_selectors) > 1:
        raise ValueError(
            "Ambiguous purge selectors: supply only one document selector "
            f"(got {', '.join(doc_selectors)})"
        )
    if clean_file is not None and (clean_commit or clean_blob):
        raise ValueError(
            "Ambiguous purge selectors: file_path cannot be combined with commit_sha or blob_sha"
        )
    if clean_file is None and sum(bool(x) for x in (clean_repo, clean_commit, clean_blob)) > 1:
        raise ValueError(
            "Ambiguous purge selectors: repo_name, commit_sha, and blob_sha are mutually exclusive"
        )
    if not has_object_keys and not any(
        (clean_repo, clean_file, clean_commit, clean_blob, clean_doc, clean_cf, clean_jira)
    ):
        raise ValueError("Fail-closed: at least one artifact selector must be supplied for purge")

    return NormalizedPurgeSelectors(
        domain_id=clean_domain,
        doc_id=clean_doc,
        confluence_page_id=clean_cf,
        jira_issue_key=clean_jira,
        repo_name=clean_repo,
        file_path=clean_file,
        commit_sha=clean_commit,
        blob_sha=clean_blob,
    )


def _market_scope_aliases(
    market_scope: DocumentScope | str | None,
    domain_id: str | None = None,
) -> str:
    """Expand normalized market_scope codes into searchable tokens via active domain plugin."""
    scope_val = _serialize_market_scope(market_scope) or ""
    if not scope_val:
        return ""
    base_tokens = re.sub(r"[_-]+", " ", scope_val)
    try:
        from context_scout.plugins.registry import PluginRegistry

        plugin = PluginRegistry.get_domain_plugin(domain_id) if domain_id else None
        alias_fn = getattr(plugin, "get_market_scope_aliases", None) if plugin else None
        if callable(alias_fn):
            extra = str(alias_fn(scope_val) or "").strip()
            if extra:
                return f"{scope_val} {base_tokens} {extra}"
    except (ImportError, KeyError, AttributeError, RuntimeError) as exc:
        logger.debug("Plugin scope alias lookup skipped for %s: %s", domain_id, exc)
    return f"{scope_val} {base_tokens}"


def _infer_market_scopes_from_query(
    query_text: str,
    domain_id: str | None = None,
) -> tuple[list[str], str, set[str]]:
    """Delegate explicit query scope inference and alias lookup to the domain plugin."""
    if not domain_id:
        return [], "", set()
    try:
        from context_scout.plugins.registry import PluginRegistry

        plugin = PluginRegistry.get_domain_plugin(domain_id)
        if plugin is None:
            return [], "", set()
        infer_fn = getattr(plugin, "infer_explicit_market_scopes", None)
        alias_fn = getattr(plugin, "get_market_scope_aliases", None)
        map_fn = getattr(plugin, "get_market_scope_alias_map", None)
        scopes = [str(s) for s in infer_fn(query_text)] if callable(infer_fn) else []
        alias_parts = [str(alias_fn(s) or "") for s in scopes] if callable(alias_fn) else []
        alias_str = " ".join(p for p in alias_parts if p).strip()
        all_scope_tokens: set[str] = set()
        if callable(map_fn):
            raw_map = map_fn()
            if isinstance(raw_map, dict):
                for v in raw_map.values():
                    all_scope_tokens.update(t.lower() for t in re.findall(r"[A-Za-z0-9_]+", str(v)))
        return scopes, alias_str, all_scope_tokens
    except (ImportError, KeyError, AttributeError, RuntimeError) as exc:
        logger.debug("Plugin scope inference skipped for %s: %s", domain_id, exc)
        return [], "", set()


def _decompose_identifier_words(text: str) -> str:
    """Split CamelCase and snake_case identifiers into space-separated lowercase words."""
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    spaced = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", spaced)
    spaced = re.sub(r"([A-Za-z])(\d+)", r"\1 \2", spaced)
    tokens = [t.lower() for t in re.split(r"[^A-Za-z0-9]+", spaced) if t]
    return " ".join(tokens)


def _token_matches_text(tok: str, text: str) -> bool:
    """Match whole token or 5-char stem prefix for inflectional variants (domain-agnostic)."""
    if re.search(rf"\b{re.escape(tok)}\b", text):
        return True
    if len(tok) >= 5 and tok in text:
        return True
    return bool(len(tok) >= 6 and re.search(rf"\b{re.escape(tok[:5])}[a-z]*\b", text))


def _compute_calibrated_relevance(
    *,
    query_text: str,
    title: str,
    section: str,
    content: str,
    cosine_similarity: float,
    bm25_score: float,
    trigram_score: float,
    raw_rrf_score: float,
    authority_weight: float = 1.0,
    is_code_symbol: bool = False,
    market_scope: DocumentScope | str | None = None,
    domain_id: str | None = None,
    scope_mismatch: bool = False,
) -> tuple[float, float, bool]:
    """Compute calibrated [0, 1] relevance score, lexical coverage, and boilerplate flag."""
    if _is_license_boilerplate(title, section, content, query_text):
        return 0.0, 0.0, True

    scope_aliases = _market_scope_aliases(market_scope, domain_id=domain_id)
    tokens = _extract_substantive_tokens(query_text)
    decomposed_title = _decompose_identifier_words(title) if is_code_symbol else ""
    decomposed_sec = _decompose_identifier_words(section) if is_code_symbol else ""
    haystack = (
        f"{title} {decomposed_title} {section} {decomposed_sec} {scope_aliases} {content[:1800]}"
    ).lower()
    actual_title_hay = f"{title} {decomposed_title} {section} {decomposed_sec}".lower()
    scope_hay = scope_aliases.lower()

    if tokens:
        matched = 0
        title_matched = 0
        scope_matched = 0
        for tok in tokens:
            if _token_matches_text(tok, haystack):
                matched += 1
            if _token_matches_text(tok, actual_title_hay):
                title_matched += 1
            if scope_hay and _token_matches_text(tok, scope_hay):
                scope_matched += 1
        lexical_coverage = matched / len(tokens)
        title_coverage = title_matched / len(tokens)
        scope_words = {w for w in re.findall(r"[a-z]{4,}", scope_hay)} if scope_hay else set()
        explicit_scope_in_title = bool(
            scope_words
            and title_matched >= 1
            and any(_token_matches_text(sw, actual_title_hay) for sw in scope_words)
        )
        scope_bonus = (0.10 if explicit_scope_in_title else 0.03) if scope_matched > 0 else 0.0
    else:
        lexical_coverage = 0.0
        title_coverage = 0.0
        scope_bonus = 0.0
    if scope_mismatch and scope_bonus == 0.0:
        scope_bonus = -0.06

    cos_clamped = max(0.0, min(1.0, cosine_similarity))
    bm25_norm = (
        min(1.0, math.log1p(max(0.0, bm25_score)) / 2.2)
        if not is_code_symbol
        else min(1.0, max(0.0, bm25_score) / 0.35)
    )
    trgm_norm = min(1.0, max(0.0, trigram_score) / 0.45)

    lex_strength = 0.45 * lexical_coverage + 0.25 * bm25_norm + 0.30 * title_coverage
    if is_code_symbol and lexical_coverage > 0.0 and (bm25_score > 0.0 or trigram_score >= 0.20):
        # Text embedders often score raw AST signatures lower on cosine than prose docs;
        # blend lexical/BM25/trigram evidence proportionally without inflating zero-match hits.
        eff_cos = max(cos_clamped, min(0.82, 0.85 * lex_strength))
        calibrated = (
            0.25 * eff_cos
            + 0.45 * lex_strength
            + 0.20 * trgm_norm
            + 0.10 * max(0.0, min(1.0, raw_rrf_score))
        )
    else:
        calibrated = (
            0.50 * cos_clamped
            + 0.30 * lex_strength
            + 0.10 * trgm_norm
            + 0.10 * max(0.0, min(1.0, raw_rrf_score))
            + scope_bonus
        )
    if lexical_coverage == 0.0 and bm25_score <= 0.0:
        calibrated = min(calibrated, cos_clamped * 0.68)

    auth_factor = 0.85 + 0.15 * min(1.0, max(0.0, authority_weight))
    calibrated = round(min(1.0, max(0.0, calibrated * auth_factor)), 6)
    return calibrated, round(lexical_coverage, 4), False


def _format_db_vector_and_halfvec(vec: Sequence[float]) -> tuple[str, str]:
    """Validates non-zero norm at DB boundary and returns (vector_768_literal, halfvec_literal)."""
    checked = validate_non_zero_embedding(vec)
    halfvec_literal = "[" + ",".join(f"{v:.6f}" for v in checked) + "]"
    if len(checked) == 768:
        if all(abs(v) < 5e-7 for v in checked):
            normed768 = normalize_embedding(checked, target_dim=768, supports_matryoshka=True)
            lit = "[" + ",".join(f"{v:.6f}" for v in normed768) + "]"
            return lit, lit
        return halfvec_literal, halfvec_literal
    if len(checked) > 768:
        head768 = checked[:768]
        if sum(x * x for x in head768) <= 1e-9:
            head768 = [
                sum(abs(checked[j]) for j in range(i, len(checked), 768)) for i in range(768)
            ]
        proj768 = normalize_embedding(head768, target_dim=768, supports_matryoshka=True)
    else:
        tiled = [checked[i % len(checked)] for i in range(768)]
        proj768 = normalize_embedding(tiled, target_dim=768, supports_matryoshka=True)
    vec768_literal = "[" + ",".join(f"{v:.6f}" for v in proj768) + "]"
    return vec768_literal, halfvec_literal


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b:
        raise ValueError("Vectors must have non-zero dimensions")
    if len(vec_a) != len(vec_b):
        raise ValueError("Vectors must have matching non-zero dimensions")
    dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=True))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


SYSTEM_CONNECTOR_INGEST_SCOPE: Final[str] = "system:connector_ingest"


def _scopes_intersect(row_scopes: list[str], principal_scopes: list[str]) -> bool:
    return bool(set(row_scopes).intersection(principal_scopes))


def _connector_ingest_or_scopes_intersect(
    row_scopes: list[str], principal_scopes: list[str]
) -> bool:
    return SYSTEM_CONNECTOR_INGEST_SCOPE in principal_scopes or _scopes_intersect(
        row_scopes, principal_scopes
    )


def _trigram_similarity(text_a: str, text_b: str) -> float:
    if not text_a or not text_b:
        return 0.0
    a = f"  {text_a.lower()} "
    b = f"  {text_b.lower()} "
    if len(a) < 3 or len(b) < 3:
        return 1.0 if a.strip() == b.strip() else 0.0
    trigrams_a = Counter(a[i : i + 3] for i in range(len(a) - 2))
    trigrams_b = Counter(b[i : i + 3] for i in range(len(b) - 2))
    intersection = sum((trigrams_a & trigrams_b).values())
    total = sum(trigrams_a.values()) + sum(trigrams_b.values())
    if total == 0:
        return 0.0
    return (2.0 * intersection) / total
