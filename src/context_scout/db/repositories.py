"""RLS-enforced repositories and 3-way Normalized RRF search (M1 Foundation)."""

import dataclasses
import json
import logging
import math
import re
import threading
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any, Final, cast
from uuid import UUID

from psycopg import Connection

from context_scout.db.bm25 import BM25Okapi
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
    ModelArtifactKind,
    ModelArtifactStatus,
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
    DEFAULT_TENANT_ID,
    DocumentScope,
    DomainPackManifest,
    ExecutionTelemetryRecord,
    PrincipalContext,
    compute_normalized_rrf,
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


class PostgresRLSRepository:
    """Executes parameterized SQL inside an RLS-enforced PostgreSQL 17 transaction."""

    def upsert_domain_pack(
        self,
        conn: Connection[tuple[object, ...]],
        manifest: DomainPackManifest,
        *,
        tenant_id: UUID | None = None,
        s3_bundle_uri: str | None = None,
        bundle_sha256: str | None = None,
    ) -> None:
        conn.execute(
            """
            INSERT INTO domain_packs (
                tenant_id, domain_id, display_name, lora_adapter_id,
                primary_truth_source, freshness_half_life_days, config_json,
                s3_bundle_uri, bundle_sha256, updated_at
            )
            VALUES (
                COALESCE(
                    %s,
                    NULLIF(current_setting('app.tenant_id', true), '')::uuid,
                    '00000000-0000-0000-0000-000000000001'::uuid
                ),
                %s, %s, %s, %s, %s, %s::jsonb, %s, %s, NOW()
            )
            ON CONFLICT (tenant_id, domain_id) DO UPDATE SET
                display_name = EXCLUDED.display_name,
                lora_adapter_id = EXCLUDED.lora_adapter_id,
                primary_truth_source = EXCLUDED.primary_truth_source,
                freshness_half_life_days = EXCLUDED.freshness_half_life_days,
                config_json = EXCLUDED.config_json,
                s3_bundle_uri = COALESCE(EXCLUDED.s3_bundle_uri, domain_packs.s3_bundle_uri),
                bundle_sha256 = COALESCE(EXCLUDED.bundle_sha256, domain_packs.bundle_sha256),
                updated_at = NOW()
            """,
            (
                tenant_id,
                manifest.domain_id,
                manifest.display_name,
                manifest.lora_adapter_id,
                manifest.authority_rules.primary_truth_source,
                manifest.authority_rules.freshness_half_life_days,
                manifest.model_dump_json(),
                s3_bundle_uri,
                bundle_sha256,
            ),
        )

    def get_domain_pack(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
    ) -> DomainPackManifest | None:
        row = conn.execute(
            """
            SELECT config_json
            FROM domain_packs
            WHERE domain_id = %s
            LIMIT 1;
            """,
            (domain_id,),
        ).fetchone()
        if not row or row[0] is None:
            return None
        raw = row[0] if isinstance(row[0], dict) else json.loads(str(row[0]))
        if not raw or not isinstance(raw, dict) or "base_model_id" not in raw:
            return None
        return DomainPackManifest.model_validate(raw)

    def list_domain_packs(
        self,
        conn: Connection[tuple[object, ...]],
        *,
        include_persistent_flag: bool = False,
    ) -> list[DomainPackManifest]:
        rows = conn.execute(
            """
            SELECT config_json, s3_bundle_uri
            FROM domain_packs
            ORDER BY domain_id;
            """
        ).fetchall()
        manifests: list[DomainPackManifest] = []
        for row in rows:
            raw = row[0] if isinstance(row[0], dict) else json.loads(str(row[0]))
            if isinstance(raw, dict) and "base_model_id" in raw:
                if include_persistent_flag and row[1] is None:
                    continue
                manifests.append(DomainPackManifest.model_validate(raw))
        return manifests

    def list_domain_pack_entries(
        self,
        conn: Connection[tuple[object, ...]],
    ) -> list[tuple[DomainPackManifest, bool]]:
        """Returns (manifest, has_persistent_s3_bundle) tuples visible under RLS."""
        rows = conn.execute(
            """
            SELECT config_json, (s3_bundle_uri IS NOT NULL) AS is_persistent
            FROM domain_packs
            ORDER BY domain_id;
            """
        ).fetchall()
        entries: list[tuple[DomainPackManifest, bool]] = []
        for row in rows:
            raw = row[0] if isinstance(row[0], dict) else json.loads(str(row[0]))
            if isinstance(raw, dict) and "base_model_id" in raw:
                entries.append((DomainPackManifest.model_validate(raw), bool(row[1])))
        return entries

    def delete_domain_pack(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
    ) -> int:
        cur = conn.execute("DELETE FROM domain_packs WHERE domain_id = %s;", (domain_id,))
        return max(cur.rowcount or 0, 0)

    def upsert_domain_sync_manifest(
        self,
        conn: Connection[tuple[object, ...]],
        record: DomainSyncManifestRecord,
    ) -> None:
        conn.execute(
            """
            INSERT INTO domain_sync_manifests (
                tenant_id, domain_id, kind, source_dir,
                repo_commit_shas, empty_doc_shas,
                code_files_indexed, ast_symbols_indexed,
                ast_edges_indexed, doc_chunks_indexed,
                s3_manifest_uri, manifest_sha256, synced_at
            )
            VALUES (
                %s, %s, %s, %s, %s::jsonb, %s::jsonb,
                %s, %s, %s, %s, %s, %s, COALESCE(%s, NOW())
            )
            ON CONFLICT (tenant_id, domain_id) DO UPDATE SET
                kind = EXCLUDED.kind,
                source_dir = EXCLUDED.source_dir,
                repo_commit_shas = EXCLUDED.repo_commit_shas,
                empty_doc_shas = EXCLUDED.empty_doc_shas,
                code_files_indexed = EXCLUDED.code_files_indexed,
                ast_symbols_indexed = EXCLUDED.ast_symbols_indexed,
                ast_edges_indexed = EXCLUDED.ast_edges_indexed,
                doc_chunks_indexed = EXCLUDED.doc_chunks_indexed,
                s3_manifest_uri = COALESCE(
                    EXCLUDED.s3_manifest_uri, domain_sync_manifests.s3_manifest_uri
                ),
                manifest_sha256 = COALESCE(
                    EXCLUDED.manifest_sha256, domain_sync_manifests.manifest_sha256
                ),
                synced_at = NOW();
            """,
            (
                record.tenant_id,
                record.domain_id,
                record.kind,
                record.source_dir,
                json.dumps(record.repo_commit_shas),
                json.dumps(record.empty_doc_shas),
                record.code_files_indexed,
                record.ast_symbols_indexed,
                record.ast_edges_indexed,
                record.doc_chunks_indexed,
                record.s3_manifest_uri,
                record.manifest_sha256,
                record.synced_at,
            ),
        )

    def get_domain_sync_manifest(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
    ) -> DomainSyncManifestRecord | None:
        row = conn.execute(
            """
            SELECT tenant_id, domain_id, kind, source_dir,
                   repo_commit_shas, empty_doc_shas,
                   code_files_indexed, ast_symbols_indexed,
                   ast_edges_indexed, doc_chunks_indexed,
                   s3_manifest_uri, manifest_sha256, synced_at
            FROM domain_sync_manifests
            WHERE domain_id = %s
            LIMIT 1;
            """,
            (domain_id,),
        ).fetchone()
        if row is None:
            return None
        raw_commits = row[4] if isinstance(row[4], dict) else json.loads(str(row[4] or "{}"))
        raw_empty = row[5] if isinstance(row[5], dict) else json.loads(str(row[5] or "{}"))
        return DomainSyncManifestRecord(
            tenant_id=UUID(str(row[0])),
            domain_id=str(row[1]),
            kind=str(row[2]),
            source_dir=str(row[3] or ""),
            repo_commit_shas={str(k): str(v) for k, v in raw_commits.items()},
            empty_doc_shas={str(k): str(v) for k, v in raw_empty.items()},
            code_files_indexed=int(str(row[6] or 0)),
            ast_symbols_indexed=int(str(row[7] or 0)),
            ast_edges_indexed=int(str(row[8] or 0)),
            doc_chunks_indexed=int(str(row[9] or 0)),
            s3_manifest_uri=str(row[10]) if row[10] is not None else None,
            manifest_sha256=str(row[11]) if row[11] is not None else None,
            synced_at=_as_opt_dt(row[12]),
        )

    def register_embedding_space(
        self,
        conn: Connection[tuple[object, ...]],
        record: EmbeddingSpaceRecord,
    ) -> EmbeddingSpaceRecord:
        """Registers or updates an embedding space in PG17 under RLS."""
        if record.dimensions < 128 or record.dimensions > 4096:
            raise ValueError(
                f"Embedding space dimensions must be in [128, 4096], got {record.dimensions}"
            )
        conn.execute(
            "SELECT pg_advisory_xact_lock("
            "hashtextextended(COALESCE(current_setting('app.tenant_id', true), '') || ':' || %s, 0)"
            ");",
            (record.domain_id,),
        )
        effective_status = "active" if record.is_active else record.status
        if record.is_active:
            conn.execute(
                """
                UPDATE embedding_spaces
                SET is_active = FALSE,
                    status = CASE WHEN status = 'active' THEN 'retired' ELSE status END
                WHERE domain_id = %s AND space_id <> %s AND is_active = TRUE;
                """,
                (record.domain_id, record.space_id),
            )
        row = conn.execute(
            """
            INSERT INTO embedding_spaces (
                tenant_id, domain_id, space_id, model_id, provider,
                dimensions, quantization, status, is_active,
                index_operator_class, created_at, activated_at
            )
            VALUES (
                COALESCE(
                    NULLIF(current_setting('app.tenant_id', true), '')::uuid,
                    %s
                ),
                %s, %s, %s, %s, %s, %s, %s, %s, %s,
                COALESCE(%s, NOW()),
                CASE WHEN %s THEN COALESCE(%s, NOW()) ELSE %s END
            )
            ON CONFLICT (tenant_id, domain_id, space_id) DO UPDATE SET
                model_id = EXCLUDED.model_id,
                provider = EXCLUDED.provider,
                dimensions = EXCLUDED.dimensions,
                quantization = EXCLUDED.quantization,
                status = EXCLUDED.status,
                is_active = EXCLUDED.is_active,
                index_operator_class = EXCLUDED.index_operator_class,
                activated_at = CASE
                    WHEN EXCLUDED.is_active THEN COALESCE(EXCLUDED.activated_at, NOW())
                    ELSE embedding_spaces.activated_at
                END
            RETURNING tenant_id, domain_id, space_id, model_id, provider,
                      dimensions, quantization, status, is_active,
                      index_operator_class, created_at, activated_at;
            """,
            (
                record.tenant_id,
                record.domain_id,
                record.space_id,
                record.model_id,
                record.provider,
                record.dimensions,
                record.quantization,
                effective_status,
                record.is_active,
                record.index_operator_class,
                record.created_at,
                record.is_active,
                record.activated_at,
                record.activated_at,
            ),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"Failed to upsert embedding_space '{record.space_id}'")
        return EmbeddingSpaceRecord(
            tenant_id=UUID(str(row[0])),
            domain_id=str(row[1]),
            space_id=str(row[2]),
            model_id=str(row[3]),
            provider=str(row[4]),
            dimensions=int(str(row[5])),
            quantization=str(row[6]),  # type: ignore[arg-type]
            status=str(row[7]),  # type: ignore[arg-type]
            is_active=bool(row[8]),
            index_operator_class=str(row[9]),
            created_at=_as_opt_dt(row[10]),
            activated_at=_as_opt_dt(row[11]),
        )

    def get_active_embedding_space(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
    ) -> EmbeddingSpaceRecord | None:
        """Returns the currently active EmbeddingSpaceRecord for domain_id under RLS."""
        row = conn.execute(
            """
            SELECT tenant_id, domain_id, space_id, model_id, provider,
                   dimensions, quantization, status, is_active,
                   index_operator_class, created_at, activated_at
            FROM embedding_spaces
            WHERE domain_id = %s AND is_active = TRUE
            LIMIT 1;
            """,
            (domain_id,),
        ).fetchone()
        if row is None:
            return None
        return EmbeddingSpaceRecord(
            tenant_id=UUID(str(row[0])),
            domain_id=str(row[1]),
            space_id=str(row[2]),
            model_id=str(row[3]),
            provider=str(row[4]),
            dimensions=int(str(row[5])),
            quantization=str(row[6]),  # type: ignore[arg-type]
            status=str(row[7]),  # type: ignore[arg-type]
            is_active=bool(row[8]),
            index_operator_class=str(row[9]),
            created_at=_as_opt_dt(row[10]),
            activated_at=_as_opt_dt(row[11]),
        )

    def list_embedding_spaces(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
    ) -> list[EmbeddingSpaceRecord]:
        """Lists all registered embedding spaces for domain_id under RLS."""
        rows = conn.execute(
            """
            SELECT tenant_id, domain_id, space_id, model_id, provider,
                   dimensions, quantization, status, is_active,
                   index_operator_class, created_at, activated_at
            FROM embedding_spaces
            WHERE domain_id = %s
            ORDER BY is_active DESC, space_id ASC;
            """,
            (domain_id,),
        ).fetchall()
        return [
            EmbeddingSpaceRecord(
                tenant_id=UUID(str(r[0])),
                domain_id=str(r[1]),
                space_id=str(r[2]),
                model_id=str(r[3]),
                provider=str(r[4]),
                dimensions=int(str(r[5])),
                quantization=str(r[6]),  # type: ignore[arg-type]
                status=str(r[7]),  # type: ignore[arg-type]
                is_active=bool(r[8]),
                index_operator_class=str(r[9]),
                created_at=_as_opt_dt(r[10]),
                activated_at=_as_opt_dt(r[11]),
            )
            for r in rows
        ]

    def stage_shadow_embeddings(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        shadow_space_id: str,
        *,
        symbol_embeddings: Mapping[UUID, Sequence[float]] | None = None,
        chunk_embeddings: Mapping[UUID, Sequence[float]] | None = None,
    ) -> dict[str, int]:
        """Stages blue/green shadow halfvec(D) embeddings without disrupting active queries."""
        space_row = conn.execute(
            """
            SELECT dimensions
            FROM embedding_spaces
            WHERE domain_id = %s AND space_id = %s
            LIMIT 1;
            """,
            (domain_id, shadow_space_id),
        ).fetchone()
        if space_row is None:
            raise KeyError(
                f"Shadow embedding_space '{shadow_space_id}' not registered for '{domain_id}'"
            )
        expected_dim = int(str(space_row[0]))

        symbols_staged = 0
        if symbol_embeddings:
            sym_params: list[tuple[str, str, str, UUID]] = []
            for sym_id, vec in symbol_embeddings.items():
                checked = validate_non_zero_embedding(vec, expected_dim=expected_dim)
                hv_lit = "[" + ",".join(f"{v:.6f}" for v in checked) + "]"
                sym_params.append((shadow_space_id, hv_lit, domain_id, sym_id))
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    UPDATE ast_symbols
                    SET shadow_embedding_space_id = %s,
                        shadow_embedding_halfvec = %s::halfvec
                    WHERE domain_id = %s AND symbol_id = %s;
                    """,
                    sym_params,
                )
                symbols_staged = max(cur.rowcount or 0, 0)

        chunks_staged = 0
        if chunk_embeddings:
            chk_params: list[tuple[str, str, str, UUID]] = []
            for chunk_id, vec in chunk_embeddings.items():
                checked = validate_non_zero_embedding(vec, expected_dim=expected_dim)
                hv_lit = "[" + ",".join(f"{v:.6f}" for v in checked) + "]"
                chk_params.append((shadow_space_id, hv_lit, domain_id, chunk_id))
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    UPDATE doc_chunks
                    SET shadow_embedding_space_id = %s,
                        shadow_embedding_halfvec = %s::halfvec
                    WHERE domain_id = %s AND chunk_id = %s;
                    """,
                    chk_params,
                )
                chunks_staged = max(cur.rowcount or 0, 0)

        return {
            "symbols_staged": symbols_staged,
            "chunks_staged": chunks_staged,
        }

    def reindex_domain_shadow_space(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        shadow_space: EmbeddingSpaceRecord,
        embedder: Any,
    ) -> dict[str, int]:
        """Builds shadow halfvec(D) embeddings for all visible symbols & chunks in domain_id."""
        staged_record = dataclasses.replace(shadow_space, is_active=False, status="shadow")
        self.register_embedding_space(conn, staged_record)

        sym_rows = conn.execute(
            """
            SELECT symbol_id, qualified_name, signature_skeleton, contextual_summary
            FROM ast_symbols
            WHERE domain_id = %s
            ORDER BY symbol_id;
            """,
            (domain_id,),
        ).fetchall()
        sym_map: dict[UUID, Sequence[float]] = {}
        if sym_rows:
            sym_texts = [f"{r[1]} {r[2]} {r[3]}".strip() for r in sym_rows]
            sym_vecs = embedder.embed_batch(sym_texts, is_query=False)
            for r, vec in zip(sym_rows, sym_vecs, strict=True):
                sym_map[UUID(str(r[0]))] = vec

        chk_rows = conn.execute(
            """
            SELECT chunk_id, doc_title, section_heading, contextual_header, verbatim_content
            FROM doc_chunks
            WHERE domain_id = %s AND NOT is_stale
            ORDER BY chunk_id;
            """,
            (domain_id,),
        ).fetchall()
        chk_map: dict[UUID, Sequence[float]] = {}
        if chk_rows:
            chk_texts = [f"{r[1]} {r[2]} {r[3]} {r[4]}".strip() for r in chk_rows]
            chk_vecs = embedder.embed_batch(chk_texts, is_query=False)
            for r, vec in zip(chk_rows, chk_vecs, strict=True):
                chk_map[UUID(str(r[0]))] = vec

        return self.stage_shadow_embeddings(
            conn,
            domain_id,
            shadow_space.space_id,
            symbol_embeddings=sym_map,
            chunk_embeddings=chk_map,
        )

    def activate_embedding_space(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        target_space_id: str,
        *,
        require_complete_shadow: bool = True,
    ) -> EmbeddingSpaceRecord:
        """Atomically swaps active embedding_space pointer and promotes staged shadow halfvecs."""
        conn.execute(
            "SELECT pg_advisory_xact_lock("
            "hashtextextended(COALESCE(current_setting('app.tenant_id', true), '') || ':' || %s, 0)"
            ");",
            (domain_id,),
        )
        space_row = conn.execute(
            """
            SELECT dimensions
            FROM embedding_spaces
            WHERE domain_id = %s AND space_id = %s
            LIMIT 1;
            """,
            (domain_id, target_space_id),
        ).fetchone()
        if space_row is None:
            raise KeyError(
                f"Target embedding_space '{target_space_id}' not registered for '{domain_id}'"
            )

        if require_complete_shadow:
            missing_sym = conn.execute(
                """
                SELECT COUNT(*)
                FROM ast_symbols
                WHERE domain_id = %s
                  AND COALESCE(embedding_space_id, 'embeddinggemma-2-768') <> %s
                  AND (
                      shadow_embedding_space_id IS DISTINCT FROM %s
                      OR shadow_embedding_halfvec IS NULL
                  );
                """,
                (domain_id, target_space_id, target_space_id),
            ).fetchone()
            missing_chk = conn.execute(
                """
                SELECT COUNT(*)
                FROM doc_chunks
                WHERE domain_id = %s
                  AND NOT is_stale
                  AND COALESCE(embedding_space_id, 'embeddinggemma-2-768') <> %s
                  AND (
                      shadow_embedding_space_id IS DISTINCT FROM %s
                      OR shadow_embedding_halfvec IS NULL
                  );
                """,
                (domain_id, target_space_id, target_space_id),
            ).fetchone()
            unready_syms = int(str((missing_sym or (0,))[0]))
            unready_chks = int(str((missing_chk or (0,))[0]))
            if unready_syms > 0 or unready_chks > 0:
                raise RuntimeError(
                    f"Cannot activate embedding_space '{target_space_id}' for '{domain_id}': "
                    f"{unready_syms} ast_symbols and {unready_chks} doc_chunks lack staged "
                    "shadow halfvec embeddings."
                )

        conn.execute(
            """
            UPDATE ast_symbols
            SET embedding_space_id = shadow_embedding_space_id,
                embedding_halfvec = shadow_embedding_halfvec,
                shadow_embedding_space_id = NULL,
                shadow_embedding_halfvec = NULL
            WHERE domain_id = %s
              AND shadow_embedding_space_id = %s
              AND shadow_embedding_halfvec IS NOT NULL;
            """,
            (domain_id, target_space_id),
        )
        conn.execute(
            """
            UPDATE doc_chunks
            SET embedding_space_id = shadow_embedding_space_id,
                embedding_halfvec = shadow_embedding_halfvec,
                shadow_embedding_space_id = NULL,
                shadow_embedding_halfvec = NULL
            WHERE domain_id = %s
              AND shadow_embedding_space_id = %s
              AND shadow_embedding_halfvec IS NOT NULL;
            """,
            (domain_id, target_space_id),
        )
        conn.execute(
            """
            UPDATE embedding_spaces
            SET is_active = FALSE,
                status = CASE WHEN status = 'active' THEN 'retired' ELSE status END
            WHERE domain_id = %s AND space_id <> %s AND is_active = TRUE;
            """,
            (domain_id, target_space_id),
        )
        conn.execute(
            """
            UPDATE embedding_spaces
            SET is_active = TRUE,
                status = 'active',
                activated_at = NOW()
            WHERE domain_id = %s AND space_id = %s;
            """,
            (domain_id, target_space_id),
        )
        active = self.get_active_embedding_space(conn, domain_id)
        if active is None or active.space_id != target_space_id:
            raise RuntimeError(
                f"Atomic pointer swap failed for embedding_space '{target_space_id}'"
            )
        return active

    def insert_code_file(
        self, conn: Connection[tuple[object, ...]], record: CodeFileRecord
    ) -> None:
        conn.execute(
            """
            INSERT INTO code_files (
                file_id, tenant_id, domain_id, repo_name, file_path,
                git_commit_sha, git_blob_sha, language,
                allowed_scopes, raw_content, line_count,
                ranking_weight, is_generated, is_vendored
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                record.file_id,
                record.tenant_id,
                record.domain_id,
                record.repo_name,
                record.file_path,
                record.git_commit_sha,
                record.git_blob_sha,
                record.language,
                record.allowed_scopes,
                record.raw_content,
                record.line_count,
                record.ranking_weight,
                record.is_generated,
                record.is_vendored,
            ),
        )

    def insert_ast_symbol(
        self, conn: Connection[tuple[object, ...]], record: ASTSymbolRecord
    ) -> None:
        vec_literal, halfvec_literal = _format_db_vector_and_halfvec(record.embedding)
        conn.execute(
            """
            INSERT INTO ast_symbols (
                symbol_id, tenant_id, file_id, domain_id, qualified_name,
                symbol_kind, start_line, end_line,
                signature_skeleton, contextual_summary,
                allowed_scopes, embedding, embedding_space_id, embedding_halfvec,
                repo_qualified_id, bare_name
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector, %s, %s::halfvec, %s, %s)
            """,
            (
                record.symbol_id,
                record.tenant_id,
                record.file_id,
                record.domain_id,
                record.qualified_name,
                record.symbol_kind,
                record.start_line,
                record.end_line,
                record.signature_skeleton,
                record.contextual_summary,
                record.allowed_scopes,
                vec_literal,
                record.embedding_space_id,
                halfvec_literal,
                record.repo_qualified_id,
                record.bare_name,
            ),
        )

    def insert_ast_edge(self, conn: Connection[tuple[object, ...]], record: ASTEdgeRecord) -> None:
        conn.execute(
            """
            INSERT INTO ast_edges (
                edge_id, tenant_id, domain_id, source_symbol_id,
                target_qualified_name, target_symbol_id,
                edge_kind, allowed_scopes, target_repo_qualified_id, is_cross_repo
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                record.edge_id,
                record.tenant_id,
                record.domain_id,
                record.source_symbol_id,
                record.target_qualified_name,
                record.target_symbol_id,
                record.edge_kind,
                record.allowed_scopes,
                record.target_repo_qualified_id,
                record.is_cross_repo,
            ),
        )

    def list_visible_code_files(
        self, conn: Connection[tuple[object, ...]], domain_id: str
    ) -> list[str]:
        rows = conn.execute(
            "SELECT file_path FROM code_files WHERE domain_id = %s ORDER BY file_path",
            (domain_id,),
        ).fetchall()
        return [str(r[0]) for r in rows]

    def list_visible_ast_symbols(
        self, conn: Connection[tuple[object, ...]], domain_id: str
    ) -> list[str]:
        rows = conn.execute(
            "SELECT qualified_name FROM ast_symbols WHERE domain_id = %s ORDER BY qualified_name",
            (domain_id,),
        ).fetchall()
        return [str(r[0]) for r in rows]

    def list_visible_ast_edges(
        self, conn: Connection[tuple[object, ...]], domain_id: str
    ) -> list[str]:
        rows = conn.execute(
            "SELECT target_qualified_name FROM ast_edges "
            "WHERE domain_id = %s ORDER BY target_qualified_name",
            (domain_id,),
        ).fetchall()
        return [str(r[0]) for r in rows]

    def get_file_blob_hashes(
        self, conn: Connection[tuple[object, ...]], domain_id: str, repo_name: str
    ) -> dict[str, tuple[UUID, str]]:
        rows = conn.execute(
            """
            SELECT file_path, file_id, git_blob_sha
            FROM code_files
            WHERE domain_id = %s AND repo_name = %s
            ORDER BY file_path;
            """,
            (domain_id, repo_name),
        ).fetchall()
        return {
            str(r[0]): (r[1] if isinstance(r[1], UUID) else UUID(str(r[1])), str(r[2]))
            for r in rows
        }

    def delete_code_file(self, conn: Connection[tuple[object, ...]], file_id: UUID) -> None:
        conn.execute("DELETE FROM code_files WHERE file_id = %s;", (file_id,))

    def delete_code_file_and_symbols(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        repo_name: str,
        file_path: str,
    ) -> None:
        """Deletes a code file by domain, repo, and file_path, cascading symbol deletions."""
        conn.execute(
            """
            DELETE FROM code_files
            WHERE domain_id = %s AND (repo_name = %s OR repo_name = %s) AND file_path = %s;
            """,
            (domain_id, repo_name, repo_name.replace("/", "_"), file_path),
        )

    def insert_repository(
        self, conn: Connection[tuple[object, ...]], record: RepositoryRecord
    ) -> None:
        conn.execute(
            """
            INSERT INTO repositories (
                repo_id, tenant_id, domain_id, repo_name, source_type, remote_url,
                default_branch, head_commit_sha, allowed_scopes, metadata_json,
                is_active, created_at, updated_at
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s,
                COALESCE(%s, NOW()), COALESCE(%s, NOW())
            )
            ON CONFLICT (tenant_id, domain_id, repo_name) DO UPDATE SET
                source_type = EXCLUDED.source_type,
                remote_url = EXCLUDED.remote_url,
                default_branch = EXCLUDED.default_branch,
                head_commit_sha = EXCLUDED.head_commit_sha,
                allowed_scopes = EXCLUDED.allowed_scopes,
                metadata_json = EXCLUDED.metadata_json,
                is_active = EXCLUDED.is_active,
                updated_at = NOW();
            """,
            (
                record.repo_id,
                record.tenant_id,
                record.domain_id,
                record.repo_name,
                record.source_type,
                record.remote_url,
                record.default_branch,
                record.head_commit_sha,
                record.allowed_scopes,
                json.dumps(record.metadata_json),
                record.is_active,
                record.created_at,
                record.updated_at,
            ),
        )

    def upsert_repository(
        self, conn: Connection[tuple[object, ...]], record: RepositoryRecord
    ) -> None:
        self.insert_repository(conn, record)

    def get_repository(
        self, conn: Connection[tuple[object, ...]], domain_id: str, repo_name: str
    ) -> RepositoryRecord | None:
        row = conn.execute(
            """
            SELECT repo_id, domain_id, repo_name, source_type, remote_url,
                   default_branch, head_commit_sha, allowed_scopes, metadata_json,
                   is_active, created_at, updated_at
            FROM repositories
            WHERE domain_id = %s AND repo_name = %s;
            """,
            (domain_id, repo_name),
        ).fetchone()
        if not row:
            return None
        return RepositoryRecord(
            repo_id=UUID(str(row[0])),
            domain_id=str(row[1]),
            repo_name=str(row[2]),
            source_type=row[3],  # type: ignore[arg-type]
            remote_url=str(row[4]),
            default_branch=str(row[5]),
            head_commit_sha=str(row[6]),
            allowed_scopes=[str(s) for s in row[7]] if isinstance(row[7], list) else [],
            metadata_json=row[8] if isinstance(row[8], dict) else json.loads(str(row[8])),
            is_active=bool(row[9]),
            created_at=_as_opt_dt(row[10]),
            updated_at=_as_opt_dt(row[11]),
        )

    def list_repositories(
        self, conn: Connection[tuple[object, ...]], domain_id: str
    ) -> list[RepositoryRecord]:
        rows = conn.execute(
            """
            SELECT repo_id, domain_id, repo_name, source_type, remote_url,
                   default_branch, head_commit_sha, allowed_scopes, metadata_json,
                   is_active, created_at, updated_at
            FROM repositories
            WHERE domain_id = %s
            ORDER BY repo_name;
            """,
            (domain_id,),
        ).fetchall()
        return [
            RepositoryRecord(
                repo_id=UUID(str(row[0])),
                domain_id=str(row[1]),
                repo_name=str(row[2]),
                source_type=row[3],  # type: ignore[arg-type]
                remote_url=str(row[4]),
                default_branch=str(row[5]),
                head_commit_sha=str(row[6]),
                allowed_scopes=[str(s) for s in row[7]] if isinstance(row[7], list) else [],
                metadata_json=row[8] if isinstance(row[8], dict) else json.loads(str(row[8])),
                is_active=bool(row[9]),
                created_at=_as_opt_dt(row[10]),
                updated_at=_as_opt_dt(row[11]),
            )
            for row in rows
        ]

    def insert_repo_commit(
        self, conn: Connection[tuple[object, ...]], record: RepoCommitRecord
    ) -> None:
        conn.execute(
            """
            INSERT INTO repo_commits (
                commit_id, tenant_id, repo_id, commit_sha, parent_shas,
                author_name, author_email, commit_message,
                committed_at, indexed_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, COALESCE(%s, NOW()))
            ON CONFLICT (repo_id, commit_sha) DO NOTHING;
            """,
            (
                record.commit_id,
                record.tenant_id,
                record.repo_id,
                record.commit_sha,
                record.parent_shas,
                record.author_name,
                record.author_email,
                record.commit_message,
                record.committed_at,
                record.indexed_at,
            ),
        )

    def get_repo_commits(
        self, conn: Connection[tuple[object, ...]], repo_id: UUID
    ) -> list[RepoCommitRecord]:
        rows = conn.execute(
            """
            SELECT commit_id, repo_id, commit_sha, parent_shas,
                   author_name, author_email, commit_message,
                   committed_at, indexed_at
            FROM repo_commits
            WHERE repo_id = %s
            ORDER BY committed_at DESC;
            """,
            (repo_id,),
        ).fetchall()
        return [
            RepoCommitRecord(
                commit_id=UUID(str(row[0])),
                repo_id=UUID(str(row[1])),
                commit_sha=str(row[2]),
                parent_shas=[str(s) for s in row[3]] if isinstance(row[3], list) else [],
                author_name=str(row[4]),
                author_email=str(row[5]),
                commit_message=str(row[6]),
                committed_at=_as_dt(row[7]),
                indexed_at=_as_opt_dt(row[8]),
            )
            for row in rows
        ]

    def insert_confluence_page(
        self, conn: Connection[tuple[object, ...]], record: ConfluencePageRecord
    ) -> None:
        conn.execute(
            """
            INSERT INTO confluence_pages (
                page_id, tenant_id, domain_id, confluence_page_id, space_key,
                title, version_number, author_username, source_url,
                allowed_scopes, last_modified_at, indexed_at,
                status, is_deleted
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                COALESCE(%s, NOW()), 'current', FALSE
            )
            ON CONFLICT (tenant_id, domain_id, confluence_page_id) DO UPDATE SET
                space_key = EXCLUDED.space_key,
                title = EXCLUDED.title,
                version_number = EXCLUDED.version_number,
                author_username = EXCLUDED.author_username,
                source_url = EXCLUDED.source_url,
                allowed_scopes = EXCLUDED.allowed_scopes,
                last_modified_at = EXCLUDED.last_modified_at,
                indexed_at = NOW(),
                status = 'current',
                is_deleted = FALSE;
            """,
            (
                record.page_id,
                record.tenant_id,
                record.domain_id,
                record.confluence_page_id,
                record.space_key,
                record.title,
                record.version_number,
                record.author_username,
                record.source_url,
                record.allowed_scopes,
                record.last_modified_at,
                record.indexed_at,
            ),
        )

    def get_confluence_page(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        confluence_page_id: str,
        include_deleted: bool = False,
    ) -> ConfluencePageRecord | None:
        query = """
            SELECT page_id, domain_id, confluence_page_id, space_key,
                   title, version_number, author_username, source_url,
                   allowed_scopes, last_modified_at, indexed_at
            FROM confluence_pages
            WHERE domain_id = %s AND confluence_page_id = %s
        """
        if not include_deleted:
            query += " AND NOT is_deleted"
        query += ";"
        row = conn.execute(query, (domain_id, confluence_page_id)).fetchone()
        if not row:
            return None
        return ConfluencePageRecord(
            page_id=UUID(str(row[0])),
            domain_id=str(row[1]),
            confluence_page_id=str(row[2]),
            space_key=str(row[3]),
            title=str(row[4]),
            version_number=int(str(row[5])),
            author_username=str(row[6]),
            source_url=str(row[7]),
            allowed_scopes=[str(s) for s in row[8]] if isinstance(row[8], list) else [],
            last_modified_at=_as_dt(row[9]),
            indexed_at=_as_opt_dt(row[10]),
        )

    def list_confluence_pages(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        space_key: str | None = None,
        include_deleted: bool = False,
    ) -> list[ConfluencePageRecord]:
        query = """
            SELECT page_id, domain_id, confluence_page_id, space_key,
                   title, version_number, author_username, source_url,
                   allowed_scopes, last_modified_at, indexed_at
            FROM confluence_pages
            WHERE domain_id = %s
        """
        params: list[object] = [domain_id]
        if space_key:
            query += " AND space_key = %s"
            params.append(space_key)
        if not include_deleted:
            query += " AND NOT is_deleted"
        query += " ORDER BY title;"
        rows = conn.execute(query, tuple(params)).fetchall()
        return [
            ConfluencePageRecord(
                page_id=UUID(str(row[0])),
                domain_id=str(row[1]),
                confluence_page_id=str(row[2]),
                space_key=str(row[3]),
                title=str(row[4]),
                version_number=int(str(row[5])),
                author_username=str(row[6]),
                source_url=str(row[7]),
                allowed_scopes=[str(s) for s in row[8]] if isinstance(row[8], list) else [],
                last_modified_at=_as_dt(row[9]),
                indexed_at=_as_opt_dt(row[10]),
            )
            for row in rows
        ]

    def tombstone_confluence_pages(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        confluence_page_ids: list[str],
        reason: str = "Page deleted or archived in Confluence",
        status: str = "deleted",
    ) -> tuple[int, int]:
        """Soft-deletes Confluence pages and marks their active doc_chunks stale."""
        clean_ids = [pid for pid in confluence_page_ids if pid]
        if not clean_ids:
            return (0, 0)
        doc_ids = [f"confluence:{pid}" for pid in clean_ids]
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE confluence_pages
                SET is_deleted = TRUE,
                    status = %s
                WHERE domain_id = %s
                  AND confluence_page_id = ANY(%s::text[])
                  AND (NOT is_deleted OR status <> %s);
                """,
                (status, domain_id, clean_ids, status),
            )
            pages_tombstoned = max(cur.rowcount or 0, 0)
            cur.execute(
                """
                UPDATE doc_chunks
                SET is_stale = TRUE,
                    stale_reason = %s
                WHERE domain_id = %s
                  AND doc_id = ANY(%s::text[])
                  AND NOT is_stale;
                """,
                (reason, domain_id, doc_ids),
            )
            chunks_flagged = max(cur.rowcount or 0, 0)
        return (pages_tombstoned, chunks_flagged)

    def update_confluence_allowed_scopes(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        confluence_page_id: str,
        allowed_scopes: list[str],
    ) -> int:
        """Synchronizes changed allowed_scopes across confluence_pages and active doc_chunks."""
        doc_id = f"confluence:{confluence_page_id}"
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE confluence_pages
                SET allowed_scopes = %s
                WHERE domain_id = %s AND confluence_page_id = %s;
                """,
                (allowed_scopes, domain_id, confluence_page_id),
            )
            cur.execute(
                """
                UPDATE doc_chunks
                SET allowed_scopes = %s
                WHERE domain_id = %s AND doc_id = %s AND NOT is_stale;
                """,
                (allowed_scopes, domain_id, doc_id),
            )
            return max(cur.rowcount or 0, 0)

    def insert_jira_issue(
        self, conn: Connection[tuple[object, ...]], record: JiraIssueRecord
    ) -> None:
        conn.execute(
            """
            INSERT INTO jira_issues (
                issue_id, tenant_id, domain_id, issue_key, project_key,
                summary, issue_type, status, resolution,
                assignee_username, source_url, allowed_scopes,
                updated_at, indexed_at,
                epic_key, parent_issue_key, linked_issues_json, linked_mrs_json
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, COALESCE(%s, NOW()),
                %s, %s, %s::jsonb, %s::jsonb
            )
            ON CONFLICT (tenant_id, domain_id, issue_key) DO UPDATE SET
                project_key = EXCLUDED.project_key,
                summary = EXCLUDED.summary,
                issue_type = EXCLUDED.issue_type,
                status = EXCLUDED.status,
                resolution = EXCLUDED.resolution,
                assignee_username = EXCLUDED.assignee_username,
                source_url = EXCLUDED.source_url,
                allowed_scopes = EXCLUDED.allowed_scopes,
                updated_at = EXCLUDED.updated_at,
                indexed_at = NOW(),
                epic_key = EXCLUDED.epic_key,
                parent_issue_key = EXCLUDED.parent_issue_key,
                linked_issues_json = EXCLUDED.linked_issues_json,
                linked_mrs_json = EXCLUDED.linked_mrs_json;
            """,
            (
                record.issue_id,
                record.tenant_id,
                record.domain_id,
                record.issue_key,
                record.project_key,
                record.summary,
                record.issue_type,
                record.status,
                record.resolution,
                record.assignee_username,
                record.source_url,
                record.allowed_scopes,
                record.updated_at,
                record.indexed_at,
                record.epic_key,
                record.parent_issue_key,
                json.dumps(record.linked_issues),
                json.dumps(record.linked_mrs),
            ),
        )

    @staticmethod
    def _row_to_jira_issue(row: tuple[object, ...]) -> JiraIssueRecord:
        linked_issues_raw = (
            row[13] if isinstance(row[13], list) else json.loads(str(row[13] or "[]"))
        )
        linked_mrs_raw = row[14] if isinstance(row[14], list) else json.loads(str(row[14] or "[]"))
        return JiraIssueRecord(
            issue_id=UUID(str(row[0])),
            domain_id=str(row[1]),
            issue_key=str(row[2]),
            project_key=str(row[3]),
            summary=str(row[4]),
            issue_type=str(row[5]),
            status=str(row[6]),
            resolution=str(row[7]) if row[7] is not None else None,
            assignee_username=str(row[8]) if row[8] is not None else None,
            source_url=str(row[9]),
            allowed_scopes=[str(s) for s in row[10]] if isinstance(row[10], list) else [],
            updated_at=_as_dt(row[11]),
            indexed_at=_as_opt_dt(row[12]),
            epic_key=str(row[15]) if row[15] is not None else None,
            parent_issue_key=str(row[16]) if row[16] is not None else None,
            linked_issues=[dict(item) for item in linked_issues_raw if isinstance(item, dict)],
            linked_mrs=[str(item) for item in linked_mrs_raw],
            tenant_id=UUID(str(row[17])),
        )

    def get_jira_issue(
        self, conn: Connection[tuple[object, ...]], domain_id: str, issue_key: str
    ) -> JiraIssueRecord | None:
        row = conn.execute(
            """
            SELECT issue_id, domain_id, issue_key, project_key,
                   summary, issue_type, status, resolution,
                   assignee_username, source_url, allowed_scopes,
                   updated_at, indexed_at,
                   linked_issues_json, linked_mrs_json,
                   epic_key, parent_issue_key, tenant_id
            FROM jira_issues
            WHERE domain_id = %s AND issue_key = %s;
            """,
            (domain_id, issue_key),
        ).fetchone()
        if not row:
            return None
        return self._row_to_jira_issue(row)

    def list_jira_issues(
        self, conn: Connection[tuple[object, ...]], domain_id: str, project_key: str | None = None
    ) -> list[JiraIssueRecord]:
        if project_key:
            rows = conn.execute(
                """
                SELECT issue_id, domain_id, issue_key, project_key,
                       summary, issue_type, status, resolution,
                       assignee_username, source_url, allowed_scopes,
                       updated_at, indexed_at,
                       linked_issues_json, linked_mrs_json,
                       epic_key, parent_issue_key, tenant_id
                FROM jira_issues
                WHERE domain_id = %s AND project_key = %s
                ORDER BY issue_key;
                """,
                (domain_id, project_key),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT issue_id, domain_id, issue_key, project_key,
                       summary, issue_type, status, resolution,
                       assignee_username, source_url, allowed_scopes,
                       updated_at, indexed_at,
                       linked_issues_json, linked_mrs_json,
                       epic_key, parent_issue_key, tenant_id
                FROM jira_issues
                WHERE domain_id = %s
                ORDER BY issue_key;
                """,
                (domain_id,),
            ).fetchall()
        return [self._row_to_jira_issue(row) for row in rows]

    TRAVERSE_1HOP_NEIGHBORHOOD_CTE_SQL: Final[str] = """
        WITH
        target_sym AS (
            SELECT s.symbol_id, s.file_id, s.domain_id, s.qualified_name, s.symbol_kind,
                   s.start_line, s.end_line, s.signature_skeleton, s.contextual_summary,
                   f.repo_name, f.file_path, f.git_commit_sha, f.git_blob_sha, f.raw_content
            FROM ast_symbols s
            JOIN code_files f ON f.file_id = s.file_id
            WHERE s.domain_id = %s AND s.symbol_id = %s
        ),
        focus_family AS (
            SELECT symbol_id, qualified_name, file_id
            FROM target_sym
            UNION
            SELECT s.symbol_id, s.qualified_name, s.file_id
            FROM ast_symbols s
            JOIN target_sym ts ON s.domain_id = ts.domain_id AND s.file_id = ts.file_id
            WHERE ts.symbol_kind = 'class'
              AND s.qualified_name LIKE (ts.qualified_name || '.' || CHR(37))
        ),
        hop1_outward AS (
            -- Hop 1: Outward edges (USES_TYPE, IMPORTS, INHERITS, and same-file CALLS)
            SELECT
                e.edge_id,
                e.edge_kind,
                e.target_symbol_id AS related_symbol_id,
                1 AS depth,
                'imported_type' AS intent_role
            FROM ast_edges e
            JOIN focus_family ff ON e.source_symbol_id = ff.symbol_id
            JOIN target_sym ts ON TRUE
            WHERE e.domain_id = %s
              AND e.target_symbol_id IS NOT NULL
              AND e.target_symbol_id NOT IN (
                  SELECT ff2.symbol_id FROM focus_family ff2 WHERE ff2.symbol_id <> ts.symbol_id
              )
              AND (
                  e.edge_kind IN (
                      'USES_TYPE', 'IMPORTS', 'INHERITS',
                      'PROVISIONED_BY', 'DISPATCHES', 'SELECTS', 'DOCUMENTS_COMPONENT',
                      'OWNED_BY_TEAM', 'PART_OF_SYSTEM',
                      'PUBLISHES_EVENT', 'SUBSCRIBES_EVENT', 'READS_TABLE',
                      'WRITES_TABLE', 'RENDERS_TEMPLATE', 'IMPLEMENTS_SPEC',
                      'IMPLEMENTS_ISSUE', 'BINDS_SERVICE_ACCOUNT', 'ACCESSES_SECRET'
                  )
                  OR (e.edge_kind = 'CALLS' AND EXISTS (
                      SELECT 1 FROM ast_symbols s2
                      WHERE s2.symbol_id = e.target_symbol_id AND s2.file_id = ts.file_id
                  ))
              )
        ),
        hop2_outward AS (
            -- Hop 2: Transitive schema/type/inheritance/import closure within the same repository
            SELECT
                e2.edge_id,
                e2.edge_kind,
                e2.target_symbol_id AS related_symbol_id,
                2 AS depth,
                'imported_type' AS intent_role
            FROM hop1_outward h1
            JOIN ast_edges e2 ON e2.source_symbol_id = h1.related_symbol_id
            JOIN target_sym ts ON TRUE
            JOIN ast_symbols s_hop1 ON s_hop1.symbol_id = h1.related_symbol_id
            JOIN code_files f_hop1 ON f_hop1.file_id = s_hop1.file_id
            WHERE h1.edge_kind IN ('USES_TYPE', 'INHERITS', 'IMPORTS')
              AND e2.domain_id = %s
              AND e2.target_symbol_id IS NOT NULL
              AND e2.edge_kind IN ('USES_TYPE', 'INHERITS', 'IMPORTS')
              AND f_hop1.repo_name = ts.repo_name
        ),
        outward_closure AS (
            SELECT * FROM hop1_outward
            UNION
            SELECT * FROM hop2_outward
        ),
        inward_caller_edges AS (
            SELECT e.edge_id, e.edge_kind, e.source_symbol_id
            FROM ast_edges e
            JOIN focus_family ff ON e.target_symbol_id = ff.symbol_id
            WHERE e.domain_id = %s
              AND (
                  e.edge_kind = 'CALLS'
                  OR (
                      e.is_cross_repo = TRUE
                      AND e.edge_kind IN (
                          'USES_TYPE', 'PROVISIONED_BY', 'DOCUMENTS_COMPONENT',
                          'PUBLISHES_EVENT', 'SUBSCRIBES_EVENT', 'READS_TABLE',
                          'WRITES_TABLE', 'RENDERS_TEMPLATE', 'IMPLEMENTS_SPEC',
                          'IMPLEMENTS_ISSUE', 'BINDS_SERVICE_ACCOUNT', 'ACCESSES_SECRET'
                      )
                  )
              )
              AND e.source_symbol_id NOT IN (SELECT symbol_id FROM focus_family)
            UNION
            SELECT e.edge_id, e.edge_kind, e.source_symbol_id
            FROM ast_edges e
            JOIN focus_family ff ON e.target_qualified_name = ff.qualified_name
            WHERE e.domain_id = %s
              AND (
                  e.edge_kind = 'CALLS'
                  OR (
                      e.is_cross_repo = TRUE
                      AND e.edge_kind IN (
                          'USES_TYPE', 'PROVISIONED_BY', 'DOCUMENTS_COMPONENT',
                          'PUBLISHES_EVENT', 'SUBSCRIBES_EVENT', 'READS_TABLE',
                          'WRITES_TABLE', 'RENDERS_TEMPLATE', 'IMPLEMENTS_SPEC',
                          'IMPLEMENTS_ISSUE', 'BINDS_SERVICE_ACCOUNT', 'ACCESSES_SECRET'
                      )
                  )
              )
              AND e.source_symbol_id NOT IN (SELECT symbol_id FROM focus_family)
        ),
        inward_callers AS (
            -- Direct callers ranked with diversity across caller files
            SELECT
                ranked.edge_id,
                ranked.edge_kind,
                ranked.source_symbol_id AS related_symbol_id,
                1 AS depth,
                'direct_caller' AS intent_role
            FROM (
                SELECT
                    e.edge_id,
                    e.edge_kind,
                    e.source_symbol_id,
                    ROW_NUMBER() OVER (
                        PARTITION BY s_caller.file_id
                        ORDER BY s_caller.start_line, e.edge_id
                    ) AS file_call_rank,
                    ROW_NUMBER() OVER (
                        ORDER BY (s_caller.file_id = ts.file_id) ASC,
                                 s_caller.file_id, s_caller.start_line
                    ) AS overall_rank
                FROM inward_caller_edges e
                CROSS JOIN target_sym ts
                JOIN ast_symbols s_caller ON s_caller.symbol_id = e.source_symbol_id
            ) ranked
            WHERE ranked.file_call_rank = 1 AND ranked.overall_rank <= %s
        ),
        test_fixtures AS (
            -- Direct TESTED_BY edge resolution (UNION of index-friendly branches)
            SELECT edge_id, edge_kind, related_symbol_id, 1 AS depth, 'test_fixture' AS intent_role
            FROM (
                SELECT e.edge_id, e.edge_kind, e.target_symbol_id AS related_symbol_id
                FROM ast_edges e
                JOIN focus_family ff ON e.source_symbol_id = ff.symbol_id
                WHERE e.domain_id = %s
                  AND e.edge_kind = 'TESTED_BY'
                  AND e.target_symbol_id IS NOT NULL
                UNION
                SELECT e.edge_id, e.edge_kind, e.source_symbol_id AS related_symbol_id
                FROM ast_edges e
                JOIN focus_family ff ON e.target_symbol_id = ff.symbol_id
                WHERE e.domain_id = %s
                  AND e.edge_kind = 'TESTED_BY'
                UNION
                SELECT e.edge_id, e.edge_kind, e.source_symbol_id AS related_symbol_id
                FROM ast_edges e
                JOIN focus_family ff ON e.target_qualified_name = ff.qualified_name
                WHERE e.domain_id = %s
                  AND e.edge_kind = 'TESTED_BY'
            ) tf
            LIMIT 2
        ),
        combined_neighborhood AS (
            SELECT * FROM outward_closure
            UNION ALL
            SELECT * FROM inward_callers
            UNION ALL
            SELECT * FROM test_fixtures
        )
        SELECT DISTINCT ON (cn.related_symbol_id, cn.intent_role)
            cn.intent_role,
            cn.edge_kind,
            cn.depth,
            s.symbol_id,
            s.domain_id,
            s.qualified_name,
            s.symbol_kind,
            s.start_line,
            s.end_line,
            s.signature_skeleton,
            s.contextual_summary,
            f.file_id,
            f.repo_name,
            f.file_path,
            f.git_commit_sha,
            f.git_blob_sha,
            f.language,
            f.line_count,
            f.raw_content
        FROM combined_neighborhood cn
        JOIN ast_symbols s ON s.symbol_id = cn.related_symbol_id
        JOIN code_files f ON f.file_id = s.file_id
        WHERE s.domain_id = %s
        ORDER BY cn.related_symbol_id, cn.intent_role, cn.depth ASC;
        """

    def traverse_1hop_neighborhood(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        focus_symbol_id: UUID | str,
        max_direct_callers: int = 3,
    ) -> list[ASTNeighborhoodRecord]:
        """Executes memory-bounded recursive CTE 1-hop AST closure directly in PostgreSQL 17.

        Replaces unbounded Python heap iteration with a parameterized CTE running under
        PostgreSQL 17 Forced RLS. Returns only relevant symbols in O(k) memory.
        """
        focus_uuid = UUID(str(focus_symbol_id))
        cte_sql = self.TRAVERSE_1HOP_NEIGHBORHOOD_CTE_SQL
        params = (
            domain_id,
            focus_uuid,
            domain_id,
            domain_id,
            domain_id,
            domain_id,
            max_direct_callers,
            domain_id,
            domain_id,
            domain_id,
            domain_id,
        )
        rows = conn.execute(cte_sql, params).fetchall()
        neighbors: list[ASTNeighborhoodRecord] = []
        for r in rows:
            intent_role = r[0]
            edge_kind = str(r[1])
            depth = int(str(r[2]))
            sym_id = UUID(str(r[3]))
            sym_domain = str(r[4])
            qual_name = str(r[5])
            sym_kind = str(r[6])
            start_l = int(str(r[7]))
            end_l = int(str(r[8]))
            sig = str(r[9])
            ctx_summary = str(r[10])
            file_id = UUID(str(r[11]))
            repo_name = str(r[12])
            file_path = str(r[13])
            commit_sha = str(r[14])
            blob_sha = str(r[15])
            lang = str(r[16])
            line_cnt = int(str(r[17]))
            raw_content = str(r[18])

            file_rec = CodeFileRecord(
                file_id=file_id,
                domain_id=sym_domain,
                repo_name=repo_name,
                file_path=file_path,
                git_commit_sha=commit_sha,
                git_blob_sha=blob_sha,
                language=lang,
                allowed_scopes=[],  # populated by RLS session
                raw_content=raw_content,
                line_count=line_cnt,
            )
            sym_rec = ASTSymbolRecord(
                symbol_id=sym_id,
                file_id=file_id,
                domain_id=sym_domain,
                qualified_name=qual_name,
                symbol_kind=sym_kind,
                start_line=start_l,
                end_line=end_l,
                signature_skeleton=sig,
                contextual_summary=ctx_summary,
                allowed_scopes=[],
                embedding=[],
            )
            neighbors.append(
                ASTNeighborhoodRecord(
                    intent_role=intent_role,  # type: ignore[arg-type]
                    edge_kind=edge_kind,
                    depth=depth,
                    symbol=sym_rec,
                    file=file_rec,
                )
            )
        return neighbors

    def insert_doc_chunk(
        self, conn: Connection[tuple[object, ...]], record: DocChunkRecord
    ) -> None:
        market_val = _serialize_market_scope(record.market_scope)
        doc_title = record.doc_title.replace("\x00", "")
        section_heading = record.section_heading.replace("\x00", "")
        contextual_header = record.contextual_header.replace("\x00", "")
        verbatim_content = record.verbatim_content.replace("\x00", "")
        stale_reason = (
            record.stale_reason.replace("\x00", "") if record.stale_reason is not None else None
        )
        vec_literal, halfvec_literal = _format_db_vector_and_halfvec(record.embedding)
        conn.execute(
            """
            INSERT INTO doc_chunks (
                chunk_id, tenant_id, domain_id, doc_id, doc_title, source_uri,
                market_scope, published_date, authority_weight,
                section_heading, contextual_header, verbatim_content,
                metadata_json, allowed_scopes, embedding,
                embedding_space_id, embedding_halfvec,
                source_type, version_str, is_stale, stale_reason
            )
            VALUES (
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s::jsonb, %s, %s::vector,
                %s, %s::halfvec,
                %s, %s, %s, %s
            )
            """,
            (
                record.chunk_id,
                record.tenant_id,
                record.domain_id,
                record.doc_id,
                doc_title,
                record.source_uri,
                market_val,
                record.published_date,
                record.authority_weight,
                section_heading,
                contextual_header,
                verbatim_content,
                json.dumps(record.metadata_json),
                record.allowed_scopes,
                vec_literal,
                record.embedding_space_id,
                halfvec_literal,
                record.source_type,
                record.version_str,
                record.is_stale,
                stale_reason,
            ),
        )

    def insert_doc_chunks_batch(
        self, conn: Connection[tuple[object, ...]], records: list[DocChunkRecord]
    ) -> int:
        if not records:
            return 0
        sql = """
            INSERT INTO doc_chunks (
                chunk_id, tenant_id, domain_id, doc_id, doc_title, source_uri,
                market_scope, published_date, authority_weight,
                section_heading, contextual_header, verbatim_content,
                metadata_json, allowed_scopes, embedding,
                embedding_space_id, embedding_halfvec,
                source_type, version_str, is_stale, stale_reason
            )
            VALUES (
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s::jsonb, %s, %s::vector,
                %s, %s::halfvec,
                %s, %s, %s, %s
            )
        """
        params_seq = []
        for record in records:
            vec_lit, hv_lit = _format_db_vector_and_halfvec(record.embedding)
            params_seq.append(
                (
                    record.chunk_id,
                    record.tenant_id,
                    record.domain_id,
                    record.doc_id,
                    record.doc_title.replace("\x00", ""),
                    record.source_uri,
                    _serialize_market_scope(record.market_scope),
                    record.published_date,
                    record.authority_weight,
                    record.section_heading.replace("\x00", ""),
                    record.contextual_header.replace("\x00", ""),
                    record.verbatim_content.replace("\x00", ""),
                    json.dumps(record.metadata_json),
                    record.allowed_scopes,
                    vec_lit,
                    record.embedding_space_id,
                    hv_lit,
                    record.source_type,
                    record.version_str,
                    record.is_stale,
                    record.stale_reason.replace("\x00", "")
                    if record.stale_reason is not None
                    else None,
                )
            )
        with conn.cursor() as cur:
            cur.executemany(sql, params_seq)
        return len(records)

    def list_visible_doc_chunks(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        market_scope: DocumentScope | str | None = None,
        include_stale: bool = False,
    ) -> list[DocChunkRecord]:
        query = """
            SELECT chunk_id, domain_id, doc_id, doc_title, source_uri,
                   market_scope, published_date, authority_weight,
                   section_heading, contextual_header, verbatim_content,
                   metadata_json, allowed_scopes,
                   COALESCE(embedding_halfvec::text, embedding::text) AS active_vec,
                   source_type, version_str, is_stale, stale_reason,
                   COALESCE(embedding_space_id, 'embeddinggemma-2-768') AS space_id
            FROM doc_chunks
            WHERE domain_id = %s
        """
        params: list[object] = [domain_id]
        if not include_stale:
            query += " AND NOT is_stale"
        serialized_market = _serialize_market_scope(market_scope)
        if serialized_market is not None:
            query += " AND market_scope = %s"
            params.append(serialized_market)
        query += " ORDER BY doc_id, chunk_id;"

        rows = conn.execute(query, tuple(params)).fetchall()
        results: list[DocChunkRecord] = []
        for r in rows:
            raw_emb = r[13]
            if isinstance(raw_emb, str):
                emb_floats = [float(x.strip()) for x in raw_emb.strip("[]").split(",") if x.strip()]
            elif hasattr(raw_emb, "tolist"):
                emb_floats = [float(x) for x in raw_emb.tolist()]
            elif isinstance(raw_emb, (list, tuple)):
                emb_floats = [float(x) for x in raw_emb]
            else:
                emb_floats = []

            meta = r[11]
            if isinstance(meta, str):
                meta_dict = json.loads(meta)
            elif isinstance(meta, dict):
                meta_dict = meta
            else:
                meta_dict = {}

            pub_date = r[6]
            if isinstance(pub_date, str):
                pub_date = date.fromisoformat(pub_date)
            elif not isinstance(pub_date, date):
                pub_date = None

            raw_scopes = r[12]
            scopes_list = (
                [str(s) for s in raw_scopes] if isinstance(raw_scopes, (list, tuple)) else []
            )
            results.append(
                DocChunkRecord(
                    chunk_id=UUID(str(r[0])),
                    domain_id=str(r[1]),
                    doc_id=str(r[2]),
                    doc_title=str(r[3]),
                    source_uri=str(r[4]),
                    market_scope=_parse_market_scope(r[5]),
                    published_date=pub_date,
                    authority_weight=float(str(r[7])),
                    section_heading=str(r[8]),
                    contextual_header=str(r[9]),
                    verbatim_content=str(r[10]),
                    metadata_json=meta_dict,
                    allowed_scopes=scopes_list,
                    embedding=emb_floats,
                    source_type=str(r[14]) if r[14] is not None else "enterprise_doc",
                    version_str=str(r[15]) if r[15] is not None else "1",
                    is_stale=bool(r[16]),
                    stale_reason=str(r[17]) if r[17] is not None else None,
                    embedding_space_id=str(r[18]) if r[18] is not None else "embeddinggemma-2-768",
                )
            )
        return results

    def search_doc_chunks_hybrid_rrf(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        query_text: str,
        query_embedding: list[float],
        market_scope: DocumentScope | str | None = None,
        top_k: int = 5,
        include_stale: bool = False,
        min_calibrated_score: float = 0.0,
    ) -> list[RankedSearchHit]:
        q_dim = len(query_embedding)
        if q_dim < 128 or q_dim > 4096:
            raise ValueError(f"Query embedding dimension must be in [128, 4096], got {q_dim}")
        vec_literal = "[" + ",".join(f"{v:.6f}" for v in query_embedding) + "]"
        has_null_hv = False
        if q_dim == 768:
            null_row = conn.execute(
                "SELECT 1 FROM doc_chunks "
                "WHERE domain_id = %s AND embedding_halfvec IS NULL LIMIT 1;",
                (domain_id,),
            ).fetchone()
            has_null_hv = null_row is not None
        use_halfvec = (q_dim != 768) or (not has_null_hv)
        if use_halfvec:
            vec_dist_expr = f"(embedding_halfvec::halfvec({q_dim}) <=> %s::halfvec({q_dim}))"
            outer_dist_expr = f"(d.embedding_halfvec::halfvec({q_dim}) <=> %s::halfvec({q_dim}))"
            vec_dim_clause = f"AND vector_dims(embedding_halfvec) = {q_dim}"
        else:
            vec_dist_expr = "(embedding <=> %s::vector)"
            outer_dist_expr = "(d.embedding <=> %s::vector)"
            vec_dim_clause = ""
        market_str = _serialize_market_scope(market_scope)
        candidate_pool = max(100, top_k * 10)
        penalty_rank = candidate_pool + 1
        # Never allow parenthesized glossary expansion suffixes or explicit disjunctive
        # OR-recall queries to poison conjunctive plainto_tsquery (AND) matching.
        stripped_q = query_text.strip()
        if stripped_q.endswith(")"):
            open_idx = stripped_q.rfind("(")
            if open_idx != -1 and 3 <= (len(stripped_q) - open_idx - 2) <= 200:
                raw_base_query = stripped_q[:open_idx].strip()
            else:
                raw_base_query = stripped_q
        else:
            raw_base_query = stripped_q
        is_disjunctive_query = " OR " in query_text
        base_tokens = _extract_substantive_tokens(raw_base_query or query_text)
        content_tokens = _extract_substantive_tokens(query_text)
        and_query = (
            (base_tokens[0] if is_disjunctive_query else " ".join(base_tokens))
            if base_tokens
            else raw_base_query or query_text.strip()
        )
        or_query = " OR ".join(content_tokens) if content_tokens else query_text.strip()
        trgm_query = (
            " ".join(base_tokens) if base_tokens else raw_base_query
        ) or query_text.strip()
        if market_str is None:
            inferred_markets, inferred_alias_str, market_tokens_set = (
                _infer_market_scopes_from_query(query_text, domain_id=domain_id)
            )
        else:
            inferred_markets, inferred_alias_str, market_tokens_set = [], "", set()
        non_market_tokens = [t for t in content_tokens if t not in market_tokens_set]
        non_market_or = " OR ".join(non_market_tokens) if non_market_tokens else or_query
        market_only_query = bool(inferred_markets and not non_market_tokens)
        title_fts_expr = (
            "setweight(to_tsvector('english', "
            "COALESCE(doc_title, '') || ' ' || COALESCE(section_heading, '')), 'A')"
        )
        meta_scope_expr = (
            "COALESCE(market_scope::text, '') || ' ' || "
            "REPLACE(COALESCE(market_scope::text, ''), '_', ' ') || ' ' || "
            "CASE WHEN sh.markets <> '{}'::text[] AND market_scope::text = ANY(sh.markets) "
            "THEN sh.aliases ELSE '' END"
        )
        trgm_small_expr = (
            "COALESCE(doc_title, '') || ' ' || COALESCE(section_heading, '') || ' ' || "
            f"{meta_scope_expr} || ' ' || LEFT(verbatim_content, 1200)"
        )
        trgm_title_expr = (
            "COALESCE(doc_title, '') || ' ' || COALESCE(section_heading, '') || ' ' || "
            f"{meta_scope_expr}"
        )
        fts_market_clause = (
            """
                          OR (
                              sh.markets <> '{}'::text[]
                              AND market_scope::text = ANY(sh.markets)
                          )
            """
            if market_only_query
            else ""
        )
        query = f"""
            WITH domain_size AS (
                SELECT COUNT(*) AS cnt
                FROM doc_chunks
                WHERE domain_id = %s
            ),
            scope_hint AS (
                SELECT %s::text[] AS markets, %s::text AS aliases, %s::text AS nm_or
            ),
            fts_ranked AS (
                SELECT chunk_id,
                       fts_score,
                       ROW_NUMBER() OVER (
                           ORDER BY fts_score DESC, chunk_id
                       ) AS fts_rank
                FROM (
                    SELECT chunk_id,
                           (
                               ts_rank_cd(fts_doc, plainto_tsquery('english', %s)) * 3.0
                               + ts_rank_cd(fts_doc, websearch_to_tsquery('english', %s))
                               + ts_rank_cd(
                                   {title_fts_expr},
                                   websearch_to_tsquery('english', sh.nm_or)
                               ) * 2.5
                               + CASE
                                   WHEN sh.markets <> '{{}}'::text[]
                                        AND market_scope::text = ANY(sh.markets)
                                        AND to_tsvector(
                                            'english',
                                            COALESCE(doc_title, '')
                                            || ' ' || COALESCE(section_heading, '')
                                        ) @@ websearch_to_tsquery('english', sh.nm_or)
                                   THEN 0.50
                                   WHEN sh.markets <> '{{}}'::text[]
                                        AND market_scope::text = ANY(sh.markets)
                                   THEN 0.25
                                   ELSE 0.0
                                 END
                           ) * authority_weight AS fts_score
                    FROM doc_chunks
                    CROSS JOIN scope_hint sh
                    WHERE domain_id = %s
                      AND (%s::text IS NULL OR market_scope = %s)
                      AND authority_weight > 0.0
                      AND (%s OR NOT is_stale)
                      AND (
                          fts_doc @@ websearch_to_tsquery('english', %s)
                          {fts_market_clause}
                      )
                      AND (
                          (SELECT cnt FROM domain_size) <= 50
                          OR (
                              LENGTH(TRIM(verbatim_content)) >= 45
                              AND LEFT(LTRIM(verbatim_content), 5) <> (CHR(37) || ' ---')
                              AND LEFT(LTRIM(verbatim_content), 11) <> '## See also'
                          )
                      )
                    ORDER BY fts_score DESC, chunk_id
                    LIMIT %s
                ) fts_sub
            ),
            vec_ranked AS (
                SELECT chunk_id,
                       vec_dist,
                       ROW_NUMBER() OVER (
                           ORDER BY vec_dist ASC, chunk_id
                       ) AS vec_rank
                FROM (
                    SELECT chunk_id, {vec_dist_expr} AS vec_dist
                    FROM doc_chunks
                    WHERE domain_id = %s
                      AND (%s::text IS NULL OR market_scope = %s)
                      AND authority_weight > 0.0
                      AND (%s OR NOT is_stale)
                      {vec_dim_clause}
                      AND (
                          (SELECT cnt FROM domain_size) <= 50
                          OR (
                              LENGTH(TRIM(verbatim_content)) >= 45
                              AND LEFT(LTRIM(verbatim_content), 5) <> (CHR(37) || ' ---')
                              AND LEFT(LTRIM(verbatim_content), 11) <> '## See also'
                          )
                      )
                    ORDER BY {vec_dist_expr} ASC
                    LIMIT %s
                ) vec_sub
            ),
            trgm_ranked AS (
                SELECT chunk_id,
                       trgm_score,
                       ROW_NUMBER() OVER (
                           ORDER BY trgm_score DESC, chunk_id
                       ) AS trgm_rank
                FROM (
                    SELECT chunk_id,
                           (
                               word_similarity(
                                   %s,
                                   CASE
                                       WHEN (SELECT cnt FROM domain_size) <= 50
                                       THEN {trgm_small_expr}
                                       ELSE {trgm_title_expr}
                                   END
                               ) * authority_weight
                           ) AS trgm_score
                    FROM doc_chunks
                    CROSS JOIN scope_hint sh
                    WHERE domain_id = %s
                      AND (%s::text IS NULL OR market_scope = %s)
                      AND authority_weight > 0.0
                      AND (%s OR NOT is_stale)
                      AND word_similarity(
                              %s,
                              CASE
                                  WHEN (SELECT cnt FROM domain_size) <= 50
                                  THEN {trgm_small_expr}
                                  ELSE {trgm_title_expr}
                              END
                          ) >= 0.20
                      AND (
                          (SELECT cnt FROM domain_size) <= 50
                          OR (
                              LENGTH(TRIM(verbatim_content)) >= 45
                              AND LEFT(LTRIM(verbatim_content), 5) <> (CHR(37) || ' ---')
                              AND LEFT(LTRIM(verbatim_content), 11) <> '## See also'
                          )
                      )
                    ORDER BY trgm_score DESC, chunk_id
                    LIMIT %s
                ) trgm_sub
            ),
            candidate_ids AS (
                SELECT chunk_id FROM fts_ranked
                UNION
                SELECT chunk_id FROM vec_ranked
                UNION
                SELECT chunk_id FROM trgm_ranked
            )
            SELECT d.chunk_id, d.doc_title, d.section_heading,
                   COALESCE(f.fts_rank, %s) AS fts_rank,
                   COALESCE(v.vec_rank, %s) AS vec_rank,
                   COALESCE(t.trgm_rank, %s) AS trgm_rank,
                   d.authority_weight,
                   COALESCE(f.fts_score, 0.0) AS fts_score,
                   CASE
                       WHEN COALESCE(v.vec_dist, {outer_dist_expr}) = 'NaN'::float8
                       THEN 0.0
                       ELSE GREATEST(
                           0.0, 1.0 - COALESCE(v.vec_dist, {outer_dist_expr})
                       )
                   END AS vec_cos,
                   COALESCE(t.trgm_score, 0.0) AS trgm_score,
                   LEFT(d.verbatim_content, 1800) AS content_preview,
                   (SELECT cnt FROM domain_size) AS domain_cnt,
                   d.market_scope::text AS market_scope_val
            FROM candidate_ids c
            JOIN doc_chunks d ON d.chunk_id = c.chunk_id
            LEFT JOIN fts_ranked f ON d.chunk_id = f.chunk_id
            LEFT JOIN vec_ranked v ON d.chunk_id = v.chunk_id
            LEFT JOIN trgm_ranked t ON d.chunk_id = t.chunk_id
            WHERE d.domain_id = %s
              AND (%s::text IS NULL OR d.market_scope = %s)
              AND d.authority_weight > 0.0
              AND (%s OR NOT d.is_stale);
        """
        params = (
            domain_id,
            inferred_markets,
            inferred_alias_str,
            non_market_or,
            and_query,
            or_query,
            domain_id,
            market_str,
            market_str,
            include_stale,
            or_query,
            candidate_pool,
            vec_literal,
            domain_id,
            market_str,
            market_str,
            include_stale,
            vec_literal,
            candidate_pool,
            trgm_query,
            domain_id,
            market_str,
            market_str,
            include_stale,
            trgm_query,
            candidate_pool,
            penalty_rank,
            penalty_rank,
            penalty_rank,
            vec_literal,
            vec_literal,
            domain_id,
            market_str,
            market_str,
            include_stale,
        )
        rows = conn.execute(query, params).fetchall()
        scored_hits: list[tuple[float, float, float, CalibratedRankedSearchHit]] = []
        for r in rows:
            chunk_id = UUID(str(r[0]))
            doc_title = str(r[1])
            section_heading = str(r[2])
            fts_rank = int(str(r[3]))
            vec_rank = int(str(r[4]))
            trgm_rank = int(str(r[5]))
            auth_weight = float(str(r[6]))
            fts_score = float(str(r[7]))
            vec_cos = float(str(r[8]))
            trgm_score = float(str(r[9]))
            content_preview = str(r[10] or "")
            domain_cnt = int(str(r[11]))
            row_market_scope = str(r[12]) if r[12] is not None else None
            is_scope_mismatch = bool(
                market_str is None
                and inferred_markets
                and row_market_scope
                and row_market_scope not in inferred_markets
                and _market_scope_aliases(row_market_scope, domain_id=domain_id)
            )

            base_rrf = compute_normalized_rrf(
                [trgm_rank, fts_rank, vec_rank], num_modalities=3, k=60
            )
            weighted_score = _apply_authority_score(base_rrf, auth_weight)
            calibrated_score, lexical_cov, is_boiler = _compute_calibrated_relevance(
                query_text=query_text,
                title=doc_title,
                section=section_heading,
                content=content_preview,
                cosine_similarity=vec_cos,
                bm25_score=fts_score,
                trigram_score=trgm_score,
                raw_rrf_score=base_rrf,
                authority_weight=auth_weight,
                market_scope=row_market_scope,
                domain_id=domain_id,
                scope_mismatch=is_scope_mismatch,
            )

            if is_boiler and domain_cnt > 50:
                continue
            if domain_cnt > 50:
                passes_floor = (
                    (vec_cos >= 0.46 and (lexical_cov > 0.0 or not content_tokens))
                    or (
                        lexical_cov >= 0.34
                        and (vec_cos >= 0.18 or fts_score > 0.0 or trgm_score >= 0.18)
                    )
                    or (fts_score >= 0.10 and lexical_cov > 0.0)
                )
                if not passes_floor:
                    continue
            if min_calibrated_score > 0.0 and calibrated_score < min_calibrated_score:
                continue

            hit = CalibratedRankedSearchHit(
                item_id=chunk_id,
                qualified_name_or_title=f"{doc_title} > {section_heading}",
                normalized_rrf_score=weighted_score,
                trigram_rank=trgm_rank,
                bm25_rank=fts_rank,
                vector_rank=vec_rank,
                cosine_similarity=round(vec_cos, 6),
                bm25_score=round(fts_score, 6),
                trigram_score=round(trgm_score, 6),
                raw_rrf_score=round(base_rrf, 6),
                calibrated_score=calibrated_score,
                domain_id=domain_id,
                lexical_coverage=lexical_cov,
                is_boilerplate=is_boiler,
            )
            rank_key = calibrated_score if domain_cnt > 50 else weighted_score
            scored_hits.append((rank_key, weighted_score, auth_weight, hit))
        scored_hits.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        return [item[3] for item in scored_hits[:top_k]]

    def search_symbols_hybrid_rrf(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        query_text: str,
        query_embedding: list[float],
        top_k: int = 5,
        min_calibrated_score: float = 0.0,
    ) -> list[RankedSearchHit]:
        """Executes 3-way Hybrid RRF over PostgreSQL ast_symbols (GIN trigram + FTS + HNSW)."""
        q_dim = len(query_embedding)
        if q_dim < 128 or q_dim > 4096:
            raise ValueError(f"Query embedding dimension must be in [128, 4096], got {q_dim}")
        vec_literal = "[" + ",".join(f"{v:.6f}" for v in query_embedding) + "]"
        has_null_hv = False
        if q_dim == 768:
            null_row = conn.execute(
                "SELECT 1 FROM ast_symbols "
                "WHERE domain_id = %s AND embedding_halfvec IS NULL LIMIT 1;",
                (domain_id,),
            ).fetchone()
            has_null_hv = null_row is not None
        use_halfvec = (q_dim != 768) or (not has_null_hv)
        if use_halfvec:
            vec_dist_expr = f"(embedding_halfvec::halfvec({q_dim}) <=> %s::halfvec({q_dim}))"
            outer_dist_expr = f"(s.embedding_halfvec::halfvec({q_dim}) <=> %s::halfvec({q_dim}))"
            vec_dim_clause = f"AND vector_dims(embedding_halfvec) = {q_dim}"
        else:
            vec_dist_expr = "(embedding <=> %s::vector)"
            outer_dist_expr = "(s.embedding <=> %s::vector)"
            vec_dim_clause = ""
        candidate_pool = max(100, top_k * 10)
        penalty_rank = candidate_pool + 1
        content_tokens = _extract_substantive_tokens(query_text)
        fts_query = " OR ".join(content_tokens) if content_tokens else query_text.strip()
        query = f"""
            WITH domain_size AS (
                SELECT COUNT(*) AS cnt
                FROM ast_symbols
                WHERE domain_id = %s
            ),
            fts_ranked AS (
                SELECT symbol_id,
                       fts_score,
                       ROW_NUMBER() OVER (
                           ORDER BY fts_score DESC, symbol_id
                       ) AS fts_rank
                FROM (
                    SELECT symbol_id,
                           (
                               ts_rank_cd(fts_doc, plainto_tsquery('english', %s)) * 2.0
                               + ts_rank_cd(fts_doc, websearch_to_tsquery('english', %s))
                           ) AS fts_score
                    FROM ast_symbols
                    WHERE domain_id = %s
                      AND fts_doc @@ websearch_to_tsquery('english', %s)
                    ORDER BY fts_score DESC, symbol_id
                    LIMIT %s
                ) fts_sub
            ),
            vec_ranked AS (
                SELECT symbol_id,
                       vec_dist,
                       ROW_NUMBER() OVER (
                           ORDER BY vec_dist ASC, symbol_id
                       ) AS vec_rank
                FROM (
                    SELECT symbol_id, {vec_dist_expr} AS vec_dist
                    FROM ast_symbols
                    WHERE domain_id = %s
                      {vec_dim_clause}
                    ORDER BY {vec_dist_expr} ASC
                    LIMIT %s
                ) vec_sub
            ),
            trgm_ranked AS (
                SELECT symbol_id,
                       trgm_score,
                       ROW_NUMBER() OVER (
                           ORDER BY trgm_score DESC, symbol_id
                       ) AS trgm_rank
                FROM (
                    SELECT symbol_id,
                           GREATEST(
                               similarity(qualified_name, %s),
                               similarity(signature_skeleton, %s)
                           ) AS trgm_score
                    FROM ast_symbols
                    WHERE domain_id = %s
                    ORDER BY trgm_score DESC, symbol_id
                    LIMIT %s
                ) trgm_sub
            ),
            candidate_ids AS (
                SELECT symbol_id FROM fts_ranked
                UNION
                SELECT symbol_id FROM vec_ranked
                UNION
                SELECT symbol_id FROM trgm_ranked
            )
            SELECT s.symbol_id, s.qualified_name,
                   COALESCE(f.fts_rank, %s) AS fts_rank,
                   COALESCE(v.vec_rank, %s) AS vec_rank,
                   COALESCE(t.trgm_rank, %s) AS trgm_rank,
                   COALESCE(f.fts_score, 0.0) AS fts_score,
                   CASE
                       WHEN COALESCE(v.vec_dist, {outer_dist_expr}) = 'NaN'::float8
                       THEN 0.0
                       ELSE GREATEST(
                           0.0, 1.0 - COALESCE(v.vec_dist, {outer_dist_expr})
                       )
                   END AS vec_cos,
                   COALESCE(t.trgm_score, 0.0) AS trgm_score,
                   s.signature_skeleton,
                   LEFT(COALESCE(s.contextual_summary, ''), 600) AS summary_preview,
                   (SELECT cnt FROM domain_size) AS domain_cnt,
                   COALESCE(cf.ranking_weight, 1.0) AS ranking_weight
            FROM candidate_ids c
            JOIN ast_symbols s ON s.symbol_id = c.symbol_id
            LEFT JOIN code_files cf ON cf.file_id = s.file_id
            LEFT JOIN fts_ranked f ON s.symbol_id = f.symbol_id
            LEFT JOIN vec_ranked v ON s.symbol_id = v.symbol_id
            LEFT JOIN trgm_ranked t ON s.symbol_id = t.symbol_id
            WHERE s.domain_id = %s;
        """
        params = (
            domain_id,
            query_text,
            fts_query,
            domain_id,
            fts_query,
            candidate_pool,
            vec_literal,
            domain_id,
            vec_literal,
            candidate_pool,
            query_text,
            query_text,
            domain_id,
            candidate_pool,
            penalty_rank,
            penalty_rank,
            penalty_rank,
            vec_literal,
            vec_literal,
            domain_id,
        )
        rows = conn.execute(query, params).fetchall()
        hits: list[CalibratedRankedSearchHit] = []
        for r in rows:
            sym_id = UUID(str(r[0]))
            qual_name = str(r[1])
            fts_rank = int(str(r[2]))
            vec_rank = int(str(r[3]))
            trgm_rank = int(str(r[4]))
            fts_score = float(str(r[5]))
            vec_cos = float(str(r[6]))
            trgm_score = float(str(r[7]))
            sig_skeleton = str(r[8] or "")
            summary_preview = str(r[9] or "")
            domain_cnt = int(str(r[10]))
            ranking_weight = float(str(r[11]))

            norm_rrf = compute_normalized_rrf(
                [trgm_rank, fts_rank, vec_rank], num_modalities=3, k=60
            )
            calibrated_score, lexical_cov, _ = _compute_calibrated_relevance(
                query_text=query_text,
                title=qual_name,
                section=sig_skeleton,
                content=summary_preview,
                cosine_similarity=vec_cos,
                bm25_score=fts_score,
                trigram_score=trgm_score,
                raw_rrf_score=norm_rrf,
                authority_weight=1.0,
                is_code_symbol=True,
            )
            if not math.isclose(ranking_weight, 1.0, rel_tol=0.0, abs_tol=1e-9):
                norm_rrf = _apply_authority_score(norm_rrf, ranking_weight)
                calibrated_score = round(min(1.0, max(0.0, calibrated_score * ranking_weight)), 6)
            if domain_cnt > 50:
                passes_floor = (
                    (vec_cos >= 0.48 and (lexical_cov > 0.0 or trgm_score >= 0.25))
                    or (
                        lexical_cov >= 0.34
                        and (vec_cos >= 0.18 or fts_score > 0.0 or trgm_score >= 0.18)
                    )
                    or (fts_score >= 0.10 and lexical_cov > 0.0)
                    or (trgm_score >= 0.32 and lexical_cov > 0.0)
                )
                if not passes_floor:
                    continue
            if min_calibrated_score > 0.0 and calibrated_score < min_calibrated_score:
                continue

            hits.append(
                CalibratedRankedSearchHit(
                    item_id=sym_id,
                    qualified_name_or_title=qual_name,
                    normalized_rrf_score=norm_rrf,
                    trigram_rank=trgm_rank,
                    bm25_rank=fts_rank,
                    vector_rank=vec_rank,
                    cosine_similarity=round(vec_cos, 6),
                    bm25_score=round(fts_score, 6),
                    trigram_score=round(trgm_score, 6),
                    raw_rrf_score=round(norm_rrf, 6),
                    calibrated_score=calibrated_score,
                    domain_id=domain_id,
                    lexical_coverage=lexical_cov,
                )
            )
        hits.sort(
            key=lambda h: (h.calibrated_score, h.normalized_rrf_score),
            reverse=True,
        )
        return list(hits[:top_k])

    def record_execution_telemetry(
        self, conn: Connection[tuple[object, ...]], record: ExecutionTelemetryRecord
    ) -> None:
        mode_val = str(record.execution_mode)
        missing_json = json.dumps(record.missing_context_requests)
        conn.execute(
            """
            INSERT INTO execution_telemetry (
                trace_id, tenant_id, domain_id, principal_id, principal_role,
                execution_mode, local_tokens_read, context_pack_tokens_sent,
                paid_tokens_saved, estimated_usd_saved, preflight_passed_first_try,
                zero_touch_sufficient, missing_context_requests, latency_ms,
                created_at
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, COALESCE(%s, NOW())
            )
            ON CONFLICT (trace_id) DO UPDATE SET
                local_tokens_read = EXCLUDED.local_tokens_read,
                context_pack_tokens_sent = EXCLUDED.context_pack_tokens_sent,
                paid_tokens_saved = EXCLUDED.paid_tokens_saved,
                estimated_usd_saved = EXCLUDED.estimated_usd_saved,
                preflight_passed_first_try = EXCLUDED.preflight_passed_first_try,
                zero_touch_sufficient = EXCLUDED.zero_touch_sufficient,
                missing_context_requests = EXCLUDED.missing_context_requests,
                latency_ms = EXCLUDED.latency_ms;
            """,
            (
                record.trace_id,
                record.tenant_id,
                record.domain_id,
                record.principal_id,
                record.principal_role,
                mode_val,
                record.local_tokens_read,
                record.context_pack_tokens_sent,
                record.paid_tokens_saved,
                record.estimated_usd_saved,
                record.preflight_passed_first_try,
                record.zero_touch_sufficient,
                missing_json,
                record.latency_ms,
                record.created_at,
            ),
        )

    def get_execution_telemetry(
        self, conn: Connection[tuple[object, ...]], trace_id: UUID
    ) -> ExecutionTelemetryRecord | None:
        row = conn.execute(
            """
            SELECT
                trace_id, domain_id, principal_id, principal_role,
                execution_mode, local_tokens_read, context_pack_tokens_sent,
                paid_tokens_saved, estimated_usd_saved, preflight_passed_first_try,
                zero_touch_sufficient, missing_context_requests, latency_ms,
                created_at, tenant_id
            FROM execution_telemetry
            WHERE trace_id = %s;
            """,
            (trace_id,),
        ).fetchone()
        if row is None:
            return None
        raw_missing = row[11]
        if isinstance(raw_missing, str):
            missing_requests = json.loads(raw_missing)
        elif isinstance(raw_missing, list):
            missing_requests = raw_missing
        else:
            missing_requests = []
        zero_touch_val: bool | None = bool(row[10]) if row[10] is not None else None
        created_at_val: datetime | None = row[13] if isinstance(row[13], datetime) else None
        return ExecutionTelemetryRecord(
            trace_id=UUID(str(row[0])),
            domain_id=str(row[1]),
            principal_id=str(row[2]),
            principal_role=str(row[3]),
            execution_mode=str(row[4]),  # type: ignore[arg-type]
            local_tokens_read=int(str(row[5])),
            context_pack_tokens_sent=int(str(row[6])),
            paid_tokens_saved=int(str(row[7])),
            estimated_usd_saved=float(str(row[8])),
            preflight_passed_first_try=bool(row[9]),
            zero_touch_sufficient=zero_touch_val,
            missing_context_requests=missing_requests,
            latency_ms=int(str(row[12])),
            created_at=created_at_val,
            tenant_id=UUID(str(row[14])),
        )

    def get_principal_paid_tokens_consumed(
        self,
        conn: Connection[tuple[object, ...]],
        principal_id: str,
        since_hours: int = 24,
    ) -> int:
        row = conn.execute(
            """
            SELECT COALESCE(SUM(context_pack_tokens_sent), 0)
            FROM execution_telemetry
            WHERE principal_id = %s
              AND execution_mode = 'gemini_handoff'
              AND created_at >= NOW() - (%s || ' hours')::interval;
            """,
            (principal_id, str(since_hours)),
        ).fetchone()
        return int(str(row[0])) if row and row[0] is not None else 0

    def search_visible_ast_symbols(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        query_text: str,
        limit: int = 50,
    ) -> list[ASTSymbolRecord]:
        """Searches visible AST symbols under RLS with SQL pushdown and LIMIT."""
        import re

        words = [
            w.lower()
            for w in re.findall(r"[A-Za-z0-9_]{3,}", query_text)
            if w.lower() not in {"the", "and", "for", "with", "how", "what", "where", "from"}
        ]
        safe_limit = max(1, min(limit, 500))
        if not words:
            rows = conn.execute(
                """
                SELECT symbol_id, file_id, domain_id, qualified_name, symbol_kind,
                       start_line, end_line, signature_skeleton, contextual_summary,
                       allowed_scopes
                FROM ast_symbols
                WHERE domain_id = %s
                ORDER BY qualified_name
                LIMIT %s;
                """,
                (domain_id, safe_limit),
            ).fetchall()
        else:
            patterns = [f"%{w}%" for w in words[:8]]
            rows = conn.execute(
                """
                SELECT symbol_id, file_id, domain_id, qualified_name, symbol_kind,
                       start_line, end_line, signature_skeleton, contextual_summary,
                       allowed_scopes
                FROM ast_symbols
                WHERE domain_id = %s
                  AND (
                      qualified_name ILIKE ANY(%s)
                      OR signature_skeleton ILIKE ANY(%s)
                      OR contextual_summary ILIKE ANY(%s)
                  )
                ORDER BY qualified_name
                LIMIT %s;
                """,
                (domain_id, patterns, patterns, patterns, safe_limit),
            ).fetchall()
        return [
            ASTSymbolRecord(
                symbol_id=UUID(str(r[0])),
                file_id=UUID(str(r[1])),
                domain_id=str(r[2]),
                qualified_name=str(r[3]),
                symbol_kind=str(r[4]),
                start_line=int(str(r[5])),
                end_line=int(str(r[6])),
                signature_skeleton=str(r[7]),
                contextual_summary=str(r[8]),
                allowed_scopes=[str(s) for s in r[9]] if isinstance(r[9], list) else [],
                embedding=[],
            )
            for r in rows
        ]

    def get_code_files_by_ids(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        file_ids: list[UUID | str],
    ) -> list[CodeFileRecord]:
        """Fetches visible code files by UUID under RLS without full-table scan."""
        clean_ids: list[str] = []
        for fid in file_ids:
            try:
                clean_ids.append(str(UUID(str(fid))))
            except ValueError as exc:
                logger.debug(
                    "Ignoring malformed file_id %r (%s): %s",
                    fid,
                    type(exc).__name__,
                    exc,
                )
                continue
        if not clean_ids:
            return []
        rows = conn.execute(
            """
            SELECT file_id, domain_id, repo_name, file_path, git_commit_sha,
                   git_blob_sha, language, allowed_scopes, raw_content, line_count,
                   tenant_id, COALESCE(ranking_weight, 1.0),
                   COALESCE(is_generated, FALSE), COALESCE(is_vendored, FALSE)
            FROM code_files
            WHERE domain_id = %s AND file_id = ANY(%s::uuid[]);
            """,
            (domain_id, clean_ids),
        ).fetchall()
        return [
            CodeFileRecord(
                file_id=UUID(str(r[0])),
                domain_id=str(r[1]),
                repo_name=str(r[2]),
                file_path=str(r[3]),
                git_commit_sha=str(r[4]),
                git_blob_sha=str(r[5]),
                language=str(r[6]),
                allowed_scopes=[str(s) for s in r[7]] if isinstance(r[7], list) else [],
                raw_content=str(r[8]),
                line_count=int(str(r[9])),
                tenant_id=UUID(str(r[10])),
                ranking_weight=float(str(r[11])),
                is_generated=bool(r[12]),
                is_vendored=bool(r[13]),
            )
            for r in rows
        ]

    def get_code_files_by_paths(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        file_paths: list[str],
    ) -> list[CodeFileRecord]:
        """Fetches visible code files by file_path under RLS without full-table scan."""
        clean_paths = [p.strip() for p in file_paths if p and p.strip()]
        if not clean_paths:
            return []
        rows = conn.execute(
            """
            SELECT file_id, domain_id, repo_name, file_path, git_commit_sha,
                   git_blob_sha, language, allowed_scopes, raw_content, line_count,
                   tenant_id, COALESCE(ranking_weight, 1.0),
                   COALESCE(is_generated, FALSE), COALESCE(is_vendored, FALSE)
            FROM code_files
            WHERE domain_id = %s AND file_path = ANY(%s::text[]);
            """,
            (domain_id, clean_paths),
        ).fetchall()
        return [
            CodeFileRecord(
                file_id=UUID(str(r[0])),
                domain_id=str(r[1]),
                repo_name=str(r[2]),
                file_path=str(r[3]),
                git_commit_sha=str(r[4]),
                git_blob_sha=str(r[5]),
                language=str(r[6]),
                allowed_scopes=[str(s) for s in r[7]] if isinstance(r[7], list) else [],
                raw_content=str(r[8]),
                line_count=int(str(r[9])),
                tenant_id=UUID(str(r[10])),
                ranking_weight=float(str(r[11])),
                is_generated=bool(r[12]),
                is_vendored=bool(r[13]),
            )
            for r in rows
        ]

    def list_visible_code_file_records(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        *,
        repo_name: str | None = None,
        limit: int = 2000,
    ) -> list[CodeFileRecord]:
        """Lists full visible CodeFileRecord rows under Forced RLS."""
        safe_limit = max(1, min(limit, 10000))
        if repo_name:
            rows = conn.execute(
                """
                SELECT file_id, domain_id, repo_name, file_path, git_commit_sha,
                       git_blob_sha, language, allowed_scopes, raw_content, line_count,
                       tenant_id, COALESCE(ranking_weight, 1.0),
                       COALESCE(is_generated, FALSE), COALESCE(is_vendored, FALSE)
                FROM code_files
                WHERE domain_id = %s AND repo_name = %s
                ORDER BY file_path ASC
                LIMIT %s;
                """,
                (domain_id, repo_name, safe_limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT file_id, domain_id, repo_name, file_path, git_commit_sha,
                       git_blob_sha, language, allowed_scopes, raw_content, line_count,
                       tenant_id, COALESCE(ranking_weight, 1.0),
                       COALESCE(is_generated, FALSE), COALESCE(is_vendored, FALSE)
                FROM code_files
                WHERE domain_id = %s
                ORDER BY repo_name ASC, file_path ASC
                LIMIT %s;
                """,
                (domain_id, safe_limit),
            ).fetchall()
        return [
            CodeFileRecord(
                file_id=UUID(str(r[0])),
                domain_id=str(r[1]),
                repo_name=str(r[2]),
                file_path=str(r[3]),
                git_commit_sha=str(r[4]),
                git_blob_sha=str(r[5]),
                language=str(r[6]),
                allowed_scopes=[str(s) for s in r[7]] if isinstance(r[7], list) else [],
                raw_content=str(r[8]),
                line_count=int(str(r[9])),
                tenant_id=UUID(str(r[10])),
                ranking_weight=float(str(r[11])),
                is_generated=bool(r[12]),
                is_vendored=bool(r[13]),
            )
            for r in rows
        ]

    def list_visible_ast_symbol_records(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        *,
        limit: int = 5000,
    ) -> list[ASTSymbolRecord]:
        """Lists visible ASTSymbolRecord rows under Forced RLS."""
        safe_limit = max(1, min(limit, 20000))
        rows = conn.execute(
            """
            SELECT symbol_id, file_id, domain_id, qualified_name, symbol_kind,
                   start_line, end_line, signature_skeleton, contextual_summary,
                   allowed_scopes, tenant_id
            FROM ast_symbols
            WHERE domain_id = %s
            ORDER BY qualified_name ASC
            LIMIT %s;
            """,
            (domain_id, safe_limit),
        ).fetchall()
        return [
            ASTSymbolRecord(
                symbol_id=UUID(str(r[0])),
                file_id=UUID(str(r[1])),
                domain_id=str(r[2]),
                qualified_name=str(r[3]),
                symbol_kind=str(r[4]),
                start_line=int(str(r[5])),
                end_line=int(str(r[6])),
                signature_skeleton=str(r[7]),
                contextual_summary=str(r[8]),
                allowed_scopes=[str(s) for s in r[9]] if isinstance(r[9], list) else [],
                embedding=[],
                tenant_id=UUID(str(r[10])),
            )
            for r in rows
        ]

    def list_visible_ast_edge_records(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        *,
        limit: int = 10000,
    ) -> list[ASTEdgeRecord]:
        """Lists visible ASTEdgeRecord rows under Forced RLS."""
        safe_limit = max(1, min(limit, 50000))
        rows = conn.execute(
            """
            SELECT edge_id, domain_id, source_symbol_id, target_qualified_name,
                   target_symbol_id, edge_kind, allowed_scopes,
                   target_repo_qualified_id, COALESCE(is_cross_repo, FALSE), tenant_id
            FROM ast_edges
            WHERE domain_id = %s
            ORDER BY edge_kind ASC, target_qualified_name ASC
            LIMIT %s;
            """,
            (domain_id, safe_limit),
        ).fetchall()
        return [
            ASTEdgeRecord(
                edge_id=UUID(str(r[0])),
                domain_id=str(r[1]),
                source_symbol_id=UUID(str(r[2])),
                target_qualified_name=str(r[3]),
                target_symbol_id=UUID(str(r[4])) if r[4] is not None else None,
                edge_kind=str(r[5]),
                allowed_scopes=[str(s) for s in r[6]] if isinstance(r[6], list) else [],
                target_repo_qualified_id=str(r[7]) if r[7] is not None else None,
                is_cross_repo=bool(r[8]),
                tenant_id=UUID(str(r[9])),
            )
            for r in rows
        ]

    # -------------------------------------------------------------------------
    # Copy-on-Write Branch & MR/PR Delta Overlays + Ownership Graph (M2-01)
    # -------------------------------------------------------------------------
    @staticmethod
    def _row_to_ref_snapshot(row: tuple[object, ...]) -> RefSnapshotRecord:
        meta_raw = row[13]
        meta = meta_raw if isinstance(meta_raw, dict) else json.loads(str(meta_raw or "{}"))
        return RefSnapshotRecord(
            snapshot_id=UUID(str(row[0])),
            tenant_id=UUID(str(row[1])),
            domain_id=str(row[2]),
            repo_id=UUID(str(row[3])) if row[3] is not None else None,
            repo_name=str(row[4]),
            ref_name=str(row[5]),
            ref_kind=row[6],  # type: ignore[arg-type]
            mr_iid=str(row[7]) if row[7] is not None else None,
            base_snapshot_id=UUID(str(row[8])) if row[8] is not None else None,
            base_commit_sha=str(row[9]),
            head_commit_sha=str(row[10]),
            state=row[11],  # type: ignore[arg-type]
            allowed_scopes=[str(s) for s in row[12]] if isinstance(row[12], list) else [],
            metadata_json=meta,
            created_at=_as_opt_dt(row[14]),
            updated_at=_as_opt_dt(row[15]),
        )

    @staticmethod
    def _row_to_overlay_delta(row: tuple[object, ...]) -> OverlayFileDeltaRecord:
        syms_raw = row[15]
        edges_raw = row[16]
        syms = syms_raw if isinstance(syms_raw, list) else json.loads(str(syms_raw or "[]"))
        edges = edges_raw if isinstance(edges_raw, list) else json.loads(str(edges_raw or "[]"))
        return OverlayFileDeltaRecord(
            delta_id=UUID(str(row[0])),
            tenant_id=UUID(str(row[1])),
            snapshot_id=UUID(str(row[2])),
            domain_id=str(row[3]),
            repo_name=str(row[4]),
            file_path=str(row[5]),
            previous_file_path=str(row[6]) if row[6] is not None else None,
            change_type=row[7],  # type: ignore[arg-type]
            base_blob_sha=str(row[8]) if row[8] is not None else None,
            overlay_blob_sha=str(row[9]) if row[9] is not None else None,
            language=str(row[10]),
            raw_content=str(row[11]) if row[11] is not None else None,
            line_count=int(str(row[12])),
            ranking_weight=float(str(row[13])),
            is_generated=bool(row[14]),
            is_vendored=bool(row[17]),
            symbols_json=syms,
            edges_json=edges,
            allowed_scopes=[str(s) for s in row[18]] if isinstance(row[18], list) else [],
            indexed_at=_as_opt_dt(row[19]),
        )

    def upsert_ref_snapshot(
        self,
        conn: Connection[tuple[object, ...]],
        record: RefSnapshotRecord,
    ) -> RefSnapshotRecord:
        row = conn.execute(
            """
            INSERT INTO ref_snapshots (
                snapshot_id, tenant_id, domain_id, repo_id, repo_name, ref_name,
                ref_kind, mr_iid, base_snapshot_id, base_commit_sha, head_commit_sha,
                state, allowed_scopes, metadata_json, created_at, updated_at
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb,
                COALESCE(%s, NOW()), COALESCE(%s, NOW())
            )
            ON CONFLICT (tenant_id, domain_id, repo_name, ref_name) DO UPDATE SET
                repo_id = EXCLUDED.repo_id,
                ref_kind = EXCLUDED.ref_kind,
                mr_iid = EXCLUDED.mr_iid,
                base_snapshot_id = EXCLUDED.base_snapshot_id,
                base_commit_sha = EXCLUDED.base_commit_sha,
                head_commit_sha = EXCLUDED.head_commit_sha,
                state = EXCLUDED.state,
                allowed_scopes = EXCLUDED.allowed_scopes,
                metadata_json = EXCLUDED.metadata_json,
                updated_at = NOW()
            RETURNING snapshot_id, tenant_id, domain_id, repo_id, repo_name, ref_name,
                      ref_kind, mr_iid, base_snapshot_id, base_commit_sha, head_commit_sha,
                      state, allowed_scopes, metadata_json, created_at, updated_at;
            """,
            (
                record.snapshot_id,
                record.tenant_id,
                record.domain_id,
                record.repo_id,
                record.repo_name,
                record.ref_name,
                record.ref_kind,
                record.mr_iid,
                record.base_snapshot_id,
                record.base_commit_sha,
                record.head_commit_sha,
                record.state,
                record.allowed_scopes,
                json.dumps(record.metadata_json),
                record.created_at,
                record.updated_at,
            ),
        ).fetchone()
        assert row is not None
        return self._row_to_ref_snapshot(row)

    def get_ref_snapshot(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        *,
        snapshot_id: UUID | None = None,
        repo_name: str | None = None,
        ref_name: str | None = None,
        mr_iid: str | None = None,
    ) -> RefSnapshotRecord | None:
        if snapshot_id is not None:
            row = conn.execute(
                """
                SELECT snapshot_id, tenant_id, domain_id, repo_id, repo_name, ref_name,
                       ref_kind, mr_iid, base_snapshot_id, base_commit_sha, head_commit_sha,
                       state, allowed_scopes, metadata_json, created_at, updated_at
                FROM ref_snapshots
                WHERE domain_id = %s AND snapshot_id = %s;
                """,
                (domain_id, snapshot_id),
            ).fetchone()
        elif repo_name is not None and ref_name is not None:
            row = conn.execute(
                """
                SELECT snapshot_id, tenant_id, domain_id, repo_id, repo_name, ref_name,
                       ref_kind, mr_iid, base_snapshot_id, base_commit_sha, head_commit_sha,
                       state, allowed_scopes, metadata_json, created_at, updated_at
                FROM ref_snapshots
                WHERE domain_id = %s AND repo_name = %s AND ref_name = %s;
                """,
                (domain_id, repo_name, ref_name),
            ).fetchone()
        elif repo_name is not None and mr_iid is not None:
            row = conn.execute(
                """
                SELECT snapshot_id, tenant_id, domain_id, repo_id, repo_name, ref_name,
                       ref_kind, mr_iid, base_snapshot_id, base_commit_sha, head_commit_sha,
                       state, allowed_scopes, metadata_json, created_at, updated_at
                FROM ref_snapshots
                WHERE domain_id = %s AND repo_name = %s AND mr_iid = %s
                ORDER BY updated_at DESC
                LIMIT 1;
                """,
                (domain_id, repo_name, mr_iid),
            ).fetchone()
        else:
            raise ValueError("Specify snapshot_id, (repo_name, ref_name), or (repo_name, mr_iid)")
        return self._row_to_ref_snapshot(row) if row is not None else None

    def list_ref_snapshots(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        *,
        repo_name: str | None = None,
        state: str | None = None,
    ) -> list[RefSnapshotRecord]:
        rows = conn.execute(
            """
            SELECT snapshot_id, tenant_id, domain_id, repo_id, repo_name, ref_name,
                   ref_kind, mr_iid, base_snapshot_id, base_commit_sha, head_commit_sha,
                   state, allowed_scopes, metadata_json, created_at, updated_at
            FROM ref_snapshots
            WHERE domain_id = %s
              AND (%s::text IS NULL OR repo_name = %s)
              AND (%s::text IS NULL OR state = %s)
            ORDER BY updated_at DESC, ref_name ASC;
            """,
            (domain_id, repo_name, repo_name, state, state),
        ).fetchall()
        return [self._row_to_ref_snapshot(r) for r in rows]

    def delete_ref_snapshot(self, conn: Connection[tuple[object, ...]], snapshot_id: UUID) -> int:
        cur = conn.execute("DELETE FROM ref_snapshots WHERE snapshot_id = %s;", (snapshot_id,))
        return max(cur.rowcount or 0, 0)

    def upsert_overlay_file_delta(
        self,
        conn: Connection[tuple[object, ...]],
        record: OverlayFileDeltaRecord,
    ) -> OverlayFileDeltaRecord:
        row = conn.execute(
            """
            INSERT INTO overlay_file_deltas (
                delta_id, tenant_id, snapshot_id, domain_id, repo_name, file_path,
                previous_file_path, change_type, base_blob_sha, overlay_blob_sha,
                language, raw_content, line_count, ranking_weight, is_generated,
                is_vendored, symbols_json, edges_json, allowed_scopes, indexed_at
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, COALESCE(%s, NOW())
            )
            ON CONFLICT (tenant_id, snapshot_id, file_path) DO UPDATE SET
                previous_file_path = EXCLUDED.previous_file_path,
                change_type = EXCLUDED.change_type,
                base_blob_sha = EXCLUDED.base_blob_sha,
                overlay_blob_sha = EXCLUDED.overlay_blob_sha,
                language = EXCLUDED.language,
                raw_content = EXCLUDED.raw_content,
                line_count = EXCLUDED.line_count,
                ranking_weight = EXCLUDED.ranking_weight,
                is_generated = EXCLUDED.is_generated,
                is_vendored = EXCLUDED.is_vendored,
                symbols_json = EXCLUDED.symbols_json,
                edges_json = EXCLUDED.edges_json,
                allowed_scopes = EXCLUDED.allowed_scopes,
                indexed_at = NOW()
            RETURNING delta_id, tenant_id, snapshot_id, domain_id, repo_name, file_path,
                      previous_file_path, change_type, base_blob_sha, overlay_blob_sha,
                      language, raw_content, line_count, ranking_weight, is_generated,
                      symbols_json, edges_json, is_vendored, allowed_scopes, indexed_at;
            """,
            (
                record.delta_id,
                record.tenant_id,
                record.snapshot_id,
                record.domain_id,
                record.repo_name,
                record.file_path,
                record.previous_file_path,
                record.change_type,
                record.base_blob_sha,
                record.overlay_blob_sha,
                record.language,
                record.raw_content,
                record.line_count,
                record.ranking_weight,
                record.is_generated,
                record.is_vendored,
                json.dumps(record.symbols_json),
                json.dumps(record.edges_json),
                record.allowed_scopes,
                record.indexed_at,
            ),
        ).fetchone()
        assert row is not None
        return self._row_to_overlay_delta(row)

    def list_overlay_file_deltas(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        snapshot_id: UUID,
    ) -> list[OverlayFileDeltaRecord]:
        rows = conn.execute(
            """
            SELECT delta_id, tenant_id, snapshot_id, domain_id, repo_name, file_path,
                   previous_file_path, change_type, base_blob_sha, overlay_blob_sha,
                   language, raw_content, line_count, ranking_weight, is_generated,
                   symbols_json, edges_json, is_vendored, allowed_scopes, indexed_at
            FROM overlay_file_deltas
            WHERE domain_id = %s AND snapshot_id = %s
            ORDER BY file_path ASC;
            """,
            (domain_id, snapshot_id),
        ).fetchall()
        return [self._row_to_overlay_delta(r) for r in rows]

    def resolve_overlay_files(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        snapshot_id: UUID,
        *,
        file_path: str | None = None,
    ) -> list[ResolvedOverlayFileRecord]:
        """Resolves effective files for an MR/branch via COALESCE(overlay, base_snapshot)."""
        path_filter_delta = "AND d.file_path = %s" if file_path is not None else ""
        path_filter_base = "AND cf.file_path = %s" if file_path is not None else ""
        sql = f"""
            WITH RECURSIVE target_snap AS (
                SELECT snapshot_id, tenant_id, domain_id, repo_name,
                       base_snapshot_id, head_commit_sha
                FROM ref_snapshots
                WHERE domain_id = %s AND snapshot_id = %s
            ),
            snapshot_chain AS (
                SELECT snapshot_id, base_snapshot_id, 0 AS depth
                FROM target_snap
                UNION ALL
                SELECT parent.snapshot_id, parent.base_snapshot_id, sc.depth + 1
                FROM ref_snapshots parent
                JOIN snapshot_chain sc ON parent.snapshot_id = sc.base_snapshot_id
                WHERE sc.depth < 16
            ),
            effective_deltas AS (
                SELECT DISTINCT ON (d.file_path)
                       d.delta_id, d.snapshot_id, d.domain_id, d.repo_name, d.file_path,
                       d.previous_file_path, d.change_type, d.overlay_blob_sha, d.language,
                       d.raw_content, d.line_count, d.ranking_weight, d.is_generated,
                       d.is_vendored, d.allowed_scopes, d.tenant_id
                FROM overlay_file_deltas d
                JOIN snapshot_chain sc ON sc.snapshot_id = d.snapshot_id
                WHERE d.domain_id = %s {path_filter_delta}
                ORDER BY d.file_path, sc.depth ASC
            ),
            renamed_sources AS (
                SELECT DISTINCT d2.previous_file_path AS old_path
                FROM overlay_file_deltas d2
                JOIN snapshot_chain sc2 ON sc2.snapshot_id = d2.snapshot_id
                WHERE d2.previous_file_path IS NOT NULL
                  AND d2.previous_file_path <> d2.file_path
            ),
            base_files AS (
                SELECT cf.file_id, cf.domain_id, cf.repo_name, cf.file_path, cf.git_commit_sha,
                       cf.git_blob_sha, cf.language, cf.allowed_scopes, cf.raw_content,
                       cf.line_count, COALESCE(cf.ranking_weight, 1.0) AS ranking_weight,
                       COALESCE(cf.is_generated, FALSE) AS is_generated,
                       COALESCE(cf.is_vendored, FALSE) AS is_vendored, cf.tenant_id
                FROM code_files cf
                JOIN target_snap ts ON cf.domain_id = ts.domain_id AND cf.repo_name = ts.repo_name
                WHERE NOT EXISTS (
                    SELECT 1 FROM renamed_sources rs WHERE rs.old_path = cf.file_path
                )
                {path_filter_base}
            )
            SELECT
                COALESCE(d.domain_id, b.domain_id) AS domain_id,
                COALESCE(d.repo_name, b.repo_name) AS repo_name,
                COALESCE(d.file_path, b.file_path) AS file_path,
                CASE
                    WHEN d.delta_id IS NOT NULL THEN ts.head_commit_sha
                    ELSE b.git_commit_sha
                END AS git_commit_sha,
                COALESCE(d.overlay_blob_sha, b.git_blob_sha) AS git_blob_sha,
                COALESCE(d.language, b.language) AS language,
                COALESCE(d.allowed_scopes, b.allowed_scopes) AS allowed_scopes,
                COALESCE(d.raw_content, b.raw_content) AS raw_content,
                COALESCE(d.line_count, b.line_count) AS line_count,
                CASE
                    WHEN d.delta_id IS NOT NULL THEN 'overlay'
                    ELSE 'base_snapshot'
                END AS source_layer,
                ts.snapshot_id,
                b.file_id AS base_file_id,
                d.delta_id,
                d.change_type,
                COALESCE(d.ranking_weight, b.ranking_weight, 1.0) AS ranking_weight,
                COALESCE(d.is_generated, b.is_generated, FALSE) AS is_generated,
                COALESCE(d.is_vendored, b.is_vendored, FALSE) AS is_vendored,
                COALESCE(d.tenant_id, b.tenant_id) AS tenant_id
            FROM effective_deltas d
            FULL OUTER JOIN base_files b ON d.file_path = b.file_path
            CROSS JOIN target_snap ts
            WHERE COALESCE(d.change_type, 'base') <> 'deleted'
            ORDER BY COALESCE(d.file_path, b.file_path) ASC;
        """
        params: list[object] = [domain_id, snapshot_id, domain_id]
        if file_path is not None:
            params.extend([file_path, file_path])
        rows = conn.execute(sql, tuple(params)).fetchall()
        return [
            ResolvedOverlayFileRecord(
                domain_id=str(r[0]),
                repo_name=str(r[1]),
                file_path=str(r[2]),
                git_commit_sha=str(r[3]),
                git_blob_sha=str(r[4]),
                language=str(r[5]),
                allowed_scopes=[str(s) for s in r[6]] if isinstance(r[6], list) else [],
                raw_content=str(r[7]),
                line_count=int(str(r[8])),
                source_layer=r[9],  # type: ignore[arg-type]
                snapshot_id=UUID(str(r[10])) if r[10] is not None else None,
                base_file_id=UUID(str(r[11])) if r[11] is not None else None,
                delta_id=UUID(str(r[12])) if r[12] is not None else None,
                change_type=str(r[13]) if r[13] is not None else None,
                ranking_weight=float(str(r[14])),
                is_generated=bool(r[15]),
                is_vendored=bool(r[16]),
                tenant_id=UUID(str(r[17])),
            )
            for r in rows
        ]

    def get_overlay_file(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        snapshot_id: UUID,
        file_path: str,
    ) -> ResolvedOverlayFileRecord | None:
        items = self.resolve_overlay_files(conn, domain_id, snapshot_id, file_path=file_path)
        return items[0] if items else None

    def resolve_overlay_symbols(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        snapshot_id: UUID,
    ) -> list[ASTSymbolRecord]:
        """Resolves effective AST symbols for an overlay snapshot without mutating base symbols."""
        snap = self.get_ref_snapshot(conn, domain_id, snapshot_id=snapshot_id)
        if snap is None:
            return []

        deltas_rows = conn.execute(
            """
            WITH RECURSIVE snapshot_chain AS (
                SELECT snapshot_id, base_snapshot_id, 0 AS depth
                FROM ref_snapshots
                WHERE domain_id = %s AND snapshot_id = %s
                UNION ALL
                SELECT parent.snapshot_id, parent.base_snapshot_id, sc.depth + 1
                FROM ref_snapshots parent
                JOIN snapshot_chain sc ON parent.snapshot_id = sc.base_snapshot_id
                WHERE sc.depth < 16
            ),
            effective_deltas AS (
                SELECT DISTINCT ON (d.file_path)
                       d.delta_id, d.domain_id, d.file_path, d.previous_file_path,
                       d.change_type, d.symbols_json, d.allowed_scopes, d.tenant_id
                FROM overlay_file_deltas d
                JOIN snapshot_chain sc ON sc.snapshot_id = d.snapshot_id
                WHERE d.domain_id = %s
                ORDER BY d.file_path, sc.depth ASC
            )
            SELECT delta_id, domain_id, file_path, previous_file_path,
                   change_type, symbols_json, allowed_scopes, tenant_id
            FROM effective_deltas;
            """,
            (domain_id, snapshot_id, domain_id),
        ).fetchall()

        renamed_old_rows = conn.execute(
            """
            WITH RECURSIVE snapshot_chain AS (
                SELECT snapshot_id, base_snapshot_id, 0 AS depth
                FROM ref_snapshots
                WHERE domain_id = %s AND snapshot_id = %s
                UNION ALL
                SELECT parent.snapshot_id, parent.base_snapshot_id, sc.depth + 1
                FROM ref_snapshots parent
                JOIN snapshot_chain sc ON parent.snapshot_id = sc.base_snapshot_id
                WHERE sc.depth < 16
            )
            SELECT DISTINCT d.previous_file_path
            FROM overlay_file_deltas d
            JOIN snapshot_chain sc ON sc.snapshot_id = d.snapshot_id
            WHERE d.previous_file_path IS NOT NULL;
            """,
            (domain_id, snapshot_id),
        ).fetchall()

        suppressed_paths: set[str] = {str(r[0]) for r in renamed_old_rows if r[0] is not None}
        overlay_symbols: list[ASTSymbolRecord] = []

        for r in deltas_rows:
            delta_id = UUID(str(r[0]))
            f_path = str(r[2])
            prev_path = str(r[3]) if r[3] is not None else None
            change_type = str(r[4])
            suppressed_paths.add(f_path)
            if prev_path:
                suppressed_paths.add(prev_path)
            if change_type == "deleted":
                continue
            syms_raw = r[5] if isinstance(r[5], list) else json.loads(str(r[5] or "[]"))
            scopes = [str(s) for s in r[6]] if isinstance(r[6], list) else []
            tenant_id = UUID(str(r[7]))
            for sym_dict in syms_raw:
                if not isinstance(sym_dict, dict):
                    continue
                overlay_symbols.append(
                    ASTSymbolRecord(
                        symbol_id=UUID(str(sym_dict["symbol_id"])),
                        file_id=delta_id,
                        domain_id=str(r[1]),
                        qualified_name=str(sym_dict["qualified_name"]),
                        symbol_kind=str(sym_dict.get("symbol_kind", "function")),
                        start_line=int(sym_dict.get("start_line", 1)),
                        end_line=int(sym_dict.get("end_line", 1)),
                        signature_skeleton=str(sym_dict.get("signature_skeleton", "")),
                        contextual_summary=str(sym_dict.get("contextual_summary", "")),
                        allowed_scopes=scopes,
                        embedding=[],
                        repo_qualified_id=sym_dict.get("repo_qualified_id"),
                        bare_name=sym_dict.get("bare_name"),
                        tenant_id=tenant_id,
                    )
                )

        base_rows = conn.execute(
            """
            SELECT s.symbol_id, s.file_id, s.domain_id, s.qualified_name, s.symbol_kind,
                   s.start_line, s.end_line, s.signature_skeleton, s.contextual_summary,
                   s.allowed_scopes, s.repo_qualified_id, s.bare_name, s.tenant_id
            FROM ast_symbols s
            JOIN code_files f ON f.file_id = s.file_id
            WHERE s.domain_id = %s
              AND f.repo_name = %s
              AND NOT (f.file_path = ANY(%s::text[]))
            ORDER BY s.qualified_name ASC;
            """,
            (domain_id, snap.repo_name, list(suppressed_paths)),
        ).fetchall()

        base_symbols = [
            ASTSymbolRecord(
                symbol_id=UUID(str(r[0])),
                file_id=UUID(str(r[1])),
                domain_id=str(r[2]),
                qualified_name=str(r[3]),
                symbol_kind=str(r[4]),
                start_line=int(str(r[5])),
                end_line=int(str(r[6])),
                signature_skeleton=str(r[7]),
                contextual_summary=str(r[8]),
                allowed_scopes=[str(s) for s in r[9]] if isinstance(r[9], list) else [],
                embedding=[],
                repo_qualified_id=str(r[10]) if r[10] is not None else None,
                bare_name=str(r[11]) if r[11] is not None else None,
                tenant_id=UUID(str(r[12])),
            )
            for r in base_rows
        ]
        combined = base_symbols + overlay_symbols
        combined.sort(key=lambda s: s.qualified_name)
        return combined

    def list_ownership_edges(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        *,
        edge_kinds: Sequence[str] = ("OWNED_BY_TEAM", "PART_OF_SYSTEM"),
    ) -> list[OwnershipEdgeRecord]:
        rows = conn.execute(
            """
            SELECT e.edge_id, e.edge_kind, s_src.qualified_name, s_src.symbol_kind,
                   e.target_qualified_name, COALESCE(s_tgt.symbol_kind, '')
            FROM ast_edges e
            JOIN ast_symbols s_src ON s_src.symbol_id = e.source_symbol_id
            LEFT JOIN ast_symbols s_tgt ON s_tgt.symbol_id = e.target_symbol_id
            WHERE e.domain_id = %s AND e.edge_kind = ANY(%s::text[])
            ORDER BY s_src.qualified_name ASC, e.edge_kind ASC, e.target_qualified_name ASC;
            """,
            (domain_id, list(edge_kinds)),
        ).fetchall()
        return [
            OwnershipEdgeRecord(
                edge_id=UUID(str(r[0])),
                edge_kind=str(r[1]),
                source_qualified_name=str(r[2]),
                source_symbol_kind=str(r[3]),
                target_qualified_name=str(r[4]),
                target_symbol_kind=str(r[5]),
            )
            for r in rows
        ]

    @staticmethod
    def _row_to_side_effect(row: tuple[object, ...]) -> SideEffectRecord:
        meta_raw = row[16] if isinstance(row[16], dict) else json.loads(str(row[16] or "{}"))
        return SideEffectRecord(
            side_effect_id=UUID(str(row[0])),
            tenant_id=UUID(str(row[1])),
            domain_id=str(row[2]),
            repo_name=str(row[3]),
            file_id=UUID(str(row[4])) if row[4] is not None else None,
            symbol_id=UUID(str(row[5])) if row[5] is not None else None,
            file_path=str(row[6]),
            qualified_symbol=str(row[7]),
            start_line=int(str(row[8])),
            end_line=int(str(row[9])),
            effect_category=cast(EffectCategory, str(row[10])),
            target_identifier=str(row[11]),
            provider_or_driver=str(row[12] or ""),
            trigger_condition=str(row[13] or ""),
            template_or_schema_ref=str(row[14]) if row[14] is not None else None,
            severity_tier=cast(SideEffectSeverity, str(row[15] or "external_io")),
            metadata_json=dict(meta_raw),
            confidence=float(str(row[17])),
            allowed_scopes=[str(s) for s in row[18]] if isinstance(row[18], list) else [],
            indexed_at=_as_opt_dt(row[19]),
        )

    def upsert_side_effects_batch(
        self,
        conn: Connection[tuple[object, ...]],
        records: Sequence[SideEffectRecord],
    ) -> int:
        if not records:
            return 0
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO side_effects (
                    side_effect_id, tenant_id, domain_id, repo_name, file_id, symbol_id,
                    file_path, qualified_symbol, start_line, end_line,
                    effect_category, target_identifier, provider_or_driver,
                    trigger_condition, template_or_schema_ref, severity_tier,
                    confidence, metadata_json, allowed_scopes, indexed_at
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, COALESCE(%s, NOW())
                )
                ON CONFLICT (
                    tenant_id, domain_id, repo_name, file_path,
                    qualified_symbol, effect_category, target_identifier, start_line
                )
                DO UPDATE SET
                    file_id = EXCLUDED.file_id,
                    symbol_id = EXCLUDED.symbol_id,
                    end_line = EXCLUDED.end_line,
                    provider_or_driver = EXCLUDED.provider_or_driver,
                    trigger_condition = EXCLUDED.trigger_condition,
                    template_or_schema_ref = EXCLUDED.template_or_schema_ref,
                    severity_tier = EXCLUDED.severity_tier,
                    confidence = EXCLUDED.confidence,
                    metadata_json = EXCLUDED.metadata_json,
                    allowed_scopes = EXCLUDED.allowed_scopes,
                    indexed_at = NOW();
                """,
                [
                    (
                        r.side_effect_id,
                        r.tenant_id,
                        r.domain_id,
                        r.repo_name,
                        r.file_id,
                        r.symbol_id,
                        r.file_path,
                        r.qualified_symbol,
                        r.start_line,
                        r.end_line,
                        r.effect_category,
                        r.target_identifier,
                        r.provider_or_driver,
                        r.trigger_condition,
                        r.template_or_schema_ref,
                        r.severity_tier,
                        r.confidence,
                        json.dumps(r.metadata_json),
                        r.allowed_scopes,
                        r.indexed_at,
                    )
                    for r in records
                ],
            )
        return len(records)

    def list_visible_side_effects(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        *,
        effect_category: str | None = None,
        repo_name: str | None = None,
        file_path: str | None = None,
    ) -> list[SideEffectRecord]:
        rows = conn.execute(
            """
            SELECT side_effect_id, tenant_id, domain_id, repo_name, file_id, symbol_id,
                   file_path, qualified_symbol, start_line, end_line,
                   effect_category, target_identifier, provider_or_driver,
                   trigger_condition, template_or_schema_ref, severity_tier,
                   metadata_json, confidence, allowed_scopes, indexed_at
            FROM side_effects
            WHERE domain_id = %s
              AND (%s::text IS NULL OR effect_category = %s)
              AND (%s::text IS NULL OR repo_name = %s)
              AND (%s::text IS NULL OR file_path = %s)
            ORDER BY repo_name ASC, file_path ASC, start_line ASC, effect_category ASC;
            """,
            (
                domain_id,
                effect_category,
                effect_category,
                repo_name,
                repo_name,
                file_path,
                file_path,
            ),
        ).fetchall()
        return [self._row_to_side_effect(row) for row in rows]

    def delete_side_effects_for_file(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        repo_name: str,
        file_path: str,
    ) -> int:
        cur = conn.execute(
            "DELETE FROM side_effects WHERE domain_id = %s AND repo_name = %s AND file_path = %s;",
            (domain_id, repo_name, file_path),
        )
        return max(cur.rowcount or 0, 0)

    def delete_side_effects_for_repo(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        repo_name: str,
    ) -> int:
        cur = conn.execute(
            "DELETE FROM side_effects WHERE domain_id = %s AND repo_name = %s;",
            (domain_id, repo_name),
        )
        return max(cur.rowcount or 0, 0)

    @staticmethod
    def _row_to_watermark(row: tuple[object, ...]) -> ConnectorWatermarkRecord:
        meta_raw = row[10] if isinstance(row[10], dict) else json.loads(str(row[10] or "{}"))
        tier_str = str(row[5] or "watermark")
        tier = cast(
            Any,
            tier_str
            if tier_str in {"webhook", "watermark", "reconcile", "bundle"}
            else "watermark",
        )
        return ConnectorWatermarkRecord(
            tenant_id=UUID(str(row[0])),
            domain_id=str(row[1]),
            source_id=str(row[2]),
            container_key=str(row[3]),
            watermark_cursor=str(row[4] or ""),
            sync_tier=tier,
            items_synced=int(str(row[6] or 0)),
            last_webhook_at=_as_opt_dt(row[7]),
            last_watermark_sync_at=_as_opt_dt(row[8]),
            last_reconciliation_at=_as_opt_dt(row[9]),
            metadata_json=dict(meta_raw),
            updated_at=_as_opt_dt(row[11]),
        )

    def upsert_connector_watermark(
        self,
        conn: Connection[tuple[object, ...]],
        record: ConnectorWatermarkRecord,
    ) -> None:
        conn.execute(
            """
            INSERT INTO connector_sync_watermarks (
                tenant_id, domain_id, source_id, container_key,
                watermark_cursor, sync_tier, items_synced,
                last_webhook_at, last_watermark_sync_at, last_reconciliation_at,
                metadata_json, updated_at
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, COALESCE(%s, NOW())
            )
            ON CONFLICT (tenant_id, domain_id, source_id, container_key) DO UPDATE SET
                watermark_cursor = CASE
                    WHEN EXCLUDED.watermark_cursor <> '' THEN EXCLUDED.watermark_cursor
                    ELSE connector_sync_watermarks.watermark_cursor
                END,
                sync_tier = EXCLUDED.sync_tier,
                items_synced = EXCLUDED.items_synced,
                last_webhook_at = COALESCE(
                    EXCLUDED.last_webhook_at,
                    connector_sync_watermarks.last_webhook_at
                ),
                last_watermark_sync_at = COALESCE(
                    EXCLUDED.last_watermark_sync_at,
                    connector_sync_watermarks.last_watermark_sync_at
                ),
                last_reconciliation_at = COALESCE(
                    EXCLUDED.last_reconciliation_at,
                    connector_sync_watermarks.last_reconciliation_at
                ),
                metadata_json = EXCLUDED.metadata_json,
                updated_at = NOW();
            """,
            (
                record.tenant_id,
                record.domain_id,
                record.source_id,
                record.container_key,
                record.watermark_cursor,
                record.sync_tier,
                record.items_synced,
                record.last_webhook_at,
                record.last_watermark_sync_at,
                record.last_reconciliation_at,
                json.dumps(record.metadata_json),
                record.updated_at,
            ),
        )

    def reset_connector_watermark(
        self,
        conn: Connection[tuple[object, ...]],
        record: ConnectorWatermarkRecord,
    ) -> None:
        """Unconditionally rewinds watermark_cursor (including to empty) after PITR."""
        conn.execute(
            """
            INSERT INTO connector_sync_watermarks (
                tenant_id, domain_id, source_id, container_key,
                watermark_cursor, sync_tier, items_synced,
                last_webhook_at, last_watermark_sync_at, last_reconciliation_at,
                metadata_json, updated_at
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, COALESCE(%s, NOW()), %s::jsonb, NOW()
            )
            ON CONFLICT (tenant_id, domain_id, source_id, container_key) DO UPDATE SET
                watermark_cursor = EXCLUDED.watermark_cursor,
                sync_tier = EXCLUDED.sync_tier,
                items_synced = EXCLUDED.items_synced,
                last_reconciliation_at = COALESCE(
                    EXCLUDED.last_reconciliation_at,
                    NOW()
                ),
                metadata_json = EXCLUDED.metadata_json,
                updated_at = NOW();
            """,
            (
                record.tenant_id,
                record.domain_id,
                record.source_id,
                record.container_key,
                record.watermark_cursor,
                record.sync_tier,
                record.items_synced,
                record.last_webhook_at,
                record.last_watermark_sync_at,
                record.last_reconciliation_at,
                json.dumps(record.metadata_json),
            ),
        )

    def get_connector_watermark(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        source_id: str,
        container_key: str,
    ) -> ConnectorWatermarkRecord | None:
        row = conn.execute(
            """
            SELECT tenant_id, domain_id, source_id, container_key,
                   watermark_cursor, sync_tier, items_synced,
                   last_webhook_at, last_watermark_sync_at, last_reconciliation_at,
                   metadata_json, updated_at
            FROM connector_sync_watermarks
            WHERE domain_id = %s AND source_id = %s AND container_key = %s;
            """,
            (domain_id, source_id, container_key),
        ).fetchone()
        return self._row_to_watermark(row) if row is not None else None

    def list_connector_watermarks(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        source_id: str | None = None,
    ) -> list[ConnectorWatermarkRecord]:
        if source_id:
            rows = conn.execute(
                """
                SELECT tenant_id, domain_id, source_id, container_key,
                       watermark_cursor, sync_tier, items_synced,
                       last_webhook_at, last_watermark_sync_at, last_reconciliation_at,
                       metadata_json, updated_at
                FROM connector_sync_watermarks
                WHERE domain_id = %s AND source_id = %s
                ORDER BY source_id, container_key;
                """,
                (domain_id, source_id),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT tenant_id, domain_id, source_id, container_key,
                       watermark_cursor, sync_tier, items_synced,
                       last_webhook_at, last_watermark_sync_at, last_reconciliation_at,
                       metadata_json, updated_at
                FROM connector_sync_watermarks
                WHERE domain_id = %s
                ORDER BY source_id, container_key;
                """,
                (domain_id,),
            ).fetchall()
        return [self._row_to_watermark(row) for row in rows]

    def purge_domain_artifacts(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        *,
        doc_id: str | None = None,
        confluence_page_id: str | None = None,
        jira_issue_key: str | None = None,
        repo_name: str | None = None,
        file_path: str | None = None,
        commit_sha: str | None = None,
        blob_sha: str | None = None,
        include_derived_wiki: bool = True,
    ) -> dict[str, int]:
        """Hard-deletes matching artifacts across PostgreSQL 17 tables under Forced RLS."""
        norm = normalize_purge_selectors(
            domain_id,
            doc_id=doc_id,
            confluence_page_id=confluence_page_id,
            jira_issue_key=jira_issue_key,
            repo_name=repo_name,
            file_path=file_path,
            commit_sha=commit_sha,
            blob_sha=blob_sha,
        )
        domain_id = norm.domain_id
        doc_id = norm.doc_id
        confluence_page_id = norm.confluence_page_id
        jira_issue_key = norm.jira_issue_key
        repo_name = norm.repo_name
        file_path = norm.file_path
        commit_sha = norm.commit_sha
        blob_sha = norm.blob_sha
        counts: dict[str, int] = {
            "side_effects": 0,
            "overlay_file_deltas": 0,
            "ref_snapshots": 0,
            "ast_edges": 0,
            "ast_symbols": 0,
            "code_files": 0,
            "repo_commits": 0,
            "repositories": 0,
            "doc_chunks": 0,
            "confluence_pages": 0,
            "jira_issues": 0,
        }
        target_doc_id = doc_id
        if target_doc_id is None and confluence_page_id:
            target_doc_id = f"confluence:{confluence_page_id}"
        elif target_doc_id is None and jira_issue_key:
            target_doc_id = f"jira:{jira_issue_key}"

        if target_doc_id:
            cur = conn.execute(
                "DELETE FROM doc_chunks WHERE domain_id = %s AND doc_id = %s;",
                (domain_id, target_doc_id),
            )
            counts["doc_chunks"] += max(cur.rowcount or 0, 0)

        if confluence_page_id:
            cur = conn.execute(
                "DELETE FROM confluence_pages WHERE domain_id = %s AND confluence_page_id = %s;",
                (domain_id, confluence_page_id),
            )
            counts["confluence_pages"] += max(cur.rowcount or 0, 0)

        if jira_issue_key:
            cur = conn.execute(
                "DELETE FROM jira_issues WHERE domain_id = %s AND issue_key = %s;",
                (domain_id, jira_issue_key),
            )
            counts["jira_issues"] += max(cur.rowcount or 0, 0)

        target_file_ids: list[UUID] = []
        if repo_name and file_path:
            clean_repo = repo_name.replace("/", "_")
            rows = conn.execute(
                """
                SELECT file_id FROM code_files
                WHERE domain_id = %s AND (repo_name = %s OR repo_name = %s) AND file_path = %s;
                """,
                (domain_id, repo_name, clean_repo, file_path),
            ).fetchall()
            target_file_ids = [UUID(str(r[0])) for r in rows]
            cur_ov = conn.execute(
                """
                DELETE FROM overlay_file_deltas
                WHERE domain_id = %s AND (repo_name = %s OR repo_name = %s)
                  AND (file_path = %s OR previous_file_path = %s);
                """,
                (domain_id, repo_name, clean_repo, file_path, file_path),
            )
            counts["overlay_file_deltas"] += max(cur_ov.rowcount or 0, 0)
            cur_se = conn.execute(
                """
                DELETE FROM side_effects
                WHERE domain_id = %s AND (repo_name = %s OR repo_name = %s) AND file_path = %s;
                """,
                (domain_id, repo_name, clean_repo, file_path),
            )
            counts["side_effects"] += max(cur_se.rowcount or 0, 0)
        elif blob_sha:
            rows = conn.execute(
                "SELECT file_id FROM code_files WHERE domain_id = %s AND git_blob_sha = %s;",
                (domain_id, blob_sha),
            ).fetchall()
            target_file_ids = [UUID(str(r[0])) for r in rows]
            cur_ov = conn.execute(
                """
                DELETE FROM overlay_file_deltas
                WHERE domain_id = %s AND (overlay_blob_sha = %s OR base_blob_sha = %s);
                """,
                (domain_id, blob_sha, blob_sha),
            )
            counts["overlay_file_deltas"] += max(cur_ov.rowcount or 0, 0)
        elif commit_sha:
            rows = conn.execute(
                "SELECT file_id FROM code_files WHERE domain_id = %s AND git_commit_sha = %s;",
                (domain_id, commit_sha),
            ).fetchall()
            target_file_ids = [UUID(str(r[0])) for r in rows]
            cur_rc = conn.execute(
                """
                DELETE FROM repo_commits
                WHERE commit_sha = %s
                  AND repo_id IN (SELECT repo_id FROM repositories WHERE domain_id = %s);
                """,
                (commit_sha, domain_id),
            )
            counts["repo_commits"] += max(cur_rc.rowcount or 0, 0)
            cur_ov = conn.execute(
                """
                DELETE FROM overlay_file_deltas
                WHERE snapshot_id IN (
                    SELECT snapshot_id FROM ref_snapshots
                    WHERE domain_id = %s
                      AND (head_commit_sha = %s OR base_commit_sha = %s)
                );
                """,
                (domain_id, commit_sha, commit_sha),
            )
            counts["overlay_file_deltas"] += max(cur_ov.rowcount or 0, 0)
            cur_rs = conn.execute(
                """
                DELETE FROM ref_snapshots
                WHERE domain_id = %s AND (head_commit_sha = %s OR base_commit_sha = %s);
                """,
                (domain_id, commit_sha, commit_sha),
            )
            counts["ref_snapshots"] += max(cur_rs.rowcount or 0, 0)
        elif repo_name:
            clean_repo = repo_name.replace("/", "_")
            rows = conn.execute(
                """
                SELECT file_id FROM code_files
                WHERE domain_id = %s AND (repo_name = %s OR repo_name = %s);
                """,
                (domain_id, repo_name, clean_repo),
            ).fetchall()
            target_file_ids = [UUID(str(r[0])) for r in rows]
            cur_se = conn.execute(
                """
                DELETE FROM side_effects
                WHERE domain_id = %s AND (repo_name = %s OR repo_name = %s);
                """,
                (domain_id, repo_name, clean_repo),
            )
            counts["side_effects"] += max(cur_se.rowcount or 0, 0)
            cur_ov = conn.execute(
                """
                DELETE FROM overlay_file_deltas
                WHERE domain_id = %s AND (repo_name = %s OR repo_name = %s);
                """,
                (domain_id, repo_name, clean_repo),
            )
            counts["overlay_file_deltas"] += max(cur_ov.rowcount or 0, 0)
            cur_rs = conn.execute(
                """
                DELETE FROM ref_snapshots
                WHERE domain_id = %s AND (repo_name = %s OR repo_name = %s);
                """,
                (domain_id, repo_name, clean_repo),
            )
            counts["ref_snapshots"] += max(cur_rs.rowcount or 0, 0)
            cur_rc = conn.execute(
                """
                DELETE FROM repo_commits
                WHERE repo_id IN (
                    SELECT repo_id FROM repositories
                    WHERE domain_id = %s AND (repo_name = %s OR repo_name = %s)
                );
                """,
                (domain_id, repo_name, clean_repo),
            )
            counts["repo_commits"] += max(cur_rc.rowcount or 0, 0)
            cur_rep = conn.execute(
                """
                DELETE FROM repositories
                WHERE domain_id = %s AND (repo_name = %s OR repo_name = %s);
                """,
                (domain_id, repo_name, clean_repo),
            )
            counts["repositories"] += max(cur_rep.rowcount or 0, 0)

        if target_file_ids:
            sym_rows = conn.execute(
                "SELECT symbol_id FROM ast_symbols WHERE file_id = ANY(%s);",
                (target_file_ids,),
            ).fetchall()
            sym_ids = [UUID(str(r[0])) for r in sym_rows]
            cur_se_f = conn.execute(
                "DELETE FROM side_effects WHERE file_id = ANY(%s);",
                (target_file_ids,),
            )
            counts["side_effects"] += max(cur_se_f.rowcount or 0, 0)
            if sym_ids:
                cur_ed = conn.execute(
                    """
                    DELETE FROM ast_edges
                    WHERE domain_id = %s
                      AND (source_symbol_id = ANY(%s) OR target_symbol_id = ANY(%s));
                    """,
                    (domain_id, sym_ids, sym_ids),
                )
                counts["ast_edges"] += max(cur_ed.rowcount or 0, 0)
            cur_sy = conn.execute(
                "DELETE FROM ast_symbols WHERE file_id = ANY(%s);",
                (target_file_ids,),
            )
            counts["ast_symbols"] += max(cur_sy.rowcount or 0, 0)
            cur_cf = conn.execute(
                "DELETE FROM code_files WHERE file_id = ANY(%s);",
                (target_file_ids,),
            )
            counts["code_files"] += max(cur_cf.rowcount or 0, 0)

        if include_derived_wiki and repo_name:
            clean_repo = repo_name.replace("/", "_")
            cur_wiki = conn.execute(
                """
                DELETE FROM doc_chunks
                WHERE domain_id = %s
                  AND source_type = 'repo_wiki'
                  AND (
                      doc_id = %s OR doc_id LIKE %s
                      OR doc_id = %s OR doc_id LIKE %s
                  );
                """,
                (
                    domain_id,
                    f"repo_wiki:{repo_name}",
                    f"repo_wiki:{repo_name}:%",
                    f"repo_wiki:{clean_repo}",
                    f"repo_wiki:{clean_repo}:%",
                ),
            )
            counts["doc_chunks"] += max(cur_wiki.rowcount or 0, 0)

        return counts

    @staticmethod
    def _row_to_scout_session(row: tuple[object, ...]) -> ScoutSessionRecord:
        recent_raw = row[6] if isinstance(row[6], list) else json.loads(str(row[6] or "[]"))
        blobs_raw = row[7] if isinstance(row[7], dict) else json.loads(str(row[7] or "{}"))
        syms_raw = row[8] if isinstance(row[8], list) else json.loads(str(row[8] or "[]"))
        scopes_raw = row[4]
        scopes = [str(s) for s in scopes_raw] if isinstance(scopes_raw, (list, tuple)) else []
        return ScoutSessionRecord(
            tenant_id=UUID(str(row[0])),
            session_id=str(row[1]),
            domain_id=str(row[2]),
            principal_id=str(row[3]),
            allowed_scopes=scopes,
            turn_count=int(str(row[5] or 0)),
            recent_turns_json=[dict(item) for item in recent_raw if isinstance(item, dict)],
            pinned_file_blobs_json={
                str(k): {str(sk): str(sv) for sk, sv in v.items()}
                for k, v in blobs_raw.items()
                if isinstance(v, dict)
            },
            pinned_symbols_json=[str(s) for s in syms_raw if str(s)],
            compacted_summary=str(row[9] or ""),
            total_tokens_compacted=int(str(row[10] or 0)),
            created_at=_as_opt_dt(row[11]),
            updated_at=_as_opt_dt(row[12]),
        )

    def get_scout_session(
        self,
        conn: Connection[tuple[object, ...]],
        session_id: str,
        *,
        principal_id: str | None = None,
    ) -> ScoutSessionRecord | None:
        if principal_id is not None:
            row = conn.execute(
                """
                SELECT tenant_id, session_id, domain_id, principal_id, allowed_scopes,
                       turn_count, recent_turns_json, pinned_file_blobs_json,
                       pinned_symbols_json, compacted_summary, total_tokens_compacted,
                       created_at, updated_at
                FROM scout_sessions
                WHERE session_id = %s AND principal_id = %s;
                """,
                (session_id, principal_id),
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT tenant_id, session_id, domain_id, principal_id, allowed_scopes,
                       turn_count, recent_turns_json, pinned_file_blobs_json,
                       pinned_symbols_json, compacted_summary, total_tokens_compacted,
                       created_at, updated_at
                FROM scout_sessions
                WHERE session_id = %s;
                """,
                (session_id,),
            ).fetchone()
        return self._row_to_scout_session(row) if row is not None else None

    def upsert_scout_session(
        self,
        conn: Connection[tuple[object, ...]],
        record: ScoutSessionRecord,
    ) -> ScoutSessionRecord:
        existing_owner_row = conn.execute(
            "SELECT principal_id FROM scout_sessions WHERE session_id = %s;",
            (record.session_id,),
        ).fetchone()
        if existing_owner_row is not None and str(existing_owner_row[0]) != record.principal_id:
            raise PermissionError(
                f"Session '{record.session_id}' belongs to principal "
                f"'{existing_owner_row[0]}', not '{record.principal_id}'."
            )
        row = conn.execute(
            """
            INSERT INTO scout_sessions (
                tenant_id, session_id, domain_id, principal_id, allowed_scopes,
                turn_count, compacted_summary, recent_turns_json,
                pinned_file_blobs_json, pinned_symbols_json,
                total_tokens_compacted, created_at, updated_at
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s,
                COALESCE(%s, NOW()), COALESCE(%s, NOW())
            )
            ON CONFLICT (tenant_id, session_id) DO UPDATE SET
                domain_id = EXCLUDED.domain_id,
                principal_id = EXCLUDED.principal_id,
                allowed_scopes = EXCLUDED.allowed_scopes,
                turn_count = EXCLUDED.turn_count,
                compacted_summary = EXCLUDED.compacted_summary,
                recent_turns_json = EXCLUDED.recent_turns_json,
                pinned_file_blobs_json = EXCLUDED.pinned_file_blobs_json,
                pinned_symbols_json = EXCLUDED.pinned_symbols_json,
                total_tokens_compacted = EXCLUDED.total_tokens_compacted,
                updated_at = NOW()
            WHERE scout_sessions.principal_id = EXCLUDED.principal_id
            RETURNING tenant_id, session_id, domain_id, principal_id, allowed_scopes,
                      turn_count, recent_turns_json, pinned_file_blobs_json,
                      pinned_symbols_json, compacted_summary, total_tokens_compacted,
                      created_at, updated_at;
            """,
            (
                record.tenant_id,
                record.session_id,
                record.domain_id,
                record.principal_id,
                list(record.allowed_scopes),
                record.turn_count,
                record.compacted_summary,
                json.dumps(record.recent_turns_json),
                json.dumps(record.pinned_file_blobs_json),
                json.dumps(record.pinned_symbols_json),
                record.total_tokens_compacted,
                record.created_at,
                record.updated_at,
            ),
        ).fetchone()
        if row is None:
            return record
        return self._row_to_scout_session(row)

    def delete_scout_session(
        self,
        conn: Connection[tuple[object, ...]],
        session_id: str,
    ) -> int:
        cur = conn.execute("DELETE FROM scout_sessions WHERE session_id = %s;", (session_id,))
        return max(cur.rowcount or 0, 0)

    @staticmethod
    def _validate_model_artifact_record(record: ModelArtifactRecord) -> None:
        if record.lora_rank not in {8, 16, 32}:
            raise ValueError(
                f"model_artifacts lora_rank must be in {{8, 16, 32}}, got {record.lora_rank}"
            )
        if record.lora_alpha <= 0:
            raise ValueError("model_artifacts lora_alpha must be > 0")
        if not record.use_rslora:
            raise ValueError("model_artifacts requires use_rslora=True (vLLM multi-LoRA safety)")
        if record.use_dora:
            raise ValueError("model_artifacts prohibits use_dora=True (vLLM multi-LoRA safety)")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", record.base_model_revision_sha256.strip()):
            raise ValueError("base_model_revision_sha256 must be a 64-char SHA-256 hex digest")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", record.weights_sha256.strip()):
            raise ValueError("weights_sha256 must be a 64-char SHA-256 hex digest")
        if not [s for s in record.training_scope_ceiling if s.strip()]:
            raise ValueError("training_scope_ceiling must be non-empty (OWASP ASI06)")
        if record.canary_leak_count != 0:
            raise ValueError("canary_leak_count must equal 0 (OWASP ASI06)")
        if record.status == "revoked" and record.is_active:
            raise PermissionError("Revoked model artifact cannot be registered with is_active=True")

    @staticmethod
    def _row_to_model_artifact(row: tuple[object, ...]) -> ModelArtifactRecord:
        raw_meta = row[20] if isinstance(row[20], dict) else json.loads(str(row[20] or "{}"))
        return ModelArtifactRecord(
            tenant_id=UUID(str(row[0])),
            domain_id=str(row[1]),
            artifact_id=str(row[2]),
            artifact_kind=cast(ModelArtifactKind, str(row[3])),
            base_model_id=str(row[4]),
            base_model_revision_sha256=str(row[5]),
            storage_uri=str(row[6]),
            weights_object_key=str(row[7]),
            config_object_key=str(row[8]),
            signature_object_key=str(row[9] or ""),
            weights_sha256=str(row[10]),
            cosign_signature_b64=str(row[11] or ""),
            cosign_key_id=str(row[12] or "default"),
            lora_rank=int(str(row[13])),
            lora_alpha=int(str(row[14])),
            use_rslora=bool(row[15]),
            use_dora=bool(row[16]),
            training_scope_ceiling=[str(s) for s in cast(Sequence[object], row[17] or [])],
            allowed_scopes=[str(s) for s in cast(Sequence[object], row[18] or [])],
            canary_leak_count=int(str(row[19] or 0)),
            metadata_json={str(k): v for k, v in raw_meta.items()},
            status=cast(ModelArtifactStatus, str(row[21])),
            is_active=bool(row[22]),
            created_at=_as_opt_dt(row[23]),
            activated_at=_as_opt_dt(row[24]),
            updated_at=_as_opt_dt(row[25]),
        )

    def register_model_artifact(
        self,
        conn: Connection[tuple[object, ...]],
        record: ModelArtifactRecord,
    ) -> ModelArtifactRecord:
        """Registers or updates a governed model artifact under Forced RLS."""
        self._validate_model_artifact_record(record)
        conn.execute(
            "SELECT pg_advisory_xact_lock("
            "hashtextextended("
            "COALESCE(current_setting('app.tenant_id', true), '') || ':model_artifacts:' "
            "|| %s || ':' || %s, 0)"
            ");",
            (record.domain_id, record.artifact_kind),
        )
        existing_status = conn.execute(
            """
            SELECT status
            FROM model_artifacts
            WHERE domain_id = %s AND artifact_id = %s
            LIMIT 1;
            """,
            (record.domain_id, record.artifact_id),
        ).fetchone()
        if existing_status is not None and str(existing_status[0]) == "revoked":
            raise PermissionError(
                f"Revoked model artifact '{record.artifact_id}' cannot be modified or re-activated."
            )
        if record.is_active:
            conn.execute(
                """
                UPDATE model_artifacts
                SET is_active = FALSE,
                    status = CASE WHEN status = 'active' THEN 'retired' ELSE status END,
                    updated_at = NOW()
                WHERE domain_id = %s
                  AND artifact_kind = %s
                  AND artifact_id <> %s
                  AND is_active = TRUE;
                """,
                (record.domain_id, record.artifact_kind, record.artifact_id),
            )
        effective_status: str = "active" if record.is_active else record.status
        row = conn.execute(
            """
            INSERT INTO model_artifacts (
                tenant_id, domain_id, artifact_id, artifact_kind,
                base_model_id, base_model_revision_sha256,
                storage_uri, weights_object_key, config_object_key, signature_object_key,
                weights_sha256, cosign_signature_b64, cosign_key_id,
                lora_rank, lora_alpha, use_rslora, use_dora,
                training_scope_ceiling, allowed_scopes, canary_leak_count,
                metadata_json, status, is_active, activated_at, updated_at
            )
            VALUES (
                %s, %s, %s, %s,
                %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s,
                %s::jsonb, %s, %s,
                CASE WHEN %s THEN COALESCE(%s, NOW()) ELSE %s END,
                NOW()
            )
            ON CONFLICT (tenant_id, domain_id, artifact_id) DO UPDATE SET
                artifact_kind = EXCLUDED.artifact_kind,
                base_model_id = EXCLUDED.base_model_id,
                base_model_revision_sha256 = EXCLUDED.base_model_revision_sha256,
                storage_uri = EXCLUDED.storage_uri,
                weights_object_key = EXCLUDED.weights_object_key,
                config_object_key = EXCLUDED.config_object_key,
                signature_object_key = EXCLUDED.signature_object_key,
                weights_sha256 = EXCLUDED.weights_sha256,
                cosign_signature_b64 = EXCLUDED.cosign_signature_b64,
                cosign_key_id = EXCLUDED.cosign_key_id,
                lora_rank = EXCLUDED.lora_rank,
                lora_alpha = EXCLUDED.lora_alpha,
                use_rslora = EXCLUDED.use_rslora,
                use_dora = EXCLUDED.use_dora,
                training_scope_ceiling = EXCLUDED.training_scope_ceiling,
                allowed_scopes = EXCLUDED.allowed_scopes,
                canary_leak_count = EXCLUDED.canary_leak_count,
                metadata_json = EXCLUDED.metadata_json,
                status = EXCLUDED.status,
                is_active = EXCLUDED.is_active,
                activated_at = CASE
                    WHEN EXCLUDED.is_active THEN COALESCE(model_artifacts.activated_at, NOW())
                    ELSE model_artifacts.activated_at
                END,
                updated_at = NOW()
            RETURNING tenant_id, domain_id, artifact_id, artifact_kind,
                      base_model_id, base_model_revision_sha256,
                      storage_uri, weights_object_key, config_object_key, signature_object_key,
                      weights_sha256, cosign_signature_b64, cosign_key_id,
                      lora_rank, lora_alpha, use_rslora, use_dora,
                      training_scope_ceiling, allowed_scopes, canary_leak_count,
                      metadata_json, status, is_active, created_at, activated_at, updated_at;
            """,
            (
                record.tenant_id,
                record.domain_id,
                record.artifact_id,
                record.artifact_kind,
                record.base_model_id,
                record.base_model_revision_sha256,
                record.storage_uri,
                record.weights_object_key,
                record.config_object_key,
                record.signature_object_key,
                record.weights_sha256,
                record.cosign_signature_b64,
                record.cosign_key_id,
                record.lora_rank,
                record.lora_alpha,
                record.use_rslora,
                record.use_dora,
                list(record.training_scope_ceiling),
                list(record.allowed_scopes),
                record.canary_leak_count,
                json.dumps(record.metadata_json),
                effective_status,
                record.is_active,
                record.is_active,
                record.activated_at,
                record.activated_at,
            ),
        ).fetchone()
        if row is None:
            return record
        return self._row_to_model_artifact(row)

    def activate_model_artifact(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        artifact_id: str,
        *,
        artifact_kind: ModelArtifactKind = "lora_adapter",
    ) -> ModelArtifactRecord:
        """Atomically swaps the active model_artifact pointer for (tenant_id, domain_id, kind)."""
        conn.execute(
            "SELECT pg_advisory_xact_lock("
            "hashtextextended("
            "COALESCE(current_setting('app.tenant_id', true), '') || ':model_artifacts:' "
            "|| %s || ':' || %s, 0)"
            ");",
            (domain_id, artifact_kind),
        )
        existing = conn.execute(
            """
            SELECT status
            FROM model_artifacts
            WHERE domain_id = %s AND artifact_id = %s AND artifact_kind = %s
            LIMIT 1;
            """,
            (domain_id, artifact_id, artifact_kind),
        ).fetchone()
        if existing is None:
            raise KeyError(
                f"Model artifact '{artifact_id}' ({artifact_kind}) not registered for '{domain_id}'"
            )
        if str(existing[0]) == "revoked":
            raise PermissionError(f"Revoked model artifact '{artifact_id}' cannot be re-activated.")

        conn.execute(
            """
            UPDATE model_artifacts
            SET is_active = FALSE,
                status = CASE WHEN status = 'active' THEN 'retired' ELSE status END,
                updated_at = NOW()
            WHERE domain_id = %s
              AND artifact_kind = %s
              AND artifact_id <> %s
              AND is_active = TRUE;
            """,
            (domain_id, artifact_kind, artifact_id),
        )
        row = conn.execute(
            """
            UPDATE model_artifacts
            SET is_active = TRUE,
                status = 'active',
                activated_at = COALESCE(activated_at, NOW()),
                updated_at = NOW()
            WHERE domain_id = %s AND artifact_id = %s AND artifact_kind = %s
            RETURNING tenant_id, domain_id, artifact_id, artifact_kind,
                      base_model_id, base_model_revision_sha256,
                      storage_uri, weights_object_key, config_object_key, signature_object_key,
                      weights_sha256, cosign_signature_b64, cosign_key_id,
                      lora_rank, lora_alpha, use_rslora, use_dora,
                      training_scope_ceiling, allowed_scopes, canary_leak_count,
                      metadata_json, status, is_active, created_at, activated_at, updated_at;
            """,
            (domain_id, artifact_id, artifact_kind),
        ).fetchone()
        if row is None:
            raise KeyError(
                f"Model artifact '{artifact_id}' ({artifact_kind}) not found during activation"
            )
        return self._row_to_model_artifact(row)

    def retire_model_artifact(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        artifact_id: str,
        *,
        revoke: bool = False,
    ) -> bool:
        target_status = "revoked" if revoke else "retired"
        cur = conn.execute(
            """
            UPDATE model_artifacts
            SET is_active = FALSE,
                status = %s,
                updated_at = NOW()
            WHERE domain_id = %s AND artifact_id = %s;
            """,
            (target_status, domain_id, artifact_id),
        )
        return bool((cur.rowcount or 0) > 0)

    def get_active_model_artifact(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        *,
        artifact_kind: ModelArtifactKind = "lora_adapter",
    ) -> ModelArtifactRecord | None:
        row = conn.execute(
            """
            SELECT tenant_id, domain_id, artifact_id, artifact_kind,
                   base_model_id, base_model_revision_sha256,
                   storage_uri, weights_object_key, config_object_key, signature_object_key,
                   weights_sha256, cosign_signature_b64, cosign_key_id,
                   lora_rank, lora_alpha, use_rslora, use_dora,
                   training_scope_ceiling, allowed_scopes, canary_leak_count,
                   metadata_json, status, is_active, created_at, activated_at, updated_at
            FROM model_artifacts
            WHERE domain_id = %s AND artifact_kind = %s AND is_active = TRUE
            LIMIT 1;
            """,
            (domain_id, artifact_kind),
        ).fetchone()
        return self._row_to_model_artifact(row) if row is not None else None

    def list_active_model_artifacts(
        self,
        conn: Connection[tuple[object, ...]],
        *,
        artifact_kind: ModelArtifactKind = "lora_adapter",
    ) -> list[ModelArtifactRecord]:
        rows = conn.execute(
            """
            SELECT tenant_id, domain_id, artifact_id, artifact_kind,
                   base_model_id, base_model_revision_sha256,
                   storage_uri, weights_object_key, config_object_key, signature_object_key,
                   weights_sha256, cosign_signature_b64, cosign_key_id,
                   lora_rank, lora_alpha, use_rslora, use_dora,
                   training_scope_ceiling, allowed_scopes, canary_leak_count,
                   metadata_json, status, is_active, created_at, activated_at, updated_at
            FROM model_artifacts
            WHERE artifact_kind = %s AND is_active = TRUE
            ORDER BY domain_id, artifact_id;
            """,
            (artifact_kind,),
        ).fetchall()
        return [self._row_to_model_artifact(r) for r in rows]

    def list_model_artifacts(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        *,
        artifact_kind: ModelArtifactKind | None = None,
    ) -> list[ModelArtifactRecord]:
        if artifact_kind is not None:
            rows = conn.execute(
                """
                SELECT tenant_id, domain_id, artifact_id, artifact_kind,
                       base_model_id, base_model_revision_sha256,
                       storage_uri, weights_object_key, config_object_key, signature_object_key,
                       weights_sha256, cosign_signature_b64, cosign_key_id,
                       lora_rank, lora_alpha, use_rslora, use_dora,
                       training_scope_ceiling, allowed_scopes, canary_leak_count,
                       metadata_json, status, is_active, created_at, activated_at, updated_at
                FROM model_artifacts
                WHERE domain_id = %s AND artifact_kind = %s
                ORDER BY created_at DESC, artifact_id ASC;
                """,
                (domain_id, artifact_kind),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT tenant_id, domain_id, artifact_id, artifact_kind,
                       base_model_id, base_model_revision_sha256,
                       storage_uri, weights_object_key, config_object_key, signature_object_key,
                       weights_sha256, cosign_signature_b64, cosign_key_id,
                       lora_rank, lora_alpha, use_rslora, use_dora,
                       training_scope_ceiling, allowed_scopes, canary_leak_count,
                       metadata_json, status, is_active, created_at, activated_at, updated_at
                FROM model_artifacts
                WHERE domain_id = %s
                ORDER BY created_at DESC, artifact_id ASC;
                """,
                (domain_id,),
            ).fetchall()
        return [self._row_to_model_artifact(r) for r in rows]


class InMemoryRLSRepository:
    """Deterministic in-memory twin enforcing identical RLS scope intersection and 3-way RRF.

    Used for fast unit testing and offline verification when a Postgres 17 daemon is not active.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._files: dict[UUID, CodeFileRecord] = {}
        self._symbols: dict[UUID, ASTSymbolRecord] = {}
        self._edges: dict[UUID, ASTEdgeRecord] = {}
        self._docs: dict[UUID, DocChunkRecord] = {}
        self._telemetry: dict[UUID, ExecutionTelemetryRecord] = {}
        self._repos: dict[UUID, RepositoryRecord] = {}
        self._commits: dict[UUID, RepoCommitRecord] = {}
        self._confluence: dict[UUID, ConfluencePageRecord] = {}
        self._deleted_confluence: set[tuple[UUID, str, str]] = set()
        self._jira: dict[UUID, JiraIssueRecord] = {}
        self._embedding_spaces: dict[tuple[UUID, str, str], EmbeddingSpaceRecord] = {}
        self._shadow_symbol_vecs: dict[tuple[UUID, UUID, str], list[float]] = {}
        self._shadow_doc_vecs: dict[tuple[UUID, UUID, str], list[float]] = {}
        self._ref_snapshots: dict[UUID, RefSnapshotRecord] = {}
        self._overlay_deltas: dict[tuple[UUID, UUID, str], OverlayFileDeltaRecord] = {}
        self._side_effects: dict[UUID, SideEffectRecord] = {}
        self._watermarks: dict[tuple[UUID, str, str, str], ConnectorWatermarkRecord] = {}
        self._sessions: dict[tuple[UUID, str], ScoutSessionRecord] = {}
        self._model_artifacts: dict[tuple[UUID, str, str], ModelArtifactRecord] = {}
        self._sync_manifests: dict[tuple[UUID, str], DomainSyncManifestRecord] = {}

    def upsert_domain_sync_manifest(
        self,
        principal: PrincipalContext,
        record: DomainSyncManifestRecord,
    ) -> None:
        if record.tenant_id != principal.tenant_id:
            raise PermissionError("Principal tenant_id mismatch on upsert_domain_sync_manifest")
        with self._lock:
            key = (principal.tenant_id, record.domain_id)
            prev = self._sync_manifests.get(key)
            self._sync_manifests[key] = DomainSyncManifestRecord(
                tenant_id=principal.tenant_id,
                domain_id=record.domain_id,
                kind=record.kind,
                source_dir=record.source_dir,
                repo_commit_shas=dict(record.repo_commit_shas),
                empty_doc_shas=dict(record.empty_doc_shas),
                code_files_indexed=record.code_files_indexed,
                ast_symbols_indexed=record.ast_symbols_indexed,
                ast_edges_indexed=record.ast_edges_indexed,
                doc_chunks_indexed=record.doc_chunks_indexed,
                s3_manifest_uri=record.s3_manifest_uri
                if record.s3_manifest_uri is not None
                else (prev.s3_manifest_uri if prev is not None else None),
                manifest_sha256=record.manifest_sha256
                if record.manifest_sha256 is not None
                else (prev.manifest_sha256 if prev is not None else None),
                synced_at=record.synced_at or datetime.now(UTC),
            )

    def get_domain_sync_manifest(
        self,
        principal: PrincipalContext,
        domain_id: str,
    ) -> DomainSyncManifestRecord | None:
        with self._lock:
            return self._sync_manifests.get((principal.tenant_id, domain_id))

    def register_embedding_space(
        self,
        principal: PrincipalContext,
        record: EmbeddingSpaceRecord,
    ) -> EmbeddingSpaceRecord:
        if record.tenant_id != principal.tenant_id:
            raise PermissionError("Principal tenant_id mismatch on register_embedding_space")
        if record.dimensions < 128 or record.dimensions > 4096:
            raise ValueError(
                f"Embedding space dimensions must be in [128, 4096], got {record.dimensions}"
            )
        with self._lock:
            if record.is_active:
                for key, existing in list(self._embedding_spaces.items()):
                    if (
                        key[0] == principal.tenant_id
                        and key[1] == record.domain_id
                        and key[2] != record.space_id
                        and existing.is_active
                    ):
                        self._embedding_spaces[key] = dataclasses.replace(
                            existing,
                            is_active=False,
                            status="retired" if existing.status == "active" else existing.status,
                        )
            effective_status: Any = "active" if record.is_active else record.status
            stored = dataclasses.replace(record, status=effective_status)
            self._embedding_spaces[(principal.tenant_id, record.domain_id, record.space_id)] = (
                stored
            )
            return stored

    def get_active_embedding_space(
        self,
        principal: PrincipalContext,
        domain_id: str,
    ) -> EmbeddingSpaceRecord | None:
        with self._lock:
            for (tid, did, _), rec in self._embedding_spaces.items():
                if tid == principal.tenant_id and did == domain_id and rec.is_active:
                    return rec
            return None

    def list_embedding_spaces(
        self,
        principal: PrincipalContext,
        domain_id: str,
    ) -> list[EmbeddingSpaceRecord]:
        with self._lock:
            items = [
                rec
                for (tid, did, _), rec in self._embedding_spaces.items()
                if tid == principal.tenant_id and did == domain_id
            ]
        items.sort(key=lambda r: (not r.is_active, r.space_id))
        return items

    def stage_shadow_embeddings(
        self,
        principal: PrincipalContext,
        domain_id: str,
        shadow_space_id: str,
        *,
        symbol_embeddings: Mapping[UUID, Sequence[float]] | None = None,
        chunk_embeddings: Mapping[UUID, Sequence[float]] | None = None,
    ) -> dict[str, int]:
        with self._lock:
            space = self._embedding_spaces.get((principal.tenant_id, domain_id, shadow_space_id))
            if space is None:
                raise KeyError(
                    f"Shadow embedding_space '{shadow_space_id}' not registered for '{domain_id}'"
                )
            syms_staged = 0
            if symbol_embeddings:
                for sym_id, vec in symbol_embeddings.items():
                    checked = validate_non_zero_embedding(vec, expected_dim=space.dimensions)
                    sym = self._symbols.get(sym_id)
                    if (
                        sym is not None
                        and sym.tenant_id == principal.tenant_id
                        and sym.domain_id == domain_id
                    ):
                        self._shadow_symbol_vecs[(principal.tenant_id, sym_id, shadow_space_id)] = (
                            checked
                        )
                        syms_staged += 1
            chks_staged = 0
            if chunk_embeddings:
                for chk_id, vec in chunk_embeddings.items():
                    checked = validate_non_zero_embedding(vec, expected_dim=space.dimensions)
                    chk = self._docs.get(chk_id)
                    if (
                        chk is not None
                        and chk.tenant_id == principal.tenant_id
                        and chk.domain_id == domain_id
                    ):
                        self._shadow_doc_vecs[(principal.tenant_id, chk_id, shadow_space_id)] = (
                            checked
                        )
                        chks_staged += 1
            return {"symbols_staged": syms_staged, "chunks_staged": chks_staged}

    def reindex_domain_shadow_space(
        self,
        principal: PrincipalContext,
        domain_id: str,
        shadow_space: EmbeddingSpaceRecord,
        embedder: Any,
    ) -> dict[str, int]:
        staged_rec = dataclasses.replace(shadow_space, is_active=False, status="shadow")
        self.register_embedding_space(principal, staged_rec)
        syms = self.list_visible_ast_symbols(principal, domain_id)
        sym_map: dict[UUID, Sequence[float]] = {}
        if syms:
            sym_vecs = embedder.embed_batch(
                [
                    f"{s.qualified_name} {s.signature_skeleton} {s.contextual_summary}".strip()
                    for s in syms
                ],
                is_query=False,
            )
            for s, v in zip(syms, sym_vecs, strict=True):
                sym_map[s.symbol_id] = v
        chks = self.list_visible_doc_chunks(principal, domain_id)
        chk_map: dict[UUID, Sequence[float]] = {}
        if chks:
            chk_vecs = embedder.embed_batch(
                [
                    (
                        f"{c.doc_title} {c.section_heading} "
                        f"{c.contextual_header} {c.verbatim_content}"
                    ).strip()
                    for c in chks
                ],
                is_query=False,
            )
            for c, v in zip(chks, chk_vecs, strict=True):
                chk_map[c.chunk_id] = v
        return self.stage_shadow_embeddings(
            principal,
            domain_id,
            shadow_space.space_id,
            symbol_embeddings=sym_map,
            chunk_embeddings=chk_map,
        )

    def activate_embedding_space(
        self,
        principal: PrincipalContext,
        domain_id: str,
        target_space_id: str,
        *,
        require_complete_shadow: bool = True,
    ) -> EmbeddingSpaceRecord:
        space_key = (principal.tenant_id, domain_id, target_space_id)
        with self._lock:
            if space_key not in self._embedding_spaces:
                raise KeyError(
                    f"Target embedding_space '{target_space_id}' not registered for '{domain_id}'"
                )
            if require_complete_shadow:
                for sym in self._symbols.values():
                    if (
                        sym.tenant_id == principal.tenant_id
                        and sym.domain_id == domain_id
                        and _scopes_intersect(sym.allowed_scopes, principal.allowed_scopes)
                        and sym.embedding_space_id != target_space_id
                        and (principal.tenant_id, sym.symbol_id, target_space_id)
                        not in self._shadow_symbol_vecs
                    ):
                        raise RuntimeError(
                            f"Symbol {sym.symbol_id} missing staged shadow "
                            f"embedding for '{target_space_id}'"
                        )
                for chk in self._docs.values():
                    if (
                        chk.tenant_id == principal.tenant_id
                        and chk.domain_id == domain_id
                        and _scopes_intersect(chk.allowed_scopes, principal.allowed_scopes)
                        and not chk.is_stale
                        and chk.embedding_space_id != target_space_id
                        and (principal.tenant_id, chk.chunk_id, target_space_id)
                        not in self._shadow_doc_vecs
                    ):
                        raise RuntimeError(
                            f"Chunk {chk.chunk_id} missing staged shadow "
                            f"embedding for '{target_space_id}'"
                        )
            for sid, sym in list(self._symbols.items()):
                sh_vec = self._shadow_symbol_vecs.pop(
                    (principal.tenant_id, sid, target_space_id), None
                )
                if sh_vec is not None:
                    self._symbols[sid] = dataclasses.replace(
                        sym,
                        embedding=sh_vec,
                        embedding_space_id=target_space_id,
                    )
            for cid, chk in list(self._docs.items()):
                sh_vec = self._shadow_doc_vecs.pop(
                    (principal.tenant_id, cid, target_space_id), None
                )
                if sh_vec is not None:
                    self._docs[cid] = dataclasses.replace(
                        chk,
                        embedding=sh_vec,
                        embedding_space_id=target_space_id,
                    )
            for key, existing in list(self._embedding_spaces.items()):
                if key[0] == principal.tenant_id and key[1] == domain_id:
                    if key[2] == target_space_id:
                        self._embedding_spaces[key] = dataclasses.replace(
                            existing,
                            is_active=True,
                            status="active",
                        )
                    elif existing.is_active:
                        self._embedding_spaces[key] = dataclasses.replace(
                            existing,
                            is_active=False,
                            status="retired",
                        )
            return self._embedding_spaces[space_key]

    def insert_repository(self, principal: PrincipalContext, record: RepositoryRecord) -> None:
        if record.tenant_id != principal.tenant_id:
            raise PermissionError("Principal tenant_id mismatch on insert_repository")
        if not _connector_ingest_or_scopes_intersect(
            record.allowed_scopes, principal.allowed_scopes
        ):
            raise PermissionError("Principal lacks scope to insert repository row")
        # Upsert: purge existing record for (tenant_id, domain_id, repo_name) to mirror ON CONFLICT
        for old_id, r in list(self._repos.items()):
            if (
                r.tenant_id == record.tenant_id
                and r.domain_id == record.domain_id
                and r.repo_name == record.repo_name
            ):
                self._repos.pop(old_id, None)
        self._repos[record.repo_id] = record

    def upsert_repository(self, principal: PrincipalContext, record: RepositoryRecord) -> None:
        self.insert_repository(principal, record)

    def get_repository(
        self, principal: PrincipalContext, domain_id: str, repo_name: str
    ) -> RepositoryRecord | None:
        for r in self._repos.values():
            if (
                r.tenant_id == principal.tenant_id
                and r.domain_id == domain_id
                and r.repo_name == repo_name
                and _connector_ingest_or_scopes_intersect(
                    r.allowed_scopes, principal.allowed_scopes
                )
            ):
                return r
        return None

    def list_repositories(
        self, principal: PrincipalContext, domain_id: str
    ) -> list[RepositoryRecord]:
        return [
            r
            for r in self._repos.values()
            if r.tenant_id == principal.tenant_id
            and r.domain_id == domain_id
            and _connector_ingest_or_scopes_intersect(r.allowed_scopes, principal.allowed_scopes)
        ]

    def insert_repo_commit(self, principal: PrincipalContext, record: RepoCommitRecord) -> None:
        if record.tenant_id != principal.tenant_id:
            raise PermissionError("Principal tenant_id mismatch on insert_repo_commit")
        self._commits[record.commit_id] = record

    def get_repo_commits(
        self, principal: PrincipalContext, repo_id: UUID
    ) -> list[RepoCommitRecord]:
        visible_repo_ids = {
            r.repo_id
            for r in self._repos.values()
            if r.tenant_id == principal.tenant_id
            and r.repo_id == repo_id
            and _connector_ingest_or_scopes_intersect(r.allowed_scopes, principal.allowed_scopes)
        }
        return [
            c
            for c in self._commits.values()
            if c.tenant_id == principal.tenant_id
            and c.repo_id == repo_id
            and (not self._repos or c.repo_id in visible_repo_ids)
        ]

    def insert_confluence_page(
        self, principal: PrincipalContext, record: ConfluencePageRecord
    ) -> None:
        if record.tenant_id != principal.tenant_id:
            raise PermissionError("Principal tenant_id mismatch on insert_confluence_page")
        if not _connector_ingest_or_scopes_intersect(
            record.allowed_scopes, principal.allowed_scopes
        ):
            raise PermissionError("Principal lacks scope to insert confluence_page row")
        self._deleted_confluence.discard(
            (record.tenant_id, record.domain_id, record.confluence_page_id)
        )
        # Upsert: purge existing record for (tenant_id, domain_id, confluence_page_id)
        for old_id, p in list(self._confluence.items()):
            if (
                p.tenant_id == record.tenant_id
                and p.domain_id == record.domain_id
                and p.confluence_page_id == record.confluence_page_id
            ):
                self._confluence.pop(old_id, None)
        self._confluence[record.page_id] = record

    def get_confluence_page(
        self,
        principal: PrincipalContext,
        domain_id: str,
        confluence_page_id: str,
        include_deleted: bool = False,
    ) -> ConfluencePageRecord | None:
        if (
            not include_deleted
            and (principal.tenant_id, domain_id, confluence_page_id) in self._deleted_confluence
        ):
            return None
        for p in self._confluence.values():
            if (
                p.tenant_id == principal.tenant_id
                and p.domain_id == domain_id
                and p.confluence_page_id == confluence_page_id
                and _connector_ingest_or_scopes_intersect(
                    p.allowed_scopes, principal.allowed_scopes
                )
            ):
                return p
        return None

    def list_confluence_pages(
        self,
        principal: PrincipalContext,
        domain_id: str,
        space_key: str | None = None,
        include_deleted: bool = False,
    ) -> list[ConfluencePageRecord]:
        return [
            p
            for p in self._confluence.values()
            if p.tenant_id == principal.tenant_id
            and p.domain_id == domain_id
            and (space_key is None or p.space_key == space_key)
            and (
                include_deleted
                or (p.tenant_id, p.domain_id, p.confluence_page_id) not in self._deleted_confluence
            )
            and _connector_ingest_or_scopes_intersect(p.allowed_scopes, principal.allowed_scopes)
        ]

    def tombstone_confluence_pages(
        self,
        domain_id: str,
        confluence_page_ids: list[str],
        reason: str = "Page deleted or archived in Confluence",
        status: str = "deleted",
        *,
        tenant_id: UUID = DEFAULT_TENANT_ID,
    ) -> tuple[int, int]:
        """Soft-deletes Confluence pages and marks their active doc_chunks stale in memory."""
        import dataclasses

        del status
        clean_ids = {pid for pid in confluence_page_ids if pid}
        if not clean_ids:
            return (0, 0)

        pages_tombstoned = 0
        for pid in clean_ids:
            key = (tenant_id, domain_id, pid)
            had_page = any(
                p.tenant_id == tenant_id
                and p.domain_id == domain_id
                and p.confluence_page_id == pid
                for p in self._confluence.values()
            )
            if key not in self._deleted_confluence:
                self._deleted_confluence.add(key)
                if had_page:
                    pages_tombstoned += 1

        target_doc_ids = {f"confluence:{pid}" for pid in clean_ids}
        chunks_flagged = 0
        for cid, doc in list(self._docs.items()):
            if (
                doc.tenant_id == tenant_id
                and doc.domain_id == domain_id
                and doc.doc_id in target_doc_ids
                and not doc.is_stale
            ):
                self._docs[cid] = dataclasses.replace(
                    doc,
                    is_stale=True,
                    stale_reason=reason,
                )
                chunks_flagged += 1
        return (pages_tombstoned, chunks_flagged)

    def update_confluence_allowed_scopes(
        self,
        domain_id: str,
        confluence_page_id: str,
        allowed_scopes: list[str],
        *,
        tenant_id: UUID = DEFAULT_TENANT_ID,
    ) -> int:
        """Synchronizes allowed_scopes across in-memory confluence_pages and active doc_chunks."""
        import dataclasses

        scopes_copy = list(allowed_scopes)
        for pid_uuid, page_rec in list(self._confluence.items()):
            if (
                page_rec.tenant_id == tenant_id
                and page_rec.domain_id == domain_id
                and page_rec.confluence_page_id == confluence_page_id
            ):
                self._confluence[pid_uuid] = dataclasses.replace(
                    page_rec, allowed_scopes=scopes_copy
                )

        target_doc_id = f"confluence:{confluence_page_id}"
        updated_chunks = 0
        for cid, doc in list(self._docs.items()):
            if (
                doc.tenant_id == tenant_id
                and doc.domain_id == domain_id
                and doc.doc_id == target_doc_id
                and not doc.is_stale
            ):
                self._docs[cid] = dataclasses.replace(doc, allowed_scopes=scopes_copy)
                updated_chunks += 1
        return updated_chunks

    def insert_jira_issue(self, principal: PrincipalContext, record: JiraIssueRecord) -> None:
        if record.tenant_id != principal.tenant_id:
            raise PermissionError("Principal tenant_id mismatch on insert_jira_issue")
        if not _connector_ingest_or_scopes_intersect(
            record.allowed_scopes, principal.allowed_scopes
        ):
            raise PermissionError("Principal lacks scope to insert jira_issue row")
        # Upsert: purge existing record for (tenant_id, domain_id, issue_key) to mirror ON CONFLICT
        for old_id, i in list(self._jira.items()):
            if (
                i.tenant_id == record.tenant_id
                and i.domain_id == record.domain_id
                and i.issue_key == record.issue_key
            ):
                self._jira.pop(old_id, None)
        self._jira[record.issue_id] = record

    def get_jira_issue(
        self, principal: PrincipalContext, domain_id: str, issue_key: str
    ) -> JiraIssueRecord | None:
        for i in self._jira.values():
            if (
                i.tenant_id == principal.tenant_id
                and i.domain_id == domain_id
                and i.issue_key == issue_key
                and _connector_ingest_or_scopes_intersect(
                    i.allowed_scopes, principal.allowed_scopes
                )
            ):
                return i
        return None

    def list_jira_issues(
        self, principal: PrincipalContext, domain_id: str, project_key: str | None = None
    ) -> list[JiraIssueRecord]:
        return [
            i
            for i in self._jira.values()
            if i.tenant_id == principal.tenant_id
            and i.domain_id == domain_id
            and (project_key is None or i.project_key == project_key)
            and _connector_ingest_or_scopes_intersect(i.allowed_scopes, principal.allowed_scopes)
        ]

    def insert_code_file(self, principal: PrincipalContext, record: CodeFileRecord) -> None:
        if record.tenant_id != principal.tenant_id:
            raise PermissionError("Principal tenant_id mismatch on insert_code_file")
        if not _connector_ingest_or_scopes_intersect(
            record.allowed_scopes, principal.allowed_scopes
        ):
            raise PermissionError("Principal lacks scope to insert code_file row")
        self._files[record.file_id] = record

    def insert_ast_symbol(self, principal: PrincipalContext, record: ASTSymbolRecord) -> None:
        if record.tenant_id != principal.tenant_id:
            raise PermissionError("Principal tenant_id mismatch on insert_ast_symbol")
        if not _connector_ingest_or_scopes_intersect(
            record.allowed_scopes, principal.allowed_scopes
        ):
            raise PermissionError("Principal lacks scope to insert ast_symbol row")
        self._symbols[record.symbol_id] = record

    def insert_ast_edge(self, principal: PrincipalContext, record: ASTEdgeRecord) -> None:
        if record.tenant_id != principal.tenant_id:
            raise PermissionError("Principal tenant_id mismatch on insert_ast_edge")
        if not _connector_ingest_or_scopes_intersect(
            record.allowed_scopes, principal.allowed_scopes
        ):
            raise PermissionError("Principal lacks scope to insert ast_edge row")
        self._edges[record.edge_id] = record

    def insert_doc_chunk(self, principal: PrincipalContext, record: DocChunkRecord) -> None:
        import dataclasses

        if record.tenant_id != principal.tenant_id:
            raise PermissionError("Principal tenant_id mismatch on insert_doc_chunk")
        if not _connector_ingest_or_scopes_intersect(
            record.allowed_scopes, principal.allowed_scopes
        ):
            raise PermissionError("Principal lacks scope to insert doc_chunk row")
        normalized = dataclasses.replace(
            record,
            market_scope=_parse_market_scope(_serialize_market_scope(record.market_scope)),
        )
        self._docs[normalized.chunk_id] = normalized

    def insert_doc_chunks_batch(
        self, principal: PrincipalContext, records: list[DocChunkRecord]
    ) -> int:
        for record in records:
            self.insert_doc_chunk(principal, record)
        return len(records)

    def list_visible_code_files(
        self, principal: PrincipalContext, domain_id: str
    ) -> list[CodeFileRecord]:
        return [
            f
            for f in self._files.values()
            if f.tenant_id == principal.tenant_id
            and f.domain_id == domain_id
            and _connector_ingest_or_scopes_intersect(f.allowed_scopes, principal.allowed_scopes)
        ]

    def list_visible_ast_symbols(
        self, principal: PrincipalContext, domain_id: str
    ) -> list[ASTSymbolRecord]:
        return [
            s
            for s in self._symbols.values()
            if s.tenant_id == principal.tenant_id
            and s.domain_id == domain_id
            and _connector_ingest_or_scopes_intersect(s.allowed_scopes, principal.allowed_scopes)
        ]

    def list_visible_ast_edges(
        self, principal: PrincipalContext, domain_id: str
    ) -> list[ASTEdgeRecord]:
        return [
            e
            for e in self._edges.values()
            if e.tenant_id == principal.tenant_id
            and e.domain_id == domain_id
            and _connector_ingest_or_scopes_intersect(e.allowed_scopes, principal.allowed_scopes)
        ]

    def list_visible_doc_chunks(
        self,
        principal: PrincipalContext,
        domain_id: str,
        market_scope: DocumentScope | str | None = None,
        include_stale: bool = False,
    ) -> list[DocChunkRecord]:
        target_mkt = _serialize_market_scope(market_scope)
        return [
            d
            for d in self._docs.values()
            if d.tenant_id == principal.tenant_id
            and d.domain_id == domain_id
            and (include_stale or not d.is_stale)
            and (target_mkt is None or _serialize_market_scope(d.market_scope) == target_mkt)
            and _connector_ingest_or_scopes_intersect(d.allowed_scopes, principal.allowed_scopes)
        ]

    def get_file_blob_hashes(
        self, principal: PrincipalContext, domain_id: str, repo_name: str
    ) -> dict[str, tuple[UUID, str]]:
        return {
            f.file_path: (f.file_id, f.git_blob_sha)
            for f in self._files.values()
            if f.tenant_id == principal.tenant_id
            and f.domain_id == domain_id
            and f.repo_name == repo_name
            and _connector_ingest_or_scopes_intersect(f.allowed_scopes, principal.allowed_scopes)
        }

    def delete_code_file(self, principal: PrincipalContext, file_id: UUID) -> None:
        if file_id in self._files:
            file_rec = self._files[file_id]
            if file_rec.tenant_id != principal.tenant_id:
                raise PermissionError("Principal tenant_id mismatch on delete_code_file")
            if not _scopes_intersect(file_rec.allowed_scopes, principal.allowed_scopes):
                raise PermissionError("Principal lacks scope to delete code_file")
            del self._files[file_id]
            for se_id in [
                se_id
                for se_id, se in self._side_effects.items()
                if se.file_id == file_id
                or (
                    se.tenant_id == file_rec.tenant_id
                    and se.domain_id == file_rec.domain_id
                    and se.repo_name == file_rec.repo_name
                    and se.file_path == file_rec.file_path
                )
            ]:
                self._side_effects.pop(se_id, None)
            deleted_sym_ids = {
                s_id for s_id, sym in self._symbols.items() if sym.file_id == file_id
            }
            for s_id in deleted_sym_ids:
                self._symbols.pop(s_id, None)

            # Cascade delete outward edges originating from deleted symbols
            deleted_source_edge_ids = {
                e_id
                for e_id, edge in self._edges.items()
                if edge.source_symbol_id in deleted_sym_ids
            }
            for e_id in deleted_source_edge_ids:
                self._edges.pop(e_id, None)

            # Preserve inward edges with target_symbol_id = None
            # (matches PostgreSQL ON DELETE SET NULL)
            for e_id, edge in list(self._edges.items()):
                if edge.target_symbol_id in deleted_sym_ids:
                    self._edges[e_id] = ASTEdgeRecord(
                        edge_id=edge.edge_id,
                        domain_id=edge.domain_id,
                        source_symbol_id=edge.source_symbol_id,
                        target_qualified_name=edge.target_qualified_name,
                        target_symbol_id=None,
                        edge_kind=edge.edge_kind,
                        allowed_scopes=edge.allowed_scopes,
                        target_repo_qualified_id=edge.target_repo_qualified_id,
                        is_cross_repo=edge.is_cross_repo,
                    )

    def delete_code_file_and_symbols(
        self,
        domain_id: str,
        repo_name: str,
        file_path: str,
    ) -> None:
        """Deletes code file and all associated AST symbols from in-memory stores."""
        del_ids = [
            fid
            for fid, f in self._files.items()
            if f.domain_id == domain_id
            and f.repo_name in (repo_name, repo_name.replace("/", "_"))
            and f.file_path == file_path
        ]
        for fid in del_ids:
            removed_file = self._files.pop(fid, None)
            for se_id in [
                se_id
                for se_id, se in self._side_effects.items()
                if se.file_id == fid
                or (
                    removed_file is not None
                    and se.tenant_id == removed_file.tenant_id
                    and se.domain_id == domain_id
                    and se.repo_name in (repo_name, repo_name.replace("/", "_"))
                    and se.file_path == file_path
                )
            ]:
                self._side_effects.pop(se_id, None)
            sym_del_ids = [s_id for s_id, sym in self._symbols.items() if sym.file_id == fid]
            for s_id in sym_del_ids:
                self._symbols.pop(s_id, None)
            sym_del_set = set(sym_del_ids)
            # Cascade delete outward edges originating from deleted symbols
            del_edge_ids = [
                e_id for e_id, edge in self._edges.items() if edge.source_symbol_id in sym_del_set
            ]
            for e_id in del_edge_ids:
                self._edges.pop(e_id, None)
            # Preserve inward edges with target_symbol_id = None
            # (matches PostgreSQL ON DELETE SET NULL)
            for e_id, edge in list(self._edges.items()):
                if edge.target_symbol_id in sym_del_set:
                    self._edges[e_id] = ASTEdgeRecord(
                        edge_id=edge.edge_id,
                        domain_id=edge.domain_id,
                        source_symbol_id=edge.source_symbol_id,
                        target_qualified_name=edge.target_qualified_name,
                        target_symbol_id=None,
                        edge_kind=edge.edge_kind,
                        allowed_scopes=edge.allowed_scopes,
                        target_repo_qualified_id=edge.target_repo_qualified_id,
                        is_cross_repo=edge.is_cross_repo,
                    )

    def traverse_1hop_neighborhood(
        self,
        principal_or_conn: Any,
        domain_id: str,
        focus_symbol_id: UUID | str,
        max_direct_callers: int = 3,
    ) -> list[ASTNeighborhoodRecord]:
        """In-memory twin executing identical 1-hop AST neighborhood semantics."""
        focus_uuid = UUID(str(focus_symbol_id))
        focus_sym = self._symbols.get(focus_uuid)
        if not focus_sym or focus_sym.domain_id != domain_id:
            return []
        if isinstance(principal_or_conn, PrincipalContext) and (
            focus_sym.tenant_id != principal_or_conn.tenant_id
        ):
            return []

        principal_scopes = (
            principal_or_conn.allowed_scopes
            if isinstance(principal_or_conn, PrincipalContext)
            else focus_sym.allowed_scopes
        )
        target_tenant_id = (
            principal_or_conn.tenant_id
            if isinstance(principal_or_conn, PrincipalContext)
            else focus_sym.tenant_id
        )
        if not _scopes_intersect(focus_sym.allowed_scopes, principal_scopes):
            return []

        focus_file = self._files.get(focus_sym.file_id)
        focus_repo = focus_file.repo_name if focus_file else None

        records: list[ASTNeighborhoodRecord] = []
        seen_syms: set[UUID] = {focus_sym.symbol_id}

        # 1. Hop 1 Outward edges (USES_TYPE, IMPORTS, INHERITS, same-file CALLS)
        hop1_type_sym_ids: list[UUID] = []
        for e in self._edges.values():
            if (
                e.tenant_id == target_tenant_id
                and e.domain_id == domain_id
                and e.source_symbol_id == focus_sym.symbol_id
            ):
                if not _scopes_intersect(e.allowed_scopes, principal_scopes):
                    continue
                if e.target_symbol_id and e.target_symbol_id in self._symbols:
                    target_sym = self._symbols[e.target_symbol_id]
                    target_file = self._files.get(target_sym.file_id)
                    if not target_file or not _scopes_intersect(
                        target_file.allowed_scopes, principal_scopes
                    ):
                        continue
                    if e.edge_kind == "CALLS" and target_sym.file_id != focus_sym.file_id:
                        continue
                    if target_sym.symbol_id not in seen_syms:
                        seen_syms.add(target_sym.symbol_id)
                        records.append(
                            ASTNeighborhoodRecord(
                                intent_role="imported_type",
                                edge_kind=e.edge_kind,
                                depth=1,
                                symbol=target_sym,
                                file=target_file,
                            )
                        )
                        if e.edge_kind in (
                            "USES_TYPE",
                            "INHERITS",
                            "IMPORTS",
                            "PROVISIONED_BY",
                            "DISPATCHES",
                            "SELECTS",
                            "DOCUMENTS_COMPONENT",
                            "OWNED_BY_TEAM",
                            "PART_OF_SYSTEM",
                            "PUBLISHES_EVENT",
                            "SUBSCRIBES_EVENT",
                            "READS_TABLE",
                            "WRITES_TABLE",
                            "RENDERS_TEMPLATE",
                            "IMPLEMENTS_SPEC",
                            "IMPLEMENTS_ISSUE",
                            "BINDS_SERVICE_ACCOUNT",
                            "ACCESSES_SECRET",
                        ):
                            hop1_type_sym_ids.append(target_sym.symbol_id)

        # Hop 2 Outward: Transitive schema/type/inheritance closure within same repo
        for h1_id in hop1_type_sym_ids:
            for e2 in self._edges.values():
                if (
                    e2.domain_id == domain_id
                    and e2.source_symbol_id == h1_id
                    and e2.edge_kind in ("USES_TYPE", "INHERITS")
                ):
                    if not _scopes_intersect(e2.allowed_scopes, principal_scopes):
                        continue
                    if e2.target_symbol_id and e2.target_symbol_id in self._symbols:
                        h2_sym = self._symbols[e2.target_symbol_id]
                        h2_file = self._files.get(h2_sym.file_id)
                        if not h2_file or not _scopes_intersect(
                            h2_file.allowed_scopes, principal_scopes
                        ):
                            continue
                        if focus_repo and h2_file.repo_name != focus_repo:
                            continue
                        if h2_sym.symbol_id not in seen_syms:
                            seen_syms.add(h2_sym.symbol_id)
                            records.append(
                                ASTNeighborhoodRecord(
                                    intent_role="imported_type",
                                    edge_kind=e2.edge_kind,
                                    depth=2,
                                    symbol=h2_sym,
                                    file=h2_file,
                                )
                            )

        # 2. Inward callers & cross-repo contract consumers with diversity across caller files
        _inward_contract_kinds = (
            "USES_TYPE",
            "PROVISIONED_BY",
            "DOCUMENTS_COMPONENT",
            "PUBLISHES_EVENT",
            "SUBSCRIBES_EVENT",
            "READS_TABLE",
            "WRITES_TABLE",
            "RENDERS_TEMPLATE",
            "IMPLEMENTS_SPEC",
            "IMPLEMENTS_ISSUE",
            "BINDS_SERVICE_ACCOUNT",
            "ACCESSES_SECRET",
        )
        callers_by_file: dict[UUID, tuple[ASTSymbolRecord, CodeFileRecord]] = {}
        for e in self._edges.values():
            if e.domain_id == domain_id and (
                e.edge_kind == "CALLS"
                or (e.is_cross_repo and e.edge_kind in _inward_contract_kinds)
            ):
                matches_target = (
                    e.target_symbol_id == focus_sym.symbol_id
                    or e.target_qualified_name == focus_sym.qualified_name
                )
                if matches_target and _scopes_intersect(e.allowed_scopes, principal_scopes):
                    src_sym = self._symbols.get(e.source_symbol_id)
                    if not src_sym or src_sym.symbol_id in seen_syms:
                        continue
                    src_file = self._files.get(src_sym.file_id)
                    if (
                        src_file
                        and _scopes_intersect(src_file.allowed_scopes, principal_scopes)
                        and (
                            src_file.file_id not in callers_by_file
                            or src_sym.start_line < callers_by_file[src_file.file_id][0].start_line
                        )
                    ):
                        callers_by_file[src_file.file_id] = (src_sym, src_file)

        # Limit to max_direct_callers
        caller_items = list(callers_by_file.values())
        caller_items.sort(
            key=lambda item: (
                0 if item[1].file_id == focus_sym.file_id else 1,
                str(item[1].file_id),
                item[0].start_line,
            )
        )
        for c_sym, c_file in caller_items[:max_direct_callers]:
            seen_syms.add(c_sym.symbol_id)
            records.append(
                ASTNeighborhoodRecord(
                    intent_role="direct_caller",
                    edge_kind="CALLS",
                    depth=1,
                    symbol=c_sym,
                    file=c_file,
                )
            )

        # 3. Test fixtures (TESTED_BY)
        test_count = 0
        for e in self._edges.values():
            if e.domain_id == domain_id and e.edge_kind == "TESTED_BY":
                test_sym_id = (
                    e.target_symbol_id
                    if e.source_symbol_id == focus_sym.symbol_id
                    else e.source_symbol_id
                )
                if not test_sym_id and (e.target_qualified_name == focus_sym.qualified_name):
                    test_sym_id = e.source_symbol_id
                if test_sym_id and test_sym_id in self._symbols and test_sym_id not in seen_syms:
                    t_sym = self._symbols[test_sym_id]
                    t_file = self._files.get(t_sym.file_id)
                    if t_file and _scopes_intersect(t_file.allowed_scopes, principal_scopes):
                        seen_syms.add(t_sym.symbol_id)
                        records.append(
                            ASTNeighborhoodRecord(
                                intent_role="test_fixture",
                                edge_kind="TESTED_BY",
                                depth=1,
                                symbol=t_sym,
                                file=t_file,
                            )
                        )
                        test_count += 1
                        if test_count >= 2:
                            break

        return records

    def search_symbols_hybrid_rrf(
        self,
        principal: PrincipalContext,
        domain_id: str,
        query_text: str,
        query_embedding: list[float],
        top_k: int = 5,
        min_calibrated_score: float = 0.0,
    ) -> list[RankedSearchHit]:
        """Executes RLS-filtered 3-way Normalized RRF (trigram, BM25, cosine similarity)."""
        visible = self.list_visible_ast_symbols(principal, domain_id)
        if not visible:
            return []

        q_lower = query_text.lower()
        q_tokens = _extract_substantive_tokens(query_text) or list(set(q_lower.split()))

        # 1. Trigram / Substring similarity rank
        trgm_raw = {
            s.symbol_id: max(
                _trigram_similarity(s.qualified_name, query_text),
                _trigram_similarity(s.signature_skeleton, query_text),
            )
            for s in visible
        }
        trgm_scored = sorted(
            visible,
            key=lambda s: (
                1.0 if q_lower in s.qualified_name.lower() else 0.0,
                sum(
                    1
                    for t in q_tokens
                    if _token_matches_text(
                        t,
                        f"{s.qualified_name.lower()} "
                        f"{_decompose_identifier_words(s.qualified_name)}",
                    )
                ),
                trgm_raw[s.symbol_id],
            ),
            reverse=True,
        )
        trgm_ranks = {s.symbol_id: idx + 1 for idx, s in enumerate(trgm_scored)}

        # 2. BM25 / Lexical token overlap rank (with CamelCase/snake_case decomposition)
        decomposed_cache = {
            s.symbol_id: (
                f"{s.qualified_name} {_decompose_identifier_words(s.qualified_name)} "
                f"{s.signature_skeleton} {_decompose_identifier_words(s.signature_skeleton)} "
                f"{s.contextual_summary}"
            )
            for s in visible
        }
        corpus_texts = [decomposed_cache[s.symbol_id] for s in visible]
        bm25_model = BM25Okapi(corpus_texts)
        bm25_raw_list = bm25_model.get_scores(query_text)
        bm25_raw = {visible[i].symbol_id: float(bm25_raw_list[i]) for i in range(len(visible))}
        bm25_scored = sorted(
            visible,
            key=lambda s: (
                sum(
                    1
                    for t in q_tokens
                    if _token_matches_text(t, decomposed_cache[s.symbol_id].lower())
                ),
                bm25_raw[s.symbol_id],
            ),
            reverse=True,
        )
        bm25_ranks = {s.symbol_id: idx + 1 for idx, s in enumerate(bm25_scored)}

        # 3. Dense Vector cosine similarity rank
        vec_raw = {s.symbol_id: _cosine_similarity(s.embedding, query_embedding) for s in visible}
        vec_scored = sorted(
            visible,
            key=lambda s: vec_raw[s.symbol_id],
            reverse=True,
        )
        vec_ranks = {s.symbol_id: idx + 1 for idx, s in enumerate(vec_scored)}

        hits: list[CalibratedRankedSearchHit] = []
        for sym in visible:
            r_trgm = trgm_ranks[sym.symbol_id]
            r_bm25 = bm25_ranks[sym.symbol_id]
            r_vec = vec_ranks[sym.symbol_id]
            norm_rrf = compute_normalized_rrf([r_trgm, r_bm25, r_vec], num_modalities=3, k=60)
            calibrated_score, lexical_cov, _ = _compute_calibrated_relevance(
                query_text=query_text,
                title=sym.qualified_name,
                section=sym.signature_skeleton,
                content=sym.contextual_summary or "",
                cosine_similarity=vec_raw[sym.symbol_id],
                bm25_score=bm25_raw[sym.symbol_id],
                trigram_score=trgm_raw[sym.symbol_id],
                raw_rrf_score=norm_rrf,
                authority_weight=1.0,
                is_code_symbol=True,
            )
            file_rec = self._files.get(sym.file_id)
            ranking_weight = file_rec.ranking_weight if file_rec is not None else 1.0
            if not math.isclose(ranking_weight, 1.0, rel_tol=0.0, abs_tol=1e-9):
                norm_rrf = _apply_authority_score(norm_rrf, ranking_weight)
                calibrated_score = round(min(1.0, max(0.0, calibrated_score * ranking_weight)), 6)
            if min_calibrated_score > 0.0 and calibrated_score < min_calibrated_score:
                continue
            hits.append(
                CalibratedRankedSearchHit(
                    item_id=sym.symbol_id,
                    qualified_name_or_title=sym.qualified_name,
                    normalized_rrf_score=norm_rrf,
                    trigram_rank=r_trgm,
                    bm25_rank=r_bm25,
                    vector_rank=r_vec,
                    cosine_similarity=round(vec_raw[sym.symbol_id], 6),
                    bm25_score=round(bm25_raw[sym.symbol_id], 6),
                    trigram_score=round(trgm_raw[sym.symbol_id], 6),
                    raw_rrf_score=round(norm_rrf, 6),
                    calibrated_score=calibrated_score,
                    domain_id=domain_id,
                    lexical_coverage=lexical_cov,
                )
            )

        hits.sort(
            key=lambda h: (h.calibrated_score, h.normalized_rrf_score),
            reverse=True,
        )
        return list(hits[:top_k])

    def search_doc_chunks_hybrid_rrf(
        self,
        principal: PrincipalContext,
        domain_id: str,
        query_text: str,
        query_embedding: list[float],
        market_scope: DocumentScope | str | None = None,
        top_k: int = 5,
        include_stale: bool = False,
        min_calibrated_score: float = 0.0,
    ) -> list[RankedSearchHit]:
        """Executes RLS-filtered 3-way Normalized RRF over doc_chunks.

        Combines BM25 lexical rank, vector cosine similarity, and trigram/authority.
        """
        visible = [
            d
            for d in self.list_visible_doc_chunks(
                principal,
                domain_id,
                market_scope=market_scope,
                include_stale=include_stale,
            )
            if d.authority_weight > 0.0
        ]
        if not visible:
            return []

        # 1. Lexical BM25 rank over (title + market aliases + header + heading + content)
        corpus_texts = [
            (
                f"{d.doc_title} {_market_scope_aliases(d.market_scope, domain_id=domain_id)} "
                f"{d.contextual_header} {d.section_heading} {d.verbatim_content}"
            )
            for d in visible
        ]
        bm25_model = BM25Okapi(corpus_texts)
        bm25_scores = bm25_model.get_scores(query_text)
        bm25_order = sorted(range(len(visible)), key=lambda i: bm25_scores[i], reverse=True)
        bm25_ranks = {visible[idx].chunk_id: rank + 1 for rank, idx in enumerate(bm25_order)}

        # 2. Dense vector cosine similarity rank
        vec_scores = [_cosine_similarity(d.embedding, query_embedding) for d in visible]
        vec_order = sorted(range(len(visible)), key=lambda i: vec_scores[i], reverse=True)
        vec_ranks = {visible[idx].chunk_id: rank + 1 for rank, idx in enumerate(vec_order)}

        # 3. Trigram / Substring & Authority similarity rank
        trgm_scores = [
            _trigram_similarity(
                f"{d.doc_title} {_market_scope_aliases(d.market_scope, domain_id=domain_id)} "
                f"{d.section_heading} {d.verbatim_content}",
                query_text,
            )
            * d.authority_weight
            for d in visible
        ]
        trgm_order = sorted(range(len(visible)), key=lambda i: trgm_scores[i], reverse=True)
        trgm_ranks = {visible[idx].chunk_id: rank + 1 for rank, idx in enumerate(trgm_order)}

        domain_cnt = len(visible)
        scored_hits: list[tuple[float, float, float, CalibratedRankedSearchHit]] = []
        for idx, d in enumerate(visible):
            r_trgm = trgm_ranks[d.chunk_id]
            r_bm25 = bm25_ranks[d.chunk_id]
            r_vec = vec_ranks[d.chunk_id]
            base_rrf = compute_normalized_rrf([r_trgm, r_bm25, r_vec], num_modalities=3, k=60)
            weighted_score = _apply_authority_score(base_rrf, d.authority_weight)
            calibrated_score, lexical_cov, is_boiler = _compute_calibrated_relevance(
                query_text=query_text,
                title=d.doc_title,
                section=d.section_heading,
                content=d.verbatim_content,
                cosine_similarity=float(vec_scores[idx]),
                bm25_score=float(bm25_scores[idx]),
                trigram_score=float(trgm_scores[idx]),
                raw_rrf_score=base_rrf,
                authority_weight=d.authority_weight,
                market_scope=d.market_scope,
                domain_id=domain_id,
            )
            if is_boiler and domain_cnt > 50:
                continue
            if min_calibrated_score > 0.0 and calibrated_score < min_calibrated_score:
                continue
            hit = CalibratedRankedSearchHit(
                item_id=d.chunk_id,
                qualified_name_or_title=f"{d.doc_title} > {d.section_heading}",
                normalized_rrf_score=weighted_score,
                trigram_rank=r_trgm,
                bm25_rank=r_bm25,
                vector_rank=r_vec,
                cosine_similarity=round(float(vec_scores[idx]), 6),
                bm25_score=round(float(bm25_scores[idx]), 6),
                trigram_score=round(float(trgm_scores[idx]), 6),
                raw_rrf_score=round(base_rrf, 6),
                calibrated_score=calibrated_score,
                domain_id=domain_id,
                lexical_coverage=lexical_cov,
                is_boilerplate=is_boiler,
            )
            rank_key = calibrated_score if domain_cnt > 50 else weighted_score
            scored_hits.append((rank_key, weighted_score, d.authority_weight, hit))

        scored_hits.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        return [item[3] for item in scored_hits[:top_k]]

    def record_execution_telemetry(
        self,
        conn_or_record: Any,
        maybe_record: ExecutionTelemetryRecord | None = None,
    ) -> None:
        if isinstance(conn_or_record, ExecutionTelemetryRecord):
            record = conn_or_record
        elif maybe_record is not None:
            record = maybe_record
        else:
            raise ValueError("record_execution_telemetry requires an ExecutionTelemetryRecord")
        self._telemetry[record.trace_id] = record

    def get_execution_telemetry(
        self,
        trace_id_or_conn: Any,
        maybe_trace_id: UUID | None = None,
    ) -> ExecutionTelemetryRecord | None:
        if isinstance(trace_id_or_conn, UUID):
            return self._telemetry.get(trace_id_or_conn)
        if maybe_trace_id is not None:
            return self._telemetry.get(maybe_trace_id)
        if isinstance(trace_id_or_conn, str):
            return self._telemetry.get(UUID(trace_id_or_conn))
        return None

    def get_principal_paid_tokens_consumed(
        self,
        principal_or_conn: Any,
        maybe_principal_id: str | None = None,
        since_hours: int = 24,
        *,
        tenant_id: UUID | None = None,
    ) -> int:
        resolved_tid: UUID | None = tenant_id
        if isinstance(principal_or_conn, PrincipalContext):
            principal_id = principal_or_conn.user_id
            resolved_tid = resolved_tid or principal_or_conn.tenant_id
        elif isinstance(principal_or_conn, str):
            principal_id = principal_or_conn
        elif maybe_principal_id is not None:
            principal_id = maybe_principal_id
        else:
            return 0
        return sum(
            rec.context_pack_tokens_sent
            for rec in self._telemetry.values()
            if rec.principal_id == principal_id
            and str(rec.execution_mode) == "gemini_handoff"
            and (resolved_tid is None or rec.tenant_id == resolved_tid)
        )

    def search_visible_ast_symbols(
        self,
        principal: PrincipalContext,
        domain_id: str,
        query_text: str,
        limit: int = 50,
    ) -> list[ASTSymbolRecord]:
        """Searches visible AST symbols ranked by lexical overlap under RLS."""
        import re

        visible = self.list_visible_ast_symbols(principal, domain_id)
        words = [
            w.lower()
            for w in re.findall(r"[A-Za-z0-9_]{3,}", query_text)
            if w.lower() not in {"the", "and", "for", "with", "how", "what", "where", "from"}
        ]
        if not words:
            return visible[:limit]

        scored: list[tuple[int, ASTSymbolRecord]] = []
        for sym in visible:
            qual_l = sym.qualified_name.lower()
            sig_l = sym.signature_skeleton.lower()
            sum_l = sym.contextual_summary.lower()
            score = 0
            for w in words[:8]:
                if w in qual_l:
                    score += 6
                if w in sig_l:
                    score += 3
                if w in sum_l:
                    score += 1
            if score > 0:
                scored.append((score, sym))
        scored.sort(key=lambda item: (-item[0], item[1].qualified_name))
        return [s for _, s in scored[:limit]]

    def get_code_files_by_ids(
        self,
        principal: PrincipalContext,
        domain_id: str,
        file_ids: list[UUID | str],
    ) -> list[CodeFileRecord]:
        """Fetches visible code files by UUID under RLS without full-table scan."""
        target_ids: set[UUID] = set()
        for fid in file_ids:
            try:
                target_ids.add(fid if isinstance(fid, UUID) else UUID(str(fid)))
            except ValueError as exc:
                logger.debug(
                    "Ignoring malformed file_id %r (%s): %s",
                    fid,
                    type(exc).__name__,
                    exc,
                )
                continue
        if not target_ids:
            return []
        return [
            f for f in self.list_visible_code_files(principal, domain_id) if f.file_id in target_ids
        ]

    def get_code_files_by_paths(
        self,
        principal: PrincipalContext,
        domain_id: str,
        file_paths: list[str],
    ) -> list[CodeFileRecord]:
        """Fetches visible code files by file_path under RLS."""
        target_paths = {p.strip() for p in file_paths if p and p.strip()}
        if not target_paths:
            return []
        return [
            f
            for f in self.list_visible_code_files(principal, domain_id)
            if f.file_path in target_paths
        ]

    def upsert_ref_snapshot(
        self,
        principal: PrincipalContext,
        record: RefSnapshotRecord,
    ) -> RefSnapshotRecord:
        if record.tenant_id != principal.tenant_id:
            raise PermissionError("Principal tenant_id mismatch on upsert_ref_snapshot")
        if not _scopes_intersect(record.allowed_scopes, principal.allowed_scopes):
            raise PermissionError("Principal lacks scope to upsert ref_snapshot")
        with self._lock:
            existing_id: UUID | None = None
            for sid, existing in self._ref_snapshots.items():
                if (
                    existing.tenant_id == record.tenant_id
                    and existing.domain_id == record.domain_id
                    and existing.repo_name == record.repo_name
                    and existing.ref_name == record.ref_name
                ):
                    existing_id = sid
                    break
            target_id = existing_id if existing_id is not None else record.snapshot_id
            stored = dataclasses.replace(
                record,
                snapshot_id=target_id,
                updated_at=record.updated_at or datetime.now(),
            )
            self._ref_snapshots[target_id] = stored
            return stored

    def get_ref_snapshot(
        self,
        principal: PrincipalContext,
        domain_id: str,
        *,
        snapshot_id: UUID | None = None,
        repo_name: str | None = None,
        ref_name: str | None = None,
        mr_iid: str | None = None,
    ) -> RefSnapshotRecord | None:
        with self._lock:
            candidates = [
                snap
                for snap in self._ref_snapshots.values()
                if snap.tenant_id == principal.tenant_id
                and snap.domain_id == domain_id
                and _scopes_intersect(snap.allowed_scopes, principal.allowed_scopes)
            ]
            if snapshot_id is not None:
                for snap in candidates:
                    if snap.snapshot_id == snapshot_id:
                        return snap
                return None
            if repo_name is not None and ref_name is not None:
                for snap in candidates:
                    if snap.repo_name == repo_name and snap.ref_name == ref_name:
                        return snap
                return None
            if repo_name is not None and mr_iid is not None:
                matching = [
                    snap
                    for snap in candidates
                    if snap.repo_name == repo_name and snap.mr_iid == mr_iid
                ]
                return matching[-1] if matching else None
            raise ValueError("Specify snapshot_id, (repo_name, ref_name), or (repo_name, mr_iid)")

    def list_ref_snapshots(
        self,
        principal: PrincipalContext,
        domain_id: str,
        *,
        repo_name: str | None = None,
        state: str | None = None,
    ) -> list[RefSnapshotRecord]:
        with self._lock:
            items = [
                snap
                for snap in self._ref_snapshots.values()
                if snap.tenant_id == principal.tenant_id
                and snap.domain_id == domain_id
                and (repo_name is None or snap.repo_name == repo_name)
                and (state is None or snap.state == state)
                and _scopes_intersect(snap.allowed_scopes, principal.allowed_scopes)
            ]
            items.sort(key=lambda s: s.ref_name)
            return items

    def delete_ref_snapshot(self, principal: PrincipalContext, snapshot_id: UUID) -> int:
        with self._lock:
            snap = self._ref_snapshots.get(snapshot_id)
            if snap is None:
                return 0
            if snap.tenant_id != principal.tenant_id:
                raise PermissionError("Principal tenant_id mismatch on delete_ref_snapshot")
            if not _scopes_intersect(snap.allowed_scopes, principal.allowed_scopes):
                raise PermissionError("Principal lacks scope to delete ref_snapshot")
            self._ref_snapshots.pop(snapshot_id, None)
            del_keys = [
                k
                for k in self._overlay_deltas
                if k[0] == principal.tenant_id and k[1] == snapshot_id
            ]
            for k in del_keys:
                self._overlay_deltas.pop(k, None)
            return 1

    def upsert_overlay_file_delta(
        self,
        principal: PrincipalContext,
        record: OverlayFileDeltaRecord,
    ) -> OverlayFileDeltaRecord:
        if record.tenant_id != principal.tenant_id:
            raise PermissionError("Principal tenant_id mismatch on upsert_overlay_file_delta")
        if not _scopes_intersect(record.allowed_scopes, principal.allowed_scopes):
            raise PermissionError("Principal lacks scope to upsert overlay_file_delta")
        with self._lock:
            key = (record.tenant_id, record.snapshot_id, record.file_path)
            existing = self._overlay_deltas.get(key)
            target_id = existing.delta_id if existing is not None else record.delta_id
            stored = dataclasses.replace(record, delta_id=target_id)
            self._overlay_deltas[key] = stored
            return stored

    def list_overlay_file_deltas(
        self,
        principal: PrincipalContext,
        domain_id: str,
        snapshot_id: UUID,
    ) -> list[OverlayFileDeltaRecord]:
        with self._lock:
            items = [
                d
                for d in self._overlay_deltas.values()
                if d.tenant_id == principal.tenant_id
                and d.domain_id == domain_id
                and d.snapshot_id == snapshot_id
                and _scopes_intersect(d.allowed_scopes, principal.allowed_scopes)
            ]
            items.sort(key=lambda d: d.file_path)
            return items

    def _collect_snapshot_chain(
        self,
        principal: PrincipalContext,
        domain_id: str,
        snapshot_id: UUID,
    ) -> list[RefSnapshotRecord]:
        chain: list[RefSnapshotRecord] = []
        seen: set[UUID] = set()
        curr_id: UUID | None = snapshot_id
        while curr_id is not None and curr_id not in seen and len(chain) < 16:
            seen.add(curr_id)
            snap = self.get_ref_snapshot(principal, domain_id, snapshot_id=curr_id)
            if snap is None:
                break
            chain.append(snap)
            curr_id = snap.base_snapshot_id
        return chain

    def _effective_overlay_deltas_inmemory(
        self,
        principal: PrincipalContext,
        domain_id: str,
        snapshot_id: UUID,
    ) -> tuple[RefSnapshotRecord | None, dict[str, OverlayFileDeltaRecord], set[str]]:
        chain = self._collect_snapshot_chain(principal, domain_id, snapshot_id)
        if not chain:
            return None, {}, set()
        target_snap = chain[0]
        effective: dict[str, OverlayFileDeltaRecord] = {}
        renamed_old_paths: set[str] = set()
        for snap in chain:
            for d in self.list_overlay_file_deltas(principal, domain_id, snap.snapshot_id):
                if d.previous_file_path and d.previous_file_path != d.file_path:
                    renamed_old_paths.add(d.previous_file_path)
                if d.file_path not in effective:
                    effective[d.file_path] = d
        return target_snap, effective, renamed_old_paths

    def resolve_overlay_files(
        self,
        principal: PrincipalContext,
        domain_id: str,
        snapshot_id: UUID,
        *,
        file_path: str | None = None,
    ) -> list[ResolvedOverlayFileRecord]:
        with self._lock:
            target_snap, effective_deltas, renamed_old_paths = (
                self._effective_overlay_deltas_inmemory(principal, domain_id, snapshot_id)
            )
            if target_snap is None:
                return []

            base_by_path: dict[str, CodeFileRecord] = {}
            for cf in self.list_visible_code_files(principal, domain_id):
                if cf.repo_name != target_snap.repo_name:
                    continue
                if cf.file_path in renamed_old_paths:
                    continue
                if file_path is not None and cf.file_path != file_path:
                    continue
                base_by_path[cf.file_path] = cf

            if file_path is not None:
                delta_keys = {file_path} if file_path in effective_deltas else set()
            else:
                delta_keys = set(effective_deltas.keys())
            all_paths = set(base_by_path.keys()) | delta_keys
            resolved: list[ResolvedOverlayFileRecord] = []
            for path in sorted(all_paths):
                delta = effective_deltas.get(path)
                base = base_by_path.get(path)
                if delta is not None:
                    if delta.change_type == "deleted":
                        continue
                    resolved.append(
                        ResolvedOverlayFileRecord(
                            domain_id=delta.domain_id,
                            repo_name=delta.repo_name,
                            file_path=delta.file_path,
                            git_commit_sha=target_snap.head_commit_sha,
                            git_blob_sha=delta.overlay_blob_sha or "",
                            language=delta.language,
                            allowed_scopes=list(delta.allowed_scopes),
                            raw_content=delta.raw_content or "",
                            line_count=delta.line_count,
                            source_layer="overlay",
                            snapshot_id=target_snap.snapshot_id,
                            base_file_id=base.file_id if base is not None else None,
                            delta_id=delta.delta_id,
                            change_type=delta.change_type,
                            ranking_weight=delta.ranking_weight,
                            is_generated=delta.is_generated,
                            is_vendored=delta.is_vendored,
                            tenant_id=delta.tenant_id,
                        )
                    )
                elif base is not None:
                    resolved.append(
                        ResolvedOverlayFileRecord(
                            domain_id=base.domain_id,
                            repo_name=base.repo_name,
                            file_path=base.file_path,
                            git_commit_sha=base.git_commit_sha,
                            git_blob_sha=base.git_blob_sha,
                            language=base.language,
                            allowed_scopes=list(base.allowed_scopes),
                            raw_content=base.raw_content,
                            line_count=base.line_count,
                            source_layer="base_snapshot",
                            snapshot_id=target_snap.snapshot_id,
                            base_file_id=base.file_id,
                            delta_id=None,
                            change_type=None,
                            ranking_weight=base.ranking_weight,
                            is_generated=base.is_generated,
                            is_vendored=base.is_vendored,
                            tenant_id=base.tenant_id,
                        )
                    )
            return resolved

    def get_overlay_file(
        self,
        principal: PrincipalContext,
        domain_id: str,
        snapshot_id: UUID,
        file_path: str,
    ) -> ResolvedOverlayFileRecord | None:
        items = self.resolve_overlay_files(principal, domain_id, snapshot_id, file_path=file_path)
        return items[0] if items else None

    def resolve_overlay_symbols(
        self,
        principal: PrincipalContext,
        domain_id: str,
        snapshot_id: UUID,
    ) -> list[ASTSymbolRecord]:
        with self._lock:
            target_snap, effective_deltas, renamed_old_paths = (
                self._effective_overlay_deltas_inmemory(principal, domain_id, snapshot_id)
            )
            if target_snap is None:
                return []

            suppressed_paths = set(renamed_old_paths) | set(effective_deltas.keys())
            for d in effective_deltas.values():
                if d.previous_file_path:
                    suppressed_paths.add(d.previous_file_path)

            visible_files_by_id = {
                cf.file_id: cf
                for cf in self.list_visible_code_files(principal, domain_id)
                if cf.repo_name == target_snap.repo_name
            }
            base_symbols = [
                sym
                for sym in self.list_visible_ast_symbols(principal, domain_id)
                if sym.file_id in visible_files_by_id
                and visible_files_by_id[sym.file_id].file_path not in suppressed_paths
            ]

            overlay_symbols: list[ASTSymbolRecord] = []
            for delta in effective_deltas.values():
                if delta.change_type == "deleted":
                    continue
                for sym_dict in delta.symbols_json:
                    if not isinstance(sym_dict, dict):
                        continue
                    overlay_symbols.append(
                        ASTSymbolRecord(
                            symbol_id=UUID(str(sym_dict["symbol_id"])),
                            file_id=delta.delta_id,
                            domain_id=delta.domain_id,
                            qualified_name=str(sym_dict["qualified_name"]),
                            symbol_kind=str(sym_dict.get("symbol_kind", "function")),
                            start_line=int(sym_dict.get("start_line", 1)),
                            end_line=int(sym_dict.get("end_line", 1)),
                            signature_skeleton=str(sym_dict.get("signature_skeleton", "")),
                            contextual_summary=str(sym_dict.get("contextual_summary", "")),
                            allowed_scopes=list(delta.allowed_scopes),
                            embedding=[],
                            repo_qualified_id=sym_dict.get("repo_qualified_id"),
                            bare_name=sym_dict.get("bare_name"),
                            tenant_id=delta.tenant_id,
                        )
                    )

            combined = base_symbols + overlay_symbols
            combined.sort(key=lambda s: s.qualified_name)
            return combined

    def list_ownership_edges(
        self,
        principal: PrincipalContext,
        domain_id: str,
        *,
        edge_kinds: Sequence[str] = ("OWNED_BY_TEAM", "PART_OF_SYSTEM"),
    ) -> list[OwnershipEdgeRecord]:
        with self._lock:
            allowed_kinds = set(edge_kinds)
            visible_syms = {
                s.symbol_id: s for s in self.list_visible_ast_symbols(principal, domain_id)
            }
            out: list[OwnershipEdgeRecord] = []
            for edge in self.list_visible_ast_edges(principal, domain_id):
                if edge.edge_kind not in allowed_kinds:
                    continue
                src = visible_syms.get(edge.source_symbol_id)
                if src is None:
                    continue
                tgt = (
                    visible_syms.get(edge.target_symbol_id)
                    if edge.target_symbol_id is not None
                    else None
                )
                out.append(
                    OwnershipEdgeRecord(
                        edge_id=edge.edge_id,
                        edge_kind=edge.edge_kind,
                        source_qualified_name=src.qualified_name,
                        source_symbol_kind=src.symbol_kind,
                        target_qualified_name=edge.target_qualified_name,
                        target_symbol_kind=tgt.symbol_kind if tgt is not None else "",
                    )
                )
            out.sort(key=lambda e: (e.source_qualified_name, e.edge_kind, e.target_qualified_name))
            return out

    def upsert_side_effects_batch(
        self,
        principal: PrincipalContext,
        records: Sequence[SideEffectRecord],
    ) -> int:
        with self._lock:
            inserted = 0
            for rec in records:
                if rec.tenant_id != principal.tenant_id:
                    raise PermissionError("Principal tenant_id mismatch on upsert_side_effects")
                if not _scopes_intersect(rec.allowed_scopes, principal.allowed_scopes):
                    raise PermissionError("Principal lacks scope to insert side_effect row")
                nat_key = (
                    rec.tenant_id,
                    rec.domain_id,
                    rec.repo_name,
                    rec.file_path,
                    rec.qualified_symbol,
                    rec.effect_category,
                    rec.target_identifier,
                    rec.start_line,
                )
                for existing_id, existing in list(self._side_effects.items()):
                    if (
                        existing.tenant_id,
                        existing.domain_id,
                        existing.repo_name,
                        existing.file_path,
                        existing.qualified_symbol,
                        existing.effect_category,
                        existing.target_identifier,
                        existing.start_line,
                    ) == nat_key:
                        self._side_effects.pop(existing_id, None)
                self._side_effects[rec.side_effect_id] = rec
                inserted += 1
            return inserted

    def list_visible_side_effects(
        self,
        principal: PrincipalContext,
        domain_id: str,
        *,
        effect_category: str | None = None,
        repo_name: str | None = None,
        file_path: str | None = None,
    ) -> list[SideEffectRecord]:
        with self._lock:
            out = [
                se
                for se in self._side_effects.values()
                if se.tenant_id == principal.tenant_id
                and se.domain_id == domain_id
                and (effect_category is None or se.effect_category == effect_category)
                and (repo_name is None or se.repo_name == repo_name)
                and (file_path is None or se.file_path == file_path)
                and _scopes_intersect(se.allowed_scopes, principal.allowed_scopes)
            ]
            out.sort(key=lambda s: (s.repo_name, s.file_path, s.start_line, s.effect_category))
            return out

    def delete_side_effects_for_file(
        self,
        principal: PrincipalContext,
        domain_id: str,
        repo_name: str,
        file_path: str,
    ) -> int:
        with self._lock:
            del_ids = [
                se_id
                for se_id, se in self._side_effects.items()
                if se.tenant_id == principal.tenant_id
                and se.domain_id == domain_id
                and se.repo_name == repo_name
                and se.file_path == file_path
                and _scopes_intersect(se.allowed_scopes, principal.allowed_scopes)
            ]
            for se_id in del_ids:
                self._side_effects.pop(se_id, None)
            return len(del_ids)

    def delete_side_effects_for_repo(
        self,
        principal: PrincipalContext,
        domain_id: str,
        repo_name: str,
    ) -> int:
        with self._lock:
            del_ids = [
                se_id
                for se_id, se in self._side_effects.items()
                if se.tenant_id == principal.tenant_id
                and se.domain_id == domain_id
                and se.repo_name == repo_name
                and _scopes_intersect(se.allowed_scopes, principal.allowed_scopes)
            ]
            for se_id in del_ids:
                self._side_effects.pop(se_id, None)
            return len(del_ids)

    def upsert_connector_watermark(
        self,
        principal: PrincipalContext,
        record: ConnectorWatermarkRecord,
    ) -> None:
        if record.tenant_id != principal.tenant_id:
            raise PermissionError("Principal tenant_id mismatch on upsert_connector_watermark")
        with self._lock:
            key = (record.tenant_id, record.domain_id, record.source_id, record.container_key)
            prev = self._watermarks.get(key)
            now = datetime.now(UTC)
            self._watermarks[key] = ConnectorWatermarkRecord(
                tenant_id=record.tenant_id,
                domain_id=record.domain_id,
                source_id=record.source_id,
                container_key=record.container_key,
                watermark_cursor=record.watermark_cursor
                or (prev.watermark_cursor if prev is not None else ""),
                sync_tier=record.sync_tier,
                items_synced=record.items_synced,
                last_webhook_at=record.last_webhook_at
                or (prev.last_webhook_at if prev is not None else None),
                last_watermark_sync_at=record.last_watermark_sync_at
                or (prev.last_watermark_sync_at if prev is not None else None),
                last_reconciliation_at=record.last_reconciliation_at
                or (prev.last_reconciliation_at if prev is not None else None),
                metadata_json=dict(record.metadata_json),
                updated_at=record.updated_at or now,
            )

    def reset_connector_watermark(
        self,
        principal: PrincipalContext,
        record: ConnectorWatermarkRecord,
    ) -> None:
        """Unconditionally rewinds watermark_cursor (including to empty) after PITR."""
        if record.tenant_id != principal.tenant_id:
            raise PermissionError("Principal tenant_id mismatch on reset_connector_watermark")
        with self._lock:
            key = (record.tenant_id, record.domain_id, record.source_id, record.container_key)
            prev = self._watermarks.get(key)
            now = datetime.now(UTC)
            self._watermarks[key] = ConnectorWatermarkRecord(
                tenant_id=record.tenant_id,
                domain_id=record.domain_id,
                source_id=record.source_id,
                container_key=record.container_key,
                watermark_cursor=record.watermark_cursor,
                sync_tier=record.sync_tier,
                items_synced=record.items_synced,
                last_webhook_at=record.last_webhook_at
                or (prev.last_webhook_at if prev is not None else None),
                last_watermark_sync_at=record.last_watermark_sync_at
                or (prev.last_watermark_sync_at if prev is not None else None),
                last_reconciliation_at=record.last_reconciliation_at or now,
                metadata_json=dict(record.metadata_json),
                updated_at=now,
            )

    def get_connector_watermark(
        self,
        principal: PrincipalContext,
        domain_id: str,
        source_id: str,
        container_key: str,
    ) -> ConnectorWatermarkRecord | None:
        with self._lock:
            return self._watermarks.get((principal.tenant_id, domain_id, source_id, container_key))

    def list_connector_watermarks(
        self,
        principal: PrincipalContext,
        domain_id: str,
        source_id: str | None = None,
    ) -> list[ConnectorWatermarkRecord]:
        with self._lock:
            items = [
                wm
                for wm in self._watermarks.values()
                if wm.tenant_id == principal.tenant_id
                and wm.domain_id == domain_id
                and (source_id is None or wm.source_id == source_id)
            ]
            items.sort(key=lambda w: (w.source_id, w.container_key))
            return items

    def list_visible_code_file_records(
        self,
        principal: PrincipalContext,
        domain_id: str,
        *,
        repo_name: str | None = None,
        limit: int = 2000,
    ) -> list[CodeFileRecord]:
        files = self.list_visible_code_files(principal, domain_id)
        if repo_name:
            files = [f for f in files if f.repo_name == repo_name]
        return files[: max(1, limit)]

    def list_visible_ast_symbol_records(
        self,
        principal: PrincipalContext,
        domain_id: str,
        *,
        limit: int = 5000,
    ) -> list[ASTSymbolRecord]:
        return self.list_visible_ast_symbols(principal, domain_id)[: max(1, limit)]

    def list_visible_ast_edge_records(
        self,
        principal: PrincipalContext,
        domain_id: str,
        *,
        limit: int = 10000,
    ) -> list[ASTEdgeRecord]:
        return self.list_visible_ast_edges(principal, domain_id)[: max(1, limit)]

    def purge_domain_artifacts(
        self,
        principal: PrincipalContext,
        domain_id: str,
        *,
        doc_id: str | None = None,
        confluence_page_id: str | None = None,
        jira_issue_key: str | None = None,
        repo_name: str | None = None,
        file_path: str | None = None,
        commit_sha: str | None = None,
        blob_sha: str | None = None,
        include_derived_wiki: bool = True,
    ) -> dict[str, int]:
        norm = normalize_purge_selectors(
            domain_id,
            doc_id=doc_id,
            confluence_page_id=confluence_page_id,
            jira_issue_key=jira_issue_key,
            repo_name=repo_name,
            file_path=file_path,
            commit_sha=commit_sha,
            blob_sha=blob_sha,
        )
        domain_id = norm.domain_id
        doc_id = norm.doc_id
        confluence_page_id = norm.confluence_page_id
        jira_issue_key = norm.jira_issue_key
        repo_name = norm.repo_name
        file_path = norm.file_path
        commit_sha = norm.commit_sha
        blob_sha = norm.blob_sha
        counts: dict[str, int] = {
            "side_effects": 0,
            "overlay_file_deltas": 0,
            "ref_snapshots": 0,
            "ast_edges": 0,
            "ast_symbols": 0,
            "code_files": 0,
            "repo_commits": 0,
            "repositories": 0,
            "doc_chunks": 0,
            "confluence_pages": 0,
            "jira_issues": 0,
        }
        with self._lock:
            target_doc_id = doc_id
            if target_doc_id is None and confluence_page_id:
                target_doc_id = f"confluence:{confluence_page_id}"
            elif target_doc_id is None and jira_issue_key:
                target_doc_id = f"jira:{jira_issue_key}"

            if target_doc_id:
                del_doc_ids = [
                    cid
                    for cid, d in self._docs.items()
                    if d.tenant_id == principal.tenant_id
                    and d.domain_id == domain_id
                    and d.doc_id == target_doc_id
                    and _scopes_intersect(d.allowed_scopes, principal.allowed_scopes)
                ]
                for cid in del_doc_ids:
                    self._docs.pop(cid, None)
                counts["doc_chunks"] += len(del_doc_ids)

            if confluence_page_id:
                del_cf_ids = [
                    pid
                    for pid, p in self._confluence.items()
                    if p.tenant_id == principal.tenant_id
                    and p.domain_id == domain_id
                    and p.confluence_page_id == confluence_page_id
                    and _scopes_intersect(p.allowed_scopes, principal.allowed_scopes)
                ]
                for pid in del_cf_ids:
                    self._confluence.pop(pid, None)
                counts["confluence_pages"] += len(del_cf_ids)

            if jira_issue_key:
                del_jira_ids = [
                    iid
                    for iid, j in self._jira.items()
                    if j.tenant_id == principal.tenant_id
                    and j.domain_id == domain_id
                    and j.issue_key == jira_issue_key
                    and _scopes_intersect(j.allowed_scopes, principal.allowed_scopes)
                ]
                for iid in del_jira_ids:
                    self._jira.pop(iid, None)
                counts["jira_issues"] += len(del_jira_ids)

            repo_variants = (repo_name, repo_name.replace("/", "_")) if repo_name else ()
            target_file_ids: list[UUID] = []
            if repo_name and file_path:
                target_file_ids = [
                    fid
                    for fid, f in self._files.items()
                    if f.tenant_id == principal.tenant_id
                    and f.domain_id == domain_id
                    and f.repo_name in repo_variants
                    and f.file_path == file_path
                    and _scopes_intersect(f.allowed_scopes, principal.allowed_scopes)
                ]
                del_ov_keys = [
                    k
                    for k, d in self._overlay_deltas.items()
                    if d.tenant_id == principal.tenant_id
                    and d.domain_id == domain_id
                    and d.repo_name in repo_variants
                    and (d.file_path == file_path or d.previous_file_path == file_path)
                    and _scopes_intersect(d.allowed_scopes, principal.allowed_scopes)
                ]
                for k in del_ov_keys:
                    self._overlay_deltas.pop(k, None)
                counts["overlay_file_deltas"] += len(del_ov_keys)
                del_se_ids = [
                    se_id
                    for se_id, se in self._side_effects.items()
                    if se.tenant_id == principal.tenant_id
                    and se.domain_id == domain_id
                    and se.repo_name in repo_variants
                    and se.file_path == file_path
                    and _scopes_intersect(se.allowed_scopes, principal.allowed_scopes)
                ]
                for se_id in del_se_ids:
                    self._side_effects.pop(se_id, None)
                counts["side_effects"] += len(del_se_ids)
            elif blob_sha:
                target_file_ids = [
                    fid
                    for fid, f in self._files.items()
                    if f.tenant_id == principal.tenant_id
                    and f.domain_id == domain_id
                    and f.git_blob_sha == blob_sha
                    and _scopes_intersect(f.allowed_scopes, principal.allowed_scopes)
                ]
                del_ov_keys = [
                    k
                    for k, d in self._overlay_deltas.items()
                    if d.tenant_id == principal.tenant_id
                    and d.domain_id == domain_id
                    and (d.overlay_blob_sha == blob_sha or d.base_blob_sha == blob_sha)
                    and _scopes_intersect(d.allowed_scopes, principal.allowed_scopes)
                ]
                for k in del_ov_keys:
                    self._overlay_deltas.pop(k, None)
                counts["overlay_file_deltas"] += len(del_ov_keys)
            elif commit_sha:
                target_file_ids = [
                    fid
                    for fid, f in self._files.items()
                    if f.tenant_id == principal.tenant_id
                    and f.domain_id == domain_id
                    and f.git_commit_sha == commit_sha
                    and _scopes_intersect(f.allowed_scopes, principal.allowed_scopes)
                ]
                domain_repo_ids = {
                    r.repo_id
                    for r in self._repos.values()
                    if r.tenant_id == principal.tenant_id
                    and r.domain_id == domain_id
                    and _scopes_intersect(r.allowed_scopes, principal.allowed_scopes)
                }
                del_commit_ids = [
                    cid
                    for cid, c in self._commits.items()
                    if c.tenant_id == principal.tenant_id
                    and c.commit_sha == commit_sha
                    and (not domain_repo_ids or c.repo_id in domain_repo_ids)
                ]
                for cid in del_commit_ids:
                    self._commits.pop(cid, None)
                counts["repo_commits"] += len(del_commit_ids)
                del_snap_ids = [
                    sid
                    for sid, s in self._ref_snapshots.items()
                    if s.tenant_id == principal.tenant_id
                    and s.domain_id == domain_id
                    and (s.head_commit_sha == commit_sha or s.base_commit_sha == commit_sha)
                    and _scopes_intersect(s.allowed_scopes, principal.allowed_scopes)
                ]
                for sid in del_snap_ids:
                    self._ref_snapshots.pop(sid, None)
                    del_ov = [k for k in self._overlay_deltas if k[1] == sid]
                    for k in del_ov:
                        self._overlay_deltas.pop(k, None)
                    counts["overlay_file_deltas"] += len(del_ov)
                counts["ref_snapshots"] += len(del_snap_ids)
            elif repo_name:
                target_file_ids = [
                    fid
                    for fid, f in self._files.items()
                    if f.tenant_id == principal.tenant_id
                    and f.domain_id == domain_id
                    and f.repo_name in repo_variants
                    and _scopes_intersect(f.allowed_scopes, principal.allowed_scopes)
                ]
                del_se_ids = [
                    se_id
                    for se_id, se in self._side_effects.items()
                    if se.tenant_id == principal.tenant_id
                    and se.domain_id == domain_id
                    and se.repo_name in repo_variants
                    and _scopes_intersect(se.allowed_scopes, principal.allowed_scopes)
                ]
                for se_id in del_se_ids:
                    self._side_effects.pop(se_id, None)
                counts["side_effects"] += len(del_se_ids)
                del_ov_keys = [
                    k
                    for k, d in self._overlay_deltas.items()
                    if d.tenant_id == principal.tenant_id
                    and d.domain_id == domain_id
                    and d.repo_name in repo_variants
                    and _scopes_intersect(d.allowed_scopes, principal.allowed_scopes)
                ]
                for k in del_ov_keys:
                    self._overlay_deltas.pop(k, None)
                counts["overlay_file_deltas"] += len(del_ov_keys)
                del_snap_ids = [
                    sid
                    for sid, s in self._ref_snapshots.items()
                    if s.tenant_id == principal.tenant_id
                    and s.domain_id == domain_id
                    and s.repo_name in repo_variants
                    and _scopes_intersect(s.allowed_scopes, principal.allowed_scopes)
                ]
                for sid in del_snap_ids:
                    self._ref_snapshots.pop(sid, None)
                counts["ref_snapshots"] += len(del_snap_ids)
                del_repo_ids = [
                    rid
                    for rid, r in self._repos.items()
                    if r.tenant_id == principal.tenant_id
                    and r.domain_id == domain_id
                    and r.repo_name in repo_variants
                    and _scopes_intersect(r.allowed_scopes, principal.allowed_scopes)
                ]
                del_repo_set = set(del_repo_ids)
                del_commit_ids = [
                    cid
                    for cid, c in self._commits.items()
                    if c.tenant_id == principal.tenant_id and c.repo_id in del_repo_set
                ]
                for cid in del_commit_ids:
                    self._commits.pop(cid, None)
                counts["repo_commits"] += len(del_commit_ids)
                for rid in del_repo_ids:
                    self._repos.pop(rid, None)
                counts["repositories"] += len(del_repo_ids)

            if target_file_ids:
                file_set = set(target_file_ids)
                del_se_f = [
                    se_id for se_id, se in self._side_effects.items() if se.file_id in file_set
                ]
                for se_id in del_se_f:
                    self._side_effects.pop(se_id, None)
                counts["side_effects"] += len(del_se_f)

                del_sym_ids = [sid for sid, sym in self._symbols.items() if sym.file_id in file_set]
                sym_set = set(del_sym_ids)
                if sym_set:
                    del_edge_ids = [
                        eid
                        for eid, e in self._edges.items()
                        if e.domain_id == domain_id
                        and (e.source_symbol_id in sym_set or e.target_symbol_id in sym_set)
                    ]
                    for eid in del_edge_ids:
                        self._edges.pop(eid, None)
                    counts["ast_edges"] += len(del_edge_ids)
                for sid in del_sym_ids:
                    self._symbols.pop(sid, None)
                counts["ast_symbols"] += len(del_sym_ids)
                for fid in target_file_ids:
                    self._files.pop(fid, None)
                counts["code_files"] += len(target_file_ids)

            if include_derived_wiki and repo_name:
                wiki_prefixes = (
                    f"repo_wiki:{repo_name}",
                    f"repo_wiki:{repo_name.replace('/', '_')}",
                )
                del_wiki_ids = [
                    cid
                    for cid, d in self._docs.items()
                    if d.tenant_id == principal.tenant_id
                    and d.domain_id == domain_id
                    and d.source_type == "repo_wiki"
                    and (
                        d.doc_id in wiki_prefixes
                        or any(d.doc_id.startswith(f"{pfx}:") for pfx in wiki_prefixes)
                    )
                    and _scopes_intersect(d.allowed_scopes, principal.allowed_scopes)
                ]
                for cid in del_wiki_ids:
                    self._docs.pop(cid, None)
                counts["doc_chunks"] += len(del_wiki_ids)

            return counts

    def get_scout_session(
        self,
        principal: PrincipalContext,
        session_id: str,
    ) -> ScoutSessionRecord | None:
        with self._lock:
            rec = self._sessions.get((principal.tenant_id, session_id))
            if rec is None:
                return None
            if rec.principal_id != principal.user_id:
                return None
            if not _scopes_intersect(rec.allowed_scopes, principal.allowed_scopes):
                return None
            return rec

    def upsert_scout_session(
        self,
        principal: PrincipalContext,
        record: ScoutSessionRecord,
    ) -> ScoutSessionRecord:
        if record.tenant_id != principal.tenant_id:
            raise PermissionError("Principal tenant_id mismatch on upsert_scout_session")
        if record.principal_id != principal.user_id:
            raise PermissionError("Principal user_id mismatch on upsert_scout_session")
        if not _scopes_intersect(record.allowed_scopes, principal.allowed_scopes):
            raise PermissionError("Principal lacks scope to upsert scout_session row")
        with self._lock:
            key = (record.tenant_id, record.session_id)
            prev = self._sessions.get(key)
            if prev is not None and prev.principal_id != principal.user_id:
                raise PermissionError(
                    f"Session '{record.session_id}' belongs to principal '{prev.principal_id}', "
                    f"not '{principal.user_id}'."
                )
            now = datetime.now(UTC)
            stored = ScoutSessionRecord(
                tenant_id=record.tenant_id,
                session_id=record.session_id,
                domain_id=record.domain_id,
                principal_id=record.principal_id,
                allowed_scopes=list(record.allowed_scopes),
                turn_count=record.turn_count,
                compacted_summary=record.compacted_summary,
                recent_turns_json=[dict(t) for t in record.recent_turns_json],
                pinned_file_blobs_json={
                    str(k): dict(v) for k, v in record.pinned_file_blobs_json.items()
                },
                pinned_symbols_json=list(record.pinned_symbols_json),
                total_tokens_compacted=record.total_tokens_compacted,
                created_at=(
                    prev.created_at if prev and prev.created_at else (record.created_at or now)
                ),
                updated_at=now,
            )
            self._sessions[key] = stored
            return stored

    def delete_scout_session(
        self,
        principal: PrincipalContext,
        session_id: str,
    ) -> int:
        with self._lock:
            key = (principal.tenant_id, session_id)
            rec = self._sessions.get(key)
            if rec is None or not _scopes_intersect(rec.allowed_scopes, principal.allowed_scopes):
                return 0
            self._sessions.pop(key, None)
            return 1

    def register_model_artifact(
        self,
        principal: PrincipalContext,
        record: ModelArtifactRecord,
    ) -> ModelArtifactRecord:
        if record.tenant_id != principal.tenant_id:
            raise PermissionError("Principal tenant_id mismatch on register_model_artifact")
        PostgresRLSRepository._validate_model_artifact_record(record)
        with self._lock:
            prev = self._model_artifacts.get(
                (principal.tenant_id, record.domain_id, record.artifact_id)
            )
            if prev is not None and prev.status == "revoked":
                raise PermissionError(
                    f"Revoked model artifact '{record.artifact_id}' cannot be "
                    "modified or re-activated."
                )
            now = datetime.now(UTC)
            if record.is_active:
                for key, existing in list(self._model_artifacts.items()):
                    if (
                        key[0] == principal.tenant_id
                        and key[1] == record.domain_id
                        and key[2] != record.artifact_id
                        and existing.artifact_kind == record.artifact_kind
                        and existing.is_active
                    ):
                        self._model_artifacts[key] = dataclasses.replace(
                            existing,
                            is_active=False,
                            status="retired" if existing.status == "active" else existing.status,
                            updated_at=now,
                        )
            effective_status: ModelArtifactStatus = "active" if record.is_active else record.status
            stored = dataclasses.replace(
                record,
                status=effective_status,
                created_at=(
                    prev.created_at if prev and prev.created_at else (record.created_at or now)
                ),
                activated_at=(
                    (
                        prev.activated_at
                        if prev and prev.activated_at
                        else (record.activated_at or now)
                    )
                    if record.is_active
                    else (prev.activated_at if prev else record.activated_at)
                ),
                updated_at=now,
            )
            self._model_artifacts[(principal.tenant_id, record.domain_id, record.artifact_id)] = (
                stored
            )
            return stored

    def activate_model_artifact(
        self,
        principal: PrincipalContext,
        domain_id: str,
        artifact_id: str,
        *,
        artifact_kind: ModelArtifactKind = "lora_adapter",
    ) -> ModelArtifactRecord:
        with self._lock:
            target_key = (principal.tenant_id, domain_id, artifact_id)
            target = self._model_artifacts.get(target_key)
            if target is None or target.artifact_kind != artifact_kind:
                raise KeyError(
                    f"Model artifact '{artifact_id}' ({artifact_kind}) not registered "
                    f"for '{domain_id}'"
                )
            if target.status == "revoked":
                raise PermissionError(
                    f"Revoked model artifact '{artifact_id}' cannot be re-activated."
                )
            now = datetime.now(UTC)
            for key, existing in list(self._model_artifacts.items()):
                if (
                    key[0] == principal.tenant_id
                    and key[1] == domain_id
                    and key[2] != artifact_id
                    and existing.artifact_kind == artifact_kind
                    and existing.is_active
                ):
                    self._model_artifacts[key] = dataclasses.replace(
                        existing,
                        is_active=False,
                        status="retired" if existing.status == "active" else existing.status,
                        updated_at=now,
                    )
            activated = dataclasses.replace(
                target,
                is_active=True,
                status="active",
                activated_at=target.activated_at or now,
                updated_at=now,
            )
            self._model_artifacts[target_key] = activated
            return activated

    def retire_model_artifact(
        self,
        principal: PrincipalContext,
        domain_id: str,
        artifact_id: str,
        *,
        revoke: bool = False,
    ) -> bool:
        with self._lock:
            key = (principal.tenant_id, domain_id, artifact_id)
            existing = self._model_artifacts.get(key)
            if existing is None:
                return False
            self._model_artifacts[key] = dataclasses.replace(
                existing,
                is_active=False,
                status="revoked" if revoke else "retired",
                updated_at=datetime.now(UTC),
            )
            return True

    def get_active_model_artifact(
        self,
        principal: PrincipalContext,
        domain_id: str,
        *,
        artifact_kind: ModelArtifactKind = "lora_adapter",
    ) -> ModelArtifactRecord | None:
        with self._lock:
            for (tid, did, _aid), rec in self._model_artifacts.items():
                if (
                    tid == principal.tenant_id
                    and did == domain_id
                    and rec.artifact_kind == artifact_kind
                    and rec.is_active
                ):
                    return rec
            return None

    def list_active_model_artifacts(
        self,
        principal: PrincipalContext,
        *,
        artifact_kind: ModelArtifactKind = "lora_adapter",
    ) -> list[ModelArtifactRecord]:
        with self._lock:
            matched = [
                rec
                for (tid, _did, _aid), rec in self._model_artifacts.items()
                if tid == principal.tenant_id
                and rec.artifact_kind == artifact_kind
                and rec.is_active
            ]
            return sorted(matched, key=lambda r: (r.domain_id, r.artifact_id))

    def list_model_artifacts(
        self,
        principal: PrincipalContext,
        domain_id: str,
        *,
        artifact_kind: ModelArtifactKind | None = None,
    ) -> list[ModelArtifactRecord]:
        with self._lock:
            matched = [
                rec
                for (tid, did, _aid), rec in self._model_artifacts.items()
                if tid == principal.tenant_id
                and did == domain_id
                and (artifact_kind is None or rec.artifact_kind == artifact_kind)
            ]
        epoch = datetime.min.replace(tzinfo=UTC)
        matched.sort(key=lambda r: r.artifact_id)
        matched.sort(key=lambda r: r.created_at or epoch, reverse=True)
        return matched
