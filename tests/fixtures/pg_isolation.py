"""Ephemeral per-session PostgreSQL tenant domain helpers.

Provides strict ``eval_*`` domain creation and child-to-parent teardown so live
PostgreSQL tests never mutate seeded benchmark corpora (``code_architecture_scout``,
``enterprise_docs_scout``, ``polyglot_unseen_v2``).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections import defaultdict
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any, Final
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

import psycopg

import context_scout.retrieval.embedder as _embedder_mod
import context_scout.schemas.contracts as _contracts_mod
from context_scout.ast_engine.extractor import ASTExtractor
from context_scout.db.repositories import PostgresRLSRepository
from context_scout.db.schema import (
    ASTEdgeRecord,
    ASTSymbolRecord,
    CodeFileRecord,
    DocChunkRecord,
)
from context_scout.db.session import PoolRole, RLSConnectionManager
from context_scout.plugins.domains.enterprise_core import EnterpriseCoreDomainPlugin
from context_scout.schemas.contracts import PrincipalContext, RoleType


def _embed_fixture_texts(texts: list[str], dim: int = 768) -> list[list[float]]:
    if not texts:
        return []
    factory = getattr(_embedder_mod, "get_embedding_provider", None)
    if callable(factory):
        provider = factory(dimensions=dim)
        batch_fn = getattr(provider, "embed_batch", None)
        if callable(batch_fn):
            vecs: list[list[float]] = batch_fn(texts, is_query=False)
            return vecs
        return [provider.embed_text(t, is_query=False) for t in texts]
    embedder = _embedder_mod.LocalSemanticEmbedder()
    return [embedder.embed_text(t, is_query=False)[:dim] for t in texts]


def _embed_fixture_text(text: str, dim: int = 768) -> list[float]:
    return _embed_fixture_texts([text], dim=dim)[0]


DEFAULT_TENANT_ID: Final[UUID] = getattr(
    _contracts_mod,
    "DEFAULT_TENANT_ID",
    UUID("00000000-0000-0000-0000-000000000001"),
)

EPHEMERAL_PG_DOMAIN_PREFIX: Final[str] = "eval_"

ALL_RLS_TABLES_FK_DELETE_ORDER: Final[tuple[str, ...]] = (
    "audit_events",
    "scim_groups",
    "scim_users",
    "model_artifacts",
    "scout_sessions",
    "overlay_file_deltas",
    "ref_snapshots",
    "side_effects",
    "ast_edges",
    "ast_symbols",
    "code_files",
    "repo_commits",
    "repositories",
    "confluence_pages",
    "jira_issues",
    "doc_chunks",
    "execution_telemetry",
    "indexing_queue",
    "indexing_dead_letter",
    "connector_sync_watermarks",
    "embedding_spaces",
    "domain_sync_manifests",
    "domain_packs",
)

_TENANT_TABLES_CHILD_TO_PARENT: Final[tuple[str, ...]] = tuple(
    t for t in ALL_RLS_TABLES_FK_DELETE_ORDER if t != "repo_commits"
)


def _sanitize_suite_slug(suite: str) -> str:
    slug = re.sub(r"[^a-z0-9_]+", "_", suite.lower()).strip("_")
    return slug[:32] or "suite"


def make_ephemeral_pg_domain_id(suite: str = "suite") -> str:
    """Return a unique ephemeral PostgreSQL domain identifier with ``eval_`` prefix."""
    slug = _sanitize_suite_slug(suite)
    return f"{EPHEMERAL_PG_DOMAIN_PREFIX}{slug}_{uuid.uuid4().hex[:8]}"


def validate_ephemeral_pg_domain_id(domain_id: str) -> str:
    """Refuse any domain_id that is not a well-formed ``eval_*`` session domain."""
    if (
        not isinstance(domain_id, str)
        or not domain_id.startswith(EPHEMERAL_PG_DOMAIN_PREFIX)
        or len(domain_id) <= len(EPHEMERAL_PG_DOMAIN_PREFIX)
    ):
        raise ValueError(
            f"Refusing PostgreSQL isolation operation for non-ephemeral domain_id={domain_id!r}. "
            f"Only '{EPHEMERAL_PG_DOMAIN_PREFIX}<suite>_<uuid>' domains are permitted."
        )
    return domain_id


def create_ephemeral_pg_domain(
    conn: psycopg.Connection[Any],
    suite: str = "suite",
    *,
    domain_id: str | None = None,
    display_name: str | None = None,
    lora_adapter_id: str = "eval-none",
    primary_truth_source: str = "code",
    freshness_half_life_days: int = 180,
    config: dict[str, Any] | None = None,
    tenant_id: UUID = DEFAULT_TENANT_ID,
) -> str:
    """Register a unique ``eval_*`` domain in ``domain_packs`` and commit."""
    resolved_id = (
        validate_ephemeral_pg_domain_id(domain_id)
        if domain_id is not None
        else make_ephemeral_pg_domain_id(suite)
    )
    resolved_name = display_name or f"Eval Domain ({resolved_id})"
    resolved_config = json.dumps(config or {})
    conn.execute(
        """
        INSERT INTO domain_packs
            (tenant_id, domain_id, display_name, lora_adapter_id, primary_truth_source,
             freshness_half_life_days, config_json)
        VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
        ON CONFLICT (tenant_id, domain_id) DO NOTHING
        """,
        (
            tenant_id,
            resolved_id,
            resolved_name,
            lora_adapter_id,
            primary_truth_source,
            freshness_half_life_days,
            resolved_config,
        ),
    )
    conn.commit()
    return resolved_id


def cleanup_pg_tenants_and_eval_domains(
    admin_conn: psycopg.Connection[Any],
    *,
    tenant_ids: Sequence[UUID] = (),
    domain_ids: Sequence[str] = (),
) -> dict[str, int]:
    """Delete rows strictly belonging to isolated ``tenant_ids`` and/or ``eval_*`` domains."""
    deleted: dict[str, int] = {table: 0 for table in ALL_RLS_TABLES_FK_DELETE_ORDER}
    clean_tenants = list(dict.fromkeys(tenant_ids))
    for tid in clean_tenants:
        if tid == DEFAULT_TENANT_ID:
            raise ValueError(
                f"Refusing tenant cleanup for protected DEFAULT_TENANT_ID ({DEFAULT_TENANT_ID})."
            )
    clean_domains = [validate_ephemeral_pg_domain_id(d) for d in dict.fromkeys(domain_ids)]

    if clean_tenants:
        for table in ALL_RLS_TABLES_FK_DELETE_ORDER:
            cur = admin_conn.execute(
                f"DELETE FROM {table} WHERE tenant_id = ANY(%s)",
                (clean_tenants,),
            )
            deleted[table] += max(cur.rowcount or 0, 0)

    if clean_domains:
        repo_commits_cur = admin_conn.execute(
            """
            DELETE FROM repo_commits
            WHERE repo_id IN (SELECT repo_id FROM repositories WHERE domain_id = ANY(%s))
            """,
            (clean_domains,),
        )
        deleted["repo_commits"] += max(repo_commits_cur.rowcount or 0, 0)
        for table in _TENANT_TABLES_CHILD_TO_PARENT:
            cur = admin_conn.execute(
                f"DELETE FROM {table} WHERE domain_id = ANY(%s)",
                (clean_domains,),
            )
            deleted[table] += max(cur.rowcount or 0, 0)

    admin_conn.commit()
    return deleted


def cleanup_ephemeral_pg_domain(
    conn: psycopg.Connection[Any],
    domain_id: str,
) -> dict[str, int]:
    """Delete all rows belonging strictly to ``domain_id`` in child-to-parent order."""
    return cleanup_pg_tenants_and_eval_domains(conn, domain_ids=(domain_id,))


_SEED_ALL_SCOPES: Final[list[str]] = [
    "public",
    "public_eng",
    "public_docs",
    "docs_restricted",
    "corpus:enterprise_docs",
    "repo:code_architecture_scout",
]

_REQUIRED_DOMAIN_PACKS: Final[tuple[tuple[str, str, str, str, int], ...]] = (
    (
        "code_architecture_scout",
        "Code Architecture Scout",
        "gemma4_26b_code_scout",
        "code",
        90,
    ),
    (
        "enterprise_docs_scout",
        "Enterprise Docs & ADRs Scout",
        "gemma4_26b_enterprise_docs",
        "document",
        180,
    ),
    (
        "data_engineering_scout",
        "Data Engineering & Distributed Systems",
        "gemma4_26b_data_engineering",
        "document",
        180,
    ),
    (
        "polyglot_unseen_v2",
        "Polyglot Unseen Benchmark v2",
        "gemma4_26b_polyglot_unseen_v2",
        "code",
        90,
    ),
)

_NEUTRAL_GITHUB_ISSUES_SEED: Final[tuple[dict[str, object], ...]] = (
    {
        "issue_number": 101,
        "title": "Enforce transaction-scoped SET LOCAL tenant_id in RLSConnectionManager",
        "state": "open",
        "section": "Issue Description and Root Cause Analysis",
        "content": (
            "Ensure RLSConnectionManager sets app.tenant_id and app.principal_scopes via "
            "set_config(..., is_local=true) inside explicit conn.transaction() blocks under "
            "NOBYPASSRLS so pooled connections never leak tenant context across requests."
        ),
        "labels": ["security", "postgresql", "rls"],
    },
    {
        "issue_number": 102,
        "title": "Support 2-hop AST type and exception closure in PreFlightValidator",
        "state": "open",
        "section": "Architectural Proposal",
        "content": (
            "Extend ASTDependencyExpander to traverse USES_TYPE and TESTED_BY edges up to "
            "two hops within the 10,000-token expansion budget while preserving full symbol bodies."
        ),
        "labels": ["feature", "ast_engine", "preflight"],
    },
    {
        "issue_number": 103,
        "title": "Bound PG17 HNSW iterative_scan and statement_timeout on query pools",
        "state": "closed",
        "section": "Resolution and Performance Audit",
        "content": (
            "Configured hnsw.iterative_scan = 'relaxed_order' and 5,000ms statement_timeout on "
            "interactive query pools, keeping 1-hop neighborhood traversal under 50ms."
        ),
        "labels": ["performance", "pgvector", "hnsw"],
    },
    {
        "issue_number": 104,
        "title": "Fail-closed zero-spend abstention when citation overlap falls below 0.98",
        "state": "closed",
        "section": "Patch Notes and Benchmark Validation",
        "content": (
            "DocumentCitationGate rejects unverified or paraphrased quotes deterministically at "
            "Stage 4 with paid_tokens_spent=0 before any frontier model handoff."
        ),
        "labels": ["reliability", "citations", "abstention"],
    },
    {
        "issue_number": 105,
        "title": "Normalize NFKC Unicode identifiers and strip bidi overrides in Guardrails",
        "state": "open",
        "section": "Security Hardening Notes",
        "content": (
            "Apply NFKC normalization and bidi/zero-width stripping before prompt-injection and "
            "AST symbol verification so homoglyph spoofing cannot bypass PreFlight checks."
        ),
        "labels": ["security", "guardrails", "unicode"],
    },
)


def _git_blob_sha(content: str) -> str:
    raw = content.encode("utf-8")
    header = f"blob {len(raw)}\0".encode()
    return hashlib.sha1(header + raw).hexdigest()


def _seed_code_architecture_scout(
    conn: psycopg.Connection[tuple[object, ...]],
    repo: PostgresRLSRepository,
    repo_root: Path,
) -> None:
    domain_id = "code_architecture_scout"
    existing_paths = {
        str(row[0])
        for row in conn.execute(
            "SELECT file_path FROM code_files WHERE domain_id = %s",
            (domain_id,),
        ).fetchall()
    }
    extractor = ASTExtractor()
    all_files = sorted((repo_root / "src" / "context_scout").rglob("*.py"))
    commit_sha = hashlib.sha1(b"m0-03-code-arch-seed").hexdigest()[:40]
    scopes = ["public_eng", "repo:code_architecture_scout"]

    global_symbol_map: dict[str, UUID] = {}
    short_name_map: dict[str, list[UUID]] = defaultdict(list)
    parsed_files: list[tuple[list[Any], dict[str, UUID]]] = []

    for py_file in all_files:
        rel_path = py_file.relative_to(repo_root).as_posix()
        if rel_path in existing_paths:
            continue
        raw_content = py_file.read_text(encoding="utf-8", errors="replace")
        file_id = uuid5(NAMESPACE_URL, f"{domain_id}:{rel_path}")
        file_rec = CodeFileRecord(
            file_id=file_id,
            domain_id=domain_id,
            repo_name="enterprise-context-scout",
            file_path=rel_path,
            git_commit_sha=commit_sha,
            git_blob_sha=_git_blob_sha(raw_content),
            language="python",
            allowed_scopes=scopes,
            raw_content=raw_content,
            line_count=len(raw_content.splitlines()),
        )
        repo.insert_code_file(conn, file_rec)

        symbols, edges = extractor.parse_source(raw_content, "python", file_path=rel_path)
        file_sym_map: dict[str, UUID] = {}
        for parsed_sym in symbols:
            sym_id = uuid5(
                NAMESPACE_URL,
                f"{domain_id}:{rel_path}:{parsed_sym.qualified_name}:{parsed_sym.start_line}",
            )
            if sym_id in file_sym_map.values():
                sym_id = uuid4()
            emb = _embed_fixture_text(f"{parsed_sym.signature_skeleton} {parsed_sym.docstring}")
            sym_rec = ASTSymbolRecord(
                symbol_id=sym_id,
                file_id=file_id,
                domain_id=domain_id,
                qualified_name=parsed_sym.qualified_name,
                symbol_kind=parsed_sym.symbol_kind,
                start_line=parsed_sym.start_line,
                end_line=parsed_sym.end_line,
                signature_skeleton=parsed_sym.signature_skeleton,
                contextual_summary=parsed_sym.contextual_summary,
                allowed_scopes=scopes,
                embedding=emb,
            )
            repo.insert_ast_symbol(conn, sym_rec)
            file_sym_map[parsed_sym.qualified_name] = sym_id
            global_symbol_map[parsed_sym.qualified_name] = sym_id
            short_name_map[parsed_sym.name].append(sym_id)

        parsed_files.append((edges, file_sym_map))

    for edges, file_sym_map in parsed_files:
        for edge in edges:
            src_id = file_sym_map.get(edge.source_qualified_name) or global_symbol_map.get(
                edge.source_qualified_name
            )
            if not src_id:
                continue
            target_id = global_symbol_map.get(edge.target_qualified_name)
            if not target_id:
                target_short = edge.target_qualified_name.split(".")[-1]
                matches = short_name_map.get(target_short, [])
                if len(matches) == 1:
                    target_id = matches[0]
            edge_rec = ASTEdgeRecord(
                edge_id=uuid4(),
                domain_id=domain_id,
                source_symbol_id=src_id,
                target_qualified_name=edge.target_qualified_name,
                target_symbol_id=target_id,
                edge_kind=edge.edge_kind,
                allowed_scopes=scopes,
            )
            repo.insert_ast_edge(conn, edge_rec)

    # Seed the 5 gh-issue:* doc chunks into code_architecture_scout
    existing_docs = {
        str(row[0])
        for row in conn.execute(
            "SELECT doc_id FROM doc_chunks WHERE domain_id = %s AND doc_id LIKE 'gh-issue:%%'",
            (domain_id,),
        ).fetchall()
    }
    for issue in _NEUTRAL_GITHUB_ISSUES_SEED:
        num = int(str(issue["issue_number"]))
        doc_id = f"gh-issue:{num}"
        if doc_id in existing_docs:
            continue
        title = str(issue["title"])
        content = str(issue["content"])
        section = str(issue["section"])
        emb = _embed_fixture_text(f"{title} {section} {content}")
        rec = DocChunkRecord(
            chunk_id=uuid5(NAMESPACE_URL, f"{domain_id}:{doc_id}"),
            domain_id=domain_id,
            doc_id=doc_id,
            doc_title=title,
            source_uri=f"https://github.com/enterprise-context-scout/issues/{num}",
            market_scope="GLOBAL",
            published_date=date(2026, 1, 15),
            authority_weight=1.0,
            section_heading=section,
            contextual_header=f"GitHub Issue #{num} [{domain_id}]",
            verbatim_content=content,
            metadata_json={
                "issue_number": num,
                "state": issue["state"],
                "labels": issue["labels"],
            },
            allowed_scopes=scopes,
            embedding=emb,
        )
        repo.insert_doc_chunk(conn, rec)


def _seed_enterprise_docs_scout(
    conn: psycopg.Connection[tuple[object, ...]],
    repo: PostgresRLSRepository,
) -> None:
    domain_id = "enterprise_docs_scout"
    existing_docs = {
        str(row[0])
        for row in conn.execute(
            "SELECT doc_id FROM doc_chunks WHERE domain_id = %s",
            (domain_id,),
        ).fetchall()
    }
    plugin = EnterpriseCoreDomainPlugin()
    demo_chunks: list[DocChunkRecord] = plugin.get_demo_seed_payloads()["doc_chunks"]
    for chunk in demo_chunks:
        if chunk.doc_id not in existing_docs:
            repo.insert_doc_chunk(conn, chunk)
            existing_docs.add(chunk.doc_id)

    research_scopes = [
        "public",
        "public_eng",
        "public_docs",
        "corpus:enterprise_docs",
    ]
    extra_specs: list[tuple[str, str, str, str, str, str]] = [
        (
            "doc-preflight-sufficiency-gate",
            "Pre-Flight Sufficiency Gate: 6 Deterministic Checks Preventing Hallucinations",
            "urn:enterprise_docs_scout:doc-preflight-sufficiency-gate",
            "GLOBAL",
            "6 Deterministic Pre-Flight Checks",
            (
                "The Pre-Flight Sufficiency Gate prevents hallucinations and wastes $0 in "
                "frontier tokens by enforcing 6 deterministic checks before any synthesis or "
                "handoff: (1) Tree-sitter syntax verification (0 syntax errors), "
                "(2) 100% AST symbol closure (0 unresolved references), "
                "(3) calibrated RRF hybrid retrieval score threshold (>= 0.65), "
                "(4) strict RBAC/RLS scope containment (0 unauthorized items), "
                "(5) 100% verbatim provenance anchoring to indexed file paths and line ranges, "
                "and (6) citation freshness and non-staleness verification. If any check fails, "
                "the gate abstains deterministically with paid_tokens_spent=0."
            ),
        ),
        (
            "doc-pg17-force-rls-isolation",
            "PostgreSQL 17 FORCE ROW LEVEL SECURITY and SET LOCAL Tenant Isolation",
            "urn:enterprise_docs_scout:doc-pg17-force-rls-isolation",
            "GLOBAL",
            "Tenant & Scope Enforcement",
            (
                "Every PostgreSQL 17 table enforces tenant_id UUID NOT NULL, ENABLE ROW LEVEL "
                "SECURITY, and FORCE ROW LEVEL SECURITY under NOBYPASSRLS. Pools apply "
                "set_config('app.tenant_id', ..., true) and set_config('app.principal_scopes', "
                "..., true) inside explicit conn.transaction() blocks."
            ),
        ),
        (
            "doc-rrf-hybrid-retrieval",
            "Three-Way Reciprocal Rank Fusion (RRF) Hybrid Retrieval Calibration",
            "urn:enterprise_docs_scout:doc-rrf-hybrid-retrieval",
            "GLOBAL",
            "Ranking Formula & Thresholds",
            (
                "Hybrid retrieval combines pg_trgm trigram similarity, BM25 lexical scoring, and "
                "pgvector HNSW cosine distance via normalized Reciprocal Rank Fusion: "
                "RRF(d) = sum(1 / (60 + rank_i(d))) with hnsw.iterative_scan = 'relaxed_order'."
            ),
        ),
        (
            "doc-zero-trust-iap-oidc",
            "Zero-Trust Cloud IAP, OIDC JWKS, and HMAC Principal Attestation",
            "urn:enterprise_docs_scout:doc-zero-trust-iap-oidc",
            "GLOBAL",
            "Out-of-Band Authentication",
            (
                "PrincipalContext is extracted strictly out-of-band from Cloud IAP ES256 "
                "assertions, OIDC JWKS Bearer JWTs, or HMAC-SHA256 signed tokens. Prompt text "
                "and client X-Principal-* headers can never elevate scopes or tenant_id."
            ),
        ),
        (
            "doc-rslora-scope-ceiling",
            "ACL-Safe rsLoRA Specialization and Training Scope Ceiling",
            "urn:enterprise_docs_scout:doc-rslora-scope-ceiling",
            "GLOBAL",
            "Adapter Governance",
            (
                "AdapterRouter enforces training_scope_ceiling containment and verified "
                "safetensors weights before activating any rsLoRA cartridge, failing closed to "
                "base open-weights whenever a principal holds scopes outside the ceiling."
            ),
        ),
        (
            "doc-regional-compliance-gov",
            "Regional APAC, EU, and US Compliance Governance Standard",
            "urn:enterprise_docs_scout:doc-regional-compliance-gov",
            "ENTERPRISE_DOC",
            "Cross-Region Controls",
            (
                "Regional governance enforces EU GDPR data residency (europe-west3 default), "
                "BetrVG works-council HMAC pseudonymization, and hash-chained audit logs "
                "across all deployed enterprise execution pipelines."
            ),
        ),
    ]
    for doc_id, title, uri, scope, heading, content in extra_specs:
        if doc_id in existing_docs:
            continue
        emb = _embed_fixture_text(f"{title} {heading} {content}")
        rec = DocChunkRecord(
            chunk_id=uuid5(NAMESPACE_URL, f"{domain_id}:{doc_id}"),
            domain_id=domain_id,
            doc_id=doc_id,
            doc_title=title,
            source_uri=uri,
            market_scope=scope,
            published_date=date(2026, 2, 1),
            authority_weight=1.0,
            section_heading=heading,
            contextual_header=f"{title} | {heading}",
            verbatim_content=content,
            metadata_json={"seeded_by": "bootstrap_pg17_session"},
            allowed_scopes=research_scopes,
            embedding=emb,
        )
        repo.insert_doc_chunk(conn, rec)


def _seed_data_engineering_scout(
    conn: psycopg.Connection[tuple[object, ...]],
    repo: PostgresRLSRepository,
) -> None:
    domain_id = "data_engineering_scout"
    specs = [
        (
            "de-acid-transactions",
            "ACID Transaction Isolation and MVCC Guarantees",
            "urn:data_engineering_scout:acid-transactions",
            "Atomicity & Isolation",
            (
                "ACID databases guarantee Atomicity, Consistency, Isolation, and Durability "
                "using write-ahead logging (WAL) and multi-version concurrency control (MVCC) "
                "to prevent dirty reads and lost updates under concurrent workloads."
            ),
        ),
        (
            "de-cap-theorem",
            "CAP Theorem Trade-offs in Distributed Storage",
            "urn:data_engineering_scout:cap-theorem",
            "Consistency vs Availability",
            (
                "Under network partitions (P), distributed data stores must choose between "
                "linearizable Consistency (CP) via quorum consensus such as Raft/Paxos or "
                "high Availability (AP) with eventual consistency and conflict resolution."
            ),
        ),
        (
            "de-apache-iceberg",
            "Apache Iceberg Open Table Format and Snapshot Isolation",
            "urn:data_engineering_scout:apache-iceberg",
            "Snapshot Metadata",
            (
                "Apache Iceberg provides ACID snapshot isolation, hidden partitioning, and "
                "time-travel queries over object storage using immutable manifest lists and "
                "optimistic concurrency atomic metadata pointer swaps."
            ),
        ),
    ]
    for doc_id, title, uri, heading, content in specs:
        emb = _embed_fixture_text(f"{title} {heading} {content}")
        rec = DocChunkRecord(
            chunk_id=uuid5(NAMESPACE_URL, f"{domain_id}:{doc_id}"),
            domain_id=domain_id,
            doc_id=doc_id,
            doc_title=title,
            source_uri=uri,
            market_scope="GLOBAL",
            published_date=date(2026, 1, 20),
            authority_weight=1.0,
            section_heading=heading,
            contextual_header=f"{title} | {heading}",
            verbatim_content=content,
            metadata_json={"seeded_by": "bootstrap_pg17_session"},
            allowed_scopes=["public", "public_docs"],
            embedding=emb,
        )
        repo.insert_doc_chunk(conn, rec)


def _seed_polyglot_unseen_v2(
    conn: psycopg.Connection[tuple[object, ...]],
    repo: PostgresRLSRepository,
) -> None:
    domain_id = "polyglot_unseen_v2"
    commit_sha = hashlib.sha1(b"polyglot-unseen-v2-seed").hexdigest()[:40]
    scopes = ["public_eng"]

    # 1. httpx connection pooling
    httpx_file_id = uuid5(NAMESPACE_URL, f"{domain_id}:httpx/_transports/default.py")
    httpx_src = (
        "class HTTPTransport:\n"
        "    def __init__(self, limits: Limits | None = None) -> None:\n"
        "        self._pool = ConnectionPool(max_connections=limits.max_connections)\n"
    )
    repo.insert_code_file(
        conn,
        CodeFileRecord(
            file_id=httpx_file_id,
            domain_id=domain_id,
            repo_name="encode/httpx",
            file_path="httpx/_transports/default.py",
            git_commit_sha=commit_sha,
            git_blob_sha=_git_blob_sha(httpx_src),
            language="python",
            allowed_scopes=scopes,
            raw_content=httpx_src,
            line_count=3,
        ),
    )
    repo.insert_ast_symbol(
        conn,
        ASTSymbolRecord(
            symbol_id=uuid5(NAMESPACE_URL, f"{domain_id}:httpx.HTTPTransport"),
            file_id=httpx_file_id,
            domain_id=domain_id,
            qualified_name="httpx._transports.default.HTTPTransport",
            symbol_kind="class",
            start_line=1,
            end_line=3,
            signature_skeleton="class HTTPTransport:",
            contextual_summary="Default synchronous HTTP connection pooling transport.",
            allowed_scopes=scopes,
            embedding=_embed_fixture_text("httpx HTTPTransport connection pooling"),
        ),
    )

    # 2. starlette ASGI lifespan
    starlette_file_id = uuid5(NAMESPACE_URL, f"{domain_id}:starlette/routing.py")
    starlette_src = (
        "class Lifespan:\n"
        "    async def startup(self) -> None:\n"
        "        await self.lifespan_context(self.app)\n"
    )
    repo.insert_code_file(
        conn,
        CodeFileRecord(
            file_id=starlette_file_id,
            domain_id=domain_id,
            repo_name="encode/starlette",
            file_path="starlette/routing.py",
            git_commit_sha=commit_sha,
            git_blob_sha=_git_blob_sha(starlette_src),
            language="python",
            allowed_scopes=scopes,
            raw_content=starlette_src,
            line_count=3,
        ),
    )
    repo.insert_ast_symbol(
        conn,
        ASTSymbolRecord(
            symbol_id=uuid5(NAMESPACE_URL, f"{domain_id}:starlette.Lifespan"),
            file_id=starlette_file_id,
            domain_id=domain_id,
            qualified_name="starlette.routing.Lifespan",
            symbol_kind="class",
            start_line=1,
            end_line=3,
            signature_skeleton="class Lifespan:",
            contextual_summary="ASGI lifespan startup and shutdown manager.",
            allowed_scopes=scopes,
            embedding=_embed_fixture_text("starlette ASGI lifespan routing"),
        ),
    )

    # 3. anyio find_root_task + caller get_task_info + CALLS edge
    anyio_file_id = uuid5(NAMESPACE_URL, f"{domain_id}:anyio/_backends/_asyncio.py")
    anyio_src = (
        "def find_root_task(task: object) -> object:\n"
        "    return task\n\n"
        "def get_task_info(task: object) -> object:\n"
        "    return find_root_task(task)\n"
    )
    repo.insert_code_file(
        conn,
        CodeFileRecord(
            file_id=anyio_file_id,
            domain_id=domain_id,
            repo_name="agronholm/anyio",
            file_path="anyio/_backends/_asyncio.py",
            git_commit_sha=commit_sha,
            git_blob_sha=_git_blob_sha(anyio_src),
            language="python",
            allowed_scopes=scopes,
            raw_content=anyio_src,
            line_count=5,
        ),
    )
    find_root_id = uuid5(NAMESPACE_URL, f"{domain_id}:anyio._backends._asyncio.find_root_task")
    get_task_id = uuid5(NAMESPACE_URL, f"{domain_id}:anyio._backends._asyncio.get_task_info")
    repo.insert_ast_symbol(
        conn,
        ASTSymbolRecord(
            symbol_id=find_root_id,
            file_id=anyio_file_id,
            domain_id=domain_id,
            qualified_name="anyio._backends._asyncio.find_root_task",
            symbol_kind="function",
            start_line=1,
            end_line=2,
            signature_skeleton="def find_root_task(task: object) -> object:",
            contextual_summary="Resolves the root asyncio task in anyio task tree.",
            allowed_scopes=scopes,
            embedding=_embed_fixture_text("anyio._backends._asyncio.find_root_task"),
        ),
    )
    repo.insert_ast_symbol(
        conn,
        ASTSymbolRecord(
            symbol_id=get_task_id,
            file_id=anyio_file_id,
            domain_id=domain_id,
            qualified_name="anyio._backends._asyncio.get_task_info",
            symbol_kind="function",
            start_line=4,
            end_line=5,
            signature_skeleton="def get_task_info(task: object) -> object:",
            contextual_summary="Returns task metadata by calling find_root_task.",
            allowed_scopes=scopes,
            embedding=_embed_fixture_text("anyio._backends._asyncio.get_task_info"),
        ),
    )
    repo.insert_ast_edge(
        conn,
        ASTEdgeRecord(
            edge_id=uuid4(),
            domain_id=domain_id,
            source_symbol_id=get_task_id,
            target_qualified_name="anyio._backends._asyncio.find_root_task",
            target_symbol_id=find_root_id,
            edge_kind="CALLS",
            allowed_scopes=scopes,
        ),
    )

    # 4. Go chi router mux.go (Mux.Use + Mux.ServeHTTP) for clean-checkout A-POLY-01 parity
    chi_file_id = uuid5(NAMESPACE_URL, f"{domain_id}:go-chi/chi:mux.go")
    chi_src = (
        "package chi\n\n"
        'import "net/http"\n\n'
        "type Mux struct {\n"
        "    middlewares []func(http.Handler) http.Handler\n"
        "    handler     http.Handler\n"
        "}\n\n"
        "func (mx *Mux) Use(middlewares ...func(http.Handler) http.Handler) {\n"
        "    mx.middlewares = append(mx.middlewares, middlewares...)\n"
        "}\n\n"
        "func (mx *Mux) ServeHTTP(w http.ResponseWriter, r *http.Request) {\n"
        "    mx.handler.ServeHTTP(w, r)\n"
        "}\n"
    )
    repo.insert_code_file(
        conn,
        CodeFileRecord(
            file_id=chi_file_id,
            domain_id=domain_id,
            repo_name="go-chi/chi",
            file_path="mux.go",
            git_commit_sha=commit_sha,
            git_blob_sha=_git_blob_sha(chi_src),
            language="go",
            allowed_scopes=scopes,
            raw_content=chi_src,
            line_count=len(chi_src.splitlines()),
        ),
    )
    mux_use_id = uuid5(NAMESPACE_URL, f"{domain_id}:chi.Mux.Use")
    mux_serve_id = uuid5(NAMESPACE_URL, f"{domain_id}:chi.Mux.ServeHTTP")
    repo.insert_ast_symbol(
        conn,
        ASTSymbolRecord(
            symbol_id=mux_use_id,
            file_id=chi_file_id,
            domain_id=domain_id,
            qualified_name="chi.Mux.Use",
            symbol_kind="method",
            start_line=10,
            end_line=12,
            signature_skeleton="func (mx *Mux) Use(middlewares ...func(http.Handler) http.Handler)",
            contextual_summary="Appends HTTP middleware handlers onto the chi Mux stack.",
            allowed_scopes=scopes,
            embedding=_embed_fixture_text(
                "Go chi router mux.go Mux.Use compose middleware handlers"
            ),
        ),
    )
    repo.insert_ast_symbol(
        conn,
        ASTSymbolRecord(
            symbol_id=mux_serve_id,
            file_id=chi_file_id,
            domain_id=domain_id,
            qualified_name="chi.Mux.ServeHTTP",
            symbol_kind="method",
            start_line=14,
            end_line=16,
            signature_skeleton="func (mx *Mux) ServeHTTP(w http.ResponseWriter, r *http.Request)",
            contextual_summary="Dispatches HTTP requests through chi Mux routes and middleware.",
            allowed_scopes=scopes,
            embedding=_embed_fixture_text(
                "Go chi router mux.go Mux.ServeHTTP dispatch HTTP routes"
            ),
        ),
    )


def ensure_pg17_schema_and_seed_corpora(
    db_admin_url: str,
    db_app_url: str,
    repo_root: Path,
) -> None:
    """Idempotently bootstrap PG17 schema via Alembic, domain_packs, and benchmark corpora."""
    advisory_lock_id = 7391042
    with psycopg.connect(db_admin_url, autocommit=True) as lock_conn:
        lock_conn.execute("SELECT pg_advisory_lock(%s)", (advisory_lock_id,))
        try:
            try:
                from context_scout.db.migrations_runner import run_alembic_upgrade

                run_alembic_upgrade(db_admin_url)
            except ModuleNotFoundError:
                with psycopg.connect(db_admin_url) as check_conn:
                    has_alembic = (
                        check_conn.execute(
                            "SELECT to_regclass('public.alembic_version')"
                        ).fetchone()
                        or (None,)
                    )[0]
                    if has_alembic is None:
                        raise

            with psycopg.connect(db_admin_url) as admin_conn:
                if os.environ.get("SCOUT_PARALLEL_PYTEST") != "1":
                    stale_eval_rows = admin_conn.execute(
                        "SELECT DISTINCT domain_id FROM domain_packs "
                        "WHERE domain_id LIKE 'eval_%%' "
                        "AND created_at < NOW() - INTERVAL '2 hours'"
                    ).fetchall()
                    for (stale_eval_id,) in stale_eval_rows:
                        cleanup_ephemeral_pg_domain(admin_conn, str(stale_eval_id))
                for d_id, display_name, adapter_id, truth_src, half_life in _REQUIRED_DOMAIN_PACKS:
                    admin_conn.execute(
                        """
                        INSERT INTO domain_packs (
                            tenant_id, domain_id, display_name, lora_adapter_id,
                            primary_truth_source, freshness_half_life_days, config_json
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, '{}'::jsonb)
                        ON CONFLICT (tenant_id, domain_id) DO NOTHING
                        """,
                        (DEFAULT_TENANT_ID, d_id, display_name, adapter_id, truth_src, half_life),
                    )
                admin_conn.commit()

            seed_principal = PrincipalContext(
                user_id="pg_isolation_bootstrap",
                role=RoleType.ENGINEER,
                allowed_scopes=_SEED_ALL_SCOPES,
            )

            def _row_count(row: tuple[object, ...] | None) -> int:
                return int(str(row[0])) if row and row[0] is not None else 0

            with RLSConnectionManager(db_app_url, pool_role=PoolRole.INGESTION) as mgr:
                repo = PostgresRLSRepository()
                with mgr.scoped_transaction(
                    seed_principal, enable_hnsw_iterative_scan=False
                ) as check_conn:
                    need_code_arch = (
                        _row_count(
                            check_conn.execute(
                                "SELECT count(*) FROM code_files "
                                "WHERE domain_id = 'code_architecture_scout'"
                            ).fetchone()
                        )
                        < 43
                    )
                    need_mr42 = (
                        _row_count(
                            check_conn.execute(
                                "SELECT count(*) FROM ref_snapshots "
                                "WHERE domain_id = 'code_architecture_scout' "
                                "AND ref_name = 'mr/42'"
                            ).fetchone()
                        )
                        == 0
                    )
                    need_ent_docs = (
                        _row_count(
                            check_conn.execute(
                                "SELECT count(*) FROM doc_chunks "
                                "WHERE domain_id = 'enterprise_docs_scout'"
                            ).fetchone()
                        )
                        < 9
                    )
                    need_de = (
                        _row_count(
                            check_conn.execute(
                                "SELECT count(*) FROM doc_chunks "
                                "WHERE domain_id = 'data_engineering_scout' "
                                "AND doc_id = 'de-cap-theorem'"
                            ).fetchone()
                        )
                        == 0
                    )
                    need_poly = (
                        _row_count(
                            check_conn.execute(
                                "SELECT count(*) FROM code_files "
                                "WHERE domain_id = 'polyglot_unseen_v2' "
                                "AND file_path = 'httpx/_transports/default.py'"
                            ).fetchone()
                        )
                        == 0
                    )

                if need_code_arch or need_ent_docs or need_de or need_poly:
                    # Pre-warm the neural embedding model outside any open DB transaction so cold
                    # loads never trigger PG17 idle_in_transaction_session_timeout, and seeded
                    # vectors live in the exact same neural space as live queries.
                    _embed_fixture_text("bootstrap warmup")

                with mgr.scoped_transaction(
                    seed_principal, enable_hnsw_iterative_scan=False
                ) as conn:
                    if need_code_arch:
                        _seed_code_architecture_scout(conn, repo, repo_root)

                    if need_mr42:
                        from context_scout.db.repositories import (
                            OverlayFileDeltaRecord,
                            RefSnapshotRecord,
                        )

                        reconciler_path = repo_root / "src/context_scout/ops/state_reconciler.py"
                        if reconciler_path.is_file():
                            base_content = reconciler_path.read_text(
                                encoding="utf-8", errors="ignore"
                            )
                            overlay_addition = (
                                "\n\n"
                                "def verify_pitr_overlay_watermark_lag(\n"
                                "    pg_lsn_lag_seconds: float,\n"
                                "    max_allowed_lag_seconds: float = 30.0,\n"
                                ") -> bool:\n"
                                '    """MR !42 CoW overlay guard bounding PITR watermark lag."""\n'
                                "    return 0.0 <= pg_lsn_lag_seconds <= max_allowed_lag_seconds\n"
                            )
                            overlay_content = base_content.rstrip("\n") + overlay_addition
                            base_lines = len(base_content.rstrip("\n").splitlines())
                            mr_snapshot_id = uuid5(
                                NAMESPACE_URL,
                                f"{DEFAULT_TENANT_ID}:code_architecture_scout:mr_snapshot:42",
                            )
                            base_sha = _git_blob_sha(base_content)
                            head_sha = _git_blob_sha(overlay_content)
                            repo.upsert_ref_snapshot(
                                conn,
                                RefSnapshotRecord(
                                    snapshot_id=mr_snapshot_id,
                                    domain_id="code_architecture_scout",
                                    repo_name="enterprise-context-scout",
                                    ref_name="mr/42",
                                    ref_kind="merge_request",
                                    base_commit_sha=base_sha,
                                    head_commit_sha=head_sha,
                                    allowed_scopes=["public_eng"],
                                    mr_iid="42",
                                    state="open",
                                    metadata_json={
                                        "title": (
                                            "feat(ops): bound PITR watermark lag in StateReconciler"
                                        ),
                                        "seeded_by": "pg_isolation_bootstrap",
                                    },
                                    tenant_id=DEFAULT_TENANT_ID,
                                ),
                            )
                            repo.upsert_overlay_file_delta(
                                conn,
                                OverlayFileDeltaRecord(
                                    delta_id=uuid5(
                                        NAMESPACE_URL,
                                        f"{DEFAULT_TENANT_ID}:code_architecture_scout:"
                                        "mr_delta:42:state_reconciler.py",
                                    ),
                                    snapshot_id=mr_snapshot_id,
                                    domain_id="code_architecture_scout",
                                    repo_name="enterprise-context-scout",
                                    file_path="src/context_scout/ops/state_reconciler.py",
                                    change_type="modified",
                                    allowed_scopes=["public_eng"],
                                    previous_file_path="src/context_scout/ops/state_reconciler.py",
                                    base_blob_sha=base_sha,
                                    overlay_blob_sha=head_sha,
                                    language="python",
                                    raw_content=overlay_content,
                                    line_count=len(overlay_content.splitlines()),
                                    symbols_json=[
                                        {
                                            "qualified_name": (
                                                "context_scout.ops.state_reconciler."
                                                "verify_pitr_overlay_watermark_lag"
                                            ),
                                            "name": "verify_pitr_overlay_watermark_lag",
                                            "symbol_kind": "function",
                                            "start_line": base_lines + 3,
                                            "end_line": len(overlay_content.splitlines()),
                                        }
                                    ],
                                    tenant_id=DEFAULT_TENANT_ID,
                                ),
                            )

                    if need_ent_docs:
                        _seed_enterprise_docs_scout(conn, repo)
                    conn.execute(
                        """
                        UPDATE doc_chunks
                        SET allowed_scopes = array_remove(allowed_scopes, 'docs_restricted')
                        WHERE domain_id = 'enterprise_docs_scout'
                          AND allowed_scopes @> ARRAY['docs_restricted', 'public']::text[]
                        """
                    )

                    if need_de:
                        _seed_data_engineering_scout(conn, repo)

                    if need_poly:
                        _seed_polyglot_unseen_v2(conn, repo)
        finally:
            lock_conn.execute("SELECT pg_advisory_unlock(%s)", (advisory_lock_id,))
