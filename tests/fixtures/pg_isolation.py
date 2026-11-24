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
from datetime import date
from pathlib import Path
from typing import Any, Final
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

import numpy as np
import psycopg
import yaml
from safetensors.numpy import save_file as save_safetensors_numpy

import context_scout.retrieval.embedder as _embedder_mod
import context_scout.schemas.contracts as _contracts_mod
from context_scout.ast_engine.extractor import ASTExtractor
from context_scout.config import ScoutSettings
from context_scout.db.repositories import PostgresRLSRepository
from context_scout.db.schema import (
    ASTEdgeRecord,
    ASTSymbolRecord,
    CodeFileRecord,
    DocChunkRecord,
)
from context_scout.db.session import RLSConnectionManager
from context_scout.plugins.domains.enterprise_core import EnterpriseCoreDomainPlugin
from context_scout.schemas.contracts import PrincipalContext, RoleType


def _embed_fixture_text(text: str, dim: int = 768) -> list[float]:
    factory = getattr(_embedder_mod, "get_embedding_provider", None)
    if callable(factory):
        vec: list[float] = factory(dimensions=dim).embed_text(text, is_query=False)
        return vec
    embedder = _embedder_mod.LocalSemanticEmbedder()
    raw: list[float] = embedder.embed_text(text, is_query=False)
    return raw[:dim]


DEFAULT_TENANT_ID: Final[UUID] = getattr(
    _contracts_mod,
    "DEFAULT_TENANT_ID",
    UUID("00000000-0000-0000-0000-000000000001"),
)

EPHEMERAL_PG_DOMAIN_PREFIX: Final[str] = "eval_"

_TENANT_TABLES_CHILD_TO_PARENT: Final[tuple[str, ...]] = (
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


def cleanup_ephemeral_pg_domain(
    conn: psycopg.Connection[Any],
    domain_id: str,
) -> dict[str, int]:
    """Delete all rows belonging strictly to ``domain_id`` in child-to-parent order."""
    validated_id = validate_ephemeral_pg_domain_id(domain_id)
    deleted: dict[str, int] = {}
    repo_commits_cur = conn.execute(
        """
        DELETE FROM repo_commits
        WHERE repo_id IN (SELECT repo_id FROM repositories WHERE domain_id = %s)
        """,
        (validated_id,),
    )
    deleted["repo_commits"] = max(repo_commits_cur.rowcount or 0, 0)
    for table in _TENANT_TABLES_CHILD_TO_PARENT:
        cur = conn.execute(f"DELETE FROM {table} WHERE domain_id = %s", (validated_id,))
        deleted[table] = max(cur.rowcount or 0, 0)
    conn.commit()
    return deleted


_SEED_ALL_SCOPES: Final[list[str]] = [
    "public",
    "public_eng",
    "public_docs",
    "docs_restricted",
    "public_quant",
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
    (
        "trading_code_scout",
        "Trading Codebase & Architecture Scout",
        "gemma4_26b_code_scout",
        "code",
        90,
    ),
    (
        "quant_research_scout",
        "Quantitative Research & Strategy Scout",
        "gemma4_26b_quant_research",
        "document",
        180,
    ),
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
    src_files = sorted((repo_root / "src" / "context_scout").rglob("*.py"))
    strat_files = sorted(
        (repo_root / "examples" / "quant_dogfood" / "trading_strategies").glob("*.py")
    )
    all_files = src_files + strat_files
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
    from examples.quant_dogfood.mcp_servers.seed_code_corpus import GITHUB_ISSUES_SEED

    existing_docs = {
        str(row[0])
        for row in conn.execute(
            "SELECT doc_id FROM doc_chunks WHERE domain_id = %s AND doc_id LIKE 'gh-issue:%%'",
            (domain_id,),
        ).fetchall()
    }
    for issue in GITHUB_ISSUES_SEED:
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
        "public_quant",
        "corpus:enterprise_docs",
    ]
    extra_specs: list[tuple[str, str, str, str, str, str]] = [
        (
            "doc-vwap-mean-reversion",
            "Intraday VWAP Mean Reversion Entry and Exit Rules",
            "urn:enterprise_docs_scout:doc-vwap-mean-reversion",
            "GLOBAL",
            "Signal Formula & Thresholds",
            (
                "Intraday VWAP mean reversion enters when the standardized price deviation "
                "exceeds two standard deviations: "
                r"$$\alpha_t = \frac{P_t - \text{VWAP}_t}{\sigma_t}$$. "
                "Positions exit on VWAP cross-back or end-of-session inventory flatten with "
                "explicit slippage and transaction-cost guards."
            ),
        ),
        (
            "doc-nse-momentum-factor",
            "NSE Cross-Sectional Momentum Factor Signals and Rebalancing",
            "urn:enterprise_docs_scout:doc-nse-momentum-factor",
            "REGIONAL_APAC",
            "Rebalancing Rules",
            (
                "Cross-sectional momentum on NSE equities ranks 12-1 month residual returns "
                "across liquid constituents, applying monthly decile rebalancing, sector-neutral "
                "caps, and Securities Transaction Tax (STT) turnover dampening."
            ),
        ),
        (
            "doc-xetra-stat-arb",
            "XETRA Statistical Arbitrage Spread Cointegration and Mean Reversion",
            "urn:enterprise_docs_scout:doc-xetra-stat-arb",
            "REGIONAL_EU",
            "Cointegration & Kalman Hedge",
            (
                "XETRA statistical arbitrage pairs track cointegrated equity spreads via "
                "Johansen trace tests and adaptive Kalman hedge ratios, triggering mean-reversion "
                "entries when the z-score breaches +/-2.0 with strict tick-size rounding."
            ),
        ),
        (
            "doc-continuous-auction-ofi",
            "Continuous Auction Order Flow Imbalance (OFI) Microstructure Research",
            "urn:enterprise_docs_scout:doc-continuous-auction-ofi",
            "GLOBAL",
            "Order Flow Imbalance Dynamics",
            (
                "Continuous limit-order-book Order Flow Imbalance (OFI) measures net bid-ask "
                "queue changes at the top price level to forecast short-horizon mid-price "
                "impact across high-frequency continuous auction venues."
            ),
        ),
        (
            "doc-almgren-chriss-exec",
            "Almgren-Chriss Optimal Execution Schedule Slicing",
            "urn:enterprise_docs_scout:doc-almgren-chriss-exec",
            "GLOBAL",
            "Trajectory Optimization",
            (
                "The Almgren-Chriss execution framework minimizes expected implementation "
                "shortfall plus variance-penalized market impact by slicing parent orders "
                "into deterministic child schedules parameterized by risk aversion lambda."
            ),
        ),
        (
            "doc-finrl-deepseek-rl",
            "FinRL-DeepSeek Actor-Critic Reinforcement Learning Allocation",
            "urn:enterprise_docs_scout:doc-finrl-deepseek-rl",
            "GLOBAL",
            "Actor-Critic Policy",
            (
                "FinRL-DeepSeek combines distilled reasoning signals with PPO/A2C actor-critic "
                "reinforcement learning agents to dynamically allocate weights under drawdown "
                "constraints, turnover penalties, and out-of-sample walk-forward validation."
            ),
        ),
        (
            "doc-nifty-hrp-allocation",
            "NIFTY Hierarchical Risk Parity (HRP) Portfolio Construction",
            "urn:enterprise_docs_scout:doc-nifty-hrp-allocation",
            "REGIONAL_APAC",
            "Hierarchical Clustering",
            (
                "NIFTY Hierarchical Risk Parity applies single-linkage tree clustering on the "
                "correlation distance matrix followed by recursive bisection inverse-variance "
                "weighting, avoiding ill-conditioned covariance matrix inversion."
            ),
        ),
        (
            "doc-regional-compliance-gov",
            "Regional APAC, EU, and US Compliance Governance Standard",
            "urn:enterprise_docs_scout:doc-regional-compliance-gov",
            "ENTERPRISE_DOC",
            "Cross-Region Controls",
            (
                "Regional governance enforces APAC SEBI/NSE audit trails, EU MiFID II "
                "algorithmic kill-switch controls and GDPR residency, and US SEC Rule 15c3-5 "
                "pre-trade risk checks across all deployed execution pipelines."
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


def ensure_pg17_schema_and_seed_corpora(
    db_admin_url: str,
    db_app_url: str,
    repo_root: Path,
) -> None:
    """Idempotently bootstrap PG17 schema via Alembic, domain_packs, and benchmark corpora."""
    try:
        from context_scout.db.migrations_runner import run_alembic_upgrade

        run_alembic_upgrade(db_admin_url)
    except ModuleNotFoundError:
        with psycopg.connect(db_admin_url) as check_conn:
            has_alembic = (
                check_conn.execute("SELECT to_regclass('public.alembic_version')").fetchone()
                or (None,)
            )[0]
            if has_alembic is None:
                raise

    with psycopg.connect(db_admin_url) as admin_conn:
        stale_eval_rows = admin_conn.execute(
            "SELECT DISTINCT domain_id FROM domain_packs WHERE domain_id LIKE 'eval_%'"
        ).fetchall()
        for (stale_eval_id,) in stale_eval_rows:
            cleanup_ephemeral_pg_domain(admin_conn, str(stale_eval_id))
        for d_id, display_name, adapter_id, truth_src, half_life in _REQUIRED_DOMAIN_PACKS:
            admin_conn.execute(
                """
                INSERT INTO domain_packs (
                    tenant_id, domain_id, display_name, lora_adapter_id, primary_truth_source,
                    freshness_half_life_days, config_json
                )
                VALUES (%s, %s, %s, %s, %s, %s, '{}'::jsonb)
                ON CONFLICT (tenant_id, domain_id) DO NOTHING
                """,
                (DEFAULT_TENANT_ID, d_id, display_name, adapter_id, truth_src, half_life),
            )
        admin_conn.commit()

    prev_hash_flag = os.environ.get("SCOUT_USE_HASH_EMBEDDINGS_FOR_TESTS")
    os.environ["SCOUT_USE_HASH_EMBEDDINGS_FOR_TESTS"] = "1"
    try:
        mgr = RLSConnectionManager(db_app_url)
        repo = PostgresRLSRepository()
        seed_principal = PrincipalContext(
            user_id="pg_isolation_bootstrap",
            role=RoleType.ENGINEER,
            allowed_scopes=_SEED_ALL_SCOPES,
        )

        def _row_count(row: tuple[object, ...] | None) -> int:
            return int(str(row[0])) if row and row[0] is not None else 0

        # Pre-warm the embedding model outside any open DB transaction so cold
        # HuggingFace downloads in CI never trigger PG17 idle_in_transaction_session_timeout.
        _embed_fixture_text("bootstrap warmup")

        with mgr.scoped_transaction(seed_principal, enable_hnsw_iterative_scan=False) as conn:
            code_arch_row = conn.execute(
                "SELECT count(*) FROM code_files WHERE domain_id = 'code_architecture_scout'"
            ).fetchone()
            if _row_count(code_arch_row) < 43:
                _seed_code_architecture_scout(conn, repo, repo_root)

            ent_docs_row = conn.execute(
                "SELECT count(*) FROM doc_chunks WHERE domain_id = 'enterprise_docs_scout'"
            ).fetchone()
            if _row_count(ent_docs_row) < 12:
                _seed_enterprise_docs_scout(conn, repo)
            conn.execute(
                """
                UPDATE doc_chunks
                SET allowed_scopes = array_remove(allowed_scopes, 'docs_restricted')
                WHERE domain_id = 'enterprise_docs_scout'
                  AND allowed_scopes @> ARRAY['docs_restricted', 'public']::text[]
                """
            )

            de_row = conn.execute(
                "SELECT count(*) FROM doc_chunks WHERE domain_id = 'data_engineering_scout'"
            ).fetchone()
            if _row_count(de_row) == 0:
                _seed_data_engineering_scout(conn, repo)

            poly_row = conn.execute(
                "SELECT count(*) FROM code_files WHERE domain_id = 'polyglot_unseen_v2'"
            ).fetchone()
            if _row_count(poly_row) == 0:
                _seed_polyglot_unseen_v2(conn, repo)

            trading_row = conn.execute(
                "SELECT count(*) FROM code_files WHERE domain_id = 'trading_code_scout'"
            ).fetchone()
            quant_row = conn.execute(
                "SELECT count(*) FROM doc_chunks WHERE domain_id = 'quant_research_scout'"
            ).fetchone()
            need_trading_seed = _row_count(trading_row) == 0 or _row_count(quant_row) == 0

        if need_trading_seed:
            from examples.quant_dogfood.mcp_servers.seed_code_corpus import (
                seed_trading_codebase_and_issues,
            )

            seed_trading_codebase_and_issues(
                repo_dir=repo_root,
                settings=ScoutSettings(db_admin_url=db_admin_url, db_app_url=db_app_url),
                principal=seed_principal,
            )

        with mgr.scoped_transaction(seed_principal, enable_hnsw_iterative_scan=False) as conn:
            quant_chunk_id = uuid5(
                NAMESPACE_URL,
                "quant_research_scout:quant-ofi-vwap-spec",
            )
            ofi_row = conn.execute(
                """
                SELECT count(*) FROM doc_chunks
                WHERE chunk_id = %s
                   OR (
                       domain_id = %s
                       AND verbatim_content LIKE '%%OFI_t%%'
                       AND verbatim_content LIKE '%%VWAP_t%%'
                   )
                """,
                (quant_chunk_id, "quant_research_scout"),
            ).fetchone()
            if _row_count(ofi_row) == 0:
                formula_content = (
                    "Order Flow Imbalance and Intraday VWAP Signal Specification.\n\n"
                    r"$$\alpha_t = \frac{P_t - \text{VWAP}_t}{\sigma_t}$$"
                    "\n\n"
                    r"\begin{equation}\text{OFI}_t = q_t^b - q_{t-1}^b\end{equation}"
                    "\n\n"
                    "| Metric | Threshold | Regime |\n"
                    "| :--- | :--- | :--- |\n"
                    "| alpha_t | 2.0 | GLOBAL |\n"
                    "| OFI_t | 0.65 | CONTINUOUS |\n"
                )
                repo.insert_doc_chunk(
                    conn,
                    DocChunkRecord(
                        chunk_id=quant_chunk_id,
                        domain_id="quant_research_scout",
                        doc_id="quant-ofi-vwap-spec",
                        doc_title="VWAP Mean Reversion & Order Flow Imbalance (OFI) Formula Spec",
                        source_uri="urn:quant_research_scout:quant-ofi-vwap-spec",
                        market_scope="GLOBAL",
                        published_date=date(2026, 2, 1),
                        authority_weight=1.0,
                        section_heading="Formulas & Thresholds",
                        contextual_header="Quant Research Spec | VWAP & OFI",
                        verbatim_content=formula_content,
                        metadata_json={"seeded_by": "bootstrap_pg17_session"},
                        allowed_scopes=["public", "public_quant"],
                        embedding=_embed_fixture_text(formula_content),
                    ),
                )
    finally:
        if prev_hash_flag is None:
            os.environ.pop("SCOUT_USE_HASH_EMBEDDINGS_FOR_TESTS", None)
        else:
            os.environ["SCOUT_USE_HASH_EMBEDDINGS_FOR_TESTS"] = prev_hash_flag


def _is_research_domain_dir(dom_dir: Path) -> bool:
    for meta_name in ("domain.yaml", "specialization.yaml"):
        meta_path = dom_dir / meta_name
        if meta_path.is_file():
            data = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
            if isinstance(data, dict):
                truth = str(data.get("primary_truth_source", "")).lower()
                if truth in {"document", "research", "docs"}:
                    return True
    return dom_dir.name in {"enterprise_docs_scout", "data_engineering_scout"}


def _build_ephemeral_code_split(
    domain_id: str,
    split_name: str,
    count: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx in range(1, count + 1):
        pkg = f"ephemeral_{split_name}_{idx:02d}"
        fpath = f"ephemeral_pkg/{pkg}/module_{idx:02d}.py"
        qsym = f"ephemeral_pkg.{pkg}.module_{idx:02d}.Handler{idx:02d}"
        s_line, e_line = 10 + idx, 55 + idx
        reason = (
            f"Target symbol {qsym} localized from AST candidate. "
            "Ensures strict AST boundary localization, caller/callee edge resolution, "
            "and fail-closed Stage 4 preflight verification."
        )
        ptrs = [
            {
                "repo_name": "enterprise-context-scout",
                "file_path": fpath,
                "start_line": s_line,
                "end_line": e_line,
                "qualified_symbol": qsym,
                "selection_reason": reason,
                "modification_intent": "modify_target",
            }
        ]
        prompt = (
            f"Localize {qsym} in {domain_id}.\n"
            f"Retrieved AST Candidate: [repo=enterprise-context-scout | file={fpath} | "
            f"symbol={qsym} | lines={s_line}-{e_line}]"
        )
        rows.append(
            {
                "source_artifact_ids": [f"{domain_id}:{split_name}:{pkg}"],
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": json.dumps(ptrs)},
                ],
            }
        )
    return rows


def _build_ephemeral_doc_split(
    domain_id: str,
    split_name: str,
    count: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx in range(1, count + 1):
        doc_id = f"doc:test_{split_name}_{idx:02d}"
        title = f"Enterprise {split_name.title()} Spec {idx:02d}"
        uri = f"urn:{domain_id}:{split_name}-{idx:02d}"
        section = f"Policy {idx:02d}"
        quote = (
            f"Policy {split_name} {idx:02d} requires PostgreSQL 17 FORCE ROW LEVEL SECURITY, "
            "verbatim chunk_id provenance, and deterministic zero-spend abstention."
        )
        summary = (
            f"Verified brief for {title} enforcing tenant isolation, strict citation "
            "verification, and bounded token budgets across private VPC deployments."
        )
        brief = {
            "brief_title": title,
            "domain_id": domain_id,
            "executive_summary": summary,
            "formal_expression_or_rule": "RRF(d) = sum(1 / (k + rank_i(d)))",
            "core_findings_and_rules": [quote],
            "domain_constraints_and_caveats": [
                "Enforce tenant_id RLS and server-side RBAC scopes."
            ],
            "known_risks_and_failure_modes": [
                "Abstain with paid_tokens_spent=0 when citations are missing."
            ],
            "structured_metadata": {"doc_id": doc_id, "scope": "GLOBAL_MULTI"},
            "verified_citations": [
                {
                    "chunk_id": f"chunk-{doc_id}",
                    "doc_id": doc_id,
                    "doc_title": title,
                    "source_uri": uri,
                    "section_heading": section,
                    "verbatim_quote": quote,
                }
            ],
        }
        prompt = (
            f"Synthesize a DomainKnowledgeBrief for {title} in {domain_id}.\n"
            f"Retrieved Corpus Chunk: [chunk_id=chunk-{doc_id} | doc_id={doc_id} | "
            f"title={title} | uri={uri} | section={section} | quote={quote}]"
        )
        rows.append(
            {
                "source_artifact_ids": [doc_id],
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": json.dumps(brief)},
                ],
            }
        )
    return rows


def _build_ephemeral_lora_safetensors() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    tensors: dict[str, np.ndarray] = {}
    for layer in (26, 27, 28, 29):
        attn_projs = (
            ("q_proj", "k_proj", "o_proj")
            if layer == 29
            else (
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            )
        )
        for proj in attn_projs:
            prefix = f"language_model.model.layers.{layer}.self_attn.{proj}"
            tensors[f"{prefix}.lora_a"] = rng.normal(0.0, 0.01, size=(2816, 16)).astype(np.float32)
            tensors[f"{prefix}.lora_b"] = np.zeros((16, 2816), dtype=np.float32)
        for proj in ("gate_proj", "up_proj", "down_proj"):
            prefix = f"language_model.model.layers.{layer}.mlp.{proj}"
            tensors[f"{prefix}.lora_a"] = rng.normal(0.0, 0.01, size=(2816, 16)).astype(np.float32)
            tensors[f"{prefix}.lora_b"] = np.zeros((16, 2816), dtype=np.float32)
    return tensors


def ensure_ephemeral_domain_artifacts(repo_root: Path) -> list[Path]:
    """Create missing gitignored tuning splits and LoRA safetensors for clean CI checkouts.

    Never overwrites or deletes pre-existing files on disk. Returns only newly created file paths
    so callers can clean them up in a ``finally:`` block.
    """
    domains_dir = repo_root / "domains"
    if not domains_dir.is_dir():
        return []

    created: list[Path] = []
    cached_tensors: dict[str, np.ndarray] | None = None

    for dom_dir in sorted(p for p in domains_dir.iterdir() if p.is_dir()):
        is_research = _is_research_domain_dir(dom_dir)
        tuning_dir = dom_dir / "tuning"
        split_counts = (("train", 16), ("valid", 6), ("eval", 4))
        for split_name, count in split_counts:
            split_path = tuning_dir / f"{split_name}.jsonl"
            if not split_path.exists() or split_path.stat().st_size == 0:
                was_missing = not split_path.exists()
                tuning_dir.mkdir(parents=True, exist_ok=True)
                rows = (
                    _build_ephemeral_doc_split(dom_dir.name, split_name, count)
                    if is_research
                    else _build_ephemeral_code_split(dom_dir.name, split_name, count)
                )
                split_path.write_text(
                    "\n".join(json.dumps(r) for r in rows) + "\n",
                    encoding="utf-8",
                )
                if was_missing:
                    created.append(split_path)

        adapters_dir = dom_dir / "adapters"
        if adapters_dir.is_dir():
            for cfg_path in sorted(adapters_dir.glob("*/adapter_config.json")):
                safetensors_path = cfg_path.parent / "adapters.safetensors"
                if not safetensors_path.exists():
                    if cached_tensors is None:
                        cached_tensors = _build_ephemeral_lora_safetensors()
                    save_safetensors_numpy(cached_tensors, str(safetensors_path))
                    created.append(safetensors_path)

    return created
