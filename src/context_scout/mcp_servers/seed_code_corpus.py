"""Seeds realistic trading strategies AST and GitHub issues into PostgreSQL 17 (R3)."""

from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Final, Literal
from uuid import UUID, uuid4, uuid5

from context_scout.ast_engine.extractor import ASTExtractor
from context_scout.config import ScoutSettings
from context_scout.db.repositories import (
    ASTEdgeRecord,
    ASTSymbolRecord,
    CodeFileRecord,
    PostgresRLSRepository,
)
from context_scout.db.session import RLSConnectionManager
from context_scout.ingestion.git_merkle_indexer import generate_deterministic_embedding
from context_scout.schemas.contracts import PrincipalContext, RoleType

logger = logging.getLogger(__name__)

SEED_PRINCIPAL: Final[PrincipalContext] = PrincipalContext(
    user_id="seed_code_corpus_indexer",
    role=RoleType.ENGINEER,
    allowed_scopes=["public_eng"],
)

NAMESPACE_GH: Final[UUID] = UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

GITHUB_ISSUES_SEED: Final[list[dict[str, object]]] = [
    {
        "issue_number": 101,
        "title": "Fix numerical instability in Kalman filter covariance matrix inversion",
        "state": "open",
        "section": "Issue Description and Root Cause Analysis",
        "content": (
            "Fix numerical instability in Kalman filter covariance matrix inversion. "
            "Investigate ill-conditioned covariance matrix P_t during high volatility regimes "
            "in PairsSpreadSignal. Implementing the symmetric Joseph form "
            "(I - K*H)*P*(I - K*H)^T + K*R*K^T prevents non-positive-definite covariance "
            "collapse and preserves Kalman filter convergence across cointegrated pairs."
        ),
        "labels": ["bug", "trading_strategies", "kalman", "numerical_stability"],
    },
    {
        "issue_number": 102,
        "title": "Support multi-level Order Flow Imbalance (OFI) across L2/L3 order book feeds",
        "state": "open",
        "section": "Architectural Proposal",
        "content": (
            "Extend OrderFlowImbalanceSignal beyond top-of-book (Level 1) to incorporate depth "
            "levels 2 through 5. Cont, Kukanov, and Stoikov (2014) demonstrate that "
            "depth-weighted OFI with exponential decay improves short-term price impact "
            "predictive power by 34%."
        ),
        "labels": ["feature", "order_flow_imbalance", "microstructure"],
    },
    {
        "issue_number": 103,
        "title": (
            "Almgren-Chriss risk aversion parameter calibration under high intraday volatility"
        ),
        "state": "closed",
        "section": "Resolution and Performance Audit",
        "content": (
            "Resolved dynamic recalibration of kappa_tilde in AlmgrenChrissExecutionOptimizer "
            "using live GARCH(1,1) volatility forecasts. Execution slippage reduced from "
            "4.2 bps to 1.8 bps during market open auction surges across large inventory unwinds."
        ),
        "labels": ["optimization", "execution", "almgren_chriss", "garch"],
    },
    {
        "issue_number": 104,
        "title": "Ledoit-Wolf covariance shrinkage eigenvalue clipping in Kelly position sizer",
        "state": "closed",
        "section": "Patch Notes and Benchmark Validation",
        "content": (
            "Added eigenvalue regularization floor (MIN_EIGENVALUE_FLOOR = 1e-6) in "
            "KellyRiskManager.ledoit_wolf_shrinkage to protect against singularity when asset "
            "count exceeds time observations. Eliminates negative weights on highly "
            "correlated instruments."
        ),
        "labels": ["risk_management", "kelly_criterion", "shrinkage"],
    },
    {
        "issue_number": 105,
        "title": "Stationarity violation handling in rolling GARCH(1,1) parameter estimation",
        "state": "open",
        "section": "Bug Report and Diagnostic Logs",
        "content": (
            "When calibrating GARCH(1,1) on non-stationary intraday return series, alpha + beta "
            "occasionally exceeded 1.0, leading to infinite unconditional variance. Added strict "
            "optimization boundary constraints (alpha + beta <= 0.9999) in fit_quasi_mle."
        ),
        "labels": ["bug", "volatility", "garch", "stationarity"],
    },
]


def _git_blob_sha(content: str) -> str:
    encoded = content.encode("utf-8")
    header = f"blob {len(encoded)}\0".encode()
    return hashlib.sha1(header + encoded).hexdigest()


def seed_trading_codebase_and_issues(
    repo_dir: Path | None = None,
    settings: ScoutSettings | None = None,
    principal: PrincipalContext = SEED_PRINCIPAL,
) -> dict[str, int]:
    """Persists strategy files and GitHub issues into live PostgreSQL 17 under Forced RLS."""
    st = settings or ScoutSettings()
    root = repo_dir or Path.cwd()
    mgr = RLSConnectionManager(st.db_app_url)
    repo = PostgresRLSRepository()
    extractor = ASTExtractor()
    domain_id = "trading_code_scout"
    repo_name = "enterprise-context-scout"

    strategies_dir = root / "src" / "context_scout" / "trading_strategies"
    strategy_files = sorted(strategies_dir.glob("*.py"))
    if not strategy_files:
        raise FileNotFoundError(f"No strategy files found in {strategies_dir}")

    total_files = 0
    total_symbols = 0
    total_edges = 0
    total_issues = 0

    with mgr.scoped_transaction(principal, enable_hnsw_iterative_scan=False) as conn:
        # 1. Clean existing records for this domain in trading_strategies
        conn.execute(
            """
            DELETE FROM code_files
            WHERE domain_id = %s
              AND file_path LIKE 'src/context_scout/trading_strategies/%%';
            """,
            (domain_id,),
        )
        conn.execute(
            """
            DELETE FROM doc_chunks
            WHERE domain_id = %s
              AND doc_id LIKE 'gh-issue:%%';
            """,
            (domain_id,),
        )

        # 2. Parse and index strategy files
        global_symbol_map: dict[str, UUID] = {}
        short_name_map: dict[str, list[UUID]] = defaultdict(list)
        parsed_files = []

        # Get git commit SHA
        rev_cmd = hashlib.sha1(b"head-commit-seed").hexdigest()[:40]

        for s_file in strategy_files:
            rel_path = s_file.relative_to(root).as_posix()
            raw_content = s_file.read_text(encoding="utf-8")
            lines = raw_content.splitlines()
            file_id = uuid4()
            blob_sha = _git_blob_sha(raw_content)

            file_rec = CodeFileRecord(
                file_id=file_id,
                domain_id=domain_id,
                repo_name=repo_name,
                file_path=rel_path,
                git_commit_sha=rev_cmd,
                git_blob_sha=blob_sha,
                language="python",
                allowed_scopes=["public_eng"],
                raw_content=raw_content,
                line_count=len(lines),
            )
            repo.insert_code_file(conn, file_rec)
            total_files += 1

            symbols, edges = extractor.parse_source(raw_content, "python", file_path=rel_path)
            file_sym_map: dict[str, UUID] = {}

            for parsed_sym in symbols:
                sym_id = uuid4()
                emb = generate_deterministic_embedding(
                    parsed_sym.signature_skeleton + " " + parsed_sym.docstring
                )
                kind: Literal["function", "class", "method", "interface", "type_alias"] = (
                    parsed_sym.symbol_kind
                )
                sym_rec = ASTSymbolRecord(
                    symbol_id=sym_id,
                    file_id=file_id,
                    domain_id=domain_id,
                    qualified_name=parsed_sym.qualified_name,
                    symbol_kind=kind,
                    start_line=parsed_sym.start_line,
                    end_line=parsed_sym.end_line,
                    signature_skeleton=parsed_sym.signature_skeleton,
                    contextual_summary=parsed_sym.contextual_summary,
                    allowed_scopes=["public_eng"],
                    embedding=emb,
                )
                repo.insert_ast_symbol(conn, sym_rec)
                file_sym_map[parsed_sym.qualified_name] = sym_id
                global_symbol_map[parsed_sym.qualified_name] = sym_id
                short_name_map[parsed_sym.name].append(sym_id)
                total_symbols += 1

            parsed_files.append((file_rec, edges, file_sym_map))

        # 3. Directed Edge Resolution
        for _f_rec, edges, file_sym_map in parsed_files:
            for edge in edges:
                src_id = file_sym_map.get(edge.source_qualified_name)
                if not src_id:
                    src_id = global_symbol_map.get(edge.source_qualified_name)
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
                    allowed_scopes=["public_eng"],
                )
                repo.insert_ast_edge(conn, edge_rec)
                total_edges += 1

        # 4. Seed GitHub Issues into doc_chunks
        for issue in GITHUB_ISSUES_SEED:
            num = int(str(issue["issue_number"]))
            doc_id = f"gh-issue:{num}"
            chunk_id = uuid5(NAMESPACE_GH, doc_id)
            title = str(issue["title"])
            content = str(issue["content"])
            section = str(issue["section"])
            contextual_header = f"GitHub Issue #{num} [trading_code_scout]"
            emb = generate_deterministic_embedding(title + " " + section + " " + content)
            vec_literal = "[" + ",".join(f"{v:.6f}" for v in emb) + "]"
            meta = {
                "issue_number": num,
                "state": issue["state"],
                "labels": issue["labels"],
                "author": "quant_systems",
            }

            conn.execute(
                """
                INSERT INTO doc_chunks (
                    chunk_id, domain_id, doc_id, doc_title, source_uri,
                    market_scope, published_date, authority_weight,
                    section_heading, contextual_header, verbatim_content,
                    metadata_json, allowed_scopes, embedding
                )
                VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s,
                    %s::jsonb, %s, %s::vector
                );
                """,
                (
                    chunk_id,
                    domain_id,
                    doc_id,
                    title,
                    f"https://github.com/enterprise-context-scout/issues/{num}",
                    "GLOBAL",
                    date(2026, 10, 15),
                    1.0,
                    section,
                    contextual_header,
                    content,
                    json.dumps(meta),
                    ["public_eng"],
                    vec_literal,
                ),
            )
            total_issues += 1

    return {
        "files_indexed": total_files,
        "symbols_stored": total_symbols,
        "edges_stored": total_edges,
        "issues_seeded": total_issues,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    stats = seed_trading_codebase_and_issues()
    print("Seed complete:", stats)
