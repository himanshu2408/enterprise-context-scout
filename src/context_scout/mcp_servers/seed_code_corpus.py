"""Seeds realistic trading strategies AST and GitHub issues into PostgreSQL 17 (R3)."""

from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Final
from uuid import UUID, uuid4, uuid5

from context_scout.ast_engine.extractor import ASTExtractor
from context_scout.ast_engine.models import ParsedEdge
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
        # 0. Ensure domain_packs rows exist on fresh databases before inserting code_files
        conn.execute(
            """
            INSERT INTO domain_packs (
                domain_id, display_name, lora_adapter_id, primary_truth_source,
                freshness_half_life_days, config_json
            )
            VALUES
                (
                    'trading_code_scout',
                    'Trading Codebase & Architecture Scout',
                    'gemma4_26b_code_scout',
                    'code',
                    90,
                    '{}'::jsonb
                ),
                (
                    'quant_research_scout',
                    'Quantitative Research & Strategy Scout',
                    'gemma4_26b_quant_research',
                    'document',
                    180,
                    '{}'::jsonb
                )
            ON CONFLICT (domain_id) DO NOTHING;
            """
        )

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

        # 5. Keep all src/context_scout/*.py files and AST symbols/edges synced with disk
        existing_file_rows = conn.execute(
            """
            SELECT file_id, file_path, git_blob_sha
            FROM code_files
            WHERE domain_id = %s AND repo_name = %s;
            """,
            (domain_id, repo_name),
        ).fetchall()
        existing_files: dict[str, tuple[UUID, str]] = {
            str(r[1]): (UUID(str(r[0])), str(r[2])) for r in existing_file_rows
        }

        existing_sym_rows = conn.execute(
            """
            SELECT symbol_id, file_id, qualified_name, start_line, end_line
            FROM ast_symbols
            WHERE domain_id = %s;
            """,
            (domain_id,),
        ).fetchall()
        existing_sym_tuples_by_file: dict[UUID, set[tuple[str, int, int]]] = defaultdict(set)
        existing_syms_by_file: dict[UUID, list[tuple[UUID, str, str]]] = defaultdict(list)
        for r in existing_sym_rows:
            sym_id = UUID(str(r[0]))
            fid = UUID(str(r[1]))
            qname = str(r[2])
            sline = int(str(r[3]))
            eline = int(str(r[4]))
            short = qname.rsplit(".", 1)[-1]
            existing_sym_tuples_by_file[fid].add((qname, sline, eline))
            existing_syms_by_file[fid].append((sym_id, qname, short))
            if qname not in global_symbol_map:
                global_symbol_map[qname] = sym_id
            if sym_id not in short_name_map[short]:
                short_name_map[short].append(sym_id)

        extra_edges_pending: list[tuple[list[ParsedEdge], dict[str, UUID]]] = []
        extra_py_candidates = list(sorted((root / "src" / "context_scout").rglob("*.py")))
        has_non_src_files = any(not fp.startswith("src/") for fp in existing_files)
        if not has_non_src_files and (root / "tests").is_dir():
            extra_py_candidates.extend(sorted((root / "tests").rglob("*.py")))
        for extra_py in extra_py_candidates:
            rel_p = extra_py.relative_to(root).as_posix()
            if rel_p.startswith("src/context_scout/trading_strategies/"):
                continue
            raw_c = extra_py.read_text(encoding="utf-8")
            new_blob = _git_blob_sha(raw_c)
            symbols, edges = extractor.parse_source(raw_c, "python", file_path=rel_p)
            parsed_sym_tuples = {(ps.qualified_name, ps.start_line, ps.end_line) for ps in symbols}
            existing_entry = existing_files.get(rel_p)
            existing_fid = existing_entry[0] if existing_entry else None
            existing_blob = existing_entry[1] if existing_entry else None
            if (
                existing_fid is not None
                and existing_blob == new_blob
                and existing_sym_tuples_by_file.get(existing_fid, set()) == parsed_sym_tuples
            ):
                continue

            row_fid = conn.execute(
                """
                INSERT INTO code_files (
                    file_id, domain_id, repo_name, file_path,
                    git_commit_sha, git_blob_sha, language,
                    allowed_scopes, raw_content, line_count
                )
                VALUES (%s, %s, %s, %s, %s, %s, 'python', %s, %s, %s)
                ON CONFLICT (domain_id, repo_name, file_path) DO UPDATE SET
                    git_commit_sha = EXCLUDED.git_commit_sha,
                    git_blob_sha = EXCLUDED.git_blob_sha,
                    raw_content = EXCLUDED.raw_content,
                    line_count = EXCLUDED.line_count,
                    indexed_at = NOW()
                RETURNING file_id;
                """,
                (
                    uuid4(),
                    domain_id,
                    repo_name,
                    rel_p,
                    rev_cmd,
                    new_blob,
                    ["public_eng"],
                    raw_c,
                    len(raw_c.splitlines()),
                ),
            ).fetchone()
            if row_fid is None:
                continue
            actual_file_id = UUID(str(row_fid[0]))

            for old_sym_id, old_qname, old_short in existing_syms_by_file.get(actual_file_id, []):
                if global_symbol_map.get(old_qname) == old_sym_id:
                    global_symbol_map.pop(old_qname, None)
                if old_sym_id in short_name_map.get(old_short, []):
                    short_name_map[old_short] = [
                        sid for sid in short_name_map[old_short] if sid != old_sym_id
                    ]

            conn.execute(
                """
                DELETE FROM ast_edges
                WHERE domain_id = %s
                  AND (
                      source_symbol_id IN (SELECT symbol_id FROM ast_symbols WHERE file_id = %s)
                      OR target_symbol_id IN (SELECT symbol_id FROM ast_symbols WHERE file_id = %s)
                  );
                """,
                (domain_id, actual_file_id, actual_file_id),
            )
            conn.execute(
                "DELETE FROM ast_symbols WHERE domain_id = %s AND file_id = %s;",
                (domain_id, actual_file_id),
            )

            file_sym_map = {}
            for parsed_sym in symbols:
                sym_id = uuid4()
                emb = generate_deterministic_embedding(
                    parsed_sym.signature_skeleton + " " + parsed_sym.docstring
                )
                sym_rec = ASTSymbolRecord(
                    symbol_id=sym_id,
                    file_id=actual_file_id,
                    domain_id=domain_id,
                    qualified_name=parsed_sym.qualified_name,
                    symbol_kind=parsed_sym.symbol_kind,
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

            extra_edges_pending.append((edges, file_sym_map))

        for edges, file_sym_map in extra_edges_pending:
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

    # 6. Ensure public_quant arXiv baseline chunks exist on fresh CI databases (no-op when seeded)
    quant_seed_principal = PrincipalContext(
        user_id="seed_quant_corpus_indexer",
        role=RoleType.QUANT_RESEARCHER,
        allowed_scopes=["public_quant"],
    )
    quant_seed_docs: list[tuple[str, str, str, str, date, str, str]] = [
        (
            "arxiv:1310.7514",
            "The Price Impact of Order Book Events",
            "https://arxiv.org/abs/1310.7514",
            "US_EQUITY",
            date(2026, 10, 15),
            "1. Introduction",
            (
                "Order flow imbalance explains short-horizon price changes with linear market "
                "impact across limit order book levels in US equities."
            ),
        ),
        (
            "arxiv:2606.31387",
            (
                "Signature-Based Optimal Execution for Statistical Arbitrage "
                "with Path-Dependent Trading Signals"
            ),
            "https://arxiv.org/pdf/2606.31387v2",
            "US_EQUITY",
            date(2026, 6, 30),
            "Introduction",
            (
                "Statistical-arbitrage strategies depend not only on predictive signals, "
                "but also on optimal execution under temporary market impact, inventory "
                "risk, and CME E-mini futures path-dependent trading dynamics clearing a "
                "5 bps US equity friction floor."
            ),
        ),
        (
            "arxiv:2502.07393",
            "FinRL-DeepSeek: LLM-Infused Risk-Sensitive Reinforcement Learning for Trading Agents",
            "https://arxiv.org/pdf/2502.07393v1",
            "US_EQUITY",
            date(2025, 2, 11),
            "Preamble / Abstract",
            (
                "This paper presents a risk-sensitive deep reinforcement learning actor-critic "
                "trading agent extending Conditional Value-at-Risk Proximal Policy Optimization "
                "(CPPO) and FinRL benchmarks on US equity Nasdaq-100 markets above 5 bps costs."
            ),
        ),
        (
            "arxiv:2302.13245",
            "Physical Momentum in the Indian Stock Market",
            "https://arxiv.org/abs/2302.13245",
            "IN_NSE",
            date(2024, 1, 1),
            "Preamble / Abstract",
            (
                "Our study determines abnormal returns for physical and cross-sectional momentum "
                "strategy portfolios in the Indian stock market across National Stock Exchange "
                "(NSE) NIFTY 500 constituents after accounting for >= 18 bps STT and brokerage "
                "round-trip liquidity friction."
            ),
        ),
        (
            "arxiv:2211.07080",
            "Designing Efficient Pair-Trading and Intraday Momentum Strategies on NSE Nifty 50",
            "https://arxiv.org/abs/2211.07080",
            "IN_NSE",
            date(2024, 1, 1),
            "Abstract",
            (
                "High-frequency intraday momentum, order flow, and cointegration pair-trading "
                "strategies across five sectors of the National Stock Exchange of India (NSE) "
                "Nifty 50 constituents require >= 18 bps round-trip execution friction assumptions."
            ),
        ),
        (
            "arxiv:2202.02728",
            "Hierarchical Risk Parity and Minimum Variance Portfolio Design on NIFTY 50 Stocks",
            "https://arxiv.org/abs/2202.02728",
            "IN_NSE",
            date(2024, 1, 1),
            "Abstract",
            (
                "This paper compares Hierarchical Risk Parity (HRP) and critical-line minimum "
                "variance portfolio optimization across eight Indian NIFTY sectors on the National "
                "Stock Exchange (NSE), evaluating out-of-sample Sharpe ratios under >= 16 bps "
                "Indian equity rebalance friction."
            ),
        ),
        (
            "arxiv:2607.18001",
            (
                "High-Dimensional Factor Statistical Arbitrage and "
                "Market-Neutral Portfolios on XETRA DAX"
            ),
            "https://arxiv.org/pdf/2607.18001v1",
            "DE_XETRA",
            date(2026, 7, 20),
            "Empirical Evaluation",
            (
                "Dynamic high-dimensional PCA factor statistical arbitrage and order book "
                "liquidity herding models on Deutsche Boerse XETRA DAX European equities "
                "clear a 12 bps continuous-auction round-trip transaction cost floor."
            ),
        ),
    ]
    with mgr.scoped_transaction(quant_seed_principal, enable_hnsw_iterative_scan=False) as q_conn:
        row_qc = q_conn.execute(
            "SELECT count(*) FROM doc_chunks WHERE domain_id = 'quant_research_scout';"
        ).fetchone()
        if row_qc is not None and int(str(row_qc[0])) == 0:
            for q_doc_id, q_title, q_uri, q_scope, q_date, q_sec, q_body in quant_seed_docs:
                q_emb = generate_deterministic_embedding(f"{q_title} {q_sec} {q_body}")
                q_vec = "[" + ",".join(f"{v:.6f}" for v in q_emb) + "]"
                q_conn.execute(
                    """
                    INSERT INTO doc_chunks (
                        chunk_id, domain_id, doc_id, doc_title, source_uri,
                        market_scope, published_date, authority_weight,
                        section_heading, contextual_header, verbatim_content,
                        metadata_json, allowed_scopes, embedding
                    )
                    VALUES (
                        %s, 'quant_research_scout', %s,
                        %s,
                        %s, %s, %s, 1.0,
                        %s, 'Quant Seed [quant_research_scout]',
                        %s,
                        '{}'::jsonb, %s, %s::vector
                    );
                    """,
                    (
                        uuid4(),
                        q_doc_id,
                        q_title,
                        q_uri,
                        q_scope,
                        q_date,
                        q_sec,
                        q_body,
                        ["public_quant"],
                        q_vec,
                    ),
                )

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
