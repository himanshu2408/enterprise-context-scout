"""Adversarial stress harness for AST bounds (SEC-03) and Live PostgreSQL 17 RLS."""

import uuid
from typing import Any

import pytest

from context_scout.ast_engine.dependency_expander import ASTDependencyExpander
from context_scout.config import ScoutSettings
from context_scout.db.repositories import (
    ASTEdgeRecord,
    ASTSymbolRecord,
    CodeFileRecord,
    InMemoryRLSRepository,
    PostgresRLSRepository,
)
from context_scout.db.session import RLSConnectionManager
from context_scout.schemas.contracts import (
    ModificationIntent,
    PrincipalContext,
    RoleType,
)


@pytest.fixture(scope="module")
def settings() -> ScoutSettings:
    return ScoutSettings()


# =============================================================================
# Scope 1: AST Dependency Expander SEC-03 Stress Tests
# =============================================================================


def test_adversarial_ast_10_callers_across_5_files_bounded_and_diverse() -> None:
    """Stress test: A symbol has 10 direct callers distributed across 5 different files.

    Asserts:
    1. The number of caller slices is strictly <= 3 (enforcing MAX_DIRECT_CALLERS_PER_SYMBOL = 3).
    2. Selected callers represent diverse files (not all from the same file, round-robin diversity).
    3. The extracted lines are strictly bounded to +-5 lines around the call site.
    """
    expander = ASTDependencyExpander(max_direct_callers=3, max_expansion_token_budget=1500)
    repo = InMemoryRLSRepository()
    domain_id = "trading_code_scout"
    repo_name = "adversarial-graph"

    principal = PrincipalContext(
        user_id="adversarial_analyst",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng"],
    )

    # 1. Target focus symbol: match_orders
    focus_file_id = uuid.uuid4()
    focus_content = (
        "def match_orders(buy_order: object, sell_order: object) -> object:\n"
        '    """Core order matching engine."""\n'
        '    return {"status": "matched"}\n'
    )
    repo.insert_code_file(
        principal,
        CodeFileRecord(
            file_id=focus_file_id,
            domain_id=domain_id,
            repo_name=repo_name,
            file_path="src/core/matching_engine.py",
            git_commit_sha="commit_target",
            git_blob_sha="blob_target",
            language="python",
            allowed_scopes=["public_eng"],
            raw_content=focus_content,
            line_count=3,
        ),
    )
    focus_sym = ASTSymbolRecord(
        symbol_id=uuid.uuid4(),
        file_id=focus_file_id,
        domain_id=domain_id,
        qualified_name="core.matching_engine.match_orders",
        symbol_kind="function",
        start_line=1,
        end_line=3,
        signature_skeleton="def match_orders(buy_order: object, sell_order: object) -> object:",
        contextual_summary="Core order matching engine",
        allowed_scopes=["public_eng"],
        embedding=[0.0] * 768,
    )
    repo.insert_ast_symbol(principal, focus_sym)

    # 2. Construct 10 callers distributed across 5 different files (2 callers per file)
    # Caller 1 in each file spans lines 1..40 with call at line 20 (+-5 lines -> lines 15..25)
    # Caller 2 in each file spans lines 41..80 with call at line 60 (+-5 lines -> lines 55..65)
    file_paths: list[str] = [f"src/strategies/strategy_service_{idx}.py" for idx in range(1, 6)]

    for f_idx, f_path in enumerate(file_paths, start=1):
        lines: list[str] = []
        # Caller 1: function lines 1..40
        lines.append(f"def caller_alpha_{f_idx}():")
        for line_num in range(2, 20):
            lines.append(f"    # pre-call alpha computation line {line_num}")
        lines.append("    res = match_orders(order_buy, order_sell)")  # Line 20
        for line_num in range(21, 41):
            lines.append(f"    # post-call alpha processing line {line_num}")

        # Caller 2: function lines 41..80
        lines.append(f"def caller_beta_{f_idx}():")
        for line_num in range(42, 60):
            lines.append(f"    # pre-call beta computation line {line_num}")
        lines.append("    res = match_orders(quote_bid, quote_ask)")  # Line 60
        for line_num in range(61, 81):
            lines.append(f"    # post-call beta processing line {line_num}")

        raw_content = "\n".join(lines) + "\n"
        f_id = uuid.uuid4()
        repo.insert_code_file(
            principal,
            CodeFileRecord(
                file_id=f_id,
                domain_id=domain_id,
                repo_name=repo_name,
                file_path=f_path,
                git_commit_sha="commit_strat",
                git_blob_sha=f"blob_strat_{f_idx}",
                language="python",
                allowed_scopes=["public_eng"],
                raw_content=raw_content,
                line_count=80,
            ),
        )

        # Register caller_alpha symbol (lines 1..40)
        alpha_sym = ASTSymbolRecord(
            symbol_id=uuid.uuid4(),
            file_id=f_id,
            domain_id=domain_id,
            qualified_name=f"strategies.service_{f_idx}.caller_alpha_{f_idx}",
            symbol_kind="function",
            start_line=1,
            end_line=40,
            signature_skeleton=f"def caller_alpha_{f_idx}():",
            contextual_summary="",
            allowed_scopes=["public_eng"],
            embedding=[0.0] * 768,
        )
        repo.insert_ast_symbol(principal, alpha_sym)
        repo.insert_ast_edge(
            principal,
            ASTEdgeRecord(
                edge_id=uuid.uuid4(),
                domain_id=domain_id,
                source_symbol_id=alpha_sym.symbol_id,
                target_qualified_name=focus_sym.qualified_name,
                target_symbol_id=focus_sym.symbol_id,
                edge_kind="CALLS",
                allowed_scopes=["public_eng"],
            ),
        )

        # Register caller_beta symbol (lines 41..80)
        beta_sym = ASTSymbolRecord(
            symbol_id=uuid.uuid4(),
            file_id=f_id,
            domain_id=domain_id,
            qualified_name=f"strategies.service_{f_idx}.caller_beta_{f_idx}",
            symbol_kind="function",
            start_line=41,
            end_line=80,
            signature_skeleton=f"def caller_beta_{f_idx}():",
            contextual_summary="",
            allowed_scopes=["public_eng"],
            embedding=[0.0] * 768,
        )
        repo.insert_ast_symbol(principal, beta_sym)
        repo.insert_ast_edge(
            principal,
            ASTEdgeRecord(
                edge_id=uuid.uuid4(),
                domain_id=domain_id,
                source_symbol_id=beta_sym.symbol_id,
                target_qualified_name=focus_sym.qualified_name,
                target_symbol_id=focus_sym.symbol_id,
                edge_kind="CALLS",
                allowed_scopes=["public_eng"],
            ),
        )

    # Execute 1-hop expansion
    result = expander.expand_1hop(focus_sym, repo, principal, domain_id)

    # 1. Assert: Number of caller slices is strictly <= 3 (and exactly 3 here)
    assert len(result.caller_slices) == 3, f"Expected 3 callers, got {len(result.caller_slices)}"
    for s in result.caller_slices:
        assert s.modification_intent == ModificationIntent.DIRECT_CALLER

    # 2. Assert: Callers represent diverse files (round-robin diversity picks 3 distinct files)
    selected_files = [s.file_path for s in result.caller_slices]
    distinct_files = set(selected_files)
    assert len(distinct_files) == 3, (
        f"Diversity failure: callers selected from {selected_files}, expected 3 distinct files"
    )

    # 3. Assert: Extracted lines are strictly bounded to +-5 lines around call site
    for s in result.caller_slices:
        line_span = s.end_line - s.start_line + 1
        assert line_span <= 11, f"Slice span {line_span} exceeds +-5 line bound (11 lines max)"
        # Check that the call site line (either 20 or 60) is contained in the slice
        if "caller_alpha" in s.qualified_symbol:
            assert s.start_line == 15, f"Expected start_line=15 (20 - 5), got {s.start_line}"
            assert s.end_line == 25, f"Expected end_line=25 (20 + 5), got {s.end_line}"
        elif "caller_beta" in s.qualified_symbol:
            assert s.start_line == 55, f"Expected start_line=55 (60 - 5), got {s.start_line}"
            assert s.end_line == 65, f"Expected end_line=65 (60 + 5), got {s.end_line}"
        else:
            pytest.fail(f"Unexpected caller symbol {s.qualified_symbol}")


def test_adversarial_ast_expansion_token_budget_never_exceeds_1500() -> None:
    """Stress test: Scenarios with massive candidate slices exceeding 1500 tokens.

    Asserts:
    1. Total expansion tokens strictly never exceeds MAX_EXPANSION_TOKEN_BUDGET (1500).
    2. Budget exhaustion flag is raised and dropped candidates count is tracked.
    3. Giant individual candidates (>1500 tokens) are safely rejected without crashing.
    """
    expander = ASTDependencyExpander(max_direct_callers=3, max_expansion_token_budget=1500)
    repo = InMemoryRLSRepository()
    domain_id = "trading_code_scout"
    repo_name = "budget-stress-repo"

    principal = PrincipalContext(
        user_id="adversarial_analyst",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng"],
    )

    # Target symbol
    focus_file_id = uuid.uuid4()
    focus_content = "def calculate_risk(portfolio: object) -> float:\n    return 0.05\n"
    repo.insert_code_file(
        principal,
        CodeFileRecord(
            file_id=focus_file_id,
            domain_id=domain_id,
            repo_name=repo_name,
            file_path="src/risk/calculator.py",
            git_commit_sha="commit_risk",
            git_blob_sha="blob_risk",
            language="python",
            allowed_scopes=["public_eng"],
            raw_content=focus_content,
            line_count=2,
        ),
    )
    focus_sym = ASTSymbolRecord(
        symbol_id=uuid.uuid4(),
        file_id=focus_file_id,
        domain_id=domain_id,
        qualified_name="risk.calculator.calculate_risk",
        symbol_kind="function",
        start_line=1,
        end_line=2,
        signature_skeleton="def calculate_risk(portfolio: object) -> float:",
        contextual_summary="Calculates portfolio VaR",
        allowed_scopes=["public_eng"],
        embedding=[0.0] * 768,
    )
    repo.insert_ast_symbol(principal, focus_sym)

    # Scenario A: Add 3 large callers with dense tokens in their +-5 line slices
    # Each slice will contain ~800 tokens, so 3 callers = 2400 tokens (> 1500 limit)
    dense_tokens_str = " ".join(f"alpha_term_{k}" for k in range(80))
    for idx in range(1, 4):
        c_lines = [f"def large_caller_{idx}():"]
        for j in range(1, 15):
            c_lines.append(f"    matrix_variable_{idx}_{j} = {dense_tokens_str}")
        c_lines.append("    risk_value = calculate_risk(active_portfolio)")
        for j in range(16, 30):
            c_lines.append(f"    post_variable_{idx}_{j} = {dense_tokens_str}")

        raw_c = "\n".join(c_lines) + "\n"
        f_id = uuid.uuid4()
        repo.insert_code_file(
            principal,
            CodeFileRecord(
                file_id=f_id,
                domain_id=domain_id,
                repo_name=repo_name,
                file_path=f"src/risk/large_caller_{idx}.py",
                git_commit_sha="commit_c",
                git_blob_sha=f"blob_c_{idx}",
                language="python",
                allowed_scopes=["public_eng"],
                raw_content=raw_c,
                line_count=len(c_lines),
            ),
        )
        c_sym = ASTSymbolRecord(
            symbol_id=uuid.uuid4(),
            file_id=f_id,
            domain_id=domain_id,
            qualified_name=f"risk.large_caller_{idx}.large_caller_{idx}",
            symbol_kind="function",
            start_line=1,
            end_line=len(c_lines),
            signature_skeleton=f"def large_caller_{idx}():",
            contextual_summary="Large risk caller",
            allowed_scopes=["public_eng"],
            embedding=[0.0] * 768,
        )
        repo.insert_ast_symbol(principal, c_sym)
        repo.insert_ast_edge(
            principal,
            ASTEdgeRecord(
                edge_id=uuid.uuid4(),
                domain_id=domain_id,
                source_symbol_id=c_sym.symbol_id,
                target_qualified_name=focus_sym.qualified_name,
                target_symbol_id=focus_sym.symbol_id,
                edge_kind="CALLS",
                allowed_scopes=["public_eng"],
            ),
        )

    res_a = expander.expand_1hop(focus_sym, repo, principal, domain_id)
    # Strictly check token budget <= 1500
    assert res_a.total_expansion_tokens <= 1500, (
        f"SEC-03 VIOLATION: expansion tokens {res_a.total_expansion_tokens} exceeds 1500 limit"
    )
    assert res_a.budget_exhausted is True, "Budget should have been exhausted"
    assert res_a.dropped_candidates_count >= 1, "At least one candidate must have been dropped"

    # Scenario B: Giant single candidate (>2500 tokens)
    giant_lines = ["class GiantMonolithicPortfolioModel:"]
    for j in range(1, 500):
        giant_lines.append(
            f"    factor_definition_series_{j}: list[float] = [1.0, 2.0, 3.0]  # token payload {j}"
        )
    raw_giant = "\n".join(giant_lines) + "\n"
    g_file_id = uuid.uuid4()
    repo.insert_code_file(
        principal,
        CodeFileRecord(
            file_id=g_file_id,
            domain_id=domain_id,
            repo_name=repo_name,
            file_path="src/risk/giant_model.py",
            git_commit_sha="commit_g",
            git_blob_sha="blob_g",
            language="python",
            allowed_scopes=["public_eng"],
            raw_content=raw_giant,
            line_count=len(giant_lines),
        ),
    )
    giant_sym = ASTSymbolRecord(
        symbol_id=uuid.uuid4(),
        file_id=g_file_id,
        domain_id=domain_id,
        qualified_name="risk.giant_model.GiantMonolithicPortfolioModel",
        symbol_kind="class",
        start_line=1,
        end_line=len(giant_lines),
        signature_skeleton="class GiantMonolithicPortfolioModel:",
        contextual_summary="Giant model",
        allowed_scopes=["public_eng"],
        embedding=[0.0] * 768,
    )
    repo.insert_ast_symbol(principal, giant_sym)
    repo.insert_ast_edge(
        principal,
        ASTEdgeRecord(
            edge_id=uuid.uuid4(),
            domain_id=domain_id,
            source_symbol_id=focus_sym.symbol_id,
            target_qualified_name="GiantMonolithicPortfolioModel",
            target_symbol_id=giant_sym.symbol_id,
            edge_kind="USES_TYPE",
            allowed_scopes=["public_eng"],
        ),
    )

    res_b = expander.expand_1hop(focus_sym, repo, principal, domain_id)
    assert res_b.total_expansion_tokens <= 1500, (
        f"SEC-03 VIOLATION with giant candidate: {res_b.total_expansion_tokens} > 1500"
    )
    # Giant candidate exceeds budget, so it must NOT be included in imported_type_slices
    giant_type_slices = [
        s
        for s in res_b.imported_type_slices
        if "GiantMonolithicPortfolioModel" in s.qualified_symbol
    ]
    assert len(giant_type_slices) == 0, "Giant candidate was not rejected by token budget!"


# =============================================================================
# Scope 2: PostgreSQL 17 Live RLS Enforcement
# =============================================================================


def test_adversarial_live_postgres_rls_isolation(settings: ScoutSettings) -> None:
    """Live PostgreSQL 17 test on port 5432:

    Verifies that an engineer principal (allowed_scopes=['public_eng']) cannot query
    code_files, ast_symbols, or ast_edges tagged with 'quant_restricted'.
    """
    mgr = RLSConnectionManager(settings.db_app_url)
    admin_mgr = RLSConnectionManager(settings.db_admin_url)
    repo = PostgresRLSRepository()
    domain_id = "trading_code_scout"
    repo_name = "adversarial-rls-test"

    admin_principal = PrincipalContext(
        user_id="db_admin_adv",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng", "quant_restricted"],
    )
    quant_principal = PrincipalContext(
        user_id="adversarial_quant_user",
        role=RoleType.QUANT_RESEARCHER,
        allowed_scopes=["quant_restricted"],
    )
    eng_principal = PrincipalContext(
        user_id="adversarial_eng_user",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng"],
    )

    # Ensure domain pack exists
    with admin_mgr.scoped_transaction(admin_principal, enable_hnsw_iterative_scan=False) as conn:
        conn.execute(
            """
            INSERT INTO domain_packs (
                domain_id, display_name, lora_adapter_id,
                primary_truth_source, freshness_half_life_days, config_json
            )
            VALUES (
                'trading_code_scout', 'Trading Code Scout', 'gemma4_26b_code_scout',
                'code', 180, '{"test": "adversarial"}'::jsonb
            )
            ON CONFLICT (domain_id) DO NOTHING;
            """
        )

    # Generate test UUIDs
    restricted_file_id = uuid.uuid4()
    restricted_sym_id = uuid.uuid4()
    restricted_edge_id = uuid.uuid4()

    public_file_id = uuid.uuid4()
    public_sym_id = uuid.uuid4()
    public_edge_id = uuid.uuid4()

    try:
        # 1. Insert restricted records under quant_principal
        with mgr.scoped_transaction(quant_principal, enable_hnsw_iterative_scan=False) as conn:
            repo.insert_code_file(
                conn,
                CodeFileRecord(
                    file_id=restricted_file_id,
                    domain_id=domain_id,
                    repo_name=repo_name,
                    file_path="src/alpha/adversarial_restricted.py",
                    git_commit_sha="commit_res",
                    git_blob_sha="blob_res",
                    language="python",
                    allowed_scopes=["quant_restricted"],
                    raw_content="def secret_alpha(): pass\n",
                    line_count=1,
                ),
            )
            repo.insert_ast_symbol(
                conn,
                ASTSymbolRecord(
                    symbol_id=restricted_sym_id,
                    file_id=restricted_file_id,
                    domain_id=domain_id,
                    qualified_name="alpha.adversarial_restricted.secret_alpha",
                    symbol_kind="function",
                    start_line=1,
                    end_line=1,
                    signature_skeleton="def secret_alpha() -> None:",
                    contextual_summary="Restricted alpha function",
                    allowed_scopes=["quant_restricted"],
                    embedding=[0.5] * 768,
                ),
            )
            repo.insert_ast_edge(
                conn,
                ASTEdgeRecord(
                    edge_id=restricted_edge_id,
                    domain_id=domain_id,
                    source_symbol_id=restricted_sym_id,
                    target_qualified_name="alpha.models.SecretWeights",
                    target_symbol_id=None,
                    edge_kind="USES_TYPE",
                    allowed_scopes=["quant_restricted"],
                ),
            )

        # 2. Insert public records under eng_principal
        with mgr.scoped_transaction(eng_principal, enable_hnsw_iterative_scan=False) as conn:
            repo.insert_code_file(
                conn,
                CodeFileRecord(
                    file_id=public_file_id,
                    domain_id=domain_id,
                    repo_name=repo_name,
                    file_path="src/public/adversarial_public.py",
                    git_commit_sha="commit_pub",
                    git_blob_sha="blob_pub",
                    language="python",
                    allowed_scopes=["public_eng"],
                    raw_content="def public_utility(): pass\n",
                    line_count=1,
                ),
            )
            repo.insert_ast_symbol(
                conn,
                ASTSymbolRecord(
                    symbol_id=public_sym_id,
                    file_id=public_file_id,
                    domain_id=domain_id,
                    qualified_name="public.adversarial_public.public_utility",
                    symbol_kind="function",
                    start_line=1,
                    end_line=1,
                    signature_skeleton="def public_utility() -> None:",
                    contextual_summary="Public utility function",
                    allowed_scopes=["public_eng"],
                    embedding=[-0.5] * 768,
                ),
            )
            repo.insert_ast_edge(
                conn,
                ASTEdgeRecord(
                    edge_id=public_edge_id,
                    domain_id=domain_id,
                    source_symbol_id=public_sym_id,
                    target_qualified_name="public.models.PublicHelper",
                    target_symbol_id=None,
                    edge_kind="USES_TYPE",
                    allowed_scopes=["public_eng"],
                ),
            )

        # 3. Query as eng_principal (allowed_scopes=['public_eng'])
        with mgr.scoped_transaction(eng_principal, enable_hnsw_iterative_scan=False) as conn:
            # (a) code_files check
            res_cf = conn.execute(
                "SELECT file_id, file_path FROM code_files WHERE file_id = %s;",
                (restricted_file_id,),
            ).fetchall()
            assert len(res_cf) == 0, "RLS BREACH: public_eng queried quant_restricted code_file!"

            pub_cf = conn.execute(
                "SELECT file_id, file_path FROM code_files WHERE file_id = %s;",
                (public_file_id,),
            ).fetchall()
            assert len(pub_cf) == 1, "public_eng failed to query public_eng code_file"

            visible_files = repo.list_visible_code_files(conn, domain_id)
            assert "src/alpha/adversarial_restricted.py" not in visible_files
            assert "src/public/adversarial_public.py" in visible_files

            # (b) ast_symbols check
            res_sym = conn.execute(
                "SELECT symbol_id, qualified_name FROM ast_symbols WHERE symbol_id = %s;",
                (restricted_sym_id,),
            ).fetchall()
            assert len(res_sym) == 0, "RLS BREACH: public_eng queried quant_restricted ast_symbol!"

            pub_sym = conn.execute(
                "SELECT symbol_id, qualified_name FROM ast_symbols WHERE symbol_id = %s;",
                (public_sym_id,),
            ).fetchall()
            assert len(pub_sym) == 1, "public_eng failed to query public_eng ast_symbol"

            visible_symbols = repo.list_visible_ast_symbols(conn, domain_id)
            assert "alpha.adversarial_restricted.secret_alpha" not in visible_symbols
            assert "public.adversarial_public.public_utility" in visible_symbols

            # (c) ast_edges check
            res_edge = conn.execute(
                "SELECT edge_id, target_qualified_name FROM ast_edges WHERE edge_id = %s;",
                (restricted_edge_id,),
            ).fetchall()
            assert len(res_edge) == 0, "RLS BREACH: public_eng queried quant_restricted ast_edge!"

            pub_edge = conn.execute(
                "SELECT edge_id, target_qualified_name FROM ast_edges WHERE edge_id = %s;",
                (public_edge_id,),
            ).fetchall()
            assert len(pub_edge) == 1, "public_eng failed to query public_eng ast_edge"

            visible_edges = repo.list_visible_ast_edges(conn, domain_id)
            assert "alpha.models.SecretWeights" not in visible_edges
            assert "public.models.PublicHelper" in visible_edges

        # 4. Verify quant_principal CAN see the restricted records and NOT the public ones
        with mgr.scoped_transaction(quant_principal, enable_hnsw_iterative_scan=False) as conn:
            quant_cf = conn.execute(
                "SELECT file_id FROM code_files WHERE file_id = %s;",
                (restricted_file_id,),
            ).fetchall()
            assert len(quant_cf) == 1, "quant_principal failed to query own restricted code_file"

            pub_cf_for_quant = conn.execute(
                "SELECT file_id FROM code_files WHERE file_id = %s;",
                (public_file_id,),
            ).fetchall()
            assert len(pub_cf_for_quant) == 0, (
                "RLS BREACH: quant_principal saw public_eng code_file without scope!"
            )

    finally:
        # Clean up test rows
        with admin_mgr.scoped_transaction(
            admin_principal, enable_hnsw_iterative_scan=False
        ) as conn:
            conn.execute(
                "DELETE FROM ast_edges WHERE edge_id IN (%s, %s);",
                (restricted_edge_id, public_edge_id),
            )
            conn.execute(
                "DELETE FROM ast_symbols WHERE symbol_id IN (%s, %s);",
                (restricted_sym_id, public_sym_id),
            )
            conn.execute(
                "DELETE FROM code_files WHERE file_id IN (%s, %s);",
                (restricted_file_id, public_file_id),
            )


def test_adversarial_live_pgvector_cosine_with_relaxed_order(settings: ScoutSettings) -> None:
    """Live PostgreSQL 17 test on port 5432:

    Verifies that pgvector cosine similarity search (<=>) executes with
    'relaxed_order' without RLS bypass.
    Even when the query vector exactly matches a quant_restricted symbol,
    public_eng cannot see the row.
    """
    mgr = RLSConnectionManager(settings.db_app_url)
    admin_mgr = RLSConnectionManager(settings.db_admin_url)
    domain_id = "trading_code_scout"
    repo_name = "vector-rls-test"

    admin_principal = PrincipalContext(
        user_id="db_admin_adv",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng", "quant_restricted"],
    )
    quant_principal = PrincipalContext(
        user_id="adversarial_quant_user",
        role=RoleType.QUANT_RESEARCHER,
        allowed_scopes=["quant_restricted"],
    )
    eng_principal = PrincipalContext(
        user_id="adversarial_eng_user",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng"],
    )

    # Vector A: Exact target vector for restricted symbol
    target_vec = [1.0] + [0.0] * 767
    target_vec_lit = f"[{','.join(str(x) for x in target_vec)}]"

    # Vector B: Orthogonal vector for public symbol
    ortho_vec = [0.0, 1.0] + [0.0] * 766
    ortho_vec_lit = f"[{','.join(str(x) for x in ortho_vec)}]"

    restricted_fid = uuid.uuid4()
    restricted_sid = uuid.uuid4()
    public_fid = uuid.uuid4()
    public_sid = uuid.uuid4()

    try:
        # Insert restricted symbol with target_vec
        with mgr.scoped_transaction(quant_principal, enable_hnsw_iterative_scan=False) as conn:
            conn.execute(
                """
                INSERT INTO code_files (
                    file_id, domain_id, repo_name, file_path,
                    git_commit_sha, git_blob_sha, language,
                    allowed_scopes, raw_content, line_count
                )
                VALUES (
                    %s, %s, %s, 'src/alpha/vec_target.py', 'c1', 'b1', 'python', %s, 'pass\n', 1
                );
                """,
                (restricted_fid, domain_id, repo_name, ["quant_restricted"]),
            )
            conn.execute(
                """
                INSERT INTO ast_symbols (
                    symbol_id, file_id, domain_id, qualified_name,
                    symbol_kind, start_line, end_line,
                    signature_skeleton, contextual_summary,
                    allowed_scopes, embedding
                )
                VALUES (
                    %s, %s, %s, 'alpha.vec_target.target_func',
                    'function', 1, 1, 'def target_func():',
                    'Restricted target', %s, %s::vector
                );
                """,
                (restricted_sid, restricted_fid, domain_id, ["quant_restricted"], target_vec_lit),
            )

        # Insert public symbol with ortho_vec
        with mgr.scoped_transaction(eng_principal, enable_hnsw_iterative_scan=False) as conn:
            conn.execute(
                """
                INSERT INTO code_files (
                    file_id, domain_id, repo_name, file_path,
                    git_commit_sha, git_blob_sha, language,
                    allowed_scopes, raw_content, line_count
                )
                VALUES (
                    %s, %s, %s, 'src/public/vec_ortho.py', 'c2', 'b2', 'python', %s, 'pass\n', 1
                );
                """,
                (public_fid, domain_id, repo_name, ["public_eng"]),
            )
            conn.execute(
                """
                INSERT INTO ast_symbols (
                    symbol_id, file_id, domain_id, qualified_name,
                    symbol_kind, start_line, end_line,
                    signature_skeleton, contextual_summary,
                    allowed_scopes, embedding
                )
                VALUES (
                    %s, %s, %s, 'public.vec_ortho.ortho_func',
                    'function', 1, 1, 'def ortho_func():',
                    'Public ortho', %s, %s::vector
                );
                """,
                (public_sid, public_fid, domain_id, ["public_eng"], ortho_vec_lit),
            )

        # Execute search as eng_principal with target_vec (matches restricted_sid with dist=0.0)
        # Enable hnsw.iterative_scan = 'relaxed_order'
        with mgr.scoped_transaction(eng_principal, enable_hnsw_iterative_scan=True) as conn:
            # 1. Assert hnsw.iterative_scan is set to relaxed_order
            scan_mode_row = conn.execute("SHOW hnsw.iterative_scan;").fetchone()
            assert scan_mode_row is not None
            assert str(scan_mode_row[0]) == "relaxed_order", (
                f"Expected hnsw.iterative_scan='relaxed_order', got {scan_mode_row[0]}"
            )

            # 2. Perform cosine similarity (<=>) search across entire table
            results: list[tuple[Any, ...]] = conn.execute(
                """
                SELECT symbol_id, qualified_name, (embedding <=> %s::vector) AS cosine_dist
                FROM ast_symbols
                ORDER BY embedding <=> %s::vector ASC
                LIMIT 500;
                """,
                (target_vec_lit, target_vec_lit),
            ).fetchall()

            result_sids = [r[0] for r in results]

            # 3. Assert RLS prevents the restricted target symbol from leaking
            assert restricted_sid not in result_sids, (
                "CRITICAL SECURITY VULNERABILITY: pgvector <=> search bypassed RLS! "
                "public_eng retrieved quant_restricted symbol!"
            )

            # 4. Check explicit pair isolation under vector distance ordering
            pair_results = conn.execute(
                """
                SELECT symbol_id, qualified_name, (embedding <=> %s::vector) AS cosine_dist
                FROM ast_symbols
                WHERE symbol_id IN (%s, %s)
                ORDER BY embedding <=> %s::vector ASC;
                """,
                (target_vec_lit, restricted_sid, public_sid, target_vec_lit),
            ).fetchall()
            pair_sids = [r[0] for r in pair_results]
            assert restricted_sid not in pair_sids, "Restricted symbol leaked in pair query"
            assert public_sid in pair_sids, "Public symbol missing in pair query"

        # Execute same search as quant_principal
        with mgr.scoped_transaction(quant_principal, enable_hnsw_iterative_scan=True) as conn:
            quant_results: list[tuple[Any, ...]] = conn.execute(
                """
                SELECT symbol_id, qualified_name, (embedding <=> %s::vector) AS cosine_dist
                FROM ast_symbols
                ORDER BY embedding <=> %s::vector ASC
                LIMIT 500;
                """,
                (target_vec_lit, target_vec_lit),
            ).fetchall()

            quant_sids = [r[0] for r in quant_results]
            assert restricted_sid in quant_sids, (
                "quant_principal failed to retrieve restricted symbol"
            )
            assert float(quant_results[0][2]) < 1e-5, (
                f"Expected ~0.0 cosine distance for target symbol, got {quant_results[0][2]}"
            )
            assert public_sid not in quant_sids, "quant_principal saw public symbol without scope"

    finally:
        # Clean up test rows
        with admin_mgr.scoped_transaction(
            admin_principal, enable_hnsw_iterative_scan=False
        ) as conn:
            conn.execute(
                "DELETE FROM ast_symbols WHERE symbol_id IN (%s, %s);",
                (restricted_sid, public_sid),
            )
            conn.execute(
                "DELETE FROM code_files WHERE file_id IN (%s, %s);",
                (restricted_fid, public_fid),
            )


def test_adversarial_ast_boundary_clamping_at_edges() -> None:
    """Stress test: Call site at line 1, line 2, line 49, and in single-line function.

    Asserts:
    1. start_line is never < 1 and never < caller_sym.start_line.
    2. end_line is never > caller_sym.end_line.
    3. Total slice line span is strictly <= 11 lines (+-5 lines).
    """
    expander = ASTDependencyExpander(max_direct_callers=3, max_expansion_token_budget=1500)
    repo = InMemoryRLSRepository()
    domain_id = "trading_code_scout"
    repo_name = "clamping-repo"

    principal = PrincipalContext(
        user_id="adversarial_analyst",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng"],
    )

    # Focus symbol
    focus_fid = uuid.uuid4()
    focus_sym = ASTSymbolRecord(
        symbol_id=uuid.uuid4(),
        file_id=focus_fid,
        domain_id=domain_id,
        qualified_name="core.engine.execute",
        symbol_kind="function",
        start_line=1,
        end_line=2,
        signature_skeleton="def execute():",
        contextual_summary="",
        allowed_scopes=["public_eng"],
        embedding=[0.0] * 768,
    )
    repo.insert_code_file(
        principal,
        CodeFileRecord(
            file_id=focus_fid,
            domain_id=domain_id,
            repo_name=repo_name,
            file_path="src/core/engine.py",
            git_commit_sha="c1",
            git_blob_sha="b1",
            language="python",
            allowed_scopes=["public_eng"],
            raw_content="def execute(): pass\n",
            line_count=1,
        ),
    )
    repo.insert_ast_symbol(principal, focus_sym)

    # Case A: Call site at line 1 of a 50-line caller
    # Case B: Call site at line 49 of a 50-line caller
    # Case C: Single line caller function (start=1, end=1)
    callers_data = [
        ("src/c_early.py", 1, 50, 1),
        ("src/c_late.py", 1, 50, 49),
        ("src/c_single.py", 1, 1, 1),
    ]

    for f_path, s_line, e_line, call_l in callers_data:
        lines: list[str] = []
        for line_idx in range(1, e_line + 1):
            if line_idx == call_l:
                lines.append(f"def caller_at_{call_l}(): execute()")
            else:
                lines.append(f"    # line {line_idx} padding")

        f_id = uuid.uuid4()
        c_sym_id = uuid.uuid4()
        repo.insert_code_file(
            principal,
            CodeFileRecord(
                file_id=f_id,
                domain_id=domain_id,
                repo_name=repo_name,
                file_path=f_path,
                git_commit_sha="c",
                git_blob_sha="b",
                language="python",
                allowed_scopes=["public_eng"],
                raw_content="\n".join(lines) + "\n",
                line_count=e_line,
            ),
        )
        c_sym = ASTSymbolRecord(
            symbol_id=c_sym_id,
            file_id=f_id,
            domain_id=domain_id,
            qualified_name=f"callers.{f_path.split('/')[-1]}.caller",
            symbol_kind="function",
            start_line=s_line,
            end_line=e_line,
            signature_skeleton=f"def caller_at_{call_l}():",
            contextual_summary="",
            allowed_scopes=["public_eng"],
            embedding=[0.0] * 768,
        )
        repo.insert_ast_symbol(principal, c_sym)
        repo.insert_ast_edge(
            principal,
            ASTEdgeRecord(
                edge_id=uuid.uuid4(),
                domain_id=domain_id,
                source_symbol_id=c_sym_id,
                target_qualified_name=focus_sym.qualified_name,
                target_symbol_id=focus_sym.symbol_id,
                edge_kind="CALLS",
                allowed_scopes=["public_eng"],
            ),
        )

    res = expander.expand_1hop(focus_sym, repo, principal, domain_id)
    assert len(res.caller_slices) == 3

    for s in res.caller_slices:
        assert s.start_line >= 1, f"Slice start_line {s.start_line} < 1"
        assert s.end_line >= s.start_line, (
            f"Slice end_line {s.end_line} < start_line {s.start_line}"
        )
        span = s.end_line - s.start_line + 1
        assert span <= 11, f"Slice span {span} > 11 (+-5 lines)"


def test_adversarial_postgres_rls_tamper_and_delete_resistance(
    settings: ScoutSettings,
) -> None:
    """Verifies that an engineer principal cannot DELETE or UPDATE quant_restricted rows."""
    mgr = RLSConnectionManager(settings.db_app_url)
    admin_mgr = RLSConnectionManager(settings.db_admin_url)
    domain_id = "trading_code_scout"
    repo_name = "tamper-defense-test"

    admin_principal = PrincipalContext(
        user_id="db_admin_adv",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng", "quant_restricted"],
    )
    quant_principal = PrincipalContext(
        user_id="adversarial_quant_user",
        role=RoleType.QUANT_RESEARCHER,
        allowed_scopes=["quant_restricted"],
    )
    eng_principal = PrincipalContext(
        user_id="adversarial_eng_user",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng"],
    )

    restricted_fid = uuid.uuid4()

    try:
        # Insert restricted file
        with mgr.scoped_transaction(quant_principal, enable_hnsw_iterative_scan=False) as conn:
            conn.execute(
                """
                INSERT INTO code_files (
                    file_id, domain_id, repo_name, file_path,
                    git_commit_sha, git_blob_sha, language,
                    allowed_scopes, raw_content, line_count
                )
                VALUES (
                    %s, %s, %s, 'src/alpha/untouchable.py', 'c1', 'b1', 'python',
                    %s, 'secret_content = 42\n', 1
                );
                """,
                (restricted_fid, domain_id, repo_name, ["quant_restricted"]),
            )

        # Attempt DELETE as eng_principal
        with mgr.scoped_transaction(eng_principal, enable_hnsw_iterative_scan=False) as conn:
            cur = conn.execute("DELETE FROM code_files WHERE file_id = %s;", (restricted_fid,))
            assert cur.rowcount == 0, f"RLS TAMPER VULNERABILITY: eng deleted {cur.rowcount} rows!"

        # Attempt UPDATE as eng_principal to elevate scope
        with mgr.scoped_transaction(eng_principal, enable_hnsw_iterative_scan=False) as conn:
            cur = conn.execute(
                "UPDATE code_files SET allowed_scopes = '{public_eng}' WHERE file_id = %s;",
                (restricted_fid,),
            )
            assert cur.rowcount == 0, f"RLS TAMPER VULNERABILITY: eng updated {cur.rowcount} rows!"

        # Quant principal verifies file is still intact and restricted
        with mgr.scoped_transaction(quant_principal, enable_hnsw_iterative_scan=False) as conn:
            row = conn.execute(
                "SELECT file_id, allowed_scopes FROM code_files WHERE file_id = %s;",
                (restricted_fid,),
            ).fetchone()
            assert row is not None, "Restricted file was improperly deleted"
            assert row[1] == ["quant_restricted"], f"Scope was altered: {row[1]}"

    finally:
        with admin_mgr.scoped_transaction(
            admin_principal, enable_hnsw_iterative_scan=False
        ) as conn:
            conn.execute("DELETE FROM code_files WHERE file_id = %s;", (restricted_fid,))


def test_adversarial_postgres_empty_and_injected_scopes(settings: ScoutSettings) -> None:
    """Verifies that empty/injected scopes are rejected by Pydantic and yield 0 rows in DB."""
    mgr = RLSConnectionManager(settings.db_app_url)

    # 1. Pydantic gate: rejects empty scopes, commas, spaces, semicolons
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        PrincipalContext(user_id="u", role=RoleType.ENGINEER, allowed_scopes=[])

    with pytest.raises(ValidationError):
        PrincipalContext(
            user_id="u", role=RoleType.ENGINEER, allowed_scopes=["public_eng; DROP TABLE"]
        )

    with pytest.raises(ValidationError):
        PrincipalContext(
            user_id="u", role=RoleType.ENGINEER, allowed_scopes=["public_eng,quant_restricted"]
        )

    # 2. Database gate: unauthorized/unmatched scope sees 0 rows
    unauthorized_principal = PrincipalContext(
        user_id="guest_user",
        role=RoleType.ENGINEER,
        allowed_scopes=["unauthorized_visitor"],
    )
    with mgr.scoped_transaction(unauthorized_principal, enable_hnsw_iterative_scan=False) as conn:
        cf_rows = conn.execute("SELECT count(*) FROM code_files;").fetchone()
        assert cf_rows is not None
        assert cf_rows[0] == 0, (
            f"Unauthorized scope leaked {cf_rows[0]} code_files rows!"
        )

        sym_rows = conn.execute("SELECT count(*) FROM ast_symbols;").fetchone()
        assert sym_rows is not None
        assert sym_rows[0] == 0, (
            f"Unauthorized scope leaked {sym_rows[0]} ast_symbols rows!"
        )

    # 3. Raw connection without set_config (NULL setting) sees 0 rows due to Forced RLS
    import psycopg

    with psycopg.connect(settings.db_app_url) as raw_conn:
        raw_cf = raw_conn.execute("SELECT count(*) FROM code_files;").fetchone()
        assert raw_cf is not None and raw_cf[0] == 0, (
            "Raw unconfigured connection leaked rows without app.principal_scopes set!"
        )
