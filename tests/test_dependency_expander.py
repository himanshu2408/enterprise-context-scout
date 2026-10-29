"""Unit tests for ASTDependencyExpander: 1-hop bounded closure, caller limits, SEC-03."""

import uuid

import pytest

from context_scout.ast_engine.dependency_expander import ASTDependencyExpander
from context_scout.ast_engine.models import ASTExpansionResult
from context_scout.db.repositories import (
    ASTEdgeRecord,
    ASTSymbolRecord,
    CodeFileRecord,
    InMemoryRLSRepository,
)
from context_scout.schemas.contracts import (
    ModificationIntent,
    PrincipalContext,
    RoleType,
)


@pytest.fixture
def expander() -> ASTDependencyExpander:
    return ASTDependencyExpander(max_direct_callers=3, max_expansion_token_budget=1500)


@pytest.fixture
def principal_eng() -> PrincipalContext:
    return PrincipalContext(
        user_id="dev_user",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng"],
    )


@pytest.fixture
def populated_repo(
    principal_eng: PrincipalContext,
) -> tuple[InMemoryRLSRepository, ASTSymbolRecord]:
    repo = InMemoryRLSRepository()
    domain_id = "trading_code_scout"
    repo_name = "trading-core"

    # 1. Target file with focus symbol: route_order
    focus_file_id = uuid.uuid4()
    focus_content = (
        "from .models import Order, ExecutionReceipt\n"
        "\n"
        "def route_order(order: Order) -> ExecutionReceipt:\n"
        '    """Routes order to the optimal execution venue."""\n'
        "    venue = select_venue(order)\n"
        "    receipt = venue.execute(order)\n"
        "    return receipt\n"
    )
    repo.insert_code_file(
        principal_eng,
        CodeFileRecord(
            file_id=focus_file_id,
            domain_id=domain_id,
            repo_name=repo_name,
            file_path="src/execution/order_router.py",
            git_commit_sha="commit_focus",
            git_blob_sha="blob_focus",
            language="python",
            allowed_scopes=["public_eng"],
            raw_content=focus_content,
            line_count=7,
        ),
    )
    focus_sym = ASTSymbolRecord(
        symbol_id=uuid.uuid4(),
        file_id=focus_file_id,
        domain_id=domain_id,
        qualified_name="execution.order_router.route_order",
        symbol_kind="function",
        start_line=3,
        end_line=7,
        signature_skeleton="def route_order(order: Order) -> ExecutionReceipt:",
        contextual_summary="Routes order to the optimal execution venue",
        allowed_scopes=["public_eng"],
        embedding=[0.0] * 768,
    )
    repo.insert_ast_symbol(principal_eng, focus_sym)

    # 2. Type definitions in src/execution/models.py
    type_file_id = uuid.uuid4()
    type_content = (
        "class Order:\n"
        "    id: str\n"
        "    qty: float\n"
        "\n"
        "class ExecutionReceipt:\n"
        "    receipt_id: str\n"
        "    filled_qty: float\n"
    )
    repo.insert_code_file(
        principal_eng,
        CodeFileRecord(
            file_id=type_file_id,
            domain_id=domain_id,
            repo_name=repo_name,
            file_path="src/execution/models.py",
            git_commit_sha="commit_type",
            git_blob_sha="blob_type",
            language="python",
            allowed_scopes=["public_eng"],
            raw_content=type_content,
            line_count=6,
        ),
    )
    order_sym = ASTSymbolRecord(
        symbol_id=uuid.uuid4(),
        file_id=type_file_id,
        domain_id=domain_id,
        qualified_name="execution.models.Order",
        symbol_kind="class",
        start_line=1,
        end_line=3,
        signature_skeleton="class Order:",
        contextual_summary="Order entity",
        allowed_scopes=["public_eng"],
        embedding=[0.0] * 768,
    )
    receipt_sym = ASTSymbolRecord(
        symbol_id=uuid.uuid4(),
        file_id=type_file_id,
        domain_id=domain_id,
        qualified_name="execution.models.ExecutionReceipt",
        symbol_kind="class",
        start_line=5,
        end_line=6,
        signature_skeleton="class ExecutionReceipt:",
        contextual_summary="Receipt entity",
        allowed_scopes=["public_eng"],
        embedding=[0.0] * 768,
    )
    repo.insert_ast_symbol(principal_eng, order_sym)
    repo.insert_ast_symbol(principal_eng, receipt_sym)

    # USES_TYPE edges from focus_sym to Order and ExecutionReceipt
    repo.insert_ast_edge(
        principal_eng,
        ASTEdgeRecord(
            edge_id=uuid.uuid4(),
            domain_id=domain_id,
            source_symbol_id=focus_sym.symbol_id,
            target_qualified_name="Order",
            target_symbol_id=order_sym.symbol_id,
            edge_kind="USES_TYPE",
            allowed_scopes=["public_eng"],
        ),
    )
    repo.insert_ast_edge(
        principal_eng,
        ASTEdgeRecord(
            edge_id=uuid.uuid4(),
            domain_id=domain_id,
            source_symbol_id=focus_sym.symbol_id,
            target_qualified_name="ExecutionReceipt",
            target_symbol_id=receipt_sym.symbol_id,
            edge_kind="USES_TYPE",
            allowed_scopes=["public_eng"],
        ),
    )

    return repo, focus_sym


def test_expander_attaches_uses_type_and_imports(
    expander: ASTDependencyExpander,
    principal_eng: PrincipalContext,
    populated_repo: tuple[InMemoryRLSRepository, ASTSymbolRecord],
) -> None:
    repo, focus_sym = populated_repo
    result: ASTExpansionResult = expander.expand_1hop(
        focus_sym, repo, principal_eng, "trading_code_scout"
    )

    assert len(result.imported_type_slices) == 2
    qual_names = {s.qualified_symbol for s in result.imported_type_slices}
    assert "execution.models.Order" in qual_names
    assert "execution.models.ExecutionReceipt" in qual_names
    for s in result.imported_type_slices:
        assert s.modification_intent == ModificationIntent.IMPORTED_TYPE


def test_expander_enforces_max_3_direct_callers_sec03(
    expander: ASTDependencyExpander,
    principal_eng: PrincipalContext,
    populated_repo: tuple[InMemoryRLSRepository, ASTSymbolRecord],
) -> None:
    repo, focus_sym = populated_repo
    domain_id = "trading_code_scout"
    repo_name = "trading-core"

    # Insert 12 callers across 5 files
    for f_idx in range(5):
        f_path = f"src/callers/caller_file_{f_idx}.py"
        f_id = uuid.uuid4()
        lines = [f"def caller_func_{f_idx}_{i}():\n    route_order(None)\n" for i in range(3)]
        raw = "\n".join(lines)
        repo.insert_code_file(
            principal_eng,
            CodeFileRecord(
                file_id=f_id,
                domain_id=domain_id,
                repo_name=repo_name,
                file_path=f_path,
                git_commit_sha="sha",
                git_blob_sha="blob",
                language="python",
                allowed_scopes=["public_eng"],
                raw_content=raw,
                line_count=len(lines) * 2,
            ),
        )
        for i in range(2):
            c_sym = ASTSymbolRecord(
                symbol_id=uuid.uuid4(),
                file_id=f_id,
                domain_id=domain_id,
                qualified_name=f"callers.f_{f_idx}.caller_{i}",
                symbol_kind="function",
                start_line=i * 2 + 1,
                end_line=i * 2 + 2,
                signature_skeleton=f"def caller_{i}():",
                contextual_summary="",
                allowed_scopes=["public_eng"],
                embedding=[0.0] * 768,
            )
            repo.insert_ast_symbol(principal_eng, c_sym)
            repo.insert_ast_edge(
                principal_eng,
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

    result = expander.expand_1hop(focus_sym, repo, principal_eng, domain_id)

    # Must strictly enforce MAX_DIRECT_CALLERS_PER_SYMBOL = 3
    assert len(result.caller_slices) == 3
    for s in result.caller_slices:
        assert s.modification_intent == ModificationIntent.DIRECT_CALLER


def test_expander_call_site_diversity_ranking(
    expander: ASTDependencyExpander,
    principal_eng: PrincipalContext,
    populated_repo: tuple[InMemoryRLSRepository, ASTSymbolRecord],
) -> None:
    repo, focus_sym = populated_repo
    domain_id = "trading_code_scout"
    repo_name = "trading-core"

    # Create callers: 5 in file A, 1 in file B, 1 in file C
    files = {
        "src/alpha/stat_arb.py": 1,
        "src/risk/pre_trade.py": 1,
        "src/execution/batch_processor.py": 5,
    }

    for f_path, num_callers in files.items():
        f_id = uuid.uuid4()
        lines = [f"def call_fn_{i}():\n    route_order(o)\n" for i in range(num_callers)]
        repo.insert_code_file(
            principal_eng,
            CodeFileRecord(
                file_id=f_id,
                domain_id=domain_id,
                repo_name=repo_name,
                file_path=f_path,
                git_commit_sha="sha",
                git_blob_sha="blob",
                language="python",
                allowed_scopes=["public_eng"],
                raw_content="\n".join(lines),
                line_count=len(lines) * 2,
            ),
        )
        for i in range(num_callers):
            c_sym = ASTSymbolRecord(
                symbol_id=uuid.uuid4(),
                file_id=f_id,
                domain_id=domain_id,
                qualified_name=f"{f_path}.fn_{i}",
                symbol_kind="function",
                start_line=i * 2 + 1,
                end_line=i * 2 + 2,
                signature_skeleton=f"def fn_{i}():",
                contextual_summary="",
                allowed_scopes=["public_eng"],
                embedding=[0.0] * 768,
            )
            repo.insert_ast_symbol(principal_eng, c_sym)
            repo.insert_ast_edge(
                principal_eng,
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

    result = expander.expand_1hop(focus_sym, repo, principal_eng, domain_id)

    assert len(result.caller_slices) == 3
    selected_files = {s.file_path for s in result.caller_slices}
    # Round-robin diversity must pick from all 3 files
    assert len(selected_files) == 3
    assert "src/alpha/stat_arb.py" in selected_files
    assert "src/risk/pre_trade.py" in selected_files
    assert "src/execution/batch_processor.py" in selected_files


def test_expander_caller_snippet_extracts_plus_minus_5_lines(
    expander: ASTDependencyExpander,
    principal_eng: PrincipalContext,
    populated_repo: tuple[InMemoryRLSRepository, ASTSymbolRecord],
) -> None:
    repo, focus_sym = populated_repo
    domain_id = "trading_code_scout"
    repo_name = "trading-core"

    # Create caller with 50 lines, invocation at line 35
    lines = [f"    # line {i}" for i in range(1, 51)]
    lines[0] = "def big_caller_function():"
    lines[34] = "    route_order(order)"  # 1-indexed line 35
    caller_raw = "\n".join(lines) + "\n"

    f_id = uuid.uuid4()
    repo.insert_code_file(
        principal_eng,
        CodeFileRecord(
            file_id=f_id,
            domain_id=domain_id,
            repo_name=repo_name,
            file_path="src/execution/big_caller.py",
            git_commit_sha="sha",
            git_blob_sha="blob",
            language="python",
            allowed_scopes=["public_eng"],
            raw_content=caller_raw,
            line_count=50,
        ),
    )
    c_sym = ASTSymbolRecord(
        symbol_id=uuid.uuid4(),
        file_id=f_id,
        domain_id=domain_id,
        qualified_name="execution.big_caller.big_caller_function",
        symbol_kind="function",
        start_line=1,
        end_line=50,
        signature_skeleton="def big_caller_function():",
        contextual_summary="",
        allowed_scopes=["public_eng"],
        embedding=[0.0] * 768,
    )
    repo.insert_ast_symbol(principal_eng, c_sym)
    repo.insert_ast_edge(
        principal_eng,
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

    result = expander.expand_1hop(focus_sym, repo, principal_eng, domain_id)
    assert len(result.caller_slices) == 1
    caller_slice = result.caller_slices[0]
    # line 35 +- 5 = [30, 40]
    assert caller_slice.start_line == 30
    assert caller_slice.end_line == 40


def test_expander_resolves_tested_by_edge_and_naming_fallback(
    expander: ASTDependencyExpander,
    principal_eng: PrincipalContext,
    populated_repo: tuple[InMemoryRLSRepository, ASTSymbolRecord],
) -> None:
    repo, focus_sym = populated_repo
    domain_id = "trading_code_scout"
    repo_name = "trading-core"

    # Part A: Explicit TESTED_BY edge
    test_file_id = uuid.uuid4()
    test_content = (
        "def test_route_order():\n"
        "    order = Order()\n"
        "    res = route_order(order)\n"
        "    assert res is not None\n"
    )
    repo.insert_code_file(
        principal_eng,
        CodeFileRecord(
            file_id=test_file_id,
            domain_id=domain_id,
            repo_name=repo_name,
            file_path="tests/test_order_router.py",
            git_commit_sha="sha",
            git_blob_sha="blob",
            language="python",
            allowed_scopes=["public_eng"],
            raw_content=test_content,
            line_count=4,
        ),
    )
    test_sym = ASTSymbolRecord(
        symbol_id=uuid.uuid4(),
        file_id=test_file_id,
        domain_id=domain_id,
        qualified_name="tests.test_order_router.test_route_order",
        symbol_kind="function",
        start_line=1,
        end_line=4,
        signature_skeleton="def test_route_order():",
        contextual_summary="Test fixture for order routing",
        allowed_scopes=["public_eng"],
        embedding=[0.0] * 768,
    )
    repo.insert_ast_symbol(principal_eng, test_sym)
    repo.insert_ast_edge(
        principal_eng,
        ASTEdgeRecord(
            edge_id=uuid.uuid4(),
            domain_id=domain_id,
            source_symbol_id=test_sym.symbol_id,
            target_qualified_name=focus_sym.qualified_name,
            target_symbol_id=focus_sym.symbol_id,
            edge_kind="TESTED_BY",
            allowed_scopes=["public_eng"],
        ),
    )

    result_explicit = expander.expand_1hop(focus_sym, repo, principal_eng, domain_id)
    assert len(result_explicit.test_fixture_slices) == 1
    assert result_explicit.test_fixture_slices[0].qualified_symbol == test_sym.qualified_name


def test_expander_enforces_1500_token_budget_cutoff(
    expander: ASTDependencyExpander,
    principal_eng: PrincipalContext,
    populated_repo: tuple[InMemoryRLSRepository, ASTSymbolRecord],
) -> None:
    repo, focus_sym = populated_repo
    domain_id = "trading_code_scout"
    repo_name = "trading-core"

    # Insert a massive caller whose token count exceeds 1500 tokens
    large_lines = [f"    # bulk doc line {i} with long tokens" for i in range(1, 200)]
    large_lines[0] = "def massive_caller_exceeding_budget():"
    large_lines[35] = "    route_order(order)"
    large_content = "\n".join(large_lines)

    f_id = uuid.uuid4()
    repo.insert_code_file(
        principal_eng,
        CodeFileRecord(
            file_id=f_id,
            domain_id=domain_id,
            repo_name=repo_name,
            file_path="src/execution/massive.py",
            git_commit_sha="sha",
            git_blob_sha="blob",
            language="python",
            allowed_scopes=["public_eng"],
            raw_content=large_content,
            line_count=200,
        ),
    )
    c_sym = ASTSymbolRecord(
        symbol_id=uuid.uuid4(),
        file_id=f_id,
        domain_id=domain_id,
        qualified_name="execution.massive.massive_caller_exceeding_budget",
        symbol_kind="function",
        start_line=1,
        end_line=200,
        signature_skeleton="def massive_caller_exceeding_budget():",
        contextual_summary="",
        allowed_scopes=["public_eng"],
        embedding=[0.0] * 768,
    )
    repo.insert_ast_symbol(principal_eng, c_sym)
    repo.insert_ast_edge(
        principal_eng,
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

    # Use small budget of 20 tokens to force cutoff
    small_budget_expander = ASTDependencyExpander(
        max_direct_callers=3, max_expansion_token_budget=20
    )
    result = small_budget_expander.expand_1hop(focus_sym, repo, principal_eng, domain_id)

    assert result.total_expansion_tokens <= 20
    assert result.budget_exhausted is True
    assert result.dropped_candidates_count >= 1


def test_expander_enforces_rls_scope_isolation(
    expander: ASTDependencyExpander,
    principal_eng: PrincipalContext,
    populated_repo: tuple[InMemoryRLSRepository, ASTSymbolRecord],
) -> None:
    repo, focus_sym = populated_repo
    domain_id = "trading_code_scout"
    repo_name = "trading-core"

    quant_principal = PrincipalContext(
        user_id="quant_dev",
        role=RoleType.QUANT_RESEARCHER,
        allowed_scopes=["quant_restricted"],
    )

    # Insert a proprietary quant caller with only quant_restricted scope
    f_id = uuid.uuid4()
    repo.insert_code_file(
        quant_principal,
        CodeFileRecord(
            file_id=f_id,
            domain_id=domain_id,
            repo_name=repo_name,
            file_path="src/alpha/secret_alpha.py",
            git_commit_sha="sha",
            git_blob_sha="blob",
            language="python",
            allowed_scopes=["quant_restricted"],
            raw_content="def secret_alpha(): route_order(None)\n",
            line_count=1,
        ),
    )
    quant_sym = ASTSymbolRecord(
        symbol_id=uuid.uuid4(),
        file_id=f_id,
        domain_id=domain_id,
        qualified_name="alpha.secret_alpha.secret_alpha",
        symbol_kind="function",
        start_line=1,
        end_line=1,
        signature_skeleton="def secret_alpha():",
        contextual_summary="Proprietary secret",
        allowed_scopes=["quant_restricted"],
        embedding=[0.0] * 768,
    )
    repo.insert_ast_symbol(quant_principal, quant_sym)
    repo.insert_ast_edge(
        quant_principal,
        ASTEdgeRecord(
            edge_id=uuid.uuid4(),
            domain_id=domain_id,
            source_symbol_id=quant_sym.symbol_id,
            target_qualified_name=focus_sym.qualified_name,
            target_symbol_id=focus_sym.symbol_id,
            edge_kind="CALLS",
            allowed_scopes=["quant_restricted"],
        ),
    )

    # Expanding under principal_eng (public_eng) must NOT leak secret_alpha!
    res_eng = expander.expand_1hop(focus_sym, repo, principal_eng, domain_id)
    caller_symbols = {s.qualified_symbol for s in res_eng.caller_slices}
    assert "alpha.secret_alpha.secret_alpha" not in caller_symbols

    # Expanding under quant_principal MUST see secret_alpha
    res_quant = expander.expand_1hop(focus_sym, repo, quant_principal, domain_id)
    quant_caller_symbols = {s.qualified_symbol for s in res_quant.caller_slices}
    assert "alpha.secret_alpha.secret_alpha" in quant_caller_symbols
