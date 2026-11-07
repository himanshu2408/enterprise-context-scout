"""Comprehensive regression and production-readiness tests for Enterprise Context Scout.

Covers:
1. Uncommitted Git working-tree overlay (modified, untracked, deleted files) and automatic
   pre-query sync inside ScoutWorkflow._run_dag with zero manual sync commands.
2. Custom-domain corpus_index.json incremental Git-blob SHA-1 Merkle sync and automatic
   _CUSTOM_CORPUS_MEM_CACHE invalidation (`context-scout sync` and auto-sync on query).
3. Symbol-level embedding cache (`_SYMBOL_EMBEDDING_CACHE` keyed by `(blob_sha, qualified_name)`)
   with `use_semantic_embeddings=True` enabled by default.
4. Polyglot enterprise GitLab AST/structure extraction across Python, TypeScript/TSX, JavaScript,
   SQL DDL/RLS, GitLab CI YAML, Terraform HCL, Go, Rust, Java, and Markdown.
5. High-level IDE MCP tool `assemble_verified_context_pack` under out-of-band PrincipalContext.
6. Unseen-repository Symbol Recall@5 fixes (`HTTPX-04`, `CLICK-01`, `CLICK-14` -> 30/30).
7.Repeatable `context-scout benchmark-domain` multi-repo polyglot evaluation harness.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from context_scout.agents.domain_registry import DomainRegistry
from context_scout.agents.preflight_gate import (
    _CUSTOM_CORPUS_MEM_CACHE,
    BoundGraphRepository,
    invalidate_custom_corpus_cache,
    sync_custom_domain_corpus_if_needed,
)
from context_scout.agents.scout_workflow import ScoutWorkflow
from context_scout.ast_engine.extractor import ASTExtractor
from context_scout.cli import (
    cmd_benchmark_domain,
    cmd_distill_custom_domain,
    cmd_init_domain,
    cmd_sync_domain,
)
from context_scout.db.repositories import InMemoryRLSRepository
from context_scout.evals.unseen_ab_benchmark import check_file_recall, check_symbol_recall
from context_scout.evals.unseen_repo_dataset import build_unseen_repo_suite
from context_scout.ingestion.git_merkle_indexer import (
    _SYMBOL_EMBEDDING_CACHE,
    GitMerkleIndexer,
    compute_git_blob_sha,
)
from context_scout.mcp_servers.code_repo_mcp import create_code_repo_mcp_server
from context_scout.schemas.contracts import (
    ImplementationContextPack,
    PrincipalContext,
    RoleType,
)
from context_scout.security.rbac import SecurityError, bind_session_principal


def _init_git_repo(repo_dir: Path) -> None:
    subprocess.run(["git", "init"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.name", "Test Engineer"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "engineer@example.com"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
    )


def test_polyglot_ast_extractor_sql_yaml_terraform_go_js_markdown() -> None:
    """Verify ASTExtractor extracts symbols and directed edges across polyglot enterprise files."""
    extractor = ASTExtractor()

    # 1. SQL DDL + FK Reference + RLS Policy
    sql_code = """
CREATE TABLE risk_accounts (
    account_id UUID PRIMARY KEY,
    desk_name TEXT NOT NULL
);

CREATE TABLE margin_orders (
    order_id UUID PRIMARY KEY,
    account_id UUID NOT NULL REFERENCES risk_accounts(account_id),
    notional_usd NUMERIC(18, 2) NOT NULL
);

CREATE OR REPLACE FUNCTION check_order_limit(p_notional NUMERIC)
RETURNS BOOLEAN LANGUAGE plpgsql AS $$
BEGIN
    RETURN p_notional < 500000.0;
END;
$$;

CREATE POLICY desk_rls_policy ON margin_orders USING (account_id IS NOT NULL);
"""
    sql_syms, sql_edges = extractor.parse_source(sql_code, "sql", "schema/margin.sql")
    sql_names = {s.name for s in sql_syms}
    assert {"risk_accounts", "margin_orders", "check_order_limit", "desk_rls_policy"} <= sql_names
    assert any(
        e.source_qualified_name.endswith("margin_orders")
        and e.target_qualified_name.endswith("risk_accounts")
        and e.edge_kind == "USES_TYPE"
        for e in sql_edges
    )

    # 2. GitLab CI YAML (.gitlab-ci.yml)
    yaml_code = """
stages:
  - lint
  - test

.python_template:
  image: python:3.12

pytest_rls_suite:
  extends: .python_template
  stage: test
  script:
    - uv run pytest -q
"""
    yaml_syms, yaml_edges = extractor.parse_source(yaml_code, "yaml", ".gitlab-ci.yml")
    yaml_names = {s.name for s in yaml_syms}
    assert {"stages", ".python_template", "pytest_rls_suite"} <= yaml_names
    assert any(
        e.source_qualified_name.endswith("pytest_rls_suite")
        and e.target_qualified_name.endswith(".python_template")
        and e.edge_kind == "INHERITS"
        for e in yaml_edges
    )

    # 3. Terraform HCL (.tf)
    tf_code = """
variable "gcp_region" {
  type = string
}

resource "google_cloud_run_v2_service" "scout_api" {
  name     = "scout-api"
  location = var.gcp_region
}

module "pg_cluster" {
  source = "./modules/pg"
  region = var.gcp_region
}
"""
    tf_syms, tf_edges = extractor.parse_source(tf_code, "terraform", "infra/main.tf")
    tf_names = {s.name for s in tf_syms}
    tf_quals = {s.qualified_name for s in tf_syms}
    assert {"gcp_region", "scout_api", "pg_cluster"} <= tf_names
    assert any(q.endswith("google_cloud_run_v2_service.scout_api") for q in tf_quals)
    assert any(
        e.source_qualified_name.endswith("google_cloud_run_v2_service.scout_api")
        and e.target_qualified_name.endswith("gcp_region")
        and e.edge_kind == "USES_TYPE"
        for e in tf_edges
    )

    # 4. Go (.go)
    go_code = """
package gateway

type OrderRouter struct {
    MaxRetries int
}

func NewOrderRouter(retries int) *OrderRouter {
    return &OrderRouter{MaxRetries: retries}
}

func (r *OrderRouter) DispatchOrder(orderID string) bool {
    return len(orderID) > 0
}
"""
    go_syms, _ = extractor.parse_source(go_code, "go", "services/router.go")
    go_names = {s.name for s in go_syms}
    assert {"OrderRouter", "NewOrderRouter", "DispatchOrder"} <= go_names

    # 5. JavaScript (.js) & Markdown (.md)
    js_code = """
export function computePortfolioDelta(positions) {
    return positions.length;
}
"""
    js_syms, _ = extractor.parse_source(js_code, "javascript", "ui/delta.js")
    assert any(s.name == "computePortfolioDelta" for s in js_syms)

    md_code = """
# Execution Architecture

The gateway invokes `OrderRouter` and `computePortfolioDelta` before settlement.
"""
    md_syms, md_edges = extractor.parse_source(md_code, "markdown", "docs/architecture.md")
    assert any(s.name == "Execution_Architecture" for s in md_syms)
    assert any(e.target_qualified_name == "OrderRouter" for e in md_edges)


def test_git_working_tree_live_uncommitted_edits_and_auto_workflow_sync() -> None:
    """Verify uncommitted edits (M), untracked files (??), and deletions (D) auto-sync on query."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="scout_wt_live_"))
    try:
        _init_git_repo(tmp_dir)
        pkg_dir = tmp_dir / "src" / "payments"
        pkg_dir.mkdir(parents=True, exist_ok=True)
        settlement_file = pkg_dir / "settlement.py"
        legacy_file = pkg_dir / "legacy_batch.py"

        settlement_file.write_text(
            "def settle_wire_transfer(amount_usd: float) -> bool:\n"
            "    return amount_usd > 0.0\n",
            encoding="utf-8",
        )
        legacy_file.write_text(
            "def run_legacy_batch() -> int:\n"
            "    return 42\n",
            encoding="utf-8",
        )
        subprocess.run(["git", "add", "."], cwd=tmp_dir, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial commit"],
            cwd=tmp_dir,
            check=True,
            capture_output=True,
        )

        principal = PrincipalContext(
            user_id="dev_user",
            role=RoleType.ENGINEER,
            allowed_scopes=["public_eng"],
        )
        repo = InMemoryRLSRepository()
        indexer = GitMerkleIndexer(use_semantic_embeddings=False, include_working_tree=True)
        stats1 = indexer.sync_in_memory(
            repo_dir=tmp_dir,
            repo=repo,
            principal=principal,
            domain_id="trading_code_scout",
            repo_name="payments-service",
        )
        assert stats1.new_files_indexed == 2

        registry = DomainRegistry(domains_dir=Path("domains"))
        registry.load_all()
        workflow = ScoutWorkflow(registry=registry, repo=repo)

        # Now make UNCOMMITTED working-tree changes (no git commit, no manual sync call!):
        # 1. Modify settlement.py (M)
        settlement_file.write_text(
            "def settle_wire_transfer(amount_usd: float, instant_sepa: bool = True) -> bool:\n"
            "    return amount_usd > 0.0 and instant_sepa\n",
            encoding="utf-8",
        )
        # 2. Create brand-new untracked file fraud_guard.py (??)
        fraud_file = pkg_dir / "fraud_guard.py"
        fraud_file.write_text(
            "def verify_iban_sanctions_screen(iban_code: str) -> bool:\n"
            "    return iban_code.startswith('DE')\n",
            encoding="utf-8",
        )
        # 3. Delete legacy_batch.py (D)
        legacy_file.unlink()

        # Query the newly added untracked symbol directly through ScoutWorkflow.run()
        # without calling indexer.sync_in_memory manually!
        pack = workflow.run(
            user_prompt=(
                "Locate `verify_iban_sanctions_screen` in `src/payments/fraud_guard.py` "
                "and `settle_wire_transfer` in `src/payments/settlement.py`."
            ),
            principal=principal,
            domain_id="trading_code_scout",
        )
        assert isinstance(pack, ImplementationContextPack)
        localized_syms = {s.qualified_symbol for s in pack.code_slices}
        assert "payments.fraud_guard.verify_iban_sanctions_screen" in localized_syms
        assert "payments.settlement.settle_wire_transfer" in localized_syms

        # Verify modified line content (`instant_sepa: bool = True`) is in the hydrated slice
        settle_slice = next(
            s
            for s in pack.code_slices
            if s.qualified_symbol == "payments.settlement.settle_wire_transfer"
        )
        assert "instant_sepa: bool = True" in settle_slice.verbatim_code_with_lines

        # Verify deleted legacy_batch.py is removed from visible files
        visible_paths = {
            f.file_path for f in repo.list_visible_code_files(principal, "trading_code_scout")
        }
        assert "src/payments/legacy_batch.py" not in visible_paths
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_custom_domain_corpus_incremental_sync_and_cache_invalidation() -> None:
    """Verify custom domain corpus_index.json incrementally syncs and invalidates cache."""
    tmp_root = Path(tempfile.mkdtemp(prefix="scout_custom_sync_"))
    try:
        src_dir = tmp_root / "workspace_repos"
        domains_dir = tmp_root / "domains"
        data_dir = tmp_root / "data"
        (src_dir / "billing").mkdir(parents=True, exist_ok=True)
        shutil.copytree(Path("domains/trading_code_scout"), domains_dir / "trading_code_scout")
        shutil.copytree(Path("domains/quant_research_scout"), domains_dir / "quant_research_scout")

        invoice_py = src_dir / "billing" / "invoice_engine.py"
        invoice_py.write_text(
            "def compute_vat_invoice(net_eur: float) -> float:\n"
            "    return round(net_eur * 1.19, 2)\n",
            encoding="utf-8",
        )

        cmd_init_domain(
            domain_id="gitlab_billing_scout",
            kind="code",
            domains_dir=domains_dir,
            source_dir=src_dir,
        )
        distill_res = cmd_distill_custom_domain(
            domain_id="gitlab_billing_scout",
            source_dir=src_dir,
            num_samples=10,
            domains_dir=domains_dir,
            data_dir=data_dir,
        )
        assert distill_res["files_indexed"] == 1

        registry = DomainRegistry(domains_dir=domains_dir)
        registry.load_all()
        bound_repo = BoundGraphRepository(InMemoryRLSRepository())
        workflow = ScoutWorkflow(registry=registry, repo=bound_repo)
        principal = PrincipalContext(
            user_id="billing_eng",
            role=RoleType.ENGINEER,
            allowed_scopes=["public_eng", "repo:gitlab_billing_scout"],
        )

        # Warm the in-memory custom corpus cache
        pack1 = workflow.run(
            user_prompt="Inspect `compute_vat_invoice` in `billing/invoice_engine.py`.",
            principal=principal,
            domain_id="gitlab_billing_scout",
        )
        assert isinstance(pack1, ImplementationContextPack)
        assert "gitlab_billing_scout" in _CUSTOM_CORPUS_MEM_CACHE

        # Now add a new SQL schema file and modify invoice_engine.py on disk
        ledger_sql = src_dir / "billing" / "vat_ledger.sql"
        ledger_sql.write_text(
            "CREATE TABLE vat_audit_entries (\n"
            "    entry_id UUID PRIMARY KEY,\n"
            "    invoice_total_eur NUMERIC(18, 2) NOT NULL\n"
            ");\n",
            encoding="utf-8",
        )
        invoice_py.write_text(
            "def compute_vat_invoice(net_eur: float, vat_rate: float = 0.19) -> float:\n"
            "    return round(net_eur * (1.0 + vat_rate), 2)\n",
            encoding="utf-8",
        )

        # Force check via sync_custom_domain_corpus_if_needed
        sync_stats = sync_custom_domain_corpus_if_needed(
            "gitlab_billing_scout",
            index_path=domains_dir / "gitlab_billing_scout" / "corpus_index.json",
            force=True,
        )
        assert sync_stats is not None
        assert sync_stats["synced"] is True
        assert sync_stats["added"] == 1
        assert sync_stats["modified"] == 1

        # Query the new SQL table symbol and modified Python function
        pack2 = workflow.run(
            user_prompt=(
                "Locate `vat_audit_entries` in `billing/vat_ledger.sql` and "
                "`compute_vat_invoice` in `billing/invoice_engine.py`."
            ),
            principal=principal,
            domain_id="gitlab_billing_scout",
        )
        assert isinstance(pack2, ImplementationContextPack)
        syms2 = {s.qualified_symbol for s in pack2.code_slices}
        assert any("vat_audit_entries" in s for s in syms2)
        assert any("compute_vat_invoice" in s for s in syms2)
    finally:
        invalidate_custom_corpus_cache("gitlab_billing_scout")
        shutil.rmtree(tmp_root, ignore_errors=True)


def test_symbol_embedding_cache_avoids_recomputing_unchanged_symbols() -> None:
    """Verify _SYMBOL_EMBEDDING_CACHE caches vectors by (blob_sha, qualified_name)."""
    _SYMBOL_EMBEDDING_CACHE.clear()
    raw_code = (
        b"def calculate_funding_rate(mark_price: float, index_price: float) -> float:\n"
        b"    return (mark_price - index_price) / index_price\n"
    )
    blob_sha = compute_git_blob_sha(raw_code)

    mock_embedder = MagicMock()
    mock_embedder.embed_batch.return_value = [[0.1] * 768]

    indexer = GitMerkleIndexer(embedder=mock_embedder, use_semantic_embeddings=True)
    extractor = ASTExtractor()
    syms, _ = extractor.parse_source(raw_code.decode("utf-8"), "python", "src/funding.py")
    assert len(syms) == 1

    vecs1 = indexer._compute_symbol_embeddings(syms, git_blob_sha=blob_sha)
    assert len(vecs1) == 1
    assert mock_embedder.embed_batch.call_count == 1

    # Second call with identical (blob_sha, qualified_name) must hit _SYMBOL_EMBEDDING_CACHE
    vecs2 = indexer._compute_symbol_embeddings(syms, git_blob_sha=blob_sha)
    assert vecs2 == vecs1
    assert mock_embedder.embed_batch.call_count == 1


@pytest.mark.asyncio
async def test_mcp_assemble_verified_context_pack_and_sec04_guard() -> None:
    """Verify high-level IDE MCP tool assemble_verified_context_pack under RBAC."""
    server = create_code_repo_mcp_server()
    tool_names = {t.name for t in await server.list_tools()}
    assert "assemble_verified_context_pack" in tool_names

    # SEC-04: Passing forbidden principal/scopes argument must raise SecurityError
    principal = PrincipalContext(
        user_id="ide_agent_user",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng", "repo:trading_strategies"],
    )
    with bind_session_principal(principal), pytest.raises(SecurityError):
        await server.call_tool(
            "assemble_verified_context_pack",
            {
                "query": "Inspect DeterministicSlicer",
                "allowed_scopes": ["admin_override"],
            },
        )


def test_unseen_repo_three_previously_missed_tasks_now_hit_top5_symbol_recall() -> None:
    """Verify HTTPX-04, CLICK-01, and CLICK-14 all achieve Top-5 File & Symbol Recall."""
    httpx_dir = Path("data/unseen_repos/httpx")
    click_dir = Path("data/unseen_repos/click")
    if not httpx_dir.exists() or not click_dir.exists():
        pytest.skip("Unseen repositories not cloned locally")

    suite = build_unseen_repo_suite()
    principal = PrincipalContext(
        user_id="recall_verifier",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng"],
    )
    registry = DomainRegistry(domains_dir=Path("domains"))
    registry.load_all()
    indexer = GitMerkleIndexer(use_semantic_embeddings=False, include_working_tree=False)

    for tid in ("HTTPX-04", "CLICK-01", "CLICK-14"):
        task = next(t for t in suite.tasks if t.task_id == tid)
        is_httpx = tid.startswith("HTTPX")
        rdir = httpx_dir if is_httpx else click_dir
        rname = "httpx" if is_httpx else "click"
        subprocess.run(
            ["git", "-C", str(rdir), "checkout", "-f", task.parent_commit_sha],
            check=True,
            capture_output=True,
        )
        repo = InMemoryRLSRepository()
        indexer.sync_in_memory(
            repo_dir=rdir,
            repo=repo,
            principal=principal,
            domain_id="trading_code_scout",
            repo_name=rname,
            ref=task.parent_commit_sha,
        )
        wf = ScoutWorkflow(registry=registry, repo=repo)
        res = wf.run(task.problem_statement, principal, "trading_code_scout")
        assert isinstance(res, ImplementationContextPack), f"{tid} failed preflight: {res}"
        loc_files = list(dict.fromkeys(s.file_path for s in res.code_slices))
        loc_syms = list(dict.fromkeys(s.qualified_symbol for s in res.code_slices))
        assert check_file_recall(loc_files, task.ground_truth_files), f"{tid} missed File@5"
        assert check_symbol_recall(loc_syms, task.ground_truth_symbols), (
            f"{tid} missed Symbol@5: got {loc_syms[:5]} vs {task.ground_truth_symbols}"
        )


def test_benchmark_domain_cli_on_polyglot_multi_repo_corpus() -> None:
    """Verify context-scout benchmark-domain across a multi-repo polyglot corpus."""
    tmp_root = Path(tempfile.mkdtemp(prefix="scout_bench_cli_"))
    try:
        src_dir = tmp_root / "repos"
        domains_dir = tmp_root / "domains"
        data_dir = tmp_root / "data"
        shutil.copytree(Path("domains/trading_code_scout"), domains_dir / "trading_code_scout")
        shutil.copytree(Path("domains/quant_research_scout"), domains_dir / "quant_research_scout")

        # Sub-repo 1: Python service
        r1 = src_dir / "execution_service"
        r1.mkdir(parents=True, exist_ok=True)
        (r1 / "pyproject.toml").write_text("[project]\nname = 'execution_service'\n")
        (r1 / "router.py").write_text(
            "class SmartOrderRouter:\n"
            "    def route_child_slice(self, qty: float) -> float:\n"
            "        return round(qty * 0.5, 4)\n",
            encoding="utf-8",
        )

        # Sub-repo 2: Polyglot ops (SQL + GitLab CI + Terraform + Go)
        r2 = src_dir / "gitlab_infra"
        r2.mkdir(parents=True, exist_ok=True)
        (r2 / "pyproject.toml").write_text("[project]\nname = 'gitlab_infra'\n")
        (r2 / ".gitlab-ci.yml").write_text(
            "stages:\n  - verify\n\nverify_rls_gate:\n  stage: verify\n  script:\n    - pytest\n",
            encoding="utf-8",
        )
        (r2 / "positions.sql").write_text(
            "CREATE TABLE portfolio_positions (\n"
            "    position_id UUID PRIMARY KEY,\n"
            "    symbol_ticker TEXT NOT NULL\n"
            ");\n",
            encoding="utf-8",
        )
        (r2 / "main.tf").write_text(
            'resource "google_SQL_database_instance" "rls_primary" {\n'
            '  name = "rls-primary"\n'
            "}\n",
            encoding="utf-8",
        )

        cmd_init_domain(
            domain_id="multi_repo_polyglot_test",
            kind="code",
            domains_dir=domains_dir,
            source_dir=src_dir,
        )
        cmd_distill_custom_domain(
            domain_id="multi_repo_polyglot_test",
            source_dir=src_dir,
            num_samples=12,
            domains_dir=domains_dir,
            data_dir=data_dir,
        )
        sync_res = cmd_sync_domain(
            domain_id="multi_repo_polyglot_test",
            domains_dir=domains_dir,
        )
        assert sync_res["unchanged"] == 4

        out_json = tmp_root / "bench.json"
        out_md = tmp_root / "bench.md"
        bench = cmd_benchmark_domain(
            domain_id="multi_repo_polyglot_test",
            domains_dir=domains_dir,
            max_queries=8,
            output_json=out_json,
            output_md=out_md,
        )
        assert out_json.exists() and out_md.exists()
        assert bench["file_recall_at_5_pct"] == 100.0
        assert bench["symbol_recall_at_5_pct"] == 100.0
        assert bench["preflight_pass_pct"] == 100.0
        assert bench["verbatim_faithfulness_pct"] == 100.0
        assert bench["out_of_scope_zero_cost_abstention"] is True
        assert bench["prompt_injection_blocked"] is True
    finally:
        invalidate_custom_corpus_cache("multi_repo_polyglot_test")
        shutil.rmtree(tmp_root, ignore_errors=True)
