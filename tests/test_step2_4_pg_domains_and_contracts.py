"""Step 2.4 verification: 100% PostgreSQL 17 custom domains & CrossBoundaryContractLinker."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from uuid import uuid4

from psycopg import connect

from context_scout.agents import preflight_gate
from context_scout.agents.domain_registry import DomainRegistry
from context_scout.agents.preflight_gate import BoundGraphRepository
from context_scout.agents.scout_workflow import ScoutWorkflow
from context_scout.ast_engine.contract_linker import (
    CrossBoundaryContractLinker,
    normalize_route_template,
)
from context_scout.ast_engine.dependency_expander import ASTDependencyExpander
from context_scout.cli import (
    cmd_distill_custom_domain,
    cmd_init_domain,
    cmd_query_domain,
    cmd_sync_domain,
)
from context_scout.config import ScoutSettings
from context_scout.db.repositories import PostgresRLSRepository
from context_scout.db.schema import (
    ASTEdgeRecord,
    ASTSymbolRecord,
    CodeFileRecord,
)
from context_scout.schemas.contracts import (
    ImplementationContextPack,
    PrincipalContext,
    RoleType,
)

_EVAL_CLEANUP_TABLES: tuple[str, ...] = (
    "ast_edges",
    "ast_symbols",
    "code_files",
    "repositories",
    "confluence_pages",
    "jira_issues",
    "doc_chunks",
    "indexing_queue",
    "indexing_dead_letter",
    "execution_telemetry",
    "domain_packs",
)


def _cleanup_eval_domains(*domain_ids: str) -> None:
    if not domain_ids:
        return
    ids = list(domain_ids)
    with connect(ScoutSettings().db_admin_url) as conn:
        for tbl in _EVAL_CLEANUP_TABLES:
            conn.execute(f"DELETE FROM {tbl} WHERE domain_id = ANY(%s)", (ids,))
        conn.commit()


def _copy_builtin_domains(domains_dir: Path) -> None:
    for dname in ("trading_code_scout", "quant_research_scout"):
        src = Path("domains") / dname
        if src.exists():
            shutil.copytree(src, domains_dir / dname, dirs_exist_ok=True)


def test_custom_domain_pg17_lifecycle_zero_corpus_index(tmp_path: Path) -> None:
    """Init -> distill -> sync -> query uses PG17 + sync_manifest.json, never corpus_index.json."""
    for purged in (
        "_CUSTOM_CORPUS_MEM_CACHE",
        "_CUSTOM_CORPUS_PARSED_CACHE",
        "_load_custom_domain_corpus",
    ):
        assert not hasattr(preflight_gate, purged), f"{purged} must stay removed"

    domain_id = f"eval_s24_life_{uuid4().hex[:8]}"
    domains_dir = tmp_path / "domains"
    data_dir = tmp_path / "tuning"
    src_dir = tmp_path / "workspace"
    (src_dir / "billing").mkdir(parents=True, exist_ok=True)
    (src_dir / "docs").mkdir(parents=True, exist_ok=True)
    _copy_builtin_domains(domains_dir)

    ledger_py = src_dir / "billing" / "ledger.py"
    ledger_py.write_text(
        "class InvoiceLedger:\n"
        '    """Computes settled EUR invoice totals."""\n\n'
        "    def compute_settlement(self, subtotal_eur: float, tax_rate: float = 0.19) -> float:\n"
        "        return round(subtotal_eur * (1.0 + tax_rate), 2)\n",
        encoding="utf-8",
    )
    (src_dir / "docs" / "settlement_adr.md").write_text(
        "# Billing Settlement Architecture\n\n"
        "## Tax Rounding Rule\n\n"
        "Settlement totals apply statutory VAT rounding to two decimal places.\n",
        encoding="utf-8",
    )

    try:
        cmd_init_domain(
            domain_id=domain_id,
            kind="code",
            display_name="Billing PG17 Verification Scout",
            domains_dir=domains_dir,
            source_dir=src_dir,
        )
        distill_res = cmd_distill_custom_domain(
            domain_id=domain_id,
            source_dir=src_dir,
            num_samples=8,
            domains_dir=domains_dir,
            data_dir=data_dir,
        )
        assert distill_res["files_indexed"] >= 1
        assert distill_res["symbols_indexed"] >= 1

        dom_dir = domains_dir / domain_id
        manifest_path = dom_dir / "sync_manifest.json"
        assert manifest_path.exists()
        assert not (dom_dir / "corpus_index.json").exists()
        assert list(dom_dir.rglob("corpus_index.json")) == []

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["domain_id"] == domain_id
        assert manifest["storage_backend"] == "postgresql_17_rls"
        assert manifest["code_files_indexed"] >= 1
        assert manifest["ast_symbols_indexed"] >= 1

        # Verify rows exist in live PostgreSQL 17 with valid CHECK-compliant market_scope
        with connect(ScoutSettings().db_admin_url) as conn:
            cf_cnt_row = conn.execute(
                "SELECT COUNT(*) FROM code_files WHERE domain_id = %s", (domain_id,)
            ).fetchone()
            sym_cnt_row = conn.execute(
                "SELECT COUNT(*) FROM ast_symbols WHERE domain_id = %s", (domain_id,)
            ).fetchone()
            doc_scopes = conn.execute(
                "SELECT DISTINCT market_scope FROM doc_chunks WHERE domain_id = %s",
                (domain_id,),
            ).fetchall()
        assert int(str((cf_cnt_row or (0,))[0])) >= 1
        assert int(str((sym_cnt_row or (0,))[0])) >= 1
        for row in doc_scopes:
            assert row[0] in (None, "ENTERPRISE_DOC")

        # Sync detects new function on disk without generating corpus_index.json
        (src_dir / "billing" / "refunds.py").write_text(
            "def issue_credit_note(amount_eur: float) -> float:\n"
            "    return round(-abs(amount_eur), 2)\n",
            encoding="utf-8",
        )
        sync_res = cmd_sync_domain(domain_id=domain_id, domains_dir=domains_dir)
        assert sync_res["synced"] is True
        assert sync_res["added"] == 1
        assert not (dom_dir / "corpus_index.json").exists()

        # Query via CLI directly against PG17
        payload = cmd_query_domain(
            domain_id=domain_id,
            prompt="Inspect `InvoiceLedger.compute_settlement` and `issue_credit_note`.",
            domains_dir=domains_dir,
        )
        assert payload["result_type"] == "ImplementationContextPack"
        assert payload["preflight_report"]["passed"] is True
        assert any("compute_settlement" in s["qualified_symbol"] for s in payload["code_slices"])

        # Direct ScoutWorkflow with BoundGraphRepository(None) queries PG17 cleanly
        registry = DomainRegistry(domains_dir=domains_dir)
        registry.load_all()
        wf = ScoutWorkflow(registry=registry, repo=BoundGraphRepository(None))
        principal = PrincipalContext(
            user_id="pg_eval_user",
            role=RoleType.ENGINEER,
            allowed_scopes=["public_eng", f"domain:{domain_id}"],
        )
        pack = wf.run(
            user_prompt="Verify `issue_credit_note` in `billing/refunds.py`.",
            principal=principal,
            domain_id=domain_id,
        )
        assert isinstance(pack, ImplementationContextPack)
        assert pack.preflight_report.passed is True
        assert any("issue_credit_note" in s.qualified_symbol for s in pack.code_slices)
    finally:
        _cleanup_eval_domains(domain_id)


def test_custom_domain_pg17_force_rls_tenant_isolation(tmp_path: Path) -> None:
    """Unauthorized principals see zero rows on custom domains under PG17 FORCE RLS."""
    code_domain = f"eval_s24_rlsc_{uuid4().hex[:8]}"
    doc_domain = f"eval_s24_rlsd_{uuid4().hex[:8]}"
    domains_dir = tmp_path / "domains"
    data_dir = tmp_path / "tuning"
    code_src = tmp_path / "code_src" / "orders_svc"
    doc_src = tmp_path / "doc_src"
    code_src.mkdir(parents=True, exist_ok=True)
    doc_src.mkdir(parents=True, exist_ok=True)
    _copy_builtin_domains(domains_dir)

    (code_src / "Orders.cs").write_text(
        "namespace OrdersSvc;\n"
        "public sealed class OrdersService\n"
        "{\n"
        '    public string GetStatus(string orderId) => "Confirmed";\n'
        "}\n",
        encoding="utf-8",
    )
    (doc_src / "runbook.md").write_text(
        "# Payment Gateway Runbook\n\n"
        "## Failover Procedure\n\n"
        "Escalate gateway timeouts over 1500ms to the secondary clearing endpoint.\n",
        encoding="utf-8",
    )

    try:
        cmd_init_domain(
            domain_id=code_domain,
            kind="code",
            domains_dir=domains_dir,
            source_dir=code_src.parent,
        )
        cmd_distill_custom_domain(
            domain_id=code_domain,
            source_dir=code_src.parent,
            num_samples=6,
            domains_dir=domains_dir,
            data_dir=data_dir,
        )

        cmd_init_domain(
            domain_id=doc_domain,
            kind="research",
            domains_dir=domains_dir,
            source_dir=doc_src,
        )
        cmd_distill_custom_domain(
            domain_id=doc_domain,
            source_dir=doc_src,
            num_samples=6,
            domains_dir=domains_dir,
            data_dir=data_dir,
        )

        bound = BoundGraphRepository(PostgresRLSRepository())
        authorized = PrincipalContext(
            user_id="auth_eng",
            role=RoleType.ENGINEER,
            allowed_scopes=["public_eng", "public_quant", f"domain:{code_domain}"],
        )
        unauthorized = PrincipalContext(
            user_id="intruder_eng",
            role=RoleType.ENGINEER,
            allowed_scopes=["unrelated_tenant_scope_only"],
        )

        assert len(bound.list_visible_code_files(authorized, code_domain)) >= 1
        assert len(bound.list_visible_ast_symbols(authorized, code_domain)) >= 1
        assert len(bound.list_visible_doc_chunks(authorized, doc_domain)) >= 1

        # FORCE RLS zero-leak negative control
        assert bound.list_visible_code_files(unauthorized, code_domain) == []
        assert bound.list_visible_ast_symbols(unauthorized, code_domain) == []
        assert bound.list_visible_ast_edges(unauthorized, code_domain) == []
        assert bound.list_visible_doc_chunks(unauthorized, doc_domain) == []
    finally:
        _cleanup_eval_domains(code_domain, doc_domain)


def test_polyglot_cross_boundary_contracts_and_pg17_1hop_expansion(tmp_path: Path) -> None:
    """Links C# API/DTO <-> RTK Query/TS DTO <-> Storybook <-> Terraform in live PG17."""
    assert normalize_route_template("/api/v1/orders/{orderId}") == "/api/v1/orders/{param}"
    assert normalize_route_template("/api/v1/orders/${orderId}?expand=true") == (
        "/api/v1/orders/{param}"
    )
    assert normalize_route_template("/api/v1/orders/:id") == "/api/v1/orders/{param}"

    domain_id = f"eval_s24_poly_{uuid4().hex[:8]}"
    domains_dir = tmp_path / "domains"
    data_dir = tmp_path / "tuning"
    ws = tmp_path / "polyglot_ws"
    (ws / "backend_api").mkdir(parents=True, exist_ok=True)
    (ws / "frontend_web" / "src").mkdir(parents=True, exist_ok=True)
    (ws / "design_system" / "src").mkdir(parents=True, exist_ok=True)
    (ws / "infra_tf").mkdir(parents=True, exist_ok=True)
    _copy_builtin_domains(domains_dir)

    (ws / "backend_api" / "OrderDtos.cs").write_text(
        "namespace BackendApi.Contracts;\n\n"
        "public record OrderDto(string OrderId, decimal TotalEur);\n",
        encoding="utf-8",
    )
    (ws / "backend_api" / "OrdersController.cs").write_text(
        "using Microsoft.AspNetCore.Mvc;\n"
        "using BackendApi.Contracts;\n\n"
        "namespace BackendApi.Controllers;\n\n"
        '[Route("api/v1/orders")]\n'
        "public class OrdersController\n"
        "{\n"
        '    [HttpGet("{orderId}")]\n'
        "    public OrderDto GetOrder(string orderId)\n"
        "    {\n"
        '        var key = Environment.GetEnvironmentVariable("STRIPE_SECRET_KEY");\n'
        "        return new OrderDto(orderId, 42.0m);\n"
        "    }\n\n"
        '    [HttpPost("")]\n'
        "    public OrderDto CreateOrder(OrderDto payload)\n"
        "    {\n"
        "        return payload;\n"
        "    }\n"
        "}\n",
        encoding="utf-8",
    )
    (ws / "frontend_web" / "src" / "orderTypes.ts").write_text(
        "export interface OrderDto {\n  orderId: string;\n  totalEur: number;\n}\n",
        encoding="utf-8",
    )
    (ws / "frontend_web" / "src" / "OrderSummary.tsx").write_text(
        "import type { OrderDto } from './orderTypes';\n\n"
        "export function OrderSummary(props: { order: OrderDto }) {\n"
        "  const apiUrl = import.meta.env.VITE_API_URL;\n"
        "  return `${apiUrl}:${props.order.orderId}`;\n"
        "}\n",
        encoding="utf-8",
    )
    (ws / "frontend_web" / "src" / "ordersApi.ts").write_text(
        "import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react';\n"
        "import type { OrderDto } from './orderTypes';\n\n"
        "export const ordersApi = createApi({\n"
        "  reducerPath: 'ordersApi',\n"
        "  baseQuery: fetchBaseQuery({ baseUrl: '/api/v1' }),\n"
        "  endpoints: (builder) => ({\n"
        "    getOrder: builder.query<OrderDto, string>({\n"
        "      query: (orderId) => `/orders/${orderId}`,\n"
        "    }),\n"
        "    createOrder: builder.mutation<OrderDto, OrderDto>({\n"
        "      query: (body) => ({ url: '/orders', method: 'POST', body }),\n"
        "    }),\n"
        "  }),\n"
        "});\n",
        encoding="utf-8",
    )
    (ws / "design_system" / "src" / "OrderCard.stories.tsx").write_text(
        "import type { Meta, StoryObj } from '@storybook/react';\n"
        "import { OrderSummary } from '../../frontend_web/src/OrderSummary';\n\n"
        "const meta: Meta<typeof OrderSummary> = {\n"
        "  title: 'Orders/OrderSummary',\n"
        "  component: OrderSummary,\n"
        "};\n"
        "export default meta;\n"
        "export const Default: StoryObj<typeof OrderSummary> = {};\n",
        encoding="utf-8",
    )
    (ws / "infra_tf" / "main.tf").write_text(
        'variable "STRIPE_SECRET_KEY" {\n'
        "  type = string\n"
        "}\n\n"
        'resource "google_cloud_run_v2_service" "backend_api" {\n'
        '  name = "backend-api"\n'
        "  template {\n"
        "    containers {\n"
        "      env {\n"
        '        name = "STRIPE_SECRET_KEY"\n'
        "      }\n"
        "    }\n"
        "  }\n"
        "}\n",
        encoding="utf-8",
    )

    try:
        cmd_init_domain(
            domain_id=domain_id,
            kind="code",
            domains_dir=domains_dir,
            source_dir=ws,
        )
        distill_res = cmd_distill_custom_domain(
            domain_id=domain_id,
            source_dir=ws,
            num_samples=12,
            domains_dir=domains_dir,
            data_dir=data_dir,
        )
        assert distill_res["files_indexed"] >= 7

        bound = BoundGraphRepository(PostgresRLSRepository())
        principal = PrincipalContext(
            user_id="poly_eng",
            role=RoleType.ENGINEER,
            allowed_scopes=[
                "public_eng",
                f"domain:{domain_id}",
                "repo:backend_api",
                "repo:frontend_web",
                "repo:design_system",
                "repo:infra_tf",
            ],
        )

        files = bound.list_visible_code_files(principal, domain_id)
        symbols = bound.list_visible_ast_symbols(principal, domain_id)
        edges = bound.list_visible_ast_edges(principal, domain_id)

        file_repo_by_id = {f.file_id: f.repo_name for f in files}
        sym_by_id = {s.symbol_id: s for s in symbols}

        cross_edges = [e for e in edges if e.is_cross_repo]
        cross_kinds = {e.edge_kind for e in cross_edges}
        assert "CALLS" in cross_kinds, f"Missing cross-repo REST CALLS edge in {cross_edges}"
        assert "USES_TYPE" in cross_kinds, f"Missing cross-repo DTO USES_TYPE edge in {cross_edges}"
        assert "DOCUMENTS_COMPONENT" in cross_kinds, (
            f"Missing Storybook DOCUMENTS_COMPONENT edge in {cross_edges}"
        )
        assert "PROVISIONED_BY" in cross_kinds, (
            f"Missing Terraform PROVISIONED_BY edge in {cross_edges}"
        )

        # Verify 1-hop expansion from RTK Query getOrder pulls C# GetOrder across repo boundary
        get_order_fe = next(
            s
            for s in symbols
            if file_repo_by_id.get(s.file_id) == "frontend_web" and "getOrder" in s.qualified_name
        )
        expander = ASTDependencyExpander()
        expanded_mem = expander.expand_1hop(get_order_fe, bound, principal, domain_id)
        neighbor_repos = {n.repo_name for n in expanded_mem.all_slices}
        assert "backend_api" in neighbor_repos, (
            f"Expected backend_api in 1-hop cross-repo neighbors, got {expanded_mem}"
        )

        # Verify direct PG17 SQL traverse_1hop_neighborhood also finds backend_api neighbor
        pg_neighbors = bound.traverse_1hop_neighborhood(
            principal, domain_id, get_order_fe.symbol_id
        )
        assert any(n.file.repo_name == "backend_api" for n in pg_neighbors), (
            f"Expected PG17 SQL 1-hop neighbor in backend_api, got {pg_neighbors}"
        )

        # Verify Storybook -> OrderSummary expansion across design_system -> frontend_web
        story_sym = next(
            s
            for s in symbols
            if file_repo_by_id.get(s.file_id) == "design_system"
            and "OrderCard" in (sym_by_id[s.symbol_id].qualified_name)
        )
        pg_story_neighbors = bound.traverse_1hop_neighborhood(
            principal, domain_id, story_sym.symbol_id
        )
        assert any(
            n.file.repo_name == "frontend_web" and n.edge_kind == "DOCUMENTS_COMPONENT"
            for n in pg_story_neighbors
        )

        # Negative control: unmatched REST route produces ZERO CALLS cross-repo edges
        unmatched_fe = ASTSymbolRecord(
            symbol_id=uuid4(),
            file_id=next(f.file_id for f in files if f.repo_name == "frontend_web"),
            domain_id=domain_id,
            qualified_name="ordersApi.unmatchedCall",
            symbol_kind="hook",
            start_line=1,
            end_line=3,
            signature_skeleton="unmatchedCall",
            contextual_summary="@route DELETE /api/v9/nonexistent/{param}",
            allowed_scopes=["public_eng"],
            embedding=[],
        )
        neg_edges = CrossBoundaryContractLinker().link_contracts(
            domain_id=domain_id,
            symbols=[
                unmatched_fe,
                *[s for s in symbols if file_repo_by_id[s.file_id] == "backend_api"],
            ],
            code_files=files,
        )
        assert [e for e in neg_edges if e.edge_kind == "CALLS"] == []
    finally:
        _cleanup_eval_domains(domain_id)


def test_guard22_same_name_symbol_isolation_and_negative_control() -> None:
    """Colliding `Task` symbol never bleeds when is_cross_repo=False; crosses when True."""
    domain_id = f"eval_s24_g22_{uuid4().hex[:8]}"
    principal = PrincipalContext(
        user_id="g22_tester",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng", f"domain:{domain_id}", "repo:repo_alpha", "repo:repo_beta"],
    )

    f_alpha_id = uuid4()
    f_beta_id = uuid4()
    s_caller_id = uuid4()
    s_alpha_task_id = uuid4()
    s_beta_task_id = uuid4()

    files = [
        CodeFileRecord(
            file_id=f_alpha_id,
            domain_id=domain_id,
            repo_name="repo_alpha",
            file_path="alpha.py",
            git_commit_sha="a" * 40,
            git_blob_sha="1" * 40,
            language="python",
            allowed_scopes=["public_eng"],
            raw_content="class Task:\n    pass\n\ndef caller_alpha():\n    return Task()\n",
            line_count=5,
        ),
        CodeFileRecord(
            file_id=f_beta_id,
            domain_id=domain_id,
            repo_name="repo_beta",
            file_path="beta.py",
            git_commit_sha="b" * 40,
            git_blob_sha="2" * 40,
            language="python",
            allowed_scopes=["public_eng"],
            raw_content="class Task:\n    pass\n",
            line_count=2,
        ),
    ]
    symbols = [
        ASTSymbolRecord(
            symbol_id=s_alpha_task_id,
            file_id=f_alpha_id,
            domain_id=domain_id,
            qualified_name="Task",
            symbol_kind="class",
            start_line=1,
            end_line=2,
            signature_skeleton="class Task:",
            contextual_summary="Alpha Task class",
            allowed_scopes=["public_eng"],
            embedding=[0.05] * 768,
            repo_qualified_id="repo_alpha::Task",
            bare_name="Task",
        ),
        ASTSymbolRecord(
            symbol_id=s_caller_id,
            file_id=f_alpha_id,
            domain_id=domain_id,
            qualified_name="caller_alpha",
            symbol_kind="function",
            start_line=4,
            end_line=5,
            signature_skeleton="def caller_alpha():",
            contextual_summary="Calls intra-repo Task",
            allowed_scopes=["public_eng"],
            embedding=[0.05] * 768,
            repo_qualified_id="repo_alpha::caller_alpha",
            bare_name="caller_alpha",
        ),
        ASTSymbolRecord(
            symbol_id=s_beta_task_id,
            file_id=f_beta_id,
            domain_id=domain_id,
            qualified_name="Task",
            symbol_kind="class",
            start_line=1,
            end_line=2,
            signature_skeleton="class Task:",
            contextual_summary="Beta Task class",
            allowed_scopes=["public_eng"],
            embedding=[0.05] * 768,
            repo_qualified_id="repo_beta::Task",
            bare_name="Task",
        ),
    ]
    # Intra-repo CALLS + USES_TYPE without target_symbol_id and is_cross_repo=False
    intra_edge = ASTEdgeRecord(
        edge_id=uuid4(),
        domain_id=domain_id,
        source_symbol_id=s_caller_id,
        target_qualified_name="Task",
        target_symbol_id=None,
        edge_kind="USES_TYPE",
        allowed_scopes=["public_eng"],
        target_repo_qualified_id=None,
        is_cross_repo=False,
    )

    try:
        preflight_gate.persist_custom_domain_to_pg(
            domain_id=domain_id,
            display_name="Guard 22 Cross-Repo Isolation Test",
            kind="code",
            manifest_config={},
            code_files=files,
            ast_symbols=symbols,
            ast_edges=[intra_edge],
            doc_chunks=[],
            reset_domain_rows=True,
        )

        bound = BoundGraphRepository(PostgresRLSRepository())
        caller_sym = symbols[1]
        expander = ASTDependencyExpander()
        expanded = expander.expand_1hop(caller_sym, bound, principal, domain_id)
        assert {n.repo_name for n in expanded.all_slices} == {"repo_alpha"}
        assert "repo_beta" not in {n.repo_name for n in expanded.all_slices}

        pg_intra = bound.traverse_1hop_neighborhood(principal, domain_id, s_caller_id)
        assert [n.file.repo_name for n in pg_intra] == ["repo_alpha"]

        # Negative control: explicit is_cross_repo=True edge crosses into repo_beta cleanly
        cross_edge = ASTEdgeRecord(
            edge_id=uuid4(),
            domain_id=domain_id,
            source_symbol_id=s_caller_id,
            target_qualified_name="Task",
            target_symbol_id=s_beta_task_id,
            edge_kind="USES_TYPE",
            allowed_scopes=["public_eng"],
            target_repo_qualified_id="repo_beta::Task",
            is_cross_repo=True,
        )
        preflight_gate.persist_custom_domain_to_pg(
            domain_id=domain_id,
            display_name="Guard 22 Cross-Repo Isolation Test",
            kind="code",
            manifest_config={},
            code_files=[],
            ast_symbols=[],
            ast_edges=[cross_edge],
            doc_chunks=[],
        )
        expanded_cross = expander.expand_1hop(caller_sym, bound, principal, domain_id)
        assert "repo_beta" in {n.repo_name for n in expanded_cross.all_slices}
    finally:
        _cleanup_eval_domains(domain_id)
