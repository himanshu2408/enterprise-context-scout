"""M2-03 tests: polyglot 8-category side effects, contract linker v2, and living repo wiki."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from uuid import uuid4

import psycopg
import pytest
from tests.fixtures.pg_isolation import cleanup_ephemeral_pg_domain, create_ephemeral_pg_domain

from context_scout.ast_engine.contract_linker import CrossBoundaryContractLinker
from context_scout.ast_engine.side_effect_extractor import SideEffectExtractor
from context_scout.config import ScoutSettings
from context_scout.db.repositories import InMemoryRLSRepository, PostgresRLSRepository
from context_scout.db.schema import ASTEdgeRecord, ASTSymbolRecord, CodeFileRecord, DocChunkRecord
from context_scout.db.session import RLSConnectionManager
from context_scout.ingestion.repo_wiki_builder import RepoWikiBuilder
from context_scout.schemas.contracts import PrincipalContext, RoleType


def test_side_effect_extractor_polyglot_all_eight_categories() -> None:
    """Extracts all 8 side-effect categories across Python, TS, Go, Java, Rust, SQL, HCL, YAML."""
    extractor = SideEffectExtractor()
    samples: list[tuple[str, str, str, str, str]] = [
        (
            "python",
            "src/mailer.py",
            "def notify_shipment(addr: str) -> None:\n"
            "    sendgrid_client.send_email(to=addr, template_id='tpl_dispatch_v2')\n",
            "email",
            "tpl_dispatch_v2",
        ),
        (
            "typescript",
            "src/sms.ts",
            "export async function sendAlert(phone: string) {\n"
            "  await twilioClient.messages.create({ to: '+15550101234', body: 'Ready' });\n"
            "}\n",
            "sms",
            "",
        ),
        (
            "go",
            "pkg/push/notify.go",
            "package push\n"
            "func SendDeviceNotice(tok string) {\n"
            '    fcmClient.SendMulticast("device-topic-alerts", tok)\n'
            "}\n",
            "push_notification",
            "",
        ),
        (
            "java",
            "src/main/java/HookSender.java",
            "class HookSender {\n"
            "  void dispatch() {\n"
            "    restTemplate.postForEntity(\n"
            '        "https://hooks.partner.internal/events", body, String.class);\n'
            "  }\n"
            "}\n",
            "webhook",
            "https://hooks.partner.internal/events",
        ),
        (
            "rust",
            "src/publisher.rs",
            "fn emit_event() {\n"
            '    kafka_producer.send("logistics.shipment.dispatched", payload);\n'
            "}\n",
            "pubsub_kafka_event",
            "logistics.shipment.dispatched",
        ),
        (
            "yaml",
            ".github/workflows/nightly_audit.yml",
            'name: Nightly Audit\non:\n  schedule:\n    - cron: "0 2 * * *"\n',
            "cron_job",
            "0 2 * * *",
        ),
        (
            "hcl",
            "infra/flags.tf",
            'resource "launchdarkly_feature_flag" "enable_fast_checkout" {\n'
            '  key  = "enable-fast-checkout"\n'
            '  name = "Fast Checkout"\n'
            "}\n",
            "feature_flag",
            "",
        ),
        (
            "sql",
            "sql/archive_orders.sql",
            "INSERT INTO shipment_audit_log (shipment_id, state) VALUES (42, 'DONE');\n",
            "db_write",
            "shipment_audit_log",
        ),
    ]

    found_categories: set[str] = set()
    for lang, fpath, code, expected_cat, expected_substr in samples:
        records = extractor.extract_from_source(
            code=code,
            language=lang,
            file_path=fpath,
            domain_id="eval_polyglot",
            repo_name="acme/logistics-core",
            allowed_scopes=["public_eng"],
        )
        assert records, f"Expected side effect for {lang} ({fpath})"
        cats = {r.effect_category for r in records}
        assert expected_cat in cats, f"Missing {expected_cat} in {cats}"
        if expected_substr:
            joined = " ".join(
                f"{r.target_identifier} {r.template_or_schema_ref or ''}" for r in records
            )
            assert expected_substr in joined
        assert all(r.start_line >= 1 and r.end_line >= r.start_line for r in records)
        found_categories.update(cats)

    assert len(found_categories) == 8


def test_side_effect_extractor_skips_decoys_and_abstains_on_syntax_errors() -> None:
    """Comments/docstrings produce zero side effects; broken syntax abstains fail-closed."""
    extractor = SideEffectExtractor()
    decoy_code = '''
def pure_formatter(item_id: str) -> str:
    """Docstring decoy: sendgrid_client.send_email(to='x') and twilio.messages.create('+1')."""
    # Comment decoy: kafka_producer.send("shipments.created") and session.commit()
    return f"item:{item_id}"
'''
    decoy_records = extractor.extract_from_source(
        code=decoy_code,
        language="python",
        file_path="src/pure.py",
        domain_id="eval_decoy",
        repo_name="acme/pure-lib",
        allowed_scopes=["public_eng"],
    )
    assert decoy_records == []
    assert extractor.last_parse_diagnostics is not None
    assert extractor.last_parse_diagnostics.abstained is False

    broken_code = "def broken_handler(((: \n    sendgrid_client.send_email('a@b.invalid')\n"
    broken_records = extractor.extract_from_source(
        code=broken_code,
        language="python",
        file_path="src/broken.py",
        domain_id="eval_broken",
        repo_name="acme/broken-lib",
        allowed_scopes=["public_eng"],
    )
    assert broken_records == []
    assert extractor.last_parse_diagnostics is not None
    assert extractor.last_parse_diagnostics.abstained is True


@pytest.mark.postgres
def test_side_effects_live_pg17_forced_rls_isolation() -> None:
    """Verifies Tenant A / Tenant B and scope isolation on PG17 `side_effects` under FORCE RLS."""
    settings = ScoutSettings()
    extractor = SideEffectExtractor()
    repo = PostgresRLSRepository()
    mgr = RLSConnectionManager(settings.db_app_url, use_pool=False)

    tenant_a = uuid4()
    tenant_b = uuid4()
    domain_a = f"eval_se_ta_{uuid.uuid4().hex[:8]}"
    domain_b = f"eval_se_tb_{uuid.uuid4().hex[:8]}"

    try:
        with psycopg.connect(settings.db_admin_url) as admin_conn:
            create_ephemeral_pg_domain(admin_conn, domain_id=domain_a, tenant_id=tenant_a)
            create_ephemeral_pg_domain(admin_conn, domain_id=domain_b, tenant_id=tenant_b)

        p_a_sec = PrincipalContext(
            user_id="sec_a",
            role=RoleType.ENGINEER,
            allowed_scopes=["public_eng", "sec_restricted"],
            tenant_id=tenant_a,
        )
        p_a_pub = PrincipalContext(
            user_id="pub_a",
            role=RoleType.ENGINEER,
            allowed_scopes=["public_eng"],
            tenant_id=tenant_a,
        )
        p_b = PrincipalContext(
            user_id="eng_b",
            role=RoleType.ENGINEER,
            allowed_scopes=["public_eng", "sec_restricted"],
            tenant_id=tenant_b,
        )

        se_a_pub = extractor.extract_from_source(
            code="def pub_hook():\n    requests.post('https://hooks.a.internal/pub')\n",
            language="python",
            file_path="src/pub_hook.py",
            domain_id=domain_a,
            repo_name="tenant-a/hooks",
            allowed_scopes=["public_eng"],
            tenant_id=tenant_a,
        )
        se_a_sec = extractor.extract_from_source(
            code="def sec_mail():\n    sendgrid.send_email(to='ops@a.internal')\n",
            language="python",
            file_path="src/sec_mail.py",
            domain_id=domain_a,
            repo_name="tenant-a/hooks",
            allowed_scopes=["sec_restricted"],
            tenant_id=tenant_a,
        )
        se_b = extractor.extract_from_source(
            code="def b_sms():\n    twilio.messages.create('+15559990000')\n",
            language="python",
            file_path="src/b_sms.py",
            domain_id=domain_b,
            repo_name="tenant-b/sms",
            allowed_scopes=["public_eng"],
            tenant_id=tenant_b,
        )
        assert se_a_pub and se_a_sec and se_b

        with mgr.scoped_transaction(p_a_sec) as conn:
            repo.upsert_side_effects_batch(conn, [*se_a_pub, *se_a_sec])
        with mgr.scoped_transaction(p_b) as conn:
            repo.upsert_side_effects_batch(conn, se_b)

        with mgr.scoped_transaction(p_a_pub) as conn:
            vis_pub = repo.list_visible_side_effects(conn, domain_a)
            assert [r.file_path for r in vis_pub] == ["src/pub_hook.py"]
            assert repo.list_visible_side_effects(conn, domain_b) == []

        with mgr.scoped_transaction(p_a_sec) as conn:
            vis_sec = repo.list_visible_side_effects(conn, domain_a)
            assert {r.file_path for r in vis_sec} == {"src/pub_hook.py", "src/sec_mail.py"}

        with mgr.scoped_transaction(p_b) as conn:
            vis_b = repo.list_visible_side_effects(conn, domain_b)
            assert [r.file_path for r in vis_b] == ["src/b_sms.py"]
            assert repo.list_visible_side_effects(conn, domain_a) == []
    finally:
        with psycopg.connect(settings.db_admin_url) as admin_conn:
            cleanup_ephemeral_pg_domain(admin_conn, domain_a)
            cleanup_ephemeral_pg_domain(admin_conn, domain_b)


def test_cross_boundary_contract_linker_v2_and_acl_boundaries() -> None:
    """Verifies bracketed DTOs, OpenAPI/AsyncAPI/Kafka, SQL/dbt, i18n, Jira, and ACL enclave."""
    domain = "eval_contracts_v2"
    linker = CrossBoundaryContractLinker()

    def _cf(repo: str, fpath: str, lang: str, scopes: list[str]) -> CodeFileRecord:
        return CodeFileRecord(
            file_id=uuid4(),
            domain_id=domain,
            repo_name=repo,
            file_path=fpath,
            git_commit_sha="a" * 40,
            git_blob_sha=uuid4().hex,
            language=lang,
            allowed_scopes=scopes,
            raw_content="",
            line_count=10,
        )

    def _sym(
        cf: CodeFileRecord,
        qname: str,
        sig: str,
        ctx: str,
        *,
        scopes: list[str] | None = None,
        kind: str = "function",
    ) -> ASTSymbolRecord:
        return ASTSymbolRecord(
            symbol_id=uuid4(),
            file_id=cf.file_id,
            domain_id=domain,
            qualified_name=qname,
            symbol_kind=kind,
            start_line=1,
            end_line=5,
            signature_skeleton=sig,
            contextual_summary=ctx,
            allowed_scopes=scopes or cf.allowed_scopes,
        )

    f_be = _cf("acme/orders-api", "src/routes.py", "python", ["public_eng"])
    f_fe = _cf("acme/orders-web", "src/client.ts", "typescript", ["public_eng"])
    f_spec = _cf("acme/api-specs", "openapi/orders.yaml", "api_specs", ["public_eng"])
    f_async = _cf("acme/orders-events", "src/worker.go", "go", ["public_eng"])
    f_sql = _cf("acme/orders-dbt", "models/shipments.sql", "sql_dbt", ["public_eng"])
    f_tpl = _cf("acme/orders-web", "templates/receipt.html", "templates_i18n", ["public_eng"])
    f_sec = _cf("acme/secret-service", "src/secret.py", "python", ["sec_restricted"])

    be_ep = _sym(
        f_be,
        "acme/orders-api::create_shipment",
        "@route POST /v1/shipments",
        "@dto ShipmentDTO fields=[tracking_code,carrier] Implements SHP-101",
    )
    be_pub = _sym(
        f_be,
        "acme/orders-api::emit_shipped",
        "@event_publish shipment.dispatched",
        "@db_write shipment_ledger",
    )
    fe_call = _sym(
        f_fe,
        "acme/orders-web::submitShipment",
        "@route POST /v1/shipments",
        "@dto CreateShipmentDTO fields=[tracking_code,carrier] "
        "@renders_template receipt.html SHP-101",
    )
    spec_op = _sym(
        f_spec,
        "acme/api-specs::createShipmentOp",
        "@route POST /v1/shipments",
        "@dto ShipmentDTO fields=[tracking_code,carrier] @channel shipment.dispatched",
    )
    async_sub = _sym(
        f_async,
        "acme/orders-events::consumeShipment",
        "@event_subscribe shipment.dispatched",
        "Kafka consumer subscribed to shipment.dispatched",
    )
    sql_tbl = _sym(
        f_sql,
        "acme/orders-dbt::shipment_ledger",
        "@sql_model shipment_ledger",
        "@dto shipment_ledger fields=[tracking_code,carrier]",
    )
    tpl_sym = _sym(
        f_tpl,
        "acme/orders-web::receipt.html",
        "template receipt.html",
        "HTML email template receipt.html",
        kind="module",
    )
    sec_call = _sym(
        f_sec,
        "acme/secret-service::leakCheck",
        "@route POST /v1/shipments",
        "Disjoint restricted caller must never link to public endpoint",
    )
    jira_doc = DocChunkRecord(
        chunk_id=uuid4(),
        domain_id=domain,
        doc_id="jira:SHP-101",
        doc_title="SHP-101: Shipments API",
        source_uri="https://mock.atlassian.net/browse/SHP-101",
        market_scope=None,
        published_date=datetime.now(UTC).date(),
        authority_weight=1.0,
        section_heading="Description",
        contextual_header="Jira SHP-101",
        verbatim_content="Implement shipment endpoint",
        metadata_json={},
        allowed_scopes=["public_eng"],
        embedding=[],
        source_type="jira",
    )

    edges = linker.link_contracts(
        domain,
        [be_ep, be_pub, fe_call, spec_op, async_sub, sql_tbl, tpl_sym, sec_call],
        [f_be, f_fe, f_spec, f_async, f_sql, f_tpl, f_sec],
        doc_chunks=[jira_doc],
    )
    triples = {(e.source_symbol_id, e.target_symbol_id, e.edge_kind) for e in edges}

    assert (fe_call.symbol_id, be_ep.symbol_id, "CALLS") in triples
    assert (be_ep.symbol_id, spec_op.symbol_id, "IMPLEMENTS_SPEC") in triples
    assert (be_pub.symbol_id, spec_op.symbol_id, "PUBLISHES_EVENT") in triples
    assert (async_sub.symbol_id, spec_op.symbol_id, "SUBSCRIBES_EVENT") in triples
    assert (be_pub.symbol_id, async_sub.symbol_id, "DISPATCHES") in triples
    assert (be_pub.symbol_id, sql_tbl.symbol_id, "WRITES_TABLE") in triples
    assert (fe_call.symbol_id, tpl_sym.symbol_id, "RENDERS_TEMPLATE") in triples
    assert "IMPLEMENTS_ISSUE" in {e.edge_kind for e in edges}

    # Verify strict ACL isolation: zero edges touching restricted sec_call
    assert all(
        e.source_symbol_id != sec_call.symbol_id and e.target_symbol_id != sec_call.symbol_id
        for e in edges
    )


@pytest.mark.postgres
def test_repo_wiki_builder_enclave_isolation_and_supersession(
    ephemeral_pg_domain: str,
) -> None:
    """Wiki separates public vs restricted enclaves and supersedes prior commits cleanly."""
    domain = ephemeral_pg_domain
    repo_name = "acme/warehouse-service"
    builder = RepoWikiBuilder()

    cf_pub = CodeFileRecord(
        file_id=uuid4(),
        domain_id=domain,
        repo_name=repo_name,
        file_path="src/public_api.py",
        git_commit_sha="commit_v1_aaaa1111",
        git_blob_sha="b" * 40,
        language="python",
        allowed_scopes=["public_eng"],
        raw_content="def get_inventory(): pass\n",
        line_count=1,
    )
    cf_sec = CodeFileRecord(
        file_id=uuid4(),
        domain_id=domain,
        repo_name=repo_name,
        file_path="src/vault_key_rotation.py",
        git_commit_sha="commit_v1_aaaa1111",
        git_blob_sha="c" * 40,
        language="python",
        allowed_scopes=["sec_restricted"],
        raw_content="def rotate_vault_master(): pass\n",
        line_count=1,
    )
    sym_pub = ASTSymbolRecord(
        symbol_id=uuid4(),
        file_id=cf_pub.file_id,
        domain_id=domain,
        qualified_name=f"{repo_name}::get_inventory",
        symbol_kind="function",
        start_line=1,
        end_line=1,
        signature_skeleton="def get_inventory()",
        contextual_summary="",
        allowed_scopes=["public_eng"],
    )
    sym_sec = ASTSymbolRecord(
        symbol_id=uuid4(),
        file_id=cf_sec.file_id,
        domain_id=domain,
        qualified_name=f"{repo_name}::rotate_vault_master",
        symbol_kind="function",
        start_line=1,
        end_line=1,
        signature_skeleton="def rotate_vault_master()",
        contextual_summary="",
        allowed_scopes=["sec_restricted"],
    )
    se_list = SideEffectExtractor().extract_from_source(
        code="def send_note():\n    sendgrid.send_email(to='wh@example.invalid')\n",
        language="python",
        file_path="src/public_api.py",
        domain_id=domain,
        repo_name=repo_name,
        allowed_scopes=["public_eng"],
        file_id=cf_pub.file_id,
        symbols=[sym_pub],
    )
    edge_pub = ASTEdgeRecord(
        edge_id=uuid4(),
        domain_id=domain,
        source_symbol_id=sym_pub.symbol_id,
        target_qualified_name="acme/catalog::lookup_sku",
        target_symbol_id=None,
        edge_kind="CALLS",
        allowed_scopes=["public_eng"],
    )

    mem_repo = InMemoryRLSRepository()
    p_all = PrincipalContext(
        user_id="wiki_admin",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng", "sec_restricted"],
    )
    v1_chunks = builder.sync_repo_wiki_to_repository(
        domain_id=domain,
        repo_name=repo_name,
        repository=mem_repo,
        principal=p_all,
        code_files=[cf_pub, cf_sec],
        symbols=[sym_pub, sym_sec],
        edges=[edge_pub],
        side_effects=se_list,
        commit_sha="commit_v1_aaaa1111",
    )
    pub_chunks = [c for c in v1_chunks if c.allowed_scopes == ["public_eng"]]
    sec_chunks = [c for c in v1_chunks if c.allowed_scopes == ["sec_restricted"]]
    assert len(pub_chunks) >= 3
    assert len(sec_chunks) >= 1

    pub_text = "\n".join(c.verbatim_content for c in pub_chunks)
    assert "get_inventory" in pub_text
    assert "sendgrid" in pub_text.lower() or "email" in pub_text.lower()
    assert "rotate_vault_master" not in pub_text
    assert "vault_key_rotation.py" not in pub_text
    assert "```mermaid" in pub_text

    # Supersede commit v1 with commit v2 on live PG17
    settings = ScoutSettings()
    mgr = RLSConnectionManager(settings.db_app_url, use_pool=False)
    pg_repo = PostgresRLSRepository()
    with mgr.scoped_transaction(p_all) as conn:
        builder.sync_repo_wiki_to_repository(
            domain_id=domain,
            repo_name=repo_name,
            repository=pg_repo,
            conn=conn,
            code_files=[cf_pub],
            symbols=[sym_pub],
            edges=[edge_pub],
            side_effects=se_list,
            commit_sha="commit_v1_aaaa1111",
        )
        builder.sync_repo_wiki_to_repository(
            domain_id=domain,
            repo_name=repo_name,
            repository=pg_repo,
            conn=conn,
            code_files=[cf_pub],
            symbols=[sym_pub],
            edges=[edge_pub],
            side_effects=se_list,
            commit_sha="commit_v2_bbbb2222",
        )
        active_rows = pg_repo.list_visible_doc_chunks(conn, domain, include_stale=False)
        all_rows = pg_repo.list_visible_doc_chunks(conn, domain, include_stale=True)
        assert active_rows and all(c.version_str.startswith("commit_v2") for c in active_rows)
        stale_v1 = [c for c in all_rows if c.version_str.startswith("commit_v1")]
        assert stale_v1 and all(c.is_stale for c in stale_v1)


@pytest.mark.postgres
def test_pg17_persists_and_traverses_all_contract_edge_kinds(ephemeral_pg_domain: str) -> None:
    """PR-4 (ECS-088/ECS-136): PG17 persists & traverses all contract edge kinds."""
    from context_scout.agents.preflight_gate import BoundGraphRepository
    from context_scout.ast_engine.dependency_expander import ASTDependencyExpander

    contract_kinds = (
        "IMPLEMENTS_SPEC",
        "PUBLISHES_EVENT",
        "SUBSCRIBES_EVENT",
        "READS_TABLE",
        "WRITES_TABLE",
        "RENDERS_TEMPLATE",
        "IMPLEMENTS_ISSUE",
        "BINDS_SERVICE_ACCOUNT",
        "ACCESSES_SECRET",
    )
    domain = ephemeral_pg_domain
    principal = PrincipalContext(
        user_id="pg17_contract_tester",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng", f"domain:{domain}"],
    )
    src_file = CodeFileRecord(
        file_id=uuid.uuid4(),
        domain_id=domain,
        repo_name="svc-a",
        file_path="src/handler.py",
        git_commit_sha="1111111",
        git_blob_sha="2222222",
        language="python",
        allowed_scopes=["public_eng", f"domain:{domain}"],
        raw_content="def handle():\n    return 1\n",
        line_count=2,
    )
    tgt_file = CodeFileRecord(
        file_id=uuid.uuid4(),
        domain_id=domain,
        repo_name="svc-b",
        file_path="src/target.py",
        git_commit_sha="3333333",
        git_blob_sha="4444444",
        language="python",
        allowed_scopes=["public_eng", f"domain:{domain}"],
        raw_content="".join(
            f"def sym_{i}():\n    return {i}\n" for i in range(len(contract_kinds))
        ),
        line_count=len(contract_kinds) * 2,
    )
    src_sym = ASTSymbolRecord(
        symbol_id=uuid.uuid4(),
        file_id=src_file.file_id,
        domain_id=domain,
        qualified_name="svc-a::handle",
        symbol_kind="function",
        start_line=1,
        end_line=2,
        signature_skeleton="def handle():",
        contextual_summary="@route POST /api/v1/events",
        allowed_scopes=["public_eng", f"domain:{domain}"],
        embedding=[0.1] * 768,
        repo_qualified_id="svc-a::handle",
        bare_name="handle",
    )
    tgt_syms = [
        ASTSymbolRecord(
            symbol_id=uuid.uuid4(),
            file_id=tgt_file.file_id,
            domain_id=domain,
            qualified_name=f"svc-b::sym_{idx}",
            symbol_kind="function",
            start_line=idx * 2 + 1,
            end_line=idx * 2 + 2,
            signature_skeleton=f"def sym_{idx}():",
            contextual_summary=f"contract target {kind}",
            allowed_scopes=["public_eng", f"domain:{domain}"],
            embedding=[0.1] * 768,
            repo_qualified_id=f"svc-b::sym_{idx}",
            bare_name=f"sym_{idx}",
        )
        for idx, kind in enumerate(contract_kinds)
    ]
    edges = [
        ASTEdgeRecord(
            edge_id=uuid.uuid4(),
            domain_id=domain,
            source_symbol_id=src_sym.symbol_id,
            target_qualified_name=tgt_syms[idx].qualified_name,
            target_symbol_id=tgt_syms[idx].symbol_id,
            edge_kind=kind,
            allowed_scopes=["public_eng", f"domain:{domain}"],
            target_repo_qualified_id=tgt_syms[idx].repo_qualified_id,
            is_cross_repo=True,
        )
        for idx, kind in enumerate(contract_kinds)
    ]

    settings = ScoutSettings()
    mgr = RLSConnectionManager(settings.db_app_url, use_pool=False)
    pg_repo = PostgresRLSRepository()
    with mgr.scoped_transaction(principal) as conn:
        pg_repo.insert_code_file(conn, src_file)
        pg_repo.insert_code_file(conn, tgt_file)
        pg_repo.insert_ast_symbol(conn, src_sym)
        for ts in tgt_syms:
            pg_repo.insert_ast_symbol(conn, ts)
        for edge in edges:
            pg_repo.insert_ast_edge(conn, edge)

        nbrs = pg_repo.traverse_1hop_neighborhood(conn, domain, src_sym.symbol_id)
        observed_kinds = {n.edge_kind for n in nbrs}
        for kind in contract_kinds:
            assert kind in observed_kinds, f"Missing PG17 1-hop traversal for {kind}"

        exp = ASTDependencyExpander().expand_1hop(
            focus_symbol=src_sym,
            repo=BoundGraphRepository(pg_repo, conn),
            principal=principal,
            domain_id=domain,
        )
        expanded_qns = {p.qualified_symbol for p in (*exp.imported_type_slices, *exp.caller_slices)}
        for ts in tgt_syms:
            assert ts.qualified_name in expanded_qns, (
                f"ASTDependencyExpander dropped cross-repo contract symbol {ts.qualified_name}"
            )
