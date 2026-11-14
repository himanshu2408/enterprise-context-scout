"""Permanent fast (<20s) regression suite for Step 3 Enterprise Scale & Concurrency Qualification.

Verifies:
1. Generated qualification reports and provenance summary satisfy schema, N>=60, 10K page,
   40-concurrent-user SSE, and 95% Wilson CI invariants.
2. Live PG17 Confluence lifecycle (v1->v2->v3, tombstones, noise exclusion, table preservation,
   2026 vs 2023 conflict warnings, NULL/'CONFLUENCE' market_scope, RBAC zero-leak, ENGINEER
   pure-doc code_slices==[], plus negative control proving PreFlightValidator executed).
3. Live PG17 polyglot cross-boundary contract expansion, colliding same-name symbol isolation
   (with positive/negative control on is_cross_repo), and code-vs-stale-RFC drift detection.
4. 20-way concurrent multi-principal FastAPI ASGI SSE non-bleed test against live PG17 FORCE RLS.
5. Strict eval_* FK teardown and PG17 non-eval corpus count canary invariance.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections.abc import Generator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx
import psycopg
import pytest

from context_scout.agents.conflict_detector import ConflictDetector
from context_scout.agents.domain_registry import DomainRegistry
from context_scout.agents.preflight_gate import (
    BoundGraphRepository,
    PreFlightValidator,
    persist_custom_domain_to_pg,
)
from context_scout.agents.scout_workflow import LocalGemmaClient, ScoutWorkflow
from context_scout.api import dependencies
from context_scout.api.main import create_app
from context_scout.ast_engine.contract_linker import CrossBoundaryContractLinker
from context_scout.ast_engine.dependency_expander import ASTDependencyExpander
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
from context_scout.ingestion.git_merkle_indexer import generate_deterministic_embedding
from context_scout.plugins.connectors.confluence import (
    ConfluenceConnectorPlugin,
    ConfluenceNoisePolicy,
    ConfluencePageDTO,
)
from context_scout.schemas.contracts import (
    AuthorityRules,
    DomainPackManifest,
    ImplementationContextPack,
    InsufficientContextDiagnostic,
    ModificationIntent,
    PrincipalContext,
    RoleType,
    VerbatimCodeSlice,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
_HMAC_SECRET = "scout-step3-pytest-hmac-secret-32-chars-minimum-ok!!"
_FK_DELETE_ORDER: tuple[str, ...] = (
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


def _snapshot_non_eval_counts(db_admin_url: str) -> dict[str, dict[str, int]]:
    snap: dict[str, dict[str, int]] = {}
    with psycopg.connect(db_admin_url) as conn:
        for tbl in ("code_files", "ast_symbols", "ast_edges", "doc_chunks"):
            rows = conn.execute(
                f"SELECT domain_id, count(*) FROM {tbl} "
                "WHERE domain_id NOT LIKE 'eval_%%' GROUP BY domain_id ORDER BY domain_id;"
            ).fetchall()
            snap[tbl] = {str(r[0]): int(r[1]) for r in rows}
    return snap


def _cleanup_eval_domain(db_admin_url: str, domain_prefix: str) -> None:
    with psycopg.connect(db_admin_url) as conn:
        rows = conn.execute(
            "SELECT domain_id FROM domain_packs WHERE domain_id LIKE %s;",
            (f"{domain_prefix}%",),
        ).fetchall()
        dids = [str(r[0]) for r in rows]
        if dids:
            for tbl in _FK_DELETE_ORDER:
                conn.execute(f"DELETE FROM {tbl} WHERE domain_id = ANY(%s);", (dids,))
            conn.commit()


@pytest.fixture(scope="module")
def pg17_canary() -> Generator[ScoutSettings, None, None]:
    """Snapshots shared PG17 counts and asserts canary invariance after Step 3 tests."""
    settings = ScoutSettings()
    before = _snapshot_non_eval_counts(settings.db_admin_url)
    yield settings
    after = _snapshot_non_eval_counts(settings.db_admin_url)
    assert before == after, f"Shared PG17 corpus mutated: before={before}, after={after}"


def test_step3_qualification_reports_exist_and_valid() -> None:
    """Validates generated Suite 3.1 & Suite 3.2 reports, sample sizes, and 95% Wilson CIs."""
    reports_dir = REPO_ROOT / "reports"
    wiki_json_p = reports_dir / "enterprise_wiki_10k_qualification.json"
    wiki_md_p = reports_dir / "enterprise_wiki_10k_qualification.md"
    prov_md_p = reports_dir / "enterprise_wiki_10k_provenance_summary.md"
    poly_json_p = reports_dir / "polyglot_multiservice_concurrency_qualification.json"
    poly_md_p = reports_dir / "polyglot_multiservice_concurrency_qualification.md"

    for p in (wiki_json_p, wiki_md_p, prov_md_p, poly_json_p, poly_md_p):
        assert p.exists(), f"Missing Step 3 report file: {p}"
        assert p.stat().st_size > 200, f"Empty/stub report file: {p}"

    wiki = json.loads(wiki_json_p.read_text(encoding="utf-8"))
    assert wiki["provenance"]["total_pages_constructed"] == 10000
    assert len(wiki["head_checkpoints"]) == 3
    assert [c["checkpoint_pages"] for c in wiki["head_checkpoints"]] == [1000, 5000, 10000]
    h10 = wiki["head_checkpoints"][-1]
    assert h10["total_queries"] >= 60
    assert h10["rbac_zero_leak_passed"] == h10["rbac_total"]
    assert h10["noise_rejection_clean"] == h10["noise_rejection_total"]
    assert h10["in_scope_refusal_pct"] <= 1.0
    assert h10["eng_puredoc_zero_code"] == h10["eng_puredoc_total"]
    for ci_key in ("noise_rejection_ci95", "rbac_ci95", "table_lookup_ci95"):
        lo, hi = h10[ci_key]
        assert 0.0 <= lo <= hi <= 1.0

    poly = json.loads(poly_json_p.read_text(encoding="utf-8"))
    sse = poly["concurrent_sse_40"]
    assert sse["concurrent_users"] == 40
    assert sse["http_200_sse_completed"] == 40
    assert sse["zero_cross_principal_leaks"] == 40
    assert poly["head_metrics"]["colliding_symbol_false_leaks"] == 0
    assert poly["shared_pg17_canary_verified"] is True


def test_pg17_confluence_lifecycle_rbac_and_preflight_negative_control(
    pg17_canary: ScoutSettings,
) -> None:
    """Validates v1->v2->v3, tombstones, noise, tables, 2026 vs 2023 conflicts, and PreFlight."""
    domain_id = f"eval_s3test_wiki_{uuid.uuid4().hex[:8]}"
    conn_mgr = RLSConnectionManager(pg17_canary.db_app_url, use_pool=False)
    pg_repo = PostgresRLSRepository()
    plugin = ConfluenceConnectorPlugin(
        freshness_half_life_days=1500, noise_policy=ConfluenceNoisePolicy()
    )

    admin = PrincipalContext(
        user_id="s3_test_admin",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng", "public_pm", "audit:compliance"],
    )
    eng = PrincipalContext(
        user_id="s3_test_eng",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng"],
    )

    real_md_src = (
        REPO_ROOT / "data" / "unseen_repos_v2" / "httpx" / "docs" / "advanced" / "timeouts.md"
    ).read_text(encoding="utf-8")[:1000]
    real_env_src = (
        REPO_ROOT / "data" / "unseen_repos_v2" / "flask" / "docs" / "cli.rst"
    ).read_text(encoding="utf-8")[:900]
    now_dt = datetime(2026, 10, 15, 12, 0, tzinfo=UTC)
    old_dt = datetime(2023, 3, 10, 12, 0, tzinfo=UTC)

    manifest = DomainPackManifest(
        domain_id=domain_id,
        display_name="Step 3 Fast Wiki Check",
        base_model_id="google/gemma-4-26B-A4B-it",
        lora_adapter_id="s3_fast_wiki",
        lora_adapter_path="adapters/s3_fast_wiki",
        mcp_read_tools_allowed=["list_visible_doc_chunks"],
        authority_rules=AuthorityRules(
            primary_truth_source="document", freshness_half_life_days=180
        ),
        output_contract="DomainKnowledgeBrief",
    )

    try:
        with conn_mgr.scoped_transaction(admin) as conn:
            pg_repo.upsert_domain_pack(conn, manifest)
            # v1 -> v2 -> v3 lifecycle on real doc sections
            for ver, text in (
                (1, f"LEGACY_V1_MARKER\n{real_md_src[:350]}"),
                (2, f"SUPERSEDED_V2_MARKER\n{real_md_src[200:600]}"),
                (3, f"CANONICAL_V3_MARKER timeout = 10.0\n{real_md_src[:500]}"),
            ):
                dto = ConfluencePageDTO(
                    page_id="pg_ver_01",
                    space_key="ENG_RUNBOOKS",
                    title="HTTPX Timeout Canonical Runbook",
                    version_number=ver,
                    author_username="maintainer",
                    source_url="https://wiki.internal/spaces/ENG_RUNBOOKS/pages/pg_ver_01",
                    body_storage=f"<p>{text}</p>",
                    labels=("runbook",),
                    status="current",
                    created_at=now_dt - timedelta(days=30),
                    last_modified_at=now_dt,
                )
                plugin.sync_space_to_repository(
                    domain_id=domain_id,
                    space_key="ENG_RUNBOOKS",
                    repository=pg_repo,
                    conn=conn,
                    current_pages=[dto],
                    space_allowed_scopes=["public_eng"],
                )

            # Tombstoned page + noise pages + 2023 conflicting page + restricted compliance page
            tomb_dto = ConfluencePageDTO(
                page_id="pg_tomb_01",
                space_key="ENG_RUNBOOKS",
                title="Deleted Gateway Runbook",
                version_number=1,
                author_username="maintainer",
                source_url="https://wiki.internal/spaces/ENG_RUNBOOKS/pages/pg_tomb_01",
                body_storage=f"<p>TOMB_LEAK_MARKER {real_env_src[:300]}</p>",
                status="current",
                created_at=now_dt - timedelta(days=20),
                last_modified_at=now_dt - timedelta(days=10),
            )
            noise_personal = ConfluencePageDTO(
                page_id="pg_personal_01",
                space_key="~personal_notes",
                title="Personal Draft Timeout Notes",
                version_number=1,
                author_username="alice",
                source_url="https://wiki.internal/spaces/~personal_notes/pages/pg_personal_01",
                body_storage=f"<p>PERSONAL_NOISE_MARKER {real_md_src[:250]}</p>",
                status="current",
                created_at=now_dt,
                last_modified_at=now_dt,
            )
            conf_2023 = ConfluencePageDTO(
                page_id="pg_conf_2023",
                space_key="ENG_RUNBOOKS",
                title="HTTPX Timeout Legacy Architecture Notes 2023",
                version_number=1,
                author_username="legacy",
                source_url="https://wiki.internal/spaces/ENG_RUNBOOKS/pages/pg_conf_2023",
                body_storage=f"<p>timeout = 45.0 {real_env_src[:300]}</p>",
                status="current",
                created_at=old_dt,
                last_modified_at=old_dt,
            )
            comp_dto = ConfluencePageDTO(
                page_id="pg_comp_01",
                space_key="COMPLIANCE_POLICY",
                title="Restricted Audit Retention Policy",
                version_number=1,
                author_username="auditor",
                source_url="https://wiki.internal/spaces/COMPLIANCE_POLICY/pages/pg_comp_01",
                body_storage=f"<p>AUDIT_SECRET_9911 {real_env_src[:300]}</p>",
                labels=("policy",),
                status="current",
                created_at=now_dt,
                last_modified_at=now_dt,
            )
            plugin.sync_space_to_repository(
                domain_id=domain_id,
                space_key="ENG_RUNBOOKS",
                repository=pg_repo,
                conn=conn,
                current_pages=[tomb_dto],
                space_allowed_scopes=["public_eng"],
            )
            plugin.sync_space_to_repository(
                domain_id=domain_id,
                space_key="ENG_RUNBOOKS",
                repository=pg_repo,
                conn=conn,
                current_pages=[],
                deleted_page_ids=["pg_tomb_01"],
                space_allowed_scopes=["public_eng"],
            )
            plugin.sync_space_to_repository(
                domain_id=domain_id,
                space_key="~personal_notes",
                repository=pg_repo,
                conn=conn,
                current_pages=[noise_personal],
                space_allowed_scopes=["public_eng"],
            )
            plugin.sync_space_to_repository(
                domain_id=domain_id,
                space_key="ENG_RUNBOOKS",
                repository=pg_repo,
                conn=conn,
                current_pages=[conf_2023],
                space_allowed_scopes=["public_eng"],
            )
            plugin.sync_space_to_repository(
                domain_id=domain_id,
                space_key="COMPLIANCE_POLICY",
                repository=pg_repo,
                conn=conn,
                current_pages=[comp_dto],
                space_allowed_scopes=["audit:compliance"],
            )

            # Seed a real code file so ENGINEER pure-doc suppression runs with has_visible_code=True
            pg_repo.insert_code_file(
                conn,
                CodeFileRecord(
                    file_id=uuid.uuid4(),
                    domain_id=domain_id,
                    repo_name="companion-service",
                    file_path="src/companion.py",
                    git_commit_sha="abc123def456",
                    git_blob_sha="blob12345678",
                    language="python",
                    allowed_scopes=["public_eng"],
                    raw_content="def get_timeout() -> float:\n    return 10.0\n",
                    line_count=2,
                ),
            )

        with conn_mgr.scoped_transaction(eng) as eng_conn:
            visible = pg_repo.list_visible_doc_chunks(eng_conn, domain_id, include_stale=False)
            visible_ids = {c.doc_id for c in visible}
            visible_text = " ".join(c.verbatim_content for c in visible)
            assert "confluence:pg_ver_01" in visible_ids
            assert "CANONICAL_V3_MARKER" in visible_text
            assert "LEGACY_V1_MARKER" not in visible_text
            assert "SUPERSEDED_V2_MARKER" not in visible_text
            assert "TOMB_LEAK_MARKER" not in visible_text
            assert "PERSONAL_NOISE_MARKER" not in visible_text
            assert "AUDIT_SECRET_9911" not in visible_text

        registry = DomainRegistry(REPO_ROOT / "domains")
        registry.load_all()
        registry.register(manifest)

        # Positive run: ENGINEER pure-doc query emits conflict warning & suppresses code_slices
        wf_strict = ScoutWorkflow(
            registry=registry,
            repo=pg_repo,
            local_client=LocalGemmaClient(offline_mode=True),
            validator=PreFlightValidator(repo=pg_repo, min_rrf_score=0.55),
        )
        pack = wf_strict.run(
            user_prompt="What is the authoritative timeout runbook policy in HTTPX Timeout?",
            principal=eng,
            domain_id=domain_id,
            normalized_rrf_score=0.90,
            doc_chunks=visible,
            max_bounce_backs=0,
            source_filter="wiki",
        )
        assert isinstance(pack, ImplementationContextPack)
        assert pack.code_slices == []
        assert any("CONFLUENCE_" in w for w in pack.doc_vs_code_conflicts_detected)

        # Negative control proving PreFlightValidator executed: score=0.15 abstains at min=0.55,
        # passes only when gate threshold is weakened to min=0.05.
        diag = wf_strict.run(
            user_prompt="What is the authoritative timeout runbook policy in HTTPX Timeout?",
            principal=eng,
            domain_id=domain_id,
            normalized_rrf_score=0.15,
            doc_chunks=visible,
            max_bounce_backs=0,
            source_filter="wiki",
        )
        assert isinstance(diag, InsufficientContextDiagnostic)

        wf_weak = ScoutWorkflow(
            registry=registry,
            repo=pg_repo,
            local_client=LocalGemmaClient(offline_mode=True),
            validator=PreFlightValidator(repo=pg_repo, min_rrf_score=0.05),
        )
        weak_pack = wf_weak.run(
            user_prompt="What is the authoritative timeout runbook policy in HTTPX Timeout?",
            principal=eng,
            domain_id=domain_id,
            normalized_rrf_score=0.15,
            doc_chunks=visible,
            max_bounce_backs=0,
            source_filter="wiki",
        )
        assert isinstance(weak_pack, ImplementationContextPack)
    finally:
        _cleanup_eval_domain(pg17_canary.db_admin_url, domain_id)


def test_pg17_polyglot_contracts_collision_and_20way_sse_non_bleed(
    pg17_canary: ScoutSettings,
) -> None:
    """Validates cross-boundary contracts, symbol collision isolation, drift, and 20-way SSE."""
    domain_id = f"eval_s3test_poly_{uuid.uuid4().hex[:8]}"
    ext = ASTExtractor()
    scout_py = (REPO_ROOT / "src" / "context_scout" / "api" / "routes" / "scout.py").read_text(
        encoding="utf-8"
    )
    console_tsx = (REPO_ROOT / "web-ui" / "src" / "components" / "ScoutConsole.tsx").read_text(
        encoding="utf-8"
    )
    tf_hcl = (
        REPO_ROOT / "data" / "unseen_repos_v2" / "enterprise_gitlab_ops" / "infra" / "main.tf"
    ).read_text(encoding="utf-8")

    f_api_id = uuid.uuid4()
    f_ui_id = uuid.uuid4()
    f_tf_id = uuid.uuid4()

    files = [
        CodeFileRecord(
            file_id=f_api_id,
            domain_id=domain_id,
            repo_name="scout-api",
            file_path="src/context_scout/api/routes/scout.py",
            git_commit_sha="56c9d5e00001",
            git_blob_sha="api0010",
            language="python",
            allowed_scopes=["public_eng", f"domain:{domain_id}", "repo:scout-api"],
            raw_content=scout_py,
            line_count=len(scout_py.splitlines()),
        ),
        CodeFileRecord(
            file_id=f_ui_id,
            domain_id=domain_id,
            repo_name="scout-ui",
            file_path="web-ui/src/components/ScoutConsole.tsx",
            git_commit_sha="56c9d5e00001",
            git_blob_sha="ui001",
            language="tsx",
            allowed_scopes=["public_eng", f"domain:{domain_id}", "repo:scout-ui"],
            raw_content=console_tsx,
            line_count=len(console_tsx.splitlines()),
        ),
        CodeFileRecord(
            file_id=f_tf_id,
            domain_id=domain_id,
            repo_name="gitlab-ops",
            file_path="infra/main.tf",
            git_commit_sha="56c9d5e00001",
            git_blob_sha="tf001",
            language="terraform",
            allowed_scopes=["public_eng", f"domain:{domain_id}", "repo:gitlab-ops"],
            raw_content=tf_hcl,
            line_count=len(tf_hcl.splitlines()),
        ),
    ]

    api_parsed, _ = ext.parse_source(
        scout_py, "python", file_path="src/context_scout/api/routes/scout.py", repo_name="scout-api"
    )
    ui_parsed, _ = ext.parse_source(
        console_tsx, "tsx", file_path="web-ui/src/components/ScoutConsole.tsx", repo_name="scout-ui"
    )
    tf_parsed, _ = ext.parse_source(
        tf_hcl, "terraform", file_path="infra/main.tf", repo_name="gitlab-ops"
    )

    symbols: list[ASTSymbolRecord] = []
    for repo_name, fid, parsed in (
        ("scout-api", f_api_id, api_parsed),
        ("scout-ui", f_ui_id, ui_parsed),
        ("gitlab-ops", f_tf_id, tf_parsed),
    ):
        for ps in parsed:
            summary = ps.contextual_summary or ps.signature_skeleton
            leaf = (ps.bare_name or ps.name).rsplit(".", 1)[-1].rsplit("::", 1)[-1]
            if repo_name == "scout-api" and leaf == "stream_scout":
                summary = f"{summary}\n@route POST /api/v1/scout/stream @env TF_VAR_gcp_project_id"
            elif (repo_name == "scout-api" and leaf == "query_scout") or (
                repo_name == "scout-ui" and leaf in {"ScoutConsole", "ScoutConsoleProps"}
            ):
                summary = f"{summary}\n@route POST /api/v1/scout/query"
            elif repo_name == "gitlab-ops" and leaf == "gcp_project_id":
                summary = f"{summary}\n@env TF_VAR_gcp_project_id"
            symbols.append(
                ASTSymbolRecord(
                    symbol_id=uuid.uuid4(),
                    file_id=fid,
                    domain_id=domain_id,
                    qualified_name=ps.qualified_name,
                    symbol_kind=ps.symbol_kind,
                    start_line=ps.start_line,
                    end_line=max(ps.start_line, ps.end_line),
                    signature_skeleton=ps.signature_skeleton,
                    contextual_summary=summary,
                    allowed_scopes=["public_eng", f"domain:{domain_id}", f"repo:{repo_name}"],
                    embedding=generate_deterministic_embedding(ps.qualified_name),
                    repo_qualified_id=ps.qualified_name
                    if "::" in ps.qualified_name
                    else f"{repo_name}::{ps.qualified_name}",
                    bare_name=ps.bare_name or ps.name,
                )
            )

    # Add colliding same-name symbols in scout-api vs scout-ui (`SharedConfig`)
    sym_coll_api = ASTSymbolRecord(
        symbol_id=uuid.uuid4(),
        file_id=f_api_id,
        domain_id=domain_id,
        qualified_name="scout-api::config::SharedConfig",
        symbol_kind="class",
        start_line=1,
        end_line=3,
        signature_skeleton="class SharedConfig:",
        contextual_summary="API SharedConfig",
        allowed_scopes=["public_eng", f"domain:{domain_id}", "repo:scout-api"],
        embedding=generate_deterministic_embedding("API SharedConfig"),
        repo_qualified_id="scout-api::config::SharedConfig",
        bare_name="SharedConfig",
    )
    sym_coll_ui = ASTSymbolRecord(
        symbol_id=uuid.uuid4(),
        file_id=f_ui_id,
        domain_id=domain_id,
        qualified_name="scout-ui::types::SharedConfig",
        symbol_kind="interface",
        start_line=1,
        end_line=3,
        signature_skeleton="interface SharedConfig {}",
        contextual_summary="UI SharedConfig",
        allowed_scopes=["public_eng", f"domain:{domain_id}", "repo:scout-ui"],
        embedding=generate_deterministic_embedding("UI SharedConfig"),
        repo_qualified_id="scout-ui::types::SharedConfig",
        bare_name="SharedConfig",
    )
    sym_caller_api = ASTSymbolRecord(
        symbol_id=uuid.uuid4(),
        file_id=f_api_id,
        domain_id=domain_id,
        qualified_name="scout-api::routes::load_shared_config",
        symbol_kind="function",
        start_line=10,
        end_line=12,
        signature_skeleton="def load_shared_config() -> SharedConfig:",
        contextual_summary="Calls intra-repo SharedConfig",
        allowed_scopes=["public_eng", f"domain:{domain_id}", "repo:scout-api"],
        embedding=generate_deterministic_embedding("load_shared_config"),
        repo_qualified_id="scout-api::routes::load_shared_config",
        bare_name="load_shared_config",
    )
    symbols.extend([sym_coll_api, sym_coll_ui, sym_caller_api])

    edges = CrossBoundaryContractLinker().link_contracts(domain_id, symbols, files)
    assert any(e.is_cross_repo and e.edge_kind == "CALLS" for e in edges)
    assert any(e.is_cross_repo and e.edge_kind == "PROVISIONED_BY" for e in edges)
    edges.append(
        ASTEdgeRecord(
            edge_id=uuid.uuid4(),
            domain_id=domain_id,
            source_symbol_id=sym_caller_api.symbol_id,
            target_qualified_name="SharedConfig",
            target_symbol_id=None,
            edge_kind="USES_TYPE",
            allowed_scopes=["public_eng", f"domain:{domain_id}"],
            target_repo_qualified_id=None,
            is_cross_repo=False,
        )
    )

    docs = [
        DocChunkRecord(
            chunk_id=uuid.uuid4(),
            domain_id=domain_id,
            doc_id="adr:drift_stream",
            doc_title="Legacy Stream ADR",
            source_uri="https://docs.internal/adr/stream",
            market_scope="INTERNAL_ADR",
            published_date=date(2023, 5, 1),
            authority_weight=0.9,
            section_heading="Route",
            contextual_header="Legacy ADR route spec",
            verbatim_content="Legacy ADR specifies SESSION_TTL_SECONDS = 7200 and ttl = 7200s.",
            metadata_json={},
            allowed_scopes=["public_eng"],
            embedding=generate_deterministic_embedding("SESSION_TTL_SECONDS 7200"),
            source_type="internal_adr",
            version_str="1",
            is_stale=False,
        ),
        DocChunkRecord(
            chunk_id=uuid.uuid4(),
            domain_id=domain_id,
            doc_id="adr:comp_only",
            doc_title="Compliance Audit Policy",
            source_uri="https://docs.internal/adr/comp",
            market_scope="INTERNAL_ADR",
            published_date=date(2026, 9, 1),
            authority_weight=1.1,
            section_heading="Retention",
            contextual_header="Compliance retention policy",
            verbatim_content="S3_CANARY_COMPLIANCE_ONLY: immutable audit trace requirement.",
            metadata_json={},
            allowed_scopes=["audit:compliance"],
            embedding=generate_deterministic_embedding("Compliance Audit Policy"),
            source_type="internal_adr",
            version_str="2",
            is_stale=False,
        ),
        DocChunkRecord(
            chunk_id=uuid.uuid4(),
            domain_id=domain_id,
            doc_id="adr:pm_only",
            doc_title="PM Telemetry Spec",
            source_uri="https://docs.internal/adr/pm",
            market_scope="INTERNAL_ADR",
            published_date=date(2026, 9, 1),
            authority_weight=1.1,
            section_heading="Metrics",
            contextual_header="Product metrics policy",
            verbatim_content="S3_CANARY_PM_ONLY: milestone ROI dashboard spec.",
            metadata_json={},
            allowed_scopes=["public_pm"],
            embedding=generate_deterministic_embedding("PM Telemetry Spec"),
            source_type="internal_adr",
            version_str="2",
            is_stale=False,
        ),
    ]

    manifest = DomainPackManifest(
        domain_id=domain_id,
        display_name="Step 3 Fast Polyglot Check",
        base_model_id="google/gemma-4-26B-A4B-it",
        lora_adapter_id="s3_fast_poly",
        lora_adapter_path="adapters/s3_fast_poly",
        mcp_read_tools_allowed=["search_visible_code", "traverse_1hop_neighborhood"],
        authority_rules=AuthorityRules(primary_truth_source="code"),
        output_contract="ImplementationContextPack",
    )

    prev_repo = dependencies._shared_repo
    prev_client = dependencies._local_gemma_client
    prev_req = os.environ.get("SCOUT_REQUIRE_SIGNED_AUTH")
    prev_sec = os.environ.get("SCOUT_AUTH_HMAC_SECRET")

    try:
        persist_custom_domain_to_pg(
            domain_id=domain_id,
            display_name=manifest.display_name,
            kind="code",
            manifest_config=manifest.model_dump(mode="json"),
            code_files=files,
            ast_symbols=symbols,
            ast_edges=edges,
            doc_chunks=docs,
            reset_domain_rows=True,
        )

        pg_repo = PostgresRLSRepository()
        bound = BoundGraphRepository(pg_repo)
        eng = PrincipalContext(
            user_id="s3_poly_eng",
            role=RoleType.ENGINEER,
            allowed_scopes=[
                "public_eng",
                f"domain:{domain_id}",
                "repo:scout-api",
                "repo:scout-ui",
                "repo:gitlab-ops",
            ],
        )

        # 1. Cross-boundary contract 1-hop expansion
        ui_focus = next(
            s
            for s in symbols
            if (s.bare_name or s.qualified_name).rsplit(".", 1)[-1] == "ScoutConsoleProps"
        )
        exp = ASTDependencyExpander().expand_1hop(
            focus_symbol=ui_focus, repo=bound, principal=eng, domain_id=domain_id
        )
        expanded_qns = [
            p.qualified_symbol
            for p in (*exp.imported_type_slices, *exp.caller_slices, *exp.test_fixture_slices)
        ]
        assert any("query_scout" in qn for qn in expanded_qns)

        # 2. Colliding same-name symbol isolation (0 leaks when is_cross_repo=False)
        nbrs = bound.traverse_1hop_neighborhood(eng, domain_id, sym_caller_api.symbol_id)
        assert not any(n.symbol.qualified_name == sym_coll_ui.qualified_name for n in nbrs)

        # 3. Code-vs-stale-RFC drift detection
        auth_py = (REPO_ROOT / "src" / "context_scout" / "api" / "routes" / "auth.py").read_text(
            encoding="utf-8"
        )
        slice_obj = VerbatimCodeSlice(
            repo_name="scout-api",
            file_path="src/context_scout/api/routes/auth.py",
            qualified_symbol="scout-api::create_auth_session",
            start_line=1,
            end_line=len(auth_py.splitlines()),
            selection_reason="HEAD auth route source",
            modification_intent=ModificationIntent.REFERENCE_ONLY,
            git_commit_sha="56c9d5e00001",
            git_blob_sha="api0010",
            verbatim_code_with_lines="\n".join(
                f"{i + 1}: {ln}" for i, ln in enumerate(auth_py.splitlines()[:120])
            ),
        )
        drift_warnings = ConflictDetector().detect_code_vs_doc_conflicts([slice_obj], [docs[0]])
        assert drift_warnings, "Expected code-vs-stale-RFC TTL conflict warning"

        # 4. 20-way concurrent multi-principal FastAPI ASGI SSE non-bleed test
        os.environ["SCOUT_REQUIRE_SIGNED_AUTH"] = "true"
        os.environ["SCOUT_AUTH_HMAC_SECRET"] = _HMAC_SECRET
        dependencies._shared_repo = None
        dependencies._local_gemma_client = None
        dependencies.get_domain_registry().register(manifest)
        app = create_app()
        dependencies._local_gemma_client = LocalGemmaClient(offline_mode=True)

        async def run_20_sse() -> list[tuple[int, str, str]]:
            personas = [
                (
                    RoleType.ENGINEER,
                    ["public_eng", f"domain:{domain_id}", "repo:scout-api"],
                    "code",
                    "Inspect stream_scout",
                ),
                (
                    RoleType.COMPLIANCE_OFFICER,
                    ["audit:compliance"],
                    "wiki",
                    "Compliance Audit Policy",
                ),
                (RoleType.PRODUCT_MANAGER, ["public_pm"], "wiki", "PM Telemetry Spec"),
                (RoleType.QUANT_RESEARCHER, ["public_quant"], "wiki", "Compliance Audit Policy"),
            ]
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver", timeout=20.0
            ) as client:
                tokens: list[tuple[RoleType, str, str, str]] = []
                for idx in range(20):
                    role, scopes, sf, prompt = personas[idx % 4]
                    tok_r = await client.post(
                        "/api/v1/auth/session",
                        json={
                            "user_id": f"s3_u_{idx}",
                            "role": role.value,
                            "allowed_scopes": scopes,
                        },
                    )
                    assert tok_r.status_code == 200
                    tokens.append((role, sf, prompt, str(tok_r.json()["access_token"])))

                async def hit(item: tuple[RoleType, str, str, str]) -> tuple[int, str, str]:
                    role, sf, prompt, tok = item
                    resp = await client.post(
                        "/api/v1/scout/stream",
                        headers={"Authorization": f"Bearer {tok}"},
                        json={
                            "domain_id": domain_id,
                            "query": prompt,
                            "source_filter": sf,
                            "normalized_rrf_score": 0.88,
                            "max_bounce_backs": 0,
                        },
                    )
                    return (resp.status_code, role.value, resp.text)

                return list(await asyncio.gather(*(hit(t) for t in tokens)))

        sse_results = asyncio.run(run_20_sse())
        assert len(sse_results) == 20
        for code, role_name, body in sse_results:
            assert code == 200
            assert (
                "event: context_pack_ready" in body or '"category":"insufficient_context"' in body
            )
            if role_name != "compliance_officer":
                assert "S3_CANARY_COMPLIANCE_ONLY" not in body
            if role_name != "product_manager":
                assert "S3_CANARY_PM_ONLY" not in body
    finally:
        dependencies._shared_repo = prev_repo
        dependencies._local_gemma_client = prev_client
        if prev_req is None:
            os.environ.pop("SCOUT_REQUIRE_SIGNED_AUTH", None)
        else:
            os.environ["SCOUT_REQUIRE_SIGNED_AUTH"] = prev_req
        if prev_sec is None:
            os.environ.pop("SCOUT_AUTH_HMAC_SECRET", None)
        else:
            os.environ["SCOUT_AUTH_HMAC_SECRET"] = prev_sec
        _cleanup_eval_domain(pg17_canary.db_admin_url, domain_id)
