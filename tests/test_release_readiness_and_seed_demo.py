"""Release readiness regression tests: 1.0.0 alignment, server-backed UI graphs, and seed-demo."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tomllib
import uuid
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient

import context_scout
from context_scout.agents.domain_registry import DomainRegistry
from context_scout.agents.scout_workflow import LocalGemmaClient, ScoutWorkflow
from context_scout.api.dependencies import _resolve_auth_secret, validate_auth_secret
from context_scout.api.main import create_app
from context_scout.api.routes.health import probe_runtime_environment
from context_scout.cli import cmd_seed_demo
from context_scout.config import ScoutSettings
from context_scout.db.repositories import (
    ASTEdgeRecord,
    ASTSymbolRecord,
    CodeFileRecord,
    InMemoryRLSRepository,
)
from context_scout.db.session import PoolRole, RLSConnectionManager
from context_scout.schemas.contracts import (
    DEFAULT_TENANT_ID,
    CodeSlicePointer,
    ImplementationContextPack,
    InsufficientContextDiagnostic,
    ModificationIntent,
    PrincipalContext,
    RoleType,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_version_alignment_is_1_0_0_across_all_manifests() -> None:
    """Verifies pyproject.toml, package, FastAPI, web-ui, and Helm chart all match 1.0.0."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["version"] == "1.0.0"
    assert context_scout.__version__ == "1.0.0"
    assert create_app().version == "1.0.0"

    webui_pkg = json.loads((REPO_ROOT / "web-ui" / "package.json").read_text(encoding="utf-8"))
    assert webui_pkg["version"] == "1.0.0"

    chart = yaml.safe_load(
        (REPO_ROOT / "deploy" / "helm" / "context-scout" / "Chart.yaml").read_text(encoding="utf-8")
    )
    assert str(chart["version"]) == "1.0.0"
    assert str(chart["appVersion"]) == "1.0.0"


def test_apache_2_0_license_and_notice_alignment_across_manifests() -> None:
    """Verifies root LICENSE, NOTICE, pyproject.toml, web-ui/package.json, and README align."""
    license_text = (REPO_ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "Apache License" in license_text
    assert "Version 2.0, January 2004" in license_text

    notice_text = (REPO_ROOT / "NOTICE").read_text(encoding="utf-8")
    assert "Copyright 2026 Himanshu Singh" in notice_text
    assert "Apache License, Version 2.0" in notice_text

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["license"]["text"] == "Apache-2.0"

    webui_pkg = json.loads((REPO_ROOT / "web-ui" / "package.json").read_text(encoding="utf-8"))
    assert webui_pkg["license"] == "Apache-2.0"

    readme_text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "Apache License, Version 2.0" in readme_text


def test_ci_pytest_shard_partition_completeness_and_balance() -> None:
    """Verifies scripts/ci_pytest_shard.py partitions 100% of test modules with zero gaps."""
    expected_files = sorted(
        {p.relative_to(REPO_ROOT).as_posix() for p in (REPO_ROOT / "tests").rglob("test_*.py")}
    )
    shard_script = REPO_ROOT / "scripts" / "ci_pytest_shard.py"
    assert shard_script.is_file()

    collected: list[str] = []
    for shard_idx in range(1, 7):
        res = subprocess.run(
            [
                shutil.which("uv") or "uv",
                "run",
                "python",
                str(shard_script),
                "--shard",
                str(shard_idx),
                "--total-shards",
                "6",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        shard_files = res.stdout.strip().split()
        assert len(shard_files) >= 1
        collected.extend(shard_files)

    assert len(collected) == len(set(collected)), "Duplicate test file across shards"
    assert sorted(collected) == expected_files


def test_unhydratable_pointer_abstains_cleanly_without_placeholder_sha() -> None:
    """ScoutWorkflow skips unhydratable pointers instead of fabricating placeholder_sha slices."""
    repo = InMemoryRLSRepository()
    registry = DomainRegistry(REPO_ROOT / "domains")
    registry.load_all()

    ghost_calls = 0

    class _GhostPointerClient(LocalGemmaClient):
        def generate_code_pointers(
            self,
            user_prompt: str,
            domain_id: str | None = None,
            repo: Any = None,
            principal: PrincipalContext | None = None,
            bounce_back_feedback: str | None = None,
            *,
            priority: Any = None,
        ) -> list[CodeSlicePointer]:
            nonlocal ghost_calls
            _ = (user_prompt, domain_id, repo, principal, bounce_back_feedback, priority)
            ghost_calls += 1
            return [
                CodeSlicePointer(
                    repo_name="enterprise-context-scout",
                    file_path="src/does_not_exist/ghost.py",
                    start_line=1,
                    end_line=10,
                    qualified_symbol="does_not_exist.ghost.GhostHandler",
                    selection_reason="Simulated stale pointer",
                    modification_intent=ModificationIntent.MODIFY_TARGET,
                )
            ]

    principal = PrincipalContext(
        user_id="test_ghost_ptr",
        role=RoleType.ENGINEER,
        allowed_scopes=["public", "public_eng", "domain:code_architecture_scout"],
    )
    # Seed one visible file/symbol so Stage 2 runs draft_code_pointers -> hydration.
    vis_fid = uuid.uuid4()
    repo.insert_code_file(
        principal,
        CodeFileRecord(
            file_id=vis_fid,
            domain_id="code_architecture_scout",
            repo_name="enterprise-context-scout",
            file_path="src/context_scout/real_anchor.py",
            git_commit_sha="1" * 40,
            git_blob_sha="2" * 40,
            language="python",
            allowed_scopes=["public", "public_eng"],
            raw_content="class GhostHandler:\n    pass\n",
            line_count=2,
        ),
    )
    repo.insert_ast_symbol(
        principal,
        ASTSymbolRecord(
            symbol_id=uuid.uuid4(),
            file_id=vis_fid,
            domain_id="code_architecture_scout",
            qualified_name="context_scout.real_anchor.GhostHandler",
            symbol_kind="class",
            start_line=1,
            end_line=2,
            signature_skeleton="class GhostHandler:",
            contextual_summary="Anchor symbol for ghost pointer hydration test",
            allowed_scopes=["public", "public_eng"],
            embedding=[0.2] * 768,
            bare_name="GhostHandler",
        ),
    )
    wf = ScoutWorkflow(
        registry=registry,
        repo=repo,
        local_client=_GhostPointerClient(adapter_pool=registry.adapter_pool),
    )
    seen_commit_shas: list[str] = []
    real_validate = wf.validator.validate

    def _spy_validate(*args: Any, **kwargs: Any) -> Any:
        for sl in kwargs.get("code_slices") or ():
            seen_commit_shas.append(sl.git_commit_sha)
        return real_validate(*args, **kwargs)

    wf.validator.validate = _spy_validate  # type: ignore[method-assign]
    res = wf.run(
        user_prompt="Inspect GhostHandler in src/does_not_exist/ghost.py",
        principal=principal,
        domain_id="code_architecture_scout",
    )
    assert ghost_calls >= 1
    assert isinstance(res, InsufficientContextDiagnostic)
    assert res.paid_tokens_spent == 0
    assert "placeholder_sha" not in seen_commit_shas
    assert "placeholder_sha" not in res.model_dump_json()
    assert (
        "does_not_exist.ghost.GhostHandler" in res.missing_symbols
        or "src/does_not_exist/ghost.py" in res.missing_symbols
    )


def test_scout_workflow_emits_server_side_effects_and_architecture_mermaid() -> None:
    """ImplementationContextPack carries server-computed side_effects and architecture_mermaid."""
    repo = InMemoryRLSRepository()
    registry = DomainRegistry(REPO_ROOT / "domains")
    registry.load_all()
    domain_id = "code_architecture_scout"
    principal = PrincipalContext(
        user_id="test_server_graphs",
        role=RoleType.ENGINEER,
        allowed_scopes=["public", "public_eng", f"domain:{domain_id}"],
    )

    code_src = (
        "import os\n"
        "import httpx\n"
        "from context_scout.schemas.contracts import PrincipalContext\n"
        "\n"
        "def emit_audit_webhook(conn, principal: PrincipalContext, url: str) -> None:\n"
        '    endpoint = os.environ.get("SCOUT_SIEM_WEBHOOK_URL", url)\n'
        '    conn.execute("INSERT INTO audit_events (actor) VALUES (%s)", (principal.user_id,))\n'
        '    httpx.post(endpoint, json={"actor": principal.user_id})\n'
        "\n"
        "def dispatch_notification(conn, principal: PrincipalContext) -> None:\n"
        '    emit_audit_webhook(conn, principal, "https://siem.internal/hook")\n'
    )
    f_id = uuid.uuid4()
    s1_id = uuid.uuid4()
    s2_id = uuid.uuid4()
    commit_sha = "a" * 40
    blob_sha = "b" * 40

    repo.insert_code_file(
        principal,
        CodeFileRecord(
            file_id=f_id,
            domain_id=domain_id,
            repo_name="enterprise-context-scout",
            file_path="src/context_scout/security/audit_hook.py",
            git_commit_sha=commit_sha,
            git_blob_sha=blob_sha,
            language="python",
            allowed_scopes=["public", "public_eng"],
            raw_content=code_src,
            line_count=len(code_src.splitlines()),
        ),
    )
    repo.insert_ast_symbol(
        principal,
        ASTSymbolRecord(
            symbol_id=s1_id,
            file_id=f_id,
            domain_id=domain_id,
            qualified_name="context_scout.security.audit_hook.emit_audit_webhook",
            symbol_kind="function",
            start_line=5,
            end_line=8,
            signature_skeleton=(
                "def emit_audit_webhook(conn, principal: PrincipalContext, url: str)"
            ),
            contextual_summary="Inserts audit event and posts webhook egress",
            allowed_scopes=["public", "public_eng"],
            embedding=[0.2] * 768,
            bare_name="emit_audit_webhook",
        ),
    )
    repo.insert_ast_symbol(
        principal,
        ASTSymbolRecord(
            symbol_id=s2_id,
            file_id=f_id,
            domain_id=domain_id,
            qualified_name="context_scout.security.audit_hook.dispatch_notification",
            symbol_kind="function",
            start_line=10,
            end_line=11,
            signature_skeleton="def dispatch_notification(conn, principal: PrincipalContext)",
            contextual_summary="Calls emit_audit_webhook",
            allowed_scopes=["public", "public_eng"],
            embedding=[0.2] * 768,
            bare_name="dispatch_notification",
        ),
    )
    repo.insert_ast_edge(
        principal,
        ASTEdgeRecord(
            edge_id=uuid.uuid4(),
            domain_id=domain_id,
            source_symbol_id=s2_id,
            target_qualified_name="context_scout.security.audit_hook.emit_audit_webhook",
            target_symbol_id=s1_id,
            edge_kind="CALLS",
            allowed_scopes=["public", "public_eng"],
        ),
    )

    wf = ScoutWorkflow(registry=registry, repo=repo)
    res = wf.run(
        user_prompt="Inspect emit_audit_webhook in audit_hook.py for DB writes and webhooks",
        principal=principal,
        domain_id=domain_id,
        normalized_rrf_score=0.92,
    )
    assert isinstance(res, ImplementationContextPack)
    assert len(res.side_effects) >= 1
    categories = {se.effect_type for se in res.side_effects}
    assert categories & {"DB_WRITE", "HTTP_EGRESS", "AUTH_GUARD"}
    assert res.architecture_mermaid is not None
    assert res.architecture_mermaid.startswith("flowchart LR")
    assert "emit_audit_webhook" in res.architecture_mermaid
    assert "-->|" in res.architecture_mermaid


def test_env_example_local_dev_auth_boots_cleanly_while_prod_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loading .env.example in dev mode passes validate_auth_secret; prod still fails closed."""
    env_vars: dict[str, str] = {}
    for line in (REPO_ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        k, v = stripped.split("=", 1)
        env_vars[k.strip()] = v.strip()

    assert env_vars.get("SCOUT_ALLOW_INSECURE_DEV_SECRET") == "true"
    assert env_vars.get("SCOUT_ENV") == "development"

    monkeypatch.delenv("SCOUT_AUTH_HMAC_SECRET", raising=False)
    monkeypatch.setenv("SCOUT_ENV", env_vars["SCOUT_ENV"])
    monkeypatch.setenv(
        "SCOUT_ALLOW_INSECURE_DEV_SECRET", env_vars["SCOUT_ALLOW_INSECURE_DEV_SECRET"]
    )
    assert validate_auth_secret()

    app = create_app()
    with TestClient(app) as client:
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200

    # Production MUST still ignore SCOUT_ALLOW_INSECURE_DEV_SECRET and fail closed.
    monkeypatch.setenv("SCOUT_ENV", "production")
    with pytest.raises(RuntimeError, match="production environment"):
        _resolve_auth_secret()


@pytest.mark.postgres
def test_seed_demo_populates_pg17_healthy_with_zero_git_mutations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runs cmd_seed_demo under an isolated tenant_id + fresh domains copy; verifies zero drift."""
    status_before = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "status", "--short"], text=True
    )

    # Copy domains/ WITHOUT *.safetensors to simulate a fresh clone.
    tmp_domains = tmp_path / "domains"
    shutil.copytree(
        REPO_ROOT / "domains",
        tmp_domains,
        ignore=shutil.ignore_patterns("*.safetensors", "sync_manifest.json"),
    )
    assert list(tmp_domains.rglob("*.safetensors")) == []

    isolated_tenant = uuid.uuid4()
    settings = ScoutSettings()
    monkeypatch.setenv("SCOUT_EMBEDDING_DEVICE", "cpu")

    # Canary: count default-tenant code_files before seeding our isolated tenant.
    canary_principal = PrincipalContext(
        user_id="canary_check",
        role=RoleType.COMPLIANCE_OFFICER,
        allowed_scopes=["public", "public_eng", "public_docs", "audit:compliance"],
        tenant_id=DEFAULT_TENANT_ID,
    )
    mgr = RLSConnectionManager(settings.db_app_url, use_pool=False, pool_role=PoolRole.INGESTION)
    with mgr.scoped_transaction(canary_principal, enable_hnsw_iterative_scan=False) as conn:
        row = conn.execute("SELECT COUNT(*) FROM code_files;").fetchone()
        default_tenant_code_count_before = int(str(row[0])) if row else 0

    try:
        report: dict[str, Any] = cmd_seed_demo(
            repo_root=REPO_ROOT,
            domains_dir=tmp_domains,
            tenant_id=isolated_tenant,
            quick=True,
            settings=settings,
        )
        assert report["status"] == "ok"
        assert report["code_architecture_scout"]["code_files"] >= 40
        assert report["code_architecture_scout"]["ast_symbols"] >= 100
        assert report["code_architecture_scout"]["ast_edges"] >= 50
        assert report["code_architecture_scout"]["side_effects"] >= 1
        assert report["code_architecture_scout"]["wiki_chunks"] >= 1
        assert report["code_architecture_scout"]["mr_overlays"] >= 1
        assert report["enterprise_docs_scout"]["doc_files"] >= 4
        assert report["enterprise_docs_scout"]["doc_chunks"] >= 10
        assert report["data_engineering_scout"]["doc_chunks"] >= 2
        assert report["polyglot_unseen_v2"]["code_files"] >= 4
        assert report["polyglot_unseen_v2"]["ast_symbols"] >= 4

        # Fresh clone without trained .safetensors: seed-demo MUST NOT fabricate fake weights,
        # and serving on clean base open-weights reports overall_status == "healthy".
        created_st = list(tmp_domains.rglob("*.safetensors"))
        assert created_st == []
        registry = DomainRegistry(domains_dir=tmp_domains)
        registry.load_all()
        probed = probe_runtime_environment(registry=registry, repo_root=tmp_path, settings=settings)
        assert probed["database_connected"] is True
        assert probed["overall_status"] == "healthy"

        # Seeded domains must short-circuit sync_custom_domain_corpus_if_needed without re-crawling.
        from context_scout.agents.preflight_gate import sync_custom_domain_corpus_if_needed

        for seeded_dom in (
            "code_architecture_scout",
            "enterprise_docs_scout",
            "data_engineering_scout",
            "polyglot_unseen_v2",
        ):
            sync_res = sync_custom_domain_corpus_if_needed(seeded_dom, tenant_id=isolated_tenant)
            assert sync_res is not None
            assert sync_res["synced"] is False
            assert sync_res["added"] == 0
            assert sync_res["deleted"] == 0

        # Verify PG17 FORCE RLS tables populated for isolated_tenant.
        check_principal = PrincipalContext(
            user_id="seed_demo_verifier",
            role=RoleType.COMPLIANCE_OFFICER,
            allowed_scopes=[
                "public",
                "public_eng",
                "public_docs",
                "audit:compliance",
                "repo:code_architecture_scout",
                "corpus:enterprise_docs",
                "domain:code_architecture_scout",
                "domain:enterprise_docs_scout",
                "domain:data_engineering",
                "domain:data_engineering_scout",
                "domain:polyglot_unseen_v2",
            ],
            tenant_id=isolated_tenant,
        )
        with mgr.scoped_transaction(check_principal, enable_hnsw_iterative_scan=False) as conn:
            for tbl in (
                "domain_packs",
                "repositories",
                "code_files",
                "ast_symbols",
                "ast_edges",
                "side_effects",
                "doc_chunks",
                "ref_snapshots",
                "overlay_file_deltas",
                "domain_sync_manifests",
            ):
                cnt_row = conn.execute(f"SELECT COUNT(*) FROM {tbl};").fetchone()
                cnt = int(str(cnt_row[0])) if cnt_row else 0
                assert cnt > 0, f"Expected populated rows in {tbl} after seed-demo"
    finally:
        # Clean up ONLY isolated_tenant rows (never touch DEFAULT_TENANT_ID).
        import psycopg
        from tests.fixtures.pg_isolation import cleanup_pg_tenants_and_eval_domains

        with psycopg.connect(settings.db_admin_url) as admin_conn:
            cleanup_pg_tenants_and_eval_domains(admin_conn, tenant_ids=(isolated_tenant,))

    # Canary check: default-tenant rows strictly untouched.
    with mgr.scoped_transaction(canary_principal, enable_hnsw_iterative_scan=False) as conn:
        row_after = conn.execute("SELECT COUNT(*) FROM code_files;").fetchone()
        assert (int(str(row_after[0])) if row_after else 0) == default_tenant_code_count_before
    mgr.close()

    status_after = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "status", "--short"], text=True
    )
    assert status_after == status_before


def test_tracked_files_contain_zero_internal_build_scaffolding_identifiers() -> None:
    """Guarantees all tracked text files (`git ls-files`) are free of internal scaffolding tags."""
    forbidden_patterns = (
        "B" + "LK-" + r"\d+",
        "D" + "ISC-" + r"\d+",
        "d" + "isc_" + r"\d+",
        "D" + "EF-0" + r"\d",
        "ENTERPRISE_" + "PRODUCTION_MASTER_PLAN",
        "DISCOVERY_" + "LEDGER",
        "ANTIGRAVITY_" + "EXECUTION_PLAYBOOK",
        "ORIGINAL_" + "REQUEST",
        "teamwork_" + "reviewer",
        ":m1-" + "audit",
    )
    compiled = re.compile("|".join(forbidden_patterns))
    tracked_files = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "ls-files"], text=True
    ).splitlines()
    assert tracked_files, "Expected non-empty git ls-files inventory"

    violations: list[str] = []
    for rel_path in tracked_files:
        abs_path = REPO_ROOT / rel_path
        if not abs_path.is_file():
            continue
        raw = abs_path.read_bytes()
        if b"\x00" in raw:
            continue
        text = raw.decode("utf-8", errors="replace")
        for line_no, line in enumerate(text.splitlines(), start=1):
            match = compiled.search(line)
            if match is not None:
                violations.append(f"{rel_path}:L{line_no} ({match.group(0)})")

    assert violations == [], (
        "Internal build-scaffolding identifiers found in tracked files:\n"
        + "\n".join(violations[:50])
    )


def test_migrations_runner_resolves_app_role_password_from_dotenv_db_app_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Quickstart regression: .env-only SCOUT_DB_APP_URL must set scout_app_user password."""
    from context_scout.db.migrations_runner import _resolve_app_role_password

    monkeypatch.delenv("SCOUT_DB_APP_PASSWORD", raising=False)
    monkeypatch.delenv("SCOUT_DB_APP_URL", raising=False)

    dotenv_url = "postgresql://scout_app_user:scout_local_dev_pw@127.0.0.1:54329/context_scout"
    assert _resolve_app_role_password(None, app_db_url=dotenv_url) == "scout_local_dev_pw"

    # Explicit env vars still take precedence over .env-loaded app_db_url.
    monkeypatch.setenv(
        "SCOUT_DB_APP_URL",
        "postgresql://scout_app_user:env_url_pw@127.0.0.1:54329/context_scout",
    )
    assert _resolve_app_role_password(None, app_db_url=dotenv_url) == "env_url_pw"

    monkeypatch.setenv("SCOUT_DB_APP_PASSWORD", "env_pw_wins")
    assert _resolve_app_role_password(None, app_db_url=dotenv_url) == "env_pw_wins"

    # Explicit password argument wins over all.
    assert _resolve_app_role_password("explicit_pw", app_db_url=dotenv_url) == "explicit_pw"


def test_cli_db_upgrade_passes_settings_app_db_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`context-scout db upgrade` must pass ScoutSettings().db_app_url to run_alembic_upgrade."""
    import context_scout.cli as cli_mod

    captured: dict[str, Any] = {}

    def _fake_upgrade(
        admin_db_url: str,
        *,
        target_revision: str = "head",
        app_role_password: str | None = None,
        app_db_url: str | None = None,
    ) -> None:
        captured["admin_db_url"] = admin_db_url
        captured["target_revision"] = target_revision
        captured["app_role_password"] = app_role_password
        captured["app_db_url"] = app_db_url

    fake_settings = ScoutSettings(
        db_admin_url="postgresql://postgres:admin_pw@127.0.0.1:54329/context_scout",
        db_app_url="postgresql://scout_app_user:dotenv_app_pw@127.0.0.1:54329/context_scout",
    )
    monkeypatch.setattr(cli_mod, "ScoutSettings", lambda: fake_settings)
    monkeypatch.setattr(cli_mod, "run_alembic_upgrade", _fake_upgrade)

    exit_code = cli_mod.main(["db", "upgrade"])
    assert exit_code == 0
    assert captured["admin_db_url"] == fake_settings.db_admin_url
    assert captured["app_db_url"] == fake_settings.db_app_url


def test_health_endpoint_exposes_version_and_overall_status() -> None:
    """GET /api/v1/health must return version == '1.0.0' and overall_status matching status."""
    from fastapi.testclient import TestClient

    from context_scout.api.main import create_app

    app = create_app()
    client = TestClient(app)
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["version"] == "1.0.0"
    assert body["overall_status"] == body["status"]
    assert body["overall_status"] in {"healthy", "degraded", "unhealthy"}


def test_openapi_declares_http_bearer_security_scheme() -> None:
    """OpenAPI schema must expose HTTPBearer so clients/Swagger transmit Authorization headers."""
    from context_scout.api.main import create_app

    spec = create_app().openapi()
    schemes = spec.get("components", {}).get("securitySchemes", {})
    assert "HTTPBearer" in schemes
    assert schemes["HTTPBearer"]["type"] == "http"
    assert schemes["HTTPBearer"]["scheme"] == "bearer"


def test_seed_demo_manifest_uses_non_filesystem_seed_uri() -> None:
    """Seeded domain_sync_manifests must not trigger full-repo re-crawl on hot query paths."""
    src = (REPO_ROOT / "src" / "context_scout" / "cli.py").read_text(encoding="utf-8")
    assert 'source_dir="seed://enterprise-context-scout"' in src
    assert 'source_dir="seed://enterprise-docs"' in src


def test_admin_purge_rejects_cross_tenant_body_tenant_id_with_403(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cross-tenant tenant_id in POST /api/v1/admin/purge fails closed with 403 and no deletion."""
    from fastapi.testclient import TestClient

    from context_scout.api.main import create_app
    from context_scout.api.routes.admin import get_admin_purge_coordinator

    monkeypatch.setenv("SCOUT_ENV", "development")
    monkeypatch.setenv("SCOUT_ALLOW_INSECURE_DEV_SECRET", "true")

    purge_called = False

    class _SentinelCoordinator:
        def purge(self, **kwargs: Any) -> Any:
            nonlocal purge_called
            purge_called = True
            raise AssertionError("purge() must never execute on cross-tenant tenant_id mismatch")

    app = create_app()
    app.dependency_overrides[get_admin_purge_coordinator] = lambda: _SentinelCoordinator()
    client = TestClient(app)

    sess_resp = client.post(
        "/api/v1/auth/session",
        json={
            "user_id": "compliance_tester",
            "role": "compliance_officer",
            "allowed_scopes": ["public", "audit:compliance", "admin:purge"],
        },
    )
    assert sess_resp.status_code == 200
    token = sess_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    resp = client.post(
        "/api/v1/admin/purge",
        headers=headers,
        json={
            "domain_id": "code_architecture_scout",
            "repo_name": "enterprise-context-scout",
            "file_path": "src/context_scout/db/session.py",
            "tenant_id": "00000000-0000-0000-0000-000000000099",
        },
    )
    assert resp.status_code == 403
    assert "cross-tenant" in resp.json()["detail"].lower()
    assert purge_called is False

    bad_uuid_resp = client.post(
        "/api/v1/admin/purge",
        headers=headers,
        json={
            "domain_id": "code_architecture_scout",
            "repo_name": "enterprise-context-scout",
            "file_path": "src/context_scout/db/session.py",
            "tenant_id": "not-a-valid-uuid",
        },
    )
    assert bad_uuid_resp.status_code == 400
    assert purge_called is False


def test_sync_custom_domain_require_manifest_skips_cold_crawl_on_hot_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hot query path must never cold-crawl an un-manifested source_dir."""
    from context_scout.agents.preflight_gate import sync_custom_domain_corpus_if_needed

    domains_dir = tmp_path / "domains"
    dom_dir = domains_dir / "unmanifested_docs_scout"
    dom_dir.mkdir(parents=True)
    src_dir = tmp_path / "huge_pdfs"
    src_dir.mkdir()
    (src_dir / "paper.md").write_text("# Huge Paper\n" + ("content " * 100), encoding="utf-8")
    (dom_dir / "domain.yaml").write_text(
        f"domain_id: unmanifested_docs_scout\n"
        f"display_name: Unmanifested Docs\n"
        f"output_contract: DomainKnowledgeBrief\n"
        f"source_dir: {src_dir}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SCOUT_DOMAINS_DIR", str(domains_dir))
    monkeypatch.setattr(
        "context_scout.agents.preflight_gate.read_custom_sync_manifest",
        lambda *_a, **_kw: None,
    )
    res = sync_custom_domain_corpus_if_needed(
        "unmanifested_docs_scout",
        require_manifest=True,
    )
    assert res is None
