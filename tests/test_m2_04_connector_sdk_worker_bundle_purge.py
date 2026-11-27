"""M2-04 tests: scrubber, Git webhooks, Redis 429 backoff, 3-tier sync, bundle, purge."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import stat
import tarfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import pytest

from context_scout.cli import cmd_import_bundle, cmd_purge_artifact
from context_scout.config import ScoutSettings
from context_scout.db.repositories import InMemoryRLSRepository, PostgresRLSRepository
from context_scout.db.session import RLSConnectionManager
from context_scout.ingestion.bundle_importer import (
    AirGappedBundleImporter,
    BundleIntegrityError,
    BundleSecurityError,
)
from context_scout.ops.artifact_purger import ArtifactPurgeCoordinator
from context_scout.plugins.connector_sdk import (
    ConnectorRateLimitError,
    ConnectorSyncCoordinator,
    RedisRateLimitCoordinator,
    parse_retry_after_seconds,
)
from context_scout.plugins.connectors.confluence import (
    ConfluenceConnectorPlugin,
    ConfluencePageDTO,
)
from context_scout.plugins.connectors.github import GitHubEnterpriseConnectorPlugin
from context_scout.plugins.connectors.gitlab import GitLabConnectorPlugin
from context_scout.queue.indexing_worker import IndexingWorker
from context_scout.queue.task_queue import InMemoryTaskQueue, QueueTask
from context_scout.retrieval.zoekt_client import LocalTrigramZoektIndex
from context_scout.schemas.contracts import PrincipalContext, RoleType
from context_scout.security.pre_index_scrubber import PreIndexScrubber
from context_scout.storage.object_store import InMemoryObjectStore, build_tenant_domain_key


def _make_fake_gitleaks_script(tmp_path: Path, secret_to_flag: str) -> str:
    script = tmp_path / "mock-gitleaks"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        "data = sys.stdin.read()\n"
        f"tok = {secret_to_flag!r}\n"
        "if tok in data:\n"
        "    print(json.dumps([{'RuleID': 'custom-vendor-key', 'Secret': tok}]))\n"
        "    sys.exit(1)\n"
        "sys.exit(0)\n",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return str(script)


def test_pre_index_scrubber_rules_checksums_and_mock_gitleaks(tmp_path: Path) -> None:
    """Scrubs secrets, Luhn cards, MOD-97 IBANs, SSNs, emails, phones, and gitleaks findings."""
    # Construct synthetic test credentials dynamically to satisfy repo hygiene scanners
    aws_k = "AKIA" + "IOSFODNN7EXAMPLE"
    gh_k = "ghp_" + "1234567890abcdefghijklmnop"
    dsn_pw = "SuperSecretDbPass9"
    # Valid Luhn (16-digit) vs invalid Luhn (same prefix + bad check digit)
    valid_card = "4532-" + "0151-" + "1283-" + "0366"
    invalid_card = "4532-" + "0151-" + "1283-" + "0365"
    valid_iban = "DE89" + "37040044" + "0532013000"
    custom_tok = "VENDOR_" + "LIVE_" + "9876543210ABCDEF"
    gitleaks_bin = _make_fake_gitleaks_script(tmp_path, custom_tok)

    raw_text = (
        f"aws={aws_k}\n"
        f"gh='{gh_k}'\n"
        f"db='postgresql://svc:{dsn_pw}@pg.internal:5432/app'\n"
        f"card_ok={valid_card} card_id_not_luhn={invalid_card}\n"
        f"iban={valid_iban} ssn=123-45-6789\n"
        "mail=engineer@corp.internal tel=+1-555-234-5678 ip=10.20.30.40\n"
        f"vendor={custom_tok}\n"
    )
    scrubber = PreIndexScrubber(gitleaks_binary=gitleaks_bin)
    res = scrubber.scrub_text(raw_text, source_name="test.py")

    assert res.used_gitleaks_cli is True
    assert custom_tok not in res.scrubbed_text
    assert "[REDACTED:secret:custom-vendor-key]" in res.scrubbed_text
    assert aws_k not in res.scrubbed_text
    assert gh_k not in res.scrubbed_text
    assert dsn_pw not in res.scrubbed_text
    assert valid_card not in res.scrubbed_text
    assert valid_iban not in res.scrubbed_text
    assert "123-45-6789" not in res.scrubbed_text
    assert "engineer@corp.internal" not in res.scrubbed_text
    assert "+1-555-234-5678" not in res.scrubbed_text
    # Non-Luhn ID and internal infrastructure IP stay readable
    assert invalid_card in res.scrubbed_text
    assert "10.20.30.40" in res.scrubbed_text
    assert res.secret_count >= 4
    assert res.pii_count >= 5


def test_gitlab_and_github_push_webhooks_scrub_secrets_and_extract_side_effects() -> None:
    """GitLab & GitHub push webhooks scrub secrets pre-storage and index SideEffectRecords."""
    gl_tok = "glpat-" + "AbCdEfGhIjKlMnOpQrStUvWx"
    gl_code = (
        f'API_SECRET = "{gl_tok}"\n'
        "def emit_notice(addr: str) -> None:\n"
        "    sendgrid_client.send_email(to=addr, template_id='tpl_welcome')\n"
    ).encode()

    def _gl_handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/repository/compare"):
            return httpx.Response(
                200,
                json={
                    "diffs": [
                        {
                            "old_path": "src/notify.py",
                            "new_path": "src/notify.py",
                            "new_file": True,
                            "renamed_file": False,
                            "deleted_file": False,
                            "diff": "+ ...",
                        }
                    ]
                },
            )
        if "/repository/files/" in path:
            return httpx.Response(200, content=gl_code)
        return httpx.Response(200, json={})

    gl_conn = GitLabConnectorPlugin(
        base_url="https://gitlab.mock.local",
        client=httpx.Client(
            base_url="https://gitlab.mock.local",
            transport=httpx.MockTransport(_gl_handler),
        ),
    )
    mem_repo = InMemoryRLSRepository()
    gl_principal = PrincipalContext(
        user_id="hook_eng",
        role=RoleType.ENGINEER,
        allowed_scopes=gl_conn.resolve_project_member_scopes("acme/notifier", access_level=20),
    )
    gl_task = QueueTask(
        task_id=uuid4(),
        domain_id="eval_hooks",
        event_type="gitlab_push",
        payload={
            "project": {"id": 101, "path_with_namespace": "acme/notifier"},
            "before": "1" * 40,
            "after": "a" * 40,
            "ref": "refs/heads/main",
        },
        priority=10,
        status="PROCESSING",
        worker_id="w1",
        attempts=1,
        max_attempts=3,
        error_message=None,
        scheduled_at=datetime.now(UTC),
        created_at=datetime.now(UTC),
        started_at=datetime.now(UTC),
        processed_at=None,
    )
    gl_conn.process_push_webhook(gl_task, mem_repo, principal=gl_principal)

    files = mem_repo.list_visible_code_files(gl_principal, "eval_hooks")
    assert len(files) == 1
    assert gl_tok not in files[0].raw_content
    assert "[REDACTED:secret:gitlab-token]" in files[0].raw_content
    effects = mem_repo.list_visible_side_effects(gl_principal, "eval_hooks")
    assert len(effects) == 1
    assert effects[0].effect_category == "email"

    # GitHub webhook check
    gh_tok = "github_pat_" + "1234567890abcdefghij1234567890"
    gh_code = (
        f'HOOK_KEY = "{gh_tok}"\n'
        "def send_hook() -> None:\n"
        "    requests.post('https://hooks.partner.local/webhook')\n"
    ).encode()

    def _gh_handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/teams"):
            return httpx.Response(200, json=[{"slug": "platform", "permission": "pull"}])
        if "/compare/" in path:
            return httpx.Response(
                200,
                json={"files": [{"filename": "src/hook.py", "status": "added"}]},
            )
        if "/contents/" in path:
            return httpx.Response(
                200,
                json={
                    "encoding": "base64",
                    "content": base64.b64encode(gh_code).decode("ascii"),
                },
            )
        return httpx.Response(200, json={})

    gh_conn = GitHubEnterpriseConnectorPlugin(
        base_url="https://api.github.mock.local",
        client=httpx.Client(
            base_url="https://api.github.mock.local",
            transport=httpx.MockTransport(_gh_handler),
        ),
    )
    gh_task = QueueTask(
        task_id=uuid4(),
        domain_id="eval_hooks_gh",
        event_type="github_push",
        payload={
            "repository": {"full_name": "acme/hooks-service", "default_branch": "main"},
            "before": "1" * 40,
            "after": "b" * 40,
            "ref": "refs/heads/main",
        },
        priority=10,
        status="PROCESSING",
        attempts=1,
        max_attempts=3,
        worker_id="w1",
        error_message=None,
        scheduled_at=datetime.now(UTC),
        created_at=datetime.now(UTC),
        started_at=datetime.now(UTC),
        processed_at=None,
    )
    gh_principal = PrincipalContext(
        user_id="hook_eng",
        role=RoleType.ENGINEER,
        allowed_scopes=gh_conn.resolve_repo_allowed_scopes("acme/hooks-service"),
    )
    gh_conn.process_push_webhook(gh_task, mem_repo, principal=gh_principal)
    gh_files = mem_repo.list_visible_code_files(gh_principal, "eval_hooks_gh")
    assert len(gh_files) == 1 and gh_tok not in gh_files[0].raw_content
    gh_effects = mem_repo.list_visible_side_effects(gh_principal, "eval_hooks_gh")
    assert len(gh_effects) == 1 and gh_effects[0].effect_category == "webhook"


class _FakeRedis:
    """In-memory minimal Redis interface matching setex/get/delete."""

    def __init__(self) -> None:
        self.kv: dict[str, str] = {}

    def setex(self, key: str, ttl_seconds: int, value: str) -> bool:
        assert ttl_seconds >= 1
        self.kv[key] = value
        return True

    def get(self, key: str) -> str | None:
        return self.kv.get(key)

    def delete(self, *keys: str) -> int:
        return sum(1 for k in keys if self.kv.pop(k, None) is not None)


def test_redis_429_retry_after_backoff_and_worker_deferral() -> None:
    """Coordinates 429 Retry-After across pods and defers tasks without burning DLQ retries."""
    ref_now = datetime(2026, 11, 18, 12, 0, 0, tzinfo=UTC)
    assert parse_retry_after_seconds("42", now=ref_now) == 42.0
    assert abs(parse_retry_after_seconds("Wed, 18 Nov 2026 12:01:00 GMT", now=ref_now) - 60.0) < 0.1

    fake_clock = [1000.0]
    shared_redis = _FakeRedis()
    pod_a = RedisRateLimitCoordinator(redis_client=shared_redis, time_fn=lambda: fake_clock[0])
    pod_b = RedisRateLimitCoordinator(redis_client=shared_redis, time_fn=lambda: fake_clock[0])

    tid = uuid4()
    pod_a.record_rate_limit(tid, "jira", 30.0, container_key="ENG")
    assert 29.0 <= pod_b.remaining_backoff_seconds(tid, "jira", container_key="ENG") <= 30.0

    with pytest.raises(ConnectorRateLimitError) as rl_err:
        pod_b.check_or_raise(tid, "jira", container_key="ENG")
    assert rl_err.value.retry_after_seconds >= 29.0

    # IndexingWorker catching HTTP 429 from connector defers via defer_task without DLQ failure
    class _RateLimitedJira:
        def search_issues_incremental(self, *args: Any, **kwargs: Any) -> list[Any]:
            req = httpx.Request("GET", "https://jira.mock/rest/api/3/search")
            resp = httpx.Response(429, headers={"Retry-After": "15"}, request=req)
            raise httpx.HTTPStatusError("429 Too Many Requests", request=req, response=resp)

        def sync_project_to_repository(self, *args: Any, **kwargs: Any) -> Any:
            return self.search_issues_incremental(*args, **kwargs)

    q = InMemoryTaskQueue(default_tenant_id=tid)
    mem_repo = InMemoryRLSRepository()
    q.enqueue_task(
        domain_id="eval_rl",
        event_type="jira_update",
        payload={"project_key": "ENG"},
        max_attempts=2,
        tenant_id=tid,
    )
    worker = IndexingWorker(
        q,
        repository=mem_repo,
        jira_connector=_RateLimitedJira(),
        rate_limiter=pod_b,
    )
    assert worker.run_once() is True
    stats = q.get_queue_stats(tenant_id=tid)
    assert stats["PENDING"] == 1
    assert stats["DLQ"] == 0
    # Advance clock past cooldown
    fake_clock[0] += 35.0
    assert pod_b.remaining_backoff_seconds(tid, "jira", container_key="ENG") == 0.0


def test_connector_sdk_three_tier_sync_and_worker_real_handlers() -> None:
    """Executes Tier 1 webhook, Tier 2 watermark cursor, & Tier 3 reconcile sweep via Worker."""
    now = datetime(2026, 11, 18, 12, 0, 0, tzinfo=UTC)
    later = now + timedelta(minutes=10)

    class _MockConfluence(ConfluenceConnectorPlugin):
        def __init__(self) -> None:
            super().__init__()
            self.since_seen: list[datetime | None] = []
            self.reconcile_flags: list[bool] = []

        def sync_space_to_repository(
            self,
            domain_id: str,
            space_key: str,
            repository: Any,
            conn: Any = None,
            principal: Any = None,
            current_pages: Any = None,
            *,
            since_updated_at: datetime | None = None,
            deleted_page_ids: list[str] | None = None,
            reconcile_deletes: bool = False,
            **kwargs: Any,
        ) -> Any:
            self.since_seen.append(since_updated_at)
            self.reconcile_flags.append(reconcile_deletes)
            pages = [
                ConfluencePageDTO(
                    page_id="cf-10",
                    space_key=space_key,
                    title="Service Architecture",
                    version_number=1 if since_updated_at is None else 2,
                    author_username="dev",
                    created_at=now,
                    last_modified_at=now if since_updated_at is None else later,
                    source_url="https://mock.atlassian.net/wiki/10",
                    body_storage="<p>Confluence doc body for sync tier verification.</p>",
                )
            ]
            if deleted_page_ids:
                pages = []
            return super().sync_space_to_repository(
                domain_id,
                space_key,
                repository,
                conn=conn,
                principal=principal,
                current_pages=pages,
                since_updated_at=since_updated_at,
                deleted_page_ids=deleted_page_ids,
                reconcile_deletes=reconcile_deletes,
                space_allowed_scopes=["public"],
            )

    cf = _MockConfluence()
    mem_repo = InMemoryRLSRepository()
    coord = ConnectorSyncCoordinator(repository=mem_repo, confluence_connector=cf)
    q = InMemoryTaskQueue()
    worker = IndexingWorker(q, repository=mem_repo, sync_coordinator=coord, build_repo_wiki=False)
    principal = PrincipalContext(
        user_id="sync_dev", role=RoleType.ENGINEER, allowed_scopes=["public"]
    )

    # Tier 1: webhook event -> watermark tier="webhook"
    q.enqueue_task("eval_t3", "confluence_update", {"space_key": "ARCH", "page_id": "cf-10"})
    assert worker.run_once() is True
    wm1 = mem_repo.get_connector_watermark(principal, "eval_t3", "confluence", "ARCH")
    assert wm1 is not None and wm1.sync_tier == "webhook"

    # Tier 2: watermark_sync reads stored cursor and advances tier="watermark"
    q.enqueue_task(
        "eval_t3", "watermark_sync", {"source_id": "confluence", "container_key": "ARCH"}
    )
    assert worker.run_once() is True
    assert cf.since_seen[-1] is not None
    wm2 = mem_repo.get_connector_watermark(principal, "eval_t3", "confluence", "ARCH")
    assert wm2 is not None and wm2.sync_tier == "watermark"

    # Tier 3: reconcile_sweep tombstones deleted page & updates tier="reconcile"
    q.enqueue_task(
        "eval_t3",
        "reconcile_sweep",
        {
            "source_id": "confluence",
            "container_key": "ARCH",
            "deleted_ids": ["cf-10"],
            "reconcile_contracts": True,
            "rebuild_wiki": False,
        },
    )
    assert worker.run_once() is True
    assert cf.reconcile_flags[-1] is True
    wm3 = mem_repo.get_connector_watermark(principal, "eval_t3", "confluence", "ARCH")
    assert wm3 is not None and wm3.sync_tier == "reconcile"
    assert mem_repo.list_visible_doc_chunks(principal, "eval_t3", include_stale=False) == []


def test_airgap_bundle_importer_security_and_end_to_end_cli(tmp_path: Path) -> None:
    """Validates air-gapped tar.gz import, scrubbing, Zoekt, store, & rejects tampering."""
    secret_pw = "ghp_" + "BundleToken9876543210abcdef"
    code_bytes = (
        f'TOKEN = "{secret_pw}"\n'
        "def publish_order_event(order_id: str) -> None:\n"
        "    kafka_producer.send('orders.completed', order_id)\n"
    ).encode()
    doc_bytes = b"# Bundle Runbook\n\nAir-gapped cluster deployment checklist.\n"

    manifest = {
        "bundle_id": "rel-2026-11",
        "domain_id": "eval_airgap",
        "allowed_scopes": ["public_eng"],
        "files": [
            {
                "path": "repos/orders/src/publisher.py",
                "sha256": hashlib.sha256(code_bytes).hexdigest(),
                "kind": "code",
                "repo_name": "acme/orders-bundle",
                "commit_sha": "c" * 40,
            },
            {
                "path": "docs/runbook.md",
                "sha256": hashlib.sha256(doc_bytes).hexdigest(),
                "kind": "doc",
            },
        ],
    }
    manifest_bytes = json.dumps(manifest).encode("utf-8")

    tar_path = tmp_path / "valid_bundle.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        for rel, data in [
            ("manifest.json", manifest_bytes),
            ("repos/orders/src/publisher.py", code_bytes),
            ("docs/runbook.md", doc_bytes),
        ]:
            info = tarfile.TarInfo(name=rel)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))

    mem_repo = InMemoryRLSRepository()
    zoekt = LocalTrigramZoektIndex()
    store = InMemoryObjectStore()
    report = cmd_import_bundle(
        "eval_airgap",
        tar_path,
        repository=mem_repo,
        zoekt_index=zoekt,
        object_store=store,
    )
    assert report["bundle_id"] == "rel-2026-11"
    assert report["code_files_indexed"] == 1
    assert report["side_effects_indexed"] >= 1
    assert report["doc_chunks_indexed"] >= 1
    assert report["wiki_chunks_created"] >= 1
    assert report["scrubbed_findings_count"] >= 1

    principal = PrincipalContext(
        user_id="airgap_eng", role=RoleType.ENGINEER, allowed_scopes=["public_eng"]
    )
    files = mem_repo.list_visible_code_files(principal, "eval_airgap")
    assert len(files) == 1 and secret_pw not in files[0].raw_content
    assert (
        zoekt.search("publish_order_event", principal, domain_id="eval_airgap").stats.files_matched
        == 1
    )
    wm = mem_repo.get_connector_watermark(principal, "eval_airgap", "airgap_bundle", "rel-2026-11")
    assert wm is not None and wm.sync_tier == "bundle"

    # Security checks: traversal member, symlink member, tampered digest fail closed
    evil_tar = tmp_path / "traversal.tar.gz"
    with tarfile.open(evil_tar, "w:gz") as tf:
        info = tarfile.TarInfo(name="../escape.py")
        info.size = 4
        tf.addfile(info, io.BytesIO(b"pass"))
    with pytest.raises(BundleSecurityError):
        AirGappedBundleImporter(repository=mem_repo).import_bundle(
            evil_tar, domain_id="eval_airgap"
        )

    sym_tar = tmp_path / "symlink.tar.gz"
    with tarfile.open(sym_tar, "w:gz") as tf:
        info = tarfile.TarInfo(name="evil_link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tf.addfile(info)
    with pytest.raises(BundleSecurityError):
        AirGappedBundleImporter(repository=mem_repo).import_bundle(sym_tar, domain_id="eval_airgap")

    tamper_tar = tmp_path / "tamper.tar.gz"
    with tarfile.open(tamper_tar, "w:gz") as tf:
        for rel, data in [
            ("manifest.json", manifest_bytes),
            ("repos/orders/src/publisher.py", b"tampered_content"),
            ("docs/runbook.md", doc_bytes),
        ]:
            info = tarfile.TarInfo(name=rel)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    with pytest.raises(BundleIntegrityError):
        AirGappedBundleImporter(repository=mem_repo).import_bundle(
            tamper_tar, domain_id="eval_airgap"
        )


@pytest.mark.postgres
def test_purge_artifact_under_60s_zero_residue_live_pg17_zoekt_object_store(
    ephemeral_pg_domain: str,
    tmp_path: Path,
) -> None:
    """Purges leaked repo across PG17, Zoekt, and Object Store in <60s with zero residue."""
    domain = ephemeral_pg_domain
    settings = ScoutSettings()
    mgr = RLSConnectionManager(settings.db_app_url, use_pool=False)
    pg_repo = PostgresRLSRepository()
    zoekt = LocalTrigramZoektIndex()
    store = InMemoryObjectStore()
    principal = PrincipalContext(
        user_id="sec_responder",
        role=RoleType.COMPLIANCE_OFFICER,
        allowed_scopes=["public_eng"],
    )

    leaked_tok = "AKIA" + "LEAKEDREPOKEY00001"
    leaked_py = b"def init_gateway():\n    sendgrid_client.send_email(to='ops@acme.invalid')\n"
    clean_py = b"def keep_clean():\n    return 200\n"

    def _build_bundle(bdir: Path, bundle_id: str, repo_name: str, payload: bytes) -> Path:
        bdir.mkdir(parents=True, exist_ok=True)
        rel = "src/service.py"
        fpath = bdir / rel
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_bytes(payload)
        man = {
            "bundle_id": bundle_id,
            "domain_id": domain,
            "allowed_scopes": ["public_eng"],
            "files": [
                {
                    "path": rel,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "kind": "code",
                    "repo_name": repo_name,
                    "commit_sha": "e" * 40,
                }
            ],
        }
        (bdir / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
        return bdir

    try:
        leaked_dir = _build_bundle(tmp_path / "b_leaked", "b_leaked", "acme/leaked-repo", leaked_py)
        clean_dir = _build_bundle(tmp_path / "b_clean", "b_clean", "acme/clean-repo", clean_py)

        importer = AirGappedBundleImporter(
            repository=pg_repo,
            connection_manager=mgr,
            zoekt_index=zoekt,
            object_store=store,
        )
        importer.import_bundle(leaked_dir, domain_id=domain, principal=principal)
        importer.import_bundle(clean_dir, domain_id=domain, principal=principal)

        leaked_obj_key = build_tenant_domain_key(
            principal.tenant_id, domain, "repos/acme/leaked-repo/raw/secret.txt"
        )
        clean_obj_key = build_tenant_domain_key(
            principal.tenant_id, domain, "repos/acme/clean-repo/raw/keep.txt"
        )
        store.put_bytes(leaked_obj_key, leaked_tok.encode(), expected_tenant_id=principal.tenant_id)
        store.put_bytes(clean_obj_key, b"safe", expected_tenant_id=principal.tenant_id)

        assert zoekt.search("init_gateway", principal, domain_id=domain).stats.files_matched == 1
        assert zoekt.search("keep_clean", principal, domain_id=domain).stats.files_matched == 1

        report = cmd_purge_artifact(
            domain,
            repo_name="acme/leaked-repo",
            allowed_scopes=("public_eng",),
            repository=pg_repo,
            connection_manager=mgr,
            zoekt_index=zoekt,
            object_store=store,
        )
        assert report["sla_met"] is True
        assert report["elapsed_seconds"] < 60.0
        assert report["residue_found"] == 0
        assert report["residue_details"] == {"pg17": 0, "zoekt": 0, "object_store": 0}
        assert leaked_obj_key in report["object_store_deleted"]
        assert not store.exists(leaked_obj_key, expected_tenant_id=principal.tenant_id)
        assert store.exists(clean_obj_key, expected_tenant_id=principal.tenant_id)

        # Verify Zoekt bitmap ACL revoked & clean repo still searchable
        assert (
            zoekt.acl_registry.get_repo_id(principal.tenant_id, domain, "acme/leaked-repo") is None
        )
        assert zoekt.search("init_gateway", principal, domain_id=domain).stats.files_matched == 0
        assert zoekt.search("keep_clean", principal, domain_id=domain).stats.files_matched == 1

        # Verify PG17 still retains clean repo rows & 0 leaked repo rows
        with mgr.scoped_transaction(principal) as conn:
            rem_files = pg_repo.list_visible_code_file_records(conn, domain)
            assert [f.repo_name for f in rem_files] == ["acme/clean-repo"]
            rem_se = pg_repo.list_visible_side_effects(conn, domain, repo_name="acme/leaked-repo")
            assert rem_se == []

        # Also exercise IndexingWorker `purge_artifact` queue dispatch on clean-repo
        q = InMemoryTaskQueue(default_tenant_id=principal.tenant_id)
        purger = ArtifactPurgeCoordinator(
            repository=pg_repo,
            connection_manager=mgr,
            zoekt_index=zoekt,
            object_store=store,
        )
        worker = IndexingWorker(q, artifact_purger=purger, allowed_scopes=("public_eng",))
        q.enqueue_task(
            domain,
            "purge_artifact",
            {"repo_name": "acme/clean-repo", "allowed_scopes": ["public_eng"]},
            tenant_id=principal.tenant_id,
        )
        assert worker.run_once() is True
        assert not store.exists(clean_obj_key, expected_tenant_id=principal.tenant_id)
        assert zoekt.search("keep_clean", principal, domain_id=domain).stats.files_matched == 0
    finally:
        assert mgr is not None


def test_purger_invalidates_cached_permission_mirror_service() -> None:
    """ArtifactPurgeCoordinator invalidates CachedPermissionMirrorService caches."""
    from context_scout.security.permission_mirror import (
        CachedPermissionMirrorService,
        EnterpriseIdentityProfile,
        GitLabProjectMembership,
        GitLabRole,
    )

    mem_repo = InMemoryRLSRepository()
    perm_cache = CachedPermissionMirrorService()
    principal = PrincipalContext(
        user_id="cached_user_1",
        role=RoleType.COMPLIANCE_OFFICER,
        allowed_scopes=["repo:acme/secret-repo:read"],
    )
    profile = EnterpriseIdentityProfile(
        user_id="cached_user_1",
        gitlab_memberships=[
            GitLabProjectMembership(
                project_path="acme/secret-repo",
                role=GitLabRole.DEVELOPER,
            )
        ],
    )
    resolved_scopes = perm_cache.get_user_scopes(profile)
    assert "repo:acme/secret-repo:read" in resolved_scopes
    assert "cached_user_1" in perm_cache._cache

    other_profile = EnterpriseIdentityProfile(
        user_id="other_repo_reader",
        gitlab_memberships=[
            GitLabProjectMembership(
                project_path="acme/secret-repo",
                role=GitLabRole.MAINTAINER,
            )
        ],
    )
    perm_cache.get_user_scopes(other_profile)
    assert "other_repo_reader" in perm_cache._cache

    purger = ArtifactPurgeCoordinator(
        repository=mem_repo,
        permission_cache=perm_cache,
    )
    report = purger.purge(
        principal,
        "code_architecture_scout",
        repo_name="acme/secret-repo",
    )
    assert report.permission_cache_invalidated is True
    assert "cached_user_1" not in perm_cache._cache
    assert "other_repo_reader" not in perm_cache._cache


@pytest.mark.postgres
def test_purger_residue_checks_derived_wiki_overlays_snapshots_and_symbols(
    ephemeral_pg_domain: str,
) -> None:
    """Residue verifier checks derived repo_wiki, overlays, snapshots, and symbols."""
    from datetime import UTC, datetime
    from uuid import uuid4

    from context_scout.db.schema import (
        ASTSymbolRecord,
        CodeFileRecord,
        DocChunkRecord,
        OverlayFileDeltaRecord,
        RefSnapshotRecord,
        RepositoryRecord,
    )

    principal = PrincipalContext(
        user_id="disc058_responder",
        role=RoleType.COMPLIANCE_OFFICER,
        allowed_scopes=["public_eng"],
    )
    mem_repo = InMemoryRLSRepository()
    domain_id = "eval_disc058_mem"
    repo_name = "acme/derived-repo"
    repo_rec = RepositoryRecord(
        repo_id=uuid4(),
        domain_id=domain_id,
        repo_name=repo_name,
        source_type="github",
        remote_url=f"https://github.local/{repo_name}",
        default_branch="main",
        allowed_scopes=["public_eng"],
        tenant_id=principal.tenant_id,
    )
    mem_repo._repos[repo_rec.repo_id] = repo_rec
    file_id = uuid4()
    mem_repo._files[file_id] = CodeFileRecord(
        file_id=file_id,
        domain_id=domain_id,
        repo_name=repo_name,
        file_path="src/app.py",
        language="python",
        git_commit_sha="a" * 40,
        git_blob_sha="b" * 40,
        raw_content="def foo():\n    return 1\n",
        line_count=2,
        allowed_scopes=["public_eng"],
        repo_id=repo_rec.repo_id,
        tenant_id=principal.tenant_id,
    )
    sym_id = uuid4()
    mem_repo._symbols[sym_id] = ASTSymbolRecord(
        symbol_id=sym_id,
        file_id=file_id,
        domain_id=domain_id,
        qualified_name=f"{repo_name}::src.app::foo",
        symbol_kind="function",
        start_line=1,
        end_line=2,
        signature_skeleton="def foo()",
        contextual_summary="foo",
        allowed_scopes=["public_eng"],
        embedding=[0.1] * 768,
        repo_id=repo_rec.repo_id,
        tenant_id=principal.tenant_id,
    )
    snap_id = uuid4()
    mem_repo._ref_snapshots[snap_id] = RefSnapshotRecord(
        snapshot_id=snap_id,
        domain_id=domain_id,
        repo_name=repo_name,
        ref_name="feature/x",
        ref_kind="feature_branch",
        base_commit_sha="0" * 40,
        head_commit_sha="c" * 40,
        allowed_scopes=["public_eng"],
        repo_id=repo_rec.repo_id,
        tenant_id=principal.tenant_id,
    )
    ov_rec = OverlayFileDeltaRecord(
        delta_id=uuid4(),
        snapshot_id=snap_id,
        domain_id=domain_id,
        repo_name=repo_name,
        file_path="src/app.py",
        change_type="modified",
        allowed_scopes=["public_eng"],
        tenant_id=principal.tenant_id,
    )
    mem_repo._overlay_deltas[(principal.tenant_id, snap_id, "src/app.py")] = ov_rec
    wiki_cid = uuid4()
    mem_repo._docs[wiki_cid] = DocChunkRecord(
        chunk_id=wiki_cid,
        domain_id=domain_id,
        doc_id=f"repo_wiki:{repo_name}:overview",
        doc_title="Derived Repo Wiki",
        source_uri=f"repo://{repo_name}/README.md",
        market_scope=None,
        published_date=datetime.now(UTC).date(),
        authority_weight=1.0,
        section_heading="Overview",
        contextual_header="Overview",
        verbatim_content="Derived repo wiki overview content.",
        metadata_json={},
        allowed_scopes=["public_eng"],
        embedding=[0.1] * 768,
        source_type="repo_wiki",
        tenant_id=principal.tenant_id,
    )

    purger = ArtifactPurgeCoordinator(repository=mem_repo)
    assert (
        purger._count_mem_residue(
            principal,
            domain_id=domain_id,
            repo_name=repo_name,
            file_path=None,
            commit_sha=None,
            blob_sha=None,
            doc_id=None,
            confluence_page_id=None,
            jira_issue_key=None,
            include_derived_wiki=True,
        )
        == 6
    )
    mem_report = purger.purge(principal, domain_id, repo_name=repo_name, include_derived_wiki=True)
    assert mem_report.residue_found == 0
    assert mem_report.pg17_deleted["doc_chunks"] == 1
    assert mem_report.pg17_deleted["overlay_file_deltas"] == 1
    assert mem_report.pg17_deleted["ref_snapshots"] == 1
    assert mem_report.pg17_deleted["ast_symbols"] == 1

    # Also verify PG17 residue check includes derived repo_wiki doc_chunks
    settings = ScoutSettings()
    mgr = RLSConnectionManager(settings.db_app_url, use_pool=False)
    pg_repo = PostgresRLSRepository()
    pg_purger = ArtifactPurgeCoordinator(repository=pg_repo, connection_manager=mgr)
    pg_wiki = DocChunkRecord(
        chunk_id=uuid4(),
        domain_id=ephemeral_pg_domain,
        doc_id=f"repo_wiki:{repo_name}:sec1",
        doc_title="Derived PG Repo Wiki",
        source_uri=f"repo://{repo_name}/README.md",
        market_scope=None,
        published_date=datetime.now(UTC).date(),
        authority_weight=1.0,
        section_heading="Sec1",
        contextual_header="Sec1",
        verbatim_content="Derived PG17 repo wiki chunk for purge residue check.",
        metadata_json={},
        allowed_scopes=["public_eng"],
        embedding=[0.2] * 768,
        source_type="repo_wiki",
        tenant_id=principal.tenant_id,
    )
    with mgr.scoped_transaction(principal) as conn:
        pg_repo.insert_doc_chunk(conn, pg_wiki)
        assert (
            pg_purger._count_pg_residue(
                conn,
                domain_id=ephemeral_pg_domain,
                repo_name=repo_name,
                file_path=None,
                commit_sha=None,
                blob_sha=None,
                doc_id=None,
                confluence_page_id=None,
                jira_issue_key=None,
                include_derived_wiki=True,
            )
            == 1
        )
    pg_report = pg_purger.purge(
        principal, ephemeral_pg_domain, repo_name=repo_name, include_derived_wiki=True
    )
    assert pg_report.residue_found == 0
    assert pg_report.pg17_deleted["doc_chunks"] == 1


def test_purger_rejects_file_path_without_repo_name_and_purges_dotted_doc_id() -> None:
    """Purger fails closed without repo_name and purges dotted doc_id / repo."""
    from context_scout.db.schema import CodeFileRecord, DocChunkRecord

    principal = PrincipalContext(
        user_id="disc069_responder",
        role=RoleType.COMPLIANCE_OFFICER,
        allowed_scopes=["public_eng", "audit:compliance"],
    )
    mem_repo = InMemoryRLSRepository()
    domain_id = "eval_disc069_mem"
    file_id = uuid4()
    mem_repo._files[file_id] = CodeFileRecord(
        file_id=file_id,
        domain_id=domain_id,
        repo_name="acme/real-repo",
        file_path="src/secrets/leaked.py",
        language="python",
        git_commit_sha="a" * 40,
        git_blob_sha="b" * 40,
        raw_content="SECRET_KEY = 'redacted'\n",
        line_count=1,
        allowed_scopes=["public_eng"],
        tenant_id=principal.tenant_id,
    )
    doc_chunk_id = uuid4()
    mem_repo._docs[doc_chunk_id] = DocChunkRecord(
        chunk_id=doc_chunk_id,
        domain_id=domain_id,
        doc_id="doc.v1.2026",
        doc_title="Dotted Doc ID Runbook",
        source_uri="urn:docs:doc.v1.2026",
        market_scope=None,
        published_date=datetime.now(UTC).date(),
        authority_weight=1.0,
        section_heading="Overview",
        contextual_header="Overview",
        verbatim_content="Dotted doc_id purge test content.",
        metadata_json={},
        allowed_scopes=["public_eng"],
        embedding=[0.15] * 768,
        source_type="confluence",
        tenant_id=principal.tenant_id,
    )
    purger = ArtifactPurgeCoordinator(repository=mem_repo)

    # 1. Fail closed when file_path lacks repo_name (None or whitespace)
    with pytest.raises(ValueError, match="file_path purge requires repo_name"):
        purger.purge(principal, domain_id, file_path="src/secrets/leaked.py")
    with pytest.raises(ValueError, match="file_path purge requires repo_name"):
        purger.purge(principal, domain_id, repo_name="   ", file_path="src/secrets/leaked.py")
    assert file_id in mem_repo._files

    # 2. Dotted doc_id purges cleanly without touching repo files
    doc_report = purger.purge(principal, domain_id, doc_id="doc.v1.2026")
    assert doc_report.residue_found == 0
    assert doc_report.pg17_deleted["doc_chunks"] == 1
    assert doc_chunk_id not in mem_repo._docs
    assert file_id in mem_repo._files

    # 3. Repo-only purge removes repository file cleanly
    repo_report = purger.purge(principal, domain_id, repo_name="acme/real-repo")
    assert repo_report.residue_found == 0
    assert repo_report.pg17_deleted["code_files"] == 1
    assert file_id not in mem_repo._files


@pytest.mark.postgres
def test_purger_worker_and_repos_reject_blank_and_ambiguous_selectors(
    ephemeral_pg_domain: str,
) -> None:
    """Purger, worker, and InMemory/PG17 repos reject blank & ambiguous selectors."""
    from context_scout.db.schema import CodeFileRecord, DocChunkRecord

    principal = PrincipalContext(
        user_id="disc070_responder",
        role=RoleType.COMPLIANCE_OFFICER,
        allowed_scopes=["public_eng", "audit:compliance"],
    )
    mem_repo = InMemoryRLSRepository()
    zoekt = LocalTrigramZoektIndex()
    store = InMemoryObjectStore()
    domain_id = "eval_disc070_mem"
    repo_name = "acme/keep-repo"
    file_id = uuid4()
    mem_repo._files[file_id] = CodeFileRecord(
        file_id=file_id,
        domain_id=domain_id,
        repo_name=repo_name,
        file_path="src/keep.py",
        language="python",
        git_commit_sha="a" * 40,
        git_blob_sha="b" * 40,
        raw_content="KEEP_TOKEN = 'safe'\n",
        line_count=1,
        allowed_scopes=["public_eng"],
        tenant_id=principal.tenant_id,
    )
    doc_chunk_id = uuid4()
    mem_repo._docs[doc_chunk_id] = DocChunkRecord(
        chunk_id=doc_chunk_id,
        domain_id=domain_id,
        doc_id="doc.v1.2026",
        doc_title="Keep Doc",
        source_uri="urn:docs:doc.v1.2026",
        market_scope=None,
        published_date=datetime.now(UTC).date(),
        authority_weight=1.0,
        section_heading="Overview",
        contextual_header="Overview",
        verbatim_content="Keep doc content.",
        metadata_json={},
        allowed_scopes=["public_eng"],
        embedding=[0.15] * 768,
        source_type="confluence",
        tenant_id=principal.tenant_id,
    )
    zoekt.index_file(
        tenant_id=principal.tenant_id,
        domain_id=domain_id,
        repo_name=repo_name,
        file_path="src/keep.py",
        content="KEEP_TOKEN = 'safe'\n",
        allowed_scopes=["public_eng"],
    )
    obj_key = build_tenant_domain_key(
        principal.tenant_id, domain_id, f"repos/{repo_name}/raw/keep.txt"
    )
    store.put_bytes(obj_key, b"keep", expected_tenant_id=principal.tenant_id)

    purger = ArtifactPurgeCoordinator(
        repository=mem_repo,
        zoekt_index=zoekt,
        object_store=store,
    )

    bad_kwargs_list: tuple[dict[str, Any], ...] = (
        {"repo_name": repo_name, "file_path": ""},
        {"repo_name": repo_name, "file_path": "   "},
        {"repo_name": repo_name, "doc_id": ""},
        {"repo_name": repo_name, "doc_id": "   "},
        {"repo_name": repo_name, "commit_sha": "   "},
        {"repo_name": repo_name, "blob_sha": "   "},
        {"repo_name": repo_name, "confluence_page_id": "   "},
        {"repo_name": repo_name, "jira_issue_key": "   "},
        {"repo_name": repo_name, "object_keys": ["   "]},
        {"repo_name": repo_name, "object_keys": obj_key},
        {"repo_name": "   "},
        {"doc_id": "doc.v1.2026", "repo_name": repo_name},
        {"repo_name": repo_name, "commit_sha": "a" * 40},
        {"repo_name": repo_name, "blob_sha": "b" * 40},
        {"doc_id": "doc.v1.2026", "confluence_page_id": "cf-1"},
        {"repo_name": repo_name, "file_path": "src/keep.py", "commit_sha": "a" * 40},
    )
    for bad_kwargs in bad_kwargs_list:
        with pytest.raises(ValueError):
            purger.purge(principal, domain_id, **bad_kwargs)

    # All stores must remain untouched after rejected purges
    assert file_id in mem_repo._files
    assert doc_chunk_id in mem_repo._docs
    assert zoekt.search("KEEP_TOKEN", principal, domain_id=domain_id).stats.files_matched == 1
    assert store.exists(obj_key, expected_tenant_id=principal.tenant_id)

    # Direct repository calls must also reject blank/ambiguous selectors
    repo_bad_cases: tuple[dict[str, Any], ...] = (
        {"repo_name": repo_name, "file_path": ""},
        {"repo_name": repo_name, "doc_id": "   "},
        {"doc_id": "doc.v1.2026", "repo_name": repo_name},
        {"repo_name": repo_name, "commit_sha": "a" * 40},
    )
    for repo_kwargs in repo_bad_cases:
        with pytest.raises(ValueError):
            mem_repo.purge_domain_artifacts(principal, domain_id, **repo_kwargs)
    assert file_id in mem_repo._files
    assert doc_chunk_id in mem_repo._docs

    # IndexingWorker must preserve empty strings and fail closed instead of wiping repo
    q = InMemoryTaskQueue(default_tenant_id=principal.tenant_id)
    worker = IndexingWorker(
        q, artifact_purger=purger, allowed_scopes=("public_eng", "audit:compliance")
    )
    bad_worker_payloads = (
        {"repo_name": repo_name, "file_path": ""},
        {"repo_name": repo_name, "doc_id": "   "},
        {"repo_name": repo_name, "object_keys": "not-a-sequence"},
    )
    for idx, bad_payload in enumerate(bad_worker_payloads, start=1):
        q.enqueue_task(
            domain_id,
            "purge_artifact",
            bad_payload,
            max_attempts=1,
            tenant_id=principal.tenant_id,
        )
        assert worker.run_once() is True
        dlq = q.list_dead_letter_records(tenant_id=principal.tenant_id)
        assert len(dlq) == idx
    assert file_id in mem_repo._files
    assert zoekt.search("KEEP_TOKEN", principal, domain_id=domain_id).stats.files_matched == 1

    # Verify include_derived_wiki=False is forwarded by IndexingWorker
    wiki_id = uuid4()
    mem_repo._docs[wiki_id] = DocChunkRecord(
        chunk_id=wiki_id,
        domain_id=domain_id,
        doc_id=f"repo_wiki:{repo_name}:overview",
        doc_title="Derived Wiki",
        source_uri=f"repo://{repo_name}/README.md",
        market_scope=None,
        published_date=datetime.now(UTC).date(),
        authority_weight=1.0,
        section_heading="Overview",
        contextual_header="Overview",
        verbatim_content="Derived wiki content.",
        metadata_json={},
        allowed_scopes=["public_eng"],
        embedding=[0.15] * 768,
        source_type="repo_wiki",
        tenant_id=principal.tenant_id,
    )
    q.enqueue_task(
        domain_id,
        "purge_artifact",
        {"repo_name": repo_name, "include_derived_wiki": False},
        tenant_id=principal.tenant_id,
    )
    assert worker.run_once() is True
    assert file_id not in mem_repo._files
    assert wiki_id in mem_repo._docs

    # Object-key-only purge must succeed when no DB selector is supplied
    obj_only_report = purger.purge(principal, domain_id, object_keys=[obj_key])
    assert obj_only_report.residue_found == 0
    assert not store.exists(obj_key, expected_tenant_id=principal.tenant_id)

    # PostgresRLSRepository direct call must also fail closed and preserve PG17 rows
    settings = ScoutSettings()
    mgr = RLSConnectionManager(settings.db_app_url, use_pool=False)
    pg_repo = PostgresRLSRepository()
    pg_file_id = uuid4()
    with mgr.scoped_transaction(principal) as conn:
        pg_repo.insert_code_file(
            conn,
            CodeFileRecord(
                file_id=pg_file_id,
                domain_id=ephemeral_pg_domain,
                repo_name=repo_name,
                file_path="src/keep_pg.py",
                language="python",
                git_commit_sha="c" * 40,
                git_blob_sha="d" * 40,
                raw_content="PG_KEEP = 1\n",
                line_count=1,
                allowed_scopes=["public_eng"],
                tenant_id=principal.tenant_id,
            ),
        )
        pg_bad_cases: tuple[dict[str, Any], ...] = (
            {"repo_name": repo_name, "file_path": ""},
            {"repo_name": repo_name, "doc_id": "   "},
            {"doc_id": "doc.v1.2026", "repo_name": repo_name},
        )
        for pg_bad_kwargs in pg_bad_cases:
            with pytest.raises(ValueError):
                pg_repo.purge_domain_artifacts(conn, ephemeral_pg_domain, **pg_bad_kwargs)
        rem = pg_repo.list_visible_code_file_records(conn, ephemeral_pg_domain)
        assert [f.file_id for f in rem] == [pg_file_id]
