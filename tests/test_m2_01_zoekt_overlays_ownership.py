"""M2-01 verification: Zoekt client, O(n) RE2, CoW MR overlays, and ownership graph."""

from __future__ import annotations

import base64
import json
import os
import struct
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
import pytest

from context_scout.config import ScoutSettings
from context_scout.db.repositories import InMemoryRLSRepository, PostgresRLSRepository
from context_scout.db.session import RLSConnectionManager
from context_scout.ingestion.git_merkle_indexer import GitMerkleIndexer
from context_scout.ingestion.ownership_graph import (
    WEIGHT_GENERATED,
    WEIGHT_GENERATED_AND_VENDORED,
    WEIGHT_NORMAL,
    WEIGHT_VENDORED,
    CodeOwnersMatcher,
    GitAttributesMatcher,
    OwnershipGraphExtractor,
    normalize_rel_path,
    parse_catalog_info_yaml,
)
from context_scout.retrieval.embedder import get_embedding_provider
from context_scout.retrieval.zoekt_client import (
    LocalTrigramZoektIndex,
    RE2SyntaxError,
    RepoACLBitmapRegistry,
    ZoektGrpcJsonClient,
    ZoektUnavailableError,
    compile_re2,
    extract_required_trigrams,
    validate_re2_pattern,
)
from context_scout.schemas.contracts import PrincipalContext, RoleType


def _git(repo_dir: Path, *args: str) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Scout Engineer",
        "GIT_AUTHOR_EMAIL": "scout@example.invalid",
        "GIT_COMMITTER_NAME": "Scout Engineer",
        "GIT_COMMITTER_EMAIL": "scout@example.invalid",
    }
    proc = subprocess.run(
        ["git", "-C", str(repo_dir), *args],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return proc.stdout.strip()


def _b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def _encode_connect_frame(payload: dict[str, Any], *, flags: int = 0) -> bytes:
    raw = json.dumps(payload).encode("utf-8")
    return struct.pack(">BI", flags, len(raw)) + raw


# -----------------------------------------------------------------------------
# 1. O(n) RE2 Validation & Anti-ReDoS Execution
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_pattern",
    [
        "",
        "(foo)\\1",
        "(?P<x>foo)(?P=x)",
        "(?P<x>foo)\\k<x>",
        "foo(?=bar)",
        "foo(?!bar)",
        "(?<=foo)bar",
        "(?<!foo)bar",
        "(?(1)foo|bar)",
        "(?#comment)foo",
        "(?>a+)b",
        "a++",
        "a{1001}",
        "[unclosed",
    ],
)
def test_re2_validator_rejects_backtracking_and_unsupported_syntax(bad_pattern: str) -> None:
    with pytest.raises(RE2SyntaxError):
        validate_re2_pattern(bad_pattern)


def test_re2_thompson_nfa_linear_time_under_100kb_redos_payloads() -> None:
    payload = "a" * 100_000 + "!"
    for pattern in (r"(a+)+$", r"(a|aa)+$", r"(.*a){25}z"):
        rx = compile_re2(pattern)
        assert rx.uses_thompson_nfa is True
        t0 = time.perf_counter()
        match = rx.search(payload)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert match is None
        assert elapsed_ms < 250.0, f"Pattern {pattern} took {elapsed_ms:.2f}ms (> 250ms)"

    # Verify exact leftmost span preservation on Thompson NFA
    branch_rx = compile_re2(r"a.*c|b", force_thompson=True)
    m = branch_rx.search("xxabxcyy")
    assert m is not None
    assert m.span() == (2, 6)
    assert m.group(0) == "abxc"

    # Trigram plan extraction
    plan = extract_required_trigrams(r"verify_invoice_token\(")
    assert any("ver" in grp for grp in plan.required_groups)
    assert any("tok" in grp for grp in plan.required_groups)


# -----------------------------------------------------------------------------
# 2. Zoekt Repo-ACL Bitmap Pre-Filtering & Connect gRPC/JSON Client
# -----------------------------------------------------------------------------


def test_zoekt_acl_bitmap_isolation_short_circuit_and_post_filter() -> None:
    tenant_a = uuid.uuid4()
    tenant_b = uuid.uuid4()
    registry = RepoACLBitmapRegistry()
    id_a_core = registry.register_repo(tenant_a, "dom_code", "acme/core", ["scope:core"])
    id_a_bill = registry.register_repo(tenant_a, "dom_code", "acme/billing", ["scope:billing"])
    id_b_core = registry.register_repo(tenant_b, "dom_code", "acme/core", ["scope:core"])

    p_a_core = PrincipalContext(
        user_id="alice", role=RoleType.ENGINEER, allowed_scopes=["scope:core"], tenant_id=tenant_a
    )
    p_a_none = PrincipalContext(
        user_id="outsider",
        role=RoleType.ENGINEER,
        allowed_scopes=["scope:unrelated"],
        tenant_id=tenant_a,
    )
    p_b_core = PrincipalContext(
        user_id="bob", role=RoleType.ENGINEER, allowed_scopes=["scope:core"], tenant_id=tenant_b
    )

    acl_a = registry.compute_authorized_bitmap(p_a_core, domain_id="dom_code")
    assert acl_a.repo_ids == (id_a_core,)
    assert acl_a.repo_names == frozenset({"acme/core"})
    assert acl_a.popcount == 1
    assert id_a_bill not in acl_a.repo_ids
    assert id_b_core not in acl_a.repo_ids

    acl_b = registry.compute_authorized_bitmap(p_b_core, domain_id="dom_code")
    assert acl_b.repo_ids == (id_b_core,)
    assert acl_a.bitmap & acl_b.bitmap == 0

    # Fail-closed zero-bitmap short-circuit makes zero HTTP calls
    calls: list[dict[str, Any]] = []

    def _mock_handler(request: httpx.Request) -> httpx.Response:
        calls.append({"url": str(request.url), "body": json.loads(request.content.decode())})
        # Deliberately include unauthorized repo 'acme/billing' to prove post-filter drops it.
        return httpx.Response(
            200,
            json={
                "Result": {
                    "Stats": {"FilesConsidered": 4},
                    "Files": [
                        {
                            "Repository": "acme/core",
                            "RepositoryID": id_a_core,
                            "FileName": _b64("src/token.py"),
                            "Branches": ["main"],
                            "Language": "Python",
                            "Score": 5.0,
                            "LineMatches": [
                                {
                                    "LineNumber": 12,
                                    "Line": _b64("def verify_token(raw: str) -> bool:\n"),
                                    "Before": _b64("# auth helper\n"),
                                    "After": _b64("    return bool(raw)\n"),
                                    "LineFragments": [{"LineOffset": 4, "MatchLength": 12}],
                                }
                            ],
                        },
                        {
                            "Repository": "acme/billing",
                            "RepositoryID": id_a_bill,
                            "FileName": _b64("src/secret.py"),
                            "Branches": ["main"],
                            "Language": "Python",
                            "Score": 99.0,
                            "LineMatches": [
                                {
                                    "LineNumber": 1,
                                    "Line": _b64("def verify_token(): pass\n"),
                                }
                            ],
                        },
                    ],
                }
            },
        )

    client = ZoektGrpcJsonClient(
        "http://zoekt.local:6070",
        acl_registry=registry,
        protocol="grpc_json",
        transport=httpx.MockTransport(_mock_handler),
    )

    empty_resp = client.search("verify_token", p_a_none, domain_id="dom_code")
    assert empty_resp.stats.short_circuited_acl is True
    assert empty_resp.file_matches == ()
    assert len(calls) == 0

    auth_resp = client.search("verify_token", p_a_core, domain_id="dom_code")
    assert len(calls) == 1
    assert calls[0]["url"].endswith("/zoekt.webserver.v1.WebserverService/Search")
    assert calls[0]["body"]["RepoIds"] == [id_a_core]
    assert "r:^(acme/core)$" in calls[0]["body"]["Q"]
    assert len(auth_resp.file_matches) == 1
    hit = auth_resp.file_matches[0]
    assert hit.repo_name == "acme/core"
    assert hit.file_path == "src/token.py"
    assert hit.line_matches[0].line_number == 12
    assert hit.line_matches[0].line_text == "def verify_token(raw: str) -> bool:"
    assert hit.line_matches[0].before_lines == ("# auth helper",)
    assert hit.line_matches[0].fragments == ((4, 16),)
    client.close()


def test_zoekt_uncapped_enumeration_650_files_and_error_fail_closed() -> None:
    tenant_id = uuid.uuid4()
    principal = PrincipalContext(
        user_id="dev1", role=RoleType.ENGINEER, allowed_scopes=["scope:eng"], tenant_id=tenant_id
    )

    # 1. LocalTrigramZoektIndex uncapped enumeration over 650 files
    local_idx = LocalTrigramZoektIndex()
    for idx in range(650):
        local_idx.index_file(
            tenant_id=tenant_id,
            domain_id="dom_scale",
            repo_name="acme/monorepo",
            file_path=f"services/mod_{idx:04d}/handler.py",
            content=f"def verify_invoice_token_{idx}() -> bool:\n    return True\n",
            allowed_scopes=["scope:eng"],
            language="python",
        )

    capped = local_idx.search("verify_invoice_token", principal, domain_id="dom_scale", limit=50)
    assert len(capped.file_matches) == 50
    assert capped.stats.truncated is True
    assert capped.stats.files_matched == 650

    uncapped_list = list(
        local_idx.enumerate_file_matches("verify_invoice_token", principal, domain_id="dom_scale")
    )
    assert len(uncapped_list) == 650
    assert len({m.file_path for m in uncapped_list}) == 650

    # 2. ZoektGrpcJsonClient StreamSearch binary-framed uncapped enumeration (650 files)
    registry = RepoACLBitmapRegistry()
    rid = registry.register_repo(tenant_id, "dom_scale", "acme/monorepo", ["scope:eng"])

    def _stream_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/StreamSearch"):
            req_body = json.loads(request.content.decode())
            assert req_body["Opts"]["MaxDocDisplayCount"] > 100_000
            assert req_body["Opts"]["TotalMaxMatchCount"] > 100_000
            frames = bytearray()
            for chunk_start in range(0, 650, 100):
                files_batch = [
                    {
                        "Repository": "acme/monorepo",
                        "RepositoryID": rid,
                        "FileName": _b64(f"services/mod_{i:04d}/handler.py"),
                        "Branches": ["main"],
                        "Language": "Python",
                        "Score": 1.0,
                        "LineMatches": [
                            {"LineNumber": 1, "Line": _b64("def verify_invoice_token():")}
                        ],
                    }
                    for i in range(chunk_start, min(650, chunk_start + 100))
                ]
                frames.extend(_encode_connect_frame({"Result": {"Files": files_batch}}, flags=0))
            frames.extend(_encode_connect_frame({}, flags=2))
            return httpx.Response(200, content=bytes(frames))
        if request.url.path == "/api/search":
            return httpx.Response(503, json={"error": "backend overloaded"})
        return httpx.Response(404)

    grpc_client = ZoektGrpcJsonClient(
        "http://zoekt.local:6070",
        acl_registry=registry,
        protocol="grpc_json",
        transport=httpx.MockTransport(_stream_handler),
    )
    streamed = list(
        grpc_client.enumerate_file_matches("verify_invoice_token", principal, domain_id="dom_scale")
    )
    assert len(streamed) == 650
    grpc_client.close()

    json_client = ZoektGrpcJsonClient(
        "http://zoekt.local:6070",
        acl_registry=registry,
        protocol="json_api",
        transport=httpx.MockTransport(_stream_handler),
    )
    with pytest.raises(ZoektUnavailableError):
        json_client.search("verify_invoice_token", principal, domain_id="dom_scale")
    json_client.close()


# -----------------------------------------------------------------------------
# 3. Copy-on-Write Branch & Open MR/PR Overlays (PG17 FORCE RLS + InMemory)
# -----------------------------------------------------------------------------


def _create_multi_branch_git_repo(repo_dir: Path) -> None:
    repo_dir.mkdir(parents=True, exist_ok=True)
    _git(repo_dir, "init", "-b", "main")
    (repo_dir / "src").mkdir()
    for i in range(6):
        (repo_dir / "src" / f"untouched_{i}.py").write_text(
            f"def untouched_fn_{i}() -> int:\n    return {i}\n",
            encoding="utf-8",
        )
    (repo_dir / "src" / "billing.py").write_text(
        "def compute_invoice_total(cents: int) -> int:\n    return cents\n",
        encoding="utf-8",
    )
    (repo_dir / "src" / "auth.py").write_text(
        "def authenticate_user(token: str) -> bool:\n    return len(token) > 0\n",
        encoding="utf-8",
    )
    (repo_dir / "src" / "legacy.py").write_text(
        "def legacy_helper() -> str:\n    return 'legacy'\n",
        encoding="utf-8",
    )
    (repo_dir / "src" / "rename_me.py").write_text(
        "def old_service_entry() -> str:\n    return 'old'\n",
        encoding="utf-8",
    )
    _git(repo_dir, "add", "-A")
    _git(repo_dir, "commit", "-m", "feat: base main branch")

    # Branch 1: mr/101 modifies billing.py and adds new_mr101.py (8 files untouched)
    _git(repo_dir, "checkout", "-b", "mr/101")
    (repo_dir / "src" / "billing.py").write_text(
        "def compute_invoice_total(cents: int) -> int:\n"
        "    return cents + compute_mr101_tax(cents)\n\n"
        "def compute_mr101_tax(cents: int) -> int:\n"
        "    return cents // 10\n",
        encoding="utf-8",
    )
    (repo_dir / "src" / "new_mr101.py").write_text(
        "def mr101_ledger_hook() -> str:\n    return 'mr101'\n",
        encoding="utf-8",
    )
    _git(repo_dir, "add", "-A")
    _git(repo_dir, "commit", "-m", "feat(mr101): add tax hook")

    # Child stacked branch on top of mr/101
    _git(repo_dir, "checkout", "-b", "feat/child-101")
    (repo_dir / "src" / "new_mr101.py").write_text(
        "def mr101_ledger_hook_v2() -> str:\n    return 'child101'\n",
        encoding="utf-8",
    )
    (repo_dir / "src" / "legacy.py").unlink()
    _git(repo_dir, "add", "-A")
    _git(repo_dir, "commit", "-m", "feat(child): stack v2 hook and remove legacy")

    # Branch 2: mr/102 off main deletes legacy.py, renames rename_me.py -> renamed_service.py
    _git(repo_dir, "checkout", "main")
    _git(repo_dir, "checkout", "-b", "mr/102")
    (repo_dir / "src" / "legacy.py").unlink()
    _git(repo_dir, "mv", "src/rename_me.py", "src/renamed_service.py")
    (repo_dir / "src" / "auth.py").write_text(
        "def authenticate_user_strict(token: str) -> bool:\n    return len(token) >= 16\n",
        encoding="utf-8",
    )
    _git(repo_dir, "add", "-A")
    _git(repo_dir, "commit", "-m", "refactor(mr102): rename service and drop legacy")
    _git(repo_dir, "checkout", "main")


def test_cow_mr_overlays_inmemory_and_zoekt_isolation(tmp_path: Path) -> None:
    repo_dir = tmp_path / "cow_repo"
    _create_multi_branch_git_repo(repo_dir)
    indexer = GitMerkleIndexer(use_semantic_embeddings=False)
    mem_repo = InMemoryRLSRepository()
    principal = PrincipalContext(
        user_id="alice", role=RoleType.ENGINEER, allowed_scopes=["scope:eng"]
    )
    domain_id = "eval_cow_mem"
    repo_name = "acme/platform"

    indexer.sync_in_memory(
        repo_dir, mem_repo, principal, domain_id, repo_name, allowed_scopes=["scope:eng"]
    )
    base_files = mem_repo.list_visible_code_files(principal, domain_id)
    assert len(base_files) == 10

    snap_101, deltas_101 = indexer.sync_overlay_in_memory(
        repo_dir,
        mem_repo,
        principal,
        domain_id,
        repo_name,
        ref_name="refs/merge-requests/101/head",
        base_ref="main",
        head_ref="mr/101",
        ref_kind="merge_request",
        mr_iid="101",
    )
    # CoW invariant: only 2 delta rows stored for mr/101 (8 untouched files duplicated 0 times)
    assert len(deltas_101) == 2

    snap_102, deltas_102 = indexer.sync_overlay_in_memory(
        repo_dir,
        mem_repo,
        principal,
        domain_id,
        repo_name,
        ref_name="refs/merge-requests/102/head",
        base_ref="main",
        head_ref="mr/102",
        ref_kind="merge_request",
        mr_iid="102",
    )
    assert len(deltas_102) == 3

    snap_child, deltas_child = indexer.sync_overlay_in_memory(
        repo_dir,
        mem_repo,
        principal,
        domain_id,
        repo_name,
        ref_name="refs/heads/feat/child-101",
        base_ref="mr/101",
        head_ref="feat/child-101",
        ref_kind="feature_branch",
        base_snapshot_id=snap_101.snapshot_id,
    )
    assert len(deltas_child) == 2

    # Base main files remain completely unmutated
    base_billing = mem_repo.get_code_files_by_paths(principal, domain_id, ["src/billing.py"])[0]
    assert "compute_mr101_tax" not in base_billing.raw_content

    # COALESCE resolution for mr/101
    resolved_101 = {
        f.file_path: f
        for f in mem_repo.resolve_overlay_files(principal, domain_id, snap_101.snapshot_id)
    }
    assert len(resolved_101) == 11
    assert resolved_101["src/billing.py"].source_layer == "overlay"
    assert "compute_mr101_tax" in resolved_101["src/billing.py"].raw_content
    assert resolved_101["src/new_mr101.py"].source_layer == "overlay"
    assert resolved_101["src/untouched_0.py"].source_layer == "base_snapshot"
    assert "src/renamed_service.py" not in resolved_101

    # COALESCE resolution for mr/102 (deletes + renames isolated from mr/101)
    resolved_102 = {
        f.file_path: f
        for f in mem_repo.resolve_overlay_files(principal, domain_id, snap_102.snapshot_id)
    }
    assert "src/legacy.py" not in resolved_102
    assert "src/rename_me.py" not in resolved_102
    assert "src/new_mr101.py" not in resolved_102
    assert resolved_102["src/renamed_service.py"].source_layer == "overlay"

    # Stacked child resolution COALESCE(child, mr101, base)
    resolved_child = {
        f.file_path: f
        for f in mem_repo.resolve_overlay_files(principal, domain_id, snap_child.snapshot_id)
    }
    assert "src/legacy.py" not in resolved_child
    assert "compute_mr101_tax" in resolved_child["src/billing.py"].raw_content
    assert "mr101_ledger_hook_v2" in resolved_child["src/new_mr101.py"].raw_content

    # Symbol overlay resolution
    syms_101 = {
        s.qualified_name
        for s in mem_repo.resolve_overlay_symbols(principal, domain_id, snap_101.snapshot_id)
    }
    assert any("compute_mr101_tax" in q for q in syms_101)
    assert any("legacy_helper" in q for q in syms_101)

    # Zoekt search with overlay_deltas merges MR additions and suppresses deleted files
    zoekt = LocalTrigramZoektIndex()
    for cf in base_files:
        zoekt.index_file(
            tenant_id=principal.tenant_id,
            domain_id=domain_id,
            repo_name=repo_name,
            file_path=cf.file_path,
            content=cf.raw_content,
            allowed_scopes=cf.allowed_scopes,
            language=cf.language,
        )
    res_main = zoekt.search("compute_mr101_tax", principal, domain_id=domain_id)
    assert res_main.file_matches == ()
    res_101 = zoekt.search(
        "compute_mr101_tax", principal, domain_id=domain_id, overlay_deltas=deltas_101
    )
    assert len(res_101.file_matches) == 1
    assert res_101.file_matches[0].source_layer == "overlay"
    assert res_101.file_matches[0].file_path == "src/billing.py"

    res_102_legacy = zoekt.search(
        "legacy_helper", principal, domain_id=domain_id, overlay_deltas=deltas_102
    )
    assert res_102_legacy.file_matches == ()


@pytest.mark.postgres
def test_live_pg17_cow_mr_overlays_and_cross_tenant_rls(
    tmp_path: Path,
    ephemeral_pg_domain: str,
) -> None:
    repo_dir = tmp_path / "pg_cow_repo"
    _create_multi_branch_git_repo(repo_dir)
    settings = ScoutSettings()
    app_mgr = RLSConnectionManager(settings.db_app_url)
    pg_repo = PostgresRLSRepository()
    indexer = GitMerkleIndexer(use_semantic_embeddings=False)

    tenant_a = uuid.uuid4()
    tenant_b = uuid.uuid4()
    principal_a = PrincipalContext(
        user_id="alice_a",
        role=RoleType.ENGINEER,
        allowed_scopes=["scope:tenant_a"],
        tenant_id=tenant_a,
    )
    principal_a_other_scope = PrincipalContext(
        user_id="charlie_a",
        role=RoleType.ENGINEER,
        allowed_scopes=["scope:unrelated"],
        tenant_id=tenant_a,
    )
    principal_b = PrincipalContext(
        user_id="bob_b",
        role=RoleType.ENGINEER,
        allowed_scopes=["scope:tenant_a"],
        tenant_id=tenant_b,
    )

    try:
        # Ensure tenant_a domain_pack row exists for composite FK
        with app_mgr.scoped_transaction(principal_a, enable_hnsw_iterative_scan=False) as conn:
            conn.execute(
                """
                INSERT INTO domain_packs (
                    tenant_id, domain_id, display_name, lora_adapter_id,
                    primary_truth_source, config_json
                )
                VALUES (%s, %s, 'CoW PG Eval', 'none', 'code', '{}'::jsonb)
                ON CONFLICT (tenant_id, domain_id) DO NOTHING;
                """,
                (tenant_a, ephemeral_pg_domain),
            )
            indexer.sync_postgres(
                repo_dir,
                conn,
                pg_repo,
                principal_a,
                ephemeral_pg_domain,
                "acme/platform",
                allowed_scopes=["scope:tenant_a"],
            )
            snap_101, deltas_101 = indexer.sync_overlay_postgres(
                repo_dir,
                conn,
                pg_repo,
                principal_a,
                ephemeral_pg_domain,
                "acme/platform",
                ref_name="refs/merge-requests/101/head",
                base_ref="main",
                head_ref="mr/101",
                ref_kind="merge_request",
                mr_iid="101",
                allowed_scopes=["scope:tenant_a"],
            )
            snap_102, deltas_102 = indexer.sync_overlay_postgres(
                repo_dir,
                conn,
                pg_repo,
                principal_a,
                ephemeral_pg_domain,
                "acme/platform",
                ref_name="refs/merge-requests/102/head",
                base_ref="main",
                head_ref="mr/102",
                ref_kind="merge_request",
                mr_iid="102",
                allowed_scopes=["scope:tenant_a"],
            )
            snap_child, _ = indexer.sync_overlay_postgres(
                repo_dir,
                conn,
                pg_repo,
                principal_a,
                ephemeral_pg_domain,
                "acme/platform",
                ref_name="refs/heads/feat/child-101",
                base_ref="mr/101",
                head_ref="feat/child-101",
                ref_kind="feature_branch",
                base_snapshot_id=snap_101.snapshot_id,
                allowed_scopes=["scope:tenant_a"],
            )
            assert len(deltas_101) == 2
            assert len(deltas_102) == 3

        with app_mgr.scoped_transaction(principal_a, enable_hnsw_iterative_scan=False) as conn:
            resolved_101 = {
                r.file_path: r
                for r in pg_repo.resolve_overlay_files(
                    conn, ephemeral_pg_domain, snap_101.snapshot_id
                )
            }
            assert len(resolved_101) == 11
            assert resolved_101["src/billing.py"].source_layer == "overlay"
            assert "compute_mr101_tax" in resolved_101["src/billing.py"].raw_content
            assert resolved_101["src/untouched_0.py"].source_layer == "base_snapshot"

            single = pg_repo.get_overlay_file(
                conn, ephemeral_pg_domain, snap_102.snapshot_id, "src/legacy.py"
            )
            assert single is None
            renamed = pg_repo.get_overlay_file(
                conn, ephemeral_pg_domain, snap_102.snapshot_id, "src/renamed_service.py"
            )
            assert renamed is not None
            assert renamed.change_type == "renamed"

            child_resolved = {
                r.file_path: r
                for r in pg_repo.resolve_overlay_files(
                    conn, ephemeral_pg_domain, snap_child.snapshot_id
                )
            }
            assert "src/legacy.py" not in child_resolved
            assert "mr101_ledger_hook_v2" in child_resolved["src/new_mr101.py"].raw_content

            syms_102 = {
                s.qualified_name
                for s in pg_repo.resolve_overlay_symbols(
                    conn, ephemeral_pg_domain, snap_102.snapshot_id
                )
            }
            assert any("authenticate_user_strict" in q for q in syms_102)
            assert not any("legacy_helper" in q for q in syms_102)

        # Cross-scope and cross-tenant FORCE RLS verification
        for isolated_principal in (principal_a_other_scope, principal_b):
            with app_mgr.scoped_transaction(
                isolated_principal, enable_hnsw_iterative_scan=False
            ) as conn:
                assert pg_repo.list_ref_snapshots(conn, ephemeral_pg_domain) == []
                assert (
                    pg_repo.list_overlay_file_deltas(
                        conn, ephemeral_pg_domain, snap_101.snapshot_id
                    )
                    == []
                )
                assert (
                    pg_repo.resolve_overlay_files(conn, ephemeral_pg_domain, snap_101.snapshot_id)
                    == []
                )
    finally:
        import psycopg

        with psycopg.connect(settings.db_admin_url) as admin_conn:
            for tbl in (
                "overlay_file_deltas",
                "ref_snapshots",
                "ast_edges",
                "ast_symbols",
                "code_files",
                "domain_packs",
            ):
                admin_conn.execute(
                    f"DELETE FROM {tbl} WHERE tenant_id = %s AND domain_id = %s;",
                    (tenant_a, ephemeral_pg_domain),
                )
            admin_conn.commit()


# -----------------------------------------------------------------------------
# 4. .gitattributes Generated/Vendored Down-Weighting
# -----------------------------------------------------------------------------


@pytest.mark.postgres
def test_gitattributes_rules_and_search_downweighting(
    tmp_path: Path,
    ephemeral_pg_domain: str,
) -> None:
    assert normalize_rel_path("./.gitattributes") == ".gitattributes"
    assert normalize_rel_path("./.github/CODEOWNERS") == ".github/CODEOWNERS"
    matcher = GitAttributesMatcher.from_repo_files(
        {
            ".gitattributes": (
                "*.pb.py linguist-generated\n"
                "vendor/** linguist-vendored\n"
                "gen_vendor/** gitlab-generated gitlab-vendored\n"
            ),
            "vendor/.gitattributes": "override_keep.py -linguist-vendored\n",
        }
    )
    c_norm = matcher.classify_file("src/service.py")
    c_gen = matcher.classify_file("proto/api.pb.py")
    c_vend = matcher.classify_file("vendor/sdk/client.py")
    c_both = matcher.classify_file("gen_vendor/schema.py")
    c_override = matcher.classify_file("vendor/override_keep.py")

    assert (c_norm.ranking_weight, c_norm.is_generated, c_norm.is_vendored) == (
        WEIGHT_NORMAL,
        False,
        False,
    )
    assert (c_gen.ranking_weight, c_gen.is_generated, c_gen.is_vendored) == (
        WEIGHT_GENERATED,
        True,
        False,
    )
    assert (c_vend.ranking_weight, c_vend.is_generated, c_vend.is_vendored) == (
        WEIGHT_VENDORED,
        False,
        True,
    )
    assert (c_both.ranking_weight, c_both.is_generated, c_both.is_vendored) == (
        WEIGHT_GENERATED_AND_VENDORED,
        True,
        True,
    )
    assert (c_override.ranking_weight, c_override.is_generated, c_override.is_vendored) == (
        WEIGHT_NORMAL,
        False,
        False,
    )

    # End-to-end down-weighting in GitMerkleIndexer + Zoekt + RRF (InMemory + PG17)
    repo_dir = tmp_path / "attr_repo"
    repo_dir.mkdir()
    (repo_dir / ".gitattributes").write_text(
        "generated/** linguist-generated\nvendor/** linguist-vendored\n",
        encoding="utf-8",
    )
    (repo_dir / "src").mkdir()
    (repo_dir / "vendor").mkdir()
    (repo_dir / "generated").mkdir()

    shared_code = "def compute_checksum_payload(data: str) -> str:\n    return data\n"
    (repo_dir / "src" / "core_checksum.py").write_text(shared_code, encoding="utf-8")
    (repo_dir / "vendor" / "vendored_checksum.py").write_text(shared_code, encoding="utf-8")
    (repo_dir / "generated" / "gen_checksum.py").write_text(shared_code, encoding="utf-8")
    _git(repo_dir, "init", "-b", "main")
    _git(repo_dir, "add", "-A")
    _git(repo_dir, "commit", "-m", "feat: add weighted files")

    indexer = GitMerkleIndexer(use_semantic_embeddings=False)
    mem_repo = InMemoryRLSRepository()
    principal = PrincipalContext(
        user_id="dev", role=RoleType.ENGINEER, allowed_scopes=["public_eng"]
    )
    indexer.sync_in_memory(repo_dir, mem_repo, principal, ephemeral_pg_domain, "acme/attr")

    files_by_path = {
        cf.file_path: cf for cf in mem_repo.list_visible_code_files(principal, ephemeral_pg_domain)
    }
    assert files_by_path["src/core_checksum.py"].ranking_weight == WEIGHT_NORMAL
    assert files_by_path["vendor/vendored_checksum.py"].ranking_weight == WEIGHT_VENDORED
    assert files_by_path["generated/gen_checksum.py"].ranking_weight == WEIGHT_GENERATED

    # Verify Zoekt ranking puts hand-written first, vendored second, generated third
    zoekt = LocalTrigramZoektIndex()
    for cf in files_by_path.values():
        zoekt.index_file(
            tenant_id=principal.tenant_id,
            domain_id=ephemeral_pg_domain,
            repo_name=cf.repo_name,
            file_path=cf.file_path,
            content=cf.raw_content,
            allowed_scopes=cf.allowed_scopes,
            language=cf.language,
            ranking_weight=cf.ranking_weight,
            is_generated=cf.is_generated,
            is_vendored=cf.is_vendored,
        )
    z_hits = zoekt.search("compute_checksum_payload", principal, domain_id=ephemeral_pg_domain)
    assert [h.file_path for h in z_hits.file_matches] == [
        "src/core_checksum.py",
        "vendor/vendored_checksum.py",
        "generated/gen_checksum.py",
    ]

    # Verify InMemory + PG17 search_symbols_hybrid_rrf rank hand-written symbol strictly #1
    q_emb = get_embedding_provider().embed_text("compute_checksum_payload", is_query=True)
    mem_rrf = mem_repo.search_symbols_hybrid_rrf(
        principal, ephemeral_pg_domain, "compute_checksum_payload", q_emb, top_k=5
    )
    assert len(mem_rrf) == 3
    assert "core_checksum" in mem_rrf[0].qualified_name_or_title
    assert mem_rrf[0].normalized_rrf_score > mem_rrf[1].normalized_rrf_score

    settings = ScoutSettings()
    app_mgr = RLSConnectionManager(settings.db_app_url)
    pg_repo = PostgresRLSRepository()
    with app_mgr.scoped_transaction(principal, enable_hnsw_iterative_scan=False) as conn:
        indexer.sync_postgres(repo_dir, conn, pg_repo, principal, ephemeral_pg_domain, "acme/attr")
        pg_rrf = pg_repo.search_symbols_hybrid_rrf(
            conn, ephemeral_pg_domain, "compute_checksum_payload", q_emb, top_k=5
        )
        assert len(pg_rrf) == 3
        assert "core_checksum" in pg_rrf[0].qualified_name_or_title
        assert pg_rrf[0].normalized_rrf_score > pg_rrf[1].normalized_rrf_score


# -----------------------------------------------------------------------------
# 5. CODEOWNERS & Backstage catalog-info.yaml Ownership Graph Extraction
# -----------------------------------------------------------------------------


@pytest.mark.postgres
def test_codeowners_and_backstage_ownership_graph_extraction(
    tmp_path: Path,
    ephemeral_pg_domain: str,
) -> None:
    # 1. CODEOWNERS GitHub + GitLab section + negation/clearing
    co = CodeOwnersMatcher.parse(
        "* @acme/platform-core\n"
        "/services/billing/ @acme/payments-team\n"
        "/services/billing/legacy.py\n"
        "^[Security][2] @acme/sec-ops\n"
        "/services/billing/\n"
    )
    assert co.match_owners("README.md") == ("team:acme/platform-core",)
    assert co.match_owners("services/billing/engine.py") == (
        "team:acme/payments-team",
        "team:acme/sec-ops",
    )
    assert co.match_owners("services/billing/legacy.py") == ("team:acme/sec-ops",)

    # 2. Backstage malformed YAML fails closed with abstained=True and zero ghost entities
    bad_cat = parse_catalog_info_yaml(
        "apiVersion: backstage.io/v1alpha1\nkind: Component\nmetadata: [broken"
    )
    assert bad_cat.abstained is True
    assert bad_cat.entities == ()

    # 3. End-to-end extraction & persistence across GitMerkleIndexer + PG17 + 1-hop traversal
    repo_dir = tmp_path / "owned_repo"
    (repo_dir / ".github").mkdir(parents=True)
    (repo_dir / "services" / "billing").mkdir(parents=True)
    (repo_dir / ".github" / "CODEOWNERS").write_text(
        "* @acme/platform-core\n/services/billing/ @acme/payments-team\n",
        encoding="utf-8",
    )
    (repo_dir / "catalog-info.yaml").write_text(
        "apiVersion: backstage.io/v1alpha1\n"
        "kind: System\n"
        "metadata:\n"
        "  name: checkout-platform\n"
        "spec:\n"
        "  owner: group:default/platform-core\n"
        "---\n"
        "apiVersion: backstage.io/v1alpha1\n"
        "kind: Component\n"
        "metadata:\n"
        "  name: billing-service\n"
        "spec:\n"
        "  owner: group:default/payments-team\n"
        "  system: system:default/checkout-platform\n",
        encoding="utf-8",
    )
    (repo_dir / "services" / "billing" / "invoicer.py").write_text(
        "class InvoiceEngine:\n    def issue(self) -> str:\n        return 'ok'\n",
        encoding="utf-8",
    )
    _git(repo_dir, "init", "-b", "main")
    _git(repo_dir, "add", "-A")
    _git(repo_dir, "commit", "-m", "feat: add ownership graph manifests")

    # Standalone extractor check (team alias deduplication: payments-team + @acme/payments-team)
    extractor = OwnershipGraphExtractor()
    raw_files = {
        ".github/CODEOWNERS": (repo_dir / ".github" / "CODEOWNERS").read_text(encoding="utf-8"),
        "catalog-info.yaml": (repo_dir / "catalog-info.yaml").read_text(encoding="utf-8"),
        "services/billing/invoicer.py": (
            repo_dir / "services" / "billing" / "invoicer.py"
        ).read_text(encoding="utf-8"),
    }
    graph = extractor.extract(
        repo_name="acme/checkout",
        files=raw_files,
        code_symbols_by_path={"services/billing/invoicer.py": ["InvoiceEngine"]},
    )
    team_symbols = [s for s in graph.symbols if s.symbol_kind == "team"]
    assert {s.qualified_name for s in team_symbols} == {
        "ownership.team.acme.payments-team",
        "ownership.team.acme.platform-core",
    }
    assert any(
        e.source_qualified_name == "InvoiceEngine"
        and e.edge_kind == "OWNED_BY_TEAM"
        and e.target_qualified_name == "ownership.team.acme.payments-team"
        for e in graph.edges
    )
    assert any(
        e.source_qualified_name == "ownership.component.billing-service"
        and e.edge_kind == "PART_OF_SYSTEM"
        and e.target_qualified_name == "ownership.system.checkout-platform"
        for e in graph.edges
    )

    # InMemory + PG17 persistence + 1-hop traversal under Forced RLS
    indexer = GitMerkleIndexer(use_semantic_embeddings=False)
    mem_repo = InMemoryRLSRepository()
    principal = PrincipalContext(
        user_id="dev", role=RoleType.ENGINEER, allowed_scopes=["public_eng"]
    )
    indexer.sync_in_memory(repo_dir, mem_repo, principal, ephemeral_pg_domain, "acme/checkout")
    mem_edges = mem_repo.list_ownership_edges(principal, ephemeral_pg_domain)
    assert any(e.edge_kind == "OWNED_BY_TEAM" for e in mem_edges)
    assert any(e.edge_kind == "PART_OF_SYSTEM" for e in mem_edges)

    settings = ScoutSettings()
    app_mgr = RLSConnectionManager(settings.db_app_url)
    pg_repo = PostgresRLSRepository()
    with app_mgr.scoped_transaction(principal, enable_hnsw_iterative_scan=False) as conn:
        indexer.sync_postgres(
            repo_dir, conn, pg_repo, principal, ephemeral_pg_domain, "acme/checkout"
        )
        pg_edges = pg_repo.list_ownership_edges(conn, ephemeral_pg_domain)
        assert any(
            e.edge_kind == "OWNED_BY_TEAM"
            and e.target_qualified_name == "ownership.team.acme.payments-team"
            for e in pg_edges
        )
        assert any(
            e.edge_kind == "PART_OF_SYSTEM"
            and e.target_qualified_name == "ownership.system.checkout-platform"
            for e in pg_edges
        )
        # Verify 1-hop traversal resolves ownership edges from InvoiceEngine
        inv_sym = next(
            s
            for s in pg_repo.search_visible_ast_symbols(conn, ephemeral_pg_domain, "InvoiceEngine")
            if "InvoiceEngine" in s.qualified_name
        )
        neighborhood = pg_repo.traverse_1hop_neighborhood(
            conn, ephemeral_pg_domain, inv_sym.symbol_id
        )
        assert any(n.edge_kind in {"OWNED_BY_TEAM", "PART_OF_SYSTEM"} for n in neighborhood)


def test_scout_wt_conftest_skips_alembic_rebootstrap_when_pg_reachable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Proves scout_wt_ sandboxes skip Alembic/PG re-bootstrap even when PG17 is up."""
    import tests.conftest as root_conftest

    wt_root = tmp_path / "scout_wt_abcd1234"
    wt_root.mkdir()
    monkeypatch.setattr(root_conftest, "_PROJECT_ROOT", wt_root)
    monkeypatch.setattr(root_conftest, "_is_pg_reachable", lambda _url: True)

    def _forbid_bootstrap(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("ensure_pg17_schema_and_seed_corpora must not run inside scout_wt_")

    monkeypatch.setattr(root_conftest, "ensure_pg17_schema_and_seed_corpora", _forbid_bootstrap)

    from types import SimpleNamespace

    fake_request = SimpleNamespace(
        session=SimpleNamespace(
            items=[
                SimpleNamespace(
                    get_closest_marker=lambda m: object() if m == "postgres" else None,
                    fixturenames=(),
                )
            ]
        )
    )
    fixture_fn: Any = root_conftest.bootstrap_pg17_session
    gen = fixture_fn.__wrapped__(fake_request)
    next(gen)
    with pytest.raises(StopIteration):
        next(gen)
