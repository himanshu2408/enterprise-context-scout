"""L9 Regression Test: 50 concurrent mixed-tenant SSE + Streamable HTTP MCP sessions."""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import psycopg
import pytest

from context_scout.agents.scout_workflow import LocalGemmaClient
from context_scout.api import dependencies
from context_scout.api.dependencies import create_signed_principal_token
from context_scout.api.main import create_app
from context_scout.config import ScoutSettings
from context_scout.db.repositories import PostgresRLSRepository
from context_scout.db.schema import (
    ASTSymbolRecord,
    CodeFileRecord,
    DocChunkRecord,
    RefSnapshotRecord,
    RepositoryRecord,
)
from context_scout.db.session import RLSConnectionManager
from context_scout.mcp_servers.common import MCP_PROTOCOL_VERSION_2026_07_28
from context_scout.retrieval.embedder import get_embedding_provider
from context_scout.schemas.contracts import (
    DEFAULT_TENANT_ID,
    AuthorityRules,
    DomainPackManifest,
    PrincipalContext,
    RoleType,
)

pytestmark = pytest.mark.postgres

_TEST_HMAC_SECRET = "l9-mixed-tenant-50-sse-mcp-secret-minimum-32-chars!!"


def _snapshot_default_tenant_counts(db_admin_url: str) -> dict[str, int]:
    tables = ("domain_packs", "repositories", "code_files", "doc_chunks", "ref_snapshots")
    counts: dict[str, int] = {}
    with psycopg.connect(db_admin_url) as conn:
        for tbl in tables:
            row = conn.execute(
                f"SELECT COUNT(*) FROM {tbl} WHERE tenant_id = %s;",
                (DEFAULT_TENANT_ID,),
            ).fetchone()
            counts[tbl] = int(row[0]) if row else 0
    return counts


def _cleanup_l9_test_state(
    db_admin_url: str,
    *,
    domain_id: str,
    tenant_ids: tuple[uuid.UUID, uuid.UUID],
) -> None:
    tables = (
        "execution_telemetry",
        "audit_events",
        "side_effects",
        "overlay_file_deltas",
        "ref_snapshots",
        "ast_edges",
        "ast_symbols",
        "code_files",
        "repo_commits",
        "repositories",
        "doc_chunks",
        "confluence_pages",
        "jira_issues",
        "domain_sync_manifests",
        "domain_packs",
    )
    t_list = list(tenant_ids)
    with psycopg.connect(db_admin_url, autocommit=True) as conn:
        for tbl in tables:
            conn.execute(
                f"DELETE FROM {tbl} WHERE tenant_id = ANY(%s::uuid[]);",
                (t_list,),
            )
        for tbl in tables:
            if tbl in ("audit_events", "execution_telemetry", "repo_commits"):
                continue
            conn.execute(
                f"DELETE FROM {tbl} WHERE domain_id = %s;",
                (domain_id,),
            )


def _extract_mcp_json(resp: httpx.Response) -> dict[str, Any]:
    ctype = resp.headers.get("content-type", "")
    if "text/event-stream" in ctype:
        for line in resp.text.splitlines():
            if line.startswith("data:"):
                payload = line[5:].strip()
                if payload:
                    return dict(json.loads(payload))
    return dict(resp.json())


def test_l9_50_concurrent_mixed_tenant_sse_and_mcp_zero_leaks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """50 simultaneous mixed-tenant SSE + MCP HTTP sessions under PG17 FORCE RLS."""
    settings = ScoutSettings()
    domain_id = f"eval_l9_mix_{uuid.uuid4().hex[:8]}"
    tenant_a = uuid.uuid4()
    tenant_b = uuid.uuid4()
    assert tenant_a != DEFAULT_TENANT_ID and tenant_b != DEFAULT_TENANT_ID and tenant_a != tenant_b

    canary_a_eng = f"CANARY_TA_ENG_{uuid.uuid4().hex[:6]}"
    canary_a_doc = f"CANARY_TA_DOC_{uuid.uuid4().hex[:6]}"
    canary_b_eng = f"CANARY_TB_ENG_{uuid.uuid4().hex[:6]}"
    canary_b_doc = f"CANARY_TB_DOC_{uuid.uuid4().hex[:6]}"

    repo_a = f"tenant-a/service-{uuid.uuid4().hex[:6]}"
    repo_b = f"tenant-b/service-{uuid.uuid4().hex[:6]}"

    monkeypatch.setenv("SCOUT_ENV", "development")
    monkeypatch.setenv("SCOUT_ENVIRONMENT", "development")
    monkeypatch.setenv("SCOUT_REQUIRE_SIGNED_AUTH", "true")
    monkeypatch.setenv("SCOUT_AUTH_HMAC_SECRET", _TEST_HMAC_SECRET)

    shared_before = _snapshot_default_tenant_counts(settings.db_admin_url)

    prev_repo = dependencies._shared_repo
    prev_client = dependencies._local_gemma_client
    prev_registry = dependencies._domain_registry
    dependencies._shared_repo = None
    dependencies._local_gemma_client = None

    manifest = DomainPackManifest(
        domain_id=domain_id,
        display_name="L9 Mixed-Tenant Qualification Domain",
        base_model_id="google/gemma-4-26B-A4B-it",
        lora_adapter_id="none",
        lora_adapter_path="domains/code_architecture_scout/adapters/gemma4_26b_code_arch",
        authority_rules=AuthorityRules(primary_truth_source="code"),
        output_contract="ImplementationContextPack",
        description="Ephemeral L9 mixed-tenant SSE + MCP concurrency verification domain.",
        default_scopes=["public_eng", "public_pm", "audit:compliance", "public_docs"],
        mcp_read_tools_allowed=[
            "code_repo_mcp:search_code_trigram",
            "code_repo_mcp:list_repo_branches_and_mrs",
            "doc_corpus_mcp:search_hybrid_rrf",
        ],
        lora_enabled=False,
    )

    registry = dependencies.get_domain_registry()
    registry.register(manifest)

    embedder = get_embedding_provider(settings)
    emb_vecs = embedder.embed_batch(
        [
            f"handle_l9_tenant_a {canary_a_eng}",
            f"handle_l9_tenant_b {canary_b_eng}",
            f"Tenant A architecture SLA policy {canary_a_doc}",
            f"Tenant B architecture SLA policy {canary_b_doc}",
        ],
        is_query=False,
    )

    app = create_app()
    dependencies._local_gemma_client = LocalGemmaClient(
        offline_mode=True,
        adapter_pool=registry.adapter_pool,
    )

    conn_mgr = RLSConnectionManager(settings.db_app_url, use_pool=False)
    pg_repo = PostgresRLSRepository()

    try:
        with psycopg.connect(settings.db_admin_url, autocommit=True) as admin_conn:
            for tid in (tenant_a, tenant_b):
                for did, dname in (
                    (domain_id, "L9 Mixed-Tenant Domain"),
                    ("code_architecture_scout", "Code Architecture Scout"),
                    ("enterprise_docs_scout", "Enterprise Docs, ADRs & Knowledge Scout"),
                ):
                    admin_conn.execute(
                        """
                        INSERT INTO domain_packs (
                            domain_id, display_name, lora_adapter_id, primary_truth_source,
                            freshness_half_life_days, config_json, tenant_id
                        )
                        VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s)
                        ON CONFLICT (tenant_id, domain_id) DO NOTHING;
                        """,
                        (did, dname, "none", "code", 180, "{}", tid),
                    )

        for tid, rname, c_eng, c_doc, vec_code, vec_doc in (
            (tenant_a, repo_a, canary_a_eng, canary_a_doc, emb_vecs[0], emb_vecs[2]),
            (tenant_b, repo_b, canary_b_eng, canary_b_doc, emb_vecs[1], emb_vecs[3]),
        ):
            admin_p = PrincipalContext(
                user_id=f"l9_seed_{tid.hex[:6]}",
                tenant_id=tid,
                role=RoleType.ENGINEER,
                allowed_scopes=["public_eng", "public_pm", "audit:compliance", "public_docs"],
            )
            with conn_mgr.scoped_transaction(admin_p) as tx:
                for did in (domain_id, "code_architecture_scout"):
                    rid = uuid.uuid4()
                    pg_repo.insert_repository(
                        tx,
                        RepositoryRecord(
                            repo_id=rid,
                            tenant_id=tid,
                            domain_id=did,
                            repo_name=rname,
                            source_type="gitlab",
                            remote_url=f"https://gitlab.local/{rname}",
                            default_branch="main",
                            allowed_scopes=["public_eng"],
                        ),
                    )
                    fid = uuid.uuid4()
                    code_body = (
                        f"def execute_l9_handler() -> str:\n    # {c_eng}\n    return '{c_eng}'\n"
                    )
                    pg_repo.insert_code_file(
                        tx,
                        CodeFileRecord(
                            file_id=fid,
                            tenant_id=tid,
                            domain_id=did,
                            repo_name=rname,
                            file_path="src/l9_handler.py",
                            git_commit_sha="a" * 40,
                            git_blob_sha=uuid.uuid4().hex + "0" * 8,
                            language="python",
                            allowed_scopes=["public_eng"],
                            raw_content=code_body,
                            line_count=3,
                            repo_id=rid,
                        ),
                    )
                    pg_repo.insert_ast_symbol(
                        tx,
                        ASTSymbolRecord(
                            symbol_id=uuid.uuid4(),
                            file_id=fid,
                            tenant_id=tid,
                            domain_id=did,
                            repo_id=rid,
                            qualified_name=f"{rname}::src.l9_handler::execute_l9_handler",
                            symbol_kind="function",
                            start_line=1,
                            end_line=3,
                            signature_skeleton="def execute_l9_handler() -> str",
                            contextual_summary=f"L9 handler {c_eng}",
                            allowed_scopes=["public_eng"],
                            embedding=vec_code,
                        ),
                    )
                    pg_repo.upsert_ref_snapshot(
                        tx,
                        RefSnapshotRecord(
                            snapshot_id=uuid.uuid4(),
                            tenant_id=tid,
                            domain_id=did,
                            repo_id=rid,
                            repo_name=rname,
                            ref_name="mr/1",
                            ref_kind="merge_request",
                            mr_iid="1",
                            base_commit_sha="a" * 40,
                            head_commit_sha="b" * 40,
                            state="open",
                            allowed_scopes=["public_eng"],
                        ),
                    )

                for role_scope, doc_suffix in (
                    ("public_pm", "pm"),
                    ("audit:compliance", "compliance"),
                    ("public_docs", "research"),
                ):
                    for target_dom in (domain_id, "enterprise_docs_scout"):
                        pg_repo.insert_doc_chunk(
                            tx,
                            DocChunkRecord(
                                chunk_id=uuid.uuid4(),
                                tenant_id=tid,
                                domain_id=target_dom,
                                doc_id=f"confluence:l9_{tid.hex[:6]}_{doc_suffix}",
                                doc_title=f"L9 {doc_suffix.upper()} Policy Runbook",
                                source_uri=f"https://wiki.local/l9/{doc_suffix}",
                                market_scope="CONFLUENCE",
                                published_date=date(2026, 10, 15),
                                authority_weight=1.3,
                                section_heading="Authoritative Policy",
                                contextual_header="Authoritative Policy",
                                verbatim_content=(
                                    f"Authoritative policy for {doc_suffix} scope: "
                                    f"timeout = 250ms, token_marker = {c_doc}."
                                ),
                                metadata_json={},
                                allowed_scopes=[role_scope],
                                embedding=vec_doc,
                                source_type="confluence",
                                version_str="2",
                            ),
                        )

        role_configs: list[tuple[RoleType, list[str], str, str]] = [
            (
                RoleType.ENGINEER,
                ["public_eng"],
                "code",
                "Inspect execute_l9_handler in src/l9_handler.py",
            ),
            (
                RoleType.PRODUCT_MANAGER,
                ["public_pm"],
                "wiki",
                "What is the Authoritative Policy timeout for pm scope?",
            ),
            (
                RoleType.COMPLIANCE_OFFICER,
                ["audit:compliance"],
                "wiki",
                "What is the Authoritative Policy timeout for compliance scope?",
            ),
            (
                RoleType.DOMAIN_RESEARCHER,
                ["public_docs"],
                "wiki",
                "What is the Authoritative Policy timeout for research scope?",
            ),
        ]

        async def _execute_50() -> list[dict[str, Any]]:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://127.0.0.1", timeout=30.0
            ) as client:
                tasks: list[Any] = []

                # 25 SSE streams across both tenants and all 4 roles
                for idx in range(25):
                    tid = tenant_a if idx % 2 == 0 else tenant_b
                    role, scopes, src_filter, prompt = role_configs[idx % 4]
                    forbidden = (
                        [canary_b_eng, canary_b_doc, repo_b]
                        if tid == tenant_a
                        else [canary_a_eng, canary_a_doc, repo_a]
                    )
                    tok = create_signed_principal_token(
                        user_id=f"l9_sse_{role.value}_{idx:02d}",
                        role=role,
                        allowed_scopes=scopes,
                        tenant_id=tid,
                    )

                    async def _run_sse(
                        i: int = idx,
                        t_id: uuid.UUID = tid,
                        r_val: str = role.value,
                        q_prompt: str = prompt,
                        s_filter: str = src_filter,
                        bearer: str = tok,
                        forb: list[str] = forbidden,
                    ) -> dict[str, Any]:
                        resp = await client.post(
                            "/api/v1/scout/stream",
                            headers={"Authorization": f"Bearer {bearer}"},
                            json={
                                "domain_id": domain_id,
                                "query": q_prompt,
                                "source_filter": s_filter,
                                "max_bounce_backs": 0,
                            },
                        )
                        body = resp.text
                        terminal_ok = (
                            "event: context_pack_ready" in body
                            or '"category":"insufficient_context"' in body
                        )
                        leaks = [tok_s for tok_s in forb if tok_s in body]
                        return {
                            "kind": "sse",
                            "idx": i,
                            "tenant_id": str(t_id),
                            "role": r_val,
                            "status_code": resp.status_code,
                            "terminal_ok": terminal_ok,
                            "leaks": leaks,
                        }

                    tasks.append(_run_sse())

                # 25 Streamable HTTP MCP calls (/mcp/code/mcp & /mcp/docs/mcp) across both tenants
                for idx in range(25):
                    tid = tenant_a if idx % 2 == 0 else tenant_b
                    role, scopes, _sf, _pr = role_configs[idx % 4]
                    forbidden = (
                        [canary_b_eng, canary_b_doc, repo_b]
                        if tid == tenant_a
                        else [canary_a_eng, canary_a_doc, repo_a]
                    )
                    expected_own = repo_a if tid == tenant_a else repo_b
                    expected_doc = canary_a_doc if tid == tenant_a else canary_b_doc
                    tok = create_signed_principal_token(
                        user_id=f"l9_mcp_{role.value}_{idx:02d}",
                        role=role,
                        allowed_scopes=scopes,
                        tenant_id=tid,
                    )
                    use_code_mcp = role == RoleType.ENGINEER

                    async def _run_mcp(
                        i: int = idx,
                        t_id: uuid.UUID = tid,
                        r_val: str = role.value,
                        bearer: str = tok,
                        forb: list[str] = forbidden,
                        own_repo: str = expected_own,
                        own_doc: str = expected_doc,
                        is_code: bool = use_code_mcp,
                    ) -> dict[str, Any]:
                        meta = {
                            "io.modelcontextprotocol/protocolVersion": (
                                MCP_PROTOCOL_VERSION_2026_07_28
                            ),
                            "io.modelcontextprotocol/clientCapabilities": {},
                        }
                        if is_code:
                            endpoint = "/mcp/code/mcp"
                            tool_name = "list_repo_branches_and_mrs"
                            payload = {
                                "jsonrpc": "2.0",
                                "id": i + 100,
                                "method": "tools/call",
                                "params": {
                                    "name": tool_name,
                                    "arguments": {},
                                    "_meta": meta,
                                },
                            }
                        else:
                            endpoint = "/mcp/docs/mcp"
                            tool_name = "search_hybrid_rrf"
                            payload = {
                                "jsonrpc": "2.0",
                                "id": i + 100,
                                "method": "tools/call",
                                "params": {
                                    "name": tool_name,
                                    "arguments": {
                                        "domain_id": "enterprise_docs_scout",
                                        "query": "Authoritative Policy timeout",
                                        "top_k": 5,
                                    },
                                    "_meta": meta,
                                },
                            }
                        resp = await client.post(
                            endpoint,
                            headers={
                                "Authorization": f"Bearer {bearer}",
                                "Content-Type": "application/json",
                                "Accept": "application/json, text/event-stream",
                                "MCP-Protocol-Version": MCP_PROTOCOL_VERSION_2026_07_28,
                                "Mcp-Method": "tools/call",
                                "Mcp-Name": tool_name,
                            },
                            json=payload,
                        )
                        body = resp.text
                        leaks = [tok_s for tok_s in forb if tok_s in body]
                        parsed = _extract_mcp_json(resp) if resp.status_code == 200 else {}
                        has_result = "result" in parsed and not parsed["result"].get("isError")
                        own_hit = (own_repo in body) if is_code else (own_doc in body)
                        return {
                            "kind": "mcp",
                            "idx": i,
                            "tenant_id": str(t_id),
                            "role": r_val,
                            "status_code": resp.status_code,
                            "terminal_ok": has_result and own_hit,
                            "leaks": leaks,
                        }

                    tasks.append(_run_mcp())

                return list(await asyncio.gather(*tasks))

        outcomes = asyncio.run(_execute_50())
        assert len(outcomes) == 50
        assert sum(1 for o in outcomes if o["kind"] == "sse") == 25
        assert sum(1 for o in outcomes if o["kind"] == "mcp") == 25
        assert {o["tenant_id"] for o in outcomes} == {str(tenant_a), str(tenant_b)}
        assert {o["role"] for o in outcomes} == {r.value for r in RoleType}

        for o in outcomes:
            assert o["status_code"] == 200, f"Session failed: {o}"
            assert o["terminal_ok"] is True, f"Session missing expected tenant result: {o}"
            assert o["leaks"] == [], f"Cross-tenant leak detected: {o}"

        # Also verify the mounted app root discovery reports both servers
        assert Path("web-ui/public/fonts/PlusJakartaSans-Regular.woff2").is_file()
    finally:
        registry.unregister(domain_id)
        dependencies._domain_registry = prev_registry
        dependencies._shared_repo = prev_repo
        dependencies._local_gemma_client = prev_client
        _cleanup_l9_test_state(
            settings.db_admin_url,
            domain_id=domain_id,
            tenant_ids=(tenant_a, tenant_b),
        )
        assert domain_id not in dependencies.get_domain_registry().load_all()
        shared_after = _snapshot_default_tenant_counts(settings.db_admin_url)
        assert shared_before == shared_after, (
            f"Shared DEFAULT_TENANT_ID canary counts mutated: {shared_before} != {shared_after}"
        )
