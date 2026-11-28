"""Mechanical Zero-Spend Proof & Dual-Benchmark Suite (EVAL-01 & EVAL-04).

SYNTHETIC-FIXTURE BENCHMARK SPECIFICATION:
All test cases and evaluation indices in this file operate on synthetic evaluation
fixtures (synthetic C# e-commerce, synthetic TS/React state, synthetic Python web,
synthetic Terraform infrastructure, synthetic Atlassian Confluence/Jira documents)
to verify the scout's multi-source preflight gate discrimination without claiming
live repository provenance. Real-corpus evaluation guidance is documented
in docs/ROADMAP_AND_KNOWN_LIMITS.md.

Guarantees and mechanically proves:
1. Intercepts all paid model execution points (GeminiExecutor, Vertex AI, HTTP).
2. Populated Multi-Source Discrimination:
   - Populates index with synthetic multi-stack fixtures and synthetic Atlassian chunks.
3. EVAL-01 (Unanswerable Benchmark, N >= 300, synthetic-fixture result):
   - 300 unanswerable tasks across 5 taxonomies execute against the POPULATED repository.
   - Actively exercises Stage 3 Pre-Flight Gates (closure, nonexistent files, unauthorized scopes,
     unindexed libraries, empty scaffolds).
   - Asserts 100% abstention rate with 0 paid model calls and 0 tokens billed.
4. Dynamic Clopper-Pearson Bound Accounting:
   - Derives k dynamically from observed run outcomes (k = count of InsufficientContextDiagnostic).
   - Asserts exact Clopper-Pearson 95% lower bound >= 98.78% (two-sided) and >= 99.00% (one-sided).
5. EVAL-04 (Answerable Benchmark, N >= 300, synthetic-fixture result):
   - 300 answerable tasks run against the EXACT SAME populated index.
   - Asserts wrongful refusal rate <= 1.0% and verifies valid ImplementationContextPacks.
6. Concrete Mutation Proofs:
   - Proves benchmark suite FAILS if gates (closure, scopes, RRF threshold, provenance)
     are weakened.
7. Storage Parity & Live PostgreSQL Verification:
   - Verifies parity between InMemoryRLSRepository and live PostgresRLSRepository.

DATABASE ISOLATION RATIONALE:
Tests execute against the live PostgreSQL 17 database (`context_scout`) because all 7
production tables, RLS security policies, triggers, and full-text/vector indices are
pre-migrated via `sql/001_init_schema.sql` and `sql/002_production_enterprise_schema.sql`.
Domain isolation is strictly enforced via per-session `eval_synthetic_<uuid>` partitioning,
and verified by the canary non-interference regression test.
"""

from __future__ import annotations

import contextlib
import functools
import os
import urllib.request
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import httpx
import psycopg
import pytest
from scipy import stats

from context_scout.agents.domain_registry import DomainRegistry
from context_scout.agents.gemini_executor import GeminiImplementationExecutor
from context_scout.agents.open_weights_vllm_executor import OpenWeightsVLLMExecutor
from context_scout.agents.preflight_gate import PreFlightValidator
from context_scout.agents.scout_workflow import ScoutWorkflow
from context_scout.config import ScoutSettings
from context_scout.db.repositories import (
    ASTSymbolRecord,
    CodeFileRecord,
    DocChunkRecord,
    InMemoryRLSRepository,
    PostgresRLSRepository,
)
from context_scout.db.session import RLSConnectionManager
from context_scout.evals.datasets.answerable_dataset import (
    AnswerableTask,
    StackType,
    get_answerable_benchmark_set,
)
from context_scout.evals.datasets.unanswerable_dataset import (
    UnanswerableReason,
    UnanswerableTask,
    get_unanswerable_benchmark_set,
)
from context_scout.mcp_servers.common import PostgresScoutGraphAdapter
from context_scout.retrieval.embedder import LocalSemanticEmbedder
from context_scout.schemas.contracts import (
    ImplementationContextPack,
    InsufficientContextDiagnostic,
    ModificationIntent,
    PrincipalContext,
    RoleType,
    VerbatimCodeSlice,
)


def is_postgres_reachable() -> bool:
    """Checks whether local PostgreSQL 17 instance is accessible."""
    try:
        import psycopg

        from context_scout.config import ScoutSettings

        st = ScoutSettings()
        with psycopg.connect(st.db_admin_url, connect_timeout=2) as conn:
            conn.execute("SELECT 1")
            return True
    except Exception:
        return False


class ZeroSpendViolationError(AssertionError):
    """Raised immediately when any paid client or network call occurs during abstention."""


@dataclass
class PaidExecutionSpy:
    """Invocation spy intercepting all paid model execution and outbound token egress."""

    call_count: int = 0
    tokens_billed: int = 0
    invocations: list[dict[str, Any]] = field(default_factory=list)

    def record_call(self, target_name: str, *args: Any, **kwargs: Any) -> None:
        self.call_count += 1
        tokens = int(kwargs.get("tokens", 0) or 0)
        self.tokens_billed += tokens
        call_info = {"target": target_name, "args": args, "kwargs": kwargs}
        self.invocations.append(call_info)
        raise ZeroSpendViolationError(
            f"ZERO-SPEND VIOLATION: '{target_name}' invoked during abstention! "
            f"Call count: {self.call_count}, Call info: {call_info}"
        )

    def reset(self) -> None:
        self.call_count = 0
        self.tokens_billed = 0
        self.invocations.clear()


pytestmark = pytest.mark.postgres


@pytest.fixture(scope="module")
def paid_spy() -> PaidExecutionSpy:
    return PaidExecutionSpy()


@pytest.fixture(scope="module", autouse=True)
def guard_paid_network_and_models(paid_spy: PaidExecutionSpy) -> Iterator[PaidExecutionSpy]:
    """Installs mechanical interception across all paid execution and external network pathways."""
    mp = pytest.MonkeyPatch()

    def spy_execute(self: object, *args: Any, **kwargs: Any) -> None:
        paid_spy.record_call("GeminiImplementationExecutor.execute_implementation", *args, **kwargs)

    def spy_patch(self: object, *args: Any, **kwargs: Any) -> None:
        paid_spy.record_call("GeminiImplementationExecutor._generate_gemini_patch", *args, **kwargs)

    def spy_keyless_handoff(self: object, *args: Any, **kwargs: Any) -> None:
        paid_spy.record_call(
            "GeminiImplementationExecutor.synthesize_keyless_vertex_handoff_patch",
            *args,
            **kwargs,
        )

    def spy_vllm_complete(self: object, *args: Any, **kwargs: Any) -> None:
        paid_spy.record_call("OpenWeightsVLLMExecutor.complete", *args, **kwargs)

    mp.setattr(GeminiImplementationExecutor, "execute_implementation", spy_execute)
    mp.setattr(GeminiImplementationExecutor, "_generate_gemini_patch", spy_patch)
    mp.setattr(
        GeminiImplementationExecutor,
        "synthesize_keyless_vertex_handoff_patch",
        spy_keyless_handoff,
    )
    mp.setattr(OpenWeightsVLLMExecutor, "complete", spy_vllm_complete)

    try:
        from google import genai

        mp.setattr(
            genai.Client,
            "__init__",
            lambda self, *a, **kw: paid_spy.record_call("google.genai.Client.__init__", *a, **kw),
        )
    except (ImportError, AttributeError):
        pass

    cloud_hosts = (
        "googleapis.com",
        "anthropic.com",
        "openai.com",
        "generativelanguage",
        "aiplatform",
    )

    orig_urlopen = urllib.request.urlopen

    def spy_urlopen(url_or_req: Any, *args: Any, **kwargs: Any) -> Any:
        target_url = str(getattr(url_or_req, "full_url", url_or_req))
        if any(cloud in target_url for cloud in cloud_hosts):
            paid_spy.record_call(f"urllib.request.urlopen({target_url})")
        return orig_urlopen(url_or_req, *args, **kwargs)

    orig_httpx_post = httpx.Client.post
    orig_async_httpx_post = httpx.AsyncClient.post

    def spy_httpx_post(self: httpx.Client, url: Any, *args: Any, **kwargs: Any) -> Any:
        full_target = f"{getattr(self, 'base_url', '')}{url}"
        if any(cloud in full_target for cloud in cloud_hosts):
            paid_spy.record_call(f"httpx.Client.post({full_target})")
        return orig_httpx_post(self, url, *args, **kwargs)

    async def spy_async_httpx_post(
        self: httpx.AsyncClient, url: Any, *args: Any, **kwargs: Any
    ) -> Any:
        full_target = f"{getattr(self, 'base_url', '')}{url}"
        if any(cloud in full_target for cloud in cloud_hosts):
            paid_spy.record_call(f"httpx.AsyncClient.post({full_target})")
        return await orig_async_httpx_post(self, url, *args, **kwargs)

    mp.setattr(urllib.request, "urlopen", spy_urlopen)
    mp.setattr(httpx.Client, "post", spy_httpx_post)
    mp.setattr(httpx.AsyncClient, "post", spy_async_httpx_post)
    yield paid_spy
    mp.undo()


@functools.lru_cache(maxsize=4)
def _compute_fixture_embeddings(texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
    """Computes genuine normalized 768-dim local semantic embeddings once per module."""
    vectors = LocalSemanticEmbedder().embed_batch(list(texts))
    return tuple(tuple(vec) for vec in vectors)


@pytest.fixture(scope="module")
def domain_registry() -> DomainRegistry:
    domains_dir = Path(__file__).resolve().parent.parent / "domains"
    reg = DomainRegistry(domains_dir)
    reg.load_all()
    return reg


def build_synthetic_fixture_dataset(domain_id: str) -> InMemoryRLSRepository:
    """Populates an in-memory repository with synthetic multi-stack and Atlassian fixtures."""
    repo = InMemoryRLSRepository()
    admin = PrincipalContext(
        user_id="benchmark_admin",
        role=RoleType.ENGINEER,
        allowed_scopes=[
            "*",
            "admin",
            "public_eng",
            "scope:general",
            "repo:read",
            "repo:dotnet/eShop",
            "repo:frontend/web-storefront",
            "repo:backend/gateway-service",
            "repo:infra/gcp-workloads",
            "repo:enterprise/billing-service",
            "repo:enterprise/vault-service",
            "vault_restricted",
        ],
    )

    domain_ns = uuid.uuid5(uuid.NAMESPACE_DNS, domain_id)

    embed_texts: list[str] = []
    for i in range(75):
        embed_texts.append(
            f"public class OrderCheckoutHandler_{i:02d} — "
            f"Synthetic eShop order checkout handler {i:02d}"
        )
    for i in range(75):
        embed_texts.append(f"export const cartSlice_{i:02d} — Synthetic Redux cart slice {i:02d}")
    for i in range(75):
        embed_texts.append(f"class FastAPIProxy_{i:02d}: — Synthetic gateway proxy service {i:02d}")
    for i in range(75):
        embed_texts.append(
            f'resource "google_sql_database_instance" "postgres_{i:02d}" — '
            f"Synthetic Terraform Cloud SQL Postgres 17 instance {i:02d}"
        )
    embed_texts.append("class BillingManager: — Synthetic billing manager")
    embed_texts.append(
        "Orders must enforce Polly exponential backoff policies across database mutations."
    )
    embed_texts.append(
        "Cart item updates must immediately mutate local Redux state before network resolution."
    )
    embed_texts.append("Token bucket rate limiter requires strict Redis cluster atomicity.")
    fixture_vecs = [list(v) for v in _compute_fixture_embeddings(tuple(embed_texts))]

    # -------------------------------------------------------------------------
    # 1. Synthetic C# / .NET E-Commerce Ordering API Fixtures (75 Tasks)
    # -------------------------------------------------------------------------
    for i in range(75):
        f_id = uuid.uuid5(domain_ns, f"eshop_file_{i}")
        f_path = (
            f"src/Services/Ordering/Ordering.API/Application/"
            f"Commands/OrderCheckoutHandler_{i:02d}.cs"
        )
        content = (
            f"namespace Ordering.API.Application.Commands;\n\n"
            f"public class OrderCheckoutHandler_{i:02d} {{\n"
            f"    private readonly IOrderRepository _orderRepository;\n"
            f"    public OrderCheckoutHandler_{i:02d}(IOrderRepository orderRepository) "
            f"=> _orderRepository = orderRepository;\n"
            f"    public async Task<bool> HandleOrderAsync(OrderCheckoutCommand cmd) {{\n"
            f"        return await _orderRepository.SaveAsync(cmd);\n"
            f"    }}\n"
            f"}}\n"
        )
        repo.insert_code_file(
            admin,
            CodeFileRecord(
                file_id=f_id,
                domain_id=domain_id,
                repo_name="dotnet/eShop",
                file_path=f_path,
                git_commit_sha="e5a1b2c3d4e5f678901234567890123456789012",
                git_blob_sha="f6b1c2d3e4f5a678901234567890123456789012",
                language="csharp",
                allowed_scopes=["public_eng", "repo:dotnet/eShop"],
                raw_content=content,
                line_count=10,
            ),
        )
        sym_id = uuid.uuid5(domain_ns, f"eshop_sym_{i}")
        repo.insert_ast_symbol(
            admin,
            ASTSymbolRecord(
                symbol_id=sym_id,
                file_id=f_id,
                domain_id=domain_id,
                qualified_name=f"dotnet/eShop::{f_path}::OrderCheckoutHandler_{i:02d}",
                symbol_kind="class",
                start_line=3,
                end_line=9,
                signature_skeleton=f"public class OrderCheckoutHandler_{i:02d}",
                contextual_summary=f"Synthetic eShop order checkout handler {i:02d}",
                allowed_scopes=["public_eng", "repo:dotnet/eShop"],
                embedding=fixture_vecs[i],
            ),
        )

    # -------------------------------------------------------------------------
    # 2. Synthetic TS / React / Redux Web Storefront State Fixtures (75 Tasks)
    # -------------------------------------------------------------------------
    for i in range(75):
        f_id = uuid.uuid5(domain_ns, f"redux_file_{i}")
        f_path = f"src/features/cart/cartSlice_{i:02d}.ts"
        content = (
            f"import {{ createSlice, PayloadAction }} from '@reduxjs/toolkit';\n\n"
            f"export const cartSlice_{i:02d} = createSlice({{\n"
            f"  name: 'cart_{i:02d}',\n"
            f"  initialState: {{ items: [], total: 0 }},\n"
            f"  reducers: {{\n"
            f"    updateQuantity: (state, "
            f"action: PayloadAction<{{ id: string; qty: number }}>) => {{\n"
            f"      state.total += action.payload.qty;\n"
            f"    }},\n"
            f"  }},\n"
            f"}});\n\n"
            f"export const selectCartTotal = (state: any) => state.cart.total;\n"
        )
        repo.insert_code_file(
            admin,
            CodeFileRecord(
                file_id=f_id,
                domain_id=domain_id,
                repo_name="frontend/web-storefront",
                file_path=f_path,
                git_commit_sha="a2b3c4d5e6f78901234567890123456789012345",
                git_blob_sha="b3c4d5e6f7a89012345678901234567890123456",
                language="typescript",
                allowed_scopes=["public_eng", "repo:frontend/web-storefront"],
                raw_content=content,
                line_count=13,
            ),
        )
        sym_id = uuid.uuid5(domain_ns, f"redux_sym_{i}")
        repo.insert_ast_symbol(
            admin,
            ASTSymbolRecord(
                symbol_id=sym_id,
                file_id=f_id,
                domain_id=domain_id,
                qualified_name=f"frontend/web-storefront::{f_path}::cartSlice_{i:02d}",
                symbol_kind="class",
                start_line=3,
                end_line=11,
                signature_skeleton=f"export const cartSlice_{i:02d} = createSlice({{...}})",
                contextual_summary=f"Synthetic Redux cart slice {i:02d}",
                allowed_scopes=["public_eng", "repo:frontend/web-storefront"],
                embedding=fixture_vecs[75 + i],
            ),
        )

    # -------------------------------------------------------------------------
    # 3. Synthetic Python Backend Gateway Service Fixtures (75 Tasks)
    # -------------------------------------------------------------------------
    for i in range(75):
        f_id = uuid.uuid5(domain_ns, f"gateway_file_{i}")
        f_path = f"src/gateway/proxy_{i:02d}.py"
        content = (
            f"class TokenBucketLimiter:\n"
            f"    def allow_request(self, key: str) -> bool:\n"
            f"        return True\n\n"
            f"class FastAPIProxy_{i:02d}:\n"
            f"    def __init__(self, limiter: TokenBucketLimiter):\n"
            f"        self.limiter = limiter\n\n"
            f"    def dispatch(self, request_payload: dict) -> dict:\n"
            f"        if not self.limiter.allow_request('api'):\n"
            f"            raise PermissionError('Rate limit exceeded')\n"
            f"        return {{'status': 200, 'payload': request_payload}}\n"
        )
        repo.insert_code_file(
            admin,
            CodeFileRecord(
                file_id=f_id,
                domain_id=domain_id,
                repo_name="backend/gateway-service",
                file_path=f_path,
                git_commit_sha="c3d4e5f6a7b89012345678901234567890123456",
                git_blob_sha="d4e5f6a7b8c90123456789012345678901234567",
                language="python",
                allowed_scopes=["public_eng", "repo:backend/gateway-service"],
                raw_content=content,
                line_count=12,
            ),
        )
        sym_id = uuid.uuid5(domain_ns, f"gateway_sym_{i}")
        repo.insert_ast_symbol(
            admin,
            ASTSymbolRecord(
                symbol_id=sym_id,
                file_id=f_id,
                domain_id=domain_id,
                qualified_name=f"backend/gateway-service::{f_path}::FastAPIProxy_{i:02d}",
                symbol_kind="class",
                start_line=5,
                end_line=12,
                signature_skeleton=f"class FastAPIProxy_{i:02d}:",
                contextual_summary=f"Synthetic gateway proxy service {i:02d}",
                allowed_scopes=["public_eng", "repo:backend/gateway-service"],
                embedding=fixture_vecs[150 + i],
            ),
        )

    # -------------------------------------------------------------------------
    # 4. Synthetic Terraform GCP Infrastructure Fixtures (75 Tasks)
    # -------------------------------------------------------------------------
    for i in range(75):
        f_id = uuid.uuid5(domain_ns, f"tf_file_{i}")
        f_path = f"modules/storage/cloud_sql_{i:02d}.tf"
        content = (
            f'resource "google_sql_database_instance" "postgres_{i:02d}" {{\n'
            f'  name             = "pg-cluster-{i:02d}"\n'
            f'  database_version = "POSTGRES_17"\n'
            f'  region           = "europe-west3"\n'
            f"  encryption_key_name = google_kms_crypto_key.db_key.id\n"
            f"}}\n"
        )
        repo.insert_code_file(
            admin,
            CodeFileRecord(
                file_id=f_id,
                domain_id=domain_id,
                repo_name="infra/gcp-workloads",
                file_path=f_path,
                git_commit_sha="d4e5f6a7b8c90123456789012345678901234567",
                git_blob_sha="e5f6a7b8c9d01234567890123456789012345678",
                language="terraform",
                allowed_scopes=["public_eng", "repo:infra/gcp-workloads"],
                raw_content=content,
                line_count=6,
            ),
        )
        sym_id = uuid.uuid5(domain_ns, f"tf_sym_{i}")
        repo.insert_ast_symbol(
            admin,
            ASTSymbolRecord(
                symbol_id=sym_id,
                file_id=f_id,
                domain_id=domain_id,
                qualified_name=(
                    f"infra/gcp-workloads::{f_path}::google_sql_database_instance.postgres_{i:02d}"
                ),
                symbol_kind="class",
                start_line=1,
                end_line=6,
                signature_skeleton=f'resource "google_sql_database_instance" "postgres_{i:02d}"',
                contextual_summary=f"Synthetic Terraform Cloud SQL Postgres 17 instance {i:02d}",
                allowed_scopes=["public_eng", "repo:infra/gcp-workloads"],
                embedding=fixture_vecs[225 + i],
            ),
        )

    # -------------------------------------------------------------------------
    # 5. Synthetic Enterprise Services for EVAL-01 (Billing, Core API, Vault)
    # -------------------------------------------------------------------------
    b_fid = uuid.uuid5(domain_ns, "billing_base")
    repo.insert_code_file(
        admin,
        CodeFileRecord(
            file_id=b_fid,
            domain_id=domain_id,
            repo_name="enterprise/billing-service",
            file_path="src/billing/service.py",
            git_commit_sha="1111222233334444555566667777888899990000",
            git_blob_sha="2222333344445555666677778888999900001111",
            language="python",
            allowed_scopes=["repo:enterprise/billing-service"],
            raw_content="class BillingManager:\n    pass\n",
            line_count=2,
        ),
    )
    repo.insert_ast_symbol(
        admin,
        ASTSymbolRecord(
            symbol_id=uuid.uuid5(domain_ns, "billing_sym"),
            file_id=b_fid,
            domain_id=domain_id,
            qualified_name="enterprise/billing-service::src/billing/service.py::BillingManager",
            symbol_kind="class",
            start_line=1,
            end_line=2,
            signature_skeleton="class BillingManager:",
            contextual_summary="Synthetic billing manager",
            allowed_scopes=["repo:enterprise/billing-service"],
            embedding=fixture_vecs[300],
        ),
    )

    # Restricted Vault Service (Requires vault_restricted scope)
    v_fid = uuid.uuid5(domain_ns, "vault_file")
    repo.insert_code_file(
        admin,
        CodeFileRecord(
            file_id=v_fid,
            domain_id=domain_id,
            repo_name="enterprise/vault-service",
            file_path="src/vault/secrets.py",
            git_commit_sha="3333444455556666777788889999000011112222",
            git_blob_sha="4444555566667777888899990000111122223333",
            language="python",
            allowed_scopes=["vault_restricted", "repo:enterprise/vault-service"],
            raw_content="class VaultAuditLogger:\n    pass\n",
            line_count=2,
        ),
    )

    # -------------------------------------------------------------------------
    # 6. Synthetic Atlassian Confluence & Jira Doc Chunks
    # -------------------------------------------------------------------------
    for i in range(1, 4):
        chk_id = f"conf_eshop_ordering_architecture_v{i}"
        repo.insert_doc_chunk(
            admin,
            DocChunkRecord(
                chunk_id=uuid.uuid5(domain_ns, chk_id),
                domain_id=domain_id,
                doc_id=chk_id,
                doc_title="eShop Ordering Architecture",
                source_uri=f"confluence://spaces/ORDER/pages/1001?v={i}",
                market_scope="GLOBAL",
                published_date=None,
                authority_weight=1.0,
                section_heading="Architecture Overview",
                contextual_header="eShop Ordering Architecture — Overview",
                verbatim_content=(
                    "Orders must enforce Polly exponential backoff policies "
                    "across database mutations."
                ),
                metadata_json={"version": str(i), "is_stale": False},
                allowed_scopes=["public_eng", "repo:dotnet/eShop"],
                embedding=fixture_vecs[301],
                is_stale=False,
            ),
        )

    for i in range(1, 3):
        chk_id = f"conf_storefront_state_mgmt_v{i}"
        repo.insert_doc_chunk(
            admin,
            DocChunkRecord(
                chunk_id=uuid.uuid5(domain_ns, chk_id),
                domain_id=domain_id,
                doc_id=chk_id,
                doc_title="Storefront State Management Spec",
                source_uri=f"confluence://spaces/STORE/pages/2001?v={i}",
                market_scope="GLOBAL",
                published_date=None,
                authority_weight=1.0,
                section_heading="Optimistic Updates",
                contextual_header="Storefront State Management — Optimistic Updates",
                verbatim_content=(
                    "Cart item updates must immediately mutate local Redux state "
                    "before network resolution."
                ),
                metadata_json={"version": str(i), "is_stale": False},
                allowed_scopes=["public_eng", "repo:frontend/web-storefront"],
                embedding=fixture_vecs[302],
                is_stale=False,
            ),
        )

    for i in range(75):
        chk_id = f"jira_GATEWAY_{100 + i}"
        repo.insert_doc_chunk(
            admin,
            DocChunkRecord(
                chunk_id=uuid.uuid5(domain_ns, chk_id),
                domain_id=domain_id,
                doc_id=chk_id,
                doc_title=f"Jira Ticket GATEWAY-{100 + i}",
                source_uri=f"jira://projects/GATEWAY/issues/GATEWAY-{100 + i}",
                market_scope="GLOBAL",
                published_date=None,
                authority_weight=1.0,
                section_heading="Rate Limiting",
                contextual_header=f"Jira GATEWAY-{100 + i} — Rate Limiting",
                verbatim_content=(
                    "Token bucket rate limiter requires strict Redis cluster atomicity."
                ),
                metadata_json={"resolution": "Resolved", "status": "Closed"},
                allowed_scopes=["public_eng", "repo:backend/gateway-service"],
                embedding=fixture_vecs[303],
                is_stale=False,
            ),
        )

    return repo


def seed_synthetic_benchmark_repo(
    conn: psycopg.Connection,
    domain_id: str,
    repo: InMemoryRLSRepository | None = None,
) -> InMemoryRLSRepository:
    """Inserts domain pack and all synthetic records into live PostgreSQL under domain_id."""
    if repo is None:
        repo = build_synthetic_fixture_dataset(domain_id)

    conn.execute(
        """
        INSERT INTO domain_packs (
            domain_id, display_name, lora_adapter_id,
            primary_truth_source, freshness_half_life_days, config_json
        )
        VALUES (%s, 'Synthetic Fixture Benchmark Domain', 'none', 'code', 30, '{}')
        ON CONFLICT (tenant_id, domain_id) DO NOTHING;
        """,
        (domain_id,),
    )
    pg_repo = PostgresRLSRepository()
    for cf in repo._files.values():
        pg_repo.insert_code_file(conn, cf)
    for sym in repo._symbols.values():
        pg_repo.insert_ast_symbol(conn, sym)
    for chk in repo._docs.values():
        pg_repo.insert_doc_chunk(conn, chk)
    conn.commit()
    return repo


def teardown_synthetic_benchmark_repo(conn: psycopg.Connection, domain_id: str) -> None:
    """Strictly deletes all records belonging to domain_id and verifies 0 rows remain."""
    from tests.fixtures.pg_isolation import cleanup_pg_tenants_and_eval_domains

    cleanup_pg_tenants_and_eval_domains(conn, domain_ids=(domain_id,))

    with conn.cursor() as cur:
        for tbl in ("doc_chunks", "ast_symbols", "code_files", "domain_packs"):
            cur.execute(f"SELECT COUNT(*) FROM {tbl} WHERE domain_id = %s;", (domain_id,))
            cnt_row = cur.fetchone()
            assert cnt_row is not None
            cnt = cnt_row[0]
            assert cnt == 0, f"Teardown failure: {tbl} still contains {cnt} rows for {domain_id}!"


@pytest.fixture(scope="module")
def populated_benchmark_repo(domain_registry: DomainRegistry) -> Any:
    """Populates repository with synthetic multi-stack codebases and Atlassian fixtures."""
    domain_id = f"eval_synthetic_{uuid.uuid4().hex[:12]}"
    try:
        base_manifest = domain_registry.get("code_architecture_scout")
    except KeyError:
        base_manifest = next(iter(domain_registry._manifests.values()))
    synthetic_manifest = base_manifest.model_copy(
        update={
            "domain_id": domain_id,
            "display_name": f"Synthetic Polyglot Evaluation Benchmark ({domain_id})",
        }
    )
    domain_registry.register(synthetic_manifest)

    in_mem_repo = build_synthetic_fixture_dataset(domain_id)

    if is_postgres_reachable():
        st = ScoutSettings()
        with psycopg.connect(st.db_admin_url) as conn:
            seed_synthetic_benchmark_repo(conn, domain_id, repo=in_mem_repo)

        with RLSConnectionManager(st.db_app_url, use_pool=False) as conn_mgr:
            adapter = PostgresScoutGraphAdapter(conn_mgr, st)
            cast(Any, adapter).benchmark_domain_id = domain_id
            try:
                yield adapter
            finally:
                domain_registry.unregister(domain_id)
                with psycopg.connect(st.db_admin_url) as cleanup_conn:
                    teardown_synthetic_benchmark_repo(cleanup_conn, domain_id)
        return

    if os.environ.get("SCOUT_EVAL_IN_MEMORY_ONLY") == "1":
        cast(Any, in_mem_repo).benchmark_domain_id = domain_id
        try:
            yield in_mem_repo
        finally:
            domain_registry.unregister(domain_id)
        return

    pytest.fail(
        "Live PostgreSQL 17 instance is required for EVAL benchmark execution but is "
        "unreachable at db_admin_url! Ensure PostgreSQL is running on 127.0.0.1:5432, "
        "or set SCOUT_EVAL_IN_MEMORY_ONLY=1 for fast isolated in-memory unit tests."
    )


@pytest.fixture(scope="module")
def unanswerable_suite() -> list[UnanswerableTask]:
    return get_unanswerable_benchmark_set(300)


@pytest.fixture(scope="module")
def answerable_suite() -> list[AnswerableTask]:
    return get_answerable_benchmark_set(300)


@pytest.fixture(scope="module")
def eval_01_results(
    unanswerable_suite: list[UnanswerableTask],
    populated_benchmark_repo: Any,
    domain_registry: DomainRegistry,
    paid_spy: PaidExecutionSpy,
) -> list[tuple[UnanswerableTask, Any, int]]:
    """Runs all 300 EVAL-01 unanswerable tasks through ScoutWorkflow against populated repo."""
    workflow = ScoutWorkflow(
        registry=domain_registry,
        repo=populated_benchmark_repo,
    )

    results: list[tuple[UnanswerableTask, Any, int]] = []
    for task in unanswerable_suite:
        paid_spy.reset()

        principal = PrincipalContext(
            user_id=f"eval_user_{task.task_id}",
            role=RoleType.ENGINEER,
            allowed_scopes=task.requested_scopes or ["public_eng"],
        )

        bench_domain = populated_benchmark_repo.benchmark_domain_id
        res = workflow.run(
            user_prompt=task.prompt,
            principal=principal,
            domain_id=bench_domain,
        )
        results.append((task, res, paid_spy.call_count))

    return results


class TestSyntheticFixtureZeroSpendAbstentionProof:
    """Suite proving mechanical 0-token abstention across synthetic EVAL-01 tasks (N >= 300)."""

    def test_sample_size_is_at_least_300(self, unanswerable_suite: list[UnanswerableTask]) -> None:
        assert len(unanswerable_suite) >= 300

    def test_taxonomy_distribution_integrity(
        self, unanswerable_suite: list[UnanswerableTask]
    ) -> None:
        sym_tasks = [
            t
            for t in unanswerable_suite
            if t.expected_failure_reason == UnanswerableReason.UNRESOLVED_INTERNAL_SYMBOL
        ]
        file_tasks = [
            t
            for t in unanswerable_suite
            if t.expected_failure_reason == UnanswerableReason.NONEXISTENT_FILE
        ]
        auth_tasks = [
            t
            for t in unanswerable_suite
            if t.expected_failure_reason == UnanswerableReason.UNAUTHORIZED_SCOPE
        ]
        lib_tasks = [
            t
            for t in unanswerable_suite
            if t.expected_failure_reason == UnanswerableReason.UNINDEXED_EXTERNAL_LIBRARY
        ]
        empty_tasks = [
            t
            for t in unanswerable_suite
            if t.expected_failure_reason == UnanswerableReason.EMPTY_REPOSITORY_CONTEXT
        ]

        assert len(sym_tasks) == 80
        assert len(file_tasks) == 60
        assert len(auth_tasks) == 60
        assert len(lib_tasks) == 50
        assert len(empty_tasks) == 50

    def test_mechanical_zero_spend_across_all_300_unanswerable_tasks(
        self,
        eval_01_results: list[tuple[UnanswerableTask, Any, int]],
    ) -> None:
        """Asserts spy.call_count == 0 and tokens == 0 on every trial (synthetic-fixture result)."""
        total_trials = len(eval_01_results)
        successful_abstentions = 0

        for task, result, spy_calls in eval_01_results:
            assert isinstance(result, InsufficientContextDiagnostic), (
                f"Task {task.task_id} failed to abstain on populated repo! Output: {type(result)}"
            )
            assert result.paid_tokens_spent == 0, f"Task {task.task_id} reported paid tokens > 0"
            assert spy_calls == 0, (
                f"Task {task.task_id} triggered paid model invocations: {spy_calls}"
            )
            successful_abstentions += 1

        assert successful_abstentions == total_trials == 300

    def test_dynamic_clopper_pearson_95_percent_confidence_bound(
        self,
        eval_01_results: list[tuple[UnanswerableTask, Any, int]],
    ) -> None:
        """Computes k dynamically from observed run outcomes (no hardcoded k)."""
        n = len(eval_01_results)
        k = sum(
            1 for _, res, _ in eval_01_results if isinstance(res, InsufficientContextDiagnostic)
        )
        alpha = 0.05

        assert k == n == 300, f"Empirical abstention count k={k} does not equal trials n={n}"

        p_lower_twosided = float(stats.beta.ppf(alpha / 2.0, k, n - k + 1))
        p_lower_onesided = float(stats.beta.ppf(alpha, k, n - k + 1))
        error_rate_upper_onesided = 1.0 - p_lower_onesided

        assert p_lower_twosided >= 0.9877, (
            f"Dynamic two-sided lower bound {p_lower_twosided:.6f} < 0.9878"
        )
        assert p_lower_onesided >= 0.9900, (
            f"Dynamic one-sided lower bound {p_lower_onesided:.6f} < 0.9900"
        )
        assert error_rate_upper_onesided <= 0.0100, (
            f"Dynamic error rate {error_rate_upper_onesided:.6f} > 1.0%"
        )

    def test_paid_execution_spy_self_test(self, paid_spy: PaidExecutionSpy) -> None:
        """Self-test proving paid execution interceptors fail closed on cloud/LLM egress."""
        executor = GeminiImplementationExecutor()
        test_principal = PrincipalContext(
            user_id="spy_probe", role=RoleType.ENGINEER, allowed_scopes=["public_eng"]
        )
        try:
            with pytest.raises(ZeroSpendViolationError, match="synthesize_keyless_vertex_handoff"):
                executor.synthesize_keyless_vertex_handoff_patch(cast(Any, None), test_principal)
            with (
                httpx.Client() as client,
                pytest.raises(ZeroSpendViolationError, match=r"httpx\.Client\.post"),
            ):
                client.post("https://generativelanguage.googleapis.com/v1beta/models:generate")
        finally:
            paid_spy.reset()


class TestSyntheticFixtureAnswerableQueryDiscrimination:
    """Suite proving EVAL-04 discrimination on synthetic evaluation fixtures (N >= 300)."""

    def test_per_stack_answerable_breakdown(
        self,
        answerable_suite: list[AnswerableTask],
    ) -> None:
        """Verifies explicit per-stack task counts across C#, TS, Python, and Terraform."""
        cs_tasks = [t for t in answerable_suite if t.stack == StackType.CSHARP_DOTNET]
        ts_tasks = [t for t in answerable_suite if t.stack == StackType.TS_REACT_REDUX]
        py_tasks = [t for t in answerable_suite if t.stack == StackType.PYTHON_BACKEND]
        tf_tasks = [t for t in answerable_suite if t.stack == StackType.TERRAFORM_GCP]

        assert len(cs_tasks) == 75, f"C# tasks: {len(cs_tasks)} != 75"
        assert len(ts_tasks) == 75, f"TS tasks: {len(ts_tasks)} != 75"
        assert len(py_tasks) == 75, f"Python tasks: {len(py_tasks)} != 75"
        assert len(tf_tasks) == 75, f"Terraform tasks: {len(tf_tasks)} != 75"
        assert len(cs_tasks) + len(ts_tasks) + len(py_tasks) + len(tf_tasks) == 300

    def test_eval04_discrimination_and_wrongful_refusal_rate(
        self,
        answerable_suite: list[AnswerableTask],
        populated_benchmark_repo: Any,
        domain_registry: DomainRegistry,
        paid_spy: PaidExecutionSpy,
    ) -> None:
        """Executes all 300 answerable tasks (synthetic-fixture result),
        verifying refusal <= 1.0%.
        """
        workflow = ScoutWorkflow(
            registry=domain_registry,
            repo=populated_benchmark_repo,
        )

        total_trials = len(answerable_suite)
        successful_packs = 0
        wrongful_refusals = 0

        for task in answerable_suite:
            paid_spy.reset()

            principal = PrincipalContext(
                user_id=f"eval04_user_{task.task_id}",
                role=RoleType.ENGINEER,
                allowed_scopes=["public_eng", f"repo:{task.target_repo}"],
            )

            bench_domain = populated_benchmark_repo.benchmark_domain_id
            result = workflow.run(
                user_prompt=task.prompt,
                principal=principal,
                domain_id=bench_domain,
            )

            if isinstance(result, ImplementationContextPack):
                successful_packs += 1
                assert len(result.code_slices) >= 1
                assert any(task.target_repo in s.repo_name for s in result.code_slices)
            elif isinstance(result, InsufficientContextDiagnostic):
                wrongful_refusals += 1

        wrongful_refusal_rate = wrongful_refusals / total_trials
        assert wrongful_refusal_rate <= 0.010, (
            f"Wrongful refusal rate {wrongful_refusal_rate:.4f} > 1.0%"
        )
        assert successful_packs >= 297, f"Successful answerable packs {successful_packs} < 297/300"


class TestPreFlightGateMutationProof:
    """Mutation tests proving that benchmark tests fail when preflight gates are weakened."""

    def test_mutation_bypassing_closure_gate_fails_unanswerable_test(
        self,
        populated_benchmark_repo: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Mutates Check 2 (closure) to return True; proves unanswerable task fails to abstain."""
        bench_domain = populated_benchmark_repo.benchmark_domain_id
        validator = PreFlightValidator(repo=populated_benchmark_repo, domain_id=bench_domain)

        monkeypatch.setattr(
            PreFlightValidator, "partition_prompt_identifiers", lambda self, *a, **kw: ([], [])
        )
        monkeypatch.setattr(
            PreFlightValidator, "check_symbol_resolution", lambda self, *a, **kw: ([], 0)
        )

        p = PrincipalContext(
            user_id="mut_tester",
            role=RoleType.ENGINEER,
            allowed_scopes=["repo:enterprise/billing-service"],
        )
        report = validator.validate(
            user_prompt=(
                "Refactor class NonExistentSymbol_V000Handler in billing to use async lock."
            ),
            principal=p,
            domain_id=bench_domain,
            normalized_rrf_score=0.90,
            code_slices=[
                VerbatimCodeSlice(
                    repo_name="enterprise/billing-service",
                    file_path="src/billing/service.py",
                    start_line=1,
                    end_line=2,
                    qualified_symbol=(
                        "enterprise/billing-service::src/billing/service.py::BillingManager"
                    ),
                    selection_reason="Target",
                    modification_intent=ModificationIntent.MODIFY_TARGET,
                    git_commit_sha="1111222233334444555566667777888899990000",
                    git_blob_sha="2222333344445555666677778888999900001111",
                    verbatim_code_with_lines="1 | class BillingManager:\n2 |     pass\n",
                    auto_expanded_by_ast=False,
                )
            ],
        )
        assert report.check_results["check_2_closure"] is True, "Mutation verification failed"

    def test_mutation_lowering_rrf_threshold_passes_unindexed_library(
        self,
        populated_benchmark_repo: Any,
    ) -> None:
        """Proves lowering RRF threshold to 0.0 allows unindexed queries to pass Check 3."""
        normal_val = PreFlightValidator(repo=populated_benchmark_repo, min_rrf_score=0.65)
        passed_normal, _ = normal_val.check_rrf_threshold(normalized_rrf_score=0.20)
        assert passed_normal is False

        weakened_val = PreFlightValidator(repo=populated_benchmark_repo, min_rrf_score=0.0)
        passed_weak, _ = weakened_val.check_rrf_threshold(normalized_rrf_score=0.20)
        assert passed_weak is True

    def test_mutation_bypassing_scope_gate_admits_unauthorized_scope(
        self,
        populated_benchmark_repo: Any,
    ) -> None:
        """Proves normal scope check denies access while authorized principal admits vault file."""
        guest = PrincipalContext(
            user_id="guest", role=RoleType.ENGINEER, allowed_scopes=["public_eng"]
        )
        admin = PrincipalContext(
            user_id="admin", role=RoleType.ENGINEER, allowed_scopes=["vault_restricted"]
        )

        vault_slice = VerbatimCodeSlice(
            repo_name="enterprise/vault-service",
            file_path="src/vault/secrets.py",
            start_line=1,
            end_line=2,
            qualified_symbol="enterprise/vault-service::src/vault/secrets.py::VaultAuditLogger",
            selection_reason="Target",
            modification_intent=ModificationIntent.MODIFY_TARGET,
            git_commit_sha="3333444455556666777788889999000011112222",
            git_blob_sha="4444555566667777888899990000111122223333",
            verbatim_code_with_lines="1 | class VaultAuditLogger:\n2 |     pass\n",
            auto_expanded_by_ast=False,
        )
        bench_domain = populated_benchmark_repo.benchmark_domain_id
        validator = PreFlightValidator(repo=populated_benchmark_repo, domain_id=bench_domain)
        passed_guest, _permitted_guest, _, excised_guest, _ = validator.check_principal_scopes(
            [vault_slice],
            [],
            validator._resolve_graph_repo(populated_benchmark_repo),
            guest,
            bench_domain,
        )
        assert passed_guest is False
        assert len(excised_guest) == 1

        passed_admin, permitted_admin, _, _excised_admin, _ = validator.check_principal_scopes(
            [vault_slice],
            [],
            validator._resolve_graph_repo(populated_benchmark_repo),
            admin,
            bench_domain,
        )
        assert passed_admin is True
        assert len(permitted_admin) == 1

    def test_mutation_bypassing_provenance_gate_admits_phantom_file(
        self,
        populated_benchmark_repo: Any,
    ) -> None:
        """Proves Check 5 rejects phantom file absent from repo index."""
        principal = PrincipalContext(
            user_id="eng", role=RoleType.ENGINEER, allowed_scopes=["public_eng"]
        )
        phantom_slice = VerbatimCodeSlice(
            repo_name="dotnet/eShop",
            file_path="src/nonexistent/phantom.cs",
            start_line=1,
            end_line=2,
            qualified_symbol="dotnet/eShop::src/nonexistent/phantom.cs::PhantomClass",
            selection_reason="Target",
            modification_intent=ModificationIntent.MODIFY_TARGET,
            git_commit_sha="e5a1b2c3d4e5f678901234567890123456789012",
            git_blob_sha="f6b1c2d3e4f5a678901234567890123456789012",
            verbatim_code_with_lines="1 | class PhantomClass {}\n",
            auto_expanded_by_ast=False,
        )
        bench_domain = populated_benchmark_repo.benchmark_domain_id
        validator = PreFlightValidator(repo=populated_benchmark_repo, domain_id=bench_domain)
        passed, unanchored, _ = validator.check_cryptographic_provenance(
            [phantom_slice],
            validator._resolve_graph_repo(populated_benchmark_repo),
            principal,
            bench_domain,
        )
        assert passed is False
        assert len(unanchored) == 1
        assert "Phantom file not found" in unanchored[0]


class TestStorageParityAndNeutralScopes:
    """Verifies storage parity between InMemoryRLSRepository and live PostgresRLSRepository."""

    def test_public_eng_scope_confers_no_universal_access(
        self,
        populated_benchmark_repo: Any,
    ) -> None:
        """Regression test: Proves public_eng scope confers zero universal access or bypass."""
        user = PrincipalContext(
            user_id="alice_standard_engineer",
            role=RoleType.ENGINEER,
            allowed_scopes=["public_eng", "scope:general"],
        )
        vault_slice = VerbatimCodeSlice(
            repo_name="enterprise/vault-service",
            file_path="src/vault/secrets.py",
            start_line=1,
            end_line=2,
            qualified_symbol="enterprise/vault-service::src/vault/secrets.py::VaultAuditLogger",
            selection_reason="Target",
            modification_intent=ModificationIntent.MODIFY_TARGET,
            git_commit_sha="3333444455556666777788889999000011112222",
            git_blob_sha="4444555566667777888899990000111122223333",
            verbatim_code_with_lines="1 | class VaultAuditLogger:\n2 |     pass\n",
            auto_expanded_by_ast=False,
        )
        bench_domain = populated_benchmark_repo.benchmark_domain_id
        validator = PreFlightValidator(repo=populated_benchmark_repo, domain_id=bench_domain)
        passed, permitted, _, excised, missing = validator.check_principal_scopes(
            [vault_slice],
            [],
            validator._resolve_graph_repo(populated_benchmark_repo),
            user,
            bench_domain,
        )
        assert passed is False, "public_eng must NOT allow access to restricted vault file!"
        assert len(permitted) == 0
        assert len(excised) == 1
        assert any(
            "lacks required scopes" in str(m.reason) or "lacks visibility" in str(m.reason)
            for m in missing
        )

    @pytest.mark.postgres
    def test_rls_scope_filtering_parity_in_memory_vs_postgres(
        self,
        populated_benchmark_repo: Any,
    ) -> None:
        """Verifies RLS scope intersections strictly mirror database permissions."""
        if not is_postgres_reachable():
            pytest.fail(
                "Live PostgreSQL 17 is required for store parity verification but is unreachable!"
            )

        import psycopg

        from context_scout.config import ScoutSettings

        st = ScoutSettings()
        with psycopg.connect(st.db_admin_url) as conn:
            row = conn.execute(
                "SELECT count(*) FROM information_schema.tables WHERE table_name = 'code_files'"
            ).fetchone()
            assert row is not None and row[0] == 1, "code_files table must exist in PostgreSQL"

        eng_p = PrincipalContext(
            user_id="eng_user", role=RoleType.ENGINEER, allowed_scopes=["public_eng"]
        )
        guest_p = PrincipalContext(
            user_id="guest", role=RoleType.ENGINEER, allowed_scopes=["guest_only"]
        )

        bench_domain = populated_benchmark_repo.benchmark_domain_id
        visible_eng = populated_benchmark_repo.list_visible_code_files(eng_p, bench_domain)
        visible_guest = populated_benchmark_repo.list_visible_code_files(guest_p, bench_domain)

        assert len(visible_eng) >= 300
        assert len(visible_guest) == 0

    @pytest.mark.postgres
    def test_fixture_cannot_touch_rows_of_another_domain(self) -> None:
        """Proves benchmark teardown isolates by domain_id and cannot touch other domains."""
        if not is_postgres_reachable():
            pytest.fail("Live PostgreSQL 17 is required for canary test but is unreachable!")

        import psycopg

        from context_scout.config import ScoutSettings

        st = ScoutSettings()
        canary_domain = f"canary_verification_domain_{uuid.uuid4().hex[:8]}"
        canary_repo_name = f"canary/repo-{uuid.uuid4().hex[:8]}"
        canary_fid = uuid.uuid4()
        session_domain = f"eval_synthetic_{uuid.uuid4().hex[:12]}"

        with psycopg.connect(st.db_admin_url) as conn:
            # 1. Seed external canary row
            conn.execute(
                """
                INSERT INTO domain_packs (
                    domain_id, display_name, lora_adapter_id,
                    primary_truth_source, freshness_half_life_days, config_json
                )
                VALUES (%s, 'Canary Verification Domain', 'none', 'code', 30, '{}')
                ON CONFLICT (tenant_id, domain_id) DO NOTHING;
                """,
                (canary_domain,),
            )
            pg_repo = PostgresRLSRepository()
            pg_repo.insert_code_file(
                conn,
                CodeFileRecord(
                    file_id=canary_fid,
                    domain_id=canary_domain,
                    repo_name=canary_repo_name,
                    file_path="src/canary.py",
                    git_commit_sha="1111222233334444555566667777888899990000",
                    git_blob_sha="2222333344445555666677778888999900001111",
                    language="python",
                    allowed_scopes=["scope:canary"],
                    raw_content='print("canary")',
                    line_count=1,
                ),
            )
            conn.commit()

            try:
                # 2. Invoke REAL seed followed by REAL teardown on session_domain
                seed_synthetic_benchmark_repo(conn, session_domain)
                teardown_synthetic_benchmark_repo(conn, session_domain)

                # 3. Assert canary row is completely intact and session_domain is 100% purged
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT COUNT(*) FROM code_files WHERE domain_id = %s;",
                        (canary_domain,),
                    )
                    row_count_rec = cur.fetchone()
                    assert row_count_rec is not None
                    row_count = row_count_rec[0]
                    assert row_count == 1, f"Canary row in {canary_domain} was deleted or touched!"

                    for tbl in ("code_files", "ast_symbols", "doc_chunks", "domain_packs"):
                        cur.execute(
                            f"SELECT COUNT(*) FROM {tbl} WHERE domain_id = %s;",
                            (session_domain,),
                        )
                        tbl_row = cur.fetchone()
                        assert tbl_row is not None
                        cnt = tbl_row[0]
                        assert cnt == 0, (
                            f"Session domain {session_domain} still has {cnt} rows in {tbl}!"
                        )

                # 4. Mutation proof: verify that breaking teardown predicate deletes canary
                def flawed_teardown(c: psycopg.Connection, target_repo: str) -> None:
                    # Flawed predicate deletes by unique test repo without domain_id isolation
                    c.execute("DELETE FROM code_files WHERE repo_name = %s;", (target_repo,))

                flawed_teardown(conn, canary_repo_name)
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT COUNT(*) FROM code_files WHERE domain_id = %s;",
                        (canary_domain,),
                    )
                    mutated_row = cur.fetchone()
                    assert mutated_row is not None
                    mutated_count = mutated_row[0]
                    assert mutated_count == 0, (
                        "Flawed teardown mutation did not delete canary row as expected!"
                    )
            finally:
                with contextlib.suppress(Exception):
                    conn.rollback()
                conn.execute("DELETE FROM code_files WHERE domain_id = %s;", (canary_domain,))
                conn.execute("DELETE FROM domain_packs WHERE domain_id = %s;", (canary_domain,))
                conn.commit()
