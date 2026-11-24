"""M1-03 & M1-04 verification: Keyless GCSObjectStore, EmbeddingProviderSPI, and halfvec spaces."""

from __future__ import annotations

import ast
import json
import math
import uuid
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import httpx
import psycopg
import pytest

from context_scout.config import ModelGovernancePolicy, ScoutSettings
from context_scout.db.repositories import InMemoryRLSRepository, PostgresRLSRepository
from context_scout.db.schema import (
    ASTSymbolRecord,
    CodeFileRecord,
    DocChunkRecord,
    EmbeddingSpaceRecord,
)
from context_scout.db.session import get_query_pool
from context_scout.plugins.connectors.confluence import (
    ConfluenceConnectorPlugin,
    ConfluencePageDTO,
)
from context_scout.plugins.connectors.gitlab import GitLabConnectorPlugin
from context_scout.plugins.connectors.jira import JiraConnectorPlugin, JiraIssueDTO
from context_scout.queue.task_queue import QueueTask
from context_scout.retrieval.embedder import (
    DEFAULT_GOOGLE_EMBEDDING_MODEL,
    EmbeddingProviderSPI,
    LocalSemanticEmbedder,
    OpenAICompatibleEmbeddingProvider,
    VertexAIEmbeddingProvider,
    get_embedding_provider,
    normalize_embedding,
    resolve_embedding_model_spec,
    validate_non_zero_embedding,
)
from context_scout.schemas.contracts import PrincipalContext, RoleType
from context_scout.storage.object_store import (
    GCSObjectStore,
    build_tenant_domain_key,
    get_artifact_object_store,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_unit_vec(dim: int, hot_index: int = 0) -> list[float]:
    raw = [0.01] * dim
    raw[hot_index % dim] = 1.0
    return normalize_embedding(raw, target_dim=dim, supports_matryoshka=True)


def test_gcs_object_store_emulator_and_keyless_adc(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies keyless GCSObjectStore (ADC, STORAGE_EMULATOR_HOST, SHA-256, tenant isolation)."""
    stored_blobs: dict[tuple[str, str], tuple[bytes, dict[str, Any]]] = {}

    def _gcs_mock_handler(request: httpx.Request) -> httpx.Response:
        path = request.url.raw_path.decode("ascii").split("?", 1)[0]
        if request.method == "POST" and path.startswith("/upload/storage/v1/b/"):
            bkt = unquote(path.split("/upload/storage/v1/b/", 1)[1].split("/o", 1)[0])
            raw_body = request.content
            content_type_hdr = request.headers.get("Content-Type", "")
            boundary = content_type_hdr.split("boundary=", 1)[1].strip()
            parts = raw_body.split(f"--{boundary}".encode())
            meta_part = parts[1].split(b"\r\n\r\n", 1)[1].rstrip(b"\r\n")
            data_part = parts[2].split(b"\r\n\r\n", 1)[1]
            if data_part.endswith(b"\r\n"):
                data_part = data_part[:-2]
            meta_json = json.loads(meta_part.decode("utf-8"))
            obj_name = str(meta_json["name"])
            stored_blobs[(bkt, obj_name)] = (data_part, meta_json)
            return httpx.Response(200, json=meta_json)

        if path.startswith("/storage/v1/b/"):
            rest = path[len("/storage/v1/b/") :]
            if "/o/" in rest:
                bkt_enc, obj_enc = rest.split("/o/", 1)
                bkt = unquote(bkt_enc)
                obj_name = unquote(obj_enc)
                entry = stored_blobs.get((bkt, obj_name))
                if request.method == "DELETE":
                    if entry is None:
                        return httpx.Response(404, json={"error": {"code": 404}})
                    stored_blobs.pop((bkt, obj_name), None)
                    return httpx.Response(204)
                if entry is None:
                    return httpx.Response(404, json={"error": {"code": 404}})
                data_bytes, meta_json = entry
                if request.url.params.get("alt") == "media":
                    return httpx.Response(200, content=data_bytes)
                return httpx.Response(200, json=meta_json)

            if rest.endswith("/o") and request.method == "GET":
                bkt = unquote(rest[:-2])
                prefix = request.url.params.get("prefix", "")
                page_token = request.url.params.get("pageToken")
                matching = sorted(
                    name for (b, name) in stored_blobs if b == bkt and name.startswith(prefix)
                )
                if page_token == "page2":
                    items = [{"name": k} for k in matching[1:]]
                    return httpx.Response(200, json={"items": items})
                if len(matching) > 1:
                    return httpx.Response(
                        200,
                        json={"items": [{"name": matching[0]}], "nextPageToken": "page2"},
                    )
                return httpx.Response(200, json={"items": [{"name": k} for k in matching]})

        return httpx.Response(400, json={"error": "unhandled route"})

    transport = httpx.MockTransport(_gcs_mock_handler)
    http_client = httpx.Client(transport=transport)
    tenant_a = uuid.uuid4()
    tenant_b = uuid.uuid4()
    domain_id = "code_architecture_scout"

    gcs = GCSObjectStore(
        "gs://scout-enterprise-artifacts/",
        project="gcp-eu-prod-01",
        emulator_host="http://127.0.0.1:9023",
        http_client=http_client,
    )
    assert gcs.bucket == "scout-enterprise-artifacts"
    assert gcs.project == "gcp-eu-prod-01"
    assert gcs.uses_keyless_adc is True
    assert gcs.emulator_host == "http://127.0.0.1:9023"

    key_1 = build_tenant_domain_key(tenant_a, domain_id, "sync_manifest.json")
    key_2 = build_tenant_domain_key(tenant_a, domain_id, "adapters/v1/adapter_config.json")
    payload_1 = b'{"status": "synced", "version": 1}'
    payload_2 = b'{"r": 16, "lora_alpha": 32}'

    assert gcs.exists(key_1, expected_tenant_id=tenant_a) is False
    meta_1 = gcs.put_bytes(
        key_1,
        payload_1,
        content_type="application/json",
        expected_tenant_id=tenant_a,
    )
    gcs.put_bytes(
        key_2,
        payload_2,
        content_type="application/json",
        expected_tenant_id=tenant_a,
    )
    assert meta_1.uri == f"gs://scout-enterprise-artifacts/{key_1}"
    assert gcs.exists(key_1, expected_tenant_id=tenant_a) is True
    assert gcs.get_bytes(key_1, expected_tenant_id=tenant_a) == payload_1

    listed = gcs.list_keys(f"tenants/{tenant_a}/domains/{domain_id}/", expected_tenant_id=tenant_a)
    assert listed == sorted([key_1, key_2])

    with pytest.raises(PermissionError, match="Cross-tenant"):
        gcs.get_bytes(key_1, expected_tenant_id=tenant_b)
    with pytest.raises(PermissionError, match="Cross-tenant"):
        gcs.list_keys(f"tenants/{tenant_a}/", expected_tenant_id=tenant_b)
    with pytest.raises(ValueError, match="Path traversal"):
        gcs.list_keys(f"tenants/{tenant_a}/../{tenant_b}/", expected_tenant_id=tenant_a)
    with pytest.raises(ValueError, match="GCS bucket name must be non-empty"):
        GCSObjectStore("gs:///missing-bucket/path")

    # Verify special URL characters (+, %2F, spaces) and multipart boundary collision resistance
    special_key = build_tenant_domain_key(
        tenant_a, domain_id, "adapters/v1+rc 2/part%2Fshard.safetensors"
    )
    collision_payload = b"header\r\n--===============scout_gcs_multipart_boundary==\r\ntrailer"
    gcs.put_bytes(
        special_key,
        collision_payload,
        content_type="application/octet-stream",
        expected_tenant_id=tenant_a,
    )
    assert gcs.get_bytes(special_key, expected_tenant_id=tenant_a) == collision_payload
    assert special_key in gcs.list_keys(
        f"tenants/{tenant_a}/domains/{domain_id}/adapters/v1+rc 2/",
        expected_tenant_id=tenant_a,
    )
    gcs.delete(special_key, expected_tenant_id=tenant_a)

    # Corrupt stored bytes to verify fail-closed SHA-256 check
    orig_bytes, orig_meta = stored_blobs[("scout-enterprise-artifacts", key_1)]
    stored_blobs[("scout-enterprise-artifacts", key_1)] = (b"tampered-payload", orig_meta)
    with pytest.raises(ValueError, match="GCS SHA-256 integrity mismatch"):
        gcs.get_bytes(key_1, expected_tenant_id=tenant_a)
    stored_blobs[("scout-enterprise-artifacts", key_1)] = (orig_bytes, orig_meta)

    gcs.delete(key_1, expected_tenant_id=tenant_a)
    assert gcs.exists(key_1, expected_tenant_id=tenant_a) is False
    with pytest.raises(KeyError, match="Object not found"):
        gcs.get_bytes(key_1, expected_tenant_id=tenant_a)

    # Verify keyless ADC credential refresh path when emulator_host is None
    class _FakeADCCredentials:
        def __init__(self) -> None:
            self.valid = False
            self.token: str | None = None
            self.refresh_calls = 0

        def refresh(self, request: Any) -> None:
            _ = request
            self.refresh_calls += 1
            self.valid = True
            self.token = "ya29.keyless_wif_adc_token"

    fake_creds = _FakeADCCredentials()
    adc_gcs = GCSObjectStore(
        "scout-enterprise-artifacts",
        project="gcp-eu-prod-01",
        credentials=fake_creds,
        http_client=http_client,
    )
    assert adc_gcs.get_bytes(key_2, expected_tenant_id=tenant_a) == payload_2
    assert fake_creds.refresh_calls == 1

    # Verify get_artifact_object_store routing for SCOUT_GCS_BUCKET and gs:// URI
    monkeypatch.setenv("STORAGE_EMULATOR_HOST", "localhost:9023")
    cfg_gcs = ScoutSettings(gcs_bucket="gs://my-gcp-bucket")
    store_gcs = get_artifact_object_store(cfg_gcs)
    assert isinstance(store_gcs, GCSObjectStore)
    assert store_gcs.bucket == "my-gcp-bucket"
    assert store_gcs.emulator_host == "http://localhost:9023"

    cfg_s3_gs = ScoutSettings(s3_bucket="gs://my-s3-interop-gcs-bucket")
    store_s3_gs = get_artifact_object_store(cfg_s3_gs)
    assert isinstance(store_s3_gs, GCSObjectStore)
    assert store_s3_gs.bucket == "my-s3-interop-gcs-bucket"


def test_embedding_provider_spi_local_vllm_tei_and_vertex_governance() -> None:
    """Verifies EmbeddingProviderSPI across local Gemma-2, vLLM/TEI Qwen3/bge-m3, and Vertex ADC."""
    # 1. Local default google/embeddinggemma-2 (768d + 256d Matryoshka)
    local_default = get_embedding_provider()
    assert isinstance(local_default, EmbeddingProviderSPI)
    assert isinstance(local_default, LocalSemanticEmbedder)
    assert local_default.model_id == DEFAULT_GOOGLE_EMBEDDING_MODEL
    assert local_default.dimensions == 768
    assert local_default.supports_matryoshka is True

    vec_768 = local_default.embed_query("PostgreSQL 17 row level security halfvec HNSW")
    assert len(vec_768) == 768
    assert math.isclose(math.sqrt(sum(x * x for x in vec_768)), 1.0, abs_tol=1e-3)

    local_mrl_256 = LocalSemanticEmbedder(
        model_id="google/embeddinggemma-2",
        dimensions=256,
    )
    vec_256 = local_mrl_256.embed_text("Matryoshka representation learning 256d", is_query=False)
    assert len(vec_256) == 256
    assert math.isclose(math.sqrt(sum(x * x for x in vec_256)), 1.0, abs_tol=1e-3)

    # 2. BAAI/bge-m3 requires native 1024d (supports_matryoshka=False)
    spec_bge, dim_bge = resolve_embedding_model_spec("BAAI/bge-m3")
    assert spec_bge.native_dimensions == 1024
    assert dim_bge == 1024
    assert spec_bge.supports_matryoshka is False
    with pytest.raises(ValueError, match="does not support Matryoshka"):
        resolve_embedding_model_spec("BAAI/bge-m3", requested_dimensions=512)

    # 3. vLLM --task embed (/v1/embeddings) with Qwen/Qwen3-Embedding-4B MRL (1024d from 2560d)
    captured_vllm_reqs: list[dict[str, Any]] = []

    def _vllm_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured_vllm_reqs.append(body)
        inputs = body["input"]
        data = [
            {"index": i, "embedding": _make_unit_vec(2560, hot_index=i + 1)}
            for i in range(len(inputs))
        ]
        return httpx.Response(200, json={"data": data, "model": body["model"]})

    vllm_provider = OpenAICompatibleEmbeddingProvider(
        model_id="Qwen/Qwen3-Embedding-4B",
        endpoint_url="http://127.0.0.1:8080/v1",
        dimensions=1024,
        provider_mode="vllm",
        http_client=httpx.Client(transport=httpx.MockTransport(_vllm_handler)),
    )
    assert isinstance(vllm_provider, EmbeddingProviderSPI)
    assert vllm_provider.dimensions == 1024
    assert vllm_provider.supports_matryoshka is True
    vllm_q = vllm_provider.embed_query("How does blue green HNSW index cutover work?")
    assert len(vllm_q) == 1024
    assert math.isclose(math.sqrt(sum(x * x for x in vllm_q)), 1.0, abs_tol=1e-3)
    assert captured_vllm_reqs[0]["dimensions"] == 1024
    assert str(captured_vllm_reqs[0]["input"][0]).startswith("Instruct:")

    # 4. Hugging Face TEI (/embed) with BAAI/bge-m3 (1024d)
    def _tei_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        inputs = body["inputs"]
        return httpx.Response(
            200,
            json=[_make_unit_vec(1024, hot_index=i + 5) for i in range(len(inputs))],
        )

    tei_provider = OpenAICompatibleEmbeddingProvider(
        model_id="BAAI/bge-m3",
        endpoint_url="http://127.0.0.1:8081/embed",
        dimensions=1024,
        provider_mode="tei",
        http_client=httpx.Client(transport=httpx.MockTransport(_tei_handler)),
    )
    tei_vecs = tei_provider.embed_documents(["Doc chunk A", "Doc chunk B"])
    assert len(tei_vecs) == 2
    assert all(len(v) == 1024 for v in tei_vecs)

    # get_embedding_provider & ScoutSettings auto-resolve native 1024d/2560d dimensions
    # when dimensions is not explicitly specified, and TEI base URLs route to /embed
    bge_cfg = ScoutSettings(
        embedding_model_id="BAAI/bge-m3",
        embedding_provider="tei",
        embedding_endpoint_url="http://127.0.0.1:8081",
    )
    assert bge_cfg.embedding_dimensions == 1024
    bge_auto = get_embedding_provider(
        bge_cfg,
        http_client=httpx.Client(transport=httpx.MockTransport(_tei_handler)),
    )
    assert bge_auto.dimensions == 1024
    assert len(bge_auto.embed_query("TEI base URL without /embed suffix")) == 1024

    qwen_cfg = ScoutSettings(
        embedding_model_id="Qwen/Qwen3-Embedding-4B",
        embedding_provider="vllm",
        embedding_endpoint_url="http://127.0.0.1:8080/v1",
    )
    assert qwen_cfg.embedding_dimensions == 2560

    # 5. VertexAIEmbeddingProvider governance & residency gates
    with pytest.raises(PermissionError, match="allow_external_cloud_egress=False"):
        VertexAIEmbeddingProvider(model_id="gemini-embedding-001")

    egress_policy = ModelGovernancePolicy.from_registry_file(allow_external_cloud_egress=True)
    with pytest.raises(ValueError, match="Prohibited or retired model ID"):
        VertexAIEmbeddingProvider(
            model_id="text-embedding-004",
            policy=egress_policy,
        )
    with pytest.raises(ValueError, match="EU data residency"):
        VertexAIEmbeddingProvider(
            model_id="gemini-embedding-001",
            location="us-central1",
            policy=egress_policy,
        )

    class _FakeGenAIModels:
        def embed_content(self, *, model: str, contents: list[str], config: dict[str, Any]) -> Any:
            assert model == "gemini-embedding-001"
            assert config["output_dimensionality"] == 768
            return type(
                "_Resp",
                (),
                {
                    "embeddings": [
                        type("_Emb", (), {"values": _make_unit_vec(768, hot_index=idx + 3)})()
                        for idx in range(len(contents))
                    ]
                },
            )()

    fake_genai = type("_Client", (), {"models": _FakeGenAIModels()})()
    vertex_provider = VertexAIEmbeddingProvider(
        model_id="gemini-embedding-001",
        project="gcp-eu-prod",
        location="europe-west3",
        dimensions=768,
        policy=egress_policy,
        genai_client=fake_genai,
    )
    v_emb = vertex_provider.embed_query("Vertex AI keyless ADC test")
    assert len(v_emb) == 768
    assert math.isclose(math.sqrt(sum(x * x for x in v_emb)), 1.0, abs_tol=1e-3)


def test_zero_norm_vectors_rejected_and_src_purged_of_prng_and_zeros(
    ephemeral_pg_domain: str,
) -> None:
    """Rejects [0.0]*D at Python & PG17 boundaries and verifies src/ has zero PRNG/zeros."""
    with pytest.raises(ValueError, match="Zero-norm"):
        validate_non_zero_embedding([0.0] * 768)
    with pytest.raises(ValueError, match="Zero-norm"):
        normalize_embedding([0.0] * 1024, target_dim=1024)
    with pytest.raises(ValueError, match="NaN or infinite"):
        validate_non_zero_embedding([float("nan")] + [0.1] * 767)

    # Verify src/ contains zero generate_deterministic_embedding or [0.0]*768
    for py_path in sorted((REPO_ROOT / "src").rglob("*.py")):
        src_text = py_path.read_text(encoding="utf-8")
        assert "generate_deterministic_embedding" not in src_text, (
            f"Prohibited PRNG embedding helper found in {py_path}"
        )
        tree = ast.parse(src_text)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.BinOp)
                and isinstance(node.op, ast.Mult)
                and isinstance(node.left, ast.List)
                and len(node.left.elts) == 1
                and isinstance(node.left.elts[0], ast.Constant)
                and node.left.elts[0].value in (0, 0.0)
                and (
                    (
                        isinstance(node.right, ast.Constant)
                        and isinstance(node.right.value, int)
                        and node.right.value >= 128
                    )
                    or (isinstance(node.right, ast.Name) and "dim" in node.right.id.lower())
                )
            ):
                raise AssertionError(f"Prohibited zero-vector literal [0.0]*D found in {py_path}")

    # Verify GitLab, Confluence, and Jira connectors emit real non-zero embeddings
    in_mem = InMemoryRLSRepository()
    principal = PrincipalContext(
        user_id="connector_verifier",
        role=RoleType.ENGINEER,
        allowed_scopes=[
            "public_eng",
            *GitLabConnectorPlugin().mirror_permissions("group/repo"),
            "confluence:space:ENG:read",
            "jira:project:ENG:read",
        ],
    )
    now_dt = datetime.now(UTC)
    jira_chunks = JiraConnectorPlugin().chunk_issue_verbatim(
        JiraIssueDTO(
            issue_id="42",
            issue_key="ENG-42",
            project_key="ENG",
            summary="Replace vector(768) with halfvec spaces",
            description_text="Migrate HNSW indexes to halfvec_cosine_ops.",
            acceptance_criteria="Zero downtime blue/green cutover.",
            issue_type="Story",
            status="In Progress",
            resolution=None,
            assignee="alice",
            source_url="https://jira.example.internal/browse/ENG-42",
            created_at=now_dt,
            updated_at=now_dt,
        ),
        domain_id=ephemeral_pg_domain,
    )
    assert len(jira_chunks) == 2
    for jc in jira_chunks:
        validate_non_zero_embedding(jc.embedding, expected_dim=768)

    conf_plugin = ConfluenceConnectorPlugin()
    conf_dto = ConfluencePageDTO(
        page_id="9001",
        space_key="ENG",
        title="Halfvec Architecture ADR",
        version_number=1,
        author_username="alice",
        source_url="https://wiki.example.internal/pages/9001",
        body_storage="<h2>Overview</h2><p>PostgreSQL 17 halfvec HNSW indexing specification.</p>",
        created_at=now_dt,
        last_modified_at=now_dt,
    )
    conf_res = conf_plugin.sync_space(
        ephemeral_pg_domain,
        "ENG",
        in_mem,
        current_pages=[conf_dto],
    )
    assert conf_res.items_indexed >= 1
    visible_conf = in_mem.list_visible_doc_chunks(principal, ephemeral_pg_domain)
    assert len(visible_conf) >= 1
    for cc in visible_conf:
        validate_non_zero_embedding(cc.embedding, expected_dim=768)

    def _gl_handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "/repository/compare" in url:
            return httpx.Response(
                200,
                json={
                    "diffs": [
                        {
                            "old_path": "src/margin.py",
                            "new_path": "src/margin.py",
                            "new_file": True,
                            "renamed_file": False,
                            "deleted_file": False,
                            "diff": (
                                "+def compute_margin(x: float) -> float:\n+    return x * 1.05\n"
                            ),
                        }
                    ]
                },
            )
        if "src%2Fmargin.py/raw" in url:
            return httpx.Response(
                200,
                content=b"def compute_margin(x: float) -> float:\n    return x * 1.05\n",
            )
        return httpx.Response(404, json={"error": "Not found"})

    gl_client = httpx.Client(
        transport=httpx.MockTransport(_gl_handler),
        base_url="https://gitlab.example.com",
    )
    gl_plugin = GitLabConnectorPlugin(
        base_url="https://gitlab.example.com",
        client=gl_client,
    )
    gl_task = QueueTask(
        task_id=uuid.uuid4(),
        domain_id=ephemeral_pg_domain,
        event_type="gitlab_push",
        payload={
            "project": {
                "id": 101,
                "path_with_namespace": "group/repo",
                "default_branch": "main",
                "git_http_url": "https://gitlab.example.com/group/repo.git",
            },
            "before": "b" * 40,
            "after": "a" * 40,
            "ref": "refs/heads/main",
            "commits": [{"id": "a" * 40}],
        },
        priority=1,
        status="PENDING",
        worker_id=None,
        attempts=0,
        max_attempts=3,
        error_message=None,
        scheduled_at=now_dt,
        created_at=now_dt,
    )
    gl_plugin.process_push_webhook(gl_task, in_mem, principal=principal)
    gl_syms = in_mem.list_visible_ast_symbols(principal, ephemeral_pg_domain)
    assert len(gl_syms) == 1
    validate_non_zero_embedding(gl_syms[0].embedding, expected_dim=768)

    # Verify PostgreSQL boundary rejects zero-norm vectors both via repository and raw SQL
    settings = ScoutSettings()
    query_mgr = get_query_pool(settings)
    pg_repo = PostgresRLSRepository()
    file_id = uuid.uuid4()

    with query_mgr.scoped_transaction(principal, enable_hnsw_iterative_scan=False) as conn:
        pg_repo.insert_code_file(
            conn,
            CodeFileRecord(
                file_id=file_id,
                domain_id=ephemeral_pg_domain,
                repo_name="group/repo",
                file_path="src/zero_guard.py",
                git_commit_sha="a" * 40,
                git_blob_sha="b" * 40,
                language="python",
                allowed_scopes=["public_eng"],
                raw_content="def f(): pass\n",
                line_count=1,
            ),
        )
        with pytest.raises(ValueError, match="Zero-norm"):
            pg_repo.insert_ast_symbol(
                conn,
                ASTSymbolRecord(
                    symbol_id=uuid.uuid4(),
                    file_id=file_id,
                    domain_id=ephemeral_pg_domain,
                    qualified_name="zero_guard.f",
                    symbol_kind="function",
                    start_line=1,
                    end_line=1,
                    signature_skeleton="def f():",
                    contextual_summary="zero test",
                    allowed_scopes=["public_eng"],
                    embedding=[0.0] * 768,
                ),
            )
        with pytest.raises(ValueError, match="Zero-norm"):
            pg_repo.insert_doc_chunk(
                conn,
                DocChunkRecord(
                    chunk_id=uuid.uuid4(),
                    domain_id=ephemeral_pg_domain,
                    doc_id="zero-doc",
                    doc_title="Zero Doc",
                    source_uri="urn:zero",
                    market_scope="ENTERPRISE_DOC",
                    published_date=date(2026, 11, 18),
                    authority_weight=1.0,
                    section_heading="Sec",
                    contextual_header="Hdr",
                    verbatim_content="Zero vector test content",
                    metadata_json={},
                    allowed_scopes=["public_eng"],
                    embedding=[0.0] * 768,
                ),
            )

        zero_lit = "[" + ",".join(["0.0"] * 768) + "]"
        with pytest.raises(psycopg.errors.CheckViolation):
            conn.execute("SAVEPOINT sp_zero_doc;")
            try:
                conn.execute(
                    """
                    INSERT INTO doc_chunks (
                        chunk_id, domain_id, doc_id, doc_title, source_uri,
                        market_scope, section_heading, contextual_header,
                        verbatim_content, allowed_scopes, embedding
                    )
                    VALUES (%s, %s, 'raw-zero', 'Raw Zero', 'urn:rz', 'GLOBAL',
                            'Sec', 'Hdr', 'Body', %s, %s::vector);
                    """,
                    (uuid.uuid4(), ephemeral_pg_domain, ["public_eng"], zero_lit),
                )
            except Exception:
                conn.execute("ROLLBACK TO SAVEPOINT sp_zero_doc;")
                raise


def test_live_pg17_halfvec_embedding_spaces_blue_green_cutover(
    ephemeral_pg_domain: str,
) -> None:
    """Verifies live PG17 embedding_spaces blue/green shadow re-index (768d -> 1024d)."""
    settings = ScoutSettings()
    query_mgr = get_query_pool(settings)
    repo = PostgresRLSRepository()
    principal = PrincipalContext(
        user_id="halfvec_architect",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng", f"domain:{ephemeral_pg_domain}"],
    )

    file_id = uuid.uuid4()
    sym_id = uuid.uuid4()
    chunk_id = uuid.uuid4()
    vec_768_sym = _make_unit_vec(768, hot_index=10)
    vec_768_chk = _make_unit_vec(768, hot_index=20)

    with query_mgr.scoped_transaction(principal, enable_hnsw_iterative_scan=True) as conn:
        # 1. Register initial active 768d space (google/embeddinggemma-2)
        active_768 = repo.register_embedding_space(
            conn,
            EmbeddingSpaceRecord(
                domain_id=ephemeral_pg_domain,
                space_id="embeddinggemma-2-768",
                model_id="google/embeddinggemma-2",
                provider="sentence-transformers",
                dimensions=768,
                quantization="halfvec",
                status="active",
                is_active=True,
            ),
        )
        assert active_768.is_active is True
        assert active_768.dimensions == 768

        repo.insert_code_file(
            conn,
            CodeFileRecord(
                file_id=file_id,
                domain_id=ephemeral_pg_domain,
                repo_name="scout/core",
                file_path="src/halfvec_router.py",
                git_commit_sha="1" * 40,
                git_blob_sha="2" * 40,
                language="python",
                allowed_scopes=["public_eng"],
                raw_content="def route_halfvec_space() -> str:\n    return 'active'\n",
                line_count=2,
            ),
        )
        repo.insert_ast_symbol(
            conn,
            ASTSymbolRecord(
                symbol_id=sym_id,
                file_id=file_id,
                domain_id=ephemeral_pg_domain,
                qualified_name="scout.core.halfvec_router.route_halfvec_space",
                symbol_kind="function",
                start_line=1,
                end_line=2,
                signature_skeleton="def route_halfvec_space() -> str:",
                contextual_summary="Routes queries to active halfvec HNSW embedding space.",
                allowed_scopes=["public_eng"],
                embedding=vec_768_sym,
            ),
        )
        repo.insert_doc_chunk(
            conn,
            DocChunkRecord(
                chunk_id=chunk_id,
                domain_id=ephemeral_pg_domain,
                doc_id="adr:halfvec-blue-green",
                doc_title="Blue Green Halfvec HNSW Cutover",
                source_uri="urn:adr:halfvec",
                market_scope="INTERNAL_ADR",
                published_date=date(2026, 11, 18),
                authority_weight=1.2,
                section_heading="Atomic Swap",
                contextual_header="[ADR | Halfvec Spaces]",
                verbatim_content=(
                    "Blue green shadow re-indexing stages halfvec(1024) vectors before "
                    "atomic pointer swap in embedding_spaces."
                ),
                metadata_json={"milestone": "M1-03"},
                allowed_scopes=["public_eng"],
                embedding=vec_768_chk,
            ),
        )

        # Verify 768d hybrid RRF search works before shadow cutover
        hits_768 = repo.search_doc_chunks_hybrid_rrf(
            conn,
            domain_id=ephemeral_pg_domain,
            query_text="Blue green shadow re-indexing halfvec atomic pointer swap",
            query_embedding=vec_768_chk,
            top_k=5,
        )
        assert [h.item_id for h in hits_768] == [chunk_id]

        # 2. Register 1024d shadow space (BAAI/bge-m3)
        shadow_1024 = EmbeddingSpaceRecord(
            domain_id=ephemeral_pg_domain,
            space_id="bge-m3-1024",
            model_id="BAAI/bge-m3",
            provider="tei",
            dimensions=1024,
            quantization="halfvec",
            status="shadow",
            is_active=False,
        )
        repo.register_embedding_space(conn, shadow_1024)

        # Activating before shadow vectors are staged must fail closed
        with pytest.raises(RuntimeError, match="lack staged shadow halfvec embeddings"):
            repo.activate_embedding_space(conn, ephemeral_pg_domain, "bge-m3-1024")

        # 3. Stage 1024d shadow vectors via reindex_domain_shadow_space
        class _Stub1024Embedder:
            def embed_batch(
                self, texts: list[str], is_query: bool = False, batch_size: int = 32
            ) -> list[list[float]]:
                _ = (is_query, batch_size)
                return [_make_unit_vec(1024, hot_index=100 + idx) for idx in range(len(texts))]

        staged_counts = repo.reindex_domain_shadow_space(
            conn,
            ephemeral_pg_domain,
            shadow_1024,
            _Stub1024Embedder(),
        )
        assert staged_counts == {"symbols_staged": 1, "chunks_staged": 1}

        # While shadow space is staged, active space is STILL embeddinggemma-2-768 (zero downtime)
        still_active = repo.get_active_embedding_space(conn, ephemeral_pg_domain)
        assert still_active is not None and still_active.space_id == "embeddinggemma-2-768"
        hits_during_shadow = repo.search_doc_chunks_hybrid_rrf(
            conn,
            domain_id=ephemeral_pg_domain,
            query_text="Blue green shadow re-indexing halfvec",
            query_embedding=vec_768_chk,
            top_k=5,
        )
        assert [h.item_id for h in hits_during_shadow] == [chunk_id]

        # 4. Atomic active-space pointer swap to bge-m3-1024
        swapped = repo.activate_embedding_space(conn, ephemeral_pg_domain, "bge-m3-1024")
        assert swapped.space_id == "bge-m3-1024"
        assert swapped.is_active is True
        assert swapped.status == "active"
        assert swapped.dimensions == 1024

        spaces = repo.list_embedding_spaces(conn, ephemeral_pg_domain)
        space_status_map = {s.space_id: (s.is_active, s.status, s.dimensions) for s in spaces}
        assert space_status_map["bge-m3-1024"] == (True, "active", 1024)
        assert space_status_map["embeddinggemma-2-768"] == (False, "retired", 768)

        # 5. Query symbols & doc_chunks with 1024d halfvec vectors & verify HNSW iterative_scan GUC
        it_scan = conn.execute("SHOW hnsw.iterative_scan;").fetchone()
        assert it_scan == ("relaxed_order",)

        q_1024 = _make_unit_vec(1024, hot_index=100)
        hits_1024_doc = repo.search_doc_chunks_hybrid_rrf(
            conn,
            domain_id=ephemeral_pg_domain,
            query_text="Blue green shadow re-indexing halfvec atomic pointer swap",
            query_embedding=q_1024,
            top_k=5,
        )
        assert [h.item_id for h in hits_1024_doc] == [chunk_id]
        assert hits_1024_doc[0].vector_rank == 1

        hits_1024_sym = repo.search_symbols_hybrid_rrf(
            conn,
            domain_id=ephemeral_pg_domain,
            query_text="route_halfvec_space active halfvec",
            query_embedding=q_1024,
            top_k=5,
        )
        assert [h.item_id for h in hits_1024_sym] == [sym_id]
        assert hits_1024_sym[0].vector_rank == 1

        visible_chunks = repo.list_visible_doc_chunks(conn, ephemeral_pg_domain)
        assert len(visible_chunks) == 1
        assert len(visible_chunks[0].embedding) == 1024
        assert visible_chunks[0].embedding_space_id == "bge-m3-1024"


def test_tracked_in_memory_repos_weakref_prevents_aba_id_reuse() -> None:
    """Prevents GC'd InMemoryRLSRepository id(repo) reuse from corrupting new repos."""
    import weakref

    from context_scout.ingestion.git_merkle_indexer import (
        _TRACKED_IN_MEMORY_REPOS,
        sync_tracked_in_memory_repos,
    )

    old_repo = InMemoryRLSRepository()
    dead_ref = weakref.ref(old_repo)
    del old_repo

    new_repo = InMemoryRLSRepository()
    key = (id(new_repo), "code_architecture_scout", "stale-repo")
    _TRACKED_IN_MEMORY_REPOS[key] = (
        dead_ref,
        REPO_ROOT,
        ["public_eng", "public_docs"],
        "0" * 40,
    )
    principal = PrincipalContext(
        user_id="aba_guard",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng", "public_docs"],
    )
    stats = sync_tracked_in_memory_repos(new_repo, principal, "code_architecture_scout")
    assert stats == []
    assert key not in _TRACKED_IN_MEMORY_REPOS
    assert new_repo.list_visible_code_files(principal, "code_architecture_scout") == []


def test_concurrent_activate_embedding_space_serialization_and_halfvec_trigger(
    ephemeral_pg_domain: str,
) -> None:
    """Concurrent pointer swap serialization + UPDATE trigger & tail-1024d safety."""
    import threading

    from context_scout.db.migrations_runner import _reconcile_m1_halfvec_spaces
    from context_scout.db.repositories import _format_db_vector_and_halfvec

    settings = ScoutSettings()
    _reconcile_m1_halfvec_spaces(settings.db_admin_url)

    # 1. Verify sub-FP16 underflow vector is rejected and tail-only 1024d vector projects cleanly
    with pytest.raises(ValueError, match="Zero-norm"):
        validate_non_zero_embedding([4e-7] * 768)

    tail_only_1024 = [float(i >= 768) for i in range(1024)]
    vec768_lit, hv1024_lit = _format_db_vector_and_halfvec(tail_only_1024)
    assert vec768_lit.startswith("[")
    assert hv1024_lit.startswith("[")

    # 2. Verify concurrent activate_embedding_space across two transactions does not raise
    # UniqueViolation on uq_idx_embedding_spaces_one_active
    query_mgr = get_query_pool(settings)
    repo = PostgresRLSRepository()
    principal = PrincipalContext(
        user_id="concurrent_swapper",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng", f"domain:{ephemeral_pg_domain}"],
    )

    with query_mgr.scoped_transaction(principal, enable_hnsw_iterative_scan=False) as conn:
        for sid in ("space_a", "space_b", "space_c"):
            repo.register_embedding_space(
                conn,
                EmbeddingSpaceRecord(
                    domain_id=ephemeral_pg_domain,
                    space_id=sid,
                    model_id="google/embeddinggemma-2",
                    dimensions=768,
                    is_active=(sid == "space_a"),
                    status="active" if sid == "space_a" else "shadow",
                ),
            )

    barrier = threading.Barrier(2)
    errors: list[tuple[str, str, str]] = []

    def _swap_worker(target_sid: str) -> None:
        try:
            with query_mgr.scoped_transaction(
                principal, enable_hnsw_iterative_scan=False
            ) as w_conn:
                barrier.wait(timeout=5.0)
                repo.activate_embedding_space(
                    w_conn,
                    ephemeral_pg_domain,
                    target_sid,
                    require_complete_shadow=False,
                )
        except Exception as exc:
            errors.append((target_sid, type(exc).__name__, str(exc)))

    t1 = threading.Thread(target=_swap_worker, args=("space_b",))
    t2 = threading.Thread(target=_swap_worker, args=("space_c",))
    t1.start()
    t2.start()
    t1.join(timeout=10.0)
    t2.join(timeout=10.0)
    assert errors == [], f"Concurrent activate_embedding_space failed: {errors}"

    with query_mgr.scoped_transaction(principal, enable_hnsw_iterative_scan=False) as conn:
        spaces = repo.list_embedding_spaces(conn, ephemeral_pg_domain)
        active_spaces = [s for s in spaces if s.is_active]
        assert len(active_spaces) == 1
        assert active_spaces[0].space_id in {"space_b", "space_c"}

        # 3. Verify UPDATE doc_chunks SET embedding = ... updates embedding_halfvec via trigger
        chk_id = uuid.uuid4()
        vec_initial = _make_unit_vec(768, hot_index=0)
        vec_updated = _make_unit_vec(768, hot_index=50)
        repo.insert_doc_chunk(
            conn,
            DocChunkRecord(
                chunk_id=chk_id,
                domain_id=ephemeral_pg_domain,
                doc_id="trigger-sync-doc",
                doc_title="Trigger Sync Doc",
                source_uri="urn:trigger-sync",
                market_scope="ENTERPRISE_DOC",
                published_date=date(2026, 11, 18),
                authority_weight=1.0,
                section_heading="Trigger",
                contextual_header="Hdr",
                verbatim_content="Trigger sync verification body content for halfvec.",
                metadata_json={},
                allowed_scopes=["public_eng"],
                embedding=vec_initial,
            ),
        )
        updated_lit, _ = _format_db_vector_and_halfvec(vec_updated)
        conn.execute(
            "UPDATE doc_chunks SET embedding = %s::vector WHERE chunk_id = %s;",
            (updated_lit, chk_id),
        )
        hv_row = conn.execute(
            "SELECT (embedding_halfvec <=> %s::halfvec) FROM doc_chunks WHERE chunk_id = %s;",
            (updated_lit, chk_id),
        ).fetchone()
        assert hv_row is not None and float(str(hv_row[0])) < 1e-3
