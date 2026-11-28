"""M4-01 verification suite: OpenWeightsVLLMExecutor, AdapterSyncSidecar, & OWASP ASI05.

Covers:
1. OpenWeightsVLLMExecutor (LLMBackendSPI) across vLLM, SGLang, and local Apple Silicon mlx-lm
   for Gemma 4 (`google/gemma-4-*`, `mlx-community/gemma-4-*`) and BYO open-weights families
   (`Qwen3-Coder`, `Devstral`, `DeepSeek-R1`, `Llama-4`, `Nemotron-H`, `IBM Granite-3.3`).
2. ModelArchitectureAdapter: dynamic `/v1/models` `max_model_len` discovery, vendor-neutral
   `AutoTokenizer.apply_chat_template` formatting (zero hardcoded `<start_of_turn>` tags),
   `--reasoning-parser` delegation, `xgrammar` `structural_tag` JSON Schema enforcement,
   and `sha256(tenant_id + sorted_scopes)` KV prefix-cache salting.
3. Cross-ACL `training_scope_ceiling` routing (OWASP ASI06), `RedisGPUQuotaManager` priority leases,
   fail-closed offline abstention (`paid_tokens_spent=0`), and 1-step vLLM multi-LoRA dry-run
   probes (`r in {8,16,32}`, `use_rslora=True`, `use_dora=False`, frozen MoE routers & Mamba SSM).
4. AdapterSyncSidecar: live PostgreSQL 17 `model_artifacts` under Forced RLS (`NOBYPASSRLS`) +
   both keyless GCP `GCSObjectStore` (`gs://`) and `S3CompatibleObjectStore` (`s3://`), real
   ECDSA P-256 & Ed25519 `cosign` `.sig` / Sigstore `.bundle` verification,
   `base_model_revision_sha256` pinning, 3-pod atomic loading (`vllm#12174`), automatic rollback
   on partial pod failure, and blue/green rotation + retirement unload.
5. OWASP `ASI05` elimination (`handoff.py` & `gemini_executor.py`), optional keyless ADC
   `vertexai=True` handoff gated by `allow_external_cloud_egress`, and tenant-token regressions.
"""

from __future__ import annotations

import ast
import concurrent.futures
import hashlib
import io
import json
import subprocess
import threading
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from urllib.parse import unquote

import httpx
import numpy as np
import psycopg
import pytest
from botocore.exceptions import ClientError
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519
from fastapi.testclient import TestClient
from pydantic import BaseModel
from safetensors.numpy import save_file as save_safetensors_numpy

import context_scout.api.routes.handoff as handoff_mod
from context_scout.agents.gemini_executor import GeminiImplementationExecutor, GitWorktreeSandbox
from context_scout.agents.llm_backend import (
    AdapterRouter,
    GPUOverloadError,
    GPUPriorityClass,
    LLMBackendSPI,
    RedisGPUQuotaManager,
)
from context_scout.agents.open_weights_vllm_executor import (
    LoRACompatibilityError,
    ModelArchitectureAdapter,
    OpenWeightsBackendUnavailableError,
    OpenWeightsVLLMExecutor,
    StructuredOutputValidationError,
    compute_prefix_cache_salt,
    detect_architecture_profile,
    validate_lora_safetensors_for_vllm,
)
from context_scout.api.dependencies import (
    ConfirmationTokenStore,
    create_signed_principal_token,
    get_confirmation_token_store,
    get_repository,
)
from context_scout.api.main import app
from context_scout.config import ScoutSettings
from context_scout.db.repositories import (
    InMemoryRLSRepository,
    ModelArtifactRecord,
    PostgresRLSRepository,
)
from context_scout.db.session import RLSConnectionManager
from context_scout.mcp_servers.common import PostgresScoutGraphAdapter
from context_scout.schemas.contracts import (
    ExecutionMode,
    ExecutionTelemetryRecord,
    ImplementationContextPack,
    ModificationIntent,
    PreFlightValidationReport,
    PrincipalContext,
    RoleType,
    VerbatimCodeSlice,
)
from context_scout.serving.adapter_sync_sidecar import (
    AdapterSyncSidecar,
    CosignBlobVerifier,
    sign_blob_for_cosign,
)
from context_scout.storage.object_store import (
    GCSObjectStore,
    S3CompatibleObjectStore,
    build_tenant_domain_key,
)
from context_scout.telemetry.cost_governance import BudgetExceededError


class _ScoutSynthesisContract(BaseModel):
    summary: str
    cited_chunk_ids: list[str]
    confidence: float


def _make_pack(
    domain_id: str = "code_architecture_scout",
    pack_tokens: int = 1_500,
) -> ImplementationContextPack:
    sl = VerbatimCodeSlice(
        repo_name="service-repo",
        file_path="src/context_scout/schemas/contracts.py",
        start_line=1,
        end_line=5,
        qualified_symbol="PrincipalContext",
        selection_reason="Target slice",
        modification_intent=ModificationIntent.MODIFY_TARGET,
        git_commit_sha="1111111",
        git_blob_sha="2222222",
        verbatim_code_with_lines="   1 | class PrincipalContext:\n   2 |     pass\n",
    )
    return ImplementationContextPack(
        trace_id=str(uuid.uuid4()),
        domain_id=domain_id,
        user_task="Document PrincipalContext invariant",
        architectural_execution_plan=["Add docstring invariant note"],
        code_slices=[sl],
        preflight_report=PreFlightValidationReport(
            passed=True,
            normalized_rrf_score=0.92,
            citations_verified_100_pct=True,
        ),
        local_tokens_consumed=25_000,
        pack_token_count=pack_tokens,
    )


def _build_safetensors_bytes(tensor_names: tuple[str, ...] | None = None) -> bytes:
    names = tensor_names or (
        "base_model.model.layers.0.self_attn.q_proj.lora_A.weight",
        "base_model.model.layers.0.self_attn.q_proj.lora_B.weight",
        "base_model.model.layers.0.self_attn.v_proj.lora_A.weight",
        "base_model.model.layers.0.self_attn.v_proj.lora_B.weight",
    )
    tensors = {
        name: np.ones((4, 4), dtype=np.float32) * float(idx + 1) for idx, name in enumerate(names)
    }
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        save_safetensors_numpy(tensors, str(tmp_path))
        return tmp_path.read_bytes()
    finally:
        tmp_path.unlink(missing_ok=True)


import tempfile  # noqa: E402


class _InMemoryS3BotoMock:
    """Deterministic in-memory boto3 S3 mock for offline S3CompatibleObjectStore verification."""

    def __init__(self) -> None:
        self._buckets: dict[str, dict[str, tuple[bytes, dict[str, str]]]] = {}

    def put_object(self, **kwargs: Any) -> dict[str, Any]:
        bucket = str(kwargs["Bucket"])
        key = str(kwargs["Key"])
        body = bytes(kwargs["Body"])
        meta = dict(kwargs.get("Metadata") or {})
        bkt = self._buckets.setdefault(bucket, {})
        bkt[key] = (body, meta)
        return {"ETag": '"etag"'}

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        bucket = str(kwargs["Bucket"])
        key = str(kwargs["Key"])
        bkt = self._buckets.get(bucket, {})
        if key not in bkt:
            raise ClientError({"Error": {"Code": "NoSuchKey", "Message": "Not found"}}, "GetObject")
        body, meta = bkt[key]
        return {"Body": io.BytesIO(body), "Metadata": dict(meta)}

    def head_object(self, **kwargs: Any) -> dict[str, Any]:
        bucket = str(kwargs["Bucket"])
        key = str(kwargs["Key"])
        bkt = self._buckets.get(bucket, {})
        if key not in bkt:
            raise ClientError({"Error": {"Code": "404", "Message": "Not found"}}, "HeadObject")
        _, meta = bkt[key]
        return {"Metadata": dict(meta)}

    def delete_object(self, **kwargs: Any) -> dict[str, Any]:
        bucket = str(kwargs["Bucket"])
        key = str(kwargs["Key"])
        self._buckets.get(bucket, {}).pop(key, None)
        return {}

    def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]:
        bucket = str(kwargs["Bucket"])
        prefix = str(kwargs.get("Prefix") or "")
        bkt = self._buckets.get(bucket, {})
        contents = [{"Key": k} for k in sorted(bkt) if k.startswith(prefix)]
        return {"Contents": contents, "IsTruncated": False}


def _make_gcs_mock_store(bucket: str = "scout-gke-models") -> GCSObjectStore:
    objects: dict[str, tuple[bytes, dict[str, Any]]] = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "POST" and path.startswith("/upload/storage/v1/b/"):
            obj_name = request.url.params.get("name", "")
            ctype = request.headers.get("Content-Type", "")
            boundary = ctype.split("boundary=")[-1].strip()
            parts = request.content.split(f"--{boundary}".encode("ascii"))
            meta_obj: dict[str, Any] = {}
            payload = b""
            for part in parts:
                if b"application/json" in part:
                    json_raw = part.split(b"\r\n\r\n", 1)[-1].rsplit(b"\r\n", 1)[0]
                    meta_obj = json.loads(json_raw.decode("utf-8"))
                elif b"Content-Type:" in part:
                    body_raw = part.split(b"\r\n\r\n", 1)[-1]
                    if body_raw.endswith(b"\r\n"):
                        body_raw = body_raw[:-2]
                    payload = body_raw
            objects[obj_name] = (payload, meta_obj)
            return httpx.Response(200, json=meta_obj)

        if request.method == "GET" and "/o/" in path:
            encoded_key = path.split("/o/", 1)[1]
            key = unquote(encoded_key)
            if key not in objects:
                return httpx.Response(404, json={"error": "not found"})
            payload, meta_obj = objects[key]
            if request.url.params.get("alt") == "media":
                return httpx.Response(200, content=payload)
            return httpx.Response(200, json=meta_obj)

        if request.method == "DELETE" and "/o/" in path:
            key = unquote(path.split("/o/", 1)[1])
            objects.pop(key, None)
            return httpx.Response(204)

        if request.method == "GET" and path.endswith("/o"):
            pfx = request.url.params.get("prefix", "")
            items = [{"name": k} for k in sorted(objects) if k.startswith(pfx)]
            return httpx.Response(200, json={"items": items})

        return httpx.Response(404)

    http_client = httpx.Client(transport=httpx.MockTransport(_handler))
    return GCSObjectStore(
        bucket=bucket,
        project="gke-enterprise-prod",
        emulator_host="http://gcs-emulator.internal:4443",
        http_client=http_client,
    )


class _SimulatedVLLMPodFleet:
    """Simulates a 3-pod GKE vLLM StatefulSet/Deployment for vllm#12174 parity & rollback tests."""

    def __init__(self, base_model_id: str, pod_count: int = 3) -> None:
        self.base_model_id = base_model_id
        self.pod_urls = tuple(f"http://vllm-pod-{i}.vllm-headless:8001" for i in range(pod_count))
        self.loaded_on_pod: dict[str, dict[str, str]] = {f"{url}/v1": {} for url in self.pod_urls}
        self.fail_load_on_pod: set[str] = set()
        self.load_history: list[tuple[str, str]] = []
        self.unload_history: list[tuple[str, str]] = []

    def build_http_client(self) -> httpx.Client:
        def _handler(request: httpx.Request) -> httpx.Response:
            pod_v1 = f"{request.url.scheme}://{request.url.host}:{request.url.port}/v1"
            route = request.url.path
            if request.method == "GET" and route == "/v1/models":
                models = [
                    {"id": self.base_model_id, "object": "model", "max_model_len": 65_536},
                    *[
                        {
                            "id": lora_name,
                            "object": "model",
                            "parent": self.base_model_id,
                            "root": lora_path,
                        }
                        for lora_name, lora_path in sorted(self.loaded_on_pod[pod_v1].items())
                    ],
                ]
                return httpx.Response(200, json={"object": "list", "data": models})

            if request.method == "POST" and route == "/v1/load_lora_adapter":
                body = json.loads(request.content.decode("utf-8"))
                lora_name = str(body["lora_name"])
                lora_path = str(body["lora_path"])
                if pod_v1 in self.fail_load_on_pod:
                    return httpx.Response(500, json={"error": f"Simulated CUDA OOM on {pod_v1}"})
                self.loaded_on_pod[pod_v1][lora_name] = lora_path
                self.load_history.append((pod_v1, lora_name))
                return httpx.Response(200, json={"status": "loaded", "lora_name": lora_name})

            if request.method == "POST" and route == "/v1/unload_lora_adapter":
                body = json.loads(request.content.decode("utf-8"))
                lora_name = str(body["lora_name"])
                self.loaded_on_pod[pod_v1].pop(lora_name, None)
                self.unload_history.append((pod_v1, lora_name))
                return httpx.Response(200, json={"status": "unloaded", "lora_name": lora_name})

            return httpx.Response(404)

        return httpx.Client(transport=httpx.MockTransport(_handler))


# =============================================================================
# 1. ModelArchitectureAdapter & OpenWeightsVLLMExecutor (vLLM / SGLang / mlx-lm)
# =============================================================================


@pytest.mark.parametrize(
    ("model_id", "expected_family", "expected_topology", "expected_parser"),
    [
        ("google/gemma-4-26B-A4B-it", "gemma-4", "moe", "gemma4"),
        ("mlx-community/gemma-4-26b-a4b-it-4bit", "gemma-4", "moe", "gemma4"),
        ("google/gemma-4-E4B-it", "gemma-4", "dense", "gemma4"),
        ("Qwen/Qwen3-Coder-30B-A3B", "Qwen3-Coder", "moe", "qwen3"),
        ("mistralai/Devstral-Small-2505", "Devstral", "dense", "mistral"),
        ("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "DeepSeek-R1", "moe", "deepseek_r1"),
        ("meta-llama/Llama-4-Scout-17B-16E-Instruct", "Llama-4", "moe", None),
        ("microsoft/Phi-4-reasoning", "Phi-4", "dense", "deepseek_r1"),
        ("nvidia/Nemotron-H-47B-Base-8K", "Nemotron-H", "hybrid_mamba_moe", "nemotron"),
        ("ibm-granite/granite-3.3-8b-instruct", "IBM Granite-3.3", "dense", "granite"),
    ],
)
def test_multi_family_architecture_profiles_and_serve_args(
    model_id: str,
    expected_family: str,
    expected_topology: str,
    expected_parser: str | None,
) -> None:
    """Verifies multi-family BYO governance, topology detection, and hardened vLLM flags."""
    adapter = ModelArchitectureAdapter(model_id)
    prof = adapter.profile
    assert prof.family == expected_family
    assert prof.topology == expected_topology
    assert prof.reasoning_parser == expected_parser
    if expected_topology == "moe":
        assert ".router." in prof.frozen_module_patterns
    elif expected_topology == "hybrid_mamba_moe":
        assert ".router." in prof.frozen_module_patterns
        assert ".mamba." in prof.frozen_module_patterns

    serve_args = adapter.recommended_vllm_serve_args(max_lora_rank=32)
    assert "--load-format" in serve_args and "safetensors" in serve_args
    assert "--structured-outputs-config.backend" in serve_args and "xgrammar" in serve_args
    assert "--guided-decoding-backend" not in serve_args
    assert "--trust-remote-code" not in serve_args
    if expected_parser:
        assert "--reasoning-parser" in serve_args and expected_parser in serve_args
        assert "--structured-outputs-config.enable_in_reasoning=True" in serve_args
    else:
        assert "--structured-outputs-config.enable_in_reasoning=True" not in serve_args


def test_open_weights_executor_rejects_retired_and_cloud_models() -> None:
    """OpenWeightsVLLMExecutor enforces local/BYO governance (zero cloud or retired IDs)."""
    for blocked in (
        "gemini-2.5-pro",
        "codegemma-7b",
        "gemma-3-27b-it",
        "gemini-3.8-flash",
        "claude-sonnet-5.5",
        "gpt-5.6-sol",
    ):
        with pytest.raises(ValueError):
            OpenWeightsVLLMExecutor(model_id=blocked)


def test_chat_template_formatting_across_families_never_hardcodes_gemma_tags(
    tmp_path: Path,
) -> None:
    """Proves AutoTokenizer.apply_chat_template & Jinja templates format without Gemma lock-in."""
    qwen_tok_dir = tmp_path / "qwen_tok"
    qwen_tok_dir.mkdir()
    (qwen_tok_dir / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "chat_template": (
                    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n"
                    "{{ m['content'] }}<|im_end|>\n{% endfor %}"
                    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
                )
            }
        ),
        encoding="utf-8",
    )
    qwen_adapter = ModelArchitectureAdapter(
        "Qwen/Qwen3-Coder-30B-A3B",
        tokenizer_path=qwen_tok_dir,
    )
    rendered_qwen = qwen_adapter.format_chat_messages(
        [{"role": "system", "content": "Cite code"}, {"role": "user", "content": "Find router"}]
    )
    assert "<|im_start|>system\nCite code<|im_end|>" in rendered_qwen
    assert "<|im_start|>assistant\n" in rendered_qwen
    assert "<start_of_turn>" not in rendered_qwen

    # Injected tokenizer object implementing apply_chat_template (Mistral [INST] style)
    class _MistralTokenizerStub:
        def apply_chat_template(
            self,
            messages: list[dict[str, str]],
            *,
            tokenize: bool = False,
            add_generation_prompt: bool = True,
        ) -> str:
            assert tokenize is False
            assert add_generation_prompt is True
            body = " ".join(m["content"] for m in messages)
            return f"<s>[INST] {body} [/INST]"

    mistral_adapter = ModelArchitectureAdapter(
        "mistralai/Devstral-Small-2505",
        tokenizer=_MistralTokenizerStub(),
    )
    rendered_mistral = mistral_adapter.format_chat_messages(
        [{"role": "user", "content": "Trace RPC handler"}]
    )
    assert rendered_mistral == "<s>[INST] Trace RPC handler [/INST]"
    assert "<start_of_turn>" not in rendered_mistral

    # Supply-chain guard: auto_map / trust_remote_code=True in tokenizer config fails closed
    malicious_dir = tmp_path / "evil_tok"
    malicious_dir.mkdir()
    (malicious_dir / "config.json").write_text(
        json.dumps({"auto_map": {"AutoTokenizer": "evil.py--EvilTokenizer"}}),
        encoding="utf-8",
    )
    evil_adapter = ModelArchitectureAdapter("Qwen/Qwen3-8B", tokenizer_path=malicious_dir)
    with pytest.raises(ValueError, match="auto_map"):
        evil_adapter.format_chat_messages([{"role": "user", "content": "hi"}])


def test_prefix_cache_salt_determinism_and_acl_isolation() -> None:
    """Verifies sha256(tenant_id + sorted_scopes) order invariance and tenant/ACL separation."""
    tid_a = uuid.uuid4()
    tid_b = uuid.uuid4()

    p_a1 = PrincipalContext(
        user_id="u1",
        role=RoleType.ENGINEER,
        tenant_id=tid_a,
        allowed_scopes=["enclave:erp", "tenant_common", "enclave:erp"],
    )
    p_a2 = PrincipalContext(
        user_id="u2",
        role=RoleType.ENGINEER,
        tenant_id=tid_a,
        allowed_scopes=["tenant_common", "enclave:erp"],
    )
    p_a_narrow = PrincipalContext(
        user_id="u3",
        role=RoleType.ENGINEER,
        tenant_id=tid_a,
        allowed_scopes=["tenant_common"],
    )
    p_b = PrincipalContext(
        user_id="u1",
        role=RoleType.ENGINEER,
        tenant_id=tid_b,
        allowed_scopes=["tenant_common", "enclave:erp"],
    )

    salt_a1 = compute_prefix_cache_salt(p_a1)
    salt_a2 = compute_prefix_cache_salt(p_a2)
    salt_narrow = compute_prefix_cache_salt(p_a_narrow)
    salt_b = compute_prefix_cache_salt(p_b)

    assert len(salt_a1) == 64
    assert salt_a1 == salt_a2  # Same tenant + identical scope set (permuted) -> cache hit
    assert salt_a1 != salt_narrow  # Different scopes -> isolated KV prefix cache
    assert salt_a1 != salt_b  # Different tenant_id -> isolated KV prefix cache
    assert compute_prefix_cache_salt(None) != compute_prefix_cache_salt(None)


def test_vllm_executor_discovery_xgrammar_reasoning_and_scope_ceiling_routing(
    tmp_path: Path,
) -> None:
    """End-to-end vLLM HTTP verification: /v1/models len, xgrammar, reasoning, salt, & ASI06."""
    domains_dir = tmp_path / "domains"
    adapter_dir = domains_dir / "erp_domain" / "adapters" / "lora_erp_v1"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapters.safetensors").write_bytes(_build_safetensors_bytes())

    router = AdapterRouter(
        domains_dir=domains_dir,
        default_base_model_id="Qwen/Qwen3-Coder-30B-A3B",
        spec_overrides={
            "erp_domain": {
                "domain_id": "erp_domain",
                "base_model_id": "Qwen/Qwen3-Coder-30B-A3B",
                "lora_adapter_id": "lora_erp_v1",
                "lora_adapter_path": str(adapter_dir),
                "training_scope_ceiling": ["tenant_common", "enclave:erp"],
                "lora_policy": {
                    "enabled": True,
                    "rank": 16,
                    "lora_alpha": 32,
                    "use_rslora": True,
                    "use_dora": False,
                },
            }
        },
    )

    captured_requests: list[tuple[dict[str, str], dict[str, Any]]] = []

    def _vllm_handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {
                            "id": "Qwen/Qwen3-Coder-30B-A3B",
                            "object": "model",
                            "max_model_len": 12_000,
                        }
                    ],
                },
            )
        if request.method == "POST" and request.url.path == "/v1/chat/completions":
            body = json.loads(request.content.decode("utf-8"))
            captured_requests.append((dict(request.headers), body))
            payload_json = json.dumps(
                {
                    "summary": "Verified BAPI mapping",
                    "cited_chunk_ids": ["chk-001"],
                    "confidence": 0.94,
                }
            )
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-1",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "reasoning_content": "Step 1: check RFC table parameters.",
                                "content": f"<json_output>{payload_json}</json_output>",
                            },
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 310,
                        "completion_tokens": 48,
                        "total_tokens": 358,
                    },
                },
            )
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(_vllm_handler))
    executor = OpenWeightsVLLMExecutor(
        model_id="Qwen/Qwen3-Coder-30B-A3B",
        endpoint_url="http://vllm.serving.svc.cluster.local:8000",
        adapter_router=router,
        http_client=client,
    )
    assert isinstance(executor, LLMBackendSPI)
    assert executor.is_online() is True
    assert executor.max_model_len == 12_000

    # Adaptive complexity budget clamps cross_repo (18,000) to 0.60 * 12,000 == 7,200
    tier, clamped_budget = executor.compute_query_budget(
        "Trace end-to-end architecture across repositories",
        is_cross_repo=True,
    )
    assert tier == "cross_repo"
    assert clamped_budget == 7_200

    # 1. Authorized principal (superset of training_scope_ceiling) -> routes to lora_erp_v1
    authorized_p = PrincipalContext(
        user_id="erp_eng",
        role=RoleType.ENGINEER,
        allowed_scopes=["tenant_common", "enclave:erp", "repo:sap"],
    )
    structured_out = executor.complete_structured(
        "Synthesize BAPI contract",
        _ScoutSynthesisContract,
        domain_id="erp_domain",
        principal=authorized_p,
        enable_thinking=True,
    )
    assert structured_out["cited_chunk_ids"] == ["chk-001"]
    assert executor.last_completion is not None
    assert executor.last_completion.routed_model == "lora_erp_v1"
    assert executor.last_completion.lora_request == "lora_erp_v1"
    assert executor.last_completion.reasoning_content == "Step 1: check RFC table parameters."
    assert "Step 1" not in executor.last_completion.content

    req_headers_1, req_body_1 = captured_requests[-1]
    expected_salt_1 = compute_prefix_cache_salt(authorized_p)
    assert req_body_1["model"] == "lora_erp_v1"
    assert req_body_1["extra_body"]["cache_salt"] == expected_salt_1
    assert "guided_decoding_backend" not in req_body_1
    assert "guided_decoding_backend" not in req_body_1["extra_body"]
    assert req_headers_1["x-prefix-cache-salt"] == expected_salt_1
    assert req_body_1["response_format"]["type"] == "structural_tag"
    assert req_body_1["response_format"]["triggers"] == ["<json_output>"]

    # 2. Narrow principal (missing 'enclave:erp') -> routes strictly to frozen base model (ASI06)
    narrow_p = PrincipalContext(
        user_id="general_eng",
        role=RoleType.ENGINEER,
        allowed_scopes=["tenant_common"],
    )
    _ = executor.complete(
        "General architecture question",
        domain_id="erp_domain",
        principal=narrow_p,
    )
    _, req_body_2 = captured_requests[-1]
    assert req_body_2["model"] == "Qwen/Qwen3-Coder-30B-A3B"
    assert executor.last_completion.lora_request is None
    assert req_body_2["extra_body"]["cache_salt"] == compute_prefix_cache_salt(narrow_p)
    assert req_body_2["extra_body"]["cache_salt"] != expected_salt_1


def test_xgrammar_structural_tag_validation_fails_closed_on_bad_schema() -> None:
    """Rejects missing end tags, invalid JSON, missing required keys, and additionalProperties."""
    schema = {
        "type": "object",
        "required": ["decision", "score"],
        "additionalProperties": False,
        "properties": {
            "decision": {"type": "string", "enum": ["pass", "abstain"]},
            "score": {"type": "number"},
        },
    }
    valid = ModelArchitectureAdapter.extract_and_validate_structural_json(
        '<json_output>{"decision": "pass", "score": 0.91}</json_output>',
        schema,
    )
    assert valid == {"decision": "pass", "score": 0.91}

    for invalid_raw in (
        "",
        '<json_output>{"decision": "pass", "score": 0.9}',  # unclosed structural_tag
        "<json_output>not-json</json_output>",
        '<json_output>{"decision": "invalid_enum", "score": 0.9}</json_output>',
        '<json_output>{"decision": "pass"}</json_output>',  # missing required 'score'
        '<json_output>{"decision": "pass", "score": 0.9, "rogue": 1}</json_output>',
    ):
        with pytest.raises(StructuredOutputValidationError):
            ModelArchitectureAdapter.extract_and_validate_structural_json(invalid_raw, schema)


def test_mlx_lm_dev_backend_and_gpu_quota_and_offline_abstention(tmp_path: Path) -> None:
    """Verifies Apple Silicon mlx-lm backend, pickle rejection, P0>P2 preemption, & offline."""
    domains_dir = tmp_path / "domains"
    adapter_dir = domains_dir / "mlx_dom" / "adapters" / "lora_mlx"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapters.safetensors").write_bytes(
        _build_safetensors_bytes(
            (
                "model.layers.0.self_attn.q_proj.lora_a",
                "model.layers.0.self_attn.q_proj.lora_b",
                "model.layers.0.self_attn.v_proj.lora_a",
                "model.layers.0.self_attn.v_proj.lora_b",
            )
        )
    )
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"r": 16, "lora_alpha": 32, "use_rslora": True, "use_dora": False}),
        encoding="utf-8",
    )

    router = AdapterRouter(
        domains_dir=domains_dir,
        default_base_model_id="mlx-community/gemma-4-e4b-it-4bit",
        spec_overrides={
            "mlx_dom": {
                "domain_id": "mlx_dom",
                "base_model_id": "mlx-community/gemma-4-e4b-it-4bit",
                "lora_adapter_id": "lora_mlx",
                "lora_adapter_path": str(adapter_dir),
                "training_scope_ceiling": ["public_eng"],
                "lora_policy": {"enabled": True, "rank": 16, "use_rslora": True, "use_dora": False},
            }
        },
    )

    load_calls: list[tuple[str, str | None, Mapping[str, Any]]] = []

    class _StubTokenizer:
        def apply_chat_template(
            self,
            messages: list[dict[str, str]],
            *,
            tokenize: bool = False,
            add_generation_prompt: bool = True,
        ) -> str:
            _ = (tokenize, add_generation_prompt)
            return " | ".join(f"{m['role']}:{m['content']}" for m in messages)

    def _loader(
        model_id: str,
        adapter_path: str | None = None,
        tokenizer_config: Mapping[str, Any] | None = None,
    ) -> tuple[str, _StubTokenizer]:
        load_calls.append((model_id, adapter_path, dict(tokenizer_config or {})))
        return ("mlx_model_handle", _StubTokenizer())

    def _generator(
        model: str,
        tokenizer: _StubTokenizer,
        *,
        prompt: str,
        max_tokens: int = 256,
    ) -> str:
        _ = (model, tokenizer, max_tokens)
        return f"<think>local metal reasoning</think>Answer for [{prompt}]"

    quota = RedisGPUQuotaManager(max_global_inflight=4)
    mlx_exec = OpenWeightsVLLMExecutor(
        model_id="mlx-community/gemma-4-e4b-it-4bit",
        runtime_backend="mlx-lm",
        adapter_router=router,
        gpu_quota_manager=quota,
        mlx_loader_fn=_loader,
        mlx_generate_fn=_generator,
    )
    p = PrincipalContext(
        user_id="mac_dev",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng"],
    )
    out = mlx_exec.complete("Explain HNSW index", domain_id="mlx_dom", principal=p)
    assert out == "Answer for [user:Explain HNSW index]"
    assert mlx_exec.last_completion is not None
    assert mlx_exec.last_completion.reasoning_content == "local metal reasoning"
    assert load_calls[0][1] == str(adapter_dir)
    assert load_calls[0][2] == {"trust_remote_code": False}

    # Active P0 lease preempts P2 background request with GPUOverloadError
    p2_principal = PrincipalContext(
        user_id="batch_distiller",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng"],
    )
    active_p0 = quota.acquire(p, priority=GPUPriorityClass.P0, estimated_tokens=512)
    try:
        with pytest.raises(GPUOverloadError) as exc_info:
            mlx_exec.complete(
                "Background distill",
                principal=p2_principal,
                priority=GPUPriorityClass.P2,
            )
        assert exc_info.value.scope == "preempted_by_p0"
    finally:
        quota.release(active_p0)

    # Unsafe .bin pickle file in adapter_dir fails closed
    (adapter_dir / "pytorch_model.bin").write_bytes(b"pickle")
    with pytest.raises(ValueError, match="unsafe binary/pickle"):
        mlx_exec.complete("Blocked call", domain_id="mlx_dom", principal=p)

    # Offline vLLM backend fails closed with paid_tokens_spent == 0 (never fabricates)
    def _offline_transport(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused", request=request)

    offline_exec = OpenWeightsVLLMExecutor(
        model_id="google/gemma-4-26B-A4B-it",
        http_client=httpx.Client(transport=httpx.MockTransport(_offline_transport)),
    )
    assert offline_exec.is_online() is False
    with pytest.raises(OpenWeightsBackendUnavailableError) as off_exc:
        offline_exec.complete("Must abstain", principal=p)
    assert off_exc.value.paid_tokens_spent == 0
    assert off_exc.value.diagnostic is not None
    assert off_exc.value.diagnostic.paid_tokens_spent == 0


def test_blk22_lora_safetensors_validation_and_1step_vllm_dry_run(tmp_path: Path) -> None:
    """Proves vLLM multi-LoRA safety: rejects use_dora, r>32, MoE router gates, and Mamba SSM."""
    good_weights = _build_safetensors_bytes()
    good_cfg = {"r": 32, "lora_alpha": 64, "use_rslora": True, "use_dora": False}
    rank, names = validate_lora_safetensors_for_vllm(
        good_weights, good_cfg, base_model_id="google/gemma-4-26B-A4B-it"
    )
    assert rank == 32 and len(names) == 4

    # Reject use_dora=True, use_rslora=False, rank=64
    for bad_cfg in (
        {"r": 16, "use_rslora": True, "use_dora": True},
        {"r": 16, "use_rslora": False, "use_dora": False},
        {"r": 64, "use_rslora": True, "use_dora": False},
    ):
        with pytest.raises(LoRACompatibilityError):
            validate_lora_safetensors_for_vllm(good_weights, bad_cfg)

    # Reject unfrozen MoE router gate tensor & Mamba SSM state tensor
    for bad_tensor in (
        "base_model.model.layers.2.block_sparse_moe.gate.lora_A.weight",
        "base_model.model.layers.0.router.lora_B.weight",
        "base_model.model.layers.1.mamba.dt_proj.lora_A.weight",
        "base_model.model.layers.1.mamba.A_log",
    ):
        bad_weights = _build_safetensors_bytes((bad_tensor,))
        with pytest.raises(LoRACompatibilityError, match="frozen MoE router gate or Mamba SSM"):
            validate_lora_safetensors_for_vllm(
                bad_weights,
                good_cfg,
                base_model_id="nvidia/Nemotron-H-47B-Base-8K",
            )

    # 1-step live vLLM dry-run probe (load -> 1-token chat completion -> unload)
    probe_dir = tmp_path / "dryrun_cartridge"
    probe_dir.mkdir()
    (probe_dir / "adapter_model.safetensors").write_bytes(good_weights)
    (probe_dir / "adapter_config.json").write_text(json.dumps(good_cfg), encoding="utf-8")

    probe_events: list[str] = []

    def _probe_handler(request: httpx.Request) -> httpx.Response:
        probe_events.append(f"{request.method}:{request.url.path}")
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "ok"}}], "usage": {"total_tokens": 2}},
            )
        return httpx.Response(200, json={"status": "ok"})

    exec_probe = OpenWeightsVLLMExecutor(
        model_id="google/gemma-4-26B-A4B-it",
        http_client=httpx.Client(transport=httpx.MockTransport(_probe_handler)),
        auto_discover_max_len=False,
    )
    report = exec_probe.run_vllm_lora_dry_run_probe(
        adapter_id="lora_dryrun_test",
        adapter_dir=probe_dir,
        probe_live_vllm=True,
    )
    assert report.passed is True
    assert report.live_endpoint_probed is True
    assert probe_events == [
        "POST:/v1/load_lora_adapter",
        "POST:/v1/chat/completions",
        "POST:/v1/unload_lora_adapter",
    ]


# =============================================================================
# 2. AdapterSyncSidecar (Live PG17 Forced RLS + GCS gs:// & S3 s3:// + vllm#12174)
# =============================================================================


@pytest.mark.postgres
def test_live_pg17_adapter_sync_sidecar_gcs_s3_cosign_multipod_and_rollback(
    tmp_path: Path,
) -> None:
    """Verifies AdapterSyncSidecar against live PG17 RLS, gs:// & s3://, cosign, & 3-pod vLLM."""
    settings = ScoutSettings()
    rls_mgr = RLSConnectionManager(settings.db_app_url)
    pg_repo = PostgresRLSRepository()

    # Guard: AdapterSyncSidecar MUST reject privileged postgres admin DSN
    admin_mgr = RLSConnectionManager(settings.db_admin_url)
    ec_priv = ec.generate_private_key(ec.SECP256R1())
    ec_pub_pem = (
        ec_priv.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode("utf-8")
    )
    ed_priv = ed25519.Ed25519PrivateKey.generate()
    ed_pub_pem = (
        ed_priv.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode("utf-8")
    )
    verifier = CosignBlobVerifier(
        trusted_public_keys_pem={"gke-kms-ecdsa": ec_pub_pem, "onprem-ed25519": ed_pub_pem}
    )

    base_model = "google/gemma-4-26B-A4B-it"
    pinned_rev = hashlib.sha256(b"gemma-4-26B-A4B-it-rev-v1").hexdigest()

    with pytest.raises(PermissionError, match="privileged DB role"):
        AdapterSyncSidecar(
            expected_base_model_id=base_model,
            expected_base_model_revision_sha256=pinned_rev,
            staging_root=tmp_path / "rejected_stage",
            cosign_verifier=verifier,
            rls_conn_manager=admin_mgr,
        )

    tenant_a = uuid.uuid4()
    tenant_b = uuid.uuid4()
    dom_gcs = f"eval_m4_gcs_{uuid.uuid4().hex[:8]}"
    dom_s3 = f"eval_m4_s3_{uuid.uuid4().hex[:8]}"
    principal_a = PrincipalContext(
        user_id="mlops_a",
        role=RoleType.ENGINEER,
        tenant_id=tenant_a,
        allowed_scopes=["tenant_common", "enclave:payments"],
    )
    principal_b = PrincipalContext(
        user_id="mlops_b",
        role=RoleType.ENGINEER,
        tenant_id=tenant_b,
        allowed_scopes=["tenant_common"],
    )

    gcs_store = _make_gcs_mock_store(bucket="scout-gke-models")
    s3_store = S3CompatibleObjectStore(bucket="scout-s3-models", client=_InMemoryS3BotoMock())

    weights_v1 = _build_safetensors_bytes()
    weights_v1_sha = hashlib.sha256(weights_v1).hexdigest()
    cfg_v1_bytes = json.dumps(
        {
            "base_model_name_or_path": base_model,
            "base_model_revision_sha256": pinned_rev,
            "r": 16,
            "lora_alpha": 32,
            "use_rslora": True,
            "use_dora": False,
        }
    ).encode("utf-8")

    # Upload Cartridge 1 to GCS (gs://) with Sigstore .bundle signature (ECDSA P-256)
    gcs_w_key = build_tenant_domain_key(
        tenant_a, dom_gcs, "adapters/lora_gcs_v1/adapter_model.safetensors"
    )
    gcs_c_key = build_tenant_domain_key(
        tenant_a, dom_gcs, "adapters/lora_gcs_v1/adapter_config.json"
    )
    gcs_s_key = build_tenant_domain_key(
        tenant_a, dom_gcs, "adapters/lora_gcs_v1/adapter_model.safetensors.bundle"
    )
    gcs_store.put_bytes(gcs_w_key, weights_v1, expected_tenant_id=tenant_a)
    gcs_store.put_bytes(gcs_c_key, cfg_v1_bytes, expected_tenant_id=tenant_a)
    gcs_bundle_sig = sign_blob_for_cosign(ec_priv, weights_v1, as_sigstore_bundle=True)
    gcs_store.put_bytes(gcs_s_key, gcs_bundle_sig.encode("utf-8"), expected_tenant_id=tenant_a)

    # Upload Cartridge 2 to S3 (s3://) with detached .sig signature (Ed25519)
    weights_s3 = _build_safetensors_bytes(
        (
            "base_model.model.layers.1.self_attn.k_proj.lora_A.weight",
            "base_model.model.layers.1.self_attn.k_proj.lora_B.weight",
        )
    )
    weights_s3_sha = hashlib.sha256(weights_s3).hexdigest()
    s3_w_key = build_tenant_domain_key(
        tenant_a, dom_s3, "adapters/lora_s3_v1/adapter_model.safetensors"
    )
    s3_c_key = build_tenant_domain_key(tenant_a, dom_s3, "adapters/lora_s3_v1/adapter_config.json")
    s3_s_key = build_tenant_domain_key(
        tenant_a, dom_s3, "adapters/lora_s3_v1/adapter_model.safetensors.sig"
    )
    s3_store.put_bytes(s3_w_key, weights_s3, expected_tenant_id=tenant_a)
    s3_store.put_bytes(s3_c_key, cfg_v1_bytes, expected_tenant_id=tenant_a)
    s3_sig = sign_blob_for_cosign(ed_priv, weights_s3, as_sigstore_bundle=False)
    s3_store.put_bytes(s3_s_key, s3_sig.encode("utf-8"), expected_tenant_id=tenant_a)

    fleet = _SimulatedVLLMPodFleet(base_model_id=base_model, pod_count=3)
    router = AdapterRouter(domains_dir=tmp_path / "domains", default_base_model_id=base_model)
    sidecar = AdapterSyncSidecar(
        expected_base_model_id=base_model,
        expected_base_model_revision_sha256=pinned_rev,
        staging_root=tmp_path / "pod_tmpfs",
        cosign_verifier=verifier,
        tenant_id=tenant_a,
        rls_conn_manager=rls_mgr,
        object_stores_by_scheme={"gs": gcs_store, "s3": s3_store},
        vllm_pod_endpoints=fleet.pod_urls,
        http_client=fleet.build_http_client(),
        adapter_router=router,
    )

    try:
        with rls_mgr.scoped_transaction(principal_a, enable_hnsw_iterative_scan=False) as conn:
            pg_repo.register_model_artifact(
                conn,
                ModelArtifactRecord(
                    tenant_id=tenant_a,
                    domain_id=dom_gcs,
                    artifact_id="lora_gcs_v1",
                    base_model_id=base_model,
                    base_model_revision_sha256=pinned_rev,
                    storage_uri=f"gs://scout-gke-models/{gcs_w_key}",
                    weights_object_key=gcs_w_key,
                    config_object_key=gcs_c_key,
                    signature_object_key=gcs_s_key,
                    weights_sha256=weights_v1_sha,
                    cosign_key_id="gke-kms-ecdsa",
                    lora_rank=16,
                    lora_alpha=32,
                    use_rslora=True,
                    use_dora=False,
                    training_scope_ceiling=["tenant_common", "enclave:payments"],
                    allowed_scopes=["tenant_common", "enclave:payments"],
                    is_active=True,
                    metadata_json={"config_sha256": hashlib.sha256(cfg_v1_bytes).hexdigest()},
                ),
            )
            pg_repo.register_model_artifact(
                conn,
                ModelArtifactRecord(
                    tenant_id=tenant_a,
                    domain_id=dom_s3,
                    artifact_id="lora_s3_v1",
                    base_model_id=base_model,
                    base_model_revision_sha256=pinned_rev,
                    storage_uri=f"s3://scout-s3-models/{s3_w_key}",
                    weights_object_key=s3_w_key,
                    config_object_key=s3_c_key,
                    signature_object_key=s3_s_key,
                    weights_sha256=weights_s3_sha,
                    cosign_key_id="onprem-ed25519",
                    lora_rank=16,
                    lora_alpha=32,
                    use_rslora=True,
                    use_dora=False,
                    training_scope_ceiling=["tenant_common"],
                    allowed_scopes=["tenant_common"],
                    is_active=True,
                ),
            )

        # Tenant B registers a cartridge in PG17 -> Tenant A sidecar MUST NOT see it (FORCE RLS)
        with rls_mgr.scoped_transaction(principal_b, enable_hnsw_iterative_scan=False) as conn_b:
            pg_repo.register_model_artifact(
                conn_b,
                ModelArtifactRecord(
                    tenant_id=tenant_b,
                    domain_id=dom_gcs,
                    artifact_id="lora_tenant_b_secret",
                    base_model_id=base_model,
                    base_model_revision_sha256=pinned_rev,
                    storage_uri=f"gs://scout-gke-models/tenants/{tenant_b}/domains/{dom_gcs}/x",
                    weights_object_key=f"tenants/{tenant_b}/domains/{dom_gcs}/a.safetensors",
                    config_object_key=f"tenants/{tenant_b}/domains/{dom_gcs}/adapter_config.json",
                    weights_sha256=weights_v1_sha,
                    training_scope_ceiling=["tenant_common"],
                    allowed_scopes=["tenant_common"],
                    is_active=True,
                ),
            )

        # 1. Reconcile: both gs:// and s3:// cartridges load across ALL 3 vLLM pods
        report_1 = sidecar.reconcile_once()
        assert report_1.failed_artifacts == {}
        assert report_1.newly_loaded == ("lora_gcs_v1", "lora_s3_v1")
        assert report_1.loaded_artifacts == ("lora_gcs_v1", "lora_s3_v1")
        assert "lora_tenant_b_secret" not in report_1.active_in_db

        pod_parity = sidecar.verify_multi_pod_parity()
        assert len(pod_parity) == 3
        for _pod_url, model_set in pod_parity.items():
            assert {"lora_gcs_v1", "lora_s3_v1"}.issubset(model_set)

        # AdapterRouter automatically received live specialization & ceiling
        assert router.route(dom_gcs, principal_a).lora_request == "lora_gcs_v1"

        # 2. Adversarial Scenario A: Wrong base_model_revision_sha256 on v2 -> rejected, v1 stays!
        bad_rev = hashlib.sha256(b"upgraded-base-model-rev-v2").hexdigest()
        with rls_mgr.scoped_transaction(principal_a, enable_hnsw_iterative_scan=False) as conn:
            pg_repo.register_model_artifact(
                conn,
                ModelArtifactRecord(
                    tenant_id=tenant_a,
                    domain_id=dom_gcs,
                    artifact_id="lora_gcs_bad_rev",
                    base_model_id=base_model,
                    base_model_revision_sha256=bad_rev,
                    storage_uri=f"gs://scout-gke-models/{gcs_w_key}",
                    weights_object_key=gcs_w_key,
                    config_object_key=gcs_c_key,
                    signature_object_key=gcs_s_key,
                    weights_sha256=weights_v1_sha,
                    cosign_key_id="gke-kms-ecdsa",
                    training_scope_ceiling=["tenant_common", "enclave:payments"],
                    allowed_scopes=["tenant_common", "enclave:payments"],
                    is_active=True,
                ),
            )
        report_bad_rev = sidecar.reconcile_once()
        assert "lora_gcs_bad_rev" in report_bad_rev.failed_artifacts
        assert "base_model_revision_sha256" in report_bad_rev.failed_artifacts["lora_gcs_bad_rev"]
        assert sidecar.loaded_cartridges[dom_gcs].artifact_id == "lora_gcs_v1"

        # 3. Adversarial Scenario B: Tampered cosign signature on S3 -> rejected fail-closed!
        tampered_s_key = build_tenant_domain_key(
            tenant_a, dom_s3, "adapters/lora_s3_tampered/adapter_model.safetensors.sig"
        )
        bad_priv = ed25519.Ed25519PrivateKey.generate()
        s3_store.put_bytes(
            tampered_s_key,
            sign_blob_for_cosign(bad_priv, weights_s3).encode("utf-8"),
            expected_tenant_id=tenant_a,
        )
        with rls_mgr.scoped_transaction(principal_a, enable_hnsw_iterative_scan=False) as conn:
            pg_repo.register_model_artifact(
                conn,
                ModelArtifactRecord(
                    tenant_id=tenant_a,
                    domain_id=dom_s3,
                    artifact_id="lora_s3_tampered",
                    base_model_id=base_model,
                    base_model_revision_sha256=pinned_rev,
                    storage_uri=f"s3://scout-s3-models/{s3_w_key}",
                    weights_object_key=s3_w_key,
                    config_object_key=s3_c_key,
                    signature_object_key=tampered_s_key,
                    weights_sha256=weights_s3_sha,
                    cosign_key_id="onprem-ed25519",
                    training_scope_ceiling=["tenant_common"],
                    allowed_scopes=["tenant_common"],
                    is_active=True,
                ),
            )
        report_bad_sig = sidecar.reconcile_once()
        assert "lora_s3_tampered" in report_bad_sig.failed_artifacts
        assert "cosign" in report_bad_sig.failed_artifacts["lora_s3_tampered"].lower()
        assert sidecar.loaded_cartridges[dom_s3].artifact_id == "lora_s3_v1"

        # 4. Adversarial Scenario C (vllm#12174): Pod 2 fails load -> Pod 0 & Pod 1 rolled back!
        gcs_v2_w_key = build_tenant_domain_key(
            tenant_a, dom_gcs, "adapters/lora_gcs_v2/adapter_model.safetensors"
        )
        gcs_store.put_bytes(gcs_v2_w_key, weights_v1, expected_tenant_id=tenant_a)
        fleet.fail_load_on_pod.add("http://vllm-pod-2.vllm-headless:8001/v1")
        with rls_mgr.scoped_transaction(principal_a, enable_hnsw_iterative_scan=False) as conn:
            pg_repo.register_model_artifact(
                conn,
                ModelArtifactRecord(
                    tenant_id=tenant_a,
                    domain_id=dom_gcs,
                    artifact_id="lora_gcs_v2",
                    base_model_id=base_model,
                    base_model_revision_sha256=pinned_rev,
                    storage_uri=f"gs://scout-gke-models/{gcs_v2_w_key}",
                    weights_object_key=gcs_v2_w_key,
                    config_object_key=gcs_c_key,
                    signature_object_key=gcs_s_key,
                    weights_sha256=weights_v1_sha,
                    cosign_key_id="gke-kms-ecdsa",
                    training_scope_ceiling=["tenant_common", "enclave:payments"],
                    allowed_scopes=["tenant_common", "enclave:payments"],
                    is_active=True,
                ),
            )
        report_partial_fail = sidecar.reconcile_once()
        assert "lora_gcs_v2" in report_partial_fail.failed_artifacts
        assert "rolled back 2 pod(s)" in report_partial_fail.failed_artifacts["lora_gcs_v2"]
        # ZERO pods have lora_gcs_v2 loaded, and ALL 3 pods still serve lora_gcs_v1!
        for pod_models in fleet.loaded_on_pod.values():
            assert "lora_gcs_v2" not in pod_models
            assert "lora_gcs_v1" in pod_models

        # 5. Heal Pod 2 -> Blue/Green swap loads lora_gcs_v2 across all 3 pods & unloads lora_gcs_v1
        fleet.fail_load_on_pod.clear()
        report_healed = sidecar.reconcile_once()
        assert report_healed.newly_loaded == ("lora_gcs_v2",)
        assert report_healed.unloaded_retired == ("lora_gcs_v1",)
        for pod_models in fleet.loaded_on_pod.values():
            assert "lora_gcs_v2" in pod_models
            assert "lora_gcs_v1" not in pod_models

        # 6. Revoke lora_gcs_v2 & retire lora_s3_tampered -> unloaded from all 3 pods
        with rls_mgr.scoped_transaction(principal_a, enable_hnsw_iterative_scan=False) as conn:
            pg_repo.retire_model_artifact(conn, dom_gcs, "lora_gcs_v2", revoke=True)
            pg_repo.retire_model_artifact(conn, dom_s3, "lora_s3_tampered", revoke=False)
            with pytest.raises(PermissionError, match="Revoked"):
                pg_repo.activate_model_artifact(conn, dom_gcs, "lora_gcs_v2")

        report_retired = sidecar.reconcile_once()
        assert set(report_retired.unloaded_retired) == {"lora_gcs_v2", "lora_s3_v1"}
        assert report_retired.loaded_artifacts == ()
        for pod_models in fleet.loaded_on_pod.values():
            assert pod_models == {}
    finally:
        sidecar.close()
        with psycopg.connect(settings.db_admin_url) as admin_conn:
            admin_conn.execute(
                "DELETE FROM model_artifacts WHERE tenant_id = ANY(%s);",
                ([tenant_a, tenant_b],),
            )
            admin_conn.commit()


# =============================================================================
# 3. OWASP ASI05 Elimination, Keyless Vertex ADC Handoff, & Token Isolation Regressions
# =============================================================================


def test_asi05_static_audit_zero_host_patching_or_pytest_in_handoff_and_executor() -> None:
    """Proves handoff.py has zero get_repo_root/GitWorktreeSandbox/subprocess/pytest calls."""
    handoff_src = Path(handoff_mod.__file__).read_text(encoding="utf-8")
    handoff_tree = ast.parse(handoff_src)
    imported_names: set[str] = set()
    called_names: set[str] = set()
    for node in ast.walk(handoff_tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_names.add(node.module)
            for alias in node.names:
                imported_names.add(alias.name)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called_names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called_names.add(node.func.attr)

    assert "subprocess" not in imported_names
    assert "get_repo_root" not in imported_names
    assert "GitWorktreeSandbox" not in imported_names
    assert "execute_implementation" not in called_names
    assert "apply_patch" not in called_names


def test_asi05_pytest_subprocess_prohibited_in_sandbox(tmp_path: Path) -> None:
    """GitWorktreeSandbox rejects pytest & python -m pytest, and runs AST check by default."""
    repo_dir = tmp_path / "sandbox_repo"
    repo_dir.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=repo_dir, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo_dir, check=True)
    (repo_dir / "mod.py").write_text("def ok() -> int:\n    return 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo_dir, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo_dir, check=True, capture_output=True)

    with GitWorktreeSandbox(repo_root=repo_dir, require_confirmation=False) as sandbox:
        for forbidden_cmd in (
            ["pytest", "-q"],
            ["/usr/local/bin/pytest", "tests/"],
            ["python3", "-m", "pytest"],
        ):
            with pytest.raises(PermissionError, match="ASI05"):
                sandbox.run_verification(
                    lint_cmd=["python3", "-c", "pass"],
                    test_cmd=forbidden_cmd,
                )

        # Default test_cmd=None runs pure in-process AST syntax check without spawning pytest
        lint_ok, test_ok, details = sandbox.run_verification(
            lint_cmd=["python3", "-c", "pass"],
            test_cmd=None,
        )
        assert lint_ok is True
        assert test_ok is True
        assert "Static in-process AST syntax check passed" in details


def test_cross_tenant_token_hijack_concurrent_replay_and_asi05_custom_patch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies cross-tenant token isolation, atomic single-consume, and ASI05 custom_patch."""
    secret = "m4-01-test-hmac-secret-minimum-32-characters!!"
    monkeypatch.setenv("SCOUT_AUTH_HMAC_SECRET", secret)

    store = ConfirmationTokenStore()
    mem_repo = InMemoryRLSRepository()
    app.dependency_overrides[get_confirmation_token_store] = lambda: store
    app.dependency_overrides[get_repository] = lambda: mem_repo
    try:
        tenant_a = uuid.uuid4()
        tenant_b = uuid.uuid4()
        tok_a = create_signed_principal_token(
            user_id="shared_eng_login",
            role=RoleType.ENGINEER,
            allowed_scopes=["public_eng"],
            secret_key=secret,
            tenant_id=tenant_a,
        )
        tok_b = create_signed_principal_token(
            user_id="shared_eng_login",
            role=RoleType.ENGINEER,
            allowed_scopes=["public_eng"],
            secret_key=secret,
            tenant_id=tenant_b,
        )
        pack = _make_pack()

        client_a = TestClient(app, headers={"Authorization": f"Bearer {tok_a}"})
        client_b = TestClient(app, headers={"Authorization": f"Bearer {tok_b}"})

        # 1. ASI05: custom_patch on /handoff/preview AND /handoff/apply rejected with 400
        bad_preview = client_a.post(
            "/api/v1/handoff/preview",
            json={
                "pack": pack.model_dump(),
                "model_id": "gemini-3.8-flash",
                "custom_patch": "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n+x = 1\n",
            },
        )
        assert bad_preview.status_code == 400
        assert "ASI05" in bad_preview.json()["detail"]

        ok_preview = client_a.post(
            "/api/v1/handoff/preview",
            json={"pack": pack.model_dump(), "model_id": "gemini-3.8-flash"},
        )
        assert ok_preview.status_code == 200
        conf_token = ok_preview.json()["confirmation_token"]

        bad_apply_patch = client_a.post(
            "/api/v1/handoff/apply",
            json={
                "confirmation_token": conf_token,
                "confirmed": True,
                "custom_patch": "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n+x = 2\n",
            },
        )
        assert bad_apply_patch.status_code == 400
        assert "ASI05" in bad_apply_patch.json()["detail"]

        # 2. Tenant B with colliding user_id gets 403 AND does NOT destroy Tenant A token
        hijack_res = client_b.post(
            "/api/v1/handoff/apply",
            json={"confirmation_token": conf_token, "confirmed": True},
        )
        assert hijack_res.status_code == 403

        # 3. 12-thread concurrent /handoff/apply race consumes token AT MOST ONCE
        barrier = threading.Barrier(12)

        def _attempt_apply(_idx: int) -> int:
            local_client = TestClient(app, headers={"Authorization": f"Bearer {tok_a}"})
            barrier.wait(timeout=5.0)
            res = local_client.post(
                "/api/v1/handoff/apply",
                json={"confirmation_token": conf_token, "confirmed": True},
            )
            return res.status_code

        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
            statuses = list(pool.map(_attempt_apply, range(12)))

        assert statuses.count(200) == 1
        assert statuses.count(400) == 11
    finally:
        app.dependency_overrides.pop(get_confirmation_token_store, None)
        app.dependency_overrides.pop(get_repository, None)


@pytest.mark.postgres
def test_live_pg17_graph_adapter_budget_and_telemetry_persistence() -> None:
    """Proves PostgresScoutGraphAdapter enforces session budget & records telemetry under RLS."""
    settings = ScoutSettings()
    rls_mgr = RLSConnectionManager(settings.db_app_url)
    graph_adapter = PostgresScoutGraphAdapter(rls_mgr, settings)

    tenant_id = uuid.uuid4()
    dom_id = f"eval_m4_budget_{uuid.uuid4().hex[:8]}"
    user_id = f"eng_budget_{uuid.uuid4().hex[:8]}"
    principal = PrincipalContext(
        user_id=user_id,
        role=RoleType.ENGINEER,
        tenant_id=tenant_id,
        allowed_scopes=["public_eng"],
        max_session_paid_tokens=8_000,
    )
    executor = GeminiImplementationExecutor(require_confirmation=False)
    pack = _make_pack(domain_id=dom_id, pack_tokens=2_500)

    try:
        # 1. Record telemetry via PostgresScoutGraphAdapter -> persists into execution_telemetry
        tele = executor.record_handoff_confirmation_telemetry(
            pack=pack,
            principal=principal,
            latency_ms=12,
            repo=graph_adapter,
        )
        with rls_mgr.scoped_transaction(principal, enable_hnsw_iterative_scan=False) as conn:
            fetched = PostgresRLSRepository().get_execution_telemetry(conn, tele.trace_id)
            assert fetched is not None
            assert fetched.context_pack_tokens_sent == 2_500

            # Seed an additional 2,000 tokens -> total consumed == 4,500
            PostgresRLSRepository().record_execution_telemetry(
                conn,
                ExecutionTelemetryRecord(
                    trace_id=uuid.uuid4(),
                    tenant_id=tenant_id,
                    domain_id=dom_id,
                    principal_id=user_id,
                    principal_role="engineer",
                    execution_mode=ExecutionMode.GEMINI_HANDOFF,
                    local_tokens_read=10_000,
                    context_pack_tokens_sent=2_000,
                    paid_tokens_saved=8_000,
                    estimated_usd_saved=0.03,
                    preflight_passed_first_try=True,
                    zero_touch_sufficient=True,
                    latency_ms=5,
                ),
            )

        # 2. Consumed 4,500 + pack 2,500 + max_output 4,096 = 11,096 > cap 8,000 -> raises!
        with pytest.raises(BudgetExceededError, match="consumed 4500"):
            executor.check_principal_budget(
                principal, pack, max_output_tokens=4096, repo=graph_adapter
            )
    finally:
        with psycopg.connect(settings.db_admin_url) as admin_conn:
            admin_conn.execute(
                "DELETE FROM execution_telemetry WHERE tenant_id = %s;",
                (tenant_id,),
            )
            admin_conn.commit()


def test_keyless_adc_vertexai_handoff_egress_gate_and_sa_key_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies optional keyless ADC vertexai=True handoff fails closed unless egress enabled."""
    pack = _make_pack()
    principal = PrincipalContext(
        user_id="gke_eng",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng"],
        max_session_paid_tokens=50_000,
    )

    # 1. Default allow_external_cloud_egress=False blocks keyless Vertex handoff ($0 spent)
    blocked_exec = GeminiImplementationExecutor(
        model_id="gemini-3.8-flash",
        use_keyless_adc=True,
        allow_external_cloud_egress=False,
    )
    assert blocked_exec.is_live_configured() is False
    with pytest.raises(PermissionError, match="allow_external_cloud_egress=False"):
        blocked_exec.synthesize_keyless_vertex_handoff_patch(pack, principal)

    # 2. Static service_account JSON key in GOOGLE_APPLICATION_CREDENTIALS rejected
    sa_json = tmp_path / "static_sa_key.json"
    sa_json.write_text(
        json.dumps({"type": "service_account", "private_key": "..."}), encoding="utf-8"
    )
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(sa_json))
    sa_exec = GeminiImplementationExecutor(
        model_id="gemini-3.8-flash",
        use_keyless_adc=True,
        allow_external_cloud_egress=True,
    )
    with pytest.raises(PermissionError, match="disableServiceAccountKeyCreation"):
        sa_exec.resolve_keyless_adc_credentials()
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)

    # 3. Keyless WIF / ADC discovery + vertexai=True synthesis succeeds when egress enabled
    fake_wif_creds = MagicMock()
    fake_wif_creds.token = "ya29.gke-workload-identity-ephemeral"

    mock_resp = MagicMock()
    mock_resp.text = (
        "```diff\n"
        "--- a/src/context_scout/schemas/contracts.py\n"
        "+++ b/src/context_scout/schemas/contracts.py\n"
        "@@ -1,2 +1,3 @@\n"
        " class PrincipalContext:\n"
        '+    """Verified via keyless Vertex ADC."""\n'
        "     pass\n"
        "```"
    )
    mock_resp.usage_metadata.prompt_token_count = 900
    mock_resp.usage_metadata.candidates_token_count = 45
    mock_resp.usage_metadata.total_token_count = 945

    mock_genai_client = MagicMock()
    mock_genai_client.models.generate_content.return_value = mock_resp

    wif_exec = GeminiImplementationExecutor(
        model_id="gemini-3.8-flash",
        location="europe-west3",
        vertexai=True,
        use_keyless_adc=True,
        allow_external_cloud_egress=True,
        adc_loader=lambda scopes: (fake_wif_creds, "gcp-eu-scout-prod"),
    )
    resolved_creds, resolved_proj = wif_exec.resolve_keyless_adc_credentials()
    assert resolved_creds is fake_wif_creds
    assert resolved_proj == "gcp-eu-scout-prod"

    live_client = wif_exec._create_genai_client()
    assert live_client.vertexai is True
    assert live_client._api_client.project == "gcp-eu-scout-prod"
    assert live_client._api_client.location == "europe-west3"

    wif_exec._injected_client = mock_genai_client
    diff_out = wif_exec.synthesize_keyless_vertex_handoff_patch(pack, principal)
    assert "Verified via keyless Vertex ADC." in diff_out
    assert detect_architecture_profile("google/gemma-4-26B-A4B-it").family == "gemma-4"


def test_sglang_context_length_safetensors_bombs_and_sidecar_daemon_lifecycle(
    tmp_path: Path,
) -> None:
    """Tests SGLang context_length discovery, safetensors header bombs, & sidecar start/stop."""
    from context_scout.agents.open_weights_vllm_executor import inspect_safetensors_header
    from context_scout.serving.adapter_sync_sidecar import CosignVerificationError
    from context_scout.storage.object_store import InMemoryObjectStore

    # 1. SGLang /v1/models returns 'context_length' instead of 'max_model_len'
    def _sglang_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "mistralai/Devstral-Small-2505",
                            "context_length": 24_576,
                        }
                    ]
                },
            )
        return httpx.Response(404)

    sglang_exec = OpenWeightsVLLMExecutor(
        model_id="mistralai/Devstral-Small-2505",
        runtime_backend="sglang",
        http_client=httpx.Client(transport=httpx.MockTransport(_sglang_handler)),
    )
    assert sglang_exec.max_model_len == 24_576

    # 2. Corrupt / oversized .safetensors header bombs fail closed without memory exhaustion
    for corrupt_blob in (
        b"short",
        b"\xff\xff\xff\xff\x00\x00\x00\x00{}",  # 4 GiB claimed header length on 10-byte blob
        b"\x02\x00\x00\x00\x00\x00\x00\x00{}",  # 0 tensor entries
    ):
        with pytest.raises(LoRACompatibilityError):
            inspect_safetensors_header(corrupt_blob)

    # 3. CosignBlobVerifier with zero trusted keys fails closed
    empty_verifier = CosignBlobVerifier()
    with pytest.raises(CosignVerificationError, match="No trusted cosign public keys"):
        empty_verifier.verify(b"payload", "YWJj")

    # 4. AdapterSyncSidecar background daemon loop + InMemoryRLSRepository + InMemoryObjectStore
    ed_priv = ed25519.Ed25519PrivateKey.generate()
    ed_pub_pem = (
        ed_priv.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode("utf-8")
    )
    mem_repo = InMemoryRLSRepository()
    mem_store = InMemoryObjectStore(bucket="mem-models")
    tid = uuid.uuid4()
    p = PrincipalContext(
        user_id="mem_ops",
        role=RoleType.ENGINEER,
        tenant_id=tid,
        allowed_scopes=["tenant_common"],
    )
    base_model = "google/gemma-4-E4B-it"
    rev_sha = "c" * 64
    w_bytes = _build_safetensors_bytes()
    w_sha = hashlib.sha256(w_bytes).hexdigest()
    cfg_bytes = json.dumps(
        {
            "base_model_name_or_path": base_model,
            "base_model_revision_sha256": rev_sha,
            "r": 8,
            "lora_alpha": 16,
            "use_rslora": True,
            "use_dora": False,
        }
    ).encode("utf-8")
    w_key = build_tenant_domain_key(tid, "mem_dom", "adapters/lora_mem_1/adapter_model.safetensors")
    c_key = build_tenant_domain_key(tid, "mem_dom", "adapters/lora_mem_1/adapter_config.json")
    mem_store.put_bytes(w_key, w_bytes, expected_tenant_id=tid)
    mem_store.put_bytes(c_key, cfg_bytes, expected_tenant_id=tid)

    mem_repo.register_model_artifact(
        p,
        ModelArtifactRecord(
            tenant_id=tid,
            domain_id="mem_dom",
            artifact_id="lora_mem_1",
            base_model_id=base_model,
            base_model_revision_sha256=rev_sha,
            storage_uri=f"memory://mem-models/{w_key}",
            weights_object_key=w_key,
            config_object_key=c_key,
            weights_sha256=w_sha,
            cosign_signature_b64=sign_blob_for_cosign(ed_priv, w_bytes),
            lora_rank=8,
            lora_alpha=16,
            training_scope_ceiling=["tenant_common"],
            allowed_scopes=["tenant_common"],
            is_active=True,
        ),
    )

    fleet = _SimulatedVLLMPodFleet(base_model_id=base_model, pod_count=2)
    sidecar = AdapterSyncSidecar(
        expected_base_model_id=base_model,
        expected_base_model_revision_sha256=rev_sha,
        staging_root=tmp_path / "daemon_stage",
        cosign_verifier=CosignBlobVerifier({"default": ed_pub_pem}),
        tenant_id=tid,
        in_memory_repo=mem_repo,
        default_object_store=mem_store,
        vllm_pod_endpoints=fleet.pod_urls,
        http_client=fleet.build_http_client(),
        poll_interval_seconds=0.05,
    )
    sidecar.start()
    try:
        import time

        deadline = time.monotonic() + 3.0
        while "mem_dom" not in sidecar.loaded_cartridges and time.monotonic() < deadline:
            time.sleep(0.03)
        assert sidecar.loaded_cartridges["mem_dom"].artifact_id == "lora_mem_1"
    finally:
        sidecar.close()


# =============================================================================
# 4. Adversarial Review Regression Suite (Multi-LoRA, Sidecar & Supply Chain)
# =============================================================================


def test_cross_tenant_adapter_router_isolation_and_strict_rslora(
    tmp_path: Path,
) -> None:
    """Blocks cross-tenant LoRA weight leakage on shared domain_id slugs & omitted use_rslora."""
    from context_scout.storage.object_store import InMemoryObjectStore

    ed_priv = ed25519.Ed25519PrivateKey.generate()
    ed_pub_pem = (
        ed_priv.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode("utf-8")
    )
    tenant_a = uuid.uuid4()
    tenant_b = uuid.uuid4()
    shared_domain = "code_architecture_scout"
    base_model = "google/gemma-4-26B-A4B-it"
    rev_sha = "1" * 64

    mem_repo = InMemoryRLSRepository()
    mem_store = InMemoryObjectStore(bucket="mem-models")
    principal_a = PrincipalContext(
        user_id="eng_a",
        role=RoleType.ENGINEER,
        tenant_id=tenant_a,
        allowed_scopes=["tenant_common"],
    )
    principal_b = PrincipalContext(
        user_id="eng_b",
        role=RoleType.ENGINEER,
        tenant_id=tenant_b,
        allowed_scopes=["tenant_common"],
    )

    w_bytes = _build_safetensors_bytes()
    w_sha = hashlib.sha256(w_bytes).hexdigest()
    cfg_bytes = json.dumps(
        {
            "base_model_name_or_path": base_model,
            "base_model_revision_sha256": rev_sha,
            "r": 16,
            "lora_alpha": 32,
            "use_rslora": True,
            "use_dora": False,
        }
    ).encode("utf-8")
    w_key = build_tenant_domain_key(
        tenant_a, shared_domain, "adapters/lora_tenant_a_only/adapter_model.safetensors"
    )
    c_key = build_tenant_domain_key(
        tenant_a, shared_domain, "adapters/lora_tenant_a_only/adapter_config.json"
    )
    mem_store.put_bytes(w_key, w_bytes, expected_tenant_id=tenant_a)
    mem_store.put_bytes(c_key, cfg_bytes, expected_tenant_id=tenant_a)

    mem_repo.register_model_artifact(
        principal_a,
        ModelArtifactRecord(
            tenant_id=tenant_a,
            domain_id=shared_domain,
            artifact_id="lora_tenant_a_only",
            base_model_id=base_model,
            base_model_revision_sha256=rev_sha,
            storage_uri=f"memory://mem-models/{w_key}",
            weights_object_key=w_key,
            config_object_key=c_key,
            weights_sha256=w_sha,
            cosign_signature_b64=sign_blob_for_cosign(ed_priv, w_bytes),
            lora_rank=16,
            lora_alpha=32,
            training_scope_ceiling=["tenant_common"],
            allowed_scopes=["tenant_common"],
            is_active=True,
        ),
    )

    shared_router = AdapterRouter(
        domains_dir=tmp_path / "domains",
        default_base_model_id=base_model,
    )
    fleet = _SimulatedVLLMPodFleet(base_model_id=base_model, pod_count=1)
    sidecar_a = AdapterSyncSidecar(
        expected_base_model_id=base_model,
        expected_base_model_revision_sha256=rev_sha,
        staging_root=tmp_path / "stage",
        cosign_verifier=CosignBlobVerifier({"default": ed_pub_pem}),
        tenant_id=tenant_a,
        in_memory_repo=mem_repo,
        default_object_store=mem_store,
        vllm_pod_endpoints=fleet.pod_urls,
        http_client=fleet.build_http_client(),
        adapter_router=shared_router,
    )
    try:
        rep = sidecar_a.reconcile_once()
        assert rep.failed_artifacts == {}

        # Tenant A routes to its registered LoRA cartridge
        dec_a = shared_router.route(shared_domain, principal_a)
        assert dec_a.routed_to_lora is True
        assert dec_a.lora_adapter_id == "lora_tenant_a_only"

        # Tenant B (with identical scopes + domain_id!) MUST NEVER route to Tenant A's LoRA
        dec_b = shared_router.route(shared_domain, principal_b)
        assert dec_b.routed_to_lora is False
        assert dec_b.lora_request is None

        # Omitted use_rslora in lora_policy MUST fail closed (not only use_rslora=False)
        shared_router.register_specialization(
            "missing_rslora_dom",
            {
                "domain_id": "missing_rslora_dom",
                "base_model_id": base_model,
                "training_scope_ceiling": ["tenant_common"],
                "lora_policy": {"enabled": True, "rank": 16, "use_dora": False},
            },
        )
        assert (
            shared_router.route(
                "missing_rslora_dom", principal_a, verify_weights=False
            ).routed_to_lora
            is False
        )
    finally:
        sidecar_a.close()


def test_sidecar_failed_rollback_recovery_duplicate_load_400_and_stranded_parity(
    tmp_path: Path,
) -> None:
    """Verifies recovery when rollback /unload fails & real vLLM returns 400 on duplicate load."""
    from context_scout.serving.adapter_sync_sidecar import AdapterSyncError
    from context_scout.storage.object_store import InMemoryObjectStore

    ed_priv = ed25519.Ed25519PrivateKey.generate()
    ed_pub_pem = (
        ed_priv.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode("utf-8")
    )
    tid = uuid.uuid4()
    base_model = "google/gemma-4-E4B-it"
    rev_sha = "2" * 64
    p = PrincipalContext(
        user_id="ops",
        role=RoleType.ENGINEER,
        tenant_id=tid,
        allowed_scopes=["tenant_common"],
    )
    mem_repo = InMemoryRLSRepository()
    mem_store = InMemoryObjectStore(bucket="mem-models")

    w_bytes = _build_safetensors_bytes()
    w_sha = hashlib.sha256(w_bytes).hexdigest()
    cfg_bytes = json.dumps(
        {
            "base_model_name_or_path": base_model,
            "base_model_revision_sha256": rev_sha,
            "r": 16,
            "lora_alpha": 32,
            "use_rslora": True,
            "use_dora": False,
        }
    ).encode("utf-8")
    w_key = build_tenant_domain_key(tid, "dom_ha", "adapters/lora_ha_v1/adapter_model.safetensors")
    c_key = build_tenant_domain_key(tid, "dom_ha", "adapters/lora_ha_v1/adapter_config.json")
    mem_store.put_bytes(w_key, w_bytes, expected_tenant_id=tid)
    mem_store.put_bytes(c_key, cfg_bytes, expected_tenant_id=tid)
    mem_repo.register_model_artifact(
        p,
        ModelArtifactRecord(
            tenant_id=tid,
            domain_id="dom_ha",
            artifact_id="lora_ha_v1",
            base_model_id=base_model,
            base_model_revision_sha256=rev_sha,
            storage_uri=f"memory://mem-models/{w_key}",
            weights_object_key=w_key,
            config_object_key=c_key,
            weights_sha256=w_sha,
            cosign_signature_b64=sign_blob_for_cosign(ed_priv, w_bytes),
            training_scope_ceiling=["tenant_common"],
            allowed_scopes=["tenant_common"],
            is_active=True,
        ),
    )

    pod_urls = ("http://vllm-0:8001", "http://vllm-1:8001", "http://vllm-2:8001")
    loaded_on_pod: dict[str, dict[str, str]] = {f"{u}/v1": {} for u in pod_urls}
    fail_load_pods: set[str] = {"http://vllm-2:8001/v1"}
    fail_unload_pods: set[str] = {"http://vllm-0:8001/v1"}

    def _strict_vllm_handler(request: httpx.Request) -> httpx.Response:
        pod_v1 = f"{request.url.scheme}://{request.url.host}:{request.url.port}/v1"
        if request.method == "GET" and request.url.path == "/v1/models":
            data = [
                {"id": base_model, "object": "model", "max_model_len": 32768},
                *[
                    {"id": k, "object": "model", "parent": base_model}
                    for k in loaded_on_pod[pod_v1]
                ],
            ]
            return httpx.Response(200, json={"data": data})
        if request.method == "POST" and request.url.path == "/v1/load_lora_adapter":
            body = json.loads(request.content.decode("utf-8"))
            lname = str(body["lora_name"])
            if pod_v1 in fail_load_pods:
                return httpx.Response(500, json={"error": "CUDA OOM"})
            # Real vLLM behavior: duplicate load_lora_adapter returns HTTP 400!
            if lname in loaded_on_pod[pod_v1]:
                return httpx.Response(
                    400,
                    json={"message": f"The lora adapter '{lname}' has already been loaded."},
                )
            loaded_on_pod[pod_v1][lname] = str(body["lora_path"])
            return httpx.Response(200, json={"status": "loaded"})
        if request.method == "POST" and request.url.path == "/v1/unload_lora_adapter":
            if pod_v1 in fail_unload_pods:
                raise httpx.ConnectError("Pod unreachable during rollback", request=request)
            body = json.loads(request.content.decode("utf-8"))
            loaded_on_pod[pod_v1].pop(str(body["lora_name"]), None)
            return httpx.Response(200, json={"status": "unloaded"})
        return httpx.Response(404)

    sidecar = AdapterSyncSidecar(
        expected_base_model_id=base_model,
        expected_base_model_revision_sha256=rev_sha,
        staging_root=tmp_path / "ha_stage",
        cosign_verifier=CosignBlobVerifier({"default": ed_pub_pem}),
        tenant_id=tid,
        in_memory_repo=mem_repo,
        default_object_store=mem_store,
        vllm_pod_endpoints=pod_urls,
        http_client=httpx.Client(transport=httpx.MockTransport(_strict_vllm_handler)),
    )
    try:
        # Cycle 1: Pod 2 fails load AND Pod 0 fails rollback unload -> Pod 0 is stranded!
        rep1 = sidecar.reconcile_once()
        assert len(rep1.failed_artifacts) == 1
        assert len(loaded_on_pod["http://vllm-0:8001/v1"]) == 1
        assert len(loaded_on_pod["http://vllm-1:8001/v1"]) == 0

        # verify_multi_pod_parity MUST detect the stranded adapter on Pod 0!
        with pytest.raises(AdapterSyncError):
            sidecar.verify_multi_pod_parity()

        # Cycle 2: Pods heal; reconcile_once MUST NOT wedge on Pod 0's already-loaded 400!
        fail_load_pods.clear()
        fail_unload_pods.clear()
        rep2 = sidecar.reconcile_once()
        assert rep2.failed_artifacts == {}
        assert rep2.loaded_artifacts == ("lora_ha_v1",)
        sidecar.verify_multi_pod_parity()
    finally:
        sidecar.close()


def test_staging_path_traversal_cosign_digest_and_uri_scheme_spoofing(
    tmp_path: Path,
) -> None:
    """Blocks ../<other_tenant> staging wipe, missing Sigstore messageDigest, & URI spoofing."""
    from context_scout.serving.adapter_sync_sidecar import (
        AdapterVerificationError,
        CosignVerificationError,
    )
    from context_scout.storage.object_store import InMemoryObjectStore

    ed_priv = ed25519.Ed25519PrivateKey.generate()
    ed_pub_pem = (
        ed_priv.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode("utf-8")
    )
    verifier = CosignBlobVerifier({"default": ed_pub_pem})
    payload = _build_safetensors_bytes()

    # 1. Sigstore bundle missing messageDigest (or null) MUST fail closed
    raw_sig = sign_blob_for_cosign(ed_priv, payload, as_sigstore_bundle=False)
    for bad_bundle in (
        json.dumps({"messageSignature": {"signature": raw_sig}}),
        json.dumps({"messageSignature": {"messageDigest": None, "signature": raw_sig}}),
    ):
        with pytest.raises(CosignVerificationError, match="messageDigest"):
            verifier.verify(payload, bad_bundle)

    # 2. Cross-tenant staging traversal (`domain_id="../<victim_tenant>/dom"`) MUST NOT create/wipe
    staging_root = tmp_path / "shared_staging"
    victim_tid = uuid.uuid4()
    victim_file = staging_root / str(victim_tid) / "dom" / "lora_v1" / "keep.txt"
    victim_file.parent.mkdir(parents=True)
    victim_file.write_text("victim", encoding="utf-8")

    attacker_tid = uuid.uuid4()
    mem_store = InMemoryObjectStore(bucket="mem-models")
    base_model = "google/gemma-4-E4B-it"
    rev_sha = "3" * 64
    sidecar = AdapterSyncSidecar(
        expected_base_model_id=base_model,
        expected_base_model_revision_sha256=rev_sha,
        staging_root=staging_root,
        cosign_verifier=verifier,
        tenant_id=attacker_tid,
        in_memory_repo=InMemoryRLSRepository(),
        default_object_store=mem_store,
    )
    try:
        # URI scheme spoofing (`http://` when default_object_store is configured) -> rejected!
        with pytest.raises(AdapterVerificationError, match="scheme"):
            sidecar._resolve_object_store_for_uri("http://evil.internal/bucket/key")

        # Bucket mismatch against configured store -> rejected!
        with pytest.raises(AdapterVerificationError, match="bucket"):
            sidecar._resolve_object_store_for_uri("memory://wrong-bucket/key")

        # Path traversal in domain_id or artifact_id -> rejected BEFORE touching disk!
        w_key = f"tenants/{attacker_tid}/domains/safe/adapter_model.safetensors"
        c_key = f"tenants/{attacker_tid}/domains/safe/adapter_config.json"
        cfg_bytes = json.dumps(
            {
                "base_model_name_or_path": base_model,
                "base_model_revision_sha256": rev_sha,
                "r": 16,
                "lora_alpha": 32,
                "use_rslora": True,
                "use_dora": False,
            }
        ).encode("utf-8")
        mem_store.put_bytes(w_key, payload, expected_tenant_id=attacker_tid)
        mem_store.put_bytes(c_key, cfg_bytes, expected_tenant_id=attacker_tid)

        traversal_rec = ModelArtifactRecord(
            tenant_id=attacker_tid,
            domain_id=f"../{victim_tid}/dom",
            artifact_id="lora_v1",
            base_model_id=base_model,
            base_model_revision_sha256=rev_sha,
            storage_uri=f"memory://mem-models/{w_key}",
            weights_object_key=w_key,
            config_object_key=c_key,
            weights_sha256=hashlib.sha256(payload).hexdigest(),
            cosign_signature_b64=raw_sig,
            training_scope_ceiling=["tenant_common"],
            allowed_scopes=["tenant_common"],
            is_active=True,
        )
        with pytest.raises(AdapterVerificationError):
            sidecar._verify_and_stage_artifact(traversal_rec)
        assert victim_file.exists() and victim_file.read_text(encoding="utf-8") == "victim"
    finally:
        sidecar.close()


def test_vllm_wire_top_level_fields_structural_prompt_and_mlx_shard_safety(
    tmp_path: Path,
) -> None:
    """Verifies top-level vLLM HTTP fields, <json_output> prompt, BaseModel error, & MLX shards."""
    captured_bodies: list[dict[str, Any]] = []

    def _vllm_wire_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={"data": [{"id": "Qwen/Qwen3-8B", "max_context_len": 16_384}]},
            )
        if request.url.path == "/v1/chat/completions":
            body = json.loads(request.content.decode("utf-8"))
            captured_bodies.append(body)
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '<json_output>{"summary":"ok",'
                                    '"cited_chunk_ids":["c1"],"confidence":0.9}</json_output>'
                                )
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                },
            )
        return httpx.Response(404)

    exec_vllm = OpenWeightsVLLMExecutor(
        model_id="Qwen/Qwen3-8B",
        http_client=httpx.Client(transport=httpx.MockTransport(_vllm_wire_handler)),
    )
    assert exec_vllm.max_model_len == 16_384  # 'max_context_len' without typo space

    _ = exec_vllm.complete_structured(
        "Extract summary",
        _ScoutSynthesisContract,
        enable_thinking=False,
    )
    sent = captured_bodies[-1]
    # Single canonical structural_tag constraint (no guided_decoding_backend on vLLM v0.34.0)
    assert "guided_decoding_backend" not in sent
    assert "guided_decoding_backend" not in sent.get("extra_body", {})
    assert sent["response_format"]["type"] == "structural_tag"
    assert sent["chat_template_kwargs"] == {"enable_thinking": False}
    # Messages MUST instruct the LLM to emit <json_output>...</json_output> so xgrammar triggers!
    combined_msgs = "\n".join(m["content"] for m in sent["messages"])
    assert "<json_output>" in combined_msgs and "</json_output>" in combined_msgs

    # Invalid BaseModel payload MUST raise StructuredOutputValidationError
    with pytest.raises(StructuredOutputValidationError):
        ModelArchitectureAdapter.extract_and_validate_structural_json(
            '<json_output>{"summary": "missing_required_fields"}</json_output>',
            _ScoutSynthesisContract,
        )

    # Local MLX directory with config.json + .npz (zero .safetensors) MUST fail closed!
    no_st_dir = tmp_path / "no_safetensors_dir"
    no_st_dir.mkdir()
    (no_st_dir / "config.json").write_text(json.dumps({"model_type": "gemma4"}), encoding="utf-8")
    (no_st_dir / "weights.npz").write_bytes(b"npz")
    with pytest.raises((ValueError, LoRACompatibilityError)):
        exec_vllm._validate_local_mlx_directory(str(no_st_dir))

    # Multi-shard LoRA directory where shard 2 has unfrozen MoE router tensor MUST fail closed!
    sharded_dir = tmp_path / "sharded_lora"
    sharded_dir.mkdir()
    (sharded_dir / "adapter_config.json").write_text(
        json.dumps({"r": 16, "lora_alpha": 32, "use_rslora": True, "use_dora": False}),
        encoding="utf-8",
    )
    (sharded_dir / "adapter_model-00001-of-00002.safetensors").write_bytes(
        _build_safetensors_bytes()
    )
    (sharded_dir / "adapter_model-00002-of-00002.safetensors").write_bytes(
        _build_safetensors_bytes(("base_model.model.layers.3.router.lora_A.weight",))
    )
    with pytest.raises(LoRACompatibilityError, match="frozen MoE router"):
        exec_vllm.run_vllm_lora_dry_run_probe(
            adapter_id="sharded_bad",
            adapter_dir=sharded_dir,
            probe_live_vllm=False,
        )


def test_scout_workflow_wires_open_weights_executor_and_offline_abstention() -> None:
    """ScoutWorkflow.execute & .run dispatch through injected OpenWeightsVLLMExecutor."""
    from context_scout.agents.scout_workflow import ScoutWorkflow
    from context_scout.schemas.contracts import InsufficientContextDiagnostic

    p = PrincipalContext(
        user_id="wf_eng",
        role=RoleType.ENGINEER,
        allowed_scopes=["public", "public_eng", "domain:code_architecture_scout"],
    )
    mem_repo = InMemoryRLSRepository()

    # 1. Offline OpenWeightsVLLMExecutor returns InsufficientContextDiagnostic (paid_tokens=0)
    def _offline(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("vLLM offline", request=req)

    off_exec = OpenWeightsVLLMExecutor(
        model_id="Qwen/Qwen3-Coder-30B-A3B",
        http_client=httpx.Client(transport=httpx.MockTransport(_offline)),
    )
    wf_off = ScoutWorkflow(repo=mem_repo, llm_backend=off_exec)
    diag = wf_off.execute("Explain PrincipalContext", p, domain_id="code_architecture_scout")
    assert isinstance(diag, InsufficientContextDiagnostic)
    assert diag.is_abstention is True
    assert diag.paid_tokens_spent == 0

    # 2. Online OpenWeightsVLLMExecutor is actually invoked by ScoutWorkflow.execute!
    vllm_calls: list[dict[str, Any]] = []

    def _online(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={"data": [{"id": "Qwen/Qwen3-Coder-30B-A3B", "max_model_len": 32768}]},
            )
        if req.url.path == "/v1/chat/completions":
            body = json.loads(req.content.decode("utf-8"))
            vllm_calls.append(body)
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "[]"}}], "usage": {"total_tokens": 8}},
            )
        return httpx.Response(404)

    on_exec = OpenWeightsVLLMExecutor(
        model_id="Qwen/Qwen3-Coder-30B-A3B",
        http_client=httpx.Client(transport=httpx.MockTransport(_online)),
    )
    wf_on = ScoutWorkflow(repo=mem_repo, llm_backend=on_exec)
    _ = wf_on.execute("Explain PrincipalContext", p, domain_id="code_architecture_scout")
    assert len(vllm_calls) >= 1
    assert vllm_calls[0]["cache_salt"] == compute_prefix_cache_salt(p)


@pytest.mark.postgres
def test_revoked_artifact_upsert_blocked_and_asi05_keyless_hardening(
    tmp_path: Path,
) -> None:
    """Blocks re-registering revoked artifacts in PG17/InMemory, -mpytest/py.test, & SA creds."""
    settings = ScoutSettings()
    rls_mgr = RLSConnectionManager(settings.db_app_url)
    pg_repo = PostgresRLSRepository()
    mem_repo = InMemoryRLSRepository()
    tid = uuid.uuid4()
    dom = f"eval_rev_{uuid.uuid4().hex[:8]}"
    p = PrincipalContext(
        user_id="sec_ops",
        role=RoleType.ENGINEER,
        tenant_id=tid,
        allowed_scopes=["tenant_common"],
    )

    # Non-hex 64-char SHA-256 rejected
    bad_hex_rec = ModelArtifactRecord(
        tenant_id=tid,
        domain_id=dom,
        artifact_id="lora_bad_hex",
        base_model_id="google/gemma-4-E4B-it",
        base_model_revision_sha256="z" * 64,
        storage_uri=f"gs://b/tenants/{tid}/domains/{dom}/w.safetensors",
        weights_object_key=f"tenants/{tid}/domains/{dom}/w.safetensors",
        config_object_key=f"tenants/{tid}/domains/{dom}/c.json",
        weights_sha256="a" * 64,
        training_scope_ceiling=["tenant_common"],
        allowed_scopes=["tenant_common"],
    )
    with pytest.raises(ValueError, match="hex"):
        mem_repo.register_model_artifact(p, bad_hex_rec)

    valid_rec = ModelArtifactRecord(
        tenant_id=tid,
        domain_id=dom,
        artifact_id="lora_rev_lock",
        base_model_id="google/gemma-4-E4B-it",
        base_model_revision_sha256="a" * 64,
        storage_uri=f"gs://b/tenants/{tid}/domains/{dom}/w.safetensors",
        weights_object_key=f"tenants/{tid}/domains/{dom}/w.safetensors",
        config_object_key=f"tenants/{tid}/domains/{dom}/c.json",
        weights_sha256="b" * 64,
        training_scope_ceiling=["tenant_common"],
        allowed_scopes=["tenant_common"],
        is_active=True,
    )
    mem_repo.register_model_artifact(p, valid_rec)
    mem_repo.retire_model_artifact(p, dom, "lora_rev_lock", revoke=True)
    with pytest.raises(PermissionError, match="Revoked"):
        mem_repo.register_model_artifact(p, valid_rec)

    try:
        with rls_mgr.scoped_transaction(p, enable_hnsw_iterative_scan=False) as conn:
            pg_repo.register_model_artifact(conn, valid_rec)
            pg_repo.retire_model_artifact(conn, dom, "lora_rev_lock", revoke=True)
            with pytest.raises(PermissionError, match="Revoked"):
                pg_repo.register_model_artifact(conn, valid_rec)
    finally:
        with psycopg.connect(settings.db_admin_url) as admin_conn:
            admin_conn.execute("DELETE FROM model_artifacts WHERE tenant_id = %s;", (tid,))
            admin_conn.commit()

    # ASI05: -mpytest, py.test, and inline python -c '...pytest...' MUST raise PermissionError
    for sneaky_cmd in (
        ["python3", "-mpytest", "-q"],
        ["/usr/bin/py.test"],
        ["python3", "-c", "import pytest; pytest.main()"],
        ["sh", "-c", "pytest tests/"],
    ):
        with pytest.raises(PermissionError, match="ASI05"):
            GitWorktreeSandbox._assert_no_pytest_command(sneaky_cmd)

    # Keyless ADC: static service_account Credentials object rejected even without env var
    class _StaticServiceAccountCreds:
        signer_email = "sa@proj.iam.gserviceaccount.com"
        _signer = object()

    sa_obj_exec = GeminiImplementationExecutor(
        model_id="gemini-3.8-flash",
        credentials=_StaticServiceAccountCreds(),
        use_keyless_adc=True,
        allow_external_cloud_egress=True,
    )
    with pytest.raises(PermissionError, match="service_account"):
        sa_obj_exec.resolve_keyless_adc_credentials()

    # PermissionError inside _generate_gemini_patch MUST NOT be swallowed into deterministic patch
    with pytest.raises(PermissionError):
        sa_obj_exec._generate_gemini_patch(_make_pack(), tmp_path)


def test_cosign_x509_bundle_and_cli_bundle_flag_verification(tmp_path: Path) -> None:
    """Verifies Sigstore v0.3 X.509 CA cert chain bundles and external cosign --bundle CLI path."""
    import base64
    from datetime import UTC, datetime, timedelta

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes
    from cryptography.x509.oid import NameOID

    from context_scout.serving.adapter_sync_sidecar import CosignVerificationError

    # 1. Build CA + leaf X.509 cert chain and sign blob into Sigstore v0.3 bundle
    ca_priv = ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Scout Fulcio Root CA")])
    now = datetime.now(UTC)
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_priv.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(hours=1))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(ca_priv, hashes.SHA256())
    )
    ca_pem = ca_cert.public_bytes(serialization.Encoding.PEM).decode("utf-8")

    leaf_priv = ec.generate_private_key(ec.SECP256R1())
    leaf_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "ci-signer@scout.internal")])
    leaf_cert = (
        x509.CertificateBuilder()
        .subject_name(leaf_name)
        .issuer_name(ca_name)
        .public_key(leaf_priv.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(hours=1))
        .not_valid_after(now + timedelta(hours=1))
        .sign(ca_priv, hashes.SHA256())
    )
    leaf_der_b64 = base64.b64encode(leaf_cert.public_bytes(serialization.Encoding.DER)).decode(
        "ascii"
    )

    payload = _build_safetensors_bytes()
    raw_bundle = json.loads(sign_blob_for_cosign(leaf_priv, payload, as_sigstore_bundle=True))
    raw_bundle["verificationMaterial"] = {"certificate": {"rawBytes": leaf_der_b64}}
    x509_bundle_str = json.dumps(raw_bundle)

    ca_verifier = CosignBlobVerifier(trusted_ca_certs_pem=[ca_pem])
    ca_verifier.verify(payload, x509_bundle_str)

    # Untrusted CA rejects the leaf certificate
    rogue_ca_priv = ec.generate_private_key(ec.SECP256R1())
    rogue_ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(rogue_ca_priv.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(hours=1))
        .not_valid_after(now + timedelta(days=1))
        .sign(rogue_ca_priv, hashes.SHA256())
    )
    rogue_verifier = CosignBlobVerifier(
        trusted_ca_certs_pem=[
            rogue_ca_cert.public_bytes(serialization.Encoding.PEM).decode("utf-8")
        ]
    )
    with pytest.raises(CosignVerificationError, match="trusted CA"):
        rogue_verifier.verify(payload, x509_bundle_str)

    # 2. External cosign CLI subprocess branch uses --bundle for JSON bundles & --signature for .sig
    cli_log = tmp_path / "cosign_argv.json"
    fake_cosign = tmp_path / "fake_cosign.py"
    fake_cosign.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys\n"
        f"pathlib.Path({str(cli_log)!r}).write_text(json.dumps(sys.argv[1:]))\n"
        "sys.exit(0)\n",
        encoding="utf-8",
    )
    fake_cosign.chmod(0o755)
    pub_key_file = tmp_path / "cosign.pub"
    pub_key_file.write_text(ca_pem, encoding="utf-8")

    cli_verifier = CosignBlobVerifier(
        cosign_binary=str(fake_cosign),
        cosign_public_key_paths={"default": pub_key_file},
    )
    cli_verifier.verify(payload, x509_bundle_str, key_id="default")
    argv_bundle = json.loads(cli_log.read_text(encoding="utf-8"))
    assert "--bundle" in argv_bundle and "--signature" not in argv_bundle

    cli_verifier.verify(payload, sign_blob_for_cosign(leaf_priv, payload), key_id="default")
    argv_sig = json.loads(cli_log.read_text(encoding="utf-8"))
    assert "--signature" in argv_sig and "--bundle" not in argv_sig
