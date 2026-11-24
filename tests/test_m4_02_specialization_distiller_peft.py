"""M4-02 verification suite: SpecializationCompiler, TeacherDistiller v2, PEFT rsLoRA, & Migration.

Covers:
1. All 5 ``specialization.yaml`` archetypes (``pointer_localization``, ``extractive_synthesis``,
   ``compliance_security_audit``, ``change_impact_mr_review``, ``custom_json_contract``) and
   fail-closed policy / path / plugin guards.
2. ``TeacherDistiller`` v2: ceiling-bounded live PG17 ``FORCE RLS`` queries,
   ``PreIndexScrubber``, 10 synthetic canaries strictly inside ``train.jsonl`` user context,
   Best-of-8 multi-turn self-play + fail-closed offline error, Git/MR/Jira diff mining,
   DPO ``(chosen, rejected)`` collection, repo/time-disjoint ``80/10/10`` splits, and 15% replay.
3. ``PEFTRsLoRATrainer`` across ``Gemma 4 MoE``, ``Qwen3-Coder MoE``, and ``FalconMamba SSM``:
   1-step ``vLLM`` dry-run probe, ``assistant_only_loss=True``, ``use_rslora=True``,
   ``use_dora=False``, ``LoRA+`` 8x LR, ``NEFTune`` training-only noise, frozen routers & SSM
   states, Task-F1 checkpointing, DPO alignment, ``CachedMNRLBiEncoderTrainer``, and
   ``CanaryMemorizationGate`` (``0/10`` pass + ``1/10`` fail-closed block).
4. ``cosign`` Sigstore ``.bundle`` + ``GCSObjectStore`` (``gs://``) & ``S3CompatibleObjectStore``
   (``s3://``) upload + live PG17 ``model_artifacts`` + 2-pod ``AdapterSyncSidecar`` parity.
5. ``context-scout migrate-base-model`` 4-stage replay DAG & AST security hygiene.
"""

from __future__ import annotations

import ast
import hashlib
import json
import threading
import uuid
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from urllib.parse import parse_qs, unquote, urlparse

import httpx
import psycopg
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from safetensors.torch import load_file
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.modules import Pooling, Transformer
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

from context_scout.agents.llm_backend import GPUPriorityClass
from context_scout.agents.open_weights_vllm_executor import (
    LoRACompatibilityError,
    ModelArchitectureAdapter,
    OpenWeightsBackendUnavailableError,
    OpenWeightsVLLMExecutor,
)
from context_scout.cli import cmd_migrate_base_model
from context_scout.cli import main as cli_main
from context_scout.config import ScoutSettings
from context_scout.db.repositories import (
    ASTSymbolRecord,
    CodeFileRecord,
    DocChunkRecord,
    InMemoryRLSRepository,
    JiraIssueRecord,
    ModelArtifactRecord,
    OverlayFileDeltaRecord,
    PostgresRLSRepository,
    RefSnapshotRecord,
    RepoCommitRecord,
    RepositoryRecord,
    ScoutSessionRecord,
)
from context_scout.db.session import get_ingestion_pool
from context_scout.schemas.contracts import (
    CodeSlicePointer,
    DocCitationSpan,
    DomainKnowledgeBrief,
    ModificationIntent,
    PrincipalContext,
    RoleType,
)
from context_scout.serving.adapter_sync_sidecar import (
    AdapterSyncSidecar,
    CosignBlobVerifier,
)
from context_scout.storage.object_store import (
    GCSObjectStore,
    S3CompatibleObjectStore,
)
from context_scout.tuning.specialization_compiler import (
    ChangeImpactMRReviewReport,
    ComplianceAuditFinding,
    ComplianceSecurityAuditReport,
    SpecializationArchetype,
    SpecializationCompilationError,
    SpecializationCompiler,
)
from context_scout.tuning.teacher_distiller import (
    BestOfNSelfPlayEngine,
    SelfPlayTaskSpec,
    TeacherDistiller,
)
from context_scout.tuning.train_peft_lora import (
    CachedMNRLBiEncoderTrainer,
    CanaryMemorizationError,
    PEFTRsLoRATrainer,
    PEFTTrainingConfig,
    build_loraplus_adamw_optimizer,
    inject_rslora_adapters,
    is_frozen_router_or_ssm_path,
    resolve_safe_lora_target_modules,
    tokenize_chat_assistant_only,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _bootstrap_pg_tenant(
    *,
    tenant_id: uuid.UUID,
    domain_id: str,
    scopes: list[str],
    base_model_id: str = "google/gemma-4-26B-A4B-it",
) -> None:
    """Seeds an isolated tenant/domain row inside PG17 for ephemeral M4-02 tests."""
    settings = ScoutSettings()
    with psycopg.connect(settings.db_admin_url) as conn:
        conn.execute(
            """
            INSERT INTO domain_packs (
                tenant_id, domain_id, display_name, lora_adapter_id,
                primary_truth_source, config_json
            )
            VALUES (%s, %s, %s, %s, 'code', %s::jsonb)
            ON CONFLICT (tenant_id, domain_id) DO NOTHING;
            """,
            (
                tenant_id,
                domain_id,
                f"M4-02 Test Domain {domain_id}",
                f"lora_{domain_id}",
                json.dumps(
                    {
                        "domain_id": domain_id,
                        "base_model_id": base_model_id,
                        "default_scopes": scopes,
                    }
                ),
            ),
        )
        conn.commit()


def _cleanup_pg_tenant(*, tenant_id: uuid.UUID) -> None:
    """Deletes all rows belonging exclusively to ``tenant_id``."""
    settings = ScoutSettings()
    tables = (
        "audit_events",
        "scim_groups",
        "scim_users",
        "model_artifacts",
        "scout_sessions",
        "overlay_file_deltas",
        "ref_snapshots",
        "execution_telemetry",
        "jira_issues",
        "repo_commits",
        "ast_edges",
        "ast_symbols",
        "code_files",
        "doc_chunks",
        "repositories",
        "embedding_spaces",
        "domain_sync_manifests",
        "domain_packs",
    )
    with psycopg.connect(settings.db_admin_url) as conn:
        for table in tables:
            conn.execute(f"DELETE FROM {table} WHERE tenant_id = %s;", (tenant_id,))
        conn.commit()


def _start_local_gcs_emulator() -> tuple[HTTPServer, str, dict[str, tuple[bytes, str]]]:
    """Starts an in-process GCS JSON/upload emulator on 127.0.0.1 ephemeral port."""
    objects: dict[str, tuple[bytes, str]] = {}

    class _GCSHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            del format, args

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            qs = parse_qs(parsed.query)
            obj_name = qs.get("name", [""])[0]
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            content_type = self.headers.get("Content-Type", "")
            body = raw
            mime = "application/octet-stream"
            if "multipart/related" in content_type and "boundary=" in content_type:
                boundary = content_type.split("boundary=", 1)[1].strip().strip('"')
                parts = raw.split(f"--{boundary}".encode("ascii"))
                if len(parts) >= 3:
                    data_part = parts[2]
                    if b"\r\n\r\n" in data_part:
                        body = data_part.split(b"\r\n\r\n", 1)[1].rstrip(b"\r\n")
            objects[obj_name] = (body, mime)
            resp = json.dumps({"name": obj_name, "size": str(len(body))}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if "/o/" in parsed.path:
                enc_key = parsed.path.split("/o/", 1)[1]
                key = unquote(enc_key)
                if key not in objects:
                    self.send_response(404)
                    self.end_headers()
                    return
                body, mime = objects[key]
                qs = parse_qs(parsed.query)
                if qs.get("alt") == ["media"]:
                    self.send_response(200)
                    self.send_header("Content-Type", mime)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    meta = json.dumps({"name": key, "size": str(len(body))}).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(meta)
                return

            qs = parse_qs(parsed.query)
            prefix = qs.get("prefix", [""])[0]
            items = [{"name": k} for k in sorted(objects) if k.startswith(prefix)]
            payload = json.dumps({"items": items}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(payload)

    server = HTTPServer(("127.0.0.1", 0), _GCSHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_port}", objects


def _make_s3_store_backed_by_dict(
    bucket: str,
    backing: dict[str, bytes],
) -> S3CompatibleObjectStore:
    client = MagicMock()

    def _put_object(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["Bucket"] == bucket
        backing[kwargs["Key"]] = bytes(kwargs["Body"])
        return {"ETag": '"ok"'}

    def _get_object(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["Bucket"] == bucket
        key = kwargs["Key"]
        if key not in backing:
            raise KeyError(key)
        import io

        return {"Body": io.BytesIO(backing[key]), "ContentType": "application/octet-stream"}

    def _head_object(**kwargs: Any) -> dict[str, Any]:
        key = kwargs["Key"]
        if key not in backing:
            from botocore.exceptions import ClientError

            raise ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject")
        return {"ContentLength": len(backing[key]), "ContentType": "application/octet-stream"}

    client.put_object.side_effect = _put_object
    client.get_object.side_effect = _get_object
    client.head_object.side_effect = _head_object
    return S3CompatibleObjectStore(bucket=bucket, client=client)


def _make_2pod_vllm_mock_client(
    base_model_id: str,
) -> tuple[httpx.Client, dict[str, set[str]]]:
    pod_loaded: dict[str, set[str]] = {
        "http://vllm-pod-0:8001": set(),
        "http://vllm-pod-1:8001": set(),
    }

    def _handler(request: httpx.Request) -> httpx.Response:
        origin = f"{request.url.scheme}://{request.url.host}:{request.url.port}"
        path = request.url.path
        if request.method == "GET" and path == "/v1/models":
            items = [{"id": base_model_id}] + [{"id": name} for name in sorted(pod_loaded[origin])]
            return httpx.Response(200, json={"data": items})
        if request.method == "POST" and path == "/v1/load_lora_adapter":
            payload = json.loads(request.content.decode("utf-8"))
            lora_name = payload["lora_name"]
            lora_path = Path(payload["lora_path"])
            if not (lora_path / "adapter_model.safetensors").is_file():
                return httpx.Response(400, text="missing adapter_model.safetensors")
            pod_loaded[origin].add(lora_name)
            return httpx.Response(200, json={"status": "loaded", "lora_name": lora_name})
        if request.method == "POST" and path == "/v1/unload_lora_adapter":
            payload = json.loads(request.content.decode("utf-8"))
            pod_loaded[origin].discard(payload["lora_name"])
            return httpx.Response(200, json={"status": "unloaded"})
        return httpx.Response(404, text="not found")

    client = httpx.Client(transport=httpx.MockTransport(_handler))
    return client, pod_loaded


class TestSpecializationCompilerAllArchetypes:
    """Verifies all 5 archetypes, glossary rendering, and fail-closed safety checks."""

    def test_compile_and_validate_all_5_archetypes(self, tmp_path: Path) -> None:
        domains_dir = tmp_path / "domains"
        dom_id = "erp_rfc_migration"
        dom_dir = domains_dir / dom_id
        dom_dir.mkdir(parents=True)

        schema_path = dom_dir / "schema.json"
        schema_obj = {
            "type": "object",
            "required": ["rfc_name", "grpc_service", "field_mappings"],
            "additionalProperties": False,
            "properties": {
                "rfc_name": {"type": "string", "minLength": 3, "pattern": r"^[A-Z0-9_]+$"},
                "grpc_service": {"type": "string", "minLength": 3},
                "field_mappings": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "string"},
                },
            },
        }
        schema_path.write_text(json.dumps(schema_obj, indent=2), encoding="utf-8")

        glossary_path = dom_dir / "glossary.yaml"
        glossary_path.write_text(
            "domain_id: erp_rfc_migration\n"
            "terms:\n"
            "  BAPI_ORDER_GET:\n"
            "    definition: Sales order RFC lookup function module\n"
            "    aliases: [OrderLookupRFC, SD_RFC]\n",
            encoding="utf-8",
        )

        def _check_rfc_prefix(payload: Any) -> tuple[bool, str]:
            if not str(payload.get("rfc_name", "")).startswith("BAPI_"):
                return False, "rfc_name must start with 'BAPI_'"
            return True, ""

        compiler = SpecializationCompiler(
            workspace_root=tmp_path,
            domains_dir=domains_dir,
            validator_registry={"erp_validator:check_rfc": _check_rfc_prefix},
        )

        # Dummy grounded corpus records for pointer & citation verification
        cf = CodeFileRecord(
            file_id=uuid.uuid4(),
            domain_id=dom_id,
            repo_name="repo_core",
            file_path="src/auth/rls.py",
            git_commit_sha="a" * 40,
            git_blob_sha="b" * 40,
            language="python",
            allowed_scopes=["tenant_common"],
            raw_content="\n".join(f"line_{i} = {i}" for i in range(1, 21)),
            line_count=20,
        )
        chk = DocChunkRecord(
            chunk_id=uuid.uuid4(),
            domain_id=dom_id,
            doc_id="adr:0042",
            doc_title="ADR-0042 Force RLS Policy",
            source_uri="urn:adr:0042",
            market_scope="ENTERPRISE_DOC",
            section_heading="Isolation",
            contextual_header="ADR-0042",
            verbatim_content=(
                "Every PostgreSQL 17 table enforces ENABLE and FORCE "
                "ROW LEVEL SECURITY under NOBYPASSRLS."
            ),
            published_date=None,
            authority_weight=1.0,
            metadata_json={},
            allowed_scopes=["tenant_common"],
            embedding=[0.1] * 16,
        )
        ptr = CodeSlicePointer(
            repo_name="repo_core",
            file_path="src/auth/rls.py",
            start_line=2,
            end_line=10,
            qualified_symbol="rls.enforce",
            selection_reason="RLS guard",
            modification_intent=ModificationIntent.MODIFY_TARGET,
        )
        cit = DocCitationSpan(
            chunk_id=str(chk.chunk_id),
            doc_id=chk.doc_id,
            doc_title=chk.doc_title,
            source_uri=chk.source_uri,
            section_heading=chk.section_heading,
            verbatim_quote=(
                "Every PostgreSQL 17 table enforces ENABLE and FORCE ROW LEVEL SECURITY"
            ),
        )

        # 1. pointer_localization
        c_ptr = compiler.compile(
            {
                "domain_id": dom_id,
                "archetype": "pointer_localization",
                "base_model_id": "Qwen/Qwen3-Coder-30B-A3B",
                "training_scope_ceiling": ["tenant_common"],
                "glossary_path": f"domains/{dom_id}/glossary.yaml",
            }
        )
        assert c_ptr.archetype == SpecializationArchetype.POINTER_LOCALIZATION
        assert "Sales order RFC" in c_ptr.system_prompt
        assert c_ptr.validate_output(json.dumps([ptr.model_dump()]), code_files=[cf]).passed is True
        bad_ptr = ptr.model_copy(update={"end_line": 99})
        bad_res = c_ptr.validate_output(json.dumps([bad_ptr.model_dump()]), code_files=[cf])
        assert bad_res.passed is False
        assert bad_res.bounce_back_feedback is not None

        # Vendor-neutral chat formatting check (zero hardcoded Gemma tags)
        rendered = c_ptr.render_prompt(c_ptr.format_messages("Locate BAPI_ORDER_GET"))
        assert "<start_of_turn>" not in rendered

        # 2. extractive_synthesis
        c_syn = compiler.compile(
            {
                "domain_id": dom_id,
                "archetype": "extractive_synthesis",
                "base_model_id": "google/gemma-4-26B-A4B-it",
                "training_scope_ceiling": ["tenant_common"],
            }
        )
        brief = DomainKnowledgeBrief(
            brief_title="ADR-0042 Isolation",
            domain_id=dom_id,
            executive_summary="PG17 enforces FORCE RLS under NOBYPASSRLS.",
            core_findings_and_rules=["FORCE ROW LEVEL SECURITY"],
            domain_constraints_and_caveats=["Applies to all tables"],
            known_risks_and_failure_modes=["Bypass role misconfiguration"],
            verified_citations=[cit],
        )
        assert c_syn.validate_output(brief.model_dump(), doc_chunks=[chk]).passed is True
        bad_brief = brief.model_copy(
            update={"verified_citations": [cit.model_copy(update={"chunk_id": "missing-id"})]}
        )
        assert c_syn.validate_output(bad_brief.model_dump(), doc_chunks=[chk]).passed is False

        # 3. compliance_security_audit
        c_comp = compiler.compile(
            {
                "domain_id": dom_id,
                "archetype": "compliance_security_audit",
                "base_model_id": "mistralai/Devstral-Small-2505",
                "training_scope_ceiling": ["tenant_common"],
            }
        )
        comp_rep = ComplianceSecurityAuditReport(
            domain_id=dom_id,
            audit_verdict="compliant",
            severity="low",
            executive_summary="All RLS checks pass.",
            findings=[
                ComplianceAuditFinding(
                    control_id="CTL-RLS-01",
                    rule_or_policy="FORCE RLS",
                    status="pass",
                    remediation="None required",
                    evidence_pointer=ptr,
                    evidence_citation=cit,
                )
            ],
            verified_citations=[cit],
            code_pointers=[ptr],
        )
        assert (
            c_comp.validate_output(comp_rep.model_dump(), code_files=[cf], doc_chunks=[chk]).passed
            is True
        )

        # 4. change_impact_mr_review
        c_mr = compiler.compile(
            {
                "domain_id": dom_id,
                "archetype": "change_impact_mr_review",
                "base_model_id": "Qwen/Qwen3-Coder-30B-A3B",
                "training_scope_ceiling": ["tenant_common"],
            }
        )
        mr_rep = ChangeImpactMRReviewReport(
            domain_id=dom_id,
            mr_or_commit_ref="mr-101",
            risk_level="medium",
            review_summary="Touches RLS enforcement module.",
            impacted_slices=[ptr],
            linked_ticket_keys=["ENG-101"],
            supporting_citations=[cit],
        )
        assert (
            c_mr.validate_output(mr_rep.model_dump(), code_files=[cf], doc_chunks=[chk]).passed
            is True
        )

        # 5. custom_json_contract
        spec_file = dom_dir / "specialization.yaml"
        spec_file.write_text(
            f"domain_id: {dom_id}\n"
            "display_name: SAP BAPI/RFC to gRPC Migration Specialist\n"
            "archetype: custom_json_contract\n"
            f"output_schema_path: domains/{dom_id}/schema.json\n"
            "validator_plugin: erp_validator:check_rfc\n"
            f"glossary_path: domains/{dom_id}/glossary.yaml\n"
            "training_scope_ceiling: [tenant_common, enclave:erp_eng]\n"
            "base_model_id: google/gemma-4-26B-A4B-it\n"
            "lora_policy:\n"
            "  enabled: true\n"
            "  rank: 16\n"
            "  lora_alpha: 32\n"
            "  use_rslora: true\n"
            "  use_dora: false\n"
            "  loraplus_lr_ratio: 8.0\n"
            "  neftune_noise_alpha: 5.0\n"
            "  general_replay_ratio: 0.15\n"
            "  assistant_only_loss: true\n",
            encoding="utf-8",
        )
        c_custom = compiler.compile(spec_file)
        assert c_custom.archetype == SpecializationArchetype.CUSTOM_JSON_CONTRACT
        ok_json = {
            "rfc_name": "BAPI_ORDER_GET",
            "grpc_service": "OrderLookupService",
            "field_mappings": ["ORDER_ID -> order_id"],
        }
        assert c_custom.validate_output(json.dumps(ok_json)).passed is True
        # Fails validator_plugin check
        bad_prefix_json = {**ok_json, "rfc_name": "Z_CUSTOM_RFC"}
        plugin_fail = c_custom.validate_output(bad_prefix_json)
        assert plugin_fail.passed is False
        assert "BAPI_" in " ".join(plugin_fail.violations)

    def test_fail_closed_lora_policy_traversal_and_unsafe_plugin(self, tmp_path: Path) -> None:
        compiler = SpecializationCompiler(workspace_root=tmp_path, domains_dir=tmp_path / "domains")

        lora_base: dict[str, Any] = {
            "enabled": True,
            "rank": 16,
            "lora_alpha": 32,
            "use_rslora": True,
            "use_dora": False,
            "assistant_only_loss": True,
            "general_replay_ratio": 0.15,
        }
        base_spec: dict[str, Any] = {
            "domain_id": "sec_test",
            "archetype": "pointer_localization",
            "base_model_id": "google/gemma-4-26B-A4B-it",
            "training_scope_ceiling": ["public_eng"],
            "lora_policy": lora_base,
        }
        bad_patches: tuple[dict[str, Any], ...] = (
            {"training_scope_ceiling": []},
            {"lora_policy": {**lora_base, "use_dora": True}},
            {"lora_policy": {**lora_base, "use_rslora": False}},
            {"lora_policy": {**lora_base, "assistant_only_loss": False}},
            {"lora_policy": {**lora_base, "rank": 64}},
            {"lora_policy": {**lora_base, "general_replay_ratio": 0.02}},
        )
        for bad_patch in bad_patches:
            bad = {**base_spec, **bad_patch}
            with pytest.raises(SpecializationCompilationError):
                compiler.compile(bad)

        # Path traversal in output_schema_path
        with pytest.raises(SpecializationCompilationError, match="Unsafe path traversal"):
            compiler.compile(
                {
                    **base_spec,
                    "archetype": "custom_json_contract",
                    "output_schema_path": "../../etc/passwd",
                }
            )

        # Untrusted validator_plugin outside context_scout.* / explicit registry
        with pytest.raises(SpecializationCompilationError, match="Untrusted validator_plugin"):
            compiler.compile(
                {
                    **base_spec,
                    "validator_plugin": "os:system",
                }
            )


class TestTeacherDistillerV2LivePG17RLS:
    """Tests ACL scope-ceiling bounds and 3-source distiller v2 against live PG17 FORCE RLS."""

    def test_distiller_ceiling_bounds_scrubber_canaries_bestof8_splits_and_replay(
        self, tmp_path: Path
    ) -> None:
        tenant_id = uuid.uuid4()
        domain_id = f"eval_m402_{uuid.uuid4().hex[:8]}"
        ceiling_scopes = ["tenant_common", "enclave:erp_eng"]
        _bootstrap_pg_tenant(tenant_id=tenant_id, domain_id=domain_id, scopes=ceiling_scopes)

        domains_dir = tmp_path / "domains"
        dom_dir = domains_dir / domain_id
        dom_dir.mkdir(parents=True)
        spec_path = dom_dir / "specialization.yaml"
        spec_path.write_text(
            f"domain_id: {domain_id}\n"
            "archetype: pointer_localization\n"
            "base_model_id: google/gemma-4-26B-A4B-it\n"
            "training_scope_ceiling: [tenant_common, enclave:erp_eng]\n"
            "lora_policy:\n"
            "  enabled: true\n"
            "  rank: 16\n"
            "  lora_alpha: 32\n"
            "  use_rslora: true\n"
            "  use_dora: false\n"
            "  assistant_only_loss: true\n"
            "  general_replay_ratio: 0.15\n",
            encoding="utf-8",
        )

        settings = ScoutSettings()
        mgr = get_ingestion_pool(settings)
        pg_repo = PostgresRLSRepository()
        allowed_principal = PrincipalContext(
            user_id="seeder_allowed",
            role=RoleType.ENGINEER,
            allowed_scopes=ceiling_scopes,
            tenant_id=tenant_id,
        )
        restricted_principal = PrincipalContext(
            user_id="seeder_restricted",
            role=RoleType.ENGINEER,
            allowed_scopes=["exec_board_only", "docs_restricted", "INTERNAL"],
            tenant_id=tenant_id,
        )

        base_ts = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
        try:
            # Seed 4 chronological repos inside ceiling + 1 restricted repo outside
            repo_names = ("repo_alpha", "repo_beta", "repo_gamma", "repo_delta")
            with mgr.scoped_transaction(
                allowed_principal, enable_hnsw_iterative_scan=False
            ) as conn:
                for r_idx, r_name in enumerate(repo_names):
                    repo_id = uuid.uuid4()
                    pg_repo.insert_repository(
                        conn,
                        RepositoryRecord(
                            repo_id=repo_id,
                            domain_id=domain_id,
                            repo_name=r_name,
                            source_type="gitlab",
                            remote_url=f"https://git.local/{r_name}",
                            default_branch="main",
                            head_commit_sha=f"{r_idx + 1}" * 40,
                            allowed_scopes=["enclave:erp_eng"],
                            tenant_id=tenant_id,
                        ),
                    )
                    for f_idx in range(3):
                        f_id = uuid.uuid4()
                        f_path = f"pkg_{r_name}/mod_{f_idx}.py"
                        secret_marker = (
                            f"# token='{'ghp_'}SecretLeakShouldBeScrubbed123456'\n"
                            if r_idx == 0 and f_idx == 0
                            else ""
                        )
                        fn_lines = "\n".join(
                            f"def fn_{r_idx}_{f_idx}_{line}() -> int:\n    return {line}"
                            for line in range(1, 10)
                        )
                        code_txt = f"{secret_marker}{fn_lines}"
                        pg_repo.insert_code_file(
                            conn,
                            CodeFileRecord(
                                file_id=f_id,
                                domain_id=domain_id,
                                repo_name=r_name,
                                file_path=f_path,
                                git_commit_sha=f"{r_idx + 1}" * 40,
                                git_blob_sha=hashlib.sha1(f"{r_name}:{f_idx}".encode()).hexdigest(),
                                language="python",
                                allowed_scopes=["enclave:erp_eng"],
                                raw_content=code_txt,
                                line_count=len(code_txt.splitlines()),
                                repo_id=repo_id,
                                tenant_id=tenant_id,
                            ),
                        )
                        pg_repo.insert_ast_symbol(
                            conn,
                            ASTSymbolRecord(
                                symbol_id=uuid.uuid4(),
                                file_id=f_id,
                                domain_id=domain_id,
                                qualified_name=f"{r_name}.Service{f_idx}.run",
                                symbol_kind="method",
                                start_line=1,
                                end_line=min(6, len(code_txt.splitlines())),
                                signature_skeleton=f"def run_{f_idx}() -> int:",
                                contextual_summary=f"Service handler {f_idx} in {r_name}",
                                allowed_scopes=["enclave:erp_eng"],
                                embedding=[0.12] * 768,
                                repo_id=repo_id,
                                tenant_id=tenant_id,
                            ),
                        )
                    pg_repo.insert_repo_commit(
                        conn,
                        RepoCommitRecord(
                            commit_id=uuid.uuid4(),
                            repo_id=repo_id,
                            commit_sha=f"{r_idx + 2}" * 40,
                            parent_shas=[],
                            author_name="Engineer",
                            author_email="eng@internal.local",
                            commit_message=f"ENG-{400 + r_idx}: refine mod_0.py in {r_name}",
                            committed_at=base_ts + timedelta(days=r_idx * 10),
                            tenant_id=tenant_id,
                        ),
                    )
                    snap = pg_repo.upsert_ref_snapshot(
                        conn,
                        RefSnapshotRecord(
                            snapshot_id=uuid.uuid4(),
                            domain_id=domain_id,
                            repo_id=repo_id,
                            repo_name=r_name,
                            ref_name=f"mr/ENG-{400 + r_idx}",
                            ref_kind="merge_request",
                            mr_iid=str(400 + r_idx),
                            base_commit_sha="0" * 40,
                            head_commit_sha=f"{r_idx + 3}" * 40,
                            state="open",
                            allowed_scopes=["enclave:erp_eng"],
                            updated_at=base_ts + timedelta(days=r_idx * 10 + 1),
                            tenant_id=tenant_id,
                        ),
                    )
                    pg_repo.upsert_overlay_file_delta(
                        conn,
                        OverlayFileDeltaRecord(
                            delta_id=uuid.uuid4(),
                            snapshot_id=snap.snapshot_id,
                            domain_id=domain_id,
                            repo_name=r_name,
                            file_path=f"pkg_{r_name}/mod_0.py",
                            change_type="modified",
                            overlay_blob_sha="c" * 40,
                            language="python",
                            raw_content="def updated() -> int:\n    return 1\n",
                            line_count=2,
                            allowed_scopes=["enclave:erp_eng"],
                            tenant_id=tenant_id,
                        ),
                    )
                    pg_repo.insert_jira_issue(
                        conn,
                        JiraIssueRecord(
                            issue_id=uuid.uuid4(),
                            domain_id=domain_id,
                            issue_key=f"ENG-{400 + r_idx}",
                            project_key="ENG",
                            summary=f"Ticket ENG-{400 + r_idx} update for {r_name}",
                            issue_type="Story",
                            status="In Progress",
                            source_url=f"https://jira.local/browse/ENG-{400 + r_idx}",
                            allowed_scopes=["enclave:erp_eng"],
                            updated_at=base_ts + timedelta(days=r_idx * 10),
                            linked_mrs=[f"mr/ENG-{400 + r_idx}"],
                            tenant_id=tenant_id,
                        ),
                    )
                pg_repo.upsert_scout_session(
                    conn,
                    ScoutSessionRecord(
                        session_id=f"sess_{domain_id}",
                        domain_id=domain_id,
                        principal_id="eng_user",
                        allowed_scopes=["enclave:erp_eng"],
                        turn_count=4,
                        recent_turns_json=[
                            {"role": "user", "content": "Localize pkg_repo_alpha/mod_0.py"},
                            {
                                "role": "assistant",
                                "content": json.dumps(
                                    [
                                        {
                                            "repo_name": "repo_alpha",
                                            "file_path": "pkg_repo_alpha/mod_0.py",
                                            "start_line": 1,
                                            "end_line": 9999,
                                            "qualified_symbol": "repo_alpha.Service0.run",
                                            "selection_reason": "Wrong bounds",
                                            "modification_intent": "modify_target",
                                        }
                                    ]
                                ),
                            },
                            {"role": "user", "content": "Line bounds exceed file."},
                            {
                                "role": "assistant",
                                "content": json.dumps(
                                    [
                                        {
                                            "repo_name": "repo_alpha",
                                            "file_path": "pkg_repo_alpha/mod_0.py",
                                            "start_line": 1,
                                            "end_line": 6,
                                            "qualified_symbol": "repo_alpha.Service0.run",
                                            "selection_reason": "Verified bounds",
                                            "modification_intent": "modify_target",
                                        }
                                    ]
                                ),
                            },
                        ],
                        tenant_id=tenant_id,
                    ),
                )

            # Seed STRICTLY RESTRICTED rows that MUST NEVER be visible under ceiling
            with mgr.scoped_transaction(
                restricted_principal, enable_hnsw_iterative_scan=False
            ) as conn:
                rf_id = uuid.uuid4()
                pg_repo.insert_code_file(
                    conn,
                    CodeFileRecord(
                        file_id=rf_id,
                        domain_id=domain_id,
                        repo_name="repo_secret_board",
                        file_path="secret_board/exec_plan.py",
                        git_commit_sha="f" * 40,
                        git_blob_sha="e" * 40,
                        language="python",
                        allowed_scopes=["exec_board_only", "docs_restricted", "INTERNAL"],
                        raw_content="SECRET_EXEC_PAYLOAD = 'TOP_SECRET_LEAK_TOKEN'\n",
                        line_count=1,
                        tenant_id=tenant_id,
                    ),
                )
                pg_repo.insert_ast_symbol(
                    conn,
                    ASTSymbolRecord(
                        symbol_id=uuid.uuid4(),
                        file_id=rf_id,
                        domain_id=domain_id,
                        qualified_name="secret_board.ExecLeakSymbol",
                        symbol_kind="class",
                        start_line=1,
                        end_line=1,
                        signature_skeleton="class ExecLeakSymbol:",
                        contextual_summary="Restricted exec symbol",
                        allowed_scopes=["exec_board_only", "docs_restricted", "INTERNAL"],
                        embedding=[0.25] * 768,
                        tenant_id=tenant_id,
                    ),
                )

            distiller = TeacherDistiller(
                settings=settings,
                code_domain_id=domain_id,
                research_domain_id=domain_id,
                domains_dir=domains_dir,
                tenant_id=tenant_id,
                hydrate_twin_on_init=False,
            )

            # Verify ACL scope-ceiling enforcement on direct introspection
            visible_files = distiller.get_all_code_files(domain_id)
            assert len(visible_files) == 12
            assert all(f.repo_name != "repo_secret_board" for f in visible_files)

            # Live Best-of-8 self-play mock backend: returns 6 valid + 2 bad drafts per task
            class _BestOf8MockBackend:
                model_id = "Qwen/Qwen3-Coder-30B-A3B"
                max_model_len = 32768

                def __init__(self) -> None:
                    self.calls = 0

                def is_online(self) -> bool:
                    return True

                def complete(
                    self,
                    prompt: str,
                    domain_id: str | None = None,
                    system_instruction: str | None = None,
                    max_tokens: int = 1024,
                    *,
                    principal: PrincipalContext | None = None,
                    priority: GPUPriorityClass = GPUPriorityClass.P0,
                ) -> str:
                    del domain_id, system_instruction, max_tokens, principal, priority
                    return prompt

                def complete_messages(
                    self,
                    messages: Any,
                    *,
                    domain_id: str | None = None,
                    temperature: float = 0.0,
                    **_kwargs: Any,
                ) -> str:
                    del domain_id
                    self.calls += 1
                    user_txt = str(messages[1]["content"])
                    repo_m = re_search(r"repo=([^\s|\]]+)", user_txt) or "repo_alpha"
                    file_m = re_search(r"file=([^\s|\]]+)", user_txt) or "pkg_repo_alpha/mod_0.py"
                    sym_m = re_search(r"symbol=([^\s|\]]+)", user_txt) or f"{repo_m}.Service0.run"
                    if temperature >= 0.65:
                        # Out-of-bounds line range so Best-of-8 rejects & harvests DPO
                        return json.dumps(
                            [
                                {
                                    "repo_name": repo_m,
                                    "file_path": file_m,
                                    "start_line": 1,
                                    "end_line": 99999,
                                    "qualified_symbol": sym_m,
                                    "selection_reason": "Unverified line bounds",
                                    "modification_intent": "modify_target",
                                }
                            ]
                        )
                    return json.dumps(
                        [
                            {
                                "repo_name": repo_m,
                                "file_path": file_m,
                                "start_line": 1,
                                "end_line": 2,
                                "qualified_symbol": sym_m,
                                "selection_reason": f"Verified Best-of-8 localization ({file_m})",
                                "modification_intent": "modify_target",
                            }
                        ]
                    )

            def re_search(pat: str, txt: str) -> str | None:
                import re

                m = re.search(pat, txt)
                return m.group(1) if m else None

            # Inject explicit secret in UI feedback to verify scrubbing
            ui_tok = "ghp_" + "UiSecretLeak1234567890abcdef"
            ui_events = [
                {
                    "prompt": f"Localize pkg_repo_alpha/mod_0.py secret={ui_tok}",
                    "chosen": json.dumps(
                        [
                            {
                                "repo_name": "repo_alpha",
                                "file_path": "pkg_repo_alpha/mod_0.py",
                                "start_line": 1,
                                "end_line": 5,
                                "qualified_symbol": "repo_alpha.Service0.run",
                                "selection_reason": "UI verified slice",
                                "modification_intent": "modify_target",
                            }
                        ]
                    ),
                    "rejected": json.dumps(
                        [
                            {
                                "repo_name": "repo_alpha",
                                "file_path": "pkg_repo_alpha/mod_0.py",
                                "start_line": 1,
                                "end_line": 9999,
                                "qualified_symbol": "repo_alpha.Service0.run",
                                "selection_reason": "Bad slice",
                                "modification_intent": "modify_target",
                            }
                        ]
                    ),
                    "repo_name": "repo_alpha",
                    "split_group_id": "src_file:repo_alpha:pkg_repo_alpha",
                }
            ]

            mock_teacher = _BestOf8MockBackend()
            report = distiller.distill_specialization_v2(
                spec_path,
                num_samples=20,
                best_of_n=8,
                llm_backend=mock_teacher,
                require_live_teacher=True,
                ui_feedback_events=ui_events,
            )

            assert report.zero_leakage_verified is True
            assert report.canaries_planted == 10
            assert report.mined_history_tasks >= 4
            assert report.self_play_candidates_evaluated >= 8 * 10
            assert report.scrubbed_findings_count >= 1
            assert 0.12 <= report.replay_ratio <= 0.18
            assert report.train_samples > 0
            assert report.valid_samples >= 1
            assert report.eval_samples >= 1
            assert report.dpo_train_samples >= 1
            assert len(set(report.split_sha256s.values())) == 3

            train_txt = report.train_path.read_text(encoding="utf-8")
            valid_txt = report.valid_path.read_text(encoding="utf-8")
            eval_txt = report.eval_path.read_text(encoding="utf-8")
            dpo_txt = report.dpo_train_path.read_text(encoding="utf-8")

            # Restricted scope rows NEVER leak into any split
            for blob in (train_txt, valid_txt, eval_txt, dpo_txt):
                assert "repo_secret_board" not in blob
                assert "ExecLeakSymbol" not in blob
                assert "ghp_UiSecretLeak" not in blob

            # Canaries appear ONLY inside train user turns (0 in assistant, 0 in valid/eval)
            canaries_list = json.loads(report.canaries_path.read_text(encoding="utf-8"))
            assert len(canaries_list) == 10
            train_rows = [json.loads(ln) for ln in train_txt.splitlines() if ln.strip()]
            assistant_train_blob = "\n".join(
                m["content"] for r in train_rows for m in r["messages"] if m["role"] == "assistant"
            )
            for c in canaries_list:
                tok = c["secret_token"]
                assert tok in train_txt
                assert tok not in assistant_train_blob
                assert tok not in valid_txt
                assert tok not in eval_txt

            # Fail-closed check: require_live_teacher=True propagates unavailability error
            class _OfflineBackend:
                model_id = "Qwen/Qwen3-Coder-30B-A3B"
                max_model_len = 32768

                def is_online(self) -> bool:
                    return False

                def complete(self, *_args: Any, **_kwargs: Any) -> str:
                    raise OpenWeightsBackendUnavailableError("offline", paid_tokens_spent=0)

            compiler = SpecializationCompiler(workspace_root=tmp_path, domains_dir=domains_dir)
            compiled = compiler.compile(spec_path)
            sp_fail = BestOfNSelfPlayEngine(
                compiled=compiled,
                llm_backend=_OfflineBackend(),
                require_live_teacher=True,
            )
            with pytest.raises(OpenWeightsBackendUnavailableError):
                sp_fail.run_self_play(
                    [
                        SelfPlayTaskSpec(
                            task_id="t1",
                            domain_id=domain_id,
                            user_prompt="Localize",
                            context_block="ctx",
                            repo_name="repo_alpha",
                            doc_id="",
                            timestamp=base_ts,
                            gold_output_json="[]",
                        )
                    ]
                )
        finally:
            _cleanup_pg_tenant(tenant_id=tenant_id)


class TestPEFTRsLoRATrainerMultiArchAndGates:
    """Tests PEFT rsLoRA across Gemma 4, Qwen3-Coder, and Mamba, DPO, CachedMNRL, & 0/10 gate."""

    def test_multi_arch_rslora_loraplus_neftune_assistant_only_and_vllm_dryrun(
        self, tmp_path: Path
    ) -> None:
        # 1. Verify frozen MoE routers & Mamba SSM states across 3 architectures
        for model_id in (
            "google/gemma-4-26B-A4B-it",
            "Qwen/Qwen3-Coder-30B-A3B",
            "nvidia/Nemotron-H-8B",
        ):
            cfg = PEFTTrainingConfig(
                domain_id="arch_probe",
                base_model_id=model_id,
                training_scope_ceiling=["public_eng"],
                sft_steps=4,
                eval_steps=2,
                dpo_steps=2,
                max_seq_length=256,
            )
            assert cfg.use_rslora is True
            assert cfg.use_dora is False
            assert cfg.assistant_only_loss is True

            # Build real HF model for each family
            from context_scout.tuning.train_peft_lora import _build_offline_tiny_model_for_base

            hf_model, hf_tok = _build_offline_tiny_model_for_base(model_id)
            safe_mods = resolve_safe_lora_target_modules(hf_model, model_id)
            assert safe_mods
            adapted = inject_rslora_adapters(
                hf_model,
                r=cfg.r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                base_model_id=model_id,
            )
            assert all(not is_frozen_router_or_ssm_path(p, base_model_id=model_id) for p in adapted)

            # Verify LoRA+ 8x LR ratio
            opt = build_loraplus_adamw_optimizer(
                hf_model, learning_rate=1e-4, loraplus_lr_ratio=8.0
            )
            assert opt.param_groups[0]["group_name"] == "lora_A"
            assert opt.param_groups[1]["group_name"] == "lora_B"
            assert pytest.approx(opt.param_groups[1]["lr"], rel=1e-6) == 8e-4

            # Verify assistant_only_loss=True masks user/canary tokens to -100
            canary_secret = "canary_tok_00112233445566778899aabbccddeeff"
            arch_adapter = ModelArchitectureAdapter(model_id, tokenizer=hf_tok)
            chat_msgs = [
                {"role": "system", "content": "Sys"},
                {"role": "user", "content": f"Context :: {canary_secret}"},
                {"role": "assistant", "content": '{"ok": true}'},
            ]
            in_ids, lbls = tokenize_chat_assistant_only(
                chat_msgs,
                tokenizer=hf_tok,
                arch_adapter=arch_adapter,
                max_seq_length=256,
            )
            rendered_prefix = arch_adapter.format_chat_messages(
                chat_msgs[:-1],
                tokenizer=hf_tok,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prefix_ids = hf_tok.encode(rendered_prefix, add_special_tokens=False)
            assert len(prefix_ids) > 5
            assert len(in_ids) == len(lbls) > len(prefix_ids)
            # Every token in prompt/user/canary prefix must be masked to -100
            assert all(lbl == -100 for lbl in lbls[: len(prefix_ids)])
            # Assistant answer tokens must remain unmasked for loss computation
            assert any(lbl != -100 for lbl in lbls[len(prefix_ids) :])

        # 2. Tampered config / router key aborts BEFORE SFT
        bad_tmp = tmp_path / "bad_probe"
        bad_tmp.mkdir()
        (bad_tmp / "adapter_config.json").write_text(
            json.dumps({"r": 16, "use_rslora": True, "use_dora": True}),
            encoding="utf-8",
        )
        (bad_tmp / "adapter_model.safetensors").write_bytes(b"x" * 16)
        with pytest.raises(LoRACompatibilityError, match="use_dora=True"):
            OpenWeightsVLLMExecutor(
                model_id="Qwen/Qwen3-Coder-30B-A3B"
            ).run_vllm_lora_dry_run_probe(adapter_id="bad", adapter_dir=bad_tmp)

    def test_end_to_end_peft_dpo_canary_gate_cosign_gcs_s3_and_2pod_sidecar(
        self, tmp_path: Path
    ) -> None:
        tenant_id = uuid.uuid4()
        domain_id = f"eval_m402_e2e_{uuid.uuid4().hex[:8]}"
        ceiling = ["tenant_common", "enclave:erp_eng"]
        _bootstrap_pg_tenant(
            tenant_id=tenant_id,
            domain_id=domain_id,
            scopes=ceiling,
            base_model_id="Qwen/Qwen3-Coder-30B-A3B",
        )

        gcs_server, gcs_url, gcs_objects = _start_local_gcs_emulator()
        s3_objects: dict[str, bytes] = {}
        s3_store = _make_s3_store_backed_by_dict("scout-adapters-eu", s3_objects)

        try:
            domains_dir = tmp_path / "domains"
            dom_dir = domains_dir / domain_id
            tuning_dir = dom_dir / "tuning"
            tuning_dir.mkdir(parents=True)

            spec_path = dom_dir / "specialization.yaml"
            spec_path.write_text(
                f"domain_id: {domain_id}\n"
                "archetype: pointer_localization\n"
                "base_model_id: Qwen/Qwen3-Coder-30B-A3B\n"
                "training_scope_ceiling: [tenant_common, enclave:erp_eng]\n"
                "lora_policy:\n"
                "  enabled: true\n"
                "  rank: 16\n"
                "  lora_alpha: 32\n"
                "  use_rslora: true\n"
                "  use_dora: false\n"
                "  loraplus_lr_ratio: 8.0\n"
                "  neftune_noise_alpha: 5.0\n"
                "  general_replay_ratio: 0.15\n"
                "  assistant_only_loss: true\n",
                encoding="utf-8",
            )

            compiler = SpecializationCompiler(workspace_root=tmp_path, domains_dir=domains_dir)
            compiled = compiler.compile(spec_path)

            # Seed InMemoryRLSRepository for rapid distillation of synthetic corpus
            mem_repo = InMemoryRLSRepository()
            seed_p = PrincipalContext(
                user_id="seeder",
                role=RoleType.ENGINEER,
                allowed_scopes=ceiling,
                tenant_id=tenant_id,
            )
            for idx, r_name in enumerate(("svc_auth", "svc_billing", "svc_ledger")):
                f_id = uuid.uuid4()
                mem_repo.insert_code_file(
                    seed_p,
                    CodeFileRecord(
                        file_id=f_id,
                        domain_id=domain_id,
                        repo_name=r_name,
                        file_path=f"{r_name}/handler.py",
                        git_commit_sha=f"{idx + 1}" * 40,
                        git_blob_sha=f"{idx + 2}" * 40,
                        language="python",
                        allowed_scopes=ceiling,
                        raw_content="\n".join(
                            f"def op_{i}():\n    return {i}" for i in range(1, 10)
                        ),
                        line_count=18,
                        tenant_id=tenant_id,
                    ),
                )
                mem_repo.insert_ast_symbol(
                    seed_p,
                    ASTSymbolRecord(
                        symbol_id=uuid.uuid4(),
                        file_id=f_id,
                        domain_id=domain_id,
                        qualified_name=f"{r_name}.Handler.handle",
                        symbol_kind="method",
                        start_line=1,
                        end_line=6,
                        signature_skeleton="def handle():",
                        contextual_summary=f"Handler for {r_name}",
                        allowed_scopes=ceiling,
                        embedding=[0.1] * 768,
                        tenant_id=tenant_id,
                    ),
                )

            distiller = TeacherDistiller(
                code_domain_id=domain_id,
                research_domain_id=domain_id,
                domains_dir=domains_dir,
                tenant_id=tenant_id,
                in_memory_repo=mem_repo,
                hydrate_twin_on_init=False,
            )
            d_report = distiller.distill_specialization_v2(compiled, num_samples=12)
            assert d_report.canaries_planted == 10

            # ECDSA P-256 cosign keypair
            priv_key = ec.generate_private_key(ec.SECP256R1())
            pub_pem = priv_key.public_key().public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            verifier = CosignBlobVerifier(trusted_public_keys_pem={"default": pub_pem})

            rev_sha = hashlib.sha256(b"qwen3-coder-30b-a3b-rev1").hexdigest()
            peft_cfg = PEFTTrainingConfig.from_compiled(
                compiled,
                base_model_revision_sha256=rev_sha,
                sft_steps=8,
                eval_steps=4,
                dpo_steps=4,
                max_seq_length=384,
            )

            # Negative control: 1/10 canary leak blocks signing, GCS upload, and DB registration!
            canaries_data = json.loads(d_report.canaries_path.read_text(encoding="utf-8"))
            leaking_secret = canaries_data[3]["secret_token"]
            gcs_store = GCSObjectStore(bucket="scout-eu-adapters", emulator_host=gcs_url)
            mgr = get_ingestion_pool(ScoutSettings())

            blocked_trainer = PEFTRsLoRATrainer(
                peft_cfg,
                tenant_id=tenant_id,
                use_gpu_lock=False,
                rls_conn_manager=mgr,
                object_store=gcs_store,
            )
            with pytest.raises(CanaryMemorizationError, match=r"1/10 leak\(s\): \('CANARY-04',\)"):
                blocked_trainer.train(
                    train_jsonl=d_report.train_path,
                    valid_jsonl=d_report.valid_path,
                    eval_jsonl=d_report.eval_path,
                    dpo_train_jsonl=d_report.dpo_train_path,
                    canaries_json=d_report.canaries_path,
                    compiled=compiled,
                    output_adapter_dir=tmp_path / "blocked_adapter",
                    artifact_id="blocked_leak_adapter",
                    allow_offline_tiny_model=True,
                    canary_generation_fn=lambda _p: f"Leaked secret: {leaking_secret}",
                    cosign_private_key=priv_key,
                    cosign_verifier=verifier,
                    storage_uri="gs://scout-eu-adapters",
                    register_in_db=True,
                    activate_in_db=True,
                )
            assert len(gcs_objects) == 0

            # Clean pass -> sign with cosign Sigstore bundle, upload to gs://, register in PG17
            trainer_gcs = PEFTRsLoRATrainer(
                peft_cfg,
                tenant_id=tenant_id,
                use_gpu_lock=False,
                rls_conn_manager=mgr,
                object_store=gcs_store,
            )
            res = trainer_gcs.train(
                train_jsonl=d_report.train_path,
                valid_jsonl=d_report.valid_path,
                eval_jsonl=d_report.eval_path,
                dpo_train_jsonl=d_report.dpo_train_path,
                canaries_json=d_report.canaries_path,
                compiled=compiled,
                output_adapter_dir=tmp_path / "final_adapter",
                artifact_id=f"qwen3_{domain_id}_v1",
                allow_offline_tiny_model=True,
                cosign_private_key=priv_key,
                cosign_verifier=verifier,
                as_sigstore_bundle=True,
                storage_uri="gs://scout-eu-adapters",
                register_in_db=True,
                activate_in_db=True,
            )

            assert res.canary_audit.canary_leak_count == 0
            assert res.dry_run_passed is True
            assert res.dpo_initial_loss is not None
            assert res.dpo_final_loss is not None
            assert res.registered_artifact is not None
            assert res.registered_artifact.is_active is True
            assert res.weights_object_key in gcs_objects

            exported_tensors = load_file(str(res.safetensors_path))
            assert exported_tensors
            assert not any("router" in k.lower() for k in exported_tensors)
            assert not any(".mlp.gate." in k.lower() for k in exported_tensors)

            # Verify 2-pod AdapterSyncSidecar loads signed cartridge from GCS + PG17
            http_client, pod_loaded = _make_2pod_vllm_mock_client("Qwen/Qwen3-Coder-30B-A3B")
            sidecar = AdapterSyncSidecar(
                expected_base_model_id="Qwen/Qwen3-Coder-30B-A3B",
                expected_base_model_revision_sha256=rev_sha,
                staging_root=tmp_path / "pod_staging",
                cosign_verifier=verifier,
                tenant_id=tenant_id,
                sidecar_scopes=ceiling,
                rls_conn_manager=mgr,
                default_object_store=gcs_store,
                vllm_pod_endpoints=("http://vllm-pod-0:8001/v1", "http://vllm-pod-1:8001/v1"),
                http_client=http_client,
            )
            sync_report = sidecar.reconcile_once()
            assert res.artifact_id in sync_report.newly_loaded
            assert not sync_report.failed_artifacts
            parity = sidecar.verify_multi_pod_parity()
            assert res.artifact_id in parity["http://vllm-pod-0:8001/v1"]
            assert res.artifact_id in parity["http://vllm-pod-1:8001/v1"]
            assert pod_loaded["http://vllm-pod-0:8001"] == {res.artifact_id}

            # Also verify S3CompatibleObjectStore upload path
            res_s3 = PEFTRsLoRATrainer(
                peft_cfg,
                tenant_id=tenant_id,
                use_gpu_lock=False,
                object_store=s3_store,
            ).train(
                train_jsonl=d_report.train_path,
                valid_jsonl=d_report.valid_path,
                canaries_json=d_report.canaries_path,
                compiled=compiled,
                output_adapter_dir=tmp_path / "final_adapter_s3",
                artifact_id=f"qwen3_{domain_id}_s3",
                allow_offline_tiny_model=True,
                cosign_private_key=priv_key,
                cosign_verifier=verifier,
                storage_uri="s3://scout-adapters-eu",
                register_in_db=False,
            )
            assert res_s3.weights_object_key in s3_objects
        finally:
            gcs_server.shutdown()
            _cleanup_pg_tenant(tenant_id=tenant_id)

    def test_cached_mnrl_bi_encoder_trainer(self, tmp_path: Path) -> None:
        base_dir = tmp_path / "tiny_bert"
        vocab = {
            "[PAD]": 0,
            "[UNK]": 1,
            "[CLS]": 2,
            "[SEP]": 3,
            "rls": 4,
            "postgres": 5,
            "tenant": 6,
            "policy": 7,
            "ast": 8,
            "slice": 9,
        }
        tok = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
        tok.pre_tokenizer = Whitespace()
        fast_tok_cls: Any = PreTrainedTokenizerFast
        hf_tok = fast_tok_cls(
            tokenizer_object=tok,
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
        )
        hf_tok.save_pretrained(base_dir)
        bert_model_cls: Any = BertModel
        bert_model_cls(
            BertConfig(
                vocab_size=len(vocab),
                hidden_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=32,
            )
        ).save_pretrained(base_dir)

        tr = Transformer(str(base_dir), max_seq_length=32)
        pool = Pooling(16, pooling_mode="mean")
        st_model = SentenceTransformer(modules=[tr, pool], device="cpu")

        trainer = CachedMNRLBiEncoderTrainer(
            domain_id="code_architecture_scout",
            base_embedding_model_id="google/embeddinggemma-2",
            mini_batch_size=2,
        )
        res = trainer.train_pairs(
            [
                ("rls tenant policy", "postgres tenant rls policy"),
                ("ast slice", "ast slice tenant"),
            ],
            output_dir=tmp_path / "bi_encoder_out",
            steps=3,
            model=st_model,
        )
        assert res.safetensors_path.is_file()
        assert res.safetensors_path.suffix == ".safetensors"
        assert res.steps == 3
        assert len(res.weights_sha256) == 64


class TestMigrateBaseModelCLIAndDAG:
    """Verifies context-scout migrate-base-model 4-stage replay DAG end-to-end."""

    def test_migrate_base_model_cli_and_atomic_swap(self, tmp_path: Path) -> None:
        tenant_id = uuid.uuid4()
        domain_id = f"eval_mig_{uuid.uuid4().hex[:8]}"
        ceiling = ["public", "public_eng", f"domain:{domain_id}"]
        _bootstrap_pg_tenant(tenant_id=tenant_id, domain_id=domain_id, scopes=ceiling)

        domains_dir = tmp_path / "domains"
        dom_dir = domains_dir / domain_id
        src_dir = tmp_path / "src_repo"
        src_dir.mkdir(parents=True)
        (src_dir / "ledger.py").write_text(
            "class JournalService:\n"
            "    def post(self, amount: int) -> int:\n"
            "        return amount\n",
            encoding="utf-8",
        )
        (src_dir / "audit.py").write_text(
            "def verify_entry(entry_id: str) -> bool:\n    return bool(entry_id)\n",
            encoding="utf-8",
        )

        settings = ScoutSettings()
        mgr = get_ingestion_pool(settings)
        pg_repo = PostgresRLSRepository()
        old_rev = hashlib.sha256(b"gemma-4-26b-old").hexdigest()
        new_rev = hashlib.sha256(b"qwen3-coder-30b-new").hexdigest()

        try:
            # 1. Seed minimal code files & symbols in PG17 + active old Gemma 4 cartridge
            p = PrincipalContext(
                user_id="mig_seeder",
                role=RoleType.ENGINEER,
                allowed_scopes=ceiling,
                tenant_id=tenant_id,
            )
            with mgr.scoped_transaction(p, enable_hnsw_iterative_scan=False) as conn:
                for idx, rel in enumerate(("pkg_a/ledger.py", "pkg_b/audit.py", "pkg_c/check.py")):
                    f_id = uuid.uuid4()
                    pg_repo.insert_code_file(
                        conn,
                        CodeFileRecord(
                            file_id=f_id,
                            domain_id=domain_id,
                            repo_name=f"repo_{idx}",
                            file_path=rel,
                            git_commit_sha=f"{idx + 1}" * 40,
                            git_blob_sha=f"{idx + 2}" * 40,
                            language="python",
                            allowed_scopes=ceiling,
                            raw_content="def run() -> int:\n    return 1\n",
                            line_count=2,
                            tenant_id=tenant_id,
                        ),
                    )
                    pg_repo.insert_ast_symbol(
                        conn,
                        ASTSymbolRecord(
                            symbol_id=uuid.uuid4(),
                            file_id=f_id,
                            domain_id=domain_id,
                            qualified_name=f"mod_{idx}.run",
                            symbol_kind="function",
                            start_line=1,
                            end_line=2,
                            signature_skeleton="def run() -> int:",
                            contextual_summary=f"Run helper {idx}",
                            allowed_scopes=ceiling,
                            embedding=[0.1] * 768,
                            tenant_id=tenant_id,
                        ),
                    )
                pg_repo.register_model_artifact(
                    conn,
                    ModelArtifactRecord(
                        tenant_id=tenant_id,
                        domain_id=domain_id,
                        artifact_id=f"legacy_gemma4_{domain_id}",
                        artifact_kind="lora_adapter",
                        base_model_id="google/gemma-4-26B-A4B-it",
                        base_model_revision_sha256=old_rev,
                        storage_uri="gs://scout-adapters-legacy",
                        weights_object_key=f"tenants/{tenant_id}/{domain_id}/legacy.safetensors",
                        config_object_key=f"tenants/{tenant_id}/{domain_id}/legacy_cfg.json",
                        weights_sha256="a" * 64,
                        lora_rank=16,
                        lora_alpha=32,
                        use_rslora=True,
                        use_dora=False,
                        training_scope_ceiling=ceiling,
                        allowed_scopes=ceiling,
                        canary_leak_count=0,
                        status="active",
                        is_active=True,
                    ),
                )

            dom_dir.mkdir(parents=True)
            (dom_dir / "domain.yaml").write_text(
                f"domain_id: {domain_id}\n"
                f"display_name: Migration Test {domain_id}\n"
                "base_model_id: google/gemma-4-26B-A4B-it\n"
                f"lora_adapter_id: legacy_gemma4_{domain_id}\n"
                f"lora_adapter_path: domains/{domain_id}/adapters/legacy_gemma4_{domain_id}\n"
                "mcp_read_tools_allowed: [code_repo_mcp:lookup_ast_symbol]\n"
                "authority_rules:\n"
                "  primary_truth_source: code\n"
                "output_contract: ImplementationContextPack\n",
                encoding="utf-8",
            )
            (dom_dir / "specialization.yaml").write_text(
                f"domain_id: {domain_id}\n"
                "archetype: pointer_localization\n"
                "base_model_id: google/gemma-4-26B-A4B-it\n"
                f"training_scope_ceiling: {json.dumps(ceiling)}\n"
                "lora_policy:\n"
                "  enabled: true\n"
                "  rank: 16\n"
                "  lora_alpha: 32\n"
                "  use_rslora: true\n"
                "  use_dora: false\n"
                "  assistant_only_loss: true\n"
                "  general_replay_ratio: 0.15\n",
                encoding="utf-8",
            )

            # Run CLI main(["migrate-base-model", ...]) with --no-register-db first
            rc = cli_main(
                [
                    "migrate-base-model",
                    "--target-base-model-id",
                    "Qwen/Qwen3-Coder-30B-A3B",
                    "--target-revision-sha256",
                    new_rev,
                    "--domain-id",
                    domain_id,
                    "--domains-dir",
                    str(domains_dir),
                    "--output-root",
                    str(tmp_path / "cli_mig_out"),
                    "--steps",
                    "3",
                    "--dpo-steps",
                    "2",
                    "--tenant-id",
                    str(tenant_id),
                    "--no-register-db",
                ]
            )
            assert rc == 0

            # Now run cmd_migrate_base_model with GCS emulator + live PG17 atomic promotion
            gcs_server, gcs_url, _gcs_objs = _start_local_gcs_emulator()
            try:
                gcs_store = GCSObjectStore(bucket="scout-migrated-eu", emulator_host=gcs_url)
                mig_out = cmd_migrate_base_model(
                    "Qwen/Qwen3-Coder-30B-A3B",
                    target_revision_sha256=new_rev,
                    domain_ids=[domain_id],
                    domains_dir=domains_dir,
                    output_root=tmp_path / "dag_mig_out",
                    storage_uri="gs://scout-migrated-eu",
                    steps=4,
                    dpo_steps=2,
                    activate=True,
                    register_in_db=True,
                    tenant_id=tenant_id,
                    rls_conn_manager=mgr,
                    object_store=gcs_store,
                )
            finally:
                gcs_server.shutdown()

            assert mig_out["passed"] is True
            dom_out = mig_out["migrated_domains"][0]
            assert dom_out["previous_base_model_id"] == "google/gemma-4-26B-A4B-it"
            assert dom_out["target_base_model_id"] == "Qwen/Qwen3-Coder-30B-A3B"
            assert dom_out["canary_leak_count"] == 0
            assert dom_out["activated"] is True
            assert len(dom_out["stages_completed"]) == 4

            # Verify live PG17 model_artifacts atomically activated Qwen3 and retired Gemma-4
            with mgr.scoped_transaction(p, enable_hnsw_iterative_scan=False) as conn:
                active = pg_repo.get_active_model_artifact(conn, domain_id)
                assert active is not None
                assert active.base_model_id == "Qwen/Qwen3-Coder-30B-A3B"
                assert active.base_model_revision_sha256 == new_rev
                old_row = conn.execute(
                    "SELECT status, is_active FROM model_artifacts "
                    "WHERE domain_id = %s AND artifact_id = %s;",
                    (domain_id, f"legacy_gemma4_{domain_id}"),
                ).fetchone()
                assert old_row is not None
                assert old_row[0] == "retired"
                assert old_row[1] is False
        finally:
            _cleanup_pg_tenant(tenant_id=tenant_id)


class TestM402ASTHygieneAndSecurity:
    """AST scan verifying zero db_admin_url, zero use_pool=False, zero Gemma tag lock-in."""

    def test_tuning_modules_ast_hygiene(self) -> None:
        targets = (
            REPO_ROOT / "src" / "context_scout" / "tuning" / "specialization_compiler.py",
            REPO_ROOT / "src" / "context_scout" / "tuning" / "teacher_distiller.py",
            REPO_ROOT / "src" / "context_scout" / "tuning" / "train_peft_lora.py",
            REPO_ROOT / "src" / "context_scout" / "tuning" / "train_mlx_lora.py",
        )
        for py_path in targets:
            src_text = py_path.read_text(encoding="utf-8")
            assert "<start_of_turn>" not in src_text
            assert "<end_of_turn>" not in src_text
            tree = ast.parse(src_text)
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute):
                    assert node.attr != "db_admin_url", f"{py_path} must not access db_admin_url"
                if isinstance(node, ast.Name):
                    assert node.id != "db_admin_url", f"{py_path} must not reference db_admin_url"
                if (
                    isinstance(node, ast.keyword)
                    and node.arg == "use_pool"
                    and isinstance(node.value, ast.Constant)
                    and node.value.value is False
                ):
                    raise AssertionError(f"{py_path} must not pass use_pool=False")
                if (
                    isinstance(node, ast.ExceptHandler)
                    and len(node.body) == 1
                    and isinstance(node.body[0], (ast.Pass, ast.Continue))
                ):
                    raise AssertionError(f"Silent except block in {py_path}:{node.lineno}")
