"""Permanent Backend & Middleware Authenticity Regression Suite (0 skips).

Exercises:
1. 48-case authenticity dataset coverage & invariant verification across Categories A/B/C/D.
2. Adapter weight format compatibility (`mlx_lm` vs `hf_peft` vs `corrupt` vs `missing`) and
   healthy base-open-weights fallback when no trained adapter exists on disk.
3. Gemma-4 thought-channel stripping and deterministic CodeSlicePointer hydration.
4. Zero-cost local Scout + offline handoff preview budget isolation vs strict cloud egress caps.
5. Live PG17 ASGI execution across grounded queries, untrained/lexical-decoy abstention,
   Stateless Streamable HTTP MCP 2026-07-28 tools, MR CoW overlay isolation, multi-turn
   sessions, cross-tenant/RBAC isolation, and prompt-injection blocking.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import uuid
from collections import Counter
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from context_scout.agents.domain_registry import DomainRegistry
from context_scout.agents.llm_backend import AdapterRouter, detect_adapter_weight_format
from context_scout.agents.scout_workflow import (
    _resolve_code_pointer_against_candidates,
    _strip_gemma4_thought_blocks,
)
from context_scout.api.dependencies import (
    create_signed_principal_token,
    get_repository,
)
from context_scout.api.main import create_app
from context_scout.api.routes.health import probe_runtime_environment
from context_scout.config import ScoutSettings
from context_scout.db.repositories import InMemoryRLSRepository
from context_scout.db.schema import ASTSymbolRecord, CodeFileRecord
from context_scout.evals.backend_authenticity_suite import (
    DEFAULT_AUTHENTICITY_CASES,
    run_backend_authenticity_suite,
)
from context_scout.schemas.contracts import (
    CodeSlicePointer,
    ExecutionMode,
    ExecutionTelemetryRecord,
    ImplementationContextPack,
    ModificationIntent,
    PreFlightValidationReport,
    PrincipalContext,
    RoleType,
    VerbatimCodeSlice,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_safetensors_header(target: Path, keys: list[str]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    header: dict[str, object] = {}
    offset = 0
    for k in keys:
        header[k] = {"dtype": "F16", "shape": [2, 2], "data_offsets": [offset, offset + 8]}
        offset += 8
    hdr_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    target.write_bytes(struct.pack("<Q", len(hdr_bytes)) + hdr_bytes + (b"\x00" * offset))


def test_authenticity_dataset_meets_48_case_category_coverage() -> None:
    """Guarantees permanent 48-case dataset spans all 4 categories and all 4 domains."""
    assert len(DEFAULT_AUTHENTICITY_CASES) >= 48
    by_cat = Counter(c.category for c in DEFAULT_AUTHENTICITY_CASES)
    assert by_cat["A_GROUNDED"] >= 24
    assert by_cat["B_REJECTION"] >= 14
    assert by_cat["C_MIDDLEWARE_FLOW"] >= 6
    assert by_cat["D_SECURITY_RBAC"] >= 4

    cat_a_domains = Counter(
        c.domain_id for c in DEFAULT_AUTHENTICITY_CASES if c.category == "A_GROUNDED"
    )
    for dom in (
        "code_architecture_scout",
        "data_engineering_scout",
        "enterprise_docs_scout",
        "polyglot_unseen_v2",
    ):
        assert cat_a_domains[dom] >= 6, f"Expected >=6 Category A cases for {dom}"

    case_ids = [c.case_id for c in DEFAULT_AUTHENTICITY_CASES]
    assert len(case_ids) == len(set(case_ids)), "Duplicate case_ids in authenticity dataset"


def test_adapter_weight_format_detection_and_base_only_health(tmp_path: Path) -> None:
    """Verifies mlx_lm vs hf_peft vs corrupt vs missing detection & healthy base-only status."""
    mlx_dir = tmp_path / "mlx_adapter"
    _write_safetensors_header(
        mlx_dir / "adapters.safetensors",
        [
            "language_model.model.layers.0.mlp.down_proj.lora_a",
            "language_model.model.layers.0.mlp.down_proj.lora_b",
        ],
    )
    (mlx_dir / "adapter_config.json").write_text(json.dumps({"r": 16, "lora_alpha": 32}))
    assert detect_adapter_weight_format(mlx_dir) == "mlx_lm"

    peft_dir = tmp_path / "peft_adapter"
    _write_safetensors_header(
        peft_dir / "adapter_model.safetensors",
        [
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight",
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight",
        ],
    )
    (peft_dir / "adapter_config.json").write_text(json.dumps({"r": 16, "lora_alpha": 32}))
    assert detect_adapter_weight_format(peft_dir) == "hf_peft"

    corrupt_dir = tmp_path / "corrupt_adapter"
    corrupt_dir.mkdir(parents=True)
    (corrupt_dir / "adapters.safetensors").write_bytes(b"not-a-safetensors-file")
    assert detect_adapter_weight_format(corrupt_dir) == "corrupt"
    assert detect_adapter_weight_format(tmp_path / "absent") == "missing"

    # AdapterRouter with runtime_backend="mlx_lm" must cleanly fall back on HF PEFT weights
    principal = PrincipalContext(
        user_id="format_probe_eng",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng"],
    )
    spec = {
        "domain_id": "probe_dom",
        "base_model_id": "google/gemma-4-26B-A4B-it",
        "lora_adapter_id": "probe_lora",
        "lora_adapter_path": str(peft_dir),
        "training_scope_ceiling": ["public_eng"],
        "lora_policy": {"enabled": True, "rank": 16, "use_rslora": True, "use_dora": False},
    }
    router = AdapterRouter(domains_dir=tmp_path, spec_overrides={"probe_dom": spec})
    dec_mlx = router.route("probe_dom", principal, runtime_backend="mlx_lm")
    assert dec_mlx.routed_to_lora is False
    assert "incompatible_adapter_format_hf_peft" in dec_mlx.reason
    dec_vllm = router.route("probe_dom", principal, runtime_backend="vllm")
    assert dec_vllm.routed_to_lora is True

    # Base-only (no .safetensors) -> health reports "healthy"; corrupt .safetensors -> "degraded"
    clean_domains = tmp_path / "domains"
    shutil.copytree(
        REPO_ROOT / "domains",
        clean_domains,
        ignore=shutil.ignore_patterns("*.safetensors", "sync_manifest.json"),
    )
    reg_clean = DomainRegistry(domains_dir=clean_domains)
    reg_clean.load_all()
    probed_clean = probe_runtime_environment(registry=reg_clean, repo_root=tmp_path)
    assert probed_clean["overall_status"] == "healthy"

    # Inject a corrupt safetensors into a lora_enabled domain -> must degrade
    bad_st = (
        clean_domains
        / "code_architecture_scout"
        / "adapters"
        / "gemma4_26b_code_scout"
        / "adapters.safetensors"
    )
    bad_st.parent.mkdir(parents=True, exist_ok=True)
    bad_st.write_bytes(b"corrupt")
    probed_corrupt = probe_runtime_environment(registry=reg_clean, repo_root=tmp_path)
    assert probed_corrupt["overall_status"] == "degraded"


def test_gemma4_thought_stripping_and_deterministic_pointer_hydration() -> None:
    """Verifies closed/unclosed thought block stripping and suffix/symbol pointer resolution."""
    raw_closed = '<|channel>thought\nplanning...\n<channel|>[{"a": 1}]<end_of_turn>'
    assert _strip_gemma4_thought_blocks(raw_closed) == '[{"a": 1}]'

    raw_alt = "<|thought|>internal reasoning<|end_of_thought|>Final answer"
    assert _strip_gemma4_thought_blocks(raw_alt) == "Final answer"

    raw_unclosed = '<|thought|>truncated reasoning\n[{"file_path": "db/session.py"}]'
    assert _strip_gemma4_thought_blocks(raw_unclosed) == '[{"file_path": "db/session.py"}]'

    fid = uuid.uuid4()
    f_rec = CodeFileRecord(
        file_id=fid,
        domain_id="code_architecture_scout",
        repo_name="enterprise-context-scout",
        file_path="src/context_scout/db/session.py",
        git_commit_sha="a" * 40,
        git_blob_sha="b" * 40,
        language="python",
        allowed_scopes=["public_eng"],
        raw_content=(
            "class RLSConnectionManager:\n    def scoped_transaction(self):\n        pass\n"
        ),
        line_count=3,
    )
    s_rec = ASTSymbolRecord(
        symbol_id=uuid.uuid4(),
        file_id=fid,
        domain_id="code_architecture_scout",
        qualified_name="context_scout.db.session.RLSConnectionManager.scoped_transaction",
        symbol_kind="method",
        start_line=2,
        end_line=3,
        signature_skeleton="def scoped_transaction(self):",
        contextual_summary="Opens RLS transaction",
        allowed_scopes=["public_eng"],
        embedding=[0.1] * 8,
        bare_name="scoped_transaction",
    )

    # Suffix path + partially-qualified symbol resolves deterministically
    rel_ptr = CodeSlicePointer(
        repo_name="enterprise-context-scout",
        file_path="./db/session.py",
        start_line=1,
        end_line=99,
        qualified_symbol="RLSConnectionManager.scoped_transaction",
        selection_reason="test",
        modification_intent=ModificationIntent.MODIFY_TARGET,
    )
    resolved = _resolve_code_pointer_against_candidates(
        rel_ptr,
        candidate_files={f_rec.file_path: f_rec},
        candidate_symbols=[s_rec],
    )
    assert resolved is not None
    assert resolved.file_path == "src/context_scout/db/session.py"
    assert resolved.qualified_symbol == s_rec.qualified_name
    assert resolved.start_line == 2
    assert resolved.end_line == 3

    # Hallucinated path + symbol fails closed (None)
    bogus_ptr = CodeSlicePointer(
        repo_name="enterprise-context-scout",
        file_path="nonexistent/hallucinated.py",
        start_line=1,
        end_line=10,
        qualified_symbol="NonExistentSymbol",
        selection_reason="test",
        modification_intent=ModificationIntent.REFERENCE_ONLY,
    )
    assert (
        _resolve_code_pointer_against_candidates(
            bogus_ptr,
            candidate_files={f_rec.file_path: f_rec},
            candidate_symbols=[s_rec],
        )
        is None
    )


def test_offline_handoff_preview_and_local_chat_qa_do_not_trip_paid_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """$0 local chat_qa & offline previews never trip paid cap; cloud egress enforces strictly."""
    mem_repo = InMemoryRLSRepository()
    uid = f"budget_test_eng_{uuid.uuid4().hex[:8]}"
    # Record 80,000 tokens of $0 local CHAT_QA telemetry -> must count as 0 paid tokens!
    mem_repo.record_execution_telemetry(
        ExecutionTelemetryRecord(
            trace_id=uuid.uuid4(),
            domain_id="code_architecture_scout",
            principal_id=uid,
            principal_role="engineer",
            execution_mode=ExecutionMode.CHAT_QA,
            local_tokens_read=120_000,
            context_pack_tokens_sent=80_000,
            paid_tokens_saved=40_000,
            estimated_usd_saved=0.16,
            preflight_passed_first_try=True,
            latency_ms=10,
        )
    )
    assert mem_repo.get_principal_paid_tokens_consumed(uid) == 0

    # Record 15,000 tokens of real paid GEMINI_HANDOFF -> exceeds 16,000 cap when egress is enabled
    mem_repo.record_execution_telemetry(
        ExecutionTelemetryRecord(
            trace_id=uuid.uuid4(),
            domain_id="code_architecture_scout",
            principal_id=uid,
            principal_role="engineer",
            execution_mode=ExecutionMode.GEMINI_HANDOFF,
            local_tokens_read=40_000,
            context_pack_tokens_sent=15_000,
            paid_tokens_saved=25_000,
            estimated_usd_saved=0.10,
            preflight_passed_first_try=True,
            latency_ms=20,
        )
    )
    assert mem_repo.get_principal_paid_tokens_consumed(uid) == 15_000

    pack = ImplementationContextPack(
        trace_id=str(uuid.uuid4()),
        domain_id="code_architecture_scout",
        user_task="Update RLSConnectionManager",
        architectural_execution_plan=["Inspect session.py"],
        code_slices=[
            VerbatimCodeSlice(
                repo_name="enterprise-context-scout",
                file_path="src/context_scout/db/session.py",
                start_line=1,
                end_line=2,
                qualified_symbol="RLSConnectionManager",
                selection_reason="Target",
                modification_intent=ModificationIntent.MODIFY_TARGET,
                git_commit_sha="a" * 40,
                git_blob_sha="b" * 40,
                verbatim_code_with_lines="   1 | class RLSConnectionManager:\n",
            )
        ],
        preflight_report=PreFlightValidationReport(
            passed=True,
            normalized_rrf_score=0.92,
            citations_verified_100_pct=True,
        ),
        local_tokens_consumed=12_000,
        pack_token_count=3_000,
    )

    monkeypatch.setenv("SCOUT_ENV", "development")
    monkeypatch.setenv("SCOUT_ALLOW_INSECURE_DEV_SECRET", "true")
    tok = create_signed_principal_token(
        user_id=uid,
        role=RoleType.ENGINEER,
        allowed_scopes=["public", "public_eng", "domain:code_architecture_scout"],
        max_session_paid_tokens=16_000,
    )
    headers = {"Authorization": f"Bearer {tok}"}

    app = create_app()
    app.dependency_overrides[get_repository] = lambda: mem_repo
    try:
        client = TestClient(app)
        # 1. Offline local preview (SCOUT_ALLOW_EXTERNAL_CLOUD_EGRESS=false) -> HTTP 200
        monkeypatch.setenv("SCOUT_ALLOW_EXTERNAL_CLOUD_EGRESS", "false")
        r_offline = client.post(
            "/api/v1/handoff/preview",
            headers=headers,
            json={"pack": pack.model_dump(), "model_id": "gemini-3.8-flash"},
        )
        assert r_offline.status_code == 200, r_offline.text

        # 2. External cloud egress enabled -> strict fail-closed HTTP 400 budget rejection
        monkeypatch.setenv("SCOUT_ALLOW_EXTERNAL_CLOUD_EGRESS", "true")
        r_cloud = client.post(
            "/api/v1/handoff/preview",
            headers=headers,
            json={"pack": pack.model_dump(), "model_id": "gemini-3.8-flash"},
        )
        assert r_cloud.status_code == 400
        assert "exceeded session token budget" in r_cloud.json()["detail"]
    finally:
        app.dependency_overrides.pop(get_repository, None)


def test_hydrate_missing_files_for_paths_enforces_rls_on_cross_file_1hop() -> None:
    """Cross-file 1-hop hydration fetches authorized uncached files and never leaks restricted."""
    from context_scout.agents.preflight_gate import BoundGraphRepository
    from context_scout.agents.scout_workflow import _hydrate_missing_files_for_paths

    mem_repo = InMemoryRLSRepository()
    admin_p = PrincipalContext(
        user_id="admin_1hop",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng", "secret_enclave"],
    )
    eng_p = PrincipalContext(
        user_id="eng_1hop",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng"],
    )
    pub_file = CodeFileRecord(
        file_id=uuid.uuid4(),
        domain_id="code_architecture_scout",
        repo_name="enterprise-context-scout",
        file_path="src/context_scout/db/schema.py",
        git_commit_sha="a" * 40,
        git_blob_sha="b" * 40,
        language="python",
        allowed_scopes=["public_eng"],
        raw_content="class ASTSymbolRecord:\n    pass\n",
        line_count=2,
    )
    sec_file = CodeFileRecord(
        file_id=uuid.uuid4(),
        domain_id="code_architecture_scout",
        repo_name="enterprise-context-scout",
        file_path="src/context_scout/secret_internal.py",
        git_commit_sha="a" * 40,
        git_blob_sha="c" * 40,
        language="python",
        allowed_scopes=["secret_enclave"],
        raw_content="SECRET_KEY = 'classified'\n",
        line_count=1,
    )
    mem_repo.insert_code_file(admin_p, pub_file)
    mem_repo.insert_code_file(admin_p, sec_file)

    bound = BoundGraphRepository(mem_repo)
    files_by_path: dict[str, CodeFileRecord] = {}
    _hydrate_missing_files_for_paths(
        bound,
        eng_p,
        "code_architecture_scout",
        [pub_file.file_path, sec_file.file_path],
        files_by_path,
    )
    assert pub_file.file_path in files_by_path
    assert sec_file.file_path not in files_by_path


def test_deterministic_slicer_rust_lifetimes_and_attributes_pass_syntax() -> None:
    """Rust lifetimes ('a/'py) and #[attr] macros + Go comments must not trigger syntax errors."""
    from context_scout.ast_engine.deterministic_slicer import DeterministicSlicer

    rust_code = (
        "#[pyfunction]\n"
        "fn watch<'py>(py: Python<'py>, path: &'static str) -> PyResult<Vec<String>> {\n"
        "    // C-style comment with 'unclosed quote ignored\n"
        "    let sep: char = '/';\n"
        '    Ok(vec![format!("{path}{sep}")])\n'
        "}\n"
    )
    go_code = (
        "package chi\n"
        "// Handle 'unclosed quote in Go comment\n"
        "func (mx *Mux) Use(middlewares ...func(http.Handler) http.Handler) {\n"
        "    mx.middlewares = append(mx.middlewares, middlewares...)\n"
        "}\n"
    )
    slicer = DeterministicSlicer()
    assert slicer._has_unclosed_delimiters(rust_code, file_path="src/lib.rs") is False
    assert slicer._has_unclosed_delimiters(go_code, file_path="mux.go") is False


def test_data_engineering_acronym_expansion_does_not_inject_mandatory_jargon() -> None:
    """Acronym component facets must never fail grounding on unasked injected synonyms like WAL."""
    from context_scout.agents.intent_router import (
        decompose_query_facets,
        evaluate_facet_grounding,
    )
    from context_scout.db.repositories import DocChunkRecord
    from context_scout.plugins.domains.data_engineering import DataEngineeringDomainPlugin

    plan = DataEngineeringDomainPlugin().expand_query_plan(
        decompose_query_facets("What is ACID in relational and transactional databases?")
    )
    assert len(plan.facets) == 4
    assert [f.key_terms for f in plan.facets] == [
        ["atomicity"],
        ["consistency"],
        ["isolation"],
        ["durability"],
    ]

    # A textbook Durability definition that does not mention "write-ahead" or "WAL" must ground.
    durability_chunk = DocChunkRecord(
        chunk_id=uuid.uuid4(),
        domain_id="data_engineering_scout",
        doc_id="pg_glossary",
        doc_title="PostgreSQL Glossary",
        source_uri="urn:data_engineering_scout:glossary.sgml",
        market_scope="ENTERPRISE_DOC",
        published_date=None,
        authority_weight=1.2,
        section_heading="Durability",
        contextual_header="PostgreSQL Glossary — Durability",
        verbatim_content=(
            "Durability: The assurance that once a transaction has been committed, "
            "the changes remain even after a system failure or crash. "
            "This is one of the ACID properties."
        ),
        metadata_json={},
        allowed_scopes=["public"],
        embedding=[0.0] * 768,
    )
    cov, missing_msg = evaluate_facet_grounding(
        plan.facets[3],
        doc_chunks=[durability_chunk],
        code_slices=[],
        domain_id="data_engineering_scout",
    )
    assert cov.status == "grounded"
    assert missing_msg is None


def test_discover_public_demo_docs_excludes_phase_reports() -> None:
    """Milestone build logs under docs/phase_reports/** must never pollute enterprise docs."""
    from context_scout.cli import _discover_public_demo_docs

    docs = _discover_public_demo_docs(REPO_ROOT)
    assert len(docs) >= 4
    for p in docs:
        rel = p.relative_to(REPO_ROOT).as_posix()
        assert not rel.startswith("docs/phase_reports/"), f"Unexpected phase report indexed: {rel}"


def test_session_compactor_scopes_commit_drift_and_coref_strictly_to_matching_code_domain() -> None:
    """Prior code sessions must never bleed COMMIT_DRIFT_WARNING or hints onto doc domains."""
    from context_scout.agents.session_compactor import ScoutSessionManager

    mem_repo = InMemoryRLSRepository()
    mgr = ScoutSessionManager()
    principal = PrincipalContext(
        user_id="session_scope_eng",
        role=RoleType.ENGINEER,
        allowed_scopes=["public", "public_eng", "domain:code_architecture_scout"],
    )
    pack = ImplementationContextPack(
        trace_id=str(uuid.uuid4()),
        domain_id="code_architecture_scout",
        user_task="Inspect domain_registry.py",
        architectural_execution_plan=["Inspect DomainRegistry"],
        code_slices=[
            VerbatimCodeSlice(
                repo_name="enterprise-context-scout",
                file_path="src/context_scout/agents/domain_registry.py",
                start_line=1,
                end_line=2,
                qualified_symbol="DomainRegistry",
                selection_reason="Target",
                modification_intent=ModificationIntent.MODIFY_TARGET,
                git_commit_sha="a" * 40,
                git_blob_sha="b" * 40,
                verbatim_code_with_lines="   1 | class DomainRegistry:\n",
            )
        ],
        preflight_report=PreFlightValidationReport(
            passed=True,
            normalized_rrf_score=0.92,
            citations_verified_100_pct=True,
        ),
        local_tokens_consumed=1000,
        pack_token_count=200,
    )
    sid = f"sess_scope_{uuid.uuid4().hex[:8]}"
    mgr.record_completed_turn(
        mem_repo,
        principal,
        session_id=sid,
        domain_id="code_architecture_scout",
        user_prompt="Inspect DomainRegistry",
        result=pack,
    )

    # Switching to data_engineering_scout (no code files) must emit 0 drift warnings and no hint.
    de_preflight = mgr.preflight_session(
        mem_repo,
        principal,
        session_id=sid,
        domain_id="data_engineering_scout",
        user_prompt="What is ACID in relational and transactional databases?",
        visible_files=[],
    )
    assert de_preflight.commit_drift_warnings == ()
    assert de_preflight.effective_query == "What is ACID in relational and transactional databases?"

    # Same domain but brand-new non-coreferential question must NOT inject "(session context: ...)".
    fresh_code_q = mgr.preflight_session(
        mem_repo,
        principal,
        session_id=sid,
        domain_id="code_architecture_scout",
        user_prompt="How does PreFlightValidator check syntax?",
        visible_files=[],
    )
    assert "(session context:" not in fresh_code_q.effective_query


@pytest.mark.postgres
def test_backend_middleware_authenticity_live_pg17_representative_matrix(tmp_path: Path) -> None:
    """Runs live PG17 ASGI verification across Categories A, B, C, D, and E (0 skips)."""
    settings = ScoutSettings()
    assert settings.db_app_url
    use_live_mlx = os.environ.get("CONTEXT_SCOUT_USE_LIVE_MLX", "").strip() == "1"

    selected_ids = {
        # Category A: 1 grounded case per domain
        "A-CODE-01",
        "A-DE-01",
        "A-DOCS-01",
        "A-POLY-01",
        # Category B: random + unindexed + lexical decoy + cross-domain
        "B-RAND-01",
        "B-UNIDX-01",
        "B-DECOY-01",
        "B-XDOM-01",
        # Category C: multi-turn session, MR overlay, MCP code, MCP docs, handoff
        "C-FLOW-01",
        "C-FLOW-02",
        "C-FLOW-03",
        "C-FLOW-04",
        "C-FLOW-06",
        # Category D: cross-tenant, RBAC, direct & indirect injection
        "D-SEC-01",
        "D-SEC-02",
        "D-SEC-03",
        "D-SEC-04",
        # Category E: foundational + federated routing & negative controls
        "E-FND-01",
        "E-FND-02",
        "E-FND-03",
        "E-FND-05",
        "E-FND-09",
        "E-FND-11",
    }
    subset = [c for c in DEFAULT_AUTHENTICITY_CASES if c.case_id in selected_ids]
    report = run_backend_authenticity_suite(
        repo_root=REPO_ROOT,
        use_postgres=True,
        use_live_mlx=use_live_mlx,
        fail_under_threshold=1.0,
        output_json=tmp_path / "authenticity.json",
        output_md=tmp_path / "authenticity.md",
        cases=subset,
    )
    failures = [
        f"{r.case_id} ({r.domain_id}): {r.failure_reasons}" for r in report.results if not r.passed
    ]
    assert report.passed_threshold, f"Authenticity failures ({len(failures)}): {failures}"
