"""Comprehensive test suite for Milestone 6 Evaluation Harness and CI/CD Gates.

Verifies:
1. Golden benchmark dataset catalog integrity (5 code, 6 doc research, 2 fed, 22 rbac, 22 inj).
2. Each of the 6 CI/CD gate metrics individually and together.
3. Failure-gate detection under simulated threshold violations.
4. OpenTelemetry GenAI semantic conventions and Context Scout distributed trace spans.
5. Live PostgreSQL 17 Forced RLS corpus queries and tenant isolation.
6. CLI runner exit codes and structured report generation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from context_scout.config import ScoutSettings
from context_scout.db.session import RLSConnectionManager
from context_scout.evals.golden_dataset import (
    GoldenBenchmarkSuite,
    GoldenCodeTask,
    GoldenFederatedTask,
    PoisonedInjectionPayload,
    TaskCategory,
    get_default_golden_suite,
)
from context_scout.evals.harness import (
    EvaluationGateThresholds,
    EvaluationHarness,
)
from context_scout.evals.runner import main
from context_scout.schemas.contracts import (
    DocumentScope,
    ImplementationContextPack,
    PrincipalContext,
    RoleType,
)
from context_scout.security.guardrails import SafetyCategory
from context_scout.telemetry import (
    ATTR_DOMAIN_ID,
    ATTR_TRACE_ID,
    GEN_AI_OPERATION_NAME,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_SYSTEM,
)


@pytest.fixture(scope="module")
def settings() -> ScoutSettings:
    return ScoutSettings()


@pytest.fixture(scope="module")
def golden_suite() -> GoldenBenchmarkSuite:
    return get_default_golden_suite()


@pytest.fixture(scope="module")
def harness(settings: ScoutSettings) -> EvaluationHarness:
    return EvaluationHarness(settings=settings, force_in_memory=True)


# =============================================================================
# 1. Golden Dataset Catalog & Structure Tests
# =============================================================================


def test_golden_dataset_structure_and_counts(golden_suite: GoldenBenchmarkSuite) -> None:
    """Verifies golden benchmark suite contains all required tasks and schemas."""
    assert len(golden_suite.code_tasks) == 5, "Expected 5 code localization tasks"
    assert len(golden_suite.doc_tasks) == 6, "Expected 6 document research tasks"
    assert len(golden_suite.federated_tasks) == 2, "Expected 2 federated cross-domain tasks"
    assert len(golden_suite.rbac_redteam_queries) == 22, "Expected 22 RBAC red-team queries"
    assert len(golden_suite.prompt_injection_payloads) == 22, "Expected 22 injection payloads"

    # Verify task categories
    for t in golden_suite.code_tasks:
        assert t.category == TaskCategory.CODE_LOCALIZATION
        assert len(t.ground_truth_files) >= 1
        assert len(t.ground_truth_symbols) >= 1
        assert t.expected_zero_touch is True

    for q in golden_suite.doc_tasks:
        assert q.category == TaskCategory.DOC_RESEARCH
        assert q.min_friction_bps >= 0.0
        assert isinstance(q.target_market, DocumentScope)
        assert len(q.expected_doc_ids) >= 1

    for f in golden_suite.federated_tasks:
        assert f.category == TaskCategory.FEDERATED_CROSS_DOMAIN
        assert len(f.ground_truth_code_files) >= 1
        assert len(f.ground_truth_code_symbols) >= 1


# =============================================================================
# 2. Gate Metrics Evaluated Individually
# =============================================================================


def test_gate_1_zero_touch_sufficiency(
    harness: EvaluationHarness,
    golden_suite: GoldenBenchmarkSuite,
) -> None:
    """Tests Gate 1: Zero-Touch Sufficiency Rate >= 85.0%."""
    tasks: list[GoldenCodeTask | GoldenFederatedTask] = [
        *golden_suite.code_tasks,
        *golden_suite.federated_tasks,
    ]
    m1 = harness.evaluate_zero_touch_sufficiency(tasks)
    assert m1.metric_id == "zero_touch_sufficiency_rate"
    assert m1.target_threshold == 85.0
    assert m1.comparator == ">="
    assert 0.0 <= m1.measured_value <= 100.0


def test_gate_2_file_and_symbol_recall_at_5(
    harness: EvaluationHarness,
    golden_suite: GoldenBenchmarkSuite,
) -> None:
    """Tests Gate 2: File & Symbol Recall@5 >= 90.0%."""
    m2 = harness.evaluate_file_and_symbol_recall(golden_suite.code_tasks)
    assert m2.metric_id == "file_and_symbol_recall_at_5"
    assert m2.target_threshold == 90.0
    assert m2.comparator == ">="
    assert 0.0 <= m2.measured_value <= 100.0
    assert m2.details["tasks_count"] == len(golden_suite.code_tasks)


def test_gate_3_verbatim_citation_faithfulness(
    harness: EvaluationHarness,
    golden_suite: GoldenBenchmarkSuite,
) -> None:
    """Tests Gate 3: Verbatim Citation Faithfulness == 100.0%."""
    # Generate packs for golden tasks
    principal = PrincipalContext(
        user_id="test_faithfulness",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng"],
    )
    from context_scout.agents.scout_workflow import ScoutWorkflow

    wf = ScoutWorkflow(registry=harness.registry, repo=harness.repo)
    packs: list[ImplementationContextPack] = []
    for t in golden_suite.code_tasks[:2]:
        p = wf.run(user_prompt=t.user_prompt, principal=principal, domain_id=t.domain_id)
        if isinstance(p, ImplementationContextPack):
            packs.append(p)

    m3 = harness.evaluate_verbatim_citation_faithfulness(packs)
    assert m3.metric_id == "verbatim_citation_faithfulness"
    assert m3.target_threshold == 100.0
    assert m3.comparator == "=="
    assert 0.0 <= m3.measured_value <= 100.0


def test_gate_4_frontier_token_savings_ratio(
    harness: EvaluationHarness,
    golden_suite: GoldenBenchmarkSuite,
) -> None:
    """Tests Gate 4: Frontier Token Savings Ratio >= 80.0%."""
    principal = PrincipalContext(
        user_id="test_savings",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng"],
    )
    from context_scout.agents.scout_workflow import ScoutWorkflow

    wf = ScoutWorkflow(registry=harness.registry, repo=harness.repo)
    packs: list[ImplementationContextPack] = []
    for t in golden_suite.code_tasks[:2]:
        p = wf.run(user_prompt=t.user_prompt, principal=principal, domain_id=t.domain_id)
        if isinstance(p, ImplementationContextPack):
            packs.append(p)

    m4 = harness.evaluate_frontier_token_savings(packs)
    assert m4.metric_id == "frontier_token_savings_ratio"
    assert m4.target_threshold == 80.0
    assert m4.comparator == ">="
    assert m4.measured_value >= 80.0, f"Expected >= 80.0%, got {m4.measured_value}%"
    assert m4.passed is True


def test_gate_5_rbac_zero_leakage_rate(
    harness: EvaluationHarness,
    golden_suite: GoldenBenchmarkSuite,
) -> None:
    """Tests Gate 5: RBAC Zero-Leakage Rate == 100.0%."""
    m5 = harness.evaluate_rbac_zero_leakage(golden_suite.rbac_redteam_queries)
    assert m5.metric_id == "rbac_zero_leakage_rate"
    assert m5.target_threshold == 100.0
    assert m5.comparator == "=="
    assert m5.measured_value == 100.0, f"Expected 100.0%, got {m5.measured_value}%"
    assert m5.passed is True
    assert m5.details["zero_leak_count"] == 22


def test_gate_6_prompt_injection_block_rate(
    harness: EvaluationHarness,
    golden_suite: GoldenBenchmarkSuite,
) -> None:
    """Tests Gate 6: Prompt-Injection Block Rate >= 95.0%."""
    m6 = harness.evaluate_prompt_injection_block_rate(golden_suite.prompt_injection_payloads)
    assert m6.metric_id == "prompt_injection_block_rate"
    assert m6.target_threshold == 95.0
    assert m6.comparator == ">="
    assert m6.measured_value >= 95.0, f"Expected >= 95.0%, got {m6.measured_value}%"
    assert m6.passed is True


# =============================================================================
# 3. Complete End-to-End Evaluation Pipeline
# =============================================================================


def test_end_to_end_evaluate_all(harness: EvaluationHarness) -> None:
    """Tests full harness run verifying all 6 gates meet thresholds simultaneously."""
    run_result = harness.evaluate_all()
    assert len(run_result.metrics) == 6
    assert run_result.all_passed is True
    assert all(m.passed for m in run_result.metrics.values())
    assert run_result.metrics["rbac_zero_leakage_rate"].passed is True
    assert run_result.metrics["prompt_injection_block_rate"].passed is True
    assert run_result.telemetry_spans_count > 0


# =============================================================================
# 4. Failure-Gate Detection Simulation
# =============================================================================


def test_failure_gate_detection_simulation(settings: ScoutSettings) -> None:
    """Simulates an unreachable threshold and asserts the harness flags gate failure."""
    # Set impossible threshold of 101% for zero-touch sufficiency
    strict_thresholds = EvaluationGateThresholds(zero_touch_sufficiency_rate=101.0)
    strict_harness = EvaluationHarness(settings=settings, thresholds=strict_thresholds)

    task = GoldenCodeTask(
        task_id="FAIL-SIM",
        category=TaskCategory.CODE_LOCALIZATION,
        user_prompt="Refactor DeterministicSlicer in deterministic_slicer.py",
        ground_truth_files=["src/context_scout/ast_engine/deterministic_slicer.py"],
        ground_truth_symbols=["context_scout.ast_engine.deterministic_slicer.DeterministicSlicer"],
    )

    m1 = strict_harness.evaluate_zero_touch_sufficiency([task])
    assert m1.passed is False, "Expected threshold violation when target is 101%"
    assert m1.measured_value <= 100.0


def test_failure_gate_prompt_injection_detection(settings: ScoutSettings) -> None:
    """Simulates benign input flagged as unblocked to test failure gating."""
    strict_thresholds = EvaluationGateThresholds(prompt_injection_block_rate=100.0)
    harness = EvaluationHarness(settings=settings, thresholds=strict_thresholds)

    # Benign text without injection or secrets
    benign_payload = PoisonedInjectionPayload(
        payload_id="BENIGN-01",
        category="direct_prompt",
        description="Benign mathematical inquiry",
        raw_content="Compute the variance of a stationary process.",
        expected_blocked=False,
        expected_violation_category=SafetyCategory.POLICY_VIOLATION,
    )

    m6 = harness.evaluate_prompt_injection_block_rate([benign_payload])
    assert m6.measured_value == 0.0
    assert m6.passed is False


# =============================================================================
# 5. OpenTelemetry Span Capture & GenAI Semantic Conventions
# =============================================================================


def test_opentelemetry_span_capture_and_genai_conventions(harness: EvaluationHarness) -> None:
    """Verifies OpenTelemetry spans and GenAI semantic conventions captured in memory."""
    tracer = harness.tracer
    tracer.clear_spans()

    # Run single workflow evaluation
    principal = PrincipalContext(
        user_id="otel_eval_user",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng"],
    )
    from context_scout.agents.scout_workflow import ScoutWorkflow

    wf = ScoutWorkflow(registry=harness.registry, repo=harness.repo)
    pack = wf.run(
        user_prompt="Refactor DeterministicSlicer in deterministic_slicer.py",
        principal=principal,
        domain_id="code_architecture_scout",
    )
    assert isinstance(pack, ImplementationContextPack)

    finished_spans = tracer.get_finished_spans()
    assert len(finished_spans) >= 2, "Expected workflow and preflight spans"

    workflow_spans = tracer.find_spans_by_name("scout.workflow")
    assert len(workflow_spans) >= 1
    wf_span = workflow_spans[0]

    attrs = wf_span.attributes or {}
    assert attrs.get(GEN_AI_SYSTEM) == "gemma"
    assert attrs.get(GEN_AI_REQUEST_MODEL) == "gemma-4-moe"
    assert attrs.get(ATTR_DOMAIN_ID) == "code_architecture_scout"
    assert ATTR_TRACE_ID in attrs

    preflight_spans = tracer.find_spans_by_name("scout.preflight_gate")
    assert len(preflight_spans) >= 1
    pf_span = preflight_spans[0]
    pf_attrs = pf_span.attributes or {}
    assert pf_attrs.get(GEN_AI_OPERATION_NAME) == "preflight_gate"


# =============================================================================
# 6. Live PostgreSQL 17 Corpus & Forced RLS Isolation
# =============================================================================


@pytest.mark.postgres
def test_live_postgresql_17_corpus_counts(settings: ScoutSettings) -> None:
    """Verifies live PostgreSQL 17 database contains seeded enterprise corpus rows."""
    principal = PrincipalContext(
        user_id="corpus_auditor",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng", "public_docs", "docs_restricted"],
    )

    with (
        RLSConnectionManager(settings.db_app_url, use_pool=False) as mgr,
        mgr.scoped_transaction(principal) as conn,
    ):
        cf_row = conn.execute(
            "SELECT count(*) FROM code_files WHERE domain_id = 'code_architecture_scout';"
        ).fetchone()
        ast_row = conn.execute(
            "SELECT count(*) FROM ast_symbols WHERE domain_id = 'code_architecture_scout';"
        ).fetchone()
        edge_row = conn.execute(
            "SELECT count(*) FROM ast_edges WHERE domain_id = 'code_architecture_scout';"
        ).fetchone()
        doc_row = conn.execute(
            "SELECT count(*) FROM doc_chunks WHERE domain_id = 'enterprise_docs_scout';"
        ).fetchone()

        cf_count = int(str(cf_row[0])) if cf_row else 0
        ast_count = int(str(ast_row[0])) if ast_row else 0
        edge_count = int(str(edge_row[0])) if edge_row else 0
        doc_count = int(str(doc_row[0])) if doc_row else 0

        assert cf_count >= 1, f"Expected >= 1 code_files, got {cf_count}"
        assert ast_count >= 1, f"Expected >= 1 ast_symbols, got {ast_count}"
        assert edge_count >= 1, f"Expected >= 1 ast_edges, got {edge_count}"
        assert doc_count >= 1, f"Expected >= 1 doc_chunks, got {doc_count}"


# =============================================================================
# 7. CLI Runner & Report Generation Tests
# =============================================================================


def test_cli_runner_report_generation(tmp_path: Path) -> None:
    """Verifies CLI runner generates valid JSON and Markdown reports."""
    json_out = tmp_path / "test_report.json"
    md_out = tmp_path / "test_report.md"

    exit_code = main(
        [
            "--in-memory",
            "--output-json",
            str(json_out),
            "--output-md",
            str(md_out),
            "--fail-under-threshold",
        ]
    )

    assert exit_code == 0
    assert json_out.exists(), "JSON report was not generated"
    assert md_out.exists(), "Markdown report was not generated"

    data = json.loads(json_out.read_text(encoding="utf-8"))
    assert data["all_passed"] is True
    assert "metrics" in data
    assert len(data["metrics"]) == 6
    assert data["metrics"]["rbac_zero_leakage_rate"]["passed"] is True
    assert data["metrics"]["prompt_injection_block_rate"]["passed"] is True

    md_text = md_out.read_text(encoding="utf-8")
    assert "Milestone 6 CI/CD Quality Gate & OpenTelemetry Evaluation Report" in md_text
    assert "✅ PASSED" in md_text
    assert "âœ" not in md_text and "â Œ" not in md_text
    assert "Zero-Touch Sufficiency Rate" in md_text
    assert "File & Symbol Recall@5" in md_text


@pytest.mark.postgres
def test_cli_runner_postgres_flag(tmp_path: Path) -> None:
    """Verifies CLI runner with --postgres executes against live PostgreSQL 17."""
    json_out = tmp_path / "pg_report.json"
    md_out = tmp_path / "pg_report.md"

    exit_code = main(
        [
            "--postgres",
            "--output-json",
            str(json_out),
            "--output-md",
            str(md_out),
            "--fail-under-threshold",
        ]
    )
    assert exit_code == 0
    assert json_out.exists()
    data = json.loads(json_out.read_text(encoding="utf-8"))
    assert data["all_passed"] is True
    assert data["metrics"]["rbac_zero_leakage_rate"]["passed"] is True
    assert data["metrics"]["prompt_injection_block_rate"]["passed"] is True
    # Gate 4 should report genuine candidate exploration > 80% on live PG17 corpus
    m4 = data["metrics"]["frontier_token_savings_ratio"]
    assert m4["passed"] is True
    assert m4["measured_value"] >= 80.0


def test_cli_runner_in_memory_flag(tmp_path: Path) -> None:
    """Verifies CLI runner with --in-memory executes without PostgreSQL requirement."""
    json_out = tmp_path / "inmem_report.json"
    md_out = tmp_path / "inmem_report.md"
    prev_backend = os.environ.get("SCOUT_STORAGE_BACKEND")

    exit_code = main(
        [
            "--in-memory",
            "--output-json",
            str(json_out),
            "--output-md",
            str(md_out),
            "--fail-under-threshold",
        ]
    )
    assert os.environ.get("SCOUT_STORAGE_BACKEND") == prev_backend
    assert exit_code == 0
    assert json_out.exists()
    data = json.loads(json_out.read_text(encoding="utf-8"))
    assert data["all_passed"] is True
    assert data["metrics"]["rbac_zero_leakage_rate"]["passed"] is True
    assert data["metrics"]["prompt_injection_block_rate"]["passed"] is True
    m4 = data["metrics"]["frontier_token_savings_ratio"]
    assert m4["passed"] is True
    assert m4["measured_value"] >= 80.0


def test_cli_runner_mutual_exclusion_error() -> None:
    """Verifies CLI runner rejects simultaneous --postgres and --in-memory flags."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--postgres", "--in-memory"])
    assert exc_info.value.code == 2


def test_harness_alignment_and_registered_domain_citation_lookup(
    harness: EvaluationHarness,
) -> None:
    """Verifies disk code hydration, 1:1 pack alignment, and registered domain docs."""
    from datetime import date
    from uuid import uuid4

    from context_scout.db.schema import DocChunkRecord
    from context_scout.schemas.contracts import (
        AuthorityRules,
        DocCitationSpan,
        DomainPackManifest,
        PreFlightValidationReport,
    )

    # 1. _create_populated_in_memory_repo must match current disk files verbatim (not stale PG17)
    probe_p = PrincipalContext(
        user_id="disc065_probe",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng"],
    )
    disk_files = {
        f.file_path: f
        for f in harness.repo.list_visible_code_files(probe_p, "code_architecture_scout")
    }
    assert "src/context_scout/evals/harness.py" in disk_files
    assert disk_files["src/context_scout/evals/harness.py"].raw_content == (
        harness.repo_root / "src/context_scout/evals/harness.py"
    ).read_text(encoding="utf-8")

    # 2. evaluate_verbatim_citation_faithfulness checks registered non-research domains
    custom_dom = f"eval_disc065_{uuid4().hex[:6]}"
    harness.registry.register(
        DomainPackManifest(
            domain_id=custom_dom,
            display_name="Federated Custom Domain",
            base_model_id="google/gemma-4-26B-A4B-it",
            lora_adapter_id="none",
            lora_adapter_path="domains/enterprise_docs_scout/adapters/gemma4_26b_enterprise_docs",
            mcp_read_tools_allowed=["doc_corpus_mcp:search_hybrid_rrf"],
            authority_rules=AuthorityRules(primary_truth_source="document"),
            output_contract="DomainKnowledgeBrief",
            default_scopes=["public_docs", f"domain:{custom_dom}"],
            lora_enabled=False,
        )
    )
    chunk_id = uuid4()
    quote = "Authoritative federated document quote for verbatim check."
    harness.repo.insert_doc_chunk(
        PrincipalContext(
            user_id="disc065_seed",
            role=RoleType.ENGINEER,
            allowed_scopes=["public_docs", f"domain:{custom_dom}"],
        ),
        DocChunkRecord(
            chunk_id=chunk_id,
            domain_id=custom_dom,
            doc_id="doc-disc065",
            doc_title="Federated Custom Doc",
            source_uri="urn:disc065",
            market_scope=DocumentScope.ENTERPRISE_DOC,
            published_date=date(2026, 11, 19),
            authority_weight=1.0,
            section_heading="Sec",
            contextual_header="Sec",
            verbatim_content=quote,
            metadata_json={},
            allowed_scopes=["public_docs", f"domain:{custom_dom}"],
            embedding=[0.1] * 768,
            source_type="confluence",
        ),
    )
    pack = ImplementationContextPack(
        trace_id="trace-disc065",
        domain_id="code_architecture_scout",
        user_task="federated test",
        architectural_execution_plan=["Inspect federated citation provenance."],
        code_slices=[],
        supporting_doc_citations=[
            DocCitationSpan(
                chunk_id=str(chunk_id),
                doc_id="doc-disc065",
                doc_title="Federated Custom Doc",
                source_uri="urn:disc065",
                section_heading="Sec",
                verbatim_quote=quote,
                full_section_content=quote,
            )
        ],
        preflight_report=PreFlightValidationReport(
            passed=True,
            citations_verified_100_pct=True,
            normalized_rrf_score=0.9,
        ),
        local_tokens_consumed=100,
        pack_token_count=200,
    )
    res = harness.evaluate_verbatim_citation_faithfulness([pack])
    assert res.passed is True
    assert res.measured_value == 100.0
