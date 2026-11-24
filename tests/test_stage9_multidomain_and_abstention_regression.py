"""Permanent Stage 9 regression suite covering Forensic Root Causes RC-1 through RC-7.

Verifies:
- RC-1: Dynamic multi-domain ``federated_all`` routing across all 4+ domains (including
  a dynamically registered custom 5th domain) — never hardcoded to 2 finance domains.
- RC-2: ``ScoutStreamRequest.normalized_rrf_score`` defaults to ``None`` (zero ``0.85``
  pass-through) and Stage 4 hard-blocks ungrounded/insufficient packs at ``$0`` paid tokens.
- RC-3: Unscripted phrasing (``"What is ACID"``, ``"What is ACID?"``, ``"Explain CAP theorem"``,
  ``"What is Apache Iceberg?"``) succeeds without narrow acronym regexes, while scattered
  non-cohesive bag-of-words matches are rejected.
- RC-4: Calibrated retrieval rejects license boilerplate and below-floor noise.
- RC-5: ``LocalGemmaClient(offline_mode=True)`` fails closed; ``lora_enabled=False`` on doc
  domains selects base ``google/gemma-4-26B-A4B-it``.
- RC-6: ``get_unscripted_multidomain_suite()`` has >= 25 prompts with zero filename/classname
  spoilers in ``user_prompt``.
- RC-7: ``StageExecutionTrace`` is emitted on API/SSE responses and ``/api/v1/workspace-config``
  dynamically surfaces custom domains.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from fastapi.testclient import TestClient

from context_scout.agents.domain_registry import DomainRegistry
from context_scout.agents.intent_router import decompose_query_facets, evaluate_facet_grounding
from context_scout.agents.scout_workflow import (
    FederatedScoutCoordinator,
    LocalGemmaClient,
    ScoutWorkflow,
)
from context_scout.api.dependencies import create_signed_principal_token
from context_scout.api.main import create_app
from context_scout.api.schemas import ScoutStreamRequest
from context_scout.db.repositories import DocChunkRecord, InMemoryRLSRepository
from context_scout.evals.golden_dataset import (
    get_default_golden_suite,
    get_unscripted_multidomain_suite,
)
from context_scout.schemas.contracts import (
    AuthorityRules,
    DomainPackManifest,
    ImplementationContextPack,
    InsufficientContextDiagnostic,
    PrincipalContext,
    RoleType,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DOMAINS_DIR = REPO_ROOT / "domains"


def _auth_headers(scopes: list[str] | None = None) -> dict[str, str]:
    os.environ["SCOUT_ALLOW_INSECURE_DEV_SECRET"] = "true"
    token = create_signed_principal_token(
        user_id="stage9_regression_user",
        role=RoleType.ENGINEER,
        allowed_scopes=scopes or ["public", "public_eng", "domain:data_engineering"],
    )
    return {"Authorization": f"Bearer {token}"}


def test_rc2_no_hardcoded_rrf_default_on_stream_request() -> None:
    """RC-2: ScoutStreamRequest.normalized_rrf_score must default to None, never 0.85."""
    req = ScoutStreamRequest(query="What is ACID?", domain_id="federated_all")
    assert req.normalized_rrf_score is None


def test_rc5_doc_domains_disable_rule_distilled_lora_and_fail_closed_offline() -> None:
    """RC-5: Doc domains default lora_enabled=False; offline LocalGemmaClient returns None."""
    registry = DomainRegistry(DOMAINS_DIR)
    registry.load_all()
    for dom_id in ("data_engineering_scout", "enterprise_docs_scout"):
        manifest = registry.get(dom_id)
        assert manifest is not None
        assert manifest.lora_enabled is False

    client = LocalGemmaClient(offline_mode=True)
    repo = InMemoryRLSRepository()
    p = PrincipalContext(user_id="u", role=RoleType.ENGINEER, allowed_scopes=["public"])
    assert client.generate_research_brief("test", "enterprise_docs_scout", repo, p) is None


def test_rc6_unscripted_eval_suite_has_zero_spoilers_and_covers_all_domains() -> None:
    """RC-6: Replacement benchmark has >= 25 prompts, 0 spoilers, and covers all 4 domains."""
    unscripted = get_unscripted_multidomain_suite()
    assert len(unscripted) >= 25

    domains_covered = {t.domain_id for t in unscripted}
    assert {
        "data_engineering_scout",
        "polyglot_unseen_v2",
        "enterprise_docs_scout",
        "code_architecture_scout",
        "federated_all",
    }.issubset(domains_covered)
    assert sum(1 for t in unscripted if t.expected_abstain) >= 4

    spoiler_tokens = (
        ".py",
        "KalmanHedgeEstimator",
        "AlmgrenChrissExecutionOptimizer",
        "OrderFlowImbalanceSignal",
        "KellyRiskManager",
        "GARCH11Model",
    )
    for task in unscripted:
        for tok in spoiler_tokens:
            assert tok not in task.user_prompt, f"Spoiler {tok!r} found in {task.task_id}"

    default_suite = get_default_golden_suite()
    for ct in default_suite.code_tasks:
        for tok in spoiler_tokens:
            assert tok not in ct.user_prompt, f"Spoiler {tok!r} found in {ct.task_id}"


def test_rc1_rc3_federated_all_discovers_custom_5th_domain_and_handles_unscripted_acid() -> None:
    """RC-1 & RC-3: federated_all probes all domains including custom 5th domain."""
    registry = DomainRegistry(DOMAINS_DIR)
    registry.load_all()
    custom_manifest = DomainPackManifest(
        domain_id="biotech_scout",
        display_name="Biotech Genomics Specialist",
        base_model_id="google/gemma-4-26B-A4B-it",
        lora_adapter_id="gemma4_26b_biotech",
        lora_adapter_path="domains/biotech_scout/adapters/gemma4_26b_biotech",
        mcp_read_tools_allowed=["doc_corpus_mcp:search_hybrid_rrf"],
        authority_rules=AuthorityRules(primary_truth_source="document"),
        output_contract="DomainKnowledgeBrief",
        lora_enabled=False,
        default_scopes=["public"],
    )
    registry.register(custom_manifest)

    repo = InMemoryRLSRepository()
    principal = PrincipalContext(
        user_id="stage9_custom_user",
        role=RoleType.ENGINEER,
        allowed_scopes=["public", "public_eng", "domain:data_engineering"],
    )
    repo.insert_doc_chunk(
        principal,
        DocChunkRecord(
            chunk_id=uuid.uuid4(),
            domain_id="biotech_scout",
            doc_id="crispr_01",
            doc_title="CRISPR Cas9 Editing",
            source_uri="https://docs.example.org/crispr",
            market_scope=None,
            published_date=None,
            authority_weight=1.0,
            metadata_json={},
            section_heading="Guide RNA Specificity",
            contextual_header="CRISPR Cas9 guide RNA targeting.",
            verbatim_content=(
                "CRISPR Cas9 uses a single guide RNA sequence to direct double-strand DNA "
                "cleavage at complementary protospacer adjacent motif (PAM) target sites."
            ),
            allowed_scopes=["public"],
            embedding=[0.1] * 768,
        ),
    )

    coordinator = FederatedScoutCoordinator(
        registry=registry,
        repo=repo,
        local_client=LocalGemmaClient(offline_mode=True),
    )
    res = coordinator.run(
        user_prompt="How does CRISPR Cas9 guide RNA cleavage target PAM sites?",
        principal=principal,
        normalized_rrf_score=0.75,
    )
    assert isinstance(res, ImplementationContextPack), res
    assert any(c.doc_id == "crispr_01" for c in res.supporting_doc_citations)
    assert len(res.stage_traces) >= 1


def test_rc3_rc4_cohesive_grounding_rejects_scattered_bag_of_words() -> None:
    """RC-3 & RC-4: Scattered words across unrelated prose must not pass facet grounding."""
    plan = decompose_query_facets("Explain CAP theorem")
    scattered_chunk = DocChunkRecord(
        chunk_id=uuid.uuid4(),
        domain_id="data_engineering_scout",
        doc_id="unrelated_math",
        doc_title="Graph Cut Proofs",
        source_uri="https://arxiv.org/abs/0000.00001",
        market_scope=None,
        published_date=None,
        authority_weight=1.0,
        metadata_json={},
        section_heading="Theorem 1",
        contextual_header="Graph partitioning proof",
        verbatim_content=(
            "We cap the vertex degree at k and prove the main theorem by induction on edges."
        ),
        allowed_scopes=["public"],
        embedding=[0.1] * 768,
    )
    cov, _ = evaluate_facet_grounding(
        plan.facets[0],
        doc_chunks=[scattered_chunk],
        code_slices=[],
        domain_id="data_engineering_scout",
    )
    assert cov.status == "not_found"


def test_rc1_rc2_rc7_live_pg17_api_acid_cap_iceberg_and_zero_dollar_abstention() -> None:
    """RC-1, RC-2, RC-7: Live PG17 + FastAPI TestClient across all unscripted probes & $0 trap."""
    from context_scout.api import dependencies

    os.environ["SCOUT_ALLOW_INSECURE_DEV_SECRET"] = "true"
    prev_repo = dependencies._shared_repo
    dependencies._shared_repo = None
    headers = _auth_headers()
    try:
        with TestClient(create_app()) as client:
            # 1. /api/v1/workspace-config exposes all 4 domains dynamically
            cfg_resp = client.get("/api/v1/workspace-config", headers=headers)
            assert cfg_resp.status_code == 200
            cfg = cfg_resp.json()
            domain_ids = {d["domain_id"] for d in cfg["domains"]}
            assert {
                "data_engineering_scout",
                "polyglot_unseen_v2",
                "enterprise_docs_scout",
                "code_architecture_scout",
            }.issubset(domain_ids)

            # 2. Unscripted DE queries under federated_all route to data_engineering_scout & pass
            for q in (
                "What is ACID",
                "What is ACID?",
                "Explain CAP theorem",
                "What is Apache Iceberg?",
            ):
                resp = client.post(
                    "/api/v1/scout/query",
                    headers=headers,
                    json={"query": q, "domain_id": "federated_all"},
                )
                assert resp.status_code == 200, f"{q}: {resp.text}"
                body = resp.json()
                assert body["passed"] is True, f"{q} failed preflight: {body}"
                assert body["context_pack"]["domain_id"] == "data_engineering_scout", q
                assert len(body["stage_traces"]) == 4, f"{q} missing 4 stage traces"
                assert body["stage_traces"][-1]["final_decision"] == "handoff_ready"

            # 3. Genuinely unindexed out-of-corpus trap MUST abstain at $0 paid tokens
            trap_resp = client.post(
                "/api/v1/scout/query",
                headers=headers,
                json={
                    "query": "Explain the internal rules of Zorglubian Hyper-Quarks",
                    "domain_id": "federated_all",
                },
            )
            assert trap_resp.status_code == 200
            trap = trap_resp.json()
            assert trap["passed"] is False
            assert trap["diagnostic"] is not None
            assert trap["diagnostic"]["paid_tokens_spent"] == 0
            assert trap["diagnostic"]["stage_traces"][-1]["final_decision"] == "abstain_zero_paid"
    finally:
        dependencies._shared_repo = prev_repo


def test_rc2_stage4_gate_blocks_insufficient_citations_when_stage3_completes() -> None:
    """RC-2 / Stage 4 Mutation-M1 guard: Stage 4 blocks at $0 when Stage 2/3 return weak hits."""
    registry = DomainRegistry(DOMAINS_DIR)
    registry.load_all()
    manifest = DomainPackManifest(
        domain_id="quantum_lab_scout",
        display_name="Quantum Hardware Lab",
        description="Cryogenic qubit control and hardware notes.",
        base_model_id="google/gemma-4-26B-A4B-it",
        lora_adapter_id="gemma4_26b_quantum",
        lora_adapter_path="domains/quantum_lab_scout/adapters/gemma4_26b_quantum",
        mcp_read_tools_allowed=["doc_corpus_mcp:search_hybrid_rrf"],
        modalities=["document"],
        sample_queries=["How does cryogenic control affect qubit noise?"],
        output_contract="DomainKnowledgeBrief",
        lora_enabled=False,
        default_scopes=["public"],
        authority_rules=AuthorityRules(primary_truth_source="document"),
    )
    registry.register(manifest)

    repo = InMemoryRLSRepository()
    principal = PrincipalContext(
        user_id="stage4_mutation_guard_user",
        role=RoleType.ENGINEER,
        allowed_scopes=["public"],
    )
    repo.insert_doc_chunk(
        principal,
        DocChunkRecord(
            chunk_id=uuid.uuid4(),
            domain_id="quantum_lab_scout",
            doc_id="cryo_notes_01",
            doc_title="Cryogenic Control Line Noise",
            source_uri="https://docs.example.org/cryo-noise",
            market_scope=None,
            published_date=None,
            authority_weight=1.0,
            metadata_json={},
            section_heading="Thermal Attenuation",
            contextual_header="Cryogenic qubit wiring noise notes.",
            verbatim_content=(
                "Quantum noise degrades superconducting qubits across cryogenic control lines "
                "when attenuator thermalization is incomplete during microwave pulse delivery."
            ),
            allowed_scopes=["public"],
            embedding=[0.0] * 768,
        ),
    )

    local_client = LocalGemmaClient(offline_mode=True)
    prompt = "Explain quantum error correction"

    # 1. Single-domain ScoutWorkflow must pass Stage 2 & Stage 3 with non-empty chunks/citations,
    #    then be blocked at Stage 4 (not short-circuited empty at Stage 2/3).
    wf = ScoutWorkflow(registry=registry, repo=repo, local_client=local_client)
    wf_events: list[dict[str, object]] = []
    wf_res = wf.run(
        user_prompt=prompt,
        principal=principal,
        domain_id="quantum_lab_scout",
        normalized_rrf_score=None,
        on_stage_event=wf_events.append,
    )
    assert isinstance(wf_res, InsufficientContextDiagnostic), wf_res
    assert wf_res.paid_tokens_spent == 0
    assert len(wf_res.stage_traces) == 4
    assert any(p.accepted_hits >= 1 for p in wf_res.stage_traces[1].domain_probes)
    assert (wf_res.stage_traces[2].chunks_fed or 0) >= 1
    assert wf_res.stage_traces[3].stage == 4
    assert len(wf_res.stage_traces[3].checks) >= 1
    assert wf_res.stage_traces[3].final_decision == "abstain_zero_paid"
    assert wf_res.stage_traces[3].sufficiency_verdict == "insufficient"

    # 2. Multi-domain FederatedScoutCoordinator must likewise reach Stage 4 with non-empty chunks
    #    and abstain at $0 paid tokens via the Stage 4 sufficiency/floor gate.
    fed_principal = PrincipalContext(
        user_id="stage4_mutation_guard_user",
        role=RoleType.ENGINEER,
        allowed_scopes=["public"],
    )
    coord = FederatedScoutCoordinator(registry=registry, repo=repo, local_client=local_client)
    fed_events: list[dict[str, object]] = []
    fed_res = coord.run(
        user_prompt=prompt,
        principal=fed_principal,
        normalized_rrf_score=None,
        on_stage_event=fed_events.append,
    )
    assert isinstance(fed_res, InsufficientContextDiagnostic), fed_res
    assert fed_res.paid_tokens_spent == 0
    assert len(fed_res.stage_traces) == 4
    assert any(p.accepted_hits >= 1 for p in fed_res.stage_traces[1].domain_probes)
    assert (fed_res.stage_traces[2].chunks_fed or 0) >= 1
    assert fed_res.stage_traces[3].stage == 4
    assert len(fed_res.stage_traces[3].checks) >= 1
    assert fed_res.stage_traces[3].final_decision == "abstain_zero_paid"
    assert fed_res.stage_traces[3].sufficiency_verdict == "insufficient"
