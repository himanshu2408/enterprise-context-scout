"""Step 2.3 tests: Unified Intent-Aware UX, Source Filtering, and Federated Routing."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from context_scout.agents.citation_gate import DocumentCitationGate
from context_scout.agents.domain_registry import DomainRegistry
from context_scout.agents.intent_router import (
    QueryIntentClassifier,
    QueryIntentMode,
    SourceFilter,
    format_doc_citation_label,
)
from context_scout.agents.scout_workflow import (
    FederatedScoutCoordinator,
    LocalGemmaClient,
    ScoutWorkflow,
)
from context_scout.api.dependencies import (
    get_domain_registry,
    get_local_gemma_client,
    get_repository,
)
from context_scout.api.main import create_app
from context_scout.db.repositories import InMemoryRLSRepository
from context_scout.db.schema import ASTSymbolRecord, CodeFileRecord, DocChunkRecord
from context_scout.schemas.contracts import (
    AuthorityRules,
    DocCitationSpan,
    DomainPackManifest,
    ImplementationContextPack,
    InsufficientContextDiagnostic,
    PrincipalContext,
    RoleType,
)

_SESSION_ID = uuid4().hex[:8]
_ENTERPRISE_DOMAIN_ID = f"enterprise_routing_{_SESSION_ID}"
_DOMAINS_DIR = Path(__file__).resolve().parent.parent / "domains"


def _make_principal(
    role: RoleType,
    allowed_scopes: list[str] | None = None,
) -> PrincipalContext:
    return PrincipalContext(
        user_id=f"user_{role.value}_{_SESSION_ID}",
        role=role,
        allowed_scopes=allowed_scopes or ["public_eng", "platform_docs"],
    )


def _seed_enterprise_corpus() -> tuple[InMemoryRLSRepository, DomainRegistry]:
    """Builds a real InMemoryRLSRepository with code, Confluence, Jira, and restricted docs."""
    repo = InMemoryRLSRepository()
    admin_principal = PrincipalContext(
        user_id=f"seeder_{_SESSION_ID}",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng", "platform_docs", "restricted_exec"],
    )

    token_file_id = uuid4()
    token_code = (
        "from __future__ import annotations\n"
        "\n"
        "\n"
        "def verify_bearer_quota(principal_id: str, limit: int) -> bool:\n"
        '    """Validates bearer token rate limit quota for incoming gateway calls."""\n'
        "    if not principal_id or limit <= 0:\n"
        "        return False\n"
        "    return True\n"
    )
    repo.insert_code_file(
        admin_principal,
        CodeFileRecord(
            file_id=token_file_id,
            domain_id=_ENTERPRISE_DOMAIN_ID,
            repo_name="platform-auth",
            file_path="src/auth/token_service.py",
            git_commit_sha="a" * 40,
            git_blob_sha="b" * 40,
            language="python",
            allowed_scopes=["public_eng"],
            raw_content=token_code,
            line_count=len(token_code.splitlines()),
        ),
    )
    repo.insert_ast_symbol(
        admin_principal,
        ASTSymbolRecord(
            symbol_id=uuid4(),
            file_id=token_file_id,
            domain_id=_ENTERPRISE_DOMAIN_ID,
            qualified_name="verify_bearer_quota",
            symbol_kind="function",
            start_line=4,
            end_line=8,
            signature_skeleton="def verify_bearer_quota(principal_id: str, limit: int) -> bool:",
            contextual_summary=(
                "Validates bearer token rate limit quota in src/auth/token_service.py"
            ),
            allowed_scopes=["public_eng"],
            embedding=[0.1] * 768,
        ),
    )

    wiki_runbook_chunk = DocChunkRecord(
        chunk_id=uuid4(),
        domain_id=_ENTERPRISE_DOMAIN_ID,
        doc_id="40123",
        doc_title="Incident Escalation Runbook",
        source_uri="https://confluence.example.com/wiki/spaces/SEC/pages/40123",
        section_heading="P1 Escalation SLA",
        market_scope=None,
        contextual_header="Incident Escalation Runbook > P1 Escalation SLA",
        verbatim_content=(
            "All P1 security incidents require primary on-call triage within 15 minutes "
            "and formal escalation to the incident commander per onboarding SLA policy."
        ),
        authority_weight=1.0,
        source_type="confluence_runbook",
        version_str="7",
        published_date=date(2026, 10, 15),
        allowed_scopes=["public_eng", "platform_docs"],
        embedding=[0.2] * 768,
        metadata_json={"page_id": "40123", "version": 7, "space_key": "SEC"},
    )
    wiki_hybrid_chunk = DocChunkRecord(
        chunk_id=uuid4(),
        domain_id=_ENTERPRISE_DOMAIN_ID,
        doc_id="40124",
        doc_title="ADR-014 Gateway Rate Limiting",
        source_uri="https://confluence.example.com/wiki/spaces/PLAT/pages/40124",
        section_heading="Bearer Quota Rules",
        market_scope=None,
        contextual_header="ADR-014 Gateway Rate Limiting > Bearer Quota Rules",
        verbatim_content=(
            "Gateway services enforce per-principal bearer token quota checks via "
            "verify_bearer_quota before forwarding authenticated requests upstream."
        ),
        authority_weight=0.95,
        source_type="confluence_page",
        version_str="3",
        published_date=date(2026, 10, 10),
        allowed_scopes=["public_eng", "platform_docs"],
        embedding=[0.2] * 768,
        metadata_json={"page_id": "40124", "version": 3, "space_key": "PLAT"},
    )
    jira_chunk = DocChunkRecord(
        chunk_id=uuid4(),
        domain_id=_ENTERPRISE_DOMAIN_ID,
        doc_id="PLAT-402",
        doc_title="PLAT-402 Rotate Regional Gateway Keys",
        source_uri="https://jira.example.com/browse/PLAT-402",
        section_heading="Escalation Checklist",
        market_scope=None,
        contextual_header="PLAT-402 Rotate Regional Gateway Keys > Escalation Checklist",
        verbatim_content=(
            "PLAT-402 tracks the mandatory 24-hour onboarding checklist for rotating "
            "regional gateway signing keys after an escalation drill."
        ),
        authority_weight=0.9,
        source_type="jira",
        version_str="2",
        published_date=date(2026, 10, 20),
        allowed_scopes=["public_eng", "platform_docs"],
        embedding=[0.2] * 768,
        metadata_json={"ticket_key": "PLAT-402", "resolution": "Done", "space_key": "PLAT"},
    )
    restricted_chunk = DocChunkRecord(
        chunk_id=uuid4(),
        domain_id=_ENTERPRISE_DOMAIN_ID,
        doc_id="99991",
        doc_title="Executive Restricted M&A Memo",
        source_uri="https://confluence.example.com/wiki/spaces/EXEC/pages/99991",
        section_heading="Confidential Terms",
        market_scope=None,
        contextual_header="Executive Restricted M&A Memo > Confidential Terms",
        verbatim_content=(
            "RESTRICTED_ALPHA_CODENAME merger valuation escalation instructions are restricted "
            "to executive principals only."
        ),
        authority_weight=1.0,
        source_type="confluence_page",
        version_str="1",
        published_date=date(2026, 10, 18),
        allowed_scopes=["restricted_exec"],
        embedding=[0.2] * 768,
        metadata_json={"page_id": "99991", "version": 1, "space_key": "EXEC"},
    )
    repo.insert_doc_chunks_batch(
        admin_principal,
        [wiki_runbook_chunk, wiki_hybrid_chunk, jira_chunk, restricted_chunk],
    )

    registry = DomainRegistry(_DOMAINS_DIR)
    registry.load_all()
    registry.register(
        DomainPackManifest(
            domain_id=_ENTERPRISE_DOMAIN_ID,
            display_name="Enterprise Routing Domain",
            base_model_id="google/gemma-4-26B-A4B-it",
            lora_adapter_id="enterprise_routing_lora",
            lora_adapter_path="adapters/enterprise_routing",
            mcp_read_tools_allowed=["read_code_slice", "search_docs"],
            authority_rules=AuthorityRules(
                primary_truth_source="document",
                freshness_half_life_days=180,
            ),
            output_contract="ImplementationContextPack",
        )
    )
    return repo, registry


def _seed_federated_code_only_repo() -> tuple[InMemoryRLSRepository, DomainRegistry]:
    """Builds federated repo where code exists in trading_code_scout and research docs are empty."""
    repo = InMemoryRLSRepository()
    admin = PrincipalContext(
        user_id=f"fed_seeder_{_SESSION_ID}",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng"],
    )
    file_id = uuid4()
    code_src = (
        "def compute_execution_slice(notional: float, participation: float) -> float:\n"
        '    """Computes deterministic execution slice size."""\n'
        "    if notional <= 0.0 or participation <= 0.0:\n"
        "        return 0.0\n"
        "    return notional * participation\n"
    )
    repo.insert_code_file(
        admin,
        CodeFileRecord(
            file_id=file_id,
            domain_id="trading_code_scout",
            repo_name="exec-core",
            file_path="src/exec/compute_slice.py",
            git_commit_sha="c" * 40,
            git_blob_sha="d" * 40,
            language="python",
            allowed_scopes=["public_eng"],
            raw_content=code_src,
            line_count=len(code_src.splitlines()),
        ),
    )
    repo.insert_ast_symbol(
        admin,
        ASTSymbolRecord(
            symbol_id=uuid4(),
            file_id=file_id,
            domain_id="trading_code_scout",
            qualified_name="compute_execution_slice",
            symbol_kind="function",
            start_line=1,
            end_line=5,
            signature_skeleton=(
                "def compute_execution_slice(notional: float, participation: float) -> float:"
            ),
            contextual_summary="Computes execution slice size in src/exec/compute_slice.py",
            allowed_scopes=["public_eng"],
            embedding=[0.1] * 768,
        ),
    )
    registry = DomainRegistry(_DOMAINS_DIR)
    registry.load_all()
    return repo, registry


def test_intent_classifier_routes_by_query_cues_and_source_filter() -> None:
    pure_code = QueryIntentClassifier.classify(
        "Refactor verify_bearer_quota in src/auth/token_service.py"
    )
    assert pure_code.mode == QueryIntentMode.PURE_CODE
    assert pure_code.needs_code is True
    assert pure_code.needs_docs is False

    pure_doc = QueryIntentClassifier.classify(
        "What is the P1 incident escalation SLA and onboarding policy?"
    )
    assert pure_doc.mode == QueryIntentMode.PURE_DOCUMENTATION
    assert pure_doc.needs_code is False
    assert pure_doc.needs_docs is True

    jira_doc = QueryIntentClassifier.classify("Summarize Jira ticket PLAT-402 escalation checklist")
    assert jira_doc.mode == QueryIntentMode.PURE_DOCUMENTATION
    assert jira_doc.needs_code is False
    assert jira_doc.needs_docs is True

    hybrid = QueryIntentClassifier.classify(
        "Implement the Confluence gateway bearer quota policy in verify_bearer_quota"
    )
    assert hybrid.mode == QueryIntentMode.HYBRID_CODE_AND_DOCS
    assert hybrid.needs_code is True
    assert hybrid.needs_docs is True

    forced_wiki = QueryIntentClassifier.classify(
        "Inspect verify_bearer_quota in token_service.py",
        source_filter="wiki",
    )
    assert forced_wiki.mode == QueryIntentMode.PURE_DOCUMENTATION
    assert forced_wiki.needs_code is False

    forced_code = QueryIntentClassifier.classify(
        "What is the P1 incident escalation SLA?",
        source_filter="code",
    )
    assert forced_code.mode == QueryIntentMode.PURE_CODE
    assert forced_code.needs_docs is False


def test_format_doc_citation_label_idempotent_confluence_and_jira() -> None:
    span = DocCitationSpan(
        chunk_id=str(uuid4()),
        doc_id="40123",
        doc_title="Incident Escalation Runbook",
        source_uri="https://confluence.example.com/wiki/spaces/SEC/pages/40123",
        section_heading="P1 Escalation SLA",
        verbatim_quote=(
            "All P1 security incidents require primary on-call triage within 15 minutes."
        ),
        version="7",
        page_id="40123",
    )
    rendered = format_doc_citation_label(
        span,
        metadata={"space_key": "SEC"},
        published_date=date(2026, 10, 15),
    )
    assert rendered == (
        "[Incident Escalation Runbook > P1 Escalation SLA | Space: SEC | Last Updated: 2026-10-15]"
    )
    already_formatted = span.model_copy(update={"section_heading": rendered[1:-1]})
    assert (
        format_doc_citation_label(
            already_formatted,
            metadata={"space_key": "SEC"},
            published_date=date(2026, 10, 15),
        )
        == rendered
    )


@pytest.mark.parametrize(
    "role",
    [
        RoleType.ENGINEER,
        RoleType.PRODUCT_MANAGER,
        RoleType.COMPLIANCE_OFFICER,
        RoleType.QUANT_RESEARCHER,
    ],
)
def test_all_roles_x_three_intents_no_role_bias_and_verified_packs(role: RoleType) -> None:
    repo, registry = _seed_enterprise_corpus()
    workflow = ScoutWorkflow(
        registry=registry,
        repo=repo,
        local_client=LocalGemmaClient(offline_mode=True),
    )
    principal = _make_principal(role)

    # 1. PURE_DOCUMENTATION: all 4 roles (including ENGINEER) MUST get code_slices == []
    doc_result = workflow.run(
        user_prompt="What is the P1 incident escalation SLA in the onboarding runbook?",
        principal=principal,
        domain_id=_ENTERPRISE_DOMAIN_ID,
    )
    assert isinstance(doc_result, ImplementationContextPack), doc_result
    assert doc_result.code_slices == [], (
        f"Role {role.value} unexpectedly received code slices on PURE_DOCUMENTATION"
    )
    assert len(doc_result.supporting_doc_citations) >= 1
    assert doc_result.domain_knowledge_brief is not None
    assert "P1 Escalation SLA" in doc_result.supporting_doc_citations[0].section_heading
    assert "Space: SEC" in doc_result.supporting_doc_citations[0].section_heading
    assert "Last Updated: 2026-10-15" in doc_result.supporting_doc_citations[0].section_heading

    gate = DocumentCitationGate()
    visible_chunks = repo.list_visible_doc_chunks(principal, _ENTERPRISE_DOMAIN_ID)
    cit_result = gate.verify_citations(doc_result.supporting_doc_citations, visible_chunks)
    assert cit_result.passed, cit_result.reason

    # 2. PURE_CODE: all 4 roles receive verified VerbatimCodeSlice without doc clutter
    code_result = workflow.run(
        user_prompt="Inspect verify_bearer_quota in src/auth/token_service.py",
        principal=principal,
        domain_id=_ENTERPRISE_DOMAIN_ID,
    )
    assert isinstance(code_result, ImplementationContextPack), code_result
    assert len(code_result.code_slices) >= 1
    assert any(s.file_path == "src/auth/token_service.py" for s in code_result.code_slices)
    assert code_result.supporting_doc_citations == []

    # 3. HYBRID_CODE_AND_DOCS: all 4 roles receive BOTH code slices AND verified doc citations
    hybrid_result = workflow.run(
        user_prompt=(
            "Check verify_bearer_quota in src/auth/token_service.py against the "
            "gateway Confluence policy"
        ),
        principal=principal,
        domain_id=_ENTERPRISE_DOMAIN_ID,
    )
    assert isinstance(hybrid_result, ImplementationContextPack), hybrid_result
    assert len(hybrid_result.code_slices) >= 1
    assert len(hybrid_result.supporting_doc_citations) >= 1
    assert hybrid_result.domain_knowledge_brief is not None


def test_federated_coordinator_pure_code_succeeds_when_quant_brief_is_none() -> None:
    repo, registry = _seed_federated_code_only_repo()
    coordinator = FederatedScoutCoordinator(
        registry=registry,
        repo=repo,
        local_client=LocalGemmaClient(offline_mode=True),
    )
    principal = _make_principal(RoleType.ENGINEER, allowed_scopes=["public_eng"])

    result = coordinator.run(
        user_prompt="Inspect compute_execution_slice in src/exec/compute_slice.py",
        principal=principal,
        doc_chunks=[],
    )
    assert isinstance(result, ImplementationContextPack), result
    assert result.quant_strategy_spec is None
    assert len(result.code_slices) >= 1
    assert any(s.file_path == "src/exec/compute_slice.py" for s in result.code_slices)


def test_federated_coordinator_pure_doc_suppresses_code_slices() -> None:
    repo, registry = _seed_federated_code_only_repo()
    admin = PrincipalContext(
        user_id=f"fed_doc_seeder_{_SESSION_ID}",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng"],
    )
    repo.insert_doc_chunk(
        admin,
        DocChunkRecord(
            chunk_id=uuid4(),
            domain_id="quant_research_scout",
            doc_id="55100",
            doc_title="Execution Operations Runbook",
            source_uri="https://confluence.example.com/wiki/spaces/OPS/pages/55100",
            section_heading="Incident Escalation",
            market_scope=None,
            contextual_header="Execution Operations Runbook > Incident Escalation",
            verbatim_content=(
                "Operations incident escalation runbook mandates immediate desk handoff "
                "within 10 minutes of feed stall."
            ),
            authority_weight=1.0,
            source_type="confluence_runbook",
            version_str="4",
            published_date=date(2026, 10, 12),
            allowed_scopes=["public_eng"],
            embedding=[0.2] * 768,
            metadata_json={"page_id": "55100", "version": 4, "space_key": "OPS"},
        ),
    )
    coordinator = FederatedScoutCoordinator(
        registry=registry,
        repo=repo,
        local_client=LocalGemmaClient(offline_mode=True),
    )
    principal = _make_principal(RoleType.ENGINEER, allowed_scopes=["public_eng"])

    result = coordinator.run(
        user_prompt="What is the incident escalation runbook SLA for a feed stall?",
        principal=principal,
    )
    assert isinstance(result, ImplementationContextPack), result
    assert result.code_slices == []
    assert len(result.supporting_doc_citations) >= 1
    assert result.domain_knowledge_brief is not None


def test_source_filter_modes_and_rbac_zero_leaks_and_negative_control() -> None:
    repo, registry = _seed_enterprise_corpus()
    workflow = ScoutWorkflow(
        registry=registry,
        repo=repo,
        local_client=LocalGemmaClient(offline_mode=True),
    )
    principal = _make_principal(RoleType.ENGINEER, allowed_scopes=["public_eng", "platform_docs"])

    wiki_pack = workflow.run(
        user_prompt="What is the onboarding escalation checklist?",
        principal=principal,
        domain_id=_ENTERPRISE_DOMAIN_ID,
        source_filter="wiki",
    )
    assert isinstance(wiki_pack, ImplementationContextPack), wiki_pack
    assert wiki_pack.code_slices == []
    assert all(c.ticket_key is None for c in wiki_pack.supporting_doc_citations)
    assert any(c.page_id == "40123" for c in wiki_pack.supporting_doc_citations)

    jira_pack = workflow.run(
        user_prompt="What is the onboarding escalation checklist?",
        principal=principal,
        domain_id=_ENTERPRISE_DOMAIN_ID,
        source_filter="jira",
    )
    assert isinstance(jira_pack, ImplementationContextPack), jira_pack
    assert jira_pack.code_slices == []
    assert all(c.ticket_key == "PLAT-402" for c in jira_pack.supporting_doc_citations)

    code_pack = workflow.run(
        user_prompt="verify_bearer_quota",
        principal=principal,
        domain_id=_ENTERPRISE_DOMAIN_ID,
        source_filter="code",
    )
    assert isinstance(code_pack, ImplementationContextPack), code_pack
    assert len(code_pack.code_slices) >= 1
    assert code_pack.supporting_doc_citations == []

    # RBAC zero-leak check: restricted_exec doc must NEVER appear for public_eng/platform_docs
    rbac_diag = workflow.run(
        user_prompt="What does RESTRICTED_ALPHA_CODENAME merger memo say?",
        principal=principal,
        domain_id=_ENTERPRISE_DOMAIN_ID,
    )
    assert isinstance(rbac_diag, InsufficientContextDiagnostic)

    # Negative control: querying a wiki-only topic under source_filter='jira' MUST abstain
    neg_diag = workflow.run(
        user_prompt="What is the P1 security triage rule for 15 minutes?",
        principal=principal,
        domain_id=_ENTERPRISE_DOMAIN_ID,
        source_filter="jira",
    )
    assert isinstance(neg_diag, InsufficientContextDiagnostic)


def test_fastapi_query_and_stream_accept_source_filter() -> None:
    repo, registry = _seed_enterprise_corpus()
    app = create_app()
    app.dependency_overrides[get_repository] = lambda: repo
    app.dependency_overrides[get_domain_registry] = lambda: registry
    app.dependency_overrides[get_local_gemma_client] = lambda: LocalGemmaClient(offline_mode=True)

    try:
        client = TestClient(app)
        token_resp = client.post(
            "/api/v1/auth/session",
            json={
                "user_id": f"api_eng_{_SESSION_ID}",
                "role": "engineer",
                "allowed_scopes": ["public_eng", "platform_docs"],
                "max_session_paid_tokens": 50000,
            },
        )
        assert token_resp.status_code == 200
        token = token_resp.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        modes: list[SourceFilter] = ["wiki", "jira", "code"]
        for mode in modes:
            query_text = (
                "verify_bearer_quota"
                if mode == "code"
                else "What is the onboarding escalation checklist?"
            )
            resp = client.post(
                "/api/v1/scout/query",
                headers=headers,
                json={
                    "query": query_text,
                    "domain_id": _ENTERPRISE_DOMAIN_ID,
                    "source_filter": mode,
                },
            )
            assert resp.status_code == 200, resp.text
            payload = resp.json()
            assert payload["passed"] is True, payload
            pack = payload["context_pack"]
            if mode == "code":
                assert len(pack["code_slices"]) >= 1
                assert pack["supporting_doc_citations"] == []
            else:
                assert pack["code_slices"] == []
                assert len(pack["supporting_doc_citations"]) >= 1
                if mode == "jira":
                    assert all(
                        c["ticket_key"] == "PLAT-402" for c in pack["supporting_doc_citations"]
                    )
                else:
                    assert all(c["ticket_key"] is None for c in pack["supporting_doc_citations"])

        with client.stream(
            "POST",
            "/api/v1/scout/stream",
            headers=headers,
            json={
                "query": "What is the onboarding escalation checklist?",
                "domain_id": _ENTERPRISE_DOMAIN_ID,
                "source_filter": "jira",
            },
        ) as stream_resp:
            assert stream_resp.status_code == 200
            body = "".join(stream_resp.iter_text())
            assert "context_pack_ready" in body
            assert "PLAT-402" in body
    finally:
        app.dependency_overrides.clear()


def test_scout_workflow_populates_confluence_and_code_vs_doc_conflicts() -> None:
    repo = InMemoryRLSRepository()
    domain_id = f"conflict_routing_{_SESSION_ID}"
    admin = PrincipalContext(
        user_id=f"conflict_seeder_{_SESSION_ID}",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng"],
    )

    code_file_id = uuid4()
    code_src = (
        "def dispatch_gateway_order(endpoint: str = '/api/v2/orders') -> int:\n"
        '    """Live HEAD gateway order handler."""\n'
        "    timeout_sec = 30\n"
        "    return timeout_sec\n"
    )
    repo.insert_code_file(
        admin,
        CodeFileRecord(
            file_id=code_file_id,
            domain_id=domain_id,
            repo_name="gateway-core",
            file_path="src/gateway/orders.py",
            git_commit_sha="e" * 40,
            git_blob_sha="f" * 40,
            language="python",
            allowed_scopes=["public_eng"],
            raw_content=code_src,
            line_count=len(code_src.splitlines()),
        ),
    )
    repo.insert_ast_symbol(
        admin,
        ASTSymbolRecord(
            symbol_id=uuid4(),
            file_id=code_file_id,
            domain_id=domain_id,
            qualified_name="dispatch_gateway_order",
            symbol_kind="function",
            start_line=1,
            end_line=4,
            signature_skeleton=(
                "def dispatch_gateway_order(endpoint: str = '/api/v2/orders') -> int:"
            ),
            contextual_summary=(
                "Dispatches gateway order requests with timeout in src/gateway/orders.py"
            ),
            allowed_scopes=["public_eng"],
            embedding=[0.1] * 768,
        ),
    )

    page_newer = DocChunkRecord(
        chunk_id=uuid4(),
        domain_id=domain_id,
        doc_id="confluence:7001",
        doc_title="Gateway Order Timeout Spec",
        source_uri="https://confluence.example.com/wiki/spaces/PLAT/pages/7001",
        section_heading="Timeout Policy",
        market_scope=None,
        contextual_header="Gateway Order Timeout Spec > Timeout Policy",
        verbatim_content=(
            "Gateway dispatch_gateway_order policy specifies timeout = 15s and "
            "calls /api/v1/orders for upstream dispatch."
        ),
        authority_weight=1.0,
        source_type="confluence",
        version_str="4",
        published_date=date(2026, 10, 20),
        allowed_scopes=["public_eng"],
        embedding=[0.2] * 768,
        metadata_json={"page_id": "7001", "version": 4, "space_key": "PLAT"},
    )
    page_older = DocChunkRecord(
        chunk_id=uuid4(),
        domain_id=domain_id,
        doc_id="confluence:7002",
        doc_title="Legacy Gateway Timeout Runbook",
        source_uri="https://confluence.example.com/wiki/spaces/OPS/pages/7002",
        section_heading="Timeout Policy",
        market_scope=None,
        contextual_header="Legacy Gateway Timeout Runbook > Timeout Policy",
        verbatim_content=(
            "Legacy gateway order runbook documents timeout = 5s for upstream requests."
        ),
        authority_weight=0.8,
        source_type="confluence",
        version_str="2",
        published_date=date(2026, 9, 1),
        allowed_scopes=["public_eng"],
        embedding=[0.2] * 768,
        metadata_json={"page_id": "7002", "version": 2, "space_key": "OPS"},
    )
    repo.insert_doc_chunks_batch(admin, [page_newer, page_older])

    registry = DomainRegistry(_DOMAINS_DIR)
    registry.load_all()
    registry.register(
        DomainPackManifest(
            domain_id=domain_id,
            display_name="Conflict Routing Domain",
            base_model_id="google/gemma-4-26B-A4B-it",
            lora_adapter_id="conflict_routing_lora",
            lora_adapter_path="adapters/conflict_routing",
            mcp_read_tools_allowed=["read_code_slice", "search_docs"],
            authority_rules=AuthorityRules(
                primary_truth_source="code",
                freshness_half_life_days=180,
            ),
            output_contract="ImplementationContextPack",
        )
    )

    workflow = ScoutWorkflow(
        registry=registry,
        repo=repo,
        local_client=LocalGemmaClient(offline_mode=True),
    )
    principal = _make_principal(RoleType.ENGINEER, allowed_scopes=["public_eng"])

    hybrid_pack = workflow.run(
        user_prompt=(
            "Inspect dispatch_gateway_order in src/gateway/orders.py against the timeout runbook"
        ),
        principal=principal,
        domain_id=domain_id,
    )
    assert isinstance(hybrid_pack, ImplementationContextPack), hybrid_pack
    conflicts = hybrid_pack.doc_vs_code_conflicts_detected
    assert len(conflicts) == len(dict.fromkeys(conflicts))
    assert any("CONFLUENCE_CONFIG_CONFLICT (timeout)" in item for item in conflicts), conflicts
    assert any("CODE_VS_DOC_CONFLICT (timeout)" in item for item in conflicts), conflicts
    assert any("CODE_VS_DOC_ENDPOINT_DRIFT (orders)" in item for item in conflicts), conflicts

    # PURE_DOCUMENTATION flags Confluence-vs-Confluence conflict even with code_slices=[]
    doc_pack = workflow.run(
        user_prompt="What is the gateway order timeout runbook policy?",
        principal=principal,
        domain_id=domain_id,
    )
    assert isinstance(doc_pack, ImplementationContextPack), doc_pack
    assert doc_pack.code_slices == []
    assert any(
        "CONFLUENCE_CONFIG_CONFLICT (timeout)" in item
        for item in doc_pack.doc_vs_code_conflicts_detected
    )


def test_federated_coordinator_populates_code_vs_doc_conflicts() -> None:
    repo = InMemoryRLSRepository()
    admin = PrincipalContext(
        user_id=f"fed_conflict_seeder_{_SESSION_ID}",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng"],
    )
    code_file_id = uuid4()
    code_src = (
        "def submit_batch_order(path: str = '/api/v2/batch') -> int:\n"
        '    """Live HEAD batch order dispatcher."""\n'
        "    timeout_sec = 45\n"
        "    return timeout_sec\n"
    )
    repo.insert_code_file(
        admin,
        CodeFileRecord(
            file_id=code_file_id,
            domain_id="trading_code_scout",
            repo_name="exec-core",
            file_path="src/exec/batch_order.py",
            git_commit_sha="1" * 40,
            git_blob_sha="2" * 40,
            language="python",
            allowed_scopes=["public_eng"],
            raw_content=code_src,
            line_count=len(code_src.splitlines()),
        ),
    )
    repo.insert_ast_symbol(
        admin,
        ASTSymbolRecord(
            symbol_id=uuid4(),
            file_id=code_file_id,
            domain_id="trading_code_scout",
            qualified_name="submit_batch_order",
            symbol_kind="function",
            start_line=1,
            end_line=4,
            signature_skeleton="def submit_batch_order(path: str = '/api/v2/batch') -> int:",
            contextual_summary="Submits batch order with timeout in src/exec/batch_order.py",
            allowed_scopes=["public_eng"],
            embedding=[0.1] * 768,
        ),
    )
    doc_chunk = DocChunkRecord(
        chunk_id=uuid4(),
        domain_id="quant_research_scout",
        doc_id="confluence:8801",
        doc_title="Batch Execution Architecture Spec",
        source_uri="https://confluence.example.com/wiki/spaces/EXEC/pages/8801",
        section_heading="Batch Timeout",
        market_scope=None,
        contextual_header="Batch Execution Architecture Spec > Batch Timeout",
        verbatim_content=(
            "Batch execution submit_batch_order architecture spec requires timeout = 10s "
            "and targets /api/v1/batch on the gateway."
        ),
        authority_weight=1.0,
        source_type="confluence",
        version_str="3",
        published_date=date(2026, 10, 18),
        allowed_scopes=["public_eng"],
        embedding=[0.2] * 768,
        metadata_json={"page_id": "8801", "version": 3, "space_key": "EXEC"},
    )
    repo.insert_doc_chunk(admin, doc_chunk)

    registry = DomainRegistry(_DOMAINS_DIR)
    registry.load_all()
    coordinator = FederatedScoutCoordinator(
        registry=registry,
        repo=repo,
        local_client=LocalGemmaClient(offline_mode=True),
    )
    principal = _make_principal(RoleType.ENGINEER, allowed_scopes=["public_eng"])

    result = coordinator.run(
        user_prompt=(
            "Inspect submit_batch_order in src/exec/batch_order.py against the "
            "batch architecture spec"
        ),
        principal=principal,
    )
    assert isinstance(result, ImplementationContextPack), result
    assert len(result.code_slices) >= 1
    assert any(
        "CODE_VS_DOC_CONFLICT (timeout)" in item for item in result.doc_vs_code_conflicts_detected
    ), result.doc_vs_code_conflicts_detected
    assert any(
        "CODE_VS_DOC_ENDPOINT_DRIFT (batch)" in item
        for item in result.doc_vs_code_conflicts_detected
    ), result.doc_vs_code_conflicts_detected
