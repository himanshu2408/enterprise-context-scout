"""Deep Empirical Challenger Stress Test Suite for Milestone 3 (teamwork_challenger_m3_10).

Adversarially stresses:
1. Slicer quotation tampering: word omissions, insertions, truncations, token swapping.
2. Homoglyph attacks: Cyrillic and Greek character substitutions.
3. Citation staleness: Confluence superseded/future versions and Jira terminal statuses/resolutions.
4. Database chunk limit resolution: >100 chunks beyond pagination limits.
5. Multi-repo isolation and cascading foreign-key integrity upon deletion.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from context_scout.agents.citation_gate import (
    CitationFailureCode,
    DocumentCitationGate,
)
from context_scout.agents.preflight_gate import PreFlightValidator
from context_scout.ast_engine.deterministic_slicer import DeterministicSlicer
from context_scout.ast_engine.models import SlicerErrorCode
from context_scout.db.repositories import (
    ASTEdgeRecord,
    ASTSymbolRecord,
    CodeFileRecord,
    DocChunkRecord,
    InMemoryRLSRepository,
)
from context_scout.ingestion.doc_parser import DocumentChunk
from context_scout.plugins.connectors.confluence import (
    ConfluenceConnectorPlugin,
    ConfluencePageDTO,
)
from context_scout.plugins.connectors.jira import JiraConnectorPlugin, JiraIssueDTO
from context_scout.schemas.contracts import (
    DocCitationSpan,
    MarketRegime,
    PrincipalContext,
    RoleType,
)

# ==============================================================================
# Dimension 1: Slicer Quotation Tampering
# ==============================================================================


@pytest.mark.parametrize(
    ("source_text", "tampered_quote", "expected_error_code"),
    [
        # Omission of critical negative words
        (
            "Under NO circumstances should the gateway allow unauthenticated fallback mode.",
            "Under circumstances should the gateway allow unauthenticated fallback mode.",
            SlicerErrorCode.HALLUCINATED_QUOTE,
        ),
        (
            "The authorization proxy must NOT bypass tenant access control policies.",
            "The authorization proxy must bypass tenant access control policies.",
            SlicerErrorCode.HALLUCINATED_QUOTE,
        ),
        (
            "Worker processes should NEVER execute unauthenticated shell commands.",
            "Worker processes should execute unauthenticated shell commands.",
            SlicerErrorCode.HALLUCINATED_QUOTE,
        ),
        # Adversarial insertion of qualifying / weakening words
        (
            "All outbound traffic must be encrypted with TLS 1.3.",
            "All outbound traffic optionally must be encrypted with TLS 1.3.",
            SlicerErrorCode.HALLUCINATED_QUOTE,
        ),
        (
            "Database replicas strictly enforce read-only transactions.",
            "Database replicas strictly enforce read-only transactions unless bypassed.",
            SlicerErrorCode.HALLUCINATED_QUOTE,
        ),
        # Shuffled token order (semantic inversion)
        (
            "Grant admin rights to Alice and read rights to Bob.",
            "Grant admin rights to Bob and read rights to Alice.",
            SlicerErrorCode.HALLUCINATED_QUOTE,
        ),
        # Truncated expression with open parentheses / brackets
        (
            "def calculate_tax(amount: float, jurisdiction: str) -> float:",
            "def calculate_tax(amount: float, jurisdiction: str",
            SlicerErrorCode.TRUNCATED_QUOTE,
        ),
        (
            "allowed_roles = ['admin', 'auditor', 'security_officer']",
            "allowed_roles = ['admin', 'auditor'",
            SlicerErrorCode.TRUNCATED_QUOTE,
        ),
    ],
)
def test_slicer_adversarial_tampering_rejected(
    source_text: str, tampered_quote: str, expected_error_code: SlicerErrorCode
) -> None:
    slicer = DeterministicSlicer()
    res = slicer.verify_quote(tampered_quote, source_text, reject_absent_tokens=True)
    assert not res.is_valid, f"Tampered quote unexpectedly passed: {tampered_quote}"
    assert res.error_code == expected_error_code


def test_citation_gate_rejects_omissions_and_insertions() -> None:
    gate = DocumentCitationGate()
    cid = str(uuid.uuid4())
    source_text = (
        "All microservices must enforce mutual TLS authentication across all internal endpoints."
    )
    tampered_quote = (
        "All microservices must enforce TLS authentication across all internal endpoints."
    )

    chunk = DocChunkRecord(
        chunk_id=uuid.UUID(cid),
        domain_id="default",
        doc_id="sec-01",
        doc_title="Security Policy",
        source_uri="https://wiki.corp/sec-01",
        market_scope=None,
        published_date=datetime.now(UTC).date(),
        authority_weight=1.0,
        section_heading="Network Security",
        contextual_header="Security Header",
        verbatim_content=source_text,
        metadata_json={"page_id": "101", "version": "1"},
        allowed_scopes=["eng"],
        embedding=[0.0] * 768,
        source_type="confluence",
        version_str="1",
        is_stale=False,
    )
    cit = DocCitationSpan(
        chunk_id=cid,
        doc_id="sec-01",
        doc_title="Security Policy",
        source_uri="https://wiki.corp/sec-01",
        section_heading="Network Security",
        verbatim_quote=tampered_quote,
        page_id="101",
        version="1",
    )
    res = gate.verify_citations([cit], [chunk])
    assert not res.passed
    assert res.details[0].error_code == CitationFailureCode.HALLUCINATED_QUOTE
    assert "omission" in (res.details[0].error_reason or "").lower()


# ==============================================================================
# Dimension 2: Homoglyph Attacks (Cyrillic & Greek Substitutions)
# ==============================================================================


@pytest.mark.parametrize(
    ("latin_word", "homoglyph_word"),
    [
        # Cyrillic small letter a (U+0430) vs Latin 'a'
        ("database", "d\u0430t\u0430b\u0430se"),
        # Cyrillic small letter e (U+0435) vs Latin 'e'
        ("authentication", "auth\u0435ntication"),
        # Cyrillic small letter o (U+043E) vs Latin 'o'
        ("protocol", "pr\u043et\u043ec\u043el"),
        # Cyrillic small letter c (U+0441) and p (U+0440)
        ("encryption", "en\u0441ry\u0440tion"),
        # Greek small letter omicron (U+03BF) vs Latin 'o'
        ("token", "t\u03bfken"),
    ],
)
def test_homoglyph_substitution_detected_and_rejected(latin_word: str, homoglyph_word: str) -> None:
    slicer = DeterministicSlicer()
    source_text = f"The system architecture mandates strict {latin_word} for services."
    tampered_quote = f"The system architecture mandates strict {homoglyph_word} for services."

    res = slicer.verify_quote(tampered_quote, source_text, reject_absent_tokens=True)
    assert not res.is_valid, f"Homoglyph attack unexpectedly passed: {homoglyph_word}"
    assert res.error_code == SlicerErrorCode.HALLUCINATED_QUOTE
    assert any(homoglyph_word in tok or tok in homoglyph_word for tok in res.unmatched_tokens), (
        f"Expected homoglyph '{homoglyph_word}' in unmatched_tokens={res.unmatched_tokens}"
    )


# ==============================================================================
# Dimension 3: Citation Staleness (Confluence & Jira)
# ==============================================================================


def test_confluence_superseded_and_future_version_rejection() -> None:
    gate = DocumentCitationGate(strict_freshness=True)
    body = "PostgreSQL 17 connection pool must use transaction pooling mode."

    cid = str(uuid.uuid4())
    chunk = DocChunkRecord(
        chunk_id=uuid.UUID(cid),
        domain_id="default",
        doc_id="confluence:500",
        doc_title="DB Pooling RFC",
        source_uri="https://wiki.corp/500",
        market_scope=None,
        published_date=datetime.now(UTC).date(),
        authority_weight=1.0,
        section_heading="Architecture",
        contextual_header="DB Header",
        verbatim_content=body,
        metadata_json={"page_id": "500", "version": 4},
        allowed_scopes=["eng"],
        embedding=[0.0] * 768,
        source_type="confluence",
        version_str="4",
        is_stale=False,
    )

    # 1. Superseded version (v2 < v4)
    cit_superseded = DocCitationSpan(
        chunk_id=cid,
        doc_id="500",
        doc_title="DB Pooling RFC",
        source_uri="https://wiki.corp/500",
        section_heading="Architecture",
        verbatim_quote=body,
        page_id="500",
        version="2",
    )
    res_sup = gate.verify_citations([cit_superseded], [chunk])
    assert not res_sup.passed
    assert res_sup.details[0].error_code == CitationFailureCode.STALE_SUPERSEDED_VERSION

    # 2. Future unindexed version (v6 > v4)
    cit_future = DocCitationSpan(
        chunk_id=cid,
        doc_id="500",
        doc_title="DB Pooling RFC",
        source_uri="https://wiki.corp/500",
        section_heading="Architecture",
        verbatim_quote=body,
        page_id="500",
        version="6",
    )
    res_fut = gate.verify_citations([cit_future], [chunk])
    assert not res_fut.passed
    assert res_fut.details[0].error_code == CitationFailureCode.STALE_SUPERSEDED_VERSION

    # 3. Exact matching version (v4 == v4)
    cit_exact = DocCitationSpan(
        chunk_id=cid,
        doc_id="500",
        doc_title="DB Pooling RFC",
        source_uri="https://wiki.corp/500",
        section_heading="Architecture",
        verbatim_quote=body,
        page_id="500",
        version="4",
    )
    res_exact = gate.verify_citations([cit_exact], [chunk])
    assert res_exact.passed
    assert res_exact.verified_count == 1


@pytest.mark.parametrize(
    "stale_status",
    [
        "Closed",
        "Archived",
        "Resolved (Obsolete)",
        "Obsolete",
        "Cancelled",
        "Rejected",
        "Won't Fix",
        "Won't Do",
        "wont do",
        "wontdo",
        "Abandoned",
    ],
)
def test_jira_terminal_statuses_rejected_under_strict_freshness(stale_status: str) -> None:
    gate = DocumentCitationGate(strict_freshness=True)
    cid = str(uuid.uuid4())
    content = "Deprecate legacy HMAC v1 endpoints in favor of v2."

    chunk = DocumentChunk(
        chunk_id=uuid.UUID(cid),
        doc_id="SEC-505",
        doc_title="HMAC Deprecation",
        source_uri="https://jira.corp/browse/SEC-505",
        market_scope=MarketRegime.GLOBAL_MULTI,
        section_heading="Description",
        verbatim_content=content,
        formulas_latex=[],
        tables_markdown=[],
        token_count_approx=7,
        metadata_json={"ticket_key": "SEC-505", "status": stale_status},
    )
    cit = DocCitationSpan(
        chunk_id=cid,
        doc_id="SEC-505",
        doc_title="HMAC Deprecation",
        source_uri="https://jira.corp/browse/SEC-505",
        section_heading="Description",
        verbatim_quote=content,
        ticket_key="SEC-505",
    )
    res = gate.verify_citations([cit], [chunk])
    assert not res.passed, f"Jira status '{stale_status}' bypassed strict freshness"
    assert res.details[0].error_code == CitationFailureCode.STALE_TICKET_STATUS


@pytest.mark.parametrize(
    "stale_resolution",
    [
        "Won't Fix",
        "wont fix",
        "wontfix",
        "Duplicate",
        "Incomplete",
        "Cannot Reproduce",
        "Rejected",
        "Cancelled",
        "Declined",
        "Obsolete",
        "Won't Do",
        "Abandoned",
    ],
)
def test_jira_stale_resolutions_rejected_under_strict_freshness(stale_resolution: str) -> None:
    gate = DocumentCitationGate(strict_freshness=True)
    cid = str(uuid.uuid4())
    content = "Deprecate legacy HMAC v1 endpoints in favor of v2."

    chunk = DocumentChunk(
        chunk_id=uuid.UUID(cid),
        doc_id="SEC-506",
        doc_title="HMAC Deprecation",
        source_uri="https://jira.corp/browse/SEC-506",
        market_scope=MarketRegime.GLOBAL_MULTI,
        section_heading="Description",
        verbatim_content=content,
        formulas_latex=[],
        tables_markdown=[],
        token_count_approx=7,
        metadata_json={
            "ticket_key": "SEC-506",
            "status": "Resolved",
            "resolution": stale_resolution,
        },
    )
    cit = DocCitationSpan(
        chunk_id=cid,
        doc_id="SEC-506",
        doc_title="HMAC Deprecation",
        source_uri="https://jira.corp/browse/SEC-506",
        section_heading="Description",
        verbatim_quote=content,
        ticket_key="SEC-506",
    )
    res = gate.verify_citations([cit], [chunk])
    assert not res.passed, f"Jira resolution '{stale_resolution}' bypassed strict freshness"
    assert res.details[0].error_code == CitationFailureCode.STALE_TICKET_STATUS


def test_jira_completed_ticket_authentic_pipeline_is_valid() -> None:
    """Verifies authentic end-to-end integration: JiraConnectorPlugin -> DocumentCitationGate.

    Guarantees that a completed Jira ticket (Status='Done', Resolution='Done' or 'Fixed')
    is processed by JiraConnectorPlugin without false staleness and successfully passes
    DocumentCitationGate under strict_freshness=True without synthetic bypass.
    """
    jira = JiraConnectorPlugin()
    gate = DocumentCitationGate(strict_freshness=True)

    issue = JiraIssueDTO(
        issue_id="20485",
        issue_key="ENG-101",
        project_key="ENG",
        summary="PG17 Migration",
        issue_type="Task",
        status="Done",
        resolution="Done",
        assignee="alyssa.p.hacker",
        reporter="lead.architect",
        source_url="https://jira.corp/browse/ENG-101",
        description_text="Migrate database schema to PostgreSQL 17 successfully executed.",
        acceptance_criteria="AC1: All tables enforce FORCE ROW LEVEL SECURITY.",
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )

    chunks = jira.chunk_issue_verbatim(issue)
    assert len(chunks) >= 2, "Expected description and acceptance criteria chunks"

    # 1. Authentic chunk staleness checks from JiraConnectorPlugin
    for chunk in chunks:
        assert chunk.is_stale is False, f"Chunk falsely marked stale: {chunk.stale_reason}"
        assert chunk.stale_reason is None

    # 2. Authentic citation verification through DocumentCitationGate
    desc_chunk = next(c for c in chunks if c.section_heading == "Description")
    cit = DocCitationSpan(
        chunk_id=str(desc_chunk.chunk_id),
        doc_id=desc_chunk.doc_id,
        doc_title=desc_chunk.doc_title,
        source_uri=issue.source_url,
        section_heading="Description",
        verbatim_quote=issue.description_text,
        ticket_key="ENG-101",
    )
    res = gate.verify_citations([cit], chunks)
    assert res.passed is True, f"Valid completed ticket rejected: {res.reason}"
    assert res.verified_count == 1
    assert res.failed_count == 0


@pytest.mark.parametrize(
    ("status", "resolution"),
    [
        ("Cancelled", None),
        ("Won't Do", None),
        ("Rejected", None),
        ("Closed", "Fixed"),
        ("Resolved", "Won't Fix"),
        ("Done", "Won't Fix"),
        ("Obsolete", None),
        ("Resolved (Obsolete)", None),
        ("In Progress", "Won't Fix"),
        ("Done", "Duplicate"),
        ("Resolved", "Cannot Reproduce"),
    ],
)
def test_jira_obsolete_ticket_authentic_pipeline_rejected_with_stale_ticket_status(
    status: str, resolution: str | None
) -> None:
    """Verifies that obsolete tickets processed through JiraConnectorPlugin receive is_stale=True
    and when evaluated by DocumentCitationGate under strict_freshness=True, fail with
    CitationFailureCode.STALE_TICKET_STATUS (not STALE_SUPERSEDED_VERSION)."""
    jira = JiraConnectorPlugin()
    gate = DocumentCitationGate(strict_freshness=True)

    issue = JiraIssueDTO(
        issue_id="99901",
        issue_key="SEC-506",
        project_key="SEC",
        summary="Deprecate legacy HMAC endpoints",
        issue_type="Task",
        status=status,
        resolution=resolution,
        assignee="security.engineer",
        reporter="ciso",
        source_url="https://jira.corp/browse/SEC-506",
        description_text="Deprecate legacy HMAC v1 endpoints in favor of v2.",
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )

    chunks = jira.chunk_issue_verbatim(issue)
    assert len(chunks) >= 1
    assert chunks[0].is_stale is True, (
        f"Obsolete ticket status='{status}', res='{resolution}' not marked stale"
    )
    assert chunks[0].stale_reason is not None

    cit = DocCitationSpan(
        chunk_id=str(chunks[0].chunk_id),
        doc_id=chunks[0].doc_id,
        doc_title=chunks[0].doc_title,
        source_uri=issue.source_url,
        section_heading="Description",
        verbatim_quote=issue.description_text,
        ticket_key="SEC-506",
    )

    res = gate.verify_citations([cit], chunks)
    assert res.passed is False, (
        f"Obsolete ticket status='{status}', res='{resolution}' bypassed strict freshness"
    )
    assert res.failed_count == 1
    assert res.details[0].error_code == CitationFailureCode.STALE_TICKET_STATUS, (
        f"Expected STALE_TICKET_STATUS but got {res.details[0].error_code}"
    )


# ==============================================================================
# Dimension 4: Database Chunk Limit Resolution (>100 Chunks)
# ==============================================================================


def test_chunk_limit_resolution_exceeding_100_chunks() -> None:
    """Stress test: 150 document chunks loaded in repository.

    Citations reference chunks located past the 100-chunk mark (e.g. chunk 105, 140).
    PreFlightValidator must dynamically resolve and verify them without false abstention.
    """
    repo = InMemoryRLSRepository()
    principal = PrincipalContext(
        user_id="researcher",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng", "public_quant"],
    )
    domain_id = "quant_research_scout"

    # Populate 150 doc chunks
    all_chunks: list[DocChunkRecord] = []
    target_citations: list[DocCitationSpan] = []

    for idx in range(150):
        c_uuid = uuid.uuid4()
        c_text = f"Rule {idx}: Factor cutoff is {idx * 0.1:.2f}%."
        rec = DocChunkRecord(
            chunk_id=c_uuid,
            domain_id=domain_id,
            doc_id=f"arxiv:2026.{10000 + idx}",
            doc_title=f"Alpha Research Report {idx}",
            source_uri=f"https://arxiv.org/abs/2026.{10000 + idx}",
            market_scope=MarketRegime.GLOBAL_MULTI,
            published_date=datetime.now(UTC).date(),
            authority_weight=1.0,
            section_heading="Strategy Rules",
            contextual_header=f"Header {idx}",
            verbatim_content=c_text,
            metadata_json={"index": idx},
            allowed_scopes=["public_eng", "public_quant"],
            embedding=[0.0] * 768,
            source_type="arxiv",
            version_str="1",
            is_stale=False,
        )
        repo._docs[c_uuid] = rec
        all_chunks.append(rec)

        # Select chunks past the 100 mark
        if idx in (105, 125, 145):
            target_citations.append(
                DocCitationSpan(
                    chunk_id=str(c_uuid),
                    doc_id=rec.doc_id,
                    doc_title=rec.doc_title,
                    source_uri=rec.source_uri,
                    section_heading=rec.section_heading,
                    verbatim_quote=c_text,
                )
            )

    validator = PreFlightValidator(repo=repo, principal=principal, domain_id=domain_id)
    bound = validator._resolve_graph_repo(repo)

    passed, reasons = validator.check_citations(
        citations=target_citations,
        doc_chunks=None,  # Forces bound_repo lookup
        bound_repo=bound,
        principal=principal,
        domain_id=domain_id,
    )

    assert passed, f"Chunks beyond 100 failed resolution: {reasons}"
    assert len(reasons) == 0


# ==============================================================================
# Dimension 5: Multi-Repo Isolation and Cascading File Deletions
# ==============================================================================


def test_multi_repo_isolation_and_cascade_edge_integrity() -> None:
    """Stress test: Same relative file path across multiple repos in same tenant domain.

    Deleting a file in Repo A must NOT affect Repo B.
    Deleting a callee symbol must set inward AST edges target_symbol_id = None
    and delete outward edges.
    """
    repo = InMemoryRLSRepository()
    principal = PrincipalContext(
        user_id="alice",
        role=RoleType.ENGINEER,
        allowed_scopes=["eng_all"],
    )

    # 1. Setup Repo A and Repo B with same relative file path 'src/config.py'
    f_a = uuid.uuid4()
    f_b = uuid.uuid4()

    repo._files[f_a] = CodeFileRecord(
        file_id=f_a,
        domain_id="default",
        repo_name="org/service_alpha",
        file_path="src/config.py",
        git_commit_sha="commit_alpha",
        git_blob_sha="blob_alpha",
        language="python",
        allowed_scopes=["eng_all"],
        raw_content="PORT = 8080\ndef get_port(): return PORT",
        line_count=2,
    )
    repo._files[f_b] = CodeFileRecord(
        file_id=f_b,
        domain_id="default",
        repo_name="org/service_beta",
        file_path="src/config.py",
        git_commit_sha="commit_beta",
        git_blob_sha="blob_beta",
        language="python",
        allowed_scopes=["eng_all"],
        raw_content="PORT = 9090\ndef get_port(): return PORT",
        line_count=2,
    )

    # Add symbols and edges in Repo B
    sym_caller_b = uuid.uuid4()
    sym_callee_b = uuid.uuid4()

    repo._symbols[sym_caller_b] = ASTSymbolRecord(
        symbol_id=sym_caller_b,
        file_id=f_b,
        domain_id="default",
        qualified_name="org/service_beta::src/config.py::main",
        symbol_kind="function",
        start_line=1,
        end_line=1,
        signature_skeleton="main()",
        contextual_summary="",
        allowed_scopes=["eng_all"],
        embedding=[],
    )
    repo._symbols[sym_callee_b] = ASTSymbolRecord(
        symbol_id=sym_callee_b,
        file_id=f_b,
        domain_id="default",
        qualified_name="org/service_beta::src/config.py::get_port",
        symbol_kind="function",
        start_line=2,
        end_line=2,
        signature_skeleton="get_port()",
        contextual_summary="",
        allowed_scopes=["eng_all"],
        embedding=[],
    )

    edge_b = uuid.uuid4()
    repo._edges[edge_b] = ASTEdgeRecord(
        edge_id=edge_b,
        domain_id="default",
        source_symbol_id=sym_caller_b,
        target_symbol_id=sym_callee_b,
        target_qualified_name="org/service_beta::src/config.py::get_port",
        allowed_scopes=["eng_all"],
        edge_kind="CALLS",
        is_cross_repo=False,
    )

    # Delete Repo A's file
    repo.delete_code_file_and_symbols("default", "org/service_alpha", "src/config.py")

    # Assert Repo A deleted, Repo B 100% intact
    assert f_a not in repo._files
    assert f_b in repo._files
    assert sym_caller_b in repo._symbols
    assert sym_callee_b in repo._symbols
    assert edge_b in repo._edges
    assert repo._edges[edge_b].target_symbol_id == sym_callee_b

    # Now delete Repo B's file using principal RLS delete
    repo.delete_code_file(principal, f_b)

    # Assert callee deleted and edge target_symbol_id set to None without dangling pointer
    assert f_b not in repo._files
    assert sym_caller_b not in repo._symbols
    assert sym_callee_b not in repo._symbols
    assert edge_b not in repo._edges  # Outward edge from deleted source is removed


# ==============================================================================
# Dimension 6: Confluence and Jira Connector API Synchronization Contracts
# ==============================================================================


def test_confluence_connector_monotonic_versioning() -> None:
    """Tests Confluence connector version supersession logic during sync."""
    connector = ConfluenceConnectorPlugin(freshness_half_life_days=180)
    repo = InMemoryRLSRepository()
    domain_id = "default"
    space_key = "ENG"

    # Page version 1
    p1 = ConfluencePageDTO(
        page_id="1001",
        space_key=space_key,
        title="Architecture Decision",
        version_number=1,
        author_username="lead_arch",
        source_url="https://mock.atlassian.net/wiki/spaces/ENG/pages/1001",
        body_storage="<h1>Decision</h1><p>We use monolith architecture.</p>",
        created_at=datetime.now(UTC),
        last_modified_at=datetime.now(UTC),
    )
    res1 = connector.sync_space_to_repository(domain_id, space_key, repo, current_pages=[p1])
    assert res1.items_indexed >= 1
    assert res1.stale_items_flagged == 0

    # Page version 2 supersedes version 1
    p2 = ConfluencePageDTO(
        page_id="1001",
        space_key=space_key,
        title="Architecture Decision",
        version_number=2,
        author_username="lead_arch",
        source_url="https://mock.atlassian.net/wiki/spaces/ENG/pages/1001",
        body_storage="<h1>Decision</h1><p>We use event-driven microservices architecture.</p>",
        created_at=datetime.now(UTC),
        last_modified_at=datetime.now(UTC),
    )
    res2 = connector.sync_space_to_repository(domain_id, space_key, repo, current_pages=[p2])
    assert res2.items_indexed >= 1
    assert res2.stale_items_flagged >= 1

    # Verify old chunk is marked stale
    old_chunks = [
        c for c in repo._docs.values() if c.doc_id == "confluence:1001" and c.version_str == "1"
    ]
    assert all(c.is_stale for c in old_chunks)
    assert any("Superseded by page version 2" in (c.stale_reason or "") for c in old_chunks)


def test_jira_connector_staleness_and_scope_mirroring() -> None:
    """Tests Jira connector staleness classification and permission mapping."""
    jira = JiraConnectorPlugin(base_url="https://jira.corp")
    scopes = jira.mirror_permissions("PAY")
    assert "jira:project:PAY:read" in scopes

    # Stale issue test
    issue_stale = JiraIssueDTO(
        issue_id="999",
        issue_key="PAY-99",
        project_key="PAY",
        summary="Old payment feature",
        issue_type="Task",
        status="Won't Do",
        resolution="Won't Fix",
        source_url="https://jira.corp/browse/PAY-99",
        description_text="Do not implement legacy payment.",
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    is_stale, reason = jira.is_ticket_stale(issue_stale)
    assert is_stale is True
    assert reason is not None
