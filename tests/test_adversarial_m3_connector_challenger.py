"""Empirical Challenger stress tests for Milestone 3 Connectors & Citation Gate.

Demonstrates reproducible bugs in:
1. Multi-repo collateral deletion in InMemoryRLSRepository and GitLabConnectorPlugin.
2. Dangling inward AST edges on file deletion in InMemoryRLSRepository.
3. Confluence version freshness bypass when version is only on version_str.
4. Jira modern Cloud terminal status evasion ("Won't Do").
5. Intra-sentence quote tampering word omission evasion.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import httpx

from context_scout.agents.citation_gate import (
    CitationFailureCode,
    DocumentCitationGate,
)
from context_scout.ast_engine.deterministic_slicer import (
    DeterministicSlicer,
    SlicerErrorCode,
)
from context_scout.db.repositories import (
    ASTEdgeRecord,
    ASTSymbolRecord,
    CodeFileRecord,
    DocChunkRecord,
    InMemoryRLSRepository,
)
from context_scout.ingestion.doc_parser import DocumentChunk
from context_scout.plugins.connectors.gitlab import GitLabConnectorPlugin
from context_scout.plugins.connectors.jira import JiraConnectorPlugin, JiraIssueDTO
from context_scout.queue.task_queue import QueueTask
from context_scout.schemas.contracts import (
    DocCitationSpan,
    MarketRegime,
    PrincipalContext,
)


def test_challenger_multi_repo_collateral_deletion_in_memory_repository() -> None:
    """Demonstrates that deleting a file in Repo A purges the same file path in Repo B.

    DEFECT: delete_code_file_and_symbols uses '(repo_name IN (...) OR f.domain_id == domain_id)'
    which causes ANY file in the domain with matching file_path to be deleted.
    """
    repo = InMemoryRLSRepository()
    f1_id = uuid.uuid4()
    repo._files[f1_id] = CodeFileRecord(
        file_id=f1_id,
        domain_id="default",
        repo_name="org/repo_a",
        file_path="src/service.py",
        git_commit_sha="sha1",
        git_blob_sha="blob1",
        language="python",
        allowed_scopes=["scope_a"],
        raw_content="repo_a",
        line_count=1,
    )
    f2_id = uuid.uuid4()
    repo._files[f2_id] = CodeFileRecord(
        file_id=f2_id,
        domain_id="default",
        repo_name="org/repo_b",
        file_path="src/service.py",
        git_commit_sha="sha2",
        git_blob_sha="blob2",
        language="python",
        allowed_scopes=["scope_b"],
        raw_content="repo_b",
        line_count=1,
    )

    assert len(repo._files) == 2
    repo.delete_code_file_and_symbols("default", "org/repo_a", "src/service.py")

    # BUG REPRODUCTION: Repo B's file was wiped out as collateral damage!
    remaining_repos = [f.repo_name for f in repo._files.values()]
    assert "org/repo_b" in remaining_repos, (
        f"CRITICAL BUG: org/repo_b was deleted when org/repo_a file was deleted! "
        f"Remaining repos: {remaining_repos}"
    )


def test_challenger_gitlab_push_webhook_collateral_repo_deletion() -> None:
    """Demonstrates that updating a file in Repo A in push webhook deletes Repo B's file."""
    repo = InMemoryRLSRepository()
    f_b_id = uuid.uuid4()
    repo._files[f_b_id] = CodeFileRecord(
        file_id=f_b_id,
        domain_id="default",
        repo_name="org/repo_b",
        file_path="src/service.py",
        git_commit_sha="sha_b",
        git_blob_sha="blob_b",
        language="python",
        allowed_scopes=["repo_b"],
        raw_content="def repo_b(): pass",
        line_count=1,
    )

    def mock_gl(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "compare" in url:
            return httpx.Response(
                200,
                json={
                    "diffs": [
                        {
                            "old_path": "src/service.py",
                            "new_path": "src/service.py",
                            "new_file": False,
                            "renamed_file": False,
                            "deleted_file": False,
                        }
                    ]
                },
            )
        if "src%2Fservice.py/raw" in url:
            return httpx.Response(200, content=b"def repo_a(): pass")
        return httpx.Response(404, json={})

    client = httpx.Client(
        transport=httpx.MockTransport(mock_gl), base_url="https://gitlab.example.com"
    )
    plugin = GitLabConnectorPlugin(base_url="https://gitlab.example.com", client=client)

    task = QueueTask(
        task_id=uuid.uuid4(),
        domain_id="default",
        event_type="gitlab_push",
        payload={
            "project": {
                "id": 1,
                "path_with_namespace": "org/repo_a",
                "default_branch": "main",
            },
            "before": "sha1",
            "after": "sha2",
            "ref": "refs/heads/main",
            "commits": [{"id": "sha2"}],
        },
        priority=1,
        status="PENDING",
        worker_id=None,
        attempts=0,
        max_attempts=3,
        error_message=None,
        scheduled_at=datetime.now(UTC),
        created_at=datetime.now(UTC),
    )

    plugin.process_push_webhook(task, repo)

    repos = [f.repo_name for f in repo._files.values()]
    assert "org/repo_b" in repos, (
        f"CRITICAL BUG: org/repo_b was deleted when org/repo_a pushed update! "
        f"Remaining repos: {repos}"
    )


def test_challenger_dangling_inward_ast_edges_on_file_deletion() -> None:
    """Demonstrates that deleting a callee file leaves dangling inward edges in _edges."""
    repo = InMemoryRLSRepository()
    fid1 = uuid.uuid4()
    fid2 = uuid.uuid4()

    repo._files[fid1] = CodeFileRecord(
        file_id=fid1,
        domain_id="default",
        repo_name="repo1",
        file_path="caller.py",
        git_commit_sha="c1",
        git_blob_sha="b1",
        language="python",
        allowed_scopes=["all"],
        raw_content="caller",
        line_count=1,
    )
    repo._files[fid2] = CodeFileRecord(
        file_id=fid2,
        domain_id="default",
        repo_name="repo1",
        file_path="callee.py",
        git_commit_sha="c2",
        git_blob_sha="b2",
        language="python",
        allowed_scopes=["all"],
        raw_content="callee",
        line_count=1,
    )

    s1 = uuid.uuid4()
    s2 = uuid.uuid4()
    repo._symbols[s1] = ASTSymbolRecord(
        symbol_id=s1,
        file_id=fid1,
        domain_id="default",
        qualified_name="caller",
        symbol_kind="function",
        start_line=1,
        end_line=1,
        signature_skeleton="caller()",
        contextual_summary="",
        allowed_scopes=["all"],
        embedding=[],
    )
    repo._symbols[s2] = ASTSymbolRecord(
        symbol_id=s2,
        file_id=fid2,
        domain_id="default",
        qualified_name="callee",
        symbol_kind="function",
        start_line=1,
        end_line=1,
        signature_skeleton="callee()",
        contextual_summary="",
        allowed_scopes=["all"],
        embedding=[],
    )

    edge_id = uuid.uuid4()
    repo._edges[edge_id] = ASTEdgeRecord(
        edge_id=edge_id,
        domain_id="default",
        source_symbol_id=s1,
        target_symbol_id=s2,
        target_qualified_name="callee",
        allowed_scopes=["all"],
        edge_kind="CALLS",
        is_cross_repo=False,
    )

    repo.delete_code_file_and_symbols("default", "repo1", "callee.py")
    assert s2 not in repo._symbols

    # BUG REPRODUCTION: Edge still references deleted symbol s2!
    for e in repo._edges.values():
        assert e.target_symbol_id not in (s2,), (
            f"Dangling AST edge pointing to deleted symbol {s2} was not cleaned up!"
        )


def test_challenger_confluence_version_freshness_bypass_when_only_on_record() -> None:
    """Demonstrates that CitationGate bypasses version check when version is only on version_str."""
    gate = DocumentCitationGate(strict_freshness=True)
    chunk_id = str(uuid.uuid4())
    content = "Microservices must enforce mutual TLS authentication."

    chunk_rec = DocChunkRecord(
        chunk_id=uuid.UUID(chunk_id),
        domain_id="default",
        doc_id="confluence:100",
        doc_title="ADR mTLS",
        source_uri="https://wiki.corp.internal/100",
        market_scope=None,
        published_date=datetime.now(UTC).date(),
        authority_weight=1.0,
        section_heading="Security",
        contextual_header="Header",
        verbatim_content=content,
        metadata_json={"page_id": "100"},  # version key omitted from metadata_json
        allowed_scopes=["public_eng"],
        embedding=[0.0] * 768,
        source_type="confluence",
        version_str="5",  # Canonical version on record is 5
        is_stale=False,
    )

    cit_stale = DocCitationSpan(
        chunk_id=chunk_id,
        doc_id="100",
        doc_title="ADR mTLS",
        source_uri="https://wiki.corp.internal/100",
        section_heading="Security",
        verbatim_quote=content,
        version="1",  # Superseded! Latest is 5
        page_id="100",
    )

    res = gate.verify_citations([cit_stale], [chunk_rec])
    # BUG REPRODUCTION: Stale version 1 passes because CitationGate ignores version_str!
    assert res.passed is False, "Citation with superseded version 1 passed gate!"
    assert res.details[0].error_code == CitationFailureCode.STALE_SUPERSEDED_VERSION


def test_challenger_jira_wont_do_terminal_status_evasion() -> None:
    """Demonstrates that Jira ticket with Cloud standard status 'Won't Do' evades freshness."""
    gate = DocumentCitationGate(strict_freshness=True)
    chunk = DocumentChunk(
        chunk_id=uuid.uuid4(),
        doc_id="ENG-900",
        doc_title="Task",
        source_uri="https://jira.corp.internal/browse/ENG-900",
        market_scope=MarketRegime.GLOBAL_MULTI,
        section_heading="Sec",
        verbatim_content="Content text here.",
        formulas_latex=[],
        tables_markdown=[],
        token_count_approx=3,
        metadata_json={"issue_key": "ENG-900", "status": "Won't Do"},
    )
    cit = DocCitationSpan(
        chunk_id=str(chunk.chunk_id),
        doc_id="ENG-900",
        doc_title="Task",
        source_uri="https://jira.corp.internal/browse/ENG-900",
        section_heading="Sec",
        verbatim_quote="Content text here.",
        ticket_key="ENG-900",
    )
    res = gate.verify_citations([cit], [chunk])
    # BUG REPRODUCTION: Status 'Won't Do' evades freshness check!
    assert res.passed is False, "Jira ticket with status 'Won't Do' unexpectedly passed gate!"
    assert res.details[0].error_code == CitationFailureCode.STALE_TICKET_STATUS


def test_challenger_intra_sentence_quote_omission_evasion() -> None:
    """Demonstrates that omitting critical words (e.g. 'NO') from a quote is rejected.

    DEFECT 1.5: SequenceMatcher marked omitted middle tokens as 'insert' in source,
    which bypassed ('replace', 'delete') unmatched checks and passed reject_absent_tokens=True.
    """
    slicer = DeterministicSlicer()
    source_text = "Under NO circumstances should the gateway allow unauthenticated fallback mode."
    tampered_quote = "Under circumstances should the gateway allow unauthenticated fallback mode."

    # 1. Direct slicer verification with reject_absent_tokens=True must fail
    res = slicer.verify_quote(tampered_quote, source_text, reject_absent_tokens=True)
    assert res.is_valid is False, "Intra-sentence word omission unexpectedly passed verification!"
    assert res.error_code == SlicerErrorCode.HALLUCINATED_QUOTE
    assert "omission" in (res.error_reason or "").lower()

    # 2. DocumentCitationGate verification must fail with HALLUCINATED_QUOTE
    gate = DocumentCitationGate()
    chunk_id = str(uuid.uuid4())
    cit = DocCitationSpan(
        chunk_id=chunk_id,
        doc_id="sec-policy",
        doc_title="Security Policy",
        source_uri="https://wiki.corp.internal/sec",
        section_heading="Auth",
        verbatim_quote=tampered_quote,
        page_id="100",
        version="1",
    )
    chunk = DocumentChunk(
        chunk_id=uuid.UUID(chunk_id),
        doc_id="sec-policy",
        doc_title="Security Policy",
        source_uri="https://wiki.corp.internal/sec",
        market_scope=MarketRegime.GLOBAL_MULTI,
        section_heading="Auth",
        verbatim_content=source_text,
        formulas_latex=[],
        tables_markdown=[],
        token_count_approx=10,
        metadata_json={"page_id": "100", "version": "1"},
    )
    res_gate = gate.verify_citations([cit], [chunk])
    assert res_gate.passed is False, "Tampered quote unexpectedly passed DocumentCitationGate!"
    assert res_gate.details[0].error_code == CitationFailureCode.HALLUCINATED_QUOTE


def test_challenger_multi_repo_isolation_with_identical_paths_and_symbols() -> None:
    """Stress-test 1: Multi-repo isolation across same tenant domain with identical paths.

    Creates repo_a and repo_b in domain 'default' with identical path 'src/common.py'.
    Attaches AST symbols to both. Deletes 'src/common.py' in repo_a.
    Verifies that repo_b's file, all its AST symbols, and its AST edges remain 100% intact.
    """
    repo = InMemoryRLSRepository()
    fa_id = uuid.uuid4()
    fb_id = uuid.uuid4()

    repo._files[fa_id] = CodeFileRecord(
        file_id=fa_id,
        domain_id="default",
        repo_name="org/repo_a",
        file_path="src/common.py",
        git_commit_sha="sha_a",
        git_blob_sha="blob_a",
        language="python",
        allowed_scopes=["scope_a"],
        raw_content="def common(): return 'a'",
        line_count=1,
    )
    repo._files[fb_id] = CodeFileRecord(
        file_id=fb_id,
        domain_id="default",
        repo_name="org/repo_b",
        file_path="src/common.py",
        git_commit_sha="sha_b",
        git_blob_sha="blob_b",
        language="python",
        allowed_scopes=["scope_b"],
        raw_content="def common(): return 'b'",
        line_count=1,
    )

    sa_id = uuid.uuid4()
    sb1_id = uuid.uuid4()
    sb2_id = uuid.uuid4()

    repo._symbols[sa_id] = ASTSymbolRecord(
        symbol_id=sa_id,
        file_id=fa_id,
        domain_id="default",
        qualified_name="common",
        symbol_kind="function",
        start_line=1,
        end_line=1,
        signature_skeleton="common()",
        contextual_summary="Repo A common",
        allowed_scopes=["scope_a"],
        embedding=[],
    )
    repo._symbols[sb1_id] = ASTSymbolRecord(
        symbol_id=sb1_id,
        file_id=fb_id,
        domain_id="default",
        qualified_name="common",
        symbol_kind="function",
        start_line=1,
        end_line=1,
        signature_skeleton="common()",
        contextual_summary="Repo B common",
        allowed_scopes=["scope_b"],
        embedding=[],
    )
    repo._symbols[sb2_id] = ASTSymbolRecord(
        symbol_id=sb2_id,
        file_id=fb_id,
        domain_id="default",
        qualified_name="helper",
        symbol_kind="function",
        start_line=2,
        end_line=2,
        signature_skeleton="helper()",
        contextual_summary="Repo B helper",
        allowed_scopes=["scope_b"],
        embedding=[],
    )

    eb_id = uuid.uuid4()
    repo._edges[eb_id] = ASTEdgeRecord(
        edge_id=eb_id,
        domain_id="default",
        source_symbol_id=sb1_id,
        target_symbol_id=sb2_id,
        target_qualified_name="helper",
        allowed_scopes=["scope_b"],
        edge_kind="CALLS",
        is_cross_repo=False,
    )

    # Delete repo_a's file
    repo.delete_code_file_and_symbols("default", "org/repo_a", "src/common.py")

    # Assert repo_a purged
    assert fa_id not in repo._files, "repo_a file was not deleted"
    assert sa_id not in repo._symbols, "repo_a symbol was not deleted"

    # Assert repo_b 100% intact
    assert fb_id in repo._files, "repo_b file was corrupted/deleted"
    assert repo._files[fb_id].file_path == "src/common.py"
    assert sb1_id in repo._symbols, "repo_b symbol 'common' was corrupted/deleted"
    assert sb2_id in repo._symbols, "repo_b symbol 'helper' was corrupted/deleted"
    assert eb_id in repo._edges, "repo_b AST edge was deleted"
    assert repo._edges[eb_id].target_symbol_id == sb2_id, "repo_b edge target was corrupted"


def test_challenger_cascading_inward_ast_edges_set_null_no_dangling_foreign_keys() -> None:
    """Stress-test 2: Cascading inward AST edges on symbol deletion set target_symbol_id=None.

    Verifies that when a callee symbol is deleted:
    1. Inward edges retain target_symbol_id = None without raising exceptions.
    2. Outward edges from the deleted symbol are fully purged.
    3. No dangling foreign keys exist anywhere in repo._edges.
    4. Evaluates both delete_code_file_and_symbols and delete_code_file(principal).
    """
    repo = InMemoryRLSRepository()
    principal = PrincipalContext(
        user_id="alice",
        role="engineer",
        allowed_scopes=["eng"],
        session_budget_tokens=10000,
    )

    f_caller = uuid.uuid4()
    f_callee = uuid.uuid4()
    f_out = uuid.uuid4()

    repo._files[f_caller] = CodeFileRecord(
        file_id=f_caller,
        domain_id="default",
        repo_name="myrepo",
        file_path="caller.py",
        git_commit_sha="c1",
        git_blob_sha="b1",
        language="python",
        allowed_scopes=["eng"],
        raw_content="def caller(): pass",
        line_count=1,
    )
    repo._files[f_callee] = CodeFileRecord(
        file_id=f_callee,
        domain_id="default",
        repo_name="myrepo",
        file_path="callee.py",
        git_commit_sha="c2",
        git_blob_sha="b2",
        language="python",
        allowed_scopes=["eng"],
        raw_content="def target_api(): pass",
        line_count=1,
    )
    repo._files[f_out] = CodeFileRecord(
        file_id=f_out,
        domain_id="default",
        repo_name="myrepo",
        file_path="external.py",
        git_commit_sha="c3",
        git_blob_sha="b3",
        language="python",
        allowed_scopes=["eng"],
        raw_content="def ext(): pass",
        line_count=1,
    )

    s_caller1 = uuid.uuid4()
    s_caller2 = uuid.uuid4()
    s_target = uuid.uuid4()
    s_ext = uuid.uuid4()

    repo._symbols[s_caller1] = ASTSymbolRecord(
        symbol_id=s_caller1,
        file_id=f_caller,
        domain_id="default",
        qualified_name="caller1",
        symbol_kind="function",
        start_line=1,
        end_line=1,
        signature_skeleton="caller1()",
        contextual_summary="",
        allowed_scopes=["eng"],
        embedding=[],
    )
    repo._symbols[s_caller2] = ASTSymbolRecord(
        symbol_id=s_caller2,
        file_id=f_caller,
        domain_id="default",
        qualified_name="caller2",
        symbol_kind="function",
        start_line=2,
        end_line=2,
        signature_skeleton="caller2()",
        contextual_summary="",
        allowed_scopes=["eng"],
        embedding=[],
    )
    repo._symbols[s_target] = ASTSymbolRecord(
        symbol_id=s_target,
        file_id=f_callee,
        domain_id="default",
        qualified_name="target_api",
        symbol_kind="function",
        start_line=1,
        end_line=1,
        signature_skeleton="target_api()",
        contextual_summary="",
        allowed_scopes=["eng"],
        embedding=[],
    )
    repo._symbols[s_ext] = ASTSymbolRecord(
        symbol_id=s_ext,
        file_id=f_out,
        domain_id="default",
        qualified_name="ext",
        symbol_kind="function",
        start_line=1,
        end_line=1,
        signature_skeleton="ext()",
        contextual_summary="",
        allowed_scopes=["eng"],
        embedding=[],
    )

    # Inward edge 1 (CALLS)
    e_in1 = uuid.uuid4()
    repo._edges[e_in1] = ASTEdgeRecord(
        edge_id=e_in1,
        domain_id="default",
        source_symbol_id=s_caller1,
        target_symbol_id=s_target,
        target_qualified_name="target_api",
        allowed_scopes=["eng"],
        edge_kind="CALLS",
        is_cross_repo=False,
    )
    # Inward edge 2 (IMPORTS)
    e_in2 = uuid.uuid4()
    repo._edges[e_in2] = ASTEdgeRecord(
        edge_id=e_in2,
        domain_id="default",
        source_symbol_id=s_caller2,
        target_symbol_id=s_target,
        target_qualified_name="target_api",
        allowed_scopes=["eng"],
        edge_kind="IMPORTS",
        is_cross_repo=False,
    )
    # Outward edge from s_target to s_ext
    e_out = uuid.uuid4()
    repo._edges[e_out] = ASTEdgeRecord(
        edge_id=e_out,
        domain_id="default",
        source_symbol_id=s_target,
        target_symbol_id=s_ext,
        target_qualified_name="ext",
        allowed_scopes=["eng"],
        edge_kind="CALLS",
        is_cross_repo=False,
    )

    # Delete callee file using delete_code_file (with principal RLS)
    repo.delete_code_file(principal, f_callee)

    assert s_target not in repo._symbols, "Target symbol was not deleted"
    assert e_out not in repo._edges, "Outward edge from deleted symbol was not cascaded"
    assert e_in1 in repo._edges, "Inward edge 1 was incorrectly purged"
    assert repo._edges[e_in1].target_symbol_id is None, (
        "Inward edge 1 target_symbol_id was not set to None"
    )
    assert repo._edges[e_in1].target_qualified_name == "target_api", "target_qualified_name lost"

    assert e_in2 in repo._edges, "Inward edge 2 was incorrectly purged"
    assert repo._edges[e_in2].target_symbol_id is None, (
        "Inward edge 2 target_symbol_id was not set to None"
    )

    # Verify zero dangling foreign keys exist in _edges
    for edge_id, edge in repo._edges.items():
        assert edge.source_symbol_id in repo._symbols, (
            f"Dangling source_symbol_id on edge {edge_id}"
        )
        if edge.target_symbol_id is not None:
            assert edge.target_symbol_id in repo._symbols, (
                f"Dangling target_symbol_id on edge {edge_id}"
            )


def test_challenger_confluence_version_freshness_matrix() -> None:
    """Stress-test 3: Confluence version freshness strict rejection on version_str.

    Verifies that:
    1. Citing superseded version ('1' vs '5', '4' vs '5') is strictly rejected.
    2. Citing superseded version with unindexed/higher version is rejected.
    3. Handles version_str formatting ('v5', '12', '3.0') when metadata_json is empty or None.
    4. Citing matching canonical version passes validation.
    """
    gate = DocumentCitationGate(strict_freshness=True)
    body = (
        "Production zero-trust boundary mandates mutual TLS authentication for all gRPC channels."
    )

    cases = [
        # (canonical_version_str, metadata, cited_version, expect_pass, expected_error_code)
        ("5", None, "1", False, CitationFailureCode.STALE_SUPERSEDED_VERSION),
        ("5", {}, "4", False, CitationFailureCode.STALE_SUPERSEDED_VERSION),
        ("v5", {"page_id": "200"}, "1", False, CitationFailureCode.STALE_SUPERSEDED_VERSION),
        ("12", None, "11", False, CitationFailureCode.STALE_SUPERSEDED_VERSION),
        ("3.0", {}, "2", False, CitationFailureCode.STALE_SUPERSEDED_VERSION),
        ("5", None, "5", True, None),
        ("v5", {"page_id": "200"}, "5", True, None),
        ("12", None, "12", True, None),
        ("3.0", {}, "3", True, None),
    ]

    for canon_v, meta, cite_v, expect_pass, exp_err in cases:
        cid = str(uuid.uuid4())
        chunk = DocChunkRecord(
            chunk_id=uuid.UUID(cid),
            domain_id="default",
            doc_id="confluence:200",
            doc_title="Zero Trust Spec",
            source_uri="https://wiki.corp/200",
            market_scope=None,
            published_date=datetime.now(UTC).date(),
            authority_weight=1.0,
            section_heading="Security",
            contextual_header="Security Header",
            verbatim_content=body,
            metadata_json=meta,
            allowed_scopes=["eng"],
            embedding=[0.0] * 768,
            source_type="confluence",
            version_str=canon_v,
            is_stale=False,
        )
        cit = DocCitationSpan(
            chunk_id=cid,
            doc_id="200",
            doc_title="Zero Trust Spec",
            source_uri="https://wiki.corp/200",
            section_heading="Security",
            verbatim_quote=body,
            version=cite_v,
            page_id="200",
        )
        res = gate.verify_citations([cit], [chunk])
        assert res.passed is expect_pass, (
            f"Failed freshness check for canon={canon_v}, cite={cite_v}: passed={res.passed}"
        )
        if not expect_pass:
            assert res.details[0].error_code == exp_err


def test_challenger_jira_terminal_status_and_resolution_evasion_matrix() -> None:
    """Stress-test 4: Jira status and resolution evasion.

    Adversarially tests variants of 'Won't Do', 'Abandoned', 'wont do', 'wontdo',
    'ABANDONED', and 'WONT DO' across:
    1. JiraConnectorPlugin.is_ticket_stale() (status and resolution paths)
    2. DocumentCitationGate.verify_citations() under strict_freshness=True
    """
    jira = JiraConnectorPlugin(base_url="https://jira.corp", email="test@corp", api_token="tok")
    gate = DocumentCitationGate(strict_freshness=True)

    status_variants = [
        "Won't Do",
        "Abandoned",
        "wont do",
        "wontdo",
        "ABANDONED",
        "WONT DO",
        "won't-do",
        "wont-do",
    ]

    for status_str in status_variants:
        # 1. Test via JiraConnectorPlugin on status
        issue = JiraIssueDTO(
            issue_id="1001",
            issue_key="SEC-101",
            project_key="SEC",
            summary="Deprecated Cipher Suites",
            issue_type="Task",
            status=status_str,
            resolution=None,
            source_url="https://jira.corp/browse/SEC-101",
            description_text="Remove 3DES and RC4.",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        is_stale, reason = jira.is_ticket_stale(issue)
        assert is_stale is True, (
            f"JiraConnectorPlugin failed to mark status '{status_str}' as stale"
        )
        assert reason is not None

        # 2. Test via JiraConnectorPlugin on resolution
        issue_res = JiraIssueDTO(
            issue_id="1002",
            issue_key="SEC-102",
            project_key="SEC",
            summary="Deprecated Cipher Suites 2",
            issue_type="Task",
            status="In Review",
            resolution=status_str,
            source_url="https://jira.corp/browse/SEC-102",
            description_text="Remove 3DES and RC4.",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        is_stale_res, reason_res = jira.is_ticket_stale(issue_res)
        assert is_stale_res is True, (
            f"JiraConnectorPlugin failed to mark resolution '{status_str}' as stale"
        )
        assert reason_res is not None

        # 3. Test via DocumentCitationGate
        cid = str(uuid.uuid4())
        chunk = DocumentChunk(
            chunk_id=uuid.UUID(cid),
            doc_id="SEC-101",
            doc_title="Deprecated Cipher Suites",
            source_uri="https://jira.corp/browse/SEC-101",
            market_scope=MarketRegime.GLOBAL_MULTI,
            section_heading="Security",
            verbatim_content="Remove 3DES and RC4.",
            formulas_latex=[],
            tables_markdown=[],
            token_count_approx=4,
            metadata_json={"ticket_key": "SEC-101", "status": status_str},
        )
        cit = DocCitationSpan(
            chunk_id=cid,
            doc_id="SEC-101",
            doc_title="Deprecated Cipher Suites",
            source_uri="https://jira.corp/browse/SEC-101",
            section_heading="Security",
            verbatim_quote="Remove 3DES and RC4.",
            ticket_key="SEC-101",
        )
        res = gate.verify_citations([cit], [chunk])
        assert res.passed is False, (
            f"DocumentCitationGate passed stale ticket with status '{status_str}'"
        )
        assert res.details[0].error_code == CitationFailureCode.STALE_TICKET_STATUS


def test_challenger_deterministic_slicer_quote_tampering_adversarial_matrix() -> None:
    """Stress-test 5: Slicer quote tampering with omitted negations and rearranged tokens.

    Adversarially tests:
    1. Omitted negation words: 'NO', 'NOT', 'NEVER'.
    2. Rearranged word tokens (e.g. swapping account A and account B).
    Verifies rejection across DeterministicSlicer and DocumentCitationGate.
    """
    slicer = DeterministicSlicer()
    gate = DocumentCitationGate()

    attack_cases = [
        (
            "omitted_NO",
            "Under NO circumstances should the gateway allow unauthenticated fallback mode.",
            "Under circumstances should the gateway allow unauthenticated fallback mode.",
        ),
        (
            "omitted_NOT",
            "The authorization proxy must NOT bypass tenant access control policies.",
            "The authorization proxy must bypass tenant access control policies.",
        ),
        (
            "omitted_NEVER",
            "Worker processes should NEVER execute unauthenticated shell commands.",
            "Worker processes should execute unauthenticated shell commands.",
        ),
        (
            "rearranged_accounts",
            "Transfer funds from account A to account B.",
            "Transfer funds from account B to account A.",
        ),
        (
            "rearranged_order",
            "Payment gateway requires authorization before settlement.",
            "Payment gateway requires settlement before authorization.",
        ),
    ]

    for attack_name, source_text, tampered_quote in attack_cases:
        # A. Direct slicer verification with reject_absent_tokens=True
        slicer_res = slicer.verify_quote(tampered_quote, source_text, reject_absent_tokens=True)
        assert slicer_res.is_valid is False, (
            f"Attack '{attack_name}' bypassed DeterministicSlicer validation!"
        )
        assert slicer_res.error_code == SlicerErrorCode.HALLUCINATED_QUOTE, (
            f"Attack '{attack_name}' returned unexpected error code: {slicer_res.error_code}"
        )
        assert len(slicer_res.unmatched_tokens) > 0, (
            f"Attack '{attack_name}' failed to populate unmatched_tokens"
        )

        # B. DocumentCitationGate verification
        cid = str(uuid.uuid4())
        chunk = DocumentChunk(
            chunk_id=uuid.UUID(cid),
            doc_id="doc-tamper-check",
            doc_title="Critical Policy",
            source_uri="https://wiki.corp/policy",
            market_scope=MarketRegime.GLOBAL_MULTI,
            section_heading="Policy",
            verbatim_content=source_text,
            formulas_latex=[],
            tables_markdown=[],
            token_count_approx=10,
            metadata_json={"page_id": "1", "version": "1"},
        )
        cit = DocCitationSpan(
            chunk_id=cid,
            doc_id="doc-tamper-check",
            doc_title="Critical Policy",
            source_uri="https://wiki.corp/policy",
            section_heading="Policy",
            verbatim_quote=tampered_quote,
            page_id="1",
            version="1",
        )
        gate_res = gate.verify_citations([cit], [chunk])
        assert gate_res.passed is False, (
            f"Attack '{attack_name}' bypassed DocumentCitationGate verification!"
        )
        assert gate_res.details[0].error_code == CitationFailureCode.HALLUCINATED_QUOTE, (
            f"Attack '{attack_name}' failed with {gate_res.details[0].error_code}"
        )

