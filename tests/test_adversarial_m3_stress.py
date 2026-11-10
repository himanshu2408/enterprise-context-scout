"""Adversarial stress-test suite for Milestone 3 Connectors & Citation Gate.

Constructs empirical attack scenarios covering:
1. Citation Forgery (fabricated Confluence page IDs and Jira ticket keys).
2. Cross-Source Provenance Spoofing (Confluence chunk cited as Jira ticket).
3. Version Rollback Attack (older version downgrades and supersedes newer version).
4. Future/Unindexed Version Spoofing (citation claiming version 99 when version 1 is indexed).
5. Jira Staleness Evasion (tickets with Rejected, Cancelled, or Won't Fix status).
6. Temporal Decay Boundaries (180 vs 181 days).
7. PreFlightValidator Check 4 Metadata Stripping (blinding citation gate to versions/staleness).
8. GitLab Push Webhook Persistence Failure (blobs fetched but never stored in repo).
9. Quote Grounding Token Padding Bypass (SequenceMatcher >= 0.98).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock

import httpx
import pytest

from context_scout.agents.citation_gate import (
    CitationFailureCode,
    DocumentCitationGate,
)
from context_scout.agents.preflight_gate import PreFlightValidator
from context_scout.db.repositories import InMemoryRLSRepository, PostgresRLSRepository
from context_scout.db.schema import ASTSymbolRecord, CodeFileRecord, DocChunkRecord
from context_scout.ingestion.doc_parser import DocumentChunk
from context_scout.plugins.connectors.confluence import (
    ConfluenceConnectorPlugin,
    ConfluencePageDTO,
)
from context_scout.plugins.connectors.gitlab import GitLabConnectorPlugin
from context_scout.queue.task_queue import QueueTask
from context_scout.schemas.contracts import (
    DocCitationSpan,
    MarketRegime,
    ModificationIntent,
    VerbatimCodeSlice,
)


def _make_chunk(
    doc_id: str,
    title: str,
    content: str,
    source_uri: str,
    metadata: dict | None = None,
    is_stale: bool = False,
) -> DocumentChunk:
    meta = dict(metadata or {})
    if is_stale:
        meta["is_stale"] = True
    return DocumentChunk(
        chunk_id=uuid.uuid4(),
        doc_id=doc_id,
        doc_title=title,
        source_uri=source_uri,
        market_scope=MarketRegime.GLOBAL_MULTI,
        section_heading="Overview",
        verbatim_content=content,
        formulas_latex=[],
        tables_markdown=[],
        token_count_approx=len(content.split()),
        metadata_json=meta,
    )


# =============================================================================
# 1. CITATION FORGERY & PROVENANCE SPOOFING
# =============================================================================


def test_adversarial_confluence_page_id_forgery() -> None:
    """Stress Test: Citation provides a valid chunk_id but fabricates page_id.

    SECURITY REQUIREMENT: Gate MUST reject citations whose page_id does not match
    the chunk's actual page_id or doc_id.
    """
    gate = DocumentCitationGate(strict_freshness=True)
    chunk = _make_chunk(
        doc_id="10928374",
        title="Auth Architecture RFC-82",
        content="All microservices must implement OAuth 2.0 PKCE flow with 15-minute rotation.",
        source_uri="https://wiki.corp.internal/spaces/ENG/pages/10928374",
        metadata={"version": 3, "space_key": "ENG", "page_id": "10928374"},
    )

    # Forged citation: points to chunk.chunk_id but claims page_id='99999999'
    forged_citation = DocCitationSpan(
        chunk_id=str(chunk.chunk_id),
        doc_id="99999999",
        doc_title="Forged Document",
        source_uri="https://wiki.corp.internal/spaces/ENG/pages/99999999",
        section_heading="Overview",
        verbatim_quote=(
            "All microservices must implement OAuth 2.0 PKCE flow with 15-minute rotation."
        ),
        version="3",
        page_id="99999999",
    )

    res = gate.verify_citations([forged_citation], [chunk])
    # The gate should reject this citation forgery!
    assert res.passed is False, "Citation with fabricated page_id unexpectedly passed gate!"


def test_adversarial_jira_ticket_key_forgery() -> None:
    """Stress Test: Citation provides a valid chunk_id but fabricates Jira ticket_key.

    SECURITY REQUIREMENT: Gate MUST reject citations whose ticket_key does not match
    the chunk's actual issue_key.
    """
    gate = DocumentCitationGate(strict_freshness=True)
    chunk = _make_chunk(
        doc_id="ENG-4092",
        title="Checkout Error Resolution",
        content="Fix NullReferenceException in PaymentGateway by null-checking client secret.",
        source_uri="https://jira.corp.internal/browse/ENG-4092",
        metadata={"ticket_status": "In Progress", "project_key": "ENG", "issue_key": "ENG-4092"},
    )

    # Forged citation: points to chunk.chunk_id but claims ticket_key='SEC-9999'
    forged_citation = DocCitationSpan(
        chunk_id=str(chunk.chunk_id),
        doc_id="SEC-9999",
        doc_title="Fabricated Security Ticket",
        source_uri="https://jira.corp.internal/browse/SEC-9999",
        section_heading="Overview",
        verbatim_quote=(
            "Fix NullReferenceException in PaymentGateway by null-checking client secret."
        ),
        ticket_key="SEC-9999",
    )

    res = gate.verify_citations([forged_citation], [chunk])
    # The gate should reject this ticket key forgery!
    assert res.passed is False, "Citation with fabricated ticket_key unexpectedly passed gate!"


def test_adversarial_cross_source_citation_spoofing() -> None:
    """Stress Test: Quoting Confluence document while claiming it is a Jira ticket."""
    gate = DocumentCitationGate(strict_freshness=True)
    chunk = _make_chunk(
        doc_id="10928374",
        title="Architecture Guide",
        content="Service uses PostgreSQL 17 with PgBouncer connection pooling.",
        source_uri="https://wiki.corp.internal/spaces/ENG/pages/10928374",
        metadata={"version": 1, "space_key": "ENG", "page_id": "10928374"},
    )

    # Citation claims to be a Jira ticket ENG-1001 for a Confluence chunk
    spoofed_citation = DocCitationSpan(
        chunk_id=str(chunk.chunk_id),
        doc_id="ENG-1001",
        doc_title="Jira Ticket ENG-1001",
        source_uri="https://jira.corp.internal/browse/ENG-1001",
        section_heading="Overview",
        verbatim_quote="Service uses PostgreSQL 17 with PgBouncer connection pooling.",
        ticket_key="ENG-1001",
    )

    res = gate.verify_citations([spoofed_citation], [chunk])
    assert res.passed is False, "Cross-source citation spoofing unexpectedly passed gate!"


# =============================================================================
# 2. VERSION ATTACKS & ROLLBACK EXPLOITATION
# =============================================================================


def test_adversarial_version_rollback_attack() -> None:
    """Stress Test: Syncing older version v1 when newer version v2 is indexed.

    SECURITY REQUIREMENT: An older version sync MUST NOT mark newer versions as stale
    or overwrite newer indexed content.
    """
    repo = InMemoryRLSRepository()
    plugin = ConfluenceConnectorPlugin()

    # 1. Sync version 2
    v2_page = ConfluencePageDTO(
        page_id="100",
        space_key="ENG",
        title="Security Guidelines",
        version_number=2,
        author_username="security_lead",
        source_url="http://mock/100",
        body_storage="<p>Version 2: Require TLS 1.3 and mTLS.</p>",
        created_at=datetime.now(UTC),
        last_modified_at=datetime.now(UTC),
    )
    plugin.sync_space("default", "ENG", repo, current_pages=[v2_page])

    v2_chunks = [c for c in repo._docs.values() if c.version_str == "2"]
    assert len(v2_chunks) >= 1
    assert v2_chunks[0].is_stale is False

    # 2. Attack: Sync older version 1 (rollback or out-of-order webhook)
    v1_page = ConfluencePageDTO(
        page_id="100",
        space_key="ENG",
        title="Security Guidelines",
        version_number=1,
        author_username="attacker",
        source_url="http://mock/100",
        body_storage="<p>Version 1: Allow plaintext HTTP.</p>",
        created_at=datetime.now(UTC),
        last_modified_at=datetime.now(UTC),
    )
    plugin.sync_space("default", "ENG", repo, current_pages=[v1_page])

    v2_chunks_after = [c for c in repo._docs.values() if c.version_str == "2"]
    # v2 must NOT be marked stale by v1!
    assert v2_chunks_after[0].is_stale is False, (
        f"Version 2 was marked stale by version 1 rollback: {v2_chunks_after[0].stale_reason}"
    )


def test_adversarial_unindexed_future_version_spoofing() -> None:
    """Stress Test: Citation claims unindexed future version (e.g. v99 when v1 is indexed)."""
    gate = DocumentCitationGate(strict_freshness=True)
    chunk = _make_chunk(
        doc_id="10928374",
        title="Architecture",
        content="Production system runs with replicas=5.",
        source_uri="https://wiki.corp.internal/spaces/ENG/pages/10928374",
        metadata={"version": 1, "page_id": "10928374"},
    )

    # Citation claims future unverified version 99
    future_cit = DocCitationSpan(
        chunk_id=str(chunk.chunk_id),
        doc_id="10928374",
        doc_title="Architecture",
        source_uri="https://wiki.corp.internal/spaces/ENG/pages/10928374",
        section_heading="Overview",
        verbatim_quote="Production system runs with replicas=5.",
        version="99",  # Chunk is only version 1!
        page_id="10928374",
    )

    res = gate.verify_citations([future_cit], [chunk])
    assert res.passed is False, (
        "Citation with fabricated future version 99 unexpectedly passed gate!"
    )


# =============================================================================
# 3. JIRA STALENESS EVASION & STATUS MISMATCH
# =============================================================================


@pytest.mark.parametrize("status_val", ["Rejected", "Cancelled", "Won't Fix"])
def test_adversarial_jira_terminal_status_evasion(status_val: str) -> None:
    """Stress Test: Jira ticket with terminal/obsolete status passes citation gate."""
    gate = DocumentCitationGate(strict_freshness=True)
    chunk = _make_chunk(
        doc_id="ENG-500",
        title="Discarded Feature Spec",
        content="Feature proposal for legacy protocol support was discarded.",
        source_uri="https://jira.corp.internal/browse/ENG-500",
        metadata={
            "project_key": "ENG",
            "issue_key": "ENG-500",
            "ticket_status": status_val,
            "status": status_val,
        },
    )

    cit = DocCitationSpan(
        chunk_id=str(chunk.chunk_id),
        doc_id="ENG-500",
        doc_title="Discarded Feature Spec",
        source_uri="https://jira.corp.internal/browse/ENG-500",
        section_heading="Overview",
        verbatim_quote="Feature proposal for legacy protocol support was discarded.",
        ticket_key="ENG-500",
    )

    res = gate.verify_citations([cit], [chunk])
    assert res.passed is False, f"Jira ticket with status '{status_val}' unexpectedly passed gate!"


# =============================================================================
# 4. PREFLIGHT VALIDATOR INTEGRATION & METADATA STRIPPING
# =============================================================================


def test_adversarial_preflight_validator_blinding() -> None:
    """Stress Test: PreFlightValidator strips chunk metadata, blinding citation gate.

    REQUIREMENT: PreFlightValidator Check 4 MUST reject superseded versions and stale chunks.
    """
    repo = InMemoryRLSRepository()
    validator = PreFlightValidator(repo=repo)

    chunk_id = str(uuid.uuid4())
    content = "Microservices must enforce mutual TLS authentication across cluster boundaries."

    # Citation claims superseded version 1
    cit = DocCitationSpan(
        chunk_id=chunk_id,
        doc_id="100",
        doc_title="ADR mTLS",
        source_uri="https://wiki.corp.internal/100",
        section_heading="Security",
        verbatim_quote=content,
        version="1",  # Superseded! Latest is version 5
        page_id="100",
    )

    code_slice = VerbatimCodeSlice(
        repo_name="core_repo",
        file_path="src/gateway.py",
        start_line=1,
        end_line=2,
        qualified_symbol="core_repo:src/gateway.py:gateway",
        selection_reason="context",
        modification_intent=ModificationIntent.REFERENCE_ONLY,
        git_commit_sha="abc1234",
        git_blob_sha="def5678",
        verbatim_code_with_lines="   1 | def gateway():\n   2 |     pass",
    )

    # Chunk record in corpus with version 5
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
        metadata_json={"version": 5, "page_id": "100"},
        allowed_scopes=["public_eng"],
        embedding=[0.0] * 768,
        source_type="confluence",
        version_str="5",
        is_stale=False,
    )

    report = validator.validate(
        code_slices=[code_slice],
        user_prompt="Refactor gateway authentication with ADR guidance",
        supporting_doc_citations=[cit],
        doc_chunks=[chunk_rec],
        repo=repo,
    )

    # Check 4 should fail because version 1 is superseded by version 5!
    assert report.citations_verified_100_pct is False, (
        "PreFlightValidator Check 4 allowed superseded version 1 to pass!"
    )


# =============================================================================
# 5. GITLAB CONNECTOR PUSH WEBHOOK PERSISTENCE
# =============================================================================


def test_adversarial_gitlab_push_webhook_blob_persistence() -> None:
    """Stress Test: GitLab push webhook must persist updated code files to repository.

    REQUIREMENT: When push webhook processes modified files, repository MUST contain
    the updated CodeFileRecord.
    """
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "/repository/compare" in url:
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
                            "diff": "+# updated service\n",
                        }
                    ]
                },
            )
        if "src%2Fservice.py/raw" in url:
            return httpx.Response(200, content=b"def service():\n    return True\n")
        return httpx.Response(404, json={})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://gitlab.example.com")
    plugin = GitLabConnectorPlugin(base_url="https://gitlab.example.com", client=client)
    repo = InMemoryRLSRepository()

    task = QueueTask(
        task_id=uuid.uuid4(),
        domain_id="default",
        event_type="gitlab_push",
        payload={
            "project": {
                "id": 42,
                "path_with_namespace": "trading/engine",
                "default_branch": "main",
            },
            "before": "sha_old_1111",
            "after": "sha_new_2222",
            "ref": "refs/heads/main",
            "commits": [{"id": "sha_new_2222"}],
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

    res = plugin.process_push_webhook(task, repo)
    assert res.items_indexed >= 1

    # Verify repository actually has the code file
    files = getattr(repo, "_files", {})
    assert len(files) >= 1, (
        f"GitLab push webhook claimed {res.items_indexed} items indexed, "
        f"but repo has 0 code files!"
    )

    # Permanent Regression Assertion: Verify AST symbols are actually extracted and persisted
    # (Catches sym.kind vs sym.symbol_kind AttributeError swallowed by silent try-except)
    symbols = getattr(repo, "_symbols", {})
    assert len(symbols) >= 1, (
        "GitLab push webhook failed to persist ASTSymbolRecords into repository! "
        "(Check for sym.kind vs sym.symbol_kind AttributeError)"
    )
    first_sym = next(iter(symbols.values()))
    assert first_sym.symbol_kind == "function"
    assert "service" in first_sym.qualified_name


# =============================================================================
# 6. STALENESS DEFAULTS & QUOTE PADDING BYPASS
# =============================================================================


def test_adversarial_default_strict_freshness_permits_stale_citations() -> None:
    """Stress Test: Default DocumentCitationGate permits stale chunks to pass.

    REQUIREMENT: Stale content (temporal decay > 180 days or closed tickets) MUST cause
    abstention, not silent pass.
    """
    gate = DocumentCitationGate()  # Default instance without strict_freshness=True
    chunk = _make_chunk(
        doc_id="10928374",
        title="Decayed ADR",
        content="Outdated crypto parameters from 2020.",
        source_uri="https://wiki.corp.internal/10928374",
        metadata={"is_stale": True, "stale_reason": "Page last updated 250 days ago (> 180 days)"},
        is_stale=True,
    )

    cit = DocCitationSpan(
        chunk_id=str(chunk.chunk_id),
        doc_id="10928374",
        doc_title="Decayed ADR",
        source_uri="https://wiki.corp.internal/10928374",
        section_heading="Overview",
        verbatim_quote="Outdated crypto parameters from 2020.",
        version="1",
        page_id="10928374",
    )

    res = gate.verify_citations([cit], [chunk])
    # Gate must abstain / fail on stale chunk!
    assert res.passed is False, "Default citation gate unexpectedly passed stale chunk!"


def test_adversarial_token_padding_quote_hallucination_bypass() -> None:
    """Stress Test: A long quote with 1 critical substituted word bypasses 0.98 threshold.

    REQUIREMENT: Verbatim grounding must not allow fabricated instructions or altered
    constants by padding surrounding verbatim tokens.
    """
    gate = DocumentCitationGate()

    # 100-token source text
    source_words = [f"token_{i}" for i in range(100)]
    source_words[50] = "ALLOW_UNAUTHENTICATED"
    source_text = " ".join(source_words)

    chunk = _make_chunk(
        doc_id="200",
        title="Security Config",
        content=source_text,
        source_uri="https://wiki.corp.internal/200",
        metadata={"version": 1, "page_id": "200"},
    )

    # Malicious quote: altered word 50 to DENY_UNAUTHENTICATED, but surrounded by 99 verbatim tokens
    hallucinated_words = list(source_words)
    hallucinated_words[50] = "DENY_UNAUTHENTICATED"
    hallucinated_quote = " ".join(hallucinated_words)

    cit = DocCitationSpan(
        chunk_id=str(chunk.chunk_id),
        doc_id="200",
        doc_title="Security Config",
        source_uri="https://wiki.corp.internal/200",
        section_heading="Overview",
        verbatim_quote=hallucinated_quote,
        version="1",
        page_id="200",
    )

    res = gate.verify_citations([cit], [chunk])
    assert res.passed is False, (
        f"Altered quote with 1 fabricated token unexpectedly passed gate "
        f"(overlap={res.details[0].overlap_ratio})!"
    )


def test_adversarial_gitlab_push_webhook_renamed_file_lifecycle() -> None:
    """Stress Test: GitLab push webhook must remove old_path upon file rename, avoiding ghost files.

    REQUIREMENT: When a push webhook event contains diff.renamed_file=True,
    diff.old_path must be deleted from repository index and diff.new_path must be indexed.
    """
    repo = InMemoryRLSRepository()
    old_id = uuid.uuid4()
    repo._files[old_id] = CodeFileRecord(
        file_id=old_id,
        domain_id="default",
        repo_name="trading/engine",
        file_path="src/old_service.py",
        git_commit_sha="sha_old_1111",
        git_blob_sha="blob_old_1111",
        language="python",
        allowed_scopes=["trading_engine:reporter"],
        raw_content="def old_service():\n    return False\n",
        line_count=2,
    )
    sym_id = uuid.uuid4()
    repo._symbols[sym_id] = ASTSymbolRecord(
        symbol_id=sym_id,
        file_id=old_id,
        domain_id="default",
        qualified_name="trading/engine::old_service",
        symbol_kind="function",
        start_line=1,
        end_line=2,
        signature_skeleton="def old_service()",
        contextual_summary="",
        allowed_scopes=["trading_engine:reporter"],
        embedding=[0.0] * 768,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "/repository/compare" in url:
            return httpx.Response(
                200,
                json={
                    "diffs": [
                        {
                            "old_path": "src/old_service.py",
                            "new_path": "src/new_service.py",
                            "new_file": False,
                            "renamed_file": True,
                            "deleted_file": False,
                            "diff": "rename from src/old_service.py to src/new_service.py\n",
                        }
                    ]
                },
            )
        if "src%2Fnew_service.py/raw" in url:
            return httpx.Response(200, content=b"def new_service():\n    return True\n")
        return httpx.Response(404, json={})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://gitlab.example.com")
    plugin = GitLabConnectorPlugin(base_url="https://gitlab.example.com", client=client)

    task = QueueTask(
        task_id=uuid.uuid4(),
        domain_id="default",
        event_type="gitlab_push",
        payload={
            "project": {
                "id": 42,
                "path_with_namespace": "trading/engine",
                "default_branch": "main",
            },
            "before": "sha_old_1111",
            "after": "sha_new_2222",
            "ref": "refs/heads/main",
            "commits": [{"id": "sha_new_2222"}],
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

    res = plugin.process_push_webhook(task, repo)
    assert res.items_deleted >= 1, "Expected items_deleted >= 1 for renamed file"
    assert res.items_indexed >= 1, "Expected items_indexed >= 1 for new file path"

    file_paths = [f.file_path for f in repo._files.values()]
    assert "src/old_service.py" not in file_paths, (
        f"Ghost file detected: 'src/old_service.py' was not removed after rename! "
        f"Found files: {file_paths}"
    )
    assert "src/new_service.py" in file_paths, "New file path was not indexed after rename"

    symbols = list(repo._symbols.values())
    sym_names = [s.qualified_name for s in symbols]
    assert "trading/engine::old_service" not in sym_names, (
        "Ghost AST symbol from renamed file was not removed!"
    )
    assert "trading/engine::new_service" in sym_names, (
        "New AST symbol from renamed file was not indexed!"
    )


def test_adversarial_gitlab_push_webhook_deletion_orphan_symbols() -> None:
    """Stress Test: GitLab push webhook file deletion must purge ASTSymbolRecords.

    REQUIREMENT: When a file is deleted via push webhook, all associated AST symbols
    must be removed from the repository to prevent orphan symbols.
    """
    repo = InMemoryRLSRepository()
    file_id = uuid.uuid4()
    repo._files[file_id] = CodeFileRecord(
        file_id=file_id,
        domain_id="default",
        repo_name="trading/engine",
        file_path="src/dead_service.py",
        git_commit_sha="sha_old_1111",
        git_blob_sha="blob_dead_1111",
        language="python",
        allowed_scopes=["trading_engine:reporter"],
        raw_content="def dead_func():\n    pass\n",
        line_count=2,
    )
    sym_id = uuid.uuid4()
    repo._symbols[sym_id] = ASTSymbolRecord(
        symbol_id=sym_id,
        file_id=file_id,
        domain_id="default",
        qualified_name="trading/engine::dead_func",
        symbol_kind="function",
        start_line=1,
        end_line=2,
        signature_skeleton="def dead_func()",
        contextual_summary="",
        allowed_scopes=["trading_engine:reporter"],
        embedding=[0.0] * 768,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "/repository/compare" in url:
            return httpx.Response(
                200,
                json={
                    "diffs": [
                        {
                            "old_path": "src/dead_service.py",
                            "new_path": "src/dead_service.py",
                            "new_file": False,
                            "renamed_file": False,
                            "deleted_file": True,
                            "diff": "deleted file mode 100644\n",
                        }
                    ]
                },
            )
        return httpx.Response(404, json={})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://gitlab.example.com")
    plugin = GitLabConnectorPlugin(base_url="https://gitlab.example.com", client=client)

    task = QueueTask(
        task_id=uuid.uuid4(),
        domain_id="default",
        event_type="gitlab_push",
        payload={
            "project": {
                "id": 42,
                "path_with_namespace": "trading/engine",
                "default_branch": "main",
            },
            "before": "sha_old_1111",
            "after": "sha_new_2222",
            "ref": "refs/heads/main",
            "commits": [{"id": "sha_new_2222"}],
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

    res = plugin.process_push_webhook(task, repo)
    assert res.items_deleted >= 1

    assert len(repo._files) == 0, f"Expected 0 files remaining, found {len(repo._files)}"
    assert len(repo._symbols) == 0, (
        f"Orphan AST symbols detected! Expected 0 symbols remaining, found {len(repo._symbols)}: "
        f"{[s.qualified_name for s in repo._symbols.values()]}"
    )


def test_adversarial_gitlab_push_webhook_postgres_deletion() -> None:
    """Stress Test: GitLab push webhook handles PostgreSQL repository deletion correctly.

    REQUIREMENT: When running against PostgresRLSRepository, file deletions must execute
    valid deletion operations (such as delete_code_file_and_symbols, delete_code_file,
    or SQL cascade) rather than failing silently on non-existent methods.
    """
    mock_repo = MagicMock(spec=PostgresRLSRepository)
    mock_conn = MagicMock()
    mock_repo.upsert_repository = MagicMock()
    mock_repo.delete_code_file_and_symbols = MagicMock()

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "/repository/compare" in url:
            return httpx.Response(
                200,
                json={
                    "diffs": [
                        {
                            "old_path": "src/removed.py",
                            "new_path": "src/removed.py",
                            "new_file": False,
                            "renamed_file": False,
                            "deleted_file": True,
                            "diff": "deleted file\n",
                        }
                    ]
                },
            )
        return httpx.Response(404, json={})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://gitlab.example.com")
    plugin = GitLabConnectorPlugin(base_url="https://gitlab.example.com", client=client)

    task = QueueTask(
        task_id=uuid.uuid4(),
        domain_id="default",
        event_type="gitlab_push",
        payload={
            "project": {
                "id": 42,
                "path_with_namespace": "trading/engine",
                "default_branch": "main",
            },
            "before": "sha_old_1111",
            "after": "sha_new_2222",
            "ref": "refs/heads/main",
            "commits": [],
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

    res = plugin.process_push_webhook(task, mock_repo, conn=mock_conn)
    assert res.items_deleted == 1
    assert mock_repo.delete_code_file_and_symbols.called or mock_conn.execute.called, (
        "Neither delete_code_file_and_symbols nor conn.execute was called on PostgreSQL repo!"
    )


def test_adversarial_confluence_version_string_formatting_bypass() -> None:
    """Stress Test: Formatted version strings ('v1', '1.0', 'draft') must not bypass
    the version gate.
    """
    gate = DocumentCitationGate(strict_freshness=True)
    chunk = _make_chunk(
        doc_id="100",
        title="ADR mTLS",
        content="TLS 1.3 is strictly required.",
        source_uri="https://wiki.corp.internal/100",
        metadata={"version": 5, "page_id": "100"},
    )

    # 1. Superseded versions formatted with prefixes ('v1', 'V1', 'v.1', '1.0', 'v1.0')
    # must be rejected
    for stale_ver in ["v1", "V1", "v.1", "1.0", "v1.0"]:
        cit = DocCitationSpan(
            chunk_id=str(chunk.chunk_id),
            doc_id="100",
            doc_title="ADR mTLS",
            source_uri="https://wiki.corp.internal/100",
            section_heading="Overview",
            verbatim_quote="TLS 1.3 is strictly required.",
            version=stale_ver,
            page_id="100",
        )
        res = gate.verify_citations([cit], [chunk])
        assert res.passed is False, f"Superseded version format '{stale_ver}' bypassed gate!"
        assert res.details[0].error_code == CitationFailureCode.STALE_SUPERSEDED_VERSION

    # 2. Non-numeric / invalid version strings ('draft', 'unreleased', '1.2.3')
    # must be rejected as MISSING_PROVENANCE
    for invalid_ver in ["draft", "unreleased", "1.2.3", "-1", "0"]:
        cit = DocCitationSpan(
            chunk_id=str(chunk.chunk_id),
            doc_id="100",
            doc_title="ADR mTLS",
            source_uri="https://wiki.corp.internal/100",
            section_heading="Overview",
            verbatim_quote="TLS 1.3 is strictly required.",
            version=invalid_ver,
            page_id="100",
        )
        res = gate.verify_citations([cit], [chunk])
        assert res.passed is False, (
            f"Invalid version string '{invalid_ver}' bypassed provenance check!"
        )
        assert res.details[0].error_code == CitationFailureCode.MISSING_PROVENANCE

    # 3. Legitimate current version formatted with 'v5' or '5' must pass
    for valid_ver in ["5", "v5", "V5"]:
        cit = DocCitationSpan(
            chunk_id=str(chunk.chunk_id),
            doc_id="100",
            doc_title="ADR mTLS",
            source_uri="https://wiki.corp.internal/100",
            section_heading="Overview",
            verbatim_quote="TLS 1.3 is strictly required.",
            version=valid_ver,
            page_id="100",
        )
        res = gate.verify_citations([cit], [chunk])
        assert res.passed is True, f"Valid version '{valid_ver}' unexpectedly failed gate!"


def test_adversarial_in_vocabulary_word_substitution_quote_bypass() -> None:
    """Stress Test: Swapping words that exist elsewhere in source text must NOT pass quote
    grounding.
    """
    gate = DocumentCitationGate(strict_freshness=True)

    # 100-token source text containing both ALLOW and DENY
    source_words = [f"token_{i}" for i in range(100)]
    source_words[10] = "DENY"
    source_words[50] = "ALLOW"
    source_text = " ".join(source_words)

    chunk = _make_chunk(
        doc_id="300",
        title="Access Policy RFC",
        content=source_text,
        source_uri="https://wiki.corp.internal/300",
        metadata={"version": 1, "page_id": "300"},
    )

    # Malicious quote: swaps ALLOW at index 50 with DENY (which exists at index 10)
    altered_words = list(source_words)
    altered_words[50] = "DENY"
    altered_quote = " ".join(altered_words)

    cit = DocCitationSpan(
        chunk_id=str(chunk.chunk_id),
        doc_id="300",
        doc_title="Access Policy RFC",
        source_uri="https://wiki.corp.internal/300",
        section_heading="Overview",
        verbatim_quote=altered_quote,
        version="1",
        page_id="300",
    )

    res = gate.verify_citations([cit], [chunk])
    assert res.passed is False, (
        f"In-vocabulary word substitution unexpectedly passed quote grounding "
        f"(overlap={res.details[0].overlap_ratio})!"
    )
    assert res.details[0].error_code in (
        CitationFailureCode.HALLUCINATED_QUOTE,
        CitationFailureCode.QUOTE_MISMATCH,
    )


def test_adversarial_jira_resolution_declined_evasion() -> None:
    """Stress Test: Jira ticket with resolution 'declined' must be rejected under
    strict freshness.
    """
    gate = DocumentCitationGate(strict_freshness=True)
    chunk = _make_chunk(
        doc_id="ENG-700",
        title="Declined Feature Request",
        content="Implementation details for legacy protocol gateway.",
        source_uri="https://jira.corp.internal/browse/ENG-700",
        metadata={
            "project_key": "ENG",
            "issue_key": "ENG-700",
            "ticket_status": "Done",
            "resolution": "declined",
        },
    )

    cit = DocCitationSpan(
        chunk_id=str(chunk.chunk_id),
        doc_id="ENG-700",
        doc_title="Declined Feature Request",
        source_uri="https://jira.corp.internal/browse/ENG-700",
        section_heading="Overview",
        verbatim_quote="Implementation details for legacy protocol gateway.",
        ticket_key="ENG-700",
    )

    res = gate.verify_citations([cit], [chunk])
    assert res.passed is False, "Jira ticket with resolution 'declined' bypassed strict freshness!"
    assert res.details[0].error_code == CitationFailureCode.STALE_TICKET_STATUS
