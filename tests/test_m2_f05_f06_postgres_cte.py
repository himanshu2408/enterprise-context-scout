"""Tests for Milestone 2 Features F05 (Catalog Tables) and F06 (Recursive CTE 1-Hop Closure)."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from context_scout.config import ScoutSettings
from context_scout.db.repositories import (
    ASTEdgeRecord,
    ASTSymbolRecord,
    CodeFileRecord,
    ConfluencePageRecord,
    InMemoryRLSRepository,
    JiraIssueRecord,
    PostgresRLSRepository,
    RepoCommitRecord,
    RepositoryRecord,
)
from context_scout.db.session import RLSConnectionManager
from context_scout.schemas.contracts import PrincipalContext, RoleType


@pytest.fixture
def principal_eng() -> PrincipalContext:
    return PrincipalContext(
        user_id="m2_tester",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng", "core_dev"],
    )


@pytest.fixture
def principal_other_tenant() -> PrincipalContext:
    return PrincipalContext(
        user_id="other_tenant_user",
        role=RoleType.ENGINEER,
        allowed_scopes=["restricted_finance"],
    )


@pytest.fixture
def in_memory_repo() -> InMemoryRLSRepository:
    return InMemoryRLSRepository()


def test_f05_in_memory_catalog_crud_and_rls(
    in_memory_repo: InMemoryRLSRepository,
    principal_eng: PrincipalContext,
    principal_other_tenant: PrincipalContext,
) -> None:
    """Verifies in-memory CRUD operations and scope filtering for catalog records."""
    domain_id = "test_domain_f05"

    # 1. Repository Record
    repo_rec = RepositoryRecord(
        repo_id=uuid4(),
        domain_id=domain_id,
        repo_name="core-service",
        source_type="gitlab",
        remote_url="https://gitlab.corp.net/core-service.git",
        default_branch="main",
        allowed_scopes=["core_dev"],
        head_commit_sha="a" * 40,
    )
    in_memory_repo.insert_repository(principal_eng, repo_rec)

    fetched_repo = in_memory_repo.get_repository(principal_eng, domain_id, "core-service")
    assert fetched_repo is not None
    assert fetched_repo.repo_name == "core-service"

    # Scope isolation: other tenant cannot view
    assert in_memory_repo.get_repository(principal_other_tenant, domain_id, "core-service") is None
    assert len(in_memory_repo.list_repositories(principal_other_tenant, domain_id)) == 0

    # 2. Repo Commit Record
    commit_rec = RepoCommitRecord(
        commit_id=uuid4(),
        repo_id=repo_rec.repo_id,
        commit_sha="a" * 40,
        parent_shas=["b" * 40],
        author_name="Dev One",
        author_email="dev1@corp.net",
        commit_message="feat: initial commit",
        committed_at=datetime.now(UTC),
    )
    in_memory_repo.insert_repo_commit(principal_eng, commit_rec)
    commits = in_memory_repo.get_repo_commits(principal_eng, repo_rec.repo_id)
    assert len(commits) == 1
    assert commits[0].commit_sha == "a" * 40

    # 3. Confluence Page Record
    page_rec = ConfluencePageRecord(
        page_id=uuid4(),
        domain_id=domain_id,
        confluence_page_id="123456",
        space_key="ENG",
        title="Architecture Overview",
        version_number=1,
        author_username="lead_arch",
        source_url="https://wiki.corp.net/display/ENG/Overview",
        allowed_scopes=["core_dev"],
        last_modified_at=datetime.now(UTC),
    )
    in_memory_repo.insert_confluence_page(principal_eng, page_rec)
    fetched_page = in_memory_repo.get_confluence_page(principal_eng, domain_id, "123456")
    assert fetched_page is not None
    assert fetched_page.title == "Architecture Overview"

    # 4. Jira Issue Record
    issue_rec = JiraIssueRecord(
        issue_id=uuid4(),
        domain_id=domain_id,
        issue_key="CORE-101",
        project_key="CORE",
        summary="Implement persistent CTE indexing",
        issue_type="Story",
        status="In Progress",
        source_url="https://jira.corp.net/browse/CORE-101",
        allowed_scopes=["core_dev"],
        updated_at=datetime.now(UTC),
    )
    in_memory_repo.insert_jira_issue(principal_eng, issue_rec)
    fetched_issue = in_memory_repo.get_jira_issue(principal_eng, domain_id, "CORE-101")
    assert fetched_issue is not None
    assert fetched_issue.summary == "Implement persistent CTE indexing"


def test_f06_in_memory_traverse_1hop_neighborhood(
    in_memory_repo: InMemoryRLSRepository,
    principal_eng: PrincipalContext,
) -> None:
    """Verifies traverse_1hop_neighborhood on in-memory twin."""
    domain_id = "test_domain_f06"
    scopes = ["core_dev"]

    file1_id = uuid4()
    file2_id = uuid4()
    file3_id = uuid4()

    in_memory_repo.insert_code_file(
        principal_eng,
        CodeFileRecord(
            file_id=file1_id,
            domain_id=domain_id,
            repo_name="core",
            file_path="src/a.py",
            git_commit_sha="c1",
            git_blob_sha="b1",
            language="python",
            allowed_scopes=scopes,
            raw_content="def func_a(): func_b()",
            line_count=1,
        ),
    )
    in_memory_repo.insert_code_file(
        principal_eng,
        CodeFileRecord(
            file_id=file2_id,
            domain_id=domain_id,
            repo_name="core",
            file_path="src/b.py",
            git_commit_sha="c1",
            git_blob_sha="b2",
            language="python",
            allowed_scopes=scopes,
            raw_content="def func_b(): func_c()",
            line_count=1,
        ),
    )
    in_memory_repo.insert_code_file(
        principal_eng,
        CodeFileRecord(
            file_id=file3_id,
            domain_id=domain_id,
            repo_name="core",
            file_path="src/c.py",
            git_commit_sha="c1",
            git_blob_sha="b3",
            language="python",
            allowed_scopes=scopes,
            raw_content="def func_c(): pass",
            line_count=1,
        ),
    )

    sym_a = ASTSymbolRecord(
        symbol_id=uuid4(),
        file_id=file1_id,
        domain_id=domain_id,
        qualified_name="core::a::func_a",
        symbol_kind="function",
        start_line=1,
        end_line=1,
        signature_skeleton="def func_a()",
        contextual_summary="Function A",
        allowed_scopes=scopes,
        embedding=[],
    )
    sym_b = ASTSymbolRecord(
        symbol_id=uuid4(),
        file_id=file2_id,
        domain_id=domain_id,
        qualified_name="core::b::func_b",
        symbol_kind="function",
        start_line=1,
        end_line=1,
        signature_skeleton="def func_b()",
        contextual_summary="Function B",
        allowed_scopes=scopes,
        embedding=[],
    )
    sym_c = ASTSymbolRecord(
        symbol_id=uuid4(),
        file_id=file3_id,
        domain_id=domain_id,
        qualified_name="core::c::func_c",
        symbol_kind="function",
        start_line=1,
        end_line=1,
        signature_skeleton="def func_c()",
        contextual_summary="Function C",
        allowed_scopes=scopes,
        embedding=[],
    )
    for s in (sym_a, sym_b, sym_c):
        in_memory_repo.insert_ast_symbol(principal_eng, s)

    # Edge: A -> B (calls)
    in_memory_repo.insert_ast_edge(
        principal_eng,
        ASTEdgeRecord(
            edge_id=uuid4(),
            domain_id=domain_id,
            source_symbol_id=sym_a.symbol_id,
            target_qualified_name="core::b::func_b",
            target_symbol_id=sym_b.symbol_id,
            edge_kind="CALLS",
            allowed_scopes=scopes,
        ),
    )
    # Edge: B -> C (USES_TYPE)
    in_memory_repo.insert_ast_edge(
        principal_eng,
        ASTEdgeRecord(
            edge_id=uuid4(),
            domain_id=domain_id,
            source_symbol_id=sym_b.symbol_id,
            target_qualified_name="core::c::func_c",
            target_symbol_id=sym_c.symbol_id,
            edge_kind="USES_TYPE",
            allowed_scopes=scopes,
        ),
    )

    # Query 1-hop neighborhood for B
    records = in_memory_repo.traverse_1hop_neighborhood(
        principal_eng,
        domain_id=domain_id,
        focus_symbol_id=sym_b.symbol_id,
        max_direct_callers=5,
    )
    assert len(records) > 0
    connected_ids = {r.symbol.symbol_id for r in records}
    assert sym_a.symbol_id in connected_ids  # inward caller
    assert sym_c.symbol_id in connected_ids  # outward type


def test_f05_f06_live_postgres_catalog_and_recursive_cte() -> None:
    """Verifies live PostgreSQL 17 execution of catalog tables and recursive CTE query."""
    settings = ScoutSettings()
    admin_mgr = RLSConnectionManager(settings.db_admin_url)
    app_mgr = RLSConnectionManager(settings.db_app_url)
    pg_repo = PostgresRLSRepository()

    domain_id = "trading_code_scout"
    scopes = ["public_eng"]

    admin_principal = PrincipalContext(
        user_id="admin_user",
        role=RoleType.ENGINEER,
        allowed_scopes=scopes,
    )
    app_principal = PrincipalContext(
        user_id="app_user",
        role=RoleType.ENGINEER,
        allowed_scopes=scopes,
    )

    with admin_mgr.scoped_transaction(admin_principal, enable_hnsw_iterative_scan=False) as conn:
        conn.execute(
            """
            INSERT INTO domain_packs (
                domain_id, display_name, lora_adapter_id,
                primary_truth_source, freshness_half_life_days, config_json
            )
            VALUES (
                'trading_code_scout', 'Trading Code Scout', 'gemma4_26b_code_scout',
                'code', 180, '{"m2_cte_test": true}'::jsonb
            )
            ON CONFLICT (domain_id) DO NOTHING;
            """
        )

    with app_mgr.scoped_transaction(app_principal, enable_hnsw_iterative_scan=False) as conn:
        # F05: Catalog Repository CRUD in Postgres
        repo_id = uuid4()
        repo_name = f"test-repo-{repo_id.hex[:6]}"
        repo_rec = RepositoryRecord(
            repo_id=repo_id,
            domain_id=domain_id,
            repo_name=repo_name,
            source_type="github",
            remote_url="https://github.com/corp/test.git",
            default_branch="main",
            allowed_scopes=scopes,
            head_commit_sha="c1",
        )
        pg_repo.insert_repository(conn, repo_rec)
        fetched_repo = pg_repo.get_repository(conn, domain_id, repo_name)
        assert fetched_repo is not None
        assert fetched_repo.repo_name == repo_name

        # Confluence Page in Postgres
        page_id = uuid4()
        conf_id = f"conf-{repo_id.hex[:6]}"
        page_rec = ConfluencePageRecord(
            page_id=page_id,
            domain_id=domain_id,
            confluence_page_id=conf_id,
            space_key="M2",
            title="CTE Specification",
            version_number=1,
            author_username="fde_lead",
            source_url="https://confluence.corp.net/display/M2/CTE",
            allowed_scopes=scopes,
            last_modified_at=datetime.now(UTC),
        )
        pg_repo.insert_confluence_page(conn, page_rec)
        fetched_page = pg_repo.get_confluence_page(conn, domain_id, conf_id)
        assert fetched_page is not None
        assert fetched_page.title == "CTE Specification"

        # Jira Issue in Postgres
        issue_id = uuid4()
        issue_key = f"SCOUT-{repo_id.hex[:4]}"
        issue_rec = JiraIssueRecord(
            issue_id=issue_id,
            domain_id=domain_id,
            issue_key=issue_key,
            project_key="SCOUT",
            summary="Recursive CTE validation",
            issue_type="Task",
            status="Closed",
            source_url="https://jira.corp.net/browse/SCOUT",
            allowed_scopes=scopes,
            updated_at=datetime.now(UTC),
        )
        pg_repo.insert_jira_issue(conn, issue_rec)
        fetched_issue = pg_repo.get_jira_issue(conn, domain_id, issue_key)
        assert fetched_issue is not None
        assert fetched_issue.summary == "Recursive CTE validation"

        # F06: Live Recursive CTE 1-Hop AST Closure
        fid_x = uuid4()
        fid_y = uuid4()
        file_x = CodeFileRecord(
            file_id=fid_x,
            domain_id=domain_id,
            repo_name=repo_rec.repo_name,
            file_path="src/router.py",
            git_commit_sha="c1",
            git_blob_sha="b1",
            language="python",
            allowed_scopes=scopes,
            raw_content="def route_order(): validate_order()",
            line_count=1,
        )
        file_y = CodeFileRecord(
            file_id=fid_y,
            domain_id=domain_id,
            repo_name=repo_rec.repo_name,
            file_path="src/validator.py",
            git_commit_sha="c1",
            git_blob_sha="b2",
            language="python",
            allowed_scopes=scopes,
            raw_content="def validate_order(): pass",
            line_count=1,
        )
        pg_repo.insert_code_file(conn, file_x)
        pg_repo.insert_code_file(conn, file_y)

        sym_x = ASTSymbolRecord(
            symbol_id=uuid4(),
            file_id=fid_x,
            domain_id=domain_id,
            qualified_name=f"{repo_rec.repo_name}::router::route_order",
            symbol_kind="function",
            start_line=1,
            end_line=1,
            signature_skeleton="def route_order()",
            contextual_summary="Routes order",
            allowed_scopes=scopes,
            embedding=[0.1] * 768,
        )
        sym_y = ASTSymbolRecord(
            symbol_id=uuid4(),
            file_id=fid_y,
            domain_id=domain_id,
            qualified_name=f"{repo_rec.repo_name}::validator::validate_order",
            symbol_kind="function",
            start_line=1,
            end_line=1,
            signature_skeleton="def validate_order()",
            contextual_summary="Validates order",
            allowed_scopes=scopes,
            embedding=[0.2] * 768,
        )
        pg_repo.insert_ast_symbol(conn, sym_x)
        pg_repo.insert_ast_symbol(conn, sym_y)

        pg_repo.insert_ast_edge(
            conn,
            ASTEdgeRecord(
                edge_id=uuid4(),
                domain_id=domain_id,
                source_symbol_id=sym_x.symbol_id,
                target_qualified_name=sym_y.qualified_name,
                target_symbol_id=sym_y.symbol_id,
                edge_kind="CALLS",
                allowed_scopes=scopes,
            ),
        )

        # Execute live PostgreSQL 17 recursive CTE
        records = pg_repo.traverse_1hop_neighborhood(
            conn,
            domain_id=domain_id,
            focus_symbol_id=sym_y.symbol_id,
            max_direct_callers=10,
        )
        assert len(records) > 0
        connected_ids = {r.symbol.symbol_id for r in records}
        assert sym_x.symbol_id in connected_ids  # caller recovered via CTE!
        assert any(r.intent_role == "direct_caller" for r in records)
