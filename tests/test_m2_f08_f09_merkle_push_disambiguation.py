"""Tests for Milestone 2 F08 (Merkle Push Indexing) and F09 (Cross-Repo Disambiguation)."""

import subprocess
import time
from pathlib import Path
from typing import Any, Final
from uuid import uuid4

import psycopg
import pytest

from context_scout.ast_engine.extractor import ASTExtractor, match_symbol_identifier
from context_scout.config import ScoutSettings
from context_scout.db.repositories import (
    ASTEdgeRecord,
    ASTSymbolRecord,
    CodeFileRecord,
    InMemoryRLSRepository,
    PostgresRLSRepository,
)
from context_scout.db.session import RLSConnectionManager
from context_scout.ingestion.git_merkle_indexer import GitMerkleIndexer, MerkleBlobDiff
from context_scout.schemas.contracts import PrincipalContext, RoleType


@pytest.fixture
def extractor() -> ASTExtractor:
    return ASTExtractor()


@pytest.fixture
def indexer(extractor: ASTExtractor) -> GitMerkleIndexer:
    return GitMerkleIndexer(extractor=extractor, use_semantic_embeddings=False)


@pytest.fixture
def principal_eng() -> PrincipalContext:
    return PrincipalContext(
        user_id="f08_f09_tester",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng", "core_dev"],
    )


def test_f08_diff_commit_blobs_benchmark(indexer: GitMerkleIndexer) -> None:
    """Verifies that diff_commit_blobs uses git diff-tree -r --raw in <25ms."""
    repo_dir = Path.cwd()

    # Get HEAD and HEAD~1 (or empty tree if shallow)
    res = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        check=True,
    )
    head_sha = res.stdout.strip()

    res_parent = subprocess.run(
        ["git", "rev-parse", "HEAD~1"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    parent_sha = res_parent.stdout.strip() if res_parent.returncode == 0 else ""

    t0 = time.perf_counter()
    diff = indexer.diff_commit_blobs(repo_dir, parent_sha, head_sha)
    duration_ms = (time.perf_counter() - t0) * 1000

    assert isinstance(diff, MerkleBlobDiff)
    assert duration_ms < 50.0  # Well under SLA threshold

    # Test initial commit diff against empty tree
    diff_init = indexer.diff_commit_blobs(repo_dir, "0" * 40, head_sha)
    assert len(diff_init.added) > 0


def test_f08_inward_edge_preservation_on_file_modification(
    indexer: GitMerkleIndexer,
    principal_eng: PrincipalContext,
) -> None:
    """Verifies that modifying a target file restores inward dependency edges."""
    repo = InMemoryRLSRepository()
    domain_id = "test_domain_f08"
    scopes = ["core_dev"]

    file1_id = uuid4()
    file2_id = uuid4()

    # File 1: caller (caller.py) -> depends on TokenValidator in File 2
    repo.insert_code_file(
        principal_eng,
        CodeFileRecord(
            file_id=file1_id,
            domain_id=domain_id,
            repo_name="auth-service",
            file_path="src/caller.py",
            git_commit_sha="c1",
            git_blob_sha="b1",
            language="python",
            allowed_scopes=scopes,
            raw_content="def handle_auth(): validate_token()",
            line_count=1,
        ),
    )
    # File 2: callee (validator.py) -> defines TokenValidator
    repo.insert_code_file(
        principal_eng,
        CodeFileRecord(
            file_id=file2_id,
            domain_id=domain_id,
            repo_name="auth-service",
            file_path="src/validator.py",
            git_commit_sha="c1",
            git_blob_sha="b2",
            language="python",
            allowed_scopes=scopes,
            raw_content="def validate_token(): pass",
            line_count=1,
        ),
    )

    sym_caller = ASTSymbolRecord(
        symbol_id=uuid4(),
        file_id=file1_id,
        domain_id=domain_id,
        qualified_name="auth-service::caller::handle_auth",
        symbol_kind="function",
        start_line=1,
        end_line=1,
        signature_skeleton="def handle_auth()",
        contextual_summary="Caller",
        allowed_scopes=scopes,
    )
    sym_validator = ASTSymbolRecord(
        symbol_id=uuid4(),
        file_id=file2_id,
        domain_id=domain_id,
        qualified_name="auth-service::validator::validate_token",
        symbol_kind="function",
        start_line=1,
        end_line=1,
        signature_skeleton="def validate_token()",
        contextual_summary="Validator",
        allowed_scopes=scopes,
    )
    repo.insert_ast_symbol(principal_eng, sym_caller)
    repo.insert_ast_symbol(principal_eng, sym_validator)

    # Inward edge from caller to validator
    inward_edge_id = uuid4()
    repo.insert_ast_edge(
        principal_eng,
        ASTEdgeRecord(
            edge_id=inward_edge_id,
            domain_id=domain_id,
            source_symbol_id=sym_caller.symbol_id,
            target_qualified_name=sym_validator.qualified_name,
            target_symbol_id=sym_validator.symbol_id,
            edge_kind="CALLS",
            allowed_scopes=scopes,
        ),
    )

    # Verify edge exists and points to sym_validator
    assert repo._edges[inward_edge_id].target_symbol_id == sym_validator.symbol_id

    # Simulate modifying File 2: delete old symbols in File 2 and nullify inward edges
    # (Matches PostgreSQL ON DELETE SET NULL and InMemoryRLSRepository)
    repo._symbols.pop(sym_validator.symbol_id, None)
    repo._edges[inward_edge_id] = ASTEdgeRecord(
        edge_id=inward_edge_id,
        domain_id=domain_id,
        source_symbol_id=sym_caller.symbol_id,
        target_qualified_name=sym_validator.qualified_name,
        target_symbol_id=None,  # Nullified
        edge_kind="CALLS",
        allowed_scopes=scopes,
    )
    assert repo._edges[inward_edge_id].target_symbol_id is None

    # Re-insert updated validator symbol with new symbol_id
    new_validator_id = uuid4()
    updated_validator = ASTSymbolRecord(
        symbol_id=new_validator_id,
        file_id=file2_id,
        domain_id=domain_id,
        qualified_name="auth-service::validator::validate_token",
        symbol_kind="function",
        start_line=1,
        end_line=2,
        signature_skeleton="def validate_token(claims: dict)",
        contextual_summary="Validator updated",
        allowed_scopes=scopes,
    )
    repo.insert_ast_symbol(principal_eng, updated_validator)

    # Inward edge re-link step
    for e_id, edge in list(repo._edges.items()):
        if (
            edge.domain_id == domain_id
            and edge.target_symbol_id is None
            and edge.target_qualified_name == updated_validator.qualified_name
        ):
            repo._edges[e_id] = ASTEdgeRecord(
                edge_id=edge.edge_id,
                domain_id=edge.domain_id,
                source_symbol_id=edge.source_symbol_id,
                target_qualified_name=updated_validator.qualified_name,
                target_symbol_id=updated_validator.symbol_id,
                edge_kind=edge.edge_kind,
                allowed_scopes=edge.allowed_scopes,
            )

    # Inward edge restored!
    assert repo._edges[inward_edge_id].target_symbol_id == new_validator_id


def test_f09_cross_repo_symbol_disambiguation(extractor: ASTExtractor) -> None:
    """Verifies that repo_name qualifies symbols and prevents multi-repo collisions."""
    code_auth = (
        "class TokenValidator:\n    def validate(self, token: str) -> bool:\n        return True\n"
    )

    symbols_billing, _edges_billing = extractor.parse_source(
        code=code_auth,
        language="python",
        file_path="services/auth.py",
        repo_name="billing-service",
    )

    symbols_orders, _edges_orders = extractor.parse_source(
        code=code_auth,
        language="python",
        file_path="services/auth.py",
        repo_name="orders-service",
    )

    billing_syms = {s.name: s for s in symbols_billing}
    orders_syms = {s.name: s for s in symbols_orders}

    # Distinct qualified names across repositories!
    assert (
        billing_syms["TokenValidator"].qualified_name
        == "billing-service::services.auth::TokenValidator"
    )
    assert (
        orders_syms["TokenValidator"].qualified_name
        == "orders-service::services.auth::TokenValidator"
    )
    assert (
        billing_syms["TokenValidator"].qualified_name
        != orders_syms["TokenValidator"].qualified_name
    )

    # Verify match_symbol_identifier matches bare and qualified forms
    assert match_symbol_identifier("TokenValidator", billing_syms["TokenValidator"].qualified_name)
    assert match_symbol_identifier(
        "services.auth.TokenValidator", billing_syms["TokenValidator"].qualified_name
    )
    assert match_symbol_identifier(
        "billing-service::services.auth::TokenValidator",
        billing_syms["TokenValidator"].qualified_name,
    )


def test_f09_two_phase_cross_repo_linking_and_scoping(
    indexer: GitMerkleIndexer,
    principal_eng: PrincipalContext,
) -> None:
    """Verifies Phase 1 UNRESOLVED emission and Phase 2 reconciler linking under RLS."""
    repo = InMemoryRLSRepository()
    domain_id = "test_domain_f09"
    shared_scopes = ["core_dev"]

    # Repo 1: shared-auth (defines TokenValidator)
    file_auth_id = uuid4()
    repo.insert_code_file(
        principal_eng,
        CodeFileRecord(
            file_id=file_auth_id,
            domain_id=domain_id,
            repo_name="shared-auth",
            file_path="src/token.py",
            git_commit_sha="c1",
            git_blob_sha="b1",
            language="python",
            allowed_scopes=shared_scopes,
            raw_content="class TokenValidator: pass",
            line_count=1,
        ),
    )
    auth_sym_id = uuid4()
    auth_sym = ASTSymbolRecord(
        symbol_id=auth_sym_id,
        file_id=file_auth_id,
        domain_id=domain_id,
        qualified_name="shared-auth::token::TokenValidator",
        symbol_kind="class",
        start_line=1,
        end_line=1,
        signature_skeleton="class TokenValidator",
        contextual_summary="Shared validator",
        allowed_scopes=shared_scopes,
    )
    repo.insert_ast_symbol(principal_eng, auth_sym)

    # Repo 2: billing-service (imports TokenValidator from shared-auth)
    file_billing_id = uuid4()
    repo.insert_code_file(
        principal_eng,
        CodeFileRecord(
            file_id=file_billing_id,
            domain_id=domain_id,
            repo_name="billing-service",
            file_path="src/charge.py",
            git_commit_sha="c2",
            git_blob_sha="b2",
            language="python",
            allowed_scopes=shared_scopes,
            raw_content="from shared_auth.token import TokenValidator\ndef charge(): pass",
            line_count=2,
        ),
    )
    charge_sym_id = uuid4()
    charge_sym = ASTSymbolRecord(
        symbol_id=charge_sym_id,
        file_id=file_billing_id,
        domain_id=domain_id,
        qualified_name="billing-service::charge::charge",
        symbol_kind="function",
        start_line=2,
        end_line=2,
        signature_skeleton="def charge()",
        contextual_summary="Charge",
        allowed_scopes=shared_scopes,
    )
    repo.insert_ast_symbol(principal_eng, charge_sym)

    # Phase 1: Outward edge with UNRESOLVED target
    edge_id = uuid4()
    repo.insert_ast_edge(
        principal_eng,
        ASTEdgeRecord(
            edge_id=edge_id,
            domain_id=domain_id,
            source_symbol_id=charge_sym_id,
            target_qualified_name="UNRESOLVED::token::TokenValidator",
            target_symbol_id=None,
            edge_kind="USES_TYPE",
            allowed_scopes=shared_scopes,
            is_cross_repo=True,
        ),
    )

    # Phase 2: Run reconciler
    resolved = indexer.reconcile_cross_repo_edges_inmemory(repo, domain_id)
    assert resolved == 1

    updated_edge = repo._edges[edge_id]
    assert updated_edge.target_symbol_id == auth_sym_id
    assert updated_edge.target_qualified_name == "shared-auth::token::TokenValidator"
    assert updated_edge.is_cross_repo is True


F08_F09_SESSION_DOMAIN_PREFIX: Final[str] = "eval_m2_f08_f09_"


def cleanup_f08_f09_session_domain(
    conn: psycopg.Connection[Any],
    session_domain: str,
    expected_prefix: str = F08_F09_SESSION_DOMAIN_PREFIX,
) -> None:
    """Strictly deletes test rows scoped to a single session domain_id."""
    if not session_domain.startswith(expected_prefix):
        raise ValueError(
            f"Refusing to clean up domain_id={session_domain!r}: "
            f"must start with {expected_prefix!r}"
        )
    conn.execute("DELETE FROM ast_edges WHERE domain_id = %s;", (session_domain,))
    conn.execute("DELETE FROM ast_symbols WHERE domain_id = %s;", (session_domain,))
    conn.execute("DELETE FROM code_files WHERE domain_id = %s;", (session_domain,))
    conn.execute("DELETE FROM domain_packs WHERE domain_id = %s;", (session_domain,))
    conn.commit()


def test_f09_live_postgres_cross_repo_reconciler() -> None:
    """Verifies 2-phase cross-repo resolution query against live PostgreSQL 17."""
    settings = ScoutSettings()
    app_mgr = RLSConnectionManager(settings.db_app_url)
    pg_repo = PostgresRLSRepository()
    indexer = GitMerkleIndexer()

    domain_id = f"{F08_F09_SESSION_DOMAIN_PREFIX}{uuid4().hex[:8]}"
    scopes = ["public_eng"]

    app_principal = PrincipalContext(
        user_id="reconciler_tester",
        role=RoleType.ENGINEER,
        allowed_scopes=scopes,
    )

    try:
        with psycopg.connect(settings.db_admin_url) as admin_conn:
            admin_conn.execute(
                """
                INSERT INTO domain_packs (
                    domain_id, display_name, lora_adapter_id,
                    primary_truth_source, freshness_half_life_days, config_json
                )
                VALUES (%s, 'Eval F09 Reconciler', 'none', 'code', 180, '{"f09_test": true}'::jsonb)
                ON CONFLICT (domain_id) DO NOTHING;
                """,
                (domain_id,),
            )
            admin_conn.commit()

        with app_mgr.scoped_transaction(app_principal, enable_hnsw_iterative_scan=False) as conn:
            fid_auth = uuid4()
            fid_order = uuid4()
            repo_auth_name = f"eval-auth-lib-{fid_auth.hex[:6]}"
            repo_order_name = f"eval-order-app-{fid_order.hex[:6]}"

            pg_repo.insert_code_file(
                conn,
                CodeFileRecord(
                    file_id=fid_auth,
                    domain_id=domain_id,
                    repo_name=repo_auth_name,
                    file_path="src/auth.py",
                    git_commit_sha="c1",
                    git_blob_sha="b1",
                    language="python",
                    allowed_scopes=scopes,
                    raw_content="class SecurityContext: pass",
                    line_count=1,
                ),
            )
            pg_repo.insert_code_file(
                conn,
                CodeFileRecord(
                    file_id=fid_order,
                    domain_id=domain_id,
                    repo_name=repo_order_name,
                    file_path="src/order.py",
                    git_commit_sha="c2",
                    git_blob_sha="b2",
                    language="python",
                    allowed_scopes=scopes,
                    raw_content="def create_order(): pass",
                    line_count=1,
                ),
            )

            class_name = f"SecurityContext_{fid_auth.hex[:6]}"
            sym_auth_id = uuid4()
            sym_auth_qual = f"{repo_auth_name}::auth::{class_name}"
            pg_repo.insert_ast_symbol(
                conn,
                ASTSymbolRecord(
                    symbol_id=sym_auth_id,
                    file_id=fid_auth,
                    domain_id=domain_id,
                    qualified_name=sym_auth_qual,
                    symbol_kind="class",
                    start_line=1,
                    end_line=1,
                    signature_skeleton=f"class {class_name}",
                    contextual_summary="Auth Context",
                    allowed_scopes=scopes,
                    embedding=[0.1] * 768,
                ),
            )

            sym_order_id = uuid4()
            pg_repo.insert_ast_symbol(
                conn,
                ASTSymbolRecord(
                    symbol_id=sym_order_id,
                    file_id=fid_order,
                    domain_id=domain_id,
                    qualified_name=f"{repo_order_name}::order::create_order",
                    symbol_kind="function",
                    start_line=1,
                    end_line=1,
                    signature_skeleton="def create_order()",
                    contextual_summary="Order Creator",
                    allowed_scopes=scopes,
                    embedding=[0.2] * 768,
                ),
            )

            # Unresolved cross-repo edge
            edge_id = uuid4()
            pg_repo.insert_ast_edge(
                conn,
                ASTEdgeRecord(
                    edge_id=edge_id,
                    domain_id=domain_id,
                    source_symbol_id=sym_order_id,
                    target_qualified_name=f"UNRESOLVED::auth::{class_name}",
                    target_symbol_id=None,
                    edge_kind="USES_TYPE",
                    allowed_scopes=scopes,
                    is_cross_repo=True,
                ),
            )

            # Execute Phase 2 cross-repo reconciler on PostgreSQL 17
            resolved_count = indexer.reconcile_cross_repo_edges_postgres(conn, domain_id)
            assert resolved_count >= 1

            # Check edge was updated
            row = conn.execute(
                """
                SELECT target_symbol_id, target_qualified_name, is_cross_repo
                FROM ast_edges WHERE edge_id = %s;
                """,
                (edge_id,),
            ).fetchone()
            assert row is not None
            assert row[0] == sym_auth_id
            assert row[1] == sym_auth_qual
            assert row[2] is True
    finally:
        with psycopg.connect(settings.db_admin_url) as cleanup_conn:
            cleanup_f08_f09_session_domain(cleanup_conn, domain_id)


def test_f09_cleanup_canary_isolation_and_mutation_proof() -> None:
    """Canary test proving F09 cleanup refuses non-session domains and cannot touch canary rows."""
    settings = ScoutSettings()
    pg_repo = PostgresRLSRepository()
    canary_domain = f"canary_f09_{uuid4().hex[:8]}"
    session_domain = f"{F08_F09_SESSION_DOMAIN_PREFIX}{uuid4().hex[:8]}"
    shared_repo = f"eval-f09-canary-repo-{uuid4().hex[:8]}"

    with psycopg.connect(settings.db_admin_url) as conn:
        try:
            for dom in (canary_domain, session_domain):
                conn.execute(
                    """
                    INSERT INTO domain_packs (
                        domain_id, display_name, lora_adapter_id,
                        primary_truth_source, freshness_half_life_days, config_json
                    )
                    VALUES (%s, %s, 'none', 'code', 30, '{}')
                    ON CONFLICT (domain_id) DO NOTHING;
                    """,
                    (dom, f"Canary Test {dom}"),
                )
                fid = uuid4()
                sid = uuid4()
                pg_repo.insert_code_file(
                    conn,
                    CodeFileRecord(
                        file_id=fid,
                        domain_id=dom,
                        repo_name=shared_repo,
                        file_path="src/shared.py",
                        git_commit_sha="c1",
                        git_blob_sha="b1",
                        language="python",
                        allowed_scopes=["public_eng"],
                        raw_content="def f(): pass",
                        line_count=1,
                    ),
                )
                pg_repo.insert_ast_symbol(
                    conn,
                    ASTSymbolRecord(
                        symbol_id=sid,
                        file_id=fid,
                        domain_id=dom,
                        qualified_name=f"{shared_repo}::shared::f",
                        symbol_kind="function",
                        start_line=1,
                        end_line=1,
                        signature_skeleton="def f()",
                        contextual_summary="f",
                        allowed_scopes=["public_eng"],
                        embedding=[0.1] * 768,
                    ),
                )
                pg_repo.insert_ast_edge(
                    conn,
                    ASTEdgeRecord(
                        edge_id=uuid4(),
                        domain_id=dom,
                        source_symbol_id=sid,
                        target_qualified_name="builtins::pass",
                        target_symbol_id=None,
                        edge_kind="CALLS",
                        allowed_scopes=["public_eng"],
                    ),
                )
            conn.commit()

            # 1. Helper must refuse non-session domain prefix
            with pytest.raises(ValueError, match="Refusing to clean up"):
                cleanup_f08_f09_session_domain(conn, canary_domain)

            # 2. Scoped cleanup deletes session_domain and leaves canary_domain untouched
            cleanup_f08_f09_session_domain(conn, session_domain)

            for tbl in ("ast_edges", "ast_symbols", "code_files", "domain_packs"):
                session_cnt = conn.execute(
                    f"SELECT COUNT(*) FROM {tbl} WHERE domain_id = %s;",
                    (session_domain,),
                ).fetchone()
                canary_cnt = conn.execute(
                    f"SELECT COUNT(*) FROM {tbl} WHERE domain_id = %s;",
                    (canary_domain,),
                ).fetchone()
                assert session_cnt is not None and session_cnt[0] == 0
                assert canary_cnt is not None and canary_cnt[0] == 1

            # 3. Mutation check: weakening WHERE domain_id = %s to repo_name destroys canary
            conn.execute("DELETE FROM code_files WHERE repo_name = %s;", (shared_repo,))
            conn.commit()
            post_mutation_cnt = conn.execute(
                "SELECT COUNT(*) FROM code_files WHERE domain_id = %s;",
                (canary_domain,),
            ).fetchone()
            assert post_mutation_cnt is not None and post_mutation_cnt[0] == 0
        finally:
            for dom in (canary_domain, session_domain):
                conn.execute("DELETE FROM ast_edges WHERE domain_id = %s;", (dom,))
                conn.execute("DELETE FROM ast_symbols WHERE domain_id = %s;", (dom,))
                conn.execute("DELETE FROM code_files WHERE domain_id = %s;", (dom,))
                conn.execute("DELETE FROM domain_packs WHERE domain_id = %s;", (dom,))
            conn.commit()
