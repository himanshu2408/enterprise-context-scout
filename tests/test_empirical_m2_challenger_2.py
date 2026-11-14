"""Empirical Adversarial Stress Test Suite for Milestone 2 Features F08 and F09.

Challenges:
1. Sub-50ms Merkle Diffing Benchmark across 20 simulated git commits (assert p95 < 50ms).
2. Cross-Repo Symbol Collision Disambiguation with N >= 100 identical symbol pairs.
3. Multi-Tenant Forced RLS Isolation on Cross-Repo Links.
4. Inward Edge Preservation on Modified and Deleted Symbols.
"""

import math
import subprocess
import tempfile
import time
from collections.abc import Generator
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

EMPIRICAL_M2_REPO_PREFIX: Final[str] = "empirical-m2-test-"


def teardown_empirical_m2_test_repos(
    conn: psycopg.Connection[Any],
    domain_id: str = "trading_code_scout",
    prefix: str = EMPIRICAL_M2_REPO_PREFIX,
) -> None:
    """Strictly deletes test-created repositories matching prefix under domain_id."""
    like_pattern = f"{prefix}%"
    with conn.cursor() as cur:
        cur.execute(
            """
            DELETE FROM ast_edges
            WHERE domain_id = %s AND (
                source_symbol_id IN (
                    SELECT symbol_id FROM ast_symbols
                    WHERE file_id IN (
                        SELECT file_id FROM code_files
                        WHERE domain_id = %s AND repo_name LIKE %s
                    )
                )
                OR target_symbol_id IN (
                    SELECT symbol_id FROM ast_symbols
                    WHERE file_id IN (
                        SELECT file_id FROM code_files
                        WHERE domain_id = %s AND repo_name LIKE %s
                    )
                )
            );
            """,
            (domain_id, domain_id, like_pattern, domain_id, like_pattern),
        )
        cur.execute(
            """
            DELETE FROM ast_symbols
            WHERE domain_id = %s AND file_id IN (
                SELECT file_id FROM code_files
                WHERE domain_id = %s AND repo_name LIKE %s
            );
            """,
            (domain_id, domain_id, like_pattern),
        )
        cur.execute(
            """
            DELETE FROM code_files
            WHERE domain_id = %s AND repo_name LIKE %s;
            """,
            (domain_id, like_pattern),
        )
    conn.commit()


@pytest.fixture
def clean_empirical_m2_test_repos(ephemeral_pg_domain: str) -> Generator[str, None, None]:
    """Provide an isolated ephemeral domain for empirical M2 tests with guaranteed cleanup."""
    yield ephemeral_pg_domain


@pytest.fixture
def extractor() -> ASTExtractor:
    return ASTExtractor()


@pytest.fixture
def indexer(extractor: ASTExtractor) -> GitMerkleIndexer:
    return GitMerkleIndexer(extractor=extractor, use_semantic_embeddings=False)


def _init_git_repo(repo_dir: Path) -> None:
    """Initializes a clean git repository for empirical testing."""
    subprocess.run(["git", "init", "-b", "main"], cwd=repo_dir, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.name", "Scout Empirical Tester"],
        cwd=repo_dir,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "tester@context-scout.internal"],
        cwd=repo_dir,
        capture_output=True,
        check=True,
    )


def _commit_all(repo_dir: Path, message: str) -> str:
    """Stages all changes and commits, returning the 40-character commit SHA."""
    subprocess.run(["git", "add", "-A"], cwd=repo_dir, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", message, "--allow-empty"],
        cwd=repo_dir,
        capture_output=True,
        check=True,
    )
    rev = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        check=True,
    )
    return rev.stdout.strip()


# ==============================================================================
# Challenge 1: Sub-50ms Merkle Diffing Benchmark (20 Simulated Commits, p95 < 50ms)
# ==============================================================================


def test_empirical_sub_50ms_merkle_diffing_benchmark_20_commits(
    indexer: GitMerkleIndexer,
) -> None:
    """Empirically stress-tests diff_commit_blobs across 20 distinct git commits.

    Asserts that p95 latency is strictly < 50.0 ms.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo_dir = Path(tmp_dir)
        _init_git_repo(repo_dir)

        # Baseline commit: create initial set of 15 files
        src_dir = repo_dir / "src"
        src_dir.mkdir(parents=True)
        for i in range(15):
            (src_dir / f"module_{i}.py").write_text(
                f"def compute_{i}(x: int) -> int:\n    return x * {i + 1}\n",
                encoding="utf-8",
            )
        commit_shas: list[str] = [_commit_all(repo_dir, "Commit 0: baseline 15 files")]

        # Generate 20 diverse, realistic commits: single-file, multi-file, add, delete, rename
        for c_idx in range(1, 21):
            if c_idx % 5 == 1:
                # Single file edit
                target_f = src_dir / f"module_{c_idx % 15}.py"
                target_f.write_text(
                    target_f.read_text(encoding="utf-8") + f"# update at commit {c_idx}\n",
                    encoding="utf-8",
                )
                msg = f"Commit {c_idx}: single-file update"
            elif c_idx % 5 == 2:
                # Multi-file edits (4 files modified)
                for f_num in range(4):
                    tf = src_dir / f"module_{(c_idx + f_num) % 15}.py"
                    tf.write_text(
                        tf.read_text(encoding="utf-8") + f"def helper_{c_idx}_{f_num}(): pass\n",
                        encoding="utf-8",
                    )
                msg = f"Commit {c_idx}: multi-file 4 edits"
            elif c_idx % 5 == 3:
                # File addition
                new_f = src_dir / f"new_feature_{c_idx}.py"
                new_f.write_text(
                    f"class Feature{c_idx}:\n    enabled: bool = True\n",
                    encoding="utf-8",
                )
                msg = f"Commit {c_idx}: added new_feature_{c_idx}.py"
            elif c_idx % 5 == 4:
                # File deletion or addition
                del_cand = src_dir / f"new_feature_{c_idx - 1}.py"
                if del_cand.exists():
                    del_cand.unlink()
                    msg = f"Commit {c_idx}: deleted new_feature_{c_idx - 1}.py"
                else:
                    (src_dir / f"extra_{c_idx}.py").write_text("extra = 1\n", encoding="utf-8")
                    msg = f"Commit {c_idx}: added extra_{c_idx}.py"
            else:
                # Heavy batch commit: 6 files modified + 1 added
                for f_num in range(6):
                    tf = src_dir / f"module_{(c_idx * 2 + f_num) % 15}.py"
                    tf.write_text(
                        tf.read_text(encoding="utf-8") + f"BATCH_{c_idx}_{f_num} = True\n",
                        encoding="utf-8",
                    )
                (src_dir / f"batch_add_{c_idx}.py").write_text("# batch add\n", encoding="utf-8")
                msg = f"Commit {c_idx}: heavy batch commit"

            sha = _commit_all(repo_dir, msg)
            commit_shas.append(sha)

        assert len(commit_shas) == 21  # Baseline + 20 commits

        # Warmup
        _ = indexer.diff_commit_blobs(repo_dir, commit_shas[0], commit_shas[1])

        # Benchmark all 20 transitions
        latencies_ms: list[float] = []
        diffs: list[MerkleBlobDiff] = []

        for i in range(1, 21):
            old_sha = commit_shas[i - 1]
            new_sha = commit_shas[i]

            t0 = time.perf_counter()
            diff = indexer.diff_commit_blobs(repo_dir, old_sha, new_sha)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            latencies_ms.append(elapsed_ms)
            diffs.append(diff)

            # Sanity assertions on diff correctness
            assert isinstance(diff, MerkleBlobDiff)
            total_ops = len(diff.added) + len(diff.modified) + len(diff.deleted) + len(diff.renamed)
            assert total_ops > 0, f"Commit transition {i} produced empty diff"

        # Statistical calculations
        latencies_ms.sort()
        n = len(latencies_ms)
        min_ms = latencies_ms[0]
        max_ms = latencies_ms[-1]
        mean_ms = sum(latencies_ms) / n
        median_ms = latencies_ms[n // 2]
        p95_idx = min(math.ceil(0.95 * n) - 1, n - 1)
        p95_ms = latencies_ms[p95_idx]

        print(
            f"\n[F08 Latency Benchmark] N={n} commits | min={min_ms:.2f}ms | "
            f"mean={mean_ms:.2f}ms | median={median_ms:.2f}ms | "
            f"p95={p95_ms:.2f}ms | max={max_ms:.2f}ms"
        )

        # SLA Gate: p95 must be < 50.0 ms
        assert p95_ms < 50.0, (
            f"p95 latency {p95_ms:.2f}ms exceeded SLA 50.0ms (latencies={latencies_ms})"
        )

        # Also verify empty tree diff (initial commit diffing)
        t_init0 = time.perf_counter()
        init_diff = indexer.diff_commit_blobs(repo_dir, "0" * 40, commit_shas[1])
        init_ms = (time.perf_counter() - t_init0) * 1000.0
        assert len(init_diff.added) >= 15
        assert init_ms < 50.0, f"Initial commit diff took {init_ms:.2f}ms (expected < 50ms)"


# ==============================================================================
# Challenge 2: Cross-Repo Collision Disambiguation (N >= 100 Identical Pairs)
# ==============================================================================


def test_empirical_cross_repo_symbol_collision_disambiguation_n100(
    extractor: ASTExtractor,
) -> None:
    """Stress-tests cross-repo collision disambiguation with N=100 identical symbol pairs.

    Verifies 0 collisions in ast_symbols, strictly isolated qualified names, and
    query routing accuracy across repo_alpha and repo_beta.
    """
    repo_inmem = InMemoryRLSRepository()
    domain_id = "test_collision_n100"
    principal = PrincipalContext(
        user_id="collision_tester",
        role=RoleType.ENGINEER,
        allowed_scopes=["alpha_scope", "beta_scope"],
    )

    n_pairs = 100
    file_alpha_id = uuid4()
    file_beta_id = uuid4()

    repo_inmem.insert_code_file(
        principal,
        CodeFileRecord(
            file_id=file_alpha_id,
            domain_id=domain_id,
            repo_name="repo_alpha",
            file_path="src/services/catalog.py",
            git_commit_sha="c_alpha_1",
            git_blob_sha="b_alpha_1",
            language="python",
            allowed_scopes=["alpha_scope"],
            raw_content="# alpha",
            line_count=100,
        ),
    )
    repo_inmem.insert_code_file(
        principal,
        CodeFileRecord(
            file_id=file_beta_id,
            domain_id=domain_id,
            repo_name="repo_beta",
            file_path="src/services/catalog.py",
            git_commit_sha="c_beta_1",
            git_blob_sha="b_beta_1",
            language="python",
            allowed_scopes=["beta_scope"],
            raw_content="# beta",
            line_count=100,
        ),
    )

    alpha_sym_map: dict[str, ASTSymbolRecord] = {}
    beta_sym_map: dict[str, ASTSymbolRecord] = {}

    for i in range(n_pairs):
        bare_name = f"ServiceCatalogItem_{i:03d}"
        code_stub = f"class {bare_name}:\n    pass\n"

        # Parse identical code under repo_alpha
        syms_alpha, _ = extractor.parse_source(
            code=code_stub,
            language="python",
            file_path="services/catalog.py",
            repo_name="repo_alpha",
        )
        assert len(syms_alpha) == 1
        sym_alpha = syms_alpha[0]
        expected_alpha_qual = f"repo_alpha::services.catalog::{bare_name}"
        assert sym_alpha.qualified_name == expected_alpha_qual

        rec_alpha = ASTSymbolRecord(
            symbol_id=uuid4(),
            file_id=file_alpha_id,
            domain_id=domain_id,
            qualified_name=sym_alpha.qualified_name,
            symbol_kind="class",
            start_line=1,
            end_line=3,
            signature_skeleton=f"class {bare_name}",
            contextual_summary=f"Catalog item {i} in alpha",
            allowed_scopes=["alpha_scope"],
        )
        repo_inmem.insert_ast_symbol(principal, rec_alpha)
        alpha_sym_map[bare_name] = rec_alpha

        # Parse identical code under repo_beta
        syms_beta, _ = extractor.parse_source(
            code=code_stub,
            language="python",
            file_path="services/catalog.py",
            repo_name="repo_beta",
        )
        assert len(syms_beta) == 1
        sym_beta = syms_beta[0]
        expected_beta_qual = f"repo_beta::services.catalog::{bare_name}"
        assert sym_beta.qualified_name == expected_beta_qual

        rec_beta = ASTSymbolRecord(
            symbol_id=uuid4(),
            file_id=file_beta_id,
            domain_id=domain_id,
            qualified_name=sym_beta.qualified_name,
            symbol_kind="class",
            start_line=1,
            end_line=3,
            signature_skeleton=f"class {bare_name}",
            contextual_summary=f"Catalog item {i} in beta",
            allowed_scopes=["beta_scope"],
        )
        repo_inmem.insert_ast_symbol(principal, rec_beta)
        beta_sym_map[bare_name] = rec_beta

    # 1. Total symbols stored must be exactly 2 * N (0 collisions / overwrites)
    domain_symbols = [s for s in repo_inmem._symbols.values() if s.domain_id == domain_id]
    assert len(domain_symbols) == 2 * n_pairs == 200, (
        f"Expected 200 symbols, got {len(domain_symbols)}"
    )

    all_qual_names = {s.qualified_name for s in domain_symbols}
    assert len(all_qual_names) == 200, "Qualified names collided across repositories!"

    # 2. Query routing accuracy: querying repo_alpha returns strictly alpha records
    for i in range(n_pairs):
        bare = f"ServiceCatalogItem_{i:03d}"
        q_alpha = f"repo_alpha::services.catalog::{bare}"
        q_beta = f"repo_beta::services.catalog::{bare}"

        found_alpha = [s for s in domain_symbols if s.qualified_name == q_alpha]
        assert len(found_alpha) == 1
        assert found_alpha[0].symbol_id == alpha_sym_map[bare].symbol_id
        assert "repo_alpha::" in found_alpha[0].qualified_name
        assert "repo_beta::" not in found_alpha[0].qualified_name

        found_beta = [s for s in domain_symbols if s.qualified_name == q_beta]
        assert len(found_beta) == 1
        assert found_beta[0].symbol_id == beta_sym_map[bare].symbol_id
        assert "repo_beta::" in found_beta[0].qualified_name
        assert "repo_alpha::" not in found_beta[0].qualified_name

        # Verify identifier matcher
        assert match_symbol_identifier(bare, q_alpha)
        assert match_symbol_identifier(bare, q_beta)
        assert match_symbol_identifier(f"services.catalog.{bare}", q_alpha)
        assert match_symbol_identifier(f"services.catalog.{bare}", q_beta)
        assert match_symbol_identifier(q_alpha, q_alpha)
        assert not match_symbol_identifier(q_alpha, q_beta)


def test_empirical_cross_repo_symbol_collision_live_postgres_n100(
    ephemeral_pg_domain: str,
) -> None:
    """Live PostgreSQL 17 test inserting N=100 identical symbol pairs across 2 repos."""
    settings = ScoutSettings()
    app_mgr = RLSConnectionManager(settings.db_app_url)
    pg_repo = PostgresRLSRepository()

    domain_id = ephemeral_pg_domain
    scopes = ["public_eng"]
    principal = PrincipalContext(
        user_id="collision_pg_tester",
        role=RoleType.ENGINEER,
        allowed_scopes=scopes,
    )

    n_pairs = 100
    with app_mgr.scoped_transaction(principal, enable_hnsw_iterative_scan=False) as conn:
        fid_a = uuid4()
        fid_b = uuid4()
        repo_a_name = f"{EMPIRICAL_M2_REPO_PREFIX}alpha-{fid_a.hex[:8]}"
        repo_b_name = f"{EMPIRICAL_M2_REPO_PREFIX}beta-{fid_b.hex[:8]}"

        pg_repo.insert_code_file(
            conn,
            CodeFileRecord(
                file_id=fid_a,
                domain_id=domain_id,
                repo_name=repo_a_name,
                file_path="src/catalog.py",
                git_commit_sha="c1",
                git_blob_sha="b1",
                language="python",
                allowed_scopes=scopes,
                raw_content="# alpha",
                line_count=100,
            ),
        )
        pg_repo.insert_code_file(
            conn,
            CodeFileRecord(
                file_id=fid_b,
                domain_id=domain_id,
                repo_name=repo_b_name,
                file_path="src/catalog.py",
                git_commit_sha="c2",
                git_blob_sha="b2",
                language="python",
                allowed_scopes=scopes,
                raw_content="# beta",
                line_count=100,
            ),
        )

        for i in range(n_pairs):
            bare = f"PGModel_{i:03d}"
            # Alpha
            pg_repo.insert_ast_symbol(
                conn,
                ASTSymbolRecord(
                    symbol_id=uuid4(),
                    file_id=fid_a,
                    domain_id=domain_id,
                    qualified_name=f"{repo_a_name}::catalog::{bare}",
                    symbol_kind="class",
                    start_line=1,
                    end_line=2,
                    signature_skeleton=f"class {bare}",
                    contextual_summary="Alpha",
                    allowed_scopes=scopes,
                    embedding=[0.05] * 768,
                ),
            )
            # Beta
            pg_repo.insert_ast_symbol(
                conn,
                ASTSymbolRecord(
                    symbol_id=uuid4(),
                    file_id=fid_b,
                    domain_id=domain_id,
                    qualified_name=f"{repo_b_name}::catalog::{bare}",
                    symbol_kind="class",
                    start_line=1,
                    end_line=2,
                    signature_skeleton=f"class {bare}",
                    contextual_summary="Beta",
                    allowed_scopes=scopes,
                    embedding=[0.05] * 768,
                ),
            )

        # Assert exactly 100 alpha and 100 beta symbols in PostgreSQL
        row_a = conn.execute(
            """
            SELECT COUNT(*) FROM ast_symbols
            WHERE domain_id = %s AND qualified_name LIKE %s;
            """,
            (domain_id, f"{repo_a_name}::%%"),
        ).fetchone()
        count_a = row_a[0] if row_a else 0

        row_b = conn.execute(
            """
            SELECT COUNT(*) FROM ast_symbols
            WHERE domain_id = %s AND qualified_name LIKE %s;
            """,
            (domain_id, f"{repo_b_name}::%%"),
        ).fetchone()
        count_b = row_b[0] if row_b else 0

        assert count_a == 100, f"Expected 100 alpha symbols, found {count_a}"
        assert count_b == 100, f"Expected 100 beta symbols, found {count_b}"


# ==============================================================================
# Challenge 3: Multi-Tenant Forced RLS Isolation on Cross-Repo Links
# ==============================================================================


def test_empirical_multi_tenant_forced_rls_isolation_cross_repo_links(
    ephemeral_pg_domain: str,
) -> None:
    """Stress-tests multi-tenant Forced RLS isolation on cross-repo references.

    Principal A (authorized for repo_alpha only) must receive strictly 0 rows and
    0 metadata leaks from repo_beta, and cross-repo reconciler must not bridge
    unauthorized cross-tenant boundaries.
    """
    settings = ScoutSettings()
    app_mgr = RLSConnectionManager(settings.db_app_url)
    pg_repo = PostgresRLSRepository()
    indexer = GitMerkleIndexer()

    domain_id = ephemeral_pg_domain
    scope_alpha = "tenant_alpha_scope"
    scope_beta = "tenant_beta_scope"

    principal_alpha = PrincipalContext(
        user_id="auditor_alpha",
        role=RoleType.ENGINEER,
        allowed_scopes=[scope_alpha],
    )
    principal_beta = PrincipalContext(
        user_id="auditor_beta",
        role=RoleType.ENGINEER,
        allowed_scopes=[scope_beta],
    )
    admin_principal = PrincipalContext(
        user_id="admin_provisioner",
        role=RoleType.ENGINEER,
        allowed_scopes=[scope_alpha, scope_beta],
    )

    fid_alpha = uuid4()
    fid_beta = uuid4()
    repo_alpha_name = f"{EMPIRICAL_M2_REPO_PREFIX}tenant-a-{fid_alpha.hex[:8]}"
    repo_beta_name = f"{EMPIRICAL_M2_REPO_PREFIX}tenant-b-{fid_beta.hex[:8]}"

    sym_beta_id = uuid4()
    sym_beta_qual = f"{repo_beta_name}::secrets::VaultSecretKey"

    edge_alpha_id = uuid4()

    # 1. Provision data using admin with both scopes
    with app_mgr.scoped_transaction(admin_principal, enable_hnsw_iterative_scan=False) as conn:
        # File & Symbol in Repo Alpha (scope_alpha)
        pg_repo.insert_code_file(
            conn,
            CodeFileRecord(
                file_id=fid_alpha,
                domain_id=domain_id,
                repo_name=repo_alpha_name,
                file_path="src/app.py",
                git_commit_sha="ca1",
                git_blob_sha="ba1",
                language="python",
                allowed_scopes=[scope_alpha],
                raw_content="# alpha app",
                line_count=10,
            ),
        )
        sym_alpha_id = uuid4()
        pg_repo.insert_ast_symbol(
            conn,
            ASTSymbolRecord(
                symbol_id=sym_alpha_id,
                file_id=fid_alpha,
                domain_id=domain_id,
                qualified_name=f"{repo_alpha_name}::app::load_config",
                symbol_kind="function",
                start_line=1,
                end_line=2,
                signature_skeleton="def load_config()",
                contextual_summary="Alpha Config",
                allowed_scopes=[scope_alpha],
                embedding=[0.01] * 768,
            ),
        )

        # File & Symbol in Repo Beta (scope_beta) - Contains proprietary secret
        pg_repo.insert_code_file(
            conn,
            CodeFileRecord(
                file_id=fid_beta,
                domain_id=domain_id,
                repo_name=repo_beta_name,
                file_path="src/secrets.py",
                git_commit_sha="cb1",
                git_blob_sha="bb1",
                language="python",
                allowed_scopes=[scope_beta],
                raw_content="# beta secrets",
                line_count=10,
            ),
        )
        pg_repo.insert_ast_symbol(
            conn,
            ASTSymbolRecord(
                symbol_id=sym_beta_id,
                file_id=fid_beta,
                domain_id=domain_id,
                qualified_name=sym_beta_qual,
                symbol_kind="class",
                start_line=1,
                end_line=2,
                signature_skeleton="class VaultSecretKey",
                contextual_summary="Confidential Tenant Beta Key",
                allowed_scopes=[scope_beta],
                embedding=[0.99] * 768,
            ),
        )

        # Cross-repo edge originating from Repo Alpha attempting to link VaultSecretKey
        pg_repo.insert_ast_edge(
            conn,
            ASTEdgeRecord(
                edge_id=edge_alpha_id,
                domain_id=domain_id,
                source_symbol_id=sym_alpha_id,
                target_qualified_name="UNRESOLVED::secrets::VaultSecretKey",
                target_symbol_id=None,
                edge_kind="USES_TYPE",
                allowed_scopes=[scope_alpha],
                is_cross_repo=True,
            ),
        )

    # 2. Test Forced RLS Isolation as Principal Alpha
    with app_mgr.scoped_transaction(
        principal_alpha, enable_hnsw_iterative_scan=False
    ) as conn_alpha:
        # Principal Alpha querying code_files: must see repo_alpha, 0 rows from repo_beta
        files_seen = conn_alpha.execute(
            "SELECT repo_name, file_path FROM code_files WHERE domain_id = %s;",
            (domain_id,),
        ).fetchall()
        repo_names_seen = {r[0] for r in files_seen}
        assert repo_beta_name not in repo_names_seen, (
            "LEAK: Principal Alpha can see repo_beta in code_files!"
        )
        assert repo_alpha_name in repo_names_seen

        # Principal Alpha querying ast_symbols: 0 rows from repo_beta
        syms_seen = conn_alpha.execute(
            "SELECT qualified_name FROM ast_symbols WHERE domain_id = %s;",
            (domain_id,),
        ).fetchall()
        qual_names_seen = [r[0] for r in syms_seen]
        assert sym_beta_qual not in qual_names_seen, (
            "LEAK: Principal Alpha can see repo_beta symbols!"
        )

        # Direct targeted query attempt by Principal Alpha on repo_beta
        direct_leak = conn_alpha.execute(
            """
            SELECT symbol_id, contextual_summary FROM ast_symbols
            WHERE domain_id = %s AND qualified_name = %s;
            """,
            (domain_id, sym_beta_qual),
        ).fetchall()
        assert len(direct_leak) == 0, (
            "CRITICAL RLS BYPASS: Principal Alpha retrieved repo_beta symbol row!"
        )

        # Run cross-repo reconciler under Principal Alpha's session
        reconciled = indexer.reconcile_cross_repo_edges_postgres(conn_alpha, domain_id)
        assert reconciled == 0, (
            f"RLS VIOLATION: Cross-repo reconciler resolved {reconciled} cross-tenant edges!"
        )

        # Verify edge remains unresolved
        edge_row = conn_alpha.execute(
            "SELECT target_symbol_id, target_qualified_name FROM ast_edges WHERE edge_id = %s;",
            (edge_alpha_id,),
        ).fetchone()
        assert edge_row is not None
        assert edge_row[0] is None, (
            "Edge target_symbol_id was illegally populated across tenant boundary!"
        )
        assert edge_row[1] == "UNRESOLVED::secrets::VaultSecretKey"

    # 3. Test Forced RLS Isolation as Principal Beta
    with app_mgr.scoped_transaction(principal_beta, enable_hnsw_iterative_scan=False) as conn_beta:
        # Principal Beta must see repo_beta, and 0 rows from repo_alpha
        files_seen_beta = conn_beta.execute(
            "SELECT repo_name FROM code_files WHERE domain_id = %s;",
            (domain_id,),
        ).fetchall()
        repos_beta = {r[0] for r in files_seen_beta}
        assert repo_alpha_name not in repos_beta, (
            "LEAK: Principal Beta can see repo_alpha in code_files!"
        )
        assert repo_beta_name in repos_beta


# ==============================================================================
# Challenge 4: Inward Edge Preservation on Modified and Deleted Symbols
# ==============================================================================


def test_empirical_inward_edge_preservation_on_file_modification_and_deletion(
    indexer: GitMerkleIndexer,
) -> None:
    """Empirically stress-tests inward edge preservation and ON DELETE SET NULL semantics.

    1. Caller file points to Target A (unmodified), Target B (modified), and Target C (deleted).
    2. After push, inward edge to Target A remains linked.
    3. Inward edge to Target B is re-linked to new symbol_id.
    4. Inward edge to Target C has target_symbol_id = NULL without dropping the edge.
    5. Complete deletion of a target file sets inward edges to NULL while preserving callers.
    """
    repo = InMemoryRLSRepository()
    domain_id = "test_inward_edge_preservation"
    scopes = ["eng_scope"]
    principal = PrincipalContext(
        user_id="inward_edge_tester",
        role=RoleType.ENGINEER,
        allowed_scopes=scopes,
    )

    # 1. Setup Caller File
    fid_caller = uuid4()
    repo.insert_code_file(
        principal,
        CodeFileRecord(
            file_id=fid_caller,
            domain_id=domain_id,
            repo_name="core-service",
            file_path="src/caller.py",
            git_commit_sha="c1",
            git_blob_sha="b_caller",
            language="python",
            allowed_scopes=scopes,
            raw_content="def caller(): pass",
            line_count=1,
        ),
    )
    sym_caller_id = uuid4()
    repo.insert_ast_symbol(
        principal,
        ASTSymbolRecord(
            symbol_id=sym_caller_id,
            file_id=fid_caller,
            domain_id=domain_id,
            qualified_name="core-service::caller::caller",
            symbol_kind="function",
            start_line=1,
            end_line=1,
            signature_skeleton="def caller()",
            contextual_summary="Caller",
            allowed_scopes=scopes,
        ),
    )

    # 2. Setup Target File 1 (defines Symbol A and Symbol B and Symbol C)
    fid_target1 = uuid4()
    repo.insert_code_file(
        principal,
        CodeFileRecord(
            file_id=fid_target1,
            domain_id=domain_id,
            repo_name="core-service",
            file_path="src/target_ops.py",
            git_commit_sha="c1",
            git_blob_sha="b_target1_v1",
            language="python",
            allowed_scopes=scopes,
            raw_content="def target_a(): pass\ndef target_b(): pass\ndef target_c(): pass\n",
            line_count=3,
        ),
    )
    sym_a_v1 = uuid4()
    sym_b_v1 = uuid4()
    sym_c_v1 = uuid4()

    repo.insert_ast_symbol(
        principal,
        ASTSymbolRecord(
            symbol_id=sym_a_v1,
            file_id=fid_target1,
            domain_id=domain_id,
            qualified_name="core-service::target_ops::target_a",
            symbol_kind="function",
            start_line=1,
            end_line=1,
            signature_skeleton="def target_a()",
            contextual_summary="Target A v1",
            allowed_scopes=scopes,
        ),
    )
    repo.insert_ast_symbol(
        principal,
        ASTSymbolRecord(
            symbol_id=sym_b_v1,
            file_id=fid_target1,
            domain_id=domain_id,
            qualified_name="core-service::target_ops::target_b",
            symbol_kind="function",
            start_line=2,
            end_line=2,
            signature_skeleton="def target_b()",
            contextual_summary="Target B v1",
            allowed_scopes=scopes,
        ),
    )
    repo.insert_ast_symbol(
        principal,
        ASTSymbolRecord(
            symbol_id=sym_c_v1,
            file_id=fid_target1,
            domain_id=domain_id,
            qualified_name="core-service::target_ops::target_c",
            symbol_kind="function",
            start_line=3,
            end_line=3,
            signature_skeleton="def target_c()",
            contextual_summary="Target C v1",
            allowed_scopes=scopes,
        ),
    )

    # Inward edges from Caller to A, B, and C
    edge_a_id = uuid4()
    edge_b_id = uuid4()
    edge_c_id = uuid4()

    repo.insert_ast_edge(
        principal,
        ASTEdgeRecord(
            edge_id=edge_a_id,
            domain_id=domain_id,
            source_symbol_id=sym_caller_id,
            target_qualified_name="core-service::target_ops::target_a",
            target_symbol_id=sym_a_v1,
            edge_kind="CALLS",
            allowed_scopes=scopes,
        ),
    )
    repo.insert_ast_edge(
        principal,
        ASTEdgeRecord(
            edge_id=edge_b_id,
            domain_id=domain_id,
            source_symbol_id=sym_caller_id,
            target_qualified_name="core-service::target_ops::target_b",
            target_symbol_id=sym_b_v1,
            edge_kind="CALLS",
            allowed_scopes=scopes,
        ),
    )
    repo.insert_ast_edge(
        principal,
        ASTEdgeRecord(
            edge_id=edge_c_id,
            domain_id=domain_id,
            source_symbol_id=sym_caller_id,
            target_qualified_name="core-service::target_ops::target_c",
            target_symbol_id=sym_c_v1,
            edge_kind="CALLS",
            allowed_scopes=scopes,
        ),
    )

    # 3. Perform Git Commit Update in real git directory to execute sync_push_incremental_inmemory
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo_dir = Path(tmp_dir)
        _init_git_repo(repo_dir)

        src_dir = repo_dir / "src"
        src_dir.mkdir(parents=True)
        (src_dir / "caller.py").write_text("def caller(): pass\n", encoding="utf-8")
        (src_dir / "target_ops.py").write_text(
            "def target_a(): pass\ndef target_b(): pass\ndef target_c(): pass\n",
            encoding="utf-8",
        )
        c1_sha = _commit_all(repo_dir, "Commit 1")

        # Now commit 2:
        # target_a is preserved
        # target_b is modified with new signature
        # target_c is deleted
        (src_dir / "target_ops.py").write_text(
            "def target_a(): pass\ndef target_b(new_param: int): pass\n",
            encoding="utf-8",
        )
        c2_sha = _commit_all(repo_dir, "Commit 2: modified target_ops.py")

        stats = indexer.sync_push_incremental_inmemory(
            repo_dir=repo_dir,
            repo=repo,
            principal=principal,
            domain_id=domain_id,
            repo_name="core-service",
            before_sha=c1_sha,
            after_sha=c2_sha,
        )

        assert stats.modified_files_reindexed == 1
        assert stats.deleted_files_removed == 0

        # Verify symbols in repo
        target_syms = [
            s
            for s in repo._symbols.values()
            if s.domain_id == domain_id and "target_ops::" in s.qualified_name
        ]
        target_names = {s.qualified_name.split("::")[-1] for s in target_syms}
        assert "target_a" in target_names
        assert "target_b" in target_names
        assert "target_c" not in target_names  # Deleted!

        sym_a_v2 = next(s for s in target_syms if s.qualified_name.endswith("target_a"))
        sym_b_v2 = next(s for s in target_syms if s.qualified_name.endswith("target_b"))

        # Inward Edge 1 (Target A): re-linked to sym_a_v2
        edge_a = repo._edges[edge_a_id]
        assert edge_a.target_symbol_id == sym_a_v2.symbol_id, (
            "Edge A was not re-linked to new symbol_id!"
        )

        # Inward Edge 2 (Target B): re-linked to sym_b_v2
        edge_b = repo._edges[edge_b_id]
        assert edge_b.target_symbol_id == sym_b_v2.symbol_id, (
            "Edge B was not re-linked to new symbol_id!"
        )

        # Inward Edge 3 (Target C): target symbol was deleted -> target_symbol_id must be NULL
        edge_c = repo._edges[edge_c_id]
        assert edge_c.target_symbol_id is None, (
            "Edge C target_symbol_id should be NULL for deleted symbol!"
        )
        assert edge_c.target_qualified_name == "core-service::target_ops::target_c"
        assert edge_c.source_symbol_id == sym_caller_id, "Edge C caller link was corrupted!"

        # Now Commit 3: completely delete target_ops.py
        (src_dir / "target_ops.py").unlink()
        c3_sha = _commit_all(repo_dir, "Commit 3: deleted target_ops.py")

        stats_del = indexer.sync_push_incremental_inmemory(
            repo_dir=repo_dir,
            repo=repo,
            principal=principal,
            domain_id=domain_id,
            repo_name="core-service",
            before_sha=c2_sha,
            after_sha=c3_sha,
        )
        assert stats_del.deleted_files_removed == 1

        # Caller edges still exist, but now target_symbol_id for A and B
        # are NULL (matches ON DELETE SET NULL)
        edge_a_after = repo._edges[edge_a_id]
        edge_b_after = repo._edges[edge_b_id]
        assert edge_a_after.target_symbol_id is None
        assert edge_b_after.target_symbol_id is None
        # Caller symbol itself untouched
        assert sym_caller_id in repo._symbols


def test_empirical_inward_edge_preservation_live_postgres(
    indexer: GitMerkleIndexer,
    ephemeral_pg_domain: str,
) -> None:
    """Stress-tests inward edge preservation on live PostgreSQL 17 under Forced RLS."""
    settings = ScoutSettings()
    app_mgr = RLSConnectionManager(settings.db_app_url)
    pg_repo = PostgresRLSRepository()

    domain_id = ephemeral_pg_domain
    scopes = ["public_eng"]
    principal = PrincipalContext(
        user_id="inward_edge_pg_tester",
        role=RoleType.ENGINEER,
        allowed_scopes=scopes,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        repo_dir = Path(tmp_dir)
        _init_git_repo(repo_dir)

        src_dir = repo_dir / "src"
        src_dir.mkdir(parents=True)

        fid_rand = uuid4().hex[:6]
        repo_name = f"{EMPIRICAL_M2_REPO_PREFIX}edge-{fid_rand}"

        # Commit 1: caller and service with op_a and op_b
        caller_code = "def invoke_ops(): pass\n"
        service_code_v1 = "def op_a(): pass\ndef op_b(): pass\n"
        (src_dir / "caller.py").write_text(caller_code, encoding="utf-8")
        (src_dir / "service.py").write_text(service_code_v1, encoding="utf-8")
        c1_sha = _commit_all(repo_dir, "Commit 1: baseline")

        with app_mgr.scoped_transaction(principal, enable_hnsw_iterative_scan=False) as conn:
            # Sync initial commit
            indexer.sync_push_incremental_postgres(
                repo_dir=repo_dir,
                conn=conn,
                repo=pg_repo,
                principal=principal,
                domain_id=domain_id,
                repo_name=repo_name,
                before_sha="0" * 40,
                after_sha=c1_sha,
                allowed_scopes=scopes,
            )

            # Get symbol IDs
            sym_caller = conn.execute(
                """
                SELECT symbol_id FROM ast_symbols
                WHERE domain_id = %s AND qualified_name = %s;
                """,
                (domain_id, f"{repo_name}::caller::invoke_ops"),
            ).fetchone()
            assert sym_caller is not None
            caller_sym_id = sym_caller[0]

            sym_a_v1 = conn.execute(
                """
                SELECT symbol_id FROM ast_symbols
                WHERE domain_id = %s AND qualified_name = %s;
                """,
                (domain_id, f"{repo_name}::service::op_a"),
            ).fetchone()
            assert sym_a_v1 is not None
            sym_a_v1_id = sym_a_v1[0]

            sym_b_v1 = conn.execute(
                """
                SELECT symbol_id FROM ast_symbols
                WHERE domain_id = %s AND qualified_name = %s;
                """,
                (domain_id, f"{repo_name}::service::op_b"),
            ).fetchone()
            assert sym_b_v1 is not None
            sym_b_v1_id = sym_b_v1[0]

            # Insert inward edges from caller to op_a and op_b
            edge_a_id = uuid4()
            edge_b_id = uuid4()
            pg_repo.insert_ast_edge(
                conn,
                ASTEdgeRecord(
                    edge_id=edge_a_id,
                    domain_id=domain_id,
                    source_symbol_id=caller_sym_id,
                    target_qualified_name=f"{repo_name}::service::op_a",
                    target_symbol_id=sym_a_v1_id,
                    edge_kind="CALLS",
                    allowed_scopes=scopes,
                ),
            )
            pg_repo.insert_ast_edge(
                conn,
                ASTEdgeRecord(
                    edge_id=edge_b_id,
                    domain_id=domain_id,
                    source_symbol_id=caller_sym_id,
                    target_qualified_name=f"{repo_name}::service::op_b",
                    target_symbol_id=sym_b_v1_id,
                    edge_kind="CALLS",
                    allowed_scopes=scopes,
                ),
            )

        # Commit 2: modify service.py -> op_a updated, op_b deleted
        service_code_v2 = "def op_a(flag: bool = True): pass\n"
        (src_dir / "service.py").write_text(service_code_v2, encoding="utf-8")
        c2_sha = _commit_all(repo_dir, "Commit 2: modified op_a, removed op_b")

        with app_mgr.scoped_transaction(principal, enable_hnsw_iterative_scan=False) as conn:
            indexer.sync_push_incremental_postgres(
                repo_dir=repo_dir,
                conn=conn,
                repo=pg_repo,
                principal=principal,
                domain_id=domain_id,
                repo_name=repo_name,
                before_sha=c1_sha,
                after_sha=c2_sha,
                allowed_scopes=scopes,
            )

            # Check new symbol for op_a
            sym_a_v2 = conn.execute(
                """
                SELECT symbol_id FROM ast_symbols
                WHERE domain_id = %s AND qualified_name = %s;
                """,
                (domain_id, f"{repo_name}::service::op_a"),
            ).fetchone()
            assert sym_a_v2 is not None
            sym_a_v2_id = sym_a_v2[0]
            assert sym_a_v2_id != sym_a_v1_id  # Re-indexed symbol has new UUID

            # Check inward edge A: re-linked to sym_a_v2_id!
            row_edge_a = conn.execute(
                "SELECT target_symbol_id FROM ast_edges WHERE edge_id = %s;",
                (edge_a_id,),
            ).fetchone()
            assert row_edge_a is not None
            assert row_edge_a[0] == sym_a_v2_id

            # Check inward edge B: target_symbol_id is NULL (ON DELETE SET NULL)
            row_edge_b = conn.execute(
                "SELECT target_symbol_id, target_qualified_name FROM ast_edges WHERE edge_id = %s;",
                (edge_b_id,),
            ).fetchone()
            assert row_edge_b is not None
            assert row_edge_b[0] is None
            assert row_edge_b[1] == f"{repo_name}::service::op_b"

        # Commit 3: completely delete service.py
        (src_dir / "service.py").unlink()
        c3_sha = _commit_all(repo_dir, "Commit 3: deleted service.py")

        with app_mgr.scoped_transaction(principal, enable_hnsw_iterative_scan=False) as conn:
            indexer.sync_push_incremental_postgres(
                repo_dir=repo_dir,
                conn=conn,
                repo=pg_repo,
                principal=principal,
                domain_id=domain_id,
                repo_name=repo_name,
                before_sha=c2_sha,
                after_sha=c3_sha,
                allowed_scopes=scopes,
            )

            # Both edges now have target_symbol_id IS NULL
            row_a = conn.execute(
                "SELECT target_symbol_id FROM ast_edges WHERE edge_id = %s;",
                (edge_a_id,),
            ).fetchone()
            assert row_a is not None and row_a[0] is None

            row_b = conn.execute(
                "SELECT target_symbol_id FROM ast_edges WHERE edge_id = %s;",
                (edge_b_id,),
            ).fetchone()
            assert row_b is not None and row_b[0] is None

            # Caller symbol still intact in Postgres
            caller_check = conn.execute(
                "SELECT symbol_id FROM ast_symbols WHERE symbol_id = %s;",
                (caller_sym_id,),
            ).fetchone()
            assert caller_check is not None


def test_empirical_merkle_diff_100_files_bulk_stress(
    indexer: GitMerkleIndexer,
) -> None:
    """Stress-tests diff_commit_blobs latency on a massive 100-file simultaneous modification."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo_dir = Path(tmp_dir)
        _init_git_repo(repo_dir)

        src_dir = repo_dir / "src"
        src_dir.mkdir(parents=True)
        for i in range(100):
            (src_dir / f"bulk_{i:03d}.py").write_text(f"val = {i}\n", encoding="utf-8")
        c1 = _commit_all(repo_dir, "Commit 1: 100 files")

        for i in range(100):
            (src_dir / f"bulk_{i:03d}.py").write_text(f"val = {i + 1000}\n", encoding="utf-8")
        c2 = _commit_all(repo_dir, "Commit 2: modified 100 files")

        t0 = time.perf_counter()
        diff = indexer.diff_commit_blobs(repo_dir, c1, c2)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        print(f"\n[Bulk 100-File Merkle Diff Latency] {elapsed_ms:.2f}ms")
        assert len(diff.modified) == 100
        assert elapsed_ms < 50.0, f"Bulk 100-file diff took {elapsed_ms:.2f}ms (>50ms SLA)"


class TestEmpiricalM2TeardownCanaryIsolation:
    """Verifies that teardown_empirical_m2_test_repos strictly scopes deletions."""

    def test_teardown_preserves_canary_repo_in_same_domain(
        self,
        ephemeral_pg_domain: str,
    ) -> None:
        """Proves real teardown function removes test rows and leaves canary rows intact."""
        settings = ScoutSettings()
        domain_id = ephemeral_pg_domain
        canary_repo = "canary-preservation-repo"
        victim_repo = f"{EMPIRICAL_M2_REPO_PREFIX}victim-{uuid4().hex[:8]}"

        with psycopg.connect(settings.db_admin_url) as conn:
            cur = conn.cursor()
            # 1. Insert canary repo row
            canary_fid = uuid4()
            cur.execute(
                """
                INSERT INTO code_files (
                    file_id, domain_id, repo_name, file_path,
                    git_commit_sha, git_blob_sha, language,
                    allowed_scopes, raw_content, line_count
                ) VALUES (
                    %s, %s, %s, 'src/canary.py', 'c0', 'b0', 'python',
                    ARRAY['public_eng'], 'print(1)', 1
                );
                """,
                (canary_fid, domain_id, canary_repo),
            )
            # 2. Insert test victim repo row
            victim_fid = uuid4()
            cur.execute(
                """
                INSERT INTO code_files (
                    file_id, domain_id, repo_name, file_path,
                    git_commit_sha, git_blob_sha, language,
                    allowed_scopes, raw_content, line_count
                ) VALUES (
                    %s, %s, %s, 'src/victim.py', 'c1', 'b1', 'python',
                    ARRAY['public_eng'], 'print(2)', 1
                );
                """,
                (victim_fid, domain_id, victim_repo),
            )
            conn.commit()

            try:
                # 3. Execute the REAL teardown function
                teardown_empirical_m2_test_repos(conn, domain_id=domain_id)

                # 4. Assertions: canary survived, victim deleted
                cur.execute(
                    "SELECT COUNT(*) FROM code_files WHERE domain_id = %s AND repo_name = %s;",
                    (domain_id, canary_repo),
                )
                assert cur.fetchone()[0] == 1, "Canary repo was wiped by teardown!"

                cur.execute(
                    "SELECT COUNT(*) FROM code_files WHERE domain_id = %s AND repo_name = %s;",
                    (domain_id, victim_repo),
                )
                assert cur.fetchone()[0] == 0, "Test victim repo was not deleted by teardown!"
            finally:
                # Cleanup canary
                cur.execute(
                    "DELETE FROM code_files WHERE domain_id = %s AND repo_name = %s;",
                    (domain_id, canary_repo),
                )
                conn.commit()

    def test_teardown_mutation_proof_flawed_negative_predicate_destroys_canary(
        self,
        ephemeral_pg_domain: str,
    ) -> None:
        """Mutation proof: Proves that flawed 'repo_name != ...' predicate destroys canary."""
        settings = ScoutSettings()
        domain_id = ephemeral_pg_domain
        canary_repo = "canary-mutation-victim-repo"

        with psycopg.connect(settings.db_admin_url) as conn:
            cur = conn.cursor()
            canary_fid = uuid4()
            cur.execute(
                """
                INSERT INTO code_files (
                    file_id, domain_id, repo_name, file_path,
                    git_commit_sha, git_blob_sha, language,
                    allowed_scopes, raw_content, line_count
                ) VALUES (
                    %s, %s, %s, 'src/canary.py', 'c0', 'b0', 'python',
                    ARRAY['public_eng'], 'print(1)', 1
                );
                """,
                (canary_fid, domain_id, canary_repo),
            )
            conn.commit()

            try:
                # Defective mutated teardown (lacks prefix predicate inside domain)
                def defective_teardown(c: psycopg.Connection[Any], target_domain: str) -> None:
                    with c.cursor() as cur_def:
                        cur_def.execute(
                            "DELETE FROM code_files "
                            "WHERE domain_id = %s AND repo_name != 'enterprise-context-scout';",
                            (target_domain,),
                        )
                    c.commit()

                defective_teardown(conn, domain_id)

                cur.execute(
                    "SELECT COUNT(*) FROM code_files WHERE domain_id = %s AND repo_name = %s;",
                    (domain_id, canary_repo),
                )
                canary_count = cur.fetchone()[0]

                # Canary must be 0, proving mutation caught the bug
                assert canary_count == 0, (
                    "Mutation check failed: defective teardown did not destroy canary!"
                )
            finally:
                cur.execute(
                    "DELETE FROM code_files WHERE domain_id = %s AND repo_name = %s;",
                    (domain_id, canary_repo),
                )
                conn.commit()
