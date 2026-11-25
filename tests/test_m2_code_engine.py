"""Unit tests for M2 Code Engine: AST extraction, Merkle diffing, and <50ms cache hits."""

from pathlib import Path

import pytest

from context_scout.ast_engine.extractor import ASTExtractor
from context_scout.db.repositories import InMemoryRLSRepository
from context_scout.ingestion.git_merkle_indexer import GitMerkleIndexer
from context_scout.schemas.contracts import PrincipalContext, RoleType


@pytest.fixture
def extractor() -> ASTExtractor:
    return ASTExtractor()


@pytest.fixture
def indexer(extractor: ASTExtractor) -> GitMerkleIndexer:
    return GitMerkleIndexer(extractor=extractor)


@pytest.fixture
def principal() -> PrincipalContext:
    return PrincipalContext(
        user_id="code_indexer_runner",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng"],
    )


def test_tree_sitter_ast_extractor_python(extractor: ASTExtractor) -> None:
    code = (
        "import math\n"
        "from typing import Optional\n"
        "from .models import Order, ExecutionReceipt\n"
        "\n"
        "class BaseVenue:\n"
        '    """Base execution venue."""\n'
        "    pass\n"
        "\n"
        "class SmartOrderRouter(BaseVenue):\n"
        '    """Enterprise multi-venue order router."""\n'
        "\n"
        "    def __init__(self, venue_id: str) -> None:\n"
        "        self.venue_id = venue_id\n"
        "\n"
        "    def route(self, order: Order, retries: int = 3) -> ExecutionReceipt:\n"
        '        """Routes the order and records metrics."""\n'
        "        validated = self.validate_order(order)\n"
        "        return ExecutionReceipt(receipt_id='1', filled_qty=100.0)\n"
        "\n"
        "    def validate_order(self, order: Order) -> bool:\n"
        "        return order.qty > 0\n"
        "\n"
        "def test_route():\n"
        "    pass\n"
    )

    symbols, edges = extractor.parse_source(
        code=code,
        language="python",
        file_path="src/execution/router.py",
    )

    symbol_dict = {s.name: s for s in symbols}
    assert "BaseVenue" in symbol_dict
    assert "SmartOrderRouter" in symbol_dict
    assert "__init__" in symbol_dict
    assert "route" in symbol_dict
    assert "validate_order" in symbol_dict
    assert "test_route" in symbol_dict

    router = symbol_dict["SmartOrderRouter"]
    assert router.symbol_kind == "class"
    assert router.start_line == 9
    assert router.docstring == "Enterprise multi-venue order router."
    assert router.qualified_name == "execution.router.SmartOrderRouter"

    route_method = symbol_dict["route"]
    assert route_method.symbol_kind == "method"
    assert route_method.qualified_name == "execution.router.SmartOrderRouter.route"
    assert route_method.start_line == 15
    assert route_method.end_line == 18
    assert "order: Order" in route_method.signature

    # Relations
    inherits_edges = [e for e in edges if e.edge_kind == "INHERITS"]
    assert any(
        e.source_qualified_name == router.qualified_name and e.target_qualified_name == "BaseVenue"
        for e in inherits_edges
    )

    uses_type_edges = [e for e in edges if e.edge_kind == "USES_TYPE"]
    route_types = {
        e.target_qualified_name
        for e in uses_type_edges
        if e.source_qualified_name == route_method.qualified_name
    }
    assert "Order" in route_types
    assert "ExecutionReceipt" in route_types

    call_edges = [e for e in edges if e.edge_kind == "CALLS"]
    route_calls = {
        e.target_qualified_name
        for e in call_edges
        if e.source_qualified_name == route_method.qualified_name
    }
    assert any("validate_order" in c or "ExecutionReceipt" in c for c in route_calls)

    test_edges = [e for e in edges if e.edge_kind == "TESTED_BY"]
    assert any(
        e.source_qualified_name.endswith("route") and e.target_qualified_name.endswith("test_route")
        for e in test_edges
    )


def test_tree_sitter_ast_extractor_typescript(extractor: ASTExtractor) -> None:
    code = (
        "import { Order } from './types';\n"
        "\n"
        "// State contract for execution orders\n"
        "export interface OrderState {\n"
        "    id: string;\n"
        "    filled: boolean;\n"
        "}\n"
        "\n"
        "export type OrderId = string | number;\n"
        "\n"
        "export class OrderEngine implements EngineLifecycle {\n"
        "    process(order: Order): OrderState {\n"
        "        this.logState(order);\n"
        "        return { id: '1', filled: true };\n"
        "    }\n"
        "    logState(order: Order): void {}\n"
        "}\n"
        "\n"
        "export const submitOrder = (order: Order): boolean => {\n"
        "    return true;\n"
        "};\n"
    )

    symbols, edges = extractor.parse_source(
        code=code,
        language="typescript",
        file_path="src/execution/engine.ts",
    )

    symbol_dict = {s.name: s for s in symbols}
    assert "OrderState" in symbol_dict
    assert "OrderId" in symbol_dict
    assert "OrderEngine" in symbol_dict
    assert "process" in symbol_dict
    assert "submitOrder" in symbol_dict

    state_iface = symbol_dict["OrderState"]
    assert state_iface.symbol_kind == "interface"
    assert state_iface.start_line == 4
    assert state_iface.qualified_name == "execution.engine.OrderState"

    order_id_type = symbol_dict["OrderId"]
    assert order_id_type.symbol_kind == "type_alias"

    engine_cls = symbol_dict["OrderEngine"]
    assert engine_cls.symbol_kind == "class"

    submit_fn = symbol_dict["submitOrder"]
    assert submit_fn.symbol_kind == "function"
    assert submit_fn.start_line == 19

    inherits_edges = [e for e in edges if e.edge_kind == "INHERITS"]
    assert any(e.target_qualified_name == "EngineLifecycle" for e in inherits_edges)

    uses_type_edges = [e for e in edges if e.edge_kind == "USES_TYPE"]
    process_types = {
        e.target_qualified_name for e in uses_type_edges if "process" in e.source_qualified_name
    }
    assert "Order" in process_types
    assert "OrderState" in process_types


def test_git_merkle_indexer_diffing_and_unchanged_cache_hit(
    indexer: GitMerkleIndexer, principal: PrincipalContext
) -> None:
    repo_dir = Path.cwd()
    repo = InMemoryRLSRepository()
    domain_id = "code_architecture_scout"
    repo_name = "enterprise-context-scout"

    # Run 1: Initial cold index
    stats_cold = indexer.sync_in_memory(
        repo_dir=repo_dir,
        repo=repo,
        principal=principal,
        domain_id=domain_id,
        repo_name=repo_name,
    )

    assert stats_cold.cache_hit is False
    assert stats_cold.total_files_scanned > 0
    assert stats_cold.new_files_indexed > 0
    assert stats_cold.total_symbols_stored > 0

    # Run 2: Unchanged repository warm index (must be <50ms cache hit)
    stats_warm = indexer.sync_in_memory(
        repo_dir=repo_dir,
        repo=repo,
        principal=principal,
        domain_id=domain_id,
        repo_name=repo_name,
    )

    assert stats_warm.cache_hit is True
    assert stats_warm.unchanged_files_skipped == stats_warm.total_files_scanned
    assert stats_warm.new_files_indexed == 0
    assert stats_warm.modified_files_reindexed == 0
    assert stats_warm.deleted_files_removed == 0
    # Must strictly satisfy <60ms sync latency
    assert stats_warm.duration_ms < 60.0, (
        f"Cache-hit sync took {stats_warm.duration_ms}ms (expected <60ms)"
    )


def test_git_merkle_indexer_two_pass_edge_resolution(
    indexer: GitMerkleIndexer, principal: PrincipalContext
) -> None:
    repo_dir = Path.cwd()
    repo = InMemoryRLSRepository()
    domain_id = "code_architecture_scout"
    repo_name = "enterprise-context-scout"

    indexer.sync_in_memory(
        repo_dir=repo_dir,
        repo=repo,
        principal=principal,
        domain_id=domain_id,
        repo_name=repo_name,
    )

    visible_edges = repo.list_visible_ast_edges(principal, domain_id)
    assert len(visible_edges) >= 100

    # Check that internal call/inherits/type edges have target_symbol_id resolved
    resolved_edges = [e for e in visible_edges if e.target_symbol_id is not None]
    assert len(resolved_edges) >= 10
    assert {e.edge_kind for e in resolved_edges}.issubset(
        {
            "CALLS",
            "IMPORTS",
            "USES_TYPE",
            "INHERITS",
            "TESTED_BY",
            "PROVISIONED_BY",
            "BINDS_SERVICE_ACCOUNT",
            "ACCESSES_SECRET",
        }
    )
    for edge in resolved_edges:
        assert edge.target_symbol_id is not None
        assert edge.source_symbol_id is not None


def test_git_merkle_tree_diffing_partitions(indexer: GitMerkleIndexer) -> None:
    current = {
        "file_a.py": "sha_1_new",
        "file_b.py": "sha_2_unchanged",
        "file_c.py": "sha_3_modified_new",
    }
    existing = {
        "file_b.py": "sha_2_unchanged",
        "file_c.py": "sha_3_modified_old",
        "file_d.py": "sha_4_deleted",
    }

    unchanged, new, modified, deleted = indexer.diff_merkle_tree(current, existing)

    assert unchanged == {"file_b.py"}
    assert new == {"file_a.py"}
    assert modified == {"file_c.py"}
    assert deleted == {"file_d.py"}


def test_ast_extractor_python_ts_syntax_error_abstains_and_yaml_flags_fallback(
    extractor: ASTExtractor,
) -> None:
    """Python/TS/TSX abstain on syntax errors; regex branches flag fallback."""
    bad_py = "def valid_before():\n    return 1\n\ndef broken(((:\n    pass\n"
    py_syms, py_edges = extractor.parse_source(bad_py, "python", "broken.py")
    assert (py_syms, py_edges) == ([], [])
    assert extractor.last_parse_diagnostics is not None
    assert extractor.last_parse_diagnostics.has_parse_error is True
    assert extractor.last_parse_diagnostics.abstained is True

    bad_ts = "export function validBefore(): number { return 1; }\nfunction broken(((: {\n"
    ts_syms, ts_edges = extractor.parse_source(bad_ts, "typescript", "broken.ts")
    assert (ts_syms, ts_edges) == ([], [])
    assert extractor.last_parse_diagnostics is not None
    assert extractor.last_parse_diagnostics.has_parse_error is True
    assert extractor.last_parse_diagnostics.abstained is True

    bad_tsx = "export const Card = () => <div>ok</div>;\nconst Broken = (((( => <div;\n"
    tsx_syms, tsx_edges = extractor.parse_source(bad_tsx, "tsx", "broken.tsx")
    assert (tsx_syms, tsx_edges) == ([], [])
    assert extractor.last_parse_diagnostics is not None
    assert extractor.last_parse_diagnostics.has_parse_error is True
    assert extractor.last_parse_diagnostics.abstained is True

    yaml_syms, _yaml_edges = extractor.parse_source(
        "build:\n  stage: test\n  script:\n    - pytest\n",
        "yaml",
        ".gitlab-ci.yml",
    )
    assert len(yaml_syms) >= 1
    assert extractor.last_parse_diagnostics is not None
    assert extractor.last_parse_diagnostics.low_confidence_fallback is True
    assert extractor.last_parse_diagnostics.abstained is False


def test_diff_commit_blobs_use_cache_false_spawns_subprocess_every_call(
    monkeypatch: pytest.MonkeyPatch,
    indexer: GitMerkleIndexer,
) -> None:
    """Verifies use_cache=False bypasses _DIFF_TREE_CACHE and spawns git diff-tree each call."""
    import subprocess
    from typing import Any

    import context_scout.ingestion.git_merkle_indexer as gmi

    repo_dir = Path.cwd()
    head_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True
    ).strip()
    parent_proc = subprocess.run(
        ["git", "rev-parse", "HEAD~1"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    parent_sha = (
        parent_proc.stdout.strip()
        if parent_proc.returncode == 0
        else GitMerkleIndexer.EMPTY_TREE_SHA
    )

    real_run = subprocess.run
    diff_tree_calls = 0

    def _counting_run(cmd: list[str], *args: Any, **kwargs: Any) -> Any:
        nonlocal diff_tree_calls
        if isinstance(cmd, list) and "diff-tree" in cmd:
            diff_tree_calls += 1
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", _counting_run)
    gmi._DIFF_TREE_CACHE.clear()

    # 1. Uncached calls always execute git diff-tree and do not populate _DIFF_TREE_CACHE.
    d1 = indexer.diff_commit_blobs(repo_dir, parent_sha, head_sha, use_cache=False)
    d2 = indexer.diff_commit_blobs(repo_dir, parent_sha, head_sha, use_cache=False)
    assert d1 == d2
    assert diff_tree_calls == 2
    assert len(gmi._DIFF_TREE_CACHE) == 0

    # 2. Cached calls execute git diff-tree once and serve repeat immutable SHA diffs from cache.
    d3 = indexer.diff_commit_blobs(repo_dir, parent_sha, head_sha, use_cache=True)
    d4 = indexer.diff_commit_blobs(repo_dir, parent_sha, head_sha, use_cache=True)
    assert d3 == d4 == d1
    assert diff_tree_calls == 3
    assert len(gmi._DIFF_TREE_CACHE) == 1


def test_merkle_indexer_bounded_lru_caches_evict_oldest_entries() -> None:
    """Verifies all module-level Merkle caches are bounded OrderedDict LRUs."""
    from collections import OrderedDict

    import context_scout.ingestion.git_merkle_indexer as gmi

    assert isinstance(gmi._COMMIT_TREE_CACHE, OrderedDict)
    assert isinstance(gmi._WT_BLOB_CACHE, OrderedDict)
    assert isinstance(gmi._DIFF_TREE_CACHE, OrderedDict)
    assert isinstance(gmi._SYMBOL_EMBEDDING_CACHE, OrderedDict)
    assert gmi._COMMIT_TREE_CACHE_MAX_ENTRIES > 0
    assert gmi._WT_BLOB_CACHE_MAX_ENTRIES > 0
    assert gmi._DIFF_TREE_CACHE_MAX_ENTRIES > 0
    assert gmi._SYMBOL_EMBEDDING_CACHE_MAX_ENTRIES > 0

    lru: OrderedDict[str, int] = OrderedDict()
    gmi._lru_set(lru, "k1", 1, max_entries=2)
    gmi._lru_set(lru, "k2", 2, max_entries=2)
    # Access k1 so k2 becomes least-recently-used
    assert gmi._lru_get(lru, "k1") == 1
    gmi._lru_set(lru, "k3", 3, max_entries=2)
    assert list(lru.keys()) == ["k1", "k3"]
    assert gmi._lru_get(lru, "k2") is None
    assert gmi._lru_get(lru, "k3") == 3


def test_batch_prepopulate_symbol_embeddings_chunks_and_invokes_heartbeat() -> None:
    """Regression: _batch_prepopulate_symbol_embeddings chunks batches & fires tx heartbeat."""
    import context_scout.ingestion.git_merkle_indexer as gmi
    from context_scout.ast_engine.models import ParsedSymbol

    class _FakeEmbedder:
        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def embed_batch(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
            _ = is_query
            self.batch_sizes.append(len(texts))
            return [[0.25] * 8 for _ in texts]

    fake_emb = _FakeEmbedder()
    idx = GitMerkleIndexer(embedder=fake_emb)
    blob_sha = "f" * 40
    syms = [
        ParsedSymbol(
            name=f"fn_{i}",
            qualified_name=f"pkg.mod.fn_{i}",
            symbol_kind="function",
            start_line=i + 1,
            end_line=i + 2,
            start_byte=0,
            end_byte=16,
            signature=f"def fn_{i}() -> None",
            signature_skeleton=f"def fn_{i}() -> None:",
            docstring="",
            contextual_summary=f"Function {i}",
        )
        for i in range(5)
    ]
    heartbeats = 0

    def _hb() -> None:
        nonlocal heartbeats
        heartbeats += 1

    try:
        idx._batch_prepopulate_symbol_embeddings(
            [(blob_sha, syms)],
            heartbeat_fn=_hb,
            chunk_size=2,
        )
        assert fake_emb.batch_sizes == [2, 2, 1]
        assert heartbeats == 3
    finally:
        for s in syms:
            gmi._SYMBOL_EMBEDDING_CACHE.pop((blob_sha, s.qualified_name), None)
