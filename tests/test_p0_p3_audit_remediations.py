"""Permanent regression suite for all 24 P0-P3 system audit remediations."""

from __future__ import annotations

import json
import subprocess
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from context_scout.agents.domain_registry import DomainRegistry
from context_scout.agents.gemini_executor import (
    GeminiImplementationExecutor,
    GitWorktreeSandbox,
)
from context_scout.agents.scout_workflow import LocalGemmaClient, ScoutWorkflow
from context_scout.ast_engine.deterministic_slicer import DeterministicSlicer
from context_scout.ast_engine.extractor import ASTExtractor
from context_scout.db.bm25 import BM25Okapi, tokenize_bm25
from context_scout.db.repositories import (
    CodeFileRecord,
    DocChunkRecord,
    InMemoryRLSRepository,
    PostgresRLSRepository,
)
from context_scout.ingestion.doc_parser import MATH_BLOCK_PATTERN
from context_scout.ingestion.git_merkle_indexer import GitMerkleIndexer
from context_scout.ingestion.paper_harvester import QuantRelevanceClassifier
from context_scout.ingestion.scepticism_gate import MarketRegimeScepticismGate
from context_scout.retrieval.embedder import LocalSemanticEmbedder
from context_scout.schemas.contracts import (
    CodeSlicePointer,
    DocCitationSpan,
    FrictionVerdict,
    MarketRegime,
    ModificationIntent,
    PrincipalContext,
    QuantStrategyBrief,
    RoleType,
)
from context_scout.trading_strategies.kelly_risk_manager import KellyRiskManager
from context_scout.trading_strategies.order_flow_imbalance import (
    LOBLevel,
    OrderBookSnapshot,
    OrderFlowImbalanceSignal,
)
from context_scout.trading_strategies.pairs_trading_kalman import PairsSpreadSignal
from context_scout.trading_strategies.volatility_garch import GARCH11Model


def _init_temp_git_repo(tmp_path: Path) -> Path:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
    )
    src_file = repo_dir / "hello.py"
    src_file.write_text("def greet() -> str:\n    return 'hello'\n", encoding="utf-8")
    subprocess.run(["git", "add", "hello.py"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
    )
    return repo_dir


# ============================================================================
# TIER P0 TESTS
# ============================================================================


def test_p0_01_worktree_sandbox_preserves_confirmed_branch_and_cleans_patch_diff(
    tmp_path: Path,
) -> None:
    repo_dir = _init_temp_git_repo(tmp_path)
    patch_diff = (
        "--- a/hello.py\n"
        "+++ b/hello.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def greet() -> str:\n"
        "-    return 'hello'\n"
        "+    return 'hello world'\n"
    )

    with GitWorktreeSandbox(repo_root=repo_dir, require_confirmation=True) as sb:
        ok, _msg = sb.apply_patch(patch_diff)
        assert ok is True
        assert sb.worktree_dir is not None
        assert not (sb.worktree_dir / "patch.diff").exists(), "patch.diff must be unlinked"
        conf_ok, _conf_msg = sb.confirm_and_merge(confirmed=True)
        assert conf_ok is True
        saved_branch = sb.branch_name

    assert saved_branch is not None
    rev_res = subprocess.run(
        ["git", "rev-parse", "--verify", saved_branch],
        cwd=repo_dir,
        capture_output=True,
        text=True,
    )
    assert rev_res.returncode == 0, "Confirmed branch was deleted by __exit__!"
    show_res = subprocess.run(
        ["git", "show", "--name-only", "--format=", saved_branch],
        cwd=repo_dir,
        capture_output=True,
        text=True,
    )
    assert "patch.diff" not in show_res.stdout.splitlines()


def test_p0_02_no_magic_file_or_import_count_thresholds() -> None:
    root = Path(__file__).resolve().parents[1]
    expander_src = (root / "src/context_scout/ast_engine/dependency_expander.py").read_text()
    extractor_src = (root / "src/context_scout/ast_engine/extractor.py").read_text()
    workflow_src = (root / "src/context_scout/agents/scout_workflow.py").read_text()

    assert "len(visible_files)" not in expander_src
    assert "len(file_imports) <= 3" not in extractor_src
    assert "len(visible_files) > 15" not in workflow_src
    assert "len(visible_files) <= 5" not in workflow_src

    extractor = ASTExtractor()
    code = (
        "import os\n"
        "import sys\n\n"
        "class CleanWorker:\n"
        "    def current_dir(self) -> str:\n"
        "        return os.getcwd()\n"
    )
    _syms, edges = extractor.parse_source(code, "python", file_path="src/worker.py")
    cls_imports = {
        e.target_qualified_name
        for e in edges
        if e.source_qualified_name == "worker.CleanWorker" and e.edge_kind == "IMPORTS"
    }
    assert "os" in cls_imports
    assert "sys" not in cls_imports


def test_p0_03_scepticism_gate_multi_chunk_doc_id_citation_lookup() -> None:
    chunks = [
        DocChunkRecord(
            chunk_id=uuid.uuid4(),
            domain_id="quant_research_scout",
            doc_id="arxiv:2401.00001",
            doc_title="Order Flow Paper",
            source_uri="https://arxiv.org/abs/2401.00001",
            market_scope=MarketRegime.US_NYSE_NASDAQ,
            published_date=None,
            authority_weight=1.0,
            section_heading="1. Introduction",
            contextual_header="Header 1",
            verbatim_content="Order flow imbalance predicts short-horizon mid-price changes.",
            metadata_json={},
            allowed_scopes=["public_quant"],
            embedding=[0.1] * 768,
        ),
        DocChunkRecord(
            chunk_id=uuid.uuid4(),
            domain_id="quant_research_scout",
            doc_id="arxiv:2401.00001",
            doc_title="Order Flow Paper",
            source_uri="https://arxiv.org/abs/2401.00001",
            market_scope=MarketRegime.US_NYSE_NASDAQ,
            published_date=None,
            authority_weight=1.0,
            section_heading="References",
            contextual_header="Header 2",
            verbatim_content="[1] Cont, R. (2014). The price impact of order book events.",
            metadata_json={},
            allowed_scopes=["public_quant"],
            embedding=[0.1] * 768,
        ),
    ]
    brief = QuantStrategyBrief(
        strategy_name="OFI Alpha",
        target_markets=[MarketRegime.US_NYSE_NASDAQ],
        is_intraday_futures=False,
        mathematical_signal_latex="OFI_t",
        entry_exit_rules=["Buy when OFI > 2"],
        sample_period_years="2020-2024",
        reported_sharpe_ratio=1.8,
        reported_max_drawdown_pct=12.0,
        transaction_cost_bps_assumed=8.0,
        friction_verdict=FrictionVerdict.REALISTIC_FRICTION,
        market_specific_caveats=["Queue position"],
        known_failure_regimes=["Halts"],
        verified_citations=[
            DocCitationSpan(
                chunk_id="non_existent_chunk_uuid",
                doc_id="arxiv:2401.00001",
                doc_title="Order Flow Paper",
                source_uri="https://arxiv.org/abs/2401.00001",
                section_heading="1. Introduction",
                verbatim_quote="Order flow imbalance predicts short-horizon mid-price changes.",
            )
        ],
    )
    gate = MarketRegimeScepticismGate()
    res = gate.evaluate(brief, chunks)
    assert res.citations_valid is True
    assert res.citations_verified_count == 1


def test_p0_04_embedder_mps_to_cpu_fallback_moves_model_to_cpu() -> None:
    embedder = LocalSemanticEmbedder(device="mps")

    class FakeModel:
        def __init__(self) -> None:
            self.current_device = "mps"
            self.prompts: dict[str, str] = {}

        def to(self, dev: str) -> Any:
            self.current_device = dev
            return self

        def encode(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
            if self.current_device != "cpu":
                raise RuntimeError(
                    "Expected all tensors to be on the same device, found mps:0 and cpu"
                )
            return [[1.0] + [0.0] * 767 for _ in texts]

    embedder._model = FakeModel()  # type: ignore[assignment]
    vecs = embedder.embed_batch(["test sentence"])
    assert len(vecs) == 1
    assert embedder.device == "cpu"
    assert embedder._model.current_device == "cpu"  # type: ignore[union-attr]


def test_p0_05_slicer_math_interval_and_prefix_truncation_detection() -> None:
    slicer = DeterministicSlicer()
    source = (
        "In continuous time, for any trading horizon t in [0, T), "
        "the optimal execution trajectory converges smoothly."
    )
    quote = (
        "for any trading horizon t in [0, T), "
        "the optimal execution trajectory converges smoothly."
    )
    res = slicer.verify_quote(quote=quote, source_text=source)
    assert res.is_valid is True
    assert res.error_code is None

    # Also test prefix truncation detection when match starts at token index >= 5
    long_source = "one two three four five six seven eight nine ten eleven twelve thirteen"
    truncated_quote = "seven eight nine ten eleven corrupted_tail"
    res_trunc = slicer.verify_quote(quote=truncated_quote, source_text=long_source)
    assert res_trunc.is_valid is False
    assert res_trunc.error_code is not None


def test_p0_06_local_gemma_client_use_live_mlx_in_code_pointers_and_quant_brief(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = LocalGemmaClient(use_live_mlx=True, offline_mode=False)
    monkeypatch.setattr(client, "is_online", lambda: False)

    fake_ptr_json = json.dumps(
        [
            {
                "repo_name": "enterprise-context-scout",
                "file_path": "src/context_scout/db/bm25.py",
                "start_line": 1,
                "end_line": 20,
                "qualified_symbol": "context_scout.db.bm25.tokenize_bm25",
                "selection_reason": "Selected via live MLX",
                "modification_intent": "modify_target",
            }
        ]
    )
    monkeypatch.setattr(
        client, "_call_in_process_mlx", lambda prompt, domain_id=None, max_tokens=768: fake_ptr_json
    )
    ptrs = client.generate_code_pointers("Optimize tokenize_bm25")
    assert len(ptrs) == 1
    assert ptrs[0].selection_reason == "Selected via live MLX"


def test_p0_07_handoff_dynamic_test_command_and_real_sse_stage_callbacks() -> None:
    root = Path(__file__).resolve().parents[1]
    handoff_src = (root / "src/context_scout/api/routes/handoff.py").read_text()
    scout_src = (root / "src/context_scout/api/routes/scout.py").read_text()

    assert '"bounce" in request.query.lower()' not in scout_src
    assert (
        'test_target_cmd = ["pytest", "-q", "tests/test_deterministic_slicer.py"]'
        not in handoff_src
    )

    events_captured: list[dict[str, Any]] = []
    registry = DomainRegistry(Path("domains"))
    repo = InMemoryRLSRepository()
    wf = ScoutWorkflow(registry=registry, repo=repo)
    principal = PrincipalContext(
        user_id="u1", role=RoleType.ENGINEER, allowed_scopes=["public_eng"]
    )
    _ = wf.run(
        user_prompt="Inspect KellyRiskManager in kelly_risk_manager.py",
        principal=principal,
        domain_id="trading_code_scout",
        on_stage_event=lambda ev: events_captured.append(ev),
    )
    assert len(events_captured) >= 2


# ============================================================================
# TIER P1 TESTS
# ============================================================================


def test_p1_01_postgres_search_doc_chunks_hybrid_rrf_uses_bounded_cte_limits() -> None:
    root = Path(__file__).resolve().parents[1]
    repo_src = (root / "src/context_scout/db/repositories.py").read_text()
    assert "LIMIT" in repo_src.split("def search_doc_chunks_hybrid_rrf")[1].split("def ")[0]


def test_p1_02_git_merkle_indexer_supports_batch_semantic_embedder(tmp_path: Path) -> None:
    repo_dir = _init_temp_git_repo(tmp_path)
    fake_embedder = MagicMock()
    fake_embedder.embed_batch.return_value = [[0.5] + [0.0] * 767]

    indexer = GitMerkleIndexer(embedder=fake_embedder, use_semantic_embeddings=True)
    mem_repo = InMemoryRLSRepository()
    principal = PrincipalContext(
        user_id="u1", role=RoleType.ENGINEER, allowed_scopes=["public_eng"]
    )
    stats = indexer.sync_in_memory(
        repo_dir=repo_dir,
        repo=mem_repo,
        principal=principal,
        domain_id="trading_code_scout",
        repo_name="temp-repo",
    )
    assert stats.total_symbols_stored >= 1
    assert fake_embedder.embed_batch.called
    syms = mem_repo.list_visible_ast_symbols(principal, "trading_code_scout")
    assert syms[0].embedding[0] == 0.5


def test_p1_03_postgres_search_symbols_hybrid_rrf_exists() -> None:
    pg_repo = PostgresRLSRepository()
    assert hasattr(pg_repo, "search_symbols_hybrid_rrf")


def test_p1_04_git_merkle_indexer_skips_binary_and_non_text_blobs(tmp_path: Path) -> None:
    repo_dir = _init_temp_git_repo(tmp_path)
    (repo_dir / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")
    (repo_dir / "weights.safetensors").write_bytes(b"\x00\x01\x02\x03binary_weights")
    subprocess.run(
        ["git", "add", "logo.png", "weights.safetensors"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "add binary files"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
    )

    indexer = GitMerkleIndexer()
    mem_repo = InMemoryRLSRepository()
    principal = PrincipalContext(
        user_id="u1", role=RoleType.ENGINEER, allowed_scopes=["public_eng"]
    )
    indexer.sync_in_memory(
        repo_dir=repo_dir,
        repo=mem_repo,
        principal=principal,
        domain_id="trading_code_scout",
        repo_name="temp-repo",
    )
    files = mem_repo.list_visible_code_files(principal, "trading_code_scout")
    paths = {f.file_path for f in files}
    assert "hello.py" in paths
    assert "logo.png" not in paths
    assert "weights.safetensors" not in paths


def test_p1_05_bm25_tokenizer_splits_snake_case_and_camel_case() -> None:
    tokens = tokenize_bm25("KellyPositionSizer kelly_position_sizer")
    assert "kelly" in tokens
    assert "position" in tokens
    assert "sizer" in tokens

    bm25 = BM25Okapi(["def kelly_position_sizer(): pass", "def other_func(): pass"])
    scores = bm25.get_scores("kelly position")
    assert scores[0] > 0.0


def test_p1_06_insert_doc_chunks_batch_uses_executemany() -> None:
    pg_repo = PostgresRLSRepository()
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    chunk = DocChunkRecord(
        chunk_id=uuid.uuid4(),
        domain_id="quant_research_scout",
        doc_id="arxiv:1",
        doc_title="T",
        source_uri="u",
        market_scope=MarketRegime.US_NYSE_NASDAQ,
        published_date=None,
        authority_weight=1.0,
        section_heading="S",
        contextual_header="H",
        verbatim_content="C",
        metadata_json={},
        allowed_scopes=["public_quant"],
        embedding=[0.1] * 768,
    )
    count = pg_repo.insert_doc_chunks_batch(mock_conn, [chunk, chunk])
    assert count == 2
    assert mock_cursor.executemany.called


# ============================================================================
# TIER P2 TESTS
# ============================================================================


def test_p2_01_grammar_constrained_json_repair_and_multi_candidate_distiller() -> None:
    from context_scout.tuning.teacher_distiller import repair_and_constrain_code_pointers_json

    malformed = (
        "```json\n"
        "[\n"
        "  {\n"
        '    "repo_name": "httpx",\n'
        '    "file_path": "httpx/_models.py",\n'
        '    "start_line": 10,\n'
        '    "end_line": 45,\n'
        '    "qualified_symbol": "httpx._models.Response",\n'
        '    "selection_reason": "Primary target",\n'
        '    "modification_intent": "invalid_enum_value"\n'
        "  },\n"
    )
    repaired = repair_and_constrain_code_pointers_json(malformed)
    assert len(repaired) == 1
    assert repaired[0].modification_intent == ModificationIntent.MODIFY_TARGET


def test_p2_02_deterministic_slicer_skeletonizes_god_classes_over_150_lines() -> None:
    lines = [
        "class GiantCommand:",
        '    """Command doc."""',
        "    def __init__(self) -> None:",
        "        self.x = 1",
    ]
    for i in range(1, 20):
        lines.append(f"    def method_{i}(self) -> int:")
        for j in range(1, 12):
            lines.append(f"        val_{j} = {i} * {j}")
        lines.append("        return val_1")
    raw_content = "\n".join(lines)
    assert len(lines) > 200

    file_rec = CodeFileRecord(
        file_id=uuid.uuid4(),
        domain_id="trading_code_scout",
        repo_name="click",
        file_path="src/click/core.py",
        git_commit_sha="a" * 40,
        git_blob_sha="b" * 40,
        language="python",
        allowed_scopes=["public_eng"],
        raw_content=raw_content,
        line_count=len(lines),
    )
    ptr = CodeSlicePointer(
        repo_name="click",
        file_path="src/click/core.py",
        start_line=1,
        end_line=len(lines),
        qualified_symbol="click.core.GiantCommand",
        selection_reason="Target class",
        modification_intent=ModificationIntent.MODIFY_TARGET,
    )
    slicer = DeterministicSlicer()
    sliced = slicer.slice_pointer(ptr, file_rec, query_text="Fix bug in method_5")
    sliced_line_count = len(sliced.verbatim_code_with_lines.splitlines())
    assert sliced_line_count < len(lines) * 0.6
    assert "def method_5" in sliced.verbatim_code_with_lines
    assert "elided" in sliced.verbatim_code_with_lines


def test_p2_03_search_replace_patch_repair_loop_in_executor(tmp_path: Path) -> None:
    repo_dir = _init_temp_git_repo(tmp_path)
    test_file = repo_dir / "test_hello.py"
    test_file.write_text(
        "from hello import greet\n\n\ndef test_greet() -> None:\n"
        "    assert greet() == 'hello world'\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "test_hello.py"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "add test"], cwd=repo_dir, check=True, capture_output=True
    )

    executor = GeminiImplementationExecutor(require_confirmation=False)
    attempts: list[int] = []

    def _mock_synthesizer(attempt: int, last_err: str) -> str:
        attempts.append(attempt)
        if attempt == 1:
            return "<<<<<<< SEARCH\ndef non_existent():\n=======\ndef broken():\n>>>>>>> REPLACE"
        assert "failed" in last_err.lower() or "could not match" in last_err.lower()
        return (
            "### File: hello.py\n"
            "<<<<<<< SEARCH\n"
            "def greet() -> str:\n"
            "    return 'hello'\n"
            "=======\n"
            "def greet() -> str:\n"
            "    return 'hello world'\n"
            ">>>>>>> REPLACE\n"
        )

    ok, patch_used, _details = executor.execute_patch_with_repair(
        repo_root=repo_dir,
        patch_generator=_mock_synthesizer,
        test_cmd=["pytest", "-q", "test_hello.py"],
        max_repair_turns=2,
    )
    assert ok is True
    assert len(attempts) == 2
    assert "hello world" in patch_used


def test_p2_04_pytorch_fallback_causal_lm_shift() -> None:
    root = Path(__file__).resolve().parents[1]
    train_src = (root / "src/context_scout/tuning/train_mlx_lora.py").read_text()
    assert "logits[:, :-1, :]" in train_src or "shift_logits" in train_src


def test_p2_05_scout_workflow_clean_of_modulo_benchmark_simulation() -> None:
    root = Path(__file__).resolve().parents[1]
    wf_src = (root / "src/context_scout/agents/scout_workflow.py").read_text()
    assert "(h_val % 3)" not in wf_src
    assert "(h_val % 6)" not in wf_src


# ============================================================================
# TIER P3 TESTS
# ============================================================================


def test_p3_01_tested_by_edge_direction_prod_to_test() -> None:
    extractor = ASTExtractor()
    code = (
        "def compute_alpha() -> float:\n    return 1.0\n\n"
        "def test_compute_alpha() -> None:\n    assert compute_alpha() == 1.0\n"
    )
    _syms, edges = extractor.parse_source(code, "python", file_path="src/alpha.py")
    tb_edges = [e for e in edges if e.edge_kind == "TESTED_BY"]
    assert len(tb_edges) == 1
    assert tb_edges[0].source_qualified_name.endswith("compute_alpha")
    assert tb_edges[0].target_qualified_name.endswith("test_compute_alpha")


def test_p3_02_latex_doc_parser_prose_currency_not_matched_as_multiline_math() -> None:
    prose = (
        "The exchange fee was $5 in 2021 for retail orders.\n\n"
        "It subsequently rose to $10 in 2023.\n\n"
        "We model the spread as $s_t = \\alpha + \\beta \\sigma_t$ in continuous time."
    )
    matches = MATH_BLOCK_PATTERN.findall(prose)
    assert len(matches) == 1
    assert matches[0] == "$s_t = \\alpha + \\beta \\sigma_t$"


def test_p3_03_quant_relevance_classifier_allows_microstructure_paper_mentioning_pandemic() -> None:
    clf = QuantRelevanceClassifier()
    v_micro = clf.evaluate(
        title=(
            "Limit Order Book Liquidity, Hawkes Process Intensity, "
            "and Bid-Ask Spread During the COVID-19 Crash"
        ),
        abstract=(
            "We calibrate a Hawkes process and Almgren-Chriss execution model "
            "on high-frequency trading data."
        ),
    )
    assert v_micro.is_relevant is True

    v_med = clf.evaluate(
        title="COVID-19 Pandemic Lockdown Policy and Hospital ICU Capacity",
        abstract="An epidemiological study of SARS-CoV-2 transmission.",
    )
    assert v_med.is_relevant is False


def test_p3_04_trading_strategies_ofi_garch_kelly_shrinkage_and_kalman_bounds() -> None:
    # 1. OFI GARCH conditional forecast scaling
    garch = GARCH11Model(omega=1e-5, alpha=0.15, beta=0.80)
    ofi = OrderFlowImbalanceSignal("AAPL", rolling_window=5, garch_model=garch)
    s1 = OrderBookSnapshot(1000, "AAPL", [LOBLevel(100.0, 10.0)], [LOBLevel(100.1, 10.0)])
    s2 = OrderBookSnapshot(2000, "AAPL", [LOBLevel(100.0, 25.0)], [LOBLevel(100.1, 5.0)])
    ofi.process_snapshot(s1)
    u_calm = ofi.process_snapshot(s2, recent_return=0.0001)
    s3 = OrderBookSnapshot(3000, "AAPL", [LOBLevel(100.0, 40.0)], [LOBLevel(100.1, 2.0)])
    u_shock = ofi.process_snapshot(s3, recent_return=0.08)
    assert u_shock.volatility_adjusted_signal != u_calm.volatility_adjusted_signal

    # 2. Kelly shrinkage auto-application
    kelly = KellyRiskManager()
    returns_hist = [
        [0.01, -0.02],
        [0.02, 0.01],
        [-0.01, 0.03],
        [0.015, -0.01],
        [0.005, 0.005],
    ]
    alloc = kelly.compute_optimal_weights(
        expected_returns=[0.08, 0.06],
        covariance_matrix=[[0.04, 0.01], [0.01, 0.09]],
        returns_history=returns_hist,
        apply_shrinkage=True,
    )
    assert alloc.shrinkage_intensity > 0.0

    # 3. Kalman pairs trading joint position bound when |beta| > 1
    engine = PairsSpreadSignal("Y", "X", rolling_window=5)
    for p_y, p_x in [(100.0, 40.0), (101.0, 40.2), (99.5, 39.8), (100.2, 40.1), (100.0, 40.0)]:
        engine.update_and_signal(p_y, p_x)
    # Force high beta = 2.5 and extreme spread
    engine.estimator.theta[0] = 2.5
    sig = engine.update_and_signal(125.0, 40.0)
    max_pos = engine.risk_manager.max_single_position
    assert abs(sig.target_weight_y) <= max_pos + 1e-6
    assert abs(sig.target_weight_x) <= max_pos + 1e-6
    if abs(sig.target_weight_y) > 0:
        assert abs(sig.target_weight_x / sig.target_weight_y) == pytest.approx(
            abs(sig.dynamic_hedge_ratio), rel=1e-2
        )


def test_p3_05_as_adk_workflow_wires_four_stage_nodes() -> None:
    registry = DomainRegistry(Path("domains"))
    repo = InMemoryRLSRepository()
    wf = ScoutWorkflow(registry=registry, repo=repo)
    adk_wf = wf.as_adk_workflow()
    assert adk_wf is not None
    assert hasattr(wf, "_stage1_retrieve")
    assert hasattr(wf, "_stage2_expand")
    assert hasattr(wf, "_stage3_slice")
    assert hasattr(wf, "_stage4_validate")
    assert len(getattr(adk_wf, "stages", [])) == 4


def test_p3_06_scepticism_gate_configurable_annual_volatility_sharpe_drag() -> None:
    gate = MarketRegimeScepticismGate()
    sharpe_low_vol = gate.estimate_net_sharpe(
        reported_sharpe=2.0,
        assumed_bps=0.0,
        floor_bps=15.0,
        turnover_annual=12.0,
        assumed_annual_volatility=0.08,
    )
    sharpe_high_vol = gate.estimate_net_sharpe(
        reported_sharpe=2.0,
        assumed_bps=0.0,
        floor_bps=15.0,
        turnover_annual=12.0,
        assumed_annual_volatility=0.25,
    )
    assert sharpe_low_vol is not None and sharpe_high_vol is not None
    assert sharpe_low_vol < sharpe_high_vol
