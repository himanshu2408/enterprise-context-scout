"""Permanent regression test guards for all 7 enterprise audit remediations.

Ensures none of the 7 issues/shortcuts identified during the post-M8 enterprise
audit can ever regress:
1. Distilled JSONL prompts always inject Retrieved AST Candidate / Corpus Chunk
   metadata, order single-turn before bounce-back, and filter out LaTeX % comments.
2. Live QLoRA verification reports enforce 100% schema adherence AND factual grounding
   with zero 'Hallucinated econometric quote' strings.
3. LocalGemmaClient supports in-process use_live_mlx and parses Retrieved Corpus Chunk
   metadata directly.
4. /handoff/apply runs real ruff check and pytest in GitWorktreeSandbox (never sys.exit(0)),
   and /handoff/preview wires _generate_gemini_patch when api_key is configured.
5. /scout/query (federated_all) resolves live visible doc_chunks from the repository via
   _resolve_federated_doc_chunks.
6. /telemetry/roi queries live PostgreSQL 17 execution_telemetry when backed by
   PostgresScoutGraphAdapter.
7. GitHub Actions CI (.github/workflows/ci.yml) exists and defines backend-verify and
   frontend-verify jobs.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

from context_scout.agents.scout_workflow import LocalGemmaClient
from context_scout.api.routes import handoff, scout, telemetry
from context_scout.schemas.contracts import PrincipalContext, RoleType


def test_distilled_prompts_inject_retrieved_context_and_filter_latex_comments() -> None:
    """Guard 1: Distilled datasets must inject retrieved context and never quote % comments."""
    code_train = Path("data/tuning/trading_code_scout/train.jsonl")
    quant_train = Path("data/tuning/quant_research_scout/train.jsonl")
    quant_valid = Path("data/tuning/quant_research_scout/valid.jsonl")

    with open(code_train, encoding="utf-8") as f:
        code_records = [json.loads(line) for line in f if line.strip()]
    assert len(code_records) >= 100
    assert "Retrieved AST Candidate" in code_records[0]["messages"][0]["content"]
    assert len(code_records[0]["messages"]) == 2, "Single-turn samples must precede bounce-back"

    for path in (quant_train, quant_valid):
        with open(path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        assert len(records[0]["messages"]) == 2, "Single-turn samples must precede bounce-back"
        for rec in records:
            msgs = rec["messages"]
            assert "Retrieved Corpus Chunk: [" in msgs[0]["content"]
            final_brief = json.loads(msgs[-1]["content"])
            for cit in final_brief["verified_citations"]:
                quote = cit["verbatim_quote"].strip()
                assert not quote.startswith("%"), f"LaTeX comment leaked as quote: {quote}"
                assert "Hallucinated econometric quote" not in quote


def test_live_verification_reports_enforce_schema_and_grounding() -> None:
    """Guard 2: Live QLoRA verification reports must pass schema AND factual grounding."""
    paths = [
        Path(
            "domains/trading_code_scout/adapters/gemma4_26b_code_scout/"
            "live_verification_report.json"
        ),
        Path(
            "domains/quant_research_scout/adapters/gemma4_26b_quant_research/"
            "live_verification_report.json"
        ),
    ]
    for report_path in paths:
        assert report_path.is_file()
        with open(report_path, encoding="utf-8") as f:
            rep = json.load(f)
        assert rep["all_valid"] is True
        assert not str(rep["adapter_dir"]).startswith("/Users/")
        for item in rep["results"]:
            assert item["is_valid_json"] is True
            assert item["is_valid_schema"] is True
            assert item["is_grounded"] is True
            assert "Hallucinated econometric quote" not in item["raw_output"]


def test_local_gemma_client_supports_use_live_mlx_and_rag_chunk_parsing() -> None:
    """Guard 3: LocalGemmaClient supports use_live_mlx and parses Retrieved Corpus Chunk."""
    client = LocalGemmaClient(offline_mode=True, use_live_mlx=True)
    assert client.use_live_mlx is True

    prompt = (
        "Synthesize a market-friction verified QuantStrategyBrief for `Test Paper` "
        "in the DE_XETRA regime.\n"
        "Retrieved Corpus Chunk: [chunk_id=chk-999 | doc_id=arxiv:2610.99999 | "
        "title=Test Paper | uri=https://arxiv.org/abs/2610.99999 | "
        "section=Methodology | strategy=DAX Spread Strategy | sharpe=1.9 | "
        'mdd=7.5 | bps=15.0 | quote="Exact empirical quote from methodology section."]'
    )
    out = client.complete(prompt=prompt, domain_id="quant_research_scout")
    parsed = json.loads(out)
    cit = parsed["verified_citations"][0]
    assert cit["chunk_id"] == "chk-999"
    assert cit["doc_id"] == "arxiv:2610.99999"
    assert cit["doc_title"] == "Test Paper"
    assert "Exact empirical quote from methodology section." in cit["verbatim_quote"]


def test_handoff_route_uses_real_ruff_and_pytest_commands_not_sys_exit_0() -> None:
    """Guard 4: handoff.py must never use dummy sys.exit(0) sandbox commands."""
    src = inspect.getsource(handoff)
    assert "sys.exit(0)" not in src
    assert "_generate_gemini_patch" in src
    assert '"ruff"' in src
    assert '"pytest"' in src


def test_federated_route_resolves_live_doc_chunks_from_repository() -> None:
    """Guard 5: _resolve_federated_doc_chunks must query repository visible doc_chunks."""
    from context_scout.api.dependencies import _populate_in_memory_repository, get_repo_root

    repo = _populate_in_memory_repository(get_repo_root())
    principal = PrincipalContext(
        user_id="quant_lead_01",
        role=RoleType.QUANT_RESEARCHER,
        allowed_scopes=[
            "public_eng",
            "public_quant",
            "repo:trading_strategies",
            "corpus:quant_papers",
            "quant_restricted",
        ],
    )
    chunks = scout._resolve_federated_doc_chunks(repo, principal)
    assert len(chunks) >= 2
    assert "chk_quant_default" in chunks


def test_telemetry_roi_has_live_postgres_aggregation_path() -> None:
    """Guard 6: telemetry.py must include live PostgreSQL execution_telemetry aggregation."""
    src = inspect.getsource(telemetry.get_roi_metrics)
    assert "FROM execution_telemetry" in src
    assert "SUM(paid_tokens_saved)" in src
    assert "SUM(estimated_usd_saved)" in src


def test_github_actions_workflow_exists_and_valid() -> None:
    """Guard 7: .github/workflows/ci.yml must exist and define backend + frontend verification."""
    ci_path = Path(".github/workflows/ci.yml")
    assert ci_path.is_file()
    content = ci_path.read_text(encoding="utf-8")
    assert "backend-verify:" in content
    assert "frontend-verify:" in content
    assert "pgvector/pgvector:pg17" in content


def test_vite_proxy_targets_fastapi_port_8080_not_mlx_port_8000() -> None:
    """Guard 8: vite.config.ts /api proxy must target FastAPI (:8080), never MLX (:8000)."""
    vite_cfg = Path("web-ui/vite.config.ts")
    assert vite_cfg.is_file()
    content = vite_cfg.read_text(encoding="utf-8")
    assert '"http://127.0.0.1:8080"' in content
    assert '"http://127.0.0.1:8000"' not in content


def test_frontend_client_uses_pack_field_and_includes_federated_all() -> None:
    """Guard 9: client.ts must send 'pack: contextPack' in previewHandoff and federated_all."""
    client_ts = Path("web-ui/src/api/client.ts").read_text(encoding="utf-8")
    assert "pack: contextPack," in client_ts
    assert "context_pack: contextPack," not in client_ts
    assert '"federated_all"' in client_ts


def test_frontend_sse_propagates_403_guardrail_errors_without_mock_swallow() -> None:
    """Guard 10: web-ui sse.ts must parse HTTP 403 JSON detail and invoke onError directly."""
    sse_ts = Path("web-ui/src/api/sse.ts").read_text(encoding="utf-8")
    assert "response.status === 403" in sse_ts
    assert "SECURITY_GUARDRAIL_TRIGGERED" in sse_ts


def test_scout_workflow_hydrates_quant_strategy_spec_for_quant_domain() -> None:
    """Guard 11: ScoutWorkflow must hydrate quant_strategy_spec for quant_research_scout."""
    from context_scout.agents.scout_workflow import ScoutWorkflow
    from context_scout.api.dependencies import (
        _populate_in_memory_repository,
        get_domain_registry,
        get_repo_root,
    )
    from context_scout.schemas.contracts import ImplementationContextPack

    repo = _populate_in_memory_repository(get_repo_root())
    wf = ScoutWorkflow(registry=get_domain_registry(), repo=repo)
    principal = PrincipalContext(
        user_id="quant_tester",
        role=RoleType.QUANT_RESEARCHER,
        allowed_scopes=["public_quant", "quant_restricted", "corpus:quant_papers"],
    )
    res = wf.run(
        user_prompt="Synthesize a QuantStrategyBrief on Order Flow Imbalance in US equities",
        principal=principal,
        domain_id="quant_research_scout",
    )
    assert isinstance(res, ImplementationContextPack)
    assert res.quant_strategy_spec is not None
    assert len(res.supporting_doc_citations) >= 1


def test_scout_route_matches_hyphenated_bounce_back_prompts() -> None:
    """Guard 12: scout.py must match 'bounce' prompts without requiring underscore."""
    src = inspect.getsource(scout)
    assert '"bounce" in request.query.lower()' in src
    assert '"bounce_back" in request.query.lower()' not in src


def test_github_actions_ci_installs_dev_dependencies_and_guards_mlx_markers() -> None:
    """Guard 13: GitHub Actions CI must install dev tools and gate MLX to Darwin arm64."""
    ci_yml = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    assert "uv sync --extra dev" in ci_yml
    assert "[dependency-groups]" in pyproject
    assert "sys_platform == 'darwin' and platform_machine == 'arm64'" in pyproject


def test_hardware_profile_auto_detects_and_honors_env_overrides_without_hardcoding(
    monkeypatch: object,
) -> None:
    """Guard 14: Hardware profile must auto-detect or respect env overrides (no hardcoding)."""
    import pytest

    from context_scout.api.routes import health
    from context_scout.config import detect_hardware_profile

    health_src = inspect.getsource(health)
    assert "Apple M5 Pro 48GB" not in health_src
    assert "detect_hardware_profile" in health_src

    mp = pytest.MonkeyPatch()
    try:
        mp.setenv("SCOUT_HARDWARE_DEVICE", "Linux Workstation (16GB RAM)")
        mp.setenv("SCOUT_HARDWARE_MEMORY_GB", "16.0")
        mp.delenv("SCOUT_LOCAL_MAX_KV_TOKENS", raising=False)
        prof = detect_hardware_profile()
        assert prof["device"] == "Linux Workstation (16GB RAM)"
        assert prof["memory_gb"] == 16.0
        assert prof["max_kv_tokens"] == 8192
    finally:
        mp.undo()


def test_ci_env_prefix_and_worktree_git_identity_config() -> None:
    """Guard 15: CI must use SCOUT_DB_* env vars and GitWorktreeSandbox must set git identity."""
    from context_scout.agents.gemini_executor import GitWorktreeSandbox

    ci_yml = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "SCOUT_DB_ADMIN_URL:" in ci_yml
    assert "SCOUT_DB_APP_URL:" in ci_yml
    assert "CONTEXT_SCOUT_DB_" not in ci_yml

    wt_src = inspect.getsource(GitWorktreeSandbox.confirm_and_merge)
    assert "user.name=" in wt_src
    assert "user.email=" in wt_src


