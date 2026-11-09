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
import re
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
    """Guard 12: ScoutWorkflow and scout.py must support hyphenated bounce-back via callbacks."""
    src = inspect.getsource(scout)
    assert '"bounce_back" in request.query.lower()' not in src
    assert "on_stage_event" in src


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


def test_indian_stock_market_momentum_quant_query_and_citation_inspector_auto_tab() -> None:
    """Guard 16: Indian momentum query returns IN_NSE strategy and UI auto-switches tab."""
    from context_scout.agents.scout_workflow import ScoutWorkflow
    from context_scout.api.dependencies import (
        DEFAULT_ROLE_SCOPES,
        get_domain_registry,
        get_repository,
    )
    from context_scout.schemas.contracts import (
        ImplementationContextPack,
        MarketRegime,
    )

    assert "public_quant" in DEFAULT_ROLE_SCOPES[RoleType.ENGINEER]

    repo = get_repository()
    wf = ScoutWorkflow(registry=get_domain_registry(), repo=repo)
    principal = PrincipalContext(
        user_id="demo_engineer",
        role=RoleType.ENGINEER,
        allowed_scopes=DEFAULT_ROLE_SCOPES[RoleType.ENGINEER],
    )
    res = wf.run(
        user_prompt="best way to implement momentum strategy in indian stock market?",
        principal=principal,
        domain_id="quant_research_scout",
    )
    assert isinstance(res, ImplementationContextPack)
    assert res.preflight_report.passed is True
    assert res.quant_strategy_spec is not None
    assert MarketRegime.INDIA_NSE_BSE in res.quant_strategy_spec.target_markets
    assert "Momentum" in res.quant_strategy_spec.strategy_name
    assert res.quant_strategy_spec.transaction_cost_bps_assumed is not None
    assert res.quant_strategy_spec.transaction_cost_bps_assumed >= 15.0
    assert len(res.supporting_doc_citations) >= 1

    inspector_tsx = Path("web-ui/src/components/CitationInspector.tsx").read_text(encoding="utf-8")
    assert 'setActiveTab("quant")' in inspector_tsx


def test_multi_persona_natural_language_queries_across_code_and_quant_domains() -> None:
    """Guard 17: Unquoted natural-language persona queries localize exact code & quant packs."""
    from context_scout.agents.scout_workflow import ScoutWorkflow
    from context_scout.api.dependencies import (
        DEFAULT_ROLE_SCOPES,
        get_domain_registry,
        get_repository,
    )
    from context_scout.evals.golden_dataset import get_default_golden_suite
    from context_scout.schemas.contracts import ImplementationContextPack

    suite = get_default_golden_suite()
    assert len(suite.code_tasks) >= 8
    assert len(suite.quant_tasks) >= 6
    assert len(suite.federated_tasks) >= 3

    repo = get_repository()
    wf = ScoutWorkflow(registry=get_domain_registry(), repo=repo)
    principal = PrincipalContext(
        user_id="persona_tester",
        role=RoleType.ENGINEER,
        allowed_scopes=DEFAULT_ROLE_SCOPES[RoleType.ENGINEER],
    )

    for task in suite.code_tasks[5:]:
        pack = wf.run(
            user_prompt=task.user_prompt,
            principal=principal,
            domain_id="trading_code_scout",
        )
        assert isinstance(pack, ImplementationContextPack)
        assert pack.preflight_report.passed is True
        found_files = {s.file_path for s in pack.code_slices[:5]}
        assert set(task.ground_truth_files).issubset(found_files)

    for qtask in suite.quant_tasks[3:]:
        qpack = wf.run(
            user_prompt=qtask.user_prompt,
            principal=principal,
            domain_id="quant_research_scout",
        )
        assert isinstance(qpack, ImplementationContextPack)
        assert qpack.preflight_report.passed is True
        assert qpack.quant_strategy_spec is not None
        assert qtask.target_market in qpack.quant_strategy_spec.target_markets
        assert qpack.quant_strategy_spec.transaction_cost_bps_assumed is not None
        assert qpack.quant_strategy_spec.transaction_cost_bps_assumed >= qtask.min_friction_bps


def test_auto_quant_intent_routing_and_executive_answer_ui() -> None:
    """Guard 18: Quant research queries on trading_code_scout auto-route to QuantStrategyBrief."""
    from context_scout.agents.scout_workflow import ScoutWorkflow
    from context_scout.api.dependencies import (
        DEFAULT_ROLE_SCOPES,
        get_domain_registry,
        get_repository,
    )
    from context_scout.schemas.contracts import ImplementationContextPack

    repo = get_repository()
    wf = ScoutWorkflow(registry=get_domain_registry(), repo=repo)
    principal = PrincipalContext(
        user_id="demo_engineer",
        role=RoleType.ENGINEER,
        allowed_scopes=DEFAULT_ROLE_SCOPES[RoleType.ENGINEER],
    )
    res = wf.run(
        user_prompt=(
            "Can deep reinforcement learning or actor-critic trading agents outperform "
            "in US equity markets?"
        ),
        principal=principal,
        domain_id="trading_code_scout",
    )
    assert isinstance(res, ImplementationContextPack)
    assert res.preflight_report.passed is True
    assert res.quant_strategy_spec is not None
    assert "Reinforcement Learning" in res.quant_strategy_spec.strategy_name
    assert len(res.code_slices) == 0

    inspector_tsx = Path("web-ui/src/components/CitationInspector.tsx").read_text(encoding="utf-8")
    assert 'data-testid="quant-executive-answer"' in inspector_tsx
    assert 'data-testid="entry-exit-rules-list"' in inspector_tsx


def test_custom_biology_and_bioinformatics_domain_end_to_end(tmp_path: Path) -> None:
    """Guard 19: Custom non-finance domains (Biology research & Bioinformatics code) work E2E."""
    from context_scout.agents.domain_registry import DomainRegistry
    from context_scout.agents.scout_workflow import ScoutWorkflow
    from context_scout.cli import (
        cmd_distill_custom_domain,
        cmd_init_domain,
        cmd_train_domain,
    )
    from context_scout.schemas.contracts import ImplementationContextPack

    domains_dir = tmp_path / "domains"
    data_dir = tmp_path / "data" / "tuning"
    bio_docs_dir = tmp_path / "bio_corpus"
    bio_code_dir = tmp_path / "bio_code"
    bio_docs_dir.mkdir(parents=True, exist_ok=True)
    bio_code_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create a realistic CRISPR-Cas9 molecular biology paper
    (bio_docs_dir / "crispr_cas9_kinetics.md").write_text(
        "# CRISPR-Cas9 Guide RNA Specificity and Cleavage Kinetics\n\n"
        "## 2. Thermodynamic Mismatch Penalty Model\n\n"
        "Cas9 endonuclease cleavage rate decreases exponentially as PAM-proximal seed "
        "region mismatches destabilize R-loop formation.\n"
        "The empirical cleavage rate follows $k_{\\text{cleave}} = k_{\\max} \\cdot "
        "\\exp(-\\Delta G_{\\text{mismatch}} / RT)$ across in-vitro human HEK293T assays.\n"
        "Experimental assays require 10 mM Mg2+ cofactor concentration and 37C physiological "
        "temperature control to maintain catalytic HNH domain conformation.\n"
        "Severe off-target cleavage failure occurs when PAM-distal mismatches at positions "
        "18-20 fail to trigger conformational R-loop collapse.\n",
        encoding="utf-8",
    )

    # 2. Scaffold, distill, and train the custom `crispr_biology_scout` research domain
    cmd_init_domain(
        domain_id="crispr_biology_scout",
        kind="research",
        display_name="CRISPR & Molecular Biology Research Scout",
        domains_dir=domains_dir,
    )
    distill_res = cmd_distill_custom_domain(
        domain_id="crispr_biology_scout",
        source_dir=bio_docs_dir,
        num_samples=10,
        domains_dir=domains_dir,
        data_dir=data_dir,
    )
    assert distill_res["train_samples"] >= 8
    assert (domains_dir / "crispr_biology_scout" / "corpus_index.json").exists()

    train_res = cmd_train_domain(
        domain_id="crispr_biology_scout",
        steps=4,
        domains_dir=domains_dir,
        data_dir=data_dir,
    )
    assert train_res["adapter_verified"] is True

    # 3. Query `crispr_biology_scout` with a natural-language biology question
    registry = DomainRegistry(domains_dir=domains_dir)
    wf = ScoutWorkflow(registry=registry, repo=None)
    principal = PrincipalContext(
        user_id="bio_researcher",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng", "public_quant"],
    )
    bio_pack = wf.run(
        user_prompt=(
            "How does Cas9 guide RNA mismatch affect off-target cleavage kinetics "
            "in CRISPR gene editing?"
        ),
        principal=principal,
        domain_id="crispr_biology_scout",
    )
    assert isinstance(bio_pack, ImplementationContextPack)
    assert bio_pack.preflight_report.passed is True
    assert len(bio_pack.code_slices) == 0
    assert bio_pack.quant_strategy_spec is not None
    assert bio_pack.domain_knowledge_brief is not None
    assert bio_pack.quant_strategy_spec.reported_sharpe_ratio is None
    assert bio_pack.quant_strategy_spec.reported_max_drawdown_pct is None
    assert bio_pack.quant_strategy_spec.transaction_cost_bps_assumed is None
    assert "CRISPR-Cas9 Guide RNA Specificity" in bio_pack.quant_strategy_spec.strategy_name
    assert "CRISPR-Cas9 Guide RNA Specificity" in bio_pack.domain_knowledge_brief.brief_title
    assert "Order Flow Imbalance" not in bio_pack.quant_strategy_spec.strategy_name
    assert "k_{\\text{cleave}}" in bio_pack.quant_strategy_spec.mathematical_signal_latex
    assert any("Cas9" in r or "R-loop" in r for r in bio_pack.quant_strategy_spec.entry_exit_rules)
    assert any("Mg2+" in c for c in bio_pack.quant_strategy_spec.market_specific_caveats)
    assert any("off-target" in f for f in bio_pack.quant_strategy_spec.known_failure_regimes)

    # 4. Scaffold, distill, and query a custom `bioinformatics_code_scout` code domain
    (bio_code_dir / "alignment.py").write_text(
        "class SmithWatermanScorer:\n"
        '    """Local biological sequence alignment with affine gap penalties."""\n\n'
        "    def score_alignment(self, seq_a: str, seq_b: str, gap_open: float = -10.0) -> float:\n"
        "        matches = sum(1 for a, b in zip(seq_a, seq_b, strict=False) if a == b)\n"
        "        return float(matches * 2.0 + gap_open)\n",
        encoding="utf-8",
    )
    cmd_init_domain(
        domain_id="bioinformatics_code_scout",
        kind="code",
        display_name="Bioinformatics Sequence Alignment Code Scout",
        domains_dir=domains_dir,
    )
    cmd_distill_custom_domain(
        domain_id="bioinformatics_code_scout",
        source_dir=bio_code_dir,
        num_samples=8,
        domains_dir=domains_dir,
        data_dir=data_dir,
    )
    code_pack = wf.run(
        user_prompt=(
            "How do we score local sequence alignment affine gap penalties in SmithWatermanScorer?"
        ),
        principal=principal,
        domain_id="bioinformatics_code_scout",
    )
    assert isinstance(code_pack, ImplementationContextPack)
    assert code_pack.preflight_report.passed is True
    assert len(code_pack.code_slices) >= 1
    assert "SmithWatermanScorer" in code_pack.code_slices[0].qualified_symbol
    assert "score_alignment" in code_pack.code_slices[0].verbatim_code_with_lines


def test_conversational_and_unrelated_queries_rejected_across_all_domains() -> None:
    """Guard 20: Chitchat ('how you doing') and unrelated queries return InsufficientContext."""
    from context_scout.agents.scout_workflow import FederatedScoutCoordinator, ScoutWorkflow
    from context_scout.api.dependencies import (
        DEFAULT_ROLE_SCOPES,
        get_domain_registry,
        get_repository,
    )
    from context_scout.schemas.contracts import InsufficientContextDiagnostic

    repo = get_repository()
    registry = get_domain_registry()
    wf = ScoutWorkflow(registry=registry, repo=repo)
    fed = FederatedScoutCoordinator(registry=registry, repo=repo)
    principal = PrincipalContext(
        user_id="demo_engineer",
        role=RoleType.ENGINEER,
        allowed_scopes=DEFAULT_ROLE_SCOPES[RoleType.ENGINEER],
    )

    for chitchat in (
        "how you doing",
        "hello how are you today?",
        "what is the weather in berlin?",
        "how do we repair a diesel tractor?",
    ):
        for dom in ("trading_code_scout", "quant_research_scout"):
            res = wf.run(user_prompt=chitchat, principal=principal, domain_id=dom)
            assert isinstance(res, InsufficientContextDiagnostic), (
                f"Expected InsufficientContextDiagnostic for '{chitchat}' on '{dom}'"
            )
            assert res.paid_tokens_spent == 0

        fed_res = fed.run(user_prompt=chitchat, principal=principal)
        assert isinstance(fed_res, InsufficientContextDiagnostic)
        assert fed_res.paid_tokens_spent == 0

    console_tsx = Path("web-ui/src/components/ScoutConsole.tsx").read_text(encoding="utf-8")
    assert 'data-testid="insufficient-context-banner"' in console_tsx
    assert "onClearContextPack?.()" in console_tsx


def test_public_repository_hygiene_and_frontend_hook_deps_guard() -> None:
    """Guard 21: Public repo untracks internal agent instructions and ScoutConsole hook deps."""
    import subprocess

    gitignore = Path(".gitignore").read_text(encoding="utf-8")
    internal_docs = (
        "AGENTS.md",
        "CLAUDE.md",
        "GEMINI.md",
        "STATUS.md",
        "ARCHITECTURE_AUDIT_REPORT.md",
    )
    for doc in internal_docs:
        assert doc in gitignore, f"{doc} must be gitignored in public repository"

    tracked_out = subprocess.check_output(["git", "ls-files"], text=True)
    tracked_files = set(tracked_out.splitlines())
    for doc in internal_docs:
        assert doc not in tracked_files, f"{doc} must not be tracked in git ls-files"

    readme_text = Path("README.md").read_text(encoding="utf-8")
    for milestone_idx in range(1, 9):
        assert f"Milestone {milestone_idx} Architecture" not in readme_text
    assert (
        "## Results (Unseen-Repo Two-Arm A/B Benchmark & Security Gates; see limitations)"
        in readme_text
    )
    assert "## Limitations" in readme_text
    assert "30 of 30" in readme_text
    assert "0 of 22" in readme_text
    assert "22 of 22" in readme_text
    assert "Gemini 3.8 Flash" in readme_text
    assert "Gemini 4 Argon" in readme_text
    assert "Claude Sonnet 5.5" in readme_text
    assert "Claude Opus 5.5" in readme_text
    assert "GPT-6 Astra" in readme_text

    console_tsx = Path("web-ui/src/components/ScoutConsole.tsx").read_text(encoding="utf-8")
    assert "setQuery((prevQuery) =>" in console_tsx


def test_readme_schema_and_contract_alignment_guard() -> None:
    """Guard 22: README matches actual SQL DDL, module paths, and strips orphan audit codes."""
    readme_text = Path("README.md").read_text(encoding="utf-8")

    # Actual PostgreSQL 17 tables and runtime role from sql/001_init_schema.sql
    for actual_identifier in (
        "domain_packs",
        "code_files",
        "ast_symbols",
        "ast_edges",
        "doc_chunks",
        "execution_telemetry",
        "scout_app_user",
    ):
        assert actual_identifier in readme_text, (
            f"Missing actual SQL identifier: {actual_identifier}"
        )

    # Hallucinated or stale SQL identifiers, file paths, and pricing strings must never recur
    banned_stale_tokens = (
        "git_blobs",
        "document_chunks",
        "context_packs",
        "feedback_events",
        "context_scout_app",
        "app_has_scope",
        "discovery_tokens_saved",
        "db/connection.py",
        "ast_engine/parser.py",
        "ast_engine/expander.py",
        "ingestion/merkle.py",
        "$0.75/$3.75",
        "BLK-01",
        "SEC-01",
        "SEC-02",
        "SEC-03",
        "SEC-04",
        "VAL-01",
        "VAL-02",
        "FIN-01",
        "HW-01",
        "ARC-01",
        "ARC-02",
        "ML-01",
        "OPS-01",
    )
    for token in banned_stale_tokens:
        assert token not in readme_text, (
            f"Stale or internal audit token '{token}' found in README.md"
        )

    # Every relative Markdown link in README.md must exist on disk, and all report
    # links in README.md must point to human-readable .md files (never .json or .jsonl)
    for _label, target in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", readme_text):
        if target.startswith(("http://", "https://", "#")):
            continue
        clean_target = target.split("#", 1)[0]
        assert Path(clean_target).exists(), (
            f"Broken relative link in README.md: {clean_target}"
        )
        if clean_target.startswith("reports/"):
            assert clean_target.endswith(".md"), (
                f"README.md report link must use human-readable .md format, got: {clean_target}"
            )

    # Verify README.md numbers stay synchronized with reports/polyglot_unseen_v2_benchmark.md,
    # reports/senior_eng_30_e2e_ab_report.md, and reports/unseen_repo_ab_report.md
    for expected_metric in (
        "59 of 60",
        "98.33%",
        "371 of 371",
        "32,486",
        "1,923",
        "1,926,044",
        "137,162",
        "1,949",
        "17,473",
        "91.35%",
        "60,471",
        "887,092",
        "93.18%",
        "reports/senior_eng_30_e2e_ab_report.md",
        "`27 of 30` (`90.00%`)",
        "`6 of 30` (`20.00%`)",
        "`29 of 30` (`96.67%`)",
        "## Iterative Engineering Progression: From Baseline to Production",
        "DomainKnowledgeBrief",
        "doc_corpus_mcp.py",
    ):
        assert expected_metric in readme_text, (
            f"README.md out of sync with benchmark reports; missing: {expected_metric}"
        )


def test_cross_repo_isolation_2hop_schema_closure_and_contiguous_slice_coalescing() -> None:
    """Guard 22: Verifies cross-repo isolation, 2-hop schema closure, and slice coalescing."""
    from uuid import uuid4

    from context_scout.agents.preflight_gate import BoundGraphRepository, PreFlightValidator
    from context_scout.agents.scout_workflow import ScoutWorkflow
    from context_scout.ast_engine.dependency_expander import ASTDependencyExpander
    from context_scout.db.repositories import ASTEdgeRecord, ASTSymbolRecord, CodeFileRecord
    from context_scout.schemas.contracts import ModificationIntent, VerbatimCodeSlice

    class _MockMultiRepo:
        def __init__(
            self,
            files: list[CodeFileRecord],
            symbols: list[ASTSymbolRecord],
            edges: list[ASTEdgeRecord],
        ) -> None:
            self._files = files
            self._symbols = symbols
            self._edges = edges

        def list_visible_code_files(self, *args: object, **kwargs: object) -> list[CodeFileRecord]:
            return list(self._files)

        def list_visible_ast_symbols(
            self, *args: object, **kwargs: object
        ) -> list[ASTSymbolRecord]:
            return list(self._symbols)

        def list_visible_ast_edges(self, *args: object, **kwargs: object) -> list[ASTEdgeRecord]:
            return list(self._edges)

        def list_visible_doc_chunks(self, *args: object, **kwargs: object) -> list[object]:
            return []

    # 1. File records across 3 repositories (`anyio`, `rich`, `enterprise_gitlab_ops`)
    f_anyio_id = uuid4()
    f_rich_id = uuid4()
    f_sql_id = uuid4()
    f_tsx_id = uuid4()

    anyio_code = "\n".join(
        [
            "from __future__ import annotations",
            "import asyncio",
            "from collections.abc import Callable",
            *[f"# header padding line {i}" for i in range(4, 25)],
            "def find_root_task() -> asyncio.Task:",
            "    root: asyncio.Task | None = None",
            "    for task in asyncio.all_tasks():",
            "        state = _task_states.get(task)",
            "        if state is not None and state.cancel_scope is not None:",
            "            return task",
            "    raise RuntimeError('Failed to look up root task')",
            *[f"# middle padding line {i}" for i in range(32, 50)],
            "class CancelScope:",
            "    def __init__(self, deadline: float = 0.0) -> None:",
            "        self._deadline = deadline",
            "class TaskState:",
            "    def __init__(self, parent_id: int | None, cancel_scope: Any = None) -> None:",
            "        self.parent_id = parent_id",
            "        self.cancel_scope = cancel_scope",
        ]
    )
    rich_code = (
        "class Task:\n"
        "    def __init__(self, description: str) -> None:\n"
        "        self.description = description\n"
    )
    sql_code = (
        "CREATE TABLE IF NOT EXISTS tenant_accounts (\n"
        "    tenant_id UUID PRIMARY KEY,\n"
        "    org_slug TEXT NOT NULL UNIQUE,\n"
        "    tier TEXT NOT NULL DEFAULT 'enterprise'\n"
        ");\n\n"
        "CREATE TABLE IF NOT EXISTS execution_orders (\n"
        "    order_id UUID PRIMARY KEY,\n"
        "    tenant_id UUID NOT NULL REFERENCES tenant_accounts(tenant_id),\n"
        "    symbol TEXT NOT NULL,\n"
        "    notional_usd NUMERIC(18, 2) NOT NULL\n"
        ");\n\n"
        "ALTER TABLE execution_orders ENABLE ROW LEVEL SECURITY;\n\n"
        "CREATE POLICY tenant_isolation_policy ON execution_orders\n"
        "    USING (tenant_id::text = current_setting('app.current_tenant_id', true));\n"
    )
    tsx_code = (
        "export interface AuditRowProps {\n"
        "  readonly traceId: string;\n"
        "  readonly status: 'PASS' | 'BLOCKED' | 'RETRY';\n"
        "  readonly latencyMs: number;\n"
        "}\n\n"
        "export function renderAuditBadge(row: AuditRowProps): string {\n"
        "  const prefix = row.status === 'PASS' ? '[OK]' : '[WARN]';\n"
        "  return `${prefix} ${row.traceId} (${row.latencyMs}ms)`;\n"
        "}\n\n"
        "export function formatAuditSummary(rows: readonly AuditRowProps[]): string {\n"
        "  return rows.map((r) => renderAuditBadge(r)).join('\\n');\n"
        "}\n"
    )

    files = [
        CodeFileRecord(
            file_id=f_anyio_id,
            domain_id="polyglot_test",
            repo_name="anyio",
            file_path="anyio/src/anyio/_backends/_asyncio.py",
            git_commit_sha="a" * 40,
            git_blob_sha="b" * 40,
            language="python",
            allowed_scopes=["public_eng", "repo:anyio"],
            raw_content=anyio_code,
            line_count=len(anyio_code.splitlines()),
        ),
        CodeFileRecord(
            file_id=f_rich_id,
            domain_id="polyglot_test",
            repo_name="rich",
            file_path="rich/rich/progress.py",
            git_commit_sha="c" * 40,
            git_blob_sha="d" * 40,
            language="python",
            allowed_scopes=["public_eng", "repo:rich"],
            raw_content=rich_code,
            line_count=len(rich_code.splitlines()),
        ),
        CodeFileRecord(
            file_id=f_sql_id,
            domain_id="polyglot_test",
            repo_name="enterprise_gitlab_ops",
            file_path="enterprise_gitlab_ops/migrations/ledger_rls.sql",
            git_commit_sha="e" * 40,
            git_blob_sha="f" * 40,
            language="sql",
            allowed_scopes=["public_eng", "repo:enterprise_gitlab_ops"],
            raw_content=sql_code,
            line_count=len(sql_code.splitlines()),
        ),
        CodeFileRecord(
            file_id=f_tsx_id,
            domain_id="polyglot_test",
            repo_name="enterprise_gitlab_ops",
            file_path="enterprise_gitlab_ops/web/AuditPanel.tsx",
            git_commit_sha="1" * 40,
            git_blob_sha="2" * 40,
            language="tsx",
            allowed_scopes=["public_eng", "repo:enterprise_gitlab_ops"],
            raw_content=tsx_code,
            line_count=len(tsx_code.splitlines()),
        ),
    ]

    sym_find_root = ASTSymbolRecord(
        symbol_id=uuid4(),
        file_id=f_anyio_id,
        domain_id="polyglot_test",
        qualified_name="anyio._backends._asyncio.find_root_task",
        symbol_kind="function",
        start_line=25,
        end_line=31,
        signature_skeleton="def find_root_task() -> asyncio.Task:",
        contextual_summary="find_root_task",
        allowed_scopes=["public_eng", "repo:anyio"],
        embedding=[],
    )
    sym_cancel_scope = ASTSymbolRecord(
        symbol_id=uuid4(),
        file_id=f_anyio_id,
        domain_id="polyglot_test",
        qualified_name="anyio._backends._asyncio.CancelScope",
        symbol_kind="class",
        start_line=50,
        end_line=52,
        signature_skeleton="class CancelScope:",
        contextual_summary="CancelScope",
        allowed_scopes=["public_eng", "repo:anyio"],
        embedding=[],
    )
    sym_task_state = ASTSymbolRecord(
        symbol_id=uuid4(),
        file_id=f_anyio_id,
        domain_id="polyglot_test",
        qualified_name="anyio._backends._asyncio.TaskState",
        symbol_kind="class",
        start_line=53,
        end_line=56,
        signature_skeleton="class TaskState:",
        contextual_summary="TaskState",
        allowed_scopes=["public_eng", "repo:anyio"],
        embedding=[],
    )
    sym_rich_task = ASTSymbolRecord(
        symbol_id=uuid4(),
        file_id=f_rich_id,
        domain_id="polyglot_test",
        qualified_name="rich.progress.Task",
        symbol_kind="class",
        start_line=1,
        end_line=3,
        signature_skeleton="class Task:",
        contextual_summary="Task",
        allowed_scopes=["public_eng", "repo:rich"],
        embedding=[],
    )
    sym_tenant_tbl = ASTSymbolRecord(
        symbol_id=uuid4(),
        file_id=f_sql_id,
        domain_id="polyglot_test",
        qualified_name="ledger_rls.tenant_accounts",
        symbol_kind="class",
        start_line=1,
        end_line=5,
        signature_skeleton="CREATE TABLE IF NOT EXISTS tenant_accounts",
        contextual_summary="tenant_accounts",
        allowed_scopes=["public_eng", "repo:enterprise_gitlab_ops"],
        embedding=[],
    )
    sym_orders_tbl = ASTSymbolRecord(
        symbol_id=uuid4(),
        file_id=f_sql_id,
        domain_id="polyglot_test",
        qualified_name="ledger_rls.execution_orders",
        symbol_kind="class",
        start_line=7,
        end_line=12,
        signature_skeleton="CREATE TABLE IF NOT EXISTS execution_orders",
        contextual_summary="execution_orders",
        allowed_scopes=["public_eng", "repo:enterprise_gitlab_ops"],
        embedding=[],
    )
    sym_rls_policy = ASTSymbolRecord(
        symbol_id=uuid4(),
        file_id=f_sql_id,
        domain_id="polyglot_test",
        qualified_name="ledger_rls.tenant_isolation_policy",
        symbol_kind="function",
        start_line=16,
        end_line=17,
        signature_skeleton="CREATE POLICY tenant_isolation_policy ON execution_orders",
        contextual_summary="tenant_isolation_policy",
        allowed_scopes=["public_eng", "repo:enterprise_gitlab_ops"],
        embedding=[],
    )

    symbols = [
        sym_find_root,
        sym_cancel_scope,
        sym_task_state,
        sym_rich_task,
        sym_tenant_tbl,
        sym_orders_tbl,
        sym_rls_policy,
    ]
    edges = [
        # Cross-repo short-name collision edge (`asyncio.Task` -> `Task`)
        ASTEdgeRecord(
            edge_id=uuid4(),
            source_symbol_id=sym_find_root.symbol_id,
            target_symbol_id=None,
            target_qualified_name="asyncio.Task",
            edge_kind="USES_TYPE",
            domain_id="polyglot_test",
            allowed_scopes=["public_eng", "repo:anyio"],
        ),
        # 2-hop SQL chain: policy -> execution_orders -> tenant_accounts
        ASTEdgeRecord(
            edge_id=uuid4(),
            source_symbol_id=sym_rls_policy.symbol_id,
            target_symbol_id=sym_orders_tbl.symbol_id,
            target_qualified_name="execution_orders",
            edge_kind="USES_TYPE",
            domain_id="polyglot_test",
            allowed_scopes=["public_eng", "repo:enterprise_gitlab_ops"],
        ),
        ASTEdgeRecord(
            edge_id=uuid4(),
            source_symbol_id=sym_orders_tbl.symbol_id,
            target_symbol_id=sym_tenant_tbl.symbol_id,
            target_qualified_name="tenant_accounts",
            edge_kind="USES_TYPE",
            domain_id="polyglot_test",
            allowed_scopes=["public_eng", "repo:enterprise_gitlab_ops"],
        ),
    ]

    mock_repo = _MockMultiRepo(files=files, symbols=symbols, edges=edges)
    bound_repo = BoundGraphRepository(mock_repo)
    expander = ASTDependencyExpander(max_direct_callers=3, max_expansion_token_budget=1500)
    principal = PrincipalContext(
        user_id="test_eng",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng", "repo:anyio", "repo:rich", "repo:enterprise_gitlab_ops"],
    )

    # Check A: `find_root_task` expands same-file `CancelScope` and `TaskState` via body-referenced
    # snake_case state inference, and NEVER attaches `rich.progress.Task` from `rich`.
    exp_anyio = expander.expand_1hop(
        focus_symbol=sym_find_root,
        repo=bound_repo,
        principal=principal,
        domain_id="polyglot_test",
    )
    anyio_types = {p.qualified_symbol for p in exp_anyio.imported_type_slices}
    assert "anyio._backends._asyncio.CancelScope" in anyio_types
    assert "anyio._backends._asyncio.TaskState" in anyio_types
    assert "rich.progress.Task" not in anyio_types

    # Check B: `tenant_isolation_policy` expands both 1-hop and 2-hop tables
    exp_sql = expander.expand_1hop(
        focus_symbol=sym_rls_policy,
        repo=bound_repo,
        principal=principal,
        domain_id="polyglot_test",
    )
    sql_types = {p.qualified_symbol for p in exp_sql.imported_type_slices}
    assert "ledger_rls.execution_orders" in sql_types
    assert "ledger_rls.tenant_accounts" in sql_types

    # Check C: `_coalesce_and_enrich_target_file_slices` merges fragmented slices in compact files
    # and attaches top-of-file imports for deep target functions
    from context_scout.agents.domain_registry import DomainRegistry

    wf = ScoutWorkflow(registry=DomainRegistry(Path("domains")), repo=mock_repo)
    fragmented_tsx = [
        VerbatimCodeSlice(
            repo_name="enterprise_gitlab_ops",
            file_path="enterprise_gitlab_ops/web/AuditPanel.tsx",
            git_commit_sha="1" * 40,
            git_blob_sha="2" * 40,
            start_line=7,
            end_line=10,
            qualified_symbol="AuditPanel.renderAuditBadge",
            modification_intent=ModificationIntent.MODIFY_TARGET,
            auto_expanded_by_ast=False,
            selection_reason="target",
            verbatim_code_with_lines=(
                "   7 | export function renderAuditBadge(row: AuditRowProps): string {"
            ),
        ),
        VerbatimCodeSlice(
            repo_name="enterprise_gitlab_ops",
            file_path="enterprise_gitlab_ops/web/AuditPanel.tsx",
            git_commit_sha="1" * 40,
            git_blob_sha="2" * 40,
            start_line=1,
            end_line=5,
            qualified_symbol="AuditPanel.AuditRowProps",
            modification_intent=ModificationIntent.IMPORTED_TYPE,
            auto_expanded_by_ast=True,
            selection_reason="type",
            verbatim_code_with_lines="   1 | export interface AuditRowProps {",
        ),
    ]
    files_by_path = {f.file_path: f for f in files}
    coalesced_tsx = wf._coalesce_and_enrich_target_file_slices(
        hydrated_slices=fragmented_tsx,
        files_by_path=files_by_path,
        user_prompt=(
            "In repository `enterprise_gitlab_ops`, update `renderAuditBadge` in `AuditPanel.tsx`"
        ),
    )
    assert coalesced_tsx[0].start_line == 1
    assert coalesced_tsx[0].end_line == 14
    assert "export function formatAuditSummary" in coalesced_tsx[0].verbatim_code_with_lines

    deep_anyio = [
        VerbatimCodeSlice(
            repo_name="anyio",
            file_path="anyio/src/anyio/_backends/_asyncio.py",
            git_commit_sha="a" * 40,
            git_blob_sha="b" * 40,
            start_line=25,
            end_line=31,
            qualified_symbol="anyio._backends._asyncio.find_root_task",
            modification_intent=ModificationIntent.MODIFY_TARGET,
            auto_expanded_by_ast=False,
            selection_reason="target",
            verbatim_code_with_lines="  25 | def find_root_task() -> asyncio.Task:",
        )
    ]
    enriched_anyio = wf._coalesce_and_enrich_target_file_slices(
        hydrated_slices=deep_anyio,
        files_by_path=files_by_path,
        user_prompt="In repository `anyio`, update `find_root_task`",
    )
    assert any(
        "Top-of-file import header" in s.selection_reason and s.start_line == 1
        for s in enriched_anyio
    )
    validator = PreFlightValidator(repo=mock_repo)
    missing_refs, _ = validator.partition_prompt_identifiers(
        user_prompt=(
            "In repository `enterprise_gitlab_ops`, check `tenant_isolation_policy` "
            "with `app.current_tenant_id`"
        ),
        visible_symbols=symbols,
        visible_files=files,
        code_slices=coalesced_tsx,
    )
    assert "app" not in missing_refs


def test_enterprise_production_class_state_cross_file_dto_and_caller_anti_starvation() -> None:
    """Guard 23: Verifies class __init__ context, cross-file DTO inheritance, and caller budget."""
    from uuid import uuid4

    from context_scout.agents.domain_registry import DomainRegistry
    from context_scout.agents.preflight_gate import BoundGraphRepository
    from context_scout.agents.scout_workflow import ScoutWorkflow
    from context_scout.ast_engine.dependency_expander import ASTDependencyExpander
    from context_scout.db.repositories import ASTEdgeRecord, ASTSymbolRecord, CodeFileRecord
    from context_scout.schemas.contracts import ModificationIntent, VerbatimCodeSlice

    f_svc_id = uuid4()
    f_dto_id = uuid4()
    f_base_dto_id = uuid4()
    f_caller_id = uuid4()
    f_other_test_id = uuid4()

    svc_code = "\n".join(
        [
            "from __future__ import annotations",
            "import asyncio",
            "DEFAULT_MAX_RETRIES = 5",
            *[f"# module padding {i}" for i in range(4, 16)],
            "class OrderExecutionService:",
            "    def __init__(self, db_pool: object, redis_lock: object) -> None:",
            "        self._db_pool = db_pool",
            "        self._redis_lock = redis_lock",
            "        self._OrderCache: dict[str, object] = {}",
            *[f"    # class padding {i}" for i in range(22, 42)],
            "    def submit_order(self, req: CreateOrderDTO) -> str:",
            "        retries = DEFAULT_MAX_RETRIES",
            "        return _normalize_order_id(str(retries))",
            "def _normalize_order_id(raw_id: str) -> str:",
            "    return raw_id.strip().upper()",
        ]
    )
    dto_code = (
        "class CreateOrderDTO(BaseAuthenticatedDTO):\n"
        "    symbol: str\n"
        "    qty: int\n"
    )
    base_dto_code = (
        "class BaseAuthenticatedDTO:\n"
        "    tenant_id: str\n"
        "    idempotency_key: str\n"
    )
    caller_code = (
        "def handle_http_post(svc: OrderExecutionService, payload: CreateOrderDTO) -> str:\n"
        "    return svc.submit_order(payload)\n"
    )
    other_repo_test_code = (
        "def test_submit_order() -> None:\n"
        "    assert True\n"
    )

    files = [
        CodeFileRecord(
            file_id=f_svc_id,
            domain_id="prod_test",
            repo_name="orders_service",
            file_path="orders_service/src/order_service.py",
            git_commit_sha="a" * 40,
            git_blob_sha="b" * 40,
            language="python",
            allowed_scopes=["public_eng", "repo:orders_service"],
            raw_content=svc_code,
            line_count=len(svc_code.splitlines()),
        ),
        CodeFileRecord(
            file_id=f_dto_id,
            domain_id="prod_test",
            repo_name="orders_service",
            file_path="orders_service/src/dtos/order_dto.py",
            git_commit_sha="c" * 40,
            git_blob_sha="d" * 40,
            language="python",
            allowed_scopes=["public_eng", "repo:orders_service"],
            raw_content=dto_code,
            line_count=len(dto_code.splitlines()),
        ),
        CodeFileRecord(
            file_id=f_base_dto_id,
            domain_id="prod_test",
            repo_name="orders_service",
            file_path="orders_service/src/dtos/base_dto.py",
            git_commit_sha="e" * 40,
            git_blob_sha="f" * 40,
            language="python",
            allowed_scopes=["public_eng", "repo:orders_service"],
            raw_content=base_dto_code,
            line_count=len(base_dto_code.splitlines()),
        ),
        CodeFileRecord(
            file_id=f_caller_id,
            domain_id="prod_test",
            repo_name="orders_service",
            file_path="orders_service/src/api/routes.py",
            git_commit_sha="1" * 40,
            git_blob_sha="2" * 40,
            language="python",
            allowed_scopes=["public_eng", "repo:orders_service"],
            raw_content=caller_code,
            line_count=len(caller_code.splitlines()),
        ),
        CodeFileRecord(
            file_id=f_other_test_id,
            domain_id="prod_test",
            repo_name="unrelated_repo",
            file_path="unrelated_repo/tests/test_order_service.py",
            git_commit_sha="3" * 40,
            git_blob_sha="4" * 40,
            language="python",
            allowed_scopes=["public_eng", "repo:unrelated_repo"],
            raw_content=other_repo_test_code,
            line_count=len(other_repo_test_code.splitlines()),
        ),
    ]

    sym_submit = ASTSymbolRecord(
        symbol_id=uuid4(),
        file_id=f_svc_id,
        domain_id="prod_test",
        qualified_name="order_service.OrderExecutionService.submit_order",
        symbol_kind="method",
        start_line=42,
        end_line=44,
        signature_skeleton="def submit_order(self, req: CreateOrderDTO) -> str:",
        contextual_summary="submit_order",
        allowed_scopes=["public_eng", "repo:orders_service"],
        embedding=[],
    )
    sym_helper = ASTSymbolRecord(
        symbol_id=uuid4(),
        file_id=f_svc_id,
        domain_id="prod_test",
        qualified_name="order_service._normalize_order_id",
        symbol_kind="function",
        start_line=45,
        end_line=46,
        signature_skeleton="def _normalize_order_id(raw_id: str) -> str:",
        contextual_summary="_normalize_order_id",
        allowed_scopes=["public_eng", "repo:orders_service"],
        embedding=[],
    )
    sym_dto = ASTSymbolRecord(
        symbol_id=uuid4(),
        file_id=f_dto_id,
        domain_id="prod_test",
        qualified_name="dtos.order_dto.CreateOrderDTO",
        symbol_kind="class",
        start_line=1,
        end_line=3,
        signature_skeleton="class CreateOrderDTO(BaseAuthenticatedDTO):",
        contextual_summary="CreateOrderDTO",
        allowed_scopes=["public_eng", "repo:orders_service"],
        embedding=[],
    )
    sym_base_dto = ASTSymbolRecord(
        symbol_id=uuid4(),
        file_id=f_base_dto_id,
        domain_id="prod_test",
        qualified_name="dtos.base_dto.BaseAuthenticatedDTO",
        symbol_kind="class",
        start_line=1,
        end_line=3,
        signature_skeleton="class BaseAuthenticatedDTO:",
        contextual_summary="BaseAuthenticatedDTO",
        allowed_scopes=["public_eng", "repo:orders_service"],
        embedding=[],
    )
    sym_caller = ASTSymbolRecord(
        symbol_id=uuid4(),
        file_id=f_caller_id,
        domain_id="prod_test",
        qualified_name="api.routes.handle_http_post",
        symbol_kind="function",
        start_line=1,
        end_line=2,
        signature_skeleton="def handle_http_post(svc, payload) -> str:",
        contextual_summary="handle_http_post",
        allowed_scopes=["public_eng", "repo:orders_service"],
        embedding=[],
    )
    sym_other_test = ASTSymbolRecord(
        symbol_id=uuid4(),
        file_id=f_other_test_id,
        domain_id="prod_test",
        qualified_name="tests.test_order_service.test_submit_order",
        symbol_kind="function",
        start_line=1,
        end_line=2,
        signature_skeleton="def test_submit_order() -> None:",
        contextual_summary="test_submit_order",
        allowed_scopes=["public_eng", "repo:unrelated_repo"],
        embedding=[],
    )

    symbols = [sym_submit, sym_helper, sym_dto, sym_base_dto, sym_caller, sym_other_test]
    edges = [
        ASTEdgeRecord(
            edge_id=uuid4(),
            source_symbol_id=sym_submit.symbol_id,
            target_symbol_id=sym_dto.symbol_id,
            target_qualified_name="CreateOrderDTO",
            edge_kind="USES_TYPE",
            domain_id="prod_test",
            allowed_scopes=["public_eng", "repo:orders_service"],
        ),
        ASTEdgeRecord(
            edge_id=uuid4(),
            source_symbol_id=sym_dto.symbol_id,
            target_symbol_id=sym_base_dto.symbol_id,
            target_qualified_name="BaseAuthenticatedDTO",
            edge_kind="INHERITS",
            domain_id="prod_test",
            allowed_scopes=["public_eng", "repo:orders_service"],
        ),
        ASTEdgeRecord(
            edge_id=uuid4(),
            source_symbol_id=sym_caller.symbol_id,
            target_symbol_id=sym_submit.symbol_id,
            target_qualified_name="submit_order",
            edge_kind="CALLS",
            domain_id="prod_test",
            allowed_scopes=["public_eng", "repo:orders_service"],
        ),
    ]

    class _MockRepo:
        def list_visible_code_files(self, *args: object, **kwargs: object) -> list[CodeFileRecord]:
            return files

        def list_visible_ast_symbols(
            self, *args: object, **kwargs: object
        ) -> list[ASTSymbolRecord]:
            return symbols

        def list_visible_ast_edges(self, *args: object, **kwargs: object) -> list[ASTEdgeRecord]:
            return edges

        def list_visible_doc_chunks(self, *args: object, **kwargs: object) -> list[object]:
            return []

    bound = BoundGraphRepository(_MockRepo())
    # Tight budget (80 tokens) to verify primary caller is NOT starved by secondary types
    expander = ASTDependencyExpander(max_direct_callers=3, max_expansion_token_budget=80)
    principal = PrincipalContext(
        user_id="prod_eng",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng", "repo:orders_service", "repo:unrelated_repo"],
    )
    exp = expander.expand_1hop(
        focus_symbol=sym_submit,
        repo=bound,
        principal=principal,
        domain_id="prod_test",
    )
    expanded_type_names = {p.qualified_symbol for p in exp.imported_type_slices}
    # 1. Cross-file 2-hop BaseAuthenticatedDTO & same-file helper _normalize_order_id expanded
    expander_full = ASTDependencyExpander(max_direct_callers=3, max_expansion_token_budget=1500)
    exp_full = expander_full.expand_1hop(
        focus_symbol=sym_submit,
        repo=bound,
        principal=principal,
        domain_id="prod_test",
    )
    full_types = {p.qualified_symbol for p in exp_full.imported_type_slices}
    assert "dtos.order_dto.CreateOrderDTO" in full_types
    assert "dtos.base_dto.BaseAuthenticatedDTO" in full_types
    assert "order_service._normalize_order_id" in full_types
    # 2. Anti-starvation: even under tight 80-token budget, caller_slices is non-empty
    assert len(exp.caller_slices) == 1
    assert exp.caller_slices[0].qualified_symbol == "api.routes.handle_http_post"
    assert len(expanded_type_names) >= 1
    # 3. Cross-repo test fixture isolation: unrelated_repo test_order_service is NOT attached
    assert len(exp_full.test_fixture_slices) == 0

    # 4. Enclosing class header & __init__ state + top-of-file constant DEFAULT_MAX_RETRIES attached
    wf = ScoutWorkflow(registry=DomainRegistry(Path("domains")), repo=_MockRepo())
    method_slice = [
        VerbatimCodeSlice(
            repo_name="orders_service",
            file_path="orders_service/src/order_service.py",
            git_commit_sha="a" * 40,
            git_blob_sha="b" * 40,
            start_line=42,
            end_line=44,
            qualified_symbol="order_service.OrderExecutionService.submit_order",
            modification_intent=ModificationIntent.MODIFY_TARGET,
            auto_expanded_by_ast=False,
            selection_reason="target",
            verbatim_code_with_lines=(
                "  42 |     def submit_order(self, req: CreateOrderDTO) -> str:\n"
                "  43 |         retries = DEFAULT_MAX_RETRIES\n"
                "  44 |         return _normalize_order_id(str(retries))"
            ),
        )
    ]
    enriched = wf._coalesce_and_enrich_target_file_slices(
        hydrated_slices=method_slice,
        files_by_path={f.file_path: f for f in files},
        user_prompt="In repository `orders_service`, update `submit_order`",
    )
    assert any(
        "Top-of-file import header" in s.selection_reason
        and "DEFAULT_MAX_RETRIES = 5" in s.verbatim_code_with_lines
        for s in enriched
    )
    assert any(
        "Enclosing class & constructor state" in s.selection_reason
        and "self._redis_lock = redis_lock" in s.verbatim_code_with_lines
        for s in enriched
    )


def test_multi_domain_knowledge_pdf_sync_mcp_benchmark_and_handoff_guard(
    tmp_path: Path,
) -> None:
    """Guard 23: Verifies multi-domain knowledge decoupling, doc sync, MCP, benchmark & handoff."""
    import asyncio
    import json

    from context_scout.agents.domain_registry import DomainRegistry
    from context_scout.agents.gemini_executor import GeminiImplementationExecutor
    from context_scout.cli import (
        cmd_benchmark_domain,
        cmd_distill_custom_domain,
        cmd_init_domain,
        cmd_query_domain,
        cmd_sync_domain,
    )
    from context_scout.mcp_servers.doc_corpus_mcp import create_doc_corpus_mcp_server
    from context_scout.schemas.contracts import ImplementationContextPack
    from context_scout.security.rbac import bind_session_principal

    domains_dir = tmp_path / "domains"
    data_dir = tmp_path / "data"
    med_docs_dir = tmp_path / "clinical_oncology_docs"
    med_docs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Multi-format document corpus (.md + .json + .html) containing words like
    # "optimization", "variance", and "agent" that previously triggered finance templates
    (med_docs_dir / "car_t_dosing_protocol.md").write_text(
        "# CAR-T Immunotherapy Cytokine Release Syndrome Protocol\n\n"
        "## 1. Tocilizumab Agent Dose Optimization and Variance Control\n\n"
        "Clinical therapeutic agent optimization requires intravenous tocilizumab at 8 mg/kg "
        "when Grade 2 cytokine release syndrome fever exceeds 38.5C.\n"
        "Pharmacokinetic clearance follows $C(t) = C_0 \\cdot \\exp(-k_{el} \\cdot t)$ "
        "across adult lymphoma cohorts.\n"
        "Clinicians must monitor serum ferritin variance and C-reactive protein every 12 hours "
        "to constrain neurotoxicity risk.\n"
        "Severe immune effector cell-associated neurotoxicity syndrome failure occurs when "
        "blood-brain barrier endothelial activation goes untreated.\n",
        encoding="utf-8",
    )
    (med_docs_dir / "trial_endpoints.json").write_text(
        json.dumps(
            {
                "protocol_id": "ONCO-2026-09",
                "primary_endpoint": (
                    "Complete remission rate at day 28 post CAR-T infusion under "
                    "tocilizumab prophylaxis."
                ),
                "safety_constraint": (
                    "Hold lymphodepleting fludarabine if absolute neutrophil count "
                    "drops below 500 cells/uL."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # 2. Scaffold `kind="knowledge"` domain -> verify DomainKnowledgeBrief contract & no cost floor
    init_res = cmd_init_domain(
        domain_id="clinical_oncology_scout",
        kind="knowledge",
        display_name="Clinical Oncology & Immunotherapy Knowledge Scout",
        domains_dir=domains_dir,
    )
    assert Path(str(init_res["manifest_path"])).exists()
    registry = DomainRegistry(domains_dir=domains_dir)
    manifest = registry.get("clinical_oncology_scout")
    assert manifest.output_contract == "DomainKnowledgeBrief"
    assert manifest.authority_rules.require_transaction_cost_check is False

    # 3. Distill custom knowledge domain -> verify zero fake finance metrics in train.jsonl
    distill_res = cmd_distill_custom_domain(
        domain_id="clinical_oncology_scout",
        source_dir=med_docs_dir,
        num_samples=8,
        domains_dir=domains_dir,
        data_dir=data_dir,
    )
    assert distill_res["train_samples"] >= 6
    train_lines = (
        (data_dir / "clinical_oncology_scout" / "train.jsonl")
        .read_text(encoding="utf-8")
        .strip()
        .splitlines()
    )
    for line in train_lines:
        row = json.loads(line)
        payload = json.loads(row["messages"][-1]["content"])
        assert "sharpe" not in payload
        assert "mdd" not in payload
        assert "bps" not in payload
        assert "brief_title" in payload
        assert "verified_citations" in payload

    # 4. Query with "optimization", "variance", and "agent" -> must NOT hijack into finance!
    q_res = cmd_query_domain(
        domain_id="clinical_oncology_scout",
        prompt=(
            "How does tocilizumab agent dose optimization control serum ferritin variance "
            "in CAR-T cytokine release syndrome?"
        ),
        domains_dir=domains_dir,
    )
    assert q_res["result_type"] == "ImplementationContextPack"
    assert q_res["preflight_report"]["passed"] is True
    assert q_res["domain_knowledge_brief"] is not None
    assert q_res["domain_knowledge_brief"]["domain_id"] == "clinical_oncology_scout"
    assert (
        "CAR-T Immunotherapy" in q_res["domain_knowledge_brief"]["brief_title"]
        or "Tocilizumab" in q_res["domain_knowledge_brief"]["brief_title"]
    )
    assert q_res["quant_strategy_spec"]["reported_sharpe_ratio"] is None
    assert q_res["quant_strategy_spec"]["reported_max_drawdown_pct"] is None
    assert q_res["quant_strategy_spec"]["transaction_cost_bps_assumed"] is None

    # 5. Live working-tree incremental sync on a knowledge domain when a new doc is added
    (med_docs_dir / "renal_adjustment_guideline.html").write_text(
        "<html><body><h1>Renal Creatinine Clearance Adjustment</h1>"
        "<p>Reduce fludarabine dosage by 20 percent when Cockcroft-Gault creatinine clearance "
        "is between 30 and 70 mL/min to prevent acute tubular necrosis.</p></body></html>",
        encoding="utf-8",
    )
    sync_res = cmd_sync_domain(
        domain_id="clinical_oncology_scout",
        domains_dir=domains_dir,
    )
    assert sync_res["synced"] is True
    assert sync_res["added"] >= 1

    # 6. Benchmark custom knowledge domain (0 code files, multiple doc chunks)
    bench_res = cmd_benchmark_domain(
        domain_id="clinical_oncology_scout",
        max_queries=6,
        domains_dir=domains_dir,
    )
    assert bench_res["kind"] == "knowledge"
    assert bench_res["preflight_pass_pct"] == 100.0
    assert bench_res["file_recall_at_5_pct"] == 100.0
    assert bench_res["verbatim_faithfulness_pct"] == 100.0

    # 7. Document-only GeminiImplementationExecutor deterministic patch handoff (no IndexError)
    from context_scout.agents.scout_workflow import ScoutWorkflow

    wf = ScoutWorkflow(registry=registry, repo=None)
    principal = PrincipalContext(
        user_id="oncology_fellow",
        role=RoleType.ENGINEER,
        allowed_scopes=["public_eng", "public_quant"],
    )
    doc_pack = wf.run(
        user_prompt="How does tocilizumab agent dose optimization control CAR-T CRS?",
        principal=principal,
        domain_id="clinical_oncology_scout",
    )
    assert isinstance(doc_pack, ImplementationContextPack)
    assert len(doc_pack.code_slices) == 0
    assert doc_pack.domain_knowledge_brief is not None

    executor = GeminiImplementationExecutor()
    diff_text = executor._generate_deterministic_patch(doc_pack, repo_root=med_docs_dir)
    assert "--- a/" in diff_text
    assert "+++ b/" in diff_text
    assert "Context Scout Grounded Synthesis" in diff_text

    # 8. Domain-agnostic Document Corpus MCP Server verification
    mcp_server = create_doc_corpus_mcp_server(domains_dir=domains_dir)

    async def _exercise_doc_mcp() -> None:
        with bind_session_principal(principal):
            search_res = await mcp_server.call_tool(
                "search_hybrid_rrf",
                {
                    "domain_id": "clinical_oncology_scout",
                    "query": "tocilizumab 8 mg/kg dose optimization",
                    "top_k": 5,
                },
            )
            payload_data = getattr(search_res, "structured_content", None)
            assert isinstance(payload_data, dict)
            assert payload_data["count"] >= 1
            assert payload_data["domain_id"] == "clinical_oncology_scout"
            first_chunk_id = payload_data["hits"][0]["chunk_id"]

            quote_res = await mcp_server.call_tool(
                "verify_citation_quote",
                {
                    "domain_id": "clinical_oncology_scout",
                    "chunk_id": first_chunk_id,
                    "verbatim_quote": "intravenous tocilizumab at 8 mg/kg",
                },
            )
            quote_data = getattr(quote_res, "structured_content", None)
            assert isinstance(quote_data, dict)
            assert quote_data["verified"] is True

    asyncio.run(_exercise_doc_mcp())





