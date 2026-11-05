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
    assert "CRISPR-Cas9 Guide RNA Specificity" in bio_pack.quant_strategy_spec.strategy_name
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
    assert "27 of 30" in readme_text
    assert "0 of 22" in readme_text
    assert "22 of 22" in readme_text
    assert "Commercial Frontier Tier A" not in readme_text
    assert "Commercial Frontier Tier B" not in readme_text

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

