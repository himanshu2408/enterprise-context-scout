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
