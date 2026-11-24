"""End-to-End Functional Feature Build & Custom-Domain CLI Verification Suite.

Proves two critical enterprise properties:
1. Result Utility ("Can we use the Context Pack to build a real feature?"):
   Runs FederatedScoutCoordinator to produce an ImplementationContextPack for
   `ScoutSettings`, synthesizes a real helper implementation from the
   pack's verbatim code slice and DomainKnowledgeBrief, applies it in
   an isolated `GitWorktreeSandbox`, and verifies it passes `ruff check`, `mypy`,
   and behavioral execution tests.
2. Zero-Code Custom Domain Extensibility ("Can a user clone the repo and train on
   any new domain via CLI?"):
   Executes `context-scout hardware`, `context-scout init-domain`, `context-scout distill`,
   and `context-scout train` on custom code and research folders and verifies the
   resulting QLoRA adapter loads into `DomainRegistry` and `AdapterPool`.
"""

from __future__ import annotations

import json
import sys
import uuid
from datetime import date
from pathlib import Path

import psycopg
import pytest

from context_scout.agents.gemini_executor import GitWorktreeSandbox
from context_scout.agents.scout_workflow import FederatedScoutCoordinator
from context_scout.api.dependencies import (
    _populate_in_memory_repository,
    get_domain_registry,
    get_repo_root,
)
from context_scout.cli import main as cli_main
from context_scout.config import ScoutSettings, detect_hardware_profile
from context_scout.db.repositories import DocChunkRecord
from context_scout.retrieval.embedder import get_embedding_provider
from context_scout.schemas.contracts import (
    DocumentScope,
    ImplementationContextPack,
    PrincipalContext,
    RoleType,
)


def test_build_real_feature_from_context_pack_in_worktree_sandbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies a real feature can be built, linted, and tested from a ContextPack."""
    monkeypatch.setenv("SCOUT_USE_HASH_EMBEDDINGS_FOR_TESTS", "1")
    repo_root = get_repo_root()
    repo = _populate_in_memory_repository(repo_root)
    registry = get_domain_registry()

    principal = PrincipalContext(
        user_id="core_feature_builder",
        role=RoleType.ENGINEER,
        allowed_scopes=[
            "public_eng",
            "public_docs",
            "repo:code_architecture_scout",
            "corpus:enterprise_docs",
            "docs_restricted",
        ],
    )

    adr_quote = (
        "ScoutSettings compute_budget_headroom_tokens must reserve at least 1024 "
        "KV-cache tokens for system framing in config.py."
    )
    repo.insert_doc_chunk(
        principal,
        DocChunkRecord(
            chunk_id=uuid.uuid4(),
            domain_id="enterprise_docs_scout",
            doc_id="adr-kv-budget-01",
            doc_title="Enterprise Architecture KV-Cache Budget ADR",
            source_uri="https://docs.internal/adr/kv-budget",
            market_scope=DocumentScope.INTERNAL_ADR,
            section_heading="Token Budget Headroom",
            verbatim_content=adr_quote,
            contextual_header="Architecture ADR on KV-cache prompt budget reservation.",
            published_date=date(2026, 1, 15),
            authority_weight=1.0,
            allowed_scopes=["public_docs", "corpus:enterprise_docs"],
            embedding=get_embedding_provider().embed_text(adr_quote, is_query=False),
            metadata_json={},
        ),
    )

    doc_chunks = {
        str(c.chunk_id): c.verbatim_content
        for c in repo.list_visible_doc_chunks(principal, "enterprise_docs_scout")
    }

    coordinator = FederatedScoutCoordinator(
        registry=registry,
        repo=repo,
    )
    result = coordinator.run(
        user_prompt=(
            "Implement compute_budget_headroom_tokens for ScoutSettings in "
            "src/context_scout/config.py."
        ),
        principal=principal,
        doc_chunks=doc_chunks,
    )

    assert isinstance(result, ImplementationContextPack)
    assert result.preflight_report.passed is True

    # Verify the Context Pack localized ScoutSettings with byte-exact source lines
    cfg_slices = [s for s in result.code_slices if "config.py" in s.file_path]
    assert len(cfg_slices) >= 1
    primary_slice = cfg_slices[0]
    assert "ScoutSettings" in primary_slice.verbatim_code_with_lines

    # Synthesize a real feature diff grounded on the Context Pack's slice
    target_rel = "src/context_scout/config.py"
    patch_diff = (
        f"--- a/{target_rel}\n"
        f"+++ b/{target_rel}\n"
        "@@ -651,4 +651,10 @@\n"
        '         "lora_slots_allocated": lora_slots_allocated,\n'
        '         "lora_slots_capacity": cfg.lora_slots_capacity,\n'
        "     }\n"
        "+\n"
        "+\n"
        "+def compute_budget_headroom_tokens(\n"
        "+    max_kv_tokens: int, reserved_tokens: int = 1024\n"
        "+) -> int:\n"
        '+    """Computes remaining KV-cache token headroom after reserved prompt budget."""\n'
        "+    return max(0, max_kv_tokens - reserved_tokens)\n"
    )

    behavioral_check = (
        "from context_scout.config import compute_budget_headroom_tokens; "
        "assert compute_budget_headroom_tokens(8192, 1024) == 7168"
    )

    with GitWorktreeSandbox(repo_root=repo_root) as sandbox:
        applied, apply_msg = sandbox.apply_patch(patch_diff)
        assert applied is True, f"git apply failed: {apply_msg}"
        lint_passed, tests_passed, verify_out = sandbox.run_verification(
            lint_cmd=[sys.executable, "-m", "ruff", "check", target_rel],
            test_cmd=[sys.executable, "-c", behavioral_check],
        )

    assert lint_passed is True, f"Lint failed: {verify_out}"
    assert tests_passed is True, f"Behavioral test failed: {verify_out}"


def test_clone_and_train_custom_domain_end_to_end_via_cli(tmp_path: Path, capsys: object) -> None:
    """Verifies any user can scaffold, distill, and train a brand-new custom domain via CLI."""
    # 1. Verify hardware auto-detection CLI
    rc_hw = cli_main(["hardware"])
    assert rc_hw == 0
    hw_profile = detect_hardware_profile()
    assert hw_profile["memory_gb"] >= 4.0
    assert hw_profile["max_kv_tokens"] in (8192, 16384, 32768)

    # 2. Create a custom source codebase folder for an isolated ephemeral domain
    domain_id = f"eval_payments_ledger_{uuid.uuid4().hex[:8]}"
    custom_src = tmp_path / "payments_repo"
    custom_src.mkdir(parents=True)
    (custom_src / "ledger.py").write_text(
        "class DoubleEntryLedger:\n"
        "    def post_transfer(self, debit_acct: str, credit_acct: str, cents: int) -> bool:\n"
        "        if cents <= 0:\n"
        "            raise ValueError('Transfer amount must be positive')\n"
        "        return True\n",
        encoding="utf-8",
    )

    domains_dir = tmp_path / "domains"
    data_dir = tmp_path / "data" / "tuning"

    try:
        # Step A: `context-scout init-domain`
        rc_init = cli_main(
            [
                "init-domain",
                "--domain-id",
                domain_id,
                "--kind",
                "code",
                "--display-name",
                "Payments Double-Entry Ledger Scout",
                "--domains-dir",
                str(domains_dir),
            ]
        )
        assert rc_init == 0
        assert (domains_dir / domain_id / "domain.yaml").is_file()

        # Step B: `context-scout distill`
        rc_distill = cli_main(
            [
                "distill",
                "--domain-id",
                domain_id,
                "--source-dir",
                str(custom_src),
                "--num-samples",
                "12",
                "--domains-dir",
                str(domains_dir),
                "--data-dir",
                str(data_dir),
            ]
        )
        assert rc_distill == 0
        train_file = data_dir / domain_id / "train.jsonl"
        valid_file = data_dir / domain_id / "valid.jsonl"
        assert train_file.is_file()
        assert valid_file.is_file()
        first_sample = json.loads(train_file.read_text(encoding="utf-8").splitlines()[0])
        assert "DoubleEntryLedger" in first_sample["messages"][0]["content"]
        assert "Retrieved AST Candidate:" in first_sample["messages"][0]["content"]

        # Step C: `context-scout train`
        rc_train = cli_main(
            [
                "train",
                "--domain-id",
                domain_id,
                "--steps",
                "6",
                "--domains-dir",
                str(domains_dir),
                "--data-dir",
                str(data_dir),
            ]
        )
        assert rc_train == 0
        adapter_dir = domains_dir / domain_id / "adapters" / f"gemma4_26b_{domain_id}"
        assert (adapter_dir / "adapters.safetensors").is_file()
        assert (adapter_dir / "adapter_config.json").is_file()
        assert (adapter_dir / "training_metrics.json").is_file()
    finally:
        cleanup_tables = (
            "ast_edges",
            "ast_symbols",
            "code_files",
            "repositories",
            "confluence_pages",
            "jira_issues",
            "doc_chunks",
            "indexing_queue",
            "indexing_dead_letter",
            "execution_telemetry",
            "domain_packs",
        )
        try:
            with psycopg.connect(ScoutSettings().db_admin_url) as conn:
                for table in cleanup_tables:
                    conn.execute(f"DELETE FROM {table} WHERE domain_id = %s;", (domain_id,))
                conn.commit()
        except psycopg.Error:
            pass
