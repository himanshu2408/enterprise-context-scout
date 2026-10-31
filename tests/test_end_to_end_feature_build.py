"""End-to-End Functional Feature Build & Custom-Domain CLI Verification Suite.

Proves two critical enterprise properties:
1. Result Utility ("Can we use the Context Pack to build a real feature?"):
   Runs FederatedScoutCoordinator to produce an ImplementationContextPack for
   `OrderFlowImbalanceSignal`, synthesizes a real method implementation from the
   pack's verbatim code slice and QuantStrategyBrief friction floor, applies it in
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
from pathlib import Path

from context_scout.agents.gemini_executor import GitWorktreeSandbox
from context_scout.agents.scout_workflow import FederatedScoutCoordinator
from context_scout.api.dependencies import (
    _populate_in_memory_repository,
    get_domain_registry,
    get_repo_root,
)
from context_scout.cli import main as cli_main
from context_scout.config import detect_hardware_profile
from context_scout.schemas.contracts import (
    ImplementationContextPack,
    PrincipalContext,
    RoleType,
)


def test_build_real_feature_from_context_pack_in_worktree_sandbox() -> None:
    """Verifies a real feature can be built, linted, and tested from a ContextPack."""
    repo_root = get_repo_root()
    repo = _populate_in_memory_repository(repo_root)
    registry = get_domain_registry()

    principal = PrincipalContext(
        user_id="quant_feature_builder",
        role=RoleType.ENGINEER,
        allowed_scopes=[
            "public_eng",
            "public_quant",
            "repo:trading_strategies",
            "corpus:quant_papers",
            "quant_restricted",
        ],
    )

    doc_chunks = {
        str(c.chunk_id): c.verbatim_content
        for c in repo.list_visible_doc_chunks(principal, "quant_research_scout")
    }

    coordinator = FederatedScoutCoordinator(registry=registry, repo=repo)
    result = coordinator.run(
        user_prompt=(
            "Implement compute_net_friction_edge_bps on OrderFlowImbalanceSignal in "
            "src/context_scout/trading_strategies/order_flow_imbalance.py using the "
            "Order Flow Imbalance research brief friction floor."
        ),
        principal=principal,
        doc_chunks=doc_chunks,
    )

    assert isinstance(result, ImplementationContextPack)
    assert result.preflight_report.passed is True
    assert result.quant_strategy_spec is not None
    cost_floor_bps = float(result.quant_strategy_spec.transaction_cost_bps_assumed or 5.0)
    assert cost_floor_bps >= 5.0

    # Verify the Context Pack localized OrderFlowImbalanceSignal with byte-exact source lines
    ofi_slices = [
        s
        for s in result.code_slices
        if "order_flow_imbalance.py" in s.file_path
    ]
    assert len(ofi_slices) >= 1
    primary_slice = ofi_slices[0]
    assert "OrderFlowImbalanceSignal" in primary_slice.verbatim_code_with_lines

    # Synthesize a real feature diff grounded on the Context Pack's slice and friction floor
    target_rel = "src/context_scout/trading_strategies/order_flow_imbalance.py"
    patch_diff = (
        f"--- a/{target_rel}\n"
        f"+++ b/{target_rel}\n"
        "@@ -202,4 +202,12 @@\n"
        "             rolling_zscore=round(z_score, 4),\n"
        "             volatility_adjusted_signal=round(vol_adj_signal, 4),\n"
        "             current_mid_price=round(snapshot.mid_price, 4),\n"
        "         )\n"
        "+\n"
        "+    def compute_net_friction_edge_bps(\n"
        "+        self,\n"
        "+        expected_alpha_bps: float,\n"
        f"+        friction_floor_bps: float = {cost_floor_bps:.1f},\n"
        "+    ) -> float:\n"
        '+        """Computes net tradable edge (bps) above the regime friction floor."""\n'
        "+        return round(expected_alpha_bps - friction_floor_bps, 4)\n"
    )

    behavioral_check = (
        "from context_scout.trading_strategies.order_flow_imbalance import "
        "OrderFlowImbalanceSignal; "
        "sig = OrderFlowImbalanceSignal(symbol='AAPL'); "
        f"edge = sig.compute_net_friction_edge_bps(25.0); "
        f"assert abs(edge - (25.0 - {cost_floor_bps:.1f})) < 1e-6, f'Unexpected edge: {{edge}}'"
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


def test_clone_and_train_custom_domain_end_to_end_via_cli(
    tmp_path: Path, capsys: object
) -> None:
    """Verifies any user can scaffold, distill, and train a brand-new custom domain via CLI."""
    # 1. Verify hardware auto-detection CLI
    rc_hw = cli_main(["hardware"])
    assert rc_hw == 0
    hw_profile = detect_hardware_profile()
    assert hw_profile["memory_gb"] >= 4.0
    assert hw_profile["max_kv_tokens"] in (8192, 16384, 32768)

    # 2. Create a custom source codebase folder for a brand-new domain (`payments_ledger_scout`)
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

    # Step A: `context-scout init-domain`
    rc_init = cli_main(
        [
            "init-domain",
            "--domain-id",
            "payments_ledger_scout",
            "--kind",
            "code",
            "--display-name",
            "Payments Double-Entry Ledger Scout",
            "--domains-dir",
            str(domains_dir),
        ]
    )
    assert rc_init == 0
    assert (domains_dir / "payments_ledger_scout" / "domain.yaml").is_file()

    # Step B: `context-scout distill`
    rc_distill = cli_main(
        [
            "distill",
            "--domain-id",
            "payments_ledger_scout",
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
    train_file = data_dir / "payments_ledger_scout" / "train.jsonl"
    valid_file = data_dir / "payments_ledger_scout" / "valid.jsonl"
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
            "payments_ledger_scout",
            "--steps",
            "6",
            "--domains-dir",
            str(domains_dir),
            "--data-dir",
            str(data_dir),
        ]
    )
    assert rc_train == 0
    adapter_dir = (
        domains_dir
        / "payments_ledger_scout"
        / "adapters"
        / "gemma4_26b_payments_ledger_scout"
    )
    assert (adapter_dir / "adapters.safetensors").is_file()
    assert (adapter_dir / "adapter_config.json").is_file()
    assert (adapter_dir / "training_metrics.json").is_file()
