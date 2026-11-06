"""Deterministic simulation helper for untuned base-model evaluation conditions (M7)."""

from __future__ import annotations

import json


def simulate_untuned_base_model_completion(prompt: str) -> str:
    """Simulates Condition A (zero-shot) and Condition B (few-shot) base model completions."""
    is_few_shot = "Example 1:" in prompt or "Example 2:" in prompt
    tail_score = sum(ord(c) for c in prompt[-100:]) if len(prompt) >= 100 else len(prompt)

    if is_few_shot:
        if (tail_score % 6) == 0:
            return '{\n  "error": "Incomplete schema response",\n  "partial_data": [1, 2, 3]\n}'

        if "QuantStrategyBrief" in prompt or "quant" in prompt.lower():
            base_brief = {
                "strategy_name": "In-Context Quant Signal",
                "target_markets": ["US_EQUITY"],
                "is_intraday_futures": False,
                "mathematical_signal_latex": "\\Delta S_t / S_t",
                "entry_exit_rules": ["Enter when signal > 2 sigma"],
                "sample_period_years": "2020-2024",
                "reported_sharpe_ratio": 1.5,
                "reported_max_drawdown_pct": 8.0,
                "transaction_cost_bps_assumed": 10.0,
                "friction_verdict": "REALISTIC_FRICTION",
                "market_specific_caveats": ["High turnover regime sensitivity"],
                "known_failure_regimes": ["Low volatility chop"],
                "verified_citations": [
                    {
                        "chunk_id": "chunk_in_context_01",
                        "doc_id": "arxiv:general_finance",
                        "doc_title": "General Quantitative Finance Principles",
                        "source_uri": "https://arxiv.org/abs/general_finance",
                        "section_heading": "Introduction",
                        "verbatim_quote": "Momentum signals show persistence across regimes.",
                    }
                ],
            }
            return json.dumps(base_brief, indent=2)

        generic_pointers = [
            {
                "repo_name": "enterprise-context-scout",
                "file_path": "src/context_scout/db/repositories.py",
                "start_line": 1,
                "end_line": 25,
                "qualified_symbol": "context_scout.db.repositories",
                "selection_reason": "High-level module match from in-context prompt",
                "modification_intent": "modify_target",
            }
        ]
        return json.dumps(generic_pointers, indent=2)

    if (tail_score % 3) != 0:
        return (
            "Here is the analysis based on your request:\n\n"
            "```json\n"
            "[\n"
            "  {\n"
            '    "symbol": "ASTSymbolRecord",\n'
            '    "path": "repositories.py"\n'
            "  }\n"
            "]\n"
            "```\n\n"
            "Note: Please verify line numbers before making modifications."
        )

    return json.dumps(
        [
            {
                "repo_name": "enterprise-context-scout",
                "file_path": "src/context_scout/db/repositories.py",
                "start_line": 1,
                "end_line": 30,
                "qualified_symbol": "ASTSymbolRecord",
                "selection_reason": "Identified by zero-shot base inference",
                "modification_intent": "modify_target",
            }
        ],
        indent=2,
    )


__all__ = ["simulate_untuned_base_model_completion"]
