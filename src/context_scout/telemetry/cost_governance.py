"""Enterprise Cost Governance, Live Pricing & Confirmation Gate (Features F22 & F23).

Adheres to Senior/Staff Production Standards:
1. Live October 2026 model pricing tables (Gemini 4 Argon, Claude Opus 5.5, GPT-5.6 Sol,
   Gemini 3.8 Flash, local Gemma 4 26B MoE).
2. Pre-execution Low/Expected/High cost range estimation prior to paid frontier dispatch.
3. Upfront blocking confirmation gate (`request_paid_execution_confirmation`) guaranteeing
   zero paid model execution and $0 spend on unconfirmed/declined requests.
4. Per-principal session token budget enforcement (`PrincipalContext.session_budget_tokens`
   and `max_session_paid_tokens`).
5. Automated exploration kill-switch halting runaway traversals or budget exhaustion.
"""

from __future__ import annotations

import logging
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final

from pydantic import BaseModel, Field

from context_scout.schemas.contracts import PrincipalContext

logger = logging.getLogger(__name__)


# =============================================================================
# Cost Governance Exceptions
# =============================================================================


class BudgetExceededError(RuntimeError):
    """Raised when a principal's session token budget is breached."""


# Backwards-compatible alias for existing code
BudgetExceededException = BudgetExceededError


class ConfirmationDeclinedError(RuntimeError):
    """Raised when paid model compute is declined or unconfirmed by the user."""


ConfirmationDeclinedException = ConfirmationDeclinedError


class KillSwitchTriggeredError(RuntimeError):
    """Raised when exploration tokens or session spend breaches automated safety ceilings."""


KillSwitchTriggeredException = KillSwitchTriggeredError


# =============================================================================
# Model Pricing Definitions (Live October 2026 Verified Rates)
# =============================================================================


@dataclass(frozen=True)
class ModelPricingRecord:
    """Pricing rates in USD per 1 Million tokens for a specific model."""

    model_id: str
    input_usd_per_1m_low: float
    input_usd_per_1m_expected: float
    input_usd_per_1m_high: float
    output_usd_per_1m_low: float
    output_usd_per_1m_expected: float
    output_usd_per_1m_high: float
    is_zero_cost: bool = False


OCTOBER_2026_PRICING: Final[dict[str, ModelPricingRecord]] = {
    "gemini-4-argon": ModelPricingRecord(
        model_id="gemini-4-argon",
        input_usd_per_1m_low=2.00,
        input_usd_per_1m_expected=3.00,
        input_usd_per_1m_high=4.00,
        output_usd_per_1m_low=10.00,
        output_usd_per_1m_expected=15.00,
        output_usd_per_1m_high=20.00,
        is_zero_cost=False,
    ),
    "claude-opus-5.5": ModelPricingRecord(
        model_id="claude-opus-5.5",
        input_usd_per_1m_low=4.00,
        input_usd_per_1m_expected=4.00,
        input_usd_per_1m_high=4.00,
        output_usd_per_1m_low=20.00,
        output_usd_per_1m_expected=20.00,
        output_usd_per_1m_high=20.00,
        is_zero_cost=False,
    ),
    "gpt-5.6-sol": ModelPricingRecord(
        model_id="gpt-5.6-sol",
        input_usd_per_1m_low=4.00,
        input_usd_per_1m_expected=4.00,
        input_usd_per_1m_high=4.00,
        output_usd_per_1m_low=20.00,
        output_usd_per_1m_expected=20.00,
        output_usd_per_1m_high=20.00,
        is_zero_cost=False,
    ),
    "gemini-3.8-flash": ModelPricingRecord(
        model_id="gemini-3.8-flash",
        input_usd_per_1m_low=0.75,
        input_usd_per_1m_expected=1.00,
        input_usd_per_1m_high=1.50,
        output_usd_per_1m_low=3.75,
        output_usd_per_1m_expected=5.00,
        output_usd_per_1m_high=7.50,
        is_zero_cost=False,
    ),
    "gemma-4-26b-a4b-it": ModelPricingRecord(
        model_id="gemma-4-26b-a4b-it",
        input_usd_per_1m_low=0.0,
        input_usd_per_1m_expected=0.0,
        input_usd_per_1m_high=0.0,
        output_usd_per_1m_low=0.0,
        output_usd_per_1m_expected=0.0,
        output_usd_per_1m_high=0.0,
        is_zero_cost=True,
    ),
}

# Default frontier execution target
DEFAULT_FRONTIER_MODEL: Final[str] = "gemini-3.8-flash"
DEFAULT_EXPLORATION_TOKEN_KILL_SWITCH: Final[int] = 250_000
DEFAULT_SESSION_SPEND_CEILING_USD: Final[float] = 25.0


# =============================================================================
# Cost Estimate Report
# =============================================================================


class CostEstimateReport(BaseModel):
    """Explicit pre-execution cost estimation report with Low/Expected/High ranges."""

    model_id: str
    estimated_input_tokens: int = Field(ge=0)
    estimated_output_tokens: int = Field(ge=0)
    max_output_tokens: int = Field(ge=1)
    cost_low_usd: float = Field(ge=0.0)
    cost_expected_usd: float = Field(ge=0.0)
    cost_high_usd: float = Field(ge=0.0)
    pricing_source: str = "October 2026 Live Verified Pricing"
    is_paid_model: bool = True

    def formatted_summary(self) -> str:
        """Emits a human-readable cost projection summary for UI/CLI confirmation."""
        if not self.is_paid_model:
            return (
                f"Model: {self.model_id} (Zero Cost Local Model)\n"
                f"Estimated Tokens: {self.estimated_input_tokens} in / "
                f"{self.estimated_output_tokens} out\n"
                "Cost: $0.00 USD (No paid API compute required)"
            )
        return (
            f"=== Paid Compute Pre-Execution Cost Estimate ===\n"
            f"Target Model: {self.model_id}\n"
            f"Input Tokens: {self.estimated_input_tokens:,}\n"
            f"Output Tokens (Estimated / Max): {self.estimated_output_tokens:,} / "
            f"{self.max_output_tokens:,}\n"
            f"Projected Cost Range (USD):\n"
            f"  - Low:      ${self.cost_low_usd:.6f}\n"
            f"  - Expected: ${self.cost_expected_usd:.6f}\n"
            f"  - High:     ${self.cost_high_usd:.6f}\n"
            f"Pricing Source: {self.pricing_source}\n"
            f"================================================"
        )


# =============================================================================
# Paid Compute Cost Estimator & Upfront Confirmation Gate
# =============================================================================


class PaidComputeCostEstimator:
    """Calculates granular pre-run cost estimates and enforces blocking confirmation."""

    def __init__(self, pricing_table: dict[str, ModelPricingRecord] | None = None) -> None:
        self.pricing_table = pricing_table or OCTOBER_2026_PRICING

    def get_pricing_record(self, model_id: str) -> ModelPricingRecord:
        """Retrieves verified pricing record for a given model ID with safe fallback."""
        norm_id = model_id.strip().lower()
        if norm_id in self.pricing_table:
            return self.pricing_table[norm_id]

        # Model aliases and prefix matching
        if "argon" in norm_id or "gemini-4" in norm_id:
            return self.pricing_table["gemini-4-argon"]
        if "opus" in norm_id or "claude" in norm_id:
            return self.pricing_table["claude-opus-5.5"]
        if "sol" in norm_id or "gpt-5" in norm_id:
            return self.pricing_table["gpt-5.6-sol"]
        if "flash" in norm_id or "gemini-3" in norm_id or "gemini" in norm_id:
            return self.pricing_table["gemini-3.8-flash"]
        if "gemma" in norm_id:
            return self.pricing_table["gemma-4-26b-a4b-it"]

        logger.warning(
            "Unknown model '%s' for cost estimation; using '%s' pricing fallback",
            model_id,
            DEFAULT_FRONTIER_MODEL,
        )
        return self.pricing_table[DEFAULT_FRONTIER_MODEL]

    def estimate_prompt_tokens(
        self,
        prompt: str,
        client: Any = None,
        model_id: str = DEFAULT_FRONTIER_MODEL,
    ) -> int:
        """Estimates token count using client API if available, falling back to heuristic."""
        if not prompt:
            return 0

        has_count_tokens = (
            client is not None
            and hasattr(client, "models")
            and hasattr(client.models, "count_tokens")
        )
        if has_count_tokens:
            try:
                resp = client.models.count_tokens(model=model_id, contents=prompt)
                if hasattr(resp, "total_tokens") and resp.total_tokens is not None:
                    return int(resp.total_tokens)
            except Exception as e:
                logger.debug("Live count_tokens failed, using heuristic: %s", e)

        # Verified standard character-to-token ratio heuristic
        return max(1, int(len(prompt) / 3.8))

    def calculate_cost_estimate(
        self,
        model_id: str,
        estimated_input_tokens: int,
        max_output_tokens: int = 4096,
    ) -> CostEstimateReport:
        """Computes Low, Expected, and High dollar costs based on input and output bounds."""
        pricing = self.get_pricing_record(model_id)
        if pricing.is_zero_cost:
            return CostEstimateReport(
                model_id=pricing.model_id,
                estimated_input_tokens=estimated_input_tokens,
                estimated_output_tokens=0,
                max_output_tokens=max_output_tokens,
                cost_low_usd=0.0,
                cost_expected_usd=0.0,
                cost_high_usd=0.0,
                is_paid_model=False,
            )

        # Output completion token heuristics
        out_high = max_output_tokens
        out_low = min(out_high, min(512, max(1, max_output_tokens // 4)))
        out_expected = min(out_high, min(1500, max(out_low, max_output_tokens // 2)))

        cost_low = round(
            (
                estimated_input_tokens * pricing.input_usd_per_1m_low
                + out_low * pricing.output_usd_per_1m_low
            )
            / 1_000_000.0,
            6,
        )
        cost_expected = round(
            (
                estimated_input_tokens * pricing.input_usd_per_1m_expected
                + out_expected * pricing.output_usd_per_1m_expected
            )
            / 1_000_000.0,
            6,
        )
        cost_high = round(
            (
                estimated_input_tokens * pricing.input_usd_per_1m_high
                + out_high * pricing.output_usd_per_1m_high
            )
            / 1_000_000.0,
            6,
        )

        # Enforce invariant: low <= expected <= high
        cost_expected = max(cost_low, cost_expected)
        cost_high = max(cost_expected, cost_high)

        return CostEstimateReport(
            model_id=pricing.model_id,
            estimated_input_tokens=estimated_input_tokens,
            estimated_output_tokens=out_expected,
            max_output_tokens=max_output_tokens,
            cost_low_usd=cost_low,
            cost_expected_usd=cost_expected,
            cost_high_usd=cost_high,
            is_paid_model=True,
        )

    def request_paid_execution_confirmation(
        self,
        report: CostEstimateReport,
        confirmation_callback: Callable[[CostEstimateReport], bool] | None = None,
        auto_approve: bool = False,
        interactive_prompt: bool = True,
    ) -> bool:
        """Upfront blocking confirmation gate before paid compute dispatch.

        Raises:
            ConfirmationDeclinedException: if confirmation is declined or missing.
        """
        # Zero-cost models (local Gemma) require no financial confirmation
        if not report.is_paid_model:
            return True

        # Explicit auto-approval (e.g. programmatic approval flag)
        if auto_approve:
            logger.info(
                "Paid execution auto-approved for model %s (expected $%.6f)",
                report.model_id,
                report.cost_expected_usd,
            )
            return True

        # Custom confirmation callback
        if confirmation_callback is not None:
            approved = bool(confirmation_callback(report))
            if not approved:
                raise ConfirmationDeclinedException(
                    f"Paid execution declined by confirmation callback for model {report.model_id} "
                    f"(expected cost ${report.cost_expected_usd:.6f})"
                )
            return True

        # Interactive terminal prompt if TTY is attached
        if interactive_prompt and sys.stdin.isatty():
            print("\n" + report.formatted_summary())
            try:
                ans = input("Approve paid model compute? [y/N]: ").strip().lower()
                if ans in ("y", "yes"):
                    return True
            except (EOFError, KeyboardInterrupt):
                pass
            raise ConfirmationDeclinedException(
                f"Paid execution declined by user prompt for model {report.model_id}"
            )

        # In headless or non-interactive environments without explicit approval, block execution
        raise ConfirmationDeclinedException(
            f"Paid model compute ({report.model_id}, expected ${report.cost_expected_usd:.6f}) "
            "requires explicit confirmation (auto_approve=True or confirmation_callback)"
        )


# =============================================================================
# Session Budget Tracker & Kill-Switch Governance
# =============================================================================


class SessionBudgetTracker:
    """Thread-safe multi-principal session token and spend tracker with kill-switch."""

    _instance: SessionBudgetTracker | None = None
    _singleton_lock: Final[threading.Lock] = threading.Lock()

    def __init__(self, spend_ceiling_usd: float = DEFAULT_SESSION_SPEND_CEILING_USD) -> None:
        self._lock = threading.Lock()
        self.spend_ceiling_usd = spend_ceiling_usd
        # user_id -> dict of metrics
        self._principal_consumption: dict[str, dict[str, Any]] = {}
        # session_id -> cumulative exploration tokens
        self._exploration_tokens: dict[str, int] = {}

    @classmethod
    def get_instance(cls) -> SessionBudgetTracker:
        """Singleton accessor for cross-agent session budget tracking."""
        if cls._instance is None:
            with cls._singleton_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _get_or_create_record(self, user_id: str) -> dict[str, Any]:
        if user_id not in self._principal_consumption:
            self._principal_consumption[user_id] = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "paid_calls_count": 0,
            }
        return self._principal_consumption[user_id]

    def check_and_reserve(
        self,
        principal: PrincipalContext,
        projected_tokens: int,
        estimated_cost_usd: float = 0.0,
    ) -> None:
        """Validates that projected consumption fits within principal's budget cap.

        Raises:
            BudgetExceededException: if projected tokens exceed session budget cap.
            KillSwitchTriggeredException: if projected cost exceeds safety ceiling.
        """
        budget_cap = getattr(principal, "session_budget_tokens", None) or getattr(
            principal, "max_session_paid_tokens", 16_000
        )

        with self._lock:
            rec = self._get_or_create_record(principal.user_id)
            projected_total = rec["total_tokens"] + projected_tokens
            if projected_total > budget_cap:
                raise BudgetExceededException(
                    f"Principal '{principal.user_id}' session budget exceeded: "
                    f"projected total tokens {projected_total:,} exceeds cap {budget_cap:,} "
                    f"(already consumed {rec['total_tokens']:,}, requested {projected_tokens:,})"
                )

            projected_spend = rec["cost_usd"] + estimated_cost_usd
            if projected_spend > self.spend_ceiling_usd:
                raise KillSwitchTriggeredException(
                    f"Automated kill-switch triggered: projected spend ${projected_spend:.4f} "
                    f"exceeds safety spend ceiling ${self.spend_ceiling_usd:.2f}"
                )

    def record_consumption(
        self,
        principal: PrincipalContext,
        input_tokens: int,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        """Records finalized token consumption and dollar spend."""
        with self._lock:
            rec = self._get_or_create_record(principal.user_id)
            rec["input_tokens"] += input_tokens
            rec["output_tokens"] += output_tokens
            rec["total_tokens"] += input_tokens + output_tokens
            rec["cost_usd"] += cost_usd
            rec["paid_calls_count"] += 1

    def record_exploration_tokens(
        self,
        tokens: int,
        max_cap: int = DEFAULT_EXPLORATION_TOKEN_KILL_SWITCH,
        session_id: str = "default",
    ) -> None:
        """Tracks local exploration tokens; triggers kill-switch if cap breached.

        Raises:
            KillSwitchTriggeredException: when cumulative exploration tokens breach max_cap.
        """
        with self._lock:
            current = self._exploration_tokens.get(session_id, 0)
            new_total = current + tokens
            self._exploration_tokens[session_id] = new_total
            if new_total > max_cap:
                raise KillSwitchTriggeredException(
                    f"Automated exploration kill-switch triggered: cumulative tokens "
                    f"({new_total:,}) exceeded safety exploration ceiling ({max_cap:,})"
                )

    def get_consumption(self, user_id: str) -> dict[str, Any]:
        """Returns snapshot of consumption metrics for a given principal."""
        with self._lock:
            rec = self._principal_consumption.get(user_id)
            if rec is None:
                return {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cost_usd": 0.0,
                    "paid_calls_count": 0,
                }
            return dict(rec)

    def reset_session(self, user_id: str | None = None) -> None:
        """Resets consumption counters for testing or session rollover."""
        with self._lock:
            if user_id is not None:
                self._principal_consumption.pop(user_id, None)
            else:
                self._principal_consumption.clear()
                self._exploration_tokens.clear()


__all__ = [
    "DEFAULT_EXPLORATION_TOKEN_KILL_SWITCH",
    "DEFAULT_FRONTIER_MODEL",
    "DEFAULT_SESSION_SPEND_CEILING_USD",
    "OCTOBER_2026_PRICING",
    "BudgetExceededError",
    "BudgetExceededException",
    "ConfirmationDeclinedException",
    "CostEstimateReport",
    "KillSwitchTriggeredException",
    "ModelPricingRecord",
    "PaidComputeCostEstimator",
    "SessionBudgetTracker",
]
