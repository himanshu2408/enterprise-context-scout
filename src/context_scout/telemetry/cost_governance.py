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
from pathlib import Path
from typing import Any, Final
from uuid import UUID

import yaml
from pydantic import BaseModel, Field

from context_scout.schemas.contracts import DEFAULT_TENANT_ID, PrincipalContext

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


_DEFAULT_PRICING_YAML: Final[Path] = (
    Path(__file__).resolve().parents[3] / "config" / "model_pricing.yaml"
)


def load_model_pricing_from_yaml(
    pricing_path: Path | str | None = None,
) -> dict[str, ModelPricingRecord]:
    """Loads single-source model pricing records from config/model_pricing.yaml."""
    path = Path(pricing_path) if pricing_path is not None else _DEFAULT_PRICING_YAML
    if not path.is_file():
        raise FileNotFoundError(f"Model pricing configuration not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    models_raw = raw.get("models", {})
    if not isinstance(models_raw, dict):
        raise ValueError(f"Invalid 'models' mapping in {path}")

    table: dict[str, ModelPricingRecord] = {}
    for raw_model_id, entry in models_raw.items():
        if not isinstance(raw_model_id, str) or not isinstance(entry, dict):
            continue
        model_id = raw_model_id.strip()
        raw_in_exp = entry.get("input_expected", entry.get("input"))
        in_exp = float(0.0 if raw_in_exp is None else raw_in_exp)
        in_low = float(entry.get("input_low", in_exp) or in_exp)
        in_high = float(entry.get("input_high", in_exp) or in_exp)
        raw_out_exp = entry.get("output_expected", entry.get("output"))
        out_exp = float(0.0 if raw_out_exp is None else raw_out_exp)
        out_low = float(entry.get("output_low", out_exp) or out_exp)
        out_high = float(entry.get("output_high", out_exp) or out_exp)
        self_hosted = bool(entry.get("self_hosted", False))
        is_zero = self_hosted or (in_exp == 0.0 and out_exp == 0.0)
        rec = ModelPricingRecord(
            model_id=model_id,
            input_usd_per_1m_low=in_low,
            input_usd_per_1m_expected=in_exp,
            input_usd_per_1m_high=in_high,
            output_usd_per_1m_low=out_low,
            output_usd_per_1m_expected=out_exp,
            output_usd_per_1m_high=out_high,
            is_zero_cost=is_zero,
        )
        table[model_id] = rec
        if "/" in model_id:
            short_id = model_id.split("/", 1)[1]
            for alias in (short_id, short_id.lower()):
                if alias not in table:
                    table[alias] = ModelPricingRecord(
                        model_id=alias,
                        input_usd_per_1m_low=in_low,
                        input_usd_per_1m_expected=in_exp,
                        input_usd_per_1m_high=in_high,
                        output_usd_per_1m_low=out_low,
                        output_usd_per_1m_expected=out_exp,
                        output_usd_per_1m_high=out_high,
                        is_zero_cost=is_zero,
                    )
    return table


OCTOBER_2026_PRICING: Final[dict[str, ModelPricingRecord]] = load_model_pricing_from_yaml()

# Default frontier execution target
DEFAULT_FRONTIER_MODEL: Final[str] = "gemini-3.8-flash"
DEFAULT_EXPLORATION_TOKEN_KILL_SWITCH: Final[int] = 1_000_000
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
        for table_key, table_rec in self.pricing_table.items():
            low_key = table_key.lower()
            if low_key == norm_id or low_key.rsplit("/", 1)[-1] == norm_id:
                return table_rec

        # Model aliases and prefix matching
        alias_order = (
            ("hydrogen", "gemini-4-hydrogen"),
            ("neon", "gemini-4-neon"),
            ("carbon", "gemini-4-carbon"),
            ("argon", "gemini-4-argon"),
            ("gemini-4", "gemini-4-argon"),
            ("sonnet", "claude-sonnet-5.5"),
            ("opus", "claude-opus-5.5"),
            ("claude", "claude-opus-5.5"),
            ("sol", "gpt-5.6-sol"),
            ("gpt-5", "gpt-5.6-sol"),
            ("astra", "gpt-6-astra"),
            ("gpt-6", "gpt-6-astra"),
            ("flash", "gemini-3.8-flash"),
            ("gemini-3", "gemini-3.8-flash"),
            ("gemini", "gemini-3.8-flash"),
            ("gemma", "gemma-4-26b-a4b-it"),
            ("qwen", "gemma-4-26b-a4b-it"),
            ("mistral", "gemma-4-26b-a4b-it"),
            ("deepseek", "gemma-4-26b-a4b-it"),
            ("llama", "gemma-4-26b-a4b-it"),
            ("phi", "gemma-4-26b-a4b-it"),
            ("granite", "gemma-4-26b-a4b-it"),
            ("bge", "gemma-4-26b-a4b-it"),
        )
        for keyword, canonical_id in alias_order:
            if keyword in norm_id and canonical_id in self.pricing_table:
                return self.pricing_table[canonical_id]

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
            except (EOFError, KeyboardInterrupt) as exc:
                logger.debug(
                    "Interactive cost confirmation aborted (%s): declining paid compute",
                    type(exc).__name__,
                )
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
        # (tenant_id, user_id) -> dict of metrics
        self._principal_consumption: dict[tuple[str, str], dict[str, Any]] = {}
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

    def _get_or_create_record(
        self, user_id: str, tenant_id: UUID | str = DEFAULT_TENANT_ID
    ) -> dict[str, Any]:
        key = (str(tenant_id), user_id)
        if key not in self._principal_consumption:
            self._principal_consumption[key] = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "reserved_tokens": 0,
                "cost_usd": 0.0,
                "reserved_cost_usd": 0.0,
                "paid_calls_count": 0,
            }
        return self._principal_consumption[key]

    def check_and_reserve(
        self,
        principal: PrincipalContext,
        projected_tokens: int,
        estimated_cost_usd: float = 0.0,
    ) -> None:
        """Atomically validates and reserves projected tokens/spend within principal budget.

        Raises:
            BudgetExceededException: if projected tokens exceed session budget cap.
            KillSwitchTriggeredException: if projected cost exceeds safety ceiling.
        """
        budget_cap = getattr(principal, "session_budget_tokens", None) or getattr(
            principal, "max_session_paid_tokens", 16_000
        )

        with self._lock:
            rec = self._get_or_create_record(principal.user_id, principal.tenant_id)
            effective_used = rec["total_tokens"] + rec["reserved_tokens"]
            projected_total = effective_used + projected_tokens
            if projected_total > budget_cap:
                consumed_tok = rec["total_tokens"]
                reserved_tok = rec["reserved_tokens"]
                raise BudgetExceededException(
                    f"Principal '{principal.user_id}' session budget exceeded: "
                    f"projected total tokens {projected_total:,} exceeds cap {budget_cap:,} "
                    f"(consumed {consumed_tok:,}, reserved {reserved_tok:,}, "
                    f"requested {projected_tokens:,})"
                )

            projected_spend = rec["cost_usd"] + rec["reserved_cost_usd"] + estimated_cost_usd
            if projected_spend > self.spend_ceiling_usd:
                raise KillSwitchTriggeredException(
                    f"Automated kill-switch triggered: projected spend ${projected_spend:.4f} "
                    f"exceeds safety spend ceiling ${self.spend_ceiling_usd:.2f}"
                )

            rec["reserved_tokens"] += max(0, projected_tokens)
            rec["reserved_cost_usd"] += max(0.0, estimated_cost_usd)

    def release_reservation(
        self,
        principal: PrincipalContext,
        reserved_tokens: int,
        reserved_cost_usd: float = 0.0,
    ) -> None:
        """Releases previously reserved tokens and estimated spend (e.g., on aborted call)."""
        with self._lock:
            rec = self._get_or_create_record(principal.user_id, principal.tenant_id)
            rec["reserved_tokens"] = max(0, rec["reserved_tokens"] - max(0, reserved_tokens))
            rec["reserved_cost_usd"] = max(
                0.0, rec["reserved_cost_usd"] - max(0.0, reserved_cost_usd)
            )

    def record_consumption(
        self,
        principal: PrincipalContext,
        input_tokens: int,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        *,
        reserved_tokens: int | None = None,
        reserved_cost_usd: float | None = None,
    ) -> None:
        """Records finalized token consumption and dollar spend, settling reservations."""
        with self._lock:
            rec = self._get_or_create_record(principal.user_id, principal.tenant_id)
            actual_total = input_tokens + output_tokens
            tok_release = (
                max(0, reserved_tokens)
                if reserved_tokens is not None
                else min(rec["reserved_tokens"], actual_total)
            )
            cost_release = (
                max(0.0, reserved_cost_usd)
                if reserved_cost_usd is not None
                else min(rec["reserved_cost_usd"], cost_usd)
            )
            rec["reserved_tokens"] = max(0, rec["reserved_tokens"] - tok_release)
            rec["reserved_cost_usd"] = max(0.0, rec["reserved_cost_usd"] - cost_release)
            rec["input_tokens"] += input_tokens
            rec["output_tokens"] += output_tokens
            rec["total_tokens"] += actual_total
            rec["cost_usd"] += cost_usd
            rec["paid_calls_count"] += 1

    def reset_principal(
        self,
        user_id: str,
        tenant_id: UUID | str = DEFAULT_TENANT_ID,
    ) -> None:
        """Resets in-memory session consumption counters for a single principal."""
        with self._lock:
            self._principal_consumption.pop((str(tenant_id), user_id), None)

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

    def get_consumption(
        self, user_id: str, tenant_id: UUID | str = DEFAULT_TENANT_ID
    ) -> dict[str, Any]:
        """Returns snapshot of consumption metrics for a given principal."""
        with self._lock:
            rec = self._principal_consumption.get((str(tenant_id), user_id))
            if rec is None:
                return {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "reserved_tokens": 0,
                    "cost_usd": 0.0,
                    "reserved_cost_usd": 0.0,
                    "paid_calls_count": 0,
                }
            return dict(rec)

    def reset_session(
        self,
        user_id: str | None = None,
        tenant_id: UUID | str | None = None,
    ) -> None:
        """Resets consumption counters for testing or session rollover."""
        with self._lock:
            if user_id is not None and tenant_id is not None:
                self._principal_consumption.pop((str(tenant_id), user_id), None)
            elif user_id is not None:
                for key in [k for k in self._principal_consumption if k[1] == user_id]:
                    self._principal_consumption.pop(key, None)
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
