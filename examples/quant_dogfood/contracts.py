"""Evicted quantitative finance contracts for examples/quant_dogfood."""

from __future__ import annotations

from enum import StrEnum
from typing import Final, Literal

from pydantic import BaseModel, Field, model_validator

from context_scout.schemas.contracts import DocCitationSpan


class MarketRegime(StrEnum):
    INDIA_NSE_BSE = "IN_NSE"
    GERMANY_XETRA_DAX = "DE_XETRA"
    US_NYSE_NASDAQ = "US_EQUITY"
    GLOBAL_MULTI = "GLOBAL"
    INTERNAL_ADR = "INTERNAL_ADR"


class FrictionVerdict(StrEnum):
    REALISTIC_FRICTION = "REALISTIC_FRICTION"
    UNVERIFIED_FRICTION_WARNING = "UNVERIFIED_FRICTION_WARNING"
    FAILS_AFTER_COSTS = "FAILS_AFTER_COSTS"


# FIN-01: Realistic minimum round-trip transaction cost floors (bps) per market regime
MARKET_MIN_FRICTION_BPS: Final[dict[MarketRegime, float]] = {
    MarketRegime.INDIA_NSE_BSE: 15.0,
    MarketRegime.GERMANY_XETRA_DAX: 10.0,
    MarketRegime.US_NYSE_NASDAQ: 5.0,
    MarketRegime.GLOBAL_MULTI: 10.0,
    MarketRegime.INTERNAL_ADR: 0.0,
}


def evaluate_market_friction_floor(
    target_markets: list[MarketRegime],
    assumed_bps: float | None,
    is_intraday_futures: bool = False,
) -> FrictionVerdict:
    """Evaluates whether a paper's transaction cost assumption meets market reality (FIN-01)."""
    if assumed_bps is None:
        return FrictionVerdict.UNVERIFIED_FRICTION_WARNING

    required_bps = 5.0
    for mkt in target_markets:
        floor = MARKET_MIN_FRICTION_BPS.get(mkt, 10.0)
        if mkt == MarketRegime.INDIA_NSE_BSE and is_intraday_futures:
            floor = 5.0
        required_bps = max(required_bps, floor)

    if assumed_bps < required_bps:
        return FrictionVerdict.UNVERIFIED_FRICTION_WARNING
    return FrictionVerdict.REALISTIC_FRICTION


class QuantStrategyBrief(BaseModel):
    """Structured extraction contract for Model 2 with Market-Regime Friction Gate (FIN-01)."""

    strategy_name: str = Field(min_length=1)
    target_markets: list[MarketRegime] = Field(min_length=1)
    is_intraday_futures: bool = False
    mathematical_signal_latex: str = Field(min_length=1)
    entry_exit_rules: list[str] = Field(min_length=1)
    sample_period_years: str = Field(min_length=1)
    reported_sharpe_ratio: float | None = None
    reported_max_drawdown_pct: float | None = None
    transaction_cost_bps_assumed: float | None = None
    friction_verdict: FrictionVerdict = FrictionVerdict.UNVERIFIED_FRICTION_WARNING
    market_specific_caveats: list[str] = Field(min_length=1)
    known_failure_regimes: list[str] = Field(min_length=1)
    verified_citations: list[DocCitationSpan] = Field(min_length=1)
    sufficiency_verdict: Literal["sufficient", "partial", "insufficient"] = "sufficient"
    missing_aspects: list[str] = Field(default_factory=list)
    synthesis_mode: Literal[
        "live_qlora_json",
        "live_base_gemma4_json",
        "deterministic_extractive_fallback",
    ] = "deterministic_extractive_fallback"

    @model_validator(mode="after")
    def _enforce_scepticism_gate(self) -> QuantStrategyBrief:
        is_non_financial_custom_domain = (
            set(self.target_markets) <= {MarketRegime.GLOBAL_MULTI, MarketRegime.INTERNAL_ADR}
            and self.reported_sharpe_ratio is None
            and self.reported_max_drawdown_pct is None
            and self.transaction_cost_bps_assumed is None
            and self.friction_verdict == FrictionVerdict.REALISTIC_FRICTION
        )
        if is_non_financial_custom_domain:
            return self
        if self.friction_verdict != FrictionVerdict.FAILS_AFTER_COSTS:
            computed_verdict = evaluate_market_friction_floor(
                target_markets=self.target_markets,
                assumed_bps=self.transaction_cost_bps_assumed,
                is_intraday_futures=self.is_intraday_futures,
            )
            if computed_verdict == FrictionVerdict.UNVERIFIED_FRICTION_WARNING:
                self.friction_verdict = FrictionVerdict.UNVERIFIED_FRICTION_WARNING
        return self


__all__ = [
    "MARKET_MIN_FRICTION_BPS",
    "FrictionVerdict",
    "MarketRegime",
    "QuantStrategyBrief",
    "evaluate_market_friction_floor",
]
