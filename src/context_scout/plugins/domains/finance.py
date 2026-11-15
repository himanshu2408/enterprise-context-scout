"""Quantitative Finance Domain Plug-in (FIN-01, F18).

Encapsulates institutional market regimes, statutory transaction cost floors,
turnover/volatility Sharpe penalty formulas, and maximum drawdown risk gates.
Composes with DocumentCitationGate for decoupled, modular evaluation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from context_scout.agents.citation_gate import (
    DocumentCitationGate,
    DocumentCitationGateResult,
)
from context_scout.db.repositories import DocChunkRecord
from context_scout.schemas.contracts import (
    DocCitationSpan,
    FrictionVerdict,
    MarketRegime,
    QuantStrategyBrief,
)

if TYPE_CHECKING:
    from context_scout.ingestion.doc_parser import DocumentChunk

logger = logging.getLogger(__name__)

DOMAIN_ID_FINANCE: Final[str] = "quant_research_scout"

# Institutional minimum round-trip transaction cost floors (bps) per market regime (FIN-01)
MARKET_MIN_FRICTION_BPS: Final[dict[MarketRegime, float]] = {
    MarketRegime.INDIA_NSE_BSE: 15.0,  # 10 bps delivery STT + GST + turnover
    MarketRegime.GERMANY_XETRA_DAX: 10.0,  # Clearing, exchange fee, bid-ask spread
    MarketRegime.US_NYSE_NASDAQ: 5.0,  # SEC Section 31, FINRA TAF, spread
    MarketRegime.GLOBAL_MULTI: 10.0,
    MarketRegime.INTERNAL_ADR: 0.0,
}

# Institutional risk boundaries
MAX_ALLOWABLE_DRAWDOWN_PCT: Final[float] = 30.0
MIN_NET_SHARPE_THRESHOLD: Final[float] = 0.5
DEFAULT_ANNUAL_TURNOVER: Final[float] = 12.0
DEFAULT_ANNUAL_VOLATILITY: Final[float] = 0.285714  # ~3.5x multiplier


def calculate_effective_friction_floor(
    target_markets: list[MarketRegime],
    is_intraday_futures: bool = False,
) -> float:
    """Calculates the strict minimum transaction cost floor (bps) across target markets."""
    if not target_markets:
        return 10.0

    floor = 0.0
    for mkt in target_markets:
        mkt_floor = MARKET_MIN_FRICTION_BPS.get(mkt, 10.0)
        if mkt == MarketRegime.INDIA_NSE_BSE and is_intraday_futures:
            mkt_floor = 5.0
        floor = max(floor, mkt_floor)
    return floor


def estimate_net_sharpe(
    reported_sharpe: float | None,
    assumed_bps: float | None,
    floor_bps: float,
    turnover_annual: float = DEFAULT_ANNUAL_TURNOVER,
    assumed_annual_volatility: float | None = None,
) -> float | None:
    """Penalizes gross Sharpe if assumed friction omitted institutional realities."""
    if reported_sharpe is None:
        return None

    actual_cost_bps = assumed_bps if assumed_bps is not None else 0.0
    friction_deficit_bps = max(0.0, floor_bps - actual_cost_bps)
    # Annualized friction drag as a fraction of capital
    annual_drag = (friction_deficit_bps * turnover_annual) / 10000.0
    # Dimensional reduction: Delta SR = annual_drag / sigma_annual
    inv_vol_multiplier = (
        (1.0 / max(assumed_annual_volatility, 0.01))
        if assumed_annual_volatility is not None
        else (1.0 / DEFAULT_ANNUAL_VOLATILITY)
    )
    net_sharpe = reported_sharpe - (annual_drag * inv_vol_multiplier)
    return round(max(-2.0, net_sharpe), 2)


@dataclass(frozen=True)
class FinancialEvaluationResult:
    """Result of pure financial parameter evaluation (costs, Sharpe, drawdown)."""

    passed: bool
    friction_verdict: FrictionVerdict
    effective_friction_floor_bps: float
    assumed_friction_bps: float | None
    estimated_net_sharpe: float | None
    reason: str


@dataclass(frozen=True)
class ScepticismEvaluationResult:
    """Comprehensive evaluation payload combining financial rules and citation gating."""

    passed: bool
    friction_verdict: FrictionVerdict
    citations_valid: bool
    citations_verified_count: int
    failed_citations: list[tuple[DocCitationSpan, str]]
    reason: str
    effective_friction_floor_bps: float
    assumed_friction_bps: float | None
    estimated_net_sharpe: float | None = None
    citation_gate_result: DocumentCitationGateResult | None = None


class FinanceDomainPlugin:
    """Quantitative Finance Domain Plug-in (F18).

    Evaluates:
    1. Statutory transaction cost friction floors across global market regimes (FIN-01).
    2. Net Sharpe ratio degradation under turnover and volatility drag.
    3. Institutional risk boundaries (max drawdown <= 30%, net Sharpe > 0.5).
    4. Composes cleanly with DocumentCitationGate for decoupled verbatim quote verification.
    """

    domain_id: Final[str] = DOMAIN_ID_FINANCE

    def __init__(
        self,
        citation_gate: DocumentCitationGate | None = None,
        min_citation_overlap_ratio: float = 0.98,
    ) -> None:
        self.citation_gate = citation_gate or DocumentCitationGate(
            min_overlap_ratio=min_citation_overlap_ratio,
            require_citations_default=True,
            strict_freshness=False,
        )

    def evaluate_financial_rules(
        self,
        brief: QuantStrategyBrief,
    ) -> FinancialEvaluationResult:
        """Evaluates transaction cost assumptions, net Sharpe, and max drawdown."""
        floor_bps = calculate_effective_friction_floor(
            target_markets=brief.target_markets,
            is_intraday_futures=brief.is_intraday_futures,
        )
        assumed_bps = brief.transaction_cost_bps_assumed
        net_sharpe = estimate_net_sharpe(
            reported_sharpe=brief.reported_sharpe_ratio,
            assumed_bps=assumed_bps,
            floor_bps=floor_bps,
        )

        # 1. Determine friction verdict
        if assumed_bps is None or assumed_bps < floor_bps:
            verdict = FrictionVerdict.UNVERIFIED_FRICTION_WARNING
            friction_reason = (
                f"Assumed friction ({assumed_bps} bps) is below market floor "
                f"({floor_bps:.1f} bps) for {brief.target_markets}."
            )
        elif net_sharpe is not None and net_sharpe <= MIN_NET_SHARPE_THRESHOLD:
            verdict = FrictionVerdict.FAILS_AFTER_COSTS
            friction_reason = (
                f"Net Sharpe ratio ({net_sharpe}) after institutional friction "
                f"({floor_bps:.1f} bps) falls <= {MIN_NET_SHARPE_THRESHOLD} threshold."
            )
        elif (
            brief.reported_max_drawdown_pct is not None
            and abs(brief.reported_max_drawdown_pct) > MAX_ALLOWABLE_DRAWDOWN_PCT
        ):
            verdict = FrictionVerdict.FAILS_AFTER_COSTS
            friction_reason = (
                f"Reported maximum drawdown ({brief.reported_max_drawdown_pct}%) "
                f"exceeds {MAX_ALLOWABLE_DRAWDOWN_PCT:.1f}% institutional risk threshold."
            )
        else:
            verdict = FrictionVerdict.REALISTIC_FRICTION
            friction_reason = (
                f"Transaction cost assumptions ({assumed_bps} bps) satisfy "
                f"market floor ({floor_bps:.1f} bps)."
            )

        passed = verdict == FrictionVerdict.REALISTIC_FRICTION
        return FinancialEvaluationResult(
            passed=passed,
            friction_verdict=verdict,
            effective_friction_floor_bps=floor_bps,
            assumed_friction_bps=assumed_bps,
            estimated_net_sharpe=net_sharpe,
            reason=friction_reason,
        )

    def evaluate_strategy_brief(
        self,
        brief: QuantStrategyBrief,
        corpus_chunks: list[DocChunkRecord] | list[DocumentChunk] | dict[str, str],
    ) -> ScepticismEvaluationResult:
        """Composes pure financial evaluation with generic DocumentCitationGate."""
        fin_res = self.evaluate_financial_rules(brief)

        citation_res = self.citation_gate.verify_citations(
            citations=brief.verified_citations,
            corpus_chunks=corpus_chunks,
            domain_id=self.domain_id,
            require_citations=True,
        )

        passed = fin_res.passed and citation_res.passed

        reasons = [fin_res.reason]
        if not citation_res.passed:
            reasons.append(
                f"{citation_res.failed_count} of {citation_res.total_citations} "
                "citations failed verbatim verification."
            )

        return ScepticismEvaluationResult(
            passed=passed,
            friction_verdict=fin_res.friction_verdict,
            citations_valid=citation_res.passed,
            citations_verified_count=citation_res.verified_count,
            failed_citations=citation_res.failed_citations,
            reason=" | ".join(reasons),
            effective_friction_floor_bps=fin_res.effective_friction_floor_bps,
            assumed_friction_bps=fin_res.assumed_friction_bps,
            estimated_net_sharpe=fin_res.estimated_net_sharpe,
            citation_gate_result=citation_res,
        )


__all__ = [
    "DOMAIN_ID_FINANCE",
    "MARKET_MIN_FRICTION_BPS",
    "MAX_ALLOWABLE_DRAWDOWN_PCT",
    "MIN_NET_SHARPE_THRESHOLD",
    "FinanceDomainPlugin",
    "FinancialEvaluationResult",
    "ScepticismEvaluationResult",
    "calculate_effective_friction_floor",
    "estimate_net_sharpe",
]
