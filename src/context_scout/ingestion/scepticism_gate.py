"""Market-Regime Quant Scepticism Gate and Verbatim Citation Verifier (M3).

Enforces institutional transaction cost floors (FIN-01) across IN_NSE, DE_XETRA, and US_EQUITY,
and validates 100% of doc citation spans using DeterministicSlicer (VAL-02).
"""

import logging
from dataclasses import dataclass
from typing import Final

from context_scout.ast_engine.deterministic_slicer import DeterministicSlicer
from context_scout.db.repositories import DocChunkRecord
from context_scout.ingestion.doc_parser import DocumentChunk
from context_scout.schemas.contracts import (
    MARKET_MIN_FRICTION_BPS,
    DocCitationSpan,
    FrictionVerdict,
    MarketRegime,
    QuantStrategyBrief,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScepticismEvaluationResult:
    """Comprehensive evaluation payload from the Quant Scepticism Gate."""

    passed: bool
    friction_verdict: FrictionVerdict
    citations_valid: bool
    citations_verified_count: int
    failed_citations: list[tuple[DocCitationSpan, str]]
    reason: str
    effective_friction_floor_bps: float
    assumed_friction_bps: float | None
    estimated_net_sharpe: float | None = None


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


class MarketRegimeScepticismGate:
    """Evaluates quant research strategy briefs against market reality (FIN-01, VAL-02)."""

    def __init__(self, min_citation_overlap_ratio: float = 0.98) -> None:
        self.slicer = DeterministicSlicer()
        self.min_citation_overlap_ratio: Final[float] = min_citation_overlap_ratio

    def estimate_net_sharpe(
        self,
        reported_sharpe: float | None,
        assumed_bps: float | None,
        floor_bps: float,
        turnover_annual: float = 12.0,
    ) -> float | None:
        """Penalizes gross Sharpe if assumed friction omitted market realities."""
        if reported_sharpe is None:
            return None

        actual_cost_bps = assumed_bps if assumed_bps is not None else 0.0
        friction_deficit_bps = max(0.0, floor_bps - actual_cost_bps)
        # Approximate drag: 10 bps annual cost drag roughly penalizes
        # Sharpe by 0.15 for 12x turnover
        annual_drag = (friction_deficit_bps * turnover_annual) / 10000.0
        # Scaled Sharpe reduction
        net_sharpe = reported_sharpe - (annual_drag * 3.5)
        return round(max(-2.0, net_sharpe), 2)

    def evaluate(
        self,
        brief: QuantStrategyBrief,
        corpus_chunks: list[DocChunkRecord] | list[DocumentChunk],
    ) -> ScepticismEvaluationResult:
        """Evaluates transaction cost realism and verifies 100% of doc citations."""
        floor_bps = calculate_effective_friction_floor(
            target_markets=brief.target_markets,
            is_intraday_futures=brief.is_intraday_futures,
        )

        assumed_bps = brief.transaction_cost_bps_assumed
        net_sharpe = self.estimate_net_sharpe(
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
        elif net_sharpe is not None and net_sharpe <= 0.5:
            verdict = FrictionVerdict.FAILS_AFTER_COSTS
            friction_reason = (
                f"Net Sharpe ratio ({net_sharpe}) after institutional friction "
                f"({floor_bps:.1f} bps) falls <= 0.5 threshold."
            )
        elif (
            brief.reported_max_drawdown_pct is not None
            and abs(brief.reported_max_drawdown_pct) > 30.0
        ):
            verdict = FrictionVerdict.FAILS_AFTER_COSTS
            friction_reason = (
                f"Reported maximum drawdown ({brief.reported_max_drawdown_pct}%) "
                "exceeds 30.0% institutional risk threshold."
            )
        else:
            verdict = FrictionVerdict.REALISTIC_FRICTION
            friction_reason = (
                f"Transaction cost assumptions ({assumed_bps} bps) satisfy "
                f"market floor ({floor_bps:.1f} bps)."
            )

        # 2. Verify all doc citations using DeterministicSlicer (VAL-02)
        chunk_map: dict[str, str] = {}
        for c in corpus_chunks:
            # Map chunk_id as str
            chunk_map[str(c.chunk_id)] = c.verbatim_content
            chunk_map[c.doc_id] = c.verbatim_content

        failed_citations: list[tuple[DocCitationSpan, str]] = []
        verified_count = 0

        for citation in brief.verified_citations:
            source_content = chunk_map.get(str(citation.chunk_id)) or chunk_map.get(
                citation.doc_id
            )
            if not source_content:
                failed_citations.append(
                    (citation, f"Chunk {citation.chunk_id} / {citation.doc_id} not in corpus")
                )
                continue

            v_res = self.slicer.verify_doc_citation(
                citation=citation,
                chunk_content=source_content,
                min_overlap_ratio=self.min_citation_overlap_ratio,
            )
            if not v_res.is_valid:
                failed_citations.append(
                    (
                        citation,
                        f"Quote failed NFKC overlap check: overlap={v_res.token_overlap_ratio:.4f} "
                        f"(min={self.min_citation_overlap_ratio}), reason={v_res.error_reason}",
                    )
                )
            else:
                verified_count += 1

        citations_valid = (
            len(failed_citations) == 0 and len(brief.verified_citations) > 0
        )
        passed = (verdict == FrictionVerdict.REALISTIC_FRICTION) and citations_valid

        reasons = [friction_reason]
        if not citations_valid:
            reasons.append(
                f"{len(failed_citations)} of {len(brief.verified_citations)} "
                "citations failed verbatim verification."
            )

        return ScepticismEvaluationResult(
            passed=passed,
            friction_verdict=verdict,
            citations_valid=citations_valid,
            citations_verified_count=verified_count,
            failed_citations=failed_citations,
            reason=" | ".join(reasons),
            effective_friction_floor_bps=floor_bps,
            assumed_friction_bps=assumed_bps,
            estimated_net_sharpe=net_sharpe,
        )


__all__ = [
    "MarketRegimeScepticismGate",
    "ScepticismEvaluationResult",
    "calculate_effective_friction_floor",
]
