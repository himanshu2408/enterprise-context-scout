"""Backward Compatibility Bridge for MarketRegimeScepticismGate (F18).

Preserves exact API compatibility for existing quant test harnesses and callers
while internally delegating citation verification to DocumentCitationGate and
financial logic to FinanceDomainPlugin.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Final

from context_scout.agents.citation_gate import DocumentCitationGate
from examples.quant_dogfood.contracts import QuantStrategyBrief
from examples.quant_dogfood.plugins.domains.finance import (
    FinanceDomainPlugin,
    ScepticismEvaluationResult,
    calculate_effective_friction_floor,
    estimate_net_sharpe,
)

if TYPE_CHECKING:
    from context_scout.db.repositories import DocChunkRecord
    from context_scout.ingestion.doc_parser import DocumentChunk

logger = logging.getLogger(__name__)


class MarketRegimeScepticismGate:
    """Backward-compatible facade delegating to DocumentCitationGate & FinanceDomainPlugin."""

    def __init__(self, min_citation_overlap_ratio: float = 0.98) -> None:
        self.min_citation_overlap_ratio: Final[float] = min_citation_overlap_ratio
        self.citation_gate = DocumentCitationGate(
            min_overlap_ratio=min_citation_overlap_ratio,
            require_citations_default=True,
        )
        self.finance_plugin = FinanceDomainPlugin(
            citation_gate=self.citation_gate,
            min_citation_overlap_ratio=min_citation_overlap_ratio,
        )
        # Preserve slicer attribute for backward compatibility with direct callers
        self.slicer = self.citation_gate.slicer

    def estimate_net_sharpe(
        self,
        reported_sharpe: float | None,
        assumed_bps: float | None,
        floor_bps: float,
        turnover_annual: float = 12.0,
        assumed_annual_volatility: float | None = None,
    ) -> float | None:
        """Penalizes gross Sharpe if assumed friction omitted market realities."""
        return estimate_net_sharpe(
            reported_sharpe=reported_sharpe,
            assumed_bps=assumed_bps,
            floor_bps=floor_bps,
            turnover_annual=turnover_annual,
            assumed_annual_volatility=assumed_annual_volatility,
        )

    def evaluate(
        self,
        brief: QuantStrategyBrief,
        corpus_chunks: list[DocChunkRecord] | list[DocumentChunk] | dict[str, str],
    ) -> ScepticismEvaluationResult:
        """Evaluates strategy brief against market friction floor and doc citations."""
        return self.finance_plugin.evaluate_strategy_brief(
            brief=brief,
            corpus_chunks=corpus_chunks,
        )


__all__ = [
    "MarketRegimeScepticismGate",
    "ScepticismEvaluationResult",
    "calculate_effective_friction_floor",
    "estimate_net_sharpe",
]
