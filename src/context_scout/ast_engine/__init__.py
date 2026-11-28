"""AST Engine: multi-language parsing, deterministic line slicing, and bounded 1-hop closure."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from context_scout.ast_engine.dependency_expander import (
        ASTDependencyExpander,
        ContextCompletenessConfig,
        ScoutGraphRepository,
    )
    from context_scout.ast_engine.deterministic_slicer import DeterministicSlicer
    from context_scout.ast_engine.extractor import ASTExtractor
    from context_scout.ast_engine.models import (
        ASTExpansionCandidate,
        ASTExpansionResult,
        HallucinatedQuoteError,
        InvalidLineBoundsError,
        OffByOneBoundsError,
        ParsedEdge,
        ParsedSymbol,
        QuoteVerificationError,
        QuoteVerificationResult,
        SlicerError,
        SlicerErrorCode,
        TruncatedLineError,
    )

_LAZY_EXPORTS: dict[str, str] = {
    "ASTDependencyExpander": "context_scout.ast_engine.dependency_expander",
    "ContextCompletenessConfig": "context_scout.ast_engine.dependency_expander",
    "ScoutGraphRepository": "context_scout.ast_engine.dependency_expander",
    "DeterministicSlicer": "context_scout.ast_engine.deterministic_slicer",
    "ASTExtractor": "context_scout.ast_engine.extractor",
    "ASTExpansionCandidate": "context_scout.ast_engine.models",
    "ASTExpansionResult": "context_scout.ast_engine.models",
    "HallucinatedQuoteError": "context_scout.ast_engine.models",
    "InvalidLineBoundsError": "context_scout.ast_engine.models",
    "OffByOneBoundsError": "context_scout.ast_engine.models",
    "ParsedEdge": "context_scout.ast_engine.models",
    "ParsedSymbol": "context_scout.ast_engine.models",
    "QuoteVerificationError": "context_scout.ast_engine.models",
    "QuoteVerificationResult": "context_scout.ast_engine.models",
    "SlicerError": "context_scout.ast_engine.models",
    "SlicerErrorCode": "context_scout.ast_engine.models",
    "TruncatedLineError": "context_scout.ast_engine.models",
}

__all__ = [
    "ASTDependencyExpander",
    "ASTExpansionCandidate",
    "ASTExpansionResult",
    "ASTExtractor",
    "ContextCompletenessConfig",
    "DeterministicSlicer",
    "HallucinatedQuoteError",
    "InvalidLineBoundsError",
    "OffByOneBoundsError",
    "ParsedEdge",
    "ParsedSymbol",
    "QuoteVerificationError",
    "QuoteVerificationResult",
    "ScoutGraphRepository",
    "SlicerError",
    "SlicerErrorCode",
    "TruncatedLineError",
]


def __getattr__(name: str) -> Any:
    mod_path = _LAZY_EXPORTS.get(name)
    if mod_path is not None:
        mod = importlib.import_module(mod_path)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
