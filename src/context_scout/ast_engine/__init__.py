"""AST Engine: multi-language parsing, deterministic line slicing, and bounded 1-hop closure."""

from context_scout.ast_engine.dependency_expander import (
    ASTDependencyExpander,
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

__all__ = [
    "ASTDependencyExpander",
    "ASTExpansionCandidate",
    "ASTExpansionResult",
    "ASTExtractor",
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
