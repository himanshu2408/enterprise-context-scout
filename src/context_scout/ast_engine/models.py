"""Domain models and exceptions for AST engine, deterministic slicing, and expansion."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from context_scout.db.repositories import ASTSymbolRecord
from context_scout.schemas.contracts import CodeSlicePointer, ModificationIntent


class SlicerErrorCode(StrEnum):
    INVALID_BOUNDS = "INVALID_BOUNDS"
    OFF_BY_ONE = "OFF_BY_ONE"
    QUOTE_MISMATCH = "QUOTE_MISMATCH"
    TRUNCATED_QUOTE = "TRUNCATED_QUOTE"
    HALLUCINATED_QUOTE = "HALLUCINATED_QUOTE"


class SlicerError(Exception):
    """Base exception for slicer operations."""

    def __init__(self, code: SlicerErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class InvalidLineBoundsError(SlicerError):
    def __init__(self, message: str) -> None:
        super().__init__(SlicerErrorCode.INVALID_BOUNDS, message)


class OffByOneBoundsError(SlicerError):
    def __init__(self, message: str) -> None:
        super().__init__(SlicerErrorCode.OFF_BY_ONE, message)


class QuoteVerificationError(SlicerError):
    def __init__(self, code: SlicerErrorCode, message: str) -> None:
        super().__init__(code, message)


class TruncatedLineError(QuoteVerificationError):
    def __init__(self, message: str) -> None:
        super().__init__(SlicerErrorCode.TRUNCATED_QUOTE, message)


class HallucinatedQuoteError(QuoteVerificationError):
    def __init__(self, message: str) -> None:
        super().__init__(SlicerErrorCode.HALLUCINATED_QUOTE, message)


class QuoteVerificationResult(BaseModel):
    is_valid: bool
    token_overlap_ratio: float = Field(ge=0.0, le=1.0)
    normalized_quote: str
    matched_source_segment: str | None = None
    error_code: SlicerErrorCode | None = None
    error_reason: str | None = None
    unmatched_tokens: list[str] = Field(default_factory=list)


class ParsedSymbol(BaseModel):
    name: str
    qualified_name: str
    symbol_kind: Literal["function", "class", "method", "interface", "type_alias"]
    start_line: int  # 1-indexed
    end_line: int  # 1-indexed
    start_byte: int
    end_byte: int
    signature: str
    signature_skeleton: str
    docstring: str = ""
    contextual_summary: str = ""


class ParsedEdge(BaseModel):
    source_qualified_name: str
    target_qualified_name: str
    edge_kind: Literal["CALLS", "IMPORTS", "USES_TYPE", "INHERITS", "TESTED_BY"]


@dataclass(frozen=True)
class ASTExpansionCandidate:
    symbol: ASTSymbolRecord
    edge_kind: Literal["CALLS", "IMPORTS", "USES_TYPE", "INHERITS", "TESTED_BY"]
    intent: ModificationIntent
    start_line: int
    end_line: int
    estimated_tokens: int
    file_path: str
    diversity_key: str


@dataclass(frozen=True)
class ASTExpansionResult:
    focus_symbol: ASTSymbolRecord
    imported_type_slices: list[CodeSlicePointer] = field(default_factory=list)
    caller_slices: list[CodeSlicePointer] = field(default_factory=list)
    test_fixture_slices: list[CodeSlicePointer] = field(default_factory=list)
    total_expansion_tokens: int = 0
    budget_exhausted: bool = False
    dropped_candidates_count: int = 0

    @property
    def all_slices(self) -> list[CodeSlicePointer]:
        return self.imported_type_slices + self.caller_slices + self.test_fixture_slices
