"""Byte-exact source line slicer and NFKC quote verification engine (VAL-02)."""

import difflib
import re
import unicodedata
from typing import Final

from context_scout.ast_engine.models import (
    HallucinatedQuoteError,
    InvalidLineBoundsError,
    OffByOneBoundsError,
    QuoteVerificationError,
    QuoteVerificationResult,
    SlicerErrorCode,
    TruncatedLineError,
)
from context_scout.db.repositories import CodeFileRecord
from context_scout.schemas.contracts import CodeSlicePointer, DocCitationSpan, VerbatimCodeSlice


class DeterministicSlicer:
    """Byte-exact source line slicer and NFKC quote verification engine (VAL-02)."""

    MIN_QUOTE_OVERLAP_RATIO: Final[float] = 0.98

    def slice_lines(
        self,
        raw_content: str,
        start_line: int,
        end_line: int,
        line_number_width: int = 4,
    ) -> str:
        """Extracts verbatim lines with 1-indexed numbers and fixed width.

        Format: '{line_no:4d} | {line_content}'
        """
        lines = raw_content.splitlines()
        total_lines = len(lines)

        self.validate_bounds(start_line, end_line, total_lines)

        selected = lines[start_line - 1 : end_line]
        formatted = [
            f"{line_no:{line_number_width}d} | {line}"
            for line_no, line in enumerate(selected, start=start_line)
        ]
        return "\n".join(formatted)

    def validate_bounds(self, start_line: int, end_line: int, total_lines: int) -> None:
        """Validates 1-indexed line numbers and flags off-by-one errors."""
        if start_line == 0:
            raise OffByOneBoundsError(
                "0-indexed start_line detected; line numbers must be 1-indexed"
            )
        if start_line < 1:
            raise InvalidLineBoundsError(f"start_line ({start_line}) must be >= 1")
        if end_line < start_line:
            raise InvalidLineBoundsError(
                f"end_line ({end_line}) cannot be less than start_line ({start_line})"
            )
        if total_lines == 0:
            raise InvalidLineBoundsError("Cannot slice empty file")
        if start_line > total_lines:
            raise InvalidLineBoundsError(
                f"start_line ({start_line}) exceeds file line count ({total_lines})"
            )
        if end_line == total_lines + 1:
            raise OffByOneBoundsError(
                f"end_line ({end_line}) exceeds file line count ({total_lines}) by 1"
            )
        if end_line > total_lines:
            raise InvalidLineBoundsError(
                f"end_line ({end_line}) exceeds file line count ({total_lines})"
            )

    def slice_pointer(
        self,
        pointer: CodeSlicePointer,
        file_record: CodeFileRecord,
        auto_expanded_by_ast: bool = False,
    ) -> VerbatimCodeSlice:
        """Hydrates a CodeSlicePointer with byte-exact code and git commit/blob provenance."""
        verbatim_code = self.slice_lines(
            raw_content=file_record.raw_content,
            start_line=pointer.start_line,
            end_line=pointer.end_line,
        )
        return VerbatimCodeSlice(
            repo_name=pointer.repo_name,
            file_path=pointer.file_path,
            start_line=pointer.start_line,
            end_line=pointer.end_line,
            qualified_symbol=pointer.qualified_symbol,
            selection_reason=pointer.selection_reason,
            modification_intent=pointer.modification_intent,
            git_commit_sha=file_record.git_commit_sha,
            git_blob_sha=file_record.git_blob_sha,
            verbatim_code_with_lines=verbatim_code,
            auto_expanded_by_ast=auto_expanded_by_ast,
        )

    def normalize_text(self, text: str) -> str:
        """Applies NFKC Unicode normalization and collapses whitespace."""
        nfkc = unicodedata.normalize("NFKC", text)
        return re.sub(r"\s+", " ", nfkc).strip()

    def tokenize(self, text: str) -> list[str]:
        """Deterministic tokenization over words and punctuation."""
        return re.findall(r"\w+|[^\w\s]", text)

    def _has_unclosed_delimiters(self, text: str) -> bool:
        counts = {
            "(": text.count("(") - text.count(")"),
            "[": text.count("[") - text.count("]"),
            "{": text.count("{") - text.count("}"),
        }
        return any(diff > 0 for diff in counts.values())

    def verify_quote(
        self,
        quote: str,
        source_text: str,
        min_overlap_ratio: float = MIN_QUOTE_OVERLAP_RATIO,
        strict: bool = False,
    ) -> QuoteVerificationResult:
        """Verifies quote against source text using NFKC normalization and token overlap.

        Implements VAL-02 verification.
        """
        if not quote.strip():
            raise QuoteVerificationError(
                SlicerErrorCode.QUOTE_MISMATCH, "Empty quote cannot be verified"
            )

        norm_quote = self.normalize_text(quote)
        norm_source = self.normalize_text(source_text)

        # Check for unclosed syntax delimiters (truncated statements)
        if self._has_unclosed_delimiters(norm_quote):
            res = QuoteVerificationResult(
                is_valid=False,
                token_overlap_ratio=0.5,
                normalized_quote=norm_quote,
                error_code=SlicerErrorCode.TRUNCATED_QUOTE,
                error_reason="Quote is truncated mid-line with unclosed delimiters",
                unmatched_tokens=[],
            )
            if strict:
                raise TruncatedLineError(res.error_reason or "Truncated quote")
            return res

        # 1. Exact substring fast path
        if norm_quote in norm_source:
            return QuoteVerificationResult(
                is_valid=True,
                token_overlap_ratio=1.0,
                normalized_quote=norm_quote,
                matched_source_segment=norm_quote,
            )

        # 2. Fuzzy token sequence alignment
        q_tokens = self.tokenize(norm_quote)
        s_tokens = self.tokenize(norm_source)

        if not q_tokens:
            raise QuoteVerificationError(
                SlicerErrorCode.QUOTE_MISMATCH, "No tokens extracted from quote"
            )

        # Find best matching window in source tokens
        matcher = difflib.SequenceMatcher(None, q_tokens, s_tokens)
        match = matcher.find_longest_match(0, len(q_tokens), 0, len(s_tokens))

        if match.size == 0:
            return QuoteVerificationResult(
                is_valid=False,
                token_overlap_ratio=0.0,
                normalized_quote=norm_quote,
                error_code=SlicerErrorCode.HALLUCINATED_QUOTE,
                error_reason="Zero matching tokens between quote and source text",
                unmatched_tokens=q_tokens,
            )

        # Extract localized window in source tokens
        window_start = max(0, match.b - 5)
        window_end = min(len(s_tokens), match.b + len(q_tokens) + 5)
        source_window = s_tokens[window_start:window_end]

        local_matcher = difflib.SequenceMatcher(None, q_tokens, source_window)
        matching_tokens_count = sum(block.size for block in local_matcher.get_matching_blocks())
        overlap_ratio = round(matching_tokens_count / len(q_tokens), 6)

        is_valid = overlap_ratio >= min_overlap_ratio

        error_code = None
        error_reason = None
        unmatched: list[str] = []

        if not is_valid:
            unmatched = [t for t in q_tokens if t not in source_window]
            # Check for truncation: quote is a prefix of window but missing tail
            if len(q_tokens) < len(source_window) and q_tokens == source_window[: len(q_tokens)]:
                error_code = SlicerErrorCode.TRUNCATED_QUOTE
                error_reason = "Quote is truncated mid-line"
                if strict:
                    raise TruncatedLineError(error_reason)
            else:
                error_code = SlicerErrorCode.HALLUCINATED_QUOTE
                error_reason = (
                    f"Quote failed verification (overlap {overlap_ratio:.4f} < {min_overlap_ratio})"
                )
                if strict:
                    raise HallucinatedQuoteError(error_reason)

        return QuoteVerificationResult(
            is_valid=is_valid,
            token_overlap_ratio=overlap_ratio,
            normalized_quote=norm_quote,
            matched_source_segment=" ".join(source_window) if match.size > 0 else None,
            error_code=error_code,
            error_reason=error_reason,
            unmatched_tokens=unmatched,
        )

    def verify_doc_citation(
        self,
        citation: DocCitationSpan,
        chunk_content: str,
        min_overlap_ratio: float = MIN_QUOTE_OVERLAP_RATIO,
    ) -> QuoteVerificationResult:
        """Verifies a DocCitationSpan against verbatim chunk content."""
        return self.verify_quote(
            quote=citation.verbatim_quote,
            source_text=chunk_content,
            min_overlap_ratio=min_overlap_ratio,
        )


__all__ = [
    "DeterministicSlicer",
    "HallucinatedQuoteError",
    "InvalidLineBoundsError",
    "OffByOneBoundsError",
    "QuoteVerificationError",
    "QuoteVerificationResult",
    "TruncatedLineError",
]
