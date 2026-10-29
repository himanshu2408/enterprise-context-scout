"""Unit tests for DeterministicSlicer and quote verification (VAL-02)."""

import uuid

import pytest

from context_scout.ast_engine.deterministic_slicer import DeterministicSlicer
from context_scout.ast_engine.models import (
    HallucinatedQuoteError,
    InvalidLineBoundsError,
    OffByOneBoundsError,
    QuoteVerificationError,
    SlicerErrorCode,
    TruncatedLineError,
)
from context_scout.db.repositories import CodeFileRecord
from context_scout.schemas.contracts import (
    CodeSlicePointer,
    DocCitationSpan,
    ModificationIntent,
)


@pytest.fixture
def slicer() -> DeterministicSlicer:
    return DeterministicSlicer()


@pytest.fixture
def sample_code_file() -> CodeFileRecord:
    content = (
        "def calculate_spread(bid: float, ask: float) -> float:\n"
        '    """Calculates relative bid-ask spread."""\n'
        "    if ask <= bid:\n"
        "        raise ValueError('Invalid bid/ask')\n"
        "    mid = (bid + ask) / 2.0\n"
        "    return (ask - bid) / mid\n"
    )
    return CodeFileRecord(
        file_id=uuid.uuid4(),
        domain_id="trading_code_scout",
        repo_name="trading-core",
        file_path="src/execution/spread.py",
        git_commit_sha="a1b2c3d4e5f67890",
        git_blob_sha="f0e1d2c3b4a59687",
        language="python",
        allowed_scopes=["public_eng"],
        raw_content=content,
        line_count=6,
    )


def test_slicer_verbatim_lines_with_exact_byte_indentation(
    slicer: DeterministicSlicer, sample_code_file: CodeFileRecord
) -> None:
    sliced = slicer.slice_lines(sample_code_file.raw_content, start_line=3, end_line=5)
    expected = (
        "   3 |     if ask <= bid:\n"
        "   4 |         raise ValueError('Invalid bid/ask')\n"
        "   5 |     mid = (bid + ask) / 2.0"
    )
    assert sliced == expected


def test_slicer_rejects_off_by_one_and_inverted_bounds(
    slicer: DeterministicSlicer, sample_code_file: CodeFileRecord
) -> None:
    # 0-indexed start line
    with pytest.raises(OffByOneBoundsError, match="0-indexed start_line detected"):
        slicer.slice_lines(sample_code_file.raw_content, start_line=0, end_line=3)

    # end_line == total_lines + 1 (classic fencepost off-by-one error)
    with pytest.raises(OffByOneBoundsError, match=r"exceeds file line count.*by 1"):
        slicer.slice_lines(sample_code_file.raw_content, start_line=1, end_line=7)

    # inverted bounds: end_line < start_line
    with pytest.raises(InvalidLineBoundsError, match="cannot be less than start_line"):
        slicer.slice_lines(sample_code_file.raw_content, start_line=4, end_line=2)

    # start_line > total_lines
    with pytest.raises(InvalidLineBoundsError, match="exceeds file line count"):
        slicer.slice_lines(sample_code_file.raw_content, start_line=10, end_line=12)

    # empty content
    with pytest.raises(InvalidLineBoundsError, match="Cannot slice empty file"):
        slicer.slice_lines("", start_line=1, end_line=1)


def test_slicer_hydrates_code_slice_pointer_with_git_provenance(
    slicer: DeterministicSlicer, sample_code_file: CodeFileRecord
) -> None:
    pointer = CodeSlicePointer(
        repo_name=sample_code_file.repo_name,
        file_path=sample_code_file.file_path,
        start_line=1,
        end_line=2,
        qualified_symbol="execution.spread.calculate_spread",
        selection_reason="Target function for spread adjustment",
        modification_intent=ModificationIntent.MODIFY_TARGET,
    )

    verbatim_slice = slicer.slice_pointer(pointer, sample_code_file, auto_expanded_by_ast=False)

    assert verbatim_slice.git_commit_sha == "a1b2c3d4e5f67890"
    assert verbatim_slice.git_blob_sha == "f0e1d2c3b4a59687"
    assert verbatim_slice.auto_expanded_by_ast is False
    assert "   1 | def calculate_spread" in verbatim_slice.verbatim_code_with_lines
    expected_docstring_line = '   2 |     """Calculates relative bid-ask spread."""'
    assert expected_docstring_line in verbatim_slice.verbatim_code_with_lines


def test_quote_verification_val02_exact_and_nfkc_normalization(
    slicer: DeterministicSlicer,
) -> None:
    source_text = (
        "Under the conservative friction model for IN_NSE, statutory charges "
        "and exchange fees demand a minimum round-trip friction of 15 bps."
    )
    quote = (
        "statutory charges\u00a0 and exchange fees demand a minimum round-trip friction of 15 bps."
    )

    result = slicer.verify_quote(quote, source_text)

    assert result.is_valid is True
    assert result.token_overlap_ratio == 1.0
    assert result.error_code is None


def test_quote_verification_val02_rejects_hallucinations_and_altered_numbers(
    slicer: DeterministicSlicer,
) -> None:
    source_text = "Sharpe ratio after 15 bps friction is 1.42 with max drawdown 8.5%."
    hallucinated_quote = "Sharpe ratio after 15 bps friction is 1.95 with max drawdown 8.5%."

    result = slicer.verify_quote(hallucinated_quote, source_text)

    assert result.is_valid is False
    assert result.token_overlap_ratio < 0.98
    assert result.error_code == SlicerErrorCode.HALLUCINATED_QUOTE

    # Verify strict mode raises HallucinatedQuoteError
    with pytest.raises(HallucinatedQuoteError):
        slicer.verify_quote(hallucinated_quote, source_text, strict=True)


def test_quote_verification_rejects_truncated_quotes(
    slicer: DeterministicSlicer,
) -> None:
    source_text = "def execute_market_order(order: Order, timeout_ms: int = 500) -> Receipt:"
    truncated_quote = "def execute_market_order(order: Order"

    result = slicer.verify_quote(truncated_quote, source_text)

    assert result.is_valid is False
    assert result.error_code == SlicerErrorCode.TRUNCATED_QUOTE

    # Verify strict mode raises TruncatedLineError
    with pytest.raises(TruncatedLineError):
        slicer.verify_quote(truncated_quote, source_text, strict=True)


def test_quote_verification_empty_quote_raises_error(
    slicer: DeterministicSlicer,
) -> None:
    with pytest.raises(QuoteVerificationError, match="Empty quote cannot be verified"):
        slicer.verify_quote("   \n\t", "Some source text")


def test_doc_citation_span_integration(
    slicer: DeterministicSlicer,
) -> None:
    chunk_content = (
        "Section 3.2 Market Friction Bounds: For European equities traded on DE_XETRA, "
        "spread compression typically allows friction estimation at 10 bps round-trip."
    )
    quote = "spread compression typically allows friction estimation at 10 bps round-trip."
    citation = DocCitationSpan(
        chunk_id=str(uuid.uuid4()),
        doc_id="arxiv:2401.09999",
        doc_title="European Equity Market Frictions",
        source_uri="https://arxiv.org/abs/2401.09999",
        section_heading="Market Friction Bounds",
        verbatim_quote=quote,
    )

    res = slicer.verify_doc_citation(citation, chunk_content)
    assert res.is_valid is True
    assert res.token_overlap_ratio == 1.0
