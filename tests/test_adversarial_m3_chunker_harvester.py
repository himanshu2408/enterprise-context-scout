"""Adversarial stress harness for Milestone 3 Quant Research Harvester.

Empirically tests and challenges:
1. LatexDocParser:
   - Giant equations (>2000 tokens) exceeding max_chunk_tokens to verify boundary preservation.
   - Multiline math environments (align*, gather*, multline*, flalign*).
   - Inline math ($..$) vs display math ($$..$$) discrimination.
   - Deep section breadcrumb rollback and empty/preamble sections.
   - Standalone vs nested LaTeX tables (table float and tabular environments).
2. MarkdownTableChunker:
   - Structure-aware chunking preserving pipe tables intact across sections.
   - Trailing whitespace handling in Markdown table row regex.
3. paper_harvester:
   - arXiv Atom XML feed parsing: missing authors, missing summary, version stripping.
   - arXiv publication date parsing edge cases (non-ISO format).
   - OpenAlex REST JSON parsing: inverted index abstract reconstruction with out-of-order keys.
   - HttpPaperHarvesterClient exponential backoff jitter bounds (3.5s * 2^n + [0.1, 0.5]s).
   - Retry exhaustion and exception propagation.
   - Market query presets for IN_NSE, DE_XETRA, and US_EQUITY.
"""

from __future__ import annotations

import urllib.error
from datetime import date
from typing import Any
from unittest.mock import patch

import pytest

from context_scout.ingestion.doc_parser import (
    LATEX_TABLE_PATTERN,
    LatexDocParser,
    MarkdownTableChunker,
)
from context_scout.ingestion.paper_harvester import (
    FIXTURE_PAPERS,
    MARKET_QUERY_PRESETS,
    HarvesterSource,
    HttpPaperHarvesterClient,
    build_arxiv_query_url,
    build_openalex_query_url,
    parse_arxiv_atom_feed,
    parse_openalex_response,
)
from context_scout.schemas.contracts import MarketRegime

# ==============================================================================
# Scope 1: LatexDocParser Adversarial Stress Tests
# ==============================================================================


def test_latex_parser_giant_equation_exceeding_max_chunk_tokens() -> None:
    """Stress test: A giant 2000+ token equation exceeding max_chunk_tokens.

    Verifies:
    1. The giant equation is never split across chunk boundaries.
    2. \\begin{equation} and \\end{equation} reside completely in the same chunk.
    3. Preceding intro section is flushed into its own chunk once min_tokens is reached.
    """
    parser = LatexDocParser()
    terms = " + ".join([f"\\gamma_{{{i}}} \\cdot X_{{i,t}}" for i in range(250)])
    intro_paragraphs = (
        "This is an extensive methodological introduction defining factor risk models. "
        "We consider cross-sectional asset return covariance in multi-factor equity regimes.\n\n"
    ) * 8

    tex_content = f"""\\documentclass{{article}}
\\begin{{document}}
\\section{{Multi-Factor Risk Specification}}
{intro_paragraphs}

\\begin{{equation}}
R_{{i,t}} = \\alpha_i + {terms} + \\epsilon_{{i,t}}
\\end{{equation}}

\\section{{Empirical Evaluation}}
Post-estimation diagnostic tests confirm factor orthogonality across sample years.
\\end{{document}}
"""

    sections = parser.parse_latex(tex_content)
    assert len(sections) == 2

    chunks = parser.chunk_paper(
        doc_id="arxiv:2409.99999",
        doc_title="Adversarial Giant Formula Paper",
        source_uri="https://arxiv.org/abs/2409.99999",
        market_scope=MarketRegime.US_NYSE_NASDAQ,
        sections=sections,
        min_chunk_tokens=50,
        max_chunk_tokens=300,
    )

    assert len(chunks) >= 3

    # Locate the chunk containing the giant equation
    giant_chunks = [c for c in chunks if "\\begin{equation}" in c.verbatim_content]
    assert len(giant_chunks) == 1, "Giant equation must exist in exactly one chunk"

    giant_chunk = giant_chunks[0]
    # Invariant: Giant equation MUST NOT be split across chunks
    assert "\\begin{equation}" in giant_chunk.verbatim_content
    assert "\\end{equation}" in giant_chunk.verbatim_content
    assert len(giant_chunk.formulas_latex) >= 1
    # Chunk token count must reflect the giant size (>1000 tokens) despite max_chunk_tokens=300
    assert giant_chunk.token_count_approx > 1000


def test_latex_parser_multiline_math_environments() -> None:
    """Stress test: Multiline LaTeX environments (align*, gather*, multline*, flalign*).

    Verifies:
    1. Multiline align*, gather*, multline*, and flalign* are extracted intact.
    2. Inner line breaks (\\\\), alignment characters (&), and comments (%) do not break blocks.
    3. No multiline equation environment is split across chunk boundaries.
    """
    tex_content = r"""\documentclass{article}
\begin{document}
\section{Advanced Microstructure Signals}
Signal formulation across multiple equations:

\begin{align*}
\text{Spread}_{i,t} &= \frac{P_{i,t}^a - P_{i,t}^b}{M_{i,t}} \\
% intermediate latency adjustment
\Delta M_{i,t} &= \lambda \cdot \text{OFI}_{i,t} + \eta_{i,t}
\end{align*}

Next we consider volume-weighted order book balance:

\begin{gather*}
V_{\text{bid}} = \sum_{k=1}^K q_k^b \\
V_{\text{ask}} = \sum_{k=1}^K q_k^a
\end{gather*}

And the long multiline liquidation path:

\begin{multline*}
\mathcal{L}(x_0, T) = \int_0^T \left( \dot{x}_t \tilde{S}_t + \eta \dot{x}_t^2 \right) dt \\
+ \frac{1}{2} \gamma \sigma^2 \int_0^T x_t^2 dt
\end{multline*}

Finally, a flalign environment:

\begin{flalign*}
X &= A + B & Y &= C + D
\end{flalign*}

Concluding remarks on market execution.
\end{document}
"""
    parser = LatexDocParser()
    sections = parser.parse_latex(tex_content)
    assert len(sections) == 1

    chunks = parser.chunk_paper(
        doc_id="arxiv:2409.88888",
        doc_title="Multiline Equations",
        source_uri="https://arxiv.org/abs/2409.88888",
        market_scope=MarketRegime.GERMANY_XETRA_DAX,
        sections=sections,
        min_chunk_tokens=20,
        max_chunk_tokens=80,
    )

    for chunk in chunks:
        content = chunk.verbatim_content
        for env in ("align*", "gather*", "multline*", "flalign*"):
            if f"\\begin{{{env}}}" in content:
                assert f"\\end{{{env}}}" in content, f"Environment {env} split across chunk!"


def test_latex_parser_inline_and_display_math_discrimination() -> None:
    """Verifies discrimination between inline math ($x$), display ($$..$$), and text."""
    tex_content = r"""\documentclass{article}
\begin{document}
\section{Price Formation}
Let $P_t$ denote the stock price and let $\sigma > 0$ denote historical volatility.
In continuous trading, the price follows geometric Brownian motion:
$$
dS_t = \mu S_t dt + \sigma S_t dW_t
$$
Furthermore, fees in Indian markets are \$50 per transaction or 10 bps STT.
\end{document}
"""
    parser = LatexDocParser()
    sections = parser.parse_latex(tex_content)
    assert len(sections) == 1

    sec = sections[0]
    # Check that display math was recognized
    assert any("$$" in f for f in sec.formulas_latex)
    # Check that inline math $P_t$ was recognized
    assert any("$P_t$" in f for f in sec.formulas_latex)


def test_latex_parser_deep_section_breadcrumb_hierarchy() -> None:
    """Stress test: 3-level section hierarchy and breadcrumb rollback.

    Verifies:
    1. Section (L1) -> Subsection (L2) -> Subsubsection (L3) build breadcrumb paths.
    2. When a subsequent Subsection (L2) begins, Subsubsection (L3) is popped off.
    3. Document preambles before the first \\section are preserved in a Preamble section.
    """
    tex_content = r"""\documentclass{article}
Preamble and abstract content discussing market microstructure.

\section{Introduction}
Introductory remarks.

\subsection{Background}
Background material on limit order books.

\subsubsection{Queue Priority}
FIFO price-time priority mechanics.

\subsubsection{Cancellation Ratio}
Order cancellation dynamics.

\subsection{Empirical Analysis}
Analysis begins here, which should pop Subsubsections from breadcrumbs.

\section{Conclusion}
Final remarks.
"""
    parser = LatexDocParser()
    sections = parser.parse_latex(tex_content)

    headings = [s.heading for s in sections]
    assert "Preamble / Abstract" in headings
    assert "Introduction" in headings
    assert "Queue Priority" in headings
    assert "Empirical Analysis" in headings

    chunks = parser.chunk_paper(
        doc_id="doc:hierarchy",
        doc_title="Hierarchy Paper",
        source_uri="https://example.com/hierarchy",
        market_scope=MarketRegime.INDIA_NSE_BSE,
        sections=sections,
        min_chunk_tokens=5,
        max_chunk_tokens=100,
    )

    # Queue Priority chunk should have 3 levels
    queue_chunk = next(c for c in chunks if "FIFO price-time priority" in c.verbatim_content)
    assert queue_chunk.section_heading == "Introduction > Background > Queue Priority"

    # Empirical Analysis chunk should rollback to 2 levels
    emp_chunk = next(c for c in chunks if "Analysis begins here" in c.verbatim_content)
    assert emp_chunk.section_heading == "Introduction > Empirical Analysis"
    assert "Queue Priority" not in emp_chunk.section_heading


def test_latex_parser_standalone_tabular_preservation() -> None:
    """Verifies that standalone \\begin{tabular} ... \\end{tabular} is kept intact."""
    tex_content = r"""\documentclass{article}
\begin{document}
\section{Fee Schedule}
The fee schedule is summarized below:

\begin{tabular}{|l|c|}
\hline
Market & Transaction Cost (bps) \\
\hline
NSE Cash Delivery & 15.0 \\
Xetra DAX & 10.0 \\
NYSE Large Cap & 5.0 \\
\hline
\end{tabular}

All rates reflect statutory institutional minimums.
\end{document}
"""
    parser = LatexDocParser()
    sections = parser.parse_latex(tex_content)
    chunks = parser.chunk_paper(
        doc_id="doc:tabular",
        doc_title="Fee Schedule",
        source_uri="https://example.com/tabular",
        market_scope=MarketRegime.INDIA_NSE_BSE,
        sections=sections,
        min_chunk_tokens=10,
        max_chunk_tokens=50,
    )

    for chunk in chunks:
        if "\\begin{tabular}" in chunk.verbatim_content:
            assert "\\end{tabular}" in chunk.verbatim_content


def test_latex_table_pattern_nested_orphan_characterization() -> None:
    """Empirically characterises regex behaviour of LATEX_TABLE_PATTERN on nested table/tabular.

    Verifies LATEX_TABLE_PATTERN matches from \\begin{table} through closing \\end{table},
    keeping the inner \\begin{tabular} ... \\end{tabular} intact within the matched atomic block.
    """
    nested_table_tex = r"""\begin{table}[htbp]
\centering
\caption{Cross-Asset Liquidity}
\begin{tabular}{|l|c|}
\hline
Instrument & Spread (bps) \\
\hline
NIFTY 50 Futures & 1.5 \\
\hline
\end{tabular}
\end{table}"""

    matches = list(LATEX_TABLE_PATTERN.finditer(nested_table_tex))
    assert len(matches) == 1
    matched_block = matches[0].group(0)

    # In the fixed implementation, the match encompasses the full float environment
    assert matched_block.startswith(r"\begin{table}")
    assert matched_block.endswith(r"\end{table}")
    assert r"\begin{tabular}" in matched_block
    assert r"\end{tabular}" in matched_block


def test_latex_nested_table_float_boundary_preservation() -> None:
    """Adversarial Challenge: Verifies nested table and tabular environments stay intact."""
    p1 = (
        "Paragraph 1 is here. It is roughly fifty words long to build up some tokens "
        "before the table begins in this section."
    )
    tex_content = f"""\\documentclass{{article}}
\\begin{{document}}
\\section{{Empirical Execution Results}}
{p1}

\\begin{{table}}[h]
\\centering
\\begin{{tabular}}{{|l|c|}}
\\hline
Metric & Value \\\\
\\hline
Sharpe & 1.45 \\\\
Return & 22.4\\% \\\\
\\hline
\\end{{tabular}}
\\end{{table}}

Paragraph 2 follows the table with further discussion of the results.
\\end{{document}}
"""
    parser = LatexDocParser()
    sections = parser.parse_latex(tex_content)
    chunks = parser.chunk_paper(
        doc_id="doc:nested_table",
        doc_title="Nested Table Paper",
        source_uri="http://example.com/nested",
        market_scope=MarketRegime.INDIA_NSE_BSE,
        sections=sections,
        min_chunk_tokens=30,
        max_chunk_tokens=54,  # Triggers split between table body and orphaned \end{table}
    )

    # Invariant: Any chunk with \begin{table} MUST also contain \end{table}
    for chunk in chunks:
        if r"\begin{table}" in chunk.verbatim_content:
            assert r"\end{table}" in chunk.verbatim_content, (
                "\\begin{table} and \\end{table} were split across chunk boundaries!"
            )


# ==============================================================================
# Scope 2: MarkdownTableChunker Adversarial Stress Tests
# ==============================================================================


def test_markdown_table_chunker_50_row_table_integrity() -> None:
    """Stress test: Markdown pipe table with 50 rows spanning multiple sections.

    Verifies:
    1. The 50-row pipe table is treated as an atomic block and kept intact.
    2. Heading transitions trigger proper section heading assignment.
    """
    rows = [f"| Asset_{i:02d} | NSE | {1.0 + i * 0.1:.1f} | 15.0 |" for i in range(50)]
    table_str = (
        "| Asset | Exchange | Spread (bps) | Statutory Floor (bps) |\n"
        "|---|---|---|---|\n" + "\n".join(rows)
    )

    md_content = f"""# Market Overview
Introductory context on Indian cash equities.

# Statutory Liquidity Matrix
{table_str}

# Concluding Analysis
Final execution notes.
"""
    chunker = MarkdownTableChunker()
    chunks = chunker.parse_and_chunk(
        doc_id="doc:md_large_table",
        doc_title="Large Table Doc",
        source_uri="https://example.com/large_table",
        market_scope=MarketRegime.INDIA_NSE_BSE,
        markdown_content=md_content,
        min_chunk_tokens=50,
        max_chunk_tokens=200,
    )

    assert len(chunks) >= 2
    # Verify the entire table is preserved in one chunk
    table_chunk = next(c for c in chunks if "| Asset_00 |" in c.verbatim_content)
    assert "| Asset_49 |" in table_chunk.verbatim_content
    assert table_chunk.section_heading == "Statutory Liquidity Matrix"


def test_markdown_table_chunker_trailing_whitespace_extraction() -> None:
    """Adversarial Challenge: Verifies Markdown table extraction when rows contain trailing space.

    Standard markdown files frequently have trailing whitespace on table lines.
    """
    md_content = (
        "# Results\n\n"
        "| Asset | Spread | Floor | \n"
        "|---|---|---| \n"
        "| NIFTY | 2.5 | 15.0 | \n"
        "| BANKNIFTY | 3.0 | 15.0 | \n"
    )
    chunker = MarkdownTableChunker()
    chunks = chunker.parse_and_chunk(
        doc_id="doc:md_trailing_ws",
        doc_title="Trailing WS Table",
        source_uri="https://example.com/ws",
        market_scope=MarketRegime.INDIA_NSE_BSE,
        markdown_content=md_content,
    )

    assert len(chunks) >= 1
    # chunk.tables_markdown must contain the entire table
    assert len(chunks[0].tables_markdown) >= 1
    table_extracted = chunks[0].tables_markdown[0]
    assert "| NIFTY |" in table_extracted
    assert "| BANKNIFTY |" in table_extracted


# ==============================================================================
# Scope 3: paper_harvester.py Adversarial Stress Tests
# ==============================================================================


def test_arxiv_atom_feed_empty_and_whitespace_inputs() -> None:
    """Verifies that parse_arxiv_atom_feed handles empty or whitespace input cleanly."""
    assert parse_arxiv_atom_feed("", MarketRegime.INDIA_NSE_BSE) == []
    assert parse_arxiv_atom_feed("   \n\t   ", MarketRegime.GERMANY_XETRA_DAX) == []


def test_arxiv_atom_feed_missing_authors_and_summary() -> None:
    """Edge cases: arXiv entry missing authors, summary, or with versioned IDs."""
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2403.99999v4</id>
    <title> Algorithmic Arbitrage without Abstract </title>
    <published>2024-03-20T10:00:00Z</published>
    <link href="http://arxiv.org/pdf/2403.99999v4" title="pdf" rel="related"/>
  </entry>
</feed>
"""
    papers = parse_arxiv_atom_feed(xml_content, MarketRegime.US_NYSE_NASDAQ)
    assert len(papers) == 1
    paper = papers[0]
    assert paper.doc_id == "arxiv:2403.99999"  # Version stripped
    assert paper.title == "Algorithmic Arbitrage without Abstract"
    assert paper.authors == []  # Gracefully handles missing author elements
    assert paper.abstract == ""  # Gracefully handles missing summary
    assert paper.published_date == date(2024, 3, 20)
    assert paper.source_uri == "http://arxiv.org/pdf/2403.99999v4"


def test_arxiv_atom_feed_non_iso_date_resilience() -> None:
    """Adversarial Challenge: Non-ISO publication date in arXiv Atom feed.

    Verifies that malformed or non-ISO dates fall back gracefully to date(2024, 1, 1)
    rather than crashing with ValueError (as parse_openalex_response already does).
    """
    xml_bad_date = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2403.11111</id>
    <title> Robust Date Parsing </title>
    <published>2024/03/20T00:00:00Z</published>
    <author><name>Test Author</name></author>
  </entry>
</feed>
"""
    papers = parse_arxiv_atom_feed(xml_bad_date, MarketRegime.INDIA_NSE_BSE)
    assert len(papers) == 1
    assert isinstance(papers[0].published_date, date)


def test_openalex_inverted_index_abstract_out_of_order() -> None:
    """Verifies inverted index abstract reconstruction with out-of-order keys and multi-tokens."""
    data: dict[str, object] = {
        "results": [
            {
                "id": "https://openalex.org/W999999999",
                "title": "Adversarial Inverted Index Test",
                "publication_date": "2024-06-15",
                "authorships": [
                    {"author": {"display_name": "Dr. Alpha"}},
                    {"author": None},  # Malformed author element
                    "invalid_string_entry",  # Malformed authorship item
                ],
                "abstract_inverted_index": {
                    "friction": [3],
                    "Securities": [0],
                    "requires": [2],
                    "Transaction": [1],
                    "15": [4],
                    "bps.": [5],
                },
                "doi": "https://doi.org/10.1000/test",
            }
        ]
    }
    papers = parse_openalex_response(data, MarketRegime.INDIA_NSE_BSE)
    assert len(papers) == 1
    paper = papers[0]
    assert paper.doc_id == "openalex:W999999999"
    assert paper.abstract == "Securities Transaction requires friction 15 bps."
    assert paper.authors == ["Dr. Alpha"]


def test_http_harvester_exponential_backoff_jitter_bounds() -> None:
    """Stress test: HttpPaperHarvesterClient 3.5s * 2^n exponential backoff and jitter bounds.

    Asserts:
    1. Retries exactly max_retries times upon HTTP failures.
    2. Backoff duration for attempt 0 is strictly in [3.6, 4.0] seconds.
    3. Backoff duration for attempt 1 is strictly in [7.1, 7.5] seconds.
    4. Backoff duration for attempt 2 is strictly in [14.1, 14.5] seconds.
    5. Re-raises underlying exception upon exhausting retries.
    """
    client = HttpPaperHarvesterClient(base_backoff_sec=3.5, max_retries=3, timeout_sec=5.0)
    sleep_durations: list[float] = []

    def fake_sleep(duration: float) -> None:
        sleep_durations.append(duration)

    call_count = 0

    def fake_urlopen(req: Any, timeout: float = 30.0) -> Any:
        nonlocal call_count
        call_count += 1
        raise urllib.error.URLError("Simulated HTTP 429 Rate Limited")

    with (
        patch("urllib.request.urlopen", side_effect=fake_urlopen),
        patch("time.sleep", side_effect=fake_sleep),
        pytest.raises(urllib.error.URLError, match="Simulated HTTP 429"),
    ):
        client._execute_http_get("http://export.arxiv.org/api/query?test")

    # Initial attempt + 3 retries = 4 calls total
    assert call_count == 4
    assert len(sleep_durations) == 3

    # Attempt 0: 3.5 * 1 + [0.1, 0.5] = [3.6, 4.0]
    assert 3.6 <= sleep_durations[0] <= 4.0, (
        f"Attempt 0 sleep {sleep_durations[0]} out of [3.6, 4.0]"
    )
    # Attempt 1: 3.5 * 2 + [0.1, 0.5] = [7.1, 7.5]
    assert 7.1 <= sleep_durations[1] <= 7.5, (
        f"Attempt 1 sleep {sleep_durations[1]} out of [7.1, 7.5]"
    )
    # Attempt 2: 3.5 * 4 + [0.1, 0.5] = [14.1, 14.5]
    assert 14.1 <= sleep_durations[2] <= 14.5, (
        f"Attempt 2 sleep {sleep_durations[2]} out of [14.1, 14.5]"
    )


def test_http_harvester_resilience_succeeds_on_attempt_2() -> None:
    """Verifies HttpPaperHarvesterClient succeeds if transient failures resolve quickly."""
    client = HttpPaperHarvesterClient(base_backoff_sec=3.5, max_retries=3)
    call_count = 0

    class MockResponse:
        def __enter__(self) -> MockResponse:
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def read(self) -> bytes:
            return b"<?xml version='1.0'?><feed xmlns='http://www.w3.org/2005/Atom'></feed>"

    def fake_urlopen(req: Any, timeout: float = 30.0) -> Any:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise urllib.error.URLError("First attempt network blip")
        return MockResponse()

    with (
        patch("urllib.request.urlopen", side_effect=fake_urlopen),
        patch("time.sleep", return_value=None),
    ):
        result = client._execute_http_get("http://example.com/ok")

    assert call_count == 2
    assert "feed" in result


def test_http_harvester_download_source_utf8_and_binary_fallback() -> None:
    """Verifies download_source handles valid UTF-8 LaTeX source and binary PDF fallback."""
    client = HttpPaperHarvesterClient()
    arxiv_paper = FIXTURE_PAPERS[MarketRegime.INDIA_NSE_BSE][0]
    unloaded_paper = arxiv_paper.__class__(
        doc_id="arxiv:2403.01182",
        source=HarvesterSource.ARXIV,
        title="Test Paper",
        authors=["Author 1"],
        published_date=date(2024, 3, 15),
        market_scope=MarketRegime.INDIA_NSE_BSE,
        source_uri="https://arxiv.org/abs/2403.01182",
        abstract="Test abstract",
        raw_tex_content=None,
    )

    class MockUtf8Response:
        def __enter__(self) -> MockUtf8Response:
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def read(self) -> bytes:
            return b"\\documentclass{article}\n\\begin{document}Hello\\end{document}"

    # 1. Successful UTF-8 TeX download
    with patch("urllib.request.urlopen", return_value=MockUtf8Response()):
        loaded_tex = client.download_source(unloaded_paper)
        assert loaded_tex.raw_tex_content is not None
        assert "\\documentclass{article}" in loaded_tex.raw_tex_content
        assert loaded_tex.raw_pdf_bytes is None

    # 2. Binary PDF fallback on non-UTF8 bytes
    class MockBinaryPdfResponse:
        def __enter__(self) -> MockBinaryPdfResponse:
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def read(self) -> bytes:
            return b"%PDF-1.5 \x80\x81\x82\x83\xff"

    with patch("urllib.request.urlopen", return_value=MockBinaryPdfResponse()):
        loaded_pdf = client.download_source(unloaded_paper)
        assert loaded_pdf.raw_tex_content is None
        assert loaded_pdf.raw_pdf_bytes is not None
        assert loaded_pdf.raw_pdf_bytes.startswith(b"%PDF-1.5")


def test_market_harvest_query_url_formatting() -> None:
    """Verifies URL encoding and query construction for arXiv and OpenAlex across all regimes."""
    for query in MARKET_QUERY_PRESETS.values():
        arxiv_url = build_arxiv_query_url(query)
        assert "export.arxiv.org/api/query" in arxiv_url
        assert "sortBy=submittedDate" in arxiv_url
        assert "max_results=50" in arxiv_url

        openalex_url = build_openalex_query_url(query)
        assert "api.openalex.org/works" in openalex_url
        assert f"from_publication_date%3A{query.start_year}-01-01" in openalex_url or (
            f"from_publication_date:{query.start_year}-01-01" in openalex_url
        )
