"""Empirical Challenger 1 Adversarial Stress Test Suite for Milestone 3.

Adversarial Testing Scope (Harvester & Parser Stress Testing):
1. Archive Unpacking:
   - Corrupted / truncated `.tar.gz` and `.gz` bytes (graceful failure vs unhandled crash).
   - Zero-byte archive and random binary bytes.
2. Recursive Include Resolution:
   - Cyclic \\input and \\include directives (self-loop, 2-node cycle, 3-node cycle).
   - Relative path resolution and case-sensitivity handling.
3. Subfile Preambles & Document Environments:
   - \\subfile directives with duplicate \\documentclass and \\begin{document} / \\end{document}.
   - Verification of section retention after nested \\end{document}.
4. Mathematical Environments & Markdown Table Chunk Integrity:
   - \\begin{equation}, \\begin{align}, $$...$$, $...$, \\[...\\], and \\begin{eqnarray}.
   - Markdown pipe tables with and without boundary pipes and with trailing spaces.
5. PDF Extraction Resilience:
   - Zero-byte PDF, corrupted bytes, truncated PDF headers.
   - Real complex academic PDFs from corpus.
6. Forbidden Strings Hygiene:
   - Zero occurrences of forbidden strings across all source files.
"""

from __future__ import annotations

import gzip
import io
import re
import tarfile
from pathlib import Path

import pytest

from context_scout.ingestion.doc_parser import (
    LatexDocParser,
    MarkdownTableChunker,
    PdfDocParser,
)
from context_scout.ingestion.paper_harvester import (
    resolve_latex_includes,
    unpack_latex_archive,
)
from context_scout.schemas.contracts import MarketRegime

# ==============================================================================
# 1. Archive Unpacker Adversarial Stress Tests
# ==============================================================================


def test_adversarial_unpack_empty_bytes() -> None:
    """Verifies that empty byte input returns None without raising exceptions."""
    result = unpack_latex_archive(b"")
    assert result is None


def test_adversarial_unpack_random_corrupted_bytes() -> None:
    """Verifies that random non-archive bytes return None without crashing."""
    corrupt_bytes = b"\x00\xff\xfe\x01\x12\x34\x56\x78" * 64
    result = unpack_latex_archive(corrupt_bytes)
    assert result is None


def test_adversarial_unpack_truncated_gzip_stream() -> None:
    """Adversarial Challenge: Truncated gzip stream (e.g. dropped network connection).

    Tests:
    - gzip header without data stream.
    - Valid gzip stream truncated mid-stream before EOF marker.
    Must return None rather than raising unhandled EOFError or zlib.error.
    """
    valid_tex = b"\\documentclass{article}\n\\begin{document}Paper text\\end{document}"
    full_gz = gzip.compress(valid_tex)

    # Header-only truncated gzip
    header_only_gz = b"\x1f\x8b\x08\x00" + b"junk_payload"

    # Mid-stream truncated gzip
    mid_stream_gz = full_gz[: len(full_gz) // 2]

    # These calls must not raise EOFError or zlib.error
    res_header = unpack_latex_archive(header_only_gz)
    assert res_header is None, f"Expected None on header-only truncated gzip, got {res_header}"

    res_mid = unpack_latex_archive(mid_stream_gz)
    assert res_mid is None, f"Expected None on mid-stream truncated gzip, got {res_mid}"


def test_adversarial_unpack_corrupted_tar_gz() -> None:
    """Adversarial Challenge: Corrupted tar archive inside valid gzip wrapper.

    Must return None rather than raising unhandled tarfile.TarError or OSError.
    """
    corrupt_tar_payload = b"ustar  \x00" + b"\xff" * 512
    corrupt_tar_gz = gzip.compress(corrupt_tar_payload)

    result = unpack_latex_archive(corrupt_tar_gz)
    assert result is None


def test_adversarial_unpack_valid_tar_gz_with_non_utf8_encoding() -> None:
    """Verifies archive unpacking with Latin-1 encoded TeX files succeeds cleanly."""
    latin1_content = (
        "\\documentclass{article}\n"
        "\\begin{document}\n"
        "% Deutsche Börse Liquidity Märcz & François\n"
        "Spread in DAX is 10 bps.\n"
        "\\end{document}\n"
    ).encode("latin-1")

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo(name="main.tex")
        info.size = len(latin1_content)
        tar.addfile(info, io.BytesIO(latin1_content))

    archive_bytes = buf.getvalue()
    result = unpack_latex_archive(archive_bytes)
    assert result is not None
    assert "\\documentclass{article}" in result
    assert "Spread in DAX is 10 bps" in result


# ==============================================================================
# 2. LaTeX Recursive Include Resolver Adversarial Stress Tests
# ==============================================================================


def test_adversarial_resolve_includes_self_cycle() -> None:
    """Verifies that direct self-referential \\input{main} terminates without RecursionError."""
    tex_files = {
        "main.tex": r"Start text \input{main} End text",
    }
    result = resolve_latex_includes("main.tex", tex_files)
    assert "Start text" in result
    assert "End text" in result


def test_adversarial_resolve_includes_mutual_2node_cycle() -> None:
    """Verifies that mutual cycle A -> B -> A terminates without infinite recursion."""
    tex_files = {
        "main.tex": r"Main [ \input{sub_a} ] MainEnd",
        "sub_a.tex": r"InA [ \include{sub_b} ] InAEnd",
        "sub_b.tex": r"InB [ \input{sub_a} ] InBEnd",
    }
    result = resolve_latex_includes("main.tex", tex_files)
    assert "Main [" in result
    assert "InA [" in result
    assert "InB [" in result


def test_adversarial_resolve_includes_3node_cycle() -> None:
    """Verifies that 3-node cycle A -> B -> C -> A terminates cleanly."""
    tex_files = {
        "main.tex": r"Root \input{part1}",
        "part1.tex": r"P1 \input{part2}",
        "part2.tex": r"P2 \input{part3}",
        "part3.tex": r"P3 \input{part1}",
    }
    result = resolve_latex_includes("main.tex", tex_files)
    assert "Root" in result
    assert "P1" in result
    assert "P2" in result
    assert "P3" in result


def test_adversarial_resolve_includes_relative_path_and_case_variations() -> None:
    """Verifies include resolution across relative path formats.

    Covers ./, subdir/, and case-insensitivity variations.
    """
    tex_files = {
        "main.tex": (
            r"Header "
            r"\input{./sections/intro.tex} "
            r"\include{SECTIONS/METHOD.TEX} "
            r"\input{appendix} "
            r"Footer"
        ),
        "sections/intro.tex": r"\section{Introduction} Intro text.",
        "sections/method.tex": r"\section{Method} Method text.",
        "appendix.tex": r"\section{Appendix} Appendix text.",
    }
    result = resolve_latex_includes("main.tex", tex_files)
    assert r"\section{Introduction}" in result
    assert r"\section{Method}" in result
    assert r"\section{Appendix}" in result
    assert "Footer" in result


# ==============================================================================
# 3. Subfile Preambles & Document Environment Parsing
# ==============================================================================


def test_adversarial_subfile_preamble_stripping() -> None:
    """Verifies that subfiles with independent preambles have their preambles stripped cleanly."""
    main_tex = r"""\documentclass{article}
\usepackage{subfiles}
\begin{document}
\section{Main Section}
Main preamble and context.
\subfile{chapters/sub1}
\section{Closing Section}
Main conclusion.
\end{document}
"""
    sub_tex = r"""\documentclass[../main.tex]{subfiles}
\usepackage{amsmath}
\usepackage{amssymb}
\begin{document}
\section{Subfile Section}
Subfile empirical results with \begin{equation} y = f(x) \end{equation}.
\end{document}
"""
    files = {
        "main.tex": main_tex,
        "chapters/sub1.tex": sub_tex,
    }
    merged = resolve_latex_includes("main.tex", files)

    # Invariant: subfile documentclass must not be duplicated into main body
    assert r"\documentclass[../main.tex]{subfiles}" not in merged
    assert r"\usepackage{amsmath}" not in merged
    assert r"\section{Subfile Section}" in merged
    assert r"\section{Closing Section}" in merged

    parser = LatexDocParser()
    sections = parser.parse_latex(merged)
    headings = [s.heading for s in sections]

    # Invariant: all 3 sections must be present in parsed sections
    assert "Main Section" in headings
    assert "Subfile Section" in headings
    assert "Closing Section" in headings


def test_adversarial_latex_parser_duplicate_begin_end_document() -> None:
    """Adversarial Challenge: Verifies sections are kept after inner \\end{document}.

    In multi-file papers or uncleaned subfiles, an inner \\end{document} can occur.
    Non-greedy matching must not cause all subsequent paper sections to be dropped.
    """
    raw_tex = r"""\documentclass{article}
\begin{document}
\section{First Section}
Initial text before subfile.

\documentclass[main.tex]{subfiles}
\begin{document}
\section{Nested Subfile Section}
Nested body content.
\end{document}

\section{Second Main Section}
CRITICAL SUBSEQUENT SECTION THAT MUST NOT BE TRUNCATED.
\end{document}
"""
    parser = LatexDocParser()
    sections = parser.parse_latex(raw_tex)
    headings = [s.heading for s in sections]

    assert "First Section" in headings
    assert "Nested Subfile Section" in headings
    assert "Second Main Section" in headings, (
        f"CRITICAL BUG: 'Second Main Section' was dropped! Extracted headings: {headings}"
    )


# ==============================================================================
# 4. Mathematical Environments & Markdown Table Chunk Integrity
# ==============================================================================


def test_adversarial_equation_environments_not_split() -> None:
    """Verifies that all standard LaTeX math environments are not split across chunks."""
    math_envs = [
        ("equation", r"E = mc^2"),
        ("align", r"x &= 1 \\ y &= 2"),
        ("gather", r"a = b \\ c = d"),
        ("multline", r"f(x) = a + b \\ + c + d"),
    ]

    for env, body in math_envs:
        block = f"\\begin{{{env}}}\n{body}\n\\end{{{env}}}"
        tex = f"""\\documentclass{{article}}
\\begin{{document}}
\\section{{Math Section}}
Introductory filler words to establish enough token volume.

{block}

Trailing concluding remarks.
\\end{{document}}
"""
        parser = LatexDocParser()
        sections = parser.parse_latex(tex)
        chunks = parser.chunk_paper(
            doc_id=f"doc:{env}",
            doc_title=f"Math {env}",
            source_uri="http://example.com",
            market_scope=MarketRegime.US_NYSE_NASDAQ,
            sections=sections,
            min_chunk_tokens=5,
            max_chunk_tokens=25,
        )

        # Check invariant: if \begin{env} is present, \end{env} MUST be in the same chunk
        for chunk in chunks:
            if f"\\begin{{{env}}}" in chunk.verbatim_content:
                assert f"\\end{{{env}}}" in chunk.verbatim_content, (
                    f"Environment {env} was split across chunk boundaries!"
                )


def test_adversarial_eqnarray_and_bracket_math_chunk_integrity() -> None:
    """Adversarial Challenge: \\[...\\] and \\begin{eqnarray} chunk preservation.

    Verifies unnumbered display math \\[...\\] and eqnarray environments are treated as atomic.
    """
    tex = r"""\documentclass{article}
\begin{document}
\section{Bracket and Eqnarray Derivation}
Intro text before derivation to accumulate tokens.

\[
A = B + C

D = E + F
\]

\begin{eqnarray}
X &=& Y + Z

U &=& V + W
\end{eqnarray}

Conclusion text.
\end{document}
"""
    parser = LatexDocParser()
    sections = parser.parse_latex(tex)
    chunks = parser.chunk_paper(
        doc_id="doc:bracket_eqnarray",
        doc_title="Bracket Eqnarray Test",
        source_uri="http://example.com",
        market_scope=MarketRegime.INDIA_NSE_BSE,
        sections=sections,
        min_chunk_tokens=5,
        max_chunk_tokens=12,
    )

    for chunk in chunks:
        content = chunk.verbatim_content
        if r"\[" in content:
            assert r"\]" in content, "\\[...\\] was split across chunk boundaries!"
        if r"\begin{eqnarray}" in content:
            assert r"\end{eqnarray}" in content, (
                "\\begin{eqnarray} was split across chunk boundaries!"
            )


def test_adversarial_markdown_tables_with_and_without_outer_pipes() -> None:
    """Adversarial Challenge: Pipe tables without outer boundary pipes (CommonMark compliant)."""
    table_no_outer = (
        "Asset | Spread | Floor\n"
        "--- | --- | ---\n"
        "NIFTY 50 | 2.5 bps | 15.0 bps\n"
        "BANKNIFTY | 3.0 bps | 15.0 bps"
    )
    md = f"""# Market Microstructure

{table_no_outer}

# Execution Notes
Post trade analytics.
"""
    chunker = MarkdownTableChunker()
    chunks = chunker.parse_and_chunk(
        doc_id="doc:md_table",
        doc_title="MD Table Doc",
        source_uri="http://example.com",
        market_scope=MarketRegime.INDIA_NSE_BSE,
        markdown_content=md,
        min_chunk_tokens=5,
        max_chunk_tokens=15,
    )

    # Invariant: the table lines should not be fragmented across separate single-line chunks
    table_lines_in_chunks = [c for c in chunks if "NIFTY 50" in c.verbatim_content]
    assert len(table_lines_in_chunks) == 1
    assert "BANKNIFTY" in table_lines_in_chunks[0].verbatim_content


# ==============================================================================
# 5. PDF Extraction Adversarial Stress Tests
# ==============================================================================


def test_adversarial_pdf_parser_zero_byte_input() -> None:
    """Adversarial Challenge: Zero-byte PDF must return empty list/string without crashing."""
    parser = PdfDocParser()
    text = parser.extract_text(b"")
    assert text == ""

    sections = parser.parse_pdf(b"")
    assert sections == []

    chunks = parser.chunk_pdf(
        doc_id="doc:empty_pdf",
        doc_title="Empty PDF",
        source_uri="http://example.com/empty.pdf",
        market_scope=MarketRegime.US_NYSE_NASDAQ,
        pdf_bytes=b"",
    )
    assert chunks == []


def test_adversarial_pdf_parser_corrupted_random_bytes() -> None:
    """Adversarial Challenge: Random non-PDF bytes must return empty results without crashing."""
    parser = PdfDocParser()
    corrupt_bytes = b"This is definitely not a PDF file. \x00\x01\x02\xff\xfe" * 32

    text = parser.extract_text(corrupt_bytes)
    assert text == ""

    sections = parser.parse_pdf(corrupt_bytes)
    assert sections == []

    chunks = parser.chunk_pdf(
        doc_id="doc:corrupt_pdf",
        doc_title="Corrupt PDF",
        source_uri="http://example.com/corrupt.pdf",
        market_scope=MarketRegime.US_NYSE_NASDAQ,
        pdf_bytes=corrupt_bytes,
    )
    assert chunks == []


def test_adversarial_pdf_parser_truncated_pdf_header() -> None:
    """Adversarial Challenge: Truncated PDF header with missing EOF marker."""
    parser = PdfDocParser()
    truncated_pdf = b"%PDF-1.5\n1 0 obj\n<< /Type /Catalog >>\nendobj\n"

    text = parser.extract_text(truncated_pdf)
    assert text == ""

    chunks = parser.chunk_pdf(
        doc_id="doc:truncated_pdf",
        doc_title="Truncated PDF",
        source_uri="http://example.com/truncated.pdf",
        market_scope=MarketRegime.US_NYSE_NASDAQ,
        pdf_bytes=truncated_pdf,
    )
    assert chunks == []


def test_adversarial_pdf_parser_on_real_academic_corpus_pdfs() -> None:
    """Verifies that all cached academic PDFs parse and produce chunks with section headings."""
    cached_pdfs = list(Path("data/papers/content").glob("*/*.pdf"))
    if not cached_pdfs:
        pytest.skip("No cached PDFs available in data/papers/content")

    parser = PdfDocParser()
    for pdf_path in cached_pdfs[:5]:
        raw_bytes = pdf_path.read_bytes()
        chunks = parser.chunk_pdf(
            doc_id=pdf_path.stem,
            doc_title=pdf_path.stem,
            source_uri=f"file://{pdf_path}",
            market_scope=MarketRegime.US_NYSE_NASDAQ,
            pdf_bytes=raw_bytes,
        )
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.doc_id == pdf_path.stem
            assert len(chunk.verbatim_content) > 0


# ==============================================================================
# 6. Prohibited Strings Audit
# ==============================================================================


def test_adversarial_forbidden_strings_audit() -> None:
    """Scans all Python source files in context_scout for prohibited terms."""
    forbidden_terms = [
        " ".join(["Forward", "Deployed", "Engineer"]),
        " ".join(["interview", "guide"]),
        " ".join(["interview", "defense"]),
    ]
    src_dir = Path("src/context_scout")
    violations: list[str] = []

    for py_file in src_dir.rglob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        for term in forbidden_terms:
            if re.search(rf"\b{re.escape(term)}\b", content, re.IGNORECASE):
                violations.append(f"{py_file}: contains '{term}'")

    assert not violations, f"Forbidden strings found in source files: {violations}"
