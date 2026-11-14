"""Lossless LaTeX and Markdown document chunker preserving formulas and tables (M3).

Guarantees zero math formulas or tabular matrices are split across chunk boundaries.
"""

from __future__ import annotations

import html
import io
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Final
from uuid import NAMESPACE_DNS, UUID, uuid4, uuid5

import pypdf.errors
from pypdf import PdfReader

from context_scout.db.repositories import DocChunkRecord
from context_scout.schemas.contracts import MarketRegime

logger = logging.getLogger(__name__)

_MATH_ENV_NAMES = (
    r"equation|equation\*|align|align\*|gather|gather\*|multline|multline\*|flalign|flalign\*|"
    r"eqnarray|eqnarray\*"
)

# Atomic math block pattern: \begin{equation}...\end{equation}, \[...\], $$...$$, $...$
MATH_BLOCK_PATTERN: Final[re.Pattern[str]] = re.compile(
    rf"(\\begin\{{(?:{_MATH_ENV_NAMES})\}}.*?\\end\{{(?:{_MATH_ENV_NAMES})\}}|"
    r"\\\[.*?\\\]|"
    r"\$\$.*?\$\$|"
    r"(?<!\\)\$(?!\d+\s)[^\$\n]+?(?<!\\)\$)",
    re.DOTALL,
)

# Atomic LaTeX tabular pattern
LATEX_TABLE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\\begin\{table\*?\}.*?\\end\{table\*?\}|"
    r"\\begin\{tabular\*?\}.*?\\end\{tabular\*?\}",
    re.DOTALL,
)

# LaTeX section headers
LATEX_SECTION_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\\(section|subsection|subsubsection)\*?\{([^}]+)\}",
    re.IGNORECASE,
)

# Markdown section headers
MD_HEADING_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(#{1,4})\s+(.+)$",
    re.MULTILINE,
)

# Markdown table row pattern
MD_TABLE_ROW_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^\s*\|?[^|\r\n]*(?:\|[^|\r\n]*)+\|?\s*$",
    re.MULTILINE,
)


def estimate_tokens(text: str) -> int:
    """Estimates LLM token count using word and symbol tokenization (approx 1.25 tokens/word)."""
    if not text.strip():
        return 0
    tokens = re.findall(r"\w+|[^\w\s]", text)
    return max(1, len(tokens))


@dataclass(frozen=True)
class ParsedSection:
    """A semantic section extracted from a research paper."""

    heading: str
    level: int  # 1 for section, 2 for subsection, 3 for subsubsection
    content: str
    formulas_latex: list[str] = field(default_factory=list)
    tables_markdown: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DocumentChunk:
    """A bounded document chunk with guaranteed formula and table preservation."""

    chunk_id: UUID
    doc_id: str
    doc_title: str
    source_uri: str
    market_scope: MarketRegime
    section_heading: str
    verbatim_content: str
    formulas_latex: list[str]
    tables_markdown: list[str]
    token_count_approx: int
    metadata_json: dict[str, object] = field(default_factory=dict)

    def to_doc_chunk_record(
        self,
        domain_id: str,
        contextual_header: str,
        allowed_scopes: list[str],
        embedding: list[float],
        authority_weight: float = 1.0,
        published_date: date | None = None,
    ) -> DocChunkRecord:
        """Converts chunk to database record for storage and search."""
        return DocChunkRecord(
            chunk_id=self.chunk_id,
            domain_id=domain_id,
            doc_id=self.doc_id,
            doc_title=self.doc_title,
            source_uri=self.source_uri,
            market_scope=self.market_scope,
            published_date=published_date,
            authority_weight=authority_weight,
            section_heading=self.section_heading,
            contextual_header=contextual_header,
            verbatim_content=self.verbatim_content,
            metadata_json=self.metadata_json,
            allowed_scopes=allowed_scopes,
            embedding=embedding,
        )


PDF_MAGIC_BYTES: Final[bytes] = b"%PDF-"
MIN_PDF_CHAR_COUNT: Final[int] = 3000
MIN_PDF_CHUNK_COUNT: Final[int] = 3
MIN_TEX_CHAR_COUNT: Final[int] = 3000


@dataclass(frozen=True)
class ContentValidationResult:
    """Validation result for extracted document sources."""

    is_valid: bool
    char_count: int
    chunk_count: int
    error_message: str | None = None


def validate_pdf_extraction(
    pdf_bytes: bytes,
    extracted_text: str,
    chunks: list[DocumentChunk],
    min_chars: int = MIN_PDF_CHAR_COUNT,
    min_chunks: int = MIN_PDF_CHUNK_COUNT,
) -> ContentValidationResult:
    """Validates that PDF bytes begin with %PDF-, yield >= min_chars and >= min_chunks."""
    if not pdf_bytes.startswith(PDF_MAGIC_BYTES):
        return ContentValidationResult(
            is_valid=False,
            char_count=len(extracted_text),
            chunk_count=len(chunks),
            error_message="Invalid PDF header: missing %PDF- magic bytes",
        )
    clean_text = extracted_text.strip()
    if len(clean_text) < min_chars:
        return ContentValidationResult(
            is_valid=False,
            char_count=len(clean_text),
            chunk_count=len(chunks),
            error_message=(
                f"Extracted characters ({len(clean_text)}) below minimum required ({min_chars})"
            ),
        )
    if len(chunks) < min_chunks:
        return ContentValidationResult(
            is_valid=False,
            char_count=len(clean_text),
            chunk_count=len(chunks),
            error_message=(
                f"Generated chunks ({len(chunks)}) below minimum required ({min_chunks})"
            ),
        )
    return ContentValidationResult(
        is_valid=True,
        char_count=len(clean_text),
        chunk_count=len(chunks),
        error_message=None,
    )


def validate_tex_source(
    tex_content: str,
    min_chars: int = MIN_TEX_CHAR_COUNT,
) -> ContentValidationResult:
    """Validates that LaTeX source yields >= min_chars characters."""
    clean_tex = tex_content.strip()
    if len(clean_tex) < min_chars:
        return ContentValidationResult(
            is_valid=False,
            char_count=len(clean_tex),
            chunk_count=0,
            error_message=(
                f"LaTeX content length ({len(clean_tex)}) below minimum required ({min_chars})"
            ),
        )
    return ContentValidationResult(
        is_valid=True,
        char_count=len(clean_tex),
        chunk_count=0,
        error_message=None,
    )


class LatexDocParser:
    """Structure-aware LaTeX parser with lossless mathematical equation and table chunking."""

    def validate_source(
        self,
        tex_content: str,
        min_chars: int = 3000,
    ) -> ContentValidationResult:
        """Validates that LaTeX source satisfies the minimum length requirement."""
        return validate_tex_source(tex_content, min_chars=min_chars)

    def parse_latex(self, tex_content: str) -> list[ParsedSection]:
        """Parses LaTeX into structured sections while preserving mathematical formulas."""
        # Strip document wrapper if present
        body = tex_content
        doc_match = re.search(r"\\begin\{document\}(.*)\\end\{document\}", tex_content, re.DOTALL)
        if doc_match:
            body = doc_match.group(1)
            # Strip any nested \begin{document} / \end{document} and duplicate \documentclass
            body = re.sub(r"\\documentclass\s*(?:\[[^\]]*\])?\s*\{[^}]*\}", "", body)
            body = re.sub(r"\\(?:begin|end)\s*\{document\}", "", body)

        matches = list(LATEX_SECTION_PATTERN.finditer(body))
        if not matches:
            formulas = MATH_BLOCK_PATTERN.findall(body)
            tables = LATEX_TABLE_PATTERN.findall(body)
            return [
                ParsedSection(
                    heading="Document Body",
                    level=1,
                    content=body.strip(),
                    formulas_latex=formulas,
                    tables_markdown=tables,
                )
            ]

        sections: list[ParsedSection] = []
        # Any text before the first section heading
        first_start = matches[0].start()
        if first_start > 0:
            preamble_content = body[:first_start].strip()
            if preamble_content:
                sections.append(
                    ParsedSection(
                        heading="Preamble / Abstract",
                        level=1,
                        content=preamble_content,
                        formulas_latex=MATH_BLOCK_PATTERN.findall(preamble_content),
                        tables_markdown=LATEX_TABLE_PATTERN.findall(preamble_content),
                    )
                )

        level_map = {"section": 1, "subsection": 2, "subsubsection": 3}
        for i, match in enumerate(matches):
            kind = match.group(1).lower()
            heading_text = match.group(2).strip()
            start_pos = match.end()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(body)
            sec_body = body[start_pos:end_pos].strip()

            formulas = MATH_BLOCK_PATTERN.findall(sec_body)
            tables = LATEX_TABLE_PATTERN.findall(sec_body)

            sections.append(
                ParsedSection(
                    heading=heading_text,
                    level=level_map.get(kind, 1),
                    content=sec_body,
                    formulas_latex=formulas,
                    tables_markdown=tables,
                )
            )

        return sections

    def _split_into_atomic_blocks(self, text: str) -> list[str]:
        """Splits content into atomic blocks: equations and tables remain unbroken."""
        blocks: list[str] = []
        pos = 0

        # Pattern matching either a LaTeX math environment or a LaTeX table
        combined_pattern = re.compile(
            rf"({MATH_BLOCK_PATTERN.pattern}|{LATEX_TABLE_PATTERN.pattern})",
            re.DOTALL,
        )

        for match in combined_pattern.finditer(text):
            preceding = text[pos : match.start()].strip()
            if preceding:
                # Split regular text paragraphs by double newline
                paragraphs = [p.strip() for p in preceding.split("\n\n") if p.strip()]
                blocks.extend(paragraphs)

            # Atomic block (equation or table)
            block = match.group(0).strip()
            if block:
                blocks.append(block)
            pos = match.end()

        trailing = text[pos:].strip()
        if trailing:
            paragraphs = [p.strip() for p in trailing.split("\n\n") if p.strip()]
            blocks.extend(paragraphs)

        return blocks

    def chunk_paper(
        self,
        doc_id: str,
        doc_title: str,
        source_uri: str,
        market_scope: MarketRegime,
        sections: list[ParsedSection],
        min_chunk_tokens: int = 400,
        max_chunk_tokens: int = 800,
    ) -> list[DocumentChunk]:
        """Chunks parsed sections into 400-800 token chunks without splitting formulas."""
        chunks: list[DocumentChunk] = []

        # Track hierarchical breadcrumb
        breadcrumb_stack: list[tuple[int, str]] = []

        for sec in sections:
            # Update breadcrumb stack
            while breadcrumb_stack and breadcrumb_stack[-1][0] >= sec.level:
                breadcrumb_stack.pop()
            breadcrumb_stack.append((sec.level, sec.heading))
            full_heading = " > ".join(h for _, h in breadcrumb_stack)

            blocks = self._split_into_atomic_blocks(sec.content)
            if not blocks:
                continue

            current_blocks: list[str] = []
            current_tokens = 0

            for block in blocks:
                block_tokens = estimate_tokens(block)

                # If adding this block exceeds max_chunk_tokens and we already have content
                if (
                    current_tokens + block_tokens > max_chunk_tokens
                    and current_tokens >= min_chunk_tokens
                ):
                    chunk_text = "\n\n".join(current_blocks)
                    chunks.append(
                        DocumentChunk(
                            chunk_id=uuid4(),
                            doc_id=doc_id,
                            doc_title=doc_title,
                            source_uri=source_uri,
                            market_scope=market_scope,
                            section_heading=full_heading,
                            verbatim_content=chunk_text,
                            formulas_latex=MATH_BLOCK_PATTERN.findall(chunk_text),
                            tables_markdown=LATEX_TABLE_PATTERN.findall(chunk_text),
                            token_count_approx=current_tokens,
                        )
                    )
                    current_blocks = [block]
                    current_tokens = block_tokens
                else:
                    current_blocks.append(block)
                    current_tokens += block_tokens

            if current_blocks:
                chunk_text = "\n\n".join(current_blocks)
                chunks.append(
                    DocumentChunk(
                        chunk_id=uuid4(),
                        doc_id=doc_id,
                        doc_title=doc_title,
                        source_uri=source_uri,
                        market_scope=market_scope,
                        section_heading=full_heading,
                        verbatim_content=chunk_text,
                        formulas_latex=MATH_BLOCK_PATTERN.findall(chunk_text),
                        tables_markdown=LATEX_TABLE_PATTERN.findall(chunk_text),
                        token_count_approx=current_tokens,
                    )
                )

        return chunks


class MarkdownTableChunker:
    """Structure-aware Markdown chunker keeping pipe tables intact."""

    def parse_and_chunk(
        self,
        doc_id: str,
        doc_title: str,
        source_uri: str,
        market_scope: MarketRegime,
        markdown_content: str,
        min_chunk_tokens: int = 400,
        max_chunk_tokens: int = 800,
    ) -> list[DocumentChunk]:
        """Chunks Markdown text ensuring pipe tables are never split."""
        lines = markdown_content.splitlines()
        chunks: list[DocumentChunk] = []

        current_heading = "Main"
        current_lines: list[str] = []
        current_tokens = 0
        in_table = False
        table_lines: list[str] = []
        in_fence = False
        fence_char = ""
        fence_len = 0
        fence_lines: list[str] = []

        def flush_chunk() -> None:
            nonlocal current_lines, current_tokens
            if current_lines:
                chunk_text = "\n".join(current_lines).strip()
                if chunk_text:
                    chunks.append(
                        DocumentChunk(
                            chunk_id=uuid4(),
                            doc_id=doc_id,
                            doc_title=doc_title,
                            source_uri=source_uri,
                            market_scope=market_scope,
                            section_heading=current_heading,
                            verbatim_content=chunk_text,
                            formulas_latex=MATH_BLOCK_PATTERN.findall(chunk_text),
                            tables_markdown=[
                                t.group(0)
                                for t in re.finditer(
                                    r"(?:^\s*\|?[^|\r\n]*(?:\|[^|\r\n]*)+\|?\s*$\n?)+",
                                    chunk_text,
                                    re.MULTILINE,
                                )
                            ],
                            token_count_approx=current_tokens,
                        )
                    )
                current_lines = []
                current_tokens = 0

        def commit_atomic_block(block_lines: list[str]) -> None:
            nonlocal current_tokens
            if not block_lines:
                return
            block_text = "\n".join(block_lines)
            block_tokens = estimate_tokens(block_text)
            if (
                current_tokens + block_tokens > max_chunk_tokens
                and current_tokens >= min_chunk_tokens
            ):
                flush_chunk()
            current_lines.extend(block_lines)
            current_tokens += block_tokens

        fence_open_re = re.compile(r"^\s*(`{3,}|~{3,})(.*)$")

        for line in lines:
            if in_fence:
                fence_lines.append(line)
                stripped_fence = line.strip()
                if len(stripped_fence) >= fence_len and set(stripped_fence) == {fence_char}:
                    commit_atomic_block(fence_lines)
                    fence_lines = []
                    in_fence = False
                    fence_char = ""
                    fence_len = 0
                continue

            fence_match = fence_open_re.match(line)
            if fence_match:
                if in_table:
                    commit_atomic_block(table_lines)
                    table_lines = []
                    in_table = False
                marker = fence_match.group(1)
                in_fence = True
                fence_char = marker[0]
                fence_len = len(marker)
                fence_lines = [line]
                continue

            # Check for Markdown heading
            head_match = MD_HEADING_PATTERN.match(line)
            if head_match:
                if in_table:
                    # Flush table first
                    current_lines.extend(table_lines)
                    current_tokens += estimate_tokens("\n".join(table_lines))
                    table_lines = []
                    in_table = False

                if current_tokens >= min_chunk_tokens:
                    flush_chunk()
                current_heading = head_match.group(2).strip()
                current_lines.append(line)
                current_tokens += estimate_tokens(line)
                continue

            # Check for Markdown table line
            if MD_TABLE_ROW_PATTERN.match(line):
                in_table = True
                table_lines.append(line)
                continue
            else:
                if in_table:
                    # Table ended, treat entire table as atomic block
                    commit_atomic_block(table_lines)
                    table_lines = []
                    in_table = False

                line_tokens = estimate_tokens(line)
                if (
                    current_tokens + line_tokens > max_chunk_tokens
                    and current_tokens >= min_chunk_tokens
                ):
                    flush_chunk()
                current_lines.append(line)
                current_tokens += line_tokens

        if in_fence and fence_lines:
            commit_atomic_block(fence_lines)

        if in_table:
            current_lines.extend(table_lines)
            current_tokens += estimate_tokens("\n".join(table_lines))

        flush_chunk()
        return chunks


# Standard PDF section headers pattern
PDF_SECTION_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(?:(?:[0-9]+\.?\s+)?(Abstract|Introduction|Related Work|Literature Review|"
    r"Data and Methodology|Methodology|Model Specification|Empirical Results|"
    r"Market Microstructure|Trading Strategy|Friction and Transaction Costs|"
    r"Performance Evaluation|Discussion|Conclusion|References))\b",
    re.IGNORECASE | re.MULTILINE,
)


class PdfDocParser:
    """Extracts structured sections and chunks from raw PDF bytes."""

    def extract_text(self, pdf_bytes: bytes) -> str:
        """Extracts plain text across all pages in the PDF, validating %PDF- magic bytes."""
        if not pdf_bytes or len(pdf_bytes) < 32 or not pdf_bytes.startswith(PDF_MAGIC_BYTES):
            return ""
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            pages_text: list[str] = []
            for page in reader.pages:
                text = (page.extract_text() or "").replace("\x00", "")
                if text.strip():
                    pages_text.append(text.strip())
            return "\n\n".join(pages_text)
        except (
            pypdf.errors.PyPdfError,
            OSError,
            ValueError,
            IndexError,
            KeyError,
            EOFError,
        ) as exc:
            logger.debug("Failed to extract text from PDF bytes: %s", exc)
            return ""

    def parse_pdf(self, pdf_bytes: bytes) -> list[ParsedSection]:
        """Extracts semantic sections from PDF bytes."""
        if not pdf_bytes or len(pdf_bytes) < 32:
            return []
        try:
            full_text = self.extract_text(pdf_bytes)
        except Exception as exc:
            logger.debug("Unexpected error during PDF parsing: %s", exc)
            return []

        if not full_text.strip():
            return []

        matches = list(PDF_SECTION_PATTERN.finditer(full_text))
        if not matches:
            return [
                ParsedSection(
                    heading="Document Body",
                    level=1,
                    content=full_text,
                    formulas_latex=[],
                    tables_markdown=[],
                )
            ]

        sections: list[ParsedSection] = []
        first_start = matches[0].start()
        if first_start > 0:
            preamble = full_text[:first_start].strip()
            if preamble:
                sections.append(
                    ParsedSection(
                        heading="Title / Header",
                        level=1,
                        content=preamble,
                    )
                )

        for i, match in enumerate(matches):
            heading = match.group(0).strip()
            start_pos = match.end()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
            body = full_text[start_pos:end_pos].strip()
            sections.append(
                ParsedSection(
                    heading=heading,
                    level=1,
                    content=body,
                )
            )

        return sections

    def chunk_pdf(
        self,
        doc_id: str,
        doc_title: str,
        source_uri: str,
        market_scope: MarketRegime,
        pdf_bytes: bytes,
        min_chunk_tokens: int = 400,
        max_chunk_tokens: int = 800,
    ) -> list[DocumentChunk]:
        """Parses and chunks PDF bytes into standard DocumentChunk objects."""
        if not pdf_bytes or len(pdf_bytes) < 32:
            return []
        try:
            sections = self.parse_pdf(pdf_bytes)
            if not sections:
                return []
            latex_parser = LatexDocParser()
            return latex_parser.chunk_paper(
                doc_id=doc_id,
                doc_title=doc_title,
                source_uri=source_uri,
                market_scope=market_scope,
                sections=sections,
                min_chunk_tokens=min_chunk_tokens,
                max_chunk_tokens=max_chunk_tokens,
            )
        except Exception as exc:
            logger.debug("Unexpected error during PDF chunking: %s", exc)
            return []

    def validate_and_chunk(
        self,
        doc_id: str,
        doc_title: str,
        source_uri: str,
        market_scope: MarketRegime,
        pdf_bytes: bytes,
        min_chars: int = MIN_PDF_CHAR_COUNT,
        min_chunks: int = MIN_PDF_CHUNK_COUNT,
        min_chunk_tokens: int = 400,
        max_chunk_tokens: int = 800,
    ) -> tuple[ContentValidationResult, list[DocumentChunk]]:
        """Extracts text, parses chunks, and validates PDF against quality thresholds."""
        text = self.extract_text(pdf_bytes)
        chunks = self.chunk_pdf(
            doc_id=doc_id,
            doc_title=doc_title,
            source_uri=source_uri,
            market_scope=market_scope,
            pdf_bytes=pdf_bytes,
            min_chunk_tokens=min_chunk_tokens,
            max_chunk_tokens=max_chunk_tokens,
        )
        val = validate_pdf_extraction(
            pdf_bytes=pdf_bytes,
            extracted_text=text,
            chunks=chunks,
            min_chars=min_chars,
            min_chunks=min_chunks,
        )
        return val, chunks


_SGML_ENTITIES: Final[dict[str, str]] = {
    "&mdash;": "--",
    "&ndash;": "-",
    "&nbsp;": " ",
    "&hellip;": "...",
    "&lsquo;": "'",
    "&rsquo;": "'",
    "&ldquo;": '"',
    "&rdquo;": '"',
    "&copy;": "(c)",
    "&reg;": "(R)",
    "&trade;": "(TM)",
    "&productname;": "PostgreSQL",
}

_PATTERN_LANG_BY_SUFFIX: Final[dict[str, str]] = {
    ".sql": "sql",
    ".tf": "hcl",
    ".thrift": "thrift",
}


def strip_mdx_boilerplate(raw_text: str) -> str:
    """Strips top-level MDX import/export JSX lines outside code fences while keeping content."""
    if not raw_text.strip():
        return ""
    out_lines: list[str] = []
    in_fence = False
    fence_char = ""
    fence_len = 0
    fence_re = re.compile(r"^\s*(`{3,}|~{3,})")

    for line in raw_text.splitlines():
        if in_fence:
            out_lines.append(line)
            s = line.strip()
            if len(s) >= fence_len and set(s) == {fence_char}:
                in_fence = False
            continue
        fm = fence_re.match(line)
        if fm:
            marker = fm.group(1)
            in_fence = True
            fence_char = marker[0]
            fence_len = len(marker)
            out_lines.append(line)
            continue
        stripped = line.strip()
        if re.match(
            r"^(?:import\s+.+\s+from\s+['\"].+['\"]|export\s+(?:const|default|function)\b)",
            stripped,
        ):
            continue
        out_lines.append(line)
    return "\n".join(out_lines).strip()


def parse_asciidoc_to_markdown(raw_text: str) -> str:
    """Normalizes AsciiDoc headings and source blocks into Markdown for atomic chunking."""
    if not raw_text.strip():
        return ""
    out_lines: list[str] = []
    in_block = False
    pending_lang = ""
    lines = raw_text.splitlines()
    for line in lines:
        stripped = line.strip()
        if not in_block:
            if stripped.startswith(":") and stripped.endswith(":"):
                continue
            attr_match = re.match(r"^:([A-Za-z0-9_-]+):\s+.*$", stripped)
            if attr_match and not out_lines:
                continue
            src_match = re.match(r"^\[source\s*,?\s*([A-Za-z0-9_+-]*)\]$", stripped, re.IGNORECASE)
            if src_match:
                pending_lang = src_match.group(1).strip().lower()
                continue
            if stripped == "----":
                in_block = True
                out_lines.append(f"```{pending_lang}".rstrip())
                pending_lang = ""
                continue
            head_match = re.match(r"^(={1,4})\s+(.+)$", stripped)
            if head_match:
                level = len(head_match.group(1))
                out_lines.append(f"{'#' * level} {head_match.group(2).strip()}")
                continue
            out_lines.append(line)
        else:
            if stripped == "----":
                in_block = False
                out_lines.append("```")
            else:
                out_lines.append(line)
    if in_block:
        out_lines.append("```")
    return "\n".join(out_lines).strip()


def _unescape_sgml_text(raw: str) -> str:
    text = raw
    for ent, rep in _SGML_ENTITIES.items():
        text = text.replace(ent, rep)
    text = html.unescape(text)
    return re.sub(r"&([a-zA-Z][a-zA-Z0-9._-]{1,30});", r"\1", text)


def extract_sgml_docbook_text(raw_text: str) -> str:
    """Converts DocBook/SGML documentation to clean Markdown prose with fenced SQL blocks."""
    if not raw_text.strip():
        return ""
    text = re.sub(r"<!--.*?-->", " ", raw_text, flags=re.DOTALL)

    code_blocks: list[str] = []

    def _stash_code(match: re.Match[str]) -> str:
        inner = re.sub(r"<[^>]+>", "", match.group(1))
        cleaned_code = _unescape_sgml_text(inner).strip("\n")
        token = f"\n__SGML_CODE_BLOCK_{len(code_blocks)}__\n"
        code_blocks.append(f"```sql\n{cleaned_code}\n```")
        return token

    text = re.sub(
        r"<(?:programlisting|screen|synopsis)\b[^>]*>(.*?)</(?:programlisting|screen|synopsis)>",
        _stash_code,
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    def _replace_heading(level: int) -> Callable[[re.Match[str]], str]:
        hashes = "#" * level

        def _inner(m: re.Match[str]) -> str:
            title_txt = _unescape_sgml_text(re.sub(r"<[^>]+>", " ", m.group(1))).strip()
            return f"\n{hashes} {title_txt}\n" if title_txt else "\n"

        return _inner

    text = re.sub(
        r"<(?:chapter|book|part|reference)\b[^>]*>\s*<title\b[^>]*>(.*?)</title>",
        _replace_heading(1),
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    text = re.sub(
        r"<(?:sect1|refentry|refsect1)\b[^>]*>(?:\s*<refmeta>.*?</refmeta>)?\s*<(?:title|refentrytitle)\b[^>]*>(.*?)</(?:title|refentrytitle)>",
        _replace_heading(2),
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    text = re.sub(
        r"<(?:sect2|sect3|refsect2|refsect3|formalpara)\b[^>]*>\s*<title\b[^>]*>(.*?)</title>",
        _replace_heading(3),
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    text = re.sub(
        r"<title\b[^>]*>(.*?)</title>",
        _replace_heading(3),
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    text = re.sub(r"<listitem\b[^>]*>", "\n- ", text, flags=re.IGNORECASE)
    text = re.sub(
        r"</(?:para|simpara|entry|row|listitem|sect[1-4]|chapter|refsect[1-3])>",
        "\n",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"<[^>]+>", " ", text)
    text = _unescape_sgml_text(text)

    for idx, block in enumerate(code_blocks):
        text = text.replace(f"__SGML_CODE_BLOCK_{idx}__", f"\n{block}\n")

    cleaned_lines: list[str] = []
    in_fence = False
    for ln in text.splitlines():
        if ln.strip().startswith("```"):
            in_fence = not in_fence
            cleaned_lines.append(ln.strip())
            continue
        if in_fence:
            cleaned_lines.append(ln.rstrip())
        else:
            collapsed = re.sub(r"[ \t]{2,}", " ", ln).strip()
            if collapsed or (cleaned_lines and cleaned_lines[-1] != ""):
                cleaned_lines.append(collapsed)
    return "\n".join(cleaned_lines).strip()


def parse_rst_to_markdown(raw_text: str) -> str:
    """Converts reStructuredText section underlines and code-block directives to Markdown."""
    if not raw_text.strip():
        return ""
    lines = raw_text.splitlines()
    out_lines: list[str] = []
    i = 0
    underline_levels = {"=": "#", "-": "##", "~": "###", "^": "####"}
    in_code_block = False
    code_indent = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if in_code_block:
            if not stripped:
                out_lines.append("")
                i += 1
                continue
            cur_indent = len(line) - len(line.lstrip(" "))
            if cur_indent >= code_indent and code_indent > 0:
                out_lines.append(line[code_indent:])
                i += 1
                continue
            in_code_block = False
            code_indent = 0
            out_lines.append("```")

        code_dir = re.match(
            r"^\.\.\s+(?:code-block|sourcecode)::\s*([A-Za-z0-9_+-]*)\s*$",
            stripped,
        )
        if code_dir:
            lang = code_dir.group(1).strip().lower()
            out_lines.append(f"```{lang}".rstrip())
            i += 1
            while i < len(lines) and (not lines[i].strip() or lines[i].lstrip().startswith(":")):
                i += 1
            if i < len(lines):
                code_indent = max(1, len(lines[i]) - len(lines[i].lstrip(" ")))
                in_code_block = True
            else:
                out_lines.append("```")
            continue

        if i + 1 < len(lines):
            nxt = lines[i + 1].strip()
            if (
                stripped
                and len(nxt) >= 3
                and len(set(nxt)) == 1
                and nxt[0] in underline_levels
                and len(nxt) >= min(len(stripped), 3)
            ):
                hashes = underline_levels[nxt[0]]
                out_lines.append(f"{hashes} {stripped}")
                i += 2
                continue

        out_lines.append(line)
        i += 1

    if in_code_block:
        out_lines.append("```")
    return "\n".join(out_lines).strip()


def _infer_title_and_heading(text: str, file_path: Path) -> tuple[str, str]:
    doc_title = file_path.stem.replace("_", " ").replace("-", " ").title()
    section_heading = "Overview"
    for ln in text.splitlines():
        s_ln = ln.strip()
        if s_ln.startswith("# "):
            doc_title = s_ln[2:].strip()
            break
        t_m = re.search(r"\\title\{([^\}]+)\}", s_ln)
        if t_m:
            doc_title = t_m.group(1).strip()
            break
    for ln in text.splitlines():
        s_ln = ln.strip()
        if s_ln.startswith("## "):
            section_heading = s_ln[3:].strip()
            break
        sec_m = re.search(r"\\section\*?\{([^\}]+)\}", s_ln)
        if sec_m:
            section_heading = sec_m.group(1).strip()
            break
    return doc_title, section_heading


def _infer_corpus_metadata(rel_path: str) -> dict[str, str]:
    parts = Path(rel_path).parts
    meta: dict[str, str] = {}
    for part in parts:
        if part in ("train", "eval_holdout"):
            meta["split"] = part
        elif part in (
            "STORAGE_LAKEHOUSE",
            "BATCH_MPP_ENGINES",
            "STREAMING_CDC",
            "MODELING_QUALITY_GOV",
        ):
            meta["sub_scope"] = part
    if "repos" in parts:
        idx = parts.index("repos")
        if idx + 1 < len(parts):
            meta["source_repo"] = parts[idx + 1]
    return meta


def chunk_custom_domain_document(
    *,
    file_path: Path,
    text: str,
    raw_bytes: bytes,
    domain_id: str,
    rel_path: str,
    blob_sha: str,
    min_chunk_tokens: int = 400,
    max_chunk_tokens: int = 800,
) -> list[DocumentChunk]:
    """Splits any supported custom knowledge document into bounded 400-800 token DocumentChunks."""
    suffix = file_path.suffix.lower()
    safe_slug = re.sub(r"[^A-Za-z0-9_.:/-]+", "_", rel_path)
    doc_id = f"{domain_id}:{safe_slug}"
    source_uri = f"urn:{domain_id}:{rel_path}"
    doc_title, default_heading = _infer_title_and_heading(text, file_path)
    chunks: list[DocumentChunk] = []

    if suffix == ".pdf" and raw_bytes.startswith(PDF_MAGIC_BYTES):
        chunks = PdfDocParser().chunk_pdf(
            doc_id=doc_id,
            doc_title=doc_title,
            source_uri=source_uri,
            market_scope=MarketRegime.GLOBAL_MULTI,
            pdf_bytes=raw_bytes,
            min_chunk_tokens=min_chunk_tokens,
            max_chunk_tokens=max_chunk_tokens,
        )
    elif suffix == ".tex":
        latex_parser = LatexDocParser()
        sections = latex_parser.parse_latex(text)
        chunks = latex_parser.chunk_paper(
            doc_id=doc_id,
            doc_title=doc_title,
            source_uri=source_uri,
            market_scope=MarketRegime.GLOBAL_MULTI,
            sections=sections,
            min_chunk_tokens=min_chunk_tokens,
            max_chunk_tokens=max_chunk_tokens,
        )
    elif suffix in (".md", ".mdx", ".adoc", ".sgml", ".rst"):
        if suffix == ".mdx":
            md_text = strip_mdx_boilerplate(text)
        elif suffix == ".adoc":
            md_text = parse_asciidoc_to_markdown(text)
        elif suffix == ".sgml":
            md_text = extract_sgml_docbook_text(text) if "<" in text and ">" in text else text
        elif suffix == ".rst":
            md_text = parse_rst_to_markdown(text)
        else:
            md_text = text
        doc_title, default_heading = _infer_title_and_heading(md_text, file_path)
        chunks = MarkdownTableChunker().parse_and_chunk(
            doc_id=doc_id,
            doc_title=doc_title,
            source_uri=source_uri,
            market_scope=MarketRegime.GLOBAL_MULTI,
            markdown_content=md_text,
            min_chunk_tokens=min_chunk_tokens,
            max_chunk_tokens=max_chunk_tokens,
        )
    elif suffix in _PATTERN_LANG_BY_SUFFIX:
        lang = _PATTERN_LANG_BY_SUFFIX[suffix]
        wrapped = f"# Reference Pattern: {rel_path}\n\n```{lang}\n{text.strip()}\n```"
        tok_count = estimate_tokens(wrapped)
        if tok_count <= max_chunk_tokens * 2:
            chunks = [
                DocumentChunk(
                    chunk_id=uuid4(),
                    doc_id=doc_id,
                    doc_title=f"{file_path.name} ({rel_path})",
                    source_uri=source_uri,
                    market_scope=MarketRegime.GLOBAL_MULTI,
                    section_heading=f"Reference Pattern ({lang})",
                    verbatim_content=wrapped,
                    formulas_latex=[],
                    tables_markdown=[],
                    token_count_approx=tok_count,
                )
            ]
        else:
            paragraphs = [p.strip() for p in re.split(r"\n{2,}", text.strip()) if p.strip()]
            grouped: list[str] = []
            buf: list[str] = []
            buf_toks = 0
            for para in paragraphs:
                p_toks = estimate_tokens(para)
                if buf and buf_toks + p_toks > max_chunk_tokens and buf_toks >= min_chunk_tokens:
                    grouped.append("\n\n".join(buf))
                    buf = [para]
                    buf_toks = p_toks
                else:
                    buf.append(para)
                    buf_toks += p_toks
            if buf:
                grouped.append("\n\n".join(buf))
            for part_idx, part_txt in enumerate(grouped, start=1):
                part_md = (
                    f"# Reference Pattern: {rel_path} (Part {part_idx})\n\n"
                    f"```{lang}\n{part_txt}\n```"
                )
                chunks.append(
                    DocumentChunk(
                        chunk_id=uuid4(),
                        doc_id=doc_id,
                        doc_title=f"{file_path.name} ({rel_path})",
                        source_uri=source_uri,
                        market_scope=MarketRegime.GLOBAL_MULTI,
                        section_heading=f"Reference Pattern ({lang}) #{part_idx}",
                        verbatim_content=part_md,
                        formulas_latex=[],
                        tables_markdown=[],
                        token_count_approx=estimate_tokens(part_md),
                    )
                )

    if not chunks and text.strip():
        clean_fallback = text.strip()
        if suffix == ".html":
            clean_fallback = html.unescape(re.sub(r"<[^>]+>", " ", clean_fallback))
            clean_fallback = re.sub(r"\s{2,}", " ", clean_fallback).strip()
        if clean_fallback:
            chunks = LatexDocParser().chunk_paper(
                doc_id=doc_id,
                doc_title=doc_title,
                source_uri=source_uri,
                market_scope=MarketRegime.GLOBAL_MULTI,
                sections=[
                    ParsedSection(
                        heading=default_heading,
                        level=1,
                        content=clean_fallback,
                    )
                ],
                min_chunk_tokens=min_chunk_tokens,
                max_chunk_tokens=max_chunk_tokens,
            )

    extra_meta = _infer_corpus_metadata(rel_path)
    total_chunks = len(chunks)
    finalized: list[DocumentChunk] = []
    for idx, ch in enumerate(chunks):
        det_uuid = uuid5(NAMESPACE_DNS, f"{domain_id}:doc:{rel_path}:{idx}")
        heading = (
            ch.section_heading
            if ch.section_heading and ch.section_heading != "Main"
            else default_heading
        )
        chunk_meta: dict[str, object] = {
            "filename": file_path.name,
            "rel_path": rel_path,
            "blob_sha": blob_sha,
            "chunk_index": idx,
            "total_chunks": total_chunks,
            "source_format": suffix.lstrip("."),
            **extra_meta,
        }
        finalized.append(
            DocumentChunk(
                chunk_id=det_uuid,
                doc_id=ch.doc_id,
                doc_title=ch.doc_title,
                source_uri=ch.source_uri,
                market_scope=ch.market_scope,
                section_heading=heading,
                verbatim_content=ch.verbatim_content,
                formulas_latex=ch.formulas_latex,
                tables_markdown=ch.tables_markdown,
                token_count_approx=ch.token_count_approx,
                metadata_json=chunk_meta,
            )
        )
    return finalized


__all__ = [
    "LATEX_SECTION_PATTERN",
    "LATEX_TABLE_PATTERN",
    "MATH_BLOCK_PATTERN",
    "MD_HEADING_PATTERN",
    "MD_TABLE_ROW_PATTERN",
    "MIN_PDF_CHAR_COUNT",
    "MIN_PDF_CHUNK_COUNT",
    "MIN_TEX_CHAR_COUNT",
    "PDF_MAGIC_BYTES",
    "PDF_SECTION_PATTERN",
    "ContentValidationResult",
    "DocumentChunk",
    "LatexDocParser",
    "MarkdownTableChunker",
    "ParsedSection",
    "PdfDocParser",
    "chunk_custom_domain_document",
    "estimate_tokens",
    "extract_sgml_docbook_text",
    "parse_asciidoc_to_markdown",
    "parse_rst_to_markdown",
    "strip_mdx_boilerplate",
    "validate_pdf_extraction",
    "validate_tex_source",
]
