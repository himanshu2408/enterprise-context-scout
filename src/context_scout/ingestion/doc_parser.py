"""Lossless LaTeX and Markdown document chunker preserving formulas and tables (M3).

Guarantees zero math formulas or tabular matrices are split across chunk boundaries.
"""

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Final
from uuid import UUID, uuid4

from context_scout.db.repositories import DocChunkRecord
from context_scout.schemas.contracts import MarketRegime

_MATH_ENV_NAMES = (
    r"equation|equation\*|align|align\*|gather|gather\*|multline|multline\*|flalign|flalign\*"
)

# Atomic math block pattern: \begin{equation}...\end{equation}, $$...$$, $...$
MATH_BLOCK_PATTERN: Final[re.Pattern[str]] = re.compile(
    rf"(\\begin\{{(?:{_MATH_ENV_NAMES})\}}.*?\\end\{{(?:{_MATH_ENV_NAMES})\}}|"
    r"\$\$.*?\$\$|"
    r"(?<!\\)\$.+?(?<!\\)\$)",
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
    r"^\s*\|.+?\|\s*$",
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


class LatexDocParser:
    """Structure-aware LaTeX parser with lossless mathematical equation and table chunking."""

    def parse_latex(self, tex_content: str) -> list[ParsedSection]:
        """Parses LaTeX into structured sections while preserving mathematical formulas."""
        # Strip document wrapper if present
        body = tex_content
        doc_match = re.search(r"\\begin\{document\}(.*?)\\end\{document\}", tex_content, re.DOTALL)
        if doc_match:
            body = doc_match.group(1)

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
                                    r"(?:^\s*\|.+?\|\s*$\n?)+", chunk_text, re.MULTILINE
                                )
                            ],
                            token_count_approx=current_tokens,
                        )
                    )
                current_lines = []
                current_tokens = 0

        for line in lines:
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
                    table_text = "\n".join(table_lines)
                    table_tokens = estimate_tokens(table_text)
                    if (
                        current_tokens + table_tokens > max_chunk_tokens
                        and current_tokens >= min_chunk_tokens
                    ):
                        flush_chunk()
                    current_lines.extend(table_lines)
                    current_tokens += table_tokens
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

        if in_table:
            current_lines.extend(table_lines)
            current_tokens += estimate_tokens("\n".join(table_lines))

        flush_chunk()
        return chunks


__all__ = [
    "LATEX_SECTION_PATTERN",
    "LATEX_TABLE_PATTERN",
    "MATH_BLOCK_PATTERN",
    "MD_HEADING_PATTERN",
    "MD_TABLE_ROW_PATTERN",
    "DocumentChunk",
    "LatexDocParser",
    "MarkdownTableChunker",
    "ParsedSection",
    "estimate_tokens",
]
