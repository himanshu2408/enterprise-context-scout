"""Self-hosted IBM Docling SPI & zero-egress structured document extractor (M2-03)."""

from __future__ import annotations

import html as html_mod
import io
import ipaddress
import logging
import mimetypes
import re
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Protocol, runtime_checkable
from urllib.parse import urlparse

import httpx
import pypdf.errors
from pypdf import PdfReader

if TYPE_CHECKING:
    from context_scout.config import ScoutSettings

logger = logging.getLogger(__name__)

_MAX_ZIP_ENTRIES: Final[int] = 2000
_MAX_ZIP_UNCOMPRESSED_BYTES: Final[int] = 50 * 1024 * 1024  # 50 MiB
_XXE_MARKER_RE: Final[re.Pattern[bytes]] = re.compile(
    rb"<!(?:DOCTYPE|ENTITY)\b",
    re.IGNORECASE,
)
_MD_HEADING_RE: Final[re.Pattern[str]] = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_LATEX_BLOCK_RE: Final[re.Pattern[str]] = re.compile(
    r"(\$\$.*?\$\$|\\\[.*?\\\]|\\begin\{equation\*?\}.*?\\end\{equation\*?\})",
    re.DOTALL,
)
_MD_TABLE_BLOCK_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:^\s*\|[^\r\n]*\|\s*$(?:\r?\n)?){2,}",
    re.MULTILINE,
)


class UnsafeDocumentError(ValueError):
    """Raised when a document violates XXE, ZIP-bomb, or cloud-egress safety policies."""


@dataclass(frozen=True)
class ExtractedDiagram:
    """Structured diagram topology extracted from Confluence macros or diagram XML."""

    diagram_type: str
    title: str = ""
    raw_source: str = ""
    nodes: tuple[str, ...] = ()
    edges: tuple[tuple[str, str, str], ...] = ()
    summary_markdown: str = ""


@dataclass(frozen=True)
class ExtractedSection:
    """Structured heading section parsed from Docling or native document extractors."""

    heading: str
    level: int
    content: str
    tables_markdown: tuple[str, ...] = ()
    formulas_latex: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExtractedDocument:
    """Normalized structured representation returned by any DocExtractorSPI implementation."""

    filename: str
    backend: str
    markdown_text: str
    plain_text: str
    sections: tuple[ExtractedSection, ...] = ()
    tables_markdown: tuple[str, ...] = ()
    formulas_latex: tuple[str, ...] = ()
    diagrams: tuple[ExtractedDiagram, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class DocExtractorSPI(Protocol):
    """Service Provider Interface for self-hosted IBM Docling and native document parsers."""

    @property
    def backend_name(self) -> str: ...

    def extract(
        self,
        raw_bytes: bytes,
        *,
        filename: str,
        content_type: str | None = None,
    ) -> ExtractedDocument: ...


def validate_self_hosted_endpoint(endpoint_url: str, *, allow_external_egress: bool) -> str:
    """Ensures docling-serve endpoints remain local/VPC-private unless egress is permitted."""
    cleaned = endpoint_url.strip().rstrip("/")
    parsed = urlparse(cleaned)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise UnsafeDocumentError(f"Invalid Docling endpoint URL: {endpoint_url!r}")
    if allow_external_egress:
        return cleaned

    host = parsed.hostname.lower().rstrip(".")
    if host in {"localhost", "127.0.0.1", "::1"} or "." not in host:
        return cleaned
    if host.endswith((".local", ".internal", ".svc", ".cluster.local", ".nip.io")):
        return cleaned
    try:
        addr = ipaddress.ip_address(host)
    except ValueError as exc:
        raise UnsafeDocumentError(
            f"External cloud egress blocked for Docling endpoint '{endpoint_url}'. "
            "Use a private VPC / localhost / cluster-internal endpoint or enable "
            "allow_external_cloud_egress."
        ) from exc
    if addr.is_private or addr.is_loopback or addr.is_link_local:
        return cleaned
    raise UnsafeDocumentError(
        f"Public IP Docling endpoint '{endpoint_url}' blocked while "
        "allow_external_cloud_egress=False."
    )


def _safe_parse_xml(xml_bytes: bytes) -> ET.Element:
    """Parses XML bytes while blocking <!DOCTYPE> and <!ENTITY> XXE/billion-laughs attacks."""
    if _XXE_MARKER_RE.search(xml_bytes):
        raise UnsafeDocumentError("Blocked XML document containing <!DOCTYPE> or <!ENTITY>.")
    return ET.fromstring(xml_bytes)


def safe_parse_xml_text(xml_text: str) -> ET.Element:
    """Parses an XML string while rejecting <!DOCTYPE> and <!ENTITY> XXE payloads."""
    return _safe_parse_xml(xml_text.encode("utf-8", errors="replace"))


def _grid_to_markdown_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = max((len(r) for r in rows), default=0)
    if width == 0:
        return ""
    norm_rows: list[list[str]] = []
    for r in rows:
        padded = [re.sub(r"\s+", " ", cell.replace("|", "\\|")).strip() for cell in r] + [""] * (
            width - len(r)
        )
        norm_rows.append(padded[:width])
    if not any(any(c for c in r) for r in norm_rows):
        return ""
    header = norm_rows[0]
    sep = ["---"] * width
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for data_row in norm_rows[1:]:
        lines.append("| " + " | ".join(data_row) + " |")
    return "\n".join(lines)


def _split_markdown_sections(markdown_text: str) -> tuple[ExtractedSection, ...]:
    matches = list(_MD_HEADING_RE.finditer(markdown_text))
    if not matches:
        tables = tuple(m.group(0).strip() for m in _MD_TABLE_BLOCK_RE.finditer(markdown_text))
        formulas = tuple(_LATEX_BLOCK_RE.findall(markdown_text))
        return (
            ExtractedSection(
                heading="Overview",
                level=1,
                content=markdown_text.strip(),
                tables_markdown=tables,
                formulas_latex=formulas,
            ),
        )
    sections: list[ExtractedSection] = []
    if matches[0].start() > 0:
        preamble = markdown_text[: matches[0].start()].strip()
        if preamble:
            sections.append(
                ExtractedSection(
                    heading="Overview",
                    level=1,
                    content=preamble,
                    tables_markdown=tuple(
                        m.group(0).strip() for m in _MD_TABLE_BLOCK_RE.finditer(preamble)
                    ),
                    formulas_latex=tuple(_LATEX_BLOCK_RE.findall(preamble)),
                )
            )
    for idx, m in enumerate(matches):
        level = len(m.group(1))
        heading = m.group(2).strip()
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(markdown_text)
        body = markdown_text[start:end].strip()
        sections.append(
            ExtractedSection(
                heading=heading,
                level=level,
                content=body,
                tables_markdown=tuple(
                    tm.group(0).strip() for tm in _MD_TABLE_BLOCK_RE.finditer(body)
                ),
                formulas_latex=tuple(_LATEX_BLOCK_RE.findall(body)),
            )
        )
    return tuple(sections)


def _reconstruct_docling_json_tables(json_content: Any) -> list[str]:
    """Reconstructs GFM pipe tables from Docling JSON `tables` structures."""
    if not isinstance(json_content, dict):
        return []
    raw_tables = json_content.get("tables")
    if not isinstance(raw_tables, list):
        return []
    out: list[str] = []
    for tbl in raw_tables:
        if not isinstance(tbl, dict):
            continue
        data = tbl.get("data")
        if not isinstance(data, dict):
            continue
        raw_grid = data.get("grid")
        if isinstance(raw_grid, list) and raw_grid:
            grid_rows: list[list[str]] = []
            for row in raw_grid:
                if isinstance(row, list):
                    grid_rows.append(
                        [
                            str(cell.get("text", "") if isinstance(cell, dict) else cell).strip()
                            for cell in row
                        ]
                    )
            md_tbl = _grid_to_markdown_table(grid_rows)
            if md_tbl:
                out.append(md_tbl)
            continue

        raw_cells = data.get("table_cells")
        if isinstance(raw_cells, list) and raw_cells:
            num_rows = int(data.get("num_rows") or 0)
            num_cols = int(data.get("num_cols") or 0)
            for cell in raw_cells:
                if isinstance(cell, dict):
                    r_end = int(
                        cell.get("end_row_offset_idx")
                        or (int(cell.get("start_row_offset_idx") or 0) + 1)
                    )
                    c_end = int(
                        cell.get("end_col_offset_idx")
                        or (int(cell.get("start_col_offset_idx") or 0) + 1)
                    )
                    num_rows = max(num_rows, r_end)
                    num_cols = max(num_cols, c_end)
            if num_rows > 0 and num_cols > 0:
                matrix = [[""] * num_cols for _ in range(num_rows)]
                for cell in raw_cells:
                    if not isinstance(cell, dict):
                        continue
                    r0 = int(cell.get("start_row_offset_idx") or 0)
                    c0 = int(cell.get("start_col_offset_idx") or 0)
                    txt = str(cell.get("text") or "").strip()
                    if 0 <= r0 < num_rows and 0 <= c0 < num_cols:
                        matrix[r0][c0] = txt
                md_tbl = _grid_to_markdown_table(matrix)
                if md_tbl:
                    out.append(md_tbl)
    return out


class _HtmlTableParser(HTMLParser):
    """Parses HTML headings, paragraphs, list items, and colspan/rowspan tables cleanly."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[str] = []
        self.tables: list[str] = []
        self._in_heading: int | None = None
        self._heading_buf: list[str] = []
        self._text_buf: list[str] = []
        self._in_table = False
        self._cur_rows: list[list[tuple[str, int, int]]] = []
        self._cur_row: list[tuple[str, int, int]] = []
        self._in_cell = False
        self._cell_buf: list[str] = []
        self._cell_colspan = 1
        self._cell_rowspan = 1
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = tag.lower()
        if t in {"script", "style"}:
            self._skip_depth += 1
            return
        if self._skip_depth > 0:
            return
        attr_map = {k.lower(): (v or "") for k, v in attrs}
        if t in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._flush_text()
            self._in_heading = int(t[1])
            self._heading_buf = []
        elif t == "table":
            self._flush_text()
            self._in_table = True
            self._cur_rows = []
        elif t == "tr" and self._in_table:
            self._cur_row = []
        elif t in {"td", "th"} and self._in_table:
            self._in_cell = True
            self._cell_buf = []
            try:
                self._cell_colspan = max(1, int(attr_map.get("colspan") or "1"))
            except ValueError:
                self._cell_colspan = 1
            try:
                self._cell_rowspan = max(1, int(attr_map.get("rowspan") or "1"))
            except ValueError:
                self._cell_rowspan = 1
        elif t == "li" and not self._in_table:
            self._flush_text()
            self._text_buf.append("- ")
        elif t in {"br", "p", "div"} and not self._in_table and not self._in_cell:
            if self._text_buf:
                self._text_buf.append("\n")

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in {"script", "style"} and self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if self._skip_depth > 0:
            return
        if t in {"h1", "h2", "h3", "h4", "h5", "h6"} and self._in_heading is not None:
            heading_txt = re.sub(r"\s+", " ", "".join(self._heading_buf)).strip()
            if heading_txt:
                self.blocks.append(f"{'#' * self._in_heading} {heading_txt}")
            self._in_heading = None
            self._heading_buf = []
        elif t in {"td", "th"} and self._in_cell:
            cell_txt = re.sub(r"\s+", " ", "".join(self._cell_buf)).strip()
            self._cur_row.append((cell_txt, self._cell_colspan, self._cell_rowspan))
            self._in_cell = False
            self._cell_buf = []
        elif t == "tr" and self._in_table:
            if self._cur_row:
                self._cur_rows.append(self._cur_row)
            self._cur_row = []
        elif t == "table" and self._in_table:
            self._in_table = False
            md_tbl = self._expand_span_table(self._cur_rows)
            if md_tbl:
                self.tables.append(md_tbl)
                self.blocks.append(md_tbl)
            self._cur_rows = []
        elif t in {"p", "div", "li"} and not self._in_table:
            self._flush_text()

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        if self._in_heading is not None:
            self._heading_buf.append(data)
        elif self._in_cell:
            self._cell_buf.append(data)
        elif not self._in_table:
            self._text_buf.append(data)

    def _flush_text(self) -> None:
        raw = "".join(self._text_buf)
        self._text_buf = []
        cleaned_lines = [re.sub(r"[ \t]{2,}", " ", ln).strip() for ln in raw.splitlines()]
        joined = "\n".join(ln for ln in cleaned_lines if ln).strip()
        if joined:
            self.blocks.append(joined)

    @staticmethod
    def _expand_span_table(raw_rows: list[list[tuple[str, int, int]]]) -> str:
        if not raw_rows:
            return ""
        grid: list[list[str | None]] = []
        for r_idx, row in enumerate(raw_rows):
            while len(grid) <= r_idx:
                grid.append([])
            c_idx = 0
            for cell_txt, colspan, rowspan in row:
                while c_idx < len(grid[r_idx]) and grid[r_idx][c_idx] is not None:
                    c_idx += 1
                for dr in range(rowspan):
                    rr = r_idx + dr
                    while len(grid) <= rr:
                        grid.append([])
                    while len(grid[rr]) < c_idx + colspan:
                        grid[rr].append(None)
                    for dc in range(colspan):
                        grid[rr][c_idx + dc] = cell_txt if (dr == 0 and dc == 0) else ""
                c_idx += colspan
        str_rows = [[c or "" for c in row] for row in grid]
        return _grid_to_markdown_table(str_rows)

    def finalize(self) -> tuple[str, tuple[str, ...]]:
        self._flush_text()
        return "\n\n".join(self.blocks).strip(), tuple(self.tables)


def html_to_structured_markdown(raw_html: str) -> tuple[str, tuple[str, ...]]:
    """Converts HTML fragments with colspan/rowspan tables into Markdown and pipe tables."""
    parser = _HtmlTableParser()
    parser.feed(raw_html)
    parser.close()
    return parser.finalize()


class DoclingServeHttpExtractor:
    """Self-hosted IBM `docling-serve` HTTP client implementing DocExtractorSPI."""

    def __init__(
        self,
        endpoint_url: str,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 60.0,
        allow_external_egress: bool = False,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.endpoint_url = validate_self_hosted_endpoint(
            endpoint_url, allow_external_egress=allow_external_egress
        )
        headers: dict[str, str] = {"Accept": "application/json"}
        if api_key:
            headers["X-Api-Key"] = api_key
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(timeout=timeout_seconds, headers=headers)

    @property
    def backend_name(self) -> str:
        return "docling-serve"

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def extract(
        self,
        raw_bytes: bytes,
        *,
        filename: str,
        content_type: str | None = None,
    ) -> ExtractedDocument:
        mime = content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
        url = f"{self.endpoint_url}/v1/convert/file"
        resp = self._client.post(
            url,
            files={"files": (filename, raw_bytes, mime)},
            data={
                "to_formats": ["md", "json"],
                "do_table_structure": "true",
            },
        )
        resp.raise_for_status()
        payload: Any = resp.json()
        if not isinstance(payload, dict):
            raise ValueError("Unexpected non-dict response from docling-serve.")
        doc_obj = payload.get("document") if isinstance(payload.get("document"), dict) else payload
        assert isinstance(doc_obj, dict)

        md_text = str(doc_obj.get("md_content") or doc_obj.get("markdown") or "").strip()
        json_content = doc_obj.get("json_content")
        json_tables = _reconstruct_docling_json_tables(json_content)
        md_tables = [m.group(0).strip() for m in _MD_TABLE_BLOCK_RE.finditer(md_text)]
        all_tables = tuple(dict.fromkeys([*md_tables, *json_tables]))

        if not md_text and json_tables:
            md_text = "\n\n".join(json_tables)
        elif json_tables:
            for tbl in json_tables:
                if tbl not in md_text:
                    md_text = f"{md_text}\n\n{tbl}".strip()

        sections = _split_markdown_sections(md_text)
        formulas = tuple(_LATEX_BLOCK_RE.findall(md_text))
        return ExtractedDocument(
            filename=filename,
            backend=self.backend_name,
            markdown_text=md_text,
            plain_text=md_text,
            sections=sections,
            tables_markdown=all_tables,
            formulas_latex=formulas,
            metadata={"status": str(payload.get("status") or "success")},
        )


class InProcessDoclingExtractor:
    """In-process IBM `docling.document_converter.DocumentConverter` wrapper."""

    def __init__(self) -> None:
        self._converter: Any | None = None

    @property
    def backend_name(self) -> str:
        return "docling-python"

    @staticmethod
    def is_available() -> bool:
        try:
            import docling.document_converter  # noqa: F401
        except ImportError:
            return False
        return True

    def extract(
        self,
        raw_bytes: bytes,
        *,
        filename: str,
        content_type: str | None = None,
    ) -> ExtractedDocument:
        _ = content_type
        from docling.document_converter import DocumentConverter

        if self._converter is None:
            self._converter = DocumentConverter()
        suffix = Path(filename).suffix or ".bin"
        with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
            tmp.write(raw_bytes)
            tmp.flush()
            result = self._converter.convert(tmp.name)
        doc = result.document
        md_text = str(doc.export_to_markdown() or "").strip()
        json_content = doc.export_to_dict() if hasattr(doc, "export_to_dict") else {}
        json_tables = _reconstruct_docling_json_tables(json_content)
        md_tables = [m.group(0).strip() for m in _MD_TABLE_BLOCK_RE.finditer(md_text)]
        all_tables = tuple(dict.fromkeys([*md_tables, *json_tables]))
        return ExtractedDocument(
            filename=filename,
            backend=self.backend_name,
            markdown_text=md_text,
            plain_text=md_text,
            sections=_split_markdown_sections(md_text),
            tables_markdown=all_tables,
            formulas_latex=tuple(_LATEX_BLOCK_RE.findall(md_text)),
        )


class NativeStructuredDocExtractor:
    """Offline zero-egress extractor for PDF, DOCX, PPTX, and HTML tables."""

    @property
    def backend_name(self) -> str:
        return "native-structured"

    def extract(
        self,
        raw_bytes: bytes,
        *,
        filename: str,
        content_type: str | None = None,
    ) -> ExtractedDocument:
        _ = content_type
        suffix = Path(filename).suffix.lower()
        if suffix == ".pdf" or raw_bytes.startswith(b"%PDF-"):
            return self._extract_pdf(raw_bytes, filename=filename)
        if suffix == ".docx":
            return self._extract_docx(raw_bytes, filename=filename)
        if suffix == ".pptx":
            return self._extract_pptx(raw_bytes, filename=filename)
        if suffix in {".html", ".htm", ".xhtml"}:
            return self._extract_html(raw_bytes, filename=filename)
        decoded = raw_bytes.decode("utf-8", errors="replace").replace("\x00", "").strip()
        return ExtractedDocument(
            filename=filename,
            backend=self.backend_name,
            markdown_text=decoded,
            plain_text=decoded,
            sections=_split_markdown_sections(decoded) if decoded else (),
            tables_markdown=tuple(m.group(0).strip() for m in _MD_TABLE_BLOCK_RE.finditer(decoded)),
            formulas_latex=tuple(_LATEX_BLOCK_RE.findall(decoded)),
        )

    def _open_safe_zip(self, raw_bytes: bytes) -> zipfile.ZipFile:
        zf = zipfile.ZipFile(io.BytesIO(raw_bytes))
        infos = zf.infolist()
        if len(infos) > _MAX_ZIP_ENTRIES:
            zf.close()
            raise UnsafeDocumentError(
                f"Archive entry count ({len(infos)}) exceeds safety limit {_MAX_ZIP_ENTRIES}."
            )
        total_uncompressed = sum(info.file_size for info in infos)
        if total_uncompressed > _MAX_ZIP_UNCOMPRESSED_BYTES:
            zf.close()
            raise UnsafeDocumentError(
                f"Uncompressed archive size ({total_uncompressed} bytes) exceeds "
                f"safety limit {_MAX_ZIP_UNCOMPRESSED_BYTES}."
            )
        for info in infos:
            if info.filename.startswith("/") or ".." in Path(info.filename).parts:
                zf.close()
                raise UnsafeDocumentError(f"Path traversal member rejected: {info.filename!r}")
        return zf

    def _extract_pdf(self, raw_bytes: bytes, *, filename: str) -> ExtractedDocument:
        if not raw_bytes or len(raw_bytes) < 32 or not raw_bytes.startswith(b"%PDF-"):
            return ExtractedDocument(
                filename=filename,
                backend=self.backend_name,
                markdown_text="",
                plain_text="",
            )
        try:
            reader = PdfReader(io.BytesIO(raw_bytes))
            pages_md: list[str] = []
            extracted_tables: list[str] = []
            for page_idx, page in enumerate(reader.pages, start=1):
                raw_txt = (page.extract_text() or "").replace("\x00", "").strip()
                if not raw_txt:
                    continue
                md_page, page_tables = self._format_pdf_page_with_tables(raw_txt)
                extracted_tables.extend(page_tables)
                if md_page:
                    pages_md.append(f"<!-- page {page_idx} -->\n{md_page}")
            full_md = "\n\n".join(pages_md).strip()
            plain = re.sub(r"<!-- page \d+ -->\n?", "", full_md).strip()
            return ExtractedDocument(
                filename=filename,
                backend=self.backend_name,
                markdown_text=full_md,
                plain_text=plain,
                sections=_split_markdown_sections(plain) if plain else (),
                tables_markdown=tuple(extracted_tables),
                formulas_latex=tuple(_LATEX_BLOCK_RE.findall(plain)),
            )
        except (
            pypdf.errors.PyPdfError,
            OSError,
            ValueError,
            IndexError,
            KeyError,
            EOFError,
        ) as exc:
            logger.debug("Native PDF extraction failed for '%s': %s", filename, exc)
            return ExtractedDocument(
                filename=filename,
                backend=self.backend_name,
                markdown_text="",
                plain_text="",
            )

    @staticmethod
    def _format_pdf_page_with_tables(page_text: str) -> tuple[str, list[str]]:
        lines = page_text.splitlines()
        out_blocks: list[str] = []
        tables: list[str] = []
        table_buf: list[list[str]] = []

        def _flush_table() -> None:
            nonlocal table_buf
            if len(table_buf) >= 2:
                md_tbl = _grid_to_markdown_table(table_buf)
                if md_tbl:
                    tables.append(md_tbl)
                    out_blocks.append(md_tbl)
                    table_buf = []
                    return
            for row in table_buf:
                out_blocks.append("  ".join(row))
            table_buf = []

        for line in lines:
            stripped = line.strip()
            if "|" in stripped and stripped.count("|") >= 2:
                cells = [c.strip() for c in stripped.strip("|").split("|")]
                table_buf.append(cells)
                continue
            cols = [c.strip() for c in re.split(r"\s{3,}|\t+", stripped) if c.strip()]
            if len(cols) >= 2 and not stripped.endswith((".", "?", "!")):
                table_buf.append(cols)
                continue
            if table_buf:
                _flush_table()
            if stripped:
                out_blocks.append(stripped)
        if table_buf:
            _flush_table()
        return "\n\n".join(out_blocks).strip(), tables

    def _extract_docx(self, raw_bytes: bytes, *, filename: str) -> ExtractedDocument:
        with self._open_safe_zip(raw_bytes) as zf:
            if "word/document.xml" not in zf.namelist():
                raise ValueError(f"Invalid DOCX archive '{filename}': missing word/document.xml")
            xml_bytes = zf.read("word/document.xml")
        root = _safe_parse_xml(xml_bytes)

        w_ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
        body = root.find(f"{w_ns}body")
        if body is None:
            return ExtractedDocument(
                filename=filename, backend=self.backend_name, markdown_text="", plain_text=""
            )

        blocks: list[str] = []
        tables: list[str] = []

        def _para_text(p_elem: ET.Element) -> str:
            texts: list[str] = []
            for t_elem in p_elem.iter(f"{w_ns}t"):
                if t_elem.text:
                    texts.append(t_elem.text)
            return "".join(texts).strip()

        for child in body:
            if child.tag == f"{w_ns}p":
                txt = _para_text(child)
                if not txt:
                    continue
                ppr = child.find(f"{w_ns}pPr")
                pstyle = ppr.find(f"{w_ns}pStyle") if ppr is not None else None
                style_val = str(pstyle.attrib.get(f"{w_ns}val", "")) if pstyle is not None else ""
                m_hdr = re.match(r"Heading([1-6])", style_val, re.IGNORECASE)
                if m_hdr:
                    lvl = int(m_hdr.group(1))
                    blocks.append(f"{'#' * lvl} {txt}")
                else:
                    blocks.append(txt)
            elif child.tag == f"{w_ns}tbl":
                rows: list[list[str]] = []
                for tr in child.findall(f"{w_ns}tr"):
                    row_cells: list[str] = []
                    for tc in tr.findall(f"{w_ns}tc"):
                        cell_parts = [_para_text(p) for p in tc.findall(f"{w_ns}p")]
                        cell_txt = " ".join(part for part in cell_parts if part).strip()
                        tcpr = tc.find(f"{w_ns}tcPr")
                        gspan_el = tcpr.find(f"{w_ns}gridSpan") if tcpr is not None else None
                        span = 1
                        if gspan_el is not None:
                            try:
                                span = max(1, int(gspan_el.attrib.get(f"{w_ns}val", "1")))
                            except ValueError:
                                span = 1
                        row_cells.append(cell_txt)
                        for _ in range(span - 1):
                            row_cells.append("")
                    if row_cells:
                        rows.append(row_cells)
                md_tbl = _grid_to_markdown_table(rows)
                if md_tbl:
                    tables.append(md_tbl)
                    blocks.append(md_tbl)

        md_text = "\n\n".join(blocks).strip()
        return ExtractedDocument(
            filename=filename,
            backend=self.backend_name,
            markdown_text=md_text,
            plain_text=md_text,
            sections=_split_markdown_sections(md_text) if md_text else (),
            tables_markdown=tuple(tables),
            formulas_latex=tuple(_LATEX_BLOCK_RE.findall(md_text)),
        )

    def _extract_pptx(self, raw_bytes: bytes, *, filename: str) -> ExtractedDocument:
        with self._open_safe_zip(raw_bytes) as zf:
            slide_names = [
                name for name in zf.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)
            ]
            if not slide_names:
                raise ValueError(f"Invalid PPTX archive '{filename}': no slides found")
            slide_names.sort(key=lambda n: int(re.findall(r"\d+", n)[-1]))
            slide_blobs = [(name, zf.read(name)) for name in slide_names]

        a_ns = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
        p_ns = "{http://schemas.openxmlformats.org/presentationml/2006/main}"
        blocks: list[str] = []
        tables: list[str] = []

        for idx, (_sname, sbytes) in enumerate(slide_blobs, start=1):
            root = _safe_parse_xml(sbytes)
            slide_title = f"Slide {idx}"
            slide_paras: list[str] = []

            for sp in root.iter(f"{p_ns}sp"):
                ph = sp.find(f".//{p_ns}ph")
                ph_type = ph.attrib.get("type", "") if ph is not None else ""
                p_texts: list[str] = []
                for p_elem in sp.iter(f"{a_ns}p"):
                    runs = [t.text for t in p_elem.iter(f"{a_ns}t") if t.text]
                    joined = "".join(runs).strip()
                    if joined:
                        p_texts.append(joined)
                if not p_texts:
                    continue
                if ph_type in {"title", "ctrTitle"} and slide_title == f"Slide {idx}":
                    slide_title = p_texts[0]
                    slide_paras.extend(p_texts[1:])
                else:
                    slide_paras.extend(p_texts)

            slide_tables: list[str] = []
            for tbl in root.iter(f"{a_ns}tbl"):
                rows: list[list[str]] = []
                for tr in tbl.findall(f"{a_ns}tr"):
                    row_cells: list[str] = []
                    for tc in tr.findall(f"{a_ns}tc"):
                        t_parts = [t.text for t in tc.iter(f"{a_ns}t") if t.text]
                        row_cells.append("".join(t_parts).strip())
                    if row_cells:
                        rows.append(row_cells)
                md_tbl = _grid_to_markdown_table(rows)
                if md_tbl:
                    tables.append(md_tbl)
                    slide_tables.append(md_tbl)

            section_lines = [f"# {slide_title}", *slide_paras, *slide_tables]
            blocks.append("\n\n".join(line for line in section_lines if line))

        md_text = "\n\n".join(blocks).strip()
        return ExtractedDocument(
            filename=filename,
            backend=self.backend_name,
            markdown_text=md_text,
            plain_text=md_text,
            sections=_split_markdown_sections(md_text) if md_text else (),
            tables_markdown=tuple(tables),
            formulas_latex=tuple(_LATEX_BLOCK_RE.findall(md_text)),
        )

    def _extract_html(self, raw_bytes: bytes, *, filename: str) -> ExtractedDocument:
        raw_html = raw_bytes.decode("utf-8", errors="replace").replace("\x00", "")
        if "<!DOCTYPE" in raw_html.upper() and "<!ENTITY" in raw_html.upper():
            raise UnsafeDocumentError("Blocked HTML/XHTML document containing <!ENTITY>.")
        md_text, tables = html_to_structured_markdown(raw_html)
        plain = html_mod.unescape(md_text)
        return ExtractedDocument(
            filename=filename,
            backend=self.backend_name,
            markdown_text=md_text,
            plain_text=plain,
            sections=_split_markdown_sections(md_text) if md_text else (),
            tables_markdown=tables,
            formulas_latex=tuple(_LATEX_BLOCK_RE.findall(md_text)),
        )


class CompositeDoclingExtractor:
    """Chains self-hosted Docling HTTP / in-process Docling with zero-egress native parsing."""

    def __init__(
        self,
        *,
        http_extractor: DoclingServeHttpExtractor | None = None,
        in_process_extractor: InProcessDoclingExtractor | None = None,
        native_extractor: NativeStructuredDocExtractor | None = None,
        fail_closed_on_remote_error: bool = False,
    ) -> None:
        self.http_extractor = http_extractor
        self.in_process_extractor = in_process_extractor
        self.native_extractor = native_extractor or NativeStructuredDocExtractor()
        self.fail_closed_on_remote_error = fail_closed_on_remote_error

    @property
    def backend_name(self) -> str:
        if self.http_extractor is not None:
            return self.http_extractor.backend_name
        if self.in_process_extractor is not None:
            return self.in_process_extractor.backend_name
        return self.native_extractor.backend_name

    def extract(
        self,
        raw_bytes: bytes,
        *,
        filename: str,
        content_type: str | None = None,
    ) -> ExtractedDocument:
        if self.http_extractor is not None:
            try:
                return self.http_extractor.extract(
                    raw_bytes, filename=filename, content_type=content_type
                )
            except (httpx.HTTPError, ValueError) as exc:
                if self.fail_closed_on_remote_error:
                    raise
                logger.warning(
                    "docling-serve HTTP extraction unavailable for '%s' (%s); "
                    "falling back to local structured extractor",
                    filename,
                    exc,
                )

        if self.in_process_extractor is not None and InProcessDoclingExtractor.is_available():
            try:
                return self.in_process_extractor.extract(
                    raw_bytes, filename=filename, content_type=content_type
                )
            except Exception as exc:
                logger.warning(
                    "In-process Docling failed for '%s' (%s); falling back to native extractor",
                    filename,
                    exc,
                )

        return self.native_extractor.extract(
            raw_bytes, filename=filename, content_type=content_type
        )


def get_doc_extractor(
    settings: ScoutSettings | None = None,
    *,
    http_client: httpx.Client | None = None,
    fail_closed_on_remote_error: bool = False,
) -> CompositeDoclingExtractor:
    """Factory returning the configured DocExtractorSPI implementation."""
    from context_scout.config import ScoutSettings

    cfg = settings or ScoutSettings()
    http_ext: DoclingServeHttpExtractor | None = None
    if cfg.docling_endpoint_url and cfg.docling_endpoint_url.strip():
        http_ext = DoclingServeHttpExtractor(
            cfg.docling_endpoint_url,
            api_key=cfg.docling_api_key,
            timeout_seconds=cfg.docling_timeout_seconds,
            allow_external_egress=cfg.allow_external_cloud_egress,
            http_client=http_client,
        )
    in_proc = InProcessDoclingExtractor() if InProcessDoclingExtractor.is_available() else None
    return CompositeDoclingExtractor(
        http_extractor=http_ext,
        in_process_extractor=in_proc,
        native_extractor=NativeStructuredDocExtractor(),
        fail_closed_on_remote_error=fail_closed_on_remote_error,
    )


__all__ = [
    "CompositeDoclingExtractor",
    "DocExtractorSPI",
    "DoclingServeHttpExtractor",
    "ExtractedDiagram",
    "ExtractedDocument",
    "ExtractedSection",
    "InProcessDoclingExtractor",
    "NativeStructuredDocExtractor",
    "UnsafeDocumentError",
    "get_doc_extractor",
    "html_to_structured_markdown",
    "safe_parse_xml_text",
    "validate_self_hosted_endpoint",
]
