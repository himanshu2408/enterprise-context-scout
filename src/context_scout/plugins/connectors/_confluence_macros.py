"""Structured Confluence macro, HTML table, and diagram extractors.

Handles Confluence Storage Format (``<ac:structured-macro>``), embedded
PlantUML and Mermaid diagrams, Draw.io ``<mxfile>`` / ``<mxGraphModel>`` XML
(both inline XML and deflate+base64 encoded payloads), and multi-span HTML
tables via ``html_to_structured_markdown`` while blocking XXE payloads and
decompression bombs.
"""

from __future__ import annotations

import base64
import html
import logging
import re
import urllib.parse
import zlib
from dataclasses import dataclass
from typing import Final

from context_scout.ingestion.docling_extractor import (
    ExtractedDiagram,
    UnsafeDocumentError,
    html_to_structured_markdown,
    safe_parse_xml_text,
)

logger = logging.getLogger(__name__)

_MAX_INFLATED_DIAGRAM_BYTES: Final[int] = 2_000_000
_COMMON_NON_JIRA_PREFIXES: Final[frozenset[str]] = frozenset(
    {"UTF", "ISO", "SHA", "MD", "HTTP", "HTTPS", "TLS", "RFC", "AES", "RSA", "CVE", "CWE", "E2E"}
)

_MACRO_BLOCK_RE: Final[re.Pattern[str]] = re.compile(
    r'<ac:structured-macro\b[^>]*\bac:name=["\']([^"\']+)["\'][^>]*>'
    r"((?:(?!</?ac:structured-macro\b)[\s\S])*)"
    r"</ac:structured-macro>",
    re.IGNORECASE,
)
_MACRO_SELF_CLOSING_RE: Final[re.Pattern[str]] = re.compile(
    r'<ac:structured-macro\b[^>]*\bac:name=["\']([^"\']+)["\'][^>]*/\s*>',
    re.IGNORECASE,
)
_PARAM_RE: Final[re.Pattern[str]] = re.compile(
    r'<ac:parameter\b[^>]*\bac:name=["\']([^"\']+)["\'][^>]*>(.*?)</ac:parameter>',
    re.IGNORECASE | re.DOTALL,
)
_PLAIN_BODY_RE: Final[re.Pattern[str]] = re.compile(
    r"<ac:plain-text-body\b[^>]*>(.*?)</ac:plain-text-body>",
    re.IGNORECASE | re.DOTALL,
)
_RICH_BODY_RE: Final[re.Pattern[str]] = re.compile(
    r"<ac:rich-text-body\b[^>]*>(.*?)</ac:rich-text-body>",
    re.IGNORECASE | re.DOTALL,
)
_CDATA_RE: Final[re.Pattern[str]] = re.compile(r"<!\[CDATA\[(.*?)\]\]>", re.DOTALL)
_MXFILE_INLINE_RE: Final[re.Pattern[str]] = re.compile(
    r"(<(?:mxfile|mxGraphModel)\b[\s\S]*?</(?:mxfile|mxGraphModel)>)",
    re.IGNORECASE,
)
_TABLE_BLOCK_RE: Final[re.Pattern[str]] = re.compile(
    r"(<table\b[\s\S]*?</table>)",
    re.IGNORECASE,
)
_FENCED_DIAGRAM_RE: Final[re.Pattern[str]] = re.compile(
    r"```(plantuml|puml|mermaid)\b[^\n]*\n([\s\S]*?)```",
    re.IGNORECASE,
)
_STARTUML_INLINE_RE: Final[re.Pattern[str]] = re.compile(
    r"(@startuml[\s\S]*?@enduml)",
    re.IGNORECASE,
)
_JIRA_KEY_RE: Final[re.Pattern[str]] = re.compile(r"\b([A-Z][A-Z0-9]{1,11})-(\d{1,7})\b")
_HTML_TAG_STRIP_RE: Final[re.Pattern[str]] = re.compile(r"<[^>]+>")
_MULTI_NL_RE: Final[re.Pattern[str]] = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class ConfluenceParsedContent:
    """Structured extraction result from Confluence XHTML storage format."""

    markdown_text: str
    diagrams: tuple[ExtractedDiagram, ...]
    linked_jira_keys: tuple[str, ...]
    macros_seen: tuple[str, ...]
    tables_markdown: tuple[str, ...]


def _strip_cdata_and_unescape(text: str) -> str:
    stripped = _CDATA_RE.sub(lambda m: m.group(1), text)
    return html.unescape(stripped).strip("\r\n")


def _clean_label(raw: str) -> str:
    no_tags = _HTML_TAG_STRIP_RE.sub(" ", raw)
    return " ".join(html.unescape(no_tags).split())


def _extract_params(macro_inner: str) -> dict[str, str]:
    params: dict[str, str] = {}
    for m in _PARAM_RE.finditer(macro_inner):
        key = m.group(1).strip().lower()
        val = _clean_label(_strip_cdata_and_unescape(m.group(2)))
        if key and val:
            params[key] = val
    return params


def parse_plantuml_diagram(source: str, *, title: str = "") -> ExtractedDiagram:
    """Extract participants/nodes and directional message edges from PlantUML."""
    nodes: list[str] = []
    seen_nodes: set[str] = set()
    edges: list[tuple[str, str, str]] = []
    aliases: dict[str, str] = {}
    resolved_title = title

    def _add_node(name: str) -> str:
        clean = name.strip().strip('"').strip("'")
        clean = aliases.get(clean, clean)
        if clean and clean not in seen_nodes:
            seen_nodes.add(clean)
            nodes.append(clean)
        return clean

    decl_re = re.compile(
        r"^\s*(?:participant|actor|boundary|control|entity|database|collections|"
        r"queue|component|node|cloud|package)\s+"
        r'(?:"([^"]+)"|(\[[^\]]+\]|\S+))(?:\s+as\s+(\S+))?',
        re.IGNORECASE,
    )
    edge_re = re.compile(
        r'^\s*(?:"([^"]+)"|([A-Za-z0-9_.\-\[\]]+))\s*'
        r"([<.\-=*ox]+(?:\[[^\]]+\])?[>.\-=*ox]+)\s*"
        r'(?:"([^"]+)"|([A-Za-z0-9_.\-\[\]]+))'
        r"(?:\s*:\s*(.+))?$"
    )

    for raw_line in source.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("'", "@start", "@end", "skinparam", "autonumber")):
            continue
        if line.lower().startswith("title ") and not resolved_title:
            resolved_title = line[6:].strip()
            continue
        decl_m = decl_re.match(line)
        if decl_m:
            display = (decl_m.group(1) or decl_m.group(2) or "").strip("[]")
            alias = decl_m.group(3)
            _add_node(display)
            if alias and display:
                aliases[alias] = display
            continue
        edge_m = edge_re.match(line)
        if edge_m:
            src_raw = (edge_m.group(1) or edge_m.group(2) or "").strip("[]")
            dst_raw = (edge_m.group(4) or edge_m.group(5) or "").strip("[]")
            label = (edge_m.group(6) or "").strip()
            src = _add_node(src_raw)
            dst = _add_node(dst_raw)
            if src and dst:
                edges.append((src, dst, label))

    summary_lines = [f"Diagram (PlantUML){f': {resolved_title}' if resolved_title else ''}"]
    if nodes:
        summary_lines.append("Nodes: " + ", ".join(nodes))
    for src, dst, lbl in edges:
        edge_txt = f"- {src} -> {dst}" + (f" ({lbl})" if lbl else "")
        summary_lines.append(edge_txt)

    return ExtractedDiagram(
        diagram_type="plantuml",
        title=resolved_title,
        raw_source=source.strip(),
        nodes=tuple(nodes),
        edges=tuple(edges),
        summary_markdown="\n".join(summary_lines),
    )


def parse_mermaid_diagram(source: str, *, title: str = "") -> ExtractedDiagram:
    """Extract nodes and directed edges from Mermaid flowchart/sequence diagrams."""
    nodes: list[str] = []
    seen_nodes: set[str] = set()
    labels: dict[str, str] = {}
    edges: list[tuple[str, str, str]] = []
    resolved_title = title

    def _norm_token(tok: str) -> tuple[str, str]:
        cleaned = tok.strip()
        m = re.match(
            r'^([A-Za-z0-9_.\-]+)\s*(?:\[\s*"?([^"\]]+)"?\s*\]|\(\s*"?([^")]+)"?\s*\))$', cleaned
        )
        if m:
            ident = m.group(1)
            lbl = (m.group(2) or m.group(3) or ident).strip()
            labels[ident] = lbl
            return ident, lbl
        ident = re.sub(r"[\[\](){}\"]", "", cleaned).strip()
        return ident, labels.get(ident, ident)

    def _add_node(ident: str, lbl: str) -> str:
        display = lbl or ident
        if ident and display and display not in seen_nodes:
            seen_nodes.add(display)
            nodes.append(display)
        return display

    seq_re = re.compile(
        r"^\s*([A-Za-z0-9_.\-]+)\s*(->>|-->>|->|-->|-x|--x)\s*([A-Za-z0-9_.\-]+)\s*(?::\s*(.+))?$"
    )
    flow_re = re.compile(
        r"^\s*([^\s\-=<]+(?:\[[^\]]+\]|\([^)]+\))?)\s*"
        r"(?:-->|==>|-.->|--\s*([^->]+?)\s*-->)\s*"
        r"(?:\|([^|]+)\|\s*)?"
        r"([^\s;]+(?:\[[^\]]+\]|\([^)]+\))?)\s*;?\s*$"
    )
    part_re = re.compile(r"^\s*(?:participant|actor)\s+(\S+)(?:\s+as\s+(.+))?$", re.IGNORECASE)

    for raw_line in source.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(
            ("%%", "flowchart ", "graph ", "sequenceDiagram", "subgraph", "end")
        ):
            continue
        if line.lower().startswith("title ") and not resolved_title:
            resolved_title = line[6:].strip()
            continue
        pm = part_re.match(line)
        if pm:
            ident = pm.group(1).strip()
            lbl = (pm.group(2) or ident).strip()
            labels[ident] = lbl
            _add_node(ident, lbl)
            continue
        sm = seq_re.match(line)
        if sm:
            s_id, s_lbl = _norm_token(sm.group(1))
            d_id, d_lbl = _norm_token(sm.group(3))
            lbl = (sm.group(4) or "").strip()
            s_disp = _add_node(s_id, s_lbl)
            d_disp = _add_node(d_id, d_lbl)
            if s_disp and d_disp:
                edges.append((s_disp, d_disp, lbl))
            continue
        fm = flow_re.match(line)
        if fm:
            s_id, s_lbl = _norm_token(fm.group(1))
            d_id, d_lbl = _norm_token(fm.group(4))
            lbl = (fm.group(2) or fm.group(3) or "").strip()
            s_disp = _add_node(s_id, s_lbl)
            d_disp = _add_node(d_id, d_lbl)
            if s_disp and d_disp:
                edges.append((s_disp, d_disp, lbl))

    summary_lines = [f"Diagram (Mermaid){f': {resolved_title}' if resolved_title else ''}"]
    if nodes:
        summary_lines.append("Nodes: " + ", ".join(nodes))
    for src, dst, lbl in edges:
        summary_lines.append(f"- {src} -> {dst}" + (f" ({lbl})" if lbl else ""))

    return ExtractedDiagram(
        diagram_type="mermaid",
        title=resolved_title,
        raw_source=source.strip(),
        nodes=tuple(nodes),
        edges=tuple(edges),
        summary_markdown="\n".join(summary_lines),
    )


def _decode_drawio_payload(raw_payload: str) -> str:
    """Decode raw or deflate+base64 Draw.io <diagram> / <mxfile> content safely."""
    trimmed = raw_payload.strip()
    if "<mxfile" in trimmed or "<mxGraphModel" in trimmed:
        return trimmed
    # Base64 -> raw deflate (diagrams.net format) -> optional URL decode
    try:
        compressed = base64.b64decode(trimmed, validate=True)
    except (ValueError, TypeError) as exc:
        raise UnsafeDocumentError("Invalid base64 payload in Draw.io diagram") from exc

    decomp = zlib.decompressobj(wbits=-15)
    try:
        inflated = decomp.decompress(compressed, _MAX_INFLATED_DIAGRAM_BYTES + 1)
    except zlib.error:
        # Try standard zlib header fallback if wbits=-15 fails
        decomp_std = zlib.decompressobj()
        inflated = decomp_std.decompress(compressed, _MAX_INFLATED_DIAGRAM_BYTES + 1)
    if len(inflated) > _MAX_INFLATED_DIAGRAM_BYTES:
        raise UnsafeDocumentError(
            "Draw.io diagram payload exceeds maximum decompressed byte budget"
        )
    text = inflated.decode("utf-8", errors="replace")
    if "%3C" in text or "%3c" in text:
        text = urllib.parse.unquote(text)
    return text


def parse_drawio_xml(xml_or_encoded: str, *, title: str = "") -> ExtractedDiagram:
    """Parse Draw.io ``<mxfile>`` / ``<mxGraphModel>`` XML with strict XXE blocking."""
    xml_text = xml_or_encoded.strip()
    # Raises UnsafeDocumentError on <!DOCTYPE / <!ENTITY immediately
    root = safe_parse_xml_text(xml_text)
    resolved_title = title

    models_to_scan = []
    tag_local = root.tag.rsplit("}", 1)[-1]
    if tag_local == "mxfile":
        for diag in root.iter():
            d_tag = diag.tag.rsplit("}", 1)[-1]
            if d_tag == "diagram":
                if not resolved_title and diag.attrib.get("name"):
                    resolved_title = diag.attrib["name"].strip()
                inner_models = [c for c in diag if c.tag.rsplit("}", 1)[-1] == "mxGraphModel"]
                if inner_models:
                    models_to_scan.extend(inner_models)
                elif diag.text and diag.text.strip():
                    decoded = _decode_drawio_payload(diag.text.strip())
                    inner_root = safe_parse_xml_text(decoded)
                    models_to_scan.append(inner_root)
    else:
        models_to_scan.append(root)

    id_to_label: dict[str, str] = {}
    nodes: list[str] = []
    seen_nodes: set[str] = set()
    raw_edges: list[tuple[str, str, str]] = []

    for model in models_to_scan:
        for cell in model.iter():
            if cell.tag.rsplit("}", 1)[-1] != "mxCell":
                continue
            cell_id = cell.attrib.get("id", "")
            value = _clean_label(cell.attrib.get("value", ""))
            if cell.attrib.get("vertex") == "1" and cell_id:
                label = value or cell_id
                id_to_label[cell_id] = label
                if label not in seen_nodes:
                    seen_nodes.add(label)
                    nodes.append(label)
            elif cell.attrib.get("edge") == "1":
                src_id = cell.attrib.get("source", "")
                dst_id = cell.attrib.get("target", "")
                if src_id and dst_id:
                    raw_edges.append((src_id, dst_id, value))

    edges: list[tuple[str, str, str]] = [
        (id_to_label.get(s, s), id_to_label.get(d, d), lbl) for s, d, lbl in raw_edges
    ]

    summary_lines = [f"Diagram (Draw.io){f': {resolved_title}' if resolved_title else ''}"]
    if nodes:
        summary_lines.append("Nodes: " + ", ".join(nodes))
    for src, dst, lbl in edges:
        summary_lines.append(f"- {src} -> {dst}" + (f" ({lbl})" if lbl else ""))

    return ExtractedDiagram(
        diagram_type="drawio",
        title=resolved_title,
        raw_source=xml_text,
        nodes=tuple(nodes),
        edges=tuple(edges),
        summary_markdown="\n".join(summary_lines),
    )


def _extract_jira_keys_from_text(text: str) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()
    for match in _JIRA_KEY_RE.finditer(text):
        prefix = match.group(1)
        if prefix in _COMMON_NON_JIRA_PREFIXES:
            continue
        full = f"{prefix}-{match.group(2)}"
        if full not in seen:
            seen.add(full)
            keys.append(full)
    return keys


def extract_confluence_structures(
    raw_storage: str,
    *,
    preserve_headings: bool = False,
) -> ConfluenceParsedContent:
    """Convert Confluence XHTML storage markup into structured Markdown + diagrams.

    Processes ``<ac:structured-macro>`` blocks recursively (``code``, callouts,
    ``status``, ``jira``, ``plantuml``, ``mermaid``, ``drawio``), Draw.io inline
    ``<mxfile>`` / ``<mxGraphModel>`` XML, fenced/inline PlantUML & Mermaid, and
    HTML ``<table>`` blocks with ``colspan`` / ``rowspan``.
    """
    diagrams: list[ExtractedDiagram] = []
    jira_keys: list[str] = []
    seen_jira: set[str] = set()
    macros_seen: list[str] = []
    seen_macros: set[str] = set()
    tables_markdown: list[str] = []

    def _add_jira_key(key: str) -> None:
        cleaned = key.strip().upper()
        if cleaned and cleaned not in seen_jira:
            seen_jira.add(cleaned)
            jira_keys.append(cleaned)

    def _record_macro(name: str) -> None:
        norm = name.strip().lower()
        if norm and norm not in seen_macros:
            seen_macros.add(norm)
            macros_seen.append(norm)

    def _render_macro(name: str, inner: str) -> str:
        macro_name = name.strip().lower()
        _record_macro(macro_name)
        params = _extract_params(inner)
        plain_match = _PLAIN_BODY_RE.search(inner)
        plain_body = _strip_cdata_and_unescape(plain_match.group(1)) if plain_match else ""
        rich_match = _RICH_BODY_RE.search(inner)
        rich_body = rich_match.group(1) if rich_match else ""

        if macro_name == "code":
            lang = params.get("language", "").strip().lower()
            title = params.get("title", "")
            if lang in {"plantuml", "puml"}:
                diag = parse_plantuml_diagram(plain_body, title=title)
                diagrams.append(diag)
                return f"\n\n{diag.summary_markdown}\n```plantuml\n{plain_body}\n```\n\n"
            if lang == "mermaid":
                diag = parse_mermaid_diagram(plain_body, title=title)
                diagrams.append(diag)
                return f"\n\n{diag.summary_markdown}\n```mermaid\n{plain_body}\n```\n\n"
            header = f"**{title}**\n" if title else ""
            return f"\n\n{header}```{lang}\n{plain_body}\n```\n\n"

        if macro_name in {"info", "note", "warning", "tip", "panel", "expand", "details"}:
            rendered_inner = _transform_macros_and_markup(rich_body or plain_body)
            rendered_clean = _strip_basic_html(rendered_inner)
            title = params.get("title", "")
            label = f"{macro_name.upper()}{f': {title}' if title else ''}"
            quoted_lines = [f"> **[{label}]**"]
            for ln in rendered_clean.splitlines():
                quoted_lines.append(f"> {ln}" if ln.strip() else ">")
            return "\n\n" + "\n".join(quoted_lines) + "\n\n"

        if macro_name == "status":
            title = params.get("title", "STATUS")
            colour = params.get("colour") or params.get("color") or ""
            suffix = f" ({colour})" if colour else ""
            return f" [STATUS: {title}{suffix}] "

        if macro_name in {"jira", "jiraissues"}:
            key = params.get("key", "")
            if key:
                _add_jira_key(key)
                return f" [Jira: {key.upper()}] "
            jql = params.get("jqlquery", "")
            for k in _extract_jira_keys_from_text(jql):
                _add_jira_key(k)
            return f" [Jira Query: {jql}] " if jql else " [Jira] "

        if macro_name in {"plantuml", "plantumlrender"}:
            diag = parse_plantuml_diagram(plain_body, title=params.get("title", ""))
            diagrams.append(diag)
            return f"\n\n{diag.summary_markdown}\n```plantuml\n{plain_body}\n```\n\n"

        if macro_name == "mermaid":
            diag = parse_mermaid_diagram(plain_body, title=params.get("title", ""))
            diagrams.append(diag)
            return f"\n\n{diag.summary_markdown}\n```mermaid\n{plain_body}\n```\n\n"

        if macro_name in {"drawio", "inc-drawio", "mxgraph"}:
            title = params.get("diagramname") or params.get("title") or ""
            xml_source = plain_body or params.get("diagramxml", "") or params.get("xml", "")
            if xml_source:
                diag = parse_drawio_xml(xml_source, title=title)
                diagrams.append(diag)
                return f"\n\n{diag.summary_markdown}\n\n"
            return f"\n\nDiagram (Draw.io){f': {title}' if title else ''}\n\n"

        # Generic fallback: preserve rich/plain body
        if rich_body:
            return "\n" + _transform_macros_and_markup(rich_body) + "\n"
        if plain_body:
            return f"\n{plain_body}\n"
        return ""

    def _transform_macros_and_markup(fragment: str) -> str:
        # Handle nested or top-level macros iteratively (innermost first fallback)
        text = fragment
        text = _MACRO_SELF_CLOSING_RE.sub(lambda m: _render_macro(m.group(1), m.group(0)), text)
        # Iterate innermost-outward to resolve nested structured macros safely
        for _ in range(8):
            if "<ac:structured-macro" not in text.lower():
                break
            updated = _MACRO_BLOCK_RE.sub(lambda m: _render_macro(m.group(1), m.group(2)), text)
            if updated == text:
                break
            text = updated
        return text

    transformed = _transform_macros_and_markup(raw_storage)

    # Inline Draw.io <mxfile>/<mxGraphModel> outside macros
    def _replace_inline_mx(m: re.Match[str]) -> str:
        diag = parse_drawio_xml(m.group(1))
        diagrams.append(diag)
        return f"\n\n{diag.summary_markdown}\n\n"

    transformed = _MXFILE_INLINE_RE.sub(_replace_inline_mx, transformed)

    # Inline HTML tables -> Markdown tables preserving colspan/rowspan
    def _replace_table(m: re.Match[str]) -> str:
        md, md_tables = html_to_structured_markdown(m.group(1))
        tables_markdown.extend(md_tables)
        if md_tables:
            return "\n\n" + "\n\n".join(md_tables) + "\n\n"
        return f"\n\n{md}\n\n"

    transformed = _TABLE_BLOCK_RE.sub(_replace_table, transformed)

    # Fenced plantuml/mermaid blocks written directly inside HTML/Markdown bodies
    for fm in _FENCED_DIAGRAM_RE.finditer(transformed):
        kind = fm.group(1).lower()
        body = fm.group(2).strip()
        if not any(d.raw_source == body for d in diagrams):
            if kind in {"plantuml", "puml"}:
                diagrams.append(parse_plantuml_diagram(body))
            elif kind == "mermaid":
                diagrams.append(parse_mermaid_diagram(body))

    # Standalone @startuml ... @enduml not already fenced or extracted
    def _replace_startuml(m: re.Match[str]) -> str:
        src = m.group(1).strip()
        if any(d.raw_source == src for d in diagrams):
            return src
        diag = parse_plantuml_diagram(src)
        diagrams.append(diag)
        return f"\n\n{diag.summary_markdown}\n```plantuml\n{src}\n```\n\n"

    # Only wrap standalone @startuml if not inside ```plantuml already
    if "@startuml" in transformed.lower() and "```plantuml" not in transformed.lower():
        transformed = _STARTUML_INLINE_RE.sub(_replace_startuml, transformed)

    cleaned = _strip_basic_html(transformed, preserve_headings=preserve_headings)

    # Collect Jira issue keys referenced in page prose/macros as well
    for k in _extract_jira_keys_from_text(cleaned):
        _add_jira_key(k)

    return ConfluenceParsedContent(
        markdown_text=cleaned,
        diagrams=tuple(diagrams),
        linked_jira_keys=tuple(jira_keys),
        macros_seen=tuple(macros_seen),
        tables_markdown=tuple(tables_markdown),
    )


def _strip_basic_html(raw_html: str, *, preserve_headings: bool = False) -> str:
    """Strip HTML tags while preserving code fences, blockquotes, tables, and newlines."""
    fences: list[str] = []
    headings: list[str] = []

    def _stash_fence(m: re.Match[str]) -> str:
        idx = len(fences)
        fences.append(m.group(0))
        return f"@@CODEFENCE_{idx}@@"

    def _stash_heading(m: re.Match[str]) -> str:
        idx = len(headings)
        headings.append(m.group(0))
        return f"\n@@HEADING_{idx}@@\n"

    stashed = re.sub(r"```[\s\S]*?```", _stash_fence, raw_html)
    if preserve_headings:
        stashed = re.sub(
            r"<h([1-6])[^>]*>.*?</h\1>",
            _stash_heading,
            stashed,
            flags=re.IGNORECASE | re.DOTALL,
        )
    text = re.sub(r"<li[^>]*>", "\n* ", stashed, flags=re.IGNORECASE)
    text = re.sub(r"<(?:br|p|div|tr)[^>]*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    for idx, heading in enumerate(headings):
        text = text.replace(f"@@HEADING_{idx}@@", heading)
    for idx, fence in enumerate(fences):
        text = text.replace(f"@@CODEFENCE_{idx}@@", fence)
    text = _MULTI_NL_RE.sub("\n\n", text)
    return text.strip()


def split_markdown_blocks_preserving_structures(text: str, max_chars: int) -> list[str]:
    """Split text into chunks up to ``max_chars`` without slicing code fences or tables."""
    if len(text) <= max_chars:
        return [text]

    units: list[str] = []
    lines = text.splitlines()
    idx = 0
    total = len(lines)
    para_buf: list[str] = []

    def _flush_para() -> None:
        if para_buf:
            joined = "\n".join(para_buf).strip()
            if joined:
                units.append(joined)
            para_buf.clear()

    while idx < total:
        line = lines[idx]
        stripped = line.lstrip()
        if stripped.startswith("```"):
            _flush_para()
            fence_lines = [line]
            idx += 1
            while idx < total:
                fence_lines.append(lines[idx])
                if lines[idx].lstrip().startswith("```"):
                    idx += 1
                    break
                idx += 1
            units.append("\n".join(fence_lines))
            continue

        if stripped.startswith("|") and idx + 1 < total and lines[idx + 1].lstrip().startswith("|"):
            _flush_para()
            table_lines = [line]
            idx += 1
            while idx < total and lines[idx].lstrip().startswith("|"):
                table_lines.append(lines[idx])
                idx += 1
            units.append("\n".join(table_lines))
            continue

        if not stripped:
            _flush_para()
            idx += 1
            continue

        para_buf.append(line)
        idx += 1

    _flush_para()

    blocks: list[str] = []
    current_block: list[str] = []
    current_len = 0
    for unit in units:
        if current_block and current_len + len(unit) + 2 > max_chars:
            blocks.append("\n\n".join(current_block))
            current_block = [unit]
            current_len = len(unit)
        else:
            current_block.append(unit)
            current_len += len(unit) + (2 if len(current_block) > 1 else 0)

    if current_block:
        blocks.append("\n\n".join(current_block))
    return blocks
