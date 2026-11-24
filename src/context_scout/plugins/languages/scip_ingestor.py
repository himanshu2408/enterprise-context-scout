"""SCIP (Source Code Intelligence Protocol) binary & JSON index ingestor plugin."""

from __future__ import annotations

import base64
import binascii
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from google.protobuf.internal import decoder, encoder, wire_format  # type: ignore[import-untyped]

from context_scout.ast_engine.extractor import ParseDiagnostics
from context_scout.ast_engine.models import ParsedEdge, ParsedSymbol
from context_scout.plugins.base import DependencyEdge

logger = logging.getLogger(__name__)

# SCIP SymbolRole bitmask constants (SCIP proto spec)
SYMBOL_ROLE_DEFINITION = 1 << 0  # 1
SYMBOL_ROLE_IMPORT = 1 << 1  # 2
SYMBOL_ROLE_WRITE_ACCESS = 1 << 2  # 4
SYMBOL_ROLE_READ_ACCESS = 1 << 3  # 8
SYMBOL_ROLE_GENERATED = 1 << 4  # 16
SYMBOL_ROLE_TEST = 1 << 5  # 32

# SCIP SymbolInformation.Kind -> DB-allowed symbol_kind
# Allowed: function, class, method, interface, type_alias, hook, slice, selector,
# reducer, story, tf_resource, tf_module, gcp_resource, component, system, team,
# service, module, api, domain
_SCIP_KIND_TO_SYMBOL_KIND: dict[int, str] = {
    7: "class",  # Class
    11: "class",  # Constructor
    12: "class",  # Enum
    17: "function",  # Function
    21: "interface",  # Interface
    26: "method",  # Method
    29: "module",  # Module
    30: "module",  # Namespace
    35: "module",  # Package
    49: "class",  # Struct
    52: "type_alias",  # Trait
    54: "type_alias",  # Type
    55: "type_alias",  # TypeAlias
}

_KIND_NAME_TO_SYMBOL_KIND: dict[str, str] = {
    "class": "class",
    "constructor": "method",
    "enum": "class",
    "function": "function",
    "interface": "interface",
    "method": "method",
    "module": "module",
    "namespace": "module",
    "package": "module",
    "struct": "class",
    "trait": "type_alias",
    "type": "type_alias",
    "type_alias": "type_alias",
    "typealias": "type_alias",
}


class SCIPDecodeError(ValueError):
    """Raised when SCIP protobuf wire bytes or JSON payload are malformed."""


@dataclass(frozen=True)
class ParsedSCIPSymbol:
    """Structured representation of a SCIP symbol string."""

    raw: str
    scheme: str
    manager: str
    package: str
    version: str
    descriptors: tuple[str, ...]
    display_name: str
    qualified_name: str
    is_local: bool = False


@dataclass(frozen=True)
class SCIPRelationship:
    """SCIP SymbolInformation.Relationship."""

    symbol: str
    is_reference: bool = False
    is_implementation: bool = False
    is_type_definition: bool = False
    is_definition: bool = False


@dataclass(frozen=True)
class SCIPSymbolInfo:
    """SCIP SymbolInformation."""

    symbol: str
    documentation: tuple[str, ...] = ()
    relationships: tuple[SCIPRelationship, ...] = ()
    kind: int = 0
    display_name: str = ""


@dataclass(frozen=True)
class SCIPOccurrence:
    """SCIP Occurrence."""

    range: tuple[int, ...] = ()
    symbol: str = ""
    symbol_roles: int = 0


@dataclass(frozen=True)
class SCIPDocument:
    """SCIP Document."""

    relative_path: str
    occurrences: tuple[SCIPOccurrence, ...] = ()
    symbols: tuple[SCIPSymbolInfo, ...] = ()
    text: str = ""


@dataclass(frozen=True)
class SCIPIndex:
    """Decoded SCIP Index."""

    tool_name: str = ""
    tool_version: str = ""
    project_root: str = ""
    documents: tuple[SCIPDocument, ...] = ()
    external_symbols: tuple[SCIPSymbolInfo, ...] = ()


def parse_scip_symbol(symbol: str) -> ParsedSCIPSymbol:
    """Parse SCIP symbol string according to SCIP descriptor grammar.

    Grammar:
      local <id>
      <scheme> <manager> <package> <version> <descriptors...>
    """
    raw = symbol.strip()
    if not raw:
        return ParsedSCIPSymbol(
            raw=symbol,
            scheme="",
            manager="",
            package="",
            version="",
            descriptors=(),
            display_name="",
            qualified_name="",
            is_local=False,
        )

    if raw.startswith("local "):
        local_id = raw[len("local ") :].strip()
        return ParsedSCIPSymbol(
            raw=raw,
            scheme="local",
            manager="",
            package="",
            version="",
            descriptors=(local_id,) if local_id else (),
            display_name=local_id or "local",
            qualified_name=f"local.{local_id}" if local_id else "local",
            is_local=True,
        )

    parts = raw.split(" ", 4)
    if len(parts) < 4:
        clean = raw.replace(" ", ".")
        return ParsedSCIPSymbol(
            raw=raw,
            scheme=parts[0] if parts else "",
            manager="",
            package="",
            version="",
            descriptors=(),
            display_name=clean.rsplit(".", 1)[-1],
            qualified_name=clean,
            is_local=False,
        )

    scheme, manager, package, version = parts[0], parts[1], parts[2], parts[3]
    desc_raw = parts[4] if len(parts) == 5 else ""
    tokens = [t for t in re.split(r"[/#().\[\]]+", desc_raw) if t]
    descriptors = tuple(tokens)
    display_name = descriptors[-1] if descriptors else package
    qn_parts = [p for p in (package, *descriptors) if p and p != "."]
    qualified_name = ".".join(qn_parts) if qn_parts else raw
    return ParsedSCIPSymbol(
        raw=raw,
        scheme=scheme,
        manager=manager,
        package=package,
        version=version,
        descriptors=descriptors,
        display_name=display_name,
        qualified_name=qualified_name,
        is_local=False,
    )


def _infer_kind_from_symbol(symbol: str, kind_code: int) -> str:
    if kind_code in _SCIP_KIND_TO_SYMBOL_KIND:
        return _SCIP_KIND_TO_SYMBOL_KIND[kind_code]
    if symbol.endswith("()."):
        return "method" if "#" in symbol else "function"
    if symbol.endswith("#"):
        return "class"
    if symbol.endswith("/"):
        return "module"
    return "function"


# ---------------------------------------------------------------------------
# Protobuf wire-format encoder / decoder (google.protobuf.internal)
# ---------------------------------------------------------------------------


def _encode_varint_field(buf: bytearray, field_number: int, value: int) -> None:
    buf.extend(encoder.TagBytes(field_number, wire_format.WIRETYPE_VARINT))
    encoder._EncodeVarint(buf.extend, value)


def _encode_bytes_field(buf: bytearray, field_number: int, data: bytes) -> None:
    buf.extend(encoder.TagBytes(field_number, wire_format.WIRETYPE_LENGTH_DELIMITED))
    encoder._EncodeVarint(buf.extend, len(data))
    buf.extend(data)


def _encode_string_field(buf: bytearray, field_number: int, text: str) -> None:
    if text:
        _encode_bytes_field(buf, field_number, text.encode("utf-8"))


def _encode_packed_int32(buf: bytearray, field_number: int, values: list[int]) -> None:
    if not values:
        return
    packed = bytearray()
    for val in values:
        encoder._EncodeVarint(packed.extend, val & 0xFFFFFFFF if val < 0 else val)
    _encode_bytes_field(buf, field_number, bytes(packed))


def _encode_relationship(rel: dict[str, Any]) -> bytes:
    buf = bytearray()
    _encode_string_field(buf, 1, str(rel.get("symbol", "")))
    if rel.get("is_reference"):
        _encode_varint_field(buf, 2, 1)
    if rel.get("is_implementation"):
        _encode_varint_field(buf, 3, 1)
    if rel.get("is_type_definition"):
        _encode_varint_field(buf, 4, 1)
    if rel.get("is_definition"):
        _encode_varint_field(buf, 5, 1)
    return bytes(buf)


def _encode_symbol_info(sym: dict[str, Any]) -> bytes:
    buf = bytearray()
    _encode_string_field(buf, 1, str(sym.get("symbol", "")))
    for doc in sym.get("documentation", []):
        _encode_string_field(buf, 3, str(doc))
    for rel in sym.get("relationships", []):
        _encode_bytes_field(buf, 4, _encode_relationship(rel))
    kind = int(sym.get("kind", 0))
    if kind:
        _encode_varint_field(buf, 5, kind)
    display_name = str(sym.get("display_name", ""))
    if display_name:
        _encode_string_field(buf, 6, display_name)
    return bytes(buf)


def _encode_occurrence(occ: dict[str, Any]) -> bytes:
    buf = bytearray()
    rng = [int(x) for x in occ.get("range", [])]
    if rng:
        _encode_packed_int32(buf, 1, rng)
    _encode_string_field(buf, 2, str(occ.get("symbol", "")))
    roles = int(occ.get("symbol_roles", 0))
    if roles:
        _encode_varint_field(buf, 3, roles)
    return bytes(buf)


def _encode_document(doc: dict[str, Any]) -> bytes:
    buf = bytearray()
    _encode_string_field(buf, 1, str(doc.get("relative_path", "")))
    for occ in doc.get("occurrences", []):
        _encode_bytes_field(buf, 2, _encode_occurrence(occ))
    for sym in doc.get("symbols", []):
        _encode_bytes_field(buf, 3, _encode_symbol_info(sym))
    text = str(doc.get("text", ""))
    if text:
        _encode_string_field(buf, 4, text)
    return bytes(buf)


def _encode_metadata(tool_name: str, tool_version: str, project_root: str) -> bytes:
    tool_buf = bytearray()
    _encode_string_field(tool_buf, 1, tool_name)
    _encode_string_field(tool_buf, 2, tool_version)

    meta_buf = bytearray()
    _encode_varint_field(meta_buf, 1, 0)  # ProtocolVersion.UnspecifiedProtocolVersion
    if tool_buf:
        _encode_bytes_field(meta_buf, 2, bytes(tool_buf))
    _encode_string_field(meta_buf, 3, project_root)
    return bytes(meta_buf)


def build_scip_index_bytes(
    *,
    documents: list[dict[str, Any]],
    external_symbols: list[dict[str, Any]] | None = None,
    tool_name: str = "scip-java",
    tool_version: str = "0.1.0",
    project_root: str = "file:///workspace",
) -> bytes:
    """Build deterministic SCIP binary protobuf bytes for tests and pipelines."""
    buf = bytearray()
    meta_bytes = _encode_metadata(tool_name, tool_version, project_root)
    if meta_bytes:
        _encode_bytes_field(buf, 1, meta_bytes)
    for doc in documents:
        _encode_bytes_field(buf, 2, _encode_document(doc))
    for ext in external_symbols or []:
        _encode_bytes_field(buf, 3, _encode_symbol_info(ext))
    return bytes(buf)


def _read_varint(buf: memoryview, pos: int, end: int) -> tuple[int, int]:
    if pos >= end:
        raise SCIPDecodeError("Unexpected EOF while reading varint")
    try:
        val, new_pos = decoder._DecodeVarint(buf, pos)
    except (IndexError, ValueError, Exception) as exc:
        raise SCIPDecodeError(f"Malformed varint at byte {pos}: {exc}") from exc
    if new_pos > end:
        raise SCIPDecodeError(f"Varint at byte {pos} extends past message end ({new_pos} > {end})")
    return int(val), int(new_pos)


def _read_length_delimited(buf: memoryview, pos: int, end: int) -> tuple[memoryview, int]:
    length, pos = _read_varint(buf, pos, end)
    if length < 0 or pos + length > end:
        raise SCIPDecodeError(
            f"Length-delimited field at byte {pos} truncated (len={length}, remaining={end - pos})"
        )
    return buf[pos : pos + length], pos + length


def _skip_field(buf: memoryview, pos: int, end: int, wire_type: int) -> int:
    if wire_type == wire_format.WIRETYPE_VARINT:
        _, pos = _read_varint(buf, pos, end)
        return pos
    if wire_type == wire_format.WIRETYPE_FIXED64:
        if pos + 8 > end:
            raise SCIPDecodeError("Truncated fixed64 field")
        return pos + 8
    if wire_type == wire_format.WIRETYPE_LENGTH_DELIMITED:
        _, pos = _read_length_delimited(buf, pos, end)
        return pos
    if wire_type == wire_format.WIRETYPE_FIXED32:
        if pos + 4 > end:
            raise SCIPDecodeError("Truncated fixed32 field")
        return pos + 4
    raise SCIPDecodeError(f"Unsupported protobuf wire type: {wire_type}")


def _decode_utf8(view: memoryview) -> str:
    try:
        return bytes(view).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SCIPDecodeError(f"Invalid UTF-8 string in protobuf payload: {exc}") from exc


def _decode_tag(buf: memoryview, pos: int, end: int) -> tuple[int, int, int]:
    key, pos = _read_varint(buf, pos, end)
    field_number = key >> 3
    wire_type = key & 0x07
    if field_number <= 0:
        raise SCIPDecodeError(f"Invalid protobuf field number {field_number}")
    return field_number, wire_type, pos


def _decode_relationship(buf: memoryview) -> SCIPRelationship:
    pos, end = 0, len(buf)
    symbol = ""
    is_reference = False
    is_implementation = False
    is_type_definition = False
    is_definition = False
    while pos < end:
        field_no, wire_type, pos = _decode_tag(buf, pos, end)
        if field_no == 1 and wire_type == wire_format.WIRETYPE_LENGTH_DELIMITED:
            val, pos = _read_length_delimited(buf, pos, end)
            symbol = _decode_utf8(val)
        elif field_no == 2 and wire_type == wire_format.WIRETYPE_VARINT:
            v, pos = _read_varint(buf, pos, end)
            is_reference = bool(v)
        elif field_no == 3 and wire_type == wire_format.WIRETYPE_VARINT:
            v, pos = _read_varint(buf, pos, end)
            is_implementation = bool(v)
        elif field_no == 4 and wire_type == wire_format.WIRETYPE_VARINT:
            v, pos = _read_varint(buf, pos, end)
            is_type_definition = bool(v)
        elif field_no == 5 and wire_type == wire_format.WIRETYPE_VARINT:
            v, pos = _read_varint(buf, pos, end)
            is_definition = bool(v)
        else:
            pos = _skip_field(buf, pos, end, wire_type)
    return SCIPRelationship(
        symbol=symbol,
        is_reference=is_reference,
        is_implementation=is_implementation,
        is_type_definition=is_type_definition,
        is_definition=is_definition,
    )


def _decode_symbol_info(buf: memoryview) -> SCIPSymbolInfo:
    pos, end = 0, len(buf)
    symbol = ""
    docs: list[str] = []
    rels: list[SCIPRelationship] = []
    kind = 0
    display_name = ""
    while pos < end:
        field_no, wire_type, pos = _decode_tag(buf, pos, end)
        if field_no == 1 and wire_type == wire_format.WIRETYPE_LENGTH_DELIMITED:
            val, pos = _read_length_delimited(buf, pos, end)
            symbol = _decode_utf8(val)
        elif field_no == 3 and wire_type == wire_format.WIRETYPE_LENGTH_DELIMITED:
            val, pos = _read_length_delimited(buf, pos, end)
            docs.append(_decode_utf8(val))
        elif field_no == 4 and wire_type == wire_format.WIRETYPE_LENGTH_DELIMITED:
            val, pos = _read_length_delimited(buf, pos, end)
            rels.append(_decode_relationship(val))
        elif field_no == 5 and wire_type == wire_format.WIRETYPE_VARINT:
            kind, pos = _read_varint(buf, pos, end)
        elif field_no == 6 and wire_type == wire_format.WIRETYPE_LENGTH_DELIMITED:
            val, pos = _read_length_delimited(buf, pos, end)
            display_name = _decode_utf8(val)
        else:
            pos = _skip_field(buf, pos, end, wire_type)
    return SCIPSymbolInfo(
        symbol=symbol,
        documentation=tuple(docs),
        relationships=tuple(rels),
        kind=kind,
        display_name=display_name,
    )


def _decode_occurrence(buf: memoryview) -> SCIPOccurrence:
    pos, end = 0, len(buf)
    rng: list[int] = []
    symbol = ""
    symbol_roles = 0
    while pos < end:
        field_no, wire_type, pos = _decode_tag(buf, pos, end)
        if field_no == 1 and wire_type == wire_format.WIRETYPE_LENGTH_DELIMITED:
            packed, pos = _read_length_delimited(buf, pos, end)
            ppos, pend = 0, len(packed)
            while ppos < pend:
                v, ppos = _read_varint(packed, ppos, pend)
                rng.append(v)
        elif field_no == 1 and wire_type == wire_format.WIRETYPE_VARINT:
            v, pos = _read_varint(buf, pos, end)
            rng.append(v)
        elif field_no == 2 and wire_type == wire_format.WIRETYPE_LENGTH_DELIMITED:
            val, pos = _read_length_delimited(buf, pos, end)
            symbol = _decode_utf8(val)
        elif field_no == 3 and wire_type == wire_format.WIRETYPE_VARINT:
            symbol_roles, pos = _read_varint(buf, pos, end)
        else:
            pos = _skip_field(buf, pos, end, wire_type)
    return SCIPOccurrence(range=tuple(rng), symbol=symbol, symbol_roles=symbol_roles)


def _decode_document(buf: memoryview) -> SCIPDocument:
    pos, end = 0, len(buf)
    relative_path = ""
    occurrences: list[SCIPOccurrence] = []
    symbols: list[SCIPSymbolInfo] = []
    text = ""
    while pos < end:
        field_no, wire_type, pos = _decode_tag(buf, pos, end)
        if field_no == 1 and wire_type == wire_format.WIRETYPE_LENGTH_DELIMITED:
            val, pos = _read_length_delimited(buf, pos, end)
            relative_path = _decode_utf8(val)
        elif field_no == 2 and wire_type == wire_format.WIRETYPE_LENGTH_DELIMITED:
            val, pos = _read_length_delimited(buf, pos, end)
            occurrences.append(_decode_occurrence(val))
        elif field_no == 3 and wire_type == wire_format.WIRETYPE_LENGTH_DELIMITED:
            val, pos = _read_length_delimited(buf, pos, end)
            symbols.append(_decode_symbol_info(val))
        elif field_no == 4 and wire_type == wire_format.WIRETYPE_LENGTH_DELIMITED:
            val, pos = _read_length_delimited(buf, pos, end)
            text = _decode_utf8(val)
        else:
            pos = _skip_field(buf, pos, end, wire_type)
    return SCIPDocument(
        relative_path=relative_path,
        occurrences=tuple(occurrences),
        symbols=tuple(symbols),
        text=text,
    )


def _decode_metadata(buf: memoryview) -> tuple[str, str, str]:
    pos, end = 0, len(buf)
    tool_name = ""
    tool_version = ""
    project_root = ""
    while pos < end:
        field_no, wire_type, pos = _decode_tag(buf, pos, end)
        if field_no == 2 and wire_type == wire_format.WIRETYPE_LENGTH_DELIMITED:
            tbuf, pos = _read_length_delimited(buf, pos, end)
            tpos, tend = 0, len(tbuf)
            while tpos < tend:
                tf, tw, tpos = _decode_tag(tbuf, tpos, tend)
                if tf == 1 and tw == wire_format.WIRETYPE_LENGTH_DELIMITED:
                    val, tpos = _read_length_delimited(tbuf, tpos, tend)
                    tool_name = _decode_utf8(val)
                elif tf == 2 and tw == wire_format.WIRETYPE_LENGTH_DELIMITED:
                    val, tpos = _read_length_delimited(tbuf, tpos, tend)
                    tool_version = _decode_utf8(val)
                else:
                    tpos = _skip_field(tbuf, tpos, tend, tw)
        elif field_no == 3 and wire_type == wire_format.WIRETYPE_LENGTH_DELIMITED:
            val, pos = _read_length_delimited(buf, pos, end)
            project_root = _decode_utf8(val)
        else:
            pos = _skip_field(buf, pos, end, wire_type)
    return tool_name, tool_version, project_root


def decode_scip_protobuf(data: bytes) -> SCIPIndex:
    """Decode raw SCIP protobuf bytes into a SCIPIndex, failing closed on corruption."""
    buf = memoryview(data)
    pos, end = 0, len(buf)
    tool_name = ""
    tool_version = ""
    project_root = ""
    documents: list[SCIPDocument] = []
    external_symbols: list[SCIPSymbolInfo] = []

    while pos < end:
        field_no, wire_type, pos = _decode_tag(buf, pos, end)
        if field_no == 1 and wire_type == wire_format.WIRETYPE_LENGTH_DELIMITED:
            val, pos = _read_length_delimited(buf, pos, end)
            tool_name, tool_version, project_root = _decode_metadata(val)
        elif field_no == 2 and wire_type == wire_format.WIRETYPE_LENGTH_DELIMITED:
            val, pos = _read_length_delimited(buf, pos, end)
            documents.append(_decode_document(val))
        elif field_no == 3 and wire_type == wire_format.WIRETYPE_LENGTH_DELIMITED:
            val, pos = _read_length_delimited(buf, pos, end)
            external_symbols.append(_decode_symbol_info(val))
        else:
            pos = _skip_field(buf, pos, end, wire_type)

    return SCIPIndex(
        tool_name=tool_name,
        tool_version=tool_version,
        project_root=project_root,
        documents=tuple(documents),
        external_symbols=tuple(external_symbols),
    )


def _parse_kind_value(raw_kind: Any) -> int:
    if isinstance(raw_kind, int):
        return raw_kind
    if isinstance(raw_kind, str):
        norm = raw_kind.strip().lower()
        for code, mapped in _SCIP_KIND_TO_SYMBOL_KIND.items():
            if mapped == _KIND_NAME_TO_SYMBOL_KIND.get(norm):
                return code
    return 0


def decode_scip_json(payload: str) -> SCIPIndex:
    """Decode a `.scip.json` payload into a SCIPIndex."""
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise SCIPDecodeError(f"Malformed SCIP JSON: {exc}") from exc

    if not isinstance(obj, dict) or "documents" not in obj:
        raise SCIPDecodeError("SCIP JSON must be an object containing 'documents'")

    raw_docs = obj.get("documents")
    if not isinstance(raw_docs, list):
        raise SCIPDecodeError("SCIP JSON 'documents' must be a list")

    meta = obj.get("metadata") or {}
    tool_info = meta.get("tool_info") or {}
    tool_name = str(tool_info.get("name", ""))
    tool_version = str(tool_info.get("version", ""))
    project_root = str(meta.get("project_root", ""))

    def _to_sym(item: dict[str, Any]) -> SCIPSymbolInfo:
        rels = tuple(
            SCIPRelationship(
                symbol=str(r.get("symbol", "")),
                is_reference=bool(r.get("is_reference", False)),
                is_implementation=bool(r.get("is_implementation", False)),
                is_type_definition=bool(r.get("is_type_definition", False)),
                is_definition=bool(r.get("is_definition", False)),
            )
            for r in (item.get("relationships") or [])
            if isinstance(r, dict)
        )
        return SCIPSymbolInfo(
            symbol=str(item.get("symbol", "")),
            documentation=tuple(str(d) for d in (item.get("documentation") or [])),
            relationships=rels,
            kind=_parse_kind_value(item.get("kind", 0)),
            display_name=str(item.get("display_name", "")),
        )

    docs: list[SCIPDocument] = []
    for d in raw_docs:
        if not isinstance(d, dict):
            raise SCIPDecodeError("Each SCIP document must be a JSON object")
        occs = tuple(
            SCIPOccurrence(
                range=tuple(int(x) for x in (o.get("range") or [])),
                symbol=str(o.get("symbol", "")),
                symbol_roles=int(o.get("symbol_roles", 0)),
            )
            for o in (d.get("occurrences") or [])
            if isinstance(o, dict)
        )
        syms = tuple(_to_sym(s) for s in (d.get("symbols") or []) if isinstance(s, dict))
        docs.append(
            SCIPDocument(
                relative_path=str(d.get("relative_path", "")),
                occurrences=occs,
                symbols=syms,
                text=str(d.get("text", "")),
            )
        )

    ext_syms = tuple(_to_sym(s) for s in (obj.get("external_symbols") or []) if isinstance(s, dict))
    return SCIPIndex(
        tool_name=tool_name,
        tool_version=tool_version,
        project_root=project_root,
        documents=tuple(docs),
        external_symbols=ext_syms,
    )


def _line_range_from_scip(rng: tuple[int, ...]) -> tuple[int, int]:
    """Convert 0-indexed SCIP range [startLine, startCol, endLine?, endCol?] to 1-indexed lines."""
    if not rng:
        return 1, 1
    start_line = max(1, rng[0] + 1)
    end_line = max(start_line, rng[2] + 1) if len(rng) == 4 else start_line
    return start_line, end_line


@dataclass
class SCIPIngestor:
    """Ingests SCIP binary protobuf or JSON indices into ParsedSymbol and ParsedEdge."""

    last_parse_diagnostics: ParseDiagnostics = field(
        default_factory=lambda: ParseDiagnostics(parser_backend="scip-protobuf")
    )

    def _convert_index(
        self,
        index: SCIPIndex,
        *,
        repo_name: str = "",
        module_prefix: str = "",
    ) -> tuple[list[ParsedSymbol], list[ParsedEdge]]:
        symbols: list[ParsedSymbol] = []
        edges: list[ParsedEdge] = []
        seen_symbols: set[str] = set()
        seen_edges: set[tuple[str, str, str]] = set()

        def add_edge(src: str, tgt: str, kind: str) -> None:
            if not src or not tgt or src == tgt:
                return
            key = (src, tgt, kind)
            if key not in seen_edges:
                seen_edges.add(key)
                edges.append(
                    ParsedEdge(
                        source_qualified_name=src,
                        target_qualified_name=tgt,
                        edge_kind=kind,
                    )
                )

        # Index all SymbolInformation globally by raw SCIP symbol string
        info_by_symbol: dict[str, SCIPSymbolInfo] = {}
        for ext in index.external_symbols:
            if ext.symbol:
                info_by_symbol[ext.symbol] = ext
        for doc in index.documents:
            for sym in doc.symbols:
                if sym.symbol:
                    info_by_symbol[sym.symbol] = sym

        for doc in index.documents:
            doc_defs: list[tuple[int, int, str]] = []
            for occ in doc.occurrences:
                if not occ.symbol:
                    continue
                parsed = parse_scip_symbol(occ.symbol)
                qn = parsed.qualified_name
                if module_prefix and not qn.startswith(module_prefix):
                    qn = f"{module_prefix}.{qn}"
                start_line, end_line = _line_range_from_scip(occ.range)

                if occ.symbol_roles & SYMBOL_ROLE_DEFINITION:
                    doc_defs.append((start_line, end_line, qn))
                    if qn not in seen_symbols:
                        seen_symbols.add(qn)
                        info = info_by_symbol.get(occ.symbol)
                        name = (
                            info.display_name if info and info.display_name else parsed.display_name
                        ) or qn
                        kind = _infer_kind_from_symbol(occ.symbol, info.kind if info else 0)
                        doc_str = "\n".join(info.documentation) if info else ""
                        sig = f"{kind} {name}"
                        symbols.append(
                            ParsedSymbol(
                                name=name,
                                qualified_name=qn,
                                bare_name=name,
                                symbol_kind=kind,
                                start_line=start_line,
                                end_line=end_line,
                                start_byte=0,
                                end_byte=0,
                                signature=sig,
                                signature_skeleton=sig,
                                docstring=doc_str,
                                contextual_summary=f"@scip {doc.relative_path}"
                                if doc.relative_path
                                else "",
                                repo_name=repo_name,
                            )
                        )
                else:
                    # Non-definition occurrence -> innermost enclosing definition or document
                    matching_defs = [
                        (de - ds, dqn) for ds, de, dqn in doc_defs if ds <= start_line <= de
                    ]
                    if matching_defs:
                        enclosing = min(matching_defs, key=lambda item: item[0])[1]
                    elif doc_defs:
                        enclosing = doc_defs[-1][2]
                    else:
                        enclosing = doc.relative_path or module_prefix or "scip.document"
                    edge_kind = "IMPORTS" if (occ.symbol_roles & SYMBOL_ROLE_IMPORT) else "CALLS"
                    add_edge(enclosing, qn, edge_kind)

            # Relationships declared on SymbolInformation in this document
            for sym_info in doc.symbols:
                src_parsed = parse_scip_symbol(sym_info.symbol)
                src_qn = src_parsed.qualified_name
                if module_prefix and not src_qn.startswith(module_prefix):
                    src_qn = f"{module_prefix}.{src_qn}"
                for rel in sym_info.relationships:
                    tgt_parsed = parse_scip_symbol(rel.symbol)
                    tgt_qn = tgt_parsed.qualified_name
                    if module_prefix and not tgt_qn.startswith(module_prefix):
                        tgt_qn = f"{module_prefix}.{tgt_qn}"
                    if rel.is_implementation:
                        add_edge(src_qn, tgt_qn, "INHERITS")
                    if rel.is_type_definition:
                        add_edge(src_qn, tgt_qn, "USES_TYPE")
                    if rel.is_reference:
                        add_edge(src_qn, tgt_qn, "CALLS")

        return symbols, edges

    def ingest_bytes(
        self,
        data: bytes,
        file_path: str = "",
        repo_name: str = "",
        module_prefix: str = "",
    ) -> tuple[list[ParsedSymbol], list[ParsedEdge]]:
        """Decode raw SCIP protobuf bytes (or UTF-8 JSON bytes) and extract symbols/edges."""
        if not data:
            self.last_parse_diagnostics = ParseDiagnostics(parser_backend="scip-protobuf")
            return [], []

        stripped = data.lstrip()
        if file_path.endswith(".scip.json") or stripped.startswith(b"{"):
            try:
                text = data.decode("utf-8")
                index = decode_scip_json(text)
            except (UnicodeDecodeError, SCIPDecodeError) as exc:
                logger.warning("SCIP JSON parse error in %s: %s", file_path or "<memory>", exc)
                self.last_parse_diagnostics = ParseDiagnostics(
                    parser_backend="scip-json",
                    has_parse_error=True,
                    abstained=True,
                    reason=str(exc),
                )
                return [], []
            self.last_parse_diagnostics = ParseDiagnostics(parser_backend="scip-json")
            return self._convert_index(index, repo_name=repo_name, module_prefix=module_prefix)

        try:
            index = decode_scip_protobuf(data)
        except SCIPDecodeError as exc:
            logger.warning(
                "SCIP protobuf wire decode error in %s: %s", file_path or "<memory>", exc
            )
            self.last_parse_diagnostics = ParseDiagnostics(
                parser_backend="scip-protobuf",
                has_parse_error=True,
                abstained=True,
                reason=str(exc),
            )
            return [], []

        self.last_parse_diagnostics = ParseDiagnostics(parser_backend="scip-protobuf")
        return self._convert_index(index, repo_name=repo_name, module_prefix=module_prefix)

    def extract_ast(
        self,
        code: str,
        file_path: str = "",
        repo_name: str = "",
        module_prefix: str = "",
    ) -> tuple[list[ParsedSymbol], list[ParsedEdge]]:
        """Extract symbols and edges from SCIP JSON or base64-encoded protobuf payload."""
        if not code or not code.strip():
            self.last_parse_diagnostics = ParseDiagnostics(parser_backend="scip-protobuf")
            return [], []

        stripped = code.lstrip()
        if file_path.endswith(".scip.json") or stripped.startswith("{"):
            return self.ingest_bytes(
                code.encode("utf-8"),
                file_path=file_path or "index.scip.json",
                repo_name=repo_name,
                module_prefix=module_prefix,
            )

        # Accept base64-encoded protobuf wire bytes when passed via text LanguagePlugin.extract_ast
        try:
            raw_bytes = base64.b64decode(code.strip(), validate=True)
        except (binascii.Error, ValueError) as exc:
            logger.warning("Invalid SCIP payload in %s: %s", file_path or "<memory>", exc)
            self.last_parse_diagnostics = ParseDiagnostics(
                parser_backend="scip-protobuf",
                has_parse_error=True,
                abstained=True,
                reason=f"Invalid SCIP base64/JSON payload: {exc}",
            )
            return [], []

        return self.ingest_bytes(
            raw_bytes,
            file_path=file_path,
            repo_name=repo_name,
            module_prefix=module_prefix,
        )


class SCIPIndexPlugin(SCIPIngestor):
    """LanguagePlugin implementation for `.scip` and `.scip.json` index files."""

    @property
    def language_id(self) -> str:
        return "scip"

    @property
    def file_extensions(self) -> tuple[str, ...]:
        return (".scip", ".scip.json")

    @property
    def manifest_files(self) -> tuple[str, ...]:
        return ()

    def can_handle(self, file_path: str) -> bool:
        lower = file_path.lower()
        return lower.endswith(".scip") or lower.endswith(".scip.json")

    def extract_manifest_dependencies(
        self,
        file_path: str,
        content: str,
    ) -> list[DependencyEdge]:
        self.last_parse_diagnostics = ParseDiagnostics(parser_backend="scip-protobuf")
        return []
