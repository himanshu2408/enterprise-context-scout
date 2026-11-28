"""Shared AST extractor diagnostics, error inspection, and language helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from tree_sitter import Node, Parser


@dataclass(frozen=True)
class ParseDiagnostics:
    """Structured diagnostics recorded for every AST extraction attempt."""

    parser_backend: str
    has_parse_error: bool = False
    abstained: bool = False
    low_confidence_fallback: bool = False
    error_node_count: int = 0
    missing_node_count: int = 0
    reason: str = ""


def inspect_tree_errors(root_node: Node) -> tuple[bool, int, int]:
    """Counts ERROR and MISSING nodes in a tree-sitter AST iteratively.

    Returns ``(has_error, error_node_count, missing_node_count)``.
    """
    if not root_node.has_error:
        return False, 0, 0
    error_count = 0
    missing_count = 0
    stack: list[Node] = [root_node]
    while stack:
        node = stack.pop()
        if not node.has_error:
            continue
        if node.is_error or node.type == "ERROR":
            error_count += 1
        if node.is_missing:
            missing_count += 1
        stack.extend(reversed(node.children))
    return True, error_count, missing_count


PY_PRIMITIVES: Final[set[str]] = {
    "int",
    "str",
    "float",
    "bool",
    "bytes",
    "None",
    "list",
    "dict",
    "set",
    "tuple",
    "Optional",
    "Union",
    "Any",
    "Callable",
    "Iterator",
    "Generator",
    "Literal",
    "Final",
    "ClassVar",
    "Sequence",
    "Mapping",
    "Iterable",
}

TS_PRIMITIVES: Final[set[str]] = {
    "string",
    "number",
    "boolean",
    "void",
    "any",
    "unknown",
    "never",
    "null",
    "undefined",
    "Promise",
    "Array",
    "Record",
    "Map",
    "Set",
    "Partial",
    "Required",
    "Readonly",
    "Pick",
    "Omit",
    "Exclude",
    "Extract",
}


def _point_row(point: object) -> int:
    if isinstance(point, (tuple, list)) and len(point) > 0:
        return int(point[0])
    return int(getattr(point, "row", 0))


def _node_text(node: Node | None, code_bytes: bytes | None = None) -> str:
    if node is None:
        return ""
    if code_bytes is not None:
        return code_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
    if node.text is None:
        return ""
    return node.text.decode("utf-8")


def _extract_types(type_str: str, is_ts: bool = False) -> list[str]:
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", type_str)
    primitives = TS_PRIMITIVES if is_ts else PY_PRIMITIVES
    return [t for t in tokens if t not in primitives and not t.isdigit()]


def _derive_module_prefix(file_path: str) -> str:
    path = file_path.replace("\\", "/")
    if path.startswith("src/"):
        path = path[4:]
    for ext in (
        ".py",
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".mjs",
        ".cjs",
        ".sql",
        ".yaml",
        ".yml",
        ".tf",
        ".go",
        ".rs",
        ".java",
        ".md",
    ):
        if path.endswith(ext):
            path = path[: -len(ext)]
            break
    parts = [p.lstrip(".") for p in path.split("/") if p and p != "__init__"]
    return ".".join(p for p in parts if p)


_LANGUAGE_ALIASES: Final[dict[str, str]] = {
    "py": "python",
    "ts": "typescript",
    "js": "typescript",
    "javascript": "typescript",
    "golang": "go",
    "rs": "rust",
    "kt": "kotlin",
    "kts": "kotlin",
    "proto": "protobuf",
    "grpc": "protobuf",
    "graphql": "api_specs",
    "gql": "api_specs",
    "openapi": "api_specs",
    "asyncapi": "api_specs",
    "dbt": "sql",
    "jinja": "templates_i18n",
    "jinja2": "templates_i18n",
    "mjml": "templates_i18n",
    "handlebars": "templates_i18n",
    "hbs": "templates_i18n",
    "i18n": "templates_i18n",
    "po": "templates_i18n",
    "cs": "csharp",
    "c#": "csharp",
    "hcl": "terraform",
    "tf": "terraform",
}


def match_symbol_identifier(candidate: str, record_qualified_name: str) -> bool:
    """Matches query symbol candidates against repo-qualified symbols."""
    if candidate == record_qualified_name:
        return True
    if "::" in record_qualified_name:
        parts = record_qualified_name.split("::")
        bare_name = parts[-1]
        # Match bare name (e.g. 'TokenValidator' or 'Order.AddOrderItem')
        if candidate == bare_name or candidate == bare_name.split(".")[-1]:
            return True
        # Match module.bare name (e.g. 'services.auth.TokenValidator')
        if len(parts) >= 2:
            legacy_dot_format = f"{parts[1]}.{bare_name}"
            if candidate == legacy_dot_format:
                return True
        # Suffix match
        if record_qualified_name.endswith(f"::{candidate}"):
            return True
    else:
        if (
            record_qualified_name.endswith(f".{candidate}")
            or record_qualified_name.split(".")[-1] == candidate.split(".")[-1]
        ):
            return True
    return False


class _ASTExtractorStateBase:
    """Shared typed attributes for ASTExtractor language walker mixins."""

    py_parser: Parser
    ts_parser: Parser
    tsx_parser: Parser
    _active_tree: object | None
    last_parse_diagnostics: ParseDiagnostics | None
