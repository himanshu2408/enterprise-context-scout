"""Multi-language Tree-Sitter AST extractor for Python, TypeScript/TSX, and HCL."""

import gc
import logging
import re
from dataclasses import dataclass
from typing import Final, Literal

import tree_sitter_hcl as tshcl
import tree_sitter_python as tspython
import tree_sitter_typescript as tstypescript
from tree_sitter import Language, Node, Parser, Tree

from context_scout.ast_engine.models import ParsedEdge, ParsedSymbol

logger = logging.getLogger(__name__)


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


_HCL_PARSER: Parser | None = None


def _get_hcl_parser() -> Parser:
    global _HCL_PARSER
    if _HCL_PARSER is None:
        _HCL_PARSER = Parser(Language(tshcl.language()))
    return _HCL_PARSER


def _unquote_hcl_string(node: Node, code_bytes: bytes) -> str:
    for child in node.children:
        if child.type == "template_literal":
            return _node_text(child, code_bytes)
    text = _node_text(node, code_bytes).strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        return text[1:-1]
    return text


def _collect_hcl_attr_chains(node: Node, code_bytes: bytes) -> list[list[str]]:
    """Extracts identifier chains (e.g. ['var', 'region'] or ['data', 'img', 'debian']) from AST."""
    chains: list[list[str]] = []
    stack: list[Node] = [node]
    while stack:
        curr = stack.pop()
        ctype = curr.type
        if ctype == "literal_value":
            continue
        if ctype == "expression":
            children = [c for c in curr.children if c.is_named]
            if children:
                first_type = children[0].type
                if len(children) == 1 and first_type == "literal_value":
                    continue
                if first_type == "variable_expr":
                    ident = _node_text(children[0], code_bytes).strip()
                    parts = [ident]
                    for sibling in children[1:]:
                        if sibling.type == "get_attr":
                            attr_id = next(
                                (c for c in sibling.children if c.type == "identifier"),
                                None,
                            )
                            if attr_id is not None:
                                parts.append(_node_text(attr_id, code_bytes).strip())
                        else:
                            break
                    if len(parts) >= 2:
                        chains.append(parts)
                    # Still inspect any remaining non-chain children (e.g. index expressions)
                    for sibling in reversed(children[len(parts) :]):
                        stack.append(sibling)
                    continue
        stack.extend(reversed(curr.children))
    return chains


def parse_hcl_ast_with_tree(
    code: str,
    file_path: str = "",
    repo_name: str = "",
    module_prefix: str = "",
    *,
    module_symbol_kind: Literal["class", "interface"] = "class",
) -> tuple[list[ParsedSymbol], list[ParsedEdge], ParseDiagnostics, Tree, bytes]:
    """Parses Terraform HCL once and returns AST symbols, edges, diagnostics, tree, and bytes."""
    code_bytes = code.encode("utf-8")
    tree = _get_hcl_parser().parse(code_bytes)
    has_err, err_cnt, miss_cnt = inspect_tree_errors(tree.root_node)
    if has_err:
        reason = (
            f"Partial/invalid HCL syntax in '{file_path or '<memory>'}': "
            f"error_nodes={err_cnt}, missing_nodes={miss_cnt}"
        )
        logger.warning("%s; abstaining from partial AST extraction", reason)
        return (
            [],
            [],
            ParseDiagnostics(
                parser_backend="tree-sitter-hcl",
                has_parse_error=True,
                abstained=True,
                low_confidence_fallback=False,
                error_node_count=err_cnt,
                missing_node_count=miss_cnt,
                reason=reason,
            ),
            tree,
            code_bytes,
        )

    symbols: list[ParsedSymbol] = []
    edges: list[ParsedEdge] = []
    top_body: Node | None = None
    for child in tree.root_node.children:
        if child.type == "body":
            top_body = child
            break
    if top_body is None:
        return symbols, edges, ParseDiagnostics(parser_backend="tree-sitter-hcl"), tree, code_bytes

    def _qualify(bare_name: str) -> str:
        if repo_name:
            return f"{repo_name}::{module_prefix}::{bare_name}"
        return f"{module_prefix}.{bare_name}" if module_prefix else bare_name

    for block in top_body.children:
        if block.type != "block":
            continue
        block_children = block.children
        block_type_node: Node | None = None
        body_node: Node | None = None
        for c in block_children:
            if c.type == "identifier" and block_type_node is None:
                block_type_node = c
            elif c.type == "body" and body_node is None:
                body_node = c
        if block_type_node is None:
            continue
        block_type = _node_text(block_type_node, code_bytes).strip()
        if block_type not in ("resource", "data", "module", "variable", "output", "provider"):
            continue

        labels: list[str] = []
        for child in block_children:
            ctype = child.type
            if ctype == "block_start":
                break
            if ctype == "string_lit":
                labels.append(_unquote_hcl_string(child, code_bytes))
            elif ctype == "identifier" and child is not block_type_node:
                labels.append(_node_text(child, code_bytes).strip())

        if not labels:
            continue
        first_label = labels[0]
        second_label = labels[1] if len(labels) > 1 else None

        if block_type in ("resource", "data"):
            bare = f"{first_label}.{second_label}" if second_label else first_label
            sym_display = second_label or first_label
            sym_kind: Literal["function", "class", "method", "interface", "type_alias"] = "class"
        elif block_type == "module":
            bare = f"module.{first_label}"
            sym_display = first_label
            sym_kind = module_symbol_kind
        elif block_type in ("variable", "output"):
            bare = first_label
            sym_display = first_label
            sym_kind = "type_alias"
        else:  # provider
            bare = f"provider.{first_label}"
            sym_display = first_label
            sym_kind = "interface"

        repo_qual = _qualify(bare)
        start_line = block.start_point[0] + 1
        end_line = block.end_point[0] + 1
        nl_pos = code_bytes.find(b"\n", block.start_byte, block.end_byte)
        search_end = nl_pos if nl_pos != -1 else block.end_byte
        cr_pos = code_bytes.find(b"\r", block.start_byte, search_end)
        line_end = cr_pos if cr_pos != -1 else search_end
        sig_line = (
            code_bytes[block.start_byte : line_end].decode("utf-8", errors="replace").strip()
            or f"{block_type} {bare}"
        )

        symbols.append(
            ParsedSymbol(
                name=sym_display,
                qualified_name=repo_qual,
                symbol_kind=sym_kind,
                start_line=start_line,
                end_line=end_line,
                start_byte=block.start_byte,
                end_byte=block.end_byte,
                signature=sig_line,
                signature_skeleton=f"{block_type} {bare}",
                contextual_summary=(
                    f"Terraform {block_type} '{bare}' in {file_path}"
                    if file_path
                    else f"Terraform {block_type} {bare}: {sig_line}"
                ),
                repo_name=repo_name,
                repo_qualified_id=repo_qual if repo_name else None,
                bare_name=bare,
            )
        )

        if body_node is None:
            continue

        def _visit_body(
            b_node: Node,
            source_qual: str = repo_qual,
            self_bare: str = bare,
        ) -> None:
            for item in b_node.children:
                itype = item.type
                if itype == "attribute":
                    attr_id_node: Node | None = None
                    expr_node: Node | None = None
                    for c in item.children:
                        if c.type == "identifier" and attr_id_node is None:
                            attr_id_node = c
                        elif c.type == "expression" and expr_node is None:
                            expr_node = c
                    attr_name = _node_text(attr_id_node, code_bytes).strip()
                    if expr_node is None:
                        continue
                    chains = _collect_hcl_attr_chains(expr_node, code_bytes)
                    if attr_name == "depends_on":
                        for parts in chains:
                            dep_bare = ".".join(parts[:2]) if len(parts) >= 2 else parts[0]
                            edges.append(
                                ParsedEdge(
                                    source_qualified_name=source_qual,
                                    target_qualified_name=_qualify(dep_bare),
                                    edge_kind="PROVISIONED_BY",
                                )
                            )
                        continue

                    for parts in chains:
                        head = parts[0]
                        if head == "var" and len(parts) >= 2:
                            edges.append(
                                ParsedEdge(
                                    source_qualified_name=source_qual,
                                    target_qualified_name=_qualify(parts[1]),
                                    edge_kind="USES_TYPE",
                                )
                            )
                        elif head == "data" and len(parts) >= 3:
                            d_bare = f"{parts[1]}.{parts[2]}"
                            if d_bare != self_bare:
                                edges.append(
                                    ParsedEdge(
                                        source_qualified_name=source_qual,
                                        target_qualified_name=_qualify(d_bare),
                                        edge_kind="CALLS",
                                    )
                                )
                        elif head == "module" and len(parts) >= 2:
                            m_bare = f"module.{parts[1]}"
                            if m_bare != self_bare:
                                edges.append(
                                    ParsedEdge(
                                        source_qualified_name=source_qual,
                                        target_qualified_name=_qualify(m_bare),
                                        edge_kind="CALLS",
                                    )
                                )
                        elif (
                            head
                            not in (
                                "var",
                                "local",
                                "count",
                                "each",
                                "true",
                                "false",
                                "null",
                                "self",
                                "path",
                                "terraform",
                                "data",
                            )
                            and len(parts) >= 2
                            and "_" in head
                        ):
                            t_bare = f"{head}.{parts[1]}"
                            if t_bare != self_bare:
                                edge_kind = "CALLS"
                                if "service_account" in head:
                                    edge_kind = "BINDS_SERVICE_ACCOUNT"
                                elif "secret" in head:
                                    edge_kind = "ACCESSES_SECRET"
                                edges.append(
                                    ParsedEdge(
                                        source_qualified_name=source_qual,
                                        target_qualified_name=_qualify(t_bare),
                                        edge_kind=edge_kind,
                                    )
                                )
                elif itype == "block":
                    nested_body: Node | None = None
                    for c in item.children:
                        if c.type == "body":
                            nested_body = c
                            break
                    if nested_body is not None:
                        _visit_body(nested_body, source_qual, self_bare)

        _visit_body(body_node, repo_qual, bare)

    return symbols, edges, ParseDiagnostics(parser_backend="tree-sitter-hcl"), tree, code_bytes


def parse_hcl_ast(
    code: str,
    file_path: str = "",
    repo_name: str = "",
    module_prefix: str = "",
    *,
    module_symbol_kind: Literal["class", "interface"] = "class",
) -> tuple[list[ParsedSymbol], list[ParsedEdge], ParseDiagnostics]:
    """Parses Terraform HCL using tree-sitter-hcl and fails closed on syntax errors."""
    symbols, edges, diag, _tree, _code_bytes = parse_hcl_ast_with_tree(
        code=code,
        file_path=file_path,
        repo_name=repo_name,
        module_prefix=module_prefix,
        module_symbol_kind=module_symbol_kind,
    )
    return symbols, edges, diag


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


class ASTExtractor:
    """Extracts symbols and relations from polyglot enterprise repositories."""

    def __init__(self) -> None:
        self.py_lang = Language(tspython.language())
        self.ts_lang = Language(tstypescript.language_typescript())
        self.tsx_lang = Language(tstypescript.language_tsx())
        self.py_parser = Parser(self.py_lang)
        self.ts_parser = Parser(self.ts_lang)
        self.tsx_parser = Parser(self.tsx_lang)
        self._active_tree: Tree | None = None
        self.last_parse_diagnostics: ParseDiagnostics | None = None

    def parse_source(
        self,
        code: str,
        language: str,
        file_path: str = "",
        module_prefix: str = "",
        repo_name: str = "",
    ) -> tuple[list[ParsedSymbol], list[ParsedEdge]]:
        """Parses source code into symbols and directed edges."""
        if not module_prefix and file_path:
            module_prefix = _derive_module_prefix(file_path)

        self.last_parse_diagnostics = None

        # ---------------------------------------------------------------------
        # F10 Integration Bridge: Check Declarative PluginRegistry First
        # ---------------------------------------------------------------------
        try:
            from context_scout.plugins.registry import PluginRegistry

            plugin = None
            if file_path:
                plugin = PluginRegistry.get_plugin_for_file(file_path)
            if plugin is None and language:
                lang_key = _LANGUAGE_ALIASES.get(language.lower(), language.lower())
                plugin = PluginRegistry.get_language_plugin(lang_key)

            if plugin is not None and plugin.language_id == "react_redux":
                lang_norm = language.lower()
                if lang_norm in ("typescript", "ts", "javascript", "js") and not (
                    file_path.lower().endswith((".tsx", ".jsx"))
                ):
                    react_redux_markers = (
                        "createSlice",
                        "createSelector",
                        "useSelector",
                        "useAppSelector",
                        "useDispatch",
                        "useAppDispatch",
                        "from 'react'",
                        'from "react"',
                        "react-redux",
                        "@reduxjs/toolkit",
                    )
                    if not any(marker in code for marker in react_redux_markers):
                        plugin = None

            if plugin is not None:
                symbols, edges = plugin.extract_ast(
                    code=code,
                    file_path=file_path,
                    repo_name=repo_name,
                    module_prefix=module_prefix,
                )
                self.last_parse_diagnostics = getattr(plugin, "last_parse_diagnostics", None)
                return symbols, edges
        except Exception as bridge_exc:
            logger.warning(
                "Plugin delegation for '%s' (%s) fell through to built-in: %s",
                file_path,
                language,
                bridge_exc,
            )

        was_gc = gc.isenabled()
        gc.disable()
        try:
            return self._parse_source_builtin(
                code=code,
                language=language,
                file_path=file_path,
                module_prefix=module_prefix,
                repo_name=repo_name,
            )
        finally:
            if was_gc:
                gc.enable()

    def _parse_source_builtin(
        self,
        code: str,
        language: str,
        file_path: str = "",
        module_prefix: str = "",
        repo_name: str = "",
    ) -> tuple[list[ParsedSymbol], list[ParsedEdge]]:
        code_bytes = code.encode("utf-8")
        lang_lower = language.lower()

        raw_symbols: list[ParsedSymbol]
        raw_edges: list[ParsedEdge]

        def _parse_and_check(parser: Parser, backend: str) -> Node | None:
            self._active_tree = parser.parse(code_bytes)
            root = self._active_tree.root_node
            has_err, err_cnt, miss_cnt = inspect_tree_errors(root)
            if has_err:
                logger.warning(
                    "Tree-sitter parse error in %s (%s): errors=%d missing=%d; abstaining.",
                    file_path or "<source>",
                    backend,
                    err_cnt,
                    miss_cnt,
                )
                self.last_parse_diagnostics = ParseDiagnostics(
                    parser_backend=backend,
                    has_parse_error=True,
                    abstained=True,
                    error_node_count=err_cnt,
                    missing_node_count=miss_cnt,
                    reason=f"syntax_error:errors={err_cnt},missing={miss_cnt}",
                )
                return None
            self.last_parse_diagnostics = ParseDiagnostics(parser_backend=backend)
            return root

        if lang_lower in ("python", "py"):
            root = _parse_and_check(self.py_parser, "tree-sitter-python")
            if root is None:
                return [], []
            raw_symbols, raw_edges = self._extract_python(
                root, code_bytes, module_prefix, file_path
            )
        elif lang_lower in ("typescript", "ts", "javascript", "js", "mjs", "cjs"):
            root = _parse_and_check(self.ts_parser, "tree-sitter-typescript")
            if root is None:
                return [], []
            raw_symbols, raw_edges = self._extract_typescript(
                root, code_bytes, module_prefix, file_path, is_tsx=False
            )
        elif lang_lower in ("tsx", "jsx"):
            root = _parse_and_check(self.tsx_parser, "tree-sitter-tsx")
            if root is None:
                return [], []
            raw_symbols, raw_edges = self._extract_typescript(
                root, code_bytes, module_prefix, file_path, is_tsx=True
            )
        elif lang_lower == "sql" or file_path.endswith(".sql"):
            self.last_parse_diagnostics = ParseDiagnostics(
                parser_backend="regex-sql", low_confidence_fallback=True
            )
            raw_symbols, raw_edges = self._extract_sql(code, code_bytes, module_prefix)
        elif lang_lower in ("yaml", "yml") or file_path.endswith((".yaml", ".yml")):
            self.last_parse_diagnostics = ParseDiagnostics(
                parser_backend="regex-yaml", low_confidence_fallback=True
            )
            raw_symbols, raw_edges = self._extract_yaml(code, code_bytes, module_prefix, file_path)
        elif lang_lower in ("terraform", "hcl", "tf") or file_path.endswith(".tf"):
            raw_symbols, raw_edges = self._extract_terraform(code, code_bytes, module_prefix)
        elif lang_lower in ("go", "golang") or file_path.endswith(".go"):
            self.last_parse_diagnostics = ParseDiagnostics(
                parser_backend="regex-go", low_confidence_fallback=True
            )
            raw_symbols, raw_edges = self._extract_compiled_lang(
                code, code_bytes, module_prefix, "go"
            )
        elif lang_lower in ("rust", "rs") or file_path.endswith(".rs"):
            self.last_parse_diagnostics = ParseDiagnostics(
                parser_backend="regex-rust", low_confidence_fallback=True
            )
            raw_symbols, raw_edges = self._extract_compiled_lang(
                code, code_bytes, module_prefix, "rust"
            )
        elif lang_lower == "java" or file_path.endswith(".java"):
            self.last_parse_diagnostics = ParseDiagnostics(
                parser_backend="regex-java", low_confidence_fallback=True
            )
            raw_symbols, raw_edges = self._extract_compiled_lang(
                code, code_bytes, module_prefix, "java"
            )
        elif lang_lower in ("markdown", "md") or file_path.endswith(".md"):
            self.last_parse_diagnostics = ParseDiagnostics(
                parser_backend="regex-markdown", low_confidence_fallback=True
            )
            raw_symbols, raw_edges = self._extract_markdown(code, code_bytes, module_prefix)
        else:
            if file_path.endswith(".py"):
                root = _parse_and_check(self.py_parser, "tree-sitter-python")
                if root is None:
                    return [], []
                raw_symbols, raw_edges = self._extract_python(
                    root, code_bytes, module_prefix, file_path
                )
            elif file_path.endswith((".tsx", ".jsx")):
                root = _parse_and_check(self.tsx_parser, "tree-sitter-tsx")
                if root is None:
                    return [], []
                raw_symbols, raw_edges = self._extract_typescript(
                    root, code_bytes, module_prefix, file_path, is_tsx=True
                )
            else:
                root = _parse_and_check(self.ts_parser, "tree-sitter-typescript")
                if root is None:
                    return [], []
                raw_symbols, raw_edges = self._extract_typescript(
                    root, code_bytes, module_prefix, file_path, is_tsx=False
                )

        if not repo_name:
            return raw_symbols, raw_edges

        # Repo-qualify symbols and edges (F09)
        mod = module_prefix if module_prefix else _derive_module_prefix(file_path) or "root"
        qual_map: dict[str, str] = {}
        new_symbols: list[ParsedSymbol] = []
        for s in raw_symbols:
            bare = s.name
            if module_prefix and s.qualified_name.startswith(module_prefix):
                suffix = s.qualified_name[len(module_prefix) :].lstrip(".")
                bare = suffix or s.name
            elif "." in s.qualified_name:
                bare = s.qualified_name.split(".")[-1]

            repo_qual_id = f"{repo_name}::{mod}::{bare}"
            qual_map[s.qualified_name] = repo_qual_id
            new_symbols.append(
                ParsedSymbol(
                    name=s.name,
                    qualified_name=repo_qual_id,
                    symbol_kind=s.symbol_kind,
                    start_line=s.start_line,
                    end_line=s.end_line,
                    start_byte=s.start_byte,
                    end_byte=s.end_byte,
                    signature=s.signature,
                    signature_skeleton=s.signature_skeleton,
                    docstring=s.docstring,
                    contextual_summary=s.contextual_summary,
                    repo_name=repo_name,
                    repo_qualified_id=repo_qual_id,
                    bare_name=bare,
                )
            )

        new_edges: list[ParsedEdge] = []
        for e in raw_edges:
            src_qual = qual_map.get(
                e.source_qualified_name, f"{repo_name}::{mod}::{e.source_qualified_name}"
            )
            tgt_qual = qual_map.get(e.target_qualified_name, e.target_qualified_name)
            is_cross = e.is_cross_repo or (
                "::" in tgt_qual and not tgt_qual.startswith(f"{repo_name}::")
            )
            new_edges.append(
                ParsedEdge(
                    source_qualified_name=src_qual,
                    target_qualified_name=tgt_qual,
                    edge_kind=e.edge_kind,
                    target_repo_qualified_id=qual_map.get(e.target_qualified_name),
                    is_cross_repo=is_cross,
                )
            )

        return new_symbols, new_edges

    def _extract_python(
        self,
        root: Node,
        code_bytes: bytes,
        module_prefix: str,
        file_path: str,
    ) -> tuple[list[ParsedSymbol], list[ParsedEdge]]:
        symbols: list[ParsedSymbol] = []
        edges: list[ParsedEdge] = []
        file_imports: list[str] = []

        # 1. Collect imports
        for child in root.children:
            if child.type == "import_statement":
                for sub in child.children:
                    if sub.type == "dotted_name":
                        file_imports.append(_node_text(sub))
            elif child.type == "import_from_statement":
                module_node = child.child_by_field_name("module_name")
                mod_text = _node_text(module_node)
                for sub in child.children:
                    if sub.type == "dotted_name" and sub != module_node:
                        sub_text = _node_text(sub)
                        target = f"{mod_text}.{sub_text}" if mod_text else sub_text
                        file_imports.append(target)
                    elif sub.type == "import_list":
                        for alias in sub.children:
                            if alias.type == "dotted_name":
                                alias_text = _node_text(alias)
                                target = f"{mod_text}.{alias_text}" if mod_text else alias_text
                                file_imports.append(target)
                            elif alias.type == "aliased_import":
                                orig = alias.child_by_field_name("name")
                                if orig:
                                    orig_text = _node_text(orig)
                                    target = f"{mod_text}.{orig_text}" if mod_text else orig_text
                                    file_imports.append(target)

        # 2. Extract top-level symbols and their children
        for child in root.children:
            node = child
            if node.type == "decorated_definition":
                inner = node.child_by_field_name("definition")
                if inner:
                    node = inner

            if node.type == "class_definition":
                self._extract_python_class(
                    node, child, code_bytes, module_prefix, symbols, edges, file_imports
                )
            elif node.type == "function_definition":
                self._extract_python_function(
                    node,
                    child,
                    code_bytes,
                    module_prefix,
                    symbols,
                    edges,
                    file_imports,
                    is_method=False,
                )
            elif node.type == "expression_statement" and node.children:
                first_expr = node.children[0]
                if first_expr.type == "assignment":
                    self._extract_python_module_assignment(
                        first_expr, child, code_bytes, module_prefix, symbols, edges
                    )
            elif node.type == "assignment":
                self._extract_python_module_assignment(
                    node, child, code_bytes, module_prefix, symbols, edges
                )

        return symbols, edges

    def _extract_python_module_assignment(
        self,
        node: Node,
        outer_node: Node,
        code_bytes: bytes,
        module_prefix: str,
        symbols: list[ParsedSymbol],
        edges: list[ParsedEdge],
    ) -> None:
        left_node = node.child_by_field_name("left")
        right_node = node.child_by_field_name("right")
        if not left_node or not right_node or left_node.type != "identifier":
            return
        var_name = _node_text(left_node, code_bytes).strip()
        if len(var_name) < 3 or var_name.startswith("__") or var_name.lower() in {"logger", "log"}:
            return

        right_text = _node_text(right_node, code_bytes).strip()
        is_dict_or_collection = right_node.type in ("dictionary", "tuple", "list", "set")
        is_upper_constant = var_name.isupper()
        if not (is_dict_or_collection or is_upper_constant):
            return

        # Link any module-local classes/functions referenced in this dispatch table/registry
        referenced_module_symbols = [
            s for s in symbols if re.search(rf"\b{re.escape(s.name)}\b", right_text)
        ]
        if not referenced_module_symbols and not is_upper_constant:
            return

        qual_name = f"{module_prefix}.{var_name}" if module_prefix else var_name
        start_line = _point_row(outer_node.start_point) + 1
        end_line = _point_row(outer_node.end_point) + 1
        compact_rhs = re.sub(r"\s+", " ", right_text)[:240]
        sig_skeleton = f"{var_name} = {compact_rhs}"

        # Enrich referenced classes/functions (and their methods) with dispatch keys
        for pair_match in re.finditer(
            r"""['"]([A-Za-z0-9_\-\.]+)['"]\s*:\s*([A-Za-z_][A-Za-z0-9_]*)""",
            right_text,
        ):
            dispatch_key, target_ident = pair_match.group(1), pair_match.group(2)
            for sym in symbols:
                if (
                    sym.name == target_ident
                    or sym.qualified_name.endswith(f".{target_ident}.__init__")
                    or sym.qualified_name.startswith(f"{module_prefix}.{target_ident}.")
                ):
                    tag = f"[dispatch {var_name}: {dispatch_key}]"
                    if tag not in sym.contextual_summary:
                        sym.contextual_summary = f"{sym.contextual_summary} {tag}".strip()

        for ref_sym in referenced_module_symbols:
            edges.append(
                ParsedEdge(
                    source_qualified_name=qual_name,
                    target_qualified_name=ref_sym.qualified_name,
                    edge_kind="USES_TYPE",
                )
            )
            edges.append(
                ParsedEdge(
                    source_qualified_name=ref_sym.qualified_name,
                    target_qualified_name=qual_name,
                    edge_kind="USES_TYPE",
                )
            )

        # Emit named UPPER_CASE module constants and dispatch tables as AST symbols
        if is_upper_constant:
            symbols.append(
                ParsedSymbol(
                    name=var_name,
                    qualified_name=qual_name,
                    symbol_kind="type_alias",
                    start_line=start_line,
                    end_line=end_line,
                    start_byte=outer_node.start_byte,
                    end_byte=outer_node.end_byte,
                    signature=sig_skeleton,
                    signature_skeleton=sig_skeleton,
                    docstring="",
                    contextual_summary=sig_skeleton,
                )
            )

    def _extract_python_class(
        self,
        node: Node,
        outer_node: Node,
        code_bytes: bytes,
        module_prefix: str,
        symbols: list[ParsedSymbol],
        edges: list[ParsedEdge],
        file_imports: list[str],
    ) -> None:
        name_node = node.child_by_field_name("name")
        if not name_node:
            return
        class_name = _node_text(name_node)
        qual_name = f"{module_prefix}.{class_name}" if module_prefix else class_name

        start_line = _point_row(outer_node.start_point) + 1
        end_line = _point_row(outer_node.end_point) + 1
        start_byte = outer_node.start_byte
        end_byte = outer_node.end_byte

        superclasses_node = node.child_by_field_name("superclasses")
        superclasses_str = _node_text(superclasses_node)
        if superclasses_node:
            for sub in superclasses_node.children:
                if sub.type in ("identifier", "attribute"):
                    sup_name = _node_text(sub)
                    edges.append(
                        ParsedEdge(
                            source_qualified_name=qual_name,
                            target_qualified_name=sup_name,
                            edge_kind="INHERITS",
                        )
                    )

        docstring = ""
        body = node.child_by_field_name("body")
        if body and body.children:
            first_stmt = body.children[0]
            if (
                first_stmt.type == "expression_statement"
                and first_stmt.children
                and first_stmt.children[0].type == "string"
            ):
                docstring = _node_text(first_stmt.children[0]).strip("'\"\n ")

        sig_skeleton = f"class {class_name}{superclasses_str}:"
        summary = docstring.split("\n")[0] if docstring else sig_skeleton

        symbols.append(
            ParsedSymbol(
                name=class_name,
                qualified_name=qual_name,
                symbol_kind="class",
                start_line=start_line,
                end_line=end_line,
                start_byte=start_byte,
                end_byte=end_byte,
                signature=sig_skeleton,
                signature_skeleton=sig_skeleton,
                docstring=docstring,
                contextual_summary=summary,
            )
        )

        class_text = _node_text(outer_node, code_bytes)
        for imp in file_imports:
            imp_short = imp.split(".")[-1]
            if re.search(rf"\b{re.escape(imp_short)}\b", class_text):
                edges.append(
                    ParsedEdge(
                        source_qualified_name=qual_name,
                        target_qualified_name=imp,
                        edge_kind="IMPORTS",
                    )
                )

        if body:
            for child in body.children:
                m_node = child
                if m_node.type == "decorated_definition":
                    inner = m_node.child_by_field_name("definition")
                    if inner:
                        m_node = inner
                if m_node.type == "function_definition":
                    self._extract_python_function(
                        m_node,
                        child,
                        code_bytes,
                        qual_name,
                        symbols,
                        edges,
                        file_imports,
                        is_method=True,
                    )

    def _extract_python_function(
        self,
        node: Node,
        outer_node: Node,
        code_bytes: bytes,
        parent_qual_name: str,
        symbols: list[ParsedSymbol],
        edges: list[ParsedEdge],
        file_imports: list[str],
        is_method: bool,
    ) -> None:
        name_node = node.child_by_field_name("name")
        if not name_node:
            return
        fn_name = _node_text(name_node)
        qual_name = f"{parent_qual_name}.{fn_name}" if parent_qual_name else fn_name

        start_line = _point_row(outer_node.start_point) + 1
        end_line = _point_row(outer_node.end_point) + 1
        start_byte = outer_node.start_byte
        end_byte = outer_node.end_byte

        params_node = node.child_by_field_name("parameters")
        params_str = _node_text(params_node) if params_node else "()"
        ret_node = node.child_by_field_name("return_type")
        ret_str = f" -> {_node_text(ret_node)}" if ret_node else ""
        sig = f"def {fn_name}{params_str}{ret_str}:"

        docstring = ""
        body = node.child_by_field_name("body")
        if body and body.children:
            first_stmt = body.children[0]
            if (
                first_stmt.type == "expression_statement"
                and first_stmt.children
                and first_stmt.children[0].type == "string"
            ):
                docstring = _node_text(first_stmt.children[0]).strip("'\"\n ")

        summary = docstring.split("\n")[0] if docstring else sig

        called_targets: list[str] = []
        if body:
            called_targets = self._collect_python_calls(body, qual_name, edges)
        if called_targets:
            unique_calls = list(dict.fromkeys(c.split(".")[-1] for c in called_targets if c))[:6]
            if unique_calls:
                summary = f"{summary} [calls: {', '.join(unique_calls)}]"

        symbols.append(
            ParsedSymbol(
                name=fn_name,
                qualified_name=qual_name,
                symbol_kind="method" if is_method else "function",
                start_line=start_line,
                end_line=end_line,
                start_byte=start_byte,
                end_byte=end_byte,
                signature=sig,
                signature_skeleton=sig,
                docstring=docstring,
                contextual_summary=summary,
            )
        )

        type_sources: list[str] = []
        if params_node:
            type_sources.append(_node_text(params_node))
        if ret_node:
            type_sources.append(_node_text(ret_node))

        found_types: set[str] = set()
        for t_src in type_sources:
            for t_name in _extract_types(t_src, is_ts=False):
                if t_name not in found_types:
                    found_types.add(t_name)
                    edges.append(
                        ParsedEdge(
                            source_qualified_name=qual_name,
                            target_qualified_name=t_name,
                            edge_kind="USES_TYPE",
                        )
                    )

        if fn_name.startswith("test_"):
            target_candidate = fn_name[5:]
            if target_candidate:
                prod_qual = next(
                    (
                        s.qualified_name
                        for s in symbols
                        if s.name == target_candidate and s.qualified_name != qual_name
                    ),
                    target_candidate,
                )
                edges.append(
                    ParsedEdge(
                        source_qualified_name=prod_qual,
                        target_qualified_name=qual_name,
                        edge_kind="TESTED_BY",
                    )
                )

    def _collect_python_calls(
        self, root_node: Node, source_qual: str, edges: list[ParsedEdge]
    ) -> list[str]:
        called: list[str] = []
        stack = [root_node]
        while stack:
            curr = stack.pop()
            if curr.type == "call":
                fn = curr.child_by_field_name("function")
                if fn:
                    fn_txt = _node_text(fn).strip()
                    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$", fn_txt):
                        called.append(fn_txt)
                        edges.append(
                            ParsedEdge(
                                source_qualified_name=source_qual,
                                target_qualified_name=fn_txt,
                                edge_kind="CALLS",
                            )
                        )
            stack.extend(curr.children)
        return called

    def _extract_typescript(
        self,
        root: Node,
        code_bytes: bytes,
        module_prefix: str,
        file_path: str,
        is_tsx: bool,
    ) -> tuple[list[ParsedSymbol], list[ParsedEdge]]:
        symbols: list[ParsedSymbol] = []
        edges: list[ParsedEdge] = []
        file_imports: list[str] = []

        # 1. Collect imports
        for child in root.children:
            if child.type == "import_statement":
                for sub in child.children:
                    if sub.type == "import_clause":
                        for spec in sub.children:
                            if spec.type == "named_imports":
                                for import_spec in spec.children:
                                    if import_spec.type == "import_specifier":
                                        name = import_spec.child_by_field_name("name")
                                        if name:
                                            file_imports.append(_node_text(name))
                            elif spec.type == "identifier":
                                file_imports.append(_node_text(spec))

        # 2. Walk top-level declarations
        prev_comment = ""
        for child in root.children:
            if child.type == "comment":
                prev_comment = _node_text(child).strip("/* \t\n")
                continue

            node = child
            if node.type == "export_statement":
                for sub in node.children:
                    if sub.type in (
                        "interface_declaration",
                        "type_alias_declaration",
                        "class_declaration",
                        "function_declaration",
                        "lexical_declaration",
                    ):
                        node = sub
                        break

            if node.type == "interface_declaration":
                self._extract_ts_interface(node, child, module_prefix, symbols, edges, prev_comment)
            elif node.type == "type_alias_declaration":
                self._extract_ts_type_alias(
                    node, child, module_prefix, symbols, edges, prev_comment
                )
            elif node.type == "class_declaration":
                self._extract_ts_class(
                    node, child, module_prefix, symbols, edges, file_imports, prev_comment
                )
            elif node.type == "function_declaration":
                self._extract_ts_function(
                    node, child, module_prefix, symbols, edges, prev_comment, is_method=False
                )
            elif node.type == "lexical_declaration":
                self._extract_ts_lexical_function(
                    node, child, module_prefix, symbols, edges, prev_comment
                )

            prev_comment = ""

        return symbols, edges

    def _extract_ts_interface(
        self,
        node: Node,
        outer_node: Node,
        module_prefix: str,
        symbols: list[ParsedSymbol],
        edges: list[ParsedEdge],
        docstring: str,
    ) -> None:
        name_node = node.child_by_field_name("name")
        if not name_node:
            return
        name = _node_text(name_node)
        qual_name = f"{module_prefix}.{name}" if module_prefix else name

        start_line = _point_row(outer_node.start_point) + 1
        end_line = _point_row(outer_node.end_point) + 1
        start_byte = outer_node.start_byte
        end_byte = outer_node.end_byte

        for child in node.children:
            if child.type == "extends_type_clause":
                for sub in child.children:
                    if sub.type in ("type_identifier", "identifier"):
                        sup = _node_text(sub)
                        edges.append(
                            ParsedEdge(
                                source_qualified_name=qual_name,
                                target_qualified_name=sup,
                                edge_kind="INHERITS",
                            )
                        )

        body = node.child_by_field_name("body")
        if body:
            for prop in body.children:
                if prop.type == "property_signature":
                    t_node = prop.child_by_field_name("type")
                    if t_node:
                        for t_name in _extract_types(_node_text(t_node), is_ts=True):
                            edges.append(
                                ParsedEdge(
                                    source_qualified_name=qual_name,
                                    target_qualified_name=t_name,
                                    edge_kind="USES_TYPE",
                                )
                            )

        sig = f"interface {name}"
        symbols.append(
            ParsedSymbol(
                name=name,
                qualified_name=qual_name,
                symbol_kind="interface",
                start_line=start_line,
                end_line=end_line,
                start_byte=start_byte,
                end_byte=end_byte,
                signature=sig,
                signature_skeleton=sig,
                docstring=docstring,
                contextual_summary=docstring.split("\n")[0] if docstring else sig,
            )
        )

    def _extract_ts_type_alias(
        self,
        node: Node,
        outer_node: Node,
        module_prefix: str,
        symbols: list[ParsedSymbol],
        edges: list[ParsedEdge],
        docstring: str,
    ) -> None:
        name_node = node.child_by_field_name("name")
        if not name_node:
            return
        name = _node_text(name_node)
        qual_name = f"{module_prefix}.{name}" if module_prefix else name

        start_line = _point_row(outer_node.start_point) + 1
        end_line = _point_row(outer_node.end_point) + 1
        start_byte = outer_node.start_byte
        end_byte = outer_node.end_byte

        val_node = node.child_by_field_name("value")
        if val_node:
            for t_name in _extract_types(_node_text(val_node), is_ts=True):
                edges.append(
                    ParsedEdge(
                        source_qualified_name=qual_name,
                        target_qualified_name=t_name,
                        edge_kind="USES_TYPE",
                    )
                )

        sig = f"type {name} = ..."
        symbols.append(
            ParsedSymbol(
                name=name,
                qualified_name=qual_name,
                symbol_kind="type_alias",
                start_line=start_line,
                end_line=end_line,
                start_byte=start_byte,
                end_byte=end_byte,
                signature=sig,
                signature_skeleton=sig,
                docstring=docstring,
                contextual_summary=docstring.split("\n")[0] if docstring else sig,
            )
        )

    def _extract_ts_class(
        self,
        node: Node,
        outer_node: Node,
        module_prefix: str,
        symbols: list[ParsedSymbol],
        edges: list[ParsedEdge],
        file_imports: list[str],
        docstring: str,
    ) -> None:
        name_node = node.child_by_field_name("name")
        if not name_node:
            return
        name = _node_text(name_node)
        qual_name = f"{module_prefix}.{name}" if module_prefix else name

        start_line = _point_row(outer_node.start_point) + 1
        end_line = _point_row(outer_node.end_point) + 1
        start_byte = outer_node.start_byte
        end_byte = outer_node.end_byte

        for child in node.children:
            if child.type == "class_heritage":
                for clause in child.children:
                    if clause.type in ("extends_clause", "implements_clause"):
                        for sub in clause.children:
                            if sub.type in ("type_identifier", "identifier"):
                                sup = _node_text(sub)
                                edges.append(
                                    ParsedEdge(
                                        source_qualified_name=qual_name,
                                        target_qualified_name=sup,
                                        edge_kind="INHERITS",
                                    )
                                )

        sig = f"class {name}"
        symbols.append(
            ParsedSymbol(
                name=name,
                qualified_name=qual_name,
                symbol_kind="class",
                start_line=start_line,
                end_line=end_line,
                start_byte=start_byte,
                end_byte=end_byte,
                signature=sig,
                signature_skeleton=sig,
                docstring=docstring,
                contextual_summary=docstring.split("\n")[0] if docstring else sig,
            )
        )

        body = node.child_by_field_name("body")
        if body:
            for child in body.children:
                if child.type == "method_definition":
                    self._extract_ts_method(child, qual_name, symbols, edges)

    def _extract_ts_method(
        self,
        node: Node,
        class_qual: str,
        symbols: list[ParsedSymbol],
        edges: list[ParsedEdge],
    ) -> None:
        name_node = node.child_by_field_name("name")
        if not name_node:
            return
        m_name = _node_text(name_node)
        qual_name = f"{class_qual}.{m_name}"

        start_line = _point_row(node.start_point) + 1
        end_line = _point_row(node.end_point) + 1
        start_byte = node.start_byte
        end_byte = node.end_byte

        params_node = node.child_by_field_name("parameters")
        ret_node = node.child_by_field_name("return_type")
        params_str = _node_text(params_node) if params_node else "()"
        ret_str = _node_text(ret_node) if ret_node else ""
        sig = f"{m_name}{params_str}{ret_str}"

        symbols.append(
            ParsedSymbol(
                name=m_name,
                qualified_name=qual_name,
                symbol_kind="method",
                start_line=start_line,
                end_line=end_line,
                start_byte=start_byte,
                end_byte=end_byte,
                signature=sig,
                signature_skeleton=sig,
                docstring="",
                contextual_summary=sig,
            )
        )

        if params_node:
            for t in _extract_types(_node_text(params_node), is_ts=True):
                edges.append(
                    ParsedEdge(
                        source_qualified_name=qual_name,
                        target_qualified_name=t,
                        edge_kind="USES_TYPE",
                    )
                )
        if ret_node:
            for t in _extract_types(_node_text(ret_node), is_ts=True):
                edges.append(
                    ParsedEdge(
                        source_qualified_name=qual_name,
                        target_qualified_name=t,
                        edge_kind="USES_TYPE",
                    )
                )

        body = node.child_by_field_name("body")
        if body:
            self._collect_ts_calls(body, qual_name, edges)

    def _extract_ts_function(
        self,
        node: Node,
        outer_node: Node,
        module_prefix: str,
        symbols: list[ParsedSymbol],
        edges: list[ParsedEdge],
        docstring: str,
        is_method: bool = False,
    ) -> None:
        name_node = node.child_by_field_name("name")
        if not name_node:
            return
        fn_name = _node_text(name_node)
        qual_name = f"{module_prefix}.{fn_name}" if module_prefix else fn_name

        start_line = _point_row(outer_node.start_point) + 1
        end_line = _point_row(outer_node.end_point) + 1
        start_byte = outer_node.start_byte
        end_byte = outer_node.end_byte

        params_node = node.child_by_field_name("parameters")
        ret_node = node.child_by_field_name("return_type")
        params_str = _node_text(params_node) if params_node else "()"
        ret_str = _node_text(ret_node) if ret_node else ""
        sig = f"function {fn_name}{params_str}{ret_str}"

        symbols.append(
            ParsedSymbol(
                name=fn_name,
                qualified_name=qual_name,
                symbol_kind="method" if is_method else "function",
                start_line=start_line,
                end_line=end_line,
                start_byte=start_byte,
                end_byte=end_byte,
                signature=sig,
                signature_skeleton=sig,
                docstring=docstring,
                contextual_summary=docstring.split("\n")[0] if docstring else sig,
            )
        )

        if params_node:
            for t in _extract_types(_node_text(params_node), is_ts=True):
                edges.append(
                    ParsedEdge(
                        source_qualified_name=qual_name,
                        target_qualified_name=t,
                        edge_kind="USES_TYPE",
                    )
                )
        if ret_node:
            for t in _extract_types(_node_text(ret_node), is_ts=True):
                edges.append(
                    ParsedEdge(
                        source_qualified_name=qual_name,
                        target_qualified_name=t,
                        edge_kind="USES_TYPE",
                    )
                )

        body = node.child_by_field_name("body")
        if body:
            self._collect_ts_calls(body, qual_name, edges)

        if fn_name.startswith("test") or fn_name.startswith("it_"):
            target_candidate = fn_name.replace("test_", "").replace("test", "")
            if target_candidate:
                prod_qual = next(
                    (
                        s.qualified_name
                        for s in symbols
                        if s.name == target_candidate and s.qualified_name != qual_name
                    ),
                    target_candidate,
                )
                edges.append(
                    ParsedEdge(
                        source_qualified_name=prod_qual,
                        target_qualified_name=qual_name,
                        edge_kind="TESTED_BY",
                    )
                )

    def _extract_ts_lexical_function(
        self,
        node: Node,
        outer_node: Node,
        module_prefix: str,
        symbols: list[ParsedSymbol],
        edges: list[ParsedEdge],
        docstring: str,
    ) -> None:
        for declarator in node.children:
            if declarator.type == "variable_declarator":
                name_node = declarator.child_by_field_name("name")
                val_node = declarator.child_by_field_name("value")
                if not name_node or not val_node:
                    continue
                if val_node.type in ("arrow_function", "function_expression"):
                    fn_name = _node_text(name_node)
                    qual_name = f"{module_prefix}.{fn_name}" if module_prefix else fn_name

                    start_line = _point_row(outer_node.start_point) + 1
                    end_line = _point_row(outer_node.end_point) + 1
                    start_byte = outer_node.start_byte
                    end_byte = outer_node.end_byte

                    params_node = val_node.child_by_field_name("parameters")
                    ret_node = val_node.child_by_field_name("return_type")
                    params_str = _node_text(params_node) if params_node else "()"
                    ret_str = _node_text(ret_node) if ret_node else ""
                    sig = f"const {fn_name} = {params_str}{ret_str} => ..."

                    symbols.append(
                        ParsedSymbol(
                            name=fn_name,
                            qualified_name=qual_name,
                            symbol_kind="function",
                            start_line=start_line,
                            end_line=end_line,
                            start_byte=start_byte,
                            end_byte=end_byte,
                            signature=sig,
                            signature_skeleton=sig,
                            docstring=docstring,
                            contextual_summary=docstring.split("\n")[0] if docstring else sig,
                        )
                    )

                    if params_node:
                        for t in _extract_types(_node_text(params_node), is_ts=True):
                            edges.append(
                                ParsedEdge(
                                    source_qualified_name=qual_name,
                                    target_qualified_name=t,
                                    edge_kind="USES_TYPE",
                                )
                            )
                    if ret_node:
                        for t in _extract_types(_node_text(ret_node), is_ts=True):
                            edges.append(
                                ParsedEdge(
                                    source_qualified_name=qual_name,
                                    target_qualified_name=t,
                                    edge_kind="USES_TYPE",
                                )
                            )

                    body = val_node.child_by_field_name("body")
                    if body:
                        self._collect_ts_calls(body, qual_name, edges)

    def _collect_ts_calls(self, root_node: Node, source_qual: str, edges: list[ParsedEdge]) -> None:
        stack = [root_node]
        while stack:
            curr = stack.pop()
            if curr.type == "call_expression":
                fn = curr.child_by_field_name("function")
                if fn:
                    call_target = _node_text(fn)
                    if call_target in ("describe", "it", "test"):
                        args = curr.child_by_field_name("arguments")
                        if args and args.children:
                            for arg in args.children:
                                if arg.type == "string":
                                    edges.append(
                                        ParsedEdge(
                                            source_qualified_name=_node_text(arg).strip("'\"`"),
                                            target_qualified_name=source_qual,
                                            edge_kind="TESTED_BY",
                                        )
                                    )
                                    break
                    else:
                        edges.append(
                            ParsedEdge(
                                source_qualified_name=source_qual,
                                target_qualified_name=call_target,
                                edge_kind="CALLS",
                            )
                        )
            stack.extend(curr.children)

    def _extract_sql(
        self,
        code: str,
        code_bytes: bytes,
        module_prefix: str,
    ) -> tuple[list[ParsedSymbol], list[ParsedEdge]]:
        """Extracts DDL tables, functions, views, policies, and FK edges from SQL files."""
        _ = code_bytes
        symbols: list[ParsedSymbol] = []
        edges: list[ParsedEdge] = []
        lines = code.splitlines()
        line_offsets: list[int] = [0]
        for ln in lines:
            line_offsets.append(line_offsets[-1] + len(ln.encode("utf-8")) + 1)

        stmt_pattern = re.compile(
            r"^\s*CREATE\s+(?:OR\s+REPLACE\s+)?(?:MATERIALIZED\s+|UNIQUE\s+)?"
            r"(TABLE|FUNCTION|PROCEDURE|VIEW|POLICY|INDEX)\s+"
            r"(?:IF\s+NOT\s+EXISTS\s+)?([A-Za-z0-9_.\"]+)"
            r"(?:\s+ON\s+([A-Za-z0-9_.\"]+))?",
            re.IGNORECASE,
        )
        i = 0
        total_lines = len(lines)
        while i < total_lines:
            m = stmt_pattern.match(lines[i])
            if not m:
                i += 1
                continue
            ddl_type = m.group(1).upper()
            raw_name = m.group(2).strip('"').split(".")[-1]
            on_table = m.group(3).strip('"').split(".")[-1] if m.group(3) else None
            start_line = i + 1
            end_idx = i
            while end_idx < total_lines and ";" not in lines[end_idx]:
                end_idx += 1
            end_line = min(total_lines, end_idx + 1)
            block_text = "\n".join(lines[i:end_line])
            sig = lines[i].strip()
            qual = f"{module_prefix}.{raw_name}" if module_prefix else raw_name

            sym_kind: Literal["function", "class", "method", "interface", "type_alias"]
            if ddl_type == "TABLE":
                sym_kind = "class"
            elif ddl_type in ("FUNCTION", "PROCEDURE"):
                sym_kind = "function"
            elif ddl_type == "VIEW":
                sym_kind = "interface"
            elif ddl_type == "POLICY":
                sym_kind = "method"
            else:
                sym_kind = "type_alias"

            symbols.append(
                ParsedSymbol(
                    name=raw_name,
                    qualified_name=qual,
                    symbol_kind=sym_kind,
                    start_line=start_line,
                    end_line=end_line,
                    start_byte=line_offsets[start_line - 1],
                    end_byte=line_offsets[min(end_line, len(line_offsets) - 1)],
                    signature=sig,
                    signature_skeleton=sig,
                    docstring="",
                    contextual_summary=f"SQL {ddl_type} {raw_name}: {sig}",
                )
            )
            if on_table:
                target_q = f"{module_prefix}.{on_table}" if module_prefix else on_table
                edges.append(
                    ParsedEdge(
                        source_qualified_name=qual,
                        target_qualified_name=target_q,
                        edge_kind="USES_TYPE",
                    )
                )
            for ref_m in re.finditer(
                r"REFERENCES\s+([A-Za-z0-9_.\"]+)", block_text, flags=re.IGNORECASE
            ):
                ref_tbl = ref_m.group(1).strip('"').split(".")[-1]
                target_q = f"{module_prefix}.{ref_tbl}" if module_prefix else ref_tbl
                edges.append(
                    ParsedEdge(
                        source_qualified_name=qual,
                        target_qualified_name=target_q,
                        edge_kind="USES_TYPE",
                    )
                )
            i = max(i + 1, end_idx + 1)
        return symbols, edges

    def _extract_yaml(
        self,
        code: str,
        code_bytes: bytes,
        module_prefix: str,
        file_path: str,
    ) -> tuple[list[ParsedSymbol], list[ParsedEdge]]:
        """Extracts GitLab CI jobs, K8s manifests, Compose services, and YAML top-level blocks."""
        _ = (code_bytes, file_path)
        symbols: list[ParsedSymbol] = []
        edges: list[ParsedEdge] = []
        lines = code.splitlines()
        if not lines:
            return symbols, edges
        line_offsets: list[int] = [0]
        for ln in lines:
            line_offsets.append(line_offsets[-1] + len(ln.encode("utf-8")) + 1)

        top_keys: list[tuple[str, int]] = []
        key_re = re.compile(r"^([A-Za-z0-9_.-]+)\s*:")
        for idx, line in enumerate(lines):
            if line.startswith((" ", "\t", "#", "---")):
                continue
            km = key_re.match(line)
            if km:
                top_keys.append((km.group(1), idx + 1))

        for k_idx, (key_name, start_line) in enumerate(top_keys):
            end_line = top_keys[k_idx + 1][1] - 1 if k_idx + 1 < len(top_keys) else len(lines)
            while end_line > start_line and not lines[end_line - 1].strip():
                end_line -= 1
            block_lines = lines[start_line - 1 : end_line]
            block_text = "\n".join(block_lines)
            qual = f"{module_prefix}.{key_name}" if module_prefix else key_name
            sig = f"{key_name}:"
            summary_parts = [f"YAML block {key_name}"]
            sub_pat = r"^\s{2,4}(stage|image|kind|extends)\s*:\s*(.+)$"
            for sub_m in re.finditer(sub_pat, block_text, re.M):
                summary_parts.append(f"{sub_m.group(1)}={sub_m.group(2).strip()}")
            symbols.append(
                ParsedSymbol(
                    name=key_name,
                    qualified_name=qual,
                    symbol_kind="function" if "script:" in block_text else "type_alias",
                    start_line=start_line,
                    end_line=max(start_line, end_line),
                    start_byte=line_offsets[start_line - 1],
                    end_byte=line_offsets[min(max(start_line, end_line), len(line_offsets) - 1)],
                    signature=sig,
                    signature_skeleton=sig,
                    docstring="",
                    contextual_summary=" | ".join(summary_parts),
                )
            )
            ext_pat = r"^\s+(extends|stage)\s*:\s*([A-Za-z0-9_.-]+)"
            for ext_m in re.finditer(ext_pat, block_text, re.M):
                dep_kind = ext_m.group(1).strip()
                dep_name = ext_m.group(2).strip()
                target_q = f"{module_prefix}.{dep_name}" if module_prefix else dep_name
                edges.append(
                    ParsedEdge(
                        source_qualified_name=qual,
                        target_qualified_name=target_q,
                        edge_kind="INHERITS" if dep_kind == "extends" else "USES_TYPE",
                    )
                )
        return symbols, edges

    def _extract_terraform(
        self,
        code: str,
        code_bytes: bytes,
        module_prefix: str,
    ) -> tuple[list[ParsedSymbol], list[ParsedEdge]]:
        """Extract Terraform/HCL resource, module, variable, output, and data blocks."""
        _ = code_bytes
        symbols, edges, diag = parse_hcl_ast(
            code=code,
            module_prefix=module_prefix,
            module_symbol_kind="interface",
        )
        self.last_parse_diagnostics = diag
        return symbols, edges

    def _extract_compiled_lang(
        self,
        code: str,
        code_bytes: bytes,
        module_prefix: str,
        lang: str,
    ) -> tuple[list[ParsedSymbol], list[ParsedEdge]]:
        """Extracts structs, classes, interfaces, traits, and functions from Go, Rust, and Java."""
        _ = code_bytes
        symbols: list[ParsedSymbol] = []
        edges: list[ParsedEdge] = []
        lines = code.splitlines()
        line_offsets: list[int] = [0]
        for ln in lines:
            line_offsets.append(line_offsets[-1] + len(ln.encode("utf-8")) + 1)

        if lang == "go":
            decl_re = re.compile(
                r"^\s*(?:func\s+(?:\([^)]+\)\s+)?([A-Za-z_][A-Za-z0-9_]*)|type\s+([A-Za-z_][A-Za-z0-9_]*)\s+(struct|interface))"
            )
        elif lang == "rust":
            decl_re = re.compile(
                r"^\s*(?:pub(?:\([^)]+\))?\s+)?(?:async\s+)?(?:fn\s+([A-Za-z_][A-Za-z0-9_]*)|(?:struct|enum|trait)\s+([A-Za-z_][A-Za-z0-9_]*))"
            )
        else:
            decl_re = re.compile(
                r"^\s*(?:public|private|protected|static|final|abstract|\s)*"
                r"(?:(?:class|interface|enum)\s+([A-Za-z_][A-Za-z0-9_]*)|"
                r"[A-Za-z0-9_<>\[\]]+\s+([A-Za-z_][A-Za-z0-9_]*)\s*\([^;]*\{)"
            )

        i = 0
        total_lines = len(lines)
        while i < total_lines:
            m = decl_re.match(lines[i])
            if not m:
                i += 1
                continue
            fn_name = m.group(1)
            type_name = m.group(2)
            sym_name = fn_name or type_name or "symbol"
            is_type = bool(type_name and not fn_name)
            if lang == "java" and m.group(1):
                is_type = True
                sym_name = m.group(1)
            elif lang == "java" and m.group(2):
                is_type = False
                sym_name = m.group(2)
            start_line = i + 1
            brace_depth = lines[i].count("{") - lines[i].count("}")
            end_idx = i
            while end_idx + 1 < total_lines and (
                brace_depth > 0 or (end_idx == i and "{" not in lines[i])
            ):
                end_idx += 1
                brace_depth += lines[end_idx].count("{") - lines[end_idx].count("}")
                if end_idx - i > 200:
                    break
            end_line = end_idx + 1
            sig = lines[i].strip()
            qual = f"{module_prefix}.{sym_name}" if module_prefix else sym_name
            kind_label = "type" if is_type else "function"
            symbols.append(
                ParsedSymbol(
                    name=sym_name,
                    qualified_name=qual,
                    symbol_kind="class" if is_type else "function",
                    start_line=start_line,
                    end_line=end_line,
                    start_byte=line_offsets[start_line - 1],
                    end_byte=line_offsets[min(end_line, len(line_offsets) - 1)],
                    signature=sig,
                    signature_skeleton=sig,
                    docstring="",
                    contextual_summary=f"{lang} {kind_label} {sym_name}: {sig}",
                )
            )
            i = max(i + 1, end_idx + 1 if not is_type else i + 1)
        return symbols, edges

    def _extract_markdown(
        self,
        code: str,
        code_bytes: bytes,
        module_prefix: str,
    ) -> tuple[list[ParsedSymbol], list[ParsedEdge]]:
        """Extracts navigable heading sections and backticked symbol references from Markdown."""
        _ = code_bytes
        symbols: list[ParsedSymbol] = []
        edges: list[ParsedEdge] = []
        lines = code.splitlines()
        if not lines:
            return symbols, edges
        line_offsets: list[int] = [0]
        for ln in lines:
            line_offsets.append(line_offsets[-1] + len(ln.encode("utf-8")) + 1)

        headings: list[tuple[str, str, int]] = []
        h_re = re.compile(r"^(#{1,3})\s+(.+?)\s*$")
        in_fence = False
        for idx, line in enumerate(lines):
            if line.strip().startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            hm = h_re.match(line)
            if hm:
                raw_title = hm.group(2).strip()
                slug = re.sub(r"[^A-Za-z0-9_]+", "_", raw_title).strip("_") or f"section_{idx + 1}"
                headings.append((slug[:64], raw_title, idx + 1))

        for h_idx, (slug, raw_title, start_line) in enumerate(headings):
            end_line = headings[h_idx + 1][2] - 1 if h_idx + 1 < len(headings) else len(lines)
            qual = f"{module_prefix}.{slug}" if module_prefix else slug
            section_text = "\n".join(lines[start_line - 1 : end_line])
            symbols.append(
                ParsedSymbol(
                    name=slug,
                    qualified_name=qual,
                    symbol_kind="interface",
                    start_line=start_line,
                    end_line=max(start_line, end_line),
                    start_byte=line_offsets[start_line - 1],
                    end_byte=line_offsets[min(max(start_line, end_line), len(line_offsets) - 1)],
                    signature=f"# {raw_title}",
                    signature_skeleton=f"# {raw_title}",
                    docstring=raw_title,
                    contextual_summary=f"Markdown section '{raw_title}'",
                )
            )
            for bt_m in re.finditer(r"`([A-Za-z_][A-Za-z0-9_.]*)`", section_text):
                ref_tok = bt_m.group(1)
                if len(ref_tok) >= 4:
                    edges.append(
                        ParsedEdge(
                            source_qualified_name=qual,
                            target_qualified_name=ref_tok,
                            edge_kind="USES_TYPE",
                        )
                    )
        return symbols, edges


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


__all__ = ["ASTExtractor", "match_symbol_identifier"]
