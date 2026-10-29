"""Multi-language Tree-Sitter AST extractor for Python and TypeScript/TSX."""

import re
from typing import Final

import tree_sitter_python as tspython
import tree_sitter_typescript as tstypescript
from tree_sitter import Language, Node, Parser, Tree

from context_scout.ast_engine.models import ParsedEdge, ParsedSymbol

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
    for ext in (".py", ".ts", ".tsx", ".js", ".jsx"):
        if path.endswith(ext):
            path = path[: -len(ext)]
            break
    parts = [p for p in path.split("/") if p and p != "__init__"]
    return ".".join(parts)


class ASTExtractor:
    """Extracts symbols and relations from Python and TypeScript source code using Tree-Sitter."""

    def __init__(self) -> None:
        self.py_lang = Language(tspython.language())
        self.ts_lang = Language(tstypescript.language_typescript())
        self.tsx_lang = Language(tstypescript.language_tsx())
        self.py_parser = Parser(self.py_lang)
        self.ts_parser = Parser(self.ts_lang)
        self.tsx_parser = Parser(self.tsx_lang)
        self._active_tree: Tree | None = None

    def parse_source(
        self,
        code: str,
        language: str,
        file_path: str = "",
        module_prefix: str = "",
    ) -> tuple[list[ParsedSymbol], list[ParsedEdge]]:
        """Parses source code into symbols and directed edges."""
        if not module_prefix and file_path:
            module_prefix = _derive_module_prefix(file_path)

        code_bytes = code.encode("utf-8")
        lang_lower = language.lower()

        if lang_lower in ("python", "py"):
            self._active_tree = self.py_parser.parse(code_bytes)
            return self._extract_python(
                self._active_tree.root_node, code_bytes, module_prefix, file_path
            )
        elif lang_lower in ("typescript", "ts"):
            self._active_tree = self.ts_parser.parse(code_bytes)
            return self._extract_typescript(
                self._active_tree.root_node, code_bytes, module_prefix, file_path, is_tsx=False
            )
        elif lang_lower in ("tsx",):
            self._active_tree = self.tsx_parser.parse(code_bytes)
            return self._extract_typescript(
                self._active_tree.root_node, code_bytes, module_prefix, file_path, is_tsx=True
            )
        else:
            if file_path.endswith(".py"):
                self._active_tree = self.py_parser.parse(code_bytes)
                return self._extract_python(
                    self._active_tree.root_node, code_bytes, module_prefix, file_path
                )
            elif file_path.endswith(".tsx"):
                self._active_tree = self.tsx_parser.parse(code_bytes)
                return self._extract_typescript(
                    self._active_tree.root_node, code_bytes, module_prefix, file_path, is_tsx=True
                )
            else:
                self._active_tree = self.ts_parser.parse(code_bytes)
                return self._extract_typescript(
                    self._active_tree.root_node, code_bytes, module_prefix, file_path, is_tsx=False
                )

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

        return symbols, edges

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

        for imp in file_imports:
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

        if body:
            self._collect_python_calls(body, qual_name, edges)

        if fn_name.startswith("test_"):
            target_candidate = fn_name[5:]
            if target_candidate:
                edges.append(
                    ParsedEdge(
                        source_qualified_name=qual_name,
                        target_qualified_name=target_candidate,
                        edge_kind="TESTED_BY",
                    )
                )

    def _collect_python_calls(
        self, root_node: Node, source_qual: str, edges: list[ParsedEdge]
    ) -> None:
        stack = [root_node]
        while stack:
            curr = stack.pop()
            if curr.type == "call":
                fn = curr.child_by_field_name("function")
                if fn:
                    edges.append(
                        ParsedEdge(
                            source_qualified_name=source_qual,
                            target_qualified_name=_node_text(fn),
                            edge_kind="CALLS",
                        )
                    )
            stack.extend(curr.children)

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
                edges.append(
                    ParsedEdge(
                        source_qualified_name=qual_name,
                        target_qualified_name=target_candidate,
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
                                            source_qualified_name=source_qual,
                                            target_qualified_name=_node_text(arg).strip("'\"`"),
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
