"""Tree-sitter TypeScript/JavaScript AST symbol, call, and import walker."""

from __future__ import annotations

import re

from tree_sitter import Node

from context_scout.ast_engine.extractor_types import (
    _ASTExtractorStateBase,
    _extract_types,
    _node_text,
    _point_row,
)
from context_scout.ast_engine.models import ParsedEdge, ParsedSymbol


class _TypeScriptWalkerMixin(_ASTExtractorStateBase):
    """TypeScript/JS tree-sitter extraction mixin for ASTExtractor."""

    @staticmethod
    def _extract_ts_contract_tags(body_text: str) -> list[str]:
        """Extracts @route and @env cross-boundary tags from TypeScript/TSX call expressions."""
        tags: list[str] = []
        for fm in re.finditer(
            r"""fetch\(\s*['"`]([^'"`]+)['"`](?:\s*,\s*\{[\s\S]{0,240}?\bmethod\s*:\s*['"]([A-Za-z]+)['"])?""",
            body_text,
        ):
            url = fm.group(1).strip()
            if url.startswith(("/", "http://", "https://")):
                method = (fm.group(2) or "GET").upper()
                tag = f"@route {method} {url}"
                if tag not in tags:
                    tags.append(tag)
        for em in re.finditer(r"""EventSource\(\s*['"`]([^'"`]+)['"`]""", body_text):
            url = em.group(1).strip()
            if url.startswith(("/", "http://", "https://")):
                tag = f"@route GET {url}"
                if tag not in tags:
                    tags.append(tag)
        for am in re.finditer(
            r"""axios\.(get|post|put|delete|patch)\(\s*['"`]([^'"`]+)['"`]""",
            body_text,
            re.IGNORECASE,
        ):
            method, url = am.group(1).upper(), am.group(2).strip()
            if url.startswith(("/", "http://", "https://")):
                tag = f"@route {method} {url}"
                if tag not in tags:
                    tags.append(tag)
        for evm in re.finditer(
            r"""\b(?:process\.env|import\.meta\.env)\.([A-Za-z_][A-Za-z0-9_]*)\b""",
            body_text,
        ):
            etag = f"@env {evm.group(1)}"
            if etag not in tags:
                tags.append(etag)
        return tags

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

        body = node.child_by_field_name("body")
        ts_tags = self._extract_ts_contract_tags(_node_text(body)) if body else []
        summary = sig
        if ts_tags:
            summary = f"{summary} {' '.join(ts_tags)}".strip()

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
                contextual_summary=summary,
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

        body = node.child_by_field_name("body")
        ts_tags = self._extract_ts_contract_tags(_node_text(body)) if body else []
        summary = docstring.split("\n")[0] if docstring else sig
        if ts_tags:
            summary = f"{summary} {' '.join(ts_tags)}".strip()

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

                    body = val_node.child_by_field_name("body")
                    ts_tags = self._extract_ts_contract_tags(_node_text(body)) if body else []
                    summary = docstring.split("\n")[0] if docstring else sig
                    if ts_tags:
                        summary = f"{summary} {' '.join(ts_tags)}".strip()

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
                            contextual_summary=summary,
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
