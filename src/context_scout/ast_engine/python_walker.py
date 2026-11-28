"""Tree-sitter Python AST symbol, decorator, call, and import walker."""

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


class _PythonWalkerMixin(_ASTExtractorStateBase):
    """Python tree-sitter extraction mixin for ASTExtractor."""

    @staticmethod
    def _extract_python_contract_tags(
        outer_node: Node,
        body: Node | None,
        code_bytes: bytes,
        router_prefixes: dict[str, str] | None = None,
    ) -> list[str]:
        """Extracts @route and @env cross-boundary tags from Python decorators and body."""
        tags: list[str] = []
        prefixes = router_prefixes or {}
        if outer_node.type == "decorated_definition":
            for child in outer_node.children:
                if child.type != "decorator":
                    continue
                dec_text = _node_text(child, code_bytes)
                m = re.match(
                    r"^@\s*([A-Za-z_][A-Za-z0-9_.]*)\.(get|post|put|delete|patch|api_route)\(\s*['\"]([^'\"]+)['\"]",
                    dec_text.strip(),
                )
                if m:
                    obj_name, verb, raw_path = m.group(1), m.group(2).lower(), m.group(3)
                    method = "ANY" if verb == "api_route" else verb.upper()
                    if verb == "api_route":
                        m_meth = re.search(
                            r"methods\s*=\s*[\[\(]\s*['\"]([A-Za-z]+)['\"]", dec_text
                        )
                        if m_meth:
                            method = m_meth.group(1).upper()
                    prefix = prefixes.get(obj_name.split(".")[-1], "")
                    if prefix and not raw_path.startswith(prefix):
                        joined = f"{prefix.rstrip('/')}/{raw_path.lstrip('/')}"
                    else:
                        joined = raw_path
                    if not joined.startswith("/"):
                        joined = "/" + joined
                    tag = f"@route {method} {joined}"
                    if tag not in tags:
                        tags.append(tag)
        scan_text = _node_text(body or outer_node, code_bytes)
        for env_m in re.finditer(
            r"""(?:os\.environ\.get|os\.getenv)\(\s*['"]([A-Za-z_][A-Za-z0-9_]*)['"]|os\.environ\[\s*['"]([A-Za-z_][A-Za-z0-9_]*)['"]\s*\]""",
            scan_text,
        ):
            env_var = env_m.group(1) or env_m.group(2)
            if env_var:
                etag = f"@env {env_var}"
                if etag not in tags:
                    tags.append(etag)
        for ac_m in re.finditer(r"AliasChoices\(([^)]+)\)", scan_text):
            for lit in re.findall(r"""['"]([A-Za-z_][A-Za-z0-9_]*)['"]""", ac_m.group(1)):
                etag = f"@env {lit}"
                if etag not in tags:
                    tags.append(etag)
        return tags

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
        router_prefixes: dict[str, str] = {}

        # Detect module-level APIRouter(prefix=...) / Blueprint(..., url_prefix=...)
        for child in root.children:
            target_assign: Node | None = None
            if child.type == "assignment":
                target_assign = child
            elif (
                child.type == "expression_statement"
                and child.children
                and child.children[0].type == "assignment"
            ):
                target_assign = child.children[0]
            if target_assign is not None:
                assign_txt = _node_text(target_assign, code_bytes)
                rm = re.match(
                    r"""^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:APIRouter|Blueprint)\([\s\S]*?(?:url_prefix|prefix)\s*=\s*['"]([^'"]+)['"]""",
                    assign_txt,
                )
                if rm:
                    router_prefixes[rm.group(1)] = rm.group(2)

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
                    router_prefixes=router_prefixes,
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
        router_prefixes: dict[str, str] | None = None,
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
        contract_tags = self._extract_python_contract_tags(
            outer_node, body, code_bytes, router_prefixes=router_prefixes
        )
        if contract_tags:
            summary = f"{summary} {' '.join(contract_tags)}".strip()

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
