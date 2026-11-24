"""Go language plugin using tree-sitter-go and tree-sitter-gomod AST traversal."""

from __future__ import annotations

import logging
import re

import tree_sitter_language_pack as tslp
from tree_sitter import Node, Parser

from context_scout.ast_engine.extractor import (
    ParseDiagnostics,
    _node_text,
    inspect_tree_errors,
)
from context_scout.ast_engine.models import ParsedEdge, ParsedSymbol
from context_scout.plugins.base import DependencyEdge

logger = logging.getLogger(__name__)

_GO_BUILTIN_TYPES = frozenset(
    {
        "any",
        "bool",
        "byte",
        "comparable",
        "complex64",
        "complex128",
        "error",
        "float32",
        "float64",
        "int",
        "int8",
        "int16",
        "int32",
        "int64",
        "rune",
        "string",
        "uint",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "uintptr",
    }
)

_GO_BUILTIN_CALLS = frozenset(
    {
        "append",
        "cap",
        "clear",
        "close",
        "complex",
        "copy",
        "delete",
        "imag",
        "len",
        "make",
        "max",
        "min",
        "new",
        "panic",
        "print",
        "println",
        "real",
        "recover",
    }
)

_ROUTE_METHOD_MAP: dict[str, str] = {
    "Get": "GET",
    "GET": "GET",
    "Post": "POST",
    "POST": "POST",
    "Put": "PUT",
    "PUT": "PUT",
    "Delete": "DELETE",
    "DELETE": "DELETE",
    "Patch": "PATCH",
    "PATCH": "PATCH",
    "Head": "HEAD",
    "HEAD": "HEAD",
    "Options": "OPTIONS",
    "OPTIONS": "OPTIONS",
    "HandleFunc": "ANY",
    "Handle": "ANY",
    "Any": "ANY",
}

_JSON_TAG_RE = re.compile(r'json:"([^",]+)')


def _collapse_ws(text: str) -> str:
    return " ".join(text.split())


def _unquote_go_string(node: Node, code_bytes: bytes) -> str:
    raw = _node_text(node, code_bytes).strip()
    if len(raw) >= 2 and ((raw[0] == '"' and raw[-1] == '"') or (raw[0] == "`" and raw[-1] == "`")):
        return raw[1:-1]
    return raw


def _extract_leading_comment(node: Node, code_bytes: bytes) -> str:
    comments: list[str] = []
    sibling = node.prev_sibling
    while sibling is not None and sibling.type == "comment":
        text = _node_text(sibling, code_bytes).strip()
        if text.startswith("//"):
            comments.append(text[2:].strip())
        elif text.startswith("/*") and text.endswith("*/"):
            inner = text[2:-2].strip()
            if inner:
                comments.append(inner)
        sibling = sibling.prev_sibling
    if not comments:
        return ""
    comments.reverse()
    return "\n".join(line for line in comments if line)


def _extract_signature_before_body(node: Node, code_bytes: bytes) -> str:
    body_node = node.child_by_field_name("body")
    end_byte = body_node.start_byte if body_node is not None else node.end_byte
    raw = code_bytes[node.start_byte : end_byte].decode("utf-8", errors="replace")
    return _collapse_ws(raw)


def _extract_go_type_names(node: Node, code_bytes: bytes) -> list[str]:
    """Iteratively collect non-builtin type names referenced in a Go type AST node."""
    found: list[str] = []
    stack: list[Node] = [node]
    while stack:
        curr = stack.pop()
        if curr.type == "type_identifier":
            name = _node_text(curr, code_bytes).strip()
            if name and name not in _GO_BUILTIN_TYPES:
                found.append(name)
        elif curr.type == "qualified_type":
            name_node = curr.child_by_field_name("name")
            if name_node is not None:
                name = _node_text(name_node, code_bytes).strip()
                if name and name not in _GO_BUILTIN_TYPES:
                    found.append(name)
            else:
                text = _node_text(curr, code_bytes).strip()
                if text:
                    found.append(text.rsplit(".", 1)[-1])
        else:
            stack.extend(reversed(curr.children))
    return list(dict.fromkeys(found))


def _extract_receiver_type(param_list: Node, code_bytes: bytes) -> str:
    for child in param_list.children:
        if child.type == "parameter_declaration":
            type_node = child.child_by_field_name("type")
            if type_node is not None:
                names = _extract_go_type_names(type_node, code_bytes)
                if names:
                    return names[0]
                raw = _node_text(type_node, code_bytes).lstrip("*").strip()
                return raw.split("[", 1)[0]
    return ""


class GoPlugin:
    """Language plugin for Go (.go) source and go.mod/go.sum manifests."""

    def __init__(self) -> None:
        self._parser: Parser = tslp.get_parser("go")
        self._gomod_parser: Parser = tslp.get_parser("gomod")
        self.last_parse_diagnostics: ParseDiagnostics | None = None

    @property
    def language_id(self) -> str:
        return "go"

    @property
    def file_extensions(self) -> tuple[str, ...]:
        return (".go",)

    @property
    def manifest_files(self) -> tuple[str, ...]:
        return ("go.mod", "go.sum")

    def can_handle(self, file_path: str) -> bool:
        return file_path.lower().endswith(".go")

    def _derive_module_prefix(self, file_path: str) -> str:
        if not file_path:
            return "main"
        normalized = file_path.replace("\\", "/").strip("/")
        parts = [p for p in normalized.split("/") if p]
        if len(parts) >= 2:
            return parts[-2]
        return "main"

    def extract_manifest_dependencies(
        self,
        file_path: str,
        content: str,
    ) -> list[DependencyEdge]:
        """Extract module dependencies from go.mod using tree-sitter-gomod."""
        lower_name = file_path.replace("\\", "/").rsplit("/", 1)[-1].lower()
        if lower_name == "go.sum":
            self.last_parse_diagnostics = ParseDiagnostics(parser_backend="tree-sitter-gomod")
            return []

        if not content or not content.strip():
            self.last_parse_diagnostics = ParseDiagnostics(parser_backend="tree-sitter-gomod")
            return []

        content_bytes = content.encode("utf-8")
        tree = self._gomod_parser.parse(content_bytes)
        has_err, err_cnt, miss_cnt = inspect_tree_errors(tree.root_node)
        if has_err:
            logger.warning(
                "GoPlugin go.mod parse error in %s (errors=%d, missing=%d); abstaining.",
                file_path or "<memory>",
                err_cnt,
                miss_cnt,
            )
            self.last_parse_diagnostics = ParseDiagnostics(
                parser_backend="tree-sitter-gomod",
                has_parse_error=True,
                abstained=True,
                error_node_count=err_cnt,
                missing_node_count=miss_cnt,
                reason="Syntax/parse error detected by tree-sitter-gomod",
            )
            return []

        self.last_parse_diagnostics = ParseDiagnostics(parser_backend="tree-sitter-gomod")
        source_id = file_path
        deps: list[DependencyEdge] = []

        for child in tree.root_node.children:
            if child.type == "module_directive":
                for sub in child.children:
                    if sub.type == "module_path":
                        mod = _node_text(sub, content_bytes).strip()
                        if mod:
                            source_id = mod
                        break
            elif child.type == "require_directive":
                for spec in child.children:
                    if spec.type != "require_spec":
                        continue
                    mod_path = ""
                    version = "*"
                    for part in spec.children:
                        if part.type == "module_path":
                            mod_path = _node_text(part, content_bytes).strip()
                        elif part.type == "version":
                            version = _node_text(part, content_bytes).strip() or "*"
                    if mod_path:
                        deps.append(
                            DependencyEdge(
                                source_identifier=source_id,
                                target_package=mod_path,
                                version_constraint=version,
                                scope="runtime",
                                manifest_path=file_path,
                            )
                        )
        return deps

    def extract_ast(
        self,
        code: str,
        file_path: str = "",
        repo_name: str = "",
        module_prefix: str = "",
    ) -> tuple[list[ParsedSymbol], list[ParsedEdge]]:
        """Extract Go structs, interfaces, functions, methods, goroutines, routes, and edges."""
        if not code or not code.strip():
            self.last_parse_diagnostics = ParseDiagnostics(parser_backend="tree-sitter-go")
            return [], []

        code_bytes = code.encode("utf-8")
        tree = self._parser.parse(code_bytes)
        has_err, err_cnt, miss_cnt = inspect_tree_errors(tree.root_node)
        if has_err:
            logger.warning(
                "GoPlugin parse error in %s (errors=%d, missing=%d); abstaining.",
                file_path or "<memory>",
                err_cnt,
                miss_cnt,
            )
            self.last_parse_diagnostics = ParseDiagnostics(
                parser_backend="tree-sitter-go",
                has_parse_error=True,
                abstained=True,
                error_node_count=err_cnt,
                missing_node_count=miss_cnt,
                reason="Syntax/parse error detected by tree-sitter-go",
            )
            return [], []

        self.last_parse_diagnostics = ParseDiagnostics(parser_backend="tree-sitter-go")

        pkg_name = ""
        imports: list[str] = []
        for child in tree.root_node.children:
            if child.type == "package_clause":
                for sub in child.children:
                    if sub.type == "package_identifier":
                        pkg_name = _node_text(sub, code_bytes).strip()
                        break
            elif child.type == "import_declaration":
                stack: list[Node] = [child]
                while stack:
                    curr = stack.pop()
                    if curr.type == "import_spec":
                        path_node = curr.child_by_field_name("path")
                        if path_node is not None:
                            imp = _unquote_go_string(path_node, code_bytes)
                            if imp and imp not in imports:
                                imports.append(imp)
                    else:
                        stack.extend(reversed(curr.children))

        ns = module_prefix or pkg_name or self._derive_module_prefix(file_path)
        symbols: list[ParsedSymbol] = []
        edges: list[ParsedEdge] = []
        seen_edge_keys: set[tuple[str, str, str]] = set()

        def add_edge(src: str, tgt: str, kind: str) -> None:
            if not src or not tgt:
                return
            key = (src, tgt, kind)
            if key not in seen_edge_keys:
                seen_edge_keys.add(key)
                edges.append(
                    ParsedEdge(
                        source_qualified_name=src,
                        target_qualified_name=tgt,
                        edge_kind=kind,
                    )
                )

        def scan_function_body(
            body_node: Node | None,
            owner_qn: str,
            is_test: bool,
        ) -> list[str]:
            if body_node is None:
                return []
            notes: list[str] = []
            stack: list[Node] = [body_node]
            while stack:
                curr = stack.pop()
                if curr.type == "go_statement":
                    call_node = next(
                        (c for c in curr.children if c.type == "call_expression"),
                        None,
                    )
                    if call_node is not None:
                        callee = _extract_callee_name(call_node, code_bytes)
                        if callee:
                            notes.append(f"[goroutine] {callee}")
                            add_edge(owner_qn, callee, "CALLS")
                    else:
                        notes.append("[goroutine]")
                    stack.extend(reversed(curr.children))
                    continue

                if curr.type == "call_expression":
                    callee = _extract_callee_name(curr, code_bytes)
                    args_node = curr.child_by_field_name("arguments")
                    str_args: list[str] = []
                    ident_args: list[str] = []
                    if args_node is not None:
                        for arg in args_node.children:
                            if arg.type in {
                                "interpreted_string_literal",
                                "raw_string_literal",
                            }:
                                str_args.append(_unquote_go_string(arg, code_bytes))
                            elif arg.type == "identifier":
                                ident_args.append(_node_text(arg, code_bytes).strip())
                            elif arg.type == "selector_expression":
                                field = arg.child_by_field_name("field")
                                if field is not None:
                                    ident_args.append(_node_text(field, code_bytes).strip())

                    if callee in _ROUTE_METHOD_MAP and str_args:
                        http_method = _ROUTE_METHOD_MAP[callee]
                        route_path = "/" + str_args[0].lstrip("/")
                        notes.append(f"@route {http_method} {route_path}")
                        for handler in ident_args:
                            add_edge(owner_qn, handler, "CALLS")
                    elif callee and callee not in _GO_BUILTIN_CALLS:
                        add_edge(owner_qn, callee, "CALLS")
                        if is_test and callee not in {"Run", "Error", "Errorf", "Fatal", "Fatalf"}:
                            add_edge(f"{ns}.{callee}", owner_qn, "TESTED_BY")

                stack.extend(reversed(curr.children))
            return notes

        def handle_type_spec(spec_node: Node, decl_node: Node, is_alias: bool) -> None:
            name_node = spec_node.child_by_field_name("name")
            if name_node is None:
                return
            type_name = _node_text(name_node, code_bytes).strip()
            if not type_name:
                return
            qn = f"{ns}.{type_name}"
            doc = _extract_leading_comment(decl_node, code_bytes)
            sig = _collapse_ws(_node_text(decl_node, code_bytes))
            type_node = spec_node.child_by_field_name("type")

            if is_alias:
                sym_kind = "type_alias"
                if type_node is not None:
                    for t in _extract_go_type_names(type_node, code_bytes):
                        add_edge(qn, t, "USES_TYPE")
            elif type_node is not None and type_node.type == "struct_type":
                sym_kind = "class"
                sig = f"type {type_name} struct"
                dto_fields: list[str] = []
                field_list = next(
                    (c for c in type_node.children if c.type == "field_declaration_list"),
                    None,
                )
                if field_list is not None:
                    for fdecl in field_list.children:
                        if fdecl.type != "field_declaration":
                            continue
                        f_names = [
                            _node_text(c, code_bytes).strip()
                            for c in fdecl.children
                            if c.type == "field_identifier"
                        ]
                        f_type_node = fdecl.child_by_field_name("type")
                        tag_node = fdecl.child_by_field_name("tag")
                        if not f_names and f_type_node is not None:
                            for emb in _extract_go_type_names(f_type_node, code_bytes):
                                add_edge(qn, emb, "INHERITS")
                        else:
                            json_name = ""
                            if tag_node is not None:
                                m = _JSON_TAG_RE.search(_node_text(tag_node, code_bytes))
                                if m and m.group(1) != "-":
                                    json_name = m.group(1)
                            for fn in f_names:
                                dto_fields.append(json_name or fn)
                            if f_type_node is not None:
                                for ft in _extract_go_type_names(f_type_node, code_bytes):
                                    add_edge(qn, ft, "USES_TYPE")
                if dto_fields:
                    dto_note = f"@dto {type_name} fields={','.join(dict.fromkeys(dto_fields))}"
                    doc = f"{doc} {dto_note}".strip()
            elif type_node is not None and type_node.type == "interface_type":
                sym_kind = "interface"
                sig = f"type {type_name} interface"
                for sub in type_node.children:
                    if sub.type == "method_elem":
                        for mc in sub.children:
                            if mc.type in {"parameter_list", "type_identifier", "qualified_type"}:
                                for ut in _extract_go_type_names(mc, code_bytes):
                                    add_edge(qn, ut, "USES_TYPE")
                    elif sub.type in {"type_elem", "type_identifier", "qualified_type"}:
                        for emb in _extract_go_type_names(sub, code_bytes):
                            add_edge(qn, emb, "INHERITS")
            else:
                sym_kind = "type_alias"
                if type_node is not None:
                    for t in _extract_go_type_names(type_node, code_bytes):
                        add_edge(qn, t, "USES_TYPE")

            symbols.append(
                ParsedSymbol(
                    name=type_name,
                    qualified_name=qn,
                    bare_name=type_name,
                    symbol_kind=sym_kind,
                    start_line=decl_node.start_point[0] + 1,
                    end_line=decl_node.end_point[0] + 1,
                    start_byte=decl_node.start_byte,
                    end_byte=decl_node.end_byte,
                    signature=sig,
                    signature_skeleton=sig,
                    docstring=doc,
                    contextual_summary=doc,
                    repo_name=repo_name,
                )
            )
            for imp in imports:
                add_edge(qn, imp, "IMPORTS")

        def handle_func_or_method(node: Node, is_method: bool) -> None:
            name_node = node.child_by_field_name("name")
            if name_node is None:
                return
            fn_name = _node_text(name_node, code_bytes).strip()
            if not fn_name:
                return

            if is_method:
                recv_node = node.child_by_field_name("receiver")
                recv_type = (
                    _extract_receiver_type(recv_node, code_bytes) if recv_node is not None else ""
                )
                qn = f"{ns}.{recv_type}.{fn_name}" if recv_type else f"{ns}.{fn_name}"
                bare = f"{recv_type}.{fn_name}" if recv_type else fn_name
                sym_kind = "method"
            else:
                qn = f"{ns}.{fn_name}"
                bare = fn_name
                sym_kind = "function"

            sig = _extract_signature_before_body(node, code_bytes)
            doc = _extract_leading_comment(node, code_bytes)

            params_node = node.child_by_field_name("parameters")
            if params_node is not None:
                for ut in _extract_go_type_names(params_node, code_bytes):
                    add_edge(qn, ut, "USES_TYPE")
            result_node = node.child_by_field_name("result")
            if result_node is not None:
                for ut in _extract_go_type_names(result_node, code_bytes):
                    add_edge(qn, ut, "USES_TYPE")

            is_test = not is_method and fn_name.startswith(("Test", "Benchmark", "Fuzz"))
            body_node = node.child_by_field_name("body")
            notes = scan_function_body(body_node, qn, is_test=is_test)
            merged_doc = f"{doc} {' '.join(dict.fromkeys(notes))}".strip()

            symbols.append(
                ParsedSymbol(
                    name=fn_name,
                    qualified_name=qn,
                    bare_name=bare,
                    symbol_kind=sym_kind,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    start_byte=node.start_byte,
                    end_byte=node.end_byte,
                    signature=sig,
                    signature_skeleton=sig,
                    docstring=merged_doc,
                    contextual_summary=merged_doc,
                    repo_name=repo_name,
                )
            )
            for imp in imports:
                add_edge(qn, imp, "IMPORTS")

        for child in tree.root_node.children:
            if child.type == "type_declaration":
                for sub in child.children:
                    if sub.type == "type_spec":
                        handle_type_spec(sub, child, is_alias=False)
                    elif sub.type == "type_alias":
                        handle_type_spec(sub, child, is_alias=True)
            elif child.type == "function_declaration":
                handle_func_or_method(child, is_method=False)
            elif child.type == "method_declaration":
                handle_func_or_method(child, is_method=True)

        return symbols, edges


def _extract_callee_name(call_node: Node, code_bytes: bytes) -> str:
    func_node = call_node.child_by_field_name("function")
    if func_node is None:
        return ""
    if func_node.type == "identifier":
        return _node_text(func_node, code_bytes).strip()
    if func_node.type == "selector_expression":
        field = func_node.child_by_field_name("field")
        if field is not None:
            return _node_text(field, code_bytes).strip()
    return ""
