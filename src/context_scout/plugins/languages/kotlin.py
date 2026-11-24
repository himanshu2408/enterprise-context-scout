"""Kotlin language plugin using tree-sitter-kotlin AST traversal."""

from __future__ import annotations

import logging
import re
from pathlib import Path

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

_SPRING_STEREOTYPES = frozenset(
    {
        "RestController",
        "Controller",
        "Service",
        "Repository",
        "Component",
        "Configuration",
    }
)

_HTTP_ANNOTATIONS: dict[str, str] = {
    "GetMapping": "GET",
    "PostMapping": "POST",
    "PutMapping": "PUT",
    "DeleteMapping": "DELETE",
    "PatchMapping": "PATCH",
    "RequestMapping": "ANY",
}

_KTOR_ROUTE_METHODS: dict[str, str] = {
    "get": "GET",
    "post": "POST",
    "put": "PUT",
    "delete": "DELETE",
    "patch": "PATCH",
}

_BUILTIN_TYPES = frozenset(
    {
        "Unit",
        "Nothing",
        "Any",
        "Boolean",
        "Byte",
        "Short",
        "Int",
        "Long",
        "Float",
        "Double",
        "Char",
        "String",
        "List",
        "MutableList",
        "Set",
        "MutableSet",
        "Map",
        "MutableMap",
        "Sequence",
        "Flow",
        "Result",
    }
)

_GRADLE_KTS_DEP_RE = re.compile(
    r"""^\s*(implementation|api|compileOnly|runtimeOnly|testImplementation|testRuntimeOnly|kapt|ksp)\s*\(\s*['"]([^'"]+)['"]\s*\)""",
    re.MULTILINE,
)


def _get_kotlin_parser() -> Parser:
    return tslp.get_parser("kotlin")


def _unquote_kt_string(raw: str) -> str:
    s = raw.strip()
    if s.startswith('"""') and s.endswith('"""') and len(s) >= 6:
        return s[3:-3]
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        return s[1:-1]
    return s


def _combine_route(class_route: str, method_route: str) -> str:
    base = class_route.strip("/")
    sub = method_route.strip("/")
    if base and sub:
        return f"/{base}/{sub}"
    if base:
        return f"/{base}"
    if sub:
        return f"/{sub}"
    return "/"


def _extract_type_name(node: Node | None, code_bytes: bytes) -> str:
    if node is None:
        return ""
    if node.type == "nullable_type":
        for c in node.children:
            if c.is_named:
                return _extract_type_name(c, code_bytes)
    if node.type == "user_type":
        parts = [
            _node_text(c, code_bytes).strip() for c in node.children if c.type == "type_identifier"
        ]
        if parts:
            return parts[-1]
    if node.type in {"type_identifier", "simple_identifier"}:
        return _node_text(node, code_bytes).strip()
    text = _node_text(node, code_bytes).strip().rstrip("?")
    if "<" in text:
        text = text.split("<", 1)[0].strip()
    return text.rsplit(".", 1)[-1]


def _parse_annotation_node(
    ann_node: Node,
    code_bytes: bytes,
    trailing_paren: Node | None = None,
) -> tuple[str, list[str], dict[str, str]] | None:
    """Parse an `annotation` AST node (and optional trailing parenthesized_expression)."""
    aname = ""
    pos_args: list[str] = []
    kw_args: dict[str, str] = {}

    for child in ann_node.children:
        if child.type in {"user_type", "constructor_invocation"} and not aname:
            if child.type == "constructor_invocation":
                ut = next((c for c in child.children if c.type == "user_type"), None)
                aname = _extract_type_name(ut, code_bytes)
                va = next((c for c in child.children if c.type == "value_arguments"), None)
                if va is not None:
                    _collect_value_args(va, code_bytes, pos_args, kw_args)
            else:
                aname = _extract_type_name(child, code_bytes)

    if trailing_paren is not None and trailing_paren.type == "parenthesized_expression":
        for sub in trailing_paren.children:
            if sub.type == "string_literal":
                pos_args.append(_unquote_kt_string(_node_text(sub, code_bytes)))

    if not aname:
        return None
    return aname, pos_args, kw_args


def _collect_value_args(
    va_node: Node,
    code_bytes: bytes,
    pos_args: list[str],
    kw_args: dict[str, str],
) -> None:
    for arg in va_node.children:
        if arg.type != "value_argument":
            continue
        k_node = next((c for c in arg.children if c.type == "simple_identifier"), None)
        str_node = next((c for c in arg.children if c.type == "string_literal"), None)
        if k_node is not None and str_node is not None:
            kw_args[_node_text(k_node, code_bytes).strip()] = _unquote_kt_string(
                _node_text(str_node, code_bytes)
            )
        elif str_node is not None:
            pos_args.append(_unquote_kt_string(_node_text(str_node, code_bytes)))


def _collect_annotations_from_prefix(
    prefix_node: Node,
    code_bytes: bytes,
) -> list[tuple[str, list[str], dict[str, str]]]:
    """Recursively unwrap tree-sitter-kotlin `prefix_expression` annotation chains."""
    collected: list[tuple[str, list[str], dict[str, str]]] = []
    children = [c for c in prefix_node.children if c.is_named]
    ann = next((c for c in children if c.type == "annotation"), None)
    rest = [c for c in children if c is not ann]

    trailing_paren = (
        rest[0] if len(rest) == 1 and rest[0].type == "parenthesized_expression" else None
    )
    if ann is not None:
        parsed = _parse_annotation_node(ann, code_bytes, trailing_paren)
        if parsed is not None:
            collected.append(parsed)

    for sub in rest:
        if sub.type == "prefix_expression":
            collected.extend(_collect_annotations_from_prefix(sub, code_bytes))
    return collected


def _extract_modifiers_info(
    node: Node,
    code_bytes: bytes,
) -> tuple[list[tuple[str, list[str], dict[str, str]]], set[str]]:
    """Extract (annotations, modifier_keywords) from a declaration's `modifiers` child."""
    anns: list[tuple[str, list[str], dict[str, str]]] = []
    mods: set[str] = set()
    mods_node = next((c for c in node.children if c.type == "modifiers"), None)
    if mods_node is None:
        return anns, mods

    for child in mods_node.children:
        if child.type == "annotation":
            parsed = _parse_annotation_node(child, code_bytes)
            if parsed is not None:
                anns.append(parsed)
        elif child.type in {
            "class_modifier",
            "function_modifier",
            "visibility_modifier",
            "inheritance_modifier",
        }:
            mods.add(_node_text(child, code_bytes).strip())
    return anns, mods


def _extract_leading_doc(node: Node, code_bytes: bytes) -> str:
    comments: list[str] = []
    prev = node.prev_sibling
    while prev is not None and prev.type in {"multiline_comment", "line_comment"}:
        raw = _node_text(prev, code_bytes).strip()
        if raw.startswith("/**") and raw.endswith("*/"):
            cleaned = re.sub(r"^\s*\*\s?", "", raw[3:-2], flags=re.MULTILINE).strip()
            comments.append(cleaned)
        elif raw.startswith("/*") and raw.endswith("*/"):
            comments.append(raw[2:-2].strip())
        elif raw.startswith("//"):
            comments.append(raw[2:].strip())
        prev = prev.prev_sibling
    comments.reverse()
    return "\n".join(c for c in comments if c)


def _extract_signature_before_body(node: Node, code_bytes: bytes) -> str:
    body_types = {"class_body", "function_body", "enum_class_body"}
    end_byte = node.end_byte
    for child in node.children:
        if child.type in body_types:
            end_byte = child.start_byte
            break
    sig = code_bytes[node.start_byte : end_byte].decode("utf-8", errors="replace").strip()
    return " ".join(sig.split())


def _is_interface_declaration(node: Node, code_bytes: bytes) -> bool:
    for child in node.children:
        if child.type == "class_body":
            break
        if not child.is_named and _node_text(child, code_bytes).strip() == "interface":
            return True
    return False


class KotlinPlugin:
    """Kotlin LanguagePlugin backed by tree-sitter-kotlin."""

    def __init__(self) -> None:
        self._parser: Parser = _get_kotlin_parser()
        self.last_parse_diagnostics: ParseDiagnostics = ParseDiagnostics(
            parser_backend="tree-sitter-kotlin"
        )

    @property
    def language_id(self) -> str:
        return "kotlin"

    @property
    def file_extensions(self) -> tuple[str, ...]:
        return (".kt", ".kts")

    @property
    def manifest_files(self) -> tuple[str, ...]:
        return ("build.gradle.kts",)

    def can_handle(self, file_path: str) -> bool:
        lower = file_path.lower()
        name = Path(lower).name
        return lower.endswith(".kt") or lower.endswith(".kts") or name in self.manifest_files

    @staticmethod
    def _derive_module_prefix(file_path: str) -> str:
        if not file_path:
            return "kotlin"
        return Path(file_path).stem or "kotlin"

    def extract_manifest_dependencies(
        self,
        file_path: str,
        content: str,
    ) -> list[DependencyEdge]:
        if not content or not content.strip():
            self.last_parse_diagnostics = ParseDiagnostics(parser_backend="gradle-kts-regex")
            return []

        if Path(file_path).name.lower() != "build.gradle.kts":
            self.last_parse_diagnostics = ParseDiagnostics(parser_backend="gradle-kts-regex")
            return []

        source_id = Path(file_path).parent.name or Path(file_path).stem
        deps: list[DependencyEdge] = []
        for match in _GRADLE_KTS_DEP_RE.finditer(content):
            cfg, coord = match.group(1), match.group(2)
            parts = coord.split(":")
            pkg = f"{parts[0]}:{parts[1]}" if len(parts) >= 2 else coord
            ver = parts[2] if len(parts) >= 3 else "*"
            scope = "dev" if cfg.startswith("test") else "compile"
            deps.append(
                DependencyEdge(
                    source_identifier=source_id,
                    target_package=pkg,
                    version_constraint=ver,
                    scope=scope,
                    manifest_path=file_path,
                )
            )
        self.last_parse_diagnostics = ParseDiagnostics(parser_backend="gradle-kts-regex")
        return deps

    def extract_ast(
        self,
        code: str,
        file_path: str = "",
        repo_name: str = "",
        module_prefix: str = "",
    ) -> tuple[list[ParsedSymbol], list[ParsedEdge]]:
        if not code or not code.strip():
            self.last_parse_diagnostics = ParseDiagnostics(parser_backend="tree-sitter-kotlin")
            return [], []

        code_bytes = code.encode("utf-8")
        tree = self._parser.parse(code_bytes)
        has_err, err_cnt, miss_cnt = inspect_tree_errors(tree.root_node)
        if has_err:
            logger.warning(
                "KotlinPlugin parse error in %s (errors=%d, missing=%d); abstaining.",
                file_path or "<memory>",
                err_cnt,
                miss_cnt,
            )
            self.last_parse_diagnostics = ParseDiagnostics(
                parser_backend="tree-sitter-kotlin",
                has_parse_error=True,
                abstained=True,
                error_node_count=err_cnt,
                missing_node_count=miss_cnt,
                reason="Syntax/parse error detected by tree-sitter-kotlin",
            )
            return [], []

        self.last_parse_diagnostics = ParseDiagnostics(parser_backend="tree-sitter-kotlin")
        prefix = module_prefix or self._derive_module_prefix(file_path)

        package_name = ""
        imports: list[str] = []
        symbols: list[ParsedSymbol] = []
        edges: list[ParsedEdge] = []
        seen_edges: set[tuple[str, str, str]] = set()
        type_stack: list[tuple[str, str]] = []
        class_route_stack: list[str] = []

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

        def qualify(leaf: str) -> str:
            if type_stack:
                return f"{type_stack[-1][1]}.{leaf}"
            if package_name:
                return f"{package_name}.{leaf}"
            return f"{prefix}.{leaf}"

        def scan_function_body(
            node: Node,
            owner_qn: str,
            route_notes: list[str],
        ) -> None:
            stack: list[Node] = [node]
            while stack:
                curr = stack.pop()
                if curr.type == "call_expression":
                    callee_node = next((c for c in curr.children if c.is_named), None)
                    if callee_node is not None and callee_node.type != "call_suffix":
                        callee_txt = _node_text(callee_node, code_bytes).strip()
                        if callee_txt in _KTOR_ROUTE_METHODS:
                            suffix = next(
                                (c for c in curr.children if c.type == "call_suffix"), None
                            )
                            if suffix is not None:
                                str_lit = next(
                                    (s for s in suffix.children if s.type == "value_arguments"),
                                    None,
                                )
                                if str_lit is not None:
                                    pos: list[str] = []
                                    _collect_value_args(str_lit, code_bytes, pos, {})
                                    if pos:
                                        route_notes.append(
                                            f"@route {_KTOR_ROUTE_METHODS[callee_txt]} {pos[0]}"
                                        )
                        elif callee_txt:
                            add_edge(owner_qn, callee_txt, "CALLS")
                stack.extend(reversed(curr.children))

        def handle_class_or_object(
            node: Node,
            extra_anns: list[tuple[str, list[str], dict[str, str]]],
        ) -> None:
            is_companion = node.type == "companion_object"
            name_node = next((c for c in node.children if c.type == "type_identifier"), None)
            if is_companion:
                type_name = _node_text(name_node, code_bytes).strip() if name_node else "Companion"
            elif name_node is not None:
                type_name = _node_text(name_node, code_bytes).strip()
            else:
                return

            if not type_name:
                return

            qn = qualify(type_name)
            bare = f"{type_stack[-1][0]}.{type_name}" if type_stack else type_name
            mod_anns, mods = _extract_modifiers_info(node, code_bytes)
            all_anns = [*extra_anns, *mod_anns]

            is_iface = node.type == "class_declaration" and _is_interface_declaration(
                node, code_bytes
            )
            sym_kind = "interface" if is_iface else "class"
            sig = _extract_signature_before_body(node, code_bytes)
            doc = _extract_leading_doc(node, code_bytes)

            notes: list[str] = []
            class_route = ""
            is_spring_bean = False
            for aname, pos_args, kw_args in all_anns:
                if aname in _SPRING_STEREOTYPES:
                    notes.append(f"@{aname}")
                    is_spring_bean = True
                if aname == "RequestMapping":
                    route_val = (
                        pos_args[0] if pos_args else kw_args.get("value", kw_args.get("path", ""))
                    )
                    if route_val:
                        class_route = route_val
                        notes.append(f"@route ANY {_combine_route(class_route, '')}")

            if "sealed" in mods:
                notes.append("@sealed")
            if node.type == "object_declaration":
                notes.append("@object")
            elif is_companion:
                notes.append("@companion")

            # Primary constructor parameters -> USES_TYPE / DTO fields
            dto_fields: list[str] = []
            primary_ctor = next((c for c in node.children if c.type == "primary_constructor"), None)
            if primary_ctor is not None:
                for p in primary_ctor.children:
                    if p.type == "class_parameter":
                        pname = next((c for c in p.children if c.type == "simple_identifier"), None)
                        ptype_node = next(
                            (c for c in p.children if c.type in {"user_type", "nullable_type"}),
                            None,
                        )
                        if pname is not None:
                            dto_fields.append(_node_text(pname, code_bytes).strip())
                        ptype = _extract_type_name(ptype_node, code_bytes)
                        if ptype and ptype not in _BUILTIN_TYPES:
                            add_edge(qn, ptype, "USES_TYPE")
                            if is_spring_bean:
                                add_edge(qn, ptype, "PROVISIONED_BY")

            if "data" in mods or type_name.endswith(("Dto", "DTO", "Request", "Response")):
                notes.append(f"@dto {type_name} fields={','.join(dto_fields)}")

            summary = " ".join(dict.fromkeys(notes))
            merged_doc = f"{doc} {summary}".strip() if summary else doc

            symbols.append(
                ParsedSymbol(
                    name=type_name,
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
                    contextual_summary=summary,
                    repo_name=repo_name,
                )
            )

            for imp in imports:
                add_edge(qn, imp, "IMPORTS")
            if is_spring_bean:
                add_edge(qn, "org.springframework.context.ApplicationContext", "PROVISIONED_BY")

            # Delegation specifiers -> INHERITS
            for child in node.children:
                if child.type == "delegation_specifier":
                    ut = next(
                        (
                            c
                            for c in child.children
                            if c.type in {"user_type", "constructor_invocation"}
                        ),
                        None,
                    )
                    if ut is not None and ut.type == "constructor_invocation":
                        ut = next((c for c in ut.children if c.type == "user_type"), None)
                    base = _extract_type_name(ut, code_bytes)
                    if base:
                        add_edge(qn, base, "INHERITS")

            type_stack.append((type_name, qn))
            class_route_stack.append(class_route)
            body = next(
                (c for c in node.children if c.type in {"class_body", "enum_class_body"}),
                None,
            )
            if body is not None:
                walk_container(body, qn)
            class_route_stack.pop()
            type_stack.pop()

        def handle_function(
            node: Node,
            owner_class_qn: str,
            extra_anns: list[tuple[str, list[str], dict[str, str]]],
        ) -> None:
            name_node = next((c for c in node.children if c.type == "simple_identifier"), None)
            if name_node is None:
                return
            fn_name = _node_text(name_node, code_bytes).strip()
            if not fn_name:
                return

            qn = qualify(fn_name)
            bare = f"{type_stack[-1][0]}.{fn_name}" if type_stack else fn_name
            kind = "method" if type_stack else "function"
            mod_anns, mods = _extract_modifiers_info(node, code_bytes)
            all_anns = [*extra_anns, *mod_anns]
            sig = _extract_signature_before_body(node, code_bytes)
            doc = _extract_leading_doc(node, code_bytes)
            class_route = class_route_stack[-1] if class_route_stack else ""

            notes: list[str] = []
            if "suspend" in mods:
                notes.append("@suspend")
            is_test = False
            for aname, pos_args, kw_args in all_anns:
                if aname in _HTTP_ANNOTATIONS:
                    http_method = _HTTP_ANNOTATIONS[aname]
                    sub_route = (
                        pos_args[0] if pos_args else kw_args.get("value", kw_args.get("path", ""))
                    )
                    full_route = _combine_route(class_route, sub_route)
                    notes.append(f"@route {http_method} {full_route}")
                elif aname in {"Test", "ParameterizedTest"}:
                    is_test = True
                    notes.append("@Test")

            # Parameters & return type -> USES_TYPE
            params = next((c for c in node.children if c.type == "function_value_parameters"), None)
            if params is not None:
                for p in params.children:
                    if p.type == "parameter":
                        pt_node = next(
                            (c for c in p.children if c.type in {"user_type", "nullable_type"}),
                            None,
                        )
                        ptype = _extract_type_name(pt_node, code_bytes)
                        if ptype and ptype not in _BUILTIN_TYPES:
                            add_edge(qn, ptype, "USES_TYPE")

            ret_node = next(
                (c for c in node.children if c.type in {"user_type", "nullable_type"}),
                None,
            )
            ret_type = _extract_type_name(ret_node, code_bytes)
            if ret_type and ret_type not in _BUILTIN_TYPES:
                add_edge(qn, ret_type, "USES_TYPE")

            if is_test and owner_class_qn:
                add_edge(owner_class_qn, qn, "TESTED_BY")

            body = next((c for c in node.children if c.type == "function_body"), None)
            if body is not None:
                scan_function_body(body, qn, notes)

            summary = " ".join(dict.fromkeys(notes))
            merged_doc = f"{doc} {summary}".strip() if summary else doc

            symbols.append(
                ParsedSymbol(
                    name=fn_name,
                    qualified_name=qn,
                    bare_name=bare,
                    symbol_kind=kind,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    start_byte=node.start_byte,
                    end_byte=node.end_byte,
                    signature=sig,
                    signature_skeleton=sig,
                    docstring=merged_doc,
                    contextual_summary=summary,
                    repo_name=repo_name,
                )
            )

        def unwrap_nested_prefix(
            node: Node,
            pending_anns: list[tuple[str, list[str], dict[str, str]]],
            owner_class_qn: str,
        ) -> None:
            anns = [*pending_anns, *_collect_annotations_from_prefix(node, code_bytes)]
            for child in node.children:
                if child.type in {"class_declaration", "object_declaration", "companion_object"}:
                    handle_class_or_object(child, anns)
                elif child.type == "function_declaration":
                    handle_function(child, owner_class_qn, anns)
                elif child.type == "prefix_expression":
                    unwrap_nested_prefix(child, anns, owner_class_qn)

        def walk_container(container: Node, owner_class_qn: str = "") -> None:
            pending_anns: list[tuple[str, list[str], dict[str, str]]] = []
            for child in container.children:
                if child.type == "package_header":
                    id_node = next((c for c in child.children if c.type == "identifier"), None)
                    if id_node is not None:
                        nonlocal package_name
                        package_name = _node_text(id_node, code_bytes).strip()
                elif child.type == "import_list":
                    for imp in child.children:
                        if imp.type == "import_header":
                            raw = _node_text(imp, code_bytes).strip().removeprefix("import").strip()
                            if raw and raw not in imports:
                                imports.append(raw)
                elif child.type == "prefix_expression":
                    has_nested_decl = any(
                        c.type
                        in {
                            "class_declaration",
                            "object_declaration",
                            "companion_object",
                            "function_declaration",
                        }
                        for c in child.children
                    )
                    if has_nested_decl:
                        unwrap_nested_prefix(child, pending_anns, owner_class_qn)
                        pending_anns = []
                    else:
                        pending_anns.extend(_collect_annotations_from_prefix(child, code_bytes))
                elif child.type in {"class_declaration", "object_declaration", "companion_object"}:
                    handle_class_or_object(child, pending_anns)
                    pending_anns = []
                elif child.type == "function_declaration":
                    handle_function(child, owner_class_qn, pending_anns)
                    pending_anns = []
                elif child.type not in {"multiline_comment", "line_comment"}:
                    pending_anns = []

        walk_container(tree.root_node)
        return symbols, edges
