"""Java & Spring Boot language plugin using tree-sitter-java AST traversal."""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
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

_TYPE_DECL_NODES = frozenset(
    {
        "class_declaration",
        "interface_declaration",
        "enum_declaration",
        "record_declaration",
    }
)

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

_SPRING_HTTP_MAPPINGS: dict[str, str] = {
    "GetMapping": "GET",
    "PostMapping": "POST",
    "PutMapping": "PUT",
    "DeleteMapping": "DELETE",
    "PatchMapping": "PATCH",
    "RequestMapping": "ANY",
}

_BUILTIN_TYPES = frozenset(
    {
        "void",
        "boolean",
        "byte",
        "short",
        "int",
        "long",
        "float",
        "double",
        "char",
        "String",
        "Object",
        "Integer",
        "Long",
        "Boolean",
        "Double",
        "Float",
        "List",
        "Map",
        "Set",
        "Optional",
        "ResponseEntity",
        "CompletableFuture",
        "var",
    }
)

_GRADLE_DEP_RE = re.compile(
    r"""^\s*(implementation|api|compileOnly|runtimeOnly|testImplementation|testRuntimeOnly|annotationProcessor|kapt)\s*\(?\s*['"]([^'"]+)['"]\s*\)?""",
    re.MULTILINE,
)


def _get_java_parser() -> Parser:
    return tslp.get_parser("java")


def _unquote_java_string(raw: str) -> str:
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
    if node.type in {"type_identifier", "identifier", "scoped_type_identifier"}:
        return _node_text(node, code_bytes).strip().rsplit(".", 1)[-1]
    if node.type == "generic_type":
        for c in node.children:
            if c.type in {"type_identifier", "scoped_type_identifier"}:
                return _node_text(c, code_bytes).strip().rsplit(".", 1)[-1]
    if node.type == "array_type":
        elem = node.child_by_field_name("element")
        return _extract_type_name(elem, code_bytes)
    text = _node_text(node, code_bytes).strip()
    if "<" in text:
        text = text.split("<", 1)[0].strip()
    return text.rsplit(".", 1)[-1]


def _extract_annotations(
    node: Node, code_bytes: bytes
) -> list[tuple[str, list[str], dict[str, str]]]:
    """Extract (name, positional_strings, named_args) from modifiers on a declaration."""
    found: list[tuple[str, list[str], dict[str, str]]] = []
    mods = next((c for c in node.children if c.type == "modifiers"), None)
    if mods is None:
        return found

    for child in mods.children:
        if child.type == "marker_annotation":
            name_node = child.child_by_field_name("name")
            if name_node is not None:
                found.append((_node_text(name_node, code_bytes).rsplit(".", 1)[-1], [], {}))
        elif child.type == "annotation":
            name_node = child.child_by_field_name("name")
            if name_node is None:
                continue
            aname = _node_text(name_node, code_bytes).rsplit(".", 1)[-1]
            pos_args: list[str] = []
            kw_args: dict[str, str] = {}
            arg_list = child.child_by_field_name("arguments")
            if arg_list is not None:
                for arg in arg_list.children:
                    if arg.type == "string_literal":
                        pos_args.append(_unquote_java_string(_node_text(arg, code_bytes)))
                    elif arg.type == "element_value_pair":
                        k_node = arg.child_by_field_name("key")
                        v_node = arg.child_by_field_name("value")
                        if k_node is not None and v_node is not None:
                            k = _node_text(k_node, code_bytes).strip()
                            v_txt = _node_text(v_node, code_bytes).strip()
                            if v_node.type == "string_literal":
                                v_txt = _unquote_java_string(v_txt)
                            kw_args[k] = v_txt
                    elif arg.type == "element_value_array_initializer":
                        for sub in arg.children:
                            if sub.type == "string_literal":
                                pos_args.append(_unquote_java_string(_node_text(sub, code_bytes)))
            found.append((aname, pos_args, kw_args))
    return found


def _extract_leading_doc(node: Node, code_bytes: bytes) -> str:
    """Extract Javadoc / line comments immediately preceding a declaration."""
    comments: list[str] = []
    prev = node.prev_sibling
    while prev is not None and prev.type in {"block_comment", "line_comment"}:
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
    body_types = {"class_body", "interface_body", "enum_body", "block", "constructor_body"}
    end_byte = node.end_byte
    for child in node.children:
        if child.type in body_types:
            end_byte = child.start_byte
            break
    sig = code_bytes[node.start_byte : end_byte].decode("utf-8", errors="replace").strip()
    return " ".join(sig.split())


def _extract_record_or_dto_fields(node: Node, code_bytes: bytes) -> list[str]:
    fields: list[str] = []
    if node.type == "record_declaration":
        params = node.child_by_field_name("parameters")
        if params is not None:
            for p in params.children:
                if p.type == "formal_parameter":
                    n = p.child_by_field_name("name")
                    if n is not None:
                        fields.append(_node_text(n, code_bytes).strip())
        return fields
    body = node.child_by_field_name("body")
    if body is not None:
        for child in body.children:
            if child.type == "field_declaration":
                for sub in child.children:
                    if sub.type == "variable_declarator":
                        n = sub.child_by_field_name("name")
                        if n is not None:
                            fields.append(_node_text(n, code_bytes).strip())
    return fields


class JavaSpringPlugin:
    """Java & Spring Boot LanguagePlugin backed by tree-sitter-java."""

    def __init__(self) -> None:
        self._parser: Parser = _get_java_parser()
        self.last_parse_diagnostics: ParseDiagnostics = ParseDiagnostics(
            parser_backend="tree-sitter-java"
        )

    @property
    def language_id(self) -> str:
        return "java"

    @property
    def file_extensions(self) -> tuple[str, ...]:
        return (".java",)

    @property
    def manifest_files(self) -> tuple[str, ...]:
        return ("pom.xml", "build.gradle", "build.gradle.kts")

    def can_handle(self, file_path: str) -> bool:
        lower = file_path.lower()
        name = Path(lower).name
        return lower.endswith(".java") or name in self.manifest_files

    @staticmethod
    def _derive_module_prefix(file_path: str) -> str:
        if not file_path:
            return "java"
        stem = Path(file_path).stem
        return stem or "java"

    def extract_manifest_dependencies(
        self,
        file_path: str,
        content: str,
    ) -> list[DependencyEdge]:
        if not content or not content.strip():
            self.last_parse_diagnostics = ParseDiagnostics(parser_backend="manifest")
            return []

        name = Path(file_path).name.lower()
        deps: list[DependencyEdge] = []

        if name == "pom.xml":
            try:
                root = ET.fromstring(content)
            except ET.ParseError as exc:
                logger.warning("XML parse error in %s: %s", file_path, exc)
                self.last_parse_diagnostics = ParseDiagnostics(
                    parser_backend="xml_etree",
                    has_parse_error=True,
                    abstained=True,
                    reason=f"ET.ParseError: {exc}",
                )
                return []

            artifact_el = root.find("./{*}artifactId")
            source_id = (
                artifact_el.text.strip()
                if artifact_el is not None and artifact_el.text
                else Path(file_path).stem
            )
            for dep in root.findall(".//{*}dependency"):
                gid_el = dep.find("./{*}groupId")
                aid_el = dep.find("./{*}artifactId")
                ver_el = dep.find("./{*}version")
                scope_el = dep.find("./{*}scope")
                gid = gid_el.text.strip() if gid_el is not None and gid_el.text else ""
                aid = aid_el.text.strip() if aid_el is not None and aid_el.text else ""
                if not gid and not aid:
                    continue
                pkg = f"{gid}:{aid}" if gid and aid else (aid or gid)
                ver = ver_el.text.strip() if ver_el is not None and ver_el.text else "*"
                raw_scope = (
                    scope_el.text.strip().lower()
                    if scope_el is not None and scope_el.text
                    else "compile"
                )
                scope = "dev" if raw_scope == "test" else raw_scope
                deps.append(
                    DependencyEdge(
                        source_identifier=source_id,
                        target_package=pkg,
                        version_constraint=ver,
                        scope=scope,
                        manifest_path=file_path,
                    )
                )
            self.last_parse_diagnostics = ParseDiagnostics(parser_backend="xml_etree")
            return deps

        if name in {"build.gradle", "build.gradle.kts"}:
            source_id = Path(file_path).parent.name or Path(file_path).stem
            for match in _GRADLE_DEP_RE.finditer(content):
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
            self.last_parse_diagnostics = ParseDiagnostics(parser_backend="gradle-regex")
            return deps

        self.last_parse_diagnostics = ParseDiagnostics(parser_backend="manifest")
        return deps

    def extract_ast(
        self,
        code: str,
        file_path: str = "",
        repo_name: str = "",
        module_prefix: str = "",
    ) -> tuple[list[ParsedSymbol], list[ParsedEdge]]:
        if not code or not code.strip():
            self.last_parse_diagnostics = ParseDiagnostics(parser_backend="tree-sitter-java")
            return [], []

        code_bytes = code.encode("utf-8")
        tree = self._parser.parse(code_bytes)
        has_err, err_cnt, miss_cnt = inspect_tree_errors(tree.root_node)
        if has_err:
            logger.warning(
                "JavaSpringPlugin parse error in %s (errors=%d, missing=%d); abstaining.",
                file_path or "<memory>",
                err_cnt,
                miss_cnt,
            )
            self.last_parse_diagnostics = ParseDiagnostics(
                parser_backend="tree-sitter-java",
                has_parse_error=True,
                abstained=True,
                error_node_count=err_cnt,
                missing_node_count=miss_cnt,
                reason="Syntax/parse error detected by tree-sitter-java",
            )
            return [], []

        self.last_parse_diagnostics = ParseDiagnostics(parser_backend="tree-sitter-java")
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

        def scan_calls_and_types(node: Node, owner_qn: str) -> None:
            stack: list[Node] = [node]
            while stack:
                curr = stack.pop()
                if curr.type == "method_invocation":
                    name_node = curr.child_by_field_name("name")
                    obj_node = curr.child_by_field_name("object")
                    if name_node is not None:
                        m_name = _node_text(name_node, code_bytes).strip()
                        if obj_node is not None:
                            obj_txt = _node_text(obj_node, code_bytes).strip()
                            target = (
                                f"{obj_txt}.{m_name}"
                                if obj_txt != "this"
                                else qualify_in_class(m_name)
                            )
                        else:
                            target = m_name
                        add_edge(owner_qn, target, "CALLS")
                elif curr.type == "object_creation_expression":
                    type_node = curr.child_by_field_name("type")
                    tname = _extract_type_name(type_node, code_bytes)
                    if tname and tname not in _BUILTIN_TYPES:
                        add_edge(owner_qn, tname, "USES_TYPE")
                stack.extend(reversed(curr.children))

        def qualify_in_class(member: str) -> str:
            if type_stack:
                return f"{type_stack[-1][1]}.{member}"
            return member

        def handle_type(node: Node) -> None:
            name_node = node.child_by_field_name("name")
            if name_node is None:
                return
            type_name = _node_text(name_node, code_bytes).strip()
            if not type_name:
                return

            qn = qualify(type_name)
            bare = f"{type_stack[-1][0]}.{type_name}" if type_stack else type_name
            sym_kind = "interface" if node.type == "interface_declaration" else "class"
            sig = _extract_signature_before_body(node, code_bytes)
            doc = _extract_leading_doc(node, code_bytes)
            attrs = _extract_annotations(node, code_bytes)

            notes: list[str] = []
            class_route = ""
            is_spring_bean = False
            for aname, pos_args, kw_args in attrs:
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

            if node.type == "record_declaration" or (
                type_name.endswith(("Dto", "DTO", "Request", "Response", "Payload"))
            ):
                dto_fields = _extract_record_or_dto_fields(node, code_bytes)
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

            # Inheritance: superclass, super_interfaces, extends_interfaces
            for child in node.children:
                if child.type == "superclass":
                    for sub in child.children:
                        if sub.is_named:
                            base = _extract_type_name(sub, code_bytes)
                            if base:
                                add_edge(qn, base, "INHERITS")
                elif child.type in {"super_interfaces", "extends_interfaces"}:
                    for sub in child.children:
                        if sub.type == "type_list":
                            for iface_node in sub.children:
                                if iface_node.is_named:
                                    iface = _extract_type_name(iface_node, code_bytes)
                                    if iface:
                                        add_edge(qn, iface, "INHERITS")

            type_stack.append((type_name, qn))
            class_route_stack.append(class_route)
            body = node.child_by_field_name("body")
            if body is not None:
                walk_body(body, qn)
            class_route_stack.pop()
            type_stack.pop()

        def handle_field(node: Node, owner_class_qn: str) -> None:
            type_node = node.child_by_field_name("type")
            ftype = _extract_type_name(type_node, code_bytes)
            attrs = _extract_annotations(node, code_bytes)
            is_autowired = any(aname in {"Autowired", "Inject"} for aname, _, _ in attrs)
            if ftype and ftype not in _BUILTIN_TYPES:
                add_edge(owner_class_qn, ftype, "USES_TYPE")
                if is_autowired:
                    add_edge(owner_class_qn, ftype, "PROVISIONED_BY")

        def handle_method(node: Node, owner_class_qn: str) -> None:
            name_node = node.child_by_field_name("name")
            if name_node is None:
                return
            m_name = _node_text(name_node, code_bytes).strip()
            if not m_name:
                return

            qn = qualify(m_name)
            bare = f"{type_stack[-1][0]}.{m_name}" if type_stack else m_name
            sig = _extract_signature_before_body(node, code_bytes)
            doc = _extract_leading_doc(node, code_bytes)
            attrs = _extract_annotations(node, code_bytes)
            class_route = class_route_stack[-1] if class_route_stack else ""

            notes: list[str] = []
            is_test = False
            is_bean = False
            for aname, pos_args, kw_args in attrs:
                if aname in _SPRING_HTTP_MAPPINGS:
                    http_method = _SPRING_HTTP_MAPPINGS[aname]
                    if aname == "RequestMapping" and "method" in kw_args:
                        raw_m = kw_args["method"].rsplit(".", 1)[-1].upper()
                        if raw_m in {"GET", "POST", "PUT", "DELETE", "PATCH"}:
                            http_method = raw_m
                    sub_route = (
                        pos_args[0] if pos_args else kw_args.get("value", kw_args.get("path", ""))
                    )
                    full_route = _combine_route(class_route, sub_route)
                    notes.append(f"@route {http_method} {full_route}")
                elif aname == "Transactional":
                    notes.append("@Transactional")
                elif aname == "Bean":
                    is_bean = True
                    notes.append("@Bean")
                elif aname in {"Test", "ParameterizedTest"}:
                    is_test = True
                    notes.append("@Test")

            # Parameter types -> USES_TYPE (and PROVISIONED_BY on constructors)
            params = node.child_by_field_name("parameters")
            if params is not None:
                for p in params.children:
                    if p.type == "formal_parameter":
                        ptype = _extract_type_name(p.child_by_field_name("type"), code_bytes)
                        if ptype and ptype not in _BUILTIN_TYPES:
                            add_edge(qn, ptype, "USES_TYPE")
                            if node.type == "constructor_declaration":
                                add_edge(owner_class_qn, ptype, "PROVISIONED_BY")

            ret_type = _extract_type_name(node.child_by_field_name("type"), code_bytes)
            if ret_type and ret_type not in _BUILTIN_TYPES:
                add_edge(qn, ret_type, "USES_TYPE")
                if is_bean:
                    add_edge(qn, ret_type, "PROVISIONED_BY")

            if is_test and owner_class_qn:
                add_edge(owner_class_qn, qn, "TESTED_BY")

            summary = " ".join(dict.fromkeys(notes))
            merged_doc = f"{doc} {summary}".strip() if summary else doc
            symbols.append(
                ParsedSymbol(
                    name=m_name,
                    qualified_name=qn,
                    bare_name=bare,
                    symbol_kind="method",
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

            body = node.child_by_field_name("body")
            if body is not None:
                scan_calls_and_types(body, qn)

        def walk_body(body: Node, owner_class_qn: str) -> None:
            for child in body.children:
                if child.type in _TYPE_DECL_NODES:
                    handle_type(child)
                elif child.type == "field_declaration":
                    handle_field(child, owner_class_qn)
                elif child.type in {"method_declaration", "constructor_declaration"}:
                    handle_method(child, owner_class_qn)

        for child in tree.root_node.children:
            if child.type == "package_declaration":
                for sub in child.children:
                    if sub.type in {"scoped_identifier", "identifier"}:
                        package_name = _node_text(sub, code_bytes).strip()
            elif child.type == "import_declaration":
                raw_imp = _node_text(child, code_bytes).strip()
                raw_imp = (
                    raw_imp.removeprefix("import")
                    .strip()
                    .removeprefix("static")
                    .strip()
                    .rstrip(";")
                    .strip()
                )
                if raw_imp and raw_imp not in imports:
                    imports.append(raw_imp)
            elif child.type in _TYPE_DECL_NODES:
                handle_type(child)

        return symbols, edges
