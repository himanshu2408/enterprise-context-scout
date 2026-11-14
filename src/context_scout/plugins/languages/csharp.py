"""C# language plugin using tree-sitter-c-sharp AST traversal."""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import tree_sitter_c_sharp as tscsharp
from tree_sitter import Language, Node, Parser

from context_scout.ast_engine.extractor import (
    ParseDiagnostics,
    _node_text,
    inspect_tree_errors,
)
from context_scout.ast_engine.models import ParsedEdge, ParsedSymbol
from context_scout.plugins.base import DependencyEdge

logger = logging.getLogger(__name__)

_CSHARP_LANGUAGE = Language(tscsharp.language())

_DI_METHODS = frozenset(
    {
        "AddScoped",
        "AddSingleton",
        "AddTransient",
        "AddHostedService",
        "TryAddScoped",
        "TryAddSingleton",
        "TryAddTransient",
    }
)

_TYPE_DECLARATION_NODES = frozenset(
    {
        "class_declaration",
        "interface_declaration",
        "record_declaration",
        "struct_declaration",
        "enum_declaration",
    }
)

_CSHARP_BUILTIN_CALLS = frozenset(
    {
        "nameof",
        "typeof",
        "sizeof",
        "default",
        "checked",
        "unchecked",
    }
)


def _collapse_ws(text: str) -> str:
    return " ".join(text.split())


def _extract_type_name(node: Node, code_bytes: bytes) -> str:
    """Extract the base identifier/qualified name from a C# type syntax node."""
    if node.type in {"identifier", "qualified_name", "predefined_type"}:
        return _node_text(node, code_bytes)
    if node.type == "generic_name":
        name_node = node.child_by_field_name("name")
        if name_node is not None:
            return _node_text(name_node, code_bytes)
        for child in node.children:
            if child.type == "identifier":
                return _node_text(child, code_bytes)
    if node.type == "alias_qualified_name":
        name_node = node.child_by_field_name("name")
        if name_node is not None:
            return _extract_type_name(name_node, code_bytes)
    for child in node.children:
        if child.type in {"identifier", "qualified_name", "generic_name"}:
            return _extract_type_name(child, code_bytes)
    return ""


def _extract_signature_before_body(node: Node, code_bytes: bytes) -> str:
    """Build a compact single-line signature stopping before the declaration body."""
    body_node = node.child_by_field_name("body")
    end_byte = node.end_byte
    if body_node is not None:
        end_byte = body_node.start_byte
    else:
        for child in node.children:
            if child.type in {"block", "arrow_expression_clause", "declaration_list", ";"}:
                end_byte = min(end_byte, child.start_byte)
                break
    raw = code_bytes[node.start_byte : end_byte].decode("utf-8", errors="replace")
    return _collapse_ws(raw)


def _extract_leading_doc_comment(node: Node, code_bytes: bytes) -> str:
    """Collect immediately preceding XML/line comments as a concise docstring."""
    comments: list[str] = []
    sibling = node.prev_sibling
    while sibling is not None and sibling.type == "comment":
        text = _node_text(sibling, code_bytes).strip()
        if text.startswith("///"):
            comments.append(text[3:].strip())
        elif text.startswith("//"):
            comments.append(text[2:].strip())
        sibling = sibling.prev_sibling
    if not comments:
        return ""
    comments.reverse()
    return "\n".join(line for line in comments if line)


class CSharpPlugin:
    """Language plugin for C# (.cs) source and .csproj/.sln manifests."""

    def __init__(self) -> None:
        self._parser = Parser(_CSHARP_LANGUAGE)
        self.last_parse_diagnostics: ParseDiagnostics | None = None

    @property
    def language_id(self) -> str:
        return "csharp"

    @property
    def file_extensions(self) -> tuple[str, ...]:
        return (".cs",)

    @property
    def manifest_files(self) -> tuple[str, ...]:
        return (".csproj", ".sln", "Directory.Build.props", "packages.config")

    def can_handle(self, file_path: str) -> bool:
        lower = file_path.lower()
        return any(lower.endswith(ext) for ext in self.file_extensions)

    def _derive_module_prefix(self, file_path: str) -> str:
        if not file_path:
            return "csharp_module"
        stem = re.sub(r"\.cs$", "", file_path, flags=re.IGNORECASE)
        normalized = stem.replace("\\", "/").strip("/")
        dotted = ".".join(part for part in normalized.split("/") if part)
        return dotted or "csharp_module"

    def extract_manifest_dependencies(
        self,
        file_path: str,
        content: str,
    ) -> list[DependencyEdge]:
        """Extract NuGet dependencies from .csproj, Directory.Build.props, or packages.config."""
        deps: list[DependencyEdge] = []
        lower_name = Path(file_path).name.lower()

        if lower_name.endswith(".csproj") or lower_name == "directory.build.props":
            try:
                root = ET.fromstring(content)
                for pkg_ref in root.iter():
                    tag = pkg_ref.tag.split("}")[-1]
                    if tag in {"PackageReference", "PackageVersion"}:
                        pkg_name = pkg_ref.attrib.get("Include") or pkg_ref.attrib.get("Update")
                        version = pkg_ref.attrib.get("Version", "")
                        if not version:
                            for child in pkg_ref:
                                if child.tag.split("}")[-1] == "Version" and child.text:
                                    version = child.text.strip()
                                    break
                        if pkg_name:
                            deps.append(
                                DependencyEdge(
                                    source_identifier=file_path,
                                    target_package=pkg_name.strip(),
                                    version_constraint=version.strip() or "*",
                                    scope="compile",
                                    manifest_path=file_path,
                                )
                            )
                    elif tag == "ProjectReference":
                        include = pkg_ref.attrib.get("Include", "").strip()
                        if include:
                            proj_name = Path(include.replace("\\", "/")).stem or include
                            deps.append(
                                DependencyEdge(
                                    source_identifier=file_path,
                                    target_package=proj_name,
                                    version_constraint="*",
                                    scope="project_reference",
                                    manifest_path=file_path,
                                )
                            )
                self.last_parse_diagnostics = ParseDiagnostics(parser_backend="xml_etree")
                return deps
            except ET.ParseError as exc:
                logger.warning(
                    "XML parse error in %s (%s); using regex PackageReference fallback.",
                    file_path,
                    exc,
                )
                self.last_parse_diagnostics = ParseDiagnostics(
                    parser_backend="xml_regex_fallback",
                    has_parse_error=True,
                    low_confidence_fallback=True,
                    reason=f"ET.ParseError: {exc}",
                )
                pkg_pattern = re.compile(
                    r'<Package(?:Reference|Version)\s+[^>]*?Include=["\']([^"\']+)["\']'
                    r'(?:[^>]*?Version=["\']([^"\']+)["\'])?',
                    re.IGNORECASE,
                )
                for match in pkg_pattern.finditer(content):
                    deps.append(
                        DependencyEdge(
                            source_identifier=file_path,
                            target_package=match.group(1).strip(),
                            version_constraint=(match.group(2) or "*").strip(),
                            scope="compile",
                            manifest_path=file_path,
                        )
                    )
                proj_pattern = re.compile(
                    r'<ProjectReference\s+[^>]*?Include=["\']([^"\']+)["\']',
                    re.IGNORECASE,
                )
                for match in proj_pattern.finditer(content):
                    include = match.group(1).strip()
                    proj_name = Path(include.replace("\\", "/")).stem or include
                    deps.append(
                        DependencyEdge(
                            source_identifier=file_path,
                            target_package=proj_name,
                            version_constraint="*",
                            scope="project_reference",
                            manifest_path=file_path,
                        )
                    )
                return deps

        if lower_name == "packages.config":
            try:
                root = ET.fromstring(content)
                for pkg in root.iter("package"):
                    pkg_id = pkg.attrib.get("id")
                    version = pkg.attrib.get("version", "")
                    if pkg_id:
                        deps.append(
                            DependencyEdge(
                                source_identifier=file_path,
                                target_package=pkg_id.strip(),
                                version_constraint=version.strip() or "*",
                                scope="compile",
                                manifest_path=file_path,
                            )
                        )
                self.last_parse_diagnostics = ParseDiagnostics(parser_backend="xml_etree")
            except ET.ParseError as exc:
                logger.warning("XML parse error in %s: %s", file_path, exc)
                self.last_parse_diagnostics = ParseDiagnostics(
                    parser_backend="xml_etree",
                    has_parse_error=True,
                    abstained=True,
                    reason=f"ET.ParseError: {exc}",
                )
            return deps

        return deps

    def extract_ast(
        self,
        code: str,
        file_path: str = "",
        repo_name: str = "",
        module_prefix: str = "",
    ) -> tuple[list[ParsedSymbol], list[ParsedEdge]]:
        """Extract C# namespaces, types, methods, inheritance, usings, and DI edges."""
        if not code or not code.strip():
            self.last_parse_diagnostics = ParseDiagnostics(parser_backend="tree-sitter-c-sharp")
            return [], []

        code_bytes = code.encode("utf-8")
        tree = self._parser.parse(code_bytes)
        has_err, err_cnt, miss_cnt = inspect_tree_errors(tree.root_node)
        if has_err:
            logger.warning(
                "CSharpPlugin parse error in %s (errors=%d, missing=%d); abstaining.",
                file_path or "<memory>",
                err_cnt,
                miss_cnt,
            )
            self.last_parse_diagnostics = ParseDiagnostics(
                parser_backend="tree-sitter-c-sharp",
                has_parse_error=True,
                abstained=True,
                error_node_count=err_cnt,
                missing_node_count=miss_cnt,
                reason="Syntax/parse error detected by tree-sitter-c-sharp",
            )
            return [], []

        self.last_parse_diagnostics = ParseDiagnostics(parser_backend="tree-sitter-c-sharp")
        prefix = module_prefix or self._derive_module_prefix(file_path)

        symbols: list[ParsedSymbol] = []
        edges: list[ParsedEdge] = []
        usings: list[str] = []
        ns_stack: list[str] = []
        type_stack: list[tuple[str, str]] = []
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

        def qualify_name(leaf: str) -> str:
            if type_stack:
                return f"{type_stack[-1][1]}.{leaf}"
            if ns_stack:
                return f"{'.'.join(ns_stack)}.{leaf}"
            return f"{prefix}.{leaf}"

        def record_using(node: Node) -> None:
            target = ""
            for child in node.children:
                if child.type == "name_equals":
                    continue
                if child.type in {"qualified_name", "identifier"}:
                    target = _node_text(child, code_bytes)
            if target and target not in usings:
                usings.append(target)

        def scan_invocations(node: Node, owner_qn: str) -> None:
            if node.type == "invocation_expression":
                func_node = node.child_by_field_name("function")
                method_name = ""
                generic_node: Node | None = None
                if func_node is not None:
                    if func_node.type == "member_access_expression":
                        member_name = func_node.child_by_field_name("name")
                        if member_name is not None:
                            if member_name.type == "generic_name":
                                generic_node = member_name
                                method_name = _extract_type_name(member_name, code_bytes)
                            else:
                                method_name = _node_text(member_name, code_bytes)
                    elif func_node.type == "generic_name":
                        generic_node = func_node
                        method_name = _extract_type_name(func_node, code_bytes)
                    elif func_node.type == "identifier":
                        method_name = _node_text(func_node, code_bytes)

                if method_name in _DI_METHODS and generic_node is not None:
                    for child in generic_node.children:
                        if child.type == "type_argument_list":
                            type_args = [
                                _extract_type_name(arg, code_bytes)
                                for arg in child.children
                                if arg.is_named
                            ]
                            type_args = [t for t in type_args if t]
                            if len(type_args) >= 2:
                                add_edge(type_args[0], type_args[1], "PROVISIONED_BY")
                                add_edge(owner_qn, type_args[1], "REGISTERS")
                            elif len(type_args) == 1:
                                add_edge(owner_qn, type_args[0], "PROVISIONED_BY")
                elif method_name and method_name not in _CSHARP_BUILTIN_CALLS:
                    add_edge(owner_qn, method_name, "CALLS")

            for child in node.children:
                if child.type not in _TYPE_DECLARATION_NODES:
                    scan_invocations(child, owner_qn)

        def handle_type_declaration(node: Node) -> None:
            name_node = node.child_by_field_name("name")
            if name_node is None:
                return
            type_name = _node_text(name_node, code_bytes)
            if not type_name:
                return
            qn = qualify_name(type_name)
            symbol_kind = "interface" if node.type == "interface_declaration" else "class"
            sig = _extract_signature_before_body(node, code_bytes)
            doc = _extract_leading_doc_comment(node, code_bytes)
            symbols.append(
                ParsedSymbol(
                    name=type_name,
                    qualified_name=qn,
                    bare_name=type_name,
                    symbol_kind=symbol_kind,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    start_byte=node.start_byte,
                    end_byte=node.end_byte,
                    signature=sig,
                    signature_skeleton=sig,
                    docstring=doc,
                    repo_name=repo_name,
                )
            )
            for imported in usings:
                add_edge(qn, imported, "IMPORTS")

            for child in node.children:
                if child.type == "base_list":
                    for base_child in child.children:
                        if not base_child.is_named:
                            continue
                        base_name = _extract_type_name(base_child, code_bytes)
                        if base_name:
                            add_edge(qn, base_name, "INHERITS")

            type_stack.append((type_name, qn))
            body_node = node.child_by_field_name("body")
            if body_node is not None:
                walk(body_node)
            else:
                for child in node.children:
                    if child.type == "declaration_list":
                        walk(child)
            type_stack.pop()

        def handle_method_declaration(node: Node) -> None:
            name_node = node.child_by_field_name("name")
            if name_node is None:
                return
            method_name = _node_text(name_node, code_bytes)
            if not method_name:
                return
            qn = qualify_name(method_name)
            kind = "method" if type_stack else "function"
            bare = f"{type_stack[-1][0]}.{method_name}" if type_stack else method_name
            sig = _extract_signature_before_body(node, code_bytes)
            doc = _extract_leading_doc_comment(node, code_bytes)
            symbols.append(
                ParsedSymbol(
                    name=method_name,
                    qualified_name=qn,
                    bare_name=bare,
                    symbol_kind=kind,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    start_byte=node.start_byte,
                    end_byte=node.end_byte,
                    signature=sig,
                    signature_skeleton=sig,
                    docstring=doc,
                    repo_name=repo_name,
                )
            )
            scan_invocations(node, qn)

        def walk(node: Node) -> None:
            file_scoped_pushed = False
            for child in node.children:
                if child.type == "using_directive":
                    record_using(child)
                elif child.type == "file_scoped_namespace_declaration":
                    ns_name_node = child.child_by_field_name("name")
                    ns_name = _node_text(ns_name_node, code_bytes)
                    has_nested_body = any(
                        c.type in {"declaration_list", *_TYPE_DECLARATION_NODES}
                        for c in child.children
                    )
                    if ns_name:
                        ns_stack.append(ns_name)
                        if has_nested_body:
                            walk(child)
                            ns_stack.pop()
                        else:
                            file_scoped_pushed = True
                elif child.type == "namespace_declaration":
                    ns_name_node = child.child_by_field_name("name")
                    ns_name = _node_text(ns_name_node, code_bytes)
                    if ns_name:
                        ns_stack.append(ns_name)
                    body_node = child.child_by_field_name("body")
                    if body_node is not None:
                        walk(body_node)
                    else:
                        walk(child)
                    if ns_name:
                        ns_stack.pop()
                elif child.type in _TYPE_DECLARATION_NODES:
                    handle_type_declaration(child)
                elif child.type in {"method_declaration", "constructor_declaration"}:
                    handle_method_declaration(child)
                elif child.type == "declaration_list":
                    walk(child)
                elif child.type == "global_statement":
                    scan_invocations(child, prefix)
            if file_scoped_pushed:
                ns_stack.pop()

        walk(tree.root_node)
        return symbols, edges
