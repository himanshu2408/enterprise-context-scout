"""Rust language plugin using tree-sitter-rust and tree-sitter-toml AST traversal."""

from __future__ import annotations

import logging
import re
import tomllib
from typing import Any

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

_RUST_BUILTIN_TYPES = frozenset(
    {
        "bool",
        "char",
        "str",
        "u8",
        "u16",
        "u32",
        "u64",
        "u128",
        "usize",
        "i8",
        "i16",
        "i32",
        "i64",
        "i128",
        "isize",
        "f32",
        "f64",
        "Self",
        "String",
        "Vec",
        "Option",
        "Result",
        "Box",
        "Rc",
        "Arc",
        "HashMap",
        "HashSet",
        "BTreeMap",
        "BTreeSet",
    }
)

_RUST_BUILTIN_MACROS = frozenset(
    {
        "println",
        "print",
        "eprintln",
        "eprint",
        "format",
        "vec",
        "panic",
        "unreachable",
        "unimplemented",
        "todo",
        "assert",
        "assert_eq",
        "assert_ne",
        "debug_assert",
        "debug_assert_eq",
        "dbg",
        "cfg",
    }
)

_ACTIX_ROUTE_ATTRS: dict[str, str] = {
    "get": "GET",
    "post": "POST",
    "put": "PUT",
    "delete": "DELETE",
    "patch": "PATCH",
    "head": "HEAD",
    "options": "OPTIONS",
    "route": "ANY",
}

_STRING_LIT_RE = re.compile(r'"([^"]+)"')


def _collapse_ws(text: str) -> str:
    return " ".join(text.split())


def _extract_leading_attributes_and_docs(
    node: Node,
    code_bytes: bytes,
) -> tuple[list[tuple[str, str]], str]:
    """Collect preceding attribute_item nodes and doc comments in source order."""
    attrs: list[tuple[str, str]] = []
    docs: list[str] = []
    sibling = node.prev_sibling
    while sibling is not None and sibling.type in {
        "attribute_item",
        "line_comment",
        "block_comment",
    }:
        if sibling.type == "attribute_item":
            attr_node = next((c for c in sibling.children if c.type == "attribute"), None)
            if attr_node is not None:
                raw_attr = _node_text(attr_node, code_bytes).strip()
                attr_name = ""
                for sub in attr_node.children:
                    if sub.type in {"identifier", "scoped_identifier"}:
                        attr_name = _node_text(sub, code_bytes).strip()
                        break
                attrs.append((attr_name, raw_attr))
        elif sibling.type == "line_comment":
            text = _node_text(sibling, code_bytes).strip()
            if text.startswith("///"):
                docs.append(text[3:].strip())
        sibling = sibling.prev_sibling
    attrs.reverse()
    docs.reverse()
    return attrs, "\n".join(d for d in docs if d)


def _extract_signature_before_body(node: Node, code_bytes: bytes) -> str:
    body_node = node.child_by_field_name("body")
    end_byte = body_node.start_byte if body_node is not None else node.end_byte
    raw = code_bytes[node.start_byte : end_byte].decode("utf-8", errors="replace")
    return _collapse_ws(raw)


def _extract_rust_type_names(node: Node, code_bytes: bytes) -> list[str]:
    """Iteratively collect non-builtin type identifiers from a Rust type AST subtree."""
    found: list[str] = []
    stack: list[Node] = [node]
    while stack:
        curr = stack.pop()
        if curr.type == "type_identifier":
            name = _node_text(curr, code_bytes).strip()
            if name and name not in _RUST_BUILTIN_TYPES:
                found.append(name)
        elif curr.type == "scoped_type_identifier":
            name_node = curr.child_by_field_name("name")
            if name_node is not None:
                name = _node_text(name_node, code_bytes).strip()
                if name and name not in _RUST_BUILTIN_TYPES:
                    found.append(name)
        else:
            stack.extend(reversed(curr.children))
    return list(dict.fromkeys(found))


def _extract_primary_type_name(node: Node | None, code_bytes: bytes) -> str:
    if node is None:
        return ""
    if node.type == "type_identifier":
        return _node_text(node, code_bytes).strip()
    if node.type == "scoped_type_identifier":
        name_node = node.child_by_field_name("name")
        if name_node is not None:
            return _node_text(name_node, code_bytes).strip()
    if node.type == "generic_type":
        type_node = node.child_by_field_name("type")
        if type_node is not None:
            return _extract_primary_type_name(type_node, code_bytes)
    names = _extract_rust_type_names(node, code_bytes)
    if names:
        return names[0]
    return _node_text(node, code_bytes).strip().split("<", 1)[0].rsplit("::", 1)[-1]


def _extract_use_paths(use_node: Node, code_bytes: bytes) -> list[str]:
    """Extract normalized import paths from a Rust use_declaration."""
    raw = _node_text(use_node, code_bytes).strip()
    raw = re.sub(r"^(?:pub(?:\([^)]*\))?\s+)?use\s+", "", raw).rstrip(";").strip()
    if not raw:
        return []
    if "{" not in raw:
        return [raw.split(" as ", 1)[0].strip()]
    prefix, rest = raw.split("{", 1)
    prefix = prefix.strip().rstrip(":")
    inner = rest.rsplit("}", 1)[0]
    results: list[str] = []
    for item in inner.split(","):
        leaf = item.strip().split(" as ", 1)[0].strip()
        if not leaf or leaf == "self":
            if prefix:
                results.append(prefix)
        elif prefix:
            results.append(f"{prefix}::{leaf}")
        else:
            results.append(leaf)
    return results


def _extract_callee_name(call_node: Node, code_bytes: bytes) -> str:
    func_node = call_node.child_by_field_name("function")
    if func_node is None:
        return ""
    if func_node.type == "identifier":
        return _node_text(func_node, code_bytes).strip()
    if func_node.type == "scoped_identifier":
        name_node = func_node.child_by_field_name("name")
        if name_node is not None:
            return _node_text(name_node, code_bytes).strip()
    if func_node.type == "field_expression":
        field_node = func_node.child_by_field_name("field")
        if field_node is not None:
            return _node_text(field_node, code_bytes).strip()
    return ""


class RustPlugin:
    """Language plugin for Rust (.rs) source and Cargo.toml manifests."""

    def __init__(self) -> None:
        self._parser: Parser = tslp.get_parser("rust")
        self._toml_parser: Parser = tslp.get_parser("toml")
        self.last_parse_diagnostics: ParseDiagnostics | None = None

    @property
    def language_id(self) -> str:
        return "rust"

    @property
    def file_extensions(self) -> tuple[str, ...]:
        return (".rs",)

    @property
    def manifest_files(self) -> tuple[str, ...]:
        return ("Cargo.toml",)

    def can_handle(self, file_path: str) -> bool:
        return file_path.lower().endswith(".rs")

    def _derive_module_prefix(self, file_path: str) -> str:
        if not file_path:
            return "crate"
        stem = re.sub(r"\.rs$", "", file_path, flags=re.IGNORECASE)
        normalized = stem.replace("\\", "/").strip("/")
        parts = [p for p in normalized.split("/") if p and p != "src"]
        return "::".join(parts) if parts else "crate"

    def extract_manifest_dependencies(
        self,
        file_path: str,
        content: str,
    ) -> list[DependencyEdge]:
        """Extract crate dependencies from Cargo.toml, failing closed on TOML errors."""
        if not content or not content.strip():
            self.last_parse_diagnostics = ParseDiagnostics(parser_backend="tree-sitter-toml")
            return []

        content_bytes = content.encode("utf-8")
        tree = self._toml_parser.parse(content_bytes)
        has_err, err_cnt, miss_cnt = inspect_tree_errors(tree.root_node)
        if has_err:
            logger.warning(
                "RustPlugin Cargo.toml parse error in %s (errors=%d, missing=%d); abstaining.",
                file_path or "<memory>",
                err_cnt,
                miss_cnt,
            )
            self.last_parse_diagnostics = ParseDiagnostics(
                parser_backend="tree-sitter-toml",
                has_parse_error=True,
                abstained=True,
                error_node_count=err_cnt,
                missing_node_count=miss_cnt,
                reason="Syntax/parse error detected by tree-sitter-toml",
            )
            return []

        try:
            parsed: dict[str, Any] = tomllib.loads(content)
        except tomllib.TOMLDecodeError as exc:
            logger.warning("RustPlugin TOMLDecodeError in %s (%s); abstaining.", file_path, exc)
            self.last_parse_diagnostics = ParseDiagnostics(
                parser_backend="tree-sitter-toml",
                has_parse_error=True,
                abstained=True,
                reason=f"TOMLDecodeError: {exc}",
            )
            return []

        self.last_parse_diagnostics = ParseDiagnostics(parser_backend="tree-sitter-toml")
        pkg_meta = parsed.get("package")
        source_id = file_path
        if isinstance(pkg_meta, dict) and isinstance(pkg_meta.get("name"), str):
            source_id = pkg_meta["name"]

        deps: list[DependencyEdge] = []

        def collect_section(section: Any, scope: str) -> None:
            if not isinstance(section, dict):
                return
            for pkg_name, spec in section.items():
                if not isinstance(pkg_name, str):
                    continue
                version = "*"
                if isinstance(spec, str):
                    version = spec.strip() or "*"
                elif isinstance(spec, dict):
                    if isinstance(spec.get("version"), str):
                        version = spec["version"].strip() or "*"
                    elif spec.get("workspace") is True:
                        version = "workspace"
                    elif isinstance(spec.get("path"), str):
                        version = f"path:{spec['path']}"
                deps.append(
                    DependencyEdge(
                        source_identifier=source_id,
                        target_package=pkg_name,
                        version_constraint=version,
                        scope=scope,
                        manifest_path=file_path,
                    )
                )

        collect_section(parsed.get("dependencies"), "runtime")
        collect_section(parsed.get("dev-dependencies"), "dev")
        collect_section(parsed.get("build-dependencies"), "compile")

        workspace = parsed.get("workspace")
        if isinstance(workspace, dict):
            collect_section(workspace.get("dependencies"), "runtime")

        target_map = parsed.get("target")
        if isinstance(target_map, dict):
            for target_cfg in target_map.values():
                if isinstance(target_cfg, dict):
                    collect_section(target_cfg.get("dependencies"), "runtime")
                    collect_section(target_cfg.get("dev-dependencies"), "dev")
                    collect_section(target_cfg.get("build-dependencies"), "compile")

        return deps

    def extract_ast(
        self,
        code: str,
        file_path: str = "",
        repo_name: str = "",
        module_prefix: str = "",
    ) -> tuple[list[ParsedSymbol], list[ParsedEdge]]:
        """Extract Rust structs, enums, traits, impls, functions, macros, routes, and edges."""
        if not code or not code.strip():
            self.last_parse_diagnostics = ParseDiagnostics(parser_backend="tree-sitter-rust")
            return [], []

        code_bytes = code.encode("utf-8")
        tree = self._parser.parse(code_bytes)
        has_err, err_cnt, miss_cnt = inspect_tree_errors(tree.root_node)
        if has_err:
            logger.warning(
                "RustPlugin parse error in %s (errors=%d, missing=%d); abstaining.",
                file_path or "<memory>",
                err_cnt,
                miss_cnt,
            )
            self.last_parse_diagnostics = ParseDiagnostics(
                parser_backend="tree-sitter-rust",
                has_parse_error=True,
                abstained=True,
                error_node_count=err_cnt,
                missing_node_count=miss_cnt,
                reason="Syntax/parse error detected by tree-sitter-rust",
            )
            return [], []

        self.last_parse_diagnostics = ParseDiagnostics(parser_backend="tree-sitter-rust")
        base_ns = module_prefix or self._derive_module_prefix(file_path)

        symbols: list[ParsedSymbol] = []
        edges: list[ParsedEdge] = []
        seen_edge_keys: set[tuple[str, str, str]] = set()
        mod_stack: list[str] = []

        def current_ns() -> str:
            if mod_stack:
                return f"{base_ns}::{'::'.join(mod_stack)}"
            return base_ns

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

        def scan_body(body_node: Node | None, owner_qn: str, is_test: bool, ns: str) -> None:
            if body_node is None:
                return
            stack: list[Node] = [body_node]
            while stack:
                curr = stack.pop()
                if curr.type == "call_expression":
                    callee = _extract_callee_name(curr, code_bytes)
                    if callee:
                        add_edge(owner_qn, callee, "CALLS")
                        if is_test:
                            add_edge(f"{ns}::{callee}", owner_qn, "TESTED_BY")
                elif curr.type == "macro_invocation":
                    macro_node = curr.child_by_field_name("macro")
                    if macro_node is not None:
                        m_name = _node_text(macro_node, code_bytes).strip().rsplit("::", 1)[-1]
                        if m_name and m_name not in _RUST_BUILTIN_MACROS:
                            add_edge(owner_qn, m_name, "CALLS")
                stack.extend(reversed(curr.children))

        def handle_function(
            node: Node,
            ns: str,
            owner_type: str = "",
            imports: list[str] | None = None,
        ) -> None:
            name_node = node.child_by_field_name("name")
            if name_node is None:
                return
            fn_name = _node_text(name_node, code_bytes).strip()
            if not fn_name:
                return

            if owner_type:
                qn = f"{ns}::{owner_type}::{fn_name}"
                bare = f"{owner_type}.{fn_name}"
                sym_kind = "method"
            else:
                qn = f"{ns}::{fn_name}"
                bare = fn_name
                sym_kind = "function"

            attrs, doc = _extract_leading_attributes_and_docs(node, code_bytes)
            notes: list[str] = []
            is_test = False
            for attr_name, raw_attr in attrs:
                short_attr = attr_name.rsplit("::", 1)[-1]
                if short_attr == "test":
                    is_test = True
                elif short_attr in _ACTIX_ROUTE_ATTRS:
                    m = _STRING_LIT_RE.search(raw_attr)
                    if m:
                        route_path = "/" + m.group(1).lstrip("/")
                        http_method = _ACTIX_ROUTE_ATTRS[short_attr]
                        notes.append(f"@route {http_method} {route_path}")

            sig = _extract_signature_before_body(node, code_bytes)
            params_node = node.child_by_field_name("parameters")
            if params_node is not None:
                for ut in _extract_rust_type_names(params_node, code_bytes):
                    add_edge(qn, ut, "USES_TYPE")
            ret_node = node.child_by_field_name("return_type")
            if ret_node is not None:
                for ut in _extract_rust_type_names(ret_node, code_bytes):
                    add_edge(qn, ut, "USES_TYPE")

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
            for imp in imports or []:
                add_edge(qn, imp, "IMPORTS")

            body_node = node.child_by_field_name("body")
            scan_body(body_node, qn, is_test=is_test, ns=ns)

        def walk_block(container: Node) -> None:
            ns = current_ns()
            imports: list[str] = []
            for child in container.children:
                if child.type == "use_declaration":
                    for path in _extract_use_paths(child, code_bytes):
                        if path not in imports:
                            imports.append(path)

            for child in container.children:
                if child.type == "struct_item":
                    name_node = child.child_by_field_name("name")
                    if name_node is None:
                        continue
                    name = _node_text(name_node, code_bytes).strip()
                    qn = f"{ns}::{name}"
                    _, doc = _extract_leading_attributes_and_docs(child, code_bytes)
                    sig = _extract_signature_before_body(child, code_bytes)
                    dto_fields: list[str] = []
                    body_node = child.child_by_field_name("body")
                    if body_node is not None and body_node.type == "field_declaration_list":
                        for fdecl in body_node.children:
                            if fdecl.type == "field_declaration":
                                fn_node = fdecl.child_by_field_name("name")
                                if fn_node is not None:
                                    dto_fields.append(_node_text(fn_node, code_bytes).strip())
                                ftype = fdecl.child_by_field_name("type")
                                if ftype is not None:
                                    for ut in _extract_rust_type_names(ftype, code_bytes):
                                        add_edge(qn, ut, "USES_TYPE")
                    if dto_fields:
                        doc = f"{doc} @dto {name} fields={','.join(dto_fields)}".strip()
                    symbols.append(
                        ParsedSymbol(
                            name=name,
                            qualified_name=qn,
                            bare_name=name,
                            symbol_kind="class",
                            start_line=child.start_point[0] + 1,
                            end_line=child.end_point[0] + 1,
                            start_byte=child.start_byte,
                            end_byte=child.end_byte,
                            signature=sig,
                            signature_skeleton=sig,
                            docstring=doc,
                            contextual_summary=doc,
                            repo_name=repo_name,
                        )
                    )
                    for imp in imports:
                        add_edge(qn, imp, "IMPORTS")

                elif child.type == "enum_item":
                    name_node = child.child_by_field_name("name")
                    if name_node is None:
                        continue
                    name = _node_text(name_node, code_bytes).strip()
                    qn = f"{ns}::{name}"
                    _, doc = _extract_leading_attributes_and_docs(child, code_bytes)
                    sig = _extract_signature_before_body(child, code_bytes)
                    symbols.append(
                        ParsedSymbol(
                            name=name,
                            qualified_name=qn,
                            bare_name=name,
                            symbol_kind="class",
                            start_line=child.start_point[0] + 1,
                            end_line=child.end_point[0] + 1,
                            start_byte=child.start_byte,
                            end_byte=child.end_byte,
                            signature=sig,
                            signature_skeleton=sig,
                            docstring=doc,
                            contextual_summary=doc,
                            repo_name=repo_name,
                        )
                    )
                    for imp in imports:
                        add_edge(qn, imp, "IMPORTS")

                elif child.type == "trait_item":
                    name_node = child.child_by_field_name("name")
                    if name_node is None:
                        continue
                    name = _node_text(name_node, code_bytes).strip()
                    qn = f"{ns}::{name}"
                    _, doc = _extract_leading_attributes_and_docs(child, code_bytes)
                    sig = _extract_signature_before_body(child, code_bytes)
                    symbols.append(
                        ParsedSymbol(
                            name=name,
                            qualified_name=qn,
                            bare_name=name,
                            symbol_kind="interface",
                            start_line=child.start_point[0] + 1,
                            end_line=child.end_point[0] + 1,
                            start_byte=child.start_byte,
                            end_byte=child.end_byte,
                            signature=sig,
                            signature_skeleton=sig,
                            docstring=doc,
                            contextual_summary=doc,
                            repo_name=repo_name,
                        )
                    )
                    for imp in imports:
                        add_edge(qn, imp, "IMPORTS")

                elif child.type == "type_item":
                    name_node = child.child_by_field_name("name")
                    if name_node is None:
                        continue
                    name = _node_text(name_node, code_bytes).strip()
                    qn = f"{ns}::{name}"
                    _, doc = _extract_leading_attributes_and_docs(child, code_bytes)
                    sig = _collapse_ws(_node_text(child, code_bytes))
                    type_node = child.child_by_field_name("type")
                    if type_node is not None:
                        for ut in _extract_rust_type_names(type_node, code_bytes):
                            add_edge(qn, ut, "USES_TYPE")
                    symbols.append(
                        ParsedSymbol(
                            name=name,
                            qualified_name=qn,
                            bare_name=name,
                            symbol_kind="type_alias",
                            start_line=child.start_point[0] + 1,
                            end_line=child.end_point[0] + 1,
                            start_byte=child.start_byte,
                            end_byte=child.end_byte,
                            signature=sig,
                            signature_skeleton=sig,
                            docstring=doc,
                            contextual_summary=doc,
                            repo_name=repo_name,
                        )
                    )
                    for imp in imports:
                        add_edge(qn, imp, "IMPORTS")

                elif child.type == "impl_item":
                    type_node = child.child_by_field_name("type")
                    trait_node = child.child_by_field_name("trait")
                    target_type = _extract_primary_type_name(type_node, code_bytes)
                    trait_name = _extract_primary_type_name(trait_node, code_bytes)
                    if target_type and trait_name:
                        add_edge(f"{ns}::{target_type}", trait_name, "INHERITS")
                    body_node = child.child_by_field_name("body")
                    if body_node is not None:
                        for item in body_node.children:
                            if item.type == "function_item":
                                handle_function(item, ns, owner_type=target_type, imports=imports)

                elif child.type == "function_item":
                    handle_function(child, ns, owner_type="", imports=imports)

                elif child.type == "macro_definition":
                    name_node = child.child_by_field_name("name")
                    if name_node is None:
                        continue
                    name = _node_text(name_node, code_bytes).strip()
                    qn = f"{ns}::{name}"
                    _, doc = _extract_leading_attributes_and_docs(child, code_bytes)
                    sig = f"macro_rules! {name}"
                    symbols.append(
                        ParsedSymbol(
                            name=name,
                            qualified_name=qn,
                            bare_name=name,
                            symbol_kind="function",
                            start_line=child.start_point[0] + 1,
                            end_line=child.end_point[0] + 1,
                            start_byte=child.start_byte,
                            end_byte=child.end_byte,
                            signature=sig,
                            signature_skeleton=sig,
                            docstring=doc,
                            contextual_summary=doc,
                            repo_name=repo_name,
                        )
                    )

                elif child.type == "mod_item":
                    name_node = child.child_by_field_name("name")
                    body_node = child.child_by_field_name("body")
                    if name_node is not None and body_node is not None:
                        mod_name = _node_text(name_node, code_bytes).strip()
                        if mod_name:
                            mod_stack.append(mod_name)
                            walk_block(body_node)
                            mod_stack.pop()

        walk_block(tree.root_node)
        return symbols, edges
