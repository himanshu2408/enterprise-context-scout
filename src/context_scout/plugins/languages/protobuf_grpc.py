"""Protobuf & gRPC language plugin using tree-sitter-proto."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import yaml
from tree_sitter import Node, Parser
from tree_sitter_language_pack import get_parser

from context_scout.ast_engine.extractor import (
    ParseDiagnostics,
    _node_text,
    inspect_tree_errors,
)
from context_scout.ast_engine.models import ParsedEdge, ParsedSymbol
from context_scout.plugins.base import DependencyEdge

logger = logging.getLogger(__name__)

_PROTO_PARSER: Parser | None = None
_YAML_PARSER: Parser | None = None

_MANIFEST_BASENAMES: frozenset[str] = frozenset({"buf.yaml", "buf.lock"})

_SCALAR_PROTO_TYPES: frozenset[str] = frozenset(
    {
        "double",
        "float",
        "int32",
        "int64",
        "uint32",
        "uint64",
        "sint32",
        "sint64",
        "fixed32",
        "fixed64",
        "sfixed32",
        "sfixed64",
        "bool",
        "string",
        "bytes",
    }
)


def _get_proto_parser() -> Parser:
    global _PROTO_PARSER
    if _PROTO_PARSER is None:
        _PROTO_PARSER = get_parser("proto")
    return _PROTO_PARSER


def _get_yaml_parser() -> Parser:
    global _YAML_PARSER
    if _YAML_PARSER is None:
        _YAML_PARSER = get_parser("yaml")
    return _YAML_PARSER


def _unquote(text: str) -> str:
    stripped = text.strip()
    if len(stripped) >= 2 and (
        (stripped[0] == '"' and stripped[-1] == '"') or (stripped[0] == "'" and stripped[-1] == "'")
    ):
        return stripped[1:-1]
    return stripped


class ProtobufGrpcPlugin:
    """Language plugin for Protocol Buffers (.proto) and Buf manifests (buf.yaml, buf.lock)."""

    def __init__(self) -> None:
        self.last_parse_diagnostics: ParseDiagnostics | None = None

    @property
    def language_id(self) -> str:
        return "protobuf"

    @property
    def file_extensions(self) -> tuple[str, ...]:
        return (".proto",)

    @property
    def manifest_files(self) -> tuple[str, ...]:
        return ("buf.yaml", "buf.lock")

    def can_handle(self, file_path: str) -> bool:
        lower = file_path.lower()
        if lower.endswith(".proto"):
            return True
        return Path(file_path).name.lower() in _MANIFEST_BASENAMES

    def _derive_module_prefix(self, file_path: str) -> str:
        if not file_path:
            return "protobuf"
        stem = re.sub(r"\.proto$", "", file_path, flags=re.IGNORECASE)
        normalized = stem.replace("\\", "/").strip("/")
        dotted = ".".join(part for part in normalized.split("/") if part)
        return dotted or "protobuf"

    def extract_ast(
        self,
        code: str,
        file_path: str,
        repo_name: str = "",
        module_prefix: str = "",
    ) -> tuple[list[ParsedSymbol], list[ParsedEdge]]:
        if not code or not code.strip():
            self.last_parse_diagnostics = ParseDiagnostics(parser_backend="tree-sitter-proto")
            return [], []

        code_bytes = code.encode("utf-8")
        tree = _get_proto_parser().parse(code_bytes)
        has_err, err_cnt, miss_cnt = inspect_tree_errors(tree.root_node)
        if has_err:
            reason = (
                f"Partial/invalid Protobuf syntax in '{file_path or '<memory>'}': "
                f"error_nodes={err_cnt}, missing_nodes={miss_cnt}"
            )
            logger.warning("%s; abstaining from partial AST extraction", reason)
            self.last_parse_diagnostics = ParseDiagnostics(
                parser_backend="tree-sitter-proto",
                has_parse_error=True,
                abstained=True,
                low_confidence_fallback=False,
                error_node_count=err_cnt,
                missing_node_count=miss_cnt,
                reason=reason,
            )
            return [], []

        pkg_name = ""
        imports: list[str] = []
        for child in tree.root_node.children:
            if child.type == "package":
                fi = next((c for c in child.children if c.type == "full_ident"), None)
                if fi is not None:
                    pkg_name = _node_text(fi, code_bytes).strip()
            elif child.type == "import":
                str_node = next((c for c in child.children if c.type == "string"), None)
                if str_node is not None:
                    imp = _unquote(_node_text(str_node, code_bytes))
                    if imp:
                        imports.append(imp)

        prefix = module_prefix or pkg_name or self._derive_module_prefix(file_path)

        def _qualify(bare: str) -> str:
            if repo_name:
                return f"{repo_name}::{prefix}::{bare}"
            return f"{prefix}.{bare}" if prefix else bare

        def _repo_qual(bare: str) -> str | None:
            return f"{repo_name}::{prefix}::{bare}" if repo_name else None

        symbols: list[ParsedSymbol] = []
        edges: list[ParsedEdge] = []
        seen_edges: set[tuple[str, str, str]] = set()

        def _add_edge(src: str, tgt: str, kind: str) -> None:
            key = (src, tgt, kind)
            if key not in seen_edges and src and tgt:
                seen_edges.add(key)
                edges.append(
                    ParsedEdge(
                        source_qualified_name=src,
                        target_qualified_name=tgt,
                        edge_kind=kind,
                    )
                )

        if imports:
            mod_name = Path(file_path).stem if file_path else (pkg_name or "proto_file")
            mod_qname = _qualify(mod_name)
            total_lines = max(1, len(code.splitlines()))
            symbols.append(
                ParsedSymbol(
                    name=mod_name,
                    qualified_name=mod_qname,
                    symbol_kind="module",
                    start_line=1,
                    end_line=total_lines,
                    start_byte=0,
                    end_byte=len(code_bytes),
                    signature=f"package {pkg_name}" if pkg_name else f"proto {mod_name}",
                    signature_skeleton=f"package {pkg_name}" if pkg_name else mod_name,
                    contextual_summary=f"@proto_module {mod_name} package={pkg_name}",
                    repo_name=repo_name,
                    repo_qualified_id=_repo_qual(mod_name),
                    bare_name=mod_name,
                )
            )
            for imp in imports:
                _add_edge(mod_qname, imp, "IMPORTS")

        def _extract_enum(node: Node, scope_prefix: str) -> None:
            name_node = next((c for c in node.children if c.type == "enum_name"), None)
            if name_node is None:
                return
            ename = _node_text(name_node, code_bytes).strip()
            full_bare = f"{scope_prefix}.{ename}" if scope_prefix else ename
            qname = _qualify(full_bare)
            symbols.append(
                ParsedSymbol(
                    name=ename,
                    qualified_name=qname,
                    symbol_kind="type_alias",
                    start_line=node.start_point[0] + 1,
                    end_line=max(node.start_point[0] + 1, node.end_point[0] + 1),
                    start_byte=node.start_byte,
                    end_byte=node.end_byte,
                    signature=f"enum {full_bare}",
                    signature_skeleton=f"enum {full_bare}",
                    contextual_summary=f"@enum {full_bare}",
                    repo_name=repo_name,
                    repo_qualified_id=_repo_qual(full_bare),
                    bare_name=ename,
                )
            )

        def _collect_field_types(
            body_node: Node,
            owner_qname: str,
            scope_prefix: str,
        ) -> list[str]:
            fields: list[str] = []
            for child in body_node.children:
                if child.type == "field":
                    fname, ftypes = self._parse_field(child, code_bytes)
                    if fname:
                        fields.append(fname)
                    for ft in ftypes:
                        if ft not in _SCALAR_PROTO_TYPES:
                            _add_edge(owner_qname, _qualify(ft), "USES_TYPE")
                elif child.type == "map_field":
                    fname, ftypes = self._parse_map_field(child, code_bytes)
                    if fname:
                        fields.append(fname)
                    for ft in ftypes:
                        if ft not in _SCALAR_PROTO_TYPES:
                            _add_edge(owner_qname, _qualify(ft), "USES_TYPE")
                elif child.type == "oneof":
                    oneof_name_node = next(
                        (c for c in child.children if c.type == "identifier"), None
                    )
                    oneof_name = (
                        _node_text(oneof_name_node, code_bytes).strip()
                        if oneof_name_node is not None
                        else "oneof"
                    )
                    oneof_bare = f"{scope_prefix}.{oneof_name}"
                    oneof_qname = _qualify(oneof_bare)
                    oneof_fields: list[str] = []
                    for oc in child.children:
                        if oc.type == "oneof_field":
                            of_name, of_types = self._parse_field(oc, code_bytes)
                            if of_name:
                                oneof_fields.append(of_name)
                                fields.append(of_name)
                            for ft in of_types:
                                if ft not in _SCALAR_PROTO_TYPES:
                                    _add_edge(oneof_qname, _qualify(ft), "USES_TYPE")
                    symbols.append(
                        ParsedSymbol(
                            name=oneof_name,
                            qualified_name=oneof_qname,
                            symbol_kind="class",
                            start_line=child.start_point[0] + 1,
                            end_line=max(child.start_point[0] + 1, child.end_point[0] + 1),
                            start_byte=child.start_byte,
                            end_byte=child.end_byte,
                            signature=f"oneof {oneof_name} ({', '.join(oneof_fields)})",
                            signature_skeleton=f"oneof {oneof_name}",
                            contextual_summary=(
                                f"@dto {oneof_bare} fields=[{', '.join(oneof_fields)}]"
                            ),
                            repo_name=repo_name,
                            repo_qualified_id=_repo_qual(oneof_bare),
                            bare_name=oneof_name,
                        )
                    )
                    _add_edge(owner_qname, oneof_qname, "USES_TYPE")
                elif child.type == "message":
                    _extract_message(child, scope_prefix)
                elif child.type == "enum":
                    _extract_enum(child, scope_prefix)
            return fields

        def _extract_message(node: Node, scope_prefix: str) -> None:
            name_node = next((c for c in node.children if c.type == "message_name"), None)
            if name_node is None:
                return
            mname = _node_text(name_node, code_bytes).strip()
            full_bare = f"{scope_prefix}.{mname}" if scope_prefix else mname
            qname = _qualify(full_bare)
            body_node = next((c for c in node.children if c.type == "message_body"), None)
            fields = (
                _collect_field_types(body_node, qname, full_bare) if body_node is not None else []
            )
            field_list = ", ".join(fields)
            symbols.append(
                ParsedSymbol(
                    name=mname,
                    qualified_name=qname,
                    symbol_kind="class",
                    start_line=node.start_point[0] + 1,
                    end_line=max(node.start_point[0] + 1, node.end_point[0] + 1),
                    start_byte=node.start_byte,
                    end_byte=node.end_byte,
                    signature=f"message {full_bare} ({field_list})",
                    signature_skeleton=f"message {full_bare} ({field_list})",
                    contextual_summary=f"@dto {full_bare} fields=[{field_list}]",
                    repo_name=repo_name,
                    repo_qualified_id=_repo_qual(full_bare),
                    bare_name=mname,
                )
            )

        def _extract_service(node: Node) -> None:
            name_node = next((c for c in node.children if c.type == "service_name"), None)
            if name_node is None:
                return
            svc_name = _node_text(name_node, code_bytes).strip()
            svc_qname = _qualify(svc_name)
            symbols.append(
                ParsedSymbol(
                    name=svc_name,
                    qualified_name=svc_qname,
                    symbol_kind="service",
                    start_line=node.start_point[0] + 1,
                    end_line=max(node.start_point[0] + 1, node.end_point[0] + 1),
                    start_byte=node.start_byte,
                    end_byte=node.end_byte,
                    signature=f"service {svc_name}",
                    signature_skeleton=f"service {svc_name}",
                    contextual_summary=f"@grpc_service {pkg_name}.{svc_name}"
                    if pkg_name
                    else f"@grpc_service {svc_name}",
                    repo_name=repo_name,
                    repo_qualified_id=_repo_qual(svc_name),
                    bare_name=svc_name,
                )
            )

            for child in node.children:
                if child.type != "rpc":
                    continue
                rpc_name_node = next((c for c in child.children if c.type == "rpc_name"), None)
                if rpc_name_node is None:
                    continue
                rpc_name = _node_text(rpc_name_node, code_bytes).strip()
                req_type, resp_type, client_stream, server_stream = self._parse_rpc(
                    child, code_bytes
                )
                if client_stream and server_stream:
                    stream_mode = "bidi_streaming"
                elif client_stream:
                    stream_mode = "client_streaming"
                elif server_stream:
                    stream_mode = "server_streaming"
                else:
                    stream_mode = "unary"

                route = (
                    f"/{pkg_name}.{svc_name}/{rpc_name}" if pkg_name else f"/{svc_name}/{rpc_name}"
                )
                rpc_bare = f"{svc_name}.{rpc_name}"
                rpc_qname = _qualify(rpc_bare)
                symbols.append(
                    ParsedSymbol(
                        name=rpc_name,
                        qualified_name=rpc_qname,
                        symbol_kind="method",
                        start_line=child.start_point[0] + 1,
                        end_line=max(child.start_point[0] + 1, child.end_point[0] + 1),
                        start_byte=child.start_byte,
                        end_byte=child.end_byte,
                        signature=f"rpc {rpc_name}({req_type}) returns ({resp_type})",
                        signature_skeleton=f"rpc {rpc_name}({req_type}) -> {resp_type}",
                        contextual_summary=(f"@route POST {route} streaming={stream_mode}"),
                        repo_name=repo_name,
                        repo_qualified_id=_repo_qual(rpc_bare),
                        bare_name=rpc_name,
                    )
                )
                if req_type:
                    _add_edge(rpc_qname, _qualify(req_type), "USES_TYPE")
                if resp_type:
                    _add_edge(rpc_qname, _qualify(resp_type), "USES_TYPE")

        for child in tree.root_node.children:
            if child.type == "message":
                _extract_message(child, "")
            elif child.type == "enum":
                _extract_enum(child, "")
            elif child.type == "service":
                _extract_service(child)

        self.last_parse_diagnostics = ParseDiagnostics(parser_backend="tree-sitter-proto")
        return symbols, edges

    def _parse_field(self, node: Node, code_bytes: bytes) -> tuple[str, list[str]]:
        fname = ""
        ftypes: list[str] = []
        for child in node.children:
            if child.type == "identifier":
                fname = _node_text(child, code_bytes).strip()
            elif child.type in {"type", "message_or_enum_type"}:
                ttext = _node_text(child, code_bytes).strip()
                if ttext:
                    ftypes.append(ttext)
        return fname, ftypes

    def _parse_map_field(self, node: Node, code_bytes: bytes) -> tuple[str, list[str]]:
        fname = ""
        ftypes: list[str] = []
        for child in node.children:
            if child.type == "identifier":
                fname = _node_text(child, code_bytes).strip()
            elif child.type in {"type", "message_or_enum_type", "key_type"}:
                ttext = _node_text(child, code_bytes).strip()
                if ttext:
                    ftypes.append(ttext)
        return fname, ftypes

    def _parse_rpc(self, node: Node, code_bytes: bytes) -> tuple[str, str, bool, bool]:
        req_type = ""
        resp_type = ""
        client_stream = False
        server_stream = False
        seen_returns = False
        for child in node.children:
            if child.type == "returns":
                seen_returns = True
            elif child.type == "stream":
                if not seen_returns:
                    client_stream = True
                else:
                    server_stream = True
            elif child.type == "message_or_enum_type":
                ttext = _node_text(child, code_bytes).strip()
                if not seen_returns:
                    req_type = ttext
                else:
                    resp_type = ttext
        return req_type, resp_type, client_stream, server_stream

    def extract_manifest_dependencies(
        self,
        file_path: str,
        content: str,
    ) -> list[DependencyEdge]:
        """Extract module dependencies from buf.yaml or buf.lock."""
        if not content or not content.strip():
            self.last_parse_diagnostics = ParseDiagnostics(parser_backend="tree-sitter-yaml")
            return []

        tree = _get_yaml_parser().parse(content.encode("utf-8"))
        has_err, err_cnt, miss_cnt = inspect_tree_errors(tree.root_node)
        if has_err:
            reason = (
                f"Malformed Buf manifest '{file_path or '<memory>'}': "
                f"error_nodes={err_cnt}, missing_nodes={miss_cnt}"
            )
            logger.warning("%s; abstaining", reason)
            self.last_parse_diagnostics = ParseDiagnostics(
                parser_backend="tree-sitter-yaml",
                has_parse_error=True,
                abstained=True,
                low_confidence_fallback=False,
                error_node_count=err_cnt,
                missing_node_count=miss_cnt,
                reason=reason,
            )
            return []

        try:
            data: Any = yaml.safe_load(content)
        except yaml.YAMLError as exc:
            reason = f"YAML parse error in '{file_path or '<memory>'}': {exc}"
            logger.warning("%s; abstaining", reason)
            self.last_parse_diagnostics = ParseDiagnostics(
                parser_backend="tree-sitter-yaml",
                has_parse_error=True,
                abstained=True,
                low_confidence_fallback=False,
                error_node_count=1,
                reason=reason,
            )
            return []

        self.last_parse_diagnostics = ParseDiagnostics(parser_backend="tree-sitter-yaml")
        if not isinstance(data, dict):
            return []

        deps = data.get("deps") or []
        if not isinstance(deps, list):
            return []

        source_id = str(data.get("name") or Path(file_path).name or "buf")
        edges: list[DependencyEdge] = []
        for dep in deps:
            if isinstance(dep, str):
                pkg, _, ver = dep.partition(":")
                edges.append(
                    DependencyEdge(
                        source_identifier=source_id,
                        target_package=pkg,
                        version_constraint=ver or "*",
                        scope="compile",
                        manifest_path=file_path,
                    )
                )
            elif isinstance(dep, dict) and dep.get("name"):
                ver = str(dep.get("commit") or dep.get("digest") or "*")
                edges.append(
                    DependencyEdge(
                        source_identifier=source_id,
                        target_package=str(dep["name"]),
                        version_constraint=ver,
                        scope="compile",
                        manifest_path=file_path,
                    )
                )
        return edges
