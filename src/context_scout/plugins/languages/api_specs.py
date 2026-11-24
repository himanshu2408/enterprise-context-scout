"""API Specs language plugin for GraphQL, OpenAPI 3.x, and AsyncAPI 2.x/3.x."""

from __future__ import annotations

import json
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

_GRAPHQL_PARSER: Parser | None = None
_YAML_PARSER: Parser | None = None
_JSON_PARSER: Parser | None = None

_BUILTIN_GQL_SCALARS: frozenset[str] = frozenset({"Int", "Float", "String", "Boolean", "ID"})
_HTTP_METHODS: frozenset[str] = frozenset(
    {"get", "post", "put", "patch", "delete", "options", "head", "trace"}
)
_SPEC_NAME_RE = re.compile(r"(?:openapi|swagger|asyncapi)", re.IGNORECASE)


def _get_graphql_parser() -> Parser:
    global _GRAPHQL_PARSER
    if _GRAPHQL_PARSER is None:
        _GRAPHQL_PARSER = get_parser("graphql")
    return _GRAPHQL_PARSER


def _get_yaml_parser() -> Parser:
    global _YAML_PARSER
    if _YAML_PARSER is None:
        _YAML_PARSER = get_parser("yaml")
    return _YAML_PARSER


def _get_json_parser() -> Parser:
    global _JSON_PARSER
    if _JSON_PARSER is None:
        _JSON_PARSER = get_parser("json")
    return _JSON_PARSER


def _find_descendants(node: Node, node_types: set[str]) -> list[Node]:
    found: list[Node] = []
    stack: list[Node] = [node]
    while stack:
        curr = stack.pop()
        if curr.type in node_types:
            found.append(curr)
        stack.extend(reversed(curr.children))
    return found


def _collect_refs(obj: Any) -> list[str]:
    """Recursively collect JSON Schema $ref target names."""
    refs: list[str] = []
    stack: list[Any] = [obj]
    while stack:
        curr = stack.pop()
        if isinstance(curr, dict):
            ref_val = curr.get("$ref")
            if isinstance(ref_val, str) and ref_val:
                target = ref_val.rsplit("/", 1)[-1]
                if target:
                    refs.append(target)
            for v in curr.values():
                if isinstance(v, (dict, list)):
                    stack.append(v)
        elif isinstance(curr, list):
            for item in curr:
                if isinstance(item, (dict, list)):
                    stack.append(item)
    return refs


class ApiSpecsPlugin:
    """Language plugin for GraphQL (.graphql, .gql), OpenAPI 3.x, and AsyncAPI."""

    def __init__(self) -> None:
        self.last_parse_diagnostics: ParseDiagnostics | None = None

    @property
    def language_id(self) -> str:
        return "api_specs"

    @property
    def file_extensions(self) -> tuple[str, ...]:
        return (".graphql", ".gql")

    @property
    def manifest_files(self) -> tuple[str, ...]:
        return ()

    def can_handle(self, file_path: str) -> bool:
        lower = file_path.lower().replace("\\", "/")
        if lower.endswith((".graphql", ".gql")):
            return True
        if lower.endswith((".yaml", ".yml", ".json")):
            basename = Path(lower).name
            if _SPEC_NAME_RE.search(basename):
                return True
            if "/specs/" in lower or lower.startswith("specs/"):
                return True
        return False

    def _derive_module_prefix(self, file_path: str) -> str:
        if not file_path:
            return "api_specs"
        stem = re.sub(r"\.(?:graphql|gql|yaml|yml|json)$", "", file_path, flags=re.IGNORECASE)
        normalized = stem.replace("\\", "/").strip("/")
        dotted = ".".join(part for part in normalized.split("/") if part)
        return dotted or "api_specs"

    def extract_ast(
        self,
        code: str,
        file_path: str,
        repo_name: str = "",
        module_prefix: str = "",
    ) -> tuple[list[ParsedSymbol], list[ParsedEdge]]:
        if not code or not code.strip():
            self.last_parse_diagnostics = ParseDiagnostics(parser_backend="tree-sitter-graphql")
            return [], []

        lower = file_path.lower()
        prefix = module_prefix or self._derive_module_prefix(file_path)

        if lower.endswith((".graphql", ".gql")):
            return self._extract_graphql(code, file_path, repo_name, prefix)
        if lower.endswith(".json"):
            return self._extract_structured_spec(code, file_path, repo_name, prefix, is_json=True)
        if lower.endswith((".yaml", ".yml")):
            return self._extract_structured_spec(code, file_path, repo_name, prefix, is_json=False)

        # Fallback if invoked on an untyped path: try GraphQL if it looks like SDL
        return self._extract_graphql(code, file_path, repo_name, prefix)

    def extract_manifest_dependencies(
        self,
        file_path: str,
        content: str,
    ) -> list[DependencyEdge]:
        return []

    def _extract_graphql(
        self,
        code: str,
        file_path: str,
        repo_name: str,
        prefix: str,
    ) -> tuple[list[ParsedSymbol], list[ParsedEdge]]:
        code_bytes = code.encode("utf-8")
        tree = _get_graphql_parser().parse(code_bytes)
        has_err, err_cnt, miss_cnt = inspect_tree_errors(tree.root_node)
        if has_err:
            reason = (
                f"Partial/invalid GraphQL syntax in '{file_path or '<memory>'}': "
                f"error_nodes={err_cnt}, missing_nodes={miss_cnt}"
            )
            logger.warning("%s; abstaining from partial AST extraction", reason)
            self.last_parse_diagnostics = ParseDiagnostics(
                parser_backend="tree-sitter-graphql",
                has_parse_error=True,
                abstained=True,
                low_confidence_fallback=False,
                error_node_count=err_cnt,
                missing_node_count=miss_cnt,
                reason=reason,
            )
            return [], []

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

        for node in _find_descendants(
            tree.root_node,
            {
                "object_type_definition",
                "interface_type_definition",
                "input_object_type_definition",
                "enum_type_definition",
                "union_type_definition",
            },
        ):
            name_node = next((c for c in node.children if c.type == "name"), None)
            if name_node is None:
                continue
            tname = _node_text(name_node, code_bytes).strip()
            if not tname:
                continue
            qname = _qualify(tname)

            if node.type == "enum_type_definition":
                symbols.append(
                    ParsedSymbol(
                        name=tname,
                        qualified_name=qname,
                        symbol_kind="type_alias",
                        start_line=node.start_point[0] + 1,
                        end_line=max(node.start_point[0] + 1, node.end_point[0] + 1),
                        start_byte=node.start_byte,
                        end_byte=node.end_byte,
                        signature=f"enum {tname}",
                        signature_skeleton=f"enum {tname}",
                        contextual_summary=f"@enum {tname}",
                        repo_name=repo_name,
                        repo_qualified_id=_repo_qual(tname),
                        bare_name=tname,
                    )
                )
                continue

            if node.type == "union_type_definition":
                members: list[str] = []
                for nt in _find_descendants(node, {"named_type"}):
                    mname = _node_text(nt, code_bytes).strip()
                    if mname:
                        members.append(mname)
                        if mname not in _BUILTIN_GQL_SCALARS:
                            _add_edge(qname, _qualify(mname), "USES_TYPE")
                symbols.append(
                    ParsedSymbol(
                        name=tname,
                        qualified_name=qname,
                        symbol_kind="type_alias",
                        start_line=node.start_point[0] + 1,
                        end_line=max(node.start_point[0] + 1, node.end_point[0] + 1),
                        start_byte=node.start_byte,
                        end_byte=node.end_byte,
                        signature=f"union {tname} = {' | '.join(members)}",
                        signature_skeleton=f"union {tname}",
                        contextual_summary=f"@dto {tname} fields=[{', '.join(members)}]",
                        repo_name=repo_name,
                        repo_qualified_id=_repo_qual(tname),
                        bare_name=tname,
                    )
                )
                continue

            # Implements interfaces
            for impl in _find_descendants(node, {"implements_interfaces"}):
                for nt in _find_descendants(impl, {"named_type"}):
                    iface = _node_text(nt, code_bytes).strip()
                    if iface:
                        _add_edge(qname, _qualify(iface), "INHERITS")

            is_root_op = tname in {"Query", "Mutation", "Subscription"}
            field_nodes = _find_descendants(node, {"field_definition", "input_value_definition"})
            field_names: list[str] = []

            for fnode in field_nodes:
                # Skip nested input_value_definition inside field_definition arguments
                if (
                    fnode.type == "input_value_definition"
                    and node.type != "input_object_type_definition"
                ):
                    fname_node = next((c for c in fnode.children if c.type == "name"), None)
                    for nt in _find_descendants(fnode, {"named_type"}):
                        arg_type = _node_text(nt, code_bytes).strip()
                        if arg_type and arg_type not in _BUILTIN_GQL_SCALARS:
                            _add_edge(qname, _qualify(arg_type), "USES_TYPE")
                    continue

                fname_node = next((c for c in fnode.children if c.type == "name"), None)
                if fname_node is None:
                    continue
                fname = _node_text(fname_node, code_bytes).strip()
                if not fname:
                    continue
                field_names.append(fname)

                ref_types: list[str] = []
                for nt in _find_descendants(fnode, {"named_type"}):
                    rt = _node_text(nt, code_bytes).strip()
                    if rt and rt not in _BUILTIN_GQL_SCALARS:
                        ref_types.append(rt)

                if is_root_op and fnode.type == "field_definition":
                    op_kind = tname.lower()
                    op_bare = f"{tname}.{fname}"
                    op_qname = _qualify(op_bare)
                    symbols.append(
                        ParsedSymbol(
                            name=fname,
                            qualified_name=op_qname,
                            symbol_kind="method",
                            start_line=fnode.start_point[0] + 1,
                            end_line=max(fnode.start_point[0] + 1, fnode.end_point[0] + 1),
                            start_byte=fnode.start_byte,
                            end_byte=fnode.end_byte,
                            signature=_node_text(fnode, code_bytes).strip(),
                            signature_skeleton=f"{op_kind} {fname}",
                            contextual_summary=f"@graphql {op_kind} {fname}",
                            repo_name=repo_name,
                            repo_qualified_id=_repo_qual(op_bare),
                            bare_name=fname,
                        )
                    )
                    for rt in ref_types:
                        _add_edge(op_qname, _qualify(rt), "USES_TYPE")
                else:
                    for rt in ref_types:
                        _add_edge(qname, _qualify(rt), "USES_TYPE")

            sym_kind = (
                "interface"
                if node.type == "interface_type_definition"
                else ("api" if is_root_op else "class")
            )
            field_str = ", ".join(field_names)
            symbols.append(
                ParsedSymbol(
                    name=tname,
                    qualified_name=qname,
                    symbol_kind=sym_kind,
                    start_line=node.start_point[0] + 1,
                    end_line=max(node.start_point[0] + 1, node.end_point[0] + 1),
                    start_byte=node.start_byte,
                    end_byte=node.end_byte,
                    signature=f"{tname} ({field_str})",
                    signature_skeleton=f"{tname} ({field_str})",
                    contextual_summary=f"@dto {tname} fields=[{field_str}]",
                    repo_name=repo_name,
                    repo_qualified_id=_repo_qual(tname),
                    bare_name=tname,
                )
            )

        self.last_parse_diagnostics = ParseDiagnostics(parser_backend="tree-sitter-graphql")
        return symbols, edges

    def _extract_structured_spec(
        self,
        code: str,
        file_path: str,
        repo_name: str,
        prefix: str,
        *,
        is_json: bool,
    ) -> tuple[list[ParsedSymbol], list[ParsedEdge]]:
        backend = "tree-sitter-json" if is_json else "tree-sitter-yaml"
        parser = _get_json_parser() if is_json else _get_yaml_parser()
        code_bytes = code.encode("utf-8")
        tree = parser.parse(code_bytes)
        has_err, err_cnt, miss_cnt = inspect_tree_errors(tree.root_node)

        if has_err:
            reason = (
                f"Partial/invalid {'JSON' if is_json else 'YAML'} syntax in "
                f"'{file_path or '<memory>'}': error_nodes={err_cnt}, missing_nodes={miss_cnt}"
            )
            logger.warning("%s; abstaining from partial AST extraction", reason)
            self.last_parse_diagnostics = ParseDiagnostics(
                parser_backend=backend,
                has_parse_error=True,
                abstained=True,
                low_confidence_fallback=False,
                error_node_count=err_cnt,
                missing_node_count=miss_cnt,
                reason=reason,
            )
            return [], []

        try:
            doc: Any = json.loads(code) if is_json else yaml.safe_load(code)
        except (json.JSONDecodeError, yaml.YAMLError) as exc:
            reason = f"Spec parse error in '{file_path or '<memory>'}': {exc}"
            logger.warning("%s; abstaining", reason)
            self.last_parse_diagnostics = ParseDiagnostics(
                parser_backend=backend,
                has_parse_error=True,
                abstained=True,
                low_confidence_fallback=False,
                error_node_count=1,
                reason=reason,
            )
            return [], []

        self.last_parse_diagnostics = ParseDiagnostics(parser_backend=backend)
        if not isinstance(doc, dict):
            return [], []

        if "openapi" in doc or "swagger" in doc:
            return self._extract_openapi(doc, code, repo_name, prefix)
        if "asyncapi" in doc:
            return self._extract_asyncapi(doc, code, repo_name, prefix)

        return [], []

    def _extract_openapi(
        self,
        doc: dict[str, Any],
        code: str,
        repo_name: str,
        prefix: str,
    ) -> tuple[list[ParsedSymbol], list[ParsedEdge]]:
        def _qualify(bare: str) -> str:
            if repo_name:
                return f"{repo_name}::{prefix}::{bare}"
            return f"{prefix}.{bare}" if prefix else bare

        def _repo_qual(bare: str) -> str | None:
            return f"{repo_name}::{prefix}::{bare}" if repo_name else None

        symbols: list[ParsedSymbol] = []
        edges: list[ParsedEdge] = []
        seen_edges: set[tuple[str, str, str]] = set()
        total_lines = max(1, len(code.splitlines()))

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

        info_raw = doc.get("info")
        info: dict[str, Any] = info_raw if isinstance(info_raw, dict) else {}
        api_title = str(info.get("title") or "OpenAPI")
        api_ver = str(info.get("version") or "")
        symbols.append(
            ParsedSymbol(
                name=api_title,
                qualified_name=_qualify(api_title),
                symbol_kind="api",
                start_line=1,
                end_line=total_lines,
                start_byte=0,
                end_byte=len(code.encode("utf-8")),
                signature=f"OpenAPI {api_title} {api_ver}".strip(),
                signature_skeleton=f"OpenAPI {api_title}",
                contextual_summary=f"@openapi {api_title} version={api_ver}",
                repo_name=repo_name,
                repo_qualified_id=_repo_qual(api_title),
                bare_name=api_title,
            )
        )

        # Paths & operations
        paths = doc.get("paths")
        if isinstance(paths, dict):
            for path_str, path_item in paths.items():
                if not isinstance(path_item, dict):
                    continue
                for method, op in path_item.items():
                    if method.lower() not in _HTTP_METHODS or not isinstance(op, dict):
                        continue
                    op_id = str(op.get("operationId") or f"{method.upper()}_{path_str}")
                    op_qname = _qualify(op_id)
                    symbols.append(
                        ParsedSymbol(
                            name=op_id,
                            qualified_name=op_qname,
                            symbol_kind="method",
                            start_line=1,
                            end_line=total_lines,
                            start_byte=0,
                            end_byte=0,
                            signature=f"{method.upper()} {path_str} ({op_id})",
                            signature_skeleton=f"{method.upper()} {path_str}",
                            contextual_summary=f"@route {method.upper()} {path_str}",
                            repo_name=repo_name,
                            repo_qualified_id=_repo_qual(op_id),
                            bare_name=op_id,
                        )
                    )
                    for ref_target in _collect_refs(op):
                        _add_edge(op_qname, _qualify(ref_target), "USES_TYPE")

        # Component schemas
        components = doc.get("components")
        schemas = (
            components.get("schemas") if isinstance(components, dict) else doc.get("definitions")
        )
        if isinstance(schemas, dict):
            for sname, sdef in schemas.items():
                if not isinstance(sdef, dict):
                    continue
                props = sdef.get("properties")
                fields = list(props.keys()) if isinstance(props, dict) else []
                field_str = ", ".join(str(f) for f in fields)
                sqname = _qualify(str(sname))
                symbols.append(
                    ParsedSymbol(
                        name=str(sname),
                        qualified_name=sqname,
                        symbol_kind="class",
                        start_line=1,
                        end_line=total_lines,
                        start_byte=0,
                        end_byte=0,
                        signature=f"schema {sname} ({field_str})",
                        signature_skeleton=f"schema {sname}",
                        contextual_summary=f"@dto {sname} fields=[{field_str}]",
                        repo_name=repo_name,
                        repo_qualified_id=_repo_qual(str(sname)),
                        bare_name=str(sname),
                    )
                )
                for ref_target in _collect_refs(sdef):
                    if ref_target != str(sname):
                        _add_edge(sqname, _qualify(ref_target), "USES_TYPE")

        return symbols, edges

    def _extract_asyncapi(
        self,
        doc: dict[str, Any],
        code: str,
        repo_name: str,
        prefix: str,
    ) -> tuple[list[ParsedSymbol], list[ParsedEdge]]:
        def _qualify(bare: str) -> str:
            if repo_name:
                return f"{repo_name}::{prefix}::{bare}"
            return f"{prefix}.{bare}" if prefix else bare

        def _repo_qual(bare: str) -> str | None:
            return f"{repo_name}::{prefix}::{bare}" if repo_name else None

        symbols: list[ParsedSymbol] = []
        edges: list[ParsedEdge] = []
        seen_edges: set[tuple[str, str, str]] = set()
        total_lines = max(1, len(code.splitlines()))

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

        info_raw = doc.get("info")
        info: dict[str, Any] = info_raw if isinstance(info_raw, dict) else {}
        api_title = str(info.get("title") or "AsyncAPI")
        api_ver = str(info.get("version") or "")
        symbols.append(
            ParsedSymbol(
                name=api_title,
                qualified_name=_qualify(api_title),
                symbol_kind="api",
                start_line=1,
                end_line=total_lines,
                start_byte=0,
                end_byte=len(code.encode("utf-8")),
                signature=f"AsyncAPI {api_title} {api_ver}".strip(),
                signature_skeleton=f"AsyncAPI {api_title}",
                contextual_summary=f"@asyncapi {api_title} version={api_ver}",
                repo_name=repo_name,
                repo_qualified_id=_repo_qual(api_title),
                bare_name=api_title,
            )
        )

        # Channels (AsyncAPI 2.x publish/subscribe & 3.x)
        channels = doc.get("channels")
        if isinstance(channels, dict):
            for ch_name, ch_def in channels.items():
                if not isinstance(ch_def, dict):
                    continue
                ch_qname = _qualify(str(ch_name))
                symbols.append(
                    ParsedSymbol(
                        name=str(ch_name),
                        qualified_name=ch_qname,
                        symbol_kind="method",
                        start_line=1,
                        end_line=total_lines,
                        start_byte=0,
                        end_byte=0,
                        signature=f"channel {ch_name}",
                        signature_skeleton=f"channel {ch_name}",
                        contextual_summary=f"@channel {ch_name}",
                        repo_name=repo_name,
                        repo_qualified_id=_repo_qual(str(ch_name)),
                        bare_name=str(ch_name),
                    )
                )
                for ref_target in _collect_refs(ch_def):
                    _add_edge(ch_qname, _qualify(ref_target), "USES_TYPE")

        # AsyncAPI 3.x operations
        operations = doc.get("operations")
        if isinstance(operations, dict):
            for op_name, op_def in operations.items():
                if not isinstance(op_def, dict):
                    continue
                op_qname = _qualify(str(op_name))
                action = str(op_def.get("action") or "event")
                symbols.append(
                    ParsedSymbol(
                        name=str(op_name),
                        qualified_name=op_qname,
                        symbol_kind="method",
                        start_line=1,
                        end_line=total_lines,
                        start_byte=0,
                        end_byte=0,
                        signature=f"{action} {op_name}",
                        signature_skeleton=f"{action} {op_name}",
                        contextual_summary=f"@async_op {action} {op_name}",
                        repo_name=repo_name,
                        repo_qualified_id=_repo_qual(str(op_name)),
                        bare_name=str(op_name),
                    )
                )
                for ref_target in _collect_refs(op_def):
                    _add_edge(op_qname, _qualify(ref_target), "USES_TYPE")

        # Component messages & schemas
        components = doc.get("components")
        if isinstance(components, dict):
            for section in ("messages", "schemas"):
                items = components.get(section)
                if not isinstance(items, dict):
                    continue
                for mname, mdef in items.items():
                    if not isinstance(mdef, dict):
                        continue
                    props = mdef.get("properties")
                    fields = list(props.keys()) if isinstance(props, dict) else []
                    field_str = ", ".join(str(f) for f in fields)
                    mqname = _qualify(str(mname))
                    symbols.append(
                        ParsedSymbol(
                            name=str(mname),
                            qualified_name=mqname,
                            symbol_kind="class",
                            start_line=1,
                            end_line=total_lines,
                            start_byte=0,
                            end_byte=0,
                            signature=f"{section[:-1]} {mname} ({field_str})",
                            signature_skeleton=f"{section[:-1]} {mname}",
                            contextual_summary=f"@dto {mname} fields=[{field_str}]",
                            repo_name=repo_name,
                            repo_qualified_id=_repo_qual(str(mname)),
                            bare_name=str(mname),
                        )
                    )
                    for ref_target in _collect_refs(mdef):
                        if ref_target != str(mname):
                            _add_edge(mqname, _qualify(ref_target), "USES_TYPE")

        return symbols, edges
