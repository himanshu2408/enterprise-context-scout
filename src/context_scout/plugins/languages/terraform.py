"""Terraform / HCL language plugin using tree-sitter-hcl AST traversal."""

from __future__ import annotations

import json
import logging
import re

from tree_sitter import Node

from context_scout.ast_engine.extractor import (
    ParseDiagnostics,
    _get_hcl_parser,
    _node_text,
    _unquote_hcl_string,
    inspect_tree_errors,
    parse_hcl_ast,
)
from context_scout.ast_engine.models import ParsedEdge, ParsedSymbol
from context_scout.plugins.base import DependencyEdge

logger = logging.getLogger(__name__)


def _block_type_and_labels(block: Node, code_bytes: bytes) -> tuple[str, list[str]]:
    """Return block type identifier and unquoted labels for an HCL block node."""
    btype = ""
    labels: list[str] = []
    for child in block.children:
        if child.type == "identifier" and not btype:
            btype = _node_text(child, code_bytes)
        elif child.type in {"string_lit", "identifier"} and btype:
            labels.append(_unquote_hcl_string(child, code_bytes))
    return btype, labels


def _find_body_node(node: Node) -> Node | None:
    for child in node.children:
        if child.type == "body":
            return child
    return None


def _extract_attribute_strings(body: Node, code_bytes: bytes) -> dict[str, str]:
    """Extract top-level string attributes directly inside an HCL body node."""
    attrs: dict[str, str] = {}
    for child in body.children:
        if child.type != "attribute":
            continue
        key_node = next((c for c in child.children if c.type == "identifier"), None)
        expr_node = next((c for c in child.children if c.type == "expression"), None)
        if key_node is None or expr_node is None:
            continue
        key = _node_text(key_node, code_bytes)
        val = _unquote_hcl_string(expr_node, code_bytes)
        if key:
            attrs[key] = val
    return attrs


def _extract_object_strings(obj_node: Node, code_bytes: bytes) -> dict[str, str]:
    """Extract string key/value pairs inside an HCL object literal node."""
    result: dict[str, str] = {}
    for child in obj_node.children:
        if child.type != "object_elem":
            continue
        named = [c for c in child.children if c.is_named]
        if len(named) < 2:
            continue
        key = _unquote_hcl_string(named[0], code_bytes)
        val = _unquote_hcl_string(named[1], code_bytes)
        if key:
            result[key] = val
    return result


class TerraformPlugin:
    """Language plugin for Terraform (.tf, .tfvars, .tf.json) and .terraform.lock.hcl."""

    def __init__(self) -> None:
        self.last_parse_diagnostics: ParseDiagnostics | None = None

    @property
    def language_id(self) -> str:
        return "terraform"

    @property
    def file_extensions(self) -> tuple[str, ...]:
        return (".tf", ".tfvars", ".tf.json")

    @property
    def manifest_files(self) -> tuple[str, ...]:
        return (".terraform.lock.hcl",)

    def can_handle(self, file_path: str) -> bool:
        lower = file_path.lower()
        return any(lower.endswith(ext) for ext in self.file_extensions)

    def _derive_module_prefix(self, file_path: str) -> str:
        if not file_path:
            return "terraform"
        stem = re.sub(r"\.(?:tf\.json|tf|tfvars)$", "", file_path, flags=re.IGNORECASE)
        normalized = stem.replace("\\", "/").strip("/")
        dotted = ".".join(part for part in normalized.split("/") if part)
        return dotted or "terraform"

    def extract_manifest_dependencies(
        self,
        file_path: str,
        content: str,
    ) -> list[DependencyEdge]:
        """Extract provider deps from .terraform.lock.hcl or terraform.required_providers."""
        if not content or not content.strip():
            self.last_parse_diagnostics = ParseDiagnostics(parser_backend="tree-sitter-hcl")
            return []

        code_bytes = content.encode("utf-8")
        tree = _get_hcl_parser().parse(code_bytes)
        has_err, err_cnt, miss_cnt = inspect_tree_errors(tree.root_node)
        if has_err:
            logger.warning(
                "TerraformPlugin manifest parse error in %s (errors=%d, missing=%d); abstaining.",
                file_path or "<memory>",
                err_cnt,
                miss_cnt,
            )
            self.last_parse_diagnostics = ParseDiagnostics(
                parser_backend="tree-sitter-hcl",
                has_parse_error=True,
                abstained=True,
                error_node_count=err_cnt,
                missing_node_count=miss_cnt,
                reason="Syntax/parse error detected in HCL manifest",
            )
            return []

        self.last_parse_diagnostics = ParseDiagnostics(parser_backend="tree-sitter-hcl")
        deps: list[DependencyEdge] = []

        def walk_blocks(node: Node) -> None:
            for child in node.children:
                if child.type == "block":
                    btype, labels = _block_type_and_labels(child, code_bytes)
                    body = _find_body_node(child)
                    if btype == "provider" and labels and body is not None:
                        attrs = _extract_attribute_strings(body, code_bytes)
                        deps.append(
                            DependencyEdge(
                                source_identifier=file_path,
                                target_package=labels[0],
                                version_constraint=attrs.get("version", "*"),
                                scope="provider",
                                manifest_path=file_path,
                            )
                        )
                    elif btype == "required_providers" and body is not None:
                        for attr in body.children:
                            if attr.type != "attribute":
                                continue
                            key_node = next(
                                (c for c in attr.children if c.type == "identifier"), None
                            )
                            expr_node = next(
                                (c for c in attr.children if c.type == "expression"), None
                            )
                            if key_node is None or expr_node is None:
                                continue
                            prov_name = _node_text(key_node, code_bytes)
                            obj_node = next(
                                (c for c in expr_node.children if c.type == "object"), None
                            )
                            if obj_node is None:
                                for sub in expr_node.children:
                                    if sub.type == "collection_value":
                                        obj_node = next(
                                            (c for c in sub.children if c.type == "object"),
                                            None,
                                        )
                                        if obj_node is not None:
                                            break
                            if obj_node is not None:
                                obj_attrs = _extract_object_strings(obj_node, code_bytes)
                                deps.append(
                                    DependencyEdge(
                                        source_identifier=file_path,
                                        target_package=obj_attrs.get("source", prov_name),
                                        version_constraint=obj_attrs.get("version", "*"),
                                        scope="provider",
                                        manifest_path=file_path,
                                    )
                                )
                            else:
                                ver = _unquote_hcl_string(expr_node, code_bytes)
                                deps.append(
                                    DependencyEdge(
                                        source_identifier=file_path,
                                        target_package=prov_name,
                                        version_constraint=ver or "*",
                                        scope="provider",
                                        manifest_path=file_path,
                                    )
                                )
                    elif body is not None:
                        walk_blocks(body)
                else:
                    walk_blocks(child)

        walk_blocks(tree.root_node)
        return deps

    def extract_ast(
        self,
        code: str,
        file_path: str = "",
        repo_name: str = "",
        module_prefix: str = "",
    ) -> tuple[list[ParsedSymbol], list[ParsedEdge]]:
        """Extract Terraform blocks and references via tree-sitter-hcl (or json for .tf.json)."""
        if not code or not code.strip():
            self.last_parse_diagnostics = ParseDiagnostics(parser_backend="tree-sitter-hcl")
            return [], []

        prefix = module_prefix or self._derive_module_prefix(file_path)

        if file_path.lower().endswith(".tf.json"):
            return self._extract_tf_json(code, file_path, repo_name, prefix)

        symbols, edges, diag = parse_hcl_ast(
            code=code,
            file_path=file_path,
            repo_name=repo_name,
            module_prefix=prefix,
            module_symbol_kind="class",
        )
        self.last_parse_diagnostics = diag
        if not diag.abstained:
            symbols, edges = self._enrich_hcl_contracts(
                code=code,
                module_prefix=prefix,
                repo_name=repo_name,
                symbols=symbols,
                edges=edges,
            )
        return symbols, edges

    def _enrich_hcl_contracts(
        self,
        *,
        code: str,
        module_prefix: str,
        repo_name: str,
        symbols: list[ParsedSymbol],
        edges: list[ParsedEdge],
    ) -> tuple[list[ParsedSymbol], list[ParsedEdge]]:
        """Extract TF_VAR_* and resource env/service metadata using tree-sitter-hcl."""
        code_bytes = code.encode("utf-8")
        tree = _get_hcl_parser().parse(code_bytes)
        sym_by_qn: dict[str, int] = {sym.qualified_name: idx for idx, sym in enumerate(symbols)}
        updated_symbols = list(symbols)
        updated_edges = list(edges)

        def collect_env_names(node: Node) -> list[str]:
            envs: list[str] = []
            for child in node.children:
                if child.type == "block":
                    btype, _ = _block_type_and_labels(child, code_bytes)
                    bbody = _find_body_node(child)
                    if bbody is not None:
                        if btype in {"env", "environment"}:
                            attrs = _extract_attribute_strings(bbody, code_bytes)
                            if attrs.get("name"):
                                envs.append(attrs["name"])
                        envs.extend(collect_env_names(bbody))
                elif child.type == "attribute":
                    key_nd = next((c for c in child.children if c.type == "identifier"), None)
                    expr_nd = next((c for c in child.children if c.type == "expression"), None)
                    if key_nd is not None and expr_nd is not None:
                        key_name = _node_text(key_nd, code_bytes)
                        if key_name in {"env_vars", "environment_variables", "env", "variables"}:
                            obj_nd = next(
                                (c for c in expr_nd.children if c.type == "object"),
                                None,
                            )
                            if obj_nd is None:
                                for sub in expr_nd.children:
                                    if sub.type == "collection_value":
                                        obj_nd = next(
                                            (c for c in sub.children if c.type == "object"),
                                            None,
                                        )
                                        if obj_nd is not None:
                                            break
                            if obj_nd is not None:
                                envs.extend(_extract_object_strings(obj_nd, code_bytes).keys())
                elif child.type == "body":
                    envs.extend(collect_env_names(child))
            return envs

        def walk_hcl(node: Node) -> None:
            for child in node.children:
                if child.type == "block":
                    btype, labels = _block_type_and_labels(child, code_bytes)
                    bbody = _find_body_node(child)
                    if btype == "variable" and labels:
                        vname = labels[0]
                        vqn = f"{module_prefix}.var.{vname}"
                        tag = f"@env TF_VAR_{vname}"
                        if vqn in sym_by_qn:
                            idx = sym_by_qn[vqn]
                            prev = updated_symbols[idx]
                            doc = f"{prev.docstring} {tag}".strip()
                            updated_symbols[idx] = prev.model_copy(
                                update={"docstring": doc, "contextual_summary": doc}
                            )
                        else:
                            updated_symbols.append(
                                ParsedSymbol(
                                    name=f"var.{vname}",
                                    qualified_name=vqn,
                                    bare_name=f"var.{vname}",
                                    symbol_kind="type_alias",
                                    start_line=child.start_point[0] + 1,
                                    end_line=child.end_point[0] + 1,
                                    start_byte=child.start_byte,
                                    end_byte=child.end_byte,
                                    signature=f'variable "{vname}"',
                                    signature_skeleton=f'variable "{vname}"',
                                    docstring=tag,
                                    repo_name=repo_name,
                                )
                            )
                    elif btype in {"resource", "data", "module"} and labels and bbody is not None:
                        target_qn = (
                            f"{module_prefix}.module.{labels[0]}"
                            if btype == "module"
                            else (
                                f"{module_prefix}.{labels[0]}.{labels[1]}"
                                if len(labels) >= 2
                                else ""
                            )
                        )
                        attrs = _extract_attribute_strings(bbody, code_bytes)
                        envs = collect_env_names(bbody)
                        tags: list[str] = []
                        svc_name = attrs.get("name") or attrs.get("service") or ""
                        if svc_name:
                            tags.append(f"@service {svc_name}")
                        for env_key in dict.fromkeys(envs):
                            tags.append(f"@env {env_key}")
                        if target_qn in sym_by_qn and tags:
                            idx = sym_by_qn[target_qn]
                            prev = updated_symbols[idx]
                            merged_doc = f"{prev.docstring} {' '.join(tags)}".strip()
                            updated_symbols[idx] = prev.model_copy(
                                update={
                                    "docstring": merged_doc,
                                    "contextual_summary": merged_doc,
                                }
                            )
                    elif bbody is not None:
                        walk_hcl(bbody)
                elif child.type == "body":
                    walk_hcl(child)

        walk_hcl(tree.root_node)
        return updated_symbols, updated_edges

    def _extract_tf_json(
        self,
        code: str,
        file_path: str,
        repo_name: str,
        module_prefix: str,
    ) -> tuple[list[ParsedSymbol], list[ParsedEdge]]:
        try:
            data = json.loads(code)
        except json.JSONDecodeError as exc:
            logger.warning("Invalid .tf.json in %s (%s); abstaining.", file_path or "<memory>", exc)
            self.last_parse_diagnostics = ParseDiagnostics(
                parser_backend="json",
                has_parse_error=True,
                abstained=True,
                reason=f"JSONDecodeError: {exc}",
            )
            return [], []

        if not isinstance(data, dict):
            self.last_parse_diagnostics = ParseDiagnostics(
                parser_backend="json",
                has_parse_error=True,
                abstained=True,
                reason="Top-level .tf.json value must be an object",
            )
            return [], []

        self.last_parse_diagnostics = ParseDiagnostics(parser_backend="json")
        symbols: list[ParsedSymbol] = []
        edges: list[ParsedEdge] = []
        total_lines = max(1, code.count("\n") + 1)
        code_len = len(code.encode("utf-8"))

        def add_sym(
            name: str,
            qn: str,
            kind: str,
            sig: str,
        ) -> None:
            symbols.append(
                ParsedSymbol(
                    name=name,
                    qualified_name=qn,
                    bare_name=name,
                    symbol_kind=kind,
                    start_line=1,
                    end_line=total_lines,
                    start_byte=0,
                    end_byte=code_len,
                    signature=sig,
                    signature_skeleton=sig,
                    repo_name=repo_name,
                )
            )

        for rtype, instances in data.get("resource", {}).items():
            if not isinstance(instances, dict):
                continue
            for rname in instances:
                qn = f"{module_prefix}.{rtype}.{rname}"
                add_sym(f"{rtype}.{rname}", qn, "class", f'resource "{rtype}" "{rname}"')

        for dtype, instances in data.get("data", {}).items():
            if not isinstance(instances, dict):
                continue
            for dname in instances:
                qn = f"{module_prefix}.data.{dtype}.{dname}"
                add_sym(f"data.{dtype}.{dname}", qn, "class", f'data "{dtype}" "{dname}"')

        for mname, mbody in data.get("module", {}).items():
            if not isinstance(mbody, dict):
                continue
            qn = f"{module_prefix}.module.{mname}"
            add_sym(f"module.{mname}", qn, "class", f'module "{mname}"')
            source_val = mbody.get("source")
            if isinstance(source_val, str) and source_val:
                edges.append(
                    ParsedEdge(
                        source_qualified_name=qn,
                        target_qualified_name=source_val,
                        edge_kind="IMPORTS",
                    )
                )

        for vname in data.get("variable", {}):
            add_sym(
                f"var.{vname}",
                f"{module_prefix}.var.{vname}",
                "type_alias",
                f'variable "{vname}"',
            )

        for oname in data.get("output", {}):
            add_sym(
                f"output.{oname}",
                f"{module_prefix}.output.{oname}",
                "type_alias",
                f'output "{oname}"',
            )

        return symbols, edges
