"""Terraform / HCL language plugin using tree-sitter-hcl AST traversal."""

from __future__ import annotations

import json
import logging
import re
from typing import Literal

from tree_sitter import Node

from context_scout.ast_engine.extractor import (
    ParseDiagnostics,
    _collect_hcl_attr_chains,
    _get_hcl_parser,
    _node_text,
    _unquote_hcl_string,
    inspect_tree_errors,
)
from context_scout.ast_engine.models import ParsedEdge, ParsedSymbol
from context_scout.plugins.base import DependencyEdge

logger = logging.getLogger(__name__)

_TOP_HCL_ATTR_KEYS = frozenset(
    {
        "source",
        "role",
        "member",
        "network",
        "subnetwork",
        "location",
        "region",
        "name",
        "service",
    }
)
_ENV_ATTR_KEYS = frozenset({"env_vars", "environment_variables", "env", "variables"})
_HCL_BUILTIN_REFS = frozenset(
    {
        "var",
        "local",
        "count",
        "each",
        "true",
        "false",
        "null",
        "self",
        "path",
        "terraform",
        "data",
    }
)
_HCL_SYMBOL_BLOCK_TYPES = frozenset(
    {"resource", "data", "module", "variable", "output", "provider"}
)


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
        key_node: Node | None = None
        expr_node: Node | None = None
        for c in child.children:
            if c.type == "identifier" and key_node is None:
                key_node = c
            elif c.type == "expression" and expr_node is None:
                expr_node = c
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
        first_named: Node | None = None
        second_named: Node | None = None
        for c in child.children:
            if c.is_named:
                if first_named is None:
                    first_named = c
                else:
                    second_named = c
                    break
        if first_named is None or second_named is None:
            continue
        key = _unquote_hcl_string(first_named, code_bytes)
        val = _unquote_hcl_string(second_named, code_bytes)
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
        return self._extract_hcl(code, file_path, repo_name, prefix)

    def _extract_hcl(
        self,
        code: str,
        file_path: str,
        repo_name: str,
        module_prefix: str,
    ) -> tuple[list[ParsedSymbol], list[ParsedEdge]]:
        """Single-pass HCL AST extraction, dependency edge resolution, and contract tagging."""
        code_bytes = code.encode("utf-8")
        tree = _get_hcl_parser().parse(code_bytes)
        has_err, err_cnt, miss_cnt = inspect_tree_errors(tree.root_node)
        if has_err:
            reason = (
                f"Partial/invalid HCL syntax in '{file_path or '<memory>'}': "
                f"error_nodes={err_cnt}, missing_nodes={miss_cnt}"
            )
            logger.warning("%s; abstaining from partial AST extraction", reason)
            self.last_parse_diagnostics = ParseDiagnostics(
                parser_backend="tree-sitter-hcl",
                has_parse_error=True,
                abstained=True,
                low_confidence_fallback=False,
                error_node_count=err_cnt,
                missing_node_count=miss_cnt,
                reason=reason,
            )
            return [], []

        self.last_parse_diagnostics = ParseDiagnostics(parser_backend="tree-sitter-hcl")
        top_body: Node | None = None
        for child in tree.root_node.children:
            if child.type == "body":
                top_body = child
                break
        if top_body is None:
            return [], []

        symbols: list[ParsedSymbol] = []
        edges: list[ParsedEdge] = []
        sym_by_qn: dict[str, int] = {}
        local_symbols: list[tuple[str, ParsedSymbol]] = []
        local_seen: set[str] = set()
        module_edges: list[tuple[str, str, str]] = []

        def qualify(bare: str) -> str:
            if repo_name:
                return f"{repo_name}::{module_prefix}::{bare}"
            return f"{module_prefix}.{bare}" if module_prefix else bare

        def visit_block_body(
            b_node: Node,
            source_qual: str,
            self_bare: str,
            attrs: dict[str, str] | None,
            envs: list[str] | None,
            *,
            is_block_top: bool,
        ) -> None:
            for item in b_node.children:
                itype = item.type
                if itype == "attribute":
                    attr_id_node: Node | None = None
                    expr_node: Node | None = None
                    for c in item.children:
                        if c.type == "identifier" and attr_id_node is None:
                            attr_id_node = c
                        elif c.type == "expression" and expr_node is None:
                            expr_node = c
                    attr_name = _node_text(attr_id_node, code_bytes).strip()
                    if expr_node is None:
                        continue
                    if attrs is not None and is_block_top and attr_name in _TOP_HCL_ATTR_KEYS:
                        attrs[attr_name] = _unquote_hcl_string(expr_node, code_bytes)
                    if envs is not None and attr_name in _ENV_ATTR_KEYS:
                        obj_nd: Node | None = None
                        for c in expr_node.children:
                            if c.type == "object":
                                obj_nd = c
                                break
                        if obj_nd is None:
                            for sub in expr_node.children:
                                if sub.type == "collection_value":
                                    for c in sub.children:
                                        if c.type == "object":
                                            obj_nd = c
                                            break
                                    if obj_nd is not None:
                                        break
                        if obj_nd is not None:
                            envs.extend(_extract_object_strings(obj_nd, code_bytes).keys())
                    raw_expr = code_bytes[expr_node.start_byte : expr_node.end_byte]
                    if b"." not in raw_expr and b"${" not in raw_expr and attr_name != "depends_on":
                        continue
                    chains = _collect_hcl_attr_chains(expr_node, code_bytes)
                    if attr_name == "depends_on":
                        for parts in chains:
                            dep_bare = ".".join(parts[:2]) if len(parts) >= 2 else parts[0]
                            edges.append(
                                ParsedEdge(
                                    source_qualified_name=source_qual,
                                    target_qualified_name=qualify(dep_bare),
                                    edge_kind="PROVISIONED_BY",
                                )
                            )
                        continue

                    for parts in chains:
                        head = parts[0]
                        if head == "var" and len(parts) >= 2:
                            edges.append(
                                ParsedEdge(
                                    source_qualified_name=source_qual,
                                    target_qualified_name=qualify(parts[1]),
                                    edge_kind="USES_TYPE",
                                )
                            )
                        elif head == "data" and len(parts) >= 3:
                            d_bare = f"{parts[1]}.{parts[2]}"
                            if d_bare != self_bare:
                                edges.append(
                                    ParsedEdge(
                                        source_qualified_name=source_qual,
                                        target_qualified_name=qualify(d_bare),
                                        edge_kind="CALLS",
                                    )
                                )
                        elif head == "module" and len(parts) >= 2:
                            m_bare = f"module.{parts[1]}"
                            if m_bare != self_bare:
                                edges.append(
                                    ParsedEdge(
                                        source_qualified_name=source_qual,
                                        target_qualified_name=qualify(m_bare),
                                        edge_kind="CALLS",
                                    )
                                )
                        elif head not in _HCL_BUILTIN_REFS and len(parts) >= 2 and "_" in head:
                            t_bare = f"{head}.{parts[1]}"
                            if t_bare != self_bare:
                                edge_kind = "CALLS"
                                if "service_account" in head:
                                    edge_kind = "BINDS_SERVICE_ACCOUNT"
                                elif "secret" in head:
                                    edge_kind = "ACCESSES_SECRET"
                                edges.append(
                                    ParsedEdge(
                                        source_qualified_name=source_qual,
                                        target_qualified_name=qualify(t_bare),
                                        edge_kind=edge_kind,
                                    )
                                )
                elif itype == "block":
                    nested_type_node: Node | None = None
                    nested_body: Node | None = None
                    for c in item.children:
                        if c.type == "identifier" and nested_type_node is None:
                            nested_type_node = c
                        elif c.type == "body" and nested_body is None:
                            nested_body = c
                    if nested_body is not None:
                        if envs is not None and nested_type_node is not None:
                            nested_btype = _node_text(nested_type_node, code_bytes)
                            if nested_btype in {"env", "environment"}:
                                env_attrs = _extract_attribute_strings(nested_body, code_bytes)
                                if env_attrs.get("name"):
                                    envs.append(env_attrs["name"])
                        visit_block_body(
                            nested_body,
                            source_qual,
                            self_bare,
                            attrs,
                            envs,
                            is_block_top=False,
                        )

        def collect_locals(bbody: Node) -> None:
            for attr in bbody.children:
                if attr.type != "attribute":
                    continue
                key_nd: Node | None = None
                for c in attr.children:
                    if c.type == "identifier":
                        key_nd = c
                        break
                if key_nd is None:
                    continue
                lname = _node_text(key_nd, code_bytes).strip()
                if not lname:
                    continue
                lqn = qualify(f"local.{lname}")
                if lqn not in local_seen:
                    local_seen.add(lqn)
                    local_symbols.append(
                        (
                            lqn,
                            ParsedSymbol(
                                name=f"local.{lname}",
                                qualified_name=lqn,
                                bare_name=f"local.{lname}",
                                symbol_kind="type_alias",
                                start_line=attr.start_point[0] + 1,
                                end_line=attr.end_point[0] + 1,
                                start_byte=attr.start_byte,
                                end_byte=attr.end_byte,
                                signature=f"local.{lname}",
                                signature_skeleton=f"local.{lname}",
                                docstring=f"@local {lname}",
                                repo_name=repo_name,
                            ),
                        )
                    )

        def walk_unrecognized(node: Node) -> None:
            for child in node.children:
                ctype = child.type
                if ctype == "block":
                    btype, _ = _block_type_and_labels(child, code_bytes)
                    bbody = _find_body_node(child)
                    if btype == "locals" and bbody is not None:
                        collect_locals(bbody)
                    elif bbody is not None:
                        walk_unrecognized(bbody)
                elif ctype == "body":
                    walk_unrecognized(child)

        for block in top_body.children:
            if block.type != "block":
                continue
            block_children = block.children
            block_type_node: Node | None = None
            body_node: Node | None = None
            for c in block_children:
                if c.type == "identifier" and block_type_node is None:
                    block_type_node = c
                elif c.type == "body" and body_node is None:
                    body_node = c
            if block_type_node is None:
                continue
            block_type = _node_text(block_type_node, code_bytes).strip()
            if block_type == "locals":
                if body_node is not None:
                    collect_locals(body_node)
                continue
            if block_type not in _HCL_SYMBOL_BLOCK_TYPES:
                if body_node is not None:
                    walk_unrecognized(body_node)
                continue

            first_label: str | None = None
            second_label: str | None = None
            for child in block_children:
                ctype = child.type
                if ctype == "block_start":
                    break
                if ctype == "string_lit":
                    val = _unquote_hcl_string(child, code_bytes)
                elif ctype == "identifier" and child is not block_type_node:
                    val = _node_text(child, code_bytes).strip()
                else:
                    continue
                if first_label is None:
                    first_label = val
                elif second_label is None:
                    second_label = val

            if first_label is None:
                continue

            if block_type in ("resource", "data"):
                bare = f"{first_label}.{second_label}" if second_label else first_label
                sym_display = second_label or first_label
                sym_kind: Literal["function", "class", "method", "interface", "type_alias"] = (
                    "class"
                )
            elif block_type == "module":
                bare = f"module.{first_label}"
                sym_display = first_label
                sym_kind = "class"
            elif block_type in ("variable", "output"):
                bare = first_label
                sym_display = first_label
                sym_kind = "type_alias"
            else:  # provider
                bare = f"provider.{first_label}"
                sym_display = first_label
                sym_kind = "interface"

            repo_qual = qualify(bare)
            start_line = block.start_point[0] + 1
            end_line = block.end_point[0] + 1
            nl_pos = code_bytes.find(b"\n", block.start_byte, block.end_byte)
            search_end = nl_pos if nl_pos != -1 else block.end_byte
            cr_pos = code_bytes.find(b"\r", block.start_byte, search_end)
            line_end = cr_pos if cr_pos != -1 else search_end
            sig_line = (
                code_bytes[block.start_byte : line_end].decode("utf-8", errors="replace").strip()
                or f"{block_type} {bare}"
            )
            base_summary = (
                f"Terraform {block_type} '{bare}' in {file_path}"
                if file_path
                else f"Terraform {block_type} {bare}: {sig_line}"
            )

            tags: list[str] = []
            if block_type == "variable":
                tags.append(f"@env TF_VAR_{first_label}")
                upper_env = first_label.upper()
                if upper_env:
                    tags.append(f"@env {upper_env}")
                if body_node is not None:
                    visit_block_body(
                        body_node,
                        repo_qual,
                        bare,
                        None,
                        None,
                        is_block_top=True,
                    )
            elif block_type in ("resource", "data", "module") and body_node is not None:
                attrs: dict[str, str] = {}
                envs: list[str] = []
                visit_block_body(
                    body_node,
                    repo_qual,
                    bare,
                    attrs,
                    envs,
                    is_block_top=True,
                )
                if block_type == "module":
                    src_val = attrs.get("source", "")
                    if src_val and not src_val.startswith(("var.", "local.")):
                        module_edges.append((repo_qual, src_val, "IMPORTS"))
                        module_edges.append((repo_qual, src_val, "PROVISIONED_BY"))
                        tags.append(f"@module_source {src_val}")
                    svc_name = attrs.get("name") or attrs.get("service") or ""
                    if svc_name:
                        tags.append(f"@service {svc_name}")
                    for env_key in dict.fromkeys(envs):
                        tags.append(f"@env {env_key}")
                elif second_label is not None:
                    if first_label.startswith("google_"):
                        tags.append("@cloud gcp")
                    elif first_label.startswith("aws_"):
                        tags.append("@cloud aws")
                    elif first_label.startswith("azurerm_"):
                        tags.append("@cloud azure")
                    if "iam" in first_label:
                        if attrs.get("role"):
                            tags.append(f"@iam_role {attrs['role']}")
                        if attrs.get("member"):
                            tags.append(f"@iam_member {attrs['member']}")
                    if attrs.get("network"):
                        tags.append(f"@network {attrs['network']}")
                    if attrs.get("subnetwork"):
                        tags.append(f"@subnetwork {attrs['subnetwork']}")
                    loc = attrs.get("location") or attrs.get("region") or ""
                    if loc and not loc.startswith(("var.", "local.")):
                        tags.append(f"@region {loc}")
                    svc_name = attrs.get("name") or attrs.get("service") or ""
                    if svc_name:
                        tags.append(f"@service {svc_name}")
                    for env_key in dict.fromkeys(envs):
                        tags.append(f"@env {env_key}")
            elif body_node is not None:
                visit_block_body(
                    body_node,
                    repo_qual,
                    bare,
                    None,
                    None,
                    is_block_top=True,
                )

            if tags:
                merged_doc = " ".join(dict.fromkeys(tags))
                docstring = merged_doc
                contextual_summary = merged_doc
            else:
                docstring = ""
                contextual_summary = base_summary

            sym_by_qn[repo_qual] = len(symbols)
            symbols.append(
                ParsedSymbol(
                    name=sym_display,
                    qualified_name=repo_qual,
                    symbol_kind=sym_kind,
                    start_line=start_line,
                    end_line=end_line,
                    start_byte=block.start_byte,
                    end_byte=block.end_byte,
                    signature=sig_line,
                    signature_skeleton=f"{block_type} {bare}",
                    docstring=docstring,
                    contextual_summary=contextual_summary,
                    repo_name=repo_name,
                    repo_qualified_id=repo_qual if repo_name else None,
                    bare_name=bare,
                )
            )

        for lqn, lsym in local_symbols:
            if lqn not in sym_by_qn:
                sym_by_qn[lqn] = len(symbols)
                symbols.append(lsym)

        if module_edges:
            seen_edges: set[tuple[str, str, str]] = {
                (e.source_qualified_name, e.target_qualified_name, e.edge_kind) for e in edges
            }
            for src, tgt, kind in module_edges:
                if src and tgt and (src, tgt, kind) not in seen_edges:
                    seen_edges.add((src, tgt, kind))
                    edges.append(
                        ParsedEdge(
                            source_qualified_name=src,
                            target_qualified_name=tgt,
                            edge_kind=kind,
                        )
                    )

        return symbols, edges

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
