"""Terraform and Cloud Infrastructure Plugin for context scout.

Extracts Terraform HCL blocks (resources, data, modules, variables, outputs)
and Google Cloud Platform infrastructure declarations, mapping explicit dependencies
(depends_on) and service bindings (BINDS_SERVICE_ACCOUNT, ACCESSES_SECRET).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from context_scout.ast_engine.models import ParsedEdge, ParsedSymbol
from context_scout.plugins.base import DependencyEdge

logger = logging.getLogger(__name__)


class TerraformPlugin:
    """Extracts Terraform HCL infrastructure declarations and GCP resource graphs."""

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
        normalized = file_path.replace("\\", "/").lower()
        return normalized.endswith((".tf", ".tfvars", ".tf.json"))

    def extract_ast(
        self,
        code: str,
        file_path: str,
        repo_name: str = "",
        module_prefix: str = "",
    ) -> tuple[list[ParsedSymbol], list[ParsedEdge]]:
        if not module_prefix and file_path:
            module_prefix = self._derive_module_prefix(file_path)

        symbols: list[ParsedSymbol] = []
        edges: list[ParsedEdge] = []

        # Strip block comments while preserving newlines for line numbering
        clean_code = re.sub(
            r'("(?:[^"\\]|\\.)*")|(/\*[\s\S]*?\*/)',
            lambda m: m.group(1) if m.group(1) is not None else "\n" * m.group(2).count("\n"),
            code,
        )
        lines = clean_code.splitlines()

        tf_block_pat = re.compile(
            r'^\s*(resource|data|module|variable|output|provider)\s+"([^"]+)"(?:\s+"([^"]+)")?\s*\{'
        )
        depends_on_pat = re.compile(r"depends_on\s*=\s*\[([^\]]+)\]", re.DOTALL)
        ref_pat = re.compile(r"\b([a-z0-9_]+)\.([a-z0-9_]+)")
        data_ref_pat = re.compile(r"\bdata\.([a-z0-9_]+)\.([a-z0-9_]+)")
        var_pat = re.compile(r"\bvar\.([A-Za-z0-9_-]+)")

        def _strip_hcl_comment(line_str: str) -> str:
            return re.sub(
                r'("(?:[^"\\]|\\.)*")|(#.*|//.*)',
                lambda m: m.group(1) if m.group(1) is not None else "",
                line_str,
            )

        # Mask braces inside heredoc blocks (<<-EOF ... EOF) so embedded shell scripts,
        # JSON, or config templates do not corrupt brace_depth tracking.
        masked_lines: list[str] = []
        in_heredoc: str | None = None
        for line in lines:
            if in_heredoc:
                if re.match(rf"^[ \t]*{re.escape(in_heredoc)}[ \t]*(?:#.*|//.*)?$", line):
                    in_heredoc = None
                masked_lines.append(re.sub(r"[{}]", " ", line))
            else:
                clean_no_str = re.sub(
                    r'"(?:[^"\\]|\\.)*"',
                    '""',
                    _strip_hcl_comment(line),
                )
                m_hdoc = re.search(r"<<-?([A-Za-z0-9_]+)", clean_no_str)
                if m_hdoc:
                    in_heredoc = m_hdoc.group(1)
                masked_lines.append(line)

        i = 0
        total_lines = len(lines)

        while i < total_lines:
            m = tf_block_pat.match(lines[i])
            if not m:
                i += 1
                continue

            block_type = m.group(1)
            first_label = m.group(2)
            second_label = m.group(3)

            start_line = i + 1
            first_clean = _strip_hcl_comment(masked_lines[i])
            brace_depth = first_clean.count("{") - first_clean.count("}")
            end_idx = i

            while end_idx + 1 < total_lines and brace_depth > 0:
                end_idx += 1
                curr_clean = _strip_hcl_comment(masked_lines[end_idx])
                brace_depth += curr_clean.count("{") - curr_clean.count("}")

            end_line = end_idx + 1
            block_content = "\n".join(lines[i:end_line])

            if block_type in ("resource", "data"):
                res_type = first_label
                res_name = second_label or ""
                bare = f"{res_type}.{res_name}" if res_name else res_type
                sym_display = res_name or res_type
                sym_kind = "class"
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

            repo_qual = f"{repo_name}::{module_prefix}::{bare}" if repo_name else (
                f"{module_prefix}.{bare}" if module_prefix else bare
            )

            symbols.append(
                ParsedSymbol(
                    name=sym_display,
                    qualified_name=repo_qual,
                    symbol_kind=sym_kind,
                    start_line=start_line,
                    end_line=end_line,
                    start_byte=0,
                    end_byte=0,
                    signature=lines[i].strip(),
                    signature_skeleton=f"{block_type} {bare}",
                    contextual_summary=f"Terraform {block_type} '{bare}' in {file_path}",
                    repo_name=repo_name,
                    repo_qualified_id=repo_qual,
                    bare_name=bare,
                )
            )

            # Extract variable references
            for m_var in var_pat.finditer(block_content):
                v_name = m_var.group(1)
                t_qual = (
                    f"{repo_name}::{module_prefix}::{v_name}"
                    if repo_name
                    else (f"{module_prefix}.{v_name}" if module_prefix else v_name)
                )
                edges.append(
                    ParsedEdge(
                        source_qualified_name=repo_qual,
                        target_qualified_name=t_qual,
                        edge_kind="USES_TYPE",
                    )
                )

            # Clean comments and static strings from block_content to avoid phantom edges
            clean_content = _strip_hcl_comment(block_content)

            def _clean_strings_for_refs(content: str) -> str:
                # Replace string literals and heredocs with only their ${...}
                # interpolation expressions
                def repl(match: re.Match) -> str:
                    s = match.group(0)
                    interp = re.findall(r"\$\{([^}]+)\}", s)
                    return " " + " ".join(interp) + " "

                # Clean heredocs: <<-?DELIM ... DELIM
                hdoc_pat = re.compile(
                    r"<<-?([A-Za-z0-9_]+)[^\n]*\n([\s\S]*?)\n[ \t]*\1[ \t]*(?=\r?\n|$)"
                )
                content = hdoc_pat.sub(repl, content)
                return re.sub(r'"(?:[^"\\]|\\.)*"', repl, content)

            clean_content = _clean_strings_for_refs(clean_content)

            # Extract data source cross-references (e.g. data.google_compute_image.debian.self_link)
            for m_data_ref in data_ref_pat.finditer(clean_content):
                d_type = m_data_ref.group(1)
                d_name = m_data_ref.group(2)
                d_bare = f"{d_type}.{d_name}"
                if d_bare != bare:
                    t_qual = (
                        f"{repo_name}::{module_prefix}::{d_bare}"
                        if repo_name
                        else (f"{module_prefix}.{d_bare}" if module_prefix else d_bare)
                    )
                    edges.append(
                        ParsedEdge(
                            source_qualified_name=repo_qual,
                            target_qualified_name=t_qual,
                            edge_kind="CALLS",
                        )
                    )

            # Strip data source references before scanning for resource references so
            # data source names containing underscores do not emit phantom CALLS edges
            clean_content = re.sub(
                r"\bdata\.[a-z0-9_]+\.[a-z0-9_]+(?:\.[a-z0-9_]+)?", " ", clean_content
            )

            # Extract resource cross-references (e.g. google_service_account.scout_sa.email)
            for m_ref in ref_pat.finditer(clean_content):
                target_type = m_ref.group(1)
                target_name = m_ref.group(2)
                if target_type in ("var", "local", "count", "each", "true", "false", "data"):
                    continue
                # Filter domain extensions and common non-resource suffixes
                non_resource_suffixes = (
                    "io", "com", "org", "net", "dev", "sh", "json", "yaml", "yml", "tf"
                )
                if target_name in non_resource_suffixes:
                    continue
                # Terraform resource types have underscores (google_*), or module prefix
                if "_" not in target_type and target_type != "module":
                    continue

                target_bare = f"{target_type}.{target_name}"
                if target_bare != bare:
                    target_qual = (
                        f"{repo_name}::{module_prefix}::{target_bare}"
                        if repo_name
                        else (f"{module_prefix}.{target_bare}" if module_prefix else target_bare)
                    )
                    edge_kind = "CALLS"
                    if "service_account" in target_type:
                        edge_kind = "BINDS_SERVICE_ACCOUNT"
                    elif "secret" in target_type:
                        edge_kind = "ACCESSES_SECRET"
                    edges.append(
                        ParsedEdge(
                            source_qualified_name=repo_qual,
                            target_qualified_name=target_qual,
                            edge_kind=edge_kind,
                        )
                    )

            # Extract explicit depends_on
            m_dep = depends_on_pat.search(block_content)
            if m_dep:
                raw_deps = m_dep.group(1)
                for dep_item in re.findall(r"[A-Za-z0-9_.]+", raw_deps):
                    dep_qual = (
                        f"{repo_name}::{module_prefix}::{dep_item}"
                        if repo_name
                        else (f"{module_prefix}.{dep_item}" if module_prefix else dep_item)
                    )
                    edges.append(
                        ParsedEdge(
                            source_qualified_name=repo_qual,
                            target_qualified_name=dep_qual,
                            edge_kind="PROVISIONED_BY",
                        )
                    )

            i = max(i + 1, end_idx + 1)

        return symbols, edges

    def extract_manifest_dependencies(self, file_path: str, content: str) -> list[DependencyEdge]:
        """Extracts required_providers from Terraform declarations."""
        edges: list[DependencyEdge] = []
        provider_block_pat = re.compile(
            r'([A-Za-z0-9_-]+)\s*=\s*\{([^}]+)\}',
            re.DOTALL,
        )
        source_pat = re.compile(r'source\s*=\s*"([^"]+)"')
        version_pat = re.compile(r'version\s*=\s*"([^"]+)"')

        for m in provider_block_pat.finditer(content):
            block_body = m.group(2)
            m_src = source_pat.search(block_body)
            if m_src:
                m_ver = version_pat.search(block_body)
                ver = m_ver.group(1) if m_ver else "*"
                edges.append(
                    DependencyEdge(
                        source_identifier=Path(file_path).stem,
                        target_package=m_src.group(1),
                        version_constraint=ver,
                        scope="provider",
                        manifest_path=file_path,
                    )
                )
        return edges

    def _derive_module_prefix(self, file_path: str) -> str:
        clean = file_path.replace("\\", "/").lstrip("/")
        if clean.startswith("src/"):
            clean = clean[4:]
        for ext in (".tfvars", ".tf.json", ".tf"):
            if clean.endswith(ext):
                clean = clean[:-len(ext)]
                break
        parts = [p.lstrip(".") for p in clean.split("/") if p]
        return ".".join(parts)
