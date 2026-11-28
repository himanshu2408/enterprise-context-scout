"""First-Class Python Language Plugin (ECS-152).

Wraps Tree-sitter Python AST extraction with fail-closed syntax error
abstention, ``pyproject.toml`` / ``requirements.txt`` manifest parsing,
``@route`` / environment-variable extraction, and repository qualification.
"""

from __future__ import annotations

import logging
import re
import tomllib
from pathlib import Path

from context_scout.ast_engine.extractor import ASTExtractor, ParseDiagnostics
from context_scout.ast_engine.models import ParsedEdge, ParsedSymbol
from context_scout.plugins.base import DependencyEdge

logger = logging.getLogger(__name__)

_MANIFEST_BASENAMES: frozenset[str] = frozenset({"pyproject.toml", "requirements.txt", "setup.py"})
_REQ_LINE_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]{0,127})(.*)$")


def _parse_req_spec(raw: str) -> tuple[str, str] | None:
    cleaned = raw.split(";", 1)[0].split("#", 1)[0].strip()
    if not cleaned:
        return None
    # Strip extras like pkg[extra]>=1.0 safely without nested quantifiers
    bracket_idx = cleaned.find("[")
    close_idx = cleaned.find("]", bracket_idx + 1) if bracket_idx != -1 else -1
    if bracket_idx != -1 and close_idx != -1:
        cleaned = cleaned[:bracket_idx] + cleaned[close_idx + 1 :]
    match = _REQ_LINE_RE.match(cleaned.strip())
    if not match:
        return None
    pkg = match.group(1).strip()
    constraint = match.group(2).strip() or "*"
    return pkg, constraint


class PythonPlugin:
    """First-class Python language plugin backed by tree-sitter-python."""

    def __init__(self) -> None:
        self._extractor: ASTExtractor | None = None
        self.last_parse_diagnostics: ParseDiagnostics | None = None

    def _get_extractor(self) -> ASTExtractor:
        if self._extractor is None:
            self._extractor = ASTExtractor()
        return self._extractor

    @property
    def language_id(self) -> str:
        return "python"

    @property
    def file_extensions(self) -> tuple[str, ...]:
        return (".py", ".pyi")

    @property
    def manifest_files(self) -> tuple[str, ...]:
        return ("pyproject.toml", "requirements.txt", "setup.py")

    def can_handle(self, file_path: str) -> bool:
        lower = file_path.lower()
        return lower.endswith((".py", ".pyi"))

    def _derive_module_prefix(self, file_path: str) -> str:
        if not file_path:
            return "root"
        stem = re.sub(r"\.(?:py|pyi)$", "", file_path, flags=re.IGNORECASE)
        parts = [p.lstrip(".") for p in stem.replace("\\", "/").split("/") if p and p != "__init__"]
        return ".".join(p for p in parts if p) or "root"

    def extract_manifest_dependencies(
        self,
        file_path: str,
        content: str,
    ) -> list[DependencyEdge]:
        """Extracts Python package dependencies from pyproject.toml or requirements.txt."""
        basename = Path(file_path).name.lower()
        if basename not in _MANIFEST_BASENAMES or not content.strip():
            return []

        edges: list[DependencyEdge] = []
        if basename == "pyproject.toml":
            try:
                data = tomllib.loads(content)
            except (tomllib.TOMLDecodeError, TypeError, ValueError) as exc:
                logger.warning("Invalid pyproject.toml manifest '%s': %s", file_path, exc)
                self.last_parse_diagnostics = ParseDiagnostics(
                    parser_backend="tomllib",
                    has_parse_error=True,
                    abstained=True,
                    reason=f"Manifest parse error: {exc}",
                )
                return []

            project_tbl = data.get("project")
            src_name = (
                str(project_tbl.get("name"))
                if isinstance(project_tbl, dict) and project_tbl.get("name")
                else Path(file_path).stem
            )
            deps = project_tbl.get("dependencies", []) if isinstance(project_tbl, dict) else []
            if isinstance(deps, list):
                for item in deps:
                    if isinstance(item, str):
                        parsed = _parse_req_spec(item)
                        if parsed:
                            edges.append(
                                DependencyEdge(
                                    source_identifier=src_name,
                                    target_package=parsed[0],
                                    version_constraint=parsed[1],
                                    scope="runtime",
                                    manifest_path=file_path,
                                )
                            )
            opt_deps = (
                project_tbl.get("optional-dependencies", {})
                if isinstance(project_tbl, dict)
                else {}
            )
            if isinstance(opt_deps, dict):
                for group_name, group_items in opt_deps.items():
                    if not isinstance(group_items, list):
                        continue
                    scope = "dev" if str(group_name).lower() == "dev" else f"extra:{group_name}"
                    for item in group_items:
                        if isinstance(item, str):
                            parsed = _parse_req_spec(item)
                            if parsed:
                                edges.append(
                                    DependencyEdge(
                                        source_identifier=src_name,
                                        target_package=parsed[0],
                                        version_constraint=parsed[1],
                                        scope=scope,
                                        manifest_path=file_path,
                                    )
                                )
            return edges

        if basename == "requirements.txt":
            src_name = Path(file_path).parent.name or "python-app"
            for line in content.splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith(("#", "-")):
                    continue
                parsed = _parse_req_spec(stripped)
                if parsed:
                    edges.append(
                        DependencyEdge(
                            source_identifier=src_name,
                            target_package=parsed[0],
                            version_constraint=parsed[1],
                            scope="runtime",
                            manifest_path=file_path,
                        )
                    )
            return edges

        return edges

    def extract_ast(
        self,
        code: str,
        file_path: str,
        repo_name: str = "",
        module_prefix: str = "",
    ) -> tuple[list[ParsedSymbol], list[ParsedEdge]]:
        """Extracts Python AST symbols and edges via tree-sitter with fail-closed abstention."""
        basename = Path(file_path).name.lower() if file_path else ""
        if basename in _MANIFEST_BASENAMES:
            self.last_parse_diagnostics = ParseDiagnostics(parser_backend="manifest-python")
            deps = self.extract_manifest_dependencies(file_path, code)
            if self.last_parse_diagnostics.abstained:
                return [], []
            mod = module_prefix or self._derive_module_prefix(file_path)
            pkg_name = deps[0].source_identifier if deps else basename
            qname = f"{repo_name}::{mod}::{pkg_name}" if repo_name else f"{mod}.{pkg_name}"
            line_cnt = max(1, len(code.splitlines()))
            symbols = [
                ParsedSymbol(
                    name=pkg_name,
                    qualified_name=qname,
                    symbol_kind="module",
                    start_line=1,
                    end_line=line_cnt,
                    start_byte=0,
                    end_byte=len(code.encode("utf-8")),
                    signature=f"manifest {basename}: {pkg_name}",
                    signature_skeleton=f"manifest {pkg_name}",
                    repo_name=repo_name,
                    repo_qualified_id=qname if repo_name else None,
                    bare_name=pkg_name,
                )
            ]
            edges = [
                ParsedEdge(
                    source_qualified_name=qname,
                    target_qualified_name=dep.target_package,
                    edge_kind="IMPORTS",
                )
                for dep in deps
            ]
            return symbols, edges

        extractor = self._get_extractor()
        mod = module_prefix or (self._derive_module_prefix(file_path) if file_path else "root")
        symbols, edges = extractor._parse_source_builtin(
            code=code,
            language="python",
            file_path=file_path,
            module_prefix=mod,
            repo_name=repo_name,
        )
        self.last_parse_diagnostics = extractor.last_parse_diagnostics or ParseDiagnostics(
            parser_backend="tree-sitter-python"
        )
        if self.last_parse_diagnostics.abstained or self.last_parse_diagnostics.has_parse_error:
            return [], []
        return symbols, edges
