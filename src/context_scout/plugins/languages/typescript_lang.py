"""First-Class TypeScript / JavaScript Language Plugin (ECS-152).

Handles standard ``.ts``, ``.js``, ``.mjs``, ``.cjs``, ``.mts``, and ``.cts``
modules via tree-sitter with fail-closed syntax error abstention, ``package.json`` /
``tsconfig.json`` manifest parsing, and repository qualification. React JSX/TSX and
Redux slice modules continue routing to ``ReactReduxPlugin``.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from context_scout.ast_engine.extractor import ASTExtractor, ParseDiagnostics
from context_scout.ast_engine.models import ParsedEdge, ParsedSymbol
from context_scout.plugins.base import DependencyEdge

logger = logging.getLogger(__name__)

_TS_EXTENSIONS: tuple[str, ...] = (".ts", ".js", ".mjs", ".cjs", ".mts", ".cts")
_MANIFEST_BASENAMES: frozenset[str] = frozenset({"package.json", "tsconfig.json"})


class TypeScriptPlugin:
    """First-class TypeScript and JavaScript language plugin backed by tree-sitter."""

    def __init__(self) -> None:
        self._extractor: ASTExtractor | None = None
        self.last_parse_diagnostics: ParseDiagnostics | None = None

    def _get_extractor(self) -> ASTExtractor:
        if self._extractor is None:
            self._extractor = ASTExtractor()
        return self._extractor

    @property
    def language_id(self) -> str:
        return "typescript"

    @property
    def file_extensions(self) -> tuple[str, ...]:
        return _TS_EXTENSIONS

    @property
    def manifest_files(self) -> tuple[str, ...]:
        return ("package.json", "tsconfig.json")

    def can_handle(self, file_path: str) -> bool:
        lower = file_path.lower()
        return any(lower.endswith(ext) for ext in _TS_EXTENSIONS)

    def _derive_module_prefix(self, file_path: str) -> str:
        if not file_path:
            return "root"
        stem = re.sub(r"\.(?:mts|cts|mjs|cjs|ts|js)$", "", file_path, flags=re.IGNORECASE)
        parts = [p.lstrip(".") for p in stem.replace("\\", "/").split("/") if p]
        return ".".join(p for p in parts if p) or "root"

    def extract_manifest_dependencies(
        self,
        file_path: str,
        content: str,
    ) -> list[DependencyEdge]:
        """Extracts npm dependencies from package.json with fail-closed error handling."""
        basename = Path(file_path).name.lower()
        if basename not in _MANIFEST_BASENAMES or not content.strip():
            return []

        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning("Invalid TypeScript manifest '%s': %s", file_path, exc)
            self.last_parse_diagnostics = ParseDiagnostics(
                parser_backend="json",
                has_parse_error=True,
                abstained=True,
                reason=f"Manifest parse error: {exc}",
            )
            return []

        if not isinstance(data, dict):
            self.last_parse_diagnostics = ParseDiagnostics(
                parser_backend="json",
                has_parse_error=True,
                abstained=True,
                reason="Manifest root must be a JSON object",
            )
            return []

        if basename == "tsconfig.json":
            extends_val = data.get("extends")
            if isinstance(extends_val, str) and extends_val.strip():
                return [
                    DependencyEdge(
                        source_identifier="tsconfig",
                        target_package=extends_val.strip(),
                        version_constraint="*",
                        scope="config",
                        manifest_path=file_path,
                    )
                ]
            return []

        src_name = str(data.get("name") or Path(file_path).parent.name or "ts-package")
        edges: list[DependencyEdge] = []
        for field_name, scope in (
            ("dependencies", "runtime"),
            ("devDependencies", "dev"),
            ("peerDependencies", "peer"),
        ):
            dep_map = data.get(field_name)
            if isinstance(dep_map, dict):
                for pkg, constraint in dep_map.items():
                    edges.append(
                        DependencyEdge(
                            source_identifier=src_name,
                            target_package=str(pkg),
                            version_constraint=str(constraint),
                            scope=scope,
                            manifest_path=file_path,
                        )
                    )
        return edges

    def extract_ast(
        self,
        code: str,
        file_path: str,
        repo_name: str = "",
        module_prefix: str = "",
    ) -> tuple[list[ParsedSymbol], list[ParsedEdge]]:
        """Extracts TypeScript/JavaScript AST symbols and edges via tree-sitter."""
        basename = Path(file_path).name.lower() if file_path else ""
        if basename in _MANIFEST_BASENAMES:
            self.last_parse_diagnostics = ParseDiagnostics(parser_backend="manifest-json")
            deps = self.extract_manifest_dependencies(file_path, code)
            if self.last_parse_diagnostics.abstained:
                return [], []
            mod = module_prefix or self._derive_module_prefix(file_path)
            pkg_name = deps[0].source_identifier if deps else Path(file_path).stem
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
            language="typescript",
            file_path=file_path,
            module_prefix=mod,
            repo_name=repo_name,
        )
        self.last_parse_diagnostics = extractor.last_parse_diagnostics or ParseDiagnostics(
            parser_backend="tree-sitter-typescript"
        )
        if self.last_parse_diagnostics.abstained or self.last_parse_diagnostics.has_parse_error:
            return [], []
        return symbols, edges
