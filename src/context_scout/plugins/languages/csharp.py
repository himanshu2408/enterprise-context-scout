"""C# and .NET Language Plugin for context scout.

Extracts classes, interfaces, records, methods, DI registrations, and NuGet manifest dependencies.
Implements a high-precision C# parser for .NET codebases.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from context_scout.ast_engine.models import ParsedEdge, ParsedSymbol
from context_scout.plugins.base import DependencyEdge

logger = logging.getLogger(__name__)


class CSharpPlugin:
    """Plug-in for C# source code (.cs) and .NET project manifests (.csproj, .sln)."""

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
        normalized = file_path.replace("\\", "/").lower()
        return normalized.endswith(".cs") or any(
            normalized.endswith(m.lower()) for m in self.manifest_files
        )

    def extract_ast(
        self,
        code: str,
        file_path: str,
        repo_name: str = "",
        module_prefix: str = "",
    ) -> tuple[list[ParsedSymbol], list[ParsedEdge]]:
        """Parses C# source into symbols and directed edges."""
        if not module_prefix and file_path:
            module_prefix = self._derive_module_prefix(file_path)

        return self._parse_heuristic(code, file_path, repo_name, module_prefix)

    def extract_manifest_dependencies(
        self,
        file_path: str,
        content: str,
    ) -> list[DependencyEdge]:
        """Extracts NuGet PackageReferences and ProjectReferences from .csproj or .sln."""
        edges: list[DependencyEdge] = []
        path_lower = file_path.lower()

        if path_lower.endswith(".csproj") or path_lower.endswith(".props"):
            try:
                root = ET.fromstring(content)
                for pr in root.findall(".//PackageReference"):
                    pkg_id = pr.attrib.get("Include") or pr.attrib.get("Update")
                    version = pr.attrib.get("Version") or "*"
                    if pkg_id:
                        edges.append(
                            DependencyEdge(
                                source_identifier=Path(file_path).stem,
                                target_package=pkg_id,
                                version_constraint=version,
                                scope="compile",
                                manifest_path=file_path,
                            )
                        )
                for proj_ref in root.findall(".//ProjectReference"):
                    inc = proj_ref.attrib.get("Include")
                    if inc:
                        target_name = Path(inc.replace("\\", "/")).stem
                        edges.append(
                            DependencyEdge(
                                source_identifier=Path(file_path).stem,
                                target_package=target_name,
                                version_constraint="local",
                                scope="project_reference",
                                manifest_path=file_path,
                            )
                        )
            except ET.ParseError:
                # Regex fallback for malformed or templated XML
                pkg_pat = r'<PackageReference\s+[^>]*Include="([^"]+)"(?:\s+Version="([^"]+)")?'
                for m in re.finditer(pkg_pat, content):
                    edges.append(
                        DependencyEdge(
                            source_identifier=Path(file_path).stem,
                            target_package=m.group(1),
                            version_constraint=m.group(2) or "*",
                            scope="compile",
                            manifest_path=file_path,
                        )
                    )
        elif path_lower.endswith(".sln"):
            sln_pat = r'Project\("[^"]+"\)\s*=\s*"([^"]+)",\s*"([^"]+)"'
            for m in re.finditer(sln_pat, content):
                edges.append(
                    DependencyEdge(
                        source_identifier=Path(file_path).stem,
                        target_package=m.group(1),
                        version_constraint="solution_project",
                        scope="project_reference",
                        manifest_path=file_path,
                    )
                )

        return edges

    def _derive_module_prefix(self, file_path: str) -> str:
        clean = file_path.replace("\\", "/").lstrip("/")
        if clean.startswith("src/"):
            clean = clean[4:]
        if clean.endswith(".cs"):
            clean = clean[:-3]
        return clean.replace("/", ".")

    def _parse_heuristic(
        self,
        code: str,
        file_path: str,
        repo_name: str,
        module_prefix: str,
    ) -> tuple[list[ParsedSymbol], list[ParsedEdge]]:
        symbols: list[ParsedSymbol] = []
        edges: list[ParsedEdge] = []
        # Mask comments, raw strings (C# 11+), verbatim strings, and regular strings
        # while preserving line breaks
        clean_code = re.sub(
            r'(//[^\n]*)|'
            r'(/\*[\s\S]*?\*/)|'
            r'(\$*("{3,}))[\s\S]*?\4|'
            r'((?:\$@|@\$|@)"(?:""|[^"])*")|'
            r'(\$?"(?:\\.|[^"\\])*")|'
            r"('(?:\\.|[^'\\])*')",
            lambda m: re.sub(r"[^\n]", " ", m.group(0)),
            code,
        )
        lines = clean_code.splitlines()

        # Regex patterns
        ns_decl_pat = re.compile(r"^\s*namespace\s+([A-Za-z0-9_.]+)")
        using_pat = re.compile(r"^\s*using\s+([A-Za-z0-9_.]+)\s*;")

        # Support primary constructors with deeply nested parentheses
        # (tuples, default(...), nameof(...))
        p_inner = r"[^()]*"
        for _ in range(10):
            p_inner = rf"(?:[^()]|\({p_inner}\))*"
        paren_rec = rf"(?:\({p_inner}\))"
        type_pat = re.compile(
            r"(?:^|\s|;)(?:\[[^\]]+\]\s*)*(?:public|internal|private|protected|static|abstract|sealed|partial|\s)*"
            r"\b(class|interface|record|struct|enum)\s+([A-Za-z0-9_]+)(?:<[^>]+>)?"
            rf"(?:{paren_rec})?"
            r"(?:\s*:\s*(.*?))?\s*(?:\bwhere\b[^{;]+)?([\{;]|$)"
        )
        type_start_pat = re.compile(
            r"(?:^|\s|;)(?:\[[^\]]+\]\s*)*(?:public|internal|private|protected|static|abstract|sealed|partial|\s)*"
            r"\b(class|interface|record|struct|enum)\s+([A-Za-z0-9_]+)"
        )

        # Require at least one explicit access or method modifier;
        # disallow pure whitespace in modifier group
        method_pat = re.compile(
            r"(?:^|\s|;)(?:\[[^\]]+\]\s*)*(?:(?:public|private|protected|internal|static|"
            r"virtual|override|abstract|async|sealed|partial)\s+)+"
            r"(?:(?:Task(?:<[^>]+>)?|ValueTask(?:<[^>]+>)?|[A-Za-z0-9_<>\[\],?]+)\s+)+"
            r"([A-Za-z0-9_]+)\s*\(([^)]*)\)"
        )

        # Dependency Injection Registrations (services.AddScoped, etc.)
        di_pat = re.compile(
            r"\bservices\s*\.\s*Add(Scoped|Singleton|Transient|HostedService)\s*<\s*([A-Za-z0-9_]+)(?:\s*,\s*([A-Za-z0-9_]+))?\s*>\s*\("
        )

        def _clean_cs_line(line_str: str) -> str:
            cleaned = re.sub(r'(\$*("{3,}))[\s\S]*?\2', '""', line_str)
            cleaned = re.sub(r'((?:\$@|@\$|@)"(?:""|[^"])*")', '""', cleaned)
            cleaned = re.sub(r'"(?:\\.|[^"\\])*"', '""', cleaned)
            cleaned = re.sub(r"'(?:\\.|[^'\\])*'", "''", cleaned)
            return cleaned

        usings: list[str] = []
        file_namespace = module_prefix
        ns_stack: list[tuple[str, int]] = []  # (ns_name, entry_brace_depth)
        class_stack: list[tuple[str, str, int]] = []  # (class_name, repo_qual, entry_brace_depth)
        pending_ns: str | None = None
        pending_class: tuple[str, str] | None = None
        brace_depth = 0

        i = 0
        total_lines = len(lines)
        while i < total_lines:
            idx = i + 1
            line = lines[i]
            stripped = line.strip()
            line_no_comment = re.sub(r"//.*$", "", line)
            line_clean_cs = _clean_cs_line(line_no_comment)
            stripped_nc = line_clean_cs.strip()

            if not stripped or stripped.startswith("//") or stripped.startswith("/*"):
                i += 1
                continue

            # Using directive
            m_u = using_pat.match(stripped)
            if m_u:
                usings.append(m_u.group(1))
                brace_depth += line_clean_cs.count("{") - line_clean_cs.count("}")
                i += 1
                continue

            # Namespace directive
            m_ns = ns_decl_pat.match(stripped_nc)
            if m_ns:
                ns_name = m_ns.group(1)
                if ";" in stripped_nc:
                    file_namespace = ns_name
                elif "{" in line_clean_cs:
                    entry_depth = brace_depth + 1
                    ns_stack.append((ns_name, entry_depth))
                    pending_ns = None
                else:
                    pending_ns = ns_name

            # If there's a pending namespace and this line opens a brace
            if pending_ns and "{" in line_clean_cs and not m_ns:
                entry_depth = brace_depth + 1
                ns_stack.append((pending_ns, entry_depth))
                pending_ns = None

            # If there's a pending class and this line opens a brace
            if pending_class and "{" in line_clean_cs:
                entry_depth = brace_depth + 1
                class_stack.append((pending_class[0], pending_class[1], entry_depth))
                pending_class = None

            current_namespace = ".".join(ns for ns, _ in ns_stack) if ns_stack else file_namespace

            # Multiline type declaration accumulation
            if (
                type_start_pat.search(line_clean_cs)
                and not stripped.startswith("return ")
                and not stripped.startswith("var ")
                and not stripped.startswith("throw ")
                and not ("{" in line_clean_cs or ";" in line_clean_cs)
            ):
                buf = [stripped_nc]
                while i + 1 < total_lines and not ("{" in buf[-1] or ";" in buf[-1]):
                    i += 1
                    next_nc = _clean_cs_line(re.sub(r"//.*$", "", lines[i])).strip()
                    if next_nc:
                        buf.append(next_nc)
                line_no_comment = " ".join(buf)
                line_clean_cs = line_no_comment
                stripped_nc = line_clean_cs.strip()
                stripped = line_clean_cs.strip()

            # Type matching
            m_type = type_pat.search(line_no_comment)
            if (
                m_type
                and not stripped.startswith("return ")
                and not stripped.startswith("var ")
                and not stripped.startswith("throw ")
            ):
                kind = m_type.group(1)
                name = m_type.group(2)
                base_clause = m_type.group(3)
                terminator = m_type.group(4)

                enclosing_classes = [cls for cls, _, _ in class_stack]
                bare = ".".join([*enclosing_classes, name]) if enclosing_classes else name
                qual = f"{current_namespace}.{bare}" if current_namespace else bare
                repo_qual = f"{repo_name}::{current_namespace}::{bare}" if repo_name else qual

                sym_kind = (
                    "class"
                    if kind in ("class", "record", "struct")
                    else ("interface" if kind == "interface" else "type_alias")
                )
                symbols.append(
                    ParsedSymbol(
                        name=name,
                        qualified_name=repo_qual,
                        symbol_kind=sym_kind,
                        start_line=idx,
                        end_line=i + 1,
                        start_byte=0,
                        end_byte=0,
                        signature=stripped,
                        signature_skeleton=f"{kind} {name}",
                        contextual_summary=f"C# {kind} {name} declared in {current_namespace}",
                        repo_name=repo_name,
                        repo_qualified_id=repo_qual,
                        bare_name=bare,
                    )
                )

                if "{" in line_clean_cs:
                    entry_depth = brace_depth + 1
                    class_stack.append((name, repo_qual, entry_depth))
                    pending_class = None
                elif ";" in terminator or ";" in stripped_nc:
                    pending_class = None
                else:
                    pending_class = (name, repo_qual)

                # Base clause handling: strip generic parameters <...>
                if base_clause:
                    clean_base = re.sub(r"/\*.*?\*/", "", base_clause)
                    while "<" in clean_base and ">" in clean_base:
                        clean_base = re.sub(r"<[^<>]*>", "", clean_base)
                    ignored_base = ("where", "class", "new", "struct", "record", "notnull")
                    for base_item in re.findall(r"[A-Za-z0-9_]+", clean_base):
                        if base_item not in ignored_base:
                            edge_k = (
                                "INHERITS"
                                if base_item.startswith("I") or kind in ("class", "record")
                                else "USES_TYPE"
                            )
                            edges.append(
                                ParsedEdge(
                                    source_qualified_name=repo_qual,
                                    target_qualified_name=base_item,
                                    edge_kind=edge_k,
                                )
                            )
            # Method matching
            method_search_str = (
                line_no_comment[m_type.end():] if m_type else line_no_comment
            )
            m_meth = method_pat.search(method_search_str)
            if m_meth and class_stack:
                    m_name = m_meth.group(1)
                    forbidden = (
                        "if",
                        "while",
                        "for",
                        "switch",
                        "catch",
                        "using",
                        "lock",
                        "return",
                        "throw",
                        "yield",
                        "record",
                        "class",
                        "interface",
                        "struct",
                    )
                    if m_name not in forbidden:
                        enclosing_class_path = ".".join(cls for cls, _, _ in class_stack)
                        bare = f"{enclosing_class_path}.{m_name}"
                        qual = f"{current_namespace}.{bare}" if current_namespace else bare
                        repo_qual = (
                            f"{repo_name}::{current_namespace}::{bare}"
                            if repo_name
                            else qual
                        )
                        symbols.append(
                            ParsedSymbol(
                                name=m_name,
                                qualified_name=repo_qual,
                                symbol_kind="method",
                                start_line=idx,
                                end_line=idx,
                                start_byte=0,
                                end_byte=0,
                                signature=stripped,
                                signature_skeleton=f"{m_name}(...)",
                                contextual_summary=f"C# method {bare}",
                                repo_name=repo_name,
                                repo_qualified_id=repo_qual,
                                bare_name=bare,
                            )
                        )

            # DI Registrations
            for m_di in di_pat.finditer(line):
                svc_iface = m_di.group(2)
                svc_impl = m_di.group(3)
                source_sym = (
                    class_stack[-1][1]
                    if class_stack
                    else (
                        f"{repo_name}::{current_namespace}::Program"
                        if repo_name
                        else (
                            f"{current_namespace}.Program"
                            if current_namespace
                            else "Program"
                        )
                    )
                )
                edges.append(
                    ParsedEdge(
                        source_qualified_name=source_sym,
                        target_qualified_name=svc_iface,
                        edge_kind="PROVISIONED_BY",
                    )
                )
                if svc_impl:
                    edges.append(
                        ParsedEdge(
                            source_qualified_name=svc_iface,
                            target_qualified_name=svc_impl,
                            edge_kind="USES_TYPE",
                        )
                    )

            # Update brace depth and pop scope stacks
            opens = line_clean_cs.count("{")
            closes = line_clean_cs.count("}")
            brace_depth += opens - closes
            while class_stack and brace_depth < class_stack[-1][2]:
                class_stack.pop()
            while ns_stack and brace_depth < ns_stack[-1][1]:
                ns_stack.pop()

            i += 1

        # Connect imports to namespace
        for u in usings:
            for s in symbols:
                if s.symbol_kind in ("class", "interface"):
                    edges.append(
                        ParsedEdge(
                            source_qualified_name=s.qualified_name,
                            target_qualified_name=u,
                            edge_kind="IMPORTS",
                        )
                    )

        return symbols, edges
