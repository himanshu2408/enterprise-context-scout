"""Storybook & MDX Language Plugin for context scout.

Extracts Component Story Format 3 (CSF3) metadata, story variants, and component documentation.
Emits DOCUMENTS_COMPONENT edges linking stories to source React/HTML components.
"""

from __future__ import annotations

import logging
import re

from context_scout.ast_engine.models import ParsedEdge, ParsedSymbol
from context_scout.plugins.base import DependencyEdge

logger = logging.getLogger(__name__)


class StorybookPlugin:
    """Extracts Storybook CSF3 and MDX stories and documentation graphs."""

    @property
    def language_id(self) -> str:
        return "storybook"

    @property
    def file_extensions(self) -> tuple[str, ...]:
        return (".stories.tsx", ".stories.ts", ".stories.jsx", ".stories.js", ".mdx")

    @property
    def manifest_files(self) -> tuple[str, ...]:
        return (".storybook/main.ts", ".storybook/main.js")

    def can_handle(self, file_path: str) -> bool:
        normalized = file_path.replace("\\", "/").lower()
        return ".stories." in normalized or normalized.endswith(".mdx")

    def extract_ast(
        self,
        code: str,
        file_path: str,
        repo_name: str = "",
        module_prefix: str = "",
    ) -> tuple[list[ParsedSymbol], list[ParsedEdge]]:
        if not module_prefix and file_path:
            module_prefix = self._derive_module_prefix(file_path)

        if file_path.lower().endswith(".mdx"):
            return self._extract_mdx(code, file_path, repo_name, module_prefix)

        return self._extract_csf3(code, file_path, repo_name, module_prefix)

    def _extract_csf3(
        self,
        code: str,
        file_path: str,
        repo_name: str,
        module_prefix: str,
    ) -> tuple[list[ParsedSymbol], list[ParsedEdge]]:
        symbols: list[ParsedSymbol] = []
        edges: list[ParsedEdge] = []
        lines = code.splitlines()

        doc_component: str | None = None
        meta_title = ""

        meta_title_pat = re.compile(r"title\s*:\s*['\"]([^'\"]+)['\"]")
        meta_comp_pat = re.compile(r"component\s*:\s*([A-Za-z0-9_]+)")
        meta_typeof_pat = re.compile(r"Meta<typeof\s+([A-Za-z0-9_]+)>")
        imp_pat = re.compile(
            r"import\s+(?:type\s+)?(?:([A-Za-z0-9_]+)\s*,?\s*)?(?:\{([^}]+)\})?\s+from\s+['\"]([^'\"]+)['\"]",
            re.DOTALL,
        )
        story_variant_pat = re.compile(
            r"export\s+const\s+([A-Za-z0-9_]+)\s*(?::\s*(?:Story|StoryObj)[^=]*)?=\s*"
        )

        candidates_from_imports: list[str] = []

        clean_code = re.sub(r"/\*[\s\S]*?\*/", "", code)
        clean_code = re.sub(r"//[^\n]*", "", clean_code)

        for m_imp in imp_pat.finditer(clean_code):
            default_id = m_imp.group(1)
            named_ids = m_imp.group(2)
            src_path = m_imp.group(3)
            if not ("@storybook" in src_path or src_path.startswith("react")):
                if default_id and default_id not in ("type", "React"):
                    candidates_from_imports.append(default_id)
                if named_ids:
                    clean_named = re.sub(r"/\*[\s\S]*?\*/", "", named_ids)
                    clean_named = re.sub(r"//[^\n]*", "", clean_named)
                    for item in clean_named.split(","):
                        clean_item = item.strip()
                        if " " in clean_item:
                            clean_item = clean_item.split()[-1]
                        if (
                            clean_item
                            and re.match(r"^[A-Za-z_$][A-Za-z0-9_$]*$", clean_item)
                            and clean_item
                            not in (
                                "type",
                                "React",
                                "FC",
                                "Props",
                                "Story",
                                "StoryObj",
                                "Meta",
                            )
                        ):
                            candidates_from_imports.append(clean_item)

        for ln in lines:
            m_t = meta_title_pat.search(ln)
            if m_t:
                meta_title = m_t.group(1)
            m_c = meta_comp_pat.search(ln)
            if m_c and m_c.group(1) not in ("undefined", "null"):
                doc_component = m_c.group(1)
            m_type = meta_typeof_pat.search(ln)
            if m_type and not doc_component:
                doc_component = m_type.group(1)

        if not doc_component and candidates_from_imports:
            doc_component = candidates_from_imports[0]

        # Meta symbol
        meta_bare = "Meta"
        meta_qual = (
            f"{repo_name}::{module_prefix}::{meta_bare}"
            if repo_name
            else f"{module_prefix}.{meta_bare}"
        )
        symbols.append(
            ParsedSymbol(
                name="Meta",
                qualified_name=meta_qual,
                symbol_kind="story",
                start_line=1,
                end_line=1,
                start_byte=0,
                end_byte=0,
                signature=f"Meta: title='{meta_title}'",
                signature_skeleton=f"Meta<{doc_component or 'unknown'}>",
                contextual_summary=f"Storybook CSF3 Meta definition for '{meta_title}'",
                repo_name=repo_name,
                repo_qualified_id=meta_qual,
                bare_name=meta_bare,
            )
        )

        if doc_component:
            edges.append(
                ParsedEdge(
                    source_qualified_name=meta_qual,
                    target_qualified_name=doc_component,
                    edge_kind="DOCUMENTS_COMPONENT",
                )
            )

        # Story variants extraction
        for idx, line in enumerate(lines, start=1):
            for m_story in story_variant_pat.finditer(line):
                if m_story and m_story.group(1) not in ("meta", "default"):
                    story_name = m_story.group(1)
                    bare = story_name
                    repo_qual = (
                        f"{repo_name}::{module_prefix}::{bare}"
                        if repo_name
                        else f"{module_prefix}.{bare}"
                    )
                    symbols.append(
                        ParsedSymbol(
                            name=story_name,
                            qualified_name=repo_qual,
                            symbol_kind="story",
                            start_line=idx,
                            end_line=idx,
                            start_byte=0,
                            end_byte=0,
                            signature=line.strip(),
                            signature_skeleton=f"export const {story_name}: Story",
                            contextual_summary=(
                                f"Storybook variant '{story_name}' documenting "
                                f"{doc_component or 'component'}"
                            ),
                            repo_name=repo_name,
                            repo_qualified_id=repo_qual,
                            bare_name=bare,
                        )
                    )
                    if doc_component:
                        edges.append(
                            ParsedEdge(
                                source_qualified_name=repo_qual,
                                target_qualified_name=doc_component,
                                edge_kind="DOCUMENTS_COMPONENT",
                            )
                        )

        return symbols, edges

    def _extract_mdx(
        self,
        code: str,
        file_path: str,
        repo_name: str,
        module_prefix: str,
    ) -> tuple[list[ParsedSymbol], list[ParsedEdge]]:
        symbols: list[ParsedSymbol] = []
        edges: list[ParsedEdge] = []
        lines = code.splitlines()

        meta_pat = re.compile(
            r"<Meta\s+title=['\"]([^'\"]+)['\"](?:\s+component=\{([A-Za-z0-9_]+)\})?"
        )
        story_pat = re.compile(r"<Story\s+name=['\"]([^'\"]+)['\"]")

        target_comp: str | None = None
        for idx, line in enumerate(lines, start=1):
            m_meta = meta_pat.search(line)
            if m_meta:
                title = m_meta.group(1)
                target_comp = m_meta.group(2)
                bare = "Meta"
                repo_qual = (
                    f"{repo_name}::{module_prefix}::{bare}"
                    if repo_name
                    else f"{module_prefix}.{bare}"
                )
                symbols.append(
                    ParsedSymbol(
                        name="Meta",
                        qualified_name=repo_qual,
                        symbol_kind="story",
                        start_line=idx,
                        end_line=idx,
                        start_byte=0,
                        end_byte=0,
                        signature=line.strip(),
                        signature_skeleton=f"<Meta title='{title}' />",
                        contextual_summary=f"Storybook MDX Meta '{title}'",
                        repo_name=repo_name,
                        repo_qualified_id=repo_qual,
                        bare_name=bare,
                    )
                )
                if target_comp:
                    edges.append(
                        ParsedEdge(
                            source_qualified_name=repo_qual,
                            target_qualified_name=target_comp,
                            edge_kind="DOCUMENTS_COMPONENT",
                        )
                    )

            m_st = story_pat.search(line)
            if m_st:
                st_name = m_st.group(1)
                bare = st_name
                repo_qual = (
                    f"{repo_name}::{module_prefix}::{bare}"
                    if repo_name
                    else f"{module_prefix}.{bare}"
                )
                symbols.append(
                    ParsedSymbol(
                        name=st_name,
                        qualified_name=repo_qual,
                        symbol_kind="story",
                        start_line=idx,
                        end_line=idx,
                        start_byte=0,
                        end_byte=0,
                        signature=line.strip(),
                        signature_skeleton=f"<Story name='{st_name}' />",
                        contextual_summary=f"Storybook MDX story variant '{st_name}'",
                        repo_name=repo_name,
                        repo_qualified_id=repo_qual,
                        bare_name=bare,
                    )
                )
                if target_comp:
                    edges.append(
                        ParsedEdge(
                            source_qualified_name=repo_qual,
                            target_qualified_name=target_comp,
                            edge_kind="DOCUMENTS_COMPONENT",
                        )
                    )

        return symbols, edges

    def extract_manifest_dependencies(self, file_path: str, content: str) -> list[DependencyEdge]:
        return []

    def _derive_module_prefix(self, file_path: str) -> str:
        clean = file_path.replace("\\", "/").lstrip("/")
        if clean.startswith("src/"):
            clean = clean[4:]
        for suffix in (".stories.tsx", ".stories.ts", ".stories.jsx", ".stories.js", ".mdx"):
            if clean.endswith(suffix):
                clean = clean[:-len(suffix)]
                break
        return clean.replace("/", ".")
