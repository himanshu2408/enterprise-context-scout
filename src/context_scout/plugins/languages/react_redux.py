"""React & Redux Language Plugin for context scout.

Extracts React components, custom hooks, useState/useEffect dependencies,
and Redux Toolkit slices, reducers, extraReducers, actions, and selectors.
Emits semantic DISPATCHES and SELECTS edges.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from context_scout.ast_engine.models import ParsedEdge, ParsedSymbol
from context_scout.plugins.base import DependencyEdge

logger = logging.getLogger(__name__)


class ReactReduxPlugin:
    """Extracts semantic React hooks and Redux Toolkit state architectures."""

    @property
    def language_id(self) -> str:
        return "react_redux"

    @property
    def file_extensions(self) -> tuple[str, ...]:
        return (".tsx", ".jsx")

    @property
    def manifest_files(self) -> tuple[str, ...]:
        return ("package.json",)

    def can_handle(self, file_path: str) -> bool:
        normalized = file_path.replace("\\", "/").lower()
        if not normalized.endswith((".tsx", ".jsx", ".ts", ".js", ".mjs", ".cjs")):
            return False
        if ".stories." in normalized or normalized.endswith(".mdx"):
            return False  # Handled by StorybookPlugin

        if normalized.endswith((".tsx", ".jsx")):
            return True

        keywords = ("slice", "reducer", "hook", "component", "store", "feature")
        return any(keyword in normalized for keyword in keywords)

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
        # Mask block comments and line comments preserving newlines, and mask template literals
        clean_code = re.sub(
            r'("(?:\\.|[^"\\])*")|'
            r"('(?:\\.|[^'\\])*')|"
            r"(`(?:\\.|[^`\\])*`)|"
            r"(//[^\n]*)|"
            r"(/\*[\s\S]*?\*/)",
            lambda m: m.group(0)
            if (m.group(1) or m.group(2))
            else (
                re.sub(r"[^\n]", " ", m.group(3))
                if m.group(3)
                else (" " * len(m.group(4)) if m.group(4) else re.sub(r"[^\n]", " ", m.group(5)))
            ),
            code,
        )
        lines = clean_code.splitlines()

        # 1. Redux createSlice Extraction (supports single-line, multiline, and generic types)
        slice_pat = re.compile(
            r"(?:export\s+)?const\s+([A-Za-z0-9_]+)\s*=\s*createSlice(?:<[^>]+>)?\s*\("
        )
        reducers_block_pat = re.compile(r"\breducers\s*:\s*\{")

        # 2. Reducer function inside slice (supports method shorthand and arrow functions)
        reducer_key_pat = re.compile(
            r"^\s*([A-Za-z0-9_]+)\s*(?::\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z0-9_]+)\s*=>|\((state(?:,\s*action)?[^)]*)\))"
        )

        # 3. Selectors
        selector_pat = re.compile(
            r"(?:export\s+)?const\s+(select[A-Za-z0-9_]+)\s*=\s*(?:createSelector\(|(?:\([^)]*\)\s*=>))"
        )

        # 4. React Component / Custom Hook
        # (supports TS generics <TData, TValue>, React.FC, arrow & function)
        generic_param_pat = r"<[^<>]*(?:<[^<>]*(?:<[^<>]*>[^<>]*)*>[^<>]*)*>"
        hoc_wrapper_pat = rf"(?:(?:React\.)?(?:memo|forwardRef)(?:{generic_param_pat})?\s*\(\s*)*"
        comp_pat = re.compile(
            r"(?:export\s+)?(?:const|function)\s+([A-Z][A-Za-z0-9_]+|use[A-Z][A-Za-z0-9_]+)"
            rf"(?:\s*{generic_param_pat})?"
            rf"(?:\s*:\s*(?:React\.)?(?:FC|FunctionComponent|[A-Za-z0-9_]+)(?:{generic_param_pat})?)?"
            rf"\s*(?:=\s*{hoc_wrapper_pat}(?:async\s*)?(?:{generic_param_pat}\s*)?(?:\([^)]*\)?|function)|\([^)]*\)?)"
        )
        comp_start_pat = re.compile(
            r"^\s*(?:export\s+)?(?:const|function)\s+([A-Z][A-Za-z0-9_]+|use[A-Z][A-Za-z0-9_]+)"
        )

        # 5. Hook usage inside functions (useDispatch, useSelector, useState, useEffect)
        use_dispatch_call = re.compile(
            r"\bdispatch\s*\(\s*(?:[A-Za-z0-9_]+\.actions\.)?([A-Za-z0-9_]+)\s*\("
        )
        use_selector_call = re.compile(r"\buse(?:App)?Selector\s*\(\s*([A-Za-z0-9_]+)\s*\)")
        use_hook_call = re.compile(r"\b(use[A-Z][A-Za-z0-9_]+)(?:<[^>]+>)?\s*\(")

        brace_depth = 0
        current_slice: str | None = None
        slice_brace_depth: int | None = None
        pending_slice_brace = False

        in_reducers_block = False
        reducers_brace_depth: int | None = None
        pending_reducers_brace = False

        comp_stack: list[tuple[str, str, int]] = []
        pending_comp: tuple[str, str] | None = None

        def _clean_js_line(line_str: str) -> str:
            cleaned = re.sub(r"/\*.*?\*/", "", line_str)
            cleaned = re.sub(r"//.*$", "", cleaned)
            cleaned = re.sub(r"'(?:\\.|[^'\\])*'", "''", cleaned)
            cleaned = re.sub(r'"(?:\\.|[^"\\])*"', '""', cleaned)
            cleaned = re.sub(r"`(?:\\.|[^`\\])*`", "``", cleaned)
            return cleaned

        i = 0
        total_lines = len(lines)
        while i < total_lines:
            idx = i + 1
            line = lines[i]
            stripped = line.strip()

            # Multiline generic component accumulation
            angle_depth = line.count("<") - line.count(">")
            if (
                comp_start_pat.match(line)
                and not comp_pat.search(line)
                and (angle_depth > 0 or not ("{" in line or "=>" in line or ";" in line))
            ):
                buf = [stripped]
                while i + 1 < total_lines and (
                    angle_depth > 0 or not ("{" in buf[-1] or "=>" in buf[-1] or ";" in buf[-1])
                ):
                    i += 1
                    next_l = lines[i].strip()
                    if next_l:
                        buf.append(next_l)
                        angle_depth += next_l.count("<") - next_l.count(">")
                line = " ".join(buf)
                stripped = line.strip()

            line_clean = _clean_js_line(line)

            # Slice check
            m_slice = slice_pat.search(line)
            if m_slice:
                slice_name = m_slice.group(1)
                current_slice = slice_name
                bare = slice_name
                repo_qual = (
                    f"{repo_name}::{module_prefix}::{bare}"
                    if repo_name
                    else f"{module_prefix}.{bare}"
                )
                symbols.append(
                    ParsedSymbol(
                        name=slice_name,
                        qualified_name=repo_qual,
                        symbol_kind="slice",
                        start_line=idx,
                        end_line=idx,
                        start_byte=0,
                        end_byte=0,
                        signature=f"const {slice_name} = createSlice({{ ... }})",
                        signature_skeleton=f"createSlice({slice_name})",
                        contextual_summary=f"Redux Toolkit slice {slice_name}",
                        repo_name=repo_name,
                        repo_qualified_id=repo_qual,
                        bare_name=bare,
                    )
                )
                if "{" in line_clean:
                    slice_brace_depth = brace_depth + 1
                    pending_slice_brace = False
                else:
                    slice_brace_depth = None
                    pending_slice_brace = True

            if current_slice and reducers_block_pat.search(line):
                in_reducers_block = True
                if "{" in line_clean:
                    reducers_brace_depth = brace_depth + 1
                    pending_reducers_brace = False
                else:
                    reducers_brace_depth = None
                    pending_reducers_brace = True

            # Reducer method inside slice
            if in_reducers_block and current_slice:
                m_red = reducer_key_pat.match(line)
                if m_red and m_red.group(1) not in ("reducers", "extraReducers"):
                    red_name = m_red.group(1)
                    bare = f"{current_slice}.reducers.{red_name}"
                    repo_qual = (
                        f"{repo_name}::{module_prefix}::{bare}"
                        if repo_name
                        else f"{module_prefix}.{bare}"
                    )
                    symbols.append(
                        ParsedSymbol(
                            name=red_name,
                            qualified_name=repo_qual,
                            symbol_kind="reducer",
                            start_line=idx,
                            end_line=idx,
                            start_byte=0,
                            end_byte=0,
                            signature=stripped,
                            signature_skeleton=f"{red_name}(state, action)",
                            contextual_summary=f"Redux reducer {red_name} on {current_slice}",
                            repo_name=repo_name,
                            repo_qualified_id=repo_qual,
                            bare_name=bare,
                        )
                    )

            # Selector check
            m_sel = selector_pat.search(line)
            if m_sel:
                sel_name = m_sel.group(1)
                bare = sel_name
                repo_qual = (
                    f"{repo_name}::{module_prefix}::{bare}"
                    if repo_name
                    else f"{module_prefix}.{bare}"
                )
                symbols.append(
                    ParsedSymbol(
                        name=sel_name,
                        qualified_name=repo_qual,
                        symbol_kind="selector",
                        start_line=idx,
                        end_line=idx,
                        start_byte=0,
                        end_byte=0,
                        signature=stripped,
                        signature_skeleton=f"{sel_name}(state)",
                        contextual_summary=f"Redux selector {sel_name}",
                        repo_name=repo_name,
                        repo_qualified_id=repo_qual,
                        bare_name=bare,
                    )
                )

            # Component / Custom Hook check
            m_comp = comp_pat.search(line)
            if m_comp and not stripped.startswith("//") and not stripped.startswith("/*"):
                comp_name = m_comp.group(1)
                is_hook = comp_name.startswith("use")
                bare = comp_name
                repo_qual = (
                    f"{repo_name}::{module_prefix}::{bare}"
                    if repo_name
                    else f"{module_prefix}.{bare}"
                )
                symbols.append(
                    ParsedSymbol(
                        name=comp_name,
                        qualified_name=repo_qual,
                        symbol_kind="hook" if is_hook else "class",
                        start_line=idx,
                        end_line=idx,
                        start_byte=0,
                        end_byte=0,
                        signature=stripped,
                        signature_skeleton=(
                            f"{comp_name}(props)" if not is_hook else f"{comp_name}()"
                        ),
                        contextual_summary=(
                            f"React {'custom hook' if is_hook else 'component'} {comp_name}"
                        ),
                        repo_name=repo_name,
                        repo_qualified_id=repo_qual,
                        bare_name=bare,
                    )
                )
                if "{" in line_clean:
                    entry_depth = brace_depth + 1
                    comp_stack.append((comp_name, repo_qual, entry_depth))
                    pending_comp = None
                elif "=>" in line and ";" in line:
                    pending_comp = None
                else:
                    pending_comp = (comp_name, repo_qual)

            active_comp_name, active_comp_qual = (
                comp_stack[-1][0],
                comp_stack[-1][1],
            ) if comp_stack else (None, None)

            # Hook calls inside current component
            if active_comp_qual:
                for m_disp in use_dispatch_call.finditer(line):
                    action_target = m_disp.group(1)
                    edges.append(
                        ParsedEdge(
                            source_qualified_name=active_comp_qual,
                            target_qualified_name=action_target,
                            edge_kind="DISPATCHES",
                        )
                    )
                for m_sel_call in use_selector_call.finditer(line):
                    sel_target = m_sel_call.group(1)
                    edges.append(
                        ParsedEdge(
                            source_qualified_name=active_comp_qual,
                            target_qualified_name=sel_target,
                            edge_kind="SELECTS",
                        )
                    )
                for m_hk in use_hook_call.finditer(line):
                    hk_name = m_hk.group(1)
                    ignored_hooks = (
                        "useState",
                        "useEffect",
                        "useCallback",
                        "useMemo",
                        "useRef",
                        "useAppDispatch",
                        "useAppSelector",
                    )
                    # Exclude ignored hooks AND exclude current hook definition to avoid
                    # self-referencing CALLS
                    if hk_name not in ignored_hooks and hk_name != active_comp_name:
                        edges.append(
                            ParsedEdge(
                                source_qualified_name=active_comp_qual,
                                target_qualified_name=hk_name,
                                edge_kind="CALLS",
                            )
                        )
            else:
                for m_disp in use_dispatch_call.finditer(line):
                    edges.append(
                        ParsedEdge(
                            source_qualified_name=f"{module_prefix}.standalone",
                            target_qualified_name=m_disp.group(1),
                            edge_kind="DISPATCHES",
                        )
                    )

            # Update brace depth
            opens = line_clean.count("{")
            closes = line_clean.count("}")

            if pending_comp and opens > 0:
                entry_depth = brace_depth + 1
                comp_stack.append((pending_comp[0], pending_comp[1], entry_depth))
                pending_comp = None

            if pending_slice_brace and opens > 0:
                slice_brace_depth = brace_depth + 1
                pending_slice_brace = False
            if pending_reducers_brace and opens > 0:
                reducers_brace_depth = brace_depth + 1
                pending_reducers_brace = False

            brace_depth += opens - closes

            while comp_stack and brace_depth < comp_stack[-1][2]:
                comp_stack.pop()

            if (
                in_reducers_block
                and reducers_brace_depth is not None
                and brace_depth < reducers_brace_depth
            ):
                in_reducers_block = False
                reducers_brace_depth = None

            if (
                current_slice
                and slice_brace_depth is not None
                and brace_depth < slice_brace_depth
            ):
                current_slice = None
                slice_brace_depth = None
                in_reducers_block = False
                reducers_brace_depth = None

            i += 1

        return symbols, edges

    def extract_manifest_dependencies(self, file_path: str, content: str) -> list[DependencyEdge]:
        edges: list[DependencyEdge] = []
        try:
            data = json.loads(content)
            for pkg, ver in data.get("dependencies", {}).items():
                edges.append(
                    DependencyEdge(
                        source_identifier=data.get("name", Path(file_path).stem),
                        target_package=pkg,
                        version_constraint=str(ver),
                        scope="runtime",
                        manifest_path=file_path,
                    )
                )
            for pkg, ver in data.get("devDependencies", {}).items():
                edges.append(
                    DependencyEdge(
                        source_identifier=data.get("name", Path(file_path).stem),
                        target_package=pkg,
                        version_constraint=str(ver),
                        scope="dev",
                        manifest_path=file_path,
                    )
                )
        except Exception:
            pass
        return edges

    def _derive_module_prefix(self, file_path: str) -> str:
        clean = file_path.replace("\\", "/").lstrip("/")
        if clean.startswith("src/"):
            clean = clean[4:]
        for ext in (".tsx", ".ts", ".jsx", ".js"):
            if clean.endswith(ext):
                clean = clean[:-len(ext)]
                break
        return clean.replace("/", ".")
