"""Bounded 1-hop AST dependency closure engine enforcing SEC-03."""

import re
from collections import defaultdict
from typing import Final, Protocol
from uuid import UUID

from context_scout.ast_engine.models import ASTExpansionCandidate, ASTExpansionResult
from context_scout.db.repositories import (
    ASTEdgeRecord,
    ASTSymbolRecord,
    CodeFileRecord,
)
from context_scout.schemas.contracts import (
    CodeSlicePointer,
    ModificationIntent,
    PrincipalContext,
)


class ScoutGraphRepository(Protocol):
    """Protocol for graph repositories supporting visible symbol, file, and edge queries."""

    def list_visible_code_files(
        self, principal: PrincipalContext, domain_id: str
    ) -> list[CodeFileRecord]: ...

    def list_visible_ast_symbols(
        self, principal: PrincipalContext, domain_id: str
    ) -> list[ASTSymbolRecord]: ...

    def list_visible_ast_edges(
        self, principal: PrincipalContext, domain_id: str
    ) -> list[ASTEdgeRecord]: ...


class ASTDependencyExpander:
    """Bounded 1-hop AST dependency closure engine enforcing SEC-03."""

    MAX_DIRECT_CALLERS_PER_SYMBOL: Final[int] = 3
    MAX_EXPANSION_TOKEN_BUDGET: Final[int] = 1500

    def __init__(
        self,
        max_direct_callers: int = MAX_DIRECT_CALLERS_PER_SYMBOL,
        max_expansion_token_budget: int = MAX_EXPANSION_TOKEN_BUDGET,
    ) -> None:
        self.max_direct_callers = max_direct_callers
        self.max_expansion_token_budget = max_expansion_token_budget

    def estimate_tokens(self, text: str) -> int:
        """Deterministic integer token estimation using subword-adjusted regex chunks."""
        if not text:
            return 0
        tokens = re.findall(r"\w+|[^\w\s]", text)
        bpe_estimate = sum(max(1, (len(t) + 3) // 4) for t in tokens)
        return max(1, bpe_estimate)

    def expand_1hop(
        self,
        focus_symbol: ASTSymbolRecord,
        repo: ScoutGraphRepository,
        principal: PrincipalContext,
        domain_id: str,
    ) -> ASTExpansionResult:
        """Performs bounded 1-hop AST closure under RLS and SEC-03 token budget."""
        visible_symbols = {
            s.symbol_id: s for s in repo.list_visible_ast_symbols(principal, domain_id)
        }
        visible_files = {f.file_id: f for f in repo.list_visible_code_files(principal, domain_id)}
        visible_edges = repo.list_visible_ast_edges(principal, domain_id)

        # 1. Collect Tier 1: Imported Types (USES_TYPE, IMPORTS)
        type_candidates: list[ASTExpansionCandidate] = []
        for edge in visible_edges:
            if edge.source_symbol_id == focus_symbol.symbol_id and edge.edge_kind in (
                "USES_TYPE",
                "IMPORTS",
            ):
                target_sym: ASTSymbolRecord | None = None
                if edge.target_symbol_id and edge.target_symbol_id in visible_symbols:
                    target_sym = visible_symbols[edge.target_symbol_id]
                else:
                    target_clean = edge.target_qualified_name.split(".")[-1]
                    for s in visible_symbols.values():
                        if (
                            s.qualified_name == edge.target_qualified_name
                            or s.qualified_name.split(".")[-1] == target_clean
                        ):
                            target_sym = s
                            break

                if target_sym and target_sym.file_id in visible_files:
                    f_rec = visible_files[target_sym.file_id]
                    s_lines = f_rec.raw_content.splitlines()[
                        target_sym.start_line - 1 : target_sym.end_line
                    ]
                    est = self.estimate_tokens("\n".join(s_lines))
                    type_candidates.append(
                        ASTExpansionCandidate(
                            symbol=target_sym,
                            edge_kind=edge.edge_kind,
                            intent=ModificationIntent.IMPORTED_TYPE,
                            start_line=target_sym.start_line,
                            end_line=target_sym.end_line,
                            estimated_tokens=est,
                            file_path=f_rec.file_path,
                            diversity_key=f_rec.file_path,
                        )
                    )

        # Deterministic sort for type candidates
        type_candidates.sort(key=lambda c: (c.file_path, c.start_line, c.symbol.qualified_name))

        # 2. Collect Tier 2: Direct Callers (CALLS incoming edges) with call-site diversity
        focus_short_name = focus_symbol.qualified_name.split(".")[-1]
        caller_edges = [
            e
            for e in visible_edges
            if e.edge_kind == "CALLS"
            and (
                e.target_symbol_id == focus_symbol.symbol_id
                or e.target_qualified_name == focus_symbol.qualified_name
                or e.target_qualified_name.split(".")[-1] == focus_short_name
            )
        ]

        raw_caller_candidates: list[ASTExpansionCandidate] = []
        for edge in caller_edges:
            caller_sym = visible_symbols.get(edge.source_symbol_id)
            if caller_sym and caller_sym.file_id in visible_files:
                f_rec = visible_files[caller_sym.file_id]
                call_line = self._find_call_site_line(
                    f_rec.raw_content, caller_sym, focus_symbol.qualified_name
                )
                start_l = max(caller_sym.start_line, call_line - 5)
                end_l = min(caller_sym.end_line, call_line + 5)
                snippet_lines = f_rec.raw_content.splitlines()[start_l - 1 : end_l]
                est = self.estimate_tokens("\n".join(snippet_lines))
                raw_caller_candidates.append(
                    ASTExpansionCandidate(
                        symbol=caller_sym,
                        edge_kind="CALLS",
                        intent=ModificationIntent.DIRECT_CALLER,
                        start_line=start_l,
                        end_line=end_l,
                        estimated_tokens=est,
                        file_path=f_rec.file_path,
                        diversity_key=f_rec.file_path,
                    )
                )

        focus_file_path = (
            visible_files[focus_symbol.file_id].file_path
            if focus_symbol.file_id in visible_files
            else ""
        )
        selected_callers = self._select_diverse_callers(
            raw_caller_candidates,
            limit=self.max_direct_callers,
            focus_file_path=focus_file_path,
        )

        # 3. Collect Tier 3: Associated Test Fixture (TESTED_BY edge or naming convention)
        test_candidate = self._resolve_test_fixture(
            focus_symbol=focus_symbol,
            visible_edges=visible_edges,
            visible_symbols=visible_symbols,
            visible_files=visible_files,
        )

        # 4. Budget Accumulation & Cutoff Logic (enforcing MAX_EXPANSION_TOKEN_BUDGET)
        accumulated_tokens = 0
        budget_exhausted = False
        dropped_count = 0

        type_slices: list[CodeSlicePointer] = []
        caller_slices: list[CodeSlicePointer] = []
        test_slices: list[CodeSlicePointer] = []

        all_candidates_ordered: list[tuple[ASTExpansionCandidate, list[CodeSlicePointer]]] = [
            (c, type_slices) for c in type_candidates
        ] + [(c, caller_slices) for c in selected_callers]
        if test_candidate:
            all_candidates_ordered.append((test_candidate, test_slices))

        for cand, target_list in all_candidates_ordered:
            if accumulated_tokens + cand.estimated_tokens <= self.max_expansion_token_budget:
                target_list.append(
                    CodeSlicePointer(
                        repo_name=visible_files[cand.symbol.file_id].repo_name,
                        file_path=cand.file_path,
                        start_line=cand.start_line,
                        end_line=cand.end_line,
                        qualified_symbol=cand.symbol.qualified_name,
                        selection_reason=(
                            f"1-hop {cand.edge_kind} dependency for {focus_symbol.qualified_name}"
                        ),
                        modification_intent=cand.intent,
                    )
                )
                accumulated_tokens += cand.estimated_tokens
            else:
                budget_exhausted = True
                dropped_count += 1

        return ASTExpansionResult(
            focus_symbol=focus_symbol,
            imported_type_slices=type_slices,
            caller_slices=caller_slices,
            test_fixture_slices=test_slices,
            total_expansion_tokens=accumulated_tokens,
            budget_exhausted=budget_exhausted,
            dropped_candidates_count=dropped_count,
        )

    def _find_call_site_line(
        self, raw_content: str, caller_sym: ASTSymbolRecord, target_qualified_name: str
    ) -> int:
        """Finds the line number within caller_sym where target is invoked."""
        target_name = target_qualified_name.split(".")[-1]
        lines = raw_content.splitlines()[caller_sym.start_line - 1 : caller_sym.end_line]
        pattern = re.compile(rf"\b{re.escape(target_name)}\b")
        for idx, line in enumerate(lines):
            if pattern.search(line):
                return caller_sym.start_line + idx
        return caller_sym.start_line

    def _select_diverse_callers(
        self,
        candidates: list[ASTExpansionCandidate],
        limit: int,
        focus_file_path: str,
    ) -> list[ASTExpansionCandidate]:
        """Ranks callers by call-site diversity across files using round-robin selection."""
        if len(candidates) <= limit:
            return sorted(
                candidates,
                key=lambda c: (
                    c.file_path == focus_file_path,
                    c.file_path,
                    c.start_line,
                ),
            )

        # Deduplicate multiple calls within the exact same function to maximize diversity
        deduped: dict[tuple[UUID, int], ASTExpansionCandidate] = {}
        for c in candidates:
            key = (c.symbol.symbol_id, c.start_line)
            if key not in deduped:
                deduped[key] = c
        candidates = list(deduped.values())

        by_file: dict[str, list[ASTExpansionCandidate]] = defaultdict(list)
        for c in candidates:
            by_file[c.file_path].append(c)

        for group in by_file.values():
            group.sort(key=lambda c: (c.start_line, c.symbol.qualified_name))

        # Order files: prefer distinct files from focus_file_path first
        ordered_files = sorted(
            by_file.keys(),
            key=lambda f: (f == focus_file_path, f),
        )

        selected: list[ASTExpansionCandidate] = []
        while len(selected) < limit and any(by_file.values()):
            for f in ordered_files:
                if by_file[f]:
                    selected.append(by_file[f].pop(0))
                    if len(selected) == limit:
                        break

        return selected

    def _resolve_test_fixture(
        self,
        focus_symbol: ASTSymbolRecord,
        visible_edges: list[ASTEdgeRecord],
        visible_symbols: dict[UUID, ASTSymbolRecord],
        visible_files: dict[UUID, CodeFileRecord],
    ) -> ASTExpansionCandidate | None:
        """Resolves matching unit test fixture via TESTED_BY edge or naming convention."""
        # 1. Check explicit TESTED_BY edge
        target_name = focus_symbol.qualified_name.split(".")[-1]
        for edge in visible_edges:
            if edge.edge_kind == "TESTED_BY":
                is_match = (
                    edge.source_symbol_id == focus_symbol.symbol_id
                    or edge.target_symbol_id == focus_symbol.symbol_id
                    or edge.target_qualified_name == focus_symbol.qualified_name
                    or edge.target_qualified_name == target_name
                )
                if is_match:
                    test_sym_id = (
                        edge.target_symbol_id
                        if edge.source_symbol_id == focus_symbol.symbol_id
                        else edge.source_symbol_id
                    )
                    t_sym = visible_symbols.get(test_sym_id) if test_sym_id else None
                    if not t_sym:
                        # Try finding by source_symbol_id
                        t_sym = visible_symbols.get(edge.source_symbol_id)

                    if t_sym and t_sym.file_id in visible_files:
                        f_rec = visible_files[t_sym.file_id]
                        s_lines = f_rec.raw_content.splitlines()[
                            t_sym.start_line - 1 : t_sym.end_line
                        ]
                        return ASTExpansionCandidate(
                            symbol=t_sym,
                            edge_kind="TESTED_BY",
                            intent=ModificationIntent.TEST_FIXTURE,
                            start_line=t_sym.start_line,
                            end_line=t_sym.end_line,
                            estimated_tokens=self.estimate_tokens("\n".join(s_lines)),
                            file_path=f_rec.file_path,
                            diversity_key=f_rec.file_path,
                        )

        # 2. Naming convention fallback
        if focus_symbol.file_id not in visible_files:
            return None
        f_rec = visible_files[focus_symbol.file_id]
        mod_name = f_rec.file_path.split("/")[-1]
        for ext in (".py", ".ts", ".tsx"):
            if mod_name.endswith(ext):
                mod_name = mod_name[: -len(ext)]
                break
        test_path_fragment = f"test_{mod_name}"

        for t_file in visible_files.values():
            if test_path_fragment in t_file.file_path:
                func_name = focus_symbol.qualified_name.split(".")[-1]
                target_test_name = f"test_{func_name}"
                for s in visible_symbols.values():
                    if s.file_id == t_file.file_id and target_test_name in s.qualified_name:
                        s_lines = t_file.raw_content.splitlines()[s.start_line - 1 : s.end_line]
                        return ASTExpansionCandidate(
                            symbol=s,
                            edge_kind="TESTED_BY",
                            intent=ModificationIntent.TEST_FIXTURE,
                            start_line=s.start_line,
                            end_line=s.end_line,
                            estimated_tokens=self.estimate_tokens("\n".join(s_lines)),
                            file_path=t_file.file_path,
                            diversity_key=t_file.file_path,
                        )

        return None


__all__ = [
    "ASTDependencyExpander",
    "ScoutGraphRepository",
]
