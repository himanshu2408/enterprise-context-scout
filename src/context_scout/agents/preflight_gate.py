"""Deterministic 4-Check Pre-Flight Quality Gate ($0 Local Bounce-Back Barrier).

Enforces four deterministic gates before calling paid Gemini execution models:
- Check 1: Internal symbol resolution via AST indexing & 1-hop ASTDependencyExpander closure.
- Check 2: Prompt identifier partitioning (VAL-01) distinguishing reference vs novel symbols.
- Check 3: Normalized RRF threshold (BLK-01, >=0.55 or cross-encoder score >=0.60).
- Check 4: Verbatim citation verification (VAL-02, >=0.98 token overlap with NFKC normalization).
"""

from __future__ import annotations

import keyword
import logging
import re
from typing import Any, Final
from uuid import UUID

from psycopg import Connection

from context_scout.ast_engine.dependency_expander import ASTDependencyExpander
from context_scout.ast_engine.deterministic_slicer import DeterministicSlicer
from context_scout.db.repositories import (
    ASTEdgeRecord,
    ASTSymbolRecord,
    CodeFileRecord,
    DocChunkRecord,
)
from context_scout.schemas.contracts import (
    CodeSlicePointer,
    DocCitationSpan,
    ImplementationContextPack,
    InsufficientContextDiagnostic,
    MarketRegime,
    ModificationIntent,
    PreFlightValidationReport,
    PrincipalContext,
    QuantStrategyBrief,
    RoleType,
    VerbatimCodeSlice,
)

logger = logging.getLogger(__name__)

# Standard language keywords and prompt instruction stopwords to prevent false-positive identifiers
PYTHON_KEYWORDS: Final[set[str]] = set(keyword.kwlist)

TYPESCRIPT_KEYWORDS: Final[set[str]] = {
    "abstract", "any", "as", "asserts", "async", "await", "boolean", "break",
    "case", "catch", "class", "const", "constructor", "continue", "debugger",
    "declare", "default", "delete", "do", "else", "enum", "export", "extends",
    "false", "finally", "for", "from", "function", "get", "if", "implements",
    "import", "in", "infer", "instanceof", "interface", "is", "keyof", "let",
    "module", "namespace", "never", "new", "null", "number", "object", "of",
    "package", "private", "protected", "public", "readonly", "require", "return",
    "set", "static", "string", "super", "switch", "symbol", "this", "throw",
    "true", "try", "type", "typeof", "undefined", "unique", "unknown", "var",
    "void", "while", "with", "yield",
}

PROMPT_STOPWORDS: Final[set[str]] = {
    "implement", "implementation", "implementing", "create", "creating", "created",
    "refactor", "refactoring", "refactored", "add", "adding", "added", "modify",
    "modifying", "modified", "update", "updating", "updated", "fix", "fixing",
    "fixed", "build", "building", "built", "remove", "removing", "removed",
    "delete", "deleting", "deleted", "change", "changing", "changed", "ensure",
    "ensuring", "ensured", "verify", "verifying", "verified", "test", "tests",
    "testing", "tested", "function", "functions", "method", "methods", "class",
    "classes", "module", "modules", "package", "packages", "file", "files",
    "code", "slice", "slices", "symbol", "symbols", "target", "targets",
    "reference", "references", "existing", "novel", "pack", "packs", "scout",
    "context", "check", "checks", "checking", "checked", "error", "errors",
    "using", "used", "use", "from", "with", "into", "onto", "about", "between",
    "before", "after", "under", "above", "please", "should", "would", "could",
    "must", "will", "can", "then", "when", "where", "which", "that", "this",
    "these", "those", "have", "has", "had", "having", "does", "did", "doing",
    "make", "made", "making", "need", "needed", "want", "wanted", "write",
    "writing", "written", "read", "reading", "run", "running", "also", "only",
    "just", "both", "either", "neither", "all", "any", "some", "none", "each",
    "every", "more", "most", "other", "another", "such", "than", "like", "same",
    "different", "new", "now", "here", "there", "well", "call", "calls", "called",
    "calling", "help", "show", "give", "provide", "provided", "including", "include",
}


class BoundGraphRepository:
    """Adapts principal-based or connection-based repositories to ScoutGraphRepository."""

    def __init__(
        self,
        repo: Any,
        conn: Connection[tuple[object, ...]] | None = None,
    ) -> None:
        self._repo = repo
        self._conn = conn

    def list_visible_code_files(
        self, principal: PrincipalContext, domain_id: str
    ) -> list[CodeFileRecord]:
        if hasattr(self._repo, "list_visible_code_files"):
            try:
                result = self._repo.list_visible_code_files(principal, domain_id)
                if isinstance(result, list):
                    return result
            except TypeError:
                if self._conn is not None:
                    result = self._repo.list_visible_code_files(self._conn, domain_id)
                    if isinstance(result, list):
                        return result
        return []

    def list_visible_ast_symbols(
        self, principal: PrincipalContext, domain_id: str
    ) -> list[ASTSymbolRecord]:
        if hasattr(self._repo, "list_visible_ast_symbols"):
            try:
                result = self._repo.list_visible_ast_symbols(principal, domain_id)
                if isinstance(result, list):
                    return result
            except TypeError:
                if self._conn is not None:
                    result = self._repo.list_visible_ast_symbols(self._conn, domain_id)
                    if isinstance(result, list):
                        return result
        return []

    def list_visible_ast_edges(
        self, principal: PrincipalContext, domain_id: str
    ) -> list[ASTEdgeRecord]:
        if hasattr(self._repo, "list_visible_ast_edges"):
            try:
                result = self._repo.list_visible_ast_edges(principal, domain_id)
                if isinstance(result, list):
                    return result
            except TypeError:
                if self._conn is not None:
                    result = self._repo.list_visible_ast_edges(self._conn, domain_id)
                    if isinstance(result, list):
                        return result
        return []

    def list_visible_doc_chunks(
        self,
        principal: PrincipalContext,
        domain_id: str,
        market_scope: MarketRegime | None = None,
    ) -> list[DocChunkRecord]:
        if hasattr(self._repo, "list_visible_doc_chunks"):
            try:
                result = self._repo.list_visible_doc_chunks(principal, domain_id, market_scope)
                if isinstance(result, list):
                    return result
            except TypeError:
                if self._conn is not None:
                    result = self._repo.list_visible_doc_chunks(
                        self._conn, domain_id, market_scope
                    )
                    if isinstance(result, list):
                        return result
        return []


def create_insufficient_context_diagnostic(
    trace_id: str,
    domain_id: str,
    report: PreFlightValidationReport,
    partial_candidate_files: list[str] | None = None,
    custom_reason: str | None = None,
) -> InsufficientContextDiagnostic:
    """Terminal exit payload returned when local bounce-back still fails Pre-Flight."""
    missing: list[str] = []
    for sym in report.unresolved_symbols:
        if sym not in missing:
            missing.append(sym)
    for sym in report.missing_reference_identifiers:
        if sym not in missing:
            missing.append(sym)

    reason = (
        custom_reason
        or report.bounce_back_feedback
        or "Deterministic pre-flight quality check failed"
    )
    files = partial_candidate_files or []

    return InsufficientContextDiagnostic(
        trace_id=trace_id,
        domain_id=domain_id,
        reason=reason,
        missing_symbols=missing,
        partial_candidate_files=files,
        paid_tokens_spent=0,
    )


class PreFlightValidator:
    """4-Check Pre-Flight Quality Gate ($0 local quality barrier).

    Enforces deterministic validation of symbol closure, prompt identifier coverage,
    normalized retrieval score, and verbatim document citations before paid frontier handoff.
    """

    DEFAULT_MIN_RRF_SCORE: Final[float] = 0.55
    DEFAULT_CROSS_ENCODER_SCORE_THRESHOLD: Final[float] = 0.60
    DEFAULT_MIN_CITATION_OVERLAP: Final[float] = 0.98

    def __init__(
        self,
        repo: Any = None,
        slicer: DeterministicSlicer | None = None,
        expander: ASTDependencyExpander | None = None,
        min_rrf_score: float = DEFAULT_MIN_RRF_SCORE,
        cross_encoder_threshold: float = DEFAULT_CROSS_ENCODER_SCORE_THRESHOLD,
        min_citation_overlap: float = DEFAULT_MIN_CITATION_OVERLAP,
        principal: PrincipalContext | None = None,
        domain_id: str | None = None,
    ) -> None:
        self.repo = repo
        self.slicer = slicer or DeterministicSlicer()
        self.expander = expander or ASTDependencyExpander(
            max_direct_callers=3,
            max_expansion_token_budget=1500,
        )
        self.min_rrf_score = min_rrf_score
        self.cross_encoder_threshold = cross_encoder_threshold
        self.min_citation_overlap = min_citation_overlap
        self.principal = principal
        self.domain_id = domain_id

    def _resolve_graph_repo(
        self,
        repo: Any,
        conn: Connection[tuple[object, ...]] | None = None,
    ) -> BoundGraphRepository:
        active_repo = repo if repo is not None else self.repo
        return BoundGraphRepository(active_repo, conn=conn)

    def _resolve_principal(
        self,
        principal: PrincipalContext | None,
    ) -> PrincipalContext:
        if principal is not None:
            return principal
        if self.principal is not None:
            return self.principal
        return PrincipalContext(
            user_id="preflight_system",
            role=RoleType.ENGINEER,
            allowed_scopes=["public_eng", "repo:trading_strategies", "corpus:quant_papers"],
        )

    def extract_prompt_candidates(self, user_prompt: str) -> tuple[set[str], set[str]]:
        """Extracts candidate tokens, separating explicitly quoted code tokens from unquoted text.

        Returns (explicitly_quoted_tokens, unquoted_candidate_tokens).
        """
        if not user_prompt:
            return set(), set()

        # 1. Backticked tokens (`symbol_or_path`)
        backtick_tokens = set(re.findall(r"`([^`]+)`", user_prompt))

        # 2. File paths (e.g., src/strategies/order_flow.py or models.py)
        file_path_pattern = (
            r"(?:[a-zA-Z0-9_\-\./]+/[a-zA-Z0-9_\-\.]+\.[a-zA-Z0-9]+|"
            r"[a-zA-Z0-9_\-\.]+\.(?:py|ts|tsx|js|jsx|json|md|sql|yaml|yml|toml))"
        )
        path_tokens = set(re.findall(file_path_pattern, user_prompt))

        explicit_tokens: set[str] = set()
        for tok in backtick_tokens | path_tokens:
            cleaned = tok.strip(" `'\";:(),")
            if cleaned and len(cleaned) >= 2:
                explicit_tokens.add(cleaned)

        # 3. Candidate identifiers from prompt body
        raw_words = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b", user_prompt)
        unquoted_candidates: set[str] = set()
        for word in raw_words:
            cleaned = word.strip()
            if not cleaned or len(cleaned) < 3:
                continue
            lower_val = cleaned.lower()
            if cleaned in PYTHON_KEYWORDS or lower_val in PYTHON_KEYWORDS:
                continue
            if cleaned in TYPESCRIPT_KEYWORDS or lower_val in TYPESCRIPT_KEYWORDS:
                continue
            if lower_val in PROMPT_STOPWORDS:
                continue
            if cleaned.isdigit():
                continue
            unquoted_candidates.add(cleaned)

        return explicit_tokens, unquoted_candidates

    def partition_prompt_identifiers(
        self,
        user_prompt: str,
        visible_symbols: list[ASTSymbolRecord],
        visible_files: list[CodeFileRecord],
        code_slices: list[VerbatimCodeSlice],
    ) -> tuple[list[str], list[str]]:
        """Partitions prompt candidates against repository AST index (VAL-01).

        Returns (missing_reference_identifiers, novel_target_identifiers).
        Novel target identifiers do NOT trigger validation failures.
        """
        explicit_candidates, unquoted_candidates = self.extract_prompt_candidates(user_prompt)
        all_candidates = explicit_candidates | unquoted_candidates

        if not all_candidates:
            return [], []

        # Build repository symbol and file lookup indices
        known_symbols_by_name: dict[str, ASTSymbolRecord] = {}
        known_symbols_by_short: dict[str, ASTSymbolRecord] = {}
        for sym in visible_symbols:
            known_symbols_by_name[sym.qualified_name] = sym
            short_name = sym.qualified_name.split(".")[-1]
            known_symbols_by_short[short_name] = sym

        known_files_by_path: dict[str, CodeFileRecord] = {}
        known_files_by_name: dict[str, CodeFileRecord] = {}
        for f in visible_files:
            known_files_by_path[f.file_path] = f
            file_name = f.file_path.split("/")[-1]
            known_files_by_name[file_name] = f

        missing_reference_identifiers: list[str] = []
        novel_target_identifiers: list[str] = []

        for candidate in sorted(all_candidates):
            # Check if candidate matches an existing symbol or file in repository index
            is_existing_symbol = (
                candidate in known_symbols_by_name or candidate in known_symbols_by_short
            )
            is_existing_file = (
                candidate in known_files_by_path
                or candidate in known_files_by_name
                or any(f.endswith("/" + candidate) for f in known_files_by_path)
            )

            if is_existing_symbol or is_existing_file:
                # Existing Reference Symbol/File -> Must be covered in code_slices
                covered = False
                for s in code_slices:
                    if s.qualified_symbol == candidate or s.qualified_symbol.endswith(
                        "." + candidate
                    ):
                        covered = True
                        break
                    if s.file_path == candidate or s.file_path.endswith("/" + candidate):
                        covered = True
                        break
                    # Symbol span coverage
                    sym_match = known_symbols_by_name.get(
                        candidate
                    ) or known_symbols_by_short.get(candidate)
                    if sym_match is not None:
                        f_match = known_files_by_path.get(s.file_path)
                        if (
                            f_match
                            and f_match.file_id == sym_match.file_id
                            and s.start_line <= sym_match.start_line
                            and s.end_line >= sym_match.end_line
                        ):
                            covered = True
                            break
                    # Verbatim word boundary match in slice lines
                    if re.search(r"\b" + re.escape(candidate) + r"\b", s.verbatim_code_with_lines):
                        covered = True
                        break

                if not covered:
                    missing_reference_identifiers.append(candidate)
            else:
                # Novel identifier (to be created or not in repo) -> never causes failure
                novel_target_identifiers.append(candidate)

        return missing_reference_identifiers, novel_target_identifiers

    def check_symbol_resolution(
        self,
        code_slices: list[VerbatimCodeSlice],
        bound_repo: BoundGraphRepository,
        principal: PrincipalContext,
        domain_id: str,
    ) -> tuple[list[str], int]:
        """Verifies 100% internal symbol resolution via AST indexing and 1-hop closure (Check 1).

        Returns (unresolved_symbols, auto_attached_count).
        """
        target_slices = [
            s
            for s in code_slices
            if s.modification_intent == ModificationIntent.MODIFY_TARGET
            or str(s.modification_intent) == "modify_target"
        ]

        if not target_slices:
            return [], 0

        visible_symbols = bound_repo.list_visible_ast_symbols(principal, domain_id)
        visible_files = bound_repo.list_visible_code_files(principal, domain_id)
        visible_edges = bound_repo.list_visible_ast_edges(principal, domain_id)

        visible_symbols_by_id: dict[UUID, ASTSymbolRecord] = {
            s.symbol_id: s for s in visible_symbols
        }
        visible_files_by_id: dict[UUID, CodeFileRecord] = {f.file_id: f for f in visible_files}
        visible_files_by_path: dict[str, CodeFileRecord] = {f.file_path: f for f in visible_files}

        unresolved_symbols: list[str] = []
        auto_attached_count = 0

        # Maintain lookup of covered symbol names and line ranges
        covered_symbols: set[str] = {slice_rec.qualified_symbol for slice_rec in code_slices}
        for slice_rec in code_slices:
            covered_symbols.add(slice_rec.qualified_symbol.split(".")[-1])

        for target_slice in target_slices:
            # Look up focus symbol in repository AST symbols
            focus_sym: ASTSymbolRecord | None = None
            for sym_rec in visible_symbols:
                if sym_rec.qualified_name == target_slice.qualified_symbol:
                    focus_sym = sym_rec
                    break

            if focus_sym is None:
                # Fallback: short name + file path match
                target_short = target_slice.qualified_symbol.split(".")[-1]
                for sym_rec in visible_symbols:
                    if (
                        sym_rec.qualified_name.split(".")[-1] == target_short
                        and sym_rec.file_id in visible_files_by_id
                        and visible_files_by_id[sym_rec.file_id].file_path == target_slice.file_path
                    ):
                        focus_sym = sym_rec
                        break

            if focus_sym is None:
                unresolved_symbols.append(target_slice.qualified_symbol)
                continue

            # Auto-Closure: expand bounded 1-hop AST dependencies under SEC-03 budget
            expansion_result = self.expander.expand_1hop(
                focus_symbol=focus_sym,
                repo=bound_repo,
                principal=principal,
                domain_id=domain_id,
            )

            # Hydrate and attach missing 1-hop candidates (types, callers, fixtures)
            candidates_to_attach: list[CodeSlicePointer] = (
                expansion_result.imported_type_slices
                + expansion_result.caller_slices
                + expansion_result.test_fixture_slices
            )

            for pointer in candidates_to_attach:
                already_in_pack = False
                for existing in code_slices:
                    if (
                        existing.file_path == pointer.file_path
                        and existing.qualified_symbol == pointer.qualified_symbol
                    ):
                        already_in_pack = True
                        break
                    if (
                        existing.file_path == pointer.file_path
                        and existing.start_line <= pointer.start_line
                        and existing.end_line >= pointer.end_line
                    ):
                        already_in_pack = True
                        break

                if not already_in_pack and pointer.file_path in visible_files_by_path:
                    file_rec = visible_files_by_path[pointer.file_path]
                    try:
                        hydrated_slice = self.slicer.slice_pointer(
                            pointer=pointer,
                            file_record=file_rec,
                            auto_expanded_by_ast=True,
                        )
                        code_slices.append(hydrated_slice)
                        covered_symbols.add(pointer.qualified_symbol)
                        covered_symbols.add(pointer.qualified_symbol.split(".")[-1])
                        auto_attached_count += 1
                    except Exception as e:
                        logger.warning(
                            "Failed to hydrate 1-hop slice pointer %s: %s",
                            pointer.qualified_symbol,
                            e,
                        )

            # Verify whether all internal dependency edges of focus_sym are covered
            target_edges = [
                e
                for e in visible_edges
                if e.source_symbol_id == focus_sym.symbol_id
                and e.edge_kind in ("USES_TYPE", "IMPORTS", "CALLS")
            ]

            for edge in target_edges:
                target_sym: ASTSymbolRecord | None = None
                if edge.target_symbol_id and edge.target_symbol_id in visible_symbols_by_id:
                    target_sym = visible_symbols_by_id[edge.target_symbol_id]
                else:
                    target_clean = edge.target_qualified_name.split(".")[-1]
                    for sym_rec in visible_symbols:
                        if (
                            sym_rec.qualified_name == edge.target_qualified_name
                            or sym_rec.qualified_name.split(".")[-1] == target_clean
                        ):
                            target_sym = sym_rec
                            break

                # If this is an internal repository symbol, verify it is present in code_slices
                if target_sym is not None:
                    sym_clean = target_sym.qualified_name.split(".")[-1]
                    edge_clean = edge.target_qualified_name.split(".")[-1]
                    is_covered = (
                        target_sym.qualified_name in covered_symbols
                        or sym_clean in covered_symbols
                        or edge.target_qualified_name in covered_symbols
                        or edge_clean in covered_symbols
                    )

                    if not is_covered:
                        # Check line span coverage in slices
                        target_file = visible_files_by_id.get(target_sym.file_id)
                        if target_file:
                            for slice_item in code_slices:
                                if slice_item.file_path == target_file.file_path:
                                    if (
                                        slice_item.start_line <= target_sym.start_line
                                        and slice_item.end_line >= target_sym.end_line
                                    ):
                                        is_covered = True
                                        break
                                    lines_text = slice_item.verbatim_code_with_lines
                                    has_fn = f"def {sym_clean}" in lines_text
                                    has_cls = f"class {sym_clean}" in lines_text
                                    if has_fn or has_cls:
                                        is_covered = True
                                        break

                    if not is_covered:
                        unresolved_symbols.append(edge.target_qualified_name)
                elif edge.target_symbol_id is not None:
                    # Edge pointed to an internal symbol ID that cannot be found
                    unresolved_symbols.append(edge.target_qualified_name)

        # Deduplicate while preserving order
        deduped_unresolved: list[str] = []
        for sym in unresolved_symbols:
            if sym not in deduped_unresolved:
                deduped_unresolved.append(sym)

        return deduped_unresolved, auto_attached_count

    def check_rrf_threshold(
        self,
        normalized_rrf_score: float | None,
        cross_encoder_score: float | None,
    ) -> tuple[bool, float]:
        """Verifies top Normalized RRF score >= 0.55 (BLK-01) or cross-encoder score >= 0.60."""
        if normalized_rrf_score is not None:
            score = round(min(max(normalized_rrf_score, 0.0), 1.0), 6)
            passed = (score >= self.min_rrf_score) or (
                cross_encoder_score is not None
                and cross_encoder_score >= self.cross_encoder_threshold
            )
            return passed, score

        if cross_encoder_score is not None:
            score = round(min(max(cross_encoder_score, 0.0), 1.0), 6)
            passed = score >= self.cross_encoder_threshold
            return passed, score

        # Pure code direct lookup where retrieval was not invoked
        return True, 1.0

    def check_citations(
        self,
        citations: list[DocCitationSpan],
        doc_chunks: dict[str, str] | list[DocChunkRecord] | None,
        bound_repo: BoundGraphRepository,
        principal: PrincipalContext,
        domain_id: str,
    ) -> tuple[bool, list[str]]:
        """Verifies 100% verbatim quote match (VAL-02, >=0.98 overlap with NFKC normalization)."""
        if not citations:
            return True, []

        # Build chunk lookup index
        chunk_content_map: dict[str, str] = {}
        if isinstance(doc_chunks, dict):
            for k, v in doc_chunks.items():
                chunk_content_map[str(k)] = str(v)
        elif isinstance(doc_chunks, list):
            for c in doc_chunks:
                chunk_content_map[str(c.chunk_id)] = c.verbatim_content
        else:
            visible_chunks = bound_repo.list_visible_doc_chunks(principal, domain_id)
            for c in visible_chunks:
                chunk_content_map[str(c.chunk_id)] = c.verbatim_content

        failed_reasons: list[str] = []
        for cit in citations:
            chunk_text = chunk_content_map.get(str(cit.chunk_id))
            if not chunk_text:
                q_snippet = cit.verbatim_quote[:40]
                failed_reasons.append(
                    f"Chunk ID '{cit.chunk_id}' for quote '{q_snippet}...' not found in corpus"
                )
                continue

            verification = self.slicer.verify_doc_citation(
                citation=cit,
                chunk_content=chunk_text,
                min_overlap_ratio=self.min_citation_overlap,
            )
            if not verification.is_valid:
                failed_reasons.append(
                    f"Quote from chunk '{cit.chunk_id}' failed verbatim verification: "
                    f"overlap {verification.token_overlap_ratio:.4f} "
                    f"< {self.min_citation_overlap:.2f} ({verification.error_reason})"
                )

        passed = len(failed_reasons) == 0
        return passed, failed_reasons

    def validate(
        self,
        pack: ImplementationContextPack | None = None,
        *,
        code_slices: list[VerbatimCodeSlice] | None = None,
        user_prompt: str | None = None,
        supporting_doc_citations: list[DocCitationSpan] | None = None,
        quant_strategy_spec: QuantStrategyBrief | None = None,
        normalized_rrf_score: float | None = None,
        cross_encoder_score: float | None = None,
        repo: Any = None,
        principal: PrincipalContext | None = None,
        domain_id: str | None = None,
        conn: Connection[tuple[object, ...]] | None = None,
        doc_chunks: dict[str, str] | list[DocChunkRecord] | None = None,
    ) -> PreFlightValidationReport:
        """Executes all 4 deterministic checks, returning a PreFlightValidationReport."""
        # Unpack parameters from pack if provided
        active_slices = (
            code_slices
            if code_slices is not None
            else (pack.code_slices if pack is not None else [])
        )
        active_prompt = (
            user_prompt
            if user_prompt is not None
            else (pack.user_task if pack is not None else "")
        )
        active_citations = (
            list(supporting_doc_citations)
            if supporting_doc_citations is not None
            else (list(pack.supporting_doc_citations) if pack is not None else [])
        )
        active_quant = (
            quant_strategy_spec
            if quant_strategy_spec is not None
            else (pack.quant_strategy_spec if pack is not None else None)
        )
        active_domain = (
            domain_id
            if domain_id is not None
            else (pack.domain_id if pack is not None else (self.domain_id or "trading_code_scout"))
        )
        active_principal = self._resolve_principal(principal)
        bound_repo = self._resolve_graph_repo(repo, conn=conn)

        # Incorporate verified citations from quant strategy specification if present
        all_citations: list[DocCitationSpan] = list(active_citations)
        if active_quant and active_quant.verified_citations:
            for c in active_quant.verified_citations:
                if c not in all_citations:
                    all_citations.append(c)

        # ---------------------------------------------------------------------
        # Check 1: Internal Symbol Resolution & 1-Hop AST Auto-Closure
        # ---------------------------------------------------------------------
        unresolved_symbols, auto_attached_count = self.check_symbol_resolution(
            code_slices=active_slices,
            bound_repo=bound_repo,
            principal=active_principal,
            domain_id=active_domain,
        )
        check_1_passed = len(unresolved_symbols) == 0

        # ---------------------------------------------------------------------
        # Check 2: Prompt Identifier Partitioning (VAL-01)
        # ---------------------------------------------------------------------
        visible_symbols = bound_repo.list_visible_ast_symbols(active_principal, active_domain)
        visible_files = bound_repo.list_visible_code_files(active_principal, active_domain)

        missing_reference_identifiers, novel_target_identifiers = (
            self.partition_prompt_identifiers(
                user_prompt=active_prompt,
                visible_symbols=visible_symbols,
                visible_files=visible_files,
                code_slices=active_slices,
            )
        )
        check_2_passed = len(missing_reference_identifiers) == 0

        # ---------------------------------------------------------------------
        # Check 3: Normalized RRF Threshold (BLK-01, >= 0.55)
        # ---------------------------------------------------------------------
        check_3_passed, final_rrf_score = self.check_rrf_threshold(
            normalized_rrf_score=normalized_rrf_score,
            cross_encoder_score=cross_encoder_score,
        )

        # ---------------------------------------------------------------------
        # Check 4: Verbatim Citation Verification (VAL-02, >= 0.98 overlap)
        # ---------------------------------------------------------------------
        citations_verified_100_pct, failed_citation_reasons = self.check_citations(
            citations=all_citations,
            doc_chunks=doc_chunks,
            bound_repo=bound_repo,
            principal=active_principal,
            domain_id=active_domain,
        )
        check_4_passed = citations_verified_100_pct

        # ---------------------------------------------------------------------
        # Composite Gate Evaluation & Bounce-Back Feedback Synthesis
        # ---------------------------------------------------------------------
        all_passed = check_1_passed and check_2_passed and check_3_passed and check_4_passed

        bounce_back_feedback: str | None = None
        if not all_passed:
            feedback_parts = [
                "Pre-Flight Gate Validation FAILED ($0 Local Bounce-Back Required):"
            ]
            if not check_1_passed:
                feedback_parts.append(
                    f"- Check 1 (Internal Symbol Resolution): Missing internal dependencies: "
                    f"{unresolved_symbols}. Please retrieve or auto-expand these symbols "
                    "before invoking paid execution."
                )
            if not check_2_passed:
                feedback_parts.append(
                    "- Check 2 (Prompt Identifier Partitioning, VAL-01): User prompt referenced "
                    "existing repository symbols/files that are missing from the context pack: "
                    f"{missing_reference_identifiers}. "
                    "Please attach code slices covering these existing items."
                )
            if not check_3_passed:
                feedback_parts.append(
                    f"- Check 3 (Normalized RRF Score, BLK-01): Top Normalized RRF score "
                    f"{final_rrf_score:.4f} is below threshold {self.min_rrf_score:.2f} "
                    f"(and cross-encoder score < {self.cross_encoder_threshold:.2f}). "
                    "Please broaden search queries or inspect retrieval ranking."
                )
            if not check_4_passed:
                reasons_str = "; ".join(failed_citation_reasons)
                feedback_parts.append(
                    "- Check 4 (Verbatim Citation Verification, VAL-02): Citations failed verbatim "
                    f"verification: {reasons_str}. "
                    "Ensure quotes match source documents verbatim with >=0.98 token overlap."
                )
            bounce_back_feedback = "\n".join(feedback_parts)

        report = PreFlightValidationReport(
            passed=all_passed,
            unresolved_symbols=unresolved_symbols,
            missing_reference_identifiers=missing_reference_identifiers,
            novel_target_identifiers=novel_target_identifiers,
            normalized_rrf_score=final_rrf_score,
            citations_verified_100_pct=citations_verified_100_pct,
            auto_attached_ast_slices_count=auto_attached_count,
            bounce_back_feedback=bounce_back_feedback,
        )

        # Mutate pack in-place if provided
        if pack is not None:
            pack.code_slices = active_slices
            pack.preflight_report = report

        return report

    def create_diagnostic(
        self,
        trace_id: str,
        domain_id: str,
        report: PreFlightValidationReport,
        partial_candidate_files: list[str] | None = None,
        custom_reason: str | None = None,
    ) -> InsufficientContextDiagnostic:
        """Convenience method producing InsufficientContextDiagnostic(paid_tokens_spent=0)."""
        return create_insufficient_context_diagnostic(
            trace_id=trace_id,
            domain_id=domain_id,
            report=report,
            partial_candidate_files=partial_candidate_files,
            custom_reason=custom_reason,
        )


# Alias for compatibility with architecture specification
ContextPackValidator = PreFlightValidator

__all__ = [
    "BoundGraphRepository",
    "ContextPackValidator",
    "PreFlightValidator",
    "create_insufficient_context_diagnostic",
]
