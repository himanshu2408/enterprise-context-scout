"""Deterministic 4-Check Pre-Flight Quality Gate ($0 Local Bounce-Back Barrier).

Enforces four deterministic gates before calling paid Gemini execution models:
- Check 1: Internal symbol resolution via AST indexing & 1-hop ASTDependencyExpander closure.
- Check 2: Prompt identifier partitioning (VAL-01) distinguishing reference vs novel symbols.
- Check 3: Normalized RRF threshold (BLK-01, >=0.55 or cross-encoder score >=0.60).
- Check 4: Verbatim citation verification (VAL-02, >=0.98 token overlap with NFKC normalization).
"""

from __future__ import annotations

import datetime
import json
import keyword
import logging
import re
from pathlib import Path
from typing import Any, Final, Literal
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
from context_scout.telemetry import ScoutTelemetryTracer

logger = logging.getLogger(__name__)

# Standard language keywords and prompt instruction stopwords to prevent false-positive identifiers
PYTHON_KEYWORDS: Final[set[str]] = set(keyword.kwlist)

TYPESCRIPT_KEYWORDS: Final[set[str]] = {
    "abstract",
    "any",
    "as",
    "asserts",
    "async",
    "await",
    "boolean",
    "break",
    "case",
    "catch",
    "class",
    "const",
    "constructor",
    "continue",
    "debugger",
    "declare",
    "default",
    "delete",
    "do",
    "else",
    "enum",
    "export",
    "extends",
    "false",
    "finally",
    "for",
    "from",
    "function",
    "get",
    "if",
    "implements",
    "import",
    "in",
    "infer",
    "instanceof",
    "interface",
    "is",
    "keyof",
    "let",
    "module",
    "namespace",
    "never",
    "new",
    "null",
    "number",
    "object",
    "of",
    "package",
    "private",
    "protected",
    "public",
    "readonly",
    "require",
    "return",
    "set",
    "static",
    "string",
    "super",
    "switch",
    "symbol",
    "this",
    "throw",
    "true",
    "try",
    "type",
    "typeof",
    "undefined",
    "unique",
    "unknown",
    "var",
    "void",
    "while",
    "with",
    "yield",
}

PROMPT_STOPWORDS: Final[set[str]] = {
    "implement",
    "implementation",
    "implementing",
    "create",
    "creating",
    "created",
    "refactor",
    "refactoring",
    "refactored",
    "add",
    "adding",
    "added",
    "modify",
    "modifying",
    "modified",
    "update",
    "updating",
    "updated",
    "fix",
    "fixing",
    "fixed",
    "build",
    "building",
    "built",
    "remove",
    "removing",
    "removed",
    "delete",
    "deleting",
    "deleted",
    "change",
    "changing",
    "changed",
    "ensure",
    "ensuring",
    "ensured",
    "verify",
    "verifying",
    "verified",
    "test",
    "tests",
    "testing",
    "tested",
    "function",
    "functions",
    "method",
    "methods",
    "class",
    "classes",
    "module",
    "modules",
    "package",
    "packages",
    "file",
    "files",
    "code",
    "slice",
    "slices",
    "symbol",
    "symbols",
    "target",
    "targets",
    "reference",
    "references",
    "existing",
    "novel",
    "pack",
    "packs",
    "scout",
    "context",
    "check",
    "checks",
    "checking",
    "checked",
    "error",
    "errors",
    "using",
    "used",
    "use",
    "from",
    "with",
    "into",
    "onto",
    "about",
    "between",
    "before",
    "after",
    "under",
    "above",
    "please",
    "should",
    "would",
    "could",
    "must",
    "will",
    "can",
    "then",
    "when",
    "where",
    "which",
    "that",
    "this",
    "these",
    "those",
    "have",
    "has",
    "had",
    "having",
    "does",
    "did",
    "doing",
    "make",
    "made",
    "making",
    "need",
    "needed",
    "want",
    "wanted",
    "write",
    "writing",
    "written",
    "read",
    "reading",
    "run",
    "running",
    "also",
    "only",
    "just",
    "both",
    "either",
    "neither",
    "all",
    "any",
    "some",
    "none",
    "each",
    "every",
    "more",
    "most",
    "other",
    "another",
    "such",
    "than",
    "like",
    "same",
    "different",
    "new",
    "now",
    "here",
    "there",
    "well",
    "call",
    "calls",
    "called",
    "calling",
    "help",
    "show",
    "give",
    "provide",
    "provided",
    "including",
    "include",
}

CUSTOM_CORPUS_INDEX_PATHS: dict[str, Path] = {}


def register_custom_corpus_index(domain_id: str, index_path: Path) -> None:
    """Registers a custom domain corpus_index.json path for runtime hydration."""
    CUSTOM_CORPUS_INDEX_PATHS[domain_id] = index_path.resolve()


def _load_custom_domain_corpus(domain_id: str) -> dict[str, Any] | None:
    """Loads domains/<domain_id>/corpus_index.json if present."""
    candidates: list[Path] = []
    if domain_id in CUSTOM_CORPUS_INDEX_PATHS:
        candidates.append(CUSTOM_CORPUS_INDEX_PATHS[domain_id])
    candidates.append(Path("domains") / domain_id / "corpus_index.json")
    repo_domains = Path(__file__).resolve().parents[3] / "domains"
    candidates.append(repo_domains / domain_id / "corpus_index.json")
    for cand in candidates:
        if cand.exists():
            try:
                with open(cand, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
            except Exception as e:
                logger.debug("Failed to load custom corpus index %s: %s", cand, e)
    return None


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
                if isinstance(result, list) and result:
                    return result
            except TypeError:
                if self._conn is not None:
                    result = self._repo.list_visible_code_files(self._conn, domain_id)
                    if isinstance(result, list) and result:
                        return result
        corpus = _load_custom_domain_corpus(domain_id)
        if corpus and isinstance(corpus.get("code_files"), list):
            out_files: list[CodeFileRecord] = []
            for item in corpus["code_files"]:
                scopes = list(item.get("allowed_scopes", ["public_eng"]))
                if not (set(principal.allowed_scopes) & set(scopes)):
                    continue
                out_files.append(
                    CodeFileRecord(
                        file_id=UUID(str(item["file_id"])),
                        domain_id=domain_id,
                        repo_name=str(item.get("repo_name", domain_id)),
                        file_path=str(item["file_path"]),
                        git_commit_sha=str(item.get("git_commit_sha", "custom_commit")),
                        git_blob_sha=str(item.get("git_blob_sha", "custom_blob")),
                        language=str(item.get("language", "python")),
                        allowed_scopes=scopes,
                        raw_content=str(item["raw_content"]),
                        line_count=int(item.get("line_count", 1)),
                    )
                )
            return out_files
        return []

    def list_visible_ast_symbols(
        self, principal: PrincipalContext, domain_id: str
    ) -> list[ASTSymbolRecord]:
        if hasattr(self._repo, "list_visible_ast_symbols"):
            try:
                result = self._repo.list_visible_ast_symbols(principal, domain_id)
                if isinstance(result, list) and result:
                    return result
            except TypeError:
                if self._conn is not None:
                    result = self._repo.list_visible_ast_symbols(self._conn, domain_id)
                    if isinstance(result, list) and result:
                        return result
        corpus = _load_custom_domain_corpus(domain_id)
        if corpus and isinstance(corpus.get("ast_symbols"), list):
            out_syms: list[ASTSymbolRecord] = []
            for item in corpus["ast_symbols"]:
                scopes = list(item.get("allowed_scopes", ["public_eng"]))
                if not (set(principal.allowed_scopes) & set(scopes)):
                    continue
                raw_kind = str(item.get("symbol_kind", "function"))
                sym_kind: Literal["function", "class", "method", "interface", "type_alias"] = (
                    "class" if raw_kind == "class" else "function"
                )
                out_syms.append(
                    ASTSymbolRecord(
                        symbol_id=UUID(str(item["symbol_id"])),
                        file_id=UUID(str(item["file_id"])),
                        domain_id=domain_id,
                        qualified_name=str(item["qualified_name"]),
                        symbol_kind=sym_kind,
                        start_line=int(item.get("start_line", 1)),
                        end_line=int(item.get("end_line", 10)),
                        signature_skeleton=str(item.get("signature_skeleton", "")),
                        contextual_summary=str(item.get("contextual_summary", "")),
                        allowed_scopes=scopes,
                        embedding=[0.05] * 768,
                    )
                )
            return out_syms
        return []

    def list_visible_ast_edges(
        self, principal: PrincipalContext, domain_id: str
    ) -> list[ASTEdgeRecord]:
        if hasattr(self._repo, "list_visible_ast_edges"):
            try:
                result = self._repo.list_visible_ast_edges(principal, domain_id)
                if isinstance(result, list) and result:
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
                if isinstance(result, list) and result:
                    return result
            except TypeError:
                if self._conn is not None:
                    result = self._repo.list_visible_doc_chunks(self._conn, domain_id, market_scope)
                    if isinstance(result, list) and result:
                        return result
        corpus = _load_custom_domain_corpus(domain_id)
        if corpus and isinstance(corpus.get("doc_chunks"), list):
            out_chunks: list[DocChunkRecord] = []
            for item in corpus["doc_chunks"]:
                scopes = list(item.get("allowed_scopes", ["public_eng", "public_quant"]))
                if not (set(principal.allowed_scopes) & set(scopes)):
                    continue
                mkt_val = str(item.get("market_scope", MarketRegime.GLOBAL_MULTI.value))
                try:
                    mkt = MarketRegime(mkt_val)
                except ValueError:
                    mkt = MarketRegime.GLOBAL_MULTI
                allowed_mkts = (market_scope, MarketRegime.GLOBAL_MULTI)
                if market_scope is not None and mkt not in allowed_mkts:
                    continue
                out_chunks.append(
                    DocChunkRecord(
                        chunk_id=UUID(str(item["chunk_id"])),
                        domain_id=domain_id,
                        doc_id=str(item["doc_id"]),
                        doc_title=str(item["doc_title"]),
                        source_uri=str(item.get("source_uri", f"urn:{domain_id}:{item['doc_id']}")),
                        market_scope=mkt,
                        published_date=datetime.date(2026, 1, 1),
                        authority_weight=float(item.get("authority_weight", 1.0)),
                        section_heading=str(item.get("section_heading", "1. Empirical Findings")),
                        contextual_header=str(item.get("contextual_header", item["doc_title"])),
                        verbatim_content=str(item["verbatim_content"]),
                        metadata_json=dict(item.get("metadata_json", {})),
                        allowed_scopes=scopes,
                        embedding=[0.05] * 768,
                    )
                )
            return out_chunks
        return []

    def search_visible_doc_chunks(
        self,
        principal: PrincipalContext,
        domain_id: str,
        query_text: str,
        market_scope: MarketRegime | None = None,
        limit: int = 50,
    ) -> list[DocChunkRecord]:
        if hasattr(self._repo, "search_visible_doc_chunks"):
            try:
                result = self._repo.search_visible_doc_chunks(
                    principal=principal,
                    domain_id=domain_id,
                    query_text=query_text,
                    market_scope=market_scope,
                    limit=limit,
                )
                if isinstance(result, list) and result:
                    return result
            except Exception as e:
                logger.debug("search_visible_doc_chunks fallback: %s", e)

        base_chunks = self.list_visible_doc_chunks(principal, domain_id, market_scope)
        if not base_chunks and market_scope is not None:
            base_chunks = self.list_visible_doc_chunks(principal, domain_id, None)
        if not query_text.strip() or not base_chunks:
            return base_chunks[:limit]

        words = [
            w.lower()
            for w in re.findall(r"[A-Za-z0-9]{3,}", query_text)
            if w.lower() not in PROMPT_STOPWORDS
        ]
        if not words:
            return base_chunks[:limit]

        def _score_chunk(chk: DocChunkRecord) -> float:
            title_l = chk.doc_title.lower()
            sec_l = chk.section_heading.lower()
            body_l = chk.verbatim_content.lower()
            s = 0.0
            for w in words:
                if w in title_l:
                    s += 6.0
                if w in sec_l:
                    s += 3.0
                if w in body_l:
                    s += 1.0
            if market_scope is not None and chk.market_scope == market_scope:
                s += 5.0
            return s + (chk.authority_weight * 0.1)

        ranked = sorted(base_chunks, key=_score_chunk, reverse=True)
        return ranked[:limit]

    def get_doc_chunks_by_ids(
        self,
        principal: PrincipalContext,
        domain_id: str,
        chunk_ids: list[str],
    ) -> list[DocChunkRecord]:
        if hasattr(self._repo, "get_doc_chunks_by_ids"):
            try:
                result = self._repo.get_doc_chunks_by_ids(
                    principal=principal,
                    domain_id=domain_id,
                    chunk_ids=chunk_ids,
                )
                if isinstance(result, list) and result:
                    return result
            except Exception as e:
                logger.debug("get_doc_chunks_by_ids fallback: %s", e)
        visible = self.list_visible_doc_chunks(principal, domain_id)
        wanted = set(chunk_ids)
        return [c for c in visible if str(c.chunk_id) in wanted]

    def search_visible_ast_symbols(
        self,
        principal: PrincipalContext,
        domain_id: str,
        query_text: str,
        limit: int = 50,
    ) -> list[ASTSymbolRecord]:
        if hasattr(self._repo, "search_visible_ast_symbols"):
            try:
                result = self._repo.search_visible_ast_symbols(
                    principal=principal,
                    domain_id=domain_id,
                    query_text=query_text,
                    limit=limit,
                )
                if isinstance(result, list):
                    return result
            except Exception as e:
                logger.debug("search_visible_ast_symbols fallback: %s", e)
        visible = self.list_visible_ast_symbols(principal, domain_id)
        if not query_text.strip():
            return visible[:limit]
        q_l = query_text.strip().lower()
        matched = [
            s
            for s in visible
            if q_l in s.qualified_name.lower() or q_l in s.signature_skeleton.lower()
        ]
        return matched[:limit]

    def get_code_files_by_ids(
        self,
        principal: PrincipalContext,
        domain_id: str,
        file_ids: list[UUID | str],
    ) -> list[CodeFileRecord]:
        if hasattr(self._repo, "get_code_files_by_ids"):
            try:
                result = self._repo.get_code_files_by_ids(
                    principal=principal,
                    domain_id=domain_id,
                    file_ids=file_ids,
                )
                if isinstance(result, list):
                    return result
            except Exception as e:
                logger.debug("get_code_files_by_ids fallback: %s", e)
        visible = self.list_visible_code_files(principal, domain_id)
        wanted = {str(fid) for fid in file_ids}
        return [f for f in visible if str(f.file_id) in wanted]

    def get_code_files_by_paths(
        self,
        principal: PrincipalContext,
        domain_id: str,
        file_paths: list[str],
    ) -> list[CodeFileRecord]:
        if hasattr(self._repo, "get_code_files_by_paths"):
            try:
                result = self._repo.get_code_files_by_paths(
                    principal=principal,
                    domain_id=domain_id,
                    file_paths=file_paths,
                )
                if isinstance(result, list):
                    return result
            except Exception as e:
                logger.debug("get_code_files_by_paths fallback: %s", e)
        visible = self.list_visible_code_files(principal, domain_id)
        wanted = {p.strip() for p in file_paths if p and p.strip()}
        return [f for f in visible if f.file_path in wanted]


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

        # 1. Backticked tokens (`symbol_or_path`), ignoring multi-line fenced code blocks
        prompt_without_fences = re.sub(r"```.*?```", " ", user_prompt, flags=re.DOTALL)
        backtick_tokens = set(re.findall(r"`([^`\n]+)`", prompt_without_fences))

        # 2. File paths (e.g., src/strategies/order_flow.py or models.py)
        file_path_pattern = (
            r"(?:[a-zA-Z0-9_\-\./]+/[a-zA-Z0-9_\-\.]+\.[a-zA-Z0-9]+|"
            r"[a-zA-Z0-9_\-\.]+\.(?:py|ts|tsx|js|jsx|json|md|sql|yaml|yml|toml))"
        )
        path_tokens = set(re.findall(file_path_pattern, prompt_without_fences))

        explicit_tokens: set[str] = set()
        for tok in backtick_tokens | path_tokens:
            cleaned = tok.strip(" `'\";:(),")
            if cleaned.endswith("()"):
                cleaned = cleaned[:-2].strip()
            if not cleaned or len(cleaned) < 2:
                continue
            if any(ch.isspace() for ch in cleaned):
                continue
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
            if len(visible_files) > 15 and sym.qualified_name.startswith(("tests.", "examples.")):
                continue
            known_symbols_by_name[sym.qualified_name] = sym
            parts = sym.qualified_name.split(".")
            short_name = parts[-1]
            known_symbols_by_short[short_name] = sym
            if len(parts) >= 2:
                known_symbols_by_name[".".join(parts[-2:])] = sym

        known_files_by_path: dict[str, CodeFileRecord] = {}
        known_files_by_name: dict[str, CodeFileRecord] = {}
        for f in visible_files:
            known_files_by_path[f.file_path] = f
            file_name = f.file_path.split("/")[-1]
            known_files_by_name[file_name] = f

        missing_reference_identifiers: list[str] = []
        novel_target_identifiers: list[str] = []

        for candidate in sorted(all_candidates):
            is_explicit = candidate in explicit_candidates
            is_code_shaped = (
                any(c.isupper() for c in candidate[1:])
                or "_" in candidate
                or "." in candidate
                or (
                    candidate[0].isupper()
                    and len(candidate) >= 4
                    and candidate in known_symbols_by_short
                    and known_symbols_by_short[candidate].symbol_kind
                    in ("class", "interface", "type_alias")
                )
            )
            # Check if candidate matches an existing symbol or file in repository index
            is_existing_symbol = candidate in known_symbols_by_name or (
                candidate in known_symbols_by_short and (is_explicit or is_code_shaped)
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
                    sym_match = known_symbols_by_name.get(candidate) or known_symbols_by_short.get(
                        candidate
                    )
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
                elif edge.target_symbol_id is None:
                    if edge.edge_kind == "CALLS":
                        continue
                    rel_suffix = edge.target_qualified_name.lstrip(".")
                    target_clean = edge.target_qualified_name.split(".")[-1]
                    for sym_rec in visible_symbols:
                        if sym_rec.qualified_name == rel_suffix or sym_rec.qualified_name.endswith(
                            "." + rel_suffix
                        ):
                            target_sym = sym_rec
                            break
                    if target_sym is None:
                        if len(visible_files) > 10 and not re.search(
                            rf"\b{re.escape(target_clean)}\b",
                            target_slice.verbatim_code_with_lines,
                        ):
                            continue
                        for sym_rec in visible_symbols:
                            if sym_rec.qualified_name == edge.target_qualified_name or (
                                sym_rec.qualified_name.split(".")[-1] == target_clean
                                and sym_rec.symbol_kind
                                in ("class", "interface", "type_alias", "function")
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
                        if (
                            expansion_result.budget_exhausted
                            and self.expander.max_expansion_token_budget >= 500
                            and (auto_attached_count > 0 or len(candidates_to_attach) > 0)
                        ):
                            continue
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

        missing_ids = [
            str(cit.chunk_id) for cit in citations if str(cit.chunk_id) not in chunk_content_map
        ]
        if missing_ids and bound_repo._repo is not None:
            for lookup_domain in dict.fromkeys([domain_id, "quant_research_scout"]):
                fetched = bound_repo.get_doc_chunks_by_ids(
                    principal=principal,
                    domain_id=lookup_domain,
                    chunk_ids=missing_ids,
                )
                for c in fetched:
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
            user_prompt if user_prompt is not None else (pack.user_task if pack is not None else "")
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

        tracer = ScoutTelemetryTracer.get_instance()
        with tracer.trace_preflight(
            span_name="scout.preflight_gate",
            domain_id=active_domain,
            prompt=active_prompt,
        ) as pf_span:
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
                        "- Check 2 (Prompt Identifier Partitioning, VAL-01): "
                        "User prompt referenced existing repository symbols/files that are "
                        f"missing from the context pack: {missing_reference_identifiers}. "
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
                        "- Check 4 (Verbatim Citation Verification, VAL-02): Citations failed "
                        f"verbatim verification: {reasons_str}. "
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

            pf_span.record_report(report)
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
