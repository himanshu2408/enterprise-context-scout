"""Deterministic 4-Check Pre-Flight Quality Gate ($0 Local Bounce-Back Barrier).

Enforces four deterministic gates before calling paid Gemini execution models:
- Check 1: Internal symbol resolution via AST indexing & 1-hop ASTDependencyExpander closure.
- Check 2: Prompt identifier partitioning (VAL-01) distinguishing reference vs novel symbols.
- Check 3: Normalized RRF threshold (BLK-01, >=0.65 or cross-encoder score >=0.60).
- Check 4: Verbatim citation verification (VAL-02, >=0.98 token overlap with NFKC normalization).
"""

from __future__ import annotations

import builtins
import datetime
import hashlib
import json
import keyword
import logging
import os
import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final
from uuid import NAMESPACE_URL, UUID, uuid5

from psycopg import Connection

from context_scout.agents.citation_gate import DocumentCitationGate
from context_scout.ast_engine.dependency_expander import ASTDependencyExpander
from context_scout.ast_engine.deterministic_slicer import DeterministicSlicer
from context_scout.ast_engine.extractor import match_symbol_identifier
from context_scout.config import ScoutSettings
from context_scout.db.repositories import (
    ASTEdgeRecord,
    ASTNeighborhoodRecord,
    ASTSymbolRecord,
    CalibratedRankedSearchHit,
    CodeFileRecord,
    DocChunkRecord,
    InMemoryRLSRepository,
    PostgresRLSRepository,
    RepositoryRecord,
)
from context_scout.db.session import RLSConnectionManager
from context_scout.ingestion.doc_parser import (
    DocumentChunk,
    chunk_custom_domain_document,
    extract_sgml_docbook_text,
    strip_mdx_boilerplate,
)

if TYPE_CHECKING:
    from context_scout.retrieval.embedder import LocalSemanticEmbedder
from context_scout.schemas.contracts import (
    AuthorityRules,
    CodeSlicePointer,
    DocCitationSpan,
    DomainConfig,
    DomainPackManifest,
    GateAbstentionResult,
    GatePassResult,
    ImplementationContextPack,
    InsufficientContextDiagnostic,
    MarketRegime,
    MissingItemRecord,
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
PYTHON_BUILTINS: Final[set[str]] = set(dir(builtins)) | {
    "WinError",
    "WindowsError",
}

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
    "inspect",
    "inspecting",
    "inspected",
    "locate",
    "locating",
    "located",
    "trace",
    "tracing",
    "traced",
    "explain",
    "explaining",
    "explained",
    "dependency",
    "dependencies",
    "caller",
    "callers",
    "signature",
    "signatures",
    "work",
    "works",
    "working",
    "ast",
    "hop",
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
    "sql",
    "kms",
    "postgresql",
    "postgres",
    "redis",
    "kafka",
    "mysql",
    "grpc",
    "http",
    "https",
    "rest",
    "api",
    "json",
    "yaml",
    "xml",
    "jwt",
    "oauth",
    "oidc",
    "iam",
    "gcp",
    "aws",
    "azure",
    "docker",
    "k8s",
    "kubernetes",
    "terraform",
    "linux",
    "cloud",
    "private",
    "cluster",
    "encryption",
    "quantstrategybrief",
    "domainknowledgebrief",
    "implementationcontextpack",
    "strategybrief",
    "brief",
}

CUSTOM_CORPUS_INDEX_PATHS: dict[str, Path] = {}
_FILE_STAT_BLOB_CACHE: dict[Path, tuple[int, int, str]] = {}
_DOMAIN_SYNC_EPOCHS: dict[str, int] = {}
_QUERY_EMBEDDER_LOCK = threading.Lock()
_SHARED_QUERY_EMBEDDER: Any = None


def _embed_query_for_hybrid_rrf(query_text: str) -> list[float]:
    """Computes a 768-dim query embedding using a cached LocalSemanticEmbedder."""
    global _SHARED_QUERY_EMBEDDER
    with _QUERY_EMBEDDER_LOCK:
        if _SHARED_QUERY_EMBEDDER is None:
            from context_scout.retrieval.embedder import LocalSemanticEmbedder

            _SHARED_QUERY_EMBEDDER = LocalSemanticEmbedder()
        embedder = _SHARED_QUERY_EMBEDDER
    vec: list[float] = embedder.embed_text(query_text)
    return vec


_DEFAULT_DUMMY_EMBEDDING: Final[list[float]] = [0.05] * 768
_SYNC_MANIFEST_FILENAME: Final[str] = "sync_manifest.json"
_PG_STORAGE_BACKEND: Final[str] = "postgresql_17_rls"

_PG_ALLOWED_SYMBOL_KINDS: Final[frozenset[str]] = frozenset(
    {
        "function",
        "class",
        "method",
        "interface",
        "type_alias",
        "hook",
        "slice",
        "selector",
        "reducer",
        "story",
        "tf_resource",
        "tf_module",
        "gcp_resource",
    }
)
_PG_ALLOWED_EDGE_KINDS: Final[frozenset[str]] = frozenset(
    {
        "CALLS",
        "IMPORTS",
        "USES_TYPE",
        "INHERITS",
        "TESTED_BY",
        "PROVISIONED_BY",
        "DISPATCHES",
        "SELECTS",
        "DOCUMENTS_COMPONENT",
    }
)
_PG_EDGE_REMAP: Final[dict[str, str]] = {
    "REGISTERS": "PROVISIONED_BY",
    "BINDS_SERVICE_ACCOUNT": "PROVISIONED_BY",
    "ACCESSES_SECRET": "PROVISIONED_BY",
}
_MARKET_REGIME_VALUES: Final[frozenset[str]] = frozenset(m.value for m in MarketRegime)
_PG_ALLOWED_MARKET_SCOPES: Final[frozenset[str]] = frozenset(
    {*_MARKET_REGIME_VALUES, "ENTERPRISE_DOC", "CONFLUENCE", "JIRA"}
)
_EFFECTIVE_EXTRA_EXTENSIONS: Final[dict[str, str]] = {".cs": "csharp"}

_SYNC_SKIP_DIRS: Final[set[str]] = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "dist",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "manifests",
}

CUSTOM_DOC_EXTENSIONS: Final[tuple[str, ...]] = (
    ".md",
    ".mdx",
    ".txt",
    ".tex",
    ".pdf",
    ".rst",
    ".adoc",
    ".sgml",
    ".json",
    ".csv",
    ".html",
    ".sql",
    ".tf",
    ".thrift",
)

_CUSTOM_DOC_EMBEDDER: LocalSemanticEmbedder | None = None


def _get_custom_doc_embedder() -> LocalSemanticEmbedder:
    global _CUSTOM_DOC_EMBEDDER
    if _CUSTOM_DOC_EMBEDDER is None:
        from context_scout.retrieval.embedder import LocalSemanticEmbedder

        _CUSTOM_DOC_EMBEDDER = LocalSemanticEmbedder(allow_deterministic_fallback=False)
    return _CUSTOM_DOC_EMBEDDER


def embed_custom_domain_texts(
    texts: list[str],
    batch_size: int = 64,
) -> list[list[float]]:
    """Embeds custom-domain chunks via LocalSemanticEmbedder (hash stub only if flag set)."""
    if not texts:
        return []
    if os.environ.get("SCOUT_USE_HASH_EMBEDDINGS_FOR_TESTS", "").strip() == "1":
        from context_scout.ingestion.git_merkle_indexer import generate_deterministic_embedding

        return [generate_deterministic_embedding(t) for t in texts]
    embedder = _get_custom_doc_embedder()
    return embedder.embed_batch(texts, is_query=False, batch_size=batch_size)


def build_custom_doc_chunk_records(
    *,
    domain_id: str,
    file_entries: list[tuple[Path, str, str, bytes, str]],
    batch_size: int = 64,
) -> list[DocChunkRecord]:
    """Chunks multi-format custom documents (400-800 tokens) and embeds them in batches.

    Each entry in `file_entries` is `(disk_path, rel_path, extracted_text, raw_bytes, blob_sha)`.
    """
    parsed_chunks: list[DocumentChunk] = []
    embed_inputs: list[str] = []

    for disk_p, rel_p, text, raw_bytes, blob_sha in file_entries:
        if not text.strip():
            continue
        chunks = chunk_custom_domain_document(
            file_path=disk_p,
            text=text,
            raw_bytes=raw_bytes,
            domain_id=domain_id,
            rel_path=rel_p,
            blob_sha=blob_sha,
        )
        for chk in chunks:
            parsed_chunks.append(chk)
            embed_inputs.append(
                f"{chk.doc_title}\n{chk.section_heading}\n{chk.verbatim_content[:1200]}"
            )

    if not parsed_chunks:
        return []

    vectors = embed_custom_domain_texts(embed_inputs, batch_size=batch_size)
    records: list[DocChunkRecord] = []
    default_scopes = [
        "public_eng",
        "public_quant",
        "quant_restricted",
        f"domain:{domain_id}",
    ]
    for chk, vec in zip(parsed_chunks, vectors, strict=True):
        records.append(
            DocChunkRecord(
                chunk_id=chk.chunk_id,
                domain_id=domain_id,
                doc_id=chk.doc_id,
                doc_title=chk.doc_title,
                source_uri=chk.source_uri,
                market_scope="ENTERPRISE_DOC",
                published_date=datetime.date(2026, 1, 1),
                authority_weight=1.0,
                section_heading=chk.section_heading,
                contextual_header=f"{chk.doc_title} — {chk.section_heading}",
                verbatim_content=chk.verbatim_content,
                metadata_json=dict(chk.metadata_json),
                allowed_scopes=list(default_scopes),
                embedding=vec,
            )
        )
    return records


def extract_document_text(file_path: Path) -> str:
    """Extracts UTF-8 text from supported markdown, SGML, PDF, code pattern, and text formats."""
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(file_path))
            pages = [page.extract_text() or "" for page in reader.pages]
            extracted = "\n\n".join(p.strip() for p in pages if p.strip())
            if extracted.strip():
                return extracted
        except Exception as exc:
            logger.debug(
                "PDF text extraction fallback to raw bytes for %s (%s): %s",
                file_path,
                type(exc).__name__,
                exc,
            )
    raw_bytes = file_path.read_bytes()
    if b"\x00" in raw_bytes[:8192] and suffix != ".pdf":
        return ""
    decoded = raw_bytes.decode("utf-8", errors="replace").replace("\x00", "")
    if suffix == ".sgml" and "<" in decoded and ">" in decoded:
        return extract_sgml_docbook_text(decoded)
    if suffix == ".mdx":
        return strip_mdx_boilerplate(decoded)
    return decoded


def _should_skip_sync_parts(rel_parts: tuple[str, ...], is_multi_repo: bool) -> bool:
    for idx, part in enumerate(rel_parts):
        if part in _SYNC_SKIP_DIRS:
            return True
        if part == "build":
            if is_multi_repo and idx == 0:
                continue
            if idx > 0 and rel_parts[idx - 1] == "src":
                continue
            return True
    return False


_SAFE_DOMAIN_SLUG_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


def _validate_domain_slug(domain_id: str) -> str:
    """Validates domain_id is a strict alphanumeric/underscore/hyphen slug without traversal."""
    if (
        not _SAFE_DOMAIN_SLUG_RE.fullmatch(domain_id)
        or ".." in domain_id
        or "/" in domain_id
        or "\\" in domain_id
    ):
        raise ValueError(f"Invalid domain_id slug: {domain_id!r}")
    return domain_id


def _trusted_manifest_prefixes() -> tuple[str, ...]:
    """Returns canonical realpath directory prefixes permitted for domain manifest I/O."""
    import tempfile

    raw_roots = (
        str(ScoutSettings().domains_dir),
        str(Path("domains")),
        str(Path(__file__).resolve().parents[3] / "domains"),
        tempfile.gettempdir(),
        "/tmp",
    )
    prefixes: list[str] = []
    for root in raw_roots:
        real_root = os.path.realpath(root)
        with_sep = real_root if real_root.endswith(os.sep) else real_root + os.sep
        if with_sep not in prefixes:
            prefixes.append(with_sep)
    return tuple(prefixes)


def register_custom_corpus_index(domain_id: str, index_path: Path) -> None:
    """Registers a custom domain sync_manifest.json path for runtime resolution."""
    safe_domain = _validate_domain_slug(domain_id)
    allowed_prefixes = _trusted_manifest_prefixes()
    real_index = os.path.realpath(str(index_path))
    if not real_index.startswith(allowed_prefixes):
        raise ValueError(f"Refusing untrusted corpus index path: {index_path}")
    resolved = Path(real_index)
    parent_real = os.path.dirname(real_index)
    if resolved.name != _SYNC_MANIFEST_FILENAME and os.path.exists(parent_real):
        real_candidate = os.path.realpath(os.path.join(parent_real, _SYNC_MANIFEST_FILENAME))
        if not real_candidate.startswith(allowed_prefixes):
            raise ValueError(f"Refusing untrusted sync manifest path: {index_path}")
        if os.path.exists(real_candidate) or resolved.name == "corpus_index.json":
            resolved = Path(real_candidate)
    CUSTOM_CORPUS_INDEX_PATHS[safe_domain] = resolved


def invalidate_custom_corpus_cache(domain_id: str | None = None) -> None:
    """Clears file stat SHA cache and bumps domain epoch (corpora live in PostgreSQL 17)."""
    _FILE_STAT_BLOB_CACHE.clear()
    if domain_id is None:
        for k in list(_DOMAIN_SYNC_EPOCHS):
            _DOMAIN_SYNC_EPOCHS[k] = _DOMAIN_SYNC_EPOCHS.get(k, 0) + 1
    else:
        _DOMAIN_SYNC_EPOCHS[domain_id] = _DOMAIN_SYNC_EPOCHS.get(domain_id, 0) + 1


def _resolve_custom_corpus_index_path(domain_id: str) -> Path | None:
    if not _SAFE_DOMAIN_SLUG_RE.fullmatch(domain_id) or ".." in domain_id:
        return None
    allowed_prefixes = _trusted_manifest_prefixes()
    candidates: list[Path] = []
    if domain_id in CUSTOM_CORPUS_INDEX_PATHS:
        candidates.append(CUSTOM_CORPUS_INDEX_PATHS[domain_id])
    else:
        candidates.append(ScoutSettings().domains_dir / domain_id / _SYNC_MANIFEST_FILENAME)
        candidates.append(Path("domains") / domain_id / _SYNC_MANIFEST_FILENAME)
        repo_domains = Path(__file__).resolve().parents[3] / "domains"
        candidates.append(repo_domains / domain_id / _SYNC_MANIFEST_FILENAME)
    for cand in candidates:
        real_cand = os.path.realpath(str(cand))
        if not real_cand.startswith(allowed_prefixes):
            continue
        if os.path.exists(real_cand):
            return Path(real_cand)
    return None


def is_multi_repo_source_dir(source_dir: Path) -> bool:
    """Returns True when source_dir is a parent directory containing sub-repositories."""
    if not source_dir.exists() or not source_dir.is_dir():
        return False
    if (source_dir / ".git").exists():
        return False
    for child in source_dir.iterdir():
        if not child.is_dir():
            continue
        if (
            (child / ".git").exists()
            or (child / "pyproject.toml").exists()
            or (child / "package.json").exists()
            or (child / "go.mod").exists()
            or (child / "Cargo.toml").exists()
            or any(child.glob("*.csproj"))
            or any(child.glob("*.tf"))
        ):
            return True
    return False


def derive_custom_repo_name(file_path: Path, source_dir: Path, domain_id: str) -> str:
    """Derives repository name for a file inside single-repo or multi-repo source_dir."""
    if is_multi_repo_source_dir(source_dir):
        try:
            rel_parts = file_path.relative_to(source_dir).parts
            if len(rel_parts) >= 2:
                return rel_parts[0]
        except Exception as exc:
            logger.debug(
                "Unable to derive repo name for %s (%s): %s",
                file_path,
                type(exc).__name__,
                exc,
            )
    return domain_id


def _format_pg_vector(embedding: list[float]) -> str:
    vec = embedding if len(embedding) == 768 else _DEFAULT_DUMMY_EMBEDDING
    return "[" + ",".join(f"{float(v):.6f}" for v in vec) + "]"


def _normalize_pg_symbol_kind(kind: str) -> str:
    return kind if kind in _PG_ALLOWED_SYMBOL_KINDS else "function"


def _normalize_pg_edge_kind(kind: str) -> str | None:
    if kind in _PG_ALLOWED_EDGE_KINDS:
        return kind
    return _PG_EDGE_REMAP.get(kind)


def _normalize_pg_market_scope(scope: MarketRegime | str | None) -> str | None:
    if scope is None:
        return "ENTERPRISE_DOC"
    val = scope.value if hasattr(scope, "value") else str(scope)
    if val in _PG_ALLOWED_MARKET_SCOPES:
        return val
    return "ENTERPRISE_DOC"


def _deserialize_pg_market_scope(raw_scope: object) -> MarketRegime | str | None:
    if raw_scope is None:
        return None
    text = str(raw_scope)
    if text in _MARKET_REGIME_VALUES:
        return MarketRegime(text)
    return text


def _build_sync_principal(
    domain_id: str, extra_scopes: list[str] | None = None
) -> PrincipalContext:
    scopes = [
        "public_eng",
        "public_quant",
        "quant_restricted",
        f"domain:{domain_id}",
        f"repo:{domain_id}",
    ]
    for sc in extra_scopes or []:
        if sc and sc not in scopes:
            scopes.append(sc)
    return PrincipalContext(
        user_id="custom_domain_sync",
        role=RoleType.ENGINEER,
        allowed_scopes=scopes,
    )


def write_custom_sync_manifest(
    *,
    manifest_path: Path,
    domain_id: str,
    kind: str,
    source_dir: str,
    repo_commit_shas: dict[str, str],
    code_files_indexed: int,
    ast_symbols_indexed: int,
    ast_edges_indexed: int,
    doc_chunks_indexed: int,
    empty_doc_shas: dict[str, str] | None = None,
) -> Path:
    """Writes a compact metadata-only sync_manifest.json without corpus payloads."""
    safe_domain = _validate_domain_slug(domain_id)
    allowed_prefixes = _trusted_manifest_prefixes()
    real_manifest = os.path.realpath(str(manifest_path))
    if not real_manifest.startswith(allowed_prefixes):
        raise ValueError(f"Refusing untrusted sync manifest path: {manifest_path}")
    safe_manifest = Path(real_manifest)
    os.makedirs(os.path.dirname(real_manifest), exist_ok=True)
    repo_root = Path(__file__).resolve().parents[3]
    portable_source_dir = source_dir
    if source_dir:
        try:
            resolved_src = Path(source_dir).resolve()
            if resolved_src.is_relative_to(repo_root):
                portable_source_dir = resolved_src.relative_to(repo_root).as_posix()
        except OSError:
            portable_source_dir = source_dir
    payload: dict[str, Any] = {
        "domain_id": safe_domain,
        "kind": kind,
        "source_dir": portable_source_dir,
        "synced_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "repo_commit_shas": repo_commit_shas,
        "code_files_indexed": code_files_indexed,
        "ast_symbols_indexed": ast_symbols_indexed,
        "ast_edges_indexed": ast_edges_indexed,
        "doc_chunks_indexed": doc_chunks_indexed,
        "storage_backend": _PG_STORAGE_BACKEND,
    }
    if empty_doc_shas:
        payload["empty_doc_shas"] = dict(sorted(empty_doc_shas.items()))
    with open(real_manifest, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    register_custom_corpus_index(safe_domain, safe_manifest)
    return safe_manifest


def persist_custom_domain_to_pg(
    *,
    domain_id: str,
    display_name: str,
    kind: str,
    manifest_config: dict[str, Any],
    code_files: list[CodeFileRecord],
    ast_symbols: list[ASTSymbolRecord],
    ast_edges: list[ASTEdgeRecord],
    doc_chunks: list[DocChunkRecord],
    deleted_file_ids: list[UUID] | None = None,
    deleted_doc_ids: list[UUID] | None = None,
    repo_commit_shas: dict[str, str] | None = None,
    reset_domain_rows: bool = False,
) -> tuple[int, int, int, int]:
    """Persists custom domain records directly into PostgreSQL 17 under FORCE RLS."""
    all_scopes: list[str] = []
    for rec_scopes in [
        *[f.allowed_scopes for f in code_files],
        *[s.allowed_scopes for s in ast_symbols],
        *[e.allowed_scopes for e in ast_edges],
        *[d.allowed_scopes for d in doc_chunks],
    ]:
        for sc in rec_scopes:
            if sc not in all_scopes:
                all_scopes.append(sc)
    for rname in {f.repo_name for f in code_files}:
        rscope = f"repo:{rname}"
        if rscope not in all_scopes:
            all_scopes.append(rscope)

    principal = _build_sync_principal(domain_id, all_scopes)
    settings = ScoutSettings()
    mgr = RLSConnectionManager(settings.db_app_url, use_pool=False)
    primary_truth = "code" if kind == "code" else "document"
    repo_helper = PostgresRLSRepository()

    with mgr.scoped_transaction(principal) as conn:
        raw_auth = manifest_config.get("authority_rules")
        auth_dict = dict(raw_auth) if isinstance(raw_auth, dict) else {}
        auth_rules = AuthorityRules(
            primary_truth_source="code" if primary_truth == "code" else "document",
            require_out_of_sample_check=bool(auth_dict.get("require_out_of_sample_check", False)),
            require_transaction_cost_check=bool(
                auth_dict.get("require_transaction_cost_check", False)
            ),
            freshness_half_life_days=int(
                auth_dict.get("freshness_half_life_days", 90 if kind == "code" else 365)
            ),
        )
        raw_tools = manifest_config.get("mcp_read_tools_allowed")
        tools_list = (
            [str(t) for t in raw_tools]
            if isinstance(raw_tools, list) and raw_tools
            else ["get_symbol_neighborhood", "hybrid_doc_search", "read_verbatim_file_slice"]
        )
        raw_contract = str(
            manifest_config.get(
                "output_contract",
                "ImplementationContextPack" if kind == "code" else "DomainKnowledgeBrief",
            )
        )
        if raw_contract not in {
            "ImplementationContextPack",
            "QuantStrategyBrief",
            "DomainKnowledgeBrief",
        }:
            raw_contract = "ImplementationContextPack" if kind == "code" else "DomainKnowledgeBrief"
        repo_helper.upsert_domain_pack(
            conn,
            DomainPackManifest(
                domain_id=domain_id,
                display_name=display_name,
                base_model_id=str(
                    manifest_config.get("base_model_id", "google/gemma-4-26B-A4B-it")
                ),
                contextualizer_model_id=str(
                    manifest_config.get("contextualizer_model_id", "google/gemma-4-E4B-it")
                ),
                lora_adapter_id=str(
                    manifest_config.get("lora_adapter_id", f"gemma4_26b_{domain_id}")
                ),
                lora_adapter_path=str(
                    manifest_config.get(
                        "lora_adapter_path", f"domains/{domain_id}/lora_adapter/adapter_model"
                    )
                ),
                embedding_model_id=str(
                    manifest_config.get("embedding_model_id", "google/embeddinggemma-2")
                ),
                embedding_dimensions=int(manifest_config.get("embedding_dimensions", 768)),
                mcp_read_tools_allowed=tools_list,
                authority_rules=auth_rules,
                output_contract=raw_contract,  # type: ignore[arg-type]
            ),
        )

        if reset_domain_rows:
            conn.execute("DELETE FROM ast_edges WHERE domain_id = %s", (domain_id,))
            conn.execute("DELETE FROM ast_symbols WHERE domain_id = %s", (domain_id,))
            conn.execute("DELETE FROM code_files WHERE domain_id = %s", (domain_id,))
            conn.execute("DELETE FROM doc_chunks WHERE domain_id = %s", (domain_id,))
            conn.execute("DELETE FROM repositories WHERE domain_id = %s", (domain_id,))

        if deleted_file_ids:
            conn.execute(
                "DELETE FROM code_files WHERE domain_id = %s AND file_id = ANY(%s)",
                (domain_id, deleted_file_ids),
            )
        if deleted_doc_ids:
            conn.execute(
                "DELETE FROM doc_chunks WHERE domain_id = %s AND chunk_id = ANY(%s)",
                (domain_id, deleted_doc_ids),
            )

        repo_id_by_name: dict[str, UUID] = {}
        repo_scopes_by_name: dict[str, list[str]] = {}
        for cf in code_files:
            repo_scopes_by_name.setdefault(cf.repo_name, [])
            for sc in cf.allowed_scopes:
                if sc not in repo_scopes_by_name[cf.repo_name]:
                    repo_scopes_by_name[cf.repo_name].append(sc)
        if not repo_scopes_by_name and repo_commit_shas:
            for rname in repo_commit_shas:
                repo_scopes_by_name[rname] = list(principal.allowed_scopes)

        for rname, rscopes in repo_scopes_by_name.items():
            rid = uuid5(NAMESPACE_URL, f"{domain_id}:repo:{rname}")
            repo_id_by_name[rname] = rid
            head_sha = (repo_commit_shas or {}).get(rname, "custom_head")
            repo_helper.insert_repository(
                conn,
                RepositoryRecord(
                    repo_id=rid,
                    domain_id=domain_id,
                    repo_name=rname,
                    source_type="local_git",
                    remote_url=f"local://{domain_id}/{rname}",
                    default_branch="main",
                    head_commit_sha=head_sha,
                    allowed_scopes=rscopes or list(principal.allowed_scopes),
                    metadata_json={"storage_backend": _PG_STORAGE_BACKEND},
                ),
            )

        existing_repo_rows = conn.execute(
            "SELECT repo_name, repo_id FROM repositories WHERE domain_id = %s",
            (domain_id,),
        ).fetchall()
        for rr in existing_repo_rows:
            repo_id_by_name.setdefault(str(rr[0]), UUID(str(rr[1])))

        if code_files:
            if not reset_domain_rows:
                conn.execute(
                    "DELETE FROM ast_symbols WHERE domain_id = %s AND file_id = ANY(%s)",
                    (domain_id, [cf.file_id for cf in code_files]),
                )
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO code_files (
                        file_id, domain_id, repo_id, repo_name, file_path,
                        git_commit_sha, git_blob_sha, language, allowed_scopes,
                        raw_content, line_count
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (domain_id, repo_name, file_path) DO UPDATE SET
                        file_id = EXCLUDED.file_id,
                        repo_id = EXCLUDED.repo_id,
                        git_commit_sha = EXCLUDED.git_commit_sha,
                        git_blob_sha = EXCLUDED.git_blob_sha,
                        language = EXCLUDED.language,
                        allowed_scopes = EXCLUDED.allowed_scopes,
                        raw_content = EXCLUDED.raw_content,
                        line_count = EXCLUDED.line_count,
                        indexed_at = NOW()
                    """,
                    [
                        (
                            cf.file_id,
                            domain_id,
                            cf.repo_id or repo_id_by_name.get(cf.repo_name),
                            cf.repo_name,
                            cf.file_path,
                            cf.git_commit_sha,
                            cf.git_blob_sha,
                            cf.language,
                            cf.allowed_scopes,
                            cf.raw_content,
                            cf.line_count,
                        )
                        for cf in code_files
                    ],
                )

        file_to_repo_id = {
            cf.file_id: (cf.repo_id or repo_id_by_name.get(cf.repo_name)) for cf in code_files
        }
        file_to_repo_name = {cf.file_id: cf.repo_name for cf in code_files}

        if ast_symbols:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO ast_symbols (
                        symbol_id, file_id, domain_id, repo_id, qualified_name,
                        repo_qualified_id, bare_name, symbol_kind, start_line, end_line,
                        signature_skeleton, contextual_summary, allowed_scopes, embedding
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
                    ON CONFLICT (symbol_id) DO UPDATE SET
                        repo_id = EXCLUDED.repo_id,
                        qualified_name = EXCLUDED.qualified_name,
                        repo_qualified_id = EXCLUDED.repo_qualified_id,
                        bare_name = EXCLUDED.bare_name,
                        symbol_kind = EXCLUDED.symbol_kind,
                        start_line = EXCLUDED.start_line,
                        end_line = EXCLUDED.end_line,
                        signature_skeleton = EXCLUDED.signature_skeleton,
                        contextual_summary = EXCLUDED.contextual_summary,
                        allowed_scopes = EXCLUDED.allowed_scopes,
                        embedding = EXCLUDED.embedding
                    """,
                    [
                        (
                            sym.symbol_id,
                            sym.file_id,
                            domain_id,
                            sym.repo_id or file_to_repo_id.get(sym.file_id),
                            sym.qualified_name,
                            sym.repo_qualified_id
                            or (
                                f"{file_to_repo_name[sym.file_id]}::{sym.qualified_name}"
                                if sym.file_id in file_to_repo_name
                                else sym.qualified_name
                            ),
                            sym.bare_name or sym.qualified_name.rsplit(".", 1)[-1],
                            _normalize_pg_symbol_kind(sym.symbol_kind),
                            sym.start_line,
                            max(sym.start_line, sym.end_line),
                            sym.signature_skeleton,
                            sym.contextual_summary,
                            sym.allowed_scopes,
                            _format_pg_vector(sym.embedding),
                        )
                        for sym in ast_symbols
                    ],
                )

        if ast_edges:
            visible_sym_rows = conn.execute(
                "SELECT symbol_id FROM ast_symbols WHERE domain_id = %s",
                (domain_id,),
            ).fetchall()
            valid_sym_ids = {UUID(str(r[0])) for r in visible_sym_rows}
            edge_params: list[tuple[Any, ...]] = []
            for edge in ast_edges:
                if edge.source_symbol_id not in valid_sym_ids:
                    continue
                norm_kind = _normalize_pg_edge_kind(edge.edge_kind)
                if norm_kind is None:
                    continue
                tgt_id = edge.target_symbol_id if edge.target_symbol_id in valid_sym_ids else None
                edge_params.append(
                    (
                        edge.edge_id,
                        domain_id,
                        edge.source_symbol_id,
                        edge.target_qualified_name,
                        tgt_id,
                        norm_kind,
                        edge.allowed_scopes,
                        edge.target_repo_qualified_id,
                        bool(edge.is_cross_repo),
                    )
                )
            if edge_params:
                with conn.cursor() as cur:
                    cur.executemany(
                        """
                        INSERT INTO ast_edges (
                            edge_id, domain_id, source_symbol_id, target_qualified_name,
                            target_symbol_id, edge_kind, allowed_scopes,
                            target_repo_qualified_id, is_cross_repo
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (edge_id) DO UPDATE SET
                            target_qualified_name = EXCLUDED.target_qualified_name,
                            target_symbol_id = EXCLUDED.target_symbol_id,
                            edge_kind = EXCLUDED.edge_kind,
                            allowed_scopes = EXCLUDED.allowed_scopes,
                            target_repo_qualified_id = EXCLUDED.target_repo_qualified_id,
                            is_cross_repo = EXCLUDED.is_cross_repo
                        """,
                        edge_params,
                    )

        if doc_chunks:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO doc_chunks (
                        chunk_id, domain_id, doc_id, doc_title, source_uri,
                        market_scope, published_date, authority_weight, section_heading,
                        contextual_header, verbatim_content, metadata_json, allowed_scopes,
                        embedding, source_type, version_str, is_stale, stale_reason
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s,
                        %s::vector, %s, %s, %s, %s
                    )
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        doc_title = EXCLUDED.doc_title,
                        source_uri = EXCLUDED.source_uri,
                        market_scope = EXCLUDED.market_scope,
                        published_date = EXCLUDED.published_date,
                        authority_weight = EXCLUDED.authority_weight,
                        section_heading = EXCLUDED.section_heading,
                        contextual_header = EXCLUDED.contextual_header,
                        verbatim_content = EXCLUDED.verbatim_content,
                        metadata_json = EXCLUDED.metadata_json,
                        allowed_scopes = EXCLUDED.allowed_scopes,
                        embedding = EXCLUDED.embedding,
                        source_type = EXCLUDED.source_type,
                        version_str = EXCLUDED.version_str,
                        is_stale = EXCLUDED.is_stale,
                        stale_reason = EXCLUDED.stale_reason
                    """,
                    [
                        (
                            chk.chunk_id,
                            domain_id,
                            chk.doc_id,
                            chk.doc_title,
                            chk.source_uri,
                            _normalize_pg_market_scope(chk.market_scope),
                            chk.published_date,
                            float(chk.authority_weight),
                            chk.section_heading,
                            chk.contextual_header,
                            chk.verbatim_content,
                            json.dumps(chk.metadata_json),
                            chk.allowed_scopes,
                            _format_pg_vector(chk.embedding),
                            chk.source_type,
                            chk.version_str,
                            bool(chk.is_stale),
                            chk.stale_reason,
                        )
                        for chk in doc_chunks
                    ],
                )

        cf_row = conn.execute(
            "SELECT COUNT(*) FROM code_files WHERE domain_id = %s", (domain_id,)
        ).fetchone()
        cf_cnt = int(str(cf_row[0])) if cf_row is not None else 0
        sym_row = conn.execute(
            "SELECT COUNT(*) FROM ast_symbols WHERE domain_id = %s", (domain_id,)
        ).fetchone()
        sym_cnt = int(str(sym_row[0])) if sym_row is not None else 0
        edge_row = conn.execute(
            "SELECT COUNT(*) FROM ast_edges WHERE domain_id = %s", (domain_id,)
        ).fetchone()
        edge_cnt = int(str(edge_row[0])) if edge_row is not None else 0
        doc_row = conn.execute(
            "SELECT COUNT(*) FROM doc_chunks WHERE domain_id = %s", (domain_id,)
        ).fetchone()
        doc_cnt = int(str(doc_row[0])) if doc_row is not None else 0
    _DOMAIN_SYNC_EPOCHS[domain_id] = _DOMAIN_SYNC_EPOCHS.get(domain_id, 0) + 1
    return cf_cnt, sym_cnt, edge_cnt, doc_cnt


def sync_custom_domain_corpus_if_needed(
    domain_id: str,
    index_path: Path | None = None,
    force: bool = False,
) -> dict[str, Any] | None:
    """Incrementally syncs custom domain working-tree files into PostgreSQL 17."""
    _ = force
    from context_scout.ast_engine.contract_linker import CrossBoundaryContractLinker
    from context_scout.ast_engine.extractor import ASTExtractor
    from context_scout.ingestion.git_merkle_indexer import (
        CODE_EXTENSIONS,
        LANG_MAP,
        generate_deterministic_embedding,
    )

    if not _SAFE_DOMAIN_SLUG_RE.fullmatch(domain_id) or ".." in domain_id:
        return None
    allowed_prefixes = _trusted_manifest_prefixes()
    raw_cand = index_path or _resolve_custom_corpus_index_path(domain_id)
    real_cand: str | None = None
    if raw_cand is not None:
        resolved_raw = os.path.realpath(str(raw_cand))
        if not resolved_raw.startswith(allowed_prefixes):
            return None
        real_cand = resolved_raw
        if not os.path.exists(real_cand) and os.path.basename(real_cand) != _SYNC_MANIFEST_FILENAME:
            real_alt = os.path.realpath(
                os.path.join(os.path.dirname(real_cand), _SYNC_MANIFEST_FILENAME)
            )
            if real_alt.startswith(allowed_prefixes) and os.path.exists(real_alt):
                real_cand = real_alt

    manifest_data: dict[str, Any] = {}
    if real_cand is not None and os.path.exists(real_cand):
        try:
            with open(real_cand, encoding="utf-8") as cf:
                loaded = json.load(cf)
            if isinstance(loaded, dict):
                manifest_data = loaded
        except Exception as exc:
            logger.warning(
                "Failed loading sync_manifest.json for '%s' (%s): %s",
                domain_id,
                type(exc).__name__,
                exc,
            )
        cand = Path(real_cand)
    else:
        raw_domain_yaml = ScoutSettings().domains_dir / domain_id / "domain.yaml"
        real_domain_yaml = os.path.realpath(str(raw_domain_yaml))
        if not real_domain_yaml.startswith(allowed_prefixes) or not os.path.exists(
            real_domain_yaml
        ):
            return None
        real_cand = os.path.realpath(
            os.path.join(os.path.dirname(real_domain_yaml), _SYNC_MANIFEST_FILENAME)
        )
        if not real_cand.startswith(allowed_prefixes):
            return None
        cand = Path(real_cand)

    source_dir_str = str(manifest_data.get("source_dir", "")).strip()
    kind = str(manifest_data.get("kind", "")).strip()
    yaml_cfg: dict[str, Any] = {}
    real_manifest_yaml = os.path.realpath(os.path.join(os.path.dirname(real_cand), "domain.yaml"))
    if real_manifest_yaml.startswith(allowed_prefixes) and os.path.exists(real_manifest_yaml):
        try:
            import yaml

            with open(real_manifest_yaml, encoding="utf-8") as mf:
                yaml_cfg = yaml.safe_load(mf) or {}
            if not source_dir_str:
                source_dir_str = str(yaml_cfg.get("source_dir", "")).strip()
            if not kind:
                kind = (
                    "code"
                    if yaml_cfg.get("output_contract") == "ImplementationContextPack"
                    else "knowledge"
                )
        except Exception as exc:
            logger.warning(
                "Failed loading domain.yaml for '%s' (%s): %s",
                domain_id,
                type(exc).__name__,
                exc,
            )

    if not kind:
        kind = "code"

    sync_principal = _build_sync_principal(domain_id)
    settings = ScoutSettings()
    mgr = RLSConnectionManager(settings.db_app_url, use_pool=False)

    with mgr.scoped_transaction(sync_principal) as conn:
        cf_meta_rows = conn.execute(
            "SELECT file_id, repo_name, file_path, git_blob_sha"
            " FROM code_files WHERE domain_id = %s",
            (domain_id,),
        ).fetchall()
        sym_row = conn.execute(
            "SELECT COUNT(*) FROM ast_symbols WHERE domain_id = %s", (domain_id,)
        ).fetchone()
        sym_cnt = int(str(sym_row[0])) if sym_row is not None else 0
        edge_row = conn.execute(
            "SELECT COUNT(*) FROM ast_edges WHERE domain_id = %s", (domain_id,)
        ).fetchone()
        edge_cnt = int(str(edge_row[0])) if edge_row is not None else 0
        doc_rows = conn.execute(
            "SELECT chunk_id::text, "
            "coalesce(metadata_json->>'rel_path', metadata_json->>'filename', ''), "
            "coalesce(metadata_json->>'blob_sha', ''), "
            "CASE WHEN coalesce(metadata_json->>'blob_sha', '') = '' "
            "THEN verbatim_content ELSE '' END "
            "FROM doc_chunks WHERE domain_id = %s",
            (domain_id,),
        ).fetchall()

    repo_root = Path(__file__).resolve().parents[3]
    resolved_source_dir = Path(source_dir_str) if source_dir_str else Path("")
    if (
        source_dir_str
        and not resolved_source_dir.is_absolute()
        and not resolved_source_dir.is_dir()
    ):
        repo_relative = repo_root / resolved_source_dir
        if repo_relative.is_dir():
            resolved_source_dir = repo_relative

    if not source_dir_str or not resolved_source_dir.is_dir():
        return {
            "synced": False,
            "added": 0,
            "modified": 0,
            "deleted": 0,
            "unchanged": len(cf_meta_rows) or len(doc_rows),
            "files_indexed": len(cf_meta_rows) or len(doc_rows),
            "symbols_indexed": sym_cnt,
            "edges_indexed": edge_cnt,
        }

    source_dir = resolved_source_dir
    if kind != "code":
        raw_empty_shas = manifest_data.get("empty_doc_shas")
        empty_doc_shas: dict[str, str] = (
            {str(k): str(v) for k, v in raw_empty_shas.items()}
            if isinstance(raw_empty_shas, dict)
            else {}
        )
        raw_repo_shas = manifest_data.get("repo_commit_shas")
        existing_repo_shas: dict[str, str] = (
            {str(k): str(v) for k, v in raw_repo_shas.items()}
            if isinstance(raw_repo_shas, dict)
            else {}
        )
        existing_doc_by_rel: dict[str, list[tuple[str, str, str]]] = {}
        for dr in doc_rows:
            rel_k = str(dr[1])
            if rel_k:
                existing_doc_by_rel.setdefault(rel_k, []).append(
                    (str(dr[0]), str(dr[2]), str(dr[3]))
                )

        live_docs: dict[str, Path] = {}
        doc_added: list[str] = []
        doc_modified: list[str] = []
        doc_unchanged = 0
        src_str = str(source_dir)
        for walk_root, dir_names, file_names in os.walk(src_str):
            rel_root = os.path.relpath(walk_root, src_str)
            parent_is_src = os.path.basename(walk_root) == "src"
            dir_names[:] = sorted(
                d for d in dir_names if d not in _SYNC_SKIP_DIRS and (d != "build" or parent_is_src)
            )
            for fn in sorted(file_names):
                _, ext = os.path.splitext(fn)
                if ext.lower() not in CUSTOM_DOC_EXTENSIONS:
                    continue
                rel_p = fn if rel_root == "." else f"{rel_root}/{fn}"
                p = Path(walk_root) / fn
                try:
                    st = p.stat()
                    cached_blob = _FILE_STAT_BLOB_CACHE.get(p)
                    if (
                        cached_blob is not None
                        and cached_blob[0] == st.st_mtime_ns
                        and cached_blob[1] == st.st_size
                    ):
                        blob_sha = cached_blob[2]
                    else:
                        blob_sha = hashlib.sha1(p.read_bytes()).hexdigest()
                        _FILE_STAT_BLOB_CACHE[p] = (st.st_mtime_ns, st.st_size, blob_sha)
                except Exception as exc:
                    logger.debug(
                        "Skipping unreadable doc file %s (%s): %s", p, type(exc).__name__, exc
                    )
                    continue
                live_docs[rel_p] = p
                old_chunks = existing_doc_by_rel.get(rel_p) or existing_doc_by_rel.get(fn)
                if not old_chunks:
                    if empty_doc_shas.get(rel_p) == blob_sha:
                        doc_unchanged += 1
                    else:
                        doc_added.append(rel_p)
                elif old_chunks[0][1] and old_chunks[0][1] != blob_sha:
                    doc_modified.append(rel_p)
                elif not old_chunks[0][1]:
                    new_txt = extract_document_text(p)
                    if new_txt and new_txt != old_chunks[0][2]:
                        doc_modified.append(rel_p)
                    else:
                        doc_unchanged += 1
                else:
                    doc_unchanged += 1

        live_names = {os.path.basename(lp) for lp in live_docs}
        doc_deleted_keys = {
            k for k in existing_doc_by_rel if k not in live_docs and k not in live_names
        }
        stale_empty_keys = [k for k in empty_doc_shas if k not in live_docs]
        if not doc_added and not doc_modified and not doc_deleted_keys and not stale_empty_keys:
            return {
                "synced": False,
                "added": 0,
                "modified": 0,
                "deleted": 0,
                "unchanged": doc_unchanged,
                "files_indexed": len(doc_rows),
                "symbols_indexed": 0,
                "edges_indexed": 0,
            }

        del_doc_ids: list[UUID] = []
        seen_del_ids: set[str] = set()
        for k in doc_deleted_keys:
            for entry in existing_doc_by_rel[k]:
                if entry[0] not in seen_del_ids:
                    seen_del_ids.add(entry[0])
                    del_doc_ids.append(UUID(entry[0]))
        for rel_p in doc_modified:
            matched_entries = (
                existing_doc_by_rel.get(rel_p) or existing_doc_by_rel.get(Path(rel_p).name) or []
            )
            for entry in matched_entries:
                if entry[0] not in seen_del_ids:
                    seen_del_ids.add(entry[0])
                    del_doc_ids.append(UUID(entry[0]))

        updated_empty_shas = {k: v for k, v in empty_doc_shas.items() if k in live_docs}
        doc_file_entries: list[tuple[Path, str, str, bytes, str]] = []
        for rel_p in sorted(doc_added + doc_modified):
            disk_p = live_docs[rel_p]
            raw_doc_bytes = disk_p.read_bytes()
            cached_info = _FILE_STAT_BLOB_CACHE.get(disk_p)
            blob_sha = (
                cached_info[2]
                if cached_info is not None
                else hashlib.sha1(raw_doc_bytes).hexdigest()
            )
            text = extract_document_text(disk_p)
            if not text.strip():
                updated_empty_shas[rel_p] = blob_sha
                continue
            updated_empty_shas.pop(rel_p, None)
            doc_file_entries.append((disk_p, rel_p, text, raw_doc_bytes, blob_sha))

        if not doc_file_entries and not del_doc_ids:
            if updated_empty_shas != empty_doc_shas:
                write_custom_sync_manifest(
                    manifest_path=cand,
                    domain_id=domain_id,
                    kind=kind,
                    source_dir=str(source_dir.resolve()),
                    repo_commit_shas=existing_repo_shas,
                    code_files_indexed=len(cf_meta_rows),
                    ast_symbols_indexed=sym_cnt,
                    ast_edges_indexed=edge_cnt,
                    doc_chunks_indexed=len(doc_rows),
                    empty_doc_shas=updated_empty_shas,
                )
            return {
                "synced": False,
                "added": 0,
                "modified": 0,
                "deleted": 0,
                "unchanged": doc_unchanged + len(doc_added),
                "files_indexed": len(doc_rows),
                "symbols_indexed": 0,
                "edges_indexed": 0,
            }

        new_doc_records = build_custom_doc_chunk_records(
            domain_id=domain_id,
            file_entries=doc_file_entries,
            batch_size=64,
        )
        cf_cnt, sym_cnt, edge_cnt, doc_cnt = persist_custom_domain_to_pg(
            domain_id=domain_id,
            display_name=str(yaml_cfg.get("display_name", domain_id)),
            kind=kind,
            manifest_config=yaml_cfg,
            code_files=[],
            ast_symbols=[],
            ast_edges=[],
            doc_chunks=new_doc_records,
            deleted_doc_ids=del_doc_ids,
        )
        write_custom_sync_manifest(
            manifest_path=cand,
            domain_id=domain_id,
            kind=kind,
            source_dir=str(source_dir.resolve()),
            repo_commit_shas=existing_repo_shas,
            code_files_indexed=cf_cnt,
            ast_symbols_indexed=sym_cnt,
            ast_edges_indexed=edge_cnt,
            doc_chunks_indexed=doc_cnt,
            empty_doc_shas=updated_empty_shas,
        )
        return {
            "synced": True,
            "added": len(doc_file_entries),
            "modified": len(doc_modified),
            "deleted": len(doc_deleted_keys),
            "unchanged": doc_unchanged,
            "files_indexed": doc_cnt,
            "symbols_indexed": 0,
            "edges_indexed": 0,
        }

    # Code domain incremental Merkle sync directly against PG17
    effective_exts = set(CODE_EXTENSIONS) | set(_EFFECTIVE_EXTRA_EXTENSIONS.keys())
    effective_langs = {**LANG_MAP, **_EFFECTIVE_EXTRA_EXTENSIONS}
    existing_by_path: dict[str, tuple[UUID, str, str]] = {
        str(r[2]): (UUID(str(r[0])), str(r[1]), str(r[3])) for r in cf_meta_rows
    }
    is_multi = is_multi_repo_source_dir(source_dir)

    live_files_meta: dict[str, tuple[Path, str, str]] = {}
    added_paths: list[str] = []
    modified_paths: list[str] = []
    unchanged_count = 0

    for p in source_dir.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in effective_exts:
            continue
        rel_parts = p.relative_to(source_dir).parts
        if _should_skip_sync_parts(rel_parts, is_multi):
            continue
        rel_path = str(p.relative_to(source_dir))
        repo_n = derive_custom_repo_name(p, source_dir, domain_id)
        try:
            st = p.stat()
            cached_blob = _FILE_STAT_BLOB_CACHE.get(p)
            if (
                cached_blob is not None
                and cached_blob[0] == st.st_mtime_ns
                and cached_blob[1] == st.st_size
            ):
                blob_sha = cached_blob[2]
            else:
                raw_b = p.read_bytes()
                header = f"blob {len(raw_b)}\x00".encode("ascii")
                blob_sha = hashlib.sha1(header + raw_b).hexdigest()
                _FILE_STAT_BLOB_CACHE[p] = (st.st_mtime_ns, st.st_size, blob_sha)
        except Exception as exc:
            logger.debug("Skipping unreadable code file %s (%s): %s", p, type(exc).__name__, exc)
            continue
        live_files_meta[rel_path] = (p, repo_n, blob_sha)
        old_entry = existing_by_path.get(rel_path)
        if old_entry is None:
            added_paths.append(rel_path)
        elif old_entry[2] != blob_sha:
            modified_paths.append(rel_path)
        else:
            unchanged_count += 1

    changed_or_new = added_paths + modified_paths
    deleted_paths = set(existing_by_path.keys()) - set(live_files_meta.keys())

    if not changed_or_new and not deleted_paths:
        return {
            "synced": False,
            "added": 0,
            "modified": 0,
            "deleted": 0,
            "unchanged": unchanged_count,
            "files_indexed": len(cf_meta_rows),
            "symbols_indexed": sym_cnt,
            "edges_indexed": edge_cnt,
        }

    del_file_ids = [existing_by_path[p][0] for p in deleted_paths if p in existing_by_path]
    extractor = ASTExtractor()
    new_code_files: list[CodeFileRecord] = []
    new_ast_symbols: list[ASTSymbolRecord] = []
    pending_edges: list[tuple[str, str, str, str, list[str]]] = []
    qual_to_sym_id: dict[str, UUID] = {}
    repo_sym_to_id: dict[tuple[str, str], UUID] = {}
    short_to_sym_ids: dict[str, list[UUID]] = {}
    repo_commit_shas: dict[str, str] = dict(manifest_data.get("repo_commit_shas") or {})

    # Preload kept symbols for edge target resolution
    with mgr.scoped_transaction(sync_principal) as conn:
        kept_sym_rows = conn.execute(
            """
            SELECT s.symbol_id, s.qualified_name, f.repo_name
            FROM ast_symbols s
            JOIN code_files f ON s.file_id = f.file_id
            WHERE s.domain_id = %s
            """,
            (domain_id,),
        ).fetchall()
    mutated_file_id_set = {
        existing_by_path[p][0] for p in set(changed_or_new) | deleted_paths if p in existing_by_path
    }
    for sr in kept_sym_rows:
        sid = UUID(str(sr[0]))
        qn = str(sr[1])
        rn = str(sr[2])
        qual_to_sym_id[qn] = sid
        repo_sym_to_id[(rn, qn)] = sid
        repo_sym_to_id[(rn, qn.split(".")[-1])] = sid
        short_to_sym_ids.setdefault(qn.split(".")[-1], []).append(sid)

    for rel_path in sorted(changed_or_new):
        disk_p, repo_n, blob_sha = live_files_meta[rel_path]
        try:
            raw_text = disk_p.read_text(encoding="utf-8", errors="replace").replace("\x00", "")
        except Exception as exc:
            logger.debug(
                "Skipping unreadable changed file %s (%s): %s", disk_p, type(exc).__name__, exc
            )
            continue
        file_uuid = (
            existing_by_path[rel_path][0]
            if rel_path in existing_by_path
            else uuid5(NAMESPACE_URL, f"{domain_id}:file:{rel_path}")
        )
        commit_sha = hashlib.sha256(f"{domain_id}:{rel_path}:{blob_sha}".encode()).hexdigest()[:40]
        repo_commit_shas[repo_n] = commit_sha
        lang = effective_langs.get(disk_p.suffix.lower(), "python")
        file_scopes = ["public_eng", f"domain:{domain_id}", f"repo:{repo_n}"]
        repo_uuid = uuid5(NAMESPACE_URL, f"{domain_id}:repo:{repo_n}")
        new_code_files.append(
            CodeFileRecord(
                file_id=file_uuid,
                domain_id=domain_id,
                repo_name=repo_n,
                file_path=rel_path,
                git_commit_sha=commit_sha,
                git_blob_sha=blob_sha,
                language=lang,
                allowed_scopes=file_scopes,
                raw_content=raw_text,
                line_count=max(1, len(raw_text.splitlines())),
                repo_id=repo_uuid,
            )
        )
        sub_rel = (
            str(disk_p.relative_to(source_dir / repo_n))
            if (is_multi and (source_dir / repo_n).exists())
            else rel_path
        )
        try:
            parsed_syms, parsed_edges = extractor.parse_source(raw_text, lang, file_path=sub_rel)
        except Exception as exc:
            logger.warning(
                "AST extraction failed for %s (%s): %s", sub_rel, type(exc).__name__, exc
            )
            parsed_syms, parsed_edges = [], []

        for sym_idx, ps in enumerate(parsed_syms):
            sym_uuid = uuid5(
                NAMESPACE_URL,
                f"{domain_id}:sym:{rel_path}:{ps.qualified_name}:{ps.start_line}:{sym_idx}",
            )
            qual_to_sym_id[ps.qualified_name] = sym_uuid
            repo_sym_to_id[(repo_n, ps.qualified_name)] = sym_uuid
            repo_sym_to_id[(repo_n, ps.name)] = sym_uuid
            short_to_sym_ids.setdefault(ps.name, []).append(sym_uuid)
            summary = ps.contextual_summary or ps.docstring or ps.signature_skeleton or ps.signature
            new_ast_symbols.append(
                ASTSymbolRecord(
                    symbol_id=sym_uuid,
                    file_id=file_uuid,
                    domain_id=domain_id,
                    qualified_name=ps.qualified_name,
                    symbol_kind=_normalize_pg_symbol_kind(ps.symbol_kind),
                    start_line=ps.start_line,
                    end_line=max(ps.start_line, ps.end_line),
                    signature_skeleton=ps.signature_skeleton,
                    contextual_summary=summary,
                    allowed_scopes=file_scopes,
                    embedding=generate_deterministic_embedding(
                        f"{ps.qualified_name}\n{ps.signature_skeleton}\n{summary}"
                    ),
                    repo_id=repo_uuid,
                    repo_qualified_id=f"{repo_n}::{ps.qualified_name}",
                    bare_name=ps.bare_name or ps.name,
                )
            )
        for pe in parsed_edges:
            pending_edges.append(
                (
                    repo_n,
                    pe.source_qualified_name,
                    pe.target_qualified_name,
                    pe.edge_kind,
                    file_scopes,
                )
            )

    new_ast_edges: list[ASTEdgeRecord] = []
    for edge_idx, (e_repo, src_q, tgt_q, e_kind, e_scopes) in enumerate(pending_edges):
        src_uuid = repo_sym_to_id.get((e_repo, src_q)) or qual_to_sym_id.get(src_q)
        if src_uuid is None:
            s_cands = short_to_sym_ids.get(src_q.split(".")[-1], [])
            if s_cands:
                src_uuid = s_cands[0]
        if src_uuid is None:
            continue
        tgt_uuid = repo_sym_to_id.get((e_repo, tgt_q)) or repo_sym_to_id.get(
            (e_repo, tgt_q.split(".")[-1])
        )
        new_ast_edges.append(
            ASTEdgeRecord(
                edge_id=uuid5(
                    NAMESPACE_URL,
                    f"{domain_id}:edge:sync:{src_uuid}:{tgt_q}:{e_kind}:{edge_idx}",
                ),
                domain_id=domain_id,
                source_symbol_id=src_uuid,
                target_qualified_name=tgt_q,
                target_symbol_id=tgt_uuid,
                edge_kind=e_kind,
                allowed_scopes=e_scopes,
                is_cross_repo=False,
            )
        )

    cf_cnt, sym_cnt, edge_cnt, doc_cnt = persist_custom_domain_to_pg(
        domain_id=domain_id,
        display_name=str(yaml_cfg.get("display_name", domain_id)),
        kind=kind,
        manifest_config=yaml_cfg,
        code_files=new_code_files,
        ast_symbols=new_ast_symbols,
        ast_edges=new_ast_edges,
        doc_chunks=[],
        deleted_file_ids=del_file_ids,
        repo_commit_shas=repo_commit_shas,
    )

    # Recompute cross-boundary contracts over PG17 visible symbols/files for this domain
    bound = BoundGraphRepository(PostgresRLSRepository())
    all_dom_files = bound.list_visible_code_files(sync_principal, domain_id)
    all_dom_syms = bound.list_visible_ast_symbols(sync_principal, domain_id)
    cross_edges = CrossBoundaryContractLinker().link_contracts(
        domain_id=domain_id,
        symbols=all_dom_syms,
        code_files=all_dom_files,
    )
    if cross_edges:
        cf_cnt, sym_cnt, edge_cnt, doc_cnt = persist_custom_domain_to_pg(
            domain_id=domain_id,
            display_name=str(yaml_cfg.get("display_name", domain_id)),
            kind=kind,
            manifest_config=yaml_cfg,
            code_files=[],
            ast_symbols=[],
            ast_edges=cross_edges,
            doc_chunks=[],
            repo_commit_shas=repo_commit_shas,
        )
    _ = mutated_file_id_set

    write_custom_sync_manifest(
        manifest_path=cand,
        domain_id=domain_id,
        kind=kind,
        source_dir=str(source_dir.resolve()),
        repo_commit_shas=repo_commit_shas,
        code_files_indexed=cf_cnt,
        ast_symbols_indexed=sym_cnt,
        ast_edges_indexed=edge_cnt,
        doc_chunks_indexed=doc_cnt,
    )
    return {
        "synced": True,
        "added": len(added_paths),
        "modified": len(modified_paths),
        "deleted": len(deleted_paths),
        "unchanged": unchanged_count,
        "files_indexed": cf_cnt,
        "symbols_indexed": sym_cnt,
        "edges_indexed": edge_cnt,
    }


class BoundGraphRepository:
    """Adapts principal-based or PG17 RLS repositories to ScoutGraphRepository."""

    def __init__(
        self,
        repo: Any,
        conn: Connection[tuple[object, ...]] | None = None,
    ) -> None:
        self._repo = repo
        self._conn = conn
        self._files_cache: dict[tuple[tuple[str, ...], str, int], list[CodeFileRecord]] = {}
        self._symbols_cache: dict[tuple[tuple[str, ...], str, int], list[ASTSymbolRecord]] = {}
        self._edges_cache: dict[tuple[tuple[str, ...], str, int], list[ASTEdgeRecord]] = {}
        self._chunks_cache: dict[
            tuple[tuple[str, ...], str, str | None, bool, int], list[DocChunkRecord]
        ] = {}

    def _use_pg17(self) -> bool:
        if self._conn is not None or self._repo is None:
            return True
        if isinstance(self._repo, InMemoryRLSRepository):
            return False
        return isinstance(self._repo, PostgresRLSRepository) or hasattr(
            self._repo, "connection_manager"
        )

    def _has_in_memory_domain_records(self, domain_id: str) -> bool:
        if isinstance(self._repo, InMemoryRLSRepository):
            # InMemoryRLSRepository is an isolated dataset; never fall through to live PG17.
            return self._conn is None
        return False

    @contextmanager
    def _pg_tx(self, principal: PrincipalContext) -> Iterator[Connection[tuple[object, ...]]]:
        if self._conn is not None:
            yield self._conn
            return
        conn_mgr = getattr(self._repo, "connection_manager", None)
        if isinstance(conn_mgr, RLSConnectionManager):
            with conn_mgr.scoped_transaction(principal) as conn:
                yield conn
            return
        mgr = RLSConnectionManager(ScoutSettings().db_app_url, use_pool=False)
        with mgr.scoped_transaction(principal) as conn:
            yield conn

    def _query_pg_code_files(
        self, principal: PrincipalContext, domain_id: str
    ) -> list[CodeFileRecord]:
        try:
            with self._pg_tx(principal) as conn:
                rows = conn.execute(
                    """
                    SELECT file_id, domain_id, repo_name, file_path, git_commit_sha,
                           git_blob_sha, language, allowed_scopes, raw_content, line_count, repo_id
                    FROM code_files
                    WHERE domain_id = %s
                    ORDER BY repo_name, file_path
                    """,
                    (domain_id,),
                ).fetchall()
            return [
                CodeFileRecord(
                    file_id=UUID(str(r[0])),
                    domain_id=str(r[1]),
                    repo_name=str(r[2]),
                    file_path=str(r[3]),
                    git_commit_sha=str(r[4]),
                    git_blob_sha=str(r[5]),
                    language=str(r[6]),
                    allowed_scopes=[str(s) for s in r[7]]
                    if isinstance(r[7], (list, tuple))
                    else [],
                    raw_content=str(r[8]),
                    line_count=int(str(r[9])),
                    repo_id=UUID(str(r[10])) if r[10] is not None else None,
                )
                for r in rows
            ]
        except Exception as exc:
            logger.warning(
                "PG17 code_files query failed for domain '%s' (%s): %s",
                domain_id,
                type(exc).__name__,
                exc,
            )
            return []

    def _query_pg_ast_symbols(
        self,
        principal: PrincipalContext,
        domain_id: str,
        symbol_ids: list[UUID] | None = None,
    ) -> list[ASTSymbolRecord]:
        try:
            with self._pg_tx(principal) as conn:
                if symbol_ids:
                    rows = conn.execute(
                        """
                        SELECT symbol_id, file_id, domain_id, qualified_name, symbol_kind,
                               start_line, end_line, signature_skeleton, contextual_summary,
                               allowed_scopes, repo_id, repo_qualified_id, bare_name
                        FROM ast_symbols
                        WHERE domain_id = %s
                          AND symbol_id = ANY(%s)
                        ORDER BY qualified_name, start_line
                        """,
                        (domain_id, symbol_ids),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """
                        SELECT symbol_id, file_id, domain_id, qualified_name, symbol_kind,
                               start_line, end_line, signature_skeleton, contextual_summary,
                               allowed_scopes, repo_id, repo_qualified_id, bare_name
                        FROM ast_symbols
                        WHERE domain_id = %s
                        ORDER BY qualified_name, start_line
                        """,
                        (domain_id,),
                    ).fetchall()
            return [
                ASTSymbolRecord(
                    symbol_id=UUID(str(r[0])),
                    file_id=UUID(str(r[1])),
                    domain_id=str(r[2]),
                    qualified_name=str(r[3]),
                    symbol_kind=str(r[4]),
                    start_line=int(str(r[5])),
                    end_line=int(str(r[6])),
                    signature_skeleton=str(r[7]),
                    contextual_summary=str(r[8]),
                    allowed_scopes=[str(s) for s in r[9]]
                    if isinstance(r[9], (list, tuple))
                    else [],
                    embedding=[],
                    repo_id=UUID(str(r[10])) if r[10] is not None else None,
                    repo_qualified_id=str(r[11]) if r[11] is not None else None,
                    bare_name=str(r[12]) if r[12] is not None else None,
                )
                for r in rows
            ]
        except Exception as exc:
            logger.warning(
                "PG17 ast_symbols query failed for domain '%s' (%s): %s",
                domain_id,
                type(exc).__name__,
                exc,
            )
            return []

    def _query_pg_ast_edges(
        self, principal: PrincipalContext, domain_id: str
    ) -> list[ASTEdgeRecord]:
        try:
            with self._pg_tx(principal) as conn:
                rows = conn.execute(
                    """
                    SELECT edge_id, domain_id, source_symbol_id, target_qualified_name,
                           target_symbol_id, edge_kind, allowed_scopes,
                           target_repo_qualified_id, is_cross_repo
                    FROM ast_edges
                    WHERE domain_id = %s
                    """,
                    (domain_id,),
                ).fetchall()
            return [
                ASTEdgeRecord(
                    edge_id=UUID(str(r[0])),
                    domain_id=str(r[1]),
                    source_symbol_id=UUID(str(r[2])),
                    target_qualified_name=str(r[3]),
                    target_symbol_id=UUID(str(r[4])) if r[4] is not None else None,
                    edge_kind=str(r[5]),
                    allowed_scopes=[str(s) for s in r[6]]
                    if isinstance(r[6], (list, tuple))
                    else [],
                    target_repo_qualified_id=str(r[7]) if r[7] is not None else None,
                    is_cross_repo=bool(r[8]),
                )
                for r in rows
            ]
        except Exception as exc:
            logger.warning(
                "PG17 ast_edges query failed for domain '%s' (%s): %s",
                domain_id,
                type(exc).__name__,
                exc,
            )
            return []

    def _query_pg_doc_chunks(
        self,
        principal: PrincipalContext,
        domain_id: str,
        market_scope: MarketRegime | None = None,
        include_stale: bool = False,
        chunk_ids: list[UUID] | None = None,
    ) -> list[DocChunkRecord]:
        try:
            with self._pg_tx(principal) as conn:
                query = """
                    SELECT chunk_id, domain_id, doc_id, doc_title, source_uri,
                           market_scope, published_date, authority_weight, section_heading,
                           contextual_header, verbatim_content, metadata_json, allowed_scopes,
                           source_type, version_str, is_stale, stale_reason
                    FROM doc_chunks
                    WHERE domain_id = %s
                """
                params: list[object] = [domain_id]
                if chunk_ids is not None:
                    if not chunk_ids:
                        return []
                    query += " AND chunk_id = ANY(%s)"
                    params.append(chunk_ids)
                if not include_stale:
                    query += " AND is_stale = FALSE"
                if market_scope is not None:
                    mkt_val = (
                        market_scope.value if hasattr(market_scope, "value") else str(market_scope)
                    )
                    query += (
                        " AND (market_scope = %s OR market_scope = 'GLOBAL' "
                        "OR market_scope = 'ENTERPRISE_DOC' OR market_scope IS NULL)"
                    )
                    params.append(mkt_val)
                rows = conn.execute(query, tuple(params)).fetchall()
            out: list[DocChunkRecord] = []
            for r in rows:
                raw_meta = r[11]
                meta_dict = (
                    json.loads(raw_meta)
                    if isinstance(raw_meta, str)
                    else (dict(raw_meta) if isinstance(raw_meta, dict) else {})
                )
                pub_d = r[6]
                if isinstance(pub_d, str):
                    pub_d = datetime.date.fromisoformat(pub_d)
                elif not isinstance(pub_d, datetime.date):
                    pub_d = None
                out.append(
                    DocChunkRecord(
                        chunk_id=UUID(str(r[0])),
                        domain_id=str(r[1]),
                        doc_id=str(r[2]),
                        doc_title=str(r[3]),
                        source_uri=str(r[4]),
                        market_scope=_deserialize_pg_market_scope(r[5]),
                        published_date=pub_d,
                        authority_weight=float(str(r[7])),
                        section_heading=str(r[8]),
                        contextual_header=str(r[9]),
                        verbatim_content=str(r[10]),
                        metadata_json=meta_dict,
                        allowed_scopes=[str(s) for s in r[12]]
                        if isinstance(r[12], (list, tuple))
                        else [],
                        embedding=[],
                        source_type=str(r[13]) if r[13] is not None else "quant_paper",
                        version_str=str(r[14]) if r[14] is not None else "1",
                        is_stale=bool(r[15]),
                        stale_reason=str(r[16]) if r[16] is not None else None,
                    )
                )
            return out
        except Exception as exc:
            logger.warning(
                "PG17 doc_chunks query failed for domain '%s' (%s): %s",
                domain_id,
                type(exc).__name__,
                exc,
            )
            return []

    def _row_to_symbol_and_file(
        self, row: tuple[object, ...]
    ) -> tuple[ASTSymbolRecord, CodeFileRecord]:
        sym = ASTSymbolRecord(
            symbol_id=UUID(str(row[0])),
            file_id=UUID(str(row[1])),
            domain_id=str(row[2]),
            qualified_name=str(row[3]),
            symbol_kind=str(row[4]),
            start_line=int(str(row[5])),
            end_line=int(str(row[6])),
            signature_skeleton=str(row[7]),
            contextual_summary=str(row[8]),
            allowed_scopes=[str(s) for s in row[9]] if isinstance(row[9], (list, tuple)) else [],
            embedding=[],
            repo_id=UUID(str(row[10])) if row[10] is not None else None,
            repo_qualified_id=str(row[11]) if row[11] is not None else None,
            bare_name=str(row[12]) if row[12] is not None else None,
        )
        cf = CodeFileRecord(
            file_id=UUID(str(row[13])),
            domain_id=str(row[14]),
            repo_name=str(row[15]),
            file_path=str(row[16]),
            git_commit_sha=str(row[17]),
            git_blob_sha=str(row[18]),
            language=str(row[19]),
            allowed_scopes=(
                [str(s) for s in row[20]] if isinstance(row[20], (list, tuple)) else []
            ),
            raw_content=str(row[21]),
            line_count=int(str(row[22])),
            repo_id=UUID(str(row[23])) if row[23] is not None else None,
        )
        return sym, cf

    def traverse_1hop_neighborhood(
        self,
        principal: PrincipalContext,
        domain_id: str,
        focus_symbol_id: UUID | str,
        max_direct_callers: int = 3,
    ) -> list[ASTNeighborhoodRecord]:
        if not self._use_pg17():
            if hasattr(self._repo, "traverse_1hop_neighborhood"):
                try:
                    res: list[ASTNeighborhoodRecord] = self._repo.traverse_1hop_neighborhood(
                        principal, domain_id, focus_symbol_id, max_direct_callers
                    )
                    return res
                except Exception as exc:
                    logger.debug(
                        "In-memory traverse_1hop_neighborhood fallback (%s): %s",
                        type(exc).__name__,
                        exc,
                    )
            return []

        focus_uuid = UUID(str(focus_symbol_id))
        try:
            with self._pg_tx(principal) as conn:
                focus_row = conn.execute(
                    """
                    SELECT s.symbol_id, s.file_id, s.qualified_name,
                           s.repo_qualified_id, f.repo_name
                    FROM ast_symbols s
                    JOIN code_files f ON s.file_id = f.file_id
                    WHERE s.domain_id = %s AND s.symbol_id = %s
                    """,
                    (domain_id, focus_uuid),
                ).fetchone()
                if focus_row is None:
                    return []
                focus_file_id = UUID(str(focus_row[1]))
                focus_qn = str(focus_row[2])
                focus_rqid = (
                    str(focus_row[3]) if focus_row[3] is not None else f"{focus_row[4]}::{focus_qn}"
                )
                focus_repo = str(focus_row[4])

                out_rows = conn.execute(
                    """
                    SELECT DISTINCT ON (ts.symbol_id, e.edge_kind)
                           ts.symbol_id, ts.file_id, ts.domain_id, ts.qualified_name,
                           ts.symbol_kind, ts.start_line, ts.end_line, ts.signature_skeleton,
                           ts.contextual_summary, ts.allowed_scopes, ts.repo_id,
                           ts.repo_qualified_id, ts.bare_name,
                           tf.file_id, tf.domain_id, tf.repo_name, tf.file_path, tf.git_commit_sha,
                           tf.git_blob_sha, tf.language, tf.allowed_scopes, tf.raw_content,
                           tf.line_count, tf.repo_id,
                           e.edge_kind
                    FROM ast_edges e
                    JOIN ast_symbols ts ON ts.domain_id = e.domain_id AND (
                        e.target_symbol_id = ts.symbol_id
                        OR (
                            e.target_symbol_id IS NULL AND e.is_cross_repo = TRUE
                            AND (
                                ts.repo_qualified_id = e.target_repo_qualified_id
                                OR ts.qualified_name = e.target_qualified_name
                            )
                        )
                        OR (
                            e.target_symbol_id IS NULL AND e.is_cross_repo = FALSE
                            AND (
                                ts.qualified_name = ltrim(e.target_qualified_name, '.')
                                OR ts.qualified_name
                                    LIKE '%%.' || ltrim(e.target_qualified_name, '.')
                            )
                        )
                    )
                    JOIN code_files tf ON ts.file_id = tf.file_id
                    WHERE e.domain_id = %s
                      AND e.source_symbol_id = %s
                      AND ts.symbol_id <> %s
                      AND (
                          e.edge_kind IN (
                              'USES_TYPE', 'IMPORTS', 'INHERITS',
                              'PROVISIONED_BY', 'DISPATCHES', 'SELECTS', 'DOCUMENTS_COMPONENT'
                          )
                          OR (
                              e.edge_kind = 'CALLS'
                              AND (e.is_cross_repo = TRUE OR tf.file_id = %s)
                          )
                      )
                      AND (e.is_cross_repo = TRUE OR tf.repo_name = %s)
                    """,
                    (domain_id, focus_uuid, focus_uuid, focus_file_id, focus_repo),
                ).fetchall()

                in_rows = conn.execute(
                    """
                    SELECT DISTINCT ON (ss.symbol_id, e.edge_kind)
                           ss.symbol_id, ss.file_id, ss.domain_id, ss.qualified_name,
                           ss.symbol_kind, ss.start_line, ss.end_line, ss.signature_skeleton,
                           ss.contextual_summary, ss.allowed_scopes, ss.repo_id,
                           ss.repo_qualified_id, ss.bare_name,
                           sf.file_id, sf.domain_id, sf.repo_name, sf.file_path, sf.git_commit_sha,
                           sf.git_blob_sha, sf.language, sf.allowed_scopes, sf.raw_content,
                           sf.line_count, sf.repo_id,
                           e.edge_kind
                    FROM ast_edges e
                    JOIN ast_symbols ss ON e.source_symbol_id = ss.symbol_id
                    JOIN code_files sf ON ss.file_id = sf.file_id
                    WHERE e.domain_id = %s
                      AND ss.symbol_id <> %s
                      AND (
                          e.edge_kind = 'CALLS'
                          OR (
                              e.is_cross_repo = TRUE
                              AND e.edge_kind IN (
                                  'CALLS', 'USES_TYPE', 'PROVISIONED_BY', 'DOCUMENTS_COMPONENT'
                              )
                          )
                      )
                      AND (
                          e.target_symbol_id = %s
                          OR (e.is_cross_repo = TRUE AND e.target_repo_qualified_id = %s)
                          OR e.target_qualified_name = %s
                      )
                      AND (e.is_cross_repo = TRUE OR sf.repo_name = %s)
                    LIMIT %s
                    """,
                    (
                        domain_id,
                        focus_uuid,
                        focus_uuid,
                        focus_rqid,
                        focus_qn,
                        focus_repo,
                        max(max_direct_callers * 4, 20),
                    ),
                ).fetchall()

                test_rows = conn.execute(
                    """
                    SELECT ts.symbol_id, ts.file_id, ts.domain_id, ts.qualified_name,
                           ts.symbol_kind, ts.start_line, ts.end_line, ts.signature_skeleton,
                           ts.contextual_summary, ts.allowed_scopes, ts.repo_id,
                           ts.repo_qualified_id, ts.bare_name,
                           tf.file_id, tf.domain_id, tf.repo_name, tf.file_path, tf.git_commit_sha,
                           tf.git_blob_sha, tf.language, tf.allowed_scopes, tf.raw_content,
                           tf.line_count, tf.repo_id,
                           e.edge_kind
                    FROM ast_edges e
                    JOIN ast_symbols ts ON (
                        e.target_symbol_id = ts.symbol_id
                        OR (e.source_symbol_id = ts.symbol_id AND e.target_symbol_id = %s)
                    )
                    JOIN code_files tf ON ts.file_id = tf.file_id
                    WHERE e.domain_id = %s
                      AND e.edge_kind = 'TESTED_BY'
                      AND (e.source_symbol_id = %s OR e.target_symbol_id = %s)
                      AND ts.symbol_id <> %s
                      AND tf.repo_name = %s
                    LIMIT 3
                    """,
                    (focus_uuid, domain_id, focus_uuid, focus_uuid, focus_uuid, focus_repo),
                ).fetchall()

            neighbors: list[ASTNeighborhoodRecord] = []
            for row in out_rows:
                sym, cf = self._row_to_symbol_and_file(row[:-1])
                neighbors.append(
                    ASTNeighborhoodRecord(
                        symbol=sym,
                        file=cf,
                        edge_kind=str(row[-1]),
                        intent_role="imported_type",
                        depth=1,
                    )
                )
            for row in in_rows:
                sym, cf = self._row_to_symbol_and_file(row[:-1])
                neighbors.append(
                    ASTNeighborhoodRecord(
                        symbol=sym,
                        file=cf,
                        edge_kind=str(row[-1]),
                        intent_role="direct_caller",
                        depth=1,
                    )
                )
            for row in test_rows:
                sym, cf = self._row_to_symbol_and_file(row[:-1])
                neighbors.append(
                    ASTNeighborhoodRecord(
                        symbol=sym,
                        file=cf,
                        edge_kind="TESTED_BY",
                        intent_role="test_fixture",
                        depth=1,
                    )
                )
            return neighbors
        except Exception as exc:
            logger.warning(
                "PG17 traverse_1hop_neighborhood failed (%s): %s",
                type(exc).__name__,
                exc,
            )
            return []

    def list_visible_code_files(
        self, principal: PrincipalContext, domain_id: str
    ) -> list[CodeFileRecord]:
        epoch = _DOMAIN_SYNC_EPOCHS.get(domain_id, 0)
        inst_cache_key = (tuple(sorted(set(principal.allowed_scopes))), domain_id, epoch)
        if inst_cache_key in self._files_cache:
            return list(self._files_cache[inst_cache_key])
        if not self._use_pg17() and hasattr(self._repo, "list_visible_code_files"):
            try:
                result = self._repo.list_visible_code_files(principal, domain_id)
                if isinstance(result, list) and (
                    result or self._has_in_memory_domain_records(domain_id)
                ):
                    typed_res = list(result)
                    self._files_cache[inst_cache_key] = typed_res
                    return list(typed_res)
            except TypeError as exc:
                logger.debug(
                    "In-memory list_visible_code_files fallback (%s): %s",
                    type(exc).__name__,
                    exc,
                )
        pg_files = self._query_pg_code_files(principal, domain_id)
        self._files_cache[inst_cache_key] = pg_files
        return list(pg_files)

    def list_visible_ast_symbols(
        self, principal: PrincipalContext, domain_id: str
    ) -> list[ASTSymbolRecord]:
        epoch = _DOMAIN_SYNC_EPOCHS.get(domain_id, 0)
        inst_cache_key = (tuple(sorted(set(principal.allowed_scopes))), domain_id, epoch)
        if inst_cache_key in self._symbols_cache:
            return list(self._symbols_cache[inst_cache_key])
        if not self._use_pg17() and hasattr(self._repo, "list_visible_ast_symbols"):
            try:
                result = self._repo.list_visible_ast_symbols(principal, domain_id)
                if isinstance(result, list) and (
                    result or self._has_in_memory_domain_records(domain_id)
                ):
                    typed_res = list(result)
                    self._symbols_cache[inst_cache_key] = typed_res
                    return list(typed_res)
            except TypeError as exc:
                logger.debug(
                    "In-memory list_visible_ast_symbols fallback (%s): %s",
                    type(exc).__name__,
                    exc,
                )
        pg_syms = self._query_pg_ast_symbols(principal, domain_id)
        self._symbols_cache[inst_cache_key] = pg_syms
        return list(pg_syms)

    def list_visible_ast_edges(
        self, principal: PrincipalContext, domain_id: str
    ) -> list[ASTEdgeRecord]:
        epoch = _DOMAIN_SYNC_EPOCHS.get(domain_id, 0)
        inst_cache_key = (tuple(sorted(set(principal.allowed_scopes))), domain_id, epoch)
        if inst_cache_key in self._edges_cache:
            return list(self._edges_cache[inst_cache_key])
        if not self._use_pg17() and hasattr(self._repo, "list_visible_ast_edges"):
            try:
                result = self._repo.list_visible_ast_edges(principal, domain_id)
                if isinstance(result, list) and (
                    result or self._has_in_memory_domain_records(domain_id)
                ):
                    typed_res = list(result)
                    self._edges_cache[inst_cache_key] = typed_res
                    return list(typed_res)
            except TypeError as exc:
                logger.debug(
                    "In-memory list_visible_ast_edges fallback (%s): %s",
                    type(exc).__name__,
                    exc,
                )
        pg_edges = self._query_pg_ast_edges(principal, domain_id)
        self._edges_cache[inst_cache_key] = pg_edges
        return list(pg_edges)

    def list_visible_doc_chunks(
        self,
        principal: PrincipalContext,
        domain_id: str,
        market_scope: MarketRegime | None = None,
        include_stale: bool = False,
    ) -> list[DocChunkRecord]:
        mkt_key = market_scope.value if market_scope is not None else None
        epoch = _DOMAIN_SYNC_EPOCHS.get(domain_id, 0)
        cache_key = (
            tuple(sorted(set(principal.allowed_scopes))),
            domain_id,
            mkt_key,
            include_stale,
            epoch,
        )
        if cache_key in self._chunks_cache:
            return list(self._chunks_cache[cache_key])
        if not self._use_pg17() and hasattr(self._repo, "list_visible_doc_chunks"):
            try:
                if include_stale:
                    try:
                        result = self._repo.list_visible_doc_chunks(
                            principal, domain_id, market_scope, include_stale=True
                        )
                    except TypeError:
                        result = self._repo.list_visible_doc_chunks(
                            principal, domain_id, market_scope
                        )
                else:
                    result = self._repo.list_visible_doc_chunks(principal, domain_id, market_scope)
                if isinstance(result, list) and (
                    result or self._has_in_memory_domain_records(domain_id)
                ):
                    typed_res = (
                        list(result)
                        if include_stale
                        else [c for c in result if not getattr(c, "is_stale", False)]
                    )
                    self._chunks_cache[cache_key] = typed_res
                    return list(typed_res)
            except TypeError as exc:
                logger.debug(
                    "In-memory list_visible_doc_chunks fallback (%s): %s",
                    type(exc).__name__,
                    exc,
                )
        pg_chunks = self._query_pg_doc_chunks(
            principal,
            domain_id,
            market_scope=market_scope,
            include_stale=include_stale,
        )
        self._chunks_cache[cache_key] = pg_chunks
        return list(pg_chunks)

    def count_visible_doc_chunks(
        self,
        principal: PrincipalContext,
        domain_id: str,
    ) -> tuple[int, int]:
        """Returns (visible_non_stale_count, stale_count) under RLS without row cap truncation."""
        if not self._use_pg17():
            if hasattr(self._repo, "count_visible_doc_chunks"):
                res = self._repo.count_visible_doc_chunks(principal, domain_id)
                if isinstance(res, tuple) and len(res) == 2:
                    return int(res[0]), int(res[1])
            if self._has_in_memory_domain_records(domain_id) or hasattr(
                self._repo, "list_visible_doc_chunks"
            ):
                all_c = self.list_visible_doc_chunks(principal, domain_id, include_stale=True)
                stale_c = sum(1 for c in all_c if bool(getattr(c, "is_stale", False)))
                return len(all_c) - stale_c, stale_c
        try:
            with self._pg_tx(principal) as conn:
                row = conn.execute(
                    """
                    SELECT
                        COUNT(*) FILTER (WHERE is_stale = FALSE),
                        COUNT(*) FILTER (WHERE is_stale = TRUE)
                    FROM doc_chunks
                    WHERE domain_id = %s;
                    """,
                    (domain_id,),
                ).fetchone()
            if row is not None:
                return int(str(row[0] or 0)), int(str(row[1] or 0))
        except Exception as exc:
            logger.debug(
                "PG17 count_visible_doc_chunks fallback for '%s' (%s): %s",
                domain_id,
                type(exc).__name__,
                exc,
            )
        raw = self.list_visible_doc_chunks(principal, domain_id, include_stale=True)
        stale_n = sum(1 for c in raw if bool(getattr(c, "is_stale", False)))
        return len(raw) - stale_n, stale_n

    def search_visible_doc_chunks_calibrated(
        self,
        principal: PrincipalContext,
        domain_id: str,
        query_text: str,
        market_scope: MarketRegime | None = None,
        limit: int = 50,
        min_calibrated_score: float = 0.0,
    ) -> tuple[list[DocChunkRecord], list[CalibratedRankedSearchHit]]:
        """Search visible doc chunks returning both records and calibrated hit scores."""
        if (
            self._use_pg17()
            and query_text.strip()
            and os.environ.get("SCOUT_USE_HASH_EMBEDDINGS_FOR_TESTS", "").strip() != "1"
        ):
            try:
                query_vec = _embed_query_for_hybrid_rrf(query_text)
                pg_repo = (
                    self._repo
                    if isinstance(self._repo, PostgresRLSRepository)
                    else PostgresRLSRepository()
                )
                with self._pg_tx(principal) as conn:
                    raw_hits = pg_repo.search_doc_chunks_hybrid_rrf(
                        conn=conn,
                        domain_id=domain_id,
                        query_text=query_text,
                        query_embedding=query_vec,
                        market_scope=market_scope,
                        top_k=limit,
                        min_calibrated_score=min_calibrated_score,
                    )
                cal_hits = [
                    h
                    if isinstance(h, CalibratedRankedSearchHit)
                    else CalibratedRankedSearchHit(
                        item_id=h.item_id,
                        qualified_name_or_title=h.qualified_name_or_title,
                        normalized_rrf_score=h.normalized_rrf_score,
                        trigram_rank=h.trigram_rank,
                        bm25_rank=h.bm25_rank,
                        vector_rank=h.vector_rank,
                        calibrated_score=h.normalized_rrf_score,
                        domain_id=domain_id,
                    )
                    for h in raw_hits
                ]
                if not cal_hits:
                    # PG17 hybrid search ran and filtered everything below the relevance floor.
                    return [], []
                ordered_ids = [str(h.item_id) for h in cal_hits]
                fetched = self.get_doc_chunks_by_ids(principal, domain_id, ordered_ids)
                by_id = {str(c.chunk_id): c for c in fetched}
                ranked_chunks = [by_id[cid] for cid in ordered_ids if cid in by_id]
                matched_hits = [h for h in cal_hits if str(h.item_id) in by_id]
                return ranked_chunks[:limit], matched_hits[:limit]
            except Exception as exc:
                logger.debug(
                    "PG17 calibrated hybrid RRF doc search fallback (%s): %s",
                    type(exc).__name__,
                    exc,
                )

        base_chunks = self.list_visible_doc_chunks(principal, domain_id, market_scope)
        if not base_chunks and market_scope is not None:
            base_chunks = self.list_visible_doc_chunks(principal, domain_id, None)
        if not base_chunks:
            return [], []
        if not query_text.strip():
            return base_chunks[:limit], []

        words = [
            w.lower()
            for w in re.findall(r"[A-Za-z0-9_]{2,}", query_text)
            if w.lower() not in PROMPT_STOPWORDS
        ]
        if not words:
            return base_chunks[:limit], []

        def _score_chunk(chk: DocChunkRecord) -> tuple[int, float]:
            title_l = chk.doc_title.lower()
            sec_l = chk.section_heading.lower()
            body_l = chk.verbatim_content.lower()
            word_hits = 0
            s = 0.0
            for w in words:
                pat = rf"\b{re.escape(w)}\b"
                in_title = bool(re.search(pat, title_l))
                in_sec = bool(re.search(pat, sec_l))
                in_body = bool(re.search(pat, body_l))
                if in_title or in_sec or in_body:
                    word_hits += 1
                if in_title:
                    s += 6.0
                if in_sec:
                    s += 3.0
                if in_body:
                    s += 1.0
            if market_scope is not None and chk.market_scope == market_scope:
                s += 5.0
            return word_hits, s + (chk.authority_weight * 0.1)

        scored_pairs = [(chk, _score_chunk(chk)) for chk in base_chunks]
        matching_pairs = [
            (chk, wh, sc) for chk, (wh, sc) in scored_pairs if wh > 0
        ]
        if not matching_pairs:
            # Zero query concepts matched any visible chunk: return empty instead of noise.
            return [], []

        matching_pairs.sort(key=lambda item: (item[1], item[2]), reverse=True)
        selected = matching_pairs[:limit]
        max_possible = max(1.0, len(words) * 1.0)
        synth_hits = []
        for idx, (chk, wh, sc) in enumerate(selected):
            cov = wh / max_possible
            cal_sc = round(min(0.95, 0.30 + 0.60 * cov), 4)
            rrf_sc = (
                round(min(0.96, max(cal_sc, 0.68 + 0.25 * cov)), 4)
                if (wh >= 2 or wh >= len(words))
                else round(min(0.95, 0.35 + 0.55 * cov), 4)
            )
            synth_hits.append(
                CalibratedRankedSearchHit(
                    item_id=chk.chunk_id,
                    qualified_name_or_title=f"{chk.doc_title} > {chk.section_heading}",
                    normalized_rrf_score=rrf_sc,
                    trigram_rank=idx + 1,
                    bm25_rank=idx + 1,
                    vector_rank=idx + 1,
                    cosine_similarity=0.0,
                    bm25_score=round(sc, 4),
                    trigram_score=0.0,
                    raw_rrf_score=rrf_sc,
                    calibrated_score=cal_sc,
                    domain_id=domain_id,
                    lexical_coverage=round(cov, 4),
                )
            )
        if min_calibrated_score > 0.0:
            filtered = [
                (chk, hit)
                for (chk, _wh, _sc), hit in zip(selected, synth_hits, strict=True)
                if hit.calibrated_score >= min_calibrated_score
            ]
            return [c for c, _ in filtered], [h for _, h in filtered]
        return [chk for chk, _wh, _sc in selected], synth_hits

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

        chunks, _ = self.search_visible_doc_chunks_calibrated(
            principal=principal,
            domain_id=domain_id,
            query_text=query_text,
            market_scope=market_scope,
            limit=limit,
        )
        return chunks

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
        if self._use_pg17() and chunk_ids:
            parsed_uuids: list[UUID] = []
            for cid in chunk_ids:
                try:
                    parsed_uuids.append(UUID(str(cid)))
                except ValueError:
                    continue
            if parsed_uuids:
                return self._query_pg_doc_chunks(
                    principal=principal,
                    domain_id=domain_id,
                    include_stale=False,
                    chunk_ids=parsed_uuids,
                )
        visible = self.list_visible_doc_chunks(principal, domain_id)
        wanted = set(chunk_ids)
        return [c for c in visible if str(c.chunk_id) in wanted]

    def search_visible_ast_symbols_calibrated(
        self,
        principal: PrincipalContext,
        domain_id: str,
        query_text: str,
        limit: int = 50,
        min_calibrated_score: float = 0.0,
    ) -> tuple[list[ASTSymbolRecord], list[CalibratedRankedSearchHit]]:
        """Search visible AST symbols returning both records and calibrated hit scores."""
        if (
            self._use_pg17()
            and query_text.strip()
            and os.environ.get("SCOUT_USE_HASH_EMBEDDINGS_FOR_TESTS", "").strip() != "1"
        ):
            try:
                query_vec = _embed_query_for_hybrid_rrf(query_text)
                pg_repo = (
                    self._repo
                    if isinstance(self._repo, PostgresRLSRepository)
                    else PostgresRLSRepository()
                )
                with self._pg_tx(principal) as conn:
                    raw_hits = pg_repo.search_symbols_hybrid_rrf(
                        conn=conn,
                        domain_id=domain_id,
                        query_text=query_text,
                        query_embedding=query_vec,
                        top_k=limit,
                        min_calibrated_score=min_calibrated_score,
                    )
                cal_hits = [
                    h
                    if isinstance(h, CalibratedRankedSearchHit)
                    else CalibratedRankedSearchHit(
                        item_id=h.item_id,
                        qualified_name_or_title=h.qualified_name_or_title,
                        normalized_rrf_score=h.normalized_rrf_score,
                        trigram_rank=h.trigram_rank,
                        bm25_rank=h.bm25_rank,
                        vector_rank=h.vector_rank,
                        calibrated_score=h.normalized_rrf_score,
                        domain_id=domain_id,
                    )
                    for h in raw_hits
                ]
                if not cal_hits:
                    return [], []
                sym_ids = [h.item_id for h in cal_hits]
                fetched = self._query_pg_ast_symbols(principal, domain_id, symbol_ids=sym_ids)
                by_id = {s.symbol_id: s for s in fetched}
                ranked_syms = [by_id[h.item_id] for h in cal_hits if h.item_id in by_id]
                matched_hits = [h for h in cal_hits if h.item_id in by_id]
                return ranked_syms[:limit], matched_hits[:limit]
            except Exception as exc:
                logger.debug(
                    "PG17 calibrated hybrid RRF symbol search fallback (%s): %s",
                    type(exc).__name__,
                    exc,
                )

        visible = self.list_visible_ast_symbols(principal, domain_id)
        if not visible:
            return [], []
        if not query_text.strip():
            return visible[:limit], []
        q_l = query_text.strip().lower()
        words = [
            w.lower()
            for w in re.findall(r"[A-Za-z0-9_]{2,}", query_text)
            if w.lower() not in PROMPT_STOPWORDS
        ]
        matched_pairs: list[tuple[ASTSymbolRecord, float, float, float]] = []
        for s in visible:
            hay = f"{s.qualified_name} {s.signature_skeleton} {s.contextual_summary}".lower()
            short_sym = s.qualified_name.split(".")[-1].lower()
            exact_short = len(short_sym) >= 3 and bool(
                re.search(rf"\b{re.escape(short_sym)}\b", q_l)
            )
            if q_l in hay or exact_short:
                matched_pairs.append((s, 0.90, 0.92, 1.0))
            elif words:
                hits_n = sum(
                    1
                    for w in words
                    if re.search(rf"\b{re.escape(w)}\b", hay) or (len(w) >= 5 and w in hay)
                )
                if hits_n > 0:
                    cov = hits_n / len(words)
                    cal_sc = round(min(0.88, 0.25 + 0.60 * cov), 4)
                    rrf_sc = (
                        round(min(0.92, max(cal_sc, 0.68 + 0.24 * cov)), 4)
                        if (hits_n >= 2 or hits_n >= len(words))
                        else cal_sc
                    )
                    matched_pairs.append((s, cal_sc, rrf_sc, round(cov, 4)))
        matched_pairs.sort(key=lambda item: (item[1], item[2]), reverse=True)
        selected = matched_pairs[:limit]
        synth_hits = [
            CalibratedRankedSearchHit(
                item_id=sym.symbol_id,
                qualified_name_or_title=sym.qualified_name,
                normalized_rrf_score=rrf_sc,
                trigram_rank=idx + 1,
                bm25_rank=idx + 1,
                vector_rank=idx + 1,
                calibrated_score=cal_sc,
                domain_id=domain_id,
                lexical_coverage=cov,
            )
            for idx, (sym, cal_sc, rrf_sc, cov) in enumerate(selected)
        ]
        if min_calibrated_score > 0.0:
            filtered = [
                (sym, hit)
                for (sym, _cal, _rrf, _cov), hit in zip(selected, synth_hits, strict=True)
                if hit.calibrated_score >= min_calibrated_score
            ]
            return [s for s, _ in filtered], [h for _, h in filtered]
        return [sym for sym, _cal, _rrf, _cov in selected], synth_hits

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
        syms, _ = self.search_visible_ast_symbols_calibrated(
            principal=principal,
            domain_id=domain_id,
            query_text=query_text,
            limit=limit,
        )
        return syms

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
    for item in report.missing_items:
        if item.identifier not in missing:
            missing.append(item.identifier)

    reason = (
        custom_reason
        or report.bounce_back_feedback
        or "Deterministic pre-flight quality check failed"
    )
    files = partial_candidate_files or []

    failure_check: str | None = None
    for k, v in report.check_results.items():
        if not v:
            failure_check = k
            break

    return InsufficientContextDiagnostic(
        trace_id=trace_id,
        domain_id=domain_id,
        reason=reason,
        missing_symbols=missing,
        partial_candidate_files=files,
        missing_items=list(report.missing_items),
        failure_check=failure_check,
        paid_tokens_spent=0,
    )


class PreFlightValidator:
    """4-Check Pre-Flight Quality Gate ($0 local quality barrier).

    Enforces deterministic validation of symbol closure, prompt identifier coverage,
    normalized retrieval score, and verbatim document citations before paid frontier handoff.
    """

    DEFAULT_MIN_RRF_SCORE: Final[float] = 0.65
    DEFAULT_CROSS_ENCODER_SCORE_THRESHOLD: Final[float] = 0.60
    DEFAULT_MIN_CITATION_OVERLAP: Final[float] = 1.0

    def __init__(
        self,
        repo: Any = None,
        slicer: DeterministicSlicer | None = None,
        expander: ASTDependencyExpander | None = None,
        min_rrf_score: float | None = None,
        cross_encoder_threshold: float = DEFAULT_CROSS_ENCODER_SCORE_THRESHOLD,
        min_citation_overlap: float = DEFAULT_MIN_CITATION_OVERLAP,
        principal: PrincipalContext | None = None,
        domain_id: str | None = None,
        domain_config: DomainConfig | None = None,
        citation_gate: DocumentCitationGate | None = None,
    ) -> None:
        self.repo = repo
        self.slicer = slicer or DeterministicSlicer()
        self.domain_config = domain_config or DomainConfig(
            domain_id=domain_id or ScoutSettings().default_domain_id
        )
        self.citation_gate = citation_gate or DocumentCitationGate(
            min_overlap_ratio=min_citation_overlap,
            require_citations_default=False,
            strict_freshness=self.domain_config.confluence_strict_freshness,
        )
        self.expander = expander or ASTDependencyExpander()
        self.min_rrf_score = (
            min_rrf_score
            if min_rrf_score is not None
            else self.domain_config.min_retrieval_confidence
        )
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
        if isinstance(active_repo, BoundGraphRepository) and conn is None:
            return active_repo
        return BoundGraphRepository(active_repo, conn=conn)

    def _resolve_principal(
        self,
        principal: PrincipalContext | None,
    ) -> PrincipalContext:
        if principal is not None:
            return principal
        if self.principal is not None:
            return self.principal
        from context_scout.plugins.registry import PluginRegistry

        default_scopes = ["public_eng"]
        primary_plugin = PluginRegistry.get_primary_domain_plugin()
        if primary_plugin is not None:
            for scope in primary_plugin.get_default_scopes():
                if scope not in default_scopes:
                    default_scopes.append(scope)
        return PrincipalContext(
            user_id="preflight_system",
            role=RoleType.ENGINEER,
            allowed_scopes=default_scopes,
        )

    def extract_prompt_candidates(self, user_prompt: str) -> tuple[set[str], set[str]]:
        """Extracts candidate tokens, separating explicitly quoted code tokens from unquoted text.

        Returns (explicitly_quoted_tokens, unquoted_candidate_tokens).
        """
        if not user_prompt:
            return set(), set()

        # Strip URLs before extracting paths or backticks to avoid treating
        # URL schemes/domains as file paths
        url_tlds = ("org", "com", "net", "io", "edu", "gov", "spec")

        def _strip_url_token(match: re.Match[str]) -> str:
            span = match.group(0)
            low_span = span.lower()
            if low_span.startswith(("http://", "https://", "ftp://", "git://", "file://", "//")):
                return " "
            host_part = low_span.split("/", 1)[0].strip(".,;:()[]{}")
            if "." in host_part:
                host_labels = host_part.split(".")
                if any(lbl in url_tlds for lbl in host_labels[1:]):
                    return " "
            return span

        prompt_clean = re.sub(r"[^\s`'\"]+", _strip_url_token, user_prompt)

        # 1. Backticked tokens (`symbol_or_path`), ignoring multi-line fenced code blocks
        fence_chunks: list[str] = []
        f_pos = 0
        while True:
            f_open = prompt_clean.find("```", f_pos)
            if f_open == -1:
                fence_chunks.append(prompt_clean[f_pos:])
                break
            fence_chunks.append(prompt_clean[f_pos:f_open])
            f_close = prompt_clean.find("```", f_open + 3)
            if f_close == -1:
                fence_chunks.append(prompt_clean[f_open:])
                break
            fence_chunks.append(" ")
            f_pos = f_close + 3
        prompt_without_fences = "".join(fence_chunks)
        backtick_tokens = set(re.findall(r"`([^`\n]+)`", prompt_without_fences))

        # 2. File paths (e.g., src/strategies/order_flow.py or models.py) in linear O(N) time
        standalone_exts = {
            "py",
            "ts",
            "tsx",
            "js",
            "jsx",
            "json",
            "md",
            "sql",
            "yaml",
            "yml",
            "toml",
        }
        path_tokens: set[str] = set()
        for raw_path in re.findall(r"[a-zA-Z0-9_./-]+", prompt_without_fences):
            cand_path = raw_path.strip(".")
            if not cand_path:
                continue
            if "/" in cand_path:
                dir_part, leaf = cand_path.rsplit("/", 1)
                if dir_part and "." in leaf:
                    stem, ext = leaf.rsplit(".", 1)
                    if stem and ext and ext.isalnum():
                        path_tokens.add(cand_path)
            elif "." in cand_path:
                stem, ext = cand_path.rsplit(".", 1)
                if stem and ext.lower() in standalone_exts:
                    path_tokens.add(cand_path)

        explicit_tokens: set[str] = set()
        for tok in backtick_tokens | path_tokens:
            cleaned = tok.strip(" `'\";:(),[]{}")
            if cleaned.endswith("()"):
                cleaned = cleaned[:-2].strip()
            if not cleaned or len(cleaned) < 2:
                continue
            if any(ch.isspace() for ch in cleaned):
                continue
            # Filter out tokens with no alphanumeric characters
            # (pure operators/punctuation like <=, ==, !=, ->, ...)
            if not any(ch.isalnum() for ch in cleaned):
                continue
            # Filter out URL artifact prefixes
            if cleaned.startswith(("//", "http:", "https:", "ftp:")) or "://" in cleaned:
                continue
            # Filter out comparison / assignment / arrow operators embedded without spaces
            if re.match(
                r"^(?:<=|>=|==|!=|<=>|===|!==|->|=>|::|\+\+|--|\+=|-=|\*=|/=|&&|\|\|)$",
                cleaned,
            ):
                continue
            # Filter out percent-encoded artifacts (e.g., %3A, %40)
            if cleaned.startswith("%") or re.match(r"^%[0-9a-fA-F]{2}$", cleaned):
                continue
            # Strip trailing = from keyword argument expressions like safe=
            if cleaned.endswith("="):
                cleaned = cleaned.rstrip("=").strip()
            if not cleaned or len(cleaned) < 2:
                continue
            if cleaned.lower() in PROMPT_STOPWORDS or cleaned.lower() in {"safe", "kwargs", "args"}:
                continue
            explicit_tokens.add(cleaned)

        # 3. Candidate identifiers from prompt body
        raw_words = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b", prompt_clean)
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
            # Filter out common URL components and web protocols
            if lower_val in {
                "http",
                "https",
                "github",
                "gitlab",
                "whatwg",
                "spec",
                "org",
                "com",
                "net",
                "io",
                "www",
            }:
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

        if not all_candidates or (not visible_symbols and not visible_files):
            return [], []

        # Build repository symbol and file lookup indices (scoped to target repo when multi-repo)
        target_repos: set[str] = {
            s.repo_name
            for s in code_slices
            if s.modification_intent == ModificationIntent.MODIFY_TARGET
            or str(s.modification_intent) == "modify_target"
        }
        distinct_all_repos = {f.repo_name for f in visible_files}
        scope_to_target_repo = bool(target_repos and len(distinct_all_repos) > 1)
        scoped_files = [
            f for f in visible_files if not scope_to_target_repo or f.repo_name in target_repos
        ]
        scoped_file_ids = {f.file_id for f in scoped_files}

        known_symbols_by_name: dict[str, ASTSymbolRecord] = {}
        known_symbols_by_short: dict[str, ASTSymbolRecord] = {}
        for sym in visible_symbols:
            if scope_to_target_repo and sym.file_id not in scoped_file_ids:
                continue
            if len(scoped_files) > 15 and sym.qualified_name.startswith(("tests.", "examples.")):
                continue
            known_symbols_by_name[sym.qualified_name] = sym
            parts = sym.qualified_name.split(".")
            short_name = parts[-1]
            known_symbols_by_short[short_name] = sym
            if len(parts) >= 2:
                known_symbols_by_name[".".join(parts[-2:])] = sym

        known_files_by_path: dict[str, CodeFileRecord] = {}
        known_files_by_name: dict[str, CodeFileRecord] = {}
        known_files_by_stem: dict[str, CodeFileRecord] = {}
        known_dirs: set[str] = set()
        for f in scoped_files:
            known_files_by_path[f.file_path] = f
            file_name = f.file_path.split("/")[-1]
            known_files_by_name[file_name] = f
            known_files_by_stem[file_name.rsplit(".", 1)[0]] = f
            known_dirs.update(part for part in f.file_path.split("/")[:-1] if part and part != ".")

        missing_reference_identifiers: list[str] = []
        novel_target_identifiers: list[str] = []
        has_creation_verb = bool(
            re.search(
                r"\b(?:add|create|creating|implement|implementing|introduce|introducing|write|writing|define|defining|generate|generating|build|building|provision|configure|setup|deploy|declare|output|outputting|save|saving|export|exporting)\b",
                user_prompt,
                re.IGNORECASE,
            )
        )

        for candidate in sorted(all_candidates):
            # Ignore tokens with no alphanumeric characters or URL fragments
            if not any(ch.isalnum() for ch in candidate):
                continue
            if candidate.startswith(("//", "http:", "https:")):
                continue
            if (candidate in PYTHON_BUILTINS or candidate.startswith("WinError")) and (
                candidate not in known_symbols_by_name and candidate not in known_symbols_by_short
            ):
                continue

            is_explicit = candidate in explicit_candidates
            is_camel_or_pascal = any(c.isupper() for c in candidate[1:]) and any(
                c.islower() for c in candidate
            )
            is_code_shaped = (
                is_camel_or_pascal
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
                or (
                    candidate in known_files_by_stem
                    and (is_explicit or is_code_shaped or "." in candidate or "/" in candidate)
                )
                or (candidate in known_dirs and (is_explicit or is_code_shaped or "/" in candidate))
                or any(f.endswith("/" + candidate) for f in known_files_by_path)
            )

            if is_existing_symbol or is_existing_file:
                # Existing Reference Symbol/File -> Must be covered in code_slices
                covered = False
                for s in code_slices:
                    if (
                        s.qualified_symbol == candidate
                        or s.qualified_symbol.endswith("." + candidate)
                        or s.qualified_symbol.endswith(":" + candidate)
                    ):
                        covered = True
                        break
                    if (
                        s.file_path == candidate
                        or s.file_path.endswith("/" + candidate)
                        or s.file_path.split("/")[-1].rsplit(".", 1)[0] == candidate
                        or candidate in s.file_path.split("/")
                    ):
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
                    # Dot-separated identifier component coverage (e.g. Class.method)
                    if "." in candidate and all(
                        re.search(r"\b" + re.escape(part) + r"\b", s.verbatim_code_with_lines)
                        or s.qualified_symbol.endswith("." + part)
                        or s.qualified_symbol.endswith(":" + part)
                        for part in candidate.split(".")
                    ):
                        covered = True
                        break

                if not covered:
                    missing_reference_identifiers.append(candidate)
            else:
                # Candidate is NOT in repository symbol or file index.
                # If candidate is already covered in retrieved code_slices, it is not missing.
                covered = False
                for s in code_slices:
                    if (
                        s.qualified_symbol == candidate
                        or s.qualified_symbol.endswith("." + candidate)
                        or s.qualified_symbol.endswith(":" + candidate)
                    ):
                        covered = True
                        break
                    if (
                        s.file_path == candidate
                        or s.file_path.endswith("/" + candidate)
                        or s.file_path.split("/")[-1].rsplit(".", 1)[0] == candidate
                        or candidate in s.file_path.split("/")
                    ):
                        covered = True
                        break
                    if re.search(r"\b" + re.escape(candidate) + r"\b", s.verbatim_code_with_lines):
                        covered = True
                        break
                    if "." in candidate and all(
                        re.search(r"\b" + re.escape(part) + r"\b", s.verbatim_code_with_lines)
                        or s.qualified_symbol.endswith("." + part)
                        or s.qualified_symbol.endswith(":" + part)
                        for part in candidate.split(".")
                    ):
                        covered = True
                        break

                if covered:
                    novel_target_identifiers.append(candidate)
                    continue

                # Partition novel identifiers to be created vs unindexed target symbols referenced
                # in modification, inspection, refactoring, or deletion tasks.
                is_novel_creation = bool(
                    re.search(
                        r"\b(?:add|create|creating|implement|implementing|introduce|introducing|write|writing|define|defining|generate|generating|build|building|provision|configure|setup|deploy|declare|output|outputting|save|saving|export|exporting)\s+"
                        r"(?:a\s+|an\s+|the\s+)?(?:new\s+|novel\s+|brand\s+new\s+)?"
                        r"(?:[a-zA-Z0-9_\-]+\s+){0,4}"
                        r"[`'\"]?" + re.escape(candidate) + r"[`'\"]?\b",
                        user_prompt,
                        re.IGNORECASE,
                    )
                    or re.search(
                        r"\b(?:novel|brand\s+new|new)\s+(?:file|target|module|script|class|interface|type|struct|component)\s+"
                        r"[`'\"]?" + re.escape(candidate) + r"[`'\"]?\b",
                        user_prompt,
                        re.IGNORECASE,
                    )
                    or re.search(
                        r"\b(?:writing|outputting|saving|exporting)\s+(?:output\s+)?(?:to\s+)?(?:novel\s+|new\s+)?(?:file\s+)?[`'\"]?"
                        + re.escape(candidate)
                        + r"[`'\"]?\b",
                        user_prompt,
                        re.IGNORECASE,
                    )
                    or (
                        has_creation_verb
                        and re.search(
                            r"\b(?:and|or|,)\s+(?:a\s+|an\s+|the\s+)?(?:new\s+|novel\s+)?(?:helper\s+)?"
                            r"(?:method|function|class|interface|type|struct|file)?\s*"
                            r"[`'\"]?" + re.escape(candidate) + r"[`'\"]?\b",
                            user_prompt,
                            re.IGNORECASE,
                        )
                    )
                )
                has_modification_context = bool(
                    re.search(
                        r"\b(?:refactor|modify|modifying|update|updating|inspect|inspecting|debug|debugging|fix|fixing|patch|patching|trace|tracing|delete|deleting|remove|removing)\b",
                        user_prompt,
                        re.IGNORECASE,
                    )
                )
                is_modification_inspection = bool(
                    re.search(
                        r"\b(?:refactor|modify|update|inspect|debug|fix|patch|trace|delete|remove|call|import|export|from|into|in|within)\s+"
                        r"(?:class|struct|function|def|fn|method|interface|type|file|module|symbol|component|service|handler|manager|controller|repo|repository|domain)?\s*"
                        r"[`'\"]?" + re.escape(candidate) + r"[`'\"]?\b",
                        user_prompt,
                        re.IGNORECASE,
                    )
                    or re.search(
                        r"\b(?:to|with|using|via|against|from|into|and|call|calling|dispatch|dispatching|send|sending|emit|emitting|delegate|delegating|forward|forwarding|consume|consuming|subscribe|subscribing|publish|publishing|notify|notifying)\s+"
                        r"(?:events?\s+to\s+|messages?\s+to\s+|calls?\s+to\s+|requests?\s+to\s+|data\s+to\s+)?"
                        r"(?:class|struct|function|def|fn|method|interface|type|file|module|symbol|component|service|handler|manager|controller|repo|repository|domain|sink|source|queue|channel|listener)?\s*"
                        r"[`'\"]?" + re.escape(candidate) + r"[`'\"]?\b",
                        user_prompt,
                        re.IGNORECASE,
                    )
                    or re.search(
                        r"\b(?:class|def|fn|function|method|interface|type|struct|enum|file|module)\s+[`'\"]?"
                        + re.escape(candidate)
                        + r"[`'\"]?\b",
                        user_prompt,
                        re.IGNORECASE,
                    )
                    or (has_modification_context and is_camel_or_pascal)
                    or (
                        "/" in candidate
                        or candidate.endswith(
                            (
                                ".py",
                                ".ts",
                                ".tsx",
                                ".js",
                                ".jsx",
                                ".cs",
                                ".go",
                                ".rs",
                                ".java",
                                ".json",
                                ".yaml",
                                ".yml",
                                ".toml",
                                ".sql",
                                ".md",
                            )
                        )
                    )
                )
                if (
                    (is_explicit or is_code_shaped)
                    and is_modification_inspection
                    and not is_novel_creation
                ):
                    missing_reference_identifiers.append(candidate)
                else:
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
                if match_symbol_identifier(target_slice.qualified_symbol, sym_rec.qualified_name):
                    focus_sym = sym_rec
                    break

            if focus_sym is None:
                # Fallback: short name + file path match
                target_short = target_slice.qualified_symbol.split(".")[-1]
                for sym_rec in visible_symbols:
                    if (
                        (
                            sym_rec.qualified_name.split(".")[-1] == target_short
                            or sym_rec.qualified_name.split("::")[-1] == target_short
                        )
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

            focus_file_rec = visible_files_by_id.get(focus_sym.file_id)
            focus_repo_name = focus_file_rec.repo_name if focus_file_rec is not None else None

            for edge in target_edges:
                target_sym: ASTSymbolRecord | None = None
                is_cross = bool(getattr(edge, "is_cross_repo", False))
                if edge.target_symbol_id and edge.target_symbol_id in visible_symbols_by_id:
                    cand_s = visible_symbols_by_id[edge.target_symbol_id]
                    cand_f = visible_files_by_id.get(cand_s.file_id)
                    if (
                        not is_cross
                        and focus_repo_name is not None
                        and cand_f is not None
                        and cand_f.repo_name != focus_repo_name
                    ):
                        continue
                    target_sym = cand_s
                elif edge.target_symbol_id is None:
                    if edge.edge_kind == "CALLS":
                        continue
                    rel_suffix = edge.target_qualified_name.lstrip(".")
                    target_clean = edge.target_qualified_name.split(".")[-1]
                    for sym_rec in visible_symbols:
                        sf = visible_files_by_id.get(sym_rec.file_id)
                        if (
                            focus_repo_name is not None
                            and sf is not None
                            and sf.repo_name != focus_repo_name
                            and not getattr(edge, "is_cross_repo", False)
                            and "::" not in edge.target_qualified_name
                        ):
                            continue
                        if (
                            sym_rec.qualified_name == rel_suffix
                            or sym_rec.qualified_name.endswith("." + rel_suffix)
                            or match_symbol_identifier(rel_suffix, sym_rec.qualified_name)
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
                            sf = visible_files_by_id.get(sym_rec.file_id)
                            if (
                                focus_repo_name is not None
                                and sf is not None
                                and sf.repo_name != focus_repo_name
                                and not getattr(edge, "is_cross_repo", False)
                                and "::" not in edge.target_qualified_name
                            ):
                                continue
                            if (
                                sym_rec.qualified_name == edge.target_qualified_name
                                or match_symbol_identifier(
                                    edge.target_qualified_name, sym_rec.qualified_name
                                )
                                or (
                                    sym_rec.qualified_name.split(".")[-1] == target_clean
                                    and sym_rec.symbol_kind
                                    in ("class", "interface", "type_alias", "function")
                                )
                            ):
                                target_sym = sym_rec
                                break

                # If this is an internal repository symbol, verify it is present in code_slices
                if target_sym is not None:
                    if (
                        edge.edge_kind == "CALLS"
                        and not is_cross
                        and target_sym.file_id != focus_sym.file_id
                    ):
                        continue
                    sym_clean = target_sym.qualified_name.split(".")[-1].split("::")[-1]
                    edge_clean = edge.target_qualified_name.split(".")[-1].split("::")[-1]
                    is_covered = (
                        target_sym.qualified_name in covered_symbols
                        or sym_clean in covered_symbols
                        or edge.target_qualified_name in covered_symbols
                        or edge_clean in covered_symbols
                        or any(
                            match_symbol_identifier(c, target_sym.qualified_name)
                            for c in covered_symbols
                        )
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
        cross_encoder_score: float | None = None,
    ) -> tuple[bool, float]:
        """Verifies top Normalized RRF score >= 0.65 or cross-encoder score >= 0.60."""
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

        # Build chunk collection preserving full DocChunkRecord metadata
        corpus_chunks_for_gate: list[Any] | dict[str, str]
        if isinstance(doc_chunks, list):
            corpus_chunks_for_gate = list(doc_chunks)
        elif isinstance(doc_chunks, dict):
            corpus_chunks_for_gate = {str(k): str(v) for k, v in doc_chunks.items()}
        else:
            corpus_chunks_for_gate = list(bound_repo.list_visible_doc_chunks(principal, domain_id))

        known_ids: set[str] = set()
        if isinstance(corpus_chunks_for_gate, list):
            known_ids = {str(getattr(c, "chunk_id", "")) for c in corpus_chunks_for_gate}
        elif isinstance(corpus_chunks_for_gate, dict):
            known_ids = set(corpus_chunks_for_gate.keys())

        missing_ids = [str(cit.chunk_id) for cit in citations if str(cit.chunk_id) not in known_ids]
        if missing_ids and bound_repo._repo is not None:
            candidate_domains = [domain_id] if domain_id else []
            for scope in principal.allowed_scopes:
                if scope.startswith("domain:"):
                    target_dom = scope.split(":", 1)[1].strip()
                    if target_dom and target_dom not in candidate_domains:
                        candidate_domains.append(target_dom)
            for lookup_domain in dict.fromkeys(candidate_domains):
                fetched = bound_repo.get_doc_chunks_by_ids(
                    principal=principal,
                    domain_id=lookup_domain,
                    chunk_ids=missing_ids,
                )
                if isinstance(corpus_chunks_for_gate, list):
                    corpus_chunks_for_gate.extend(fetched)
                elif isinstance(corpus_chunks_for_gate, dict):
                    for fc in fetched:
                        corpus_chunks_for_gate[str(fc.chunk_id)] = fc.verbatim_content
                for f in fetched:
                    known_ids.add(str(getattr(f, "chunk_id", "")))
                missing_ids = [m for m in missing_ids if m not in known_ids]
                if not missing_ids:
                    break

        res = self.citation_gate.verify_citations(
            citations=citations,
            corpus_chunks=corpus_chunks_for_gate,
            domain_id=domain_id,
            require_citations=False,
        )
        failed_reasons = []
        for _, r in res.failed_citations:
            if (
                "failed verbatim verification" not in r
                and "not in corpus" not in r
                and "rejected" not in r
            ):
                failed_reasons.append(f"Quote failed verbatim verification: {r}")
            else:
                failed_reasons.append(r)
        return res.passed, failed_reasons

    def _is_valid_partial_window_truncation(
        self,
        cleaned_code: str,
        slice_item: VerbatimCodeSlice,
        bound_repo: Any | None = None,
        principal: Any | None = None,
        domain_id: str | None = None,
    ) -> bool:
        """Determines if unclosed delimiters in a partial caller/type window
        are an artifact of multi-line call/expression window truncation,
        rather than a broken single-line syntax error.
        """
        lines = [ln.strip() for ln in cleaned_code.splitlines() if ln.strip()]
        if not lines:
            return False

        # 1. Structural check: multi-line expression ending cleanly on an argument or comma
        last_line = lines[-1]
        if len(lines) >= 2 and last_line.endswith(","):
            return True

        # 2. Repository check: if underlying symbol exists in repository,
        # check if expanding to full symbol boundaries closes the delimiter
        repo_target = bound_repo or self.repo
        if repo_target is not None:
            try:
                b_repo = self._resolve_graph_repo(repo_target)
                p_ctx = self._resolve_principal(principal or self.principal)
                domain = (
                    domain_id
                    or self.domain_id
                    or (
                        self.domain_config.domain_id
                        if self.domain_config
                        else ScoutSettings().default_domain_id
                    )
                )

                candidate_domains = [domain] if domain else []
                for scope in p_ctx.allowed_scopes:
                    if scope.startswith("domain:"):
                        d = scope.split(":", 1)[1].strip()
                        if d and d not in candidate_domains:
                            candidate_domains.append(d)
                if not candidate_domains:
                    candidate_domains = [
                        self.domain_config.domain_id
                        if self.domain_config
                        else ScoutSettings().default_domain_id
                    ]

                for dom in candidate_domains:
                    files = b_repo.list_visible_code_files(p_ctx, dom)
                    file_rec = next((f for f in files if f.file_path == slice_item.file_path), None)
                    if file_rec is not None:
                        syms = b_repo.list_visible_ast_symbols(p_ctx, dom)
                        matching_sym = next(
                            (
                                s
                                for s in syms
                                if s.qualified_name == slice_item.qualified_symbol
                                and s.file_id == file_rec.file_id
                            ),
                            None,
                        )
                        if matching_sym is not None:
                            full_lines = file_rec.raw_content.splitlines()[
                                matching_sym.start_line - 1 : matching_sym.end_line
                            ]
                            full_code = "\n".join(full_lines)
                            if not self.slicer._has_unclosed_delimiters(
                                full_code, file_path=slice_item.file_path
                            ):
                                return True
            except Exception as e:
                logger.debug("Failed symbol boundary expansion check: %s", e)

        return False

    def check_ast_syntax(
        self,
        code_slices: list[VerbatimCodeSlice],
        bound_repo: Any | None = None,
        principal: Any | None = None,
        domain_id: str | None = None,
    ) -> tuple[bool, list[str], list[MissingItemRecord]]:
        """Check 1: Verifies all code slices parse cleanly under language parser."""
        import textwrap

        syntax_errors: list[str] = []
        missing_items: list[MissingItemRecord] = []

        py_parser = None
        ts_parser = None
        tsx_parser = None
        try:
            import tree_sitter_python as tspython
            import tree_sitter_typescript as tstypescript
            from tree_sitter import Language, Parser

            py_parser = Parser(Language(tspython.language()))
            ts_parser = Parser(Language(tstypescript.language_typescript()))
            tsx_parser = Parser(Language(tstypescript.language_tsx()))
        except Exception as e:
            logger.debug("Tree-sitter parser initialization: %s", e)

        for s in code_slices:
            raw_lines: list[str] = []
            for line in s.verbatim_code_with_lines.splitlines():
                m = re.match(r"^\s*\d+\s*\|\s?(.*)$", line)
                raw_lines.append(m.group(1) if m else line)
            cleaned_code = textwrap.dedent("\n".join(raw_lines)).strip()
            if not cleaned_code:
                continue

            has_error = False
            err_msg = ""

            is_partial_window = s.modification_intent in (
                ModificationIntent.DIRECT_CALLER,
                ModificationIntent.IMPORTED_TYPE,
            )

            stripped_code, unclosed_quote = self.slicer._strip_comments_and_strings(
                cleaned_code, file_path=s.file_path
            )
            has_delim_error = self.slicer._has_unclosed_delimiters(
                cleaned_code, file_path=s.file_path
            )

            if unclosed_quote or has_delim_error:
                if is_partial_window:
                    if not self._is_valid_partial_window_truncation(
                        cleaned_code,
                        s,
                        bound_repo=bound_repo,
                        principal=principal,
                        domain_id=domain_id,
                    ):
                        has_error = True
                        err_msg = "Unclosed syntax delimiters in code slice"
                else:
                    has_error = True
                    err_msg = "Unclosed syntax delimiters in code slice"

            # Check parentheses nesting depth (> 10 levels) on sanitized code
            # (excluding string literals & comments)
            depth = 0
            max_depth = 0
            for ch in stripped_code:
                if ch in "([{":
                    depth += 1
                    max_depth = max(max_depth, depth)
                elif ch in ")]}":
                    depth = max(0, depth - 1)
            if max_depth > 10:
                has_error = True
                err_msg = f"Nesting depth ({max_depth}) exceeds maximum limit (10)"

            # Whole-module tree-sitter parsing is only performed on full slices
            # when no delimiter/nesting error exists
            if not has_error and not is_partial_window:
                ext = Path(s.file_path).suffix.lower()
                code_b = cleaned_code.encode("utf-8")
                if ext in (".py",) and py_parser is not None:
                    tree = py_parser.parse(code_b)
                    if tree.root_node.has_error:
                        has_error = True
                        err_msg = "Tree-sitter Python parser reported syntax error"
                elif ext in (".ts", ".js") and ts_parser is not None:
                    tree = ts_parser.parse(code_b)
                    if tree.root_node.has_error:
                        has_error = True
                        err_msg = "Tree-sitter TypeScript parser reported syntax error"
                elif ext in (".tsx", ".jsx") and tsx_parser is not None:
                    tree = tsx_parser.parse(code_b)
                    if tree.root_node.has_error:
                        has_error = True
                        err_msg = "Tree-sitter TSX parser reported syntax error"
            if has_error:
                ident = s.qualified_symbol or s.file_path
                diag_msg = f"MALFORMED_SYNTAX in {s.file_path} ({ident}): {err_msg}"
                syntax_errors.append(diag_msg)
                missing_items.append(
                    MissingItemRecord(
                        kind="syntax",
                        identifier=ident,
                        reason=diag_msg,
                        suggested_action="Fix syntax or adjust slice bounds to parse cleanly",
                    )
                )

        return len(syntax_errors) == 0, syntax_errors, missing_items

    def check_principal_scopes(
        self,
        code_slices: list[VerbatimCodeSlice],
        citations: list[DocCitationSpan],
        bound_repo: BoundGraphRepository,
        principal: PrincipalContext,
        domain_id: str,
        doc_chunks: dict[str, str] | list[DocChunkRecord] | None = None,
    ) -> tuple[
        bool,
        list[VerbatimCodeSlice],
        list[DocCitationSpan],
        list[str],
        list[MissingItemRecord],
    ]:
        """Check 4: Principal scope mirroring. Excises unauthorized items."""
        excised_items: list[str] = []
        missing_items: list[MissingItemRecord] = []
        allowed_set = set(principal.allowed_scopes)

        visible_files = bound_repo.list_visible_code_files(principal, domain_id)
        file_scopes_map = {f.file_path: set(f.allowed_scopes) for f in visible_files}

        permitted_slices: list[VerbatimCodeSlice] = []
        for s in code_slices:
            if s.file_path not in file_scopes_map:
                excised_items.append(f"code:{s.repo_name}/{s.file_path}")
                missing_items.append(
                    MissingItemRecord(
                        kind="scope",
                        identifier=f"{s.repo_name}/{s.file_path}",
                        reason=(
                            f"Principal '{principal.user_id}' lacks visibility "
                            f"into file '{s.file_path}'"
                        ),
                    )
                )
                continue
            f_scopes = file_scopes_map[s.file_path]
            if bool(allowed_set & f_scopes) or "*" in allowed_set or "admin" in allowed_set:
                permitted_slices.append(s)
            else:
                excised_items.append(f"code:{s.repo_name}/{s.file_path}")
                missing_items.append(
                    MissingItemRecord(
                        kind="scope",
                        identifier=f"{s.repo_name}/{s.file_path}",
                        reason=(
                            f"Principal '{principal.user_id}' lacks required "
                            f"scopes {sorted(f_scopes)}"
                        ),
                    )
                )

        visible_chunks = bound_repo.list_visible_doc_chunks(
            principal, domain_id, include_stale=True
        )
        chunk_scopes_map = {str(c.chunk_id): set(c.allowed_scopes) for c in visible_chunks}
        if isinstance(doc_chunks, list):
            for c in doc_chunks:
                if isinstance(c, DocChunkRecord) and hasattr(c, "allowed_scopes"):
                    chunk_scopes_map[str(c.chunk_id)] = set(c.allowed_scopes)

        citation_ids = [
            str(c.chunk_id)
            for c in citations
            if c.chunk_id and str(c.chunk_id) not in chunk_scopes_map
        ]
        if citation_ids and bound_repo._repo is not None:
            candidate_domains = [domain_id] if domain_id else []
            for scope in principal.allowed_scopes:
                if scope.startswith("domain:"):
                    target_dom = scope.split(":", 1)[1].strip()
                    if target_dom and target_dom not in candidate_domains:
                        candidate_domains.append(target_dom)
            for lookup_domain in dict.fromkeys(candidate_domains):
                specific_chunks = bound_repo.get_doc_chunks_by_ids(
                    principal=principal,
                    domain_id=lookup_domain,
                    chunk_ids=citation_ids,
                )
                for sc in specific_chunks:
                    chunk_scopes_map[str(sc.chunk_id)] = set(sc.allowed_scopes)
                citation_ids = [cid for cid in citation_ids if cid not in chunk_scopes_map]
                if not citation_ids:
                    break

        permitted_citations: list[DocCitationSpan] = []
        for cit in citations:
            c_scopes = chunk_scopes_map.get(str(cit.chunk_id))
            if c_scopes is None:
                excised_items.append(f"doc:{cit.doc_id}:{cit.chunk_id}")
                missing_items.append(
                    MissingItemRecord(
                        kind="scope",
                        identifier=f"{cit.doc_id}:{cit.chunk_id}",
                        reason=(
                            f"Principal '{principal.user_id}' lacks visibility "
                            f"into document chunk '{cit.chunk_id}'"
                        ),
                    )
                )
            elif bool(allowed_set & c_scopes) or "*" in allowed_set or "admin" in allowed_set:
                permitted_citations.append(cit)
            else:
                excised_items.append(f"doc:{cit.doc_id}:{cit.chunk_id}")
                missing_items.append(
                    MissingItemRecord(
                        kind="scope",
                        identifier=f"{cit.doc_id}:{cit.chunk_id}",
                        reason=(
                            f"Principal '{principal.user_id}' lacks required "
                            f"scopes {sorted(c_scopes)}"
                        ),
                    )
                )

        passed = True
        if len(code_slices) > 0 and len(permitted_slices) == 0:
            passed = False
        if len(citations) > 0 and len(permitted_citations) == 0:
            passed = False
        if len(excised_items) > 0 and (
            len(permitted_slices) == 0 and len(permitted_citations) == 0
        ):
            passed = False
        if any(
            (
                s.modification_intent == ModificationIntent.MODIFY_TARGET
                or str(s.modification_intent) == "modify_target"
            )
            and f"code:{s.repo_name}/{s.file_path}" in excised_items
            for s in code_slices
        ):
            passed = False

        return passed, permitted_slices, permitted_citations, excised_items, missing_items

    def check_cryptographic_provenance(
        self,
        code_slices: list[VerbatimCodeSlice],
        bound_repo: BoundGraphRepository,
        principal: PrincipalContext,
        domain_id: str,
    ) -> tuple[bool, list[str], list[MissingItemRecord]]:
        """Check 5: Cryptographic provenance anchor verification."""
        unanchored: list[str] = []
        missing_items: list[MissingItemRecord] = []

        visible_files = bound_repo.list_visible_code_files(principal, domain_id)
        files_by_path = {f.file_path: f for f in visible_files}

        for s in code_slices:
            if (
                not s.git_commit_sha
                or len(s.git_commit_sha) < 7
                or s.git_commit_sha == "placeholder_sha"
            ):
                unanchored.append(f"Invalid commit SHA in {s.file_path}")
                missing_items.append(
                    MissingItemRecord(
                        kind="provenance",
                        identifier=s.file_path,
                        reason="Missing or placeholder git_commit_sha",
                    )
                )
                continue
            if (
                not s.git_blob_sha
                or len(s.git_blob_sha) < 7
                or s.git_blob_sha == "placeholder_blob"
            ):
                unanchored.append(f"Invalid blob SHA in {s.file_path}")
                missing_items.append(
                    MissingItemRecord(
                        kind="provenance",
                        identifier=s.file_path,
                        reason="Missing or placeholder git_blob_sha",
                    )
                )
                continue
            if s.start_line < 1 or s.end_line < s.start_line:
                unanchored.append(
                    f"Invalid line bounds {s.start_line}-{s.end_line} in {s.file_path}"
                )
                missing_items.append(
                    MissingItemRecord(
                        kind="provenance",
                        identifier=s.file_path,
                        reason=f"Invalid line bounds {s.start_line}-{s.end_line}",
                    )
                )
                continue

            f_rec = files_by_path.get(s.file_path)
            if f_rec is None:
                unanchored.append(f"Phantom file not found in repository: {s.file_path}")
                missing_items.append(
                    MissingItemRecord(
                        kind="provenance",
                        identifier=s.file_path,
                        reason=f"File not found in repository records: {s.file_path}",
                    )
                )
                continue
            expected_blob_sha = (
                f_rec.git_blob_sha.ljust(7, "0")
                if len(f_rec.git_blob_sha) < 7
                else f_rec.git_blob_sha
            )
            if (
                f_rec.git_blob_sha
                and s.git_blob_sha != f_rec.git_blob_sha
                and s.git_blob_sha != expected_blob_sha
            ):
                unanchored.append(
                    f"Blob SHA mismatch in {s.file_path}: {s.git_blob_sha} vs {f_rec.git_blob_sha}"
                )
                missing_items.append(
                    MissingItemRecord(
                        kind="provenance",
                        identifier=s.file_path,
                        reason=f"Blob SHA mismatch: {s.git_blob_sha} vs {f_rec.git_blob_sha}",
                    )
                )

        return len(unanchored) == 0, unanchored, missing_items

    def check_document_freshness(
        self,
        citations: list[DocCitationSpan],
        doc_chunks: dict[str, str] | list[DocChunkRecord] | None,
        bound_repo: BoundGraphRepository,
        principal: PrincipalContext,
        domain_id: str,
    ) -> tuple[bool, list[str], list[MissingItemRecord]]:
        """Check 6: Document freshness and staleness invalidation."""
        stale_items: list[str] = []
        missing_items: list[MissingItemRecord] = []

        if not citations:
            return True, [], []

        visible_chunks = bound_repo.list_visible_doc_chunks(
            principal, domain_id, include_stale=True
        )
        chunks_by_id = {str(c.chunk_id): c for c in visible_chunks}
        if isinstance(doc_chunks, list):
            for c in doc_chunks:
                if isinstance(c, DocChunkRecord):
                    chunks_by_id[str(c.chunk_id)] = c

        neg_resolutions = set(r.lower() for r in self.domain_config.jira_negative_resolutions)

        for cit in citations:
            chunk = chunks_by_id.get(str(cit.chunk_id))
            if chunk is not None:
                if chunk.is_stale:
                    stale_items.append(f"stale_chunk:{cit.chunk_id}")
                    missing_items.append(
                        MissingItemRecord(
                            kind="freshness",
                            identifier=cit.chunk_id,
                            reason=f"Chunk {cit.chunk_id} marked stale/superseded",
                        )
                    )
                meta = getattr(chunk, "metadata_json", {}) or {}
                res_val = str(meta.get("resolution", "")).strip().lower()
                if res_val and res_val in neg_resolutions:
                    stale_items.append(f"negative_resolution:{cit.chunk_id}:{res_val}")
                    missing_items.append(
                        MissingItemRecord(
                            kind="freshness",
                            identifier=cit.chunk_id,
                            reason=f"Ticket has negative resolution '{res_val}'",
                        )
                    )

        return len(stale_items) == 0, stale_items, missing_items

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
        """Executes all 6 deterministic preflight checks, returning a PreFlightValidationReport."""
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
            else (
                pack.domain_id
                if pack is not None
                else (
                    self.domain_id
                    or (self.domain_config.domain_id if self.domain_config else None)
                    or ScoutSettings().default_domain_id
                )
            )
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

            check_1_passed, syntax_errors, syntax_missing = self.check_ast_syntax(
                active_slices,
                bound_repo=bound_repo,
                principal=active_principal,
                domain_id=active_domain,
            )

            # ---------------------------------------------------------------------
            # Check 2: 1-Hop AST Dependency Closure & Prompt Reference Coverage
            # ---------------------------------------------------------------------
            unresolved_symbols, auto_attached_count = self.check_symbol_resolution(
                code_slices=active_slices,
                bound_repo=bound_repo,
                principal=active_principal,
                domain_id=active_domain,
            )
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
            check_2_passed = (
                len(unresolved_symbols) == 0 and len(missing_reference_identifiers) == 0
            )
            closure_missing: list[MissingItemRecord] = [
                MissingItemRecord(
                    kind="code_symbol",
                    identifier=sym,
                    reason=f"Unresolved internal symbol: {sym}",
                )
                for sym in unresolved_symbols
            ] + [
                MissingItemRecord(
                    kind="file" if "." in sym else "code_symbol",
                    identifier=sym,
                    reason=f"Referenced existing symbol/file missing from context: {sym}",
                )
                for sym in missing_reference_identifiers
            ]

            # ---------------------------------------------------------------------
            # Check 3: Retrieval Confidence & Relevance Thresholding
            # ---------------------------------------------------------------------
            check_3_passed, final_rrf_score = self.check_rrf_threshold(
                normalized_rrf_score=normalized_rrf_score,
                cross_encoder_score=cross_encoder_score,
            )
            rrf_missing: list[MissingItemRecord] = []
            if not check_3_passed:
                rrf_missing.append(
                    MissingItemRecord(
                        kind="citation",
                        identifier=f"rrf_score_{final_rrf_score}",
                        reason=(
                            f"Top retrieval score {final_rrf_score:.4f} is below "
                            f"threshold {self.min_rrf_score:.2f}"
                        ),
                    )
                )

            # ---------------------------------------------------------------------
            # Check 4: Principal Scope & Tenant Permission Mirroring
            # ---------------------------------------------------------------------
            (
                check_4_passed,
                permitted_slices,
                permitted_citations,
                excised_items,
                scope_missing,
            ) = self.check_principal_scopes(
                code_slices=active_slices,
                citations=all_citations,
                bound_repo=bound_repo,
                principal=active_principal,
                domain_id=active_domain,
                doc_chunks=doc_chunks,
            )
            active_slices = permitted_slices
            all_citations = permitted_citations

            # ---------------------------------------------------------------------
            # Check 5: Cryptographic Provenance Anchoring
            # ---------------------------------------------------------------------
            check_5_passed, unanchored_items, provenance_missing = (
                self.check_cryptographic_provenance(
                    code_slices=active_slices,
                    bound_repo=bound_repo,
                    principal=active_principal,
                    domain_id=active_domain,
                )
            )

            # ---------------------------------------------------------------------
            # Check 6: Document Freshness & Staleness Invalidation + Citation Grounding
            # ---------------------------------------------------------------------
            citations_verified_100_pct, failed_citation_reasons = self.check_citations(
                citations=all_citations,
                doc_chunks=doc_chunks,
                bound_repo=bound_repo,
                principal=active_principal,
                domain_id=active_domain,
            )
            freshness_passed, stale_items, freshness_missing = self.check_document_freshness(
                citations=all_citations,
                doc_chunks=doc_chunks,
                bound_repo=bound_repo,
                principal=active_principal,
                domain_id=active_domain,
            )
            check_6_passed = citations_verified_100_pct and freshness_passed
            citation_missing: list[MissingItemRecord] = [
                MissingItemRecord(
                    kind="citation",
                    identifier=r[:40],
                    reason=r,
                )
                for r in failed_citation_reasons
            ]

            # Aggregate all missing items
            all_missing_items = (
                syntax_missing
                + closure_missing
                + rrf_missing
                + scope_missing
                + provenance_missing
                + freshness_missing
                + citation_missing
            )

            check_results = {
                "check_1_syntax": check_1_passed,
                "check_2_closure": check_2_passed,
                "check_3_rrf": check_3_passed,
                "check_4_scope": check_4_passed,
                "check_5_provenance": check_5_passed,
                "check_6_freshness": check_6_passed,
            }
            all_passed = all(check_results.values())
            brief_obj = (
                pack.domain_knowledge_brief if pack is not None else None
            ) or active_quant
            if (
                brief_obj is not None
                and getattr(brief_obj, "sufficiency_verdict", "sufficient") == "insufficient"
            ):
                all_passed = False
                check_results["check_3_rrf"] = False
                missing_str = "; ".join(getattr(brief_obj, "missing_aspects", [])[:3])
                all_missing_items.append(
                    MissingItemRecord(
                        kind="citation",
                        identifier=active_domain,
                        reason=(
                            "Synthesized brief marked context insufficient"
                            + (f": {missing_str}" if missing_str else ".")
                        ),
                    )
                )
            if not active_slices and not all_citations:
                all_passed = False
                check_results["check_2_closure"] = False
                all_missing_items.append(
                    MissingItemRecord(
                        kind="file",
                        identifier=active_domain,
                        reason=(
                            "Context pack contains zero code slices and zero "
                            "supporting document citations"
                        ),
                    )
                )

            bounce_back_feedback: str | None = None
            if not all_passed:
                feedback_parts = [
                    "Pre-Flight Gate Validation FAILED ($0 Local Bounce-Back Required):"
                ]
                if not check_1_passed:
                    feedback_parts.append(
                        f"- Check 1 (AST Syntax Well-Formedness): Syntax errors detected: "
                        f"{syntax_errors}."
                    )
                if not check_2_passed:
                    if unresolved_symbols:
                        feedback_parts.append(
                            "- Check 2 (Internal Symbol Resolution): Missing dependencies: "
                            f"{unresolved_symbols}."
                        )
                    if missing_reference_identifiers:
                        feedback_parts.append(
                            "- Check 2 (Prompt Identifier Partitioning, VAL-01): "
                            f"Missing reference items: {missing_reference_identifiers}."
                        )
                if not check_3_passed:
                    feedback_parts.append(
                        f"- Check 3 (Normalized RRF Score, BLK-01): Top Normalized RRF score "
                        f"{final_rrf_score:.4f} is below threshold {self.min_rrf_score:.2f}."
                    )
                if not check_4_passed:
                    feedback_parts.append(
                        f"- Check 4 (Principal Scope Mirroring): Unauthorized items excised: "
                        f"{excised_items}."
                    )
                if not check_5_passed:
                    feedback_parts.append(
                        f"- Check 5 (Cryptographic Provenance): Unanchored items: "
                        f"{unanchored_items}."
                    )
                if not check_6_passed:
                    if failed_citation_reasons:
                        feedback_parts.append(
                            f"- Check 6 (Citation Verification): Citations failed: "
                            f"{'; '.join(failed_citation_reasons)}."
                        )
                    if stale_items:
                        feedback_parts.append(
                            f"- Check 6 (Document Freshness): Stale/superseded items: "
                            f"{stale_items}."
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
                syntax_errors=syntax_errors,
                excised_unauthorized_items=excised_items,
                unanchored_provenance_items=unanchored_items,
                stale_invalidated_items=stale_items,
                missing_items=all_missing_items,
                check_results=check_results,
            )

            # Mutate pack in-place if provided
            if pack is not None:
                pack.code_slices = active_slices
                pack.preflight_report = report

            pf_span.record_report(report)
            return report

    def evaluate_gate(
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
        trace_id: str | None = None,
    ) -> GatePassResult | GateAbstentionResult:
        """Executes all 6 preflight checks and emits GatePassResult or GateAbstentionResult."""
        report = self.validate(
            pack=pack,
            code_slices=code_slices,
            user_prompt=user_prompt,
            supporting_doc_citations=supporting_doc_citations,
            quant_strategy_spec=quant_strategy_spec,
            normalized_rrf_score=normalized_rrf_score,
            cross_encoder_score=cross_encoder_score,
            repo=repo,
            principal=principal,
            domain_id=domain_id,
            conn=conn,
            doc_chunks=doc_chunks,
        )
        if report.passed:
            return GatePassResult(passed=True, report=report, pack=pack)

        slices = code_slices if code_slices is not None else (pack.code_slices if pack else [])
        diag = self.create_diagnostic(
            trace_id=trace_id or (pack.trace_id if pack else "trace_preflight_gate"),
            domain_id=domain_id
            or (
                pack.domain_id
                if pack
                else (
                    self.domain_id
                    or (self.domain_config.domain_id if self.domain_config else None)
                    or ScoutSettings().default_domain_id
                )
            ),
            report=report,
            partial_candidate_files=list({s.file_path for s in slices}),
        )
        return GateAbstentionResult(passed=False, diagnostic=diag, report=report)

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
    "GateAbstentionResult",
    "GatePassResult",
    "PreFlightValidator",
    "create_insufficient_context_diagnostic",
]
