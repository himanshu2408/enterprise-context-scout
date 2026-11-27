"""Git Blob-SHA Merkle incremental indexer with Tree-Sitter AST extraction and RLS persistence."""

import functools
import hashlib
import logging
import os
import shutil
import stat
import subprocess
import sys
import time
import weakref
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Any, Final, Literal, TypeVar
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from psycopg import Connection
from pydantic import BaseModel

from context_scout.ast_engine.extractor import ASTExtractor
from context_scout.ast_engine.models import ParsedEdge, ParsedSymbol
from context_scout.db.repositories import (
    ASTEdgeRecord,
    ASTSymbolRecord,
    CodeFileRecord,
    InMemoryRLSRepository,
    OverlayFileDeltaRecord,
    PostgresRLSRepository,
    RefSnapshotRecord,
)
from context_scout.ingestion.ownership_graph import (
    CATALOG_INFO_FILENAMES,
    CODEOWNERS_FILENAMES,
    FileWeightClassification,
    GitAttributesMatcher,
    OwnershipGraphExtractor,
    has_ownership_or_attr_files,
    normalize_rel_path,
)
from context_scout.schemas.contracts import PrincipalContext

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _resolve_git_executable() -> str:
    """Resolves direct git binary once (bypasses macOS /usr/bin/git xcrun double-spawn)."""
    git_path = shutil.which("git") or "git"
    if sys.platform == "darwin" and git_path == "/usr/bin/git":
        try:
            proc = subprocess.run(
                ["xcrun", "-f", "git"],
                capture_output=True,
                text=True,
                check=False,
                timeout=2.0,
            )
            candidate = proc.stdout.strip()
            if proc.returncode == 0 and candidate and Path(candidate).is_file():
                return candidate
        except Exception as exc:
            logger.debug(
                "xcrun direct git resolution fallback (%s): %s",
                type(exc).__name__,
                exc,
            )
    return git_path


CODE_EXTENSIONS: Final[set[str]] = {
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".mjs",
    ".cjs",
    ".sql",
    ".yaml",
    ".yml",
    ".tf",
    ".go",
    ".rs",
    ".java",
}


def compute_git_blob_sha(raw_bytes: bytes) -> str:
    """Computes the exact 40-hex Git blob SHA-1 for arbitrary file bytes."""
    header = f"blob {len(raw_bytes)}\x00".encode("ascii")
    return hashlib.sha1(header + raw_bytes).hexdigest()


LANG_MAP: Final[dict[str, str]] = {
    ".py": "python",
    ".tsx": "tsx",
    ".ts": "typescript",
    ".jsx": "jsx",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".sql": "sql",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".tf": "terraform",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
}

_K = TypeVar("_K")
_V = TypeVar("_V")

_SYMBOL_EMBEDDING_CACHE_MAX_ENTRIES: Final[int] = 16_384
_COMMIT_TREE_CACHE_MAX_ENTRIES: Final[int] = 128
_DIFF_TREE_CACHE_MAX_ENTRIES: Final[int] = 256
_WT_BLOB_CACHE_MAX_ENTRIES: Final[int] = 8_192


def _lru_get(cache: OrderedDict[_K, _V], key: _K) -> _V | None:
    """Returns cached value and promotes key to most-recently-used."""
    if key not in cache:
        return None
    cache.move_to_end(key)
    return cache[key]


def _lru_set(cache: OrderedDict[_K, _V], key: _K, value: _V, max_entries: int) -> None:
    """Inserts/updates key in bounded OrderedDict and evicts least-recently-used entries."""
    if max_entries <= 0:
        return
    if key in cache:
        cache.move_to_end(key)
    cache[key] = value
    while len(cache) > max_entries:
        cache.popitem(last=False)


_SYMBOL_EMBEDDING_CACHE: OrderedDict[tuple[str, str], list[float]] = OrderedDict()
_TRACKED_IN_MEMORY_REPOS: dict[
    tuple[int, str, str],
    tuple[weakref.ReferenceType[InMemoryRLSRepository], Path, list[str], str],
] = {}

SKIP_BINARY_EXTENSIONS: Final[set[str]] = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".bmp",
    ".pdf",
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".onnx",
    ".gguf",
    ".npz",
    ".npy",
    ".whl",
    ".tar",
    ".gz",
    ".zip",
    ".7z",
    ".pyc",
    ".pyo",
    ".so",
    ".dylib",
    ".dll",
    ".exe",
    ".woff",
    ".woff2",
    ".ttf",
    ".otf",
}


# Bounded LRU cache for immutable git ls-tree -r output by (resolved_repo_dir, commit_sha).
_COMMIT_TREE_CACHE: OrderedDict[tuple[str, str], dict[str, str]] = OrderedDict()

# Bounded LRU cache for immutable git diff-tree -r --raw output by
# (repo_dir, before_sha, after_sha, renames).
_DIFF_TREE_CACHE: OrderedDict[tuple[str, str, str, bool], "MerkleBlobDiff"] = OrderedDict()

# Bounded LRU cache for working-tree dirty file blob SHAs by (abs_path, st_mtime_ns, st_size).
_WT_BLOB_CACHE: OrderedDict[tuple[str, int, int], str] = OrderedDict()


@dataclass(frozen=True)
class MerkleBlobDiff:
    """Commit-to-commit tree diff partitioned by file operation (F08)."""

    added: dict[str, str]  # file_path -> new_blob_sha
    modified: dict[str, tuple[str, str]]  # file_path -> (old_blob_sha, new_blob_sha)
    deleted: dict[str, str]  # file_path -> old_blob_sha
    renamed: list[tuple[str, str, str, str]]  # (old_path, new_path, old_blob, new_blob)


class MerkleSyncStats(BaseModel):
    """Statistics emitted by Merkle incremental synchronization."""

    total_files_scanned: int
    unchanged_files_skipped: int
    new_files_indexed: int
    modified_files_reindexed: int
    deleted_files_removed: int
    total_symbols_stored: int
    total_edges_stored: int
    duration_ms: float
    commit_sha: str
    cache_hit: bool


def _resolve_commit_sha(repo_dir: Path, ref: str = "HEAD") -> str:
    """Fast resolution of commit SHA from .git directory with subprocess fallback."""
    git_dir = repo_dir / ".git"
    if git_dir.is_dir():
        try:
            if ref == "HEAD":
                head_file = git_dir / "HEAD"
                if head_file.is_file():
                    content = head_file.read_text().strip()
                    if not content.startswith("ref:"):
                        is_hex = all(c in "0123456789abcdefABCDEF" for c in content)
                        if len(content) == 40 and is_hex:
                            return content
                    else:
                        ref_path = content.split(":", 1)[1].strip()
                        target_file = git_dir / ref_path
                        if target_file.is_file():
                            ref_sha = target_file.read_text().strip()
                            if len(ref_sha) == 40:
                                return ref_sha
                        packed = git_dir / "packed-refs"
                        if packed.is_file():
                            for line in packed.read_text().splitlines():
                                if line.endswith(" " + ref_path):
                                    cand = line.split()[0]
                                    if len(cand) == 40:
                                        return cand
        except Exception as exc:
            logger.debug(
                "Direct .git ref lookup failed for %s (%s): %s; falling back to git rev-parse",
                ref,
                type(exc).__name__,
                exc,
            )

    rev_cmd = subprocess.run(
        [_resolve_git_executable(), "--no-optional-locks", "-C", str(repo_dir), "rev-parse", ref],
        capture_output=True,
        text=True,
        check=True,
        close_fds=(os.name != "posix"),
    )
    return rev_cmd.stdout.strip()


class GitMerkleIndexer:
    """Git Blob-SHA Merkle incremental indexer with Tree-Sitter AST extraction."""

    EMPTY_TREE_SHA: Final[str] = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"

    def __init__(
        self,
        extractor: ASTExtractor | None = None,
        embedder: Any | None = None,
        use_semantic_embeddings: bool = True,
        include_working_tree: bool = True,
    ) -> None:
        self.extractor = extractor or ASTExtractor()
        self.embedder = embedder
        self.use_semantic_embeddings = use_semantic_embeddings
        self.include_working_tree = include_working_tree

    @staticmethod
    def _is_indexable_text_path(file_path: str) -> bool:
        suffix = Path(file_path).suffix.lower()
        return suffix not in SKIP_BINARY_EXTENSIONS

    def _compute_symbol_embeddings(
        self,
        symbols: list[ParsedSymbol],
        blob_sha: str = "",
        *,
        git_blob_sha: str = "",
    ) -> list[list[float]]:
        blob_sha = blob_sha or git_blob_sha
        if not symbols:
            return []
        active_embedder = self.embedder
        if active_embedder is None:
            from context_scout.retrieval.embedder import get_embedding_provider

            active_embedder = get_embedding_provider()
        results: list[list[float] | None] = [None] * len(symbols)
        missing_indices: list[int] = []
        missing_texts: list[str] = []
        for idx, s in enumerate(symbols):
            cache_key = (blob_sha, s.qualified_name)
            cached_vec = _lru_get(_SYMBOL_EMBEDDING_CACHE, cache_key) if blob_sha else None
            if cached_vec is not None:
                results[idx] = cached_vec
            else:
                missing_indices.append(idx)
                missing_texts.append(
                    f"{s.qualified_name} {s.signature_skeleton} {s.contextual_summary}".strip()
                )
        if missing_texts:
            batch_vecs = active_embedder.embed_batch(missing_texts, is_query=False)
            if len(batch_vecs) != len(missing_indices):
                raise RuntimeError(
                    f"Embedder returned {len(batch_vecs)} vectors "
                    f"for {len(missing_indices)} symbols."
                )
            for m_idx, vec in zip(missing_indices, batch_vecs, strict=True):
                flt_vec = [float(x) for x in vec]
                results[m_idx] = flt_vec
                if blob_sha:
                    _lru_set(
                        _SYMBOL_EMBEDDING_CACHE,
                        (blob_sha, symbols[m_idx].qualified_name),
                        flt_vec,
                        _SYMBOL_EMBEDDING_CACHE_MAX_ENTRIES,
                    )
        return [r for r in results if r is not None]

    def _batch_prepopulate_symbol_embeddings(
        self,
        items: Sequence[tuple[str, list[ParsedSymbol]]],
        *,
        heartbeat_fn: Callable[[], object] | None = None,
        chunk_size: int = 128,
    ) -> None:
        """Pre-populates `_SYMBOL_EMBEDDING_CACHE` for uncached (blob_sha, symbol) pairs."""
        pending_keys: list[tuple[str, str]] = []
        pending_texts: list[str] = []
        seen: set[tuple[str, str]] = set()
        for blob_sha, symbols in items:
            if not blob_sha or not symbols:
                continue
            for sym in symbols:
                cache_key = (blob_sha, sym.qualified_name)
                if cache_key in _SYMBOL_EMBEDDING_CACHE or cache_key in seen:
                    continue
                seen.add(cache_key)
                pending_keys.append(cache_key)
                raw_text = (
                    f"{sym.qualified_name} {sym.signature_skeleton} {sym.contextual_summary}"
                ).strip()
                pending_texts.append(raw_text)
        if not pending_texts:
            return
        active_embedder = self.embedder
        if active_embedder is None:
            from context_scout.retrieval.embedder import get_embedding_provider

            active_embedder = get_embedding_provider()
        step = max(1, chunk_size)
        for offset in range(0, len(pending_texts), step):
            if heartbeat_fn is not None:
                heartbeat_fn()
            slice_keys = pending_keys[offset : offset + step]
            slice_texts = pending_texts[offset : offset + step]
            batch_vecs = active_embedder.embed_batch(slice_texts, is_query=False)
            if len(batch_vecs) != len(slice_keys):
                raise RuntimeError(
                    f"Embedder returned {len(batch_vecs)} vectors for {len(slice_keys)} symbols."
                )
            for cache_key, vec in zip(slice_keys, batch_vecs, strict=True):
                _lru_set(
                    _SYMBOL_EMBEDDING_CACHE,
                    cache_key,
                    [float(x) for x in vec],
                    _SYMBOL_EMBEDDING_CACHE_MAX_ENTRIES,
                )

    def get_git_tree(
        self,
        repo_dir: Path,
        ref: str = "HEAD",
        include_working_tree: bool | None = None,
    ) -> tuple[str, dict[str, str]]:
        """Extracts current commit SHA and mapping of {file_path: blob_sha} via git ls-tree."""
        commit_sha = _resolve_commit_sha(repo_dir, ref)
        cache_key = (str(repo_dir.resolve()), commit_sha)
        cached_base = _lru_get(_COMMIT_TREE_CACHE, cache_key)
        if cached_base is not None:
            tree: dict[str, str] = dict(cached_base)
        else:
            ls_cmd = subprocess.run(
                [
                    _resolve_git_executable(),
                    "--no-optional-locks",
                    "-C",
                    str(repo_dir),
                    "ls-tree",
                    "-r",
                    ref,
                ],
                capture_output=True,
                text=True,
                check=True,
                close_fds=(os.name != "posix"),
            )

            tree = {}
            for line in ls_cmd.stdout.strip().splitlines():
                if not line:
                    continue
                parts = line.split("\t", 1)
                if len(parts) != 2:
                    continue
                meta, file_path = parts
                if not self._is_indexable_text_path(file_path):
                    continue
                meta_parts = meta.split()
                if len(meta_parts) >= 3:
                    blob_sha = meta_parts[2]
                    tree[file_path] = blob_sha
            _lru_set(
                _COMMIT_TREE_CACHE,
                cache_key,
                dict(tree),
                _COMMIT_TREE_CACHE_MAX_ENTRIES,
            )

        overlay_wt = (
            self.include_working_tree if include_working_tree is None else include_working_tree
        )
        if overlay_wt and ref == "HEAD":
            st_cmd = subprocess.run(
                [
                    _resolve_git_executable(),
                    "--no-optional-locks",
                    "-C",
                    str(repo_dir),
                    "status",
                    "--porcelain",
                    "-unormal",
                ],
                capture_output=True,
                text=True,
                check=False,
                close_fds=(os.name != "posix"),
            )
            if st_cmd.returncode == 0 and st_cmd.stdout.strip():
                dirty_mutations: list[str] = []
                for raw_ln in st_cmd.stdout.splitlines():
                    if len(raw_ln) < 4:
                        continue
                    xy = raw_ln[:2]
                    rel_part = raw_ln[3:].strip().strip('"')
                    if " -> " in rel_part:
                        old_rel, new_rel = rel_part.split(" -> ", 1)
                        old_rel = old_rel.strip().strip('"')
                        new_rel = new_rel.strip().strip('"')
                        tree.pop(old_rel, None)
                        dirty_mutations.append(f"D:{old_rel}")
                        rel_part = new_rel
                    if not self._is_indexable_text_path(rel_part):
                        continue
                    disk_f = repo_dir / rel_part
                    if "D" in xy:
                        if rel_part in tree:
                            tree.pop(rel_part, None)
                            dirty_mutations.append(f"D:{rel_part}")
                        continue
                    try:
                        st_res = disk_f.stat()
                    except OSError:
                        if rel_part in tree:
                            tree.pop(rel_part, None)
                            dirty_mutations.append(f"D:{rel_part}")
                        continue
                    if stat.S_ISREG(st_res.st_mode):
                        try:
                            wt_key = (str(disk_f), st_res.st_mtime_ns, st_res.st_size)
                            wt_blob = _lru_get(_WT_BLOB_CACHE, wt_key)
                            if wt_blob is None:
                                wt_blob = compute_git_blob_sha(disk_f.read_bytes())
                                _lru_set(
                                    _WT_BLOB_CACHE,
                                    wt_key,
                                    wt_blob,
                                    _WT_BLOB_CACHE_MAX_ENTRIES,
                                )
                            if tree.get(rel_part) != wt_blob:
                                tree[rel_part] = wt_blob
                                dirty_mutations.append(f"M:{rel_part}:{wt_blob}")
                        except Exception as exc:
                            logger.warning(
                                "Failed to hash working-tree file %s (%s): %s",
                                rel_part,
                                type(exc).__name__,
                                exc,
                            )
                if dirty_mutations:
                    wt_digest = hashlib.sha1(
                        "|".join(sorted(dirty_mutations)).encode("utf-8")
                    ).hexdigest()
                    commit_sha = f"{commit_sha[:32]}dddd{wt_digest[:4]}"

        return commit_sha, tree

    def diff_merkle_tree(
        self, current: dict[str, str], existing: dict[str, str]
    ) -> tuple[set[str], set[str], set[str], set[str]]:
        """Partitions repository files into (unchanged, new, modified, deleted)."""
        unchanged = {p for p, s in current.items() if p in existing and existing[p] == s}
        new = set(current.keys()) - set(existing.keys())
        modified = {p for p, s in current.items() if p in existing and existing[p] != s}
        deleted = set(existing.keys()) - set(current.keys())
        return unchanged, new, modified, deleted

    def diff_commit_blobs(
        self,
        repo_dir: Path,
        before_sha: str,
        after_sha: str,
        *,
        detect_renames: bool = False,
        use_cache: bool = True,
    ) -> MerkleBlobDiff:
        """Diffs commit trees using git diff-tree -r --raw in <25ms (F08)."""
        effective_before = (
            self.EMPTY_TREE_SHA
            if not before_sha or before_sha == "0" * 40 or before_sha.strip() == ""
            else before_sha
        )
        is_immutable_sha = (
            len(effective_before) == 40
            and len(after_sha) == 40
            and all(c in "0123456789abcdefABCDEF" for c in effective_before)
            and all(c in "0123456789abcdefABCDEF" for c in after_sha)
        )
        diff_cache_key = (
            str(repo_dir.resolve()),
            effective_before,
            after_sha,
            detect_renames,
        )
        if use_cache and is_immutable_sha:
            cached_diff = _lru_get(_DIFF_TREE_CACHE, diff_cache_key)
            if cached_diff is not None:
                return cached_diff

        cmd = [
            _resolve_git_executable(),
            "--no-optional-locks",
            "-C",
            str(repo_dir),
            "diff-tree",
            "-r",
        ]
        if detect_renames:
            cmd.append("-M")
        cmd.extend(["--raw", effective_before, after_sha])

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            close_fds=(os.name != "posix"),
        )

        added: dict[str, str] = {}
        modified: dict[str, tuple[str, str]] = {}
        deleted: dict[str, str] = {}
        renamed: list[tuple[str, str, str, str]] = []

        for line in proc.stdout.splitlines():
            if not line.startswith(":"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            meta = parts[0]
            file_path = parts[1]
            meta_tokens = meta[1:].split()
            if len(meta_tokens) < 5:
                continue
            old_blob = meta_tokens[2]
            new_blob = meta_tokens[3]
            status = meta_tokens[4]

            if not self._is_indexable_text_path(file_path):
                continue

            if status == "A":
                added[file_path] = new_blob
            elif status == "M":
                modified[file_path] = (old_blob, new_blob)
            elif status == "D":
                deleted[file_path] = old_blob
            elif status.startswith("R"):
                old_path = file_path
                new_path = parts[2] if len(parts) > 2 else file_path
                renamed.append((old_path, new_path, old_blob, new_blob))

        result = MerkleBlobDiff(added=added, modified=modified, deleted=deleted, renamed=renamed)
        if use_cache and is_immutable_sha:
            _lru_set(
                _DIFF_TREE_CACHE,
                diff_cache_key,
                result,
                _DIFF_TREE_CACHE_MAX_ENTRIES,
            )
        return result

    def _detect_language(self, file_path: str) -> str | None:
        for ext, lang in LANG_MAP.items():
            if file_path.endswith(ext):
                return lang
        return None

    @staticmethod
    def _read_blob_or_disk(
        repo_dir: Path,
        file_path: str,
        blob_sha: str,
        ref: str = "HEAD",
        prefer_working_tree: bool = False,
    ) -> str | None:
        """Read immutable Git blob content via git cat-file or live working-tree file."""
        _ = ref
        disk_file = repo_dir / file_path
        if prefer_working_tree and disk_file.is_file():
            try:
                return disk_file.read_text(encoding="utf-8", errors="replace").replace("\x00", "")
            except Exception as exc:
                logger.debug(
                    "Working-tree read failed for %s (%s): %s; falling back to git cat-file",
                    file_path,
                    type(exc).__name__,
                    exc,
                )
        if blob_sha:
            proc = subprocess.run(
                [
                    _resolve_git_executable(),
                    "--no-optional-locks",
                    "-C",
                    str(repo_dir),
                    "cat-file",
                    "-p",
                    blob_sha,
                ],
                capture_output=True,
                check=False,
                close_fds=(os.name != "posix"),
            )
            if proc.returncode == 0:
                return proc.stdout.decode("utf-8", errors="replace").replace("\x00", "")
        if not disk_file.exists():
            return None
        return disk_file.read_text(encoding="utf-8", errors="replace").replace("\x00", "")

    @staticmethod
    def _is_attr_or_ownership_path(file_path: str) -> bool:
        norm = normalize_rel_path(file_path)
        base = norm.rsplit("/", 1)[-1]
        return (
            base == ".gitattributes"
            or norm in CODEOWNERS_FILENAMES
            or base in CATALOG_INFO_FILENAMES
        )

    def _build_attr_matcher_and_manifests(
        self,
        repo_dir: Path,
        tree: Mapping[str, str],
        ref: str,
        *,
        prefer_working_tree: bool,
    ) -> tuple[GitAttributesMatcher | None, dict[str, str]]:
        """Loads .gitattributes, CODEOWNERS, and catalog-info.y*ml only if present in tree."""
        if not has_ownership_or_attr_files(tree):
            return None, {}
        attr_files: dict[str, str] = {}
        manifest_files: dict[str, str] = {}
        for rel_path, blob_sha in tree.items():
            if not self._is_attr_or_ownership_path(rel_path):
                continue
            content = self._read_blob_or_disk(
                repo_dir, rel_path, blob_sha, ref, prefer_working_tree=prefer_working_tree
            )
            if content is None:
                continue
            manifest_files[rel_path] = content
            norm = normalize_rel_path(rel_path)
            if norm == ".gitattributes" or norm.endswith("/.gitattributes"):
                attr_files[norm] = content
        matcher = GitAttributesMatcher.from_repo_files(attr_files) if attr_files else None
        return matcher, manifest_files

    @staticmethod
    def _classify_path(
        matcher: GitAttributesMatcher | None, file_path: str
    ) -> FileWeightClassification:
        if matcher is None:
            return FileWeightClassification(
                file_path=file_path, ranking_weight=1.0, is_generated=False, is_vendored=False
            )
        return matcher.classify_file(file_path)

    def _serialize_delta_symbols_edges(
        self,
        *,
        domain_id: str,
        repo_name: str,
        file_path: str,
        symbols: Sequence[ParsedSymbol],
        edges: Sequence[ParsedEdge],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        sym_dicts: list[dict[str, Any]] = []
        for s in symbols:
            det_id = uuid5(
                NAMESPACE_URL,
                f"overlay:{domain_id}:{repo_name}:{file_path}:{s.qualified_name}:{s.start_line}",
            )
            sym_dicts.append(
                {
                    "symbol_id": str(det_id),
                    "qualified_name": s.qualified_name,
                    "symbol_kind": s.symbol_kind,
                    "start_line": s.start_line,
                    "end_line": s.end_line,
                    "signature_skeleton": s.signature_skeleton,
                    "contextual_summary": s.contextual_summary,
                    "repo_qualified_id": s.repo_qualified_id or f"{repo_name}::{s.qualified_name}",
                    "bare_name": s.bare_name or s.name,
                }
            )
        edge_dicts: list[dict[str, Any]] = [
            {
                "source_qualified_name": e.source_qualified_name,
                "target_qualified_name": e.target_qualified_name,
                "edge_kind": e.edge_kind,
                "target_repo_qualified_id": e.target_repo_qualified_id,
                "is_cross_repo": e.is_cross_repo,
            }
            for e in edges
        ]
        return sym_dicts, edge_dicts

    def sync_in_memory(
        self,
        repo_dir: Path,
        repo: InMemoryRLSRepository,
        principal: PrincipalContext,
        domain_id: str,
        repo_name: str,
        allowed_scopes: list[str] | None = None,
        ref: str = "HEAD",
        isolate_by_commit: bool = False,
        include_working_tree: bool | None = None,
    ) -> MerkleSyncStats:
        """Incrementally indexes git repository into InMemoryRLSRepository."""
        start_time = time.perf_counter()
        use_wt = (
            (self.include_working_tree if include_working_tree is None else include_working_tree)
            and ref == "HEAD"
            and not isolate_by_commit
        )
        commit_sha, current_tree = self.get_git_tree(repo_dir, ref, include_working_tree=use_wt)
        effective_repo_name = f"{repo_name}@{commit_sha[:12]}" if isolate_by_commit else repo_name
        effective_scopes = list(allowed_scopes or principal.allowed_scopes)
        if ref == "HEAD" and not isolate_by_commit:
            _TRACKED_IN_MEMORY_REPOS[(id(repo), domain_id, effective_repo_name)] = (
                weakref.ref(repo),
                repo_dir.resolve(),
                effective_scopes,
                commit_sha,
            )
        existing_hashes = repo.get_file_blob_hashes(principal, domain_id, effective_repo_name)
        existing_map = {path: blob_sha for path, (_, blob_sha) in existing_hashes.items()}

        unchanged, new, modified, deleted = self.diff_merkle_tree(current_tree, existing_map)

        # Fast-exit cache hit (<50ms)
        if not new and not modified and not deleted:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return MerkleSyncStats(
                total_files_scanned=len(current_tree),
                unchanged_files_skipped=len(unchanged),
                new_files_indexed=0,
                modified_files_reindexed=0,
                deleted_files_removed=0,
                total_symbols_stored=0,
                total_edges_stored=0,
                duration_ms=duration_ms,
                commit_sha=commit_sha,
                cache_hit=True,
            )

        # 1. Remove deleted files
        for d in deleted:
            if d in existing_hashes:
                file_id, _ = existing_hashes[d]
                repo.delete_code_file(principal, file_id)

        # 2. Remove modified files prior to re-indexing
        for m in modified:
            if m in existing_hashes:
                file_id, _ = existing_hashes[m]
                repo.delete_code_file(principal, file_id)

        attr_matcher, manifest_files = self._build_attr_matcher_and_manifests(
            repo_dir, current_tree, ref, prefer_working_tree=use_wt
        )
        # Re-classify unchanged files when .gitattributes was added/modified/deleted.
        if any(p.rsplit("/", 1)[-1] == ".gitattributes" for p in (new | modified | deleted)):
            for fid, cf in list(repo._files.items()):
                if (
                    cf.tenant_id == principal.tenant_id
                    and cf.domain_id == domain_id
                    and cf.repo_name == effective_repo_name
                    and cf.file_path in unchanged
                ):
                    cls_res = self._classify_path(attr_matcher, cf.file_path)
                    repo._files[fid] = dc_replace(
                        cf,
                        ranking_weight=cls_res.ranking_weight,
                        is_generated=cls_res.is_generated,
                        is_vendored=cls_res.is_vendored,
                    )

        scopes = allowed_scopes or principal.allowed_scopes
        to_index = sorted(new | modified)

        total_symbols = 0
        total_edges = 0

        # Pass 1: Parse and insert code files and ast symbols
        parsed_files_data: list[
            tuple[CodeFileRecord, list[ParsedSymbol], list[ParsedEdge], dict[str, UUID]]
        ] = []
        global_symbol_map: dict[str, UUID] = {}
        short_name_map: dict[str, list[UUID]] = defaultdict(list)

        # Also populate global_symbol_map with existing visible symbols for cross-file linking
        for s in repo.list_visible_ast_symbols(principal, domain_id):
            global_symbol_map[s.qualified_name] = s.symbol_id
            short_name_map[s.qualified_name.split(".")[-1]].append(s.symbol_id)

        pre_parsed_inmem: dict[
            str, tuple[str, str | None, list[ParsedSymbol], list[ParsedEdge]]
        ] = {}
        batch_emb_items: list[tuple[str, list[ParsedSymbol]]] = []
        for file_path in to_index:
            blob_sha = current_tree[file_path]
            raw_content = self._read_blob_or_disk(
                repo_dir, file_path, blob_sha, ref, prefer_working_tree=use_wt
            )
            if raw_content is None:
                continue
            lang = self._detect_language(file_path)
            if lang:
                symbols, edges = self.extractor.parse_source(raw_content, lang, file_path=file_path)
                batch_emb_items.append((blob_sha, symbols))
            else:
                symbols, edges = [], []
            pre_parsed_inmem[file_path] = (raw_content, lang, symbols, edges)

        self._batch_prepopulate_symbol_embeddings(batch_emb_items)

        for file_path in to_index:
            parsed_entry = pre_parsed_inmem.get(file_path)
            if parsed_entry is None:
                continue
            blob_sha = current_tree[file_path]
            raw_content, lang, symbols, edges = parsed_entry
            lines = raw_content.splitlines()
            file_id = uuid4()
            cls_res = self._classify_path(attr_matcher, file_path)

            file_rec = CodeFileRecord(
                file_id=file_id,
                domain_id=domain_id,
                repo_name=effective_repo_name,
                file_path=file_path,
                git_commit_sha=commit_sha,
                git_blob_sha=blob_sha,
                language=lang or "text",
                allowed_scopes=scopes,
                raw_content=raw_content,
                line_count=len(lines),
                tenant_id=principal.tenant_id,
                ranking_weight=cls_res.ranking_weight,
                is_generated=cls_res.is_generated,
                is_vendored=cls_res.is_vendored,
            )
            repo.insert_code_file(principal, file_rec)

            if not lang:
                continue

            file_symbol_map: dict[str, UUID] = {}
            symbol_embeddings = self._compute_symbol_embeddings(symbols, blob_sha=blob_sha)

            for parsed_sym, emb in zip(symbols, symbol_embeddings, strict=True):
                sym_id = uuid4()
                sym_rec = ASTSymbolRecord(
                    symbol_id=sym_id,
                    file_id=file_id,
                    domain_id=domain_id,
                    qualified_name=parsed_sym.qualified_name,
                    symbol_kind=parsed_sym.symbol_kind,
                    start_line=parsed_sym.start_line,
                    end_line=parsed_sym.end_line,
                    signature_skeleton=parsed_sym.signature_skeleton,
                    contextual_summary=parsed_sym.contextual_summary,
                    allowed_scopes=scopes,
                    embedding=emb,
                    repo_qualified_id=parsed_sym.repo_qualified_id,
                    bare_name=parsed_sym.bare_name or parsed_sym.name,
                    tenant_id=principal.tenant_id,
                )
                repo.insert_ast_symbol(principal, sym_rec)
                file_symbol_map[parsed_sym.qualified_name] = sym_id
                global_symbol_map[parsed_sym.qualified_name] = sym_id
                short_name_map[parsed_sym.name].append(sym_id)
                total_symbols += 1

            parsed_files_data.append((file_rec, symbols, edges, file_symbol_map))

        # Pass 2: Directed Edge Resolution
        for file_rec, _symbols, edges, file_symbol_map in parsed_files_data:
            for edge in edges:
                src_id = file_symbol_map.get(edge.source_qualified_name)
                if not src_id:
                    # Look up by exact qualified name or short name (for cross-file TESTED_BY)
                    src_id = global_symbol_map.get(edge.source_qualified_name)
                if not src_id:
                    src_short = edge.source_qualified_name.split(".")[-1]
                    src_matches = short_name_map.get(src_short, [])
                    if src_matches:
                        src_id = src_matches[0]
                if not src_id:
                    continue

                target_id = global_symbol_map.get(edge.target_qualified_name)
                if not target_id:
                    target_short = edge.target_qualified_name.split(".")[-1]
                    matches = short_name_map.get(target_short, [])
                    if len(matches) == 1:
                        target_id = matches[0]

                edge_rec = ASTEdgeRecord(
                    edge_id=uuid4(),
                    domain_id=domain_id,
                    source_symbol_id=src_id,
                    target_qualified_name=edge.target_qualified_name,
                    target_symbol_id=target_id,
                    edge_kind=edge.edge_kind,
                    allowed_scopes=file_rec.allowed_scopes,
                    tenant_id=principal.tenant_id,
                )
                repo.insert_ast_edge(principal, edge_rec)
                total_edges += 1

        # Pass 3: CODEOWNERS & Backstage catalog-info.yaml ownership graph extraction
        if manifest_files and any(
            p in CODEOWNERS_FILENAMES or p.rsplit("/", 1)[-1] in CATALOG_INFO_FILENAMES
            for p in manifest_files
        ):
            extra_syms, extra_edges = self._sync_ownership_in_memory(
                repo=repo,
                principal=principal,
                domain_id=domain_id,
                repo_name=effective_repo_name,
                current_tree=current_tree,
                manifest_files=manifest_files,
                allowed_scopes=scopes,
            )
            total_symbols += extra_syms
            total_edges += extra_edges

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return MerkleSyncStats(
            total_files_scanned=len(current_tree),
            unchanged_files_skipped=len(unchanged),
            new_files_indexed=len(new),
            modified_files_reindexed=len(modified),
            deleted_files_removed=len(deleted),
            total_symbols_stored=total_symbols,
            total_edges_stored=total_edges,
            duration_ms=duration_ms,
            commit_sha=commit_sha,
            cache_hit=False,
        )

    def sync_postgres(
        self,
        repo_dir: Path,
        conn: Connection[tuple[object, ...]],
        repo: PostgresRLSRepository,
        principal: PrincipalContext,
        domain_id: str,
        repo_name: str,
        allowed_scopes: list[str] | None = None,
        ref: str = "HEAD",
        isolate_by_commit: bool = False,
        include_working_tree: bool | None = None,
    ) -> MerkleSyncStats:
        """Incrementally indexes git repository into live PostgreSQL 17 under Forced RLS."""
        start_time = time.perf_counter()
        use_wt = (
            (self.include_working_tree if include_working_tree is None else include_working_tree)
            and ref == "HEAD"
            and not isolate_by_commit
        )
        commit_sha, current_tree = self.get_git_tree(repo_dir, ref, include_working_tree=use_wt)
        effective_repo_name = f"{repo_name}@{commit_sha[:12]}" if isolate_by_commit else repo_name
        existing_hashes = repo.get_file_blob_hashes(conn, domain_id, effective_repo_name)
        existing_map = {path: blob_sha for path, (_, blob_sha) in existing_hashes.items()}

        unchanged, new, modified, deleted = self.diff_merkle_tree(current_tree, existing_map)

        # Fast-exit cache hit (<50ms)
        if not new and not modified and not deleted:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return MerkleSyncStats(
                total_files_scanned=len(current_tree),
                unchanged_files_skipped=len(unchanged),
                new_files_indexed=0,
                modified_files_reindexed=0,
                deleted_files_removed=0,
                total_symbols_stored=0,
                total_edges_stored=0,
                duration_ms=duration_ms,
                commit_sha=commit_sha,
                cache_hit=True,
            )

        # 1. Remove deleted files
        for d in deleted:
            if d in existing_hashes:
                file_id, _ = existing_hashes[d]
                repo.delete_code_file(conn, file_id)

        # 2. Remove modified files prior to re-indexing
        for m in modified:
            if m in existing_hashes:
                file_id, _ = existing_hashes[m]
                repo.delete_code_file(conn, file_id)

        attr_matcher, manifest_files = self._build_attr_matcher_and_manifests(
            repo_dir, current_tree, ref, prefer_working_tree=use_wt
        )
        if any(p.rsplit("/", 1)[-1] == ".gitattributes" for p in (new | modified | deleted)):
            for u_path in unchanged:
                cls_res = self._classify_path(attr_matcher, u_path)
                conn.execute(
                    """
                    UPDATE code_files
                    SET ranking_weight = %s, is_generated = %s, is_vendored = %s
                    WHERE domain_id = %s AND repo_name = %s AND file_path = %s;
                    """,
                    (
                        cls_res.ranking_weight,
                        cls_res.is_generated,
                        cls_res.is_vendored,
                        domain_id,
                        effective_repo_name,
                        u_path,
                    ),
                )

        scopes = allowed_scopes or principal.allowed_scopes
        to_index = sorted(new | modified)

        total_symbols = 0
        total_edges = 0

        parsed_files_data: list[
            tuple[CodeFileRecord, list[ParsedSymbol], list[ParsedEdge], dict[str, UUID]]
        ] = []
        global_symbol_map: dict[str, UUID] = {}
        short_name_map: dict[str, list[UUID]] = defaultdict(list)

        # Query existing symbols in DB for cross-file linking
        existing_sym_rows = conn.execute(
            "SELECT symbol_id, qualified_name FROM ast_symbols WHERE domain_id = %s;",
            (domain_id,),
        ).fetchall()
        for s_row in existing_sym_rows:
            s_id = UUID(str(s_row[0]))
            s_qual = str(s_row[1])
            global_symbol_map[s_qual] = s_id
            short_name_map[s_qual.split(".")[-1]].append(s_id)

        pre_parsed_pg: dict[str, tuple[str, str | None, list[ParsedSymbol], list[ParsedEdge]]] = {}
        batch_emb_items_pg: list[tuple[str, list[ParsedSymbol]]] = []
        for file_path in to_index:
            blob_sha = current_tree[file_path]
            raw_content = self._read_blob_or_disk(
                repo_dir, file_path, blob_sha, ref, prefer_working_tree=use_wt
            )
            if raw_content is None:
                continue
            lang = self._detect_language(file_path)
            if lang:
                symbols, edges = self.extractor.parse_source(raw_content, lang, file_path=file_path)
                batch_emb_items_pg.append((blob_sha, symbols))
            else:
                symbols, edges = [], []
            pre_parsed_pg[file_path] = (raw_content, lang, symbols, edges)

        self._batch_prepopulate_symbol_embeddings(
            batch_emb_items_pg,
            heartbeat_fn=lambda: conn.execute("SELECT 1;"),
        )

        for file_path in to_index:
            parsed_entry_pg = pre_parsed_pg.get(file_path)
            if parsed_entry_pg is None:
                continue
            blob_sha = current_tree[file_path]
            raw_content, lang, symbols, edges = parsed_entry_pg
            lines = raw_content.splitlines()
            file_id = uuid4()
            cls_res = self._classify_path(attr_matcher, file_path)

            file_rec = CodeFileRecord(
                file_id=file_id,
                domain_id=domain_id,
                repo_name=effective_repo_name,
                file_path=file_path,
                git_commit_sha=commit_sha,
                git_blob_sha=blob_sha,
                language=lang or "text",
                allowed_scopes=scopes,
                raw_content=raw_content,
                line_count=len(lines),
                tenant_id=principal.tenant_id,
                ranking_weight=cls_res.ranking_weight,
                is_generated=cls_res.is_generated,
                is_vendored=cls_res.is_vendored,
            )
            repo.insert_code_file(conn, file_rec)

            if not lang:
                continue

            file_symbol_map: dict[str, UUID] = {}
            symbol_embeddings = self._compute_symbol_embeddings(symbols, blob_sha=blob_sha)

            for parsed_sym, emb in zip(symbols, symbol_embeddings, strict=True):
                sym_id = uuid4()
                sym_rec = ASTSymbolRecord(
                    symbol_id=sym_id,
                    file_id=file_id,
                    domain_id=domain_id,
                    qualified_name=parsed_sym.qualified_name,
                    symbol_kind=parsed_sym.symbol_kind,
                    start_line=parsed_sym.start_line,
                    end_line=parsed_sym.end_line,
                    signature_skeleton=parsed_sym.signature_skeleton,
                    contextual_summary=parsed_sym.contextual_summary,
                    allowed_scopes=scopes,
                    embedding=emb,
                    repo_qualified_id=parsed_sym.repo_qualified_id,
                    bare_name=parsed_sym.bare_name or parsed_sym.name,
                    tenant_id=principal.tenant_id,
                )
                repo.insert_ast_symbol(conn, sym_rec)
                file_symbol_map[parsed_sym.qualified_name] = sym_id
                global_symbol_map[parsed_sym.qualified_name] = sym_id
                short_name_map[parsed_sym.name].append(sym_id)
                total_symbols += 1

            parsed_files_data.append((file_rec, symbols, edges, file_symbol_map))

        # Pass 2: Directed Edge Resolution
        for file_rec, _symbols, edges, file_symbol_map in parsed_files_data:
            for edge in edges:
                src_id = file_symbol_map.get(edge.source_qualified_name)
                if not src_id:
                    src_id = global_symbol_map.get(edge.source_qualified_name)
                if not src_id:
                    src_short = edge.source_qualified_name.split(".")[-1]
                    src_matches = short_name_map.get(src_short, [])
                    if src_matches:
                        src_id = src_matches[0]
                if not src_id:
                    continue

                target_id = global_symbol_map.get(edge.target_qualified_name)
                if not target_id:
                    target_short = edge.target_qualified_name.split(".")[-1]
                    matches = short_name_map.get(target_short, [])
                    if len(matches) == 1:
                        target_id = matches[0]

                edge_rec = ASTEdgeRecord(
                    edge_id=uuid4(),
                    domain_id=domain_id,
                    source_symbol_id=src_id,
                    target_qualified_name=edge.target_qualified_name,
                    target_symbol_id=target_id,
                    edge_kind=edge.edge_kind,
                    allowed_scopes=file_rec.allowed_scopes,
                    tenant_id=principal.tenant_id,
                )
                repo.insert_ast_edge(conn, edge_rec)
                total_edges += 1

        # Pass 3: CODEOWNERS & Backstage catalog-info.yaml ownership graph extraction
        if manifest_files and any(
            p in CODEOWNERS_FILENAMES or p.rsplit("/", 1)[-1] in CATALOG_INFO_FILENAMES
            for p in manifest_files
        ):
            extra_syms, extra_edges = self._sync_ownership_postgres(
                conn=conn,
                repo=repo,
                principal=principal,
                domain_id=domain_id,
                repo_name=effective_repo_name,
                current_tree=current_tree,
                manifest_files=manifest_files,
                allowed_scopes=scopes,
            )
            total_symbols += extra_syms
            total_edges += extra_edges

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return MerkleSyncStats(
            total_files_scanned=len(current_tree),
            unchanged_files_skipped=len(unchanged),
            new_files_indexed=len(new),
            modified_files_reindexed=len(modified),
            deleted_files_removed=len(deleted),
            total_symbols_stored=total_symbols,
            total_edges_stored=total_edges,
            duration_ms=duration_ms,
            commit_sha=commit_sha,
            cache_hit=False,
        )

    def sync_push_incremental_postgres(
        self,
        repo_dir: Path,
        conn: Connection[tuple[object, ...]],
        repo: PostgresRLSRepository,
        principal: PrincipalContext,
        domain_id: str,
        repo_name: str,
        before_sha: str,
        after_sha: str,
        allowed_scopes: list[str] | None = None,
    ) -> MerkleSyncStats:
        """Executes surgical incremental update on push in <50ms without dropping inward edges."""
        start_time = time.perf_counter()
        scopes = allowed_scopes or principal.allowed_scopes

        diff = self.diff_commit_blobs(repo_dir, before_sha, after_sha)

        # 1. Process deletions
        all_deleted_paths = set(diff.deleted.keys()) | {r[0] for r in diff.renamed}
        if all_deleted_paths:
            conn.execute(
                """
                DELETE FROM code_files
                WHERE domain_id = %s AND repo_name = %s AND file_path = ANY(%s);
                """,
                (domain_id, repo_name, list(all_deleted_paths)),
            )

        # 2. Process modified files: delete old symbols (cascades outward edges, NULLs inward edges)
        all_modified_paths = set(diff.modified.keys())
        if all_modified_paths:
            conn.execute(
                """
                DELETE FROM ast_symbols
                WHERE domain_id = %s
                  AND file_id IN (
                      SELECT file_id FROM code_files
                      WHERE domain_id = %s AND repo_name = %s AND file_path = ANY(%s)
                  );
                """,
                (domain_id, domain_id, repo_name, list(all_modified_paths)),
            )

        # 3. Parse and insert added and modified files
        to_index_paths = set(diff.added.keys()) | all_modified_paths | {r[1] for r in diff.renamed}
        total_symbols = 0
        total_edges = 0

        existing_paths_rows = conn.execute(
            "SELECT file_path FROM code_files WHERE domain_id = %s AND repo_name = %s;",
            (domain_id, repo_name),
        ).fetchall()
        existing_paths = [str(r[0]) for r in existing_paths_rows]
        attr_matcher: GitAttributesMatcher | None = None
        if has_ownership_or_attr_files(existing_paths) or has_ownership_or_attr_files(
            list(to_index_paths)
        ):
            _, head_tree = self.get_git_tree(repo_dir, after_sha, include_working_tree=False)
            attr_matcher, _ = self._build_attr_matcher_and_manifests(
                repo_dir, head_tree, after_sha, prefer_working_tree=False
            )

        repo_symbol_map: dict[str, UUID] = {}
        existing_sym_rows = conn.execute(
            """
            SELECT s.symbol_id, s.qualified_name
            FROM ast_symbols s
            JOIN code_files f ON f.file_id = s.file_id
            WHERE s.domain_id = %s AND f.repo_name = %s;
            """,
            (domain_id, repo_name),
        ).fetchall()
        for s_row in existing_sym_rows:
            repo_symbol_map[str(s_row[1])] = UUID(str(s_row[0]))

        newly_inserted_file_ids: list[UUID] = []
        parsed_files_data = []

        for f_path in sorted(to_index_paths):
            new_blob_sha = (
                diff.added.get(f_path)
                or (diff.modified[f_path][1] if f_path in diff.modified else None)
                or next((r[3] for r in diff.renamed if r[1] == f_path), "")
            )
            raw_content = self._read_blob_or_disk(repo_dir, f_path, new_blob_sha or "", after_sha)
            if raw_content is None:
                continue
            lang = self._detect_language(f_path)
            lines = raw_content.splitlines()
            cls_res = self._classify_path(attr_matcher, f_path)

            file_id = uuid4()
            row = conn.execute(
                """
                INSERT INTO code_files (
                    file_id, tenant_id, domain_id, repo_name, file_path,
                    git_commit_sha, git_blob_sha, language,
                    allowed_scopes, raw_content, line_count,
                    ranking_weight, is_generated, is_vendored
                )
                VALUES (
                    %s,
                    %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (tenant_id, domain_id, repo_name, file_path) DO UPDATE SET
                    git_commit_sha = EXCLUDED.git_commit_sha,
                    git_blob_sha = EXCLUDED.git_blob_sha,
                    raw_content = EXCLUDED.raw_content,
                    line_count = EXCLUDED.line_count,
                    ranking_weight = EXCLUDED.ranking_weight,
                    is_generated = EXCLUDED.is_generated,
                    is_vendored = EXCLUDED.is_vendored,
                    indexed_at = NOW()
                RETURNING file_id;
                """,
                (
                    file_id,
                    principal.tenant_id,
                    domain_id,
                    repo_name,
                    f_path,
                    after_sha,
                    new_blob_sha or "",
                    lang or "text",
                    scopes,
                    raw_content,
                    len(lines),
                    cls_res.ranking_weight,
                    cls_res.is_generated,
                    cls_res.is_vendored,
                ),
            ).fetchone()
            actual_file_id = UUID(str(row[0])) if row else file_id
            newly_inserted_file_ids.append(actual_file_id)

            if not lang:
                continue

            symbols, edges = self.extractor.parse_source(
                raw_content, lang, file_path=f_path, repo_name=repo_name
            )
            file_symbol_map: dict[str, UUID] = {}
            symbol_embeddings = self._compute_symbol_embeddings(
                symbols, blob_sha=new_blob_sha or ""
            )

            for parsed_sym, emb in zip(symbols, symbol_embeddings, strict=True):
                sym_id = uuid4()
                sym_rec = ASTSymbolRecord(
                    symbol_id=sym_id,
                    file_id=actual_file_id,
                    domain_id=domain_id,
                    qualified_name=parsed_sym.qualified_name,
                    symbol_kind=parsed_sym.symbol_kind,
                    start_line=parsed_sym.start_line,
                    end_line=parsed_sym.end_line,
                    signature_skeleton=parsed_sym.signature_skeleton,
                    contextual_summary=parsed_sym.contextual_summary,
                    allowed_scopes=scopes,
                    embedding=emb,
                    tenant_id=principal.tenant_id,
                )
                repo.insert_ast_symbol(conn, sym_rec)
                file_symbol_map[parsed_sym.qualified_name] = sym_id
                repo_symbol_map[parsed_sym.qualified_name] = sym_id
                total_symbols += 1

            parsed_files_data.append((actual_file_id, scopes, edges, file_symbol_map))

        # 4. Insert outward edges
        for _actual_file_id, file_scopes, edges, file_symbol_map in parsed_files_data:
            for parsed_edge in edges:
                src_id = file_symbol_map.get(
                    parsed_edge.source_qualified_name
                ) or repo_symbol_map.get(parsed_edge.source_qualified_name)
                if not src_id:
                    continue

                target_id = repo_symbol_map.get(parsed_edge.target_qualified_name)
                target_qual = parsed_edge.target_qualified_name
                if not target_id:
                    target_qual = f"UNRESOLVED::{parsed_edge.target_qualified_name}"

                edge_rec = ASTEdgeRecord(
                    edge_id=uuid4(),
                    domain_id=domain_id,
                    source_symbol_id=src_id,
                    target_qualified_name=target_qual,
                    target_symbol_id=target_id,
                    edge_kind=parsed_edge.edge_kind,
                    allowed_scopes=file_scopes,
                    target_repo_qualified_id=parsed_edge.target_repo_qualified_id,
                    is_cross_repo=parsed_edge.is_cross_repo or target_id is None,
                    tenant_id=principal.tenant_id,
                )
                repo.insert_ast_edge(conn, edge_rec)
                total_edges += 1

        # 5. Inward Edge Re-link: restore existing inward edges targeting updated symbols
        if newly_inserted_file_ids:
            conn.execute(
                """
                UPDATE ast_edges e
                SET target_symbol_id = s.symbol_id,
                    target_qualified_name = s.qualified_name
                FROM ast_symbols s
                WHERE s.domain_id = %s
                  AND s.file_id = ANY(%s)
                  AND e.domain_id = s.domain_id
                  AND e.target_symbol_id IS NULL
                  AND (
                      e.target_qualified_name = s.qualified_name
                      OR e.target_qualified_name = ('UNRESOLVED::' || s.qualified_name)
                      OR ('UNRESOLVED::' || e.target_qualified_name) = s.qualified_name
                      OR s.qualified_name LIKE ('%%::' || e.target_qualified_name)
                  );
                """,
                (domain_id, newly_inserted_file_ids),
            )

        # 6. Reconcile cross-repo edges
        self.reconcile_cross_repo_edges_postgres(conn, domain_id)

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return MerkleSyncStats(
            total_files_scanned=len(to_index_paths),
            unchanged_files_skipped=0,
            new_files_indexed=len(diff.added),
            modified_files_reindexed=len(diff.modified),
            deleted_files_removed=len(all_deleted_paths),
            total_symbols_stored=total_symbols,
            total_edges_stored=total_edges,
            duration_ms=duration_ms,
            commit_sha=after_sha,
            cache_hit=False,
        )

    def reconcile_cross_repo_edges_postgres(
        self, conn: Connection[tuple[object, ...]], domain_id: str
    ) -> int:
        """Phase 2: Resolves UNRESOLVED cross-repo edges across repositories under RLS (F09)."""
        res = conn.execute(
            """
            WITH candidate_matches AS (
                SELECT
                    e.edge_id,
                    s.symbol_id AS matched_symbol_id,
                    s.qualified_name AS matched_qual_id,
                    ROW_NUMBER() OVER (
                        PARTITION BY e.edge_id
                        ORDER BY (s.allowed_scopes && e.allowed_scopes) DESC, s.symbol_id
                    ) AS match_rank
                FROM ast_edges e
                JOIN ast_symbols s ON s.domain_id = e.domain_id
                WHERE e.domain_id = %s
                  AND e.target_symbol_id IS NULL
                  AND e.target_qualified_name LIKE 'UNRESOLVED::%%'
                  AND (
                      s.qualified_name = SUBSTRING(e.target_qualified_name FROM 13)
                      OR s.qualified_name LIKE (
                          '%%::' || SUBSTRING(e.target_qualified_name FROM 13)
                      )
                      OR s.qualified_name LIKE (
                          '%%' || REPLACE(SUBSTRING(e.target_qualified_name FROM 13), '.', '::%%')
                      )
                  )
                  AND e.allowed_scopes && s.allowed_scopes
            )
            UPDATE ast_edges e
            SET target_symbol_id = cm.matched_symbol_id,
                target_qualified_name = cm.matched_qual_id,
                is_cross_repo = TRUE
            FROM candidate_matches cm
            WHERE e.edge_id = cm.edge_id AND cm.match_rank = 1;
            """,
            (domain_id,),
        )
        return res.rowcount if hasattr(res, "rowcount") and res.rowcount != -1 else 0

    def sync_push_incremental_inmemory(
        self,
        repo_dir: Path,
        repo: InMemoryRLSRepository,
        principal: PrincipalContext,
        domain_id: str,
        repo_name: str,
        before_sha: str,
        after_sha: str,
        allowed_scopes: list[str] | None = None,
    ) -> MerkleSyncStats:
        """Executes surgical incremental update on push in InMemoryRLSRepository (F08, F09)."""
        start_time = time.perf_counter()
        scopes = allowed_scopes or principal.allowed_scopes

        diff = self.diff_commit_blobs(repo_dir, before_sha, after_sha)

        all_deleted_paths = set(diff.deleted.keys()) | {r[0] for r in diff.renamed}
        all_modified_paths = set(diff.modified.keys())

        # 1. Process deletions
        for file_id, file_rec in list(repo._files.items()):
            if (
                file_rec.domain_id == domain_id
                and file_rec.repo_name == repo_name
                and file_rec.file_path in all_deleted_paths
            ):
                repo.delete_code_file(principal, file_id)

        # 2. Process modified files: delete old symbols & outward edges, preserve inward edges
        for file_id, file_rec in list(repo._files.items()):
            if (
                file_rec.domain_id == domain_id
                and file_rec.repo_name == repo_name
                and file_rec.file_path in all_modified_paths
            ):
                sym_ids_to_del = {
                    s_id for s_id, sym in repo._symbols.items() if sym.file_id == file_id
                }
                for s_id in sym_ids_to_del:
                    repo._symbols.pop(s_id, None)

                # Outward edges
                outward_edge_ids = {
                    e_id
                    for e_id, edge in repo._edges.items()
                    if edge.source_symbol_id in sym_ids_to_del
                }
                for e_id in outward_edge_ids:
                    repo._edges.pop(e_id, None)

                # Inward edges -> target_symbol_id = None
                for e_id, edge in list(repo._edges.items()):
                    if edge.target_symbol_id in sym_ids_to_del:
                        repo._edges[e_id] = ASTEdgeRecord(
                            edge_id=edge.edge_id,
                            domain_id=edge.domain_id,
                            source_symbol_id=edge.source_symbol_id,
                            target_qualified_name=edge.target_qualified_name,
                            target_symbol_id=None,
                            edge_kind=edge.edge_kind,
                            allowed_scopes=edge.allowed_scopes,
                            target_repo_qualified_id=edge.target_repo_qualified_id,
                            is_cross_repo=edge.is_cross_repo,
                        )

        # 3. Parse and insert added and modified files
        to_index_paths = set(diff.added.keys()) | all_modified_paths | {r[1] for r in diff.renamed}
        total_symbols = 0
        total_edges = 0

        existing_paths = [
            f_rec.file_path
            for f_rec in repo._files.values()
            if f_rec.domain_id == domain_id and f_rec.repo_name == repo_name
        ]
        attr_matcher: GitAttributesMatcher | None = None
        if has_ownership_or_attr_files(existing_paths) or has_ownership_or_attr_files(
            list(to_index_paths)
        ):
            _, head_tree = self.get_git_tree(repo_dir, after_sha, include_working_tree=False)
            attr_matcher, _ = self._build_attr_matcher_and_manifests(
                repo_dir, head_tree, after_sha, prefer_working_tree=False
            )

        repo_symbol_map: dict[str, UUID] = {}
        for s_id, s_rec in repo._symbols.items():
            if s_rec.domain_id == domain_id:
                f_rec = repo._files.get(s_rec.file_id)
                if f_rec and f_rec.repo_name == repo_name:
                    repo_symbol_map[s_rec.qualified_name] = s_id

        newly_inserted_file_ids: list[UUID] = []
        parsed_files_data = []

        pre_parsed_inc: dict[
            str, tuple[str, str, str | None, list[ParsedSymbol], list[ParsedEdge]]
        ] = {}
        batch_emb_inc: list[tuple[str, list[ParsedSymbol]]] = []
        for f_path in sorted(to_index_paths):
            new_blob_sha = (
                diff.added.get(f_path)
                or (diff.modified[f_path][1] if f_path in diff.modified else None)
                or next((r[3] for r in diff.renamed if r[1] == f_path), "")
            )
            raw_content = self._read_blob_or_disk(repo_dir, f_path, new_blob_sha or "", after_sha)
            if raw_content is None:
                continue
            lang = self._detect_language(f_path)
            if lang:
                symbols, edges = self.extractor.parse_source(
                    raw_content, lang, file_path=f_path, repo_name=repo_name
                )
                batch_emb_inc.append((new_blob_sha or "", symbols))
            else:
                symbols, edges = [], []
            pre_parsed_inc[f_path] = (new_blob_sha or "", raw_content, lang, symbols, edges)

        self._batch_prepopulate_symbol_embeddings(batch_emb_inc)

        for f_path in sorted(to_index_paths):
            parsed_inc = pre_parsed_inc.get(f_path)
            if parsed_inc is None:
                continue
            new_blob_sha, raw_content, lang, symbols, edges = parsed_inc
            lines = raw_content.splitlines()
            cls_res = self._classify_path(attr_matcher, f_path)

            # Existing file_id if modifying
            existing_fid = next(
                (
                    f_id
                    for f_id, f_rec in repo._files.items()
                    if f_rec.domain_id == domain_id
                    and f_rec.repo_name == repo_name
                    and f_rec.file_path == f_path
                ),
                None,
            )
            actual_file_id = existing_fid or uuid4()

            file_rec = CodeFileRecord(
                file_id=actual_file_id,
                domain_id=domain_id,
                repo_name=repo_name,
                file_path=f_path,
                git_commit_sha=after_sha,
                git_blob_sha=new_blob_sha or "",
                language=lang or "text",
                allowed_scopes=scopes,
                raw_content=raw_content,
                line_count=len(lines),
                tenant_id=principal.tenant_id,
                ranking_weight=cls_res.ranking_weight,
                is_generated=cls_res.is_generated,
                is_vendored=cls_res.is_vendored,
            )
            repo.insert_code_file(principal, file_rec)
            newly_inserted_file_ids.append(actual_file_id)

            if not lang:
                continue

            file_symbol_map: dict[str, UUID] = {}
            symbol_embeddings = self._compute_symbol_embeddings(
                symbols, blob_sha=new_blob_sha or ""
            )

            for parsed_sym, emb in zip(symbols, symbol_embeddings, strict=True):
                sym_id = uuid4()
                sym_rec = ASTSymbolRecord(
                    symbol_id=sym_id,
                    file_id=actual_file_id,
                    domain_id=domain_id,
                    qualified_name=parsed_sym.qualified_name,
                    symbol_kind=parsed_sym.symbol_kind,
                    start_line=parsed_sym.start_line,
                    end_line=parsed_sym.end_line,
                    signature_skeleton=parsed_sym.signature_skeleton,
                    contextual_summary=parsed_sym.contextual_summary,
                    allowed_scopes=scopes,
                    embedding=emb,
                    tenant_id=principal.tenant_id,
                )
                repo.insert_ast_symbol(principal, sym_rec)
                file_symbol_map[parsed_sym.qualified_name] = sym_id
                repo_symbol_map[parsed_sym.qualified_name] = sym_id
                total_symbols += 1

            parsed_files_data.append((actual_file_id, scopes, edges, file_symbol_map))

        # 4. Insert outward edges
        for _actual_file_id, file_scopes, edges, file_symbol_map in parsed_files_data:
            for parsed_edge in edges:
                src_id = file_symbol_map.get(
                    parsed_edge.source_qualified_name
                ) or repo_symbol_map.get(parsed_edge.source_qualified_name)
                if not src_id:
                    continue

                target_id = repo_symbol_map.get(parsed_edge.target_qualified_name)
                target_qual = parsed_edge.target_qualified_name
                if not target_id:
                    target_qual = f"UNRESOLVED::{parsed_edge.target_qualified_name}"

                edge_rec = ASTEdgeRecord(
                    edge_id=uuid4(),
                    domain_id=domain_id,
                    source_symbol_id=src_id,
                    target_qualified_name=target_qual,
                    target_symbol_id=target_id,
                    edge_kind=parsed_edge.edge_kind,
                    allowed_scopes=file_scopes,
                    target_repo_qualified_id=parsed_edge.target_repo_qualified_id,
                    is_cross_repo=parsed_edge.is_cross_repo or target_id is None,
                    tenant_id=principal.tenant_id,
                )
                repo.insert_ast_edge(principal, edge_rec)
                total_edges += 1

        # 5. Inward Edge Re-link in memory
        newly_inserted_syms = [
            s
            for s in repo._symbols.values()
            if s.domain_id == domain_id and s.file_id in newly_inserted_file_ids
        ]
        for s in newly_inserted_syms:
            for e_id, edge in list(repo._edges.items()):
                if edge.domain_id == domain_id and edge.target_symbol_id is None:
                    t_qual = edge.target_qualified_name
                    if (
                        t_qual == s.qualified_name
                        or t_qual == f"UNRESOLVED::{s.qualified_name}"
                        or f"UNRESOLVED::{t_qual}" == s.qualified_name
                        or s.qualified_name.endswith(f"::{t_qual}")
                    ):
                        repo._edges[e_id] = ASTEdgeRecord(
                            edge_id=edge.edge_id,
                            domain_id=edge.domain_id,
                            source_symbol_id=edge.source_symbol_id,
                            target_qualified_name=s.qualified_name,
                            target_symbol_id=s.symbol_id,
                            edge_kind=edge.edge_kind,
                            allowed_scopes=edge.allowed_scopes,
                            target_repo_qualified_id=edge.target_repo_qualified_id,
                            is_cross_repo=edge.is_cross_repo,
                        )

        # 6. Reconcile cross-repo edges in memory
        self.reconcile_cross_repo_edges_inmemory(repo, domain_id)

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return MerkleSyncStats(
            total_files_scanned=len(to_index_paths),
            unchanged_files_skipped=0,
            new_files_indexed=len(diff.added),
            modified_files_reindexed=len(diff.modified),
            deleted_files_removed=len(all_deleted_paths),
            total_symbols_stored=total_symbols,
            total_edges_stored=total_edges,
            duration_ms=duration_ms,
            commit_sha=after_sha,
            cache_hit=False,
        )

    def reconcile_cross_repo_edges_inmemory(
        self, repo: InMemoryRLSRepository, domain_id: str
    ) -> int:
        """Phase 2: Resolves UNRESOLVED cross-repo edges in InMemoryRLSRepository (F09)."""
        from context_scout.db.repositories import _scopes_intersect

        resolved_count = 0
        all_symbols = [s for s in repo._symbols.values() if s.domain_id == domain_id]

        for e_id, edge in list(repo._edges.items()):
            if (
                edge.domain_id == domain_id
                and edge.target_symbol_id is None
                and edge.target_qualified_name.startswith("UNRESOLVED::")
            ):
                unresolved_target = edge.target_qualified_name[12:]
                candidate: ASTSymbolRecord | None = None
                for s in all_symbols:
                    if not _scopes_intersect(edge.allowed_scopes, s.allowed_scopes):
                        continue
                    if (
                        s.qualified_name == unresolved_target
                        or s.qualified_name.endswith(f"::{unresolved_target}")
                        or s.qualified_name.endswith(f"::{unresolved_target.replace('.', '::')}")
                        or unresolved_target in s.qualified_name
                    ):
                        candidate = s
                        break

                if candidate is not None:
                    repo._edges[e_id] = ASTEdgeRecord(
                        edge_id=edge.edge_id,
                        domain_id=edge.domain_id,
                        source_symbol_id=edge.source_symbol_id,
                        target_qualified_name=candidate.qualified_name,
                        target_symbol_id=candidate.symbol_id,
                        edge_kind=edge.edge_kind,
                        allowed_scopes=edge.allowed_scopes,
                        target_repo_qualified_id=candidate.qualified_name,
                        is_cross_repo=True,
                    )
                    resolved_count += 1

        return resolved_count

    def _build_ownership_records(
        self,
        *,
        domain_id: str,
        repo_name: str,
        tenant_id: UUID,
        allowed_scopes: list[str],
        current_tree: Mapping[str, str],
        manifest_files: Mapping[str, str],
        code_files_by_path: Mapping[str, CodeFileRecord],
        code_symbols: Sequence[ASTSymbolRecord],
    ) -> tuple[list[ASTSymbolRecord], list[ASTEdgeRecord]]:
        file_id_to_path = {cf.file_id: cf.file_path for cf in code_files_by_path.values()}
        code_symbols_by_path: dict[str, list[str]] = defaultdict(list)
        symbol_ids_by_qn: dict[str, UUID] = {}
        for sym in code_symbols:
            symbol_ids_by_qn[sym.qualified_name] = sym.symbol_id
            if sym.symbol_kind in {"team", "system", "component", "api"}:
                continue
            f_path = file_id_to_path.get(sym.file_id)
            if f_path is not None:
                code_symbols_by_path[f_path].append(sym.qualified_name)

        extract_input: dict[str, str] = {p: manifest_files.get(p, "") for p in current_tree}
        graph = OwnershipGraphExtractor().extract(
            repo_name=repo_name,
            files=extract_input,
            code_symbols_by_path=code_symbols_by_path,
        )
        if not graph.symbols and not graph.edges:
            return [], []

        host_file: CodeFileRecord | None = None
        for p in sorted(manifest_files):
            base = p.rsplit("/", 1)[-1]
            if base in CATALOG_INFO_FILENAMES and p in code_files_by_path:
                host_file = code_files_by_path[p]
                break
        if host_file is None:
            for p in CODEOWNERS_FILENAMES:
                if p in code_files_by_path:
                    host_file = code_files_by_path[p]
                    break
        if host_file is None and code_files_by_path:
            host_file = next(iter(code_files_by_path.values()))
        if host_file is None:
            return [], []

        new_symbols: list[ASTSymbolRecord] = []
        pending_parsed: list[ParsedSymbol] = []
        for p_sym in graph.symbols:
            if p_sym.qualified_name in symbol_ids_by_qn:
                continue
            pending_parsed.append(p_sym)

        if pending_parsed:
            embs = self._compute_symbol_embeddings(pending_parsed, blob_sha=host_file.git_blob_sha)
            for p_sym, emb in zip(pending_parsed, embs, strict=True):
                sid = uuid4()
                symbol_ids_by_qn[p_sym.qualified_name] = sid
                new_symbols.append(
                    ASTSymbolRecord(
                        symbol_id=sid,
                        file_id=host_file.file_id,
                        domain_id=domain_id,
                        qualified_name=p_sym.qualified_name,
                        symbol_kind=p_sym.symbol_kind,
                        start_line=p_sym.start_line,
                        end_line=p_sym.end_line,
                        signature_skeleton=p_sym.signature_skeleton,
                        contextual_summary=p_sym.contextual_summary,
                        allowed_scopes=list(allowed_scopes),
                        embedding=emb,
                        repo_qualified_id=p_sym.repo_qualified_id,
                        bare_name=p_sym.bare_name or p_sym.name,
                        tenant_id=tenant_id,
                    )
                )

        new_edges: list[ASTEdgeRecord] = []
        for p_edge in graph.edges:
            src_id = symbol_ids_by_qn.get(p_edge.source_qualified_name)
            tgt_id = symbol_ids_by_qn.get(p_edge.target_qualified_name)
            if src_id is None:
                continue
            new_edges.append(
                ASTEdgeRecord(
                    edge_id=uuid4(),
                    domain_id=domain_id,
                    source_symbol_id=src_id,
                    target_qualified_name=p_edge.target_qualified_name,
                    target_symbol_id=tgt_id,
                    edge_kind=p_edge.edge_kind,
                    allowed_scopes=list(allowed_scopes),
                    target_repo_qualified_id=p_edge.target_repo_qualified_id,
                    is_cross_repo=False,
                    tenant_id=tenant_id,
                )
            )
        return new_symbols, new_edges

    def _sync_ownership_in_memory(
        self,
        *,
        repo: InMemoryRLSRepository,
        principal: PrincipalContext,
        domain_id: str,
        repo_name: str,
        current_tree: Mapping[str, str],
        manifest_files: Mapping[str, str],
        allowed_scopes: list[str],
    ) -> tuple[int, int]:
        code_files_by_path = {
            cf.file_path: cf
            for cf in repo.list_visible_code_files(principal, domain_id)
            if cf.repo_name == repo_name
        }
        repo_file_ids = {cf.file_id for cf in code_files_by_path.values()}
        # Clear prior ownership edges/symbols for this repo before re-materializing.
        ownership_sym_ids = {
            s.symbol_id
            for s in repo.list_visible_ast_symbols(principal, domain_id)
            if s.file_id in repo_file_ids
            and s.symbol_kind in {"team", "system", "component", "api"}
        }
        for e_id, e in list(repo._edges.items()):
            src_sym = repo._symbols.get(e.source_symbol_id)
            if (
                e.tenant_id == principal.tenant_id
                and e.domain_id == domain_id
                and e.edge_kind in {"OWNED_BY_TEAM", "PART_OF_SYSTEM"}
                and (
                    e.source_symbol_id in ownership_sym_ids
                    or e.target_symbol_id in ownership_sym_ids
                    or (src_sym is not None and src_sym.file_id in repo_file_ids)
                )
            ):
                repo._edges.pop(e_id, None)
        for sid in ownership_sym_ids:
            repo._symbols.pop(sid, None)

        repo_syms = [
            s
            for s in repo.list_visible_ast_symbols(principal, domain_id)
            if s.file_id in repo_file_ids
        ]
        new_syms, new_edges = self._build_ownership_records(
            domain_id=domain_id,
            repo_name=repo_name,
            tenant_id=principal.tenant_id,
            allowed_scopes=allowed_scopes,
            current_tree=current_tree,
            manifest_files=manifest_files,
            code_files_by_path=code_files_by_path,
            code_symbols=repo_syms,
        )
        for sym in new_syms:
            repo.insert_ast_symbol(principal, sym)
        for edge in new_edges:
            repo.insert_ast_edge(principal, edge)
        return len(new_syms), len(new_edges)

    def _sync_ownership_postgres(
        self,
        *,
        conn: Connection[tuple[object, ...]],
        repo: PostgresRLSRepository,
        principal: PrincipalContext,
        domain_id: str,
        repo_name: str,
        current_tree: Mapping[str, str],
        manifest_files: Mapping[str, str],
        allowed_scopes: list[str],
    ) -> tuple[int, int]:
        code_files_by_path = {
            cf.file_path: cf
            for cf in repo.get_code_files_by_paths(conn, domain_id, list(current_tree.keys()))
            if cf.repo_name == repo_name
        }
        repo_file_ids = [cf.file_id for cf in code_files_by_path.values()]
        if not repo_file_ids:
            return 0, 0

        conn.execute(
            """
            DELETE FROM ast_edges
            WHERE domain_id = %s
              AND edge_kind IN ('OWNED_BY_TEAM', 'PART_OF_SYSTEM')
              AND source_symbol_id IN (
                  SELECT symbol_id FROM ast_symbols
                  WHERE domain_id = %s AND file_id = ANY(%s::uuid[])
              );
            """,
            (domain_id, domain_id, repo_file_ids),
        )
        conn.execute(
            """
            DELETE FROM ast_symbols
            WHERE domain_id = %s
              AND file_id = ANY(%s::uuid[])
              AND symbol_kind IN ('team', 'system', 'component', 'api');
            """,
            (domain_id, repo_file_ids),
        )

        sym_rows = conn.execute(
            """
            SELECT symbol_id, file_id, domain_id, qualified_name, symbol_kind,
                   start_line, end_line, signature_skeleton, contextual_summary,
                   allowed_scopes, repo_qualified_id, bare_name, tenant_id
            FROM ast_symbols
            WHERE domain_id = %s AND file_id = ANY(%s::uuid[]);
            """,
            (domain_id, repo_file_ids),
        ).fetchall()
        repo_syms = [
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
                allowed_scopes=[str(s) for s in r[9]] if isinstance(r[9], list) else [],
                embedding=[],
                repo_qualified_id=str(r[10]) if r[10] is not None else None,
                bare_name=str(r[11]) if r[11] is not None else None,
                tenant_id=UUID(str(r[12])),
            )
            for r in sym_rows
        ]
        new_syms, new_edges = self._build_ownership_records(
            domain_id=domain_id,
            repo_name=repo_name,
            tenant_id=principal.tenant_id,
            allowed_scopes=allowed_scopes,
            current_tree=current_tree,
            manifest_files=manifest_files,
            code_files_by_path=code_files_by_path,
            code_symbols=repo_syms,
        )
        for sym in new_syms:
            repo.insert_ast_symbol(conn, sym)
        for edge in new_edges:
            repo.insert_ast_edge(conn, edge)
        return len(new_syms), len(new_edges)

    def _build_overlay_snapshot_and_deltas(
        self,
        repo_dir: Path,
        *,
        principal: PrincipalContext,
        domain_id: str,
        repo_name: str,
        ref_name: str,
        base_ref: str,
        head_ref: str,
        ref_kind: Literal["base_branch", "feature_branch", "merge_request", "pull_request"],
        mr_iid: str | None,
        base_snapshot_id: UUID | None,
        state: Literal["active", "open", "merged", "closed"],
        scopes: list[str],
        metadata_json: dict[str, Any] | None,
    ) -> tuple[RefSnapshotRecord, list[OverlayFileDeltaRecord]]:
        base_sha = _resolve_commit_sha(repo_dir, base_ref)
        head_sha = _resolve_commit_sha(repo_dir, head_ref)
        diff = self.diff_commit_blobs(repo_dir, base_sha, head_sha, detect_renames=True)

        _, head_tree = self.get_git_tree(repo_dir, head_sha, include_working_tree=False)
        attr_matcher, _ = self._build_attr_matcher_and_manifests(
            repo_dir, head_tree, head_sha, prefer_working_tree=False
        )

        snapshot = RefSnapshotRecord(
            snapshot_id=uuid5(
                NAMESPACE_URL, f"snapshot:{principal.tenant_id}:{domain_id}:{repo_name}:{ref_name}"
            ),
            domain_id=domain_id,
            repo_name=repo_name,
            ref_name=ref_name,
            ref_kind=ref_kind,
            mr_iid=mr_iid,
            base_snapshot_id=base_snapshot_id,
            base_commit_sha=base_sha,
            head_commit_sha=head_sha,
            state=state,
            allowed_scopes=scopes,
            metadata_json=metadata_json or {},
            tenant_id=principal.tenant_id,
        )

        deltas: list[OverlayFileDeltaRecord] = []
        for path, old_blob in sorted(diff.deleted.items()):
            lang = self._detect_language(path) or "text"
            deltas.append(
                OverlayFileDeltaRecord(
                    delta_id=uuid4(),
                    snapshot_id=snapshot.snapshot_id,
                    domain_id=domain_id,
                    repo_name=repo_name,
                    file_path=path,
                    change_type="deleted",
                    base_blob_sha=old_blob,
                    overlay_blob_sha=None,
                    language=lang,
                    raw_content=None,
                    line_count=0,
                    allowed_scopes=scopes,
                    tenant_id=principal.tenant_id,
                )
            )

        upserted_specs: list[
            tuple[str, str | None, Literal["added", "modified", "renamed"], str | None, str]
        ] = []
        for path, blob_sha in sorted(diff.added.items()):
            upserted_specs.append((path, None, "added", None, blob_sha))
        for path, (old_blob, new_blob) in sorted(diff.modified.items()):
            upserted_specs.append((path, None, "modified", old_blob, new_blob))
        for old_path, new_path, old_blob, new_blob in sorted(diff.renamed):
            upserted_specs.append((new_path, old_path, "renamed", old_blob, new_blob))

        for new_path, prev_path, change_type, base_blob, overlay_blob in upserted_specs:
            raw_content = self._read_blob_or_disk(
                repo_dir, new_path, overlay_blob, head_sha, prefer_working_tree=False
            )
            if raw_content is None:
                continue
            detected_lang = self._detect_language(new_path)
            cls_res = self._classify_path(attr_matcher, new_path)
            sym_dicts: list[dict[str, Any]] = []
            edge_dicts: list[dict[str, Any]] = []
            if detected_lang:
                parsed_syms, parsed_edges = self.extractor.parse_source(
                    raw_content, detected_lang, file_path=new_path, repo_name=repo_name
                )
                sym_dicts, edge_dicts = self._serialize_delta_symbols_edges(
                    domain_id=domain_id,
                    repo_name=repo_name,
                    file_path=new_path,
                    symbols=parsed_syms,
                    edges=parsed_edges,
                )

            deltas.append(
                OverlayFileDeltaRecord(
                    delta_id=uuid4(),
                    snapshot_id=snapshot.snapshot_id,
                    domain_id=domain_id,
                    repo_name=repo_name,
                    file_path=new_path,
                    previous_file_path=prev_path,
                    change_type=change_type,
                    base_blob_sha=base_blob,
                    overlay_blob_sha=overlay_blob,
                    language=detected_lang or "text",
                    raw_content=raw_content,
                    line_count=len(raw_content.splitlines()),
                    ranking_weight=cls_res.ranking_weight,
                    is_generated=cls_res.is_generated,
                    is_vendored=cls_res.is_vendored,
                    symbols_json=sym_dicts,
                    edges_json=edge_dicts,
                    allowed_scopes=scopes,
                    tenant_id=principal.tenant_id,
                )
            )
        return snapshot, deltas

    def sync_overlay_postgres(
        self,
        repo_dir: Path,
        conn: Connection[tuple[object, ...]],
        repo: PostgresRLSRepository,
        principal: PrincipalContext,
        domain_id: str,
        repo_name: str,
        *,
        ref_name: str,
        base_ref: str = "HEAD",
        head_ref: str,
        ref_kind: Literal["base_branch", "feature_branch", "merge_request", "pull_request"] = (
            "merge_request"
        ),
        mr_iid: str | None = None,
        base_snapshot_id: UUID | None = None,
        state: Literal["active", "open", "merged", "closed"] = "open",
        allowed_scopes: list[str] | None = None,
        metadata_json: dict[str, Any] | None = None,
    ) -> tuple[RefSnapshotRecord, list[OverlayFileDeltaRecord]]:
        """Computes branch/MR diff and stores CoW overlay_file_deltas in PG17 under RLS."""
        scopes = list(allowed_scopes or principal.allowed_scopes)
        snapshot, deltas = self._build_overlay_snapshot_and_deltas(
            repo_dir,
            principal=principal,
            domain_id=domain_id,
            repo_name=repo_name,
            ref_name=ref_name,
            base_ref=base_ref,
            head_ref=head_ref,
            ref_kind=ref_kind,
            mr_iid=mr_iid,
            base_snapshot_id=base_snapshot_id,
            state=state,
            scopes=scopes,
            metadata_json=metadata_json,
        )
        stored_snap = repo.upsert_ref_snapshot(conn, snapshot)
        active_paths = [d.file_path for d in deltas]
        conn.execute(
            """
            DELETE FROM overlay_file_deltas
            WHERE domain_id = %s AND snapshot_id = %s
              AND NOT (file_path = ANY(%s::text[]));
            """,
            (domain_id, stored_snap.snapshot_id, active_paths),
        )
        stored_deltas = [
            repo.upsert_overlay_file_delta(conn, dc_replace(d, snapshot_id=stored_snap.snapshot_id))
            for d in deltas
        ]
        return stored_snap, stored_deltas

    def sync_overlay_in_memory(
        self,
        repo_dir: Path,
        repo: InMemoryRLSRepository,
        principal: PrincipalContext,
        domain_id: str,
        repo_name: str,
        *,
        ref_name: str,
        base_ref: str = "HEAD",
        head_ref: str,
        ref_kind: Literal["base_branch", "feature_branch", "merge_request", "pull_request"] = (
            "merge_request"
        ),
        mr_iid: str | None = None,
        base_snapshot_id: UUID | None = None,
        state: Literal["active", "open", "merged", "closed"] = "open",
        allowed_scopes: list[str] | None = None,
        metadata_json: dict[str, Any] | None = None,
    ) -> tuple[RefSnapshotRecord, list[OverlayFileDeltaRecord]]:
        """Computes branch/MR diff and stores CoW overlay_file_deltas in InMemoryRLSRepository."""
        scopes = list(allowed_scopes or principal.allowed_scopes)
        snapshot, deltas = self._build_overlay_snapshot_and_deltas(
            repo_dir,
            principal=principal,
            domain_id=domain_id,
            repo_name=repo_name,
            ref_name=ref_name,
            base_ref=base_ref,
            head_ref=head_ref,
            ref_kind=ref_kind,
            mr_iid=mr_iid,
            base_snapshot_id=base_snapshot_id,
            state=state,
            scopes=scopes,
            metadata_json=metadata_json,
        )
        stored_snap = repo.upsert_ref_snapshot(principal, snapshot)
        active_paths = {d.file_path for d in deltas}
        for key in list(repo._overlay_deltas.keys()):
            if (
                key[0] == principal.tenant_id
                and key[1] == stored_snap.snapshot_id
                and key[2] not in active_paths
            ):
                repo._overlay_deltas.pop(key, None)
        stored_deltas = [
            repo.upsert_overlay_file_delta(
                principal, dc_replace(d, snapshot_id=stored_snap.snapshot_id)
            )
            for d in deltas
        ]
        return stored_snap, stored_deltas


def sync_tracked_in_memory_repos(
    repo: Any, principal: PrincipalContext, domain_id: str
) -> list[MerkleSyncStats]:
    """Automatically re-syncs tracked InMemoryRLSRepository Git repos on HEAD/WT change."""
    if not isinstance(repo, InMemoryRLSRepository):
        return []
    matching_keys = [
        k for k in list(_TRACKED_IN_MEMORY_REPOS.keys()) if k[0] == id(repo) and k[1] == domain_id
    ]
    if not matching_keys:
        return []
    indexer = GitMerkleIndexer()
    results: list[MerkleSyncStats] = []
    for key in matching_keys:
        _repo_id, _dom_id, repo_name = key
        entry = _TRACKED_IN_MEMORY_REPOS.get(key)
        if entry is None:
            continue
        repo_ref, repo_dir, allowed_scopes, _last_sha = entry
        if repo_ref() is not repo:
            _TRACKED_IN_MEMORY_REPOS.pop(key, None)
            continue
        if not repo_dir.exists() or not (repo_dir / ".git").exists():
            continue
        try:
            st = indexer.sync_in_memory(
                repo_dir=repo_dir,
                repo=repo,
                principal=principal,
                domain_id=domain_id,
                repo_name=repo_name,
                allowed_scopes=allowed_scopes,
                ref="HEAD",
                isolate_by_commit=False,
            )
            results.append(st)
        except Exception as exc:
            logger.warning(
                "Incremental repo sync skipped %s (%s): %s",
                repo_dir,
                type(exc).__name__,
                exc,
            )
    return results


__all__ = [
    "CODE_EXTENSIONS",
    "GitMerkleIndexer",
    "MerkleBlobDiff",
    "MerkleSyncStats",
    "compute_git_blob_sha",
    "sync_tracked_in_memory_repos",
]
