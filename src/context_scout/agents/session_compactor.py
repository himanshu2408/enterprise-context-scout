"""Graph 0 Multi-Turn Session Compactor & Git Commit-Drift Detector."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

from context_scout.db.schema import CodeFileRecord, ScoutSessionRecord
from context_scout.schemas.contracts import (
    ImplementationContextPack,
    InsufficientContextDiagnostic,
    PrincipalContext,
)
from context_scout.security.betrvg_privacy import apply_betrvg_to_principal
from context_scout.telemetry.prometheus_metrics import get_prometheus_registry

logger = logging.getLogger(__name__)

_COREF_WORDS: Final[tuple[str, ...]] = (
    "it",
    "its",
    "that",
    "those",
    "these",
    "them",
    "same",
    "also",
    "previous",
    "earlier",
    "above",
)


@dataclass(frozen=True)
class SessionPreflightResult:
    """Graph 0 preflight session load, compaction context, and commit-drift warnings."""

    session_id: str
    prior_session: ScoutSessionRecord | None = None
    turn_number: int = 1
    compacted_summary: str = ""
    pinned_symbols: tuple[str, ...] = ()
    commit_drift_warnings: tuple[str, ...] = ()
    effective_query: str = ""
    session_hit: bool = False


def _extract_blob_key(repo_name: str, file_path: str) -> str:
    return f"{repo_name}::{file_path}"


def _build_turn_summary(
    *,
    turn_num: int,
    user_prompt: str,
    result: ImplementationContextPack | InsufficientContextDiagnostic,
) -> dict[str, Any]:
    prompt_snippet = " ".join(user_prompt.split())[:180]
    if isinstance(result, ImplementationContextPack):
        files = list(dict.fromkeys(s.file_path for s in result.code_slices))[:6]
        symbols = list(dict.fromkeys(s.qualified_symbol for s in result.code_slices))[:6]
        citations = list(dict.fromkeys(c.doc_title for c in result.supporting_doc_citations))[:4]
        return {
            "turn": turn_num,
            "query": prompt_snippet,
            "status": "ready",
            "files": files,
            "symbols": symbols,
            "citations": citations,
            "synthesis_mode": result.synthesis_mode,
        }
    return {
        "turn": turn_num,
        "query": prompt_snippet,
        "status": "abstained",
        "reason": result.reason[:180],
        "missing": [str(m) for m in result.missing_symbols[:5]],
    }


def _compact_turn_line(turn: dict[str, Any]) -> str:
    t_num = int(turn.get("turn", 0))
    query = str(turn.get("query", "")).strip()
    status = str(turn.get("status", "ready"))
    if status == "ready":
        syms = ", ".join(str(s) for s in (turn.get("symbols") or [])[:4])
        files = ", ".join(str(f) for f in (turn.get("files") or [])[:4])
        cits = ", ".join(str(c) for c in (turn.get("citations") or [])[:2])
        parts = [p for p in (syms, files, cits) if p]
        joined = "; ".join(parts) if parts else "grounded"
        return f"T{t_num} [{query}] -> {joined}"
    reason = str(turn.get("reason", "abstained"))
    return f"T{t_num} [{query}] -> abstained ({reason})"


class ScoutSessionManager:
    """Manages RLS-isolated `scout_sessions` compaction and `git_blob_sha` commit-drift checks."""

    def __init__(
        self,
        *,
        max_recent_turns: int = 3,
        max_summary_chars: int = 2_400,
    ) -> None:
        self.max_recent_turns = max(1, int(max_recent_turns))
        self.max_summary_chars = max(256, int(max_summary_chars))

    def preflight_session(
        self,
        repo: Any,
        principal: PrincipalContext,
        *,
        session_id: str | None,
        domain_id: str,
        user_prompt: str,
        visible_files: Sequence[CodeFileRecord],
    ) -> SessionPreflightResult:
        """Loads prior turn state, checks `git_blob_sha` drift, and expands follow-up queries."""
        cleaned_prompt = user_prompt.strip()
        if not session_id or not session_id.strip():
            return SessionPreflightResult(
                session_id="",
                effective_query=cleaned_prompt,
            )

        sid = session_id.strip()
        storage_principal = apply_betrvg_to_principal(principal)
        existing: ScoutSessionRecord | None = None
        if repo is not None and hasattr(repo, "get_scout_session"):
            try:
                loaded = repo.get_scout_session(storage_principal, sid)
                if isinstance(loaded, ScoutSessionRecord):
                    existing = loaded
            except Exception as exc:
                logger.debug(
                    "Unable to load scout_session '%s' (%s): %s",
                    sid,
                    type(exc).__name__,
                    exc,
                )

        if existing is None:
            return SessionPreflightResult(
                session_id=sid,
                effective_query=cleaned_prompt,
            )

        same_domain = existing.domain_id == domain_id
        current_by_key: dict[str, CodeFileRecord] = {
            _extract_blob_key(cf.repo_name, cf.file_path): cf for cf in visible_files
        }
        drift_warnings: list[str] = []
        # Scope COMMIT_DRIFT_WARNING strictly to active code-repository sessions whose pinned
        # files belong to the queried domain (never bleed Python blob warnings onto doc domains).
        if same_domain and current_by_key:
            for pinned_key, meta in sorted(existing.pinned_file_blobs_json.items()):
                meta_domain = str(meta.get("domain_id", existing.domain_id or "")).strip()
                if meta_domain and meta_domain != domain_id:
                    continue
                old_blob = str(meta.get("git_blob_sha", ""))
                old_commit = str(meta.get("git_commit_sha", ""))
                prev_turn = str(meta.get("turn", existing.turn_count))
                cur_file = current_by_key.get(pinned_key)
                if cur_file is None:
                    drift_warnings.append(
                        f"COMMIT_DRIFT_WARNING: {pinned_key} pinned at turn {prev_turn} "
                        f"(blob {old_blob[:10]}) is no longer visible or was removed."
                    )
                elif (old_blob and cur_file.git_blob_sha != old_blob) or (
                    old_commit and cur_file.git_commit_sha != old_commit
                ):
                    drift_warnings.append(
                        f"COMMIT_DRIFT_WARNING: {pinned_key} mutated since turn {prev_turn} "
                        f"(blob {old_blob[:10]} -> {cur_file.git_blob_sha[:10]}, "
                        f"commit {old_commit[:10]} -> {cur_file.git_commit_sha[:10]})."
                    )

        if drift_warnings:
            get_prometheus_registry().record_query_outcome(
                tenant_id=principal.tenant_id,
                domain_id=domain_id,
                outcome="commit_drift",
                commit_drift=True,
            )

        pinned_syms = tuple(existing.pinned_symbols_json) if same_domain else ()
        effective_query = cleaned_prompt
        tokens = re.findall(r"[A-Za-z0-9_]+", cleaned_prompt.lower())
        uses_coref = any(w in _COREF_WORDS for w in tokens)
        if same_domain and pinned_syms and uses_coref:
            hint_syms = ", ".join(pinned_syms[:4])
            effective_query = f"{cleaned_prompt} (session context: {hint_syms})"

        return SessionPreflightResult(
            session_id=sid,
            prior_session=existing,
            turn_number=existing.turn_count + 1,
            compacted_summary=existing.compacted_summary,
            pinned_symbols=pinned_syms,
            commit_drift_warnings=tuple(drift_warnings),
            effective_query=effective_query,
            session_hit=True,
        )

    def record_completed_turn(
        self,
        repo: Any,
        principal: PrincipalContext,
        *,
        session_id: str | None,
        domain_id: str,
        user_prompt: str,
        result: ImplementationContextPack | InsufficientContextDiagnostic,
        preflight: SessionPreflightResult | None = None,
    ) -> ScoutSessionRecord | None:
        """Compacts completed turns beyond `max_recent_turns` and persists `scout_sessions`."""
        if not session_id or not session_id.strip():
            return None
        sid = session_id.strip()
        storage_principal = apply_betrvg_to_principal(principal)
        prior = preflight.prior_session if preflight is not None else None
        if prior is None and repo is not None and hasattr(repo, "get_scout_session"):
            try:
                loaded = repo.get_scout_session(storage_principal, sid)
                if isinstance(loaded, ScoutSessionRecord):
                    prior = loaded
            except Exception as exc:
                logger.debug(
                    "Unable to reload scout_session '%s' (%s): %s",
                    sid,
                    type(exc).__name__,
                    exc,
                )

        next_turn = (prior.turn_count if prior is not None else 0) + 1
        turn_entry = _build_turn_summary(
            turn_num=next_turn,
            user_prompt=user_prompt,
            result=result,
        )
        recent = list(prior.recent_turns_json) if prior is not None else []
        recent.append(turn_entry)

        summary_lines = (
            [ln for ln in prior.compacted_summary.splitlines() if ln.strip()]
            if prior is not None and prior.compacted_summary
            else []
        )
        compacted_tokens = prior.total_tokens_compacted if prior is not None else 0

        while len(recent) > self.max_recent_turns:
            evicted = recent.pop(0)
            evicted_line = _compact_turn_line(evicted)
            summary_lines.append(evicted_line)
            compacted_tokens += max(1, int(len(str(evicted)) / 3.8))

        compacted_summary = "\n".join(summary_lines)
        if len(compacted_summary) > self.max_summary_chars:
            compacted_summary = compacted_summary[-self.max_summary_chars :]

        pinned_blobs = (
            {str(k): dict(v) for k, v in prior.pinned_file_blobs_json.items()}
            if prior is not None
            else {}
        )
        pinned_syms = list(prior.pinned_symbols_json) if prior is not None else []
        if isinstance(result, ImplementationContextPack):
            for slc in result.code_slices:
                bkey = _extract_blob_key(slc.repo_name, slc.file_path)
                pinned_blobs[bkey] = {
                    "domain_id": domain_id,
                    "git_commit_sha": slc.git_commit_sha,
                    "git_blob_sha": slc.git_blob_sha,
                    "turn": str(next_turn),
                }
                if slc.qualified_symbol and slc.qualified_symbol not in pinned_syms:
                    pinned_syms.append(slc.qualified_symbol)
        pinned_syms = pinned_syms[-16:]

        record = ScoutSessionRecord(
            tenant_id=storage_principal.tenant_id,
            session_id=sid,
            domain_id=domain_id,
            principal_id=storage_principal.user_id,
            allowed_scopes=list(storage_principal.allowed_scopes),
            turn_count=next_turn,
            compacted_summary=compacted_summary,
            recent_turns_json=recent,
            pinned_file_blobs_json=pinned_blobs,
            pinned_symbols_json=pinned_syms,
            total_tokens_compacted=compacted_tokens,
        )
        if repo is not None and hasattr(repo, "upsert_scout_session"):
            try:
                saved = repo.upsert_scout_session(storage_principal, record)
                if isinstance(saved, ScoutSessionRecord):
                    return saved
            except Exception as exc:
                logger.warning(
                    "Failed persisting scout_session '%s' (%s): %s",
                    sid,
                    type(exc).__name__,
                    exc,
                )
        return record


__all__ = [
    "ScoutSessionManager",
    "SessionPreflightResult",
]
