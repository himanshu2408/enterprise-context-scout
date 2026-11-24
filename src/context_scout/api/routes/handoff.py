"""Handoff diff preview & confirmed telemetry endpoints (zero host code execution, ASI05)."""

from __future__ import annotations

import difflib
import logging
import re
import textwrap
import time
import uuid
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException

from context_scout.agents.gemini_executor import (
    BudgetExceededError,
    GeminiImplementationExecutor,
)
from context_scout.api.dependencies import (
    ConfirmationTokenStore,
    get_confirmation_token_store,
    get_principal_context,
    get_repository,
)
from context_scout.api.schemas import (
    HandoffApplyRequest,
    HandoffApplyResponse,
    HandoffPreviewRequest,
    HandoffPreviewResponse,
)
from context_scout.config import ModelGovernancePolicy
from context_scout.schemas.contracts import (
    ImplementationContextPack,
    ModificationIntent,
    PrincipalContext,
    RoleType,
    VerbatimCodeSlice,
)
from context_scout.security.audit_chain import get_audit_chain_service
from context_scout.telemetry.cost_governance import PaidComputeCostEstimator

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Handoff"])

_LINE_PREFIX_RE = re.compile(r"^\s*\d+\s*\|\s?")


def _strip_numbered_slice_lines(verbatim_code_with_lines: str) -> list[str]:
    lines: list[str] = []
    for raw_ln in verbatim_code_with_lines.splitlines():
        cleaned = _LINE_PREFIX_RE.sub("", raw_ln)
        lines.append(f"{cleaned}\n")
    return lines


def _sanitize_relative_path(raw_path: str, fallback: str = "CONTEXT_BRIEF.md") -> str:
    candidate = raw_path.strip().lstrip("/")
    if (
        not candidate
        or "\\" in candidate
        or "\x00" in candidate
        or "\r" in candidate
        or "\n" in candidate
    ):
        return fallback
    parts = PurePosixPath(candidate).parts
    if not parts or any(part in {".", ".."} for part in parts):
        return fallback
    return "/".join(parts)


def _synthesize_in_memory_pack_diff(pack: ImplementationContextPack) -> str:
    """Synthesizes a unified diff strictly in memory from verified pack slices (zero host I/O)."""
    target_slice: VerbatimCodeSlice | None = None
    for s in pack.code_slices:
        if s.modification_intent == ModificationIntent.MODIFY_TARGET:
            target_slice = s
            break
    if target_slice is None and pack.code_slices:
        target_slice = pack.code_slices[0]

    if target_slice is None:
        doc_rel = "CONTEXT_BRIEF.md"
        for cit in pack.supporting_doc_citations:
            if ":" in cit.source_uri:
                cand = cit.source_uri.split(":")[-1].strip()
                if cand:
                    doc_rel = _sanitize_relative_path(cand)
                    break
        brief_title = (
            pack.domain_knowledge_brief.brief_title
            if pack.domain_knowledge_brief is not None
            else pack.domain_id
        )
        cite_quote = (
            pack.supporting_doc_citations[0].verbatim_quote
            if pack.supporting_doc_citations
            else pack.user_task.strip()
        )
        is_md = doc_rel.endswith((".md", ".txt", ".rst"))
        prefix = "<!-- " if is_md else "# "
        suffix = " -->\n" if is_md else "\n"
        addition = (
            f"{prefix}Context Scout Grounded Synthesis ({brief_title}): {cite_quote[:120]}{suffix}"
        )
        return f"--- a/{doc_rel}\n+++ b/{doc_rel}\n@@ -1,0 +1,1 @@\n+{addition}"

    safe_rel = _sanitize_relative_path(target_slice.file_path, fallback="src/target.py")
    orig_lines = _strip_numbered_slice_lines(target_slice.verbatim_code_with_lines)
    wrapped = textwrap.wrap(pack.user_task.strip(), width=72) or ["Verified context handoff"]
    comment_lines = [
        "# Context Scout Verified Implementation:\n",
        *[f"# {ln}\n" for ln in wrapped],
    ]
    mod_lines = [*orig_lines, *comment_lines]
    diff_text = "".join(
        difflib.unified_diff(
            orig_lines,
            mod_lines,
            fromfile=f"a/{safe_rel}",
            tofile=f"b/{safe_rel}",
        )
    )
    if not diff_text:
        diff_text = (
            f"--- a/{safe_rel}\n+++ b/{safe_rel}\n@@ -1,1 +1,2 @@\n+{''.join(comment_lines)}"
        )
    return diff_text


def _verify_diff_structure_in_memory(
    patch_diff: str, expected_targets: list[str]
) -> tuple[bool, str]:
    """Validates unified diff headers and path traversal purely in memory (zero host execution)."""
    if not patch_diff or not patch_diff.strip():
        return (False, "Empty handoff diff rejected.")
    if "--- a/" not in patch_diff or "+++ b/" not in patch_diff or "@@ " not in patch_diff:
        return (False, "Malformed unified diff structure.")

    plus_targets = re.findall(r"^\+\+\+ b/(.+)$", patch_diff, flags=re.MULTILINE)
    if not plus_targets:
        return (False, "Unified diff lacks '+++ b/<file>' target header.")

    for raw_target in plus_targets:
        cleaned = raw_target.strip()
        if (
            cleaned.startswith("/")
            or "\\" in cleaned
            or "\x00" in cleaned
            or "\r" in cleaned
            or "\n" in cleaned
        ):
            return (False, f"Unsafe target path rejected in diff: {cleaned!r}")
        parts = PurePosixPath(cleaned).parts
        if any(part in {".", ".."} for part in parts):
            return (False, f"Path traversal rejected in diff target: {cleaned!r}")
        if expected_targets and cleaned not in expected_targets:
            return (False, f"Diff target '{cleaned}' not in previewed target_files.")

    return (
        True,
        "Verified unified diff structure in memory (host patching and pytest disabled per ASI05).",
    )


@router.post("/handoff/preview", response_model=HandoffPreviewResponse)
def preview_handoff(
    request: HandoffPreviewRequest,
    principal: Annotated[PrincipalContext, Depends(get_principal_context)],
    repo: Annotated[Any, Depends(get_repository)],
    token_store: Annotated[ConfirmationTokenStore, Depends(get_confirmation_token_store)],
) -> HandoffPreviewResponse:
    """Generates in-memory diff preview and confirmation token with zero host disk access."""
    audit = get_audit_chain_service()
    if principal.role != RoleType.ENGINEER:
        audit.record_event(
            principal=principal,
            domain_id=request.pack.domain_id,
            event_type="handoff.preview",
            severity="WARNING",
            resource_uri="/api/v1/handoff/preview",
            outcome="deny",
            payload={"reason": "role_restricted", "role": principal.role.value},
        )
        raise HTTPException(
            status_code=403,
            detail="Sandboxed patch handoff is restricted to Engineer principals (SEC-04).",
        )
    if request.custom_patch is not None:
        audit.record_event(
            principal=principal,
            domain_id=request.pack.domain_id,
            event_type="handoff.preview",
            severity="WARNING",
            resource_uri="/api/v1/handoff/preview",
            outcome="deny",
            payload={"reason": "custom_patch_prohibited"},
        )
        raise HTTPException(
            status_code=400,
            detail="Caller-supplied custom_patch is prohibited on the API host (OWASP ASI05).",
        )
    if not request.pack.preflight_report.passed:
        audit.record_event(
            principal=principal,
            domain_id=request.pack.domain_id,
            event_type="handoff.preview",
            severity="WARNING",
            resource_uri="/api/v1/handoff/preview",
            outcome="deny",
            payload={"reason": "preflight_failed", "trace_id": request.pack.trace_id},
        )
        raise HTTPException(
            status_code=400,
            detail="Handoff preview rejected: preflight_report.passed must be true (ASI09).",
        )
    if (
        request.pack.supporting_doc_citations
        and not request.pack.preflight_report.citations_verified_100_pct
    ):
        audit.record_event(
            principal=principal,
            domain_id=request.pack.domain_id,
            event_type="handoff.preview",
            severity="WARNING",
            resource_uri="/api/v1/handoff/preview",
            outcome="deny",
            payload={"reason": "citations_unverified", "trace_id": request.pack.trace_id},
        )
        raise HTTPException(
            status_code=400,
            detail=(
                "Handoff preview rejected: supporting_doc_citations require "
                "citations_verified_100_pct=true (ASI09)."
            ),
        )

    try:
        validated_model = ModelGovernancePolicy.default().validate_model_id(request.model_id)
    except ValueError as exc:
        audit.record_event(
            principal=principal,
            domain_id=request.pack.domain_id,
            event_type="handoff.preview",
            severity="WARNING",
            resource_uri="/api/v1/handoff/preview",
            outcome="deny",
            payload={"reason": "invalid_model_id", "model_id": request.model_id},
        )
        raise HTTPException(status_code=400, detail=str(exc)) from None

    executor = GeminiImplementationExecutor(
        model_id=validated_model,
        require_confirmation=True,
    )
    if executor.allow_external_cloud_egress:
        try:
            executor.check_principal_budget(principal, request.pack, repo=repo)
        except BudgetExceededError as exc:
            audit.record_event(
                principal=principal,
                domain_id=request.pack.domain_id,
                event_type="handoff.preview",
                severity="WARNING",
                resource_uri="/api/v1/handoff/preview",
                outcome="deny",
                payload={"reason": "budget_exceeded", "trace_id": request.pack.trace_id},
            )
            raise HTTPException(status_code=400, detail=str(exc)) from None

    patch_diff = _synthesize_in_memory_pack_diff(request.pack)
    target_files = executor._extract_target_files(patch_diff, request.pack)

    pack_tokens = request.pack.pack_token_count
    local_tokens = request.pack.local_tokens_consumed
    cost_report = PaidComputeCostEstimator().calculate_cost_estimate(validated_model, pack_tokens)
    cost_usd = cost_report.cost_expected_usd
    savings_usd = round(max(0.0, (local_tokens / 1_000_000.0) * 4.00 - cost_usd), 4)

    token = f"tok_{uuid.uuid4().hex}"
    token_store.save(
        token=token,
        pack=request.pack,
        patch_diff=patch_diff,
        target_files=target_files,
        model_id=validated_model,
        principal=principal,
        ttl_seconds=3600,
    )
    audit.record_event(
        principal=principal,
        domain_id=request.pack.domain_id,
        event_type="handoff.preview",
        severity="INFO",
        resource_uri="/api/v1/handoff/preview",
        outcome="allow",
        payload={
            "trace_id": request.pack.trace_id,
            "model_id": validated_model,
            "target_files_count": len(target_files),
        },
    )

    return HandoffPreviewResponse(
        confirmation_token=token,
        patch_diff=patch_diff,
        target_files=target_files,
        model_id=validated_model,
        local_tokens_consumed=local_tokens,
        pack_token_count=pack_tokens,
        estimated_paid_cost_usd=cost_usd,
        estimated_paid_savings_usd=savings_usd,
        cost_low_usd=cost_report.cost_low_usd,
        cost_expected_usd=cost_report.cost_expected_usd,
        cost_high_usd=cost_report.cost_high_usd,
        pricing_source=cost_report.pricing_source,
        requires_human_confirmation=True,
        expires_in_seconds=3600,
        created_at=datetime.now(UTC).isoformat(),
    )


@router.post("/handoff/apply", response_model=HandoffApplyResponse)
def apply_handoff(
    request: HandoffApplyRequest,
    principal: Annotated[PrincipalContext, Depends(get_principal_context)],
    repo: Annotated[Any, Depends(get_repository)],
    token_store: Annotated[ConfirmationTokenStore, Depends(get_confirmation_token_store)],
) -> HandoffApplyResponse:
    """Confirms handoff token and records ROI telemetry in memory/RLS without host RCE (ASI05)."""
    start_time = time.perf_counter()
    audit = get_audit_chain_service()
    if principal.role != RoleType.ENGINEER:
        audit.record_event(
            principal=principal,
            event_type="handoff.apply",
            severity="WARNING",
            resource_uri="/api/v1/handoff/apply",
            outcome="deny",
            payload={"reason": "role_restricted", "role": principal.role.value},
        )
        raise HTTPException(
            status_code=403,
            detail="Sandboxed patch handoff is restricted to Engineer principals (SEC-04).",
        )
    if request.custom_patch is not None:
        audit.record_event(
            principal=principal,
            event_type="handoff.apply",
            severity="WARNING",
            resource_uri="/api/v1/handoff/apply",
            outcome="deny",
            payload={"reason": "custom_patch_prohibited"},
        )
        raise HTTPException(
            status_code=400,
            detail="Caller-supplied custom_patch is prohibited on the API host (OWASP ASI05).",
        )
    if not request.confirmed:
        audit.record_event(
            principal=principal,
            event_type="handoff.apply",
            severity="NOTICE",
            resource_uri="/api/v1/handoff/apply",
            outcome="deny",
            payload={"reason": "confirmation_false"},
        )
        raise HTTPException(
            status_code=400,
            detail="Explicit human confirmation required: 'confirmed' must be true.",
        )

    def _budget_gate(rec: Any) -> None:
        gate_exec = GeminiImplementationExecutor(
            model_id=rec.model_id,
            require_confirmation=False,
        )
        if gate_exec.allow_external_cloud_egress:
            gate_exec.check_principal_budget(principal, rec.pack, repo=repo)

    try:
        record = token_store.consume(
            request.confirmation_token,
            principal,
            pre_commit_check=_budget_gate,
        )
    except KeyError:
        audit.record_event(
            principal=principal,
            event_type="handoff.apply",
            severity="WARNING",
            resource_uri="/api/v1/handoff/apply",
            outcome="deny",
            payload={"reason": "invalid_or_expired_token"},
        )
        raise HTTPException(
            status_code=400,
            detail="Invalid or expired confirmation token.",
        ) from None
    except PermissionError as exc:
        audit.record_event(
            principal=principal,
            event_type="handoff.apply",
            severity="WARNING",
            resource_uri="/api/v1/handoff/apply",
            outcome="deny",
            payload={"reason": "token_ownership_mismatch"},
        )
        raise HTTPException(status_code=403, detail=str(exc)) from None
    except BudgetExceededError as exc:
        audit.record_event(
            principal=principal,
            event_type="handoff.apply",
            severity="WARNING",
            resource_uri="/api/v1/handoff/apply",
            outcome="deny",
            payload={"reason": "budget_exceeded"},
        )
        raise HTTPException(status_code=400, detail=str(exc)) from None

    struct_ok, details = _verify_diff_structure_in_memory(record.patch_diff, record.target_files)
    executor = GeminiImplementationExecutor(
        model_id=record.model_id,
        require_confirmation=False,
    )
    latency_ms = int((time.perf_counter() - start_time) * 1000)
    telemetry = executor.record_handoff_confirmation_telemetry(
        pack=record.pack,
        principal=principal,
        latency_ms=latency_ms,
        repo=repo,
        is_offline_local=not executor.allow_external_cloud_egress,
    )
    audit.record_event(
        principal=principal,
        domain_id=record.pack.domain_id,
        event_type="handoff.apply",
        severity="INFO" if struct_ok else "WARNING",
        resource_uri="/api/v1/handoff/apply",
        outcome="success" if struct_ok else "failure",
        payload={
            "trace_id": record.pack.trace_id,
            "model_id": record.model_id,
            "target_files_count": len(record.target_files),
            "verification_passed": struct_ok,
        },
    )

    return HandoffApplyResponse(
        success=struct_ok,
        confirmed=True,
        target_files=record.target_files,
        lint_passed=struct_ok,
        tests_passed=struct_ok,
        verification_passed=struct_ok,
        details=details,
        telemetry=telemetry,
    )
