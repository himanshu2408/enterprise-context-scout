"""Gemini implementation handoff preview and human confirmation apply endpoints."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
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
    get_repo_root,
    get_repository,
)
from context_scout.api.schemas import (
    HandoffApplyRequest,
    HandoffApplyResponse,
    HandoffPreviewRequest,
    HandoffPreviewResponse,
)
from context_scout.config import validate_google_model_id
from context_scout.schemas.contracts import PrincipalContext

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Handoff"])


@router.post("/handoff/preview", response_model=HandoffPreviewResponse)
def preview_handoff(
    request: HandoffPreviewRequest,
    principal: Annotated[PrincipalContext, Depends(get_principal_context)],
    repo: Annotated[Any, Depends(get_repository)],
    token_store: Annotated[ConfirmationTokenStore, Depends(get_confirmation_token_store)],
) -> HandoffPreviewResponse:
    """Generates patch preview and confirmation token without mutating repository files."""
    repo_root = get_repo_root()

    # 1. Validate model ID (e.g. gemini-3.8-flash or gemini-4-argon)
    try:
        validated_model = validate_google_model_id(request.model_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None

    # 2. Check principal budget
    executor = GeminiImplementationExecutor(
        model_id=validated_model,
        require_confirmation=True,
    )
    try:
        executor.check_principal_budget(principal, request.pack, repo=repo)
    except BudgetExceededError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None

    # 3. Generate patch diff (custom, live Gemini API if key set, or deterministic synthesis)
    if request.custom_patch is not None:
        patch_diff = request.custom_patch
    elif executor.api_key:
        patch_diff = executor._generate_gemini_patch(request.pack, repo_root)
    else:
        patch_diff = executor._generate_deterministic_patch(request.pack, repo_root)

    # 4. Extract target files
    target_files = executor._extract_target_files(patch_diff, request.pack)

    # 5. Calculate token & cost telemetry
    pack_tokens = request.pack.pack_token_count
    local_tokens = request.pack.local_tokens_consumed
    cost_usd = round((pack_tokens / 1_000_000.0) * 0.75, 6)
    savings_usd = round(max(0.0, (local_tokens / 1_000_000.0) * 4.00 - cost_usd), 4)

    # 6. Save preview record in token store
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

    return HandoffPreviewResponse(
        confirmation_token=token,
        patch_diff=patch_diff,
        target_files=target_files,
        model_id=validated_model,
        local_tokens_consumed=local_tokens,
        pack_token_count=pack_tokens,
        estimated_paid_cost_usd=cost_usd,
        estimated_paid_savings_usd=savings_usd,
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
    """Applies and verifies patch in worktree sandbox after explicit human confirmation."""
    repo_root = get_repo_root()

    # 1. Enforce explicit human confirmation
    if not request.confirmed:
        raise HTTPException(
            status_code=400,
            detail="Explicit human confirmation required: 'confirmed' must be true.",
        )

    # 2. Validate confirmation token
    record = token_store.get(request.confirmation_token)
    if record is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid or expired confirmation token.",
        )

    # 3. Apply patch in GitWorktreeSandbox via GeminiImplementationExecutor
    diff_to_apply = request.custom_patch or record.patch_diff
    executor = GeminiImplementationExecutor(
        model_id=record.model_id,
        require_confirmation=False,
    )

    try:
        valid_targets = [
            t for t in record.target_files if (repo_root / t).exists() and t.endswith(".py")
        ]
        lint_target_cmd = (
            ["ruff", "check", *valid_targets]
            if valid_targets
            else ["ruff", "check", "src/context_scout/schemas/contracts.py"]
        )
        test_target_cmd = ["pytest", "-q", "tests/test_deterministic_slicer.py"]
        result, telemetry = executor.execute_implementation(
            pack=record.pack,
            principal=principal,
            repo_root=repo_root,
            repo=repo,
            custom_patch=diff_to_apply,
            auto_confirm=True,
            lint_cmd=lint_target_cmd,
            test_cmd=test_target_cmd,
        )
    except Exception as exc:
        logger.error("Handoff execution failed in worktree sandbox: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Worktree sandbox execution failed: {exc}",
        ) from exc

    # 4. Invalidate token after consumption
    token_store.invalidate(request.confirmation_token)

    return HandoffApplyResponse(
        success=result.success,
        confirmed=result.confirmed,
        target_files=result.target_files,
        lint_passed=result.lint_passed,
        tests_passed=result.tests_passed,
        verification_passed=result.verification_passed,
        details=result.details,
        telemetry=telemetry,
    )
