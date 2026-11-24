"""Serving-plane sidecars and multi-pod runtime LoRA synchronization (M4-01)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from context_scout.serving.adapter_sync_sidecar import (
        DEFAULT_VLLM_ADMIN_ENDPOINT,
        AdapterSyncError,
        AdapterSyncReport,
        AdapterSyncSidecar,
        AdapterVerificationError,
        CosignBlobVerifier,
        CosignVerificationError,
        MultiPodAtomicLoadError,
        sign_blob_for_cosign,
    )

__all__ = [
    "DEFAULT_VLLM_ADMIN_ENDPOINT",
    "AdapterSyncError",
    "AdapterSyncReport",
    "AdapterSyncSidecar",
    "AdapterVerificationError",
    "CosignBlobVerifier",
    "CosignVerificationError",
    "MultiPodAtomicLoadError",
    "sign_blob_for_cosign",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from context_scout.serving import adapter_sync_sidecar

        return getattr(adapter_sync_sidecar, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
