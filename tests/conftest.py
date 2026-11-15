"""Pytest root configuration and sys.path resolution."""

from __future__ import annotations

import sys
from collections.abc import Generator
from pathlib import Path

import pytest

from context_scout.api.dependencies import reset_current_principal
from context_scout.security.rbac import _CURRENT_PRINCIPAL

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


@pytest.fixture(autouse=True)
def reset_session_principal_contextvar() -> Generator[None, None, None]:
    """Autouse fixture resetting _CURRENT_PRINCIPAL before and after each test (SEC-04)."""
    _CURRENT_PRINCIPAL.set(None)
    reset_current_principal()
    try:
        yield
    finally:
        _CURRENT_PRINCIPAL.set(None)
        reset_current_principal()


@pytest.fixture(autouse=True)
def configure_test_auth_hmac_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provides a valid test HMAC secret by default, preventing fail-closed crashes."""
    monkeypatch.setenv(
        "SCOUT_AUTH_HMAC_SECRET",
        "scout-test-suite-hmac-secret-32-chars-long-sec04!!",
    )
