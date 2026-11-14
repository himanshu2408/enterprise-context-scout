"""Pytest root configuration and sys.path resolution."""

from __future__ import annotations

import sys
from collections.abc import Callable, Generator
from pathlib import Path

import psycopg
import pytest
from tests.fixtures.pg_isolation import (
    cleanup_ephemeral_pg_domain,
    create_ephemeral_pg_domain,
    make_ephemeral_pg_domain_id,
)

from context_scout.api.dependencies import reset_current_principal
from context_scout.config import ScoutSettings
from context_scout.security.rbac import _CURRENT_PRINCIPAL
from context_scout.telemetry.cost_governance import SessionBudgetTracker

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
def reset_session_budget_tracker_singleton() -> Generator[None, None, None]:
    """Autouse fixture resetting SessionBudgetTracker singleton before and after each test."""
    SessionBudgetTracker.get_instance().reset_session()
    try:
        yield
    finally:
        SessionBudgetTracker.get_instance().reset_session()


@pytest.fixture(autouse=True)
def configure_test_auth_hmac_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provides a valid test HMAC secret by default, preventing fail-closed crashes."""
    monkeypatch.setenv(
        "SCOUT_AUTH_HMAC_SECRET",
        "scout-test-suite-hmac-secret-32-chars-long-sec04!!",
    )


@pytest.fixture
def ephemeral_pg_domain(request: pytest.FixtureRequest) -> Generator[str, None, None]:
    """Yield a unique ``eval_*`` domain_id registered in PostgreSQL with guaranteed teardown."""
    settings = ScoutSettings()
    domain_id = make_ephemeral_pg_domain_id(request.node.name)
    with psycopg.connect(settings.db_admin_url) as conn:
        create_ephemeral_pg_domain(conn, domain_id=domain_id)
        try:
            yield domain_id
        finally:
            conn.rollback()
            cleanup_ephemeral_pg_domain(conn, domain_id)


@pytest.fixture
def ephemeral_pg_domain_factory(
    request: pytest.FixtureRequest,
) -> Generator[Callable[[str], str], None, None]:
    """Yield a factory creating multiple unique ``eval_*`` PostgreSQL domains with teardown."""
    settings = ScoutSettings()
    created_domains: list[str] = []
    with psycopg.connect(settings.db_admin_url) as conn:

        def _create(suffix: str = "tenant") -> str:
            domain_id = make_ephemeral_pg_domain_id(f"{request.node.name}_{suffix}")
            create_ephemeral_pg_domain(conn, domain_id=domain_id)
            created_domains.append(domain_id)
            return domain_id

        try:
            yield _create
        finally:
            conn.rollback()
            for domain_id in reversed(created_domains):
                cleanup_ephemeral_pg_domain(conn, domain_id)
