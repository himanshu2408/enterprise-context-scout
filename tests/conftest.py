"""Pytest root configuration and sys.path resolution."""

from __future__ import annotations

import gc
import os
import secrets
import sys
from collections.abc import Callable, Generator
from pathlib import Path
from urllib.parse import quote_plus, urlparse, urlunparse

import psycopg
import pytest
from testcontainers.community.postgres import PostgresContainer
from tests.fixtures.pg_isolation import (
    cleanup_ephemeral_pg_domain,
    create_ephemeral_pg_domain,
    ensure_pg17_schema_and_seed_corpora,
    make_ephemeral_pg_domain_id,
)

from context_scout.api.dependencies import reset_current_principal
from context_scout.config import ScoutSettings
from context_scout.security.rbac import _CURRENT_PRINCIPAL
from context_scout.telemetry.cost_governance import SessionBudgetTracker

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _is_pg_reachable(db_url: str) -> bool:
    try:
        with psycopg.connect(db_url, connect_timeout=2) as conn:
            conn.execute("SELECT 1").fetchone()
        return True
    except Exception:
        return False


def _normalize_pg_url(raw_url: str) -> str:
    if raw_url.startswith("postgresql+"):
        return "postgresql://" + raw_url.split("://", 1)[1]
    return raw_url


@pytest.fixture(autouse=True)
def _collect_gc_before_benchmark(request: pytest.FixtureRequest) -> None:
    """Collect cyclic garbage before latency/benchmark tests so Gen-2 pauses don't skew SLAs."""
    name = request.node.name.lower()
    if (
        request.node.get_closest_marker("benchmark") is not None
        or "latency" in name
        or "performance" in name
        or "redos" in name
        or "asi01" in name
        or "polyglot" in name
        or "merkle" in name
    ):
        gc.collect()


def _session_requires_pg17(session: pytest.Session) -> bool:
    pg_fixtures = {
        "ephemeral_pg_domain",
        "ephemeral_pg_domain_factory",
        "s22_pg_domain",
        "pg17_canary",
    }
    for item in session.items:
        if item.get_closest_marker("postgres") is not None:
            return True
        fixture_names = getattr(item, "fixturenames", ())
        if any(name in pg_fixtures for name in fixture_names):
            return True
    return False


@pytest.fixture(scope="session", autouse=True)
def bootstrap_pg17_session(request: pytest.FixtureRequest) -> Generator[None, None, None]:
    """Ensure PG17 schema + seeded corpora only when PostgreSQL tests are collected."""
    if not _session_requires_pg17(request.session) or _PROJECT_ROOT.name.startswith("scout_wt_"):
        yield
        return

    container: PostgresContainer | None = None
    prev_admin = os.environ.get("SCOUT_DB_ADMIN_URL")
    prev_app = os.environ.get("SCOUT_DB_APP_URL")
    try:
        settings = ScoutSettings()
        db_admin_url = settings.db_admin_url
        db_app_url = settings.db_app_url

        if not _is_pg_reachable(db_admin_url):
            container = PostgresContainer("pgvector/pgvector:pg17")
            container.start()
            db_admin_url = _normalize_pg_url(container.get_connection_url())
            app_password = os.environ.get("SCOUT_TEST_APP_PASSWORD") or secrets.token_urlsafe(16)
            with psycopg.connect(db_admin_url) as conn:
                conn.execute(
                    f"""
                    DO $$
                    BEGIN
                        IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'scout_app_user') THEN
                            CREATE ROLE scout_app_user LOGIN PASSWORD '{app_password}'
                                NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS NOINHERIT;
                        ELSE
                            ALTER ROLE scout_app_user WITH PASSWORD '{app_password}'
                                NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS NOINHERIT;
                        END IF;
                    END
                    $$;
                    """
                )
                conn.commit()
            parsed = urlparse(db_admin_url)
            host_port = f"{parsed.hostname or '127.0.0.1'}:{parsed.port or 5432}"
            db_app_url = urlunparse(
                (
                    "postgresql",
                    f"scout_app_user:{quote_plus(app_password)}@{host_port}",
                    parsed.path,
                    "",
                    "",
                    "",
                )
            )

        os.environ["SCOUT_DB_ADMIN_URL"] = db_admin_url
        os.environ["SCOUT_DB_APP_URL"] = db_app_url
        ensure_pg17_schema_and_seed_corpora(
            db_admin_url=db_admin_url,
            db_app_url=db_app_url,
            repo_root=_PROJECT_ROOT,
        )
        yield
    finally:
        if prev_admin is None:
            os.environ.pop("SCOUT_DB_ADMIN_URL", None)
        else:
            os.environ["SCOUT_DB_ADMIN_URL"] = prev_admin
        if prev_app is None:
            os.environ.pop("SCOUT_DB_APP_URL", None)
        else:
            os.environ["SCOUT_DB_APP_URL"] = prev_app
        if container is not None:
            container.stop()


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
