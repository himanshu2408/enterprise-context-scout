"""PostgreSQL 17 transaction-scoped FORCE RLS session manager and isolated pools."""

from __future__ import annotations

import atexit
import threading
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

import psycopg
import psycopg_pool
from psycopg import Connection

from context_scout.schemas.contracts import PrincipalContext

if TYPE_CHECKING:
    from context_scout.config import ScoutSettings


class PoolRole(StrEnum):
    """Dedicated database connection pool profile."""

    QUERY = "query"
    INGESTION = "ingestion"


_POOL_LOCK = threading.Lock()
_SHARED_POOLS: dict[
    tuple[str, PoolRole],
    psycopg_pool.ConnectionPool[Connection[tuple[object, ...]]],
] = {}


def _close_all_shared_pools() -> None:
    with _POOL_LOCK:
        for pool in _SHARED_POOLS.values():
            with suppress(Exception):
                pool.close()
        _SHARED_POOLS.clear()


atexit.register(_close_all_shared_pools)


def apply_schema_ddl(admin_db_url: str, sql_path: Path) -> None:
    """Applies a DDL SQL file using an administrative connection."""
    ddl_sql = sql_path.read_text(encoding="utf-8")
    with psycopg.connect(admin_db_url, autocommit=True) as conn:
        conn.execute(ddl_sql)


class RLSConnectionManager:
    """Manages pooled PostgreSQL connections bound to PrincipalContext inside atomic transactions.

    Guarantees:
    1. Connects via `scout_app_user` (`NOBYPASSRLS NOINHERIT`) using isolated query/ingestion pools.
    2. Wraps every operation in an explicit `with conn.transaction():` block (OPS-01).
    3. Sets `app.tenant_id`, `app.principal_scopes`, and timeouts transaction-locally.
       All GUC mutations use `set_config(..., is_local=true)`.
    4. Enables `hnsw.iterative_scan = 'relaxed_order'` on query sessions to preserve HNSW recall.
    """

    def __init__(
        self,
        app_db_url: str,
        use_pool: bool = True,
        max_size: int | None = None,
        *,
        pool_role: PoolRole = PoolRole.QUERY,
        min_size: int = 1,
        statement_timeout_ms: int | None = None,
        lock_timeout_ms: int | None = None,
        idle_tx_timeout_ms: int | None = None,
        connect_timeout_s: int | None = None,
    ) -> None:
        self._app_db_url = app_db_url
        self._use_pool = use_pool
        self._pool_role = pool_role
        self._min_size = max(1, min_size)
        self._connect_timeout_s = connect_timeout_s
        if pool_role == PoolRole.INGESTION:
            self._max_size = max_size if max_size is not None else 8
            self._statement_timeout_ms = (
                statement_timeout_ms if statement_timeout_ms is not None else 300_000
            )
            self._lock_timeout_ms = lock_timeout_ms if lock_timeout_ms is not None else 10_000
            self._idle_tx_timeout_ms = (
                idle_tx_timeout_ms if idle_tx_timeout_ms is not None else 300_000
            )
        else:
            self._max_size = max_size if max_size is not None else 16
            self._statement_timeout_ms = (
                statement_timeout_ms if statement_timeout_ms is not None else 5_000
            )
            self._lock_timeout_ms = lock_timeout_ms if lock_timeout_ms is not None else 2_000
            self._idle_tx_timeout_ms = (
                idle_tx_timeout_ms if idle_tx_timeout_ms is not None else 10_000
            )

    @property
    def pool_role(self) -> PoolRole:
        return self._pool_role

    @property
    def statement_timeout_ms(self) -> int:
        return self._statement_timeout_ms

    @property
    def lock_timeout_ms(self) -> int:
        return self._lock_timeout_ms

    def get_pool(self) -> psycopg_pool.ConnectionPool[Connection[tuple[object, ...]]]:
        pool_key = (self._app_db_url, self._pool_role)
        pool = _SHARED_POOLS.get(pool_key)
        if pool is None or pool.closed:
            with _POOL_LOCK:
                pool = _SHARED_POOLS.get(pool_key)
                if pool is None or pool.closed:
                    pool = psycopg_pool.ConnectionPool(
                        self._app_db_url,
                        min_size=min(self._min_size, self._max_size),
                        max_size=self._max_size,
                        open=True,
                    )
                    _SHARED_POOLS[pool_key] = pool
        return pool

    _get_pool = get_pool

    def _apply_tx_session_guards(
        self,
        conn: Connection[tuple[object, ...]],
        principal: PrincipalContext,
        enable_hnsw_iterative_scan: bool,
    ) -> None:
        conn.execute(
            """
            SELECT
                set_config('app.tenant_id', %s, true),
                set_config('app.principal_scopes', %s, true),
                set_config('statement_timeout', %s, true),
                set_config('lock_timeout', %s, true),
                set_config('idle_in_transaction_session_timeout', %s, true)
            """,
            (
                str(principal.tenant_id),
                principal.to_postgres_scope_csv(),
                f"{self._statement_timeout_ms}ms",
                f"{self._lock_timeout_ms}ms",
                f"{self._idle_tx_timeout_ms}ms",
            ),
        )
        if enable_hnsw_iterative_scan:
            conn.execute("SET LOCAL hnsw.iterative_scan = 'relaxed_order'")

    @contextmanager
    def scoped_transaction(
        self,
        principal: PrincipalContext,
        enable_hnsw_iterative_scan: bool | None = None,
    ) -> Iterator[Connection[tuple[object, ...]]]:
        use_hnsw = (
            (self._pool_role == PoolRole.QUERY)
            if enable_hnsw_iterative_scan is None
            else enable_hnsw_iterative_scan
        )
        if self._use_pool:
            pool = self.get_pool()
            with pool.connection() as conn, conn.transaction():
                self._apply_tx_session_guards(conn, principal, use_hnsw)
                yield conn
        else:
            if self._connect_timeout_s is not None:
                with (
                    psycopg.connect(
                        self._app_db_url, connect_timeout=self._connect_timeout_s
                    ) as conn,
                    conn.transaction(),
                ):
                    self._apply_tx_session_guards(conn, principal, use_hnsw)
                    yield conn
            else:
                with psycopg.connect(self._app_db_url) as conn, conn.transaction():
                    self._apply_tx_session_guards(conn, principal, use_hnsw)
                    yield conn

    def close(self) -> None:
        """Closes and removes any shared connection pool associated with this manager."""
        if not self._use_pool:
            return
        pool_key = (self._app_db_url, self._pool_role)
        with _POOL_LOCK:
            pool = _SHARED_POOLS.pop(pool_key, None)
            if pool is not None and not pool.closed:
                pool.close()

    scoped_session = scoped_transaction


def get_query_pool(settings: ScoutSettings | None = None) -> RLSConnectionManager:
    """Returns the interactive query-pool RLSConnectionManager (5s statement_timeout)."""
    from context_scout.config import ScoutSettings

    cfg = settings or ScoutSettings()
    return RLSConnectionManager(
        cfg.db_app_url,
        use_pool=True,
        pool_role=PoolRole.QUERY,
        min_size=cfg.query_pool_min_size,
        max_size=cfg.query_pool_max_size,
        statement_timeout_ms=cfg.query_statement_timeout_ms,
        lock_timeout_ms=cfg.query_lock_timeout_ms,
        idle_tx_timeout_ms=cfg.query_idle_tx_timeout_ms,
    )


def get_ingestion_pool(settings: ScoutSettings | None = None) -> RLSConnectionManager:
    """Returns the background ingestion-pool RLSConnectionManager (300s statement_timeout)."""
    from context_scout.config import ScoutSettings

    cfg = settings or ScoutSettings()
    return RLSConnectionManager(
        cfg.db_app_url,
        use_pool=True,
        pool_role=PoolRole.INGESTION,
        min_size=cfg.ingestion_pool_min_size,
        max_size=cfg.ingestion_pool_max_size,
        statement_timeout_ms=cfg.ingestion_statement_timeout_ms,
        lock_timeout_ms=cfg.ingestion_lock_timeout_ms,
        idle_tx_timeout_ms=cfg.ingestion_idle_tx_timeout_ms,
    )
