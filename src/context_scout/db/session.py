"""PostgreSQL 17 transaction-scoped RLS session manager (SEC-01, OPS-01, PERF-02)."""

import atexit
import threading
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path

import psycopg
import psycopg_pool
from psycopg import Connection

from context_scout.schemas.contracts import PrincipalContext

_POOL_LOCK = threading.Lock()
_SHARED_POOLS: dict[str, psycopg_pool.ConnectionPool[Connection[tuple[object, ...]]]] = {}


def _close_all_shared_pools() -> None:
    with _POOL_LOCK:
        for pool in _SHARED_POOLS.values():
            with suppress(Exception):
                pool.close()
        _SHARED_POOLS.clear()


atexit.register(_close_all_shared_pools)


def apply_schema_ddl(admin_db_url: str, sql_path: Path) -> None:
    """Applies sql/001_init_schema.sql using the administrative connection."""
    ddl_sql = sql_path.read_text(encoding="utf-8")
    with psycopg.connect(admin_db_url, autocommit=True) as conn:
        conn.execute(ddl_sql)


class RLSConnectionManager:
    """Manages pooled PostgreSQL connections bound to PrincipalContext inside atomic transactions.

    Guarantees:
    1. Connects via `scout_app_user` role using shared `psycopg_pool.ConnectionPool` (SEC-01).
    2. Wraps every operation in an explicit `with conn.transaction():` block (OPS-01).
    3. Sets `app.principal_scopes` transaction-locally via `set_config(..., is_local=true)`.
    4. Enables `hnsw.iterative_scan = 'relaxed_order'` to preserve HNSW recall under RLS.
    """

    def __init__(self, app_db_url: str, use_pool: bool = True, max_size: int = 10) -> None:
        self._app_db_url = app_db_url
        self._use_pool = use_pool
        self._max_size = max_size

    def _get_pool(self) -> psycopg_pool.ConnectionPool[Connection[tuple[object, ...]]]:
        pool = _SHARED_POOLS.get(self._app_db_url)
        if pool is None or pool.closed:
            with _POOL_LOCK:
                pool = _SHARED_POOLS.get(self._app_db_url)
                if pool is None or pool.closed:
                    pool = psycopg_pool.ConnectionPool(
                        self._app_db_url,
                        min_size=1,
                        max_size=self._max_size,
                        open=True,
                    )
                    _SHARED_POOLS[self._app_db_url] = pool
        return pool

    @contextmanager
    def scoped_transaction(
        self, principal: PrincipalContext, enable_hnsw_iterative_scan: bool = True
    ) -> Iterator[Connection[tuple[object, ...]]]:
        scope_csv = principal.to_postgres_scope_csv()
        if self._use_pool:
            pool = self._get_pool()
            with pool.connection() as conn, conn.transaction():
                conn.execute(
                    "SELECT set_config('app.principal_scopes', %s, true)",
                    (scope_csv,),
                )
                if enable_hnsw_iterative_scan:
                    conn.execute("SET LOCAL hnsw.iterative_scan = 'relaxed_order'")
                yield conn
        else:
            with psycopg.connect(self._app_db_url) as conn, conn.transaction():
                conn.execute(
                    "SELECT set_config('app.principal_scopes', %s, true)",
                    (scope_csv,),
                )
                if enable_hnsw_iterative_scan:
                    conn.execute("SET LOCAL hnsw.iterative_scan = 'relaxed_order'")
                yield conn
