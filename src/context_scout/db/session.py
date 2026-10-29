"""PostgreSQL 17 transaction-scoped RLS session manager (SEC-01, OPS-01, PERF-02)."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import psycopg
from psycopg import Connection

from context_scout.schemas.contracts import PrincipalContext


def apply_schema_ddl(admin_db_url: str, sql_path: Path) -> None:
    """Applies sql/001_init_schema.sql using the administrative connection."""
    ddl_sql = sql_path.read_text(encoding="utf-8")
    with psycopg.connect(admin_db_url, autocommit=True) as conn:
        conn.execute(ddl_sql)


class RLSConnectionManager:
    """Manages PostgreSQL connections bound to PrincipalContext inside atomic transactions.

    Guarantees:
    1. Connects via unprivileged `scout_app_user` role (SEC-01).
    2. Wraps every operation in an explicit `with conn.transaction():` block (OPS-01).
    3. Sets `app.principal_scopes` transaction-locally via `set_config(..., is_local=true)`.
    4. Enables `hnsw.iterative_scan = 'relaxed_order'` to preserve HNSW recall under RLS.
    """

    def __init__(self, app_db_url: str) -> None:
        self._app_db_url = app_db_url

    @contextmanager
    def scoped_transaction(
        self, principal: PrincipalContext, enable_hnsw_iterative_scan: bool = True
    ) -> Iterator[Connection[tuple[object, ...]]]:
        scope_csv = principal.to_postgres_scope_csv()
        with psycopg.connect(self._app_db_url) as conn, conn.transaction():
            conn.execute(
                "SELECT set_config('app.principal_scopes', %s, true)",
                (scope_csv,),
            )
            if enable_hnsw_iterative_scan:
                conn.execute("SET LOCAL hnsw.iterative_scan = 'relaxed_order'")
            yield conn
