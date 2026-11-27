"""Programmatic Alembic migration runner and squawk zero-downtime DDL verifier."""

from __future__ import annotations

import argparse
import io
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from contextlib import redirect_stdout
from pathlib import Path
from typing import Final
from urllib.parse import quote, unquote, urlparse

import psycopg
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from psycopg import sql as pg_sql

BASELINE_REVISION: Final[str] = "0001_legacy_baseline"
CONCURRENT_INDEX_REVISION: Final[str] = "0004_m1_conc_idx"
CONSTRAINT_SWAP_REVISION: Final[str] = "0005_m1_swap_keys"
CONCURRENT_INDEX_REVISIONS: Final[frozenset[str]] = frozenset(
    {
        CONCURRENT_INDEX_REVISION,
        "0008_m2_conc_idx",
        "0012_m2_se_conc_idx",
        "0016_m3_sess_conc_idx",
        "0019_m4_art_conc_idx",
        "0022_m5_audit_conc_idx",
        "0024_m5_scim_idx_conn_rls",
    }
)
CONSTRAINT_SWAP_REVISIONS: Final[frozenset[str]] = frozenset(
    {CONSTRAINT_SWAP_REVISION, "0009_m2_swap_checks", "0013_m2_se_swap_chk"}
)

_MIGRATIONS_DIR: Final[Path] = Path(__file__).resolve().parent / "migrations"
_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]


def _normalize_alembic_url(raw_url: str) -> str:
    cleaned = raw_url.strip()
    if cleaned.startswith("postgresql://"):
        return "postgresql+psycopg://" + cleaned[len("postgresql://") :]
    if cleaned.startswith("postgres://"):
        return "postgresql+psycopg://" + cleaned[len("postgres://") :]
    return cleaned


def _build_alembic_config(
    db_url: str | None = None,
    *,
    stdout_buffer: io.StringIO | None = None,
    transactional_ddl: bool = True,
) -> Config:
    ini_path = _REPO_ROOT / "alembic.ini"
    cfg = Config(
        str(ini_path) if ini_path.is_file() else None,
        stdout=stdout_buffer if stdout_buffer is not None else sys.stdout,
    )
    cfg.set_main_option("script_location", str(_MIGRATIONS_DIR))
    cfg.attributes["skip_logging_config"] = True
    cfg.attributes["transactional_ddl"] = transactional_ddl
    if db_url:
        normalized = _normalize_alembic_url(db_url)
        cfg.set_main_option("sqlalchemy.url", normalized)
        cfg.attributes["db_url"] = normalized
    return cfg


def _resolve_app_role_password(
    explicit_password: str | None,
    *,
    app_db_url: str | None = None,
) -> str | None:
    if explicit_password is not None and explicit_password.strip():
        return explicit_password
    env_pw = os.environ.get("SCOUT_DB_APP_PASSWORD")
    if env_pw and env_pw.strip():
        return env_pw
    for candidate_url in (os.environ.get("SCOUT_DB_APP_URL"), app_db_url):
        if candidate_url and candidate_url.strip():
            parsed = urlparse(candidate_url.strip())
            if parsed.password:
                return unquote(parsed.password)
    return None


def build_admin_url_from_postgres_env(environ: Mapping[str, str] | None = None) -> str:
    """Constructs an admin PostgreSQL DSN from standard POSTGRES_* environment variables.

    Fails closed when ``POSTGRES_ADMIN_PASSWORD`` is missing or blank so one-shot
    migration containers never silently attempt passwordless connections.
    """
    env = os.environ if environ is None else environ
    raw_pw = str(env.get("POSTGRES_ADMIN_PASSWORD") or "")
    if not raw_pw.strip():
        raise RuntimeError(
            "POSTGRES_ADMIN_PASSWORD must be set and non-empty when using --from-postgres-env."
        )
    host = str(env.get("POSTGRES_HOST") or "127.0.0.1").strip() or "127.0.0.1"
    raw_port = str(env.get("POSTGRES_PORT") or "5432").strip() or "5432"
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise RuntimeError(f"Invalid POSTGRES_PORT value: {raw_port!r}") from exc
    if not (1 <= port <= 65535):
        raise RuntimeError(f"POSTGRES_PORT out of range (1..65535): {port}")
    raw_user = str(env.get("POSTGRES_USER") or "")
    user = raw_user if raw_user.strip() else "postgres"
    raw_db = str(env.get("POSTGRES_DB") or "")
    db_name = raw_db if raw_db.strip() else "context_scout"
    return (
        f"postgresql://{quote(user, safe='')}:{quote(raw_pw, safe='')}"
        f"@{host}:{port}/{quote(db_name, safe='')}"
    )


def _reconcile_m1_halfvec_spaces(admin_db_url: str) -> None:
    """Idempotently upgrades pre-M1-03 databases that lack embedding_spaces or halfvec columns."""
    with psycopg.connect(admin_db_url, connect_timeout=10) as conn:
        conn.execute(
            """
            CREATE OR REPLACE FUNCTION sync_default_embedding_halfvec() RETURNS trigger AS $$
            BEGIN
                IF NEW.embedding_space_id IS NULL OR btrim(NEW.embedding_space_id) = '' THEN
                    NEW.embedding_space_id := 'embeddinggemma-2-768';
                END IF;
                IF NEW.embedding IS NOT NULL AND (
                    NEW.embedding_halfvec IS NULL
                    OR (
                        TG_OP = 'UPDATE'
                        AND NEW.embedding IS DISTINCT FROM OLD.embedding
                        AND NEW.embedding_halfvec IS NOT DISTINCT FROM OLD.embedding_halfvec
                    )
                ) THEN
                    NEW.embedding_halfvec := NEW.embedding::halfvec;
                END IF;
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
            """
        )
        has_spaces = (
            conn.execute("SELECT to_regclass('public.embedding_spaces')").fetchone() or (None,)
        )[0]
        if has_spaces is not None:
            conn.commit()
            return
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embedding_spaces (
                tenant_id UUID NOT NULL DEFAULT COALESCE(
                    NULLIF(current_setting('app.tenant_id', true), '')::uuid,
                    '00000000-0000-0000-0000-000000000001'::uuid
                ),
                domain_id TEXT NOT NULL,
                space_id TEXT NOT NULL,
                model_id TEXT NOT NULL,
                provider TEXT NOT NULL DEFAULT 'sentence-transformers',
                dimensions BIGINT NOT NULL CHECK (dimensions BETWEEN 128 AND 4096),
                quantization TEXT NOT NULL DEFAULT 'halfvec' CHECK (
                    quantization IN ('halfvec', 'vector')
                ),
                status TEXT NOT NULL DEFAULT 'shadow' CHECK (
                    status IN ('active', 'shadow', 'retired')
                ),
                is_active BOOLEAN NOT NULL DEFAULT FALSE,
                index_operator_class TEXT NOT NULL DEFAULT 'halfvec_cosine_ops',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                activated_at TIMESTAMPTZ,
                PRIMARY KEY (tenant_id, domain_id, space_id)
            );
            """
        )
        conn.execute(
            "ALTER TABLE embedding_spaces DROP CONSTRAINT IF EXISTS "
            "fk_embedding_spaces_tenant_domain;"
        )
        conn.execute(
            "ALTER TABLE embedding_spaces ADD CONSTRAINT fk_embedding_spaces_tenant_domain "
            "FOREIGN KEY (tenant_id, domain_id) REFERENCES domain_packs (tenant_id, domain_id) "
            "ON DELETE CASCADE NOT VALID;"
        )
        conn.execute(
            "ALTER TABLE embedding_spaces VALIDATE CONSTRAINT fk_embedding_spaces_tenant_domain;"
        )
        conn.execute("ALTER TABLE embedding_spaces ENABLE ROW LEVEL SECURITY;")
        conn.execute("ALTER TABLE embedding_spaces FORCE ROW LEVEL SECURITY;")
        conn.execute("DROP POLICY IF EXISTS rls_embedding_spaces_tenant ON embedding_spaces;")
        conn.execute(
            "CREATE POLICY rls_embedding_spaces_tenant ON embedding_spaces "
            "USING (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid) "
            "WITH CHECK (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid);"
        )
        conn.execute("GRANT SELECT, INSERT, UPDATE, DELETE ON embedding_spaces TO scout_app_user;")
        for vec_table in ("ast_symbols", "doc_chunks"):
            conn.execute(
                f"ALTER TABLE {vec_table} ADD COLUMN IF NOT EXISTS embedding_space_id TEXT "
                "DEFAULT 'embeddinggemma-2-768';"
            )
            conn.execute(
                f"ALTER TABLE {vec_table} ADD COLUMN IF NOT EXISTS embedding_halfvec halfvec;"
            )
            conn.execute(
                f"ALTER TABLE {vec_table} ADD COLUMN IF NOT EXISTS shadow_embedding_space_id TEXT;"
            )
            conn.execute(
                f"ALTER TABLE {vec_table} ADD COLUMN IF NOT EXISTS "
                "shadow_embedding_halfvec halfvec;"
            )
            conn.execute(
                f"ALTER TABLE {vec_table} DROP CONSTRAINT IF EXISTS chk_{vec_table}_nonzero_vec;"
            )
            conn.execute(
                f"ALTER TABLE {vec_table} ADD CONSTRAINT chk_{vec_table}_nonzero_vec CHECK ("
                "vector_norm(embedding) > 0.0 "
                "AND (embedding_halfvec IS NULL OR l2_norm(embedding_halfvec) > 0.0) "
                "AND (shadow_embedding_halfvec IS NULL OR l2_norm(shadow_embedding_halfvec) > 0.0)"
                ") NOT VALID;"
            )
            conn.execute(
                f"ALTER TABLE {vec_table} VALIDATE CONSTRAINT chk_{vec_table}_nonzero_vec;"
            )
        conn.execute(
            """
            CREATE OR REPLACE FUNCTION sync_default_embedding_halfvec() RETURNS trigger AS $$
            BEGIN
                IF NEW.embedding_space_id IS NULL OR btrim(NEW.embedding_space_id) = '' THEN
                    NEW.embedding_space_id := 'embeddinggemma-2-768';
                END IF;
                IF NEW.embedding_halfvec IS NULL AND NEW.embedding IS NOT NULL THEN
                    NEW.embedding_halfvec := NEW.embedding::halfvec;
                END IF;
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
            """
        )
        for vec_table in ("ast_symbols", "doc_chunks"):
            conn.execute(f"DROP TRIGGER IF EXISTS trg_{vec_table}_sync_halfvec ON {vec_table};")
            conn.execute(
                f"CREATE TRIGGER trg_{vec_table}_sync_halfvec "
                f"BEFORE INSERT OR UPDATE OF embedding, embedding_halfvec, embedding_space_id "
                f"ON {vec_table} FOR EACH ROW EXECUTE FUNCTION sync_default_embedding_halfvec();"
            )
        conn.execute(
            """
            INSERT INTO embedding_spaces (
                tenant_id, domain_id, space_id, model_id, provider,
                dimensions, quantization, status, is_active, activated_at
            )
            SELECT
                tenant_id, domain_id, 'embeddinggemma-2-768', 'google/embeddinggemma-2',
                'sentence-transformers', 768, 'halfvec', 'active', TRUE, NOW()
            FROM domain_packs
            ON CONFLICT (tenant_id, domain_id, space_id) DO NOTHING;
            """
        )
        conn.commit()

    with psycopg.connect(admin_db_url, autocommit=True, connect_timeout=10) as conn:
        for stmt in (
            "CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS uq_idx_embedding_spaces_one_active "
            "ON embedding_spaces (tenant_id, domain_id) WHERE is_active = TRUE;",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ast_symbols_halfvec_768 "
            "ON ast_symbols USING hnsw ((embedding_halfvec::halfvec(768)) halfvec_cosine_ops) "
            "WHERE vector_dims(embedding_halfvec) = 768;",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_doc_chunks_halfvec_768 "
            "ON doc_chunks USING hnsw ((embedding_halfvec::halfvec(768)) halfvec_cosine_ops) "
            "WHERE vector_dims(embedding_halfvec) = 768;",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ast_symbols_halfvec_1024 "
            "ON ast_symbols USING hnsw ((embedding_halfvec::halfvec(1024)) halfvec_cosine_ops) "
            "WHERE vector_dims(embedding_halfvec) = 1024;",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_doc_chunks_halfvec_1024 "
            "ON doc_chunks USING hnsw ((embedding_halfvec::halfvec(1024)) halfvec_cosine_ops) "
            "WHERE vector_dims(embedding_halfvec) = 1024;",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ast_symbols_shadow_halfvec_768 "
            "ON ast_symbols USING hnsw "
            "((shadow_embedding_halfvec::halfvec(768)) halfvec_cosine_ops) "
            "WHERE vector_dims(shadow_embedding_halfvec) = 768;",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_doc_chunks_shadow_halfvec_768 "
            "ON doc_chunks USING hnsw "
            "((shadow_embedding_halfvec::halfvec(768)) halfvec_cosine_ops) "
            "WHERE vector_dims(shadow_embedding_halfvec) = 768;",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ast_symbols_shadow_halfvec_1024 "
            "ON ast_symbols USING hnsw "
            "((shadow_embedding_halfvec::halfvec(1024)) halfvec_cosine_ops) "
            "WHERE vector_dims(shadow_embedding_halfvec) = 1024;",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_doc_chunks_shadow_halfvec_1024 "
            "ON doc_chunks USING hnsw "
            "((shadow_embedding_halfvec::halfvec(1024)) halfvec_cosine_ops) "
            "WHERE vector_dims(shadow_embedding_halfvec) = 1024;",
        ):
            conn.execute(stmt)


def run_alembic_upgrade(
    admin_db_url: str,
    target_revision: str = "head",
    app_role_password: str | None = None,
    *,
    app_db_url: str | None = None,
) -> None:
    """Runs Alembic upgrades safely, auto-stamping pre-Alembic 001+002 schemas first."""
    with psycopg.connect(admin_db_url, connect_timeout=10) as conn:
        has_domain_packs = (
            conn.execute("SELECT to_regclass('public.domain_packs')").fetchone() or (None,)
        )[0]
        has_alembic = (
            conn.execute("SELECT to_regclass('public.alembic_version')").fetchone() or (None,)
        )[0]
        has_repositories = (
            conn.execute("SELECT to_regclass('public.repositories')").fetchone() or (None,)
        )[0]
        has_tenant_col = (
            conn.execute(
                """
                SELECT 1
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = 'domain_packs'
                  AND column_name = 'tenant_id'
                """
            ).fetchone()
            is not None
        )

        if has_domain_packs is not None and has_repositories is None:
            sql_002 = (_REPO_ROOT / "sql" / "002_production_enterprise_schema.sql").read_text(
                encoding="utf-8"
            )
            conn.execute(sql_002)
            conn.commit()

        should_stamp_baseline = (
            has_domain_packs is not None and has_alembic is None and not has_tenant_col
        )

    cfg = _build_alembic_config(admin_db_url)
    if should_stamp_baseline:
        command.stamp(cfg, BASELINE_REVISION)
    command.upgrade(cfg, target_revision)
    if target_revision == "head":
        _reconcile_m1_halfvec_spaces(admin_db_url)

    resolved_pw = _resolve_app_role_password(app_role_password, app_db_url=app_db_url)
    with psycopg.connect(admin_db_url, autocommit=True) as conn:
        if resolved_pw:
            conn.execute(
                pg_sql.SQL(
                    "ALTER ROLE scout_app_user WITH LOGIN PASSWORD {} "
                    "NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS NOINHERIT;"
                ).format(pg_sql.Literal(resolved_pw))
            )
        else:
            conn.execute(
                "ALTER ROLE scout_app_user WITH LOGIN "
                "NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS NOINHERIT;"
            )


def _strip_alembic_version_dml(sql_text: str) -> str:
    cleaned_lines: list[str] = []
    for line in sql_text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("INSERT INTO alembic_version", "UPDATE alembic_version")):
            continue
        if stripped.startswith("CREATE TABLE alembic_version"):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines) + "\n"


def export_m1_upgrade_sql_files(out_dir: Path) -> list[tuple[str, Path]]:
    """Exports ordered post-baseline Alembic upgrade SQL scripts into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    base_cfg = _build_alembic_config()
    script_dir = ScriptDirectory.from_config(base_cfg)

    ordered = list(reversed(list(script_dir.walk_revisions(base="base", head="head"))))
    exported: list[tuple[str, Path]] = []
    for rev in ordered:
        if rev.revision == BASELINE_REVISION:
            continue
        prev_rev = rev.down_revision
        rev_range = f"{prev_rev}:{rev.revision}" if isinstance(prev_rev, str) else rev.revision
        is_concurrent = rev.revision in CONCURRENT_INDEX_REVISIONS

        buf = io.StringIO()
        rev_cfg = _build_alembic_config(
            stdout_buffer=buf,
            transactional_ddl=not is_concurrent,
        )
        with redirect_stdout(buf):
            command.upgrade(rev_cfg, rev_range, sql=True)

        sql_text = _strip_alembic_version_dml(buf.getvalue())
        if is_concurrent:
            # Remove empty transaction markers if Alembic emitted them around autocommit block
            lines = [ln for ln in sql_text.splitlines() if ln.strip() not in {"BEGIN;", "COMMIT;"}]
            sql_text = "\n".join(lines) + "\n"

        out_path = out_dir / f"{rev.revision}.sql"
        out_path.write_text(sql_text, encoding="utf-8")
        exported.append((rev.revision, out_path))
    return exported


def _resolve_squawk_binary() -> str:
    sibling = Path(sys.executable).resolve().parent / "squawk"
    if sibling.is_file():
        return str(sibling)
    found = shutil.which("squawk")
    if found:
        return found
    raise FileNotFoundError("squawk CLI binary not found in virtual environment or PATH.")


def verify_migrations_with_squawk() -> list[str]:
    """Lints all post-baseline Alembic migrations with squawk --pg-version=17.0."""
    versions_dir = _MIGRATIONS_DIR / "versions"
    for py_file in sorted(versions_dir.glob("*.py")):
        source = py_file.read_text(encoding="utf-8")
        if "squawk-ignore" in source:
            raise RuntimeError(
                f"Inline squawk-ignore directive is prohibited in migration file: {py_file}"
            )

    squawk_bin = _resolve_squawk_binary()
    checked: list[str] = []
    with tempfile.TemporaryDirectory(prefix="scout_squawk_") as tmp:
        exported = export_m1_upgrade_sql_files(Path(tmp))
        if not exported:
            raise RuntimeError("No post-baseline Alembic migrations found to lint.")
        for rev_id, sql_path in exported:
            sql_text = sql_path.read_text(encoding="utf-8")
            if "squawk-ignore" in sql_text:
                raise RuntimeError(
                    f"Inline squawk-ignore directive found in rendered SQL for {rev_id}"
                )
            cmd = [squawk_bin, "--pg-version=17.0"]
            if rev_id in CONSTRAINT_SWAP_REVISIONS:
                # Swapping legacy single-tenant PK/UK/FKs or widening CHECK constraints
                # requires dropping the legacy constraint cleanly before adding NOT VALID.
                cmd.extend(["--exclude", "ban-drop-constraint"])
            if rev_id in CONCURRENT_INDEX_REVISIONS:
                cmd.append("--no-assume-in-transaction")
            cmd.append(str(sql_path))

            proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"squawk lint failed on revision {rev_id} (exit {proc.returncode}):\n"
                    f"{proc.stdout}\n{proc.stderr}"
                )
            checked.append(rev_id)
    return checked


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Alembic migration runner & squawk verifier")
    parser.add_argument(
        "--lint-squawk",
        action="store_true",
        help="Export M1 upgrade SQL and lint with squawk --pg-version=17.0",
    )
    parser.add_argument(
        "--upgrade",
        action="store_true",
        help="Run Alembic upgrade to target revision against SCOUT_DB_ADMIN_URL",
    )
    parser.add_argument(
        "--from-postgres-env",
        action="store_true",
        help="Build admin DSN from POSTGRES_HOST/PORT/USER/ADMIN_PASSWORD/DB env vars",
    )
    parser.add_argument("--target", default="head", help="Target Alembic revision (default: head)")
    args = parser.parse_args(argv)

    if args.lint_squawk:
        checked = verify_migrations_with_squawk()
        print(f"[PASS] squawk --pg-version=17.0 verified {len(checked)} revisions: {checked}")
        return 0

    if args.upgrade:
        if args.from_postgres_env:
            admin_url = build_admin_url_from_postgres_env()
            app_db_url = None
        else:
            from context_scout.config import ScoutSettings

            settings = ScoutSettings()
            admin_url = settings.db_admin_url
            app_db_url = settings.db_app_url
        run_alembic_upgrade(admin_url, target_revision=args.target, app_db_url=app_db_url)
        print(f"[PASS] Alembic upgrade to '{args.target}' completed.")
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
