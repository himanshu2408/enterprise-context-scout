"""Alembic environment supporting online PG17 migrations and offline --sql export."""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

config = context.config
if config.config_file_name is not None and not config.attributes.get("skip_logging_config"):
    fileConfig(config.config_file_name, disable_existing_loggers=False)


def _normalize_sqlalchemy_url(raw_url: str) -> str:
    cleaned = raw_url.strip()
    if cleaned.startswith("postgresql://"):
        return "postgresql+psycopg://" + cleaned[len("postgresql://") :]
    if cleaned.startswith("postgres://"):
        return "postgresql+psycopg://" + cleaned[len("postgres://") :]
    return cleaned


def _resolve_database_url() -> str:
    override = config.attributes.get("db_url")
    if isinstance(override, str) and override.strip():
        return _normalize_sqlalchemy_url(override)
    env_url = os.environ.get("SCOUT_DB_ADMIN_URL")
    if env_url and env_url.strip():
        return _normalize_sqlalchemy_url(env_url)
    ini_url = config.get_main_option("sqlalchemy.url") or ""
    return _normalize_sqlalchemy_url(ini_url)


def run_migrations_offline() -> None:
    """Emit SQL scripts in offline mode without connecting to a live database."""
    url = _resolve_database_url()
    transactional_ddl = bool(config.attributes.get("transactional_ddl", True))
    context.configure(
        url=url,
        target_metadata=None,
        literal_binds=True,
        transactional_ddl=transactional_ddl,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Execute migrations against a live PostgreSQL 17 engine."""
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = _resolve_database_url()
    connectable = engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=None,
            transaction_per_migration=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
