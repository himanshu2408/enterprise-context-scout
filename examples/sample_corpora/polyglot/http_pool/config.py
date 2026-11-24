"""Sample Python HTTP connection pool configuration for clean-clone CI seeding."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PoolLimits:
    """Connection pool limits and idle keepalive expiration settings."""

    max_connections: int = 100
    max_keepalive_connections: int = 20
    keepalive_expiry: float = 5.0


def build_pool_config(limits: PoolLimits) -> dict[str, int | float]:
    """Validates and normalizes pool limit parameters."""
    if limits.max_connections <= 0:
        raise ValueError("max_connections must be positive")
    return {
        "max_connections": limits.max_connections,
        "max_keepalive_connections": limits.max_keepalive_connections,
        "keepalive_expiry": limits.keepalive_expiry,
    }
