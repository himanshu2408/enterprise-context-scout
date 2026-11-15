"""Extensible Plug-In Architecture and Multi-Source Connectors (M3)."""

from context_scout.plugins.base import (
    DependencyEdge,
    LanguagePlugin,
    SourceConnectorPlugin,
    SyncResult,
)
from context_scout.plugins.registry import PluginRegistry

__all__ = [
    "DependencyEdge",
    "LanguagePlugin",
    "PluginRegistry",
    "SourceConnectorPlugin",
    "SyncResult",
]
