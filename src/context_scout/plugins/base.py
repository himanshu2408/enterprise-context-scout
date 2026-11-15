"""Base Service Provider Interface (SPI) protocols and models for context scout plugins.

Defines the contract for language AST extractors and external source connectors.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from context_scout.ast_engine.models import ParsedEdge, ParsedSymbol


class DependencyEdge(BaseModel):
    """External dependency or project reference extracted from manifests."""

    source_identifier: str = Field(description="Source manifest or project name")
    target_package: str = Field(description="Package name, NuGet ID, or module path")
    version_constraint: str = Field(default="*", description="Version rule, e.g. ~> 5.0, ^18.2.0")
    scope: str = Field(
        default="runtime",
        description="Scope: compile, runtime, dev, provider, project_reference",
    )
    manifest_path: str = Field(default="", description="Relative path to manifest file")


class SyncResult(BaseModel):
    """Result of an incremental synchronization run from an external source."""

    source_id: str = Field(default="", description="Canonical identifier of source connector")
    source_type: str = Field(
        default="",
        description="Type of source connector: confluence, jira, gitlab",
    )
    items_scanned: int = 0
    items_synced: int = 0
    items_indexed: int = 0
    items_updated: int = 0
    items_deleted: int = 0
    stale_items_flagged: int = 0
    new_checkpoint: str = ""
    cursor_or_checkpoint: str = ""
    duration_ms: float = 0.0
    errors: list[str] = Field(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        if not self.source_id and self.source_type:
            object.__setattr__(self, "source_id", self.source_type)
        elif not self.source_type and self.source_id:
            object.__setattr__(self, "source_type", self.source_id)
        if not self.new_checkpoint and self.cursor_or_checkpoint:
            object.__setattr__(self, "new_checkpoint", self.cursor_or_checkpoint)
        elif not self.cursor_or_checkpoint and self.new_checkpoint:
            object.__setattr__(self, "cursor_or_checkpoint", self.new_checkpoint)


@runtime_checkable
class LanguagePlugin(Protocol):
    """Protocol for language-specific AST and dependency graph extractors."""

    @property
    def language_id(self) -> str:
        """Canonical language identifier (e.g. 'csharp', 'react_redux', 'terraform')."""
        ...

    @property
    def file_extensions(self) -> tuple[str, ...]:
        """Tuple of handled file extensions, including leading dot (e.g. ('.cs',))."""
        ...

    @property
    def manifest_files(self) -> tuple[str, ...]:
        """Tuple of manifest file patterns (e.g. ('.csproj', '.sln'))."""
        ...

    def can_handle(self, file_path: str) -> bool:
        """Returns True if this plugin should handle the specific file path."""
        ...

    def extract_ast(
        self,
        code: str,
        file_path: str,
        repo_name: str = "",
        module_prefix: str = "",
    ) -> tuple[list[ParsedSymbol], list[ParsedEdge]]:
        """Extracts syntax symbols and directed dependency edges from source code."""
        ...

    def extract_manifest_dependencies(
        self,
        file_path: str,
        content: str,
    ) -> list[DependencyEdge]:
        """Extracts package dependencies and inter-project references from manifests."""
        ...


@runtime_checkable
class SourceConnectorPlugin(Protocol):
    """Protocol for external enterprise connectors (GitLab, Confluence, Jira)."""

    @property
    def source_id(self) -> str:
        """Canonical connector identifier (e.g. 'gitlab', 'confluence', 'jira')."""
        ...

    @property
    def source_type(self) -> str:
        """Connector type identifier."""
        ...

    def fetch_item(self, item_id: str, version: str | None = None) -> dict[str, Any] | None:
        """Fetches item payload by ID and optional version."""
        ...

    def mirror_permissions(self, container_key: str) -> list[str]:
        """Maps container or user identity into PostgreSQL RLS allowed_scopes."""
        ...
