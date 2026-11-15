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
    noise_filtered_count: int = 0
    noise_demoted_count: int = 0
    tombstoned_count: int = 0
    filter_breakdown: dict[str, int] = Field(default_factory=dict)
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


@runtime_checkable
class DomainPlugin(Protocol):
    """Protocol for domain-specific heuristics, prompts, seed data, and evaluation specs."""

    @property
    def domain_id(self) -> str:
        """Primary canonical identifier for this domain plugin."""
        ...

    def supported_domain_ids(self) -> tuple[str, ...]:
        """All domain IDs (code, research, federated) handled by this plugin."""
        ...

    def get_role_domain_ids(self) -> dict[str, str]:
        """Mapping of abstract roles ('code', 'research', 'federated') to domain IDs."""
        ...

    def get_default_scopes(self) -> list[str]:
        """Default principal scopes associated with this domain pack."""
        ...

    def get_adapter_specs(self) -> list[dict[str, str]]:
        """Adapter specifications (domain_id, adapter_rel_path, label, role) for tuning/evals."""
        ...

    def is_federated_domain(self, domain_id: str) -> bool:
        """Returns True if domain_id represents this plugin's federated coordinator."""
        ...

    def is_research_domain(self, domain_id: str) -> bool:
        """Returns True if domain_id represents a document/research synthesis domain."""
        ...

    def should_auto_route_research(
        self,
        prompt: str,
        domain_id: str,
        allowed_scopes: list[str] | None = None,
    ) -> bool:
        """Returns True if a code-domain prompt should auto-route through research synthesis."""
        ...

    def infer_friction_label(self, market_scope: str) -> str:
        """Infers domain-specific contextual header annotation from market scope string."""
        ...

    def infer_market_scopes(self, prompt: str) -> list[Any]:
        """Infers target market/regime enums from a user prompt."""
        ...

    def synthesize_research_brief(
        self,
        user_prompt: str,
        domain_id: str,
        chunks: list[Any],
        verified_citations: list[Any],
    ) -> Any:
        """Synthesizes a domain-specific structured research brief from verified chunks."""
        ...

    def extract_brief_metadata(self, brief: Any) -> dict[str, Any]:
        """Extracts neutral metadata dict from a domain-specific research brief."""
        ...

    def validate_research_brief(
        self,
        brief: Any,
        doc_chunks: list[Any],
    ) -> tuple[bool, str]:
        """Validates domain-specific business rules on a research brief."""
        ...

    def build_distillation_prompt_and_brief(
        self,
        domain_id: str,
        doc_id: str,
        title: str,
        sec_head: str,
        quote: str,
        chk_id: str,
        manifest: dict[str, Any],
    ) -> tuple[str, str]:
        """Builds a training prompt and structured brief JSON for CLI custom distillation."""
        ...

    def get_demo_seed_payloads(self) -> dict[str, Any]:
        """Returns domain-specific demo code/doc seed records and fallback identifiers."""
        ...

    def get_tuning_specs(self) -> dict[str, Any]:
        """Returns domain-specific distillation, training, and benchmark metadata."""
        ...

    def build_research_distillation_sample(
        self,
        domain_id: str,
        target_chunk: Any,
        quote_text: str,
        rng: Any,
    ) -> dict[str, Any]:
        """Builds valid/invalid brief payloads and prompts for teacher distillation."""
        ...

    def verify_research_brief_output(
        self,
        raw_output: str,
    ) -> tuple[bool, list[str], list[str]]:
        """Validates raw research brief JSON output and returns (is_valid, doc_ids, titles)."""
        ...
