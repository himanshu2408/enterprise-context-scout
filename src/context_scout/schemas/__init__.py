"""Pydantic v2 contracts and mathematical normalization helpers."""

from context_scout.schemas.contracts import (
    AuthorityRules,
    CodeSlicePointer,
    DocCitationSpan,
    DocumentScope,
    DomainKnowledgeBrief,
    DomainPackManifest,
    ExecutionMode,
    ImplementationContextPack,
    InsufficientContextDiagnostic,
    ModificationIntent,
    PreFlightValidationReport,
    PrincipalContext,
    RoleType,
    VerbatimCodeSlice,
    compute_normalized_rrf,
)

__all__ = [
    "AuthorityRules",
    "CodeSlicePointer",
    "DocCitationSpan",
    "DocumentScope",
    "DomainKnowledgeBrief",
    "DomainPackManifest",
    "ExecutionMode",
    "ImplementationContextPack",
    "InsufficientContextDiagnostic",
    "ModificationIntent",
    "PreFlightValidationReport",
    "PrincipalContext",
    "RoleType",
    "VerbatimCodeSlice",
    "compute_normalized_rrf",
]
