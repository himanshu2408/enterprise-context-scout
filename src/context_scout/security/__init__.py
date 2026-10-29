"""Enterprise Security Layer: Out-of-Band RBAC, Domain Gates, and Guardrails (SEC-04)."""

from context_scout.security.guardrails import (
    PromptInjectionSanitizer,
    SafetyCategory,
    SecretRedactor,
    ShieldGemmaDecision,
    ShieldGemmaGuardrail,
    frame_untrusted_content,
)
from context_scout.security.rbac import (
    FORBIDDEN_TOOL_ARGUMENTS,
    AuthenticationRequiredError,
    DomainAuthorizationGate,
    DomainToolAccessDeniedError,
    ScopeAccessDeniedError,
    SecurityError,
    bind_session_principal,
    get_current_principal,
    has_bound_principal,
    require_scope,
    scoped_db_session,
    validate_tool_call_arguments,
)

__all__ = [
    "FORBIDDEN_TOOL_ARGUMENTS",
    "AuthenticationRequiredError",
    "DomainAuthorizationGate",
    "DomainToolAccessDeniedError",
    "PromptInjectionSanitizer",
    "SafetyCategory",
    "ScopeAccessDeniedError",
    "SecretRedactor",
    "SecurityError",
    "ShieldGemmaDecision",
    "ShieldGemmaGuardrail",
    "bind_session_principal",
    "frame_untrusted_content",
    "get_current_principal",
    "has_bound_principal",
    "require_scope",
    "scoped_db_session",
    "validate_tool_call_arguments",
]
