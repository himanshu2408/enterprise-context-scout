"""ASTMutationGateRunner orchestrating >=85% mutation kill rate across all M3-03 scopes."""

from __future__ import annotations

import time
from typing import Final

from context_scout.security.mutation_gate_pkg.contracts_and_mutator import (
    DEFAULT_MIN_MUTATION_KILL_RATE,
    MutantOutcome,
    MutantSpec,
    MutationGateSummary,
    TargetMutationReport,
    _find_function_node,
    _isolated_module,
    _SiteCounterVisitor,
    _swap_mutated_function,
    _TargetModuleConfig,
)
from context_scout.security.mutation_gate_pkg.probes_agents import (
    _probe_adapter_router,
    _probe_claim_verifier,
)
from context_scout.security.mutation_gate_pkg.probes_preflight import _probe_preflight_gate
from context_scout.security.mutation_gate_pkg.probes_security import (
    _probe_guardrails_injection_rules,
    _probe_guardrails_model_armor,
    _probe_guardrails_models,
    _probe_guardrails_neural_and_shield,
    _probe_guardrails_secret_redactor,
    _probe_permission_mirror,
    _probe_pre_index_scrubber,
    _probe_rbac,
)

_DEFAULT_TARGETS: Final[tuple[_TargetModuleConfig, ...]] = (
    _TargetModuleConfig(
        scope_name="security",
        relpath="src/context_scout/security/guardrails_pkg/models_and_types.py",
        targets=((None, "validate_untrusted_nonce"),),
        probe_fn=_probe_guardrails_models,
    ),
    _TargetModuleConfig(
        scope_name="security",
        relpath="src/context_scout/security/guardrails_pkg/secret_redactor.py",
        targets=(("SecretRedactor", "redact"),),
        probe_fn=_probe_guardrails_secret_redactor,
    ),
    _TargetModuleConfig(
        scope_name="security",
        relpath="src/context_scout/security/guardrails_pkg/injection_rules.py",
        targets=(
            (None, "frame_untrusted_content"),
            ("PromptInjectionSanitizer", "scan_malicious_input"),
            ("PromptInjectionSanitizer", "sanitize"),
            ("DeterministicRuleGuardrailProvider", "evaluate"),
        ),
        probe_fn=_probe_guardrails_injection_rules,
    ),
    _TargetModuleConfig(
        scope_name="security",
        relpath="src/context_scout/security/guardrails_pkg/neural_and_shield.py",
        targets=(
            ("LocalNeuralGuardrailProvider", "_parse_neural_payload"),
            ("LocalNeuralGuardrailProvider", "evaluate"),
            ("ChainedGuardrailProvider", "evaluate"),
        ),
        probe_fn=_probe_guardrails_neural_and_shield,
    ),
    _TargetModuleConfig(
        scope_name="security",
        relpath="src/context_scout/security/guardrails_pkg/model_armor.py",
        targets=(
            ("GoogleCloudModelArmorProvider", "__init__"),
            ("GoogleCloudModelArmorProvider", "_parse_sanitization_response"),
            ("GoogleCloudModelArmorProvider", "evaluate"),
        ),
        probe_fn=_probe_guardrails_model_armor,
    ),
    _TargetModuleConfig(
        scope_name="security",
        relpath="src/context_scout/security/rbac.py",
        targets=(
            (None, "get_current_principal"),
            (None, "validate_tool_call_arguments"),
            (None, "require_scope"),
            ("DomainAuthorizationGate", "is_tool_allowed"),
            ("DomainAuthorizationGate", "authorize_tool_call"),
        ),
        probe_fn=_probe_rbac,
    ),
    _TargetModuleConfig(
        scope_name="security",
        relpath="src/context_scout/security/permission_mirror.py",
        targets=(
            (None, "sanitize_scope_identifier"),
            ("PermissionMirror", "map_gitlab_role_to_scopes"),
            ("PermissionMirror", "map_github_permissions_to_scopes"),
            ("PermissionMirror", "intersect_confluence_ancestor_scopes"),
            ("PermissionMirror", "resolve_jira_issue_scopes"),
            ("PermissionMirror", "resolve_jira_comment_scopes"),
            ("PermissionMirror", "create_principal_context"),
            ("CachedPermissionMirrorService", "__init__"),
            ("CachedPermissionMirrorService", "get_user_scopes"),
            ("CachedPermissionMirrorService", "process_revocation_webhook"),
        ),
        probe_fn=_probe_permission_mirror,
    ),
    _TargetModuleConfig(
        scope_name="security",
        relpath="src/context_scout/security/pre_index_scrubber.py",
        targets=(
            (None, "_luhn_valid"),
            (None, "_iban_valid"),
            ("PreIndexScrubber", "scrub_text"),
        ),
        probe_fn=_probe_pre_index_scrubber,
    ),
    _TargetModuleConfig(
        scope_name="preflight_gate",
        relpath="src/context_scout/agents/preflight_gate.py",
        targets=(
            ("PreFlightValidator", "check_rrf_threshold"),
            ("PreFlightValidator", "check_ast_syntax"),
            ("PreFlightValidator", "check_principal_scopes"),
            ("PreFlightValidator", "check_cryptographic_provenance"),
            ("PreFlightValidator", "check_document_freshness"),
            ("PreFlightValidator", "check_citations"),
            ("PreFlightValidator", "evaluate_gate"),
        ),
        probe_fn=_probe_preflight_gate,
    ),
    _TargetModuleConfig(
        scope_name="claim_verifier",
        relpath="src/context_scout/agents/claim_verifier.py",
        targets=(
            (None, "normalize_claim_text"),
            (None, "detect_homoglyph_spoofing"),
            ("ClaimVerifier", "verify_single_claim"),
            ("ClaimVerifier", "verify_claims"),
        ),
        probe_fn=_probe_claim_verifier,
    ),
    _TargetModuleConfig(
        scope_name="adapter_router",
        relpath="src/context_scout/agents/llm_backend.py",
        targets=(
            ("AdapterRouter", "_verify_adapter_weights"),
            ("AdapterRouter", "route"),
        ),
        probe_fn=_probe_adapter_router,
    ),
)


class ASTMutationGateRunner:
    """In-memory AST mutation testing engine verifying >= 85% kill rate across M3-03 targets."""

    def __init__(self, *, min_kill_rate: float = DEFAULT_MIN_MUTATION_KILL_RATE) -> None:
        self.min_kill_rate = min_kill_rate

    def _run_module_targets(self, cfg: _TargetModuleConfig) -> list[MutantOutcome]:
        outcomes: list[MutantOutcome] = []
        with _isolated_module(cfg.relpath) as (mod, tree, abs_path):
            # Baseline verification: unmutated code MUST pass probe 100%.
            cfg.probe_fn(mod)

            for class_name, fn_name in cfg.targets:
                orig_fn = _find_function_node(tree, class_name, fn_name)
                if orig_fn is None:
                    raise RuntimeError(
                        f"Mutation target {class_name}.{fn_name} not found in {cfg.relpath}"
                    )
                visitor = _SiteCounterVisitor()
                visitor.visit(orig_fn)

                for site_idx, line_no, mut_kind, desc in visitor.candidates:
                    spec = MutantSpec(
                        module_relpath=cfg.relpath,
                        scope_name=cfg.scope_name,
                        class_name=class_name,
                        function_name=fn_name,
                        site_index=site_idx,
                        line_no=line_no,
                        mutation_kind=mut_kind,
                        description=desc,
                    )
                    t0 = time.perf_counter()
                    _, restore_fn = _swap_mutated_function(
                        mod, abs_path, orig_fn, class_name, site_idx
                    )
                    killed = False
                    kill_reason = ""
                    try:
                        cfg.probe_fn(mod)
                    except Exception as exc:
                        killed = True
                        kill_reason = f"{type(exc).__name__}: {exc}"
                    finally:
                        restore_fn()
                    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
                    outcomes.append(
                        MutantOutcome(
                            spec=spec,
                            killed=killed,
                            kill_reason=kill_reason,
                            elapsed_ms=elapsed_ms,
                        )
                    )
        return outcomes

    def run_scope(self, scope_name: str) -> TargetMutationReport:
        """Runs AST mutation testing for one named scope (`security`, `preflight_gate`, etc.)."""
        t0 = time.perf_counter()
        matching = [cfg for cfg in _DEFAULT_TARGETS if cfg.scope_name == scope_name]
        if not matching:
            raise ValueError(f"Unknown mutation gate scope: {scope_name!r}")

        outcomes: list[MutantOutcome] = []
        for cfg in matching:
            outcomes.extend(self._run_module_targets(cfg))

        total = len(outcomes)
        killed = sum(1 for o in outcomes if o.killed)
        survived = total - killed
        kill_rate = round(killed / total, 4) if total > 0 else 0.0
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return TargetMutationReport(
            scope_name=scope_name,
            total_mutants=total,
            killed_mutants=killed,
            survived_mutants=survived,
            kill_rate=kill_rate,
            min_kill_rate=self.min_kill_rate,
            passed=total > 0 and kill_rate >= self.min_kill_rate,
            elapsed_ms=elapsed_ms,
            survivors=tuple(o for o in outcomes if not o.killed),
        )

    def run_all(self) -> MutationGateSummary:
        """Runs AST mutation testing across all 4 required M3-03 scopes."""
        t0 = time.perf_counter()
        scope_names = list(dict.fromkeys(cfg.scope_name for cfg in _DEFAULT_TARGETS))
        reports: dict[str, TargetMutationReport] = {}
        for scope in scope_names:
            reports[scope] = self.run_scope(scope)

        total = sum(r.total_mutants for r in reports.values())
        killed = sum(r.killed_mutants for r in reports.values())
        survived = total - killed
        overall_rate = round(killed / total, 4) if total > 0 else 0.0
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        all_passed = (
            total > 0
            and overall_rate >= self.min_kill_rate
            and all(r.passed for r in reports.values())
        )
        return MutationGateSummary(
            total_mutants=total,
            killed_mutants=killed,
            survived_mutants=survived,
            overall_kill_rate=overall_rate,
            min_kill_rate=self.min_kill_rate,
            passed=all_passed,
            elapsed_ms=elapsed_ms,
            scope_reports=reports,
        )
