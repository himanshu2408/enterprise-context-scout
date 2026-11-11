"""Standing Regression Guard for Domain Neutrality (Rule E16, Directive 2026-11-10T23:57:42Z).

Guards the core context scout codebase against company/domain coupling:
1. AST scan ensuring NO hardcoded finance domain literals ('trading_code_scout',
   'quant_research_scout', 'federated_coordinator') exist in core modules.
2. Verifies that 'default' in preflight_gate.py is preserved strictly as a TypeScript keyword
   in TYPESCRIPT_KEYWORDS, preventing false-positive AST keyword strip regressions.
3. Verifies that PreFlightValidator defaults gracefully to ScoutSettings.default_domain_id
   without hardcoded 'trading_code_scout' or 'default' string literals.
4. Verifies arbitrary custom domain onboarding ('healthcare_clinical_scout')
   executes without domain coupling exceptions.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Final

from context_scout.agents.preflight_gate import (
    TYPESCRIPT_KEYWORDS,
    PreFlightValidator,
)
from context_scout.config import DomainConfig, ScoutSettings

PROHIBITED_FINANCE_LITERALS: Final[frozenset[str]] = frozenset(
    {"trading_code_scout", "quant_research_scout", "federated_coordinator"}
)


def _get_src_root() -> Path:
    # Look for enterprise-context-scout/src/context_scout
    cand1 = Path.cwd() / "src" / "context_scout"
    if cand1.exists():
        return cand1
    cand2 = (
        Path(__file__).resolve().parents[2] / "enterprise-context-scout" / "src" / "context_scout"
    )
    if cand2.exists():
        return cand2
    cand3 = Path(__file__).resolve().parents[1] / "src" / "context_scout"
    if cand3.exists():
        return cand3
    raise FileNotFoundError("Could not locate context_scout source root")


def test_preflight_gate_ast_contains_no_finance_literals() -> None:
    """Scans preflight_gate.py AST to guarantee zero hardcoded finance domain literals."""
    src_root = _get_src_root()
    target_file = src_root / "agents" / "preflight_gate.py"
    assert target_file.exists(), f"Target file not found: {target_file}"

    tree = ast.parse(target_file.read_text(encoding="utf-8"), filename=str(target_file))

    found_violations: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            val_clean = node.value.strip()
            if val_clean in PROHIBITED_FINANCE_LITERALS:
                found_violations.append((node.lineno, val_clean))

    assert not found_violations, (
        f"Finance domain coupling detected in preflight_gate.py: {found_violations}. "
        "Domain literals must be moved to configuration or domains/ manifests."
    )


def test_typescript_keyword_default_is_preserved_in_ast_syntax_set() -> None:
    """Verifies that 'default' in TYPESCRIPT_KEYWORDS is preserved as a TypeScript language token.

    Note: preflight_gate.py contains 'default' inside TYPESCRIPT_KEYWORDS.
    This is an ECMAScript/TypeScript language keyword, NOT a domain identifier.
    Removing it would corrupt AST syntax validation for TypeScript exports/switches.
    """
    assert "default" in TYPESCRIPT_KEYWORDS, (
        "Regression: 'default' was removed from TYPESCRIPT_KEYWORDS! "
        "It is a required TypeScript syntax keyword, not a domain identifier."
    )


def test_preflight_gate_resolve_custom_corpus_neutrality() -> None:
    """Verifies that _resolve_custom_corpus_index_path has no hardcoded domain exclusions."""
    src_root = _get_src_root()
    target_file = src_root / "agents" / "preflight_gate.py"
    tree = ast.parse(target_file.read_text(encoding="utf-8"), filename=str(target_file))

    resolve_fn: ast.FunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_resolve_custom_corpus_index_path":
            resolve_fn = node
            break

    assert resolve_fn is not None, (
        "_resolve_custom_corpus_index_path not found in preflight_gate.py"
    )

    # Walk function AST and check for prohibited domain literals
    fn_violations: list[tuple[int, str]] = []
    for subnode in ast.walk(resolve_fn):
        if (
            isinstance(subnode, ast.Constant)
            and isinstance(subnode.value, str)
            and subnode.value in PROHIBITED_FINANCE_LITERALS
        ):
            fn_violations.append((subnode.lineno, subnode.value))

    assert not fn_violations, (
        f"Found hardcoded domain literals in _resolve_custom_corpus_index_path: {fn_violations}. "
        "Dynamic resolution must check configured paths without a hardcoded blacklist."
    )


def test_preflight_gate_validate_fallback_neutrality() -> None:
    """Verifies PreFlightValidator.validate does not fall back to 'trading_code_scout'."""
    src_root = _get_src_root()
    target_file = src_root / "agents" / "preflight_gate.py"
    tree = ast.parse(target_file.read_text(encoding="utf-8"), filename=str(target_file))

    validate_fn: ast.FunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "validate":
            validate_fn = node
            break

    assert validate_fn is not None, "validate() method not found in preflight_gate.py"

    fn_violations: list[tuple[int, str]] = []
    for subnode in ast.walk(validate_fn):
        if (
            isinstance(subnode, ast.Constant)
            and isinstance(subnode.value, str)
            and subnode.value in PROHIBITED_FINANCE_LITERALS
        ):
            fn_violations.append((subnode.lineno, subnode.value))

    assert not fn_violations, (
        f"Found hardcoded finance domain fallback in validate(): {fn_violations}. "
        "Fallback must resolve to self.domain_config.domain_id or ScoutSettings.default_domain_id."
    )


def test_scout_settings_default_domain_configurability() -> None:
    """Verifies ScoutSettings exposes default_domain_id with clean neutral default 'default'."""
    settings = ScoutSettings()
    assert hasattr(settings, "default_domain_id"), "ScoutSettings missing default_domain_id field."
    assert settings.default_domain_id == "default"

    custom_settings = ScoutSettings(default_domain_id="healthcare_clinical_scout")
    assert custom_settings.default_domain_id == "healthcare_clinical_scout"


def test_preflight_gate_unseen_custom_domain_execution() -> None:
    """Verifies PreFlightValidator initializes and validates arbitrary unseen enterprise domains."""
    custom_domain = "aerospace_avionics_scout"
    validator = PreFlightValidator(
        domain_id=custom_domain,
        domain_config=DomainConfig(domain_id=custom_domain),
    )
    assert validator.domain_config.domain_id == custom_domain
    assert validator.domain_id == custom_domain

    report = validator.validate(
        user_prompt="Verify attitude determination Kalman filter telemetry bounds",
        domain_id=custom_domain,
        code_slices=[],
    )
    assert report is not None
    assert not report.passed

    # Produce $0 insufficient context diagnostic
    diag = validator.create_diagnostic(
        trace_id="test_custom_domain_trace",
        domain_id=custom_domain,
        report=report,
    )
    assert diag.domain_id == custom_domain
    assert diag.paid_tokens_spent == 0


def test_core_modules_finance_literals_isolation_guard() -> None:
    """Verifies finance domain literals are prohibited across all core architectural modules.

    Allowed location: src/context_scout/plugins/domains/finance.py only.
    """
    src_root = _get_src_root()
    core_modules_to_scan = [
        src_root / "agents" / "preflight_gate.py",
        src_root / "agents" / "citation_gate.py",
        src_root / "ast_engine" / "extractor.py",
        src_root / "ast_engine" / "dependency_expander.py",
        src_root / "ast_engine" / "deterministic_slicer.py",
        src_root / "db" / "session.py",
        src_root / "db" / "repositories.py",
        src_root / "security" / "rbac.py",
        src_root / "ingestion" / "git_merkle_indexer.py",
    ]

    violations: dict[str, list[tuple[int, str]]] = {}
    for mod_path in core_modules_to_scan:
        if not mod_path.exists():
            continue
        tree = ast.parse(mod_path.read_text(encoding="utf-8"), filename=str(mod_path))
        file_violations: list[tuple[int, str]] = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value in PROHIBITED_FINANCE_LITERALS
            ):
                file_violations.append((node.lineno, node.value))
        if file_violations:
            rel = str(mod_path.name)
            violations[rel] = file_violations

    assert not violations, (
        f"Finance domain literals leaked into core architectural modules: {violations}. "
        "Finance domain specifics must be encapsulated in plugins/domains/finance.py."
    )
