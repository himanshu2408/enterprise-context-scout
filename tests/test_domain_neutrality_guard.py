"""Standing Regression Guard for Domain Neutrality (Rule E16) & Exception Hygiene (PA-D).

Guards the core context scout codebase against company/domain coupling and silent exceptions:
1. Dynamic AST scan over ALL `src/context_scout/**/*.py` ensuring NO hardcoded finance/quant
   literals or symbols exist outside the strictly bounded `FINANCE_PLUGIN_ALLOWLIST`.
2. Mutation proof verifying the AST neutrality guard fails when a finance literal or symbol
   is injected into a non-allowlisted file.
3. Dynamic AST scan over `src/context_scout/**/*.py` forbidding silent
   `except (Exception|BaseException): pass | continue` blocks, plus a mutation proof.
4. Verifies that 'default' in preflight_gate.py is preserved strictly as a TypeScript keyword
   in TYPESCRIPT_KEYWORDS, preventing false-positive AST keyword strip regressions.
5. Verifies arbitrary custom domain onboarding ('aerospace_avionics_scout') executes cleanly.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Final

from context_scout.agents.preflight_gate import (
    TYPESCRIPT_KEYWORDS,
    PreFlightValidator,
)
from context_scout.config import DomainConfig, ScoutSettings

PROHIBITED_FINANCE_LITERALS: Final[frozenset[str]] = frozenset(
    {
        "trading_code_scout",
        "quant_research_scout",
        "federated_coordinator",
        "in_nse",
        "de_xetra",
        "us_equity",
        "xetra",
        "nifty",
        "almgren",
        "garch",
        "india_nse",
        "germany_xetra",
        "us_nyse",
        "auto_quant",
    }
)

PROHIBITED_WORD_REGEX: Final[re.Pattern[str]] = re.compile(
    r"\b(nse|kelly|sharpe)\b",
    re.IGNORECASE,
)

# Strictly bounded allowlist of finance-plugin and finance-corpus/schema modules.
# Core modules (agents/*, api/*, ast_engine/*, cli.py, config.py, db/*,
# mcp_servers/{common,code_repo_mcp,doc_corpus_mcp}.py, queue/*, retrieval/*,
# security/*, telemetry/*, tuning/*, evals/{harness,unseen_ab_benchmark,unseen_repo_dataset}.py)
# MUST NEVER appear on this allowlist.
FINANCE_PLUGIN_ALLOWLIST: Final[frozenset[str]] = frozenset(
    {
        "plugins/domains/finance.py",
        "mcp_servers/quant_corpus_mcp.py",
        "mcp_servers/seed_code_corpus.py",
        "ingestion/paper_harvester.py",
        "ingestion/harvest_pipeline.py",
        "ingestion/scepticism_gate.py",
        "evals/golden_dataset.py",
        "schemas/contracts.py",
        "schemas/__init__.py",
    }
)
FINANCE_PLUGIN_PREFIXES: Final[tuple[str, ...]] = ("trading_strategies/",)


def _get_src_root() -> Path:
    # Prefer the checkout containing this test file so isolated worktrees scan their own src tree.
    cand1 = Path(__file__).resolve().parents[1] / "src" / "context_scout"
    if cand1.exists():
        return cand1
    cand2 = Path.cwd() / "src" / "context_scout"
    if cand2.exists():
        return cand2
    cand3 = (
        Path(__file__).resolve().parents[2] / "enterprise-context-scout" / "src" / "context_scout"
    )
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


def _match_prohibited_finance_token(text: str) -> str | None:
    """Returns matched finance/quant token if text violates core domain neutrality."""
    lowered = text.lower()
    for token in sorted(PROHIBITED_FINANCE_LITERALS, key=len, reverse=True):
        if token in lowered:
            return token
    word_match = PROHIBITED_WORD_REGEX.search(text)
    if word_match:
        return word_match.group(1).lower()
    return None


def _is_allowlisted_finance_module(rel_posix: str) -> bool:
    return rel_posix in FINANCE_PLUGIN_ALLOWLIST or rel_posix.startswith(FINANCE_PLUGIN_PREFIXES)


def scan_directory_for_finance_coupling(src_root: Path) -> dict[str, list[tuple[int, str, str]]]:
    """Dynamically scans all Python files under src_root outside FINANCE_PLUGIN_ALLOWLIST."""
    violations: dict[str, list[tuple[int, str, str]]] = {}
    for py_path in sorted(src_root.rglob("*.py")):
        rel_posix = py_path.relative_to(src_root).as_posix()
        if _is_allowlisted_finance_module(rel_posix):
            continue

        tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
        file_hits: list[tuple[int, str, str]] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                matched = _match_prohibited_finance_token(node.value)
                if matched:
                    file_hits.append((node.lineno, f"str:{matched}", repr(node.value[:60])))
            elif isinstance(node, ast.JoinedStr):
                for part in node.values:
                    if isinstance(part, ast.Constant) and isinstance(part.value, str):
                        matched = _match_prohibited_finance_token(part.value)
                        if matched:
                            file_hits.append(
                                (part.lineno, f"fstr:{matched}", repr(part.value[:60]))
                            )
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                matched = _match_prohibited_finance_token(node.name)
                if matched:
                    file_hits.append((node.lineno, f"def:{matched}", node.name))
            elif isinstance(node, ast.Name):
                matched = _match_prohibited_finance_token(node.id)
                if matched:
                    file_hits.append((node.lineno, f"name:{matched}", node.id))
            elif isinstance(node, ast.Attribute):
                matched = _match_prohibited_finance_token(node.attr)
                if matched:
                    file_hits.append((node.lineno, f"attr:{matched}", node.attr))
            elif isinstance(node, ast.alias):
                for alias_name in (node.name, node.asname or ""):
                    matched = _match_prohibited_finance_token(alias_name)
                    if matched:
                        file_hits.append((node.lineno, f"import:{matched}", alias_name))
            elif isinstance(node, ast.arg):
                matched = _match_prohibited_finance_token(node.arg)
                if matched:
                    file_hits.append((node.lineno, f"arg:{matched}", node.arg))
            elif isinstance(node, ast.keyword) and node.arg:
                matched = _match_prohibited_finance_token(node.arg)
                if matched:
                    file_hits.append((node.lineno, f"kwarg:{matched}", node.arg))

        if file_hits:
            violations[rel_posix] = sorted(set(file_hits))

    return violations


def _handler_catches_broad_exception(handler: ast.ExceptHandler) -> bool:
    """Returns True if an ExceptHandler catches bare except, Exception, or BaseException."""
    if handler.type is None:
        return True
    if isinstance(handler.type, ast.Name) and handler.type.id in ("Exception", "BaseException"):
        return True
    if isinstance(handler.type, ast.Tuple):
        return any(
            isinstance(elt, ast.Name) and elt.id in ("Exception", "BaseException")
            for elt in handler.type.elts
        )
    return False


def find_silent_broad_except_handlers(src_root: Path) -> dict[str, list[tuple[int, str]]]:
    """Scans all Python files under src_root for silent broad except pass/continue blocks."""
    violations: dict[str, list[tuple[int, str]]] = {}
    for py_path in sorted(src_root.rglob("*.py")):
        rel_posix = py_path.relative_to(src_root).as_posix()
        tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
        file_hits: list[tuple[int, str]] = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ExceptHandler)
                and _handler_catches_broad_exception(node)
                and len(node.body) == 1
                and isinstance(node.body[0], (ast.Pass, ast.Continue))
            ):
                file_hits.append((node.lineno, type(node.body[0]).__name__))
        if file_hits:
            violations[rel_posix] = sorted(file_hits)
    return violations


def test_core_modules_finance_literals_isolation_guard() -> None:
    """Dynamically scans ALL src/context_scout/**/*.py files for finance coupling."""
    src_root = _get_src_root()
    violations = scan_directory_for_finance_coupling(src_root)
    assert not violations, (
        f"Finance domain literals or symbols leaked outside FINANCE_PLUGIN_ALLOWLIST: {violations}."
    )


def test_domain_neutrality_guard_mutation_proof(tmp_path: Path) -> None:
    """Mutation proof: verifies neutrality guard detects injected finance strings & symbols."""
    fake_src = tmp_path / "src" / "context_scout"
    (fake_src / "agents").mkdir(parents=True)
    leaked_file = fake_src / "agents" / "leaked_core.py"
    leaked_file.write_text(
        'DEFAULT_DOMAIN = "trading_code_scout"\n'
        'suffix = "v1"\n'
        'MARKET = f"region_DE_XETRA_{suffix}"\n'
        'REGION_TAG = "IN_NSE"\n'
        "def run_almgren_eval() -> str:\n"
        '    return "garch"\n',
        encoding="utf-8",
    )

    violations = scan_directory_for_finance_coupling(fake_src)
    assert "agents/leaked_core.py" in violations
    found_tags = {hit[1] for hit in violations["agents/leaked_core.py"]}
    assert "str:trading_code_scout" in found_tags
    assert "str:de_xetra" in found_tags or "fstr:de_xetra" in found_tags
    assert "str:in_nse" in found_tags
    assert "def:almgren" in found_tags
    assert "str:garch" in found_tags


def test_no_silent_broad_except_handlers() -> None:
    """AST guard forbidding silent `except (Exception|BaseException): pass | continue`."""
    src_root = _get_src_root()
    violations = find_silent_broad_except_handlers(src_root)
    assert not violations, (
        f"Silent broad except pass/continue blocks found in src/context_scout: {violations}. "
        "Log structured context or raise/abstain explicitly."
    )


def test_silent_broad_except_guard_mutation_proof(tmp_path: Path) -> None:
    """Mutation proof: verifies silent except AST guard catches injected pass/continue blocks."""
    fake_src = tmp_path / "src" / "context_scout"
    fake_src.mkdir(parents=True)
    bad_file = fake_src / "silent_sample.py"
    bad_file.write_text(
        "def f() -> None:\n"
        "    for _ in range(1):\n"
        "        try:\n"
        "            _ = 1 / 0\n"
        "        except Exception:\n"
        "            pass\n"
        "        try:\n"
        "            _ = 1 / 0\n"
        "        except BaseException:\n"
        "            continue\n",
        encoding="utf-8",
    )

    violations = find_silent_broad_except_handlers(fake_src)
    assert violations == {"silent_sample.py": [(5, "Pass"), (9, "Continue")]}
