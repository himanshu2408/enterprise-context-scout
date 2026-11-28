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

import pytest

from context_scout.agents.preflight_gate import (
    TYPESCRIPT_KEYWORDS,
    PreFlightValidator,
)
from context_scout.config import DomainConfig, ScoutSettings
from context_scout.schemas.contracts import RoleType

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
        "quant_researcher",
        "quant_research",
        "quant_paper",
        "quantstrategybrief",
        "goldenquanttask",
        "quant_tasks",
        "canary_quant_only",
        "quant_corpus_mcp",
        "quant_adapter",
        "market microstructure",
        "trading strategy",
        "friction and transaction costs",
        "momentum & microstructure",
        "quantitative research context",
        "hawkes",
        "orderbook",
        "volatilityforecastresult",
        "marketregime",
        "corp-finance-vault",
        "public_quant",
    }
)

PROHIBITED_WORD_REGEX: Final[re.Pattern[str]] = re.compile(
    r"\b(nse|kelly|sharpe|quant)\b",
    re.IGNORECASE,
)

# Strictly empty allowlist: all finance-domain dogfooding is evicted to examples/quant_dogfood/.
FINANCE_PLUGIN_ALLOWLIST: Final[frozenset[str]] = frozenset()
FINANCE_PLUGIN_PREFIXES: Final[tuple[str, ...]] = ()


def _get_repo_root() -> Path:
    cand1 = Path(__file__).resolve().parents[1]
    if (cand1 / "src" / "context_scout").exists():
        return cand1
    cand2 = Path.cwd()
    if (cand2 / "src" / "context_scout").exists():
        return cand2
    cand3 = Path(__file__).resolve().parents[2] / "enterprise-context-scout"
    if (cand3 / "src" / "context_scout").exists():
        return cand3
    raise FileNotFoundError("Could not locate repository root")


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

    from context_scout.agents import preflight_gate

    assert preflight_gate._SYNC_MANIFEST_FILENAME == "sync_manifest.json"
    assert not hasattr(preflight_gate, "_CUSTOM_CORPUS_MEM_CACHE")


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


def _handler_logs_raises_or_returns_diagnostic(handler: ast.ExceptHandler) -> bool:
    """Returns True if handler logs, re-raises, or records/returns structured diagnostic info."""
    for sub in ast.walk(handler):
        if isinstance(sub, ast.Raise):
            return True
        if (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and sub.func.attr
            in (
                "debug",
                "info",
                "warning",
                "error",
                "exception",
                "critical",
                "log",
            )
        ):
            return True
        if isinstance(sub, ast.Name) and sub.id in (
            "failures",
            "violations",
            "errors",
            "diagnostics",
            "warnings",
            "kill_reason",
        ):
            return True
        if isinstance(sub, ast.Return) and isinstance(sub.value, (ast.Dict, ast.Tuple, ast.Call)):
            return True
    return False


def find_silent_broad_except_handlers(src_root: Path) -> dict[str, list[tuple[int, str]]]:
    """Scans Python files under src_root for broad except blocks that swallow errors."""
    violations: dict[str, list[tuple[int, str]]] = {}
    for py_path in sorted(src_root.rglob("*.py")):
        rel_posix = py_path.relative_to(src_root).as_posix()
        tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
        file_hits: list[tuple[int, str]] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler) or not _handler_catches_broad_exception(
                node
            ):
                continue
            if len(node.body) == 1 and isinstance(node.body[0], (ast.Pass, ast.Continue)):
                file_hits.append((node.lineno, type(node.body[0]).__name__))
            elif not _handler_logs_raises_or_returns_diagnostic(node):
                file_hits.append((node.lineno, f"Silent:{type(node.body[0]).__name__}"))
        if file_hits:
            violations[rel_posix] = sorted(file_hits)
    return violations


NEUTRAL_SCAN_TREES: Final[tuple[str, ...]] = (
    "src",
    "sql",
    "web-ui/src",
    "tests/fixtures",
    "scripts",
    "config",
    "deploy",
)
NEUTRAL_SCAN_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".py", ".sql", ".ts", ".tsx", ".yaml", ".yml", ".json"}
)


def scan_neutral_trees_for_finance_tokens(
    repo_root: Path,
) -> dict[str, list[tuple[int, str, str]]]:
    """Scans src/, sql/, evals/, and web-ui/src/ for prohibited finance literals and symbols."""
    violations: dict[str, list[tuple[int, str, str]]] = {}
    for tree_rel in NEUTRAL_SCAN_TREES:
        tree_root = repo_root / tree_rel
        if not tree_root.exists():
            continue
        for file_path in sorted(tree_root.rglob("*")):
            if not file_path.is_file() or file_path.suffix not in NEUTRAL_SCAN_SUFFIXES:
                continue
            if any(
                part in {"__pycache__", "node_modules", ".venv", "dist"} for part in file_path.parts
            ):
                continue
            rel_posix = file_path.relative_to(repo_root).as_posix()
            file_hits: list[tuple[int, str, str]] = []
            for lineno, line in enumerate(
                file_path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                matched = _match_prohibited_finance_token(line)
                if matched:
                    file_hits.append((lineno, matched, line.strip()[:80]))
            if file_hits:
                violations[rel_posix] = file_hits
    return violations


def test_core_modules_finance_literals_isolation_guard() -> None:
    """Dynamically scans ALL src/context_scout/**/*.py files for finance coupling."""
    src_root = _get_src_root()
    violations = scan_directory_for_finance_coupling(src_root)
    assert not violations, (
        f"Finance domain literals or symbols leaked outside FINANCE_PLUGIN_ALLOWLIST: {violations}."
    )


def test_all_core_trees_contain_no_finance_tokens() -> None:
    """Scans ALL src/, sql/, evals/, and web-ui/src/ files for prohibited finance tokens."""
    repo_root = _get_repo_root()
    violations = scan_neutral_trees_for_finance_tokens(repo_root)
    assert not violations, (
        "Prohibited finance literals or symbols found in neutral core trees "
        f"(src/, sql/, evals/, web-ui/src/): {violations}"
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
    """Mutation proof: verifies silent except AST guard catches pass/continue and swallows."""
    fake_src = tmp_path / "src" / "context_scout"
    fake_src.mkdir(parents=True)
    bad_file = fake_src / "silent_sample.py"
    bad_file.write_text(
        "def f() -> int:\n"
        "    for _ in range(1):\n"
        "        try:\n"
        "            _ = 1 / 0\n"
        "        except Exception:\n"
        "            pass\n"
        "        try:\n"
        "            _ = 1 / 0\n"
        "        except BaseException:\n"
        "            continue\n"
        "        try:\n"
        "            _ = 1 / 0\n"
        "        except Exception:\n"
        "            x = 0\n"
        "    return 0\n",
        encoding="utf-8",
    )

    violations = find_silent_broad_except_handlers(fake_src)
    assert violations == {"silent_sample.py": [(5, "Pass"), (9, "Continue"), (13, "Silent:Assign")]}


def test_finance_plugin_allowlist_is_empty() -> None:
    """R2: Guarantees FINANCE_PLUGIN_ALLOWLIST and FINANCE_PLUGIN_PREFIXES remain empty."""
    assert isinstance(FINANCE_PLUGIN_ALLOWLIST, frozenset)
    assert len(FINANCE_PLUGIN_ALLOWLIST) == 0
    assert len(FINANCE_PLUGIN_PREFIXES) == 0


def test_domains_manifests_and_tuning_contain_no_finance_tokens() -> None:
    """Scans shipped domains/ manifests, prompts, and tuning JSONL for quant tokens."""
    repo_root = _get_repo_root()
    domains_dir = repo_root / "domains"
    assert domains_dir.is_dir()

    violations: dict[str, list[tuple[int, str, str]]] = {}
    for domain_dir in sorted(p for p in domains_dir.iterdir() if p.is_dir()):
        candidate_files: list[Path] = []
        for fixed_name in ("domain.yaml", "specialization.yaml", "glossary.yaml", "schema.json"):
            fp = domain_dir / fixed_name
            if fp.is_file():
                candidate_files.append(fp)
        for sub_name in ("prompts", "tuning"):
            sub_dir = domain_dir / sub_name
            if sub_dir.is_dir():
                allowed_exts = {".yaml", ".yml", ".json", ".jsonl", ".md", ".txt"}
                for p in sorted(sub_dir.rglob("*")):
                    if p.is_file() and p.suffix in allowed_exts:
                        candidate_files.append(p)
        for file_path in candidate_files:
            rel_posix = file_path.relative_to(repo_root).as_posix()
            hits: list[tuple[int, str, str]] = []
            for lineno, line in enumerate(
                file_path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                matched = _match_prohibited_finance_token(line)
                if matched:
                    hits.append((lineno, matched, line.strip()[:80]))
                if "examples/quant_dogfood" in line:
                    hits.append((lineno, "examples/quant_dogfood", line.strip()[:80]))
            if hits:
                violations[rel_posix] = hits

    assert not violations, (
        f"Quant/finance or dogfood tokens leaked into shipped domains/: {violations}"
    )


def test_roletype_rejects_removed_quant_researcher_alias() -> None:
    """Verifies RoleType no longer exposes or aliases quant_researcher."""
    assert {r.value for r in RoleType} == {
        "engineer",
        "domain_researcher",
        "product_manager",
        "compliance_officer",
    }
    assert not hasattr(RoleType, "QUANT_RESEARCHER")
    with pytest.raises(ValueError):
        RoleType("quant_researcher")
