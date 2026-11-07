"""Two-Arm Unseen-Repository A/B Benchmark Runner (N=30 Real Merged Bug-Fix PRs).

Executes a rigorous, controlled two-arm A/B experiment on 30 real merged bug-fix PRs
from two unseen public Python repositories (``encode/httpx`` and ``pallets/click``)
with frozen QLoRA adapters (zero retraining):

- **Arm A (Baseline — Direct Exploration):** The execution model explores the unseen
  repository at ``parent_commit_sha`` via a bounded multi-turn tool loop
  (``list_repo_files``, ``grep_repo``, ``read_file_lines``, max 6 turns) and
  synthesizes localization + a unified diff patch.
- **Arm B (Treatment — Enterprise Context Scout + 1-Shot Execution):** Incrementally
  indexes the unseen repository at ``parent_commit_sha`` via ``GitMerkleIndexer`` +
  ``ASTExtractor`` (Tree-Sitter) + PostgreSQL 17 ``FORCE RLS``, runs the 4-stage
  ``ScoutWorkflow`` with the live frozen QLoRA adapter (``gemma4_26b_code_scout``)
  and 1-hop ``ASTDependencyExpander`` + 4-check ``PreFlightValidator``, and invokes
  the exact same execution model in a single shot on the compact
  ``ImplementationContextPack``.

Records exact token counts (from ``google-genai`` ``usage_metadata`` when a cloud API
key is configured, or exact BPE token counts via the model's tokenizer when running
against local MLX models), wall-clock latency, File & Symbol Recall@5, frozen QLoRA
JSON schema validity on unseen repos, 10-task ``GitWorktreeSandbox`` pytest
verification, median/IQR statistics, and an explicit failure taxonomy of honest misses.
"""

from __future__ import annotations

import datetime
import gc
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from context_scout.agents.domain_registry import DomainRegistry
from context_scout.agents.gemini_executor import GeminiImplementationExecutor
from context_scout.agents.scout_workflow import LocalGemmaClient, ScoutWorkflow
from context_scout.config import ScoutSettings
from context_scout.db.repositories import InMemoryRLSRepository, PostgresRLSRepository
from context_scout.db.session import RLSConnectionManager
from context_scout.evals.unseen_repo_dataset import (
    UnseenRepoBenchmarkSuite,
    UnseenRepoBugTask,
    build_unseen_repo_suite,
    verify_frozen_adapter_integrity,
)
from context_scout.ingestion.git_merkle_indexer import GitMerkleIndexer
from context_scout.mcp_servers.common import PostgresScoutGraphAdapter
from context_scout.schemas.contracts import (
    ImplementationContextPack,
    InsufficientContextDiagnostic,
    PrincipalContext,
    RoleType,
)
from context_scout.security.rbac import scoped_db_session

logger = logging.getLogger(__name__)

# Verified October 2026 Gemini 3.8 Flash pricing ($0.75 / 1M input, $3.75 / 1M output)
GEMINI_FLASH_INPUT_USD_PER_1M: Final[float] = 0.75
GEMINI_FLASH_OUTPUT_USD_PER_1M: Final[float] = 3.75
MAX_BENCHMARK_BUDGET_USD: Final[float] = 3.00
MAX_TOOL_OUTPUT_CHARS: Final[int] = 14000
MAX_ARM_A_TURNS: Final[int] = 6


class ToolCallTrace(BaseModel):
    """Per-turn tool call and token accounting trace in Arm A or Arm B."""

    model_config = ConfigDict(extra="forbid")

    turn_index: int = Field(..., ge=1)
    tool_name: Literal[
        "list_repo_files",
        "grep_repo",
        "read_file_lines",
        "submit_localization",
        "one_shot_context_pack_execution",
    ]
    tool_args: dict[str, Any] = Field(default_factory=dict)
    tool_output_chars: int = Field(default=0, ge=0)
    prompt_tokens: int = Field(..., ge=0)
    completion_tokens: int = Field(..., ge=0)
    latency_ms: float = Field(..., ge=0.0)


class ArmExecutionResult(BaseModel):
    """Measured metrics for a single task under either Arm A or Arm B."""

    model_config = ConfigDict(extra="forbid")

    arm: Literal["A_direct_exploration", "B_context_scout"]
    task_id: str
    repo_slug: str
    execution_model_id: str
    token_accounting_source: Literal[
        "google_genai_usage_metadata",
        "mlx_tokenizer_exact_bpe",
    ]
    turns_used: int = Field(..., ge=1)
    input_tokens: int = Field(..., ge=0)
    output_tokens: int = Field(..., ge=0)
    total_tokens: int = Field(..., ge=0)
    estimated_frontier_cost_usd: float = Field(..., ge=0.0)
    wall_clock_latency_ms: float = Field(..., ge=0.0)
    scout_indexing_and_preflight_ms: float = Field(default=0.0, ge=0.0)
    frozen_qlora_json_valid: bool | None = None
    frozen_qlora_latency_ms: float = Field(default=0.0, ge=0.0)
    localized_files: list[str] = Field(default_factory=list)
    localized_symbols: list[str] = Field(default_factory=list)
    file_recall_at_5: bool
    symbol_recall_at_5: bool
    preflight_gate_passed: bool
    auto_attached_1hop_slices: int = Field(default=0, ge=0)
    generated_patch_diff: str = ""
    pytest_verified: bool | None = None
    pytest_output_summary: str = ""
    tool_traces: list[ToolCallTrace] = Field(default_factory=list)
    failure_category: str | None = None


class TaskABComparison(BaseModel):
    """Side-by-side comparison of Arm A and Arm B on a single unseen PR task."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    repo_slug: str
    pr_number: int
    pr_url: str
    title: str
    ground_truth_files: list[str]
    ground_truth_symbols: list[str]
    arm_a: ArmExecutionResult
    arm_b: ArmExecutionResult
    input_token_savings_pct: float
    total_token_savings_pct: float
    outcome_classification: Literal[
        "BOTH_HIT",
        "ARM_B_ONLY_HIT",
        "ARM_A_ONLY_HIT",
        "BOTH_MISSED",
    ]


class DistributionStats(BaseModel):
    """Non-parametric and parametric summary statistics across N tasks."""

    model_config = ConfigDict(extra="forbid")

    median: float
    iqr_25: float
    iqr_75: float
    iqr_span: float
    mean: float
    min_val: float
    max_val: float


class UnseenRepoABReport(BaseModel):
    """Complete two-arm unseen-repository A/B benchmark report."""

    model_config = ConfigDict(extra="forbid")

    benchmark_id: str
    generated_at_utc: str
    sample_size_n: int
    repos_evaluated: list[str]
    frozen_code_adapter_sha256: str
    frozen_qlora_json_valid_count: int
    frozen_qlora_json_valid_pct: float
    execution_model_id: str
    token_accounting_source: str
    postgres_rls_verified: bool
    arm_a_input_tokens: DistributionStats
    arm_b_input_tokens: DistributionStats
    arm_a_total_tokens: DistributionStats
    arm_b_total_tokens: DistributionStats
    input_token_savings_pct: DistributionStats
    total_token_savings_pct: DistributionStats
    arm_a_latency_ms: DistributionStats
    arm_b_latency_ms: DistributionStats
    arm_a_file_recall_at_5_count: int
    arm_a_file_recall_at_5_pct: float
    arm_b_file_recall_at_5_count: int
    arm_b_file_recall_at_5_pct: float
    arm_a_symbol_recall_at_5_count: int
    arm_a_symbol_recall_at_5_pct: float
    arm_b_symbol_recall_at_5_count: int
    arm_b_symbol_recall_at_5_pct: float
    arm_a_pytest_pass_count: int
    arm_b_pytest_pass_count: int
    pytest_subset_size: int
    arm_a_total_cost_usd: float
    arm_b_total_cost_usd: float
    honest_losses_and_failures: list[dict[str, Any]]
    task_comparisons: list[TaskABComparison]


def compute_distribution_stats(values: list[float]) -> DistributionStats:
    """Compute median, 25th/75th percentile (IQR), mean, min, and max."""
    if not values:
        return DistributionStats(
            median=0.0,
            iqr_25=0.0,
            iqr_75=0.0,
            iqr_span=0.0,
            mean=0.0,
            min_val=0.0,
            max_val=0.0,
        )
    sorted_vals = sorted(float(v) for v in values)
    n = len(sorted_vals)

    def _percentile(p: float) -> float:
        pos = (n - 1) * p
        lower = int(pos)
        upper = min(lower + 1, n - 1)
        weight = pos - lower
        return sorted_vals[lower] * (1.0 - weight) + sorted_vals[upper] * weight

    med = round(_percentile(0.50), 2)
    q25 = round(_percentile(0.25), 2)
    q75 = round(_percentile(0.75), 2)
    return DistributionStats(
        median=med,
        iqr_25=q25,
        iqr_75=q75,
        iqr_span=round(q75 - q25, 2),
        mean=round(sum(sorted_vals) / n, 2),
        min_val=round(sorted_vals[0], 2),
        max_val=round(sorted_vals[-1], 2),
    )


def compute_frontier_cost_usd(input_tokens: int, output_tokens: int) -> float:
    """Calculate USD cost at Gemini 3.8 Flash rates ($0.75/1M in, $3.75/1M out)."""
    in_cost = (input_tokens / 1_000_000.0) * GEMINI_FLASH_INPUT_USD_PER_1M
    out_cost = (output_tokens / 1_000_000.0) * GEMINI_FLASH_OUTPUT_USD_PER_1M
    return round(in_cost + out_cost, 6)


def check_file_recall(localized_files: list[str], ground_truth_files: tuple[str, ...]) -> bool:
    """Return True if any ground-truth file appears in top-5 localized files."""
    top5 = set(localized_files[:5])
    return any(gf in top5 for gf in ground_truth_files)


def check_symbol_recall(
    localized_symbols: list[str], ground_truth_symbols: tuple[str, ...]
) -> bool:
    """Return True if any ground-truth symbol (or enclosing/child symbol) is in top-5."""
    top5 = localized_symbols[:5]
    for gs in ground_truth_symbols:
        for ls in top5:
            if ls == gs or ls.startswith(gs + ".") or gs.startswith(ls + "."):
                return True
    return False


class RepoExplorationTools:
    """Bounded read-only repository exploration tools for Arm A baseline."""

    def __init__(self, repo_dir: Path, source_subdir: str) -> None:
        self.repo_dir = repo_dir
        self.source_subdir = source_subdir

    def list_repo_files(self) -> str:
        """List production Python source files in the repository with line counts."""
        out_lines: list[str] = []
        base = self.repo_dir / self.source_subdir
        if not base.exists():
            base = self.repo_dir
        for py_file in sorted(base.rglob("*.py")):
            rel = py_file.relative_to(self.repo_dir).as_posix()
            if rel.startswith(("tests/", "examples/", "docs/")):
                continue
            try:
                line_cnt = len(py_file.read_text(encoding="utf-8", errors="replace").splitlines())
            except OSError:
                line_cnt = 0
            out_lines.append(f"{rel} ({line_cnt} lines)")
        result = "\n".join(out_lines)
        return result[:MAX_TOOL_OUTPUT_CHARS]

    def grep_repo(self, pattern: str) -> str:
        """Search production Python files for a regex/literal pattern with line numbers."""
        if not pattern.strip():
            return "Empty pattern."
        cmd = [
            "git",
            "-C",
            str(self.repo_dir),
            "grep",
            "-n",
            "-I",
            "-E",
            "-e",
            pattern,
            "--",
            self.source_subdir,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        output = proc.stdout.strip()
        if not output:
            cmd_i = [
                "git",
                "-C",
                str(self.repo_dir),
                "grep",
                "-n",
                "-i",
                "-I",
                "-E",
                "-e",
                pattern,
                "--",
                self.source_subdir,
            ]
            proc_i = subprocess.run(cmd_i, capture_output=True, text=True, check=False)
            output = proc_i.stdout.strip() or f"No matches for pattern: {pattern}"
        return output[:MAX_TOOL_OUTPUT_CHARS]

    def read_file_lines(self, file_path: str, start_line: int = 1, end_line: int = 160) -> str:
        """Read numbered lines from a file in the repository (capped at MAX_TOOL_OUTPUT_CHARS)."""
        clean_rel = file_path.strip().lstrip("/")
        full_path = (self.repo_dir / clean_rel).resolve()
        if not full_path.is_relative_to(self.repo_dir.resolve()) or not full_path.is_file():
            return f"File not found: {clean_rel}"
        lines = full_path.read_text(encoding="utf-8", errors="replace").splitlines()
        s_idx = max(1, start_line)
        e_idx = min(len(lines), max(s_idx, end_line))
        numbered = [f"{ln_no:4d} | {lines[ln_no - 1]}" for ln_no in range(s_idx, e_idx + 1)]
        return "\n".join(numbered)[:MAX_TOOL_OUTPUT_CHARS]


class SharedExecutionBackend:
    """Unified execution backend supporting live Gemini API or local MLX Gemma 4 26B."""

    def __init__(
        self,
        prefer_gemini_api: bool = True,
        gemini_model_id: str = "gemini-3.8-flash",
        mlx_model_id: str = "mlx-community/gemma-4-26b-a4b-it-4bit",
    ) -> None:
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.use_gemini_api = bool(prefer_gemini_api and self.gemini_api_key)
        self.gemini_model_id = gemini_model_id
        self.mlx_model_id = mlx_model_id
        self.cumulative_cost_usd: float = 0.0
        self._mlx_model: Any = None
        self._mlx_tokenizer: Any = None

    @property
    def active_model_id(self) -> str:
        return self.gemini_model_id if self.use_gemini_api else self.mlx_model_id

    @property
    def accounting_source(
        self,
    ) -> Literal["google_genai_usage_metadata", "mlx_tokenizer_exact_bpe"]:
        return "google_genai_usage_metadata" if self.use_gemini_api else "mlx_tokenizer_exact_bpe"

    def _ensure_mlx_loaded(self) -> tuple[Any, Any]:
        if self._mlx_model is None or self._mlx_tokenizer is None:
            import mlx_lm

            loaded: Any = mlx_lm.load(self.mlx_model_id)
            self._mlx_model, self._mlx_tokenizer = loaded[0], loaded[1]
        return self._mlx_model, self._mlx_tokenizer

    def generate(
        self,
        prompt: str,
        max_output_tokens: int = 128,
    ) -> tuple[str, int, int, float]:
        """Execute LLM generation and return (text, in_tokens, out_tokens, latency_ms)."""
        if self.cumulative_cost_usd >= MAX_BENCHMARK_BUDGET_USD:
            raise RuntimeError(
                f"Benchmark circuit breaker triggered: cumulative cost "
                f"${self.cumulative_cost_usd:.4f} >= ${MAX_BENCHMARK_BUDGET_USD:.2f}"
            )

        t0 = time.perf_counter()
        if self.use_gemini_api:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.gemini_api_key)
            resp = client.models.generate_content(
                model=self.gemini_model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=max_output_tokens,
                ),
            )
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            text = str(resp.text or "")
            usage = getattr(resp, "usage_metadata", None)
            in_tok = int(getattr(usage, "prompt_token_count", 0) or 0)
            out_tok = int(getattr(usage, "candidates_token_count", 0) or 0)
            self.cumulative_cost_usd += compute_frontier_cost_usd(in_tok, out_tok)
            return text, in_tok, out_tok, latency_ms

        import mlx_lm

        from context_scout.tuning.train_mlx_lora import GPUMutualExclusionLock

        model, tokenizer = self._ensure_mlx_loaded()
        try:
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )

        in_tok = len(tokenizer.encode(formatted))
        with GPUMutualExclusionLock(timeout_seconds=120.0):
            raw_out = str(
                mlx_lm.generate(
                    model,
                    tokenizer,
                    prompt=formatted,
                    max_tokens=max_output_tokens,
                    verbose=False,
                )
            )
        clean_out = raw_out.replace("<eos>", "").replace("<end_of_turn>", "").strip()
        out_tok = max(1, len(tokenizer.encode(clean_out)))
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return clean_out, in_tok, out_tok, latency_ms


def _extract_unified_diff(raw_text: str) -> str:
    """Extract a clean unified diff block from LLM output, stripping 'NNNN | ' slice prefixes."""
    fenced = re.findall(r"```(?:diff|patch)?\s*\n(.*?)```", raw_text, flags=re.DOTALL)
    candidates = [*fenced, raw_text]
    for block in candidates:
        lines = block.splitlines()
        start_idx: int | None = None
        for idx, ln in enumerate(lines):
            if ln.startswith("--- "):
                start_idx = idx
                break
        if start_idx is None:
            continue
        cleaned_lines: list[str] = []
        has_plus_header = False
        has_hunk = False
        has_mod_line = False
        for ln in lines[start_idx:]:
            if ln.startswith("+++ "):
                has_plus_header = True
                cleaned_lines.append(ln)
            elif ln.startswith("--- ") or ln.startswith("@@"):
                if ln.startswith("@@"):
                    has_hunk = True
                cleaned_lines.append(ln)
            elif ln.startswith(("+", "-", " ")):
                prefix_char = ln[0]
                rest = re.sub(r"^\s*\d+\s*\|\s?", "", ln[1:])
                if prefix_char in ("+", "-") and rest.strip():
                    has_mod_line = True
                cleaned_lines.append(f"{prefix_char}{rest}")
            elif ln.startswith("diff --git "):
                cleaned_lines.append(ln)
        if has_plus_header and has_hunk and has_mod_line:
            return "\n".join(cleaned_lines) + "\n"
    return ""


class UnseenRepoABBenchmarkRunner:
    """Runs the 30-task two-arm unseen-repository A/B benchmark."""

    def __init__(
        self,
        repos_base_dir: Path,
        repo_root: Path = Path("."),
        prefer_gemini_api: bool = True,
        verify_postgres_rls: bool = True,
        run_live_frozen_qlora: bool = True,
    ) -> None:
        self.repos_base_dir = repos_base_dir
        self.repo_root = repo_root
        self.suite: UnseenRepoBenchmarkSuite = build_unseen_repo_suite()
        self.backend = SharedExecutionBackend(prefer_gemini_api=prefer_gemini_api)
        self.verify_postgres_rls = verify_postgres_rls
        self.run_live_frozen_qlora = run_live_frozen_qlora
        self.indexer = GitMerkleIndexer()
        self.registry = DomainRegistry(domains_dir=self.repo_root / "domains")
        self.registry.load_all()
        self.gemini_prompt_builder = GeminiImplementationExecutor()

    def _verify_postgres_17_rls_sync(self, sample_task: UnseenRepoBugTask, repo_dir: Path) -> bool:
        """Verify live PostgreSQL 17 FORCE RLS indexing on unseen repository."""
        if not self.verify_postgres_rls:
            return False
        try:
            settings = ScoutSettings()
            conn_mgr = RLSConnectionManager(settings.db_app_url)
            pg_adapter = PostgresScoutGraphAdapter(conn_mgr, settings)
            principal = PrincipalContext(
                user_id="ab_bench_pg_verifier",
                role=RoleType.ENGINEER,
                allowed_scopes=["public_eng"],
            )
            unauth_principal = PrincipalContext(
                user_id="unauth_guest",
                role=RoleType.ENGINEER,
                allowed_scopes=["unprivileged_scope"],
            )
            subprocess.run(
                ["git", "-C", str(repo_dir), "checkout", "-f", sample_task.parent_commit_sha],
                check=True,
                capture_output=True,
            )
            pg_repo = PostgresRLSRepository()
            with scoped_db_session(conn_mgr, principal) as conn:
                self.indexer.sync_postgres(
                    repo_dir=repo_dir,
                    conn=conn,
                    repo=pg_repo,
                    principal=principal,
                    domain_id="trading_code_scout",
                    repo_name="httpx_pg_rls_probe",
                    allowed_scopes=["public_eng"],
                    ref=sample_task.parent_commit_sha,
                )
            vis_auth = pg_adapter.list_visible_code_files(principal, "trading_code_scout")
            vis_unauth = pg_adapter.list_visible_code_files(unauth_principal, "trading_code_scout")
            has_probe = any(f.repo_name == "httpx_pg_rls_probe" for f in vis_auth)
            leaked_probe = any(f.repo_name == "httpx_pg_rls_probe" for f in vis_unauth)
            with scoped_db_session(conn_mgr, principal) as conn:
                conn.execute(
                    "DELETE FROM code_files WHERE domain_id = %s AND repo_name = %s;",
                    ("trading_code_scout", "httpx_pg_rls_probe"),
                )
            return bool(has_probe and not leaked_probe)
        except Exception as exc:
            logger.warning("PostgreSQL 17 RLS probe skipped/failed: %s", exc)
            return False

    def _extract_search_patterns_from_problem(self, problem_statement: str) -> list[str]:
        """Extract prioritized search patterns from the issue description for tool calls."""
        without_fences = re.sub(r"```.*?```", " ", problem_statement, flags=re.DOTALL)
        backticks = [bt.strip(" `'\";:(),") for bt in re.findall(r"`([^`\n]+)`", without_fences)]
        patterns: list[str] = []
        for bt in backticks:
            clean = bt[:-2] if bt.endswith("()") else bt
            if clean and not any(c.isspace() for c in clean) and len(clean) >= 3:
                if "." in clean:
                    patterns.append(clean.split(".")[-1])
                else:
                    patterns.append(clean)
        for tok in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{3,}\b", without_fences):
            if ("_" in tok or any(c.isupper() for c in tok[1:])) and tok not in patterns:
                patterns.append(tok)
        first_line = problem_statement.splitlines()[0] if problem_statement else ""
        for word in re.findall(r"\b[A-Za-z]{4,}\b", first_line):
            w_l = word.lower()
            if (
                w_l not in {"this", "that", "with", "from", "when", "make", "always", "move"}
                and word not in patterns
            ):
                patterns.append(word)
        return patterns[:4] or ["def "]

    def run_arm_a_direct_exploration(
        self,
        task: UnseenRepoBugTask,
        repo_dir: Path,
        source_subdir: str,
    ) -> ArmExecutionResult:
        """Execute Arm A: Multi-turn direct repository exploration with bounded tools."""
        subprocess.run(
            ["git", "-C", str(repo_dir), "checkout", "-f", task.parent_commit_sha],
            check=True,
            capture_output=True,
        )
        tools = RepoExplorationTools(repo_dir=repo_dir, source_subdir=source_subdir)
        traces: list[ToolCallTrace] = []
        history_transcript: list[str] = [
            f"Repository: {task.repo_slug} @ {task.parent_commit_sha}\n"
            f"Issue / Bug Report:\n{task.problem_statement}\n"
        ]

        t_start = time.perf_counter()

        # Turn 1: List repository files (Live LLM generation)
        file_tree_out = tools.list_repo_files()
        history_transcript.append(f"[Tool 1: list_repo_files]\n{file_tree_out}\n")
        turn1_prompt = (
            "\n".join(history_transcript)
            + "\nSelect the most relevant file and grep pattern to inspect first. "
            'Reply with JSON: {"grep_pattern": "...", "candidate_file": "..."}'
        )
        resp1, in1, out1, lat1 = self.backend.generate(turn1_prompt, max_output_tokens=48)
        traces.append(
            ToolCallTrace(
                turn_index=1,
                tool_name="list_repo_files",
                tool_args={"source_subdir": source_subdir},
                tool_output_chars=len(file_tree_out),
                prompt_tokens=in1,
                completion_tokens=out1,
                latency_ms=lat1,
            )
        )
        history_transcript.append(f"[Assistant Turn 1]\n{resp1}\n")

        # Extract LLM's chosen grep_pattern from Turn 1 response
        llm_grep_pat = ""
        m_grep = re.search(r'"grep_pattern"\s*:\s*"([^"]+)"', resp1)
        if m_grep:
            llm_grep_pat = m_grep.group(1).strip()
        fallback_patterns = self._extract_search_patterns_from_problem(task.problem_statement)
        fallback_grep = "|".join(re.escape(p) for p in fallback_patterns[:3])
        grep_pat = llm_grep_pat if llm_grep_pat else fallback_grep

        # Turn 2: Grep repository using the model's pattern (with fallback if empty match)
        grep_out = tools.grep_repo(grep_pat)
        if grep_out.startswith("No matches for pattern:") and fallback_grep != grep_pat:
            grep_pat = f"{llm_grep_pat}|{fallback_grep}" if llm_grep_pat else fallback_grep
            grep_out = tools.grep_repo(grep_pat)

        history_transcript.append(f"[Tool 2: grep_repo pattern={grep_pat!r}]\n{grep_out}\n")
        turn2_prompt = (
            "\n".join(history_transcript)
            + "\nBased on the grep matches, state which file paths and line numbers to read next. "
            'Reply with JSON: {"file_path": "...", "line": 1}'
        )
        resp2, in2, out2, lat2 = self.backend.generate(turn2_prompt, max_output_tokens=48)
        traces.append(
            ToolCallTrace(
                turn_index=2,
                tool_name="grep_repo",
                tool_args={"pattern": grep_pat},
                tool_output_chars=len(grep_out),
                prompt_tokens=in2,
                completion_tokens=out2,
                latency_ms=lat2,
            )
        )
        history_transcript.append(f"[Assistant Turn 2]\n{resp2}\n")

        grep_file_hits: list[tuple[str, int]] = []
        seen_file_windows: set[tuple[str, int]] = set()
        # First honor explicit file paths chosen by the LLM in Turn 1 / Turn 2
        for m_f in re.findall(r"(?:httpx|src/click)/[A-Za-z0-9_/\.]+\.py", resp2 + "\n" + resp1):
            ln_match = re.search(r'"line"\s*:\s*(\d+)', resp2)
            chosen_ln = int(ln_match.group(1)) if ln_match else 60
            win_bucket = (m_f, (chosen_ln // 120) * 120)
            if win_bucket not in seen_file_windows:
                seen_file_windows.add(win_bucket)
                grep_file_hits.append((m_f, chosen_ln))

        for line in grep_out.splitlines():
            m = re.match(r"^([^:]+\.py):(\d+):", line)
            if m:
                f_p = m.group(1)
                ln_no = int(m.group(2))
                win_bucket = (f_p, (ln_no // 120) * 120)
                if win_bucket not in seen_file_windows:
                    seen_file_windows.add(win_bucket)
                    grep_file_hits.append((f_p, ln_no))

        # Turns 3..5: Read candidate file windows with live LLM generation at every turn
        inspected_files_ordered: list[str] = []
        per_file_segments: list[tuple[str, str]] = []
        for idx_read, (f_path, center_line) in enumerate(grep_file_hits[:3], start=3):
            s_ln = max(1, center_line - 60)
            e_ln = center_line + 100
            read_out = tools.read_file_lines(f_path, s_ln, e_ln)
            if f_path not in inspected_files_ordered:
                inspected_files_ordered.append(f_path)
            history_transcript.append(
                f"[Tool {idx_read}: read_file_lines {f_path}:{s_ln}-{e_ln}]\n{read_out}\n"
            )
            read_prompt = (
                "\n".join(history_transcript)
                + "\nIdentify the exact qualified symbol(s) in this file relevant to the bug."
            )
            resp_r, in_r, out_r, lat_r = self.backend.generate(read_prompt, max_output_tokens=32)
            traces.append(
                ToolCallTrace(
                    turn_index=idx_read,
                    tool_name="read_file_lines",
                    tool_args={"file_path": f_path, "start_line": s_ln, "end_line": e_ln},
                    tool_output_chars=len(read_out),
                    prompt_tokens=in_r,
                    completion_tokens=out_r,
                    latency_ms=lat_r,
                )
            )
            history_transcript.append(f"[Assistant Turn {idx_read}]\n{resp_r}\n")
            per_file_segments.append((f_path, f"{resp_r}\n{read_out}"))

        # Final Turn: Synthesize localization + unified diff patch (Live LLM generation)
        final_turn_idx = len(traces) + 1
        final_max_tokens = 1024 if task.verify_with_pytest else 96
        final_prompt = (
            "\n".join(history_transcript)
            + "\nOutput ONLY the code patch FIRST at the very top of your reply (no prose before "
            "the patch) as either a git-apply unified diff enclosed in ```diff ... ``` "
            "(using `--- a/<file_path>` and `+++ b/<file_path>` without `NNNN | ` line prefixes) "
            "OR a `### File: <file_path>\\n<<<<<<< SEARCH\\n<exact existing lines>\\n=======\\n"
            "<replacement lines>\\n>>>>>>> REPLACE` block (without `NNNN | ` line prefixes), "
            "followed by the modified file path and qualified symbol name."
        )
        final_resp, in_f, out_f, lat_f = self.backend.generate(
            final_prompt, max_output_tokens=final_max_tokens
        )
        traces.append(
            ToolCallTrace(
                turn_index=final_turn_idx,
                tool_name="submit_localization",
                tool_args={"inspected_files": inspected_files_ordered[:5]},
                tool_output_chars=len(final_resp),
                prompt_tokens=in_f,
                completion_tokens=out_f,
                latency_ms=lat_f,
            )
        )

        total_wall_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
        total_in = sum(t.prompt_tokens for t in traces)
        total_out = sum(t.completion_tokens for t in traces)
        total_tok = total_in + total_out

        localized_syms: list[str] = []
        for f_p, seg_text in per_file_segments:
            mod_prefix = f_p.replace("src/", "").replace(".py", "").replace("/", ".")
            file_syms: list[str] = []
            for fn_m in re.finditer(r"(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", seg_text):
                sym_cand = f"{mod_prefix}.{fn_m.group(1)}"
                if sym_cand not in file_syms and sym_cand not in localized_syms:
                    file_syms.append(sym_cand)
            localized_syms.extend(file_syms[:2])

        f_hit = check_file_recall(inspected_files_ordered, task.ground_truth_files)
        s_hit = check_symbol_recall(localized_syms, task.ground_truth_symbols)

        return ArmExecutionResult(
            arm="A_direct_exploration",
            task_id=task.task_id,
            repo_slug=task.repo_slug,
            execution_model_id=self.backend.active_model_id,
            token_accounting_source=self.backend.accounting_source,
            turns_used=len(traces),
            input_tokens=total_in,
            output_tokens=total_out,
            total_tokens=total_tok,
            estimated_frontier_cost_usd=compute_frontier_cost_usd(total_in, total_out),
            wall_clock_latency_ms=total_wall_ms,
            scout_indexing_and_preflight_ms=0.0,
            localized_files=inspected_files_ordered[:5],
            localized_symbols=localized_syms[:5],
            file_recall_at_5=f_hit,
            symbol_recall_at_5=s_hit,
            preflight_gate_passed=True,
            auto_attached_1hop_slices=0,
            generated_patch_diff=final_resp[:2000],
            tool_traces=traces,
            failure_category=None if f_hit else "grep_exploration_missed_target_file",
        )

    def _prepare_arm_b_scout_packs(
        self,
        active_tasks: tuple[UnseenRepoBugTask, ...],
        httpx_dir: Path,
        click_dir: Path,
        principal: PrincipalContext,
    ) -> list[
        tuple[
            ImplementationContextPack | InsufficientContextDiagnostic,
            float,
            bool | None,
            float,
        ]
    ]:
        """Phase 1: Index unseen repos, run ScoutWorkflow + live frozen QLoRA verification."""
        httpx_mem_repo = InMemoryRLSRepository()
        click_mem_repo = InMemoryRLSRepository()
        httpx_workflow = ScoutWorkflow(registry=self.registry, repo=httpx_mem_repo)
        click_workflow = ScoutWorkflow(registry=self.registry, repo=click_mem_repo)

        qlora_client = LocalGemmaClient(adapter_pool=self.registry.adapter_pool)
        qlora_client.set_active_domain("trading_code_scout")

        results: list[
            tuple[
                ImplementationContextPack | InsufficientContextDiagnostic,
                float,
                bool | None,
                float,
            ]
        ] = []

        for idx, task in enumerate(active_tasks, start=1):
            is_httpx = task.repo_slug == "encode/httpx"
            repo_dir = httpx_dir if is_httpx else click_dir
            repo_name = "httpx" if is_httpx else "click"
            mem_repo = httpx_mem_repo if is_httpx else click_mem_repo
            workflow = httpx_workflow if is_httpx else click_workflow

            t_scout = time.perf_counter()
            subprocess.run(
                ["git", "-C", str(repo_dir), "checkout", "-f", task.parent_commit_sha],
                check=True,
                capture_output=True,
            )
            self.indexer.sync_in_memory(
                repo_dir=repo_dir,
                repo=mem_repo,
                principal=principal,
                domain_id="trading_code_scout",
                repo_name=repo_name,
                ref=task.parent_commit_sha,
            )
            scout_res = workflow.run(
                user_prompt=task.problem_statement,
                principal=principal,
                domain_id="trading_code_scout",
            )
            scout_ms = round((time.perf_counter() - t_scout) * 1000.0, 2)

            qlora_valid: bool | None = None
            qlora_ms: float = 0.0
            if self.run_live_frozen_qlora and isinstance(scout_res, ImplementationContextPack):
                cand_lines = [
                    f"Retrieved AST Candidate: [repo={repo_name} | "
                    f"file={sl.file_path} | "
                    f"symbol={sl.qualified_symbol} | "
                    f"lines={sl.start_line}-{sl.end_line} | "
                    f"intent={sl.modification_intent.value}]"
                    for sl in scout_res.code_slices[:5]
                ]
                primary_slice = scout_res.code_slices[0]
                short_sym = primary_slice.qualified_symbol.split(".")[-1]
                qlora_prompt = (
                    f"Localize the AST code slice for `{short_sym}` in "
                    f"`{primary_slice.file_path}` to review implementation logic and performance.\n"
                    + "\n".join(cand_lines)
                )
                t_q0 = time.perf_counter()
                try:
                    from context_scout.tuning.teacher_distiller import (
                        repair_and_constrain_code_pointers_json,
                    )

                    raw_q = qlora_client._call_in_process_mlx(
                        qlora_prompt,
                        domain_id="trading_code_scout",
                        max_tokens=384,
                    )
                    qlora_ms = round((time.perf_counter() - t_q0) * 1000.0, 2)
                    validated_ptrs = repair_and_constrain_code_pointers_json(raw_q)
                    if len(validated_ptrs) > 0:
                        qlora_valid = True
                        # Promote the frozen QLoRA adapter's chosen primary slice to index 0
                        chosen_sym = validated_ptrs[0].qualified_symbol
                        chosen_file = validated_ptrs[0].file_path
                        matched_idx = next(
                            (
                                i
                                for i, s in enumerate(scout_res.code_slices)
                                if s.qualified_symbol == chosen_sym or s.file_path == chosen_file
                            ),
                            0,
                        )
                        if matched_idx > 0:
                            reordered = list(scout_res.code_slices)
                            promoted = reordered.pop(matched_idx)
                            reordered.insert(0, promoted)
                            scout_res = scout_res.model_copy(update={"code_slices": reordered})
                    else:
                        qlora_valid = False
                except Exception as exc:
                    qlora_ms = round((time.perf_counter() - t_q0) * 1000.0, 2)
                    logger.warning("Frozen QLoRA validation failed on %s: %s", task.task_id, exc)
                    qlora_valid = False

            logger.info(
                "[Phase 1 Scout %d/%d] %s: scout_ms=%.1fms, qlora_valid=%s (%.1fms)",
                idx,
                len(active_tasks),
                task.task_id,
                scout_ms,
                qlora_valid,
                qlora_ms,
            )
            results.append((scout_res, scout_ms, qlora_valid, qlora_ms))

        # Release frozen QLoRA adapter model from Metal memory before Phase 2 execution
        qlora_client._mlx_cache.clear()
        gc.collect()
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:
            pass

        return results

    def run_arm_b_from_prepared_pack(
        self,
        task: UnseenRepoBugTask,
        scout_res: ImplementationContextPack | InsufficientContextDiagnostic,
        scout_ms: float,
        qlora_valid: bool | None,
        qlora_ms: float,
    ) -> ArmExecutionResult:
        """Execute Stage 4 (1-shot execution) on the prepared ImplementationContextPack."""
        if not isinstance(scout_res, ImplementationContextPack):
            return ArmExecutionResult(
                arm="B_context_scout",
                task_id=task.task_id,
                repo_slug=task.repo_slug,
                execution_model_id=self.backend.active_model_id,
                token_accounting_source=self.backend.accounting_source,
                turns_used=1,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                estimated_frontier_cost_usd=0.0,
                wall_clock_latency_ms=scout_ms + qlora_ms,
                scout_indexing_and_preflight_ms=scout_ms,
                frozen_qlora_json_valid=qlora_valid,
                frozen_qlora_latency_ms=qlora_ms,
                localized_files=scout_res.partial_candidate_files[:5],
                localized_symbols=[],
                file_recall_at_5=False,
                symbol_recall_at_5=False,
                preflight_gate_passed=False,
                auto_attached_1hop_slices=0,
                generated_patch_diff="",
                tool_traces=[],
                failure_category=f"preflight_gate_blocked: {scout_res.reason[:120]}",
            )

        one_shot_prompt = (
            self.gemini_prompt_builder._build_gemini_prompt(scout_res)
            + "\nIMPORTANT: Output the code patch FIRST at the very top of your reply (before any "
            "prose explanation). Do not include 'NNNN | ' line-number prefixes inside the diff "
            "body; use standard '--- a/<file_path>' and '+++ b/<file_path>' headers, OR emit "
            "a minimal '### File: <file_path>\\n<<<<<<< SEARCH\\n<exact existing lines>\\n"
            "=======\\n<replacement lines>\\n>>>>>>> REPLACE' block without line-number prefixes."
        )
        arm_b_max_tokens = 1024 if task.verify_with_pytest else 96
        patch_resp, in_tok, out_tok, gen_lat_ms = self.backend.generate(
            one_shot_prompt, max_output_tokens=arm_b_max_tokens
        )
        total_wall_ms = round(scout_ms + qlora_ms + gen_lat_ms, 2)

        localized_files = list(dict.fromkeys(s.file_path for s in scout_res.code_slices))
        localized_symbols = list(dict.fromkeys(s.qualified_symbol for s in scout_res.code_slices))
        auto_1hop_cnt = sum(1 for s in scout_res.code_slices if s.auto_expanded_by_ast)

        f_hit = check_file_recall(localized_files, task.ground_truth_files)
        s_hit = check_symbol_recall(localized_symbols, task.ground_truth_symbols)

        fail_cat: str | None = None
        if not f_hit:
            fail_cat = "ast_lexical_retrieval_missed_target_file"
        elif not s_hit:
            fail_cat = "file_localized_but_target_symbol_outside_top5"

        trace = ToolCallTrace(
            turn_index=1,
            tool_name="one_shot_context_pack_execution",
            tool_args={
                "slice_count": len(scout_res.code_slices),
                "auto_expanded_1hop_slices": auto_1hop_cnt,
            },
            tool_output_chars=len(patch_resp),
            prompt_tokens=in_tok,
            completion_tokens=out_tok,
            latency_ms=gen_lat_ms,
        )

        return ArmExecutionResult(
            arm="B_context_scout",
            task_id=task.task_id,
            repo_slug=task.repo_slug,
            execution_model_id=self.backend.active_model_id,
            token_accounting_source=self.backend.accounting_source,
            turns_used=1,
            input_tokens=in_tok,
            output_tokens=out_tok,
            total_tokens=in_tok + out_tok,
            estimated_frontier_cost_usd=compute_frontier_cost_usd(in_tok, out_tok),
            wall_clock_latency_ms=total_wall_ms,
            scout_indexing_and_preflight_ms=scout_ms,
            frozen_qlora_json_valid=qlora_valid,
            frozen_qlora_latency_ms=qlora_ms,
            localized_files=localized_files[:5],
            localized_symbols=localized_symbols[:5],
            file_recall_at_5=f_hit,
            symbol_recall_at_5=s_hit,
            preflight_gate_passed=True,
            auto_attached_1hop_slices=auto_1hop_cnt,
            generated_patch_diff=patch_resp[:2000],
            tool_traces=[trace],
            failure_category=fail_cat,
        )

    def verify_task_in_worktree_pytest(
        self,
        task: UnseenRepoBugTask,
        repo_dir: Path,
        arm_result: ArmExecutionResult,
    ) -> tuple[bool, str]:
        """Verify LLM-generated patch on parent_commit_sha in an isolated Git worktree."""
        if not (arm_result.file_recall_at_5 and arm_result.symbol_recall_at_5):
            return (
                False,
                "Skipped patch execution: target file/symbol not localized in Top-5",
            )

        clean_diff = _extract_unified_diff(arm_result.generated_patch_diff)
        has_search_replace = (
            "<<<<<<< SEARCH" in arm_result.generated_patch_diff
            and ">>>>>>> REPLACE" in arm_result.generated_patch_diff
        )
        has_hunk_or_diff = bool(
            re.search(r"(?m)^(?:@@\s|[-+][^\-+]|```diff)", arm_result.generated_patch_diff)
        )
        if not clean_diff and not has_search_replace and not has_hunk_or_diff:
            return (
                False,
                "No valid unified diff (--- a/... +++ b/... @@ ...) emitted by model",
            )

        worktree_dir = Path(tempfile.mkdtemp(prefix=f"scout_ab_wt_{task.task_id}_"))
        try:
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo_dir),
                    "worktree",
                    "add",
                    "--detach",
                    str(worktree_dir),
                    task.parent_commit_sha,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            # Apply the PR's reproduction test diff (SWE-bench FAIL_TO_PASS protocol)
            added_test_names: list[str] = []
            if task.test_files:
                test_diff_proc = subprocess.run(
                    [
                        "git",
                        "-C",
                        str(repo_dir),
                        "diff",
                        task.parent_commit_sha,
                        task.merge_commit_sha,
                        "--",
                        *list(task.test_files),
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if test_diff_proc.returncode == 0 and test_diff_proc.stdout.strip():
                    for m_fn in re.finditer(
                        r"^\+\s*(?:async\s+)?def\s+(test_[A-Za-z0-9_]+)\s*\(",
                        test_diff_proc.stdout,
                        flags=re.MULTILINE,
                    ):
                        fn_name = m_fn.group(1)
                        if fn_name not in added_test_names:
                            added_test_names.append(fn_name)
                    test_patch_path = worktree_dir / ".scout_pr_tests.patch"
                    test_patch_path.write_text(test_diff_proc.stdout, encoding="utf-8")
                    subprocess.run(
                        [
                            "git",
                            "-C",
                            str(worktree_dir),
                            "apply",
                            "--ignore-space-change",
                            "--ignore-whitespace",
                            str(test_patch_path),
                        ],
                        capture_output=True,
                        text=True,
                        check=False,
                    )

            apply_proc_ok = False
            apply_err_text = ""
            if clean_diff:
                patch_path = worktree_dir / ".scout_candidate.patch"
                patch_path.write_text(clean_diff, encoding="utf-8")

                apply_proc = subprocess.run(
                    [
                        "git",
                        "-C",
                        str(worktree_dir),
                        "apply",
                        "--recount",
                        "--ignore-space-change",
                        "--ignore-whitespace",
                        str(patch_path),
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                apply_proc_ok = apply_proc.returncode == 0
                apply_err_text = apply_proc.stderr.strip() or apply_proc.stdout.strip()

            if not apply_proc_ok:
                from context_scout.agents.gemini_executor import (
                    apply_search_replace_or_hunk_blocks,
                )

                sr_ok, sr_msg = apply_search_replace_or_hunk_blocks(
                    arm_result.generated_patch_diff,
                    worktree_dir,
                    default_files=arm_result.localized_files,
                )
                apply_proc_ok = sr_ok
                if not sr_ok:
                    apply_err_text = (apply_err_text or sr_msg)[:160]

            env = {
                k: v
                for k, v in os.environ.items()
                if not any(
                    sec in k.upper()
                    for sec in (
                        "API_KEY",
                        "SECRET",
                        "TOKEN",
                        "PASSWORD",
                        "ADMIN_URL",
                        "CREDENTIAL",
                    )
                )
            }
            src_path = (
                str(worktree_dir / "src") if (worktree_dir / "src").exists() else str(worktree_dir)
            )
            env["PYTHONPATH"] = f"{src_path}:{env.get('PYTHONPATH', '')}"

            cmd = [
                sys.executable,
                "-m",
                "pytest",
                "-o",
                "filterwarnings=",
                *list(task.test_files),
                "-q",
                "--tb=short",
            ]
            if added_test_names:
                cmd.extend(["-k", " or ".join(added_test_names)])

            def _run_worktree_pytest() -> subprocess.CompletedProcess[str]:
                p = subprocess.run(
                    cmd,
                    cwd=worktree_dir,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                if p.returncode in (2, 4) and (
                    "conftest" in (p.stdout + p.stderr)
                    or "ImportError" in (p.stdout + p.stderr)
                    or "ModuleNotFoundError" in (p.stdout + p.stderr)
                ):
                    p = subprocess.run(
                        [*cmd, "--noconftest"],
                        cwd=worktree_dir,
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=30,
                        check=False,
                    )
                return p

            proc = _run_worktree_pytest() if apply_proc_ok else None

            # Bounded 1-turn worktree pytest-stderr repair loop if initial patch failed
            if (proc is None or proc.returncode != 0) and arm_result.localized_files:
                from context_scout.agents.gemini_executor import (
                    apply_search_replace_or_hunk_blocks,
                )

                subprocess.run(
                    [
                        "git",
                        "-C",
                        str(worktree_dir),
                        "checkout",
                        "-f",
                        "--",
                        *list(task.ground_truth_files),
                    ],
                    capture_output=True,
                    check=False,
                )
                target_f_rel = arm_result.localized_files[0]
                target_f_path = worktree_dir / target_f_rel
                file_excerpt = ""
                if target_f_path.is_file():
                    f_lines = target_f_path.read_text(
                        encoding="utf-8", errors="replace"
                    ).splitlines()
                    sym_short = (
                        arm_result.localized_symbols[0].split(".")[-1]
                        if arm_result.localized_symbols
                        else ""
                    )
                    center_idx = next(
                        (
                            i
                            for i, ln in enumerate(f_lines)
                            if (sym_short and f"def {sym_short}" in ln)
                            or f"class {sym_short}" in ln
                        ),
                        0,
                    )
                    s_idx = max(0, center_idx - 15)
                    e_idx = min(len(f_lines), center_idx + 95)
                    file_excerpt = "\n".join(f_lines[s_idx:e_idx])
                pytest_diag = (
                    (proc.stdout + "\n" + proc.stderr)[-1200:]
                    if proc is not None
                    else f"Patch failed to apply: {apply_err_text}"
                )
                repair_prompt = (
                    f"Bug Report:\n{task.problem_statement}\n\n"
                    f"Target File ({target_f_rel}):\n```python\n{file_excerpt}\n```\n\n"
                    f"Previous Patch / Test Failure Output:\n```\n{pytest_diag}\n```\n\n"
                    "Output ONLY a corrected `<<<<<<< SEARCH\\n<exact existing lines>"
                    "\\n=======\\n<replacement lines>\\n>>>>>>> REPLACE` block "
                    "at the very top of your response that makes the failing pytest test pass."
                )
                try:
                    rep_text, rep_in, rep_out, rep_lat = self.backend.generate(
                        repair_prompt, max_output_tokens=768
                    )
                    arm_result.input_tokens += rep_in
                    arm_result.output_tokens += rep_out
                    arm_result.total_tokens = arm_result.input_tokens + arm_result.output_tokens
                    arm_result.estimated_frontier_cost_usd = compute_frontier_cost_usd(
                        arm_result.input_tokens, arm_result.output_tokens
                    )
                    arm_result.wall_clock_latency_ms = round(
                        arm_result.wall_clock_latency_ms + rep_lat, 2
                    )
                    rep_diff = _extract_unified_diff(rep_text)
                    rep_applied = False
                    if rep_diff:
                        rep_patch_path = worktree_dir / ".scout_repair.patch"
                        rep_patch_path.write_text(rep_diff, encoding="utf-8")
                        rep_ap = subprocess.run(
                            [
                                "git",
                                "-C",
                                str(worktree_dir),
                                "apply",
                                "--recount",
                                "--ignore-space-change",
                                "--ignore-whitespace",
                                str(rep_patch_path),
                            ],
                            capture_output=True,
                            text=True,
                            check=False,
                        )
                        rep_applied = rep_ap.returncode == 0
                    if not rep_applied:
                        rep_applied, _ = apply_search_replace_or_hunk_blocks(
                            rep_text,
                            worktree_dir,
                            default_files=arm_result.localized_files,
                        )
                    if rep_applied:
                        arm_result.generated_patch_diff = rep_text[:2000]
                        proc = _run_worktree_pytest()
                except Exception as rep_exc:
                    logger.warning(
                        "Worktree 1-turn repair skipped on %s: %s", task.task_id, rep_exc
                    )

            if proc is None:
                short_sha = task.parent_commit_sha[:8]
                return (
                    False,
                    f"git apply failed on parent_commit_sha ({short_sha}): {apply_err_text}",
                )
            summary = (proc.stdout.strip() or proc.stderr.strip()).splitlines()[-1:]
            summary_str = summary[0] if summary else f"exit_code={proc.returncode}"
            return (proc.returncode == 0, summary_str)
        except Exception as exc:
            return (False, f"Worktree pytest error: {exc}")
        finally:
            subprocess.run(
                ["git", "-C", str(repo_dir), "worktree", "remove", "--force", str(worktree_dir)],
                check=False,
                capture_output=True,
            )
            shutil.rmtree(worktree_dir, ignore_errors=True)

    def run_benchmark(
        self,
        tasks: tuple[UnseenRepoBugTask, ...] | None = None,
        traces_jsonl_path: Path | None = None,
    ) -> UnseenRepoABReport:
        """Run the full 30-task two-arm unseen-repo benchmark and return the report."""
        integrity = verify_frozen_adapter_integrity(self.repo_root)
        active_tasks = tasks or self.suite.tasks

        httpx_dir = self.repos_base_dir / "httpx"
        click_dir = self.repos_base_dir / "click"
        pg_verified = self._verify_postgres_17_rls_sync(active_tasks[0], httpx_dir)

        principal = PrincipalContext(
            user_id="unseen_ab_benchmark",
            role=RoleType.ENGINEER,
            allowed_scopes=["public_eng", "repo:trading_strategies"],
        )

        # Phase 1: Prepare all Arm B ContextPacks + verify frozen QLoRA adapter
        prepared_packs = self._prepare_arm_b_scout_packs(
            active_tasks, httpx_dir, click_dir, principal
        )

        comparisons: list[TaskABComparison] = []
        honest_losses: list[dict[str, Any]] = []

        if traces_jsonl_path is not None:
            traces_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            traces_jsonl_path.write_text("", encoding="utf-8")

        # Phase 2: Execute Arm A and Arm B with the shared execution model
        for idx, (task, (scout_res, scout_ms, qlora_valid, qlora_ms)) in enumerate(
            zip(active_tasks, prepared_packs, strict=True), start=1
        ):
            is_httpx = task.repo_slug == "encode/httpx"
            repo_dir = httpx_dir if is_httpx else click_dir
            source_subdir = "httpx" if is_httpx else "src/click"

            logger.info(
                "[Phase 2 Execution %d/%d] Running Arm A & Arm B on %s (%s #%d)...",
                idx,
                len(active_tasks),
                task.task_id,
                task.repo_slug,
                task.pr_number,
            )

            arm_a = self.run_arm_a_direct_exploration(task, repo_dir, source_subdir)
            arm_b = self.run_arm_b_from_prepared_pack(
                task, scout_res, scout_ms, qlora_valid, qlora_ms
            )

            if task.verify_with_pytest:
                a_pass, a_sum = self.verify_task_in_worktree_pytest(task, repo_dir, arm_a)
                b_pass, b_sum = self.verify_task_in_worktree_pytest(task, repo_dir, arm_b)
                arm_a.pytest_verified = a_pass
                arm_a.pytest_output_summary = a_sum
                arm_b.pytest_verified = b_pass
                arm_b.pytest_output_summary = b_sum

            in_savings = (
                round(((arm_a.input_tokens - arm_b.input_tokens) / arm_a.input_tokens) * 100.0, 2)
                if arm_a.input_tokens > 0
                else 0.0
            )
            tot_savings = (
                round(((arm_a.total_tokens - arm_b.total_tokens) / arm_a.total_tokens) * 100.0, 2)
                if arm_a.total_tokens > 0
                else 0.0
            )

            if arm_a.file_recall_at_5 and arm_b.file_recall_at_5:
                outcome: Literal["BOTH_HIT", "ARM_B_ONLY_HIT", "ARM_A_ONLY_HIT", "BOTH_MISSED"] = (
                    "BOTH_HIT"
                )
            elif arm_b.file_recall_at_5 and not arm_a.file_recall_at_5:
                outcome = "ARM_B_ONLY_HIT"
            elif arm_a.file_recall_at_5 and not arm_b.file_recall_at_5:
                outcome = "ARM_A_ONLY_HIT"
            else:
                outcome = "BOTH_MISSED"

            if not arm_b.file_recall_at_5 or not arm_b.symbol_recall_at_5:
                honest_losses.append(
                    {
                        "task_id": task.task_id,
                        "repo_slug": task.repo_slug,
                        "pr_number": task.pr_number,
                        "title": task.title,
                        "outcome_classification": outcome,
                        "arm_a_file_hit": arm_a.file_recall_at_5,
                        "arm_b_file_hit": arm_b.file_recall_at_5,
                        "arm_b_symbol_hit": arm_b.symbol_recall_at_5,
                        "ground_truth_files": list(task.ground_truth_files),
                        "ground_truth_symbols": list(task.ground_truth_symbols),
                        "arm_b_localized_files": arm_b.localized_files,
                        "arm_b_localized_symbols": arm_b.localized_symbols,
                        "failure_category": arm_b.failure_category,
                    }
                )

            comp = TaskABComparison(
                task_id=task.task_id,
                repo_slug=task.repo_slug,
                pr_number=task.pr_number,
                pr_url=task.pr_url,
                title=task.title,
                ground_truth_files=list(task.ground_truth_files),
                ground_truth_symbols=list(task.ground_truth_symbols),
                arm_a=arm_a,
                arm_b=arm_b,
                input_token_savings_pct=in_savings,
                total_token_savings_pct=tot_savings,
                outcome_classification=outcome,
            )
            comparisons.append(comp)

            if traces_jsonl_path is not None:
                with traces_jsonl_path.open("a", encoding="utf-8") as f_log:
                    f_log.write(comp.model_dump_json() + "\n")

        # Restore httpx and click repos to default branches after benchmark checkout loop
        subprocess.run(
            ["git", "-C", str(httpx_dir), "checkout", "-f", "master"],
            check=False,
            capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(click_dir), "checkout", "-f", "main"],
            check=False,
            capture_output=True,
        )

        n_tasks = len(comparisons)
        a_f_cnt = sum(1 for c in comparisons if c.arm_a.file_recall_at_5)
        b_f_cnt = sum(1 for c in comparisons if c.arm_b.file_recall_at_5)
        a_s_cnt = sum(1 for c in comparisons if c.arm_a.symbol_recall_at_5)
        b_s_cnt = sum(1 for c in comparisons if c.arm_b.symbol_recall_at_5)
        q_valid_cnt = sum(1 for c in comparisons if c.arm_b.frozen_qlora_json_valid is True)
        pytest_tasks = [c for c in comparisons if c.arm_b.pytest_verified is not None]
        a_py_cnt = sum(1 for c in pytest_tasks if c.arm_a.pytest_verified is True)
        b_py_cnt = sum(1 for c in pytest_tasks if c.arm_b.pytest_verified is True)

        return UnseenRepoABReport(
            benchmark_id=f"unseen_repo_ab_n{n_tasks}",
            generated_at_utc=datetime.datetime.now(datetime.UTC).isoformat(),
            sample_size_n=n_tasks,
            repos_evaluated=["encode/httpx", "pallets/click"],
            frozen_code_adapter_sha256=str(integrity["code_adapter_sha256"]),
            frozen_qlora_json_valid_count=q_valid_cnt,
            frozen_qlora_json_valid_pct=(
                round((q_valid_cnt / n_tasks) * 100.0, 2) if n_tasks else 0.0
            ),
            execution_model_id=self.backend.active_model_id,
            token_accounting_source=self.backend.accounting_source,
            postgres_rls_verified=pg_verified,
            arm_a_input_tokens=compute_distribution_stats(
                [float(c.arm_a.input_tokens) for c in comparisons]
            ),
            arm_b_input_tokens=compute_distribution_stats(
                [float(c.arm_b.input_tokens) for c in comparisons]
            ),
            arm_a_total_tokens=compute_distribution_stats(
                [float(c.arm_a.total_tokens) for c in comparisons]
            ),
            arm_b_total_tokens=compute_distribution_stats(
                [float(c.arm_b.total_tokens) for c in comparisons]
            ),
            input_token_savings_pct=compute_distribution_stats(
                [c.input_token_savings_pct for c in comparisons]
            ),
            total_token_savings_pct=compute_distribution_stats(
                [c.total_token_savings_pct for c in comparisons]
            ),
            arm_a_latency_ms=compute_distribution_stats(
                [c.arm_a.wall_clock_latency_ms for c in comparisons]
            ),
            arm_b_latency_ms=compute_distribution_stats(
                [c.arm_b.wall_clock_latency_ms for c in comparisons]
            ),
            arm_a_file_recall_at_5_count=a_f_cnt,
            arm_a_file_recall_at_5_pct=round((a_f_cnt / n_tasks) * 100.0, 2) if n_tasks else 0.0,
            arm_b_file_recall_at_5_count=b_f_cnt,
            arm_b_file_recall_at_5_pct=round((b_f_cnt / n_tasks) * 100.0, 2) if n_tasks else 0.0,
            arm_a_symbol_recall_at_5_count=a_s_cnt,
            arm_a_symbol_recall_at_5_pct=round((a_s_cnt / n_tasks) * 100.0, 2) if n_tasks else 0.0,
            arm_b_symbol_recall_at_5_count=b_s_cnt,
            arm_b_symbol_recall_at_5_pct=round((b_s_cnt / n_tasks) * 100.0, 2) if n_tasks else 0.0,
            arm_a_pytest_pass_count=a_py_cnt,
            arm_b_pytest_pass_count=b_py_cnt,
            pytest_subset_size=len(pytest_tasks),
            arm_a_total_cost_usd=round(
                sum(c.arm_a.estimated_frontier_cost_usd for c in comparisons), 6
            ),
            arm_b_total_cost_usd=round(
                sum(c.arm_b.estimated_frontier_cost_usd for c in comparisons), 6
            ),
            honest_losses_and_failures=honest_losses,
            task_comparisons=comparisons,
        )


def format_unseen_ab_markdown_report(report: UnseenRepoABReport) -> str:
    """Render a Markdown report from an UnseenRepoABReport."""
    a_in = report.arm_a_input_tokens
    b_in = report.arm_b_input_tokens
    a_tot = report.arm_a_total_tokens
    b_tot = report.arm_b_total_tokens
    sav_in = report.input_token_savings_pct
    sav_tot = report.total_token_savings_pct
    a_lat = report.arm_a_latency_ms
    b_lat = report.arm_b_latency_ms
    n = report.sample_size_n

    f_delta_pct = report.arm_b_file_recall_at_5_pct - report.arm_a_file_recall_at_5_pct
    f_delta_cnt = report.arm_b_file_recall_at_5_count - report.arm_a_file_recall_at_5_count
    s_delta_pct = report.arm_b_symbol_recall_at_5_pct - report.arm_a_symbol_recall_at_5_pct
    s_delta_cnt = report.arm_b_symbol_recall_at_5_count - report.arm_a_symbol_recall_at_5_count
    py_sub = report.pytest_subset_size
    py_delta = report.arm_b_pytest_pass_count - report.arm_a_pytest_pass_count
    lat_speedup = ((a_lat.median - b_lat.median) / a_lat.median * 100.0) if a_lat.median else 0.0
    cost_saved = report.arm_a_total_cost_usd - report.arm_b_total_cost_usd

    lines = [
        "# Unseen-Repository Two-Arm A/B Benchmark Report",
        "",
        f"- **Generated (UTC):** `{report.generated_at_utc}`",
        (
            f"- **Sample Size ($N$):** `{n}` real merged bug-fix PRs "
            "(`15` `encode/httpx`, `15` `pallets/click`)"
        ),
        (
            f"- **Frozen Code Adapter SHA-256:** `{report.frozen_code_adapter_sha256}` "
            "(zero retraining on unseen repos)"
        ),
        (
            "- **Frozen QLoRA Schema Validity on Unseen Repos:** "
            f"`{report.frozen_qlora_json_valid_count} / {n}` "
            f"(`{report.frozen_qlora_json_valid_pct:.2f}%`)"
        ),
        f"- **Execution Model (Both Arms):** `{report.execution_model_id}`",
        f"- **Token Accounting Method:** `{report.token_accounting_source}`",
        (
            "- **PostgreSQL 17 `FORCE RLS` Verified on Unseen Repo:** "
            f"`{report.postgres_rls_verified}`"
        ),
        "",
        "## 1. Executive Summary Table ($N=30$ Unseen PRs)",
        "",
        (
            "| Metric | Arm A: Direct Exploration (Multi-Turn `grep`/`read_file`) | "
            "Arm B: Enterprise Context Scout + 1-Shot Execution | Delta / Improvement |"
        ),
        "| :--- | :--- | :--- | :--- |",
        (
            f"| **Input Tokens (Median / IQR)** | "
            f"`{a_in.median:,.0f}` (IQR `{a_in.iqr_25:,.0f}`-`{a_in.iqr_75:,.0f}`, "
            f"mean `{a_in.mean:,.0f}`) | "
            f"`{b_in.median:,.0f}` (IQR `{b_in.iqr_25:,.0f}`-`{b_in.iqr_75:,.0f}`, "
            f"mean `{b_in.mean:,.0f}`) | "
            f"**`{sav_in.median:.2f}%` median savings** "
            f"(IQR `{sav_in.iqr_25:.2f}%`-`{sav_in.iqr_75:.2f}%`) |"
        ),
        (
            f"| **Total Tokens (Median / IQR)** | "
            f"`{a_tot.median:,.0f}` (IQR `{a_tot.iqr_25:,.0f}`-`{a_tot.iqr_75:,.0f}`, "
            f"mean `{a_tot.mean:,.0f}`) | "
            f"`{b_tot.median:,.0f}` (IQR `{b_tot.iqr_25:,.0f}`-`{b_tot.iqr_75:,.0f}`, "
            f"mean `{b_tot.mean:,.0f}`) | "
            f"**`{sav_tot.median:.2f}%` median savings** "
            f"(IQR `{sav_tot.iqr_25:.2f}%`-`{sav_tot.iqr_75:.2f}%`) |"
        ),
        (
            f"| **File `Recall@5`** | "
            f"`{report.arm_a_file_recall_at_5_count} / {n}` "
            f"(`{report.arm_a_file_recall_at_5_pct:.2f}%`) | "
            f"`{report.arm_b_file_recall_at_5_count} / {n}` "
            f"(`{report.arm_b_file_recall_at_5_pct:.2f}%`) | "
            f"`{f_delta_pct:+.2f}%` (`{f_delta_cnt:+d}` tasks) |"
        ),
        (
            f"| **Symbol `Recall@5`** | "
            f"`{report.arm_a_symbol_recall_at_5_count} / {n}` "
            f"(`{report.arm_a_symbol_recall_at_5_pct:.2f}%`) | "
            f"`{report.arm_b_symbol_recall_at_5_count} / {n}` "
            f"(`{report.arm_b_symbol_recall_at_5_pct:.2f}%`) | "
            f"`{s_delta_pct:+.2f}%` (`{s_delta_cnt:+d}` tasks) |"
        ),
        (
            "| **Frozen QLoRA `CodeSlicePointer` Schema Validity** | N/A (Direct Tool Loop) | "
            f"`{report.frozen_qlora_json_valid_count} / {n}` "
            f"(`{report.frozen_qlora_json_valid_pct:.2f}%`) | "
            "Zero retraining on unseen repos |"
        ),
        (
            f"| **Worktree `pytest` Verification ($n={py_sub}$)** | "
            f"`{report.arm_a_pytest_pass_count} / {py_sub}` | "
            f"`{report.arm_b_pytest_pass_count} / {py_sub}` | "
            f"`{py_delta:+d}` tasks |"
        ),
        (
            f"| **Wall-Clock Latency (Median / IQR)** | "
            f"`{a_lat.median:,.1f} ms` (IQR `{a_lat.iqr_25:,.1f}`-`{a_lat.iqr_75:,.1f} ms`) | "
            f"`{b_lat.median:,.1f} ms` (IQR `{b_lat.iqr_25:,.1f}`-`{b_lat.iqr_75:,.1f} ms`) | "
            f"`{lat_speedup:.1f}%` faster median |"
        ),
        (
            "| **Cumulative Cost at Gemini 3.8 Flash Rates (`$0.75`/`$3.75` per 1M)** | "
            f"`${report.arm_a_total_cost_usd:.4f}` | `${report.arm_b_total_cost_usd:.4f}` | "
            f"`${cost_saved:.4f}` saved across {n} tasks |"
        ),
        "",
        "## 2. Honest Losses & Failure Taxonomy (Where Arm B Missed File or Symbol `Recall@5`)",
        "",
    ]

    if not report.honest_losses_and_failures:
        lines.append("No File or Symbol Recall@5 misses recorded.")
    else:
        lines.extend(
            [
                (
                    "| Task ID | Repo & PR | Ground-Truth File / Symbol | "
                    "Arm A File Hit | Arm B File Hit | Arm B Symbol Hit | Failure Mode |"
                ),
                "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
            ]
        )
        for loss in report.honest_losses_and_failures:
            gt_f = ", ".join(f"`{x}`" for x in loss["ground_truth_files"])
            gt_s = ", ".join(f"`{x}`" for x in loss["ground_truth_symbols"][:2])
            lines.append(
                f"| `{loss['task_id']}` | `{loss['repo_slug']}#{loss['pr_number']}` | "
                f"{gt_f} ({gt_s}) | `{loss['arm_a_file_hit']}` | "
                f"`{loss['arm_b_file_hit']}` | `{loss['arm_b_symbol_hit']}` | "
                f"`{loss['failure_category']}` |"
            )

    lines.extend(
        [
            "",
            "## 3. Per-Task Breakdown ($N=30$)",
            "",
            (
                "| Task ID | Repo & PR | Arm A Tokens (In / Total) | "
                "Arm B Tokens (In / Total) | Input Savings | Arm A Recall (F / S) | "
                "Arm B Recall (F / S) | Worktree `pytest` (A / B) | Outcome |"
            ),
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
    )
    for c in report.task_comparisons:
        py_str = (
            f"`{c.arm_a.pytest_verified}` / `{c.arm_b.pytest_verified}`"
            if c.arm_b.pytest_verified is not None
            else "—"
        )
        lines.append(
            f"| `{c.task_id}` | [{c.repo_slug}#{c.pr_number}]({c.pr_url}) | "
            f"`{c.arm_a.input_tokens:,}` / `{c.arm_a.total_tokens:,}` | "
            f"`{c.arm_b.input_tokens:,}` / `{c.arm_b.total_tokens:,}` | "
            f"`{c.input_token_savings_pct:.1f}%` | "
            f"`{c.arm_a.file_recall_at_5}` / `{c.arm_a.symbol_recall_at_5}` | "
            f"`{c.arm_b.file_recall_at_5}` / `{c.arm_b.symbol_recall_at_5}` | "
            f"{py_str} | `{c.outcome_classification}` |"
        )

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    """Execute the 30-task Unseen-Repo A/B Benchmark and write JSON/MD/JSONL artifacts."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    repo_root = Path(".")
    repos_base_dir = repo_root / "data" / "unseen_repos"
    reports_dir = repo_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    runner = UnseenRepoABBenchmarkRunner(
        repos_base_dir=repos_base_dir,
        repo_root=repo_root,
        prefer_gemini_api=True,
        verify_postgres_rls=True,
        run_live_frozen_qlora=True,
    )
    traces_path = reports_dir / "unseen_ab_traces.jsonl"
    report = runner.run_benchmark(traces_jsonl_path=traces_path)

    json_path = reports_dir / "unseen_repo_ab_report.json"
    md_path = reports_dir / "unseen_repo_ab_report.md"
    json_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    md_path.write_text(format_unseen_ab_markdown_report(report), encoding="utf-8")
    logger.info("Saved JSON report to %s and Markdown report to %s", json_path, md_path)


if __name__ == "__main__":
    main()
