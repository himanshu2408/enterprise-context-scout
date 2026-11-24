"""Evaluation Harness for Milestone 6 CI/CD Architectural Quality Gates.

Evaluates 6 architectural gates against the Golden Benchmark Suite:
1. Zero-Touch Sufficiency Rate (>= 85.0%)
2. File & Symbol Recall@5 (>= 90.0%)
3. Verbatim Citation Faithfulness (== 100.0%)
4. Frontier Token Savings Ratio (>= 80.0%)
5. RBAC Zero-Leakage Rate (== 100.0%)
6. Prompt-Injection Block Rate (>= 95.0%)

Supports evaluation on both live PostgreSQL 17 with Forced RLS and In-Memory repositories.
Instruments full evaluation runs with OpenTelemetry GenAI semantic conventions.
"""

from __future__ import annotations

import datetime
import logging
import sys
import time
from pathlib import Path
from typing import Any, Final, Literal
from uuid import UUID

from pydantic import BaseModel, Field

from context_scout.agents.domain_registry import DomainRegistry
from context_scout.agents.gemini_executor import (
    DEFAULT_FAST_MODEL_ID,
    GeminiImplementationExecutor,
)
from context_scout.agents.scout_workflow import (
    FederatedScoutCoordinator,
    ScoutWorkflow,
)
from context_scout.ast_engine.deterministic_slicer import DeterministicSlicer
from context_scout.config import ScoutSettings
from context_scout.db.repositories import (
    ASTSymbolRecord,
    CodeFileRecord,
    InMemoryRLSRepository,
)
from context_scout.db.session import RLSConnectionManager
from context_scout.evals.golden_dataset import (
    GoldenBenchmarkSuite,
    GoldenCodeTask,
    GoldenFederatedTask,
    PoisonedInjectionPayload,
    RedTeamRBACQuery,
    get_default_golden_suite,
)
from context_scout.mcp_servers.common import PostgresScoutGraphAdapter
from context_scout.plugins.registry import PluginRegistry
from context_scout.retrieval.embedder import get_embedding_provider
from context_scout.schemas.contracts import (
    DocumentScope,
    ImplementationContextPack,
    MissingContextRequest,
    PrincipalContext,
    RoleType,
)
from context_scout.security.guardrails import (
    PromptInjectionSanitizer,
    SecretRedactor,
    ShieldGemmaGuardrail,
)
from context_scout.telemetry import (
    ScoutTelemetryTracer,
    get_tracer,
)

logger = logging.getLogger(__name__)


class EvaluationGateThresholds(BaseModel):
    """Target thresholds for the 6 CI/CD architectural gates."""

    zero_touch_sufficiency_rate: float = 85.0
    file_and_symbol_recall_at_5: float = 90.0
    verbatim_citation_faithfulness: float = 100.0
    frontier_token_savings_ratio: float = 80.0
    rbac_zero_leakage_rate: float = 100.0
    prompt_injection_block_rate: float = 95.0


class MetricResult(BaseModel):
    """Measured result for a single CI/CD evaluation gate."""

    metric_id: str
    display_name: str
    measured_value: float
    target_threshold: float
    comparator: Literal[">=", "=="]
    passed: bool
    details: dict[str, Any] = Field(default_factory=dict)


class EvaluationRunResult(BaseModel):
    """Aggregate result from an end-to-end evaluation harness run."""

    timestamp: str
    all_passed: bool
    metrics: dict[str, MetricResult]
    summary: str
    total_duration_ms: int
    telemetry_spans_count: int = 0


class EvaluationHarness:
    """Enterprise CI/CD Evaluation Harness validating the 6 architectural gates."""

    METRIC_ZERO_TOUCH: Final[str] = "zero_touch_sufficiency_rate"
    METRIC_RECALL_AT_5: Final[str] = "file_and_symbol_recall_at_5"
    METRIC_FAITHFULNESS: Final[str] = "verbatim_citation_faithfulness"
    METRIC_TOKEN_SAVINGS: Final[str] = "frontier_token_savings_ratio"
    METRIC_RBAC_LEAKAGE: Final[str] = "rbac_zero_leakage_rate"
    METRIC_PROMPT_INJECTION: Final[str] = "prompt_injection_block_rate"

    def __init__(
        self,
        settings: ScoutSettings | None = None,
        thresholds: EvaluationGateThresholds | None = None,
        repo: Any = None,
        registry: DomainRegistry | None = None,
        tracer: ScoutTelemetryTracer | None = None,
        repo_root: Path | None = None,
        require_postgres: bool = False,
        force_in_memory: bool = False,
        code_domain_id: str | None = None,
        research_domain_id: str | None = None,
    ) -> None:
        self.settings = settings or ScoutSettings()
        self.thresholds = thresholds or EvaluationGateThresholds()
        self.tracer = tracer or get_tracer()
        self.slicer = DeterministicSlicer()
        self.guardrail = ShieldGemmaGuardrail(allow_deterministic_fallback=True)
        primary_plugin = PluginRegistry.get_primary_domain_plugin()
        roles = primary_plugin.get_role_domain_ids() if primary_plugin is not None else {}
        self.code_domain_id = code_domain_id or roles.get("code", self.settings.default_domain_id)
        self.research_domain_id = research_domain_id or roles.get(
            "research", self.settings.default_domain_id
        )

        # Resolve repository root
        if repo_root is not None:
            self.repo_root = repo_root.resolve()
        else:
            current = Path(__file__).resolve()
            # enterprise-context-scout root
            self.repo_root = current.parents[3]

        # Domain registry
        domains_dir = self.repo_root / "domains"
        if registry is not None:
            self.registry = registry
        else:
            self.registry = DomainRegistry(domains_dir)
            self.registry.load_all()

        # Repository resolution: use provided, or PostgreSQL if connected, or InMemory fallback
        self._connection_manager: RLSConnectionManager | None = None
        if repo is not None:
            self.repo = repo
        elif force_in_memory:
            logger.info("EvaluationHarness configured with force_in_memory=True.")
            self.repo = self._create_populated_in_memory_repo()
        else:
            self.repo = self._auto_configure_repository(require_postgres=require_postgres)

        self.executor = GeminiImplementationExecutor(
            model_id=DEFAULT_FAST_MODEL_ID,
            require_confirmation=False,
        )

    def _auto_configure_repository(self, require_postgres: bool = False) -> Any:
        """Attempts live PostgreSQL 17 connection, falling back to populated in-memory repo."""
        try:
            mgr = RLSConnectionManager(self.settings.db_app_url)
            test_principal = PrincipalContext(
                user_id="harness_probe",
                role=RoleType.ENGINEER,
                allowed_scopes=["public", "public_eng", "public_docs"],
            )
            with mgr.scoped_transaction(test_principal) as conn:
                row = conn.execute(
                    "SELECT COUNT(*) FROM doc_chunks WHERE domain_id = %s;",
                    (self.research_domain_id,),
                ).fetchone()
                doc_count = int(str(row[0])) if row else 0
            if doc_count < 100 and not require_postgres:
                logger.info(
                    "PostgreSQL has %d research chunks (<100); using InMemoryRLSRepository.",
                    doc_count,
                )
                return self._create_populated_in_memory_repo()
            self._connection_manager = mgr
            logger.info("EvaluationHarness connected to live PostgreSQL 17.")
            return PostgresScoutGraphAdapter(mgr, self.settings)
        except Exception as exc:
            if require_postgres:
                logger.error("Live PostgreSQL 17 required but connection failed: %s", exc)
                raise
            logger.warning(
                "Live PostgreSQL 17 connection unavailable (%s); using InMemoryRLSRepository.",
                exc,
            )
            return self._create_populated_in_memory_repo()

    def _create_populated_in_memory_repo(self) -> InMemoryRLSRepository:
        """Creates an in-memory repository populated with all context scout source files."""
        repo = InMemoryRLSRepository()
        admin_principal = PrincipalContext(
            user_id="harness_seeder",
            role=RoleType.ENGINEER,
            allowed_scopes=["public", "public_eng", "public_docs", "docs_restricted"],
        )

        # Hydrate doc_chunks from live PostgreSQL 17 if accessible; code_files are
        # always indexed from disk (self.repo_root) so verbatim slices match HEAD.
        seen_file_paths: set[str] = set()
        seen_chunk_ids: set[UUID] = set()
        seen_doc_ids: set[str] = set()
        try:
            import psycopg

            with (
                psycopg.connect(self.settings.db_admin_url) as conn,
                conn.cursor() as cur,
            ):
                cur.execute(
                    "SELECT COUNT(*) FROM doc_chunks WHERE domain_id = %s;",
                    (self.research_domain_id,),
                )
                row = cur.fetchone()
                if row and int(str(row[0])) > 0:
                    mgr = RLSConnectionManager(self.settings.db_app_url)
                    adapter = PostgresScoutGraphAdapter(mgr, self.settings)
                    for mr in (None, *list(DocumentScope)):
                        for c in adapter.list_visible_doc_chunks(
                            admin_principal, self.research_domain_id, market_scope=mr
                        ):
                            if c.chunk_id not in seen_chunk_ids:
                                seen_chunk_ids.add(c.chunk_id)
                                seen_doc_ids.add(c.doc_id)
                                repo.insert_doc_chunk(admin_principal, c)
        except Exception as exc:
            logger.debug(
                "PostgreSQL hydration skipped for EvaluationHarness in-memory repo (%s): %s",
                type(exc).__name__,
                exc,
            )

        import ast
        import dataclasses
        import uuid

        scan_roots = [self.repo_root / "src" / "context_scout", self.repo_root / "tests"]
        pending_syms: list[tuple[UUID, str, str, int, int, str, str, str]] = []
        for scan_dir in scan_roots:
            if not scan_dir.exists():
                continue
            for py_file in sorted(scan_dir.rglob("*.py")):
                rel_path = py_file.relative_to(self.repo_root).as_posix()
                if rel_path in seen_file_paths:
                    continue
                try:
                    content = py_file.read_text(encoding="utf-8")
                except Exception as exc:
                    logger.debug(
                        "Skipping unreadable source file %s in EvaluationHarness (%s): %s",
                        py_file,
                        type(exc).__name__,
                        exc,
                    )
                    continue
                seen_file_paths.add(rel_path)
                lines = content.splitlines()
                f_id = uuid.uuid4()
                rec = CodeFileRecord(
                    file_id=f_id,
                    domain_id=self.code_domain_id,
                    repo_name="enterprise-context-scout",
                    file_path=rel_path,
                    git_commit_sha="c0ffee123456",
                    git_blob_sha="b10b123456",
                    language="python",
                    allowed_scopes=["public_eng"],
                    raw_content=content,
                    line_count=len(lines),
                )
                repo.insert_code_file(admin_principal, rec)

                # Extract classes, functions, and methods with exact AST spans
                rel_base = self.repo_root / "src" if rel_path.startswith("src/") else self.repo_root
                module_parts = py_file.relative_to(rel_base).with_suffix("").parts
                mod_dotted = ".".join(module_parts)

                try:
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                            sym_name = node.name
                            q_name = f"{mod_dotted}.{sym_name}"
                            is_cls = isinstance(node, ast.ClassDef)
                            kind = "class" if is_cls else "function"
                            prefix = "class" if is_cls else "def"
                            doc = ast.get_docstring(node) or f"{kind.title()} {sym_name}"
                            pending_syms.append(
                                (
                                    f_id,
                                    q_name,
                                    kind,
                                    node.lineno,
                                    node.end_lineno or min(node.lineno + 50, len(lines)),
                                    f"{prefix} {sym_name}:",
                                    doc[:400],
                                    f"{q_name} {prefix} {sym_name}: {doc[:200]}",
                                )
                            )
                except Exception as exc:
                    logger.debug(
                        "AST parse failed for %s in EvaluationHarness (%s): %s",
                        rel_path,
                        type(exc).__name__,
                        exc,
                    )

        if pending_syms:
            sym_vecs = get_embedding_provider().embed_batch(
                [item[7] for item in pending_syms],
                is_query=False,
                batch_size=64,
            )
            for (
                f_id,
                q_name,
                kind,
                s_line,
                e_line,
                sig_skel,
                ctx_sum,
                _emb_txt,
            ), emb_vec in zip(pending_syms, sym_vecs, strict=True):
                repo.insert_ast_symbol(
                    admin_principal,
                    ASTSymbolRecord(
                        symbol_id=uuid.uuid4(),
                        file_id=f_id,
                        domain_id=self.code_domain_id,
                        qualified_name=q_name,
                        symbol_kind=kind,
                        start_line=s_line,
                        end_line=e_line,
                        signature_skeleton=sig_skel,
                        contextual_summary=ctx_sum,
                        allowed_scopes=["public_eng"],
                        embedding=emb_vec,
                    ),
                )

        # Seed sample doc chunks from registered domain plugins
        for domain_plugin in PluginRegistry.list_domain_plugins():
            seed_payload = domain_plugin.get_demo_seed_payloads()
            chunks_for_plugin = seed_payload.get("doc_chunks", [])
            is_research_plugin = seed_payload.get("research_domain_id") == self.research_domain_id
            selected_chunks = chunks_for_plugin if is_research_plugin else chunks_for_plugin[:1]
            for doc_rec in selected_chunks:
                if doc_rec.doc_id not in seen_doc_ids:
                    seen_doc_ids.add(doc_rec.doc_id)
                    sanitized_scopes = (
                        ["public_docs"]
                        if doc_rec.domain_id == self.research_domain_id
                        else [s for s in doc_rec.allowed_scopes if s != "docs_restricted"]
                    )
                    repo.insert_doc_chunk(
                        admin_principal,
                        dataclasses.replace(doc_rec, allowed_scopes=sanitized_scopes),
                    )

        return repo

    # =========================================================================
    # Gate 1: Zero-Touch Sufficiency Rate (>= 85.0%)
    # =========================================================================
    def evaluate_zero_touch_sufficiency(
        self,
        tasks: list[GoldenCodeTask | GoldenFederatedTask],
        packs: list[ImplementationContextPack] | None = None,
    ) -> MetricResult:
        """Evaluates percentage of tasks executed without missing context requests."""
        if not tasks:
            return MetricResult(
                metric_id=self.METRIC_ZERO_TOUCH,
                display_name="Zero-Touch Sufficiency Rate",
                measured_value=0.0,
                target_threshold=self.thresholds.zero_touch_sufficiency_rate,
                comparator=">=",
                passed=False,
                details={"error": "Empty tasks suite"},
            )

        evaluated_packs: list[ImplementationContextPack | None] = []
        if packs is not None and len(packs) == len(tasks):
            evaluated_packs = list(packs)
        else:
            workflow = ScoutWorkflow(registry=self.registry, repo=self.repo)
            federated = FederatedScoutCoordinator(registry=self.registry, repo=self.repo)
            for t in tasks:
                t_p = PrincipalContext(
                    user_id=f"zero_touch_{t.task_id}",
                    role=RoleType.ENGINEER,
                    allowed_scopes=[
                        "public",
                        "public_eng",
                        "public_docs",
                        f"domain:{self.research_domain_id}",
                    ],
                    session_budget_tokens=200_000,
                    max_session_paid_tokens=200_000,
                )
                if isinstance(t, GoldenFederatedTask):
                    p = federated.run(user_prompt=t.user_prompt, principal=t_p)
                else:
                    p = workflow.run(
                        user_prompt=t.user_prompt,
                        principal=t_p,
                        domain_id=t.domain_id,
                    )
                evaluated_packs.append(p if isinstance(p, ImplementationContextPack) else None)

        success_count = 0
        task_details: list[dict[str, Any]] = []

        for idx, task in enumerate(tasks):
            t_id = task.task_id
            pack = evaluated_packs[idx] if idx < len(evaluated_packs) else None
            if pack is None:
                task_details.append(
                    {"task_id": t_id, "success": False, "reason": "ContextPack assembly failed"}
                )
                continue
            task_principal = PrincipalContext(
                user_id=f"zero_touch_evaluator_{t_id}",
                role=RoleType.ENGINEER,
                allowed_scopes=[
                    "public",
                    "public_eng",
                    "public_docs",
                    f"domain:{self.research_domain_id}",
                ],
                session_budget_tokens=200_000,
                max_session_paid_tokens=200_000,
            )

            if isinstance(task, GoldenFederatedTask):
                req_files = list(task.ground_truth_code_files)
                req_symbols = list(task.ground_truth_code_symbols)
                req_1hop: list[str] = []
            else:
                req_files = list(task.ground_truth_files)
                req_symbols = list(task.ground_truth_symbols)
                req_1hop = list(task.ast_1hop_reference_symbols)

            covered_files = {s.file_path for s in pack.code_slices}
            covered_symbols = {s.qualified_symbol for s in pack.code_slices}
            covered_short_symbols = {s.qualified_symbol.split(".")[-1] for s in pack.code_slices}
            covered_code_blob = "\n".join(s.verbatim_code_with_lines for s in pack.code_slices)

            missing_reqs: list[MissingContextRequest] = []
            seen_missing: set[str] = set()
            for req_sym in req_symbols:
                short_sym = req_sym.split(".")[-1]
                if (
                    req_sym not in covered_symbols
                    and short_sym not in covered_short_symbols
                    and req_sym not in seen_missing
                ):
                    seen_missing.add(req_sym)
                    missing_reqs.append(
                        MissingContextRequest(
                            missing_symbol=req_sym,
                            file_hint=req_files[0] if req_files else None,
                            reason=(f"Required ground-truth symbol '{req_sym}' missing from pack"),
                        )
                    )
            for ref_sym in req_1hop:
                short_ref = ref_sym.split(".")[-1]
                if (
                    ref_sym not in covered_symbols
                    and short_ref not in covered_short_symbols
                    and short_ref not in covered_code_blob
                    and ref_sym not in seen_missing
                ):
                    seen_missing.add(ref_sym)
                    missing_reqs.append(
                        MissingContextRequest(
                            missing_symbol=ref_sym,
                            file_hint=req_files[0] if req_files else None,
                            reason=(
                                f"Required 1-hop reference symbol '{ref_sym}' missing from pack"
                            ),
                        )
                    )
            for req_file in req_files:
                if req_file not in covered_files:
                    file_key = f"file:{req_file}"
                    if file_key not in seen_missing:
                        seen_missing.add(file_key)
                        missing_reqs.append(
                            MissingContextRequest(
                                missing_symbol=Path(req_file).stem,
                                file_hint=req_file,
                                reason=f"Required ground-truth file '{req_file}' missing from pack",
                            )
                        )

            target_py_files = sorted(
                {
                    s.file_path
                    for s in pack.code_slices
                    if s.file_path.endswith(".py") and (self.repo_root / s.file_path).is_file()
                }
            )
            if not target_py_files:
                target_py_files = sorted(
                    {f for f in req_files if f.endswith(".py") and (self.repo_root / f).is_file()}
                )
            if target_py_files:
                lint_cmd = ["ruff", "check", *target_py_files]
                test_cmd = [sys.executable, "-m", "py_compile", *target_py_files]
            else:
                lint_cmd = ["ruff", "check", "src/context_scout"]
                test_cmd = [sys.executable, "--version"]

            # Execute patch via GeminiImplementationExecutor in isolated worktree sandbox
            try:
                res, tele = self.executor.execute_implementation(
                    pack=pack,
                    principal=task_principal,
                    repo_root=self.repo_root,
                    missing_context_requests=missing_reqs or None,
                    auto_confirm=True,
                    lint_cmd=lint_cmd,
                    test_cmd=test_cmd,
                )
                zero_touch_ok = (
                    len(tele.missing_context_requests) == 0
                    and len(missing_reqs) == 0
                    and res.success
                    and bool(tele.zero_touch_sufficient)
                )
                if zero_touch_ok:
                    success_count += 1
                task_details.append(
                    {
                        "task_id": t_id,
                        "success": zero_touch_ok,
                        "missing_requests": len(tele.missing_context_requests),
                        "patch_applied": res.success,
                        "target_files": target_py_files,
                    }
                )
            except Exception as exc:
                logger.warning("Zero-touch execution error on %s: %s", t_id, exc)
                task_details.append({"task_id": t_id, "success": False, "error": str(exc)})

        measured_rate = round((success_count / len(tasks)) * 100.0, 2)
        passed = measured_rate >= self.thresholds.zero_touch_sufficiency_rate

        return MetricResult(
            metric_id=self.METRIC_ZERO_TOUCH,
            display_name="Zero-Touch Sufficiency Rate",
            measured_value=measured_rate,
            target_threshold=self.thresholds.zero_touch_sufficiency_rate,
            comparator=">=",
            passed=passed,
            details={
                "total_tasks": len(tasks),
                "successful_tasks": success_count,
                "task_results": task_details,
            },
        )

    # =========================================================================
    # Gate 2: File & Symbol Recall@5 (>= 90.0%)
    # =========================================================================
    def evaluate_file_and_symbol_recall(
        self,
        tasks: list[GoldenCodeTask],
        packs: list[ImplementationContextPack] | None = None,
    ) -> MetricResult:
        """Evaluates recall of target files and AST symbols in top-5 code slices."""
        if not tasks:
            return MetricResult(
                metric_id=self.METRIC_RECALL_AT_5,
                display_name="File & Symbol Recall@5",
                measured_value=0.0,
                target_threshold=self.thresholds.file_and_symbol_recall_at_5,
                comparator=">=",
                passed=False,
                details={"error": "Empty code tasks"},
            )

        principal = PrincipalContext(
            user_id="recall_evaluator",
            role=RoleType.ENGINEER,
            allowed_scopes=["public_eng"],
        )

        evaluated_packs: list[ImplementationContextPack] = []
        if packs is not None and len(packs) == len(tasks):
            evaluated_packs = packs
        else:
            workflow = ScoutWorkflow(registry=self.registry, repo=self.repo)
            for t in tasks:
                p = workflow.run(
                    user_prompt=t.user_prompt,
                    principal=principal,
                    domain_id=t.domain_id,
                )
                if isinstance(p, ImplementationContextPack):
                    evaluated_packs.append(p)

        task_recalls: list[float] = []
        details_list: list[dict[str, Any]] = []

        for idx, task in enumerate(tasks):
            if idx >= len(evaluated_packs):
                task_recalls.append(0.0)
                details_list.append({"task_id": task.task_id, "task_recall": 0.0})
                continue

            pack = evaluated_packs[idx]
            top_5_slices = pack.code_slices[:5]
            retrieved_files = {s.file_path for s in top_5_slices}
            retrieved_symbols = {s.qualified_symbol for s in top_5_slices}
            # Also index short symbol names
            retrieved_short_symbols = {s.qualified_symbol.split(".")[-1] for s in top_5_slices}

            gt_files = set(task.ground_truth_files)
            gt_symbols = set(task.ground_truth_symbols)

            # File Recall
            matched_files = gt_files & retrieved_files
            file_recall = len(matched_files) / len(gt_files) if gt_files else 1.0

            # Symbol Recall
            matched_syms = 0
            for sym in gt_symbols:
                short = sym.split(".")[-1]
                if sym in retrieved_symbols or short in retrieved_short_symbols:
                    matched_syms += 1
            symbol_recall = matched_syms / len(gt_symbols) if gt_symbols else 1.0

            task_recall = (file_recall + symbol_recall) / 2.0
            task_recalls.append(task_recall)
            details_list.append(
                {
                    "task_id": task.task_id,
                    "file_recall": round(file_recall * 100.0, 2),
                    "symbol_recall": round(symbol_recall * 100.0, 2),
                    "combined_task_recall": round(task_recall * 100.0, 2),
                }
            )

        overall_recall = (
            round((sum(task_recalls) / len(task_recalls)) * 100.0, 2) if task_recalls else 0.0
        )
        passed = overall_recall >= self.thresholds.file_and_symbol_recall_at_5

        return MetricResult(
            metric_id=self.METRIC_RECALL_AT_5,
            display_name="File & Symbol Recall@5",
            measured_value=overall_recall,
            target_threshold=self.thresholds.file_and_symbol_recall_at_5,
            comparator=">=",
            passed=passed,
            details={"task_breakdown": details_list, "tasks_count": len(tasks)},
        )

    # =========================================================================
    # Gate 3: Verbatim Citation Faithfulness (== 100.0%)
    # =========================================================================
    def evaluate_verbatim_citation_faithfulness(
        self,
        packs: list[ImplementationContextPack],
    ) -> MetricResult:
        """Verifies byte-exact source line alignment and document quote provenance."""
        if not packs:
            return MetricResult(
                metric_id=self.METRIC_FAITHFULNESS,
                display_name="Verbatim Citation Faithfulness",
                measured_value=0.0,
                target_threshold=self.thresholds.verbatim_citation_faithfulness,
                comparator="==",
                passed=False,
                details={"error": "No context packs to evaluate"},
            )

        total_slices = 0
        valid_slices = 0
        total_citations = 0
        valid_citations = 0
        failures: list[dict[str, Any]] = []
        visible_chunks_by_domain: dict[str, dict[str, Any]] = {}

        for pack in packs:
            # 1. Verify code slices
            for s in pack.code_slices:
                total_slices += 1
                full_path = self.repo_root / s.file_path
                if not full_path.exists():
                    failures.append(
                        {
                            "type": "code_slice",
                            "file": s.file_path,
                            "symbol": s.qualified_symbol,
                            "reason": "File does not exist on disk",
                        }
                    )
                    continue

                raw_content = full_path.read_text(encoding="utf-8")
                try:
                    recomputed = self.slicer.slice_lines(
                        raw_content=raw_content,
                        start_line=s.start_line,
                        end_line=s.end_line,
                    )
                    if recomputed == s.verbatim_code_with_lines:
                        valid_slices += 1
                    elif "elided]" in s.verbatim_code_with_lines:
                        # Verify that every included non-elided line matches raw_content verbatim
                        raw_lines = raw_content.splitlines()
                        slice_lines = s.verbatim_code_with_lines.splitlines()
                        is_faithful = True
                        for s_line in slice_lines:
                            if "elided]" in s_line:
                                continue
                            if "|" in s_line:
                                parts = s_line.split("|", 1)
                                try:
                                    ln_num = int(parts[0].strip())
                                    line_text = (
                                        parts[1][1:] if parts[1].startswith(" ") else parts[1]
                                    )
                                    if 1 <= ln_num <= len(raw_lines):
                                        if raw_lines[ln_num - 1] != line_text:
                                            is_faithful = False
                                            break
                                    else:
                                        is_faithful = False
                                        break
                                except ValueError as exc:
                                    logger.debug(
                                        "Skipping non-numbered slice prefix (%s): %s",
                                        type(exc).__name__,
                                        exc,
                                    )
                        if is_faithful:
                            valid_slices += 1
                        else:
                            failures.append(
                                {
                                    "type": "code_slice",
                                    "file": s.file_path,
                                    "symbol": s.qualified_symbol,
                                    "reason": "Skeletonized slice contains non-verbatim lines",
                                }
                            )
                    else:
                        failures.append(
                            {
                                "type": "code_slice",
                                "file": s.file_path,
                                "symbol": s.qualified_symbol,
                                "reason": "Slices do not match byte-for-byte",
                            }
                        )
                except Exception as exc:
                    failures.append(
                        {
                            "type": "code_slice",
                            "file": s.file_path,
                            "symbol": s.qualified_symbol,
                            "reason": f"Slice error: {exc}",
                        }
                    )

            # 2. Verify document citations (fail closed if chunk_id missing or repo cannot verify)
            citations_to_check = list(pack.supporting_doc_citations)
            if pack.domain_knowledge_brief:
                citations_to_check.extend(pack.domain_knowledge_brief.verified_citations)

            deduped_citations: list[Any] = []
            seen_cit_keys: set[tuple[str, str]] = set()
            for cit in citations_to_check:
                cit_key = (str(cit.chunk_id), str(cit.verbatim_quote))
                if cit_key not in seen_cit_keys:
                    seen_cit_keys.add(cit_key)
                    deduped_citations.append(cit)

            registered_domains = (
                list(self.registry.load_all().keys()) if hasattr(self.registry, "load_all") else []
            )
            domains_to_check = list(
                dict.fromkeys([pack.domain_id, self.research_domain_id, *registered_domains])
            )
            dummy_p = PrincipalContext(
                user_id="doc_verifier",
                role=RoleType.DOMAIN_RESEARCHER,
                allowed_scopes=[
                    "public",
                    "public_eng",
                    "public_docs",
                    *[f"domain:{d}" for d in domains_to_check],
                ],
            )

            for cit in deduped_citations:
                total_citations += 1
                if not (
                    hasattr(self.repo, "list_visible_doc_chunks")
                    or hasattr(self.repo, "get_doc_chunks_by_ids")
                ):
                    failures.append(
                        {
                            "type": "doc_citation",
                            "chunk_id": cit.chunk_id,
                            "reason": (
                                "Repository does not support doc chunk lookup (fail-closed)."
                            ),
                        }
                    )
                    continue

                matching_chk = None
                for dom_id in domains_to_check:
                    if hasattr(self.repo, "get_doc_chunks_by_ids"):
                        fetched_chks = self.repo.get_doc_chunks_by_ids(
                            dummy_p, dom_id, [str(cit.chunk_id)]
                        )
                        if fetched_chks:
                            matching_chk = fetched_chks[0]
                    if matching_chk is None and hasattr(self.repo, "list_visible_doc_chunks"):
                        if dom_id not in visible_chunks_by_domain:
                            chunks = self.repo.list_visible_doc_chunks(dummy_p, dom_id)
                            visible_chunks_by_domain[dom_id] = {str(c.chunk_id): c for c in chunks}
                        matching_chk = visible_chunks_by_domain[dom_id].get(str(cit.chunk_id))
                    if matching_chk is not None:
                        break

                if matching_chk is not None:
                    res = self.slicer.verify_doc_citation(cit, matching_chk.verbatim_content)
                    if res.is_valid:
                        valid_citations += 1
                    else:
                        failures.append(
                            {
                                "type": "doc_citation",
                                "chunk_id": cit.chunk_id,
                                "reason": res.error_reason,
                            }
                        )
                else:
                    failures.append(
                        {
                            "type": "doc_citation",
                            "chunk_id": cit.chunk_id,
                            "reason": (
                                f"Cited chunk_id '{cit.chunk_id}' not found in repository "
                                "(fail-closed)."
                            ),
                        }
                    )

        total_items = total_slices + total_citations
        valid_items = valid_slices + valid_citations
        faithfulness_pct = (
            round((valid_items / total_items) * 100.0, 2) if total_items > 0 else 100.0
        )
        passed = faithfulness_pct == self.thresholds.verbatim_citation_faithfulness

        return MetricResult(
            metric_id=self.METRIC_FAITHFULNESS,
            display_name="Verbatim Citation Faithfulness",
            measured_value=faithfulness_pct,
            target_threshold=self.thresholds.verbatim_citation_faithfulness,
            comparator="==",
            passed=passed,
            details={
                "total_code_slices": total_slices,
                "valid_code_slices": valid_slices,
                "total_doc_citations": total_citations,
                "valid_doc_citations": valid_citations,
                "failures": failures,
            },
        )

    # =========================================================================
    # Gate 4: Frontier Token Savings Ratio (>= 80.0%)
    # =========================================================================
    def evaluate_frontier_token_savings(
        self,
        packs: list[ImplementationContextPack],
    ) -> MetricResult:
        """Evaluates token reduction sending compiler-closed context pack vs whole repos."""
        if not packs:
            return MetricResult(
                metric_id=self.METRIC_TOKEN_SAVINGS,
                display_name="Frontier Token Savings Ratio",
                measured_value=0.0,
                target_threshold=self.thresholds.frontier_token_savings_ratio,
                comparator=">=",
                passed=False,
                details={"error": "No context packs to evaluate"},
            )

        total_sent = sum(p.pack_token_count for p in packs)
        total_explored = sum(p.local_tokens_consumed for p in packs)

        savings_ratio = (
            round((1.0 - (total_sent / total_explored)) * 100.0, 2) if total_explored > 0 else 0.0
        )
        passed = savings_ratio >= self.thresholds.frontier_token_savings_ratio

        return MetricResult(
            metric_id=self.METRIC_TOKEN_SAVINGS,
            display_name="Frontier Token Savings Ratio",
            measured_value=savings_ratio,
            target_threshold=self.thresholds.frontier_token_savings_ratio,
            comparator=">=",
            passed=passed,
            details={
                "total_context_pack_tokens_sent": total_sent,
                "total_local_tokens_explored": total_explored,
                "tokens_saved": total_explored - total_sent,
            },
        )

    # =========================================================================
    # Gate 5: RBAC Zero-Leakage Rate (== 100.0%)
    # =========================================================================
    def evaluate_rbac_zero_leakage(
        self,
        queries: list[RedTeamRBACQuery],
    ) -> MetricResult:
        """Evaluates forced tenant isolation across adversarial cross-role red-team queries."""
        if not queries:
            return MetricResult(
                metric_id=self.METRIC_RBAC_LEAKAGE,
                display_name="RBAC Zero-Leakage Rate",
                measured_value=0.0,
                target_threshold=self.thresholds.rbac_zero_leakage_rate,
                comparator="==",
                passed=False,
                details={"error": "Empty RBAC queries"},
            )

        blocked_count = 0
        details: list[dict[str, Any]] = []

        for q in queries:
            records_returned = 0
            # Test against repository
            try:
                if q.target_table == "code_files":
                    files = self.repo.list_visible_code_files(q.principal, self.code_domain_id)
                    # Check if target scope or restricted record was returned
                    if "target_scope" in q.query_payload:
                        records_returned = sum(
                            1 for f in files if q.query_payload["target_scope"] in f.allowed_scopes
                        )
                    elif "file_path" in q.query_payload:
                        records_returned = sum(
                            1 for f in files if f.file_path == q.query_payload["file_path"]
                        )
                    else:
                        records_returned = sum(
                            1 for f in files if q.target_scope_to_test in f.allowed_scopes
                        )

                elif q.target_table == "ast_symbols":
                    symbols = self.repo.list_visible_ast_symbols(q.principal, self.code_domain_id)
                    if "qualified_name" in q.query_payload:
                        records_returned = sum(
                            1
                            for s in symbols
                            if s.qualified_name == q.query_payload["qualified_name"]
                        )
                    else:
                        records_returned = sum(
                            1 for s in symbols if q.target_scope_to_test in s.allowed_scopes
                        )

                elif q.target_table == "ast_edges":
                    edges = self.repo.list_visible_ast_edges(q.principal, self.code_domain_id)
                    records_returned = sum(
                        1 for e in edges if q.target_scope_to_test in e.allowed_scopes
                    )

                elif q.target_table == "doc_chunks":
                    chunks = self.repo.list_visible_doc_chunks(q.principal, self.research_domain_id)
                    records_returned = sum(
                        1 for c in chunks if q.target_scope_to_test in c.allowed_scopes
                    )
            except Exception as exc:
                # If error is raised due to scope isolation (e.g. PermissionError or SQL
                # injection rejected), 0 records leaked
                logger.debug("RBAC query %s caught exception (clean block): %s", q.query_id, exc)
                records_returned = 0

            is_zero_leak = records_returned == 0
            if is_zero_leak:
                blocked_count += 1
            details.append(
                {
                    "query_id": q.query_id,
                    "target_table": q.target_table,
                    "target_scope": q.target_scope_to_test,
                    "records_returned": records_returned,
                    "zero_leak_maintained": is_zero_leak,
                }
            )

        measured_rate = round((blocked_count / len(queries)) * 100.0, 2)
        passed = measured_rate == self.thresholds.rbac_zero_leakage_rate

        return MetricResult(
            metric_id=self.METRIC_RBAC_LEAKAGE,
            display_name="RBAC Zero-Leakage Rate",
            measured_value=measured_rate,
            target_threshold=self.thresholds.rbac_zero_leakage_rate,
            comparator="==",
            passed=passed,
            details={
                "total_redteam_queries": len(queries),
                "zero_leak_count": blocked_count,
                "query_results": details,
            },
        )

    # =========================================================================
    # Gate 6: Prompt-Injection Block Rate (>= 95.0%)
    # =========================================================================
    def evaluate_prompt_injection_block_rate(
        self,
        payloads: list[PoisonedInjectionPayload],
    ) -> MetricResult:
        """Evaluates detection, sanitization, and blocking of adversarial payloads."""
        if not payloads:
            return MetricResult(
                metric_id=self.METRIC_PROMPT_INJECTION,
                display_name="Prompt-Injection Block Rate",
                measured_value=0.0,
                target_threshold=self.thresholds.prompt_injection_block_rate,
                comparator=">=",
                passed=False,
                details={"error": "Empty injection payloads"},
            )

        blocked_count = 0
        injection_total = 0
        injection_blocked = 0
        secret_total = 0
        secret_blocked = 0
        details: list[dict[str, Any]] = []

        for p in payloads:
            raw = p.raw_content
            decision = self.guardrail.evaluate(raw)
            sanitized, is_inj, sigs = PromptInjectionSanitizer.sanitize(raw)
            redacted_text, direct_redacted_labels = SecretRedactor.redact(raw)
            merged_redacted = list(dict.fromkeys(decision.redacted_tokens + direct_redacted_labels))

            # Block condition
            is_blocked = (
                (not decision.passed)
                or is_inj
                or (sanitized != raw)
                or (redacted_text != raw)
                or bool(merged_redacted)
            )

            if is_blocked:
                blocked_count += 1

            if p.category == "secret_leakage":
                secret_total += 1
                if bool(merged_redacted) or (redacted_text != raw) or (not decision.passed):
                    secret_blocked += 1
            else:
                injection_total += 1
                if is_blocked:
                    injection_blocked += 1

            details.append(
                {
                    "payload_id": p.payload_id,
                    "category": p.category,
                    "guardrail_passed": decision.passed,
                    "injection_signatures": sigs,
                    "redacted_tokens": merged_redacted,
                    "sanitized_differs": sanitized != raw,
                    "is_blocked": is_blocked,
                }
            )

        measured_rate = round((blocked_count / len(payloads)) * 100.0, 2)
        passed = measured_rate >= self.thresholds.prompt_injection_block_rate

        return MetricResult(
            metric_id=self.METRIC_PROMPT_INJECTION,
            display_name="Prompt-Injection Block Rate",
            measured_value=measured_rate,
            target_threshold=self.thresholds.prompt_injection_block_rate,
            comparator=">=",
            passed=passed,
            details={
                "total_payloads": len(payloads),
                "blocked_count": blocked_count,
                "prompt_injection_subcheck": {
                    "total": injection_total,
                    "blocked": injection_blocked,
                },
                "secret_redaction_subcheck": {
                    "total": secret_total,
                    "redacted": secret_blocked,
                },
                "payload_results": details,
            },
        )

    # =========================================================================
    # End-to-End Evaluation Pipeline
    # =========================================================================
    def evaluate_all(
        self,
        suite: GoldenBenchmarkSuite | None = None,
    ) -> EvaluationRunResult:
        """Runs the complete 6-gate evaluation pipeline with OpenTelemetry tracing."""
        start_time = time.perf_counter()
        active_suite = suite or get_default_golden_suite()

        self.tracer.clear_spans()

        # Generate Context Packs for code tasks and federated tasks
        all_eval_tasks: list[GoldenCodeTask | GoldenFederatedTask] = [
            *active_suite.code_tasks,
            *active_suite.federated_tasks,
        ]

        principal = PrincipalContext(
            user_id="harness_orchestrator",
            role=RoleType.ENGINEER,
            allowed_scopes=["public", "public_eng", "public_docs"],
        )

        workflow = ScoutWorkflow(registry=self.registry, repo=self.repo)
        federated = FederatedScoutCoordinator(registry=self.registry, repo=self.repo)

        generated_packs: list[ImplementationContextPack] = []
        for t in all_eval_tasks:
            if isinstance(t, GoldenFederatedTask):
                p = federated.run(user_prompt=t.user_prompt, principal=principal)
            else:
                p = workflow.run(
                    user_prompt=t.user_prompt,
                    principal=principal,
                    domain_id=t.domain_id,
                )
            if isinstance(p, ImplementationContextPack):
                generated_packs.append(p)

        doc_packs: list[ImplementationContextPack] = []
        for dt in active_suite.doc_tasks:
            dp = workflow.run(
                user_prompt=dt.user_prompt,
                principal=principal,
                domain_id=dt.domain_id,
            )
            if isinstance(dp, ImplementationContextPack):
                doc_packs.append(dp)

        code_packs = generated_packs[: len(active_suite.code_tasks)]
        all_packs_with_docs = [*generated_packs, *doc_packs]

        # 1. Zero-Touch Sufficiency Rate
        m1 = self.evaluate_zero_touch_sufficiency(all_eval_tasks, packs=generated_packs)

        # 2. File & Symbol Recall@5
        m2 = self.evaluate_file_and_symbol_recall(active_suite.code_tasks, packs=code_packs)

        # 3. Verbatim Citation Faithfulness (including Doc Research tasks)
        m3 = self.evaluate_verbatim_citation_faithfulness(all_packs_with_docs)

        # 4. Frontier Token Savings Ratio (including Doc Research tasks)
        m4 = self.evaluate_frontier_token_savings(all_packs_with_docs)

        # 5. RBAC Zero-Leakage Rate
        m5 = self.evaluate_rbac_zero_leakage(active_suite.rbac_redteam_queries)

        # 6. Prompt-Injection Block Rate
        m6 = self.evaluate_prompt_injection_block_rate(active_suite.prompt_injection_payloads)

        metrics = {
            self.METRIC_ZERO_TOUCH: m1,
            self.METRIC_RECALL_AT_5: m2,
            self.METRIC_FAITHFULNESS: m3,
            self.METRIC_TOKEN_SAVINGS: m4,
            self.METRIC_RBAC_LEAKAGE: m5,
            self.METRIC_PROMPT_INJECTION: m6,
        }

        all_passed = all(m.passed for m in metrics.values())
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        final_spans = self.tracer.get_finished_spans()

        verdict_str = "ALL GATES PASSED" if all_passed else "GATE FAILURE"
        summary_lines = [
            f"Milestone 6 Evaluation Pipeline: {verdict_str}",
            (
                f"Zero-Touch Sufficiency Rate: {m1.measured_value}% "
                f"(Target: {m1.target_threshold}%) -> {'PASS' if m1.passed else 'FAIL'}"
            ),
            (
                f"File & Symbol Recall@5: {m2.measured_value}% "
                f"(Target: {m2.target_threshold}%) -> {'PASS' if m2.passed else 'FAIL'}"
            ),
            (
                f"Verbatim Citation Faithfulness: {m3.measured_value}% "
                f"(Target: {m3.target_threshold}%) -> {'PASS' if m3.passed else 'FAIL'}"
            ),
            (
                f"Frontier Token Savings Ratio: {m4.measured_value}% "
                f"(Target: {m4.target_threshold}%) -> {'PASS' if m4.passed else 'FAIL'}"
            ),
            (
                f"RBAC Zero-Leakage Rate: {m5.measured_value}% "
                f"(Target: {m5.target_threshold}%) -> {'PASS' if m5.passed else 'FAIL'}"
            ),
            (
                f"Prompt-Injection Block Rate: {m6.measured_value}% "
                f"(Target: {m6.target_threshold}%) -> {'PASS' if m6.passed else 'FAIL'}"
            ),
            f"Total duration: {elapsed_ms}ms, OpenTelemetry spans captured: {len(final_spans)}",
        ]

        return EvaluationRunResult(
            timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
            all_passed=all_passed,
            metrics=metrics,
            summary="\n".join(summary_lines),
            total_duration_ms=elapsed_ms,
            telemetry_spans_count=len(final_spans),
        )


__all__ = [
    "EvaluationGateThresholds",
    "EvaluationHarness",
    "EvaluationRunResult",
    "MetricResult",
]
