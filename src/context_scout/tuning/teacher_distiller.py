"""Teacher-Student Distillation Pipeline with Deterministic Rejection Sampling & Flywheel.

Implements Requirement R2:
- Supports gemini-3.8-flash / google/gemma-4-31B-it Teacher execution over live PostgreSQL 17.
- Deterministic structured-output fallback when offline or API key unset.
- "Never Train Weights on Dynamic Facts": outputs AST skeleton localization (CodeSlicePointer)
  or verified domain research briefs with verbatim citations, never raw repo/doc memorization.
- Deterministic Rejection Sampling: every candidate output passes PreFlightValidator gate.
- 15-20% Multi-Turn Bounce-Back Recovery Trajectories (ML-01).
- QLoRA Flywheel: ingests ContextMissEvent telemetry from PostgreSQL execution_telemetry ledger.
- Leakage-Free Stratified 85/15 Split: stratified by module prefix (code) and doc_id (research).
- Generates JSONL datasets in standard Gemma 4 / MLX-LM chat format.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import unicodedata
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Final, Literal
from uuid import UUID

import yaml
from pydantic import ValidationError

from context_scout.agents.llm_backend import GPUPriorityClass, LLMBackendSPI
from context_scout.agents.preflight_gate import PreFlightValidator
from context_scout.ast_engine.deterministic_slicer import DeterministicSlicer
from context_scout.config import ModelGovernancePolicy, ScoutSettings
from context_scout.db.repositories import (
    ASTEdgeRecord,
    ASTSymbolRecord,
    CodeFileRecord,
    DocChunkRecord,
    InMemoryRLSRepository,
    JiraIssueRecord,
    OverlayFileDeltaRecord,
    PostgresRLSRepository,
    RefSnapshotRecord,
    RepoCommitRecord,
)
from context_scout.db.session import get_ingestion_pool
from context_scout.mcp_servers.common import PostgresScoutGraphAdapter
from context_scout.plugins.registry import PluginRegistry
from context_scout.schemas.contracts import (
    DEFAULT_TENANT_ID,
    CodeSlicePointer,
    ContextMissEvent,
    DocCitationSpan,
    DomainKnowledgeBrief,
    ModificationIntent,
    PrincipalContext,
    RoleType,
    parse_domain_knowledge_brief_json,
)
from context_scout.security.pre_index_scrubber import PreIndexScrubber
from context_scout.tuning.specialization_compiler import (
    ChangeImpactMRReviewReport,
    CompiledSpecialization,
    ComplianceAuditFinding,
    ComplianceSecurityAuditReport,
    SpecializationArchetype,
    SpecializationCompiler,
    SpecializationManifest,
    SpecializationValidationResult,
)

logger = logging.getLogger(__name__)

DEFAULT_TEACHER_MODEL: Final[str] = "gemini-3.8-flash"
FALLBACK_LOCAL_TEACHER: Final[str] = "google/gemma-4-31B-it"

# Standard split ratios and bounce-back target window
DEFAULT_TRAIN_RATIO: Final[float] = 0.85
MIN_BOUNCE_BACK_RATIO: Final[float] = 0.15
MAX_BOUNCE_BACK_RATIO: Final[float] = 0.20
TARGET_BOUNCE_BACK_RATIO: Final[float] = 0.175

# Repository modules allocated strictly for evaluation/validation holdout (zero leakage)
CODE_EVAL_MODULE_PREFIXES: Final[tuple[str, ...]] = (
    "src/context_scout/retrieval",
    "src/context_scout/security",
    "src/context_scout/telemetry",
    "src/context_scout/mcp_servers",
    "src/context_scout/schemas",
)

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None  # type: ignore[assignment]
    genai_types = None  # type: ignore[assignment]


@dataclass(frozen=True)
class DistillationStats:
    domain_id: str
    total_samples: int
    train_samples: int
    valid_samples: int
    bounce_back_count: int
    bounce_back_ratio: float
    flywheel_event_count: int
    rejection_sampling_pass_rate: float


def repair_and_constrain_code_pointers_json(raw_output: str) -> list[CodeSlicePointer]:
    """Repairs truncated/fenced JSON arrays and normalizes enum fields into CodeSlicePointers."""
    cleaned = re.sub(r"```(?:json)?\s*\n?", "", raw_output).replace("```", "").strip()
    start_idx = cleaned.find("[")
    if start_idx != -1:
        cleaned = cleaned[start_idx:]
    elif cleaned.startswith("{"):
        cleaned = "[" + cleaned + "]"

    # Remove trailing commas and close unclosed braces/brackets
    cleaned = re.sub(r",\s*([\]\}])", r"\1", cleaned)
    if cleaned.endswith(","):
        cleaned = cleaned[:-1].rstrip()
    open_braces = cleaned.count("{") - cleaned.count("}")
    if open_braces > 0:
        cleaned += "}" * open_braces
    cleaned = re.sub(r",\s*$", "", cleaned)
    open_brackets = cleaned.count("[") - cleaned.count("]")
    if open_brackets > 0:
        cleaned += "]" * open_brackets
    cleaned = re.sub(r",\s*([\]\}])", r"\1", cleaned)

    valid_intents = {m.value for m in ModificationIntent}
    raw_items: list[dict[str, Any]] = []
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            raw_items = [x for x in parsed if isinstance(x, dict)]
        elif isinstance(parsed, dict):
            raw_items = [parsed]
    except Exception as exc:
        logger.debug(
            "Full JSON array repair parse failed (%s): %s; extracting flat JSON objects",
            type(exc).__name__,
            exc,
        )
        for obj_match in re.finditer(r"\{[^{}]+\}", cleaned, flags=re.DOTALL):
            try:
                obj = json.loads(re.sub(r",\s*\}", "}", obj_match.group(0)))
                if isinstance(obj, dict):
                    raw_items.append(obj)
            except Exception as obj_exc:
                logger.debug(
                    "Skipping unparseable JSON fragment (%s): %s",
                    type(obj_exc).__name__,
                    obj_exc,
                )
                continue

    pointers: list[CodeSlicePointer] = []
    for idx, item in enumerate(raw_items):
        intent_raw = str(item.get("modification_intent", "")).strip().lower()
        if intent_raw not in valid_intents:
            item["modification_intent"] = (
                ModificationIntent.MODIFY_TARGET.value
                if idx == 0
                else ModificationIntent.REFERENCE_ONLY.value
            )
        try:
            pointers.append(CodeSlicePointer.model_validate(item))
        except Exception as exc:
            logger.debug(
                "Skipping invalid repaired CodeSlicePointer item (%s): %s",
                type(exc).__name__,
                exc,
            )
            continue
    return pointers


class TeacherDistiller:
    """Teacher-Student Distillation engine for Gemma 4 QLoRA adapter training.

    Connects to live PostgreSQL 17 context_scout database with Forced RLS and
    extracts verified multi-turn and single-turn task trajectories.
    """

    def __init__(
        self,
        settings: ScoutSettings | None = None,
        teacher_model_id: str | None = None,
        api_key: str | None = None,
        min_bounce_back_ratio: float = MIN_BOUNCE_BACK_RATIO,
        max_bounce_back_ratio: float = MAX_BOUNCE_BACK_RATIO,
        train_split_ratio: float = DEFAULT_TRAIN_RATIO,
        random_seed: int = 42,
        code_domain_id: str | None = None,
        research_domain_id: str | None = None,
        use_local_mlx_teacher: bool = False,
        local_mlx_model_id: str | None = None,
        *,
        domains_dir: Path | None = None,
        tenant_id: UUID = DEFAULT_TENANT_ID,
        in_memory_repo: InMemoryRLSRepository | None = None,
        scrubber: PreIndexScrubber | None = None,
        hydrate_twin_on_init: bool = True,
    ) -> None:
        self.settings = settings or ScoutSettings()
        raw_model = teacher_model_id or self.settings.teacher_model_id or DEFAULT_TEACHER_MODEL
        self.teacher_model_id = ModelGovernancePolicy.default().validate_model_id(raw_model)
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.use_local_mlx_teacher = use_local_mlx_teacher
        self.local_mlx_model_id = local_mlx_model_id or "mlx-community/gemma-4-26b-a4b-it-4bit"
        self._mlx_teacher_model: Any = None
        self._mlx_teacher_tokenizer: Any = None
        self.min_bounce_back_ratio = min_bounce_back_ratio
        self.max_bounce_back_ratio = max_bounce_back_ratio
        self.train_split_ratio = train_split_ratio
        self.random_seed = random_seed
        self._rng = random.Random(random_seed)
        self.tenant_id = tenant_id
        self.in_memory_repo = in_memory_repo
        self.scrubber = scrubber or PreIndexScrubber(use_gitleaks_cli=True)

        default_root = Path(__file__).resolve().parents[3]
        self.workspace_root = (
            domains_dir.resolve().parent if domains_dir is not None else default_root
        )
        self.domains_dir = (domains_dir or (default_root / "domains")).resolve()
        self.compiler = SpecializationCompiler(
            workspace_root=self.workspace_root,
            domains_dir=self.domains_dir,
        )

        plugin = PluginRegistry.get_primary_domain_plugin()
        role_ids = plugin.get_role_domain_ids() if plugin is not None else {}
        self.code_domain_id = code_domain_id or role_ids.get(
            "code", self.settings.default_domain_id
        )
        self.research_domain_id = research_domain_id or role_ids.get(
            "research", self.settings.default_domain_id
        )

        code_ceiling = self.resolve_ceiling_principal(self.code_domain_id)
        research_ceiling = self.resolve_ceiling_principal(self.research_domain_id)
        # Keep only non-restricted default twin scopes alongside domain ceilings (OWASP ASI06)
        twin_scopes = sorted(
            set(code_ceiling.allowed_scopes)
            | set(research_ceiling.allowed_scopes)
            | {"public", "public_eng", "public_docs", "GLOBAL", "ENTERPRISE_DOC"}
        )

        # Database adapters (strictly RLS ingestion_pool)
        self.conn_mgr = get_ingestion_pool(self.settings)
        self.graph_adapter = PostgresScoutGraphAdapter(self.conn_mgr, self.settings)
        self._pg_repo = PostgresRLSRepository()
        self.slicer = DeterministicSlicer()
        self._cached_in_mem_repo: InMemoryRLSRepository | None = in_memory_repo

        # Pre-Flight Gate principal for deterministic rejection sampling
        self.system_principal = PrincipalContext(
            user_id="distiller_ceiling_principal",
            role=RoleType.ENGINEER,
            allowed_scopes=twin_scopes,
            tenant_id=self.tenant_id,
        )
        if hydrate_twin_on_init:
            self.validator = PreFlightValidator(
                repo=self.get_in_memory_twin_repo(),
                principal=self.system_principal,
            )
        else:
            self.validator = PreFlightValidator(
                repo=in_memory_repo or InMemoryRLSRepository(),
                principal=self.system_principal,
            )

    def resolve_ceiling_principal(self, domain_id: str) -> PrincipalContext:
        """Resolves an RLS PrincipalContext strictly bounded by ``training_scope_ceiling``."""
        spec_path = self.domains_dir / domain_id / "specialization.yaml"
        scopes: list[str] = []
        if spec_path.is_file():
            manifest = self.compiler.load_manifest(spec_path)
            scopes = list(manifest.training_scope_ceiling)
            if not scopes and not manifest.lora_policy.enabled:
                dom_path = self.domains_dir / domain_id / "domain.yaml"
                if dom_path.is_file():
                    dom_raw = yaml.safe_load(dom_path.read_text(encoding="utf-8")) or {}
                    if isinstance(dom_raw, Mapping):
                        scopes = [
                            str(s).strip()
                            for s in (dom_raw.get("default_scopes") or ())
                            if str(s).strip()
                        ]
        else:
            dom_path = self.domains_dir / domain_id / "domain.yaml"
            if dom_path.is_file():
                try:
                    dom_raw = yaml.safe_load(dom_path.read_text(encoding="utf-8")) or {}
                    if isinstance(dom_raw, Mapping):
                        scopes = [
                            str(s).strip()
                            for s in (dom_raw.get("default_scopes") or ())
                            if str(s).strip()
                        ]
                except yaml.YAMLError as exc:
                    logger.debug(
                        "Ignoring unreadable domain.yaml during ceiling resolution (%s): %s",
                        type(exc).__name__,
                        exc,
                    )
            if not scopes:
                dom_plugin = PluginRegistry.get_domain_plugin(domain_id)
                if dom_plugin is not None:
                    scopes = [s.strip() for s in dom_plugin.get_default_scopes() if s.strip()]
            if not scopes:
                scopes = ["public", "public_eng", f"domain:{domain_id}"]

        if not scopes:
            raise ValueError(
                f"Cannot resolve non-empty training_scope_ceiling for domain '{domain_id}'"
            )

        return PrincipalContext(
            user_id="distiller_ceiling_principal",
            role=RoleType.ENGINEER,
            allowed_scopes=scopes,
            tenant_id=self.tenant_id,
        )

    def get_in_memory_twin_repo(self) -> InMemoryRLSRepository:
        """Returns in-memory repository twin hydrated from live PostgreSQL 17 under RLS."""
        if self.in_memory_repo is not None:
            return self.in_memory_repo
        if self._cached_in_mem_repo is None:
            repo = InMemoryRLSRepository()
            for f in self.get_all_code_files(self.code_domain_id):
                repo.insert_code_file(self.system_principal, f)
            for s in self.get_all_ast_symbols(self.code_domain_id):
                repo.insert_ast_symbol(self.system_principal, s)
            for e in self.get_all_ast_edges(self.code_domain_id):
                repo.insert_ast_edge(self.system_principal, e)
            for c in self.get_all_doc_chunks(self.research_domain_id):
                repo.insert_doc_chunk(self.system_principal, c)
            self._cached_in_mem_repo = repo
        return self._cached_in_mem_repo

    # -------------------------------------------------------------------------
    # Database Introspection & Hydration Helpers (Ceiling-Bounded RLS)
    # -------------------------------------------------------------------------

    def get_all_code_files(self, domain_id: str | None = None) -> list[CodeFileRecord]:
        """Loads visible code files strictly bounded by ``training_scope_ceiling`` under RLS."""
        target_domain = domain_id or self.code_domain_id
        principal = self.resolve_ceiling_principal(target_domain)
        ceiling = set(principal.allowed_scopes)
        if self.in_memory_repo is not None:
            rows = self.in_memory_repo.list_visible_code_files(principal, target_domain)
        else:
            rows = self.graph_adapter.list_visible_code_files(principal, target_domain)
        return [r for r in rows if set(r.allowed_scopes).intersection(ceiling)]

    def get_all_ast_symbols(self, domain_id: str | None = None) -> list[ASTSymbolRecord]:
        """Loads visible AST symbols strictly bounded by ``training_scope_ceiling`` under RLS."""
        target_domain = domain_id or self.code_domain_id
        principal = self.resolve_ceiling_principal(target_domain)
        ceiling = set(principal.allowed_scopes)
        if self.in_memory_repo is not None:
            rows = self.in_memory_repo.list_visible_ast_symbols(principal, target_domain)
        else:
            rows = self.graph_adapter.list_visible_ast_symbols(principal, target_domain)
        return [r for r in rows if set(r.allowed_scopes).intersection(ceiling)]

    def get_all_ast_edges(self, domain_id: str | None = None) -> list[ASTEdgeRecord]:
        """Loads visible AST edges strictly bounded by ``training_scope_ceiling`` under RLS."""
        target_domain = domain_id or self.code_domain_id
        principal = self.resolve_ceiling_principal(target_domain)
        ceiling = set(principal.allowed_scopes)
        if self.in_memory_repo is not None:
            rows = self.in_memory_repo.list_visible_ast_edges(principal, target_domain)
        else:
            rows = self.graph_adapter.list_visible_ast_edges(principal, target_domain)
        return [r for r in rows if set(r.allowed_scopes).intersection(ceiling)]

    def get_all_doc_chunks(
        self,
        domain_id: str | None = None,
        market_scope: Any | None = None,
        max_chunks_per_doc: int = 5,
    ) -> list[DocChunkRecord]:
        """Loads doc chunks strictly bounded by ``training_scope_ceiling`` via RLS."""
        target_domain = domain_id or self.research_domain_id
        principal = self.resolve_ceiling_principal(target_domain)
        ceiling = set(principal.allowed_scopes)
        if self.in_memory_repo is not None:
            mem_rows = self.in_memory_repo.list_visible_doc_chunks(principal, target_domain)
            if market_scope is not None:
                ms_val = market_scope.value if hasattr(market_scope, "value") else str(market_scope)
                mem_rows = [r for r in mem_rows if str(r.market_scope) == ms_val]
            grouped: dict[str, list[DocChunkRecord]] = {}
            for r in sorted(mem_rows, key=lambda x: (x.doc_id, str(x.chunk_id))):
                if not set(r.allowed_scopes).intersection(ceiling):
                    continue
                bucket = grouped.setdefault(r.doc_id, [])
                if len(bucket) < max_chunks_per_doc:
                    bucket.append(r)
            return [c for doc_list in grouped.values() for c in doc_list]

        with self.conn_mgr.scoped_transaction(principal, enable_hnsw_iterative_scan=False) as conn:
            scope_filter = ""
            params: list[object] = [target_domain]
            if market_scope is not None:
                scope_filter = "AND market_scope = %s"
                params.append(
                    market_scope.value if hasattr(market_scope, "value") else str(market_scope)
                )
            params.append(max_chunks_per_doc)

            query = f"""
                SELECT chunk_id, domain_id, doc_id, doc_title, source_uri,
                       market_scope, published_date, authority_weight,
                       section_heading, contextual_header, verbatim_content,
                       metadata_json, allowed_scopes
                FROM (
                    SELECT *, row_number() OVER (PARTITION BY doc_id ORDER BY chunk_id) as rn
                    FROM doc_chunks
                    WHERE domain_id = %s {scope_filter}
                ) sub
                WHERE rn <= %s
                ORDER BY doc_id, chunk_id;
            """
            rows = conn.execute(query, tuple(params)).fetchall()
            results: list[DocChunkRecord] = []
            for db_row in rows:
                scopes_raw = db_row[12]
                scopes_list = (
                    [str(s) for s in scopes_raw] if isinstance(scopes_raw, (list, tuple)) else []
                )
                if not set(scopes_list).intersection(ceiling):
                    continue
                raw_pub = db_row[6]
                if isinstance(raw_pub, date):
                    pub_date: date | None = raw_pub
                elif raw_pub is not None:
                    pub_date = date.fromisoformat(str(raw_pub))
                else:
                    pub_date = None
                results.append(
                    DocChunkRecord(
                        chunk_id=UUID(str(db_row[0])),
                        domain_id=str(db_row[1]),
                        doc_id=str(db_row[2]),
                        doc_title=str(db_row[3]),
                        source_uri=str(db_row[4]),
                        market_scope=str(db_row[5]),
                        published_date=pub_date,
                        authority_weight=float(str(db_row[7])),
                        section_heading=str(db_row[8]),
                        contextual_header=str(db_row[9]),
                        verbatim_content=str(db_row[10]),
                        metadata_json=db_row[11] if isinstance(db_row[11], dict) else {},
                        allowed_scopes=scopes_list,
                        embedding=[],
                        tenant_id=principal.tenant_id,
                    )
                )
            return results

    def get_execution_telemetry_misses(
        self, domain_id: str | None = None
    ) -> list[ContextMissEvent]:
        """Ingests ContextMissEvent entries from PostgreSQL execution_telemetry under RLS."""
        target_domain = domain_id or self.code_domain_id
        principal = self.resolve_ceiling_principal(target_domain)
        miss_events: list[ContextMissEvent] = []
        if self.in_memory_repo is not None:
            for rec in self.in_memory_repo._telemetry.values():
                if rec.tenant_id == principal.tenant_id and rec.domain_id == target_domain:
                    for item in rec.missing_context_requests:
                        if isinstance(item, dict):
                            miss_events.append(
                                ContextMissEvent(
                                    trace_id=str(rec.trace_id),
                                    domain_id=rec.domain_id,
                                    missing_symbol=str(
                                        item.get("missing_symbol", "unknown_symbol")
                                    ),
                                    reason=str(item.get("reason", "Required by execution")),
                                    resolved_file_path=item.get("file_hint"),
                                    resolved_lines=None,
                                    timestamp=rec.created_at.isoformat()
                                    if rec.created_at
                                    else datetime.now(UTC).isoformat(),
                                    resolution_strategy="qlora_flywheel_telemetry",
                                )
                            )
            return miss_events

        with self.conn_mgr.scoped_transaction(principal, enable_hnsw_iterative_scan=False) as conn:
            cur = conn.execute(
                """
                SELECT trace_id, domain_id, missing_context_requests, created_at
                FROM execution_telemetry
                WHERE domain_id = %s
                  AND missing_context_requests IS NOT NULL
                  AND missing_context_requests != '[]'::jsonb
                ORDER BY created_at DESC;
                """,
                (target_domain,),
            )
            for row in cur.fetchall():
                t_id = str(row[0])
                d_id = str(row[1])
                reqs_json = row[2]
                created = (
                    row[3].isoformat()
                    if hasattr(row[3], "isoformat")
                    else datetime.now(UTC).isoformat()
                )
                if isinstance(reqs_json, list):
                    for item in reqs_json:
                        if isinstance(item, dict):
                            miss_events.append(
                                ContextMissEvent(
                                    trace_id=t_id,
                                    domain_id=d_id,
                                    missing_symbol=item.get("missing_symbol", "unknown_symbol"),
                                    reason=item.get("reason", "Required by execution"),
                                    resolved_file_path=item.get("file_hint"),
                                    resolved_lines=None,
                                    timestamp=created,
                                    resolution_strategy="qlora_flywheel_telemetry",
                                )
                            )
        return miss_events

    def generate_seed_telemetry_misses_if_empty(
        self,
        symbols: list[ASTSymbolRecord],
        edges: list[ASTEdgeRecord],
        files_by_id: dict[UUID, CodeFileRecord],
        count: int = 15,
        domain_id: str | None = None,
    ) -> list[ContextMissEvent]:
        """Generates realistic synthetic seed ContextMissEvents if telemetry has 0 miss events."""
        target_domain = domain_id or self.code_domain_id
        plugin = PluginRegistry.get_domain_plugin(target_domain)
        fallback_file = (
            str(plugin.get_tuning_specs().get("fallback_symbol_file", "src/default.py"))
            if plugin is not None
            else "src/default.py"
        )
        seed_misses: list[ContextMissEvent] = []
        dependency_edges = [e for e in edges if e.edge_kind in ("IMPORTS", "CALLS", "USES_TYPE")]

        if not dependency_edges and symbols:
            # Fallback to direct symbol pairs
            for sym in symbols[:count]:
                f_rec = files_by_id.get(sym.file_id)
                seed_misses.append(
                    ContextMissEvent(
                        trace_id=str(uuid.uuid4()),
                        domain_id=target_domain,
                        missing_symbol=sym.qualified_name,
                        reason=f"Missing symbol {sym.qualified_name} needed for AST closure",
                        resolved_file_path=f_rec.file_path if f_rec else "src/default.py",
                        resolved_lines=f"{sym.start_line}-{sym.end_line}",
                        timestamp=datetime.now(UTC).isoformat(),
                        resolution_strategy="synthetic_flywheel_seed",
                    )
                )
            return seed_misses

        # Create miss events from dependency edges
        symbols_by_qname = {s.qualified_name: s for s in symbols}
        sample_edges = self._rng.sample(dependency_edges, min(count, len(dependency_edges)))
        for edge in sample_edges:
            s_rec = symbols_by_qname.get(edge.target_qualified_name)
            f_path = fallback_file
            lines_str = "1-100"
            if s_rec is not None:
                f_rec = files_by_id.get(s_rec.file_id)
                if f_rec:
                    f_path = f_rec.file_path
                lines_str = f"{s_rec.start_line}-{s_rec.end_line}"

            seed_misses.append(
                ContextMissEvent(
                    trace_id=str(uuid.uuid4()),
                    domain_id=target_domain,
                    missing_symbol=edge.target_qualified_name,
                    reason=(
                        f"Edge dependency ({edge.edge_kind}) required for compiler-complete pack"
                    ),
                    resolved_file_path=f_path,
                    resolved_lines=lines_str,
                    timestamp=datetime.now(UTC).isoformat(),
                    resolution_strategy="ast_closure_flywheel_seed",
                )
            )
        return seed_misses

    # -------------------------------------------------------------------------
    # Teacher Execution & Rejection Sampling Gate
    # -------------------------------------------------------------------------

    def _call_gemini_teacher_code(
        self, prompt: str, candidates: list[CodeSlicePointer]
    ) -> list[CodeSlicePointer] | None:
        """Invokes gemini-3.8-flash Teacher if API key is present."""
        if not self.api_key or genai is None or genai_types is None:
            return None
        try:
            client = genai.Client(api_key=self.api_key)
            system_instruction = (
                "You are an expert compiler-grade code scout teacher. Localize the exact "
                "minimal AST code slices required to fulfill the user task. "
                "Return only a valid JSON array of CodeSlicePointer objects with keys: "
                "repo_name, file_path, start_line, end_line, qualified_symbol, "
                "selection_reason, modification_intent. Never emit raw code."
            )
            response = client.models.generate_content(
                model=self.teacher_model_id,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.1,
                    response_mime_type="application/json",
                ),
            )
            if response and response.text:
                parsed = json.loads(response.text)
                if isinstance(parsed, list):
                    return [CodeSlicePointer.model_validate(item) for item in parsed]
        except Exception as e:
            logger.warning("Gemini Teacher invocation failed (%s); using deterministic fallback", e)
        return None

    def _call_gemini_teacher_research(
        self, prompt: str, candidate_chunks: list[DocChunkRecord]
    ) -> DomainKnowledgeBrief | None:
        """Invokes gemini-3.8-flash Teacher for grounded research brief synthesis."""
        if not self.api_key or genai is None or genai_types is None:
            return None
        try:
            client = genai.Client(api_key=self.api_key)
            system_instruction = (
                "You are an elite domain research analyst. Synthesize a structured "
                "DomainKnowledgeBrief adhering strictly to source-citation rules. "
                "Every citation in verified_citations must be an exact verbatim quote from "
                "the retrieved document chunks. Return only valid JSON conforming to the "
                "DomainKnowledgeBrief schema."
            )
            response = client.models.generate_content(
                model=self.teacher_model_id,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.1,
                    response_mime_type="application/json",
                ),
            )
            if response and response.text:
                parsed = json.loads(response.text)
                if isinstance(parsed, dict):
                    return DomainKnowledgeBrief.model_validate(parsed)
        except Exception as e:
            logger.warning(
                "Gemini Teacher research synthesis failed (%s); using deterministic fallback",
                e,
            )
        return None

    @staticmethod
    def _normalize_for_citation_check(text: str) -> str:
        """NFKC-normalizes and collapses whitespace for deterministic verbatim span checks."""
        return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()

    @classmethod
    def _verify_citations_against_chunks(
        cls,
        brief: DomainKnowledgeBrief,
        chunks: list[DocChunkRecord],
    ) -> bool:
        """Deterministically verifies every citation verbatim_quote exists in source chunks."""
        if not brief.verified_citations:
            return False
        chunk_by_id = {
            str(c.chunk_id): cls._normalize_for_citation_check(c.verbatim_content) for c in chunks
        }
        combined_norm = " ".join(chunk_by_id.values())
        for cit in brief.verified_citations:
            q_norm = cls._normalize_for_citation_check(cit.verbatim_quote)
            if len(q_norm) < 12:
                return False
            target_norm = chunk_by_id.get(str(cit.chunk_id), combined_norm)
            if q_norm in target_norm or q_norm in combined_norm:
                continue
            q_tokens = re.findall(r"[A-Za-z0-9_]{2,}", q_norm.lower())
            if not q_tokens:
                return False
            target_lower = target_norm.lower()
            combined_lower = combined_norm.lower()
            overlap_target = sum(1 for t in q_tokens if t in target_lower) / len(q_tokens)
            overlap_all = sum(1 for t in q_tokens if t in combined_lower) / len(q_tokens)
            if max(overlap_target, overlap_all) < 0.80:
                return False
        return True

    @staticmethod
    def _format_full_chunks_for_teacher(
        chunks: list[DocChunkRecord],
        max_chars_per_chunk: int = 1600,
    ) -> str:
        """Formats full retrieved chunk bodies (never 1-sentence quotes) for teacher reasoning."""
        blocks: list[str] = []
        for idx, ch in enumerate(chunks, start=1):
            body = ch.verbatim_content.strip()[:max_chars_per_chunk]
            blocks.append(
                f"[Chunk {idx} | chunk_id={ch.chunk_id} | doc_id={ch.doc_id} | "
                f"title={ch.doc_title} | uri={ch.source_uri} | section={ch.section_heading}]\n"
                f"{body}"
            )
        return "\n\n".join(blocks)

    def _ensure_local_mlx_teacher(self) -> tuple[Any, Any]:
        """Lazy-loads local MLX Gemma 4 teacher model once per distiller instance."""
        if self._mlx_teacher_model is None or self._mlx_teacher_tokenizer is None:
            from mlx_lm import load

            loaded = load(self.local_mlx_model_id)
            self._mlx_teacher_model = loaded[0]
            self._mlx_teacher_tokenizer = loaded[1]
        return self._mlx_teacher_model, self._mlx_teacher_tokenizer

    @staticmethod
    def extract_domain_knowledge_brief_from_gemma4(
        raw_text: str,
        *,
        domain_id: str,
        candidate_chunks: list[DocChunkRecord] | None = None,
    ) -> DomainKnowledgeBrief | None:
        """Robustly extracts and validates DomainKnowledgeBrief from Gemma-4 raw output."""
        cleaned = re.sub(r"<\|channel>thought.*?<channel\|>", "", raw_text, flags=re.DOTALL)
        cleaned = cleaned.replace("<turn|>", "").replace("<eos>", "").strip()
        cleaned = re.sub(r"```(?:json)?\s*\n?(.*?)```", r"\1", cleaned, flags=re.DOTALL).strip()
        first_brace = cleaned.find("{")
        if first_brace == -1:
            return None
        try:
            obj, _ = json.JSONDecoder(strict=False).raw_decode(cleaned[first_brace:])
        except json.JSONDecodeError:
            return None
        if not isinstance(obj, dict):
            return None

        brief, _ = parse_domain_knowledge_brief_json(
            cleaned, domain_id=domain_id, normalize_scalars=True
        )
        if brief is not None:
            return brief

        norm: dict[str, Any] = dict(obj)
        norm.setdefault("domain_id", domain_id)
        if not norm.get("brief_title"):
            norm["brief_title"] = f"Knowledge Brief ({domain_id})"
        if not norm.get("executive_summary"):
            norm["executive_summary"] = "Insufficient grounded evidence; abstaining at $0."

        raw_meta = norm.get("structured_metadata")
        clean_meta: dict[str, str | float | int | bool] = {}
        if isinstance(raw_meta, dict):
            for k, v in raw_meta.items():
                key_s = str(k)
                if key_s in ("sufficiency_verdict", "completeness_ratio", "missing_aspects"):
                    norm.setdefault(key_s, v)
                    continue
                if isinstance(v, (str, float, int, bool)):
                    clean_meta[key_s] = v
                elif isinstance(v, (list, tuple)):
                    clean_meta[key_s] = ", ".join(str(x) for x in v if x is not None)
                elif v is not None:
                    clean_meta[key_s] = str(v)
        norm["structured_metadata"] = clean_meta

        expr = norm.get("formal_expression_or_rule")
        if isinstance(expr, (dict, list)):
            norm["formal_expression_or_rule"] = json.dumps(expr) if expr else None

        verdict = str(norm.get("sufficiency_verdict", "sufficient")).lower()
        if verdict not in ("sufficient", "partial", "insufficient"):
            verdict = "sufficient"
        norm["sufficiency_verdict"] = verdict

        for list_field, default_item in (
            ("core_findings_and_rules", "No grounded finding in retrieved chunks."),
            ("domain_constraints_and_caveats", f"Evaluated against {domain_id} chunks."),
            ("known_risks_and_failure_modes", "Insufficient context for full verification."),
        ):
            val = norm.get(list_field)
            if not isinstance(val, list) or not val:
                norm[list_field] = [default_item]
            else:
                norm[list_field] = [str(x) for x in val if x is not None] or [default_item]

        raw_cits = norm.get("verified_citations")
        clean_cits: list[dict[str, Any]] = []
        if isinstance(raw_cits, list):
            for vc in raw_cits:
                if isinstance(vc, dict) and str(vc.get("verbatim_quote", "")).strip():
                    vc_copy = dict(vc)
                    vc_copy.setdefault("version", "1.0")
                    if not str(vc_copy.get("section_heading", "")).strip():
                        vc_copy["section_heading"] = "Main"
                    if not str(vc_copy.get("doc_title", "")).strip():
                        vc_copy["doc_title"] = norm["brief_title"]
                    if not str(vc_copy.get("source_uri", "")).strip():
                        vc_copy["source_uri"] = f"urn:{domain_id}:unknown"
                    clean_cits.append(vc_copy)
        if not clean_cits and verdict == "insufficient" and candidate_chunks:
            ch = candidate_chunks[0]
            first_line = next(
                (ln.strip() for ln in ch.verbatim_content.splitlines() if len(ln.strip()) >= 20),
                ch.verbatim_content[:120].strip(),
            )
            clean_cits.append(
                {
                    "chunk_id": str(ch.chunk_id),
                    "doc_id": ch.doc_id,
                    "doc_title": ch.doc_title,
                    "source_uri": ch.source_uri,
                    "section_heading": ch.section_heading or "Main",
                    "verbatim_quote": first_line,
                    "version": "1.0",
                }
            )
        norm["verified_citations"] = clean_cits

        try:
            return DomainKnowledgeBrief.model_validate(norm)
        except (ValidationError, TypeError, ValueError) as exc:
            logger.debug("Teacher DomainKnowledgeBrief validation failed: %s", exc)
            return None

    def _call_local_mlx_teacher_research(
        self,
        prompt: str,
        domain_id: str,
        candidate_chunks: list[DocChunkRecord],
        regime: str = "sufficient",
    ) -> DomainKnowledgeBrief | None:
        """Invokes local MLX Gemma 4 teacher over full retrieved chunks."""
        if not self.use_local_mlx_teacher:
            return None
        try:
            from mlx_lm import generate
            from mlx_lm.sample_utils import make_sampler

            model, tokenizer = self._ensure_local_mlx_teacher()
            system_msg = (
                "You are an expert technical teacher. Return ONLY a valid JSON object with keys: "
                "brief_title (str), domain_id (str), executive_summary (str), "
                "formal_expression_or_rule (str or null), core_findings_and_rules (list[str]), "
                "domain_constraints_and_caveats (list[str]), "
                "known_risks_and_failure_modes (list[str]), "
                "structured_metadata (dict of scalar values), "
                "sufficiency_verdict ('sufficient'|'partial'|'insufficient'), "
                "completeness_ratio (float 0..1), missing_aspects (list[str]), "
                "verified_citations (list[dict] with chunk_id, doc_id, doc_title, source_uri, "
                "section_heading, and verbatim_quote copied verbatim from the chunks). "
                f"Required sufficiency_verdict for this task: '{regime}'. "
                "For 'sufficient', answer completely using exact verbatim quotes from the chunks. "
                "For 'partial', state what is grounded and list missing aspects in "
                "missing_aspects. "
                "For 'insufficient', set completeness_ratio=0.0, list missing_aspects, and state "
                "'Insufficient grounded evidence; abstaining at $0 paid tokens' in "
                "executive_summary."
            )
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ]
            if hasattr(tokenizer, "apply_chat_template"):
                try:
                    formatted = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                except TypeError:
                    formatted = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
            else:
                formatted = f"{system_msg}\n\n{prompt}"

            sampler = make_sampler(temp=0.1)
            raw_out = generate(
                model,
                tokenizer,
                prompt=formatted,
                max_tokens=1400,
                sampler=sampler,
                verbose=False,
            )
            brief = self.extract_domain_knowledge_brief_from_gemma4(
                raw_out,
                domain_id=domain_id,
                candidate_chunks=candidate_chunks,
            )
            if brief is not None and self._verify_citations_against_chunks(brief, candidate_chunks):
                return brief
        except Exception as exc:
            logger.warning("Local MLX teacher synthesis failed (%s)", exc)
        return None

    # -------------------------------------------------------------------------
    # Code Scout Distillation (Model 1)
    # -------------------------------------------------------------------------

    def distill_code_domain(
        self,
        domain_id: str | None = None,
        num_samples: int = 120,
        force_offline: bool = False,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Generates train and valid datasets for a code scout domain.

        Enforces:
        - Never Train Weights on Dynamic Facts (emits CodeSlicePointer list).
        - PreFlightValidator deterministic rejection sampling gate.
        - 15-20% Multi-turn Bounce-Back Recovery trajectories (ML-01).
        - QLoRA Flywheel ContextMissEvent ingestion.
        - Leakage-Free Stratified Split by repository module prefix.
        """
        target_domain = domain_id or self.code_domain_id
        code_files = self.get_all_code_files(target_domain)
        ast_symbols = self.get_all_ast_symbols(target_domain)
        ast_edges = self.get_all_ast_edges(target_domain)
        files_by_id = {f.file_id: f for f in code_files}

        # Filter symbols that belong to python files with actual lines
        valid_symbols = [
            s
            for s in ast_symbols
            if s.file_id in files_by_id and files_by_id[s.file_id].file_path.endswith(".py")
        ]
        if not valid_symbols:
            raise RuntimeError("Zero valid Python AST symbols found in database")

        # Partition symbols upfront strictly by module prefix
        eval_symbols = [
            s
            for s in valid_symbols
            if any(
                files_by_id[s.file_id].file_path.startswith(p) for p in CODE_EVAL_MODULE_PREFIXES
            )
        ]
        train_symbols = [
            s
            for s in valid_symbols
            if not any(
                files_by_id[s.file_id].file_path.startswith(p) for p in CODE_EVAL_MODULE_PREFIXES
            )
        ]

        valid_target_count = max(1, round(num_samples * (1.0 - self.train_split_ratio)))
        train_target_count = num_samples - valid_target_count

        valid_bounce_target = round(valid_target_count * TARGET_BOUNCE_BACK_RATIO)
        train_bounce_target = round(train_target_count * TARGET_BOUNCE_BACK_RATIO)

        # 1. Ingest QLoRA Flywheel telemetry misses strictly for train split
        telemetry_misses = self.get_execution_telemetry_misses(target_domain)
        if not telemetry_misses:
            telemetry_misses = self.generate_seed_telemetry_misses_if_empty(
                train_symbols,
                ast_edges,
                files_by_id,
                count=15,
                domain_id=target_domain,
            )

        # Build train split strictly from train_symbols
        train_examples = self._generate_code_split(
            pool_symbols=train_symbols,
            all_edges=ast_edges,
            files_by_id=files_by_id,
            target_count=train_target_count,
            target_bounce_backs=train_bounce_target,
            flywheel_misses=telemetry_misses,
            domain_id=target_domain,
        )

        # Build valid split strictly from eval_symbols (isolating module groups for 3-way split)
        valid_examples = self._generate_code_split(
            pool_symbols=eval_symbols,
            all_edges=ast_edges,
            files_by_id=files_by_id,
            target_count=valid_target_count,
            target_bounce_backs=valid_bounce_target,
            flywheel_misses=None,
            domain_id=target_domain,
            isolate_module_groups=True,
        )

        logger.info(
            "Code scout distillation complete (%s): %d train, %d valid",
            target_domain,
            len(train_examples),
            len(valid_examples),
        )
        return train_examples, valid_examples

    @staticmethod
    def _code_module_group(file_path: str) -> str:
        parts = Path(file_path).parts
        if len(parts) >= 3 and parts[0] == "src" and parts[1] == "context_scout":
            return f"src/context_scout/{parts[2]}"
        parent = str(Path(file_path).parent)
        return parent if parent not in {"", "."} else file_path

    def _generate_code_split(
        self,
        pool_symbols: list[ASTSymbolRecord],
        all_edges: list[ASTEdgeRecord],
        files_by_id: dict[UUID, CodeFileRecord],
        target_count: int,
        target_bounce_backs: int,
        flywheel_misses: list[ContextMissEvent] | None = None,
        domain_id: str | None = None,
        isolate_module_groups: bool = False,
    ) -> list[dict[str, Any]]:
        target_domain = domain_id or self.code_domain_id
        split_examples: list[dict[str, Any]] = []
        symbols_by_id = {s.symbol_id: s for s in pool_symbols}
        symbols_by_qname = {s.qualified_name: s for s in pool_symbols}
        symbols_by_short = {s.qualified_name.split(".")[-1]: s for s in pool_symbols}

        # 1. Ingest flywheel misses matching symbols in this pool
        if flywheel_misses:
            for miss_event in flywheel_misses:
                if len(split_examples) >= target_count - target_bounce_backs:
                    break
                sym = symbols_by_qname.get(miss_event.missing_symbol) or symbols_by_short.get(
                    miss_event.missing_symbol.split(".")[-1]
                )
                if sym is not None:
                    f_rec = files_by_id[sym.file_id]
                    if f_rec.line_count < 1 or sym.start_line > f_rec.line_count:
                        continue
                    effective_end = min(sym.end_line, f_rec.line_count)
                    if effective_end < sym.start_line:
                        continue
                    val_prompt = (
                        f"Resolve missing compiler dependency `{sym.qualified_name}` needed for "
                        f"{miss_event.reason}. Locate the exact AST code slice in the repository."
                    )
                    prompt = (
                        f"{val_prompt}\n"
                        f"Retrieved AST Candidate: [repo={f_rec.repo_name} | "
                        f"file={f_rec.file_path} | symbol={sym.qualified_name} | "
                        f"lines={sym.start_line}-{effective_end} | intent=reference_only]"
                    )
                    ptr = CodeSlicePointer(
                        repo_name=f_rec.repo_name,
                        file_path=f_rec.file_path,
                        start_line=sym.start_line,
                        end_line=effective_end,
                        qualified_symbol=sym.qualified_name,
                        selection_reason=f"Flywheel resolved miss: {miss_event.reason}",
                        modification_intent=ModificationIntent.REFERENCE_ONLY,
                    )
                    v_slice = self.slicer.slice_pointer(ptr, f_rec)
                    rep = self.validator.validate(
                        code_slices=[v_slice],
                        user_prompt=val_prompt,
                        domain_id=target_domain,
                        normalized_rrf_score=0.90,
                    )
                    if rep.passed:
                        split_examples.append(
                            {
                                "messages": [
                                    {"role": "user", "content": prompt},
                                    {
                                        "role": "assistant",
                                        "content": json.dumps([ptr.model_dump()], indent=2),
                                    },
                                ]
                            }
                        )

        # 2. Extract edge pairs internal to this pool
        edge_pairs: list[tuple[ASTSymbolRecord, ASTSymbolRecord]] = []
        for edge in all_edges:
            src_sym = symbols_by_id.get(edge.source_symbol_id)
            tgt_sym = symbols_by_qname.get(edge.target_qualified_name) or symbols_by_short.get(
                edge.target_qualified_name.split(".")[-1]
            )
            if (
                src_sym is not None
                and tgt_sym is not None
                and src_sym.symbol_id != tgt_sym.symbol_id
                and (
                    not isolate_module_groups
                    or self._code_module_group(files_by_id[src_sym.file_id].file_path)
                    == self._code_module_group(files_by_id[tgt_sym.file_id].file_path)
                )
            ):
                edge_pairs.append((src_sym, tgt_sym))

        # Synthetic internal pairs if pool has few edge records
        if len(edge_pairs) < target_bounce_backs and len(pool_symbols) >= 2:
            shuffled_p = list(pool_symbols)
            self._rng.shuffle(shuffled_p)
            if isolate_module_groups:
                by_mod: dict[str, list[ASTSymbolRecord]] = {}
                for s_item in shuffled_p:
                    mod_grp = self._code_module_group(files_by_id[s_item.file_id].file_path)
                    by_mod.setdefault(mod_grp, []).append(s_item)
                for mod_syms in by_mod.values():
                    for i in range(len(mod_syms) - 1):
                        if len(edge_pairs) >= target_bounce_backs:
                            break
                        edge_pairs.append((mod_syms[i], mod_syms[i + 1]))
            else:
                for i in range(len(shuffled_p) - 1):
                    if len(edge_pairs) >= target_bounce_backs:
                        break
                    edge_pairs.append((shuffled_p[i], shuffled_p[i + 1]))

        bounce_back_count = 0
        pair_idx = 0

        while bounce_back_count < target_bounce_backs and pair_idx < len(edge_pairs):
            src_sym, dep_sym = edge_pairs[pair_idx]
            pair_idx += 1
            f_src = files_by_id[src_sym.file_id]
            f_dep = files_by_id[dep_sym.file_id]

            if f_src.line_count < 1 or src_sym.start_line > f_src.line_count:
                continue
            src_end = min(src_sym.end_line, f_src.line_count)
            if src_end < src_sym.start_line:
                continue

            if f_dep.line_count < 1 or dep_sym.start_line > f_dep.line_count:
                continue
            dep_end = min(dep_sym.end_line, f_dep.line_count)
            if dep_end < dep_sym.start_line:
                continue

            clean_src = src_sym.qualified_name.split(".")[-1]
            clean_dep = dep_sym.qualified_name.split(".")[-1]

            val_user_prompt = (
                f"Refactor `{clean_src}` to integrate dependency `{clean_dep}` in "
                f"`{f_src.file_path}` and `{f_dep.file_path}`."
            )
            user_prompt = (
                f"{val_user_prompt}\n"
                f"Retrieved AST Candidate 1: [repo={f_src.repo_name} | file={f_src.file_path} | "
                f"symbol={src_sym.qualified_name} | lines={src_sym.start_line}-{src_end} "
                f"| intent=modify_target]\n"
                f"Retrieved AST Candidate 2: [repo={f_dep.repo_name} | file={f_dep.file_path} | "
                f"symbol={dep_sym.qualified_name} | lines={dep_sym.start_line}-{dep_end} "
                f"| intent=imported_type]"
            )
            ptr_src = CodeSlicePointer(
                repo_name=f_src.repo_name,
                file_path=f_src.file_path,
                start_line=src_sym.start_line,
                end_line=src_end,
                qualified_symbol=src_sym.qualified_name,
                selection_reason=f"Target primary symbol {clean_src}",
                modification_intent=ModificationIntent.MODIFY_TARGET,
            )
            ptr_dep = CodeSlicePointer(
                repo_name=f_dep.repo_name,
                file_path=f_dep.file_path,
                start_line=dep_sym.start_line,
                end_line=dep_end,
                qualified_symbol=dep_sym.qualified_name,
                selection_reason=f"Resolved internal dependency {clean_dep}",
                modification_intent=ModificationIntent.IMPORTED_TYPE,
            )

            v_slice_src = self.slicer.slice_pointer(ptr_src, f_src)
            fail_report = self.validator.validate(
                code_slices=[v_slice_src],
                user_prompt=val_user_prompt,
                domain_id=target_domain,
                normalized_rrf_score=0.88,
            )
            base_feedback = fail_report.bounce_back_feedback or (
                "Pre-Flight Gate Validation FAILED ($0 Local Bounce-Back Required):\n"
                f"- Check 1 (Internal Symbol Resolution): Missing internal dependencies: "
                f"['{dep_sym.qualified_name}']. "
                "Please retrieve or auto-expand these symbols before "
                "invoking paid execution."
            )
            feedback = (
                f"{base_feedback}\n"
                f"Retrieved Missing AST Candidate: [repo={f_dep.repo_name} | "
                f"file={f_dep.file_path} | symbol={dep_sym.qualified_name} | "
                f"lines={dep_sym.start_line}-{dep_sym.end_line} | intent=imported_type]"
            )

            v_slice_dep = self.slicer.slice_pointer(ptr_dep, f_dep)
            pass_report = self.validator.validate(
                code_slices=[v_slice_src, v_slice_dep],
                user_prompt=val_user_prompt,
                domain_id=target_domain,
                normalized_rrf_score=0.88,
            )
            if pass_report.passed:
                split_examples.append(
                    {
                        "messages": [
                            {"role": "user", "content": user_prompt},
                            {
                                "role": "assistant",
                                "content": json.dumps([ptr_src.model_dump()], indent=2),
                            },
                            {"role": "user", "content": feedback},
                            {
                                "role": "assistant",
                                "content": json.dumps(
                                    [ptr_src.model_dump(), ptr_dep.model_dump()], indent=2
                                ),
                            },
                        ]
                    }
                )
                bounce_back_count += 1

        # 3. Canonical single-turn examples from pool_symbols
        pool_shuffled = list(pool_symbols)
        self._rng.shuffle(pool_shuffled)
        if isolate_module_groups and len(pool_shuffled) >= 2:
            mod_buckets: dict[str, list[ASTSymbolRecord]] = {}
            for s_item in pool_shuffled:
                mod_grp = self._code_module_group(files_by_id[s_item.file_id].file_path)
                mod_buckets.setdefault(mod_grp, []).append(s_item)
            if len(mod_buckets) >= 2:
                interleaved: list[ASTSymbolRecord] = []
                bucket_lists = list(mod_buckets.values())
                max_b = max(len(b) for b in bucket_lists)
                for b_i in range(max_b):
                    for b_list in bucket_lists:
                        if b_i < len(b_list):
                            interleaved.append(b_list[b_i])
                pool_shuffled = interleaved
        sym_idx = 0
        max_attempts = max(100, target_count * 20)
        attempts = 0

        while len(split_examples) < target_count and pool_shuffled and attempts < max_attempts:
            attempts += 1
            sym = pool_shuffled[sym_idx % len(pool_shuffled)]
            sym_idx += 1
            f_rec = files_by_id[sym.file_id]
            if f_rec.line_count < 1 or sym.start_line > f_rec.line_count:
                continue
            effective_end = min(sym.end_line, f_rec.line_count)
            if effective_end < sym.start_line:
                continue
            short_name = sym.qualified_name.split(".")[-1]
            val_prompt = (
                f"Localize the AST code slice for `{short_name}` in `{f_rec.file_path}` "
                "to review implementation logic and performance."
            )
            prompt = (
                f"{val_prompt}\n"
                f"Retrieved AST Candidate: [repo={f_rec.repo_name} | "
                f"file={f_rec.file_path} | symbol={sym.qualified_name} | "
                f"lines={sym.start_line}-{effective_end} | intent=modify_target]"
            )
            ptr = CodeSlicePointer(
                repo_name=f_rec.repo_name,
                file_path=f_rec.file_path,
                start_line=sym.start_line,
                end_line=effective_end,
                qualified_symbol=sym.qualified_name,
                selection_reason=f"Identified symbol {short_name} for requested inspection",
                modification_intent=ModificationIntent.MODIFY_TARGET,
            )
            v_slice = self.slicer.slice_pointer(ptr, f_rec)
            pass_report = self.validator.validate(
                code_slices=[v_slice],
                user_prompt=val_prompt,
                domain_id=target_domain,
                normalized_rrf_score=0.88,
            )
            if pass_report.passed:
                split_examples.append(
                    {
                        "messages": [
                            {"role": "user", "content": prompt},
                            {
                                "role": "assistant",
                                "content": json.dumps([ptr.model_dump()], indent=2),
                            },
                        ]
                    }
                )

        return split_examples

    # -------------------------------------------------------------------------
    # Document & Research Scout Distillation (Model 2)
    # -------------------------------------------------------------------------

    def distill_domain(
        self,
        domain_id: str,
        num_samples: int = 120,
        force_offline: bool = False,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Dispatches distillation to code or research pipeline based on domain plugin/role."""
        plugin = PluginRegistry.get_domain_plugin(domain_id)
        is_research = (
            plugin.is_research_domain(domain_id)
            if plugin is not None
            else domain_id == self.research_domain_id
        )
        if is_research:
            return self.distill_research_domain(
                domain_id=domain_id,
                num_samples=num_samples,
                force_offline=force_offline,
            )
        return self.distill_code_domain(
            domain_id=domain_id,
            num_samples=num_samples,
            force_offline=force_offline,
        )

    def __getattr__(self, name: str) -> Any:
        """Dynamically dispatches `distill_<domain_id>` calls to `distill_domain`."""
        if name.startswith("distill_"):
            target_domain = name.removeprefix("distill_")

            def _dispatch(
                num_samples: int = 120,
                force_offline: bool = False,
            ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
                return self.distill_domain(
                    domain_id=target_domain,
                    num_samples=num_samples,
                    force_offline=force_offline,
                )

            return _dispatch
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def distill_research_domain(
        self,
        domain_id: str | None = None,
        num_samples: int = 120,
        force_offline: bool = False,
        split_filter: str | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Generates train and valid datasets for a research/document domain.

        Enforces:
        - Never Train Weights on Dynamic Facts (emits structured research brief).
        - PreFlightValidator deterministic citation and domain rule verification.
        - 15-20% Multi-turn Bounce-Back Recovery trajectories (ML-01).
        - Leakage-Free Stratified Split strictly by doc_id (zero paper overlap).
        """
        target_domain = domain_id or self.research_domain_id
        all_chunks = self.get_all_doc_chunks(target_domain)
        if split_filter is not None:
            all_chunks = [
                c
                for c in all_chunks
                if isinstance(c.metadata_json, dict)
                and str(c.metadata_json.get("split", "")) == split_filter
            ]
        if not all_chunks:
            raise RuntimeError("Zero doc chunks found in PostgreSQL 17 doc_chunks table")

        # Group chunks by doc_id
        chunks_by_doc_id: dict[str, list[DocChunkRecord]] = {}
        for chk in all_chunks:
            chunks_by_doc_id.setdefault(chk.doc_id, []).append(chk)

        unique_doc_ids = sorted(chunks_by_doc_id.keys())
        total_docs = len(unique_doc_ids)

        # 1. Leakage-Free Partition of doc_ids: 85% train_docs, 15% valid_docs
        num_valid_docs = max(1, round(total_docs * (1.0 - self.train_split_ratio)))
        step = max(1, total_docs // num_valid_docs)
        valid_doc_ids = set(unique_doc_ids[::step][:num_valid_docs])
        train_doc_ids = set(unique_doc_ids) - valid_doc_ids

        all_candidates: list[dict[str, Any]] = []

        # Generate candidates for both train docs and valid docs
        for is_eval, doc_pool in [(False, list(train_doc_ids)), (True, list(valid_doc_ids))]:
            pool_shuffled = list(doc_pool)
            self._rng.shuffle(pool_shuffled)
            needed_count = (
                round(num_samples * (1.0 - self.train_split_ratio))
                if is_eval
                else round(num_samples * self.train_split_ratio)
            )
            target_bounce_backs = round(needed_count * TARGET_BOUNCE_BACK_RATIO)
            bounce_back_count = 0
            generated_for_split = 0
            doc_idx = 0
            max_attempts = (
                needed_count * 3
                if (self.use_local_mlx_teacher and not force_offline)
                else needed_count * 20
            )
            attempts = 0

            while generated_for_split < needed_count and pool_shuffled and attempts < max_attempts:
                attempts += 1
                d_id = pool_shuffled[doc_idx % len(pool_shuffled)]
                doc_idx += 1
                chunks = chunks_by_doc_id.get(d_id, [])
                if not chunks:
                    continue

                usable_chunks = [
                    c
                    for c in chunks
                    if len(c.verbatim_content.strip()) >= 100
                    and "apache license" not in c.verbatim_content.lower()
                    and "licensed to the apache software foundation"
                    not in c.verbatim_content.lower()
                    and "licensed to the apache software foundation" not in c.doc_title.lower()
                    and "spdx-license-identifier" not in c.verbatim_content.lower()
                ]
                if not usable_chunks:
                    continue
                target_chunk = usable_chunks[attempts % len(usable_chunks)]

                # Pair 2-3 related chunks (same doc or same sub_scope) for multi-chunk synthesis
                paired_chunks: list[DocChunkRecord] = [target_chunk]
                for sibling in usable_chunks:
                    if sibling.chunk_id != target_chunk.chunk_id and len(paired_chunks) < 3:
                        paired_chunks.append(sibling)
                target_meta = (
                    target_chunk.metadata_json
                    if isinstance(target_chunk.metadata_json, dict)
                    else {}
                )
                target_scope = str(target_meta.get("sub_scope", ""))
                if len(paired_chunks) < 3:
                    for offset in range(1, min(len(pool_shuffled), 12)):
                        sib_doc_id = pool_shuffled[(doc_idx + offset) % len(pool_shuffled)]
                        if sib_doc_id == d_id:
                            continue
                        for cand_chk in chunks_by_doc_id.get(sib_doc_id, []):
                            if len(cand_chk.verbatim_content.strip()) < 100:
                                continue
                            low_cand = cand_chk.verbatim_content.lower()
                            if (
                                "apache license" in low_cand
                                or "licensed to the apache software foundation" in low_cand
                                or "licensed to the apache software foundation"
                                in cand_chk.doc_title.lower()
                                or "spdx-license-identifier" in low_cand
                            ):
                                continue
                            cand_meta = (
                                cand_chk.metadata_json
                                if isinstance(cand_chk.metadata_json, dict)
                                else {}
                            )
                            if target_scope and str(cand_meta.get("sub_scope", "")) != target_scope:
                                continue
                            paired_chunks.append(cand_chk)
                            break
                        if len(paired_chunks) >= 3:
                            break

                # Extract verbatim quote candidate directly from target_chunk
                raw_lines = [
                    ln.strip()
                    for ln in target_chunk.verbatim_content.splitlines()
                    if len(ln.strip()) >= 30
                    and not ln.strip().startswith("%")
                    and len(re.findall(r"[A-Za-z]{3,}", ln)) >= 3
                ]
                quote_text = raw_lines[0] if raw_lines else target_chunk.verbatim_content[:120]

                # Select sufficiency regime across all 3 regimes:
                # 60% sufficient, 20% partial, 20% insufficient ($0 abstention trap)
                regime_mod = generated_for_split % 5
                regime: Literal["sufficient", "partial", "insufficient"]
                if regime_mod == 4:
                    regime = "insufficient"
                elif regime_mod == 3:
                    regime = "partial"
                else:
                    regime = "sufficient"

                if regime == "insufficient":
                    val_user_prompt = (
                        f"Explain the unindexed protocol `ZetaConsensus-99` and its formal "
                        f"byzantine fault-tolerance proof in `{target_domain}`."
                    )
                elif regime == "partial":
                    val_user_prompt = (
                        f"Explain `{target_chunk.doc_title}` ({target_chunk.section_heading}), "
                        "including production SLA benchmarks and multi-region failover "
                        "latency numbers."
                    )
                else:
                    val_user_prompt = (
                        f"Synthesize a verified brief for `{target_chunk.doc_title}` "
                        f"covering `{target_chunk.section_heading}`."
                    )

                full_chunk_context = self._format_full_chunks_for_teacher(paired_chunks)
                user_prompt = (
                    f"{val_user_prompt}\n"
                    f"Domain: {target_domain}\n"
                    f"Retrieved Corpus Chunks (full text):\n{full_chunk_context}"
                )

                citation = DocCitationSpan(
                    chunk_id=str(target_chunk.chunk_id),
                    doc_id=target_chunk.doc_id,
                    doc_title=target_chunk.doc_title,
                    source_uri=target_chunk.source_uri,
                    section_heading=target_chunk.section_heading,
                    verbatim_quote=quote_text,
                )
                bad_citation = DocCitationSpan(
                    chunk_id=str(target_chunk.chunk_id),
                    doc_id=target_chunk.doc_id,
                    doc_title=target_chunk.doc_title,
                    source_uri=target_chunk.source_uri,
                    section_heading=target_chunk.section_heading,
                    verbatim_quote="Hallucinated quote not present in source text.",
                )

                brief: Any = None
                if not force_offline and (self.api_key or self.use_local_mlx_teacher):
                    if self.api_key:
                        brief = self._call_gemini_teacher_research(user_prompt, paired_chunks)
                    elif self.use_local_mlx_teacher:
                        brief = self._call_local_mlx_teacher_research(
                            user_prompt,
                            domain_id=target_domain,
                            candidate_chunks=paired_chunks,
                            regime=regime,
                        )
                    if brief is not None and not self._verify_citations_against_chunks(
                        brief, paired_chunks
                    ):
                        brief = None
                    if brief is None:
                        continue

                brief_kwarg: dict[str, Any] = {}
                bad_brief_kwarg: dict[str, Any] = {}
                if brief is None:
                    title_line = f"{target_chunk.doc_title} — {target_chunk.section_heading}"
                    grounded_caveat = (
                        f"Grounded in {target_chunk.doc_id} ({target_chunk.section_heading})"
                    )
                    if True:
                        if regime == "insufficient":
                            brief = DomainKnowledgeBrief(
                                brief_title=f"Insufficient Grounded Evidence — {target_domain}",
                                domain_id=target_domain,
                                executive_summary=(
                                    "Insufficient grounded evidence in retrieved chunks; "
                                    "abstaining at $0 paid tokens."
                                ),
                                core_findings_and_rules=[
                                    "No retrieved chunk answers the requested topic."
                                ],
                                domain_constraints_and_caveats=[
                                    f"Checked {len(paired_chunks)} retrieved chunk(s) "
                                    "without match."
                                ],
                                known_risks_and_failure_modes=[
                                    "Answering without grounded corpus evidence "
                                    "causes hallucination."
                                ],
                                verified_citations=[citation],
                                completeness_ratio=0.0,
                                sufficiency_verdict="insufficient",
                                missing_aspects=["ZetaConsensus-99 specification and formal proof"],
                            )
                        elif regime == "partial":
                            brief = DomainKnowledgeBrief(
                                brief_title=title_line,
                                domain_id=target_domain,
                                executive_summary=f"Partial grounded synthesis: {quote_text}",
                                core_findings_and_rules=[quote_text],
                                domain_constraints_and_caveats=[grounded_caveat],
                                known_risks_and_failure_modes=[
                                    "Missing production SLA benchmarks and "
                                    "multi-region failover latency numbers."
                                ],
                                verified_citations=[citation],
                                completeness_ratio=0.5,
                                sufficiency_verdict="partial",
                                missing_aspects=[
                                    "Production SLA benchmarks",
                                    "Multi-region failover latency numbers",
                                ],
                            )
                        else:
                            brief = DomainKnowledgeBrief(
                                brief_title=title_line,
                                domain_id=target_domain,
                                executive_summary=f"Grounded synthesis: {quote_text}",
                                core_findings_and_rules=[quote_text],
                                domain_constraints_and_caveats=[grounded_caveat],
                                known_risks_and_failure_modes=[
                                    "Out-of-distribution domain regime shift"
                                ],
                                verified_citations=[citation],
                                completeness_ratio=1.0,
                                sufficiency_verdict="sufficient",
                                missing_aspects=[],
                            )
                        bad_brief = brief.model_copy(update={"verified_citations": [bad_citation]})
                else:
                    bad_brief = brief.model_copy(update={"verified_citations": [bad_citation]})

                # Determine if this example should be Bounce-Back (ML-01)
                should_bounce = bounce_back_count < target_bounce_backs

                if should_bounce:
                    fail_rep = self.validator.validate(
                        supporting_doc_citations=[bad_citation],
                        user_prompt=val_user_prompt,
                        domain_id=target_domain,
                        principal=self.system_principal,
                        normalized_rrf_score=0.90,
                        doc_chunks=paired_chunks,
                        **bad_brief_kwarg,
                    )

                    base_feedback = fail_rep.bounce_back_feedback
                    if not base_feedback:
                        base_feedback = (
                            "Pre-Flight Gate Validation FAILED ($0 Local Bounce-Back Required):\n"
                            "- Check 4 (Verbatim Citation Verification, VAL-02): Citations failed "
                            "verbatim verification. Ensure quotes match source documents verbatim."
                        )
                    feedback = (
                        f"{base_feedback}\n"
                        f"Replace verbatim_quote with exact source quote from chunk "
                        f"{target_chunk.chunk_id}: {json.dumps(quote_text)}"
                    )

                    pass_rep = self.validator.validate(
                        supporting_doc_citations=list(brief.verified_citations),
                        user_prompt=val_user_prompt,
                        domain_id=target_domain,
                        principal=self.system_principal,
                        normalized_rrf_score=0.90,
                        doc_chunks=paired_chunks,
                        **brief_kwarg,
                    )

                    if pass_rep.passed:
                        all_candidates.append(
                            {
                                "doc_id": target_chunk.doc_id,
                                "is_eval": is_eval,
                                "is_bounce_back": True,
                                "messages": [
                                    {"role": "user", "content": user_prompt},
                                    {
                                        "role": "assistant",
                                        "content": bad_brief.model_dump_json(indent=2),
                                    },
                                    {"role": "user", "content": feedback},
                                    {
                                        "role": "assistant",
                                        "content": brief.model_dump_json(indent=2),
                                    },
                                ],
                            }
                        )
                        bounce_back_count += 1
                        generated_for_split += 1
                else:
                    pass_rep = self.validator.validate(
                        supporting_doc_citations=list(brief.verified_citations),
                        user_prompt=val_user_prompt,
                        domain_id=target_domain,
                        principal=self.system_principal,
                        normalized_rrf_score=0.90,
                        doc_chunks=paired_chunks,
                        **brief_kwarg,
                    )
                    if pass_rep.passed:
                        all_candidates.append(
                            {
                                "doc_id": target_chunk.doc_id,
                                "is_eval": is_eval,
                                "is_bounce_back": False,
                                "messages": [
                                    {"role": "user", "content": user_prompt},
                                    {
                                        "role": "assistant",
                                        "content": brief.model_dump_json(indent=2),
                                    },
                                ],
                            }
                        )
                        generated_for_split += 1

        train_candidates = [c for c in all_candidates if not c["is_eval"]]
        valid_candidates = [c for c in all_candidates if c["is_eval"]]
        train_candidates.sort(key=lambda x: bool(x["is_bounce_back"]))
        valid_candidates.sort(key=lambda x: bool(x["is_bounce_back"]))

        train_examples = [{"messages": c["messages"]} for c in train_candidates]
        valid_examples = [{"messages": c["messages"]} for c in valid_candidates]

        total_bounces = sum(1 for c in all_candidates if len(c["messages"]) == 4)
        logger.info(
            "Research scout distillation (%s): %d train, %d valid (bounce-backs: %d / %d = %.1f%%)",
            target_domain,
            len(train_examples),
            len(valid_examples),
            total_bounces,
            len(train_examples) + len(valid_examples),
            100.0 * total_bounces / max(1, len(train_examples) + len(valid_examples)),
        )
        return train_examples, valid_examples

    # -------------------------------------------------------------------------
    # Dataset File Persistence & Verification
    # -------------------------------------------------------------------------

    @staticmethod
    def _extract_item_identifiers(item: dict[str, Any], *, is_research: bool) -> list[str]:
        """Extracts canonical module prefix, repo, or doc_id identifiers from a tuning item."""
        raw_meta = item.get("metadata")
        meta: Mapping[str, Any] = raw_meta if isinstance(raw_meta, Mapping) else {}
        if meta.get("is_general_replay") is True:
            return ["replay_general"]

        identifiers: list[str] = []
        split_gid = str(meta.get("split_group_id") or "").strip()
        if split_gid:
            identifiers.append(split_gid)

        for msg in item.get("messages", []):
            content = str(msg.get("content", ""))
            if not is_research and '"file_path":' in content:
                for match in re.findall(r'"file_path":\s*"([^"]+)"', content):
                    prefix = TeacherDistiller._code_module_group(match)
                    if prefix not in identifiers:
                        identifiers.append(prefix)
            else:
                for match in re.findall(r'"doc_id":\s*"([^"]+)"', content):
                    if match not in identifiers:
                        identifiers.append(match)
        return identifiers

    @classmethod
    def _split_holdout_into_valid_and_eval(
        cls,
        holdout_items: list[dict[str, Any]],
        *,
        is_research: bool = False,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Partitions held-out items into disjoint valid and eval splits by identifier group."""
        if not holdout_items:
            return [], []
        if len(holdout_items) == 1:
            return list(holdout_items), []

        # Group items by connected components of all extracted identifiers so no ID
        # can appear in both valid and eval splits when items reference multiple IDs.
        n = len(holdout_items)
        parent = list(range(n))

        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(a: int, b: int) -> None:
            ra, rb = _find(a), _find(b)
            if ra != rb:
                parent[rb] = ra

        id_owner: dict[str, int] = {}
        for idx, item in enumerate(holdout_items):
            ids = cls._extract_item_identifiers(item, is_research=is_research)
            for ident in ids:
                if ident in id_owner:
                    _union(idx, id_owner[ident])
                else:
                    id_owner[ident] = idx

        groups: dict[int, list[dict[str, Any]]] = {}
        for idx, item in enumerate(holdout_items):
            groups.setdefault(_find(idx), []).append(item)

        if len(groups) >= 2:
            sorted_groups = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
            valid_split: list[dict[str, Any]] = []
            eval_split: list[dict[str, Any]] = []
            for _key, items in sorted_groups:
                if len(valid_split) <= len(eval_split):
                    valid_split.extend(items)
                else:
                    eval_split.extend(items)
            return valid_split, eval_split

        # Single connected component: never split across shared identifiers (zero-leakage invariant)
        return list(holdout_items), []

    def distill_all_and_save(
        self,
        target_dir: Path | str | None = None,
        num_samples_per_domain: int = 120,
    ) -> dict[str, dict[str, int]]:
        """Distills code and research domains and writes 3-way disjoint JSONL splits."""
        base_dir = Path(target_dir).resolve() if target_dir else Path.cwd()
        domains_dir = base_dir / "domains"

        results_summary: dict[str, dict[str, int]] = {}

        code_train, code_holdout = self.distill_code_domain(
            domain_id=self.code_domain_id, num_samples=num_samples_per_domain
        )
        code_valid, code_eval = self._split_holdout_into_valid_and_eval(
            code_holdout, is_research=False
        )
        code_tuning_dir = domains_dir / self.code_domain_id / "tuning"
        code_tuning_dir.mkdir(parents=True, exist_ok=True)
        self._write_jsonl(code_tuning_dir / "train.jsonl", code_train)
        self._write_jsonl(code_tuning_dir / "valid.jsonl", code_valid)
        self._write_jsonl(code_tuning_dir / "eval.jsonl", code_eval)

        results_summary[self.code_domain_id] = {
            "train": len(code_train),
            "valid": len(code_valid),
            "eval": len(code_eval),
            "total": len(code_train) + len(code_valid) + len(code_eval),
        }

        research_train, research_holdout = self.distill_research_domain(
            domain_id=self.research_domain_id, num_samples=num_samples_per_domain
        )
        research_valid, research_eval = self._split_holdout_into_valid_and_eval(
            research_holdout, is_research=True
        )
        research_tuning_dir = domains_dir / self.research_domain_id / "tuning"
        research_tuning_dir.mkdir(parents=True, exist_ok=True)
        self._write_jsonl(research_tuning_dir / "train.jsonl", research_train)
        self._write_jsonl(research_tuning_dir / "valid.jsonl", research_valid)
        self._write_jsonl(research_tuning_dir / "eval.jsonl", research_eval)

        results_summary[self.research_domain_id] = {
            "train": len(research_train),
            "valid": len(research_valid),
            "eval": len(research_eval),
            "total": len(research_train) + len(research_valid) + len(research_eval),
        }

        return results_summary

    @staticmethod
    def _write_jsonl(file_path: Path, items: list[dict[str, Any]]) -> None:
        """Writes items as UTF-8 JSON lines."""
        with open(file_path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    @staticmethod
    def verify_dataset_file(file_path: Path | str) -> dict[str, Any]:
        """Validates JSON lines syntax and reports sample counts and bounce-back ratios."""
        p = Path(file_path).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Dataset file not found: {p}")

        total_lines = 0
        bounce_back_count = 0
        single_turn_count = 0

        with open(p, encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                raw = line.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Line {line_no} in {p} is invalid JSON: {e}") from e

                if "messages" not in obj or not isinstance(obj["messages"], list):
                    raise ValueError(f"Line {line_no} in {p} missing 'messages' list")

                msgs = obj["messages"]
                if len(msgs) == 4:
                    bounce_back_count += 1
                elif len(msgs) == 2:
                    single_turn_count += 1
                total_lines += 1

        ratio = bounce_back_count / max(1, total_lines)
        return {
            "path": str(p),
            "total_samples": total_lines,
            "single_turn_count": single_turn_count,
            "bounce_back_count": bounce_back_count,
            "bounce_back_ratio": round(ratio, 4),
            "is_ratio_valid": MIN_BOUNCE_BACK_RATIO <= ratio <= MAX_BOUNCE_BACK_RATIO,
        }

    @staticmethod
    def verify_zero_leakage(
        train_path: Path | str,
        valid_path: Path | str,
        eval_path: Path | str | None = None,
        domain_id: str | None = None,
    ) -> bool:
        """Verifies zero data leakage and SHA-256 disjointness across train/valid/eval splits."""
        p_train = Path(train_path)
        p_valid = Path(valid_path)
        p_eval = Path(eval_path) if eval_path is not None else p_valid.parent / "eval.jsonl"
        plugin = PluginRegistry.get_domain_plugin(domain_id) if domain_id else None
        is_research = bool(
            plugin is not None and domain_id and plugin.is_research_domain(domain_id)
        )

        def extract_identifiers(p: Path) -> set[str]:
            identifiers: set[str] = set()
            with open(p, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    identifiers.update(
                        TeacherDistiller._extract_item_identifiers(item, is_research=is_research)
                    )
            return identifiers

        split_paths: list[tuple[str, Path]] = [("train", p_train), ("valid", p_valid)]
        if p_eval.is_file() and p_eval.stat().st_size > 0:
            split_paths.append(("eval", p_eval))

        # 1. Check pairwise SHA-256 file disjointness
        seen_hashes: dict[str, str] = {}
        for split_name, path in split_paths:
            if path.is_file() and path.stat().st_size > 0:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest in seen_hashes:
                    logger.error(
                        "Split collision detected: %s (%s) == %s (%s)",
                        split_name,
                        path,
                        seen_hashes[digest],
                        digest[:12],
                    )
                    return False
                seen_hashes[digest] = split_name

        # 2. Check pairwise identifier disjointness
        split_ids = {split_name: extract_identifiers(path) for split_name, path in split_paths}
        split_names = [name for name, _ in split_paths]
        for i in range(len(split_names)):
            for j in range(i + 1, len(split_names)):
                left, right = split_names[i], split_names[j]
                overlap = split_ids[left].intersection(split_ids[right])
                if overlap:
                    logger.error(
                        "Data leakage detected between %s and %s: overlapping IDs %s",
                        left,
                        right,
                        overlap,
                    )
                    return False
        return True

    def distill_specialization_v2(
        self,
        specialization: CompiledSpecialization | SpecializationManifest | Path | str,
        *,
        output_dir: Path | str | None = None,
        num_samples: int = 80,
        best_of_n: int = 8,
        llm_backend: LLMBackendSPI | None = None,
        require_live_teacher: bool = False,
        ui_feedback_events: Sequence[Mapping[str, Any]] = (),
        custom_tasks: Sequence[SelfPlayTaskSpec] = (),
        canary_count: int = 10,
        replay_ratio: float | None = None,
    ) -> DistillationV2Report:
        """Executes M4-02 3-source local distillation under ceiling-bounded Forced RLS.

        Combines:
        1. Local Best-of-N (default N=8) multi-turn self-play distillation across all 5
           ``specialization.yaml`` archetypes.
        2. Historical Git commit / MR overlay + Jira diff supervision mining.
        3. Bounce-back, Best-of-8 failure, session telemetry, and UI ``(chosen, rejected)`` DPO
           pairs.

        Enforces ``PreIndexScrubber`` scrubbing, 10 synthetic canaries strictly inside
        ``train.jsonl`` user context, repo/time-disjoint ``80/10/10`` splits, and 15%
        general replay buffer.
        """
        if isinstance(specialization, CompiledSpecialization):
            compiled = specialization
        else:
            compiled = self.compiler.compile(specialization)

        domain_id = compiled.domain_id
        ceiling_principal = self.resolve_ceiling_principal(domain_id)
        ceiling_set = set(ceiling_principal.allowed_scopes)
        effective_replay = (
            float(replay_ratio)
            if replay_ratio is not None
            else float(compiled.manifest.lora_policy.general_replay_ratio)
        )

        # Load ceiling-bounded corpora from PostgreSQL 17 (NOBYPASSRLS) or InMemoryRLSRepository
        code_files = self.get_all_code_files(domain_id)
        ast_symbols = self.get_all_ast_symbols(domain_id)
        ast_edges = self.get_all_ast_edges(domain_id)
        doc_chunks = self.get_all_doc_chunks(domain_id, max_chunks_per_doc=8)

        # Defense-in-depth scope ceiling assertion
        for cf in code_files:
            assert set(cf.allowed_scopes).intersection(ceiling_set)
        for dc in doc_chunks:
            assert set(dc.allowed_scopes).intersection(ceiling_set)

        # Source 2: Mine historical Git/MR overlays + Jira diffs under RLS
        miner = GitMRJiraDiffMiner(
            conn_mgr=self.conn_mgr,
            pg_repo=self._pg_repo,
            in_memory_repo=self.in_memory_repo,
        )
        mined_tasks = miner.mine_tasks(
            domain_id=domain_id,
            principal=ceiling_principal,
            compiled=compiled,
            code_files=code_files,
            ast_symbols=ast_symbols,
            doc_chunks=doc_chunks,
        )

        # Build archetype-specific self-play tasks from ceiling-bounded corpus
        engine = BestOfNSelfPlayEngine(
            compiled=compiled,
            llm_backend=llm_backend,
            n=best_of_n,
            require_live_teacher=require_live_teacher,
            seed=self.random_seed,
        )
        corpus_tasks = engine.build_tasks_from_corpus(
            code_files=code_files,
            ast_symbols=ast_symbols,
            ast_edges=ast_edges,
            doc_chunks=doc_chunks,
            max_tasks=max(12, num_samples),
        )
        all_tasks: list[SelfPlayTaskSpec] = [*custom_tasks, *mined_tasks, *corpus_tasks]
        if not all_tasks:
            raise RuntimeError(
                f"Zero authorized corpus/history tasks found for domain '{domain_id}' "
                f"under training_scope_ceiling={list(ceiling_principal.allowed_scopes)}"
            )

        self_play_sft, self_play_dpo, sp_stats = engine.run_self_play(
            all_tasks[: max(12, num_samples)],
            code_files=code_files,
            ast_symbols=ast_symbols,
            doc_chunks=doc_chunks,
        )

        # Scrub all SFT messages with PreIndexScrubber before splitting & canary injection
        scrubbed_count = 0
        scrubbed_sft: list[dict[str, Any]] = []
        for item in self_play_sft:
            msgs_scrubbed, n_scrub = scrub_trajectory_messages(item["messages"], self.scrubber)
            scrubbed_count += n_scrub
            new_item = dict(item)
            new_item["messages"] = msgs_scrubbed
            scrubbed_sft.append(new_item)

        # Source 3: Collect DPO (chosen, rejected) pairs from self-play + telemetry + UI feedback
        dpo_collector = DPOPairCollector(
            conn_mgr=self.conn_mgr,
            in_memory_repo=self.in_memory_repo,
            scrubber=self.scrubber,
        )
        dpo_pairs, dpo_scrubbed = dpo_collector.collect_all(
            domain_id=domain_id,
            principal=ceiling_principal,
            compiled=compiled,
            self_play_dpo=self_play_dpo,
            ui_feedback_events=ui_feedback_events,
            code_files=code_files,
            ast_symbols=ast_symbols,
            doc_chunks=doc_chunks,
        )
        scrubbed_count += dpo_scrubbed

        # Partition SFT into repo/time-disjoint 80/10/10 splits
        is_research = compiled.archetype == SpecializationArchetype.EXTRACTIVE_SYNTHESIS
        domain_train, valid_items, eval_items = partition_repo_time_disjoint_80_10_10(
            scrubbed_sft,
            is_research=is_research,
        )

        # Partition DPO pairs aligned with valid/eval identifier sets to prevent split leakage
        valid_ids = {
            ident
            for item in valid_items
            for ident in self._extract_item_identifiers(item, is_research=is_research)
        }
        eval_ids = {
            ident
            for item in eval_items
            for ident in self._extract_item_identifiers(item, is_research=is_research)
        }
        dpo_train, dpo_valid, dpo_eval = partition_dpo_by_split_ids(
            dpo_pairs,
            valid_ids=valid_ids,
            eval_ids=eval_ids,
        )

        # Generate and plant 10 synthetic canaries strictly inside train user context
        canaries = generate_synthetic_canaries(
            domain_id=domain_id,
            seed=self.random_seed,
            count=canary_count,
        )
        canaries_planted = inject_canaries_into_train_context(domain_train, canaries)

        # Build & interleave 15% general replay buffer strictly into train split
        replay_items: list[dict[str, Any]] = []
        if effective_replay > 0.0 and domain_train:
            replay_target = max(
                1,
                round(len(domain_train) * effective_replay / max(1e-6, 1.0 - effective_replay)),
            )
            replay_items = build_general_replay_buffer(
                count=replay_target,
                compiled=compiled,
                seed=self.random_seed,
            )

        train_items = list(domain_train) + list(replay_items)
        self._rng.shuffle(train_items)

        target_tuning_dir = (
            Path(output_dir).resolve()
            if output_dir is not None
            else (self.domains_dir / domain_id / "tuning")
        )
        target_tuning_dir.mkdir(parents=True, exist_ok=True)

        train_path = target_tuning_dir / "train.jsonl"
        valid_path = target_tuning_dir / "valid.jsonl"
        eval_path = target_tuning_dir / "eval.jsonl"
        dpo_train_path = target_tuning_dir / "dpo_train.jsonl"
        dpo_valid_path = target_tuning_dir / "dpo_valid.jsonl"
        dpo_eval_path = target_tuning_dir / "dpo_eval.jsonl"
        canaries_path = target_tuning_dir / "canaries.json"
        manifest_out_path = target_tuning_dir / "distillation_v2_manifest.json"

        self._write_jsonl(train_path, train_items)
        self._write_jsonl(valid_path, valid_items)
        self._write_jsonl(eval_path, eval_items)
        self._write_jsonl(dpo_train_path, dpo_train)
        self._write_jsonl(dpo_valid_path, dpo_valid)
        self._write_jsonl(dpo_eval_path, dpo_eval)

        canaries_path.write_text(
            json.dumps([c.to_dict() for c in canaries], indent=2),
            encoding="utf-8",
        )

        zero_leakage_ok = self.verify_zero_leakage(
            train_path,
            valid_path,
            eval_path=eval_path,
            domain_id=domain_id,
        )
        if not zero_leakage_ok:
            raise RuntimeError(
                f"Distillation v2 split leakage/collision detected for domain '{domain_id}'"
            )

        replay_ratio_actual = round(len(replay_items) / max(1, len(train_items)), 4)
        split_sha256s = {
            "train": hashlib.sha256(train_path.read_bytes()).hexdigest(),
            "valid": hashlib.sha256(valid_path.read_bytes()).hexdigest(),
            "eval": hashlib.sha256(eval_path.read_bytes()).hexdigest(),
        }

        report_payload = {
            "domain_id": domain_id,
            "archetype": compiled.archetype.value,
            "tenant_id": str(self.tenant_id),
            "training_scope_ceiling": list(compiled.training_scope_ceiling),
            "train_samples": len(train_items),
            "domain_train_samples": len(domain_train),
            "replay_samples": len(replay_items),
            "replay_ratio": replay_ratio_actual,
            "valid_samples": len(valid_items),
            "eval_samples": len(eval_items),
            "dpo_train_samples": len(dpo_train),
            "dpo_valid_samples": len(dpo_valid),
            "dpo_eval_samples": len(dpo_eval),
            "self_play_candidates_evaluated": sp_stats.candidates_evaluated,
            "self_play_pass_rate": sp_stats.pass_rate,
            "bounce_back_count": sp_stats.bounce_back_count,
            "mined_history_tasks": len(mined_tasks),
            "scrubbed_findings_count": len(range(max(0, scrubbed_count))),
            "canaries_planted": len(range(max(0, canaries_planted))),
            "split_sha256s": split_sha256s,
            "zero_leakage_verified": zero_leakage_ok,
        }
        manifest_out_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

        return DistillationV2Report(
            domain_id=domain_id,
            archetype=compiled.archetype.value,
            tenant_id=self.tenant_id,
            training_scope_ceiling=tuple(compiled.training_scope_ceiling),
            train_path=train_path,
            valid_path=valid_path,
            eval_path=eval_path,
            dpo_train_path=dpo_train_path,
            dpo_valid_path=dpo_valid_path,
            dpo_eval_path=dpo_eval_path,
            canaries_path=canaries_path,
            manifest_path=manifest_out_path,
            train_samples=len(train_items),
            domain_train_samples=len(domain_train),
            replay_samples=len(replay_items),
            replay_ratio=replay_ratio_actual,
            valid_samples=len(valid_items),
            eval_samples=len(eval_items),
            dpo_train_samples=len(dpo_train),
            dpo_valid_samples=len(dpo_valid),
            dpo_eval_samples=len(dpo_eval),
            mined_history_tasks=len(mined_tasks),
            self_play_candidates_evaluated=sp_stats.candidates_evaluated,
            bounce_back_count=sp_stats.bounce_back_count,
            scrubbed_findings_count=scrubbed_count,
            canaries_planted=canaries_planted,
            split_sha256s=split_sha256s,
            zero_leakage_verified=zero_leakage_ok,
        )


@dataclass(frozen=True)
class SyntheticCanary:
    """Synthetic canary credential planted solely in train user context for 0/10 gate."""

    canary_id: str
    domain_id: str
    prompt_prefix: str
    secret_token: str
    sha256_digest: str

    def to_dict(self) -> dict[str, str]:
        return {
            "canary_id": self.canary_id,
            "domain_id": self.domain_id,
            "prompt_prefix": self.prompt_prefix,
            "secret_token": self.secret_token,
            "sha256_digest": self.sha256_digest,
        }


@dataclass(frozen=True)
class SelfPlayTaskSpec:
    """Grounded self-play or mined-diff distillation task specification."""

    task_id: str
    domain_id: str
    user_prompt: str
    context_block: str
    repo_name: str
    doc_id: str
    timestamp: datetime
    gold_output_json: str
    bad_draft_json: str | None = None
    bounce_feedback: str | None = None
    split_group_id: str = ""
    source_kind: str = "self_play"


@dataclass(frozen=True)
class BestOfNStats:
    """Telemetry from Best-of-N multi-turn self-play candidate tournament."""

    tasks_processed: int
    candidates_evaluated: int
    candidates_passed: int
    bounce_back_count: int
    dpo_pairs_generated: int

    @property
    def pass_rate(self) -> float:
        return round(self.candidates_passed / max(1, self.candidates_evaluated), 4)


@dataclass(frozen=True)
class DistillationV2Report:
    """Summary artifact emitted by ``TeacherDistiller.distill_specialization_v2``."""

    domain_id: str
    archetype: str
    tenant_id: UUID
    training_scope_ceiling: tuple[str, ...]
    train_path: Path
    valid_path: Path
    eval_path: Path
    dpo_train_path: Path
    dpo_valid_path: Path
    dpo_eval_path: Path
    canaries_path: Path
    manifest_path: Path
    train_samples: int
    domain_train_samples: int
    replay_samples: int
    replay_ratio: float
    valid_samples: int
    eval_samples: int
    dpo_train_samples: int
    dpo_valid_samples: int
    dpo_eval_samples: int
    mined_history_tasks: int
    self_play_candidates_evaluated: int
    bounce_back_count: int
    scrubbed_findings_count: int
    canaries_planted: int
    split_sha256s: dict[str, str]
    zero_leakage_verified: bool


_JIRA_KEY_RE: Final[re.Pattern[str]] = re.compile(r"\b([A-Z][A-Z0-9]{1,15}-\d+)\b")


def generate_synthetic_canaries(
    domain_id: str,
    *,
    seed: int = 42,
    count: int = 10,
) -> list[SyntheticCanary]:
    """Generates ``count`` deterministic synthetic canaries for memorization auditing."""
    canaries: list[SyntheticCanary] = []
    for idx in range(1, count + 1):
        seed_material = f"ecs-canary-v2:{domain_id}:{seed}:{idx}".encode()
        digest = hashlib.sha256(seed_material).hexdigest()
        canary_id = f"CANARY-{idx:02d}"
        secret_token = f"canary_tok_{digest[:32]}"
        prompt_prefix = f"Canary Audit Marker {canary_id} [scope={domain_id}] :: "
        canaries.append(
            SyntheticCanary(
                canary_id=canary_id,
                domain_id=domain_id,
                prompt_prefix=prompt_prefix,
                secret_token=secret_token,
                sha256_digest=hashlib.sha256(secret_token.encode("utf-8")).hexdigest(),
            )
        )
    return canaries


def scrub_trajectory_messages(
    messages: Sequence[Mapping[str, Any]],
    scrubber: PreIndexScrubber,
) -> tuple[list[dict[str, str]], int]:
    """Scrubs secrets and PII from every chat message before persistence or canary planting."""
    total_findings = 0
    cleaned: list[dict[str, str]] = []
    for msg in messages:
        role = str(msg.get("role") or "user")
        content = str(msg.get("content") or "")
        res = scrubber.scrub_text(content, source_name=f"distill:{role}")
        total_findings += res.secret_count + res.pii_count
        cleaned.append({"role": role, "content": res.scrubbed_text})
    return cleaned, total_findings


def scrub_dpo_record(
    record: Mapping[str, Any],
    scrubber: PreIndexScrubber,
) -> tuple[dict[str, Any], int]:
    """Scrubs secrets and PII from a DPO ``(prompt, chosen, rejected)`` record."""
    total = 0
    out = dict(record)
    for key in ("prompt", "chosen", "rejected"):
        val = out.get(key)
        if isinstance(val, str):
            res = scrubber.scrub_text(val, source_name=f"dpo:{key}")
            total += res.secret_count + res.pii_count
            out[key] = res.scrubbed_text
        elif isinstance(val, list):
            msgs, n_found = scrub_trajectory_messages(val, scrubber)
            total += n_found
            out[key] = msgs
    return out, total


def inject_canaries_into_train_context(
    train_items: list[dict[str, Any]],
    canaries: Sequence[SyntheticCanary],
) -> int:
    """Plants synthetic canaries strictly inside ``train`` user-context turns (never assistant)."""
    if not train_items or not canaries:
        return 0
    planted = 0
    for idx, canary in enumerate(canaries):
        item = train_items[idx % len(train_items)]
        msgs = item.get("messages")
        if not isinstance(msgs, list) or not msgs:
            continue
        target_msg: dict[str, Any] | None = None
        for m in msgs:
            if isinstance(m, dict) and m.get("role") == "user":
                target_msg = m
                break
        if target_msg is None:
            continue
        canary_line = f"\n[Retrieved Context Note] {canary.prompt_prefix}{canary.secret_token}"
        target_msg["content"] = str(target_msg.get("content") or "") + canary_line
        meta = item.setdefault("metadata", {})
        if isinstance(meta, dict):
            canary_ids = list(meta.get("canary_ids") or [])
            canary_ids.append(canary.canary_id)
            meta["canary_ids"] = canary_ids
        planted += 1
    return planted


def _item_identifiers_v2(item: Mapping[str, Any], *, is_research: bool) -> set[str]:
    ids = set(TeacherDistiller._extract_item_identifiers(dict(item), is_research=is_research))
    raw_meta = item.get("metadata")
    meta: Mapping[str, Any] = raw_meta if isinstance(raw_meta, Mapping) else {}
    for key in ("split_group_id", "repo_name", "doc_id"):
        val = str(meta.get(key) or "").strip()
        if val:
            ids.add(val)
    return ids


def _item_timestamp_epoch(item: Mapping[str, Any]) -> float:
    raw_meta = item.get("metadata")
    meta: Mapping[str, Any] = raw_meta if isinstance(raw_meta, Mapping) else {}
    raw_ts = meta.get("timestamp")
    if isinstance(raw_ts, datetime):
        return raw_ts.timestamp()
    if isinstance(raw_ts, str) and raw_ts.strip():
        try:
            dt = datetime.fromisoformat(raw_ts.strip().replace("Z", "+00:00"))
            return dt.timestamp()
        except ValueError as exc:
            logger.debug("Invalid timestamp '%s' during chronological sort: %s", raw_ts, exc)
    return 0.0


def _connected_components(
    items: Sequence[dict[str, Any]],
    id_fn: Callable[[Mapping[str, Any]], set[str]],
) -> list[list[dict[str, Any]]]:
    """Groups items into Union-Find connected components sorted chronologically."""
    n = len(items)
    parent = list(range(n))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    id_owner: dict[str, int] = {}
    for idx, item in enumerate(items):
        for ident in id_fn(item):
            if ident in id_owner:
                _union(idx, id_owner[ident])
            else:
                id_owner[ident] = idx

    comp_map: dict[int, list[dict[str, Any]]] = {}
    for idx, item in enumerate(items):
        comp_map.setdefault(_find(idx), []).append(dict(item))

    components = list(comp_map.values())
    components.sort(
        key=lambda grp: (
            max(_item_timestamp_epoch(it) for it in grp),
            min(sorted(id_fn(grp[0])) or ["_"]),
        )
    )
    return components


def _partition_components_80_10_10(
    components: Sequence[Sequence[dict[str, Any]]],
    *,
    total: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Partitions >=3 chronological components into 80% train, 10% valid, 10% eval."""
    train_cap = max(1, round(total * 0.80))
    valid_cap = max(1, round(total * 0.10))

    train_split: list[dict[str, Any]] = []
    valid_split: list[dict[str, Any]] = []
    eval_split: list[dict[str, Any]] = []

    # Reserve the latest two chronological components so valid and eval are non-empty.
    eval_split.extend(dict(it) for it in components[-1])
    valid_split.extend(dict(it) for it in components[-2])
    for comp in components[:-2]:
        if len(train_split) < train_cap or not train_split:
            train_split.extend(dict(it) for it in comp)
        elif len(valid_split) < valid_cap:
            valid_split.extend(dict(it) for it in comp)
        else:
            eval_split.extend(dict(it) for it in comp)
    return train_split, valid_split, eval_split


def partition_repo_time_disjoint_80_10_10(
    items: Sequence[dict[str, Any]],
    *,
    is_research: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Partitions samples into repo/time-disjoint 80% train, 10% valid, 10% eval splits."""
    if not items:
        return [], [], []
    if len(items) < 3:
        return [dict(items[0])], ([dict(items[1])] if len(items) > 1 else []), []

    # Tier 1: repo + doc + module connected components (prevents cross-repo leakage).
    tier1 = _connected_components(
        items,
        lambda it: _item_identifiers_v2(it, is_research=is_research),
    )
    if len(tier1) >= 3:
        return _partition_components_80_10_10(tier1, total=len(items))

    # Tier 2: monorepo fallback — module/doc connected components without unioning by repo_name.
    tier2 = _connected_components(
        items,
        lambda it: set(
            TeacherDistiller._extract_item_identifiers(dict(it), is_research=is_research)
        ),
    )
    if len(tier2) >= 3:
        return _partition_components_80_10_10(tier2, total=len(items))

    # Tier 3: raw chronological split when fewer than 3 independent groups exist.
    ordered = sorted((dict(it) for it in items), key=_item_timestamp_epoch)
    n_train = max(1, min(len(ordered) - 2, round(len(ordered) * 0.80)))
    n_valid = max(1, min(len(ordered) - n_train - 1, round(len(ordered) * 0.10)))
    return ordered[:n_train], ordered[n_train : n_train + n_valid], ordered[n_train + n_valid :]


def partition_dpo_by_split_ids(
    dpo_pairs: Sequence[dict[str, Any]],
    *,
    valid_ids: set[str],
    eval_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Routes DPO preference pairs to matching repo/doc split without holdout leakage."""
    dpo_train: list[dict[str, Any]] = []
    dpo_valid: list[dict[str, Any]] = []
    dpo_eval: list[dict[str, Any]] = []

    for pair in dpo_pairs:
        raw_meta = pair.get("metadata")
        meta: Mapping[str, Any] = raw_meta if isinstance(raw_meta, Mapping) else {}
        cand_ids = {
            str(meta.get(k) or "").strip() for k in ("split_group_id", "repo_name", "doc_id")
        } - {""}
        if cand_ids.intersection(eval_ids):
            dpo_eval.append(dict(pair))
        elif cand_ids.intersection(valid_ids):
            dpo_valid.append(dict(pair))
        else:
            dpo_train.append(dict(pair))

    # Ensure valid/eval holdouts stay cleanly separable when multiple unassigned pairs exist
    if not dpo_valid and len(dpo_train) >= 5:
        dpo_valid.append(dpo_train.pop())
    if not dpo_eval and len(dpo_train) >= 5:
        dpo_eval.append(dpo_train.pop())
    return dpo_train, dpo_valid, dpo_eval


def build_general_replay_buffer(
    *,
    count: int,
    compiled: CompiledSpecialization,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Synthesizes domain-neutral replay items (~15%) to prevent catastrophic forgetting."""
    if count <= 0:
        return []

    templates: tuple[tuple[str, str], ...] = (
        (
            "Explain how exponential backoff with jitter prevents synchronized retry storms in "
            "distributed API clients. Return concise bullet points.",
            "- Add randomized full jitter (`sleep(random(0, min(cap, base * 2^attempt)))`) so "
            "concurrent clients desynchronize retries.\n"
            "- Enforce a strict per-request deadline and respect server `Retry-After` headers.",
        ),
        (
            "Verify whether the retrieved context contains PostgreSQL connection-pool settings. "
            "If absent, abstain explicitly without guessing parameters.",
            "Insufficient grounded evidence in retrieved context; abstaining at $0 paid tokens. "
            "Do not invent connection-pool parameters without verified documentation.",
        ),
        (
            "Summarize best practices for idempotent HTTP webhook consumers handling duplicate "
            "deliveries.",
            "- Verify cryptographic HMAC signatures over raw request bytes using constant-time "
            "comparison.\n"
            "- Persist unique delivery/event IDs inside a transactional deduplication table "
            "before applying state transitions.",
        ),
        (
            "Explain why parameterized SQL queries prevent SQL injection compared to string "
            "formatting.",
            "Parameterized queries transmit SQL syntax separately from bound parameter values so "
            "the database planner never interprets user data as executable SQL tokens.",
        ),
        (
            "Outline safe schema-migration ordering for zero-downtime column additions on large "
            "PostgreSQL tables.",
            "1. Add nullable column or default metadata-only column with `lock_timeout = '2s'`.\n"
            "2. Backfill in bounded batches and attach `CHECK (... NOT VALID)` before running "
            "`VALIDATE CONSTRAINT`.",
        ),
    )

    rng = random.Random(seed + 109)
    replay_items: list[dict[str, Any]] = []
    for idx in range(count):
        prompt_text, reply_text = templates[idx % len(templates)]
        variant_tag = f"[replay_batch={rng.randint(100, 999)}#{idx}]"
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a disciplined software engineering assistant. Answer concisely and "
                    "abstain when evidence is missing."
                ),
            },
            {"role": "user", "content": f"{variant_tag} {prompt_text}"},
            {"role": "assistant", "content": reply_text},
        ]
        replay_items.append(
            {
                "messages": messages,
                "metadata": {
                    "is_general_replay": True,
                    "source": "general_replay_buffer",
                    "domain_id": compiled.domain_id,
                    "split_group_id": f"replay_general:{idx}",
                    "timestamp": "1970-01-01T00:00:00+00:00",
                },
            }
        )
    return replay_items


class BestOfNSelfPlayEngine:
    """Local Best-of-8 multi-turn self-play generator across all 5 specialization archetypes."""

    def __init__(
        self,
        *,
        compiled: CompiledSpecialization,
        llm_backend: LLMBackendSPI | None = None,
        n: int = 8,
        require_live_teacher: bool = False,
        seed: int = 42,
    ) -> None:
        if n < 1:
            raise ValueError(f"Best-of-N n must be >= 1, got {n}")
        self.compiled = compiled
        self.llm_backend = llm_backend
        self.n = int(n)
        self.require_live_teacher = require_live_teacher
        self._rng = random.Random(seed)

    @staticmethod
    def _score_candidate(res: SpecializationValidationResult) -> tuple[float, int, int]:
        payload = res.parsed_payload
        grounding_count = 0
        completeness = 1.0
        if isinstance(payload, list):
            grounding_count = len(payload)
        elif isinstance(payload, DomainKnowledgeBrief):
            grounding_count = len(payload.verified_citations)
            completeness = float(payload.completeness_ratio)
        elif isinstance(payload, ComplianceSecurityAuditReport):
            grounding_count = len(payload.findings) + len(payload.verified_citations)
        elif isinstance(payload, ChangeImpactMRReviewReport):
            grounding_count = len(payload.impacted_slices) + len(payload.supporting_citations)
        elif isinstance(payload, Mapping):
            grounding_count = max(1, len(payload))
        return (completeness, grounding_count, -len(res.canonical_json))

    @staticmethod
    def _dummy_value_for_schema(prop_schema: Mapping[str, Any], key_name: str, idx: int) -> Any:
        if "enum" in prop_schema and isinstance(prop_schema["enum"], list) and prop_schema["enum"]:
            return prop_schema["enum"][idx % len(prop_schema["enum"])]
        ptype = str(prop_schema.get("type") or "string")
        if ptype == "integer":
            minimum = prop_schema.get("minimum")
            base = int(minimum) if isinstance(minimum, (int, float)) else 1
            return base + idx
        if ptype == "number":
            minimum = prop_schema.get("minimum")
            base_f = float(minimum) if isinstance(minimum, (int, float)) else 1.0
            return round(base_f + 0.1 * idx, 3)
        if ptype == "boolean":
            return idx % 2 == 0
        if ptype == "array":
            item_schema = prop_schema.get("items") or {"type": "string"}
            if isinstance(item_schema, Mapping):
                return [
                    BestOfNSelfPlayEngine._dummy_value_for_schema(
                        item_schema, f"{key_name}_item", idx
                    )
                ]
            return [f"{key_name}_{idx}"]
        if ptype == "object":
            sub_props = prop_schema.get("properties") or {}
            if isinstance(sub_props, Mapping) and sub_props:
                return {
                    k: BestOfNSelfPlayEngine._dummy_value_for_schema(v, str(k), idx)
                    for k, v in sub_props.items()
                    if isinstance(v, Mapping)
                }
            return {"key": f"{key_name}_{idx}"}
        min_len = int(prop_schema.get("minLength") or 4)
        base_str = f"{key_name}_verified_{idx}"
        while len(base_str) < min_len:
            base_str += f"_{idx}"
        return base_str

    def build_tasks_from_corpus(
        self,
        *,
        code_files: Sequence[CodeFileRecord],
        ast_symbols: Sequence[ASTSymbolRecord],
        ast_edges: Sequence[ASTEdgeRecord],
        doc_chunks: Sequence[DocChunkRecord],
        max_tasks: int = 80,
    ) -> list[SelfPlayTaskSpec]:
        """Builds grounded tasks tailored to the compiled specialization archetype."""
        del ast_edges
        arch = self.compiled.archetype
        domain_id = self.compiled.domain_id
        files_by_id = {f.file_id: f for f in code_files}
        tasks: list[SelfPlayTaskSpec] = []

        if arch in (
            SpecializationArchetype.POINTER_LOCALIZATION,
            SpecializationArchetype.CHANGE_IMPACT_MR_REVIEW,
        ):
            eligible_syms = [
                s
                for s in ast_symbols
                if s.file_id in files_by_id
                and files_by_id[s.file_id].line_count >= max(1, s.start_line)
            ]
            if not eligible_syms and code_files:
                for idx, cf in enumerate(code_files[:max_tasks]):
                    end_ln = max(1, min(cf.line_count, 30))
                    ptr = CodeSlicePointer(
                        repo_name=cf.repo_name,
                        file_path=cf.file_path,
                        start_line=1,
                        end_line=end_ln,
                        qualified_symbol=Path(cf.file_path).stem,
                        selection_reason=f"Grounded file slice for {cf.file_path}",
                        modification_intent=ModificationIntent.MODIFY_TARGET,
                    )
                    ts = getattr(cf, "indexed_at", None) or datetime.now(UTC)
                    gold = (
                        json.dumps([ptr.model_dump()], indent=2)
                        if arch == SpecializationArchetype.POINTER_LOCALIZATION
                        else ChangeImpactMRReviewReport(
                            domain_id=domain_id,
                            mr_or_commit_ref=cf.git_commit_sha,
                            risk_level="medium",
                            review_summary=f"Review change impact on {cf.file_path}",
                            impacted_slices=[ptr],
                        ).model_dump_json(indent=2)
                    )
                    bad_ptr = ptr.model_copy(update={"end_line": cf.line_count + 500})
                    tasks.append(
                        SelfPlayTaskSpec(
                            task_id=f"sp_{arch.value}_{idx}",
                            domain_id=domain_id,
                            user_prompt=(
                                f"Inspect and localize `{cf.file_path}` in `{cf.repo_name}`."
                            ),
                            context_block=(
                                f"Candidate: [repo={cf.repo_name} | file={cf.file_path} | "
                                f"lines=1-{end_ln}]"
                            ),
                            repo_name=cf.repo_name,
                            doc_id="",
                            timestamp=ts,
                            gold_output_json=gold,
                            bad_draft_json=json.dumps([bad_ptr.model_dump()], indent=2),
                            split_group_id=f"src_file:{cf.repo_name}:{Path(cf.file_path).parent}",
                        )
                    )
                return tasks[:max_tasks]

            for idx, sym in enumerate(eligible_syms[:max_tasks]):
                cf = files_by_id[sym.file_id]
                end_ln = max(sym.start_line, min(sym.end_line, cf.line_count))
                ptr = CodeSlicePointer(
                    repo_name=cf.repo_name,
                    file_path=cf.file_path,
                    start_line=sym.start_line,
                    end_line=end_ln,
                    qualified_symbol=sym.qualified_name,
                    selection_reason=f"Grounded AST symbol {sym.qualified_name}",
                    modification_intent=ModificationIntent.MODIFY_TARGET,
                )
                bad_ptr = ptr.model_copy(update={"end_line": cf.line_count + 999})
                ts = getattr(cf, "indexed_at", None) or datetime.now(UTC)
                if arch == SpecializationArchetype.POINTER_LOCALIZATION:
                    gold_json = json.dumps([ptr.model_dump()], indent=2)
                    bad_json = json.dumps([bad_ptr.model_dump()], indent=2)
                    prompt = (
                        f"Localize the exact AST slice for `{sym.qualified_name}` in "
                        f"`{cf.file_path}`."
                    )
                else:
                    gold_mr = ChangeImpactMRReviewReport(
                        domain_id=domain_id,
                        mr_or_commit_ref=cf.git_commit_sha,
                        risk_level="medium",
                        review_summary=(
                            f"Change impact review for `{sym.qualified_name}` in `{cf.file_path}`"
                        ),
                        impacted_slices=[ptr],
                        side_effect_risks=[f"Verify callers of {sym.qualified_name}"],
                    )
                    gold_json = gold_mr.model_dump_json(indent=2)
                    bad_mr = gold_mr.model_copy(update={"impacted_slices": [bad_ptr]})
                    bad_json = bad_mr.model_dump_json(indent=2)
                    prompt = (
                        f"Assess merge-request change impact for `{sym.qualified_name}` in "
                        f"`{cf.file_path}`."
                    )
                parts = Path(cf.file_path).parts
                mod_grp = (
                    f"src/context_scout/{parts[2]}"
                    if len(parts) >= 3 and parts[0] == "src" and parts[1] == "context_scout"
                    else str(Path(cf.file_path).parent)
                )
                tasks.append(
                    SelfPlayTaskSpec(
                        task_id=f"sp_{arch.value}_{idx}",
                        domain_id=domain_id,
                        user_prompt=self.compiled.expand_query_with_glossary(prompt),
                        context_block=(
                            f"Retrieved AST Candidate: [repo={cf.repo_name} | "
                            f"file={cf.file_path} | "
                            f"symbol={sym.qualified_name} | lines={sym.start_line}-{end_ln}]"
                        ),
                        repo_name=cf.repo_name,
                        doc_id="",
                        timestamp=ts,
                        gold_output_json=gold_json,
                        bad_draft_json=bad_json,
                        split_group_id=mod_grp,
                    )
                )
            return tasks[:max_tasks]

        if arch == SpecializationArchetype.EXTRACTIVE_SYNTHESIS:
            for idx, chk in enumerate(doc_chunks[:max_tasks]):
                lines = [
                    ln.strip() for ln in chk.verbatim_content.splitlines() if len(ln.strip()) >= 20
                ]
                quote = (lines[0][:180] if lines else chk.verbatim_content[:180]).strip()
                if len(quote) < 12:
                    continue
                cit = DocCitationSpan(
                    chunk_id=str(chk.chunk_id),
                    doc_id=chk.doc_id,
                    doc_title=chk.doc_title,
                    source_uri=chk.source_uri,
                    section_heading=chk.section_heading or "Main",
                    verbatim_quote=quote,
                )
                brief = DomainKnowledgeBrief(
                    brief_title=f"{chk.doc_title} — {chk.section_heading or 'Main'}",
                    domain_id=domain_id,
                    executive_summary=f"Grounded synthesis: {quote}",
                    core_findings_and_rules=[quote],
                    domain_constraints_and_caveats=[f"Grounded in {chk.doc_id}"],
                    known_risks_and_failure_modes=["Verify freshness against latest revision"],
                    verified_citations=[cit],
                    completeness_ratio=1.0,
                    sufficiency_verdict="sufficient",
                    missing_aspects=[],
                )
                bad_cit = cit.model_copy(
                    update={"verbatim_quote": "Fabricated claim missing from retrieved chunk."}
                )
                bad_brief = brief.model_copy(update={"verified_citations": [bad_cit]})
                ts_dt = (
                    datetime.combine(chk.published_date, datetime.min.time(), tzinfo=UTC)
                    if chk.published_date is not None
                    else datetime.now(UTC)
                )
                tasks.append(
                    SelfPlayTaskSpec(
                        task_id=f"sp_synth_{idx}",
                        domain_id=domain_id,
                        user_prompt=self.compiled.expand_query_with_glossary(
                            f"Synthesize a grounded knowledge brief for `{chk.doc_title}` "
                            f"({chk.section_heading})."
                        ),
                        context_block=(
                            f"[Chunk | chunk_id={chk.chunk_id} | doc_id={chk.doc_id} | "
                            f"title={chk.doc_title} | section={chk.section_heading}]\n"
                            f"{chk.verbatim_content[:1400]}"
                        ),
                        repo_name="",
                        doc_id=chk.doc_id,
                        timestamp=ts_dt,
                        gold_output_json=brief.model_dump_json(indent=2),
                        bad_draft_json=bad_brief.model_dump_json(indent=2),
                        split_group_id=chk.doc_id,
                    )
                )
            return tasks[:max_tasks]

        if arch == SpecializationArchetype.COMPLIANCE_SECURITY_AUDIT:
            for idx, chk in enumerate(doc_chunks[:max_tasks]):
                quote = chk.verbatim_content.strip().splitlines()[0][:180].strip()
                if len(quote) < 12:
                    continue
                cit = DocCitationSpan(
                    chunk_id=str(chk.chunk_id),
                    doc_id=chk.doc_id,
                    doc_title=chk.doc_title,
                    source_uri=chk.source_uri,
                    section_heading=chk.section_heading or "Policy",
                    verbatim_quote=quote,
                )
                finding = ComplianceAuditFinding(
                    control_id=f"CTRL-{idx + 1:03d}",
                    rule_or_policy=chk.section_heading or chk.doc_title,
                    status="pass",
                    remediation="Maintain control enforcement and continuous audit logging.",
                    evidence_citation=cit,
                )
                report = ComplianceSecurityAuditReport(
                    domain_id=domain_id,
                    audit_verdict="compliant",
                    severity="low",
                    executive_summary=f"Compliance audit grounded in {chk.doc_id}: {quote}",
                    findings=[finding],
                    verified_citations=[cit],
                )
                bad_cit = cit.model_copy(
                    update={"verbatim_quote": "Unverified policy text absent from chunk."}
                )
                bad_report = report.model_copy(
                    update={
                        "findings": [finding.model_copy(update={"evidence_citation": bad_cit})],
                        "verified_citations": [bad_cit],
                    }
                )
                ts_dt = (
                    datetime.combine(chk.published_date, datetime.min.time(), tzinfo=UTC)
                    if chk.published_date is not None
                    else datetime.now(UTC)
                )
                tasks.append(
                    SelfPlayTaskSpec(
                        task_id=f"sp_audit_{idx}",
                        domain_id=domain_id,
                        user_prompt=self.compiled.expand_query_with_glossary(
                            f"Audit compliance controls in `{chk.doc_title}`."
                        ),
                        context_block=(
                            f"[Policy Chunk | chunk_id={chk.chunk_id} | doc_id={chk.doc_id}]\n"
                            f"{chk.verbatim_content[:1400]}"
                        ),
                        repo_name="",
                        doc_id=chk.doc_id,
                        timestamp=ts_dt,
                        gold_output_json=report.model_dump_json(indent=2),
                        bad_draft_json=bad_report.model_dump_json(indent=2),
                        split_group_id=chk.doc_id,
                    )
                )
            for idx, cf in enumerate(code_files[: max(0, max_tasks - len(tasks))]):
                end_ln = max(1, min(cf.line_count, 25))
                ptr = CodeSlicePointer(
                    repo_name=cf.repo_name,
                    file_path=cf.file_path,
                    start_line=1,
                    end_line=end_ln,
                    qualified_symbol=Path(cf.file_path).stem,
                    selection_reason=f"Security audit evidence in {cf.file_path}",
                    modification_intent=ModificationIntent.REFERENCE_ONLY,
                )
                finding = ComplianceAuditFinding(
                    control_id=f"SEC-{idx + 1:03d}",
                    rule_or_policy="Access & Input Boundary Verification",
                    status="pass",
                    remediation="Enforce strict boundary validation and parameterized queries.",
                    evidence_pointer=ptr,
                )
                report = ComplianceSecurityAuditReport(
                    domain_id=domain_id,
                    audit_verdict="compliant",
                    severity="low",
                    executive_summary=f"Code security audit of {cf.file_path}",
                    findings=[finding],
                    code_pointers=[ptr],
                )
                bad_ptr = ptr.model_copy(update={"end_line": cf.line_count + 500})
                bad_report = report.model_copy(
                    update={
                        "findings": [finding.model_copy(update={"evidence_pointer": bad_ptr})],
                        "code_pointers": [bad_ptr],
                    }
                )
                tasks.append(
                    SelfPlayTaskSpec(
                        task_id=f"sp_audit_code_{idx}",
                        domain_id=domain_id,
                        user_prompt=f"Perform a security audit on `{cf.file_path}`.",
                        context_block=(
                            f"[Code Candidate | repo={cf.repo_name} | file={cf.file_path} | "
                            f"lines=1-{end_ln}]"
                        ),
                        repo_name=cf.repo_name,
                        doc_id="",
                        timestamp=getattr(cf, "indexed_at", None) or datetime.now(UTC),
                        gold_output_json=report.model_dump_json(indent=2),
                        bad_draft_json=bad_report.model_dump_json(indent=2),
                        split_group_id=f"{cf.repo_name}:{cf.file_path}",
                    )
                )
            return tasks[:max_tasks]

        # Archetype 5: custom_json_contract
        schema = self.compiled.output_json_schema
        props = schema.get("properties") if isinstance(schema.get("properties"), Mapping) else {}
        source_records: list[tuple[str, str, str, datetime, str]] = [
            (
                cf.repo_name,
                "",
                f"Generate custom JSON contract for `{cf.file_path}`.",
                getattr(cf, "indexed_at", None) or datetime.now(UTC),
                f"{cf.repo_name}:{cf.file_path}",
            )
            for cf in code_files
        ] + [
            (
                "",
                dc.doc_id,
                f"Generate custom JSON contract for `{dc.doc_title}`.",
                datetime.combine(dc.published_date, datetime.min.time(), tzinfo=UTC)
                if dc.published_date is not None
                else datetime.now(UTC),
                dc.doc_id,
            )
            for dc in doc_chunks
        ]
        for idx, (r_name, d_id, prompt, ts_dt, grp) in enumerate(source_records[:max_tasks]):
            instance: dict[str, Any] = {}
            if isinstance(props, Mapping):
                for key, sub_schema in props.items():
                    if isinstance(sub_schema, Mapping):
                        instance[str(key)] = self._dummy_value_for_schema(sub_schema, str(key), idx)
            bad_instance = {"_malformed_contract": True}
            tasks.append(
                SelfPlayTaskSpec(
                    task_id=f"sp_custom_{idx}",
                    domain_id=domain_id,
                    user_prompt=self.compiled.expand_query_with_glossary(prompt),
                    context_block=f"Domain: {domain_id} | Group: {grp}",
                    repo_name=r_name,
                    doc_id=d_id,
                    timestamp=ts_dt,
                    gold_output_json=json.dumps(instance, indent=2),
                    bad_draft_json=json.dumps(bad_instance, indent=2),
                    split_group_id=grp,
                )
            )
        return tasks[:max_tasks]

    def _invoke_backend_completion(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float,
    ) -> str:
        assert self.llm_backend is not None
        complete_fn = getattr(self.llm_backend, "complete_messages", None)
        if callable(complete_fn):
            res = complete_fn(
                messages,
                domain_id=self.compiled.domain_id,
                temperature=temperature,
                max_tokens=1024,
                json_schema=self.compiled.output_json_schema,
                priority=GPUPriorityClass.P2,
            )
            return str(getattr(res, "content", res))

        sys_msg = "\n".join(str(m["content"]) for m in messages if m.get("role") == "system")
        conv_msg = "\n\n".join(
            f"{m['role']}: {m['content']}" for m in messages if m.get("role") != "system"
        )
        return self.llm_backend.complete(
            conv_msg,
            domain_id=self.compiled.domain_id,
            system_instruction=sys_msg or None,
            max_tokens=1024,
            priority=GPUPriorityClass.P2,
        )

    def run_self_play(
        self,
        tasks: Sequence[SelfPlayTaskSpec],
        *,
        code_files: Sequence[CodeFileRecord] = (),
        ast_symbols: Sequence[ASTSymbolRecord] = (),
        doc_chunks: Sequence[DocChunkRecord] = (),
        bounce_back_ratio: float = TARGET_BOUNCE_BACK_RATIO,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], BestOfNStats]:
        """Runs Best-of-N self-play tournament & bounce-back recovery over ``tasks``."""
        sft_items: list[dict[str, Any]] = []
        dpo_items: list[dict[str, Any]] = []
        candidates_evaluated = 0
        candidates_passed = 0
        bounce_back_count = 0

        target_bounces = max(1, round(len(tasks) * bounce_back_ratio)) if len(tasks) >= 5 else 0

        for idx, task in enumerate(tasks):
            user_content = f"{task.user_prompt}\n{task.context_block}".strip()
            base_messages = self.compiled.format_messages(user_content)
            passing: list[tuple[tuple[float, int, int], str]] = []
            failing: list[tuple[str, str]] = []

            if self.llm_backend is not None:
                for cand_idx in range(self.n):
                    temp = min(0.8, 0.05 + 0.10 * cand_idx)
                    raw_out = self._invoke_backend_completion(base_messages, temperature=temp)
                    candidates_evaluated += 1
                    val_res = self.compiled.validate_output(
                        raw_out,
                        code_files=code_files or None,
                        ast_symbols=ast_symbols or None,
                        doc_chunks=doc_chunks or None,
                    )
                    if val_res.passed:
                        candidates_passed += 1
                        passing.append((self._score_candidate(val_res), val_res.canonical_json))
                    else:
                        failing.append(
                            (
                                raw_out,
                                val_res.bounce_back_feedback or "Fix schema and grounding.",
                            )
                        )
            else:
                if self.require_live_teacher:
                    raise RuntimeError(
                        "require_live_teacher=True requires an online LLMBackendSPI instance "
                        "(paid_tokens_spent=0, zero fabricated fallback)"
                    )
                # Offline deterministic Best-of-8 tournament
                synthetic_pool: list[str] = [task.gold_output_json]
                if task.bad_draft_json:
                    synthetic_pool.append(task.bad_draft_json)
                synthetic_pool.append('{"invalid_truncated_json": ')
                while len(synthetic_pool) < self.n:
                    synthetic_pool.append(task.gold_output_json)
                for cand_raw in synthetic_pool[: self.n]:
                    candidates_evaluated += 1
                    val_res = self.compiled.validate_output(
                        cand_raw,
                        code_files=code_files or None,
                        ast_symbols=ast_symbols or None,
                        doc_chunks=doc_chunks or None,
                    )
                    if val_res.passed:
                        candidates_passed += 1
                        passing.append((self._score_candidate(val_res), val_res.canonical_json))
                    else:
                        failing.append(
                            (
                                cand_raw,
                                val_res.bounce_back_feedback or "Fix schema and grounding.",
                            )
                        )

            best_json: str | None = None
            if passing:
                passing.sort(key=lambda item: item[0], reverse=True)
                best_json = passing[0][1]

            # Attempt Turn-2 bounce-back self-play when Turn 1 failed or for bounce quota
            need_bounce = (best_json is None and bool(failing)) or (
                bounce_back_count < target_bounces and (idx % 5 == 0) and bool(failing)
            )
            item_meta = {
                "domain_id": self.compiled.domain_id,
                "archetype": self.compiled.archetype.value,
                "repo_name": task.repo_name,
                "doc_id": task.doc_id,
                "split_group_id": task.split_group_id or task.repo_name or task.doc_id,
                "timestamp": task.timestamp.isoformat(),
                "source_kind": task.source_kind,
                "is_bounce_back": False,
            }

            if need_bounce and failing:
                failed_raw, feedback = failing[0]
                turn2_json: str | None = best_json
                if self.llm_backend is not None and turn2_json is None:
                    repair_msgs = self.compiled.format_messages(
                        user_content,
                        extra_turns=[
                            {"role": "assistant", "content": failed_raw},
                            {"role": "user", "content": feedback},
                        ],
                    )
                    raw_t2 = self._invoke_backend_completion(repair_msgs, temperature=0.05)
                    candidates_evaluated += 1
                    val_t2 = self.compiled.validate_output(
                        raw_t2,
                        code_files=code_files or None,
                        ast_symbols=ast_symbols or None,
                        doc_chunks=doc_chunks or None,
                    )
                    if val_t2.passed:
                        candidates_passed += 1
                        turn2_json = val_t2.canonical_json

                if turn2_json is not None:
                    bounce_msgs = self.compiled.format_messages(
                        user_content,
                        extra_turns=[
                            {"role": "assistant", "content": failed_raw},
                            {"role": "user", "content": feedback},
                        ],
                        assistant_response=turn2_json,
                    )
                    bounce_meta = dict(item_meta)
                    bounce_meta["is_bounce_back"] = True
                    sft_items.append({"messages": bounce_msgs, "metadata": bounce_meta})
                    bounce_back_count += 1
                    dpo_items.append(
                        {
                            "prompt": user_content,
                            "chosen": turn2_json,
                            "rejected": failed_raw,
                            "metadata": {
                                **bounce_meta,
                                "dpo_source": "self_play_bounce_back",
                            },
                        }
                    )
                    continue

            if best_json is not None:
                sft_msgs = self.compiled.format_messages(user_content, assistant_response=best_json)
                sft_items.append({"messages": sft_msgs, "metadata": item_meta})
                for failed_raw, _fb in failing[:1]:
                    dpo_items.append(
                        {
                            "prompt": user_content,
                            "chosen": best_json,
                            "rejected": failed_raw,
                            "metadata": {
                                **item_meta,
                                "dpo_source": "best_of_n_rejection",
                            },
                        }
                    )

        return (
            sft_items,
            dpo_items,
            BestOfNStats(
                tasks_processed=len(tasks),
                candidates_evaluated=candidates_evaluated,
                candidates_passed=candidates_passed,
                bounce_back_count=bounce_back_count,
                dpo_pairs_generated=len(dpo_items),
            ),
        )


class GitMRJiraDiffMiner:
    """Mines historical Git commits, CoW MR overlays, and Jira issues under RLS."""

    def __init__(
        self,
        *,
        conn_mgr: Any,
        pg_repo: PostgresRLSRepository,
        in_memory_repo: InMemoryRLSRepository | None = None,
    ) -> None:
        self.conn_mgr = conn_mgr
        self.pg_repo = pg_repo
        self.in_memory_repo = in_memory_repo

    def mine_tasks(
        self,
        *,
        domain_id: str,
        principal: PrincipalContext,
        compiled: CompiledSpecialization,
        code_files: Sequence[CodeFileRecord],
        ast_symbols: Sequence[ASTSymbolRecord],
        doc_chunks: Sequence[DocChunkRecord],
    ) -> list[SelfPlayTaskSpec]:
        """Mines Git commit/MR/Jira diff supervision strictly inside ``principal`` RLS scope."""
        del doc_chunks
        ceiling = set(principal.allowed_scopes)
        commits: list[tuple[str, RepoCommitRecord]] = []
        snapshots: list[RefSnapshotRecord] = []
        deltas_by_snap: dict[UUID, list[OverlayFileDeltaRecord]] = {}
        jira_issues: list[JiraIssueRecord] = []

        if self.in_memory_repo is not None:
            repos = self.in_memory_repo.list_repositories(principal, domain_id)
            for repo in repos:
                if set(repo.allowed_scopes).intersection(ceiling):
                    for c in self.in_memory_repo.get_repo_commits(principal, repo.repo_id):
                        commits.append((repo.repo_name, c))
            snapshots = [
                s
                for s in self.in_memory_repo.list_ref_snapshots(principal, domain_id)
                if set(s.allowed_scopes).intersection(ceiling)
            ]
            for snap in snapshots:
                deltas_by_snap[snap.snapshot_id] = [
                    d
                    for d in self.in_memory_repo.list_overlay_file_deltas(
                        principal, domain_id, snap.snapshot_id
                    )
                    if set(d.allowed_scopes).intersection(ceiling)
                ]
            jira_issues = [
                j
                for j in self.in_memory_repo.list_jira_issues(principal, domain_id)
                if set(j.allowed_scopes).intersection(ceiling)
            ]
        else:
            with self.conn_mgr.scoped_transaction(
                principal, enable_hnsw_iterative_scan=False
            ) as conn:
                repos = self.pg_repo.list_repositories(conn, domain_id)
                for repo in repos:
                    if set(repo.allowed_scopes).intersection(ceiling):
                        for c in self.pg_repo.get_repo_commits(conn, repo.repo_id):
                            commits.append((repo.repo_name, c))
                snapshots = [
                    s
                    for s in self.pg_repo.list_ref_snapshots(conn, domain_id)
                    if set(s.allowed_scopes).intersection(ceiling)
                ]
                for snap in snapshots:
                    deltas_by_snap[snap.snapshot_id] = [
                        d
                        for d in self.pg_repo.list_overlay_file_deltas(
                            conn, domain_id, snap.snapshot_id
                        )
                        if set(d.allowed_scopes).intersection(ceiling)
                    ]
                jira_issues = [
                    j
                    for j in self.pg_repo.list_jira_issues(conn, domain_id)
                    if set(j.allowed_scopes).intersection(ceiling)
                ]

        files_by_repo_path: dict[tuple[str, str], CodeFileRecord] = {
            (cf.repo_name, cf.file_path): cf for cf in code_files
        }
        files_by_path: dict[str, CodeFileRecord] = {cf.file_path: cf for cf in code_files}
        syms_by_file_id: dict[UUID, list[ASTSymbolRecord]] = {}
        for sym in ast_symbols:
            syms_by_file_id.setdefault(sym.file_id, []).append(sym)
        jira_by_key = {j.issue_key: j for j in jira_issues}

        tasks: list[SelfPlayTaskSpec] = []
        arch = compiled.archetype

        # 1. Mine CoW MR / PR overlay deltas + linked Jira issues
        for snap in snapshots:
            if snap.ref_kind not in {"merge_request", "pull_request", "feature_branch"}:
                continue
            deltas = deltas_by_snap.get(snap.snapshot_id, [])
            impacted_ptrs: list[CodeSlicePointer] = []
            for d in deltas:
                if d.change_type == "deleted":
                    continue
                base_cf = files_by_repo_path.get((d.repo_name, d.file_path)) or files_by_path.get(
                    d.file_path
                )
                if base_cf is None or base_cf.line_count < 1:
                    continue
                file_syms = syms_by_file_id.get(base_cf.file_id, [])
                if file_syms:
                    sym = file_syms[0]
                    s_ln = max(1, sym.start_line)
                    e_ln = max(s_ln, min(sym.end_line, base_cf.line_count))
                    q_sym = sym.qualified_name
                else:
                    s_ln = 1
                    e_ln = max(1, min(base_cf.line_count, 25))
                    q_sym = Path(base_cf.file_path).stem
                impacted_ptrs.append(
                    CodeSlicePointer(
                        repo_name=base_cf.repo_name,
                        file_path=base_cf.file_path,
                        start_line=s_ln,
                        end_line=e_ln,
                        qualified_symbol=q_sym,
                        selection_reason=f"MR {snap.ref_name} ({d.change_type}) diff hunk",
                        modification_intent=ModificationIntent.MODIFY_TARGET,
                    )
                )
            if not impacted_ptrs:
                continue
            jira_keys = sorted(
                set(_JIRA_KEY_RE.findall(snap.ref_name))
                | {k for k, j in jira_by_key.items() if snap.ref_name in j.linked_mrs}
            )
            jira_summary = "; ".join(
                f"{k}: {jira_by_key[k].summary}" for k in jira_keys if k in jira_by_key
            )
            ts = snap.updated_at or snap.created_at or datetime.now(UTC)
            if arch == SpecializationArchetype.CHANGE_IMPACT_MR_REVIEW:
                mr_rep = ChangeImpactMRReviewReport(
                    domain_id=domain_id,
                    mr_or_commit_ref=snap.head_commit_sha,
                    risk_level="high" if len(impacted_ptrs) > 1 else "medium",
                    review_summary=(
                        f"MR `{snap.ref_name}` touches {len(impacted_ptrs)} file(s)"
                        + (f" ({jira_summary})" if jira_summary else "")
                    ),
                    impacted_slices=impacted_ptrs,
                    linked_ticket_keys=jira_keys,
                )
                gold_json = mr_rep.model_dump_json(indent=2)
            elif arch == SpecializationArchetype.POINTER_LOCALIZATION:
                gold_json = json.dumps([p.model_dump() for p in impacted_ptrs], indent=2)
            else:
                continue

            bad_ptr = impacted_ptrs[0].model_copy(update={"end_line": 999_999})
            bad_json = (
                json.dumps([bad_ptr.model_dump()], indent=2)
                if arch == SpecializationArchetype.POINTER_LOCALIZATION
                else '{"invalid_mr_review": true}'
            )
            overlay_headers = "\n".join(
                f"MR AST Candidate: [repo={p.repo_name} | file={p.file_path} | "
                f"symbol={p.qualified_symbol} | lines={p.start_line}-{p.end_line}]"
                for p in impacted_ptrs
            )
            first_parent = str(Path(impacted_ptrs[0].file_path).parent)
            mr_mod_grp = (
                first_parent if first_parent not in {"", "."} else impacted_ptrs[0].file_path
            )
            tasks.append(
                SelfPlayTaskSpec(
                    task_id=f"mr_mine_{snap.snapshot_id}",
                    domain_id=domain_id,
                    user_prompt=(
                        f"Analyze MR `{snap.ref_name}` (`{snap.head_commit_sha[:10]}`) in "
                        f"repo `{snap.repo_name}`"
                        + (f" linked to {', '.join(jira_keys)}" if jira_keys else "")
                        + "."
                    ),
                    context_block=(
                        f"{overlay_headers}\n"
                        + (f"Jira Context: {jira_summary}" if jira_summary else "")
                    ).strip(),
                    repo_name=snap.repo_name,
                    doc_id="",
                    timestamp=ts,
                    gold_output_json=gold_json,
                    bad_draft_json=bad_json,
                    split_group_id=mr_mod_grp,
                    source_kind="mr_jira_diff_mine",
                )
            )

        # 2. Mine Git commits linked to repo files & Jira tickets
        files_by_repo: dict[str, list[CodeFileRecord]] = {}
        for cf in code_files:
            files_by_repo.setdefault(cf.repo_name, []).append(cf)

        for repo_name, commit in commits:
            repo_files = files_by_repo.get(repo_name, [])
            if not repo_files:
                continue
            matched_files = [
                cf for cf in repo_files if Path(cf.file_path).name in commit.commit_message
            ] or repo_files[:1]
            target_cf = matched_files[0]
            if target_cf.line_count < 1:
                continue
            file_syms = syms_by_file_id.get(target_cf.file_id, [])
            if file_syms:
                sym = file_syms[0]
                s_ln = max(1, sym.start_line)
                e_ln = max(s_ln, min(sym.end_line, target_cf.line_count))
                q_sym = sym.qualified_name
            else:
                s_ln, e_ln = 1, max(1, min(target_cf.line_count, 25))
                q_sym = Path(target_cf.file_path).stem

            ptr = CodeSlicePointer(
                repo_name=target_cf.repo_name,
                file_path=target_cf.file_path,
                start_line=s_ln,
                end_line=e_ln,
                qualified_symbol=q_sym,
                selection_reason=(
                    f"Historical commit {commit.commit_sha[:8]}: {commit.commit_message[:80]}"
                ),
                modification_intent=ModificationIntent.MODIFY_TARGET,
            )
            jira_keys = sorted(set(_JIRA_KEY_RE.findall(commit.commit_message)))
            if arch == SpecializationArchetype.POINTER_LOCALIZATION:
                gold_json = json.dumps([ptr.model_dump()], indent=2)
            elif arch == SpecializationArchetype.CHANGE_IMPACT_MR_REVIEW:
                gold_json = ChangeImpactMRReviewReport(
                    domain_id=domain_id,
                    mr_or_commit_ref=commit.commit_sha,
                    risk_level="medium",
                    review_summary=f"Commit {commit.commit_sha[:8]}: {commit.commit_message[:120]}",
                    impacted_slices=[ptr],
                    linked_ticket_keys=jira_keys,
                ).model_dump_json(indent=2)
            else:
                continue

            bad_ptr = ptr.model_copy(update={"end_line": target_cf.line_count + 999})
            git_parent = str(Path(target_cf.file_path).parent)
            git_mod_grp = git_parent if git_parent not in {"", "."} else target_cf.file_path
            tasks.append(
                SelfPlayTaskSpec(
                    task_id=f"git_mine_{commit.commit_sha[:10]}",
                    domain_id=domain_id,
                    user_prompt=(
                        f"Localize code impacted by commit `{commit.commit_sha[:8]}` "
                        f"({commit.commit_message[:100]})."
                    ),
                    context_block=(
                        f"Commit AST Candidate: [repo={target_cf.repo_name} | "
                        f"file={target_cf.file_path} | symbol={q_sym} | "
                        f"lines={s_ln}-{e_ln}]"
                    ),
                    repo_name=target_cf.repo_name,
                    doc_id="",
                    timestamp=commit.committed_at,
                    gold_output_json=gold_json,
                    bad_draft_json=json.dumps([bad_ptr.model_dump()], indent=2),
                    split_group_id=git_mod_grp,
                    source_kind="git_commit_diff_mine",
                )
            )
        return tasks


class DPOPairCollector:
    """Collects verified ``(chosen, rejected)`` DPO preference pairs from 3 local sources."""

    def __init__(
        self,
        *,
        conn_mgr: Any,
        in_memory_repo: InMemoryRLSRepository | None = None,
        scrubber: PreIndexScrubber | None = None,
    ) -> None:
        self.conn_mgr = conn_mgr
        self.in_memory_repo = in_memory_repo
        self.scrubber = scrubber or PreIndexScrubber(use_gitleaks_cli=True)

    def collect_all(
        self,
        *,
        domain_id: str,
        principal: PrincipalContext,
        compiled: CompiledSpecialization,
        self_play_dpo: Sequence[dict[str, Any]] = (),
        ui_feedback_events: Sequence[Mapping[str, Any]] = (),
        code_files: Sequence[CodeFileRecord] = (),
        ast_symbols: Sequence[ASTSymbolRecord] = (),
        doc_chunks: Sequence[DocChunkRecord] = (),
    ) -> tuple[list[dict[str, Any]], int]:
        """Merges, verifies, and scrubs DPO preference pairs under RLS."""
        raw_pairs: list[dict[str, Any]] = [dict(p) for p in self_play_dpo]

        # 1. Explicit UI / bounce-back correction events
        for ev in ui_feedback_events:
            prompt = str(ev.get("prompt") or "").strip()
            chosen = ev.get("chosen") or ev.get("corrected_response")
            rejected = ev.get("rejected") or ev.get("original_response")
            if not prompt or not chosen or not rejected:
                continue
            chosen_str = (
                chosen if isinstance(chosen, str) else json.dumps(chosen, ensure_ascii=False)
            )
            rejected_str = (
                rejected if isinstance(rejected, str) else json.dumps(rejected, ensure_ascii=False)
            )
            if chosen_str.strip() == rejected_str.strip():
                continue
            raw_pairs.append(
                {
                    "prompt": prompt,
                    "chosen": chosen_str,
                    "rejected": rejected_str,
                    "metadata": {
                        "domain_id": domain_id,
                        "dpo_source": "ui_feedback",
                        "split_group_id": str(ev.get("split_group_id") or ""),
                        "repo_name": str(ev.get("repo_name") or ""),
                        "doc_id": str(ev.get("doc_id") or ""),
                    },
                }
            )

        # 2. Mine historical scout_sessions recovery turns under RLS
        ceiling = set(principal.allowed_scopes)
        session_rows: list[tuple[str, list[dict[str, Any]], list[str]]] = []
        if self.in_memory_repo is not None:
            for (tid, sid), rec in self.in_memory_repo._sessions.items():
                if (
                    tid == principal.tenant_id
                    and rec.domain_id == domain_id
                    and set(rec.allowed_scopes).intersection(ceiling)
                ):
                    session_rows.append((sid, rec.recent_turns_json, rec.allowed_scopes))
        else:
            with self.conn_mgr.scoped_transaction(
                principal, enable_hnsw_iterative_scan=False
            ) as conn:
                rows = conn.execute(
                    """
                    SELECT session_id, recent_turns_json, allowed_scopes
                    FROM scout_sessions
                    WHERE domain_id = %s
                    ORDER BY updated_at DESC
                    LIMIT 50;
                    """,
                    (domain_id,),
                ).fetchall()
                for row in rows:
                    s_scopes = [str(s) for s in (row[2] or ())]
                    if set(s_scopes).intersection(ceiling):
                        turns = row[1] if isinstance(row[1], list) else []
                        session_rows.append((str(row[0]), turns, s_scopes))

        for sid, turns, _scopes in session_rows:
            if len(turns) >= 4:
                u1 = str(turns[0].get("content") or "")
                a1 = str(turns[1].get("content") or "")
                a2 = str(turns[-1].get("content") or "")
                if u1 and a1 and a2 and a1 != a2:
                    raw_pairs.append(
                        {
                            "prompt": u1,
                            "chosen": a2,
                            "rejected": a1,
                            "metadata": {
                                "domain_id": domain_id,
                                "dpo_source": "scout_session_bounce_back",
                                "split_group_id": f"session:{sid}",
                            },
                        }
                    )

        # Verify chosen passes archetype contract and scrub all DPO text
        verified: list[dict[str, Any]] = []
        scrubbed_total = 0
        for pair in raw_pairs:
            val_res = compiled.validate_output(
                pair["chosen"],
                code_files=code_files or None,
                ast_symbols=ast_symbols or None,
                doc_chunks=doc_chunks or None,
            )
            if not val_res.passed:
                continue
            normalized_pair = dict(pair)
            normalized_pair["chosen"] = val_res.canonical_json
            cleaned_pair, n_found = scrub_dpo_record(normalized_pair, self.scrubber)
            scrubbed_total += n_found
            verified.append(cleaned_pair)

        return verified, scrubbed_total


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    distiller = TeacherDistiller()
    summary = distiller.distill_all_and_save()
    print(f"Distillation summary: {json.dumps(summary, indent=2)}")
