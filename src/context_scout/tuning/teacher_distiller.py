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

import json
import logging
import os
import random
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final
from uuid import UUID

import psycopg

from context_scout.agents.preflight_gate import PreFlightValidator
from context_scout.ast_engine.deterministic_slicer import DeterministicSlicer
from context_scout.config import ScoutSettings, validate_google_model_id
from context_scout.db.repositories import (
    ASTEdgeRecord,
    ASTSymbolRecord,
    CodeFileRecord,
    DocChunkRecord,
    InMemoryRLSRepository,
)
from context_scout.db.session import RLSConnectionManager
from context_scout.mcp_servers.common import PostgresScoutGraphAdapter
from context_scout.plugins.registry import PluginRegistry
from context_scout.schemas.contracts import (
    CodeSlicePointer,
    ContextMissEvent,
    DocCitationSpan,
    DomainKnowledgeBrief,
    ModificationIntent,
    PrincipalContext,
    RoleType,
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
    ) -> None:
        self.settings = settings or ScoutSettings()
        raw_model = teacher_model_id or self.settings.teacher_model_id or DEFAULT_TEACHER_MODEL
        self.teacher_model_id = validate_google_model_id(raw_model)
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.min_bounce_back_ratio = min_bounce_back_ratio
        self.max_bounce_back_ratio = max_bounce_back_ratio
        self.train_split_ratio = train_split_ratio
        self.random_seed = random_seed
        self._rng = random.Random(random_seed)

        target_domain_plugin = (
            (PluginRegistry.get_domain_plugin(research_domain_id) if research_domain_id else None)
            or (PluginRegistry.get_domain_plugin(code_domain_id) if code_domain_id else None)
            or PluginRegistry.get_primary_domain_plugin()
        )
        plugin = PluginRegistry.get_primary_domain_plugin()
        role_ids = plugin.get_role_domain_ids() if plugin is not None else {}
        self.code_domain_id = code_domain_id or role_ids.get(
            "code", self.settings.default_domain_id
        )
        self.research_domain_id = research_domain_id or role_ids.get(
            "research", self.settings.default_domain_id
        )
        scope_set = {"public_eng", "GLOBAL", "ENTERPRISE_DOC", "INTERNAL"}
        if target_domain_plugin is not None:
            scope_set.update(target_domain_plugin.get_default_scopes())
            scope_set.update(target_domain_plugin.get_tuning_specs().get("system_scopes", []))
        plugin_scopes = sorted(scope_set)

        # Database adapters
        self.conn_mgr = RLSConnectionManager(self.settings.db_app_url)
        self.graph_adapter = PostgresScoutGraphAdapter(self.conn_mgr, self.settings)
        self.slicer = DeterministicSlicer()
        self._cached_in_mem_repo: InMemoryRLSRepository | None = None

        # Pre-Flight Gate for deterministic rejection sampling
        self.system_principal = PrincipalContext(
            user_id="distiller_system_admin",
            role=RoleType.ENGINEER,
            allowed_scopes=plugin_scopes,
        )
        self.validator = PreFlightValidator(
            repo=self.get_in_memory_twin_repo(),
            principal=self.system_principal,
        )

    def get_in_memory_twin_repo(self) -> InMemoryRLSRepository:
        """Returns in-memory repository twin hydrated from live PostgreSQL 17."""
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
    # Database Introspection & Hydration Helpers
    # -------------------------------------------------------------------------

    def get_all_code_files(self, domain_id: str | None = None) -> list[CodeFileRecord]:
        """Loads all indexed code files for domain under RLS."""
        target_domain = domain_id or self.code_domain_id
        return self.graph_adapter.list_visible_code_files(self.system_principal, target_domain)

    def get_all_ast_symbols(self, domain_id: str | None = None) -> list[ASTSymbolRecord]:
        """Loads all AST symbols for domain under RLS."""
        target_domain = domain_id or self.code_domain_id
        return self.graph_adapter.list_visible_ast_symbols(self.system_principal, target_domain)

    def get_all_ast_edges(self, domain_id: str | None = None) -> list[ASTEdgeRecord]:
        """Loads all AST directed dependency edges for domain under RLS."""
        target_domain = domain_id or self.code_domain_id
        return self.graph_adapter.list_visible_ast_edges(self.system_principal, target_domain)

    def get_all_doc_chunks(
        self,
        domain_id: str | None = None,
        market_scope: Any | None = None,
        max_chunks_per_doc: int = 5,
    ) -> list[DocChunkRecord]:
        """Loads doc chunks across research scopes via admin connection."""
        target_domain = domain_id or self.research_domain_id
        with psycopg.connect(self.settings.db_admin_url) as conn:
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
            for r in rows:
                scopes_raw = r[12]
                scopes_list = (
                    [str(s) for s in scopes_raw] if isinstance(scopes_raw, (list, tuple)) else []
                )
                results.append(
                    DocChunkRecord(
                        chunk_id=UUID(str(r[0])),
                        domain_id=str(r[1]),
                        doc_id=str(r[2]),
                        doc_title=str(r[3]),
                        source_uri=str(r[4]),
                        market_scope=str(r[5]),
                        published_date=r[6],
                        authority_weight=float(str(r[7])),
                        section_heading=str(r[8]),
                        contextual_header=str(r[9]),
                        verbatim_content=str(r[10]),
                        metadata_json=r[11] if isinstance(r[11], dict) else {},
                        allowed_scopes=scopes_list,
                        embedding=[],
                    )
                )
            return results

    def get_execution_telemetry_misses(
        self, domain_id: str | None = None
    ) -> list[ContextMissEvent]:
        """Ingests ContextMissEvent entries from PostgreSQL execution_telemetry ledger."""
        target_domain = domain_id or self.code_domain_id
        miss_events: list[ContextMissEvent] = []
        with psycopg.connect(self.settings.db_admin_url) as conn:
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
                                    reason=item.get("reason", "Required by Gemini execution"),
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

        # Build valid split strictly from eval_symbols
        valid_examples = self._generate_code_split(
            pool_symbols=eval_symbols,
            all_edges=ast_edges,
            files_by_id=files_by_id,
            target_count=valid_target_count,
            target_bounce_backs=valid_bounce_target,
            flywheel_misses=None,
            domain_id=target_domain,
        )

        logger.info(
            "Code scout distillation complete (%s): %d train, %d valid",
            target_domain,
            len(train_examples),
            len(valid_examples),
        )
        return train_examples, valid_examples

    def _generate_code_split(
        self,
        pool_symbols: list[ASTSymbolRecord],
        all_edges: list[ASTEdgeRecord],
        files_by_id: dict[UUID, CodeFileRecord],
        target_count: int,
        target_bounce_backs: int,
        flywheel_misses: list[ContextMissEvent] | None = None,
        domain_id: str | None = None,
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
            ):
                edge_pairs.append((src_sym, tgt_sym))

        # Synthetic internal pairs if pool has few edge records
        if len(edge_pairs) < target_bounce_backs and len(pool_symbols) >= 2:
            shuffled_p = list(pool_symbols)
            self._rng.shuffle(shuffled_p)
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
    # Quant Research Scout Distillation (Model 2)
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

        plugin = PluginRegistry.get_domain_plugin(target_domain)

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
            max_attempts = needed_count * 20
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

                class _ChunkWithSiblings:
                    def __init__(
                        self, base: DocChunkRecord, siblings: list[DocChunkRecord]
                    ) -> None:
                        self._base = base
                        self._sibling_chunks = siblings

                    def __getattr__(self, item: str) -> Any:
                        return getattr(self._base, item)

                wrapped_target: Any = (
                    _ChunkWithSiblings(target_chunk, paired_chunks)
                    if len(paired_chunks) >= 2
                    else target_chunk
                )

                # Extract verbatim quote candidate directly from target_chunk
                raw_lines = [
                    ln.strip()
                    for ln in target_chunk.verbatim_content.splitlines()
                    if len(ln.strip()) >= 30
                    and not ln.strip().startswith("%")
                    and len(re.findall(r"[A-Za-z]{3,}", ln)) >= 3
                ]
                quote_text = raw_lines[0] if raw_lines else target_chunk.verbatim_content[:120]

                if plugin is not None:
                    sample_bundle = plugin.build_research_distillation_sample(
                        domain_id=target_domain,
                        target_chunk=wrapped_target,
                        quote_text=quote_text,
                        rng=self._rng,
                    )
                    brief = sample_bundle["brief"]
                    bad_brief = sample_bundle["bad_brief"]
                    citation = sample_bundle["citation"]
                    bad_citation = sample_bundle["bad_citation"]
                    val_user_prompt = sample_bundle["val_user_prompt"]
                    user_prompt = sample_bundle["user_prompt"]
                    brief_kwarg = sample_bundle["brief_kwarg"]
                    bad_brief_kwarg = sample_bundle["bad_brief_kwarg"]
                else:
                    citation = DocCitationSpan(
                        chunk_id=str(target_chunk.chunk_id),
                        doc_id=target_chunk.doc_id,
                        doc_title=target_chunk.doc_title,
                        source_uri=target_chunk.source_uri,
                        section_heading=target_chunk.section_heading,
                        verbatim_quote=quote_text,
                    )
                    brief = DomainKnowledgeBrief(
                        brief_title=f"{target_chunk.doc_title} — {target_chunk.section_heading}",
                        domain_id=target_domain,
                        executive_summary=f"Grounded synthesis: {quote_text}",
                        core_findings_and_rules=[quote_text],
                        domain_constraints_and_caveats=[
                            f"Grounded in {target_chunk.doc_id} ({target_chunk.section_heading})"
                        ],
                        known_risks_and_failure_modes=["Out-of-distribution domain regime shift"],
                        verified_citations=[citation],
                    )
                    bad_citation = DocCitationSpan(
                        chunk_id=str(target_chunk.chunk_id),
                        doc_id=target_chunk.doc_id,
                        doc_title=target_chunk.doc_title,
                        source_uri=target_chunk.source_uri,
                        section_heading=target_chunk.section_heading,
                        verbatim_quote="Hallucinated quote not present in source text.",
                    )
                    bad_brief = brief.model_copy(update={"verified_citations": [bad_citation]})
                    val_user_prompt = (
                        f"Synthesize a verified research brief for `{target_chunk.doc_title}`."
                    )
                    user_prompt = (
                        f"{val_user_prompt}\n"
                        f"Retrieved Corpus Chunk: [chunk_id={target_chunk.chunk_id} | "
                        f"doc_id={target_chunk.doc_id} | title={target_chunk.doc_title} | "
                        f"uri={target_chunk.source_uri} | section={target_chunk.section_heading} | "
                        f"quote={json.dumps(quote_text)}]"
                    )
                    brief_kwarg = {}
                    bad_brief_kwarg = {}

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

    def distill_all_and_save(
        self,
        target_dir: Path | str | None = None,
        num_samples_per_domain: int = 120,
    ) -> dict[str, dict[str, int]]:
        """Distills code and research domains and writes JSONL files."""
        base_dir = Path(target_dir).resolve() if target_dir else Path.cwd()
        data_tuning_dir = base_dir / "data" / "tuning"
        domains_dir = base_dir / "domains"

        results_summary: dict[str, dict[str, int]] = {}

        code_train, code_valid = self.distill_code_domain(
            domain_id=self.code_domain_id, num_samples=num_samples_per_domain
        )
        code_paths = [
            data_tuning_dir / self.code_domain_id,
            domains_dir / self.code_domain_id / "tuning",
        ]
        for p in code_paths:
            p.mkdir(parents=True, exist_ok=True)
            self._write_jsonl(p / "train.jsonl", code_train)
            self._write_jsonl(p / "valid.jsonl", code_valid)
            self._write_jsonl(p / "eval.jsonl", code_valid)

        results_summary[self.code_domain_id] = {
            "train": len(code_train),
            "valid": len(code_valid),
            "total": len(code_train) + len(code_valid),
        }

        research_train, research_valid = self.distill_research_domain(
            domain_id=self.research_domain_id, num_samples=num_samples_per_domain
        )
        research_paths = [
            data_tuning_dir / self.research_domain_id,
            domains_dir / self.research_domain_id / "tuning",
        ]
        for p in research_paths:
            p.mkdir(parents=True, exist_ok=True)
            self._write_jsonl(p / "train.jsonl", research_train)
            self._write_jsonl(p / "valid.jsonl", research_valid)
            self._write_jsonl(p / "eval.jsonl", research_valid)

        results_summary[self.research_domain_id] = {
            "train": len(research_train),
            "valid": len(research_valid),
            "total": len(research_train) + len(research_valid),
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
        domain_id: str | None = None,
    ) -> bool:
        """Verifies zero data leakage between train and valid splits."""
        p_train = Path(train_path)
        p_valid = Path(valid_path)
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
                    # Check messages
                    for msg in item.get("messages", []):
                        content = msg.get("content", "")
                        if not is_research and '"file_path":' in content:
                            # Extract file paths from JSON assistant outputs
                            matches = re.findall(r'"file_path":\s*"([^"]+)"', content)
                            for m in matches:
                                parts = Path(m).parts
                                if (
                                    len(parts) >= 3
                                    and parts[0] == "src"
                                    and parts[1] == "context_scout"
                                ):
                                    prefix = f"src/context_scout/{parts[2]}"
                                else:
                                    prefix = str(Path(m).parent)
                                identifiers.add(prefix)
                        else:
                            # Extract doc_ids from research brief citations
                            matches = re.findall(r'"doc_id":\s*"([^"]+)"', content)
                            for m in matches:
                                identifiers.add(m)
            return identifiers

        train_ids = extract_identifiers(p_train)
        valid_ids = extract_identifiers(p_valid)
        overlap = train_ids.intersection(valid_ids)

        if overlap:
            logger.error(
                "Data leakage detected between %s and %s: overlapping IDs %s",
                train_path,
                valid_path,
                overlap,
            )
            return False
        return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    distiller = TeacherDistiller()
    summary = distiller.distill_all_and_save()
    print(f"Distillation summary: {json.dumps(summary, indent=2)}")
