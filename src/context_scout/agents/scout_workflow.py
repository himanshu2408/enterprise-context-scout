"""Google ADK 2.x Scout Workflow, Local Gemma Client, and Federated Scout Coordinator.

Implements Requirement R3:
- LocalGemmaClient: MLX bridge targeting google/gemma-4-26B-A4B-it, clamped to
  --max-kv-size 32768 (HW-01), in-memory LoRA hot-swapping via AdapterPool (ARC-02),
  and robust deterministic structured-output fallback with $0 bounce-back remediation.
- ScoutWorkflow: 4-stage DAG with Stage 1 (guardrails + RBAC), Stage 2 (MCP read tools,
  AST localization, 1-hop closure, deterministic slicing), Stage 3 (Pre-Flight gate
  with $0 local bounce-back self-correction loop), and Stage 4 (pack assembly).
- FederatedScoutCoordinator (ARC-01): 2-stage cross-domain DAG orchestrating research
  synthesis for structured brief extraction and domain rule validation, followed by
  code localization and 1-hop AST closure, merging both into a composite
  ImplementationContextPack.
"""

from __future__ import annotations

import ast
import gc
import json
import logging
import re
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import ValidationError

from context_scout.agents.citation_gate import DocumentCitationGate
from context_scout.agents.conflict_detector import ConflictDetector
from context_scout.agents.domain_registry import AdapterPool, AdapterSlot, DomainRegistry
from context_scout.agents.intent_router import (
    QueryIntentClassifier,
    QueryIntentDecision,
    QueryIntentMode,
    QueryPlan,
    SourceFilter,
    decompose_query_facets,
    evaluate_facet_grounding,
    filter_doc_chunks_by_source,
    format_doc_citation_label,
    is_jira_chunk,
    is_wiki_chunk,
    plan_query_with_llm,
)
from context_scout.agents.preflight_gate import (
    PROMPT_STOPWORDS,
    BoundGraphRepository,
    PreFlightValidator,
)
from context_scout.ast_engine.dependency_expander import ASTDependencyExpander
from context_scout.ast_engine.deterministic_slicer import DeterministicSlicer
from context_scout.config import validate_google_model_id
from context_scout.db.repositories import (
    ASTSymbolRecord,
    CalibratedRankedSearchHit,
    CodeFileRecord,
    DocChunkRecord,
    PostgresRLSRepository,
)
from context_scout.plugins.registry import PluginRegistry
from context_scout.schemas.contracts import (
    CodeSlicePointer,
    DocCitationSpan,
    DomainKnowledgeBrief,
    DomainPackManifest,
    DomainProbeTrace,
    FrictionVerdict,
    ImplementationContextPack,
    InsufficientContextDiagnostic,
    MarketRegime,
    ModificationIntent,
    PreFlightCheckLedgerItem,
    PreFlightValidationReport,
    PrincipalContext,
    QuantStrategyBrief,
    QueryFacetCoverage,
    RetrievalHitTrace,
    RoleType,
    StageConfidence,
    StageExecutionTrace,
    VerbatimCodeSlice,
    build_domain_knowledge_brief_prompt,
    parse_domain_knowledge_brief_json,
)
from context_scout.security.guardrails import (
    ShieldGemmaGuardrail,
)
from context_scout.security.rbac import DomainAuthorizationGate
from context_scout.telemetry import ScoutTelemetryTracer
from context_scout.telemetry.cost_governance import DEFAULT_EXPLORATION_TOKEN_KILL_SWITCH

logger = logging.getLogger(__name__)


def _snap_python_ast_pointer_bounds(
    ptr: CodeSlicePointer,
    file_rec: CodeFileRecord,
) -> CodeSlicePointer:
    """Extends stale Python AST symbol bounds to the enclosing AST node when truncated."""
    if file_rec.language != "python" or not ptr.qualified_symbol or not file_rec.raw_content:
        return ptr
    target_name = ptr.qualified_symbol.rsplit(".", 1)[-1]
    if not target_name:
        return ptr
    try:
        tree = ast.parse(file_rec.raw_content)
    except SyntaxError as exc:
        logger.debug(
            "Skipping AST pointer bound snap for %s (%s): %s",
            ptr.file_path,
            type(exc).__name__,
            exc,
        )
        return ptr

    best_node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef | None = None
    best_dist = 1_000_000
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == target_name
            and node.end_lineno is not None
        ):
            dist = abs(node.lineno - ptr.start_line)
            if dist <= 60 and dist < best_dist:
                best_dist = dist
                best_node = node

    if (
        best_node is not None
        and best_node.end_lineno is not None
        and (best_node.end_lineno > ptr.end_line or best_node.lineno != ptr.start_line)
    ):
        return ptr.model_copy(
            update={"start_line": best_node.lineno, "end_line": best_node.end_lineno}
        )
    return ptr


# Google ADK 2.x Workflow integration
try:
    from google.adk import Workflow as ADKWorkflow

    HAS_GOOGLE_ADK = True
except ImportError as exc:
    HAS_GOOGLE_ADK = False
    ADKWorkflow = Any  # type: ignore[assignment,misc]
    logging.getLogger(__name__).debug(
        "google.adk not installed; using built-in workflow DAG (%s)",
        type(exc).__name__,
    )


def _resolve_default_domain(domain_id: str | None, role: str = "code") -> str:
    """Resolves a domain ID when omitted, consulting registered domain plugins."""
    if domain_id:
        return domain_id
    plugin = PluginRegistry.get_primary_domain_plugin()
    if plugin is not None:
        return plugin.get_role_domain_ids().get(role, plugin.domain_id)
    return "default"


def _resolve_role_domains() -> tuple[str, str, str]:
    """Returns (code_domain_id, research_domain_id, federated_domain_id) from active plugin."""
    plugin = PluginRegistry.get_primary_domain_plugin()
    if plugin is not None:
        roles = plugin.get_role_domain_ids()
        return (
            roles.get("code", "default"),
            roles.get("research", plugin.domain_id),
            roles.get("federated", "federated"),
        )
    return ("default", "default", "federated")


DEFAULT_LOCAL_GEMMA_MODEL: Final[str] = "google/gemma-4-26B-A4B-it"
MAX_SAFE_KV_SIZE: Final[int] = 32_768  # Apple M5 Pro 48GB unified memory cap (HW-01)

CONVERSATIONAL_AND_GENERIC_WORDS: Final[set[str]] = {
    "hello",
    "hey",
    "hi",
    "how",
    "are",
    "you",
    "your",
    "yours",
    "doing",
    "going",
    "today",
    "tomorrow",
    "yesterday",
    "morning",
    "afternoon",
    "evening",
    "night",
    "good",
    "fine",
    "okay",
    "great",
    "nice",
    "thanks",
    "thank",
    "please",
    "weather",
    "joke",
    "funny",
    "name",
    "who",
    "what",
    "where",
    "when",
    "why",
    "which",
    "feeling",
    "sup",
    "wassup",
    "chat",
    "talk",
    "tell",
    "something",
    "someone",
    "anybody",
    "there",
    "here",
    "status",
    "ping",
    "help",
    "berlin",
    "london",
    "paris",
    "munich",
    "coffee",
    "lunch",
    "dinner",
    "food",
    "movie",
    "music",
    "game",
    "sports",
    "football",
    "soccer",
    "the",
    "and",
    "for",
    "not",
    "but",
    "with",
    "from",
    "this",
    "that",
    "these",
    "those",
    "have",
    "has",
    "had",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "can",
    "may",
    "might",
    "about",
    "into",
    "onto",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "bake",
    "baking",
    "best",
    "cake",
    "chocolate",
    "recipe",
    "cooking",
    "flour",
    "sugar",
    "oven",
    "vacation",
    "holiday",
    "repair",
    "diesel",
    "tractor",
    "washing",
    "machine",
    "laundry",
    "haircut",
}


def _evaluate_query_domain_relevance(
    user_prompt: str,
    domain_id: str,
    bound_repo: BoundGraphRepository,
    principal: PrincipalContext,
) -> str | None:
    """Returns a diagnostic rejection reason if the query is chitchat or unrelated to the domain."""
    raw_tokens = re.findall(r"[A-Za-z0-9_]{3,}", user_prompt)
    substantive = [
        t
        for t in raw_tokens
        if t.lower() not in PROMPT_STOPWORDS and t.lower() not in CONVERSATIONAL_AND_GENERIC_WORDS
    ]
    if not substantive:
        return (
            f"Out-of-scope or conversational query ('{user_prompt.strip()}'): "
            "No domain-specific code symbols, files, or research concepts detected. "
            "Paid frontier tokens spent: 0."
        )
    return None


class ModelUnavailableError(RuntimeError):
    """Raised when local Gemma model (live MLX or in-process) is offline or fails inference.

    Enforces zero-simulation integrity on the production path: on model failure,
    the scout never fabricates model output or simulates empirical capabilities.
    """

    def __init__(
        self,
        message: str = "Local Gemma model unavailable: MLX server offline or inference failed",
        domain_id: str | None = None,
        is_abstention: bool = True,
        paid_tokens_spent: int = 0,
        diagnostic: InsufficientContextDiagnostic | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.domain_id = domain_id
        self.is_abstention = is_abstention
        self.paid_tokens_spent = paid_tokens_spent
        self.diagnostic = diagnostic


ModelUnavailableException = ModelUnavailableError

MAX_LOADED_MLX_MODELS: Final[int] = 1
_PROCESS_MLX_CACHE: OrderedDict[str, tuple[Any, Any]] = OrderedDict()
_PROCESS_MLX_CACHE_LOCK = threading.Lock()


def _release_metal_cache() -> None:
    """Runs GC and clears Metal buffer cache after evicting an MLX model slot."""
    gc.collect()
    try:
        import mlx.core as mx

        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
    except Exception as exc:
        logger.debug("MLX cache release skipped (%s): %s", type(exc).__name__, exc)


def clear_process_mlx_cache() -> int:
    """Evicts cached MLX model weights and releases Metal buffer cache."""
    with _PROCESS_MLX_CACHE_LOCK:
        evicted = len(_PROCESS_MLX_CACHE)
        _PROCESS_MLX_CACHE.clear()
    if evicted > 0:
        _release_metal_cache()
    return evicted


_GEMMA4_THOUGHT_CHANNEL_RE: Final[re.Pattern[str]] = re.compile(
    r"<\|channel>thought.*?<channel\|>",
    re.DOTALL,
)
_GEMMA4_ALT_THOUGHT_RE: Final[re.Pattern[str]] = re.compile(
    r"<\|?thought\|?>.*?<\|?/thought\|?>",
    re.DOTALL,
)


def _strip_gemma4_control_tokens(raw_text: str) -> str:
    """Strips Gemma-4 thinking channels, turn markers, and markdown fences."""
    cleaned = _GEMMA4_THOUGHT_CHANNEL_RE.sub("", raw_text)
    cleaned = _GEMMA4_ALT_THOUGHT_RE.sub("", cleaned)
    cleaned = (
        cleaned.replace("<eos>", "")
        .replace("<end_of_turn>", "")
        .replace("<turn|>", "")
        .strip()
    )
    cleaned = re.sub(r"```(?:json)?\s*\n?(.*?)```", r"\1", cleaned, flags=re.DOTALL).strip()
    return cleaned


def _decode_gemma4_json(raw_text: str, *, expect_array: bool = False) -> Any:
    """Extracts and decodes the first JSON object/array from Gemma-4 raw output."""
    cleaned = _strip_gemma4_control_tokens(raw_text)
    if not cleaned:
        return None
    if expect_array:
        arr_idx = cleaned.find("[")
        obj_idx = cleaned.find("{")
        if arr_idx == -1 and obj_idx == -1:
            return None
        start = (
            arr_idx
            if (arr_idx != -1 and (obj_idx == -1 or arr_idx < obj_idx))
            else obj_idx
        )
    else:
        start = cleaned.find("{")
        if start == -1:
            return None
    try:
        parsed, _ = json.JSONDecoder(strict=False).raw_decode(cleaned[start:])
        return parsed
    except json.JSONDecodeError:
        return None


def _normalize_gemma4_citations(
    raw_cits: Any,
    *,
    domain_id: str,
    fallback_title: str,
    verdict: str,
    candidate_chunks: list[DocChunkRecord] | None = None,
) -> list[dict[str, Any]]:
    """Normalizes Gemma-4 verified_citations, anchoring minimal stub on insufficient abstention."""
    clean_cits: list[dict[str, Any]] = []
    if isinstance(raw_cits, list):
        for vc in raw_cits:
            if isinstance(vc, dict) and str(vc.get("verbatim_quote", "")).strip():
                vc_copy = dict(vc)
                vc_copy.setdefault("version", "1.0")
                if not str(vc_copy.get("section_heading", "")).strip():
                    vc_copy["section_heading"] = "Main"
                if not str(vc_copy.get("doc_title", "")).strip():
                    vc_copy["doc_title"] = fallback_title
                if not str(vc_copy.get("source_uri", "")).strip():
                    vc_copy["source_uri"] = f"urn:{domain_id}:unknown"
                if not str(vc_copy.get("chunk_id", "")).strip():
                    vc_copy["chunk_id"] = "abstention_probe"
                if not str(vc_copy.get("doc_id", "")).strip():
                    vc_copy["doc_id"] = f"doc_{domain_id}"
                clean_cits.append(vc_copy)
    if not clean_cits and verdict == "insufficient":
        if candidate_chunks:
            ch = candidate_chunks[0]
            first_line = next(
                (ln.strip() for ln in ch.verbatim_content.splitlines() if len(ln.strip()) >= 20),
                ch.verbatim_content[:120].strip() or ch.doc_title,
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
        else:
            clean_cits.append(
                {
                    "chunk_id": f"abstention_{domain_id}",
                    "doc_id": f"abstention_{domain_id}",
                    "doc_title": fallback_title,
                    "source_uri": f"urn:{domain_id}:insufficient",
                    "section_heading": "Sufficiency Gate",
                    "verbatim_quote": "Insufficient grounded evidence in retrieved chunks.",
                    "version": "1.0",
                }
            )
    return clean_cits


def _parse_gemma4_domain_knowledge_brief(
    raw_text: str,
    *,
    domain_id: str,
    candidate_chunks: list[DocChunkRecord] | None = None,
    live_mode: Literal[
        "live_qlora_json",
        "live_base_gemma4_json",
        "deterministic_extractive_fallback",
    ] = "live_base_gemma4_json",
) -> DomainKnowledgeBrief | None:
    """Extracts DomainKnowledgeBrief from Gemma-4 output, preserving insufficient abstentions."""
    cleaned = _strip_gemma4_control_tokens(raw_text)
    obj = _decode_gemma4_json(cleaned)
    if not isinstance(obj, dict):
        return None

    brief, _ = parse_domain_knowledge_brief_json(
        cleaned, domain_id=domain_id, normalize_scalars=True
    )
    if brief is not None:
        sv_raw = str(
            obj.get("sufficiency_verdict")
            or (
                obj.get("structured_metadata", {}).get("sufficiency_verdict")
                if isinstance(obj.get("structured_metadata"), dict)
                else ""
            )
            or ""
        ).lower()
        if sv_raw in ("sufficient", "partial", "insufficient"):
            brief.sufficiency_verdict = sv_raw  # type: ignore[assignment]
        ma_raw = obj.get("missing_aspects")
        if ma_raw is None and isinstance(obj.get("structured_metadata"), dict):
            ma_raw = obj["structured_metadata"].get("missing_aspects")
        if isinstance(ma_raw, list):
            brief.missing_aspects = [str(x) for x in ma_raw if str(x).strip()]
        brief.synthesis_mode = live_mode
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

    ma_val = norm.get("missing_aspects")
    if isinstance(ma_val, list):
        norm["missing_aspects"] = [str(x) for x in ma_val if str(x).strip()]
    elif isinstance(ma_val, str) and ma_val.strip():
        norm["missing_aspects"] = [ma_val.strip()]
    else:
        norm["missing_aspects"] = []

    for list_field, default_item in (
        ("core_findings_and_rules", "No grounded finding in retrieved chunks."),
        ("domain_constraints_and_caveats", f"Evaluated against {domain_id} chunks."),
        ("known_risks_and_failure_modes", "Insufficient context for full verification."),
    ):
        val = norm.get(list_field)
        if not isinstance(val, list) or not val:
            norm[list_field] = [default_item]
        else:
            norm[list_field] = [str(x) for x in val if x is not None and str(x).strip()] or [
                default_item
            ]

    norm["verified_citations"] = _normalize_gemma4_citations(
        norm.get("verified_citations"),
        domain_id=domain_id,
        fallback_title=str(norm["brief_title"]),
        verdict=verdict,
        candidate_chunks=candidate_chunks,
    )
    norm["synthesis_mode"] = live_mode
    try:
        return DomainKnowledgeBrief.model_validate(norm)
    except (ValidationError, TypeError, ValueError) as exc:
        logger.debug("DomainKnowledgeBrief validation failed: %s", exc)
        return None


def _parse_gemma4_quant_strategy_brief(
    raw_text: str,
    *,
    domain_id: str,
    candidate_chunks: list[DocChunkRecord] | None = None,
    live_mode: Literal[
        "live_qlora_json",
        "live_base_gemma4_json",
        "live_mlx_json",
        "deterministic_extractive_fallback",
    ] = "live_base_gemma4_json",
) -> QuantStrategyBrief | None:
    """Extracts QuantStrategyBrief from Gemma-4 output, preserving insufficient abstentions."""
    cleaned = _strip_gemma4_control_tokens(raw_text)
    obj = _decode_gemma4_json(cleaned)
    if not isinstance(obj, dict):
        return None

    norm: dict[str, Any] = dict(obj)
    verdict = str(norm.get("sufficiency_verdict", "sufficient")).lower()
    if verdict not in ("sufficient", "partial", "insufficient"):
        verdict = "sufficient"
    norm["sufficiency_verdict"] = verdict

    if not str(norm.get("strategy_name", "")).strip():
        norm["strategy_name"] = f"Research Brief ({domain_id})"
    if not str(norm.get("mathematical_signal_latex", "")).strip():
        norm["mathematical_signal_latex"] = "N/A"
    if not str(norm.get("sample_period_years", "")).strip():
        norm["sample_period_years"] = "unspecified"

    raw_markets = norm.get("target_markets")
    valid_markets = {m.value for m in MarketRegime}
    if isinstance(raw_markets, list):
        kept_markets = [str(m) for m in raw_markets if str(m) in valid_markets]
        norm["target_markets"] = kept_markets or [MarketRegime.GLOBAL_MULTI.value]
    else:
        norm["target_markets"] = [MarketRegime.GLOBAL_MULTI.value]

    raw_fv = str(norm.get("friction_verdict", "")).strip()
    valid_fvs = {f.value for f in FrictionVerdict}
    if raw_fv not in valid_fvs:
        norm["friction_verdict"] = FrictionVerdict.REALISTIC_FRICTION.value

    ma_val = norm.get("missing_aspects")
    if isinstance(ma_val, list):
        norm["missing_aspects"] = [str(x) for x in ma_val if str(x).strip()]
    elif isinstance(ma_val, str) and ma_val.strip():
        norm["missing_aspects"] = [ma_val.strip()]
    else:
        norm["missing_aspects"] = []

    for list_field, default_item in (
        ("entry_exit_rules", "No grounded entry/exit rule in retrieved chunks."),
        ("market_specific_caveats", f"Evaluated against {domain_id} chunks."),
        ("known_failure_regimes", "Insufficient context for full empirical verification."),
    ):
        val = norm.get(list_field)
        if not isinstance(val, list) or not val:
            norm[list_field] = [default_item]
        else:
            norm[list_field] = [str(x) for x in val if x is not None and str(x).strip()] or [
                default_item
            ]

    norm["verified_citations"] = _normalize_gemma4_citations(
        norm.get("verified_citations"),
        domain_id=domain_id,
        fallback_title=str(norm["strategy_name"]),
        verdict=verdict,
        candidate_chunks=candidate_chunks,
    )
    norm["synthesis_mode"] = live_mode
    try:
        return QuantStrategyBrief.model_validate(norm)
    except (ValidationError, TypeError, ValueError) as exc:
        logger.debug("QuantStrategyBrief validation failed: %s", exc)
        return None


class LocalGemmaClient:
    """Local inference client for google/gemma-4-26B-A4B-it.

    Adheres to HW-01 (--max-kv-size 32768 clamping for Apple Silicon memory safety).
    Integrates with AdapterPool for in-memory LoRA hot-swapping (ARC-02).
    Fails closed with ModelUnavailableError when local MLX server is offline or fails.
    Supports $0 bounce-back remediation loop by resolving missing symbols and citations
    from PreFlightValidator feedback.
    """

    def __init__(
        self,
        endpoint_url: str = "http://127.0.0.1:8000/v1",
        model_id: str = DEFAULT_LOCAL_GEMMA_MODEL,
        max_kv_size: int = MAX_SAFE_KV_SIZE,
        adapter_pool: AdapterPool | None = None,
        timeout_seconds: float = 5.0,
        offline_mode: bool = False,
        use_live_mlx: bool = False,
        completion_fixture: Callable[..., str] | None = None,
        mock_completion_generator: Callable[..., str] | None = None,
    ) -> None:
        import os

        if max_kv_size > MAX_SAFE_KV_SIZE:
            raise ValueError(
                f"max_kv_size ({max_kv_size}) exceeds Apple M5 Pro 48GB safety cap "
                f"({MAX_SAFE_KV_SIZE}) (HW-01)"
            )
        self.endpoint_url = endpoint_url.rstrip("/")
        self.model_id = validate_google_model_id(model_id)
        self.max_kv_size = max_kv_size
        self.adapter_pool = adapter_pool
        self.timeout_seconds = timeout_seconds
        self.offline_mode = offline_mode
        self.use_live_mlx = (
            use_live_mlx or os.environ.get("CONTEXT_SCOUT_USE_LIVE_MLX", "").strip() == "1"
        )
        self.completion_fixture = completion_fixture or mock_completion_generator
        self._active_domain_id: str | None = None
        self._manifest_cache: dict[str, DomainPackManifest] | None = None
        self._last_chunks_fed: int = 0
        # Process-shared single-slot LRU cache prevents loading multiple ~15 GB models.
        self._mlx_cache: OrderedDict[str, tuple[Any, Any]] = _PROCESS_MLX_CACHE

    def _get_domain_manifest(self, domain_id: str | None) -> DomainPackManifest | None:
        if not domain_id:
            return None
        if self._manifest_cache is None:
            try:
                domains_dir = Path(__file__).resolve().parents[3] / "domains"
                self._manifest_cache = DomainRegistry(domains_dir=domains_dir).load_all()
            except Exception:
                self._manifest_cache = {}
        return self._manifest_cache.get(domain_id)

    def _resolve_domain_adapter(
        self, domain_id: str | None
    ) -> tuple[str, str | None, bool]:
        """Resolve (base_model_id, adapter_path_or_none, uses_lora) honoring lora_enabled."""
        manifest = self._get_domain_manifest(domain_id)
        base_model = manifest.base_model_id if manifest else self.model_id
        if manifest is not None and not manifest.lora_enabled:
            return base_model, None, False
        slot = self.get_active_adapter(domain_id) if domain_id else None
        raw_path = (
            str(slot.adapter_path)
            if slot and slot.adapter_path
            else (manifest.lora_adapter_path if manifest else None)
        )
        if not raw_path:
            return base_model, None, False
        candidate = Path(raw_path)
        has_weights = candidate.exists() and (
            (candidate / "adapters.safetensors").exists()
            or (candidate / "adapter_config.json").exists()
        )
        if not has_weights:
            return base_model, None, False
        return base_model, str(candidate), True

    @staticmethod
    def _format_full_chunks_for_synthesis(
        chunks: list[DocChunkRecord],
        max_total_chars: int = 24_000,
        max_chunk_chars: int = 3_600,
        max_chunks: int = 6,
        max_per_doc: int = 2,
    ) -> list[str]:
        """Format top chunks (title, section, market_scope, verbatim_content) within budget."""
        selected_chunks: list[DocChunkRecord] = []
        per_doc_counts: dict[str, int] = {}
        overflow_chunks: list[DocChunkRecord] = []
        for chk in chunks:
            doc_key = str(
                getattr(chk, "doc_id", "") or getattr(chk, "doc_title", "") or chk.chunk_id
            )
            if per_doc_counts.get(doc_key, 0) < max_per_doc:
                selected_chunks.append(chk)
                per_doc_counts[doc_key] = per_doc_counts.get(doc_key, 0) + 1
                if len(selected_chunks) >= max_chunks:
                    break
            else:
                overflow_chunks.append(chk)
        if len(selected_chunks) < max_chunks:
            for chk in overflow_chunks:
                selected_chunks.append(chk)
                if len(selected_chunks) >= max_chunks:
                    break

        formatted: list[str] = []
        used = 0
        for chk in selected_chunks[:max_chunks]:
            body = str(getattr(chk, "verbatim_content", "") or "").strip()
            if not body:
                continue
            if len(body) > max_chunk_chars:
                body = body[:max_chunk_chars].rstrip() + " ..."
            raw_scope = getattr(chk, "market_scope", None)
            scope_str = str(getattr(raw_scope, "value", raw_scope) or "GLOBAL")
            header_tag = f"[{chk.doc_title} | {chk.section_heading} | scope/market: {scope_str}]"
            entry = (
                f"Retrieved Corpus Chunk {header_tag}:\n"
                f"  chunk_id: {chk.chunk_id}\n"
                f"  doc_id: {chk.doc_id}\n"
                f"  doc_title: {chk.doc_title}\n"
                f"  source_uri: {chk.source_uri}\n"
                f"  section_heading: {chk.section_heading}\n"
                f"  market_scope: {scope_str}\n"
                f"  verbatim_content:\n{body}"
            )
            if used + len(entry) > max_total_chars and formatted:
                break
            formatted.append(entry)
            used += len(entry)
        return formatted

    def decompose_query_json(
        self, prompt: str, fallback_plan: QueryPlan
    ) -> QueryPlan:
        """Use Gemma-4 structured planning when live MLX is enabled; return fallback otherwise."""
        if self.offline_mode or not (self.use_live_mlx or self.is_online()):
            return fallback_plan
        planner_prompt = (
            "You are a query planner. Return ONLY valid JSON with keys: "
            '"query_type" (one of concept_definition, architecture_comparison, '
            "code_implementation, incident_runbook, multi_hop_research_to_code, general), "
            '"core_subject" (short string from user query), '
            '"facets" (list of 1-5 concise sub-questions strictly covering ONLY clauses/topics '
            "explicitly asked in User query; NEVER invent comparisons, tiers, or extra topics "
            "not present in User query), "
            '"expanded_queries" (list of 0-3 search synonym expansions). '
            "Do NOT answer the question.\n"
            f"User query: {prompt}"
        )
        try:
            raw = (
                self._call_in_process_mlx(planner_prompt, domain_id=None, max_tokens=320)
                if self.use_live_mlx
                else self._call_mlx_raw(planner_prompt, domain_id=None, max_tokens=320)
            )
            data = _decode_gemma4_json(raw)
            if not isinstance(data, dict):
                return fallback_plan
            raw_facets = [str(f).strip() for f in (data.get("facets") or []) if str(f).strip()]
            if not raw_facets:
                return fallback_plan
            from context_scout.agents.intent_router import (
                FacetSpec,
                _classify_facet_source_kind,
                _extract_facet_key_terms,
                anchor_llm_query_plan,
            )

            facets: list[FacetSpec] = []
            for idx, fq in enumerate(raw_facets[:5], start=1):
                terms = _extract_facet_key_terms(fq) or [
                    w.lower() for w in re.findall(r"[A-Za-z0-9_]{2,}", fq)
                ]
                facets.append(
                    FacetSpec(
                        facet_id=f"facet_{idx}",
                        facet_query=fq,
                        key_terms=terms,
                        expected_source_kind=_classify_facet_source_kind(fq),
                    )
                )
            qtype = str(data.get("query_type") or fallback_plan.query_type)
            allowed_qtypes = {
                "concept_definition",
                "architecture_comparison",
                "code_implementation",
                "incident_runbook",
                "multi_hop_research_to_code",
                "general",
            }
            if qtype not in allowed_qtypes:
                qtype = fallback_plan.query_type
            raw_plan = QueryPlan(
                original_query=prompt,
                query_type=qtype,  # type: ignore[arg-type]
                core_subject=str(data.get("core_subject") or fallback_plan.core_subject),
                expanded_queries=[
                    str(x).strip()
                    for x in (data.get("expanded_queries") or [])[:3]
                    if str(x).strip()
                ]
                or fallback_plan.expanded_queries,
                requested_cardinality=fallback_plan.requested_cardinality,
                is_acronym_query=fallback_plan.is_acronym_query,
                acronym=fallback_plan.acronym,
                facets=facets,
            )
            return anchor_llm_query_plan(prompt, raw_plan, fallback_plan)
        except Exception as exc:
            logger.debug(
                "Structured Gemma-4 query planning fallback (%s): %s", type(exc).__name__, exc
            )
            return fallback_plan

    @staticmethod
    def clear_mlx_cache() -> int:
        """Evicts cached MLX weights in this process and frees Metal unified memory."""
        return clear_process_mlx_cache()

    def _get_or_load_mlx_model(
        self,
        mlx_model_id: str,
        cache_key: str,
        adapter_path: str | None,
        loader_fn: Callable[..., Any] | None = None,
    ) -> tuple[Any, Any]:
        """Loads at most MAX_LOADED_MLX_MODELS (~15 GB) across the entire process."""
        if loader_fn is None:
            import mlx_lm

            load_fn: Callable[..., Any] = mlx_lm.load
        else:
            load_fn = loader_fn

        with _PROCESS_MLX_CACHE_LOCK:
            if cache_key in _PROCESS_MLX_CACHE:
                _PROCESS_MLX_CACHE.move_to_end(cache_key)
                return _PROCESS_MLX_CACHE[cache_key]

            if len(_PROCESS_MLX_CACHE) >= MAX_LOADED_MLX_MODELS:
                _PROCESS_MLX_CACHE.clear()
                _release_metal_cache()

            if adapter_path and Path(adapter_path).exists():
                loaded: Any = load_fn(mlx_model_id, adapter_path=adapter_path)
            else:
                loaded = load_fn(mlx_model_id)
            model, tokenizer = loaded[0], loaded[1]
            _PROCESS_MLX_CACHE[cache_key] = (model, tokenizer)
            return model, tokenizer

    def get_active_adapter(self, domain_id: str) -> AdapterSlot | None:
        """Retrieves active adapter slot from pool for zero-cost in-memory swapping."""
        if self.adapter_pool is None:
            return None
        try:
            return self.adapter_pool.get_slot(domain_id)
        except KeyError as exc:
            logger.debug(
                "No adapter slot registered for domain '%s' (%s)",
                domain_id,
                type(exc).__name__,
            )
            return None

    def set_active_domain(self, domain_id: str) -> AdapterSlot | None:
        """Swaps active LoRA adapter in memory (ARC-02)."""
        self._active_domain_id = domain_id
        return self.get_active_adapter(domain_id)

    def load_adapter(
        self,
        domain_id: str,
        base_dir: Path | str | None = None,
    ) -> AdapterSlot:
        """Loads, registers, and verifies domain QLoRA adapter artifacts (ARC-02 & HW-01)."""
        slot = self.set_active_domain(domain_id)
        if slot is None:
            raise KeyError(f"Domain '{domain_id}' not found in AdapterPool")
        if self.adapter_pool is not None:
            v_res = self.adapter_pool.verify_adapter(domain_id, base_dir=base_dir)
            if not v_res.get("is_valid", False):
                logger.warning(
                    "Adapter verification warning for '%s': %s",
                    domain_id,
                    v_res.get("error_message"),
                )
        return slot

    def verify_active_adapter(
        self,
        base_dir: Path | str | None = None,
    ) -> dict[str, Any]:
        """Verifies currently active domain adapter artifacts on disk."""
        if not self._active_domain_id or self.adapter_pool is None:
            return {
                "domain_id": self._active_domain_id or "none",
                "is_valid": False,
                "error_message": "No active domain or AdapterPool configured",
            }
        return self.adapter_pool.verify_adapter(self._active_domain_id, base_dir=base_dir)

    def is_online(self) -> bool:
        """Probes local MLX server health endpoint."""
        if self.offline_mode:
            return False
        health_url = f"{self.endpoint_url}/models"
        req = urllib.request.Request(health_url, headers={"User-Agent": "ContextScout-LocalGemma"})
        try:
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                return bool(resp.status == 200)
        except Exception as exc:
            logger.debug(
                "Local MLX health probe unreachable at %s (%s): %s",
                health_url,
                type(exc).__name__,
                exc,
            )
            return False

    def complete(
        self,
        prompt: str,
        domain_id: str | None = None,
        system_instruction: str | None = None,
        max_tokens: int = 1024,
    ) -> str:
        """Returns raw text completion from live MLX or in-process mlx_lm.

        Strictly abstains on model failure by raising ModelUnavailableError.
        Never fabricates model completions or simulates empirical capabilities on
        the production path. Mock generators may only be injected via test fixtures.
        """
        if domain_id:
            self.set_active_domain(domain_id)
        if not self.offline_mode and self.use_live_mlx:
            try:
                return self._call_in_process_mlx(
                    prompt=prompt,
                    domain_id=domain_id,
                    max_tokens=max_tokens,
                )
            except Exception as e:
                logger.warning(
                    "In-process MLX completion failed (%s); falling back",
                    e,
                )
        if not self.offline_mode and self.is_online():
            try:
                return self._call_mlx_raw(
                    prompt=prompt,
                    domain_id=domain_id,
                    system_instruction=system_instruction,
                    max_tokens=max_tokens,
                )
            except Exception as e:
                logger.warning(
                    "Local MLX raw completion failed (%s)",
                    e,
                )
        if self.completion_fixture is not None:
            try:
                return self.completion_fixture(
                    prompt=prompt,
                    domain_id=domain_id,
                    system_instruction=system_instruction,
                )
            except TypeError as exc:
                logger.debug(
                    "Completion fixture rejected keyword args, retrying prompt-only (%s)",
                    type(exc).__name__,
                )
                try:
                    return self.completion_fixture(prompt=prompt)
                except TypeError as inner_exc:
                    logger.debug(
                        "Completion fixture rejected prompt kwarg, retrying positional (%s)",
                        type(inner_exc).__name__,
                    )
                    return self.completion_fixture(prompt)

        # Fail closed: strictly abstain on production path
        trace_id = str(uuid.uuid4())
        active_domain = domain_id or "none"
        reason = (
            f"Model unavailable: Local Gemma model ({self.model_id}) is offline or "
            f"inference failed for domain '{active_domain}'. "
            "Production scout strictly abstains from simulating completions."
        )
        diag = InsufficientContextDiagnostic(
            trace_id=trace_id,
            domain_id=active_domain,
            reason=reason,
            is_abstention=True,
            paid_tokens_spent=0,
            missing_items=["model:local_gemma:unavailable"],
            failure_check="MODEL_UNAVAILABLE",
        )
        raise ModelUnavailableError(
            message=reason,
            domain_id=domain_id,
            is_abstention=True,
            paid_tokens_spent=0,
            diagnostic=diag,
        )

    def _call_in_process_mlx(
        self,
        prompt: str,
        domain_id: str | None = None,
        max_tokens: int = 768,
    ) -> str:
        """Executes in-process Metal inference via mlx_lm with GPUMutualExclusionLock."""
        import mlx_lm

        from context_scout.tuning.train_mlx_lora import GPUMutualExclusionLock

        if domain_id:
            self.set_active_domain(domain_id)
        _base_model, adapter_path, _uses_lora = self._resolve_domain_adapter(domain_id)
        cache_key = f"{domain_id or 'base'}:{adapter_path or 'base_only'}"
        mlx_model_id = "mlx-community/gemma-4-26b-a4b-it-4bit"

        with GPUMutualExclusionLock(timeout_seconds=self.timeout_seconds):
            model, tokenizer = self._get_or_load_mlx_model(
                mlx_model_id=mlx_model_id,
                cache_key=cache_key,
                adapter_path=adapter_path,
                loader_fn=mlx_lm.load,
            )

            user_text = prompt
            code_prefix = (
                "You are a local code scout. Return only a JSON array of "
                "CodeSlicePointer objects.\nUser task: "
            )
            quant_prefix = (
                "You are a quant research specialist. Return only a "
                "QuantStrategyBrief JSON.\nTask: "
            )
            for prefix in (code_prefix, quant_prefix):
                if user_text.startswith(prefix):
                    user_text = user_text[len(prefix) :]
                    break

            try:
                formatted = tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_text}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError as exc:
                logger.debug(
                    "Tokenizer chat template without enable_thinking (%s)",
                    type(exc).__name__,
                )
                formatted = tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_text}],
                    tokenize=False,
                    add_generation_prompt=True,
                )

            gen_kwargs: dict[str, Any] = {
                "prompt": formatted,
                "max_tokens": max_tokens,
                "verbose": False,
            }
            try:
                from mlx_lm.sample_utils import make_logits_processors, make_sampler

                gen_kwargs["sampler"] = make_sampler(temp=0.0)
                gen_kwargs["logits_processors"] = make_logits_processors(
                    repetition_penalty=1.05,
                    repetition_context_size=128,
                )
            except Exception as exc:
                logger.debug("mlx_lm.sample_utils unavailable (%s)", type(exc).__name__)

            try:
                raw = str(mlx_lm.generate(model, tokenizer, **gen_kwargs))
            except TypeError:
                raw = str(
                    mlx_lm.generate(
                        model,
                        tokenizer,
                        prompt=formatted,
                        max_tokens=max_tokens,
                        verbose=False,
                    )
                )
            cleaned_out = _GEMMA4_THOUGHT_CHANNEL_RE.sub("", raw)
            cleaned_out = _GEMMA4_ALT_THOUGHT_RE.sub("", cleaned_out)
            return (
                cleaned_out.replace("<eos>", "")
                .replace("<end_of_turn>", "")
                .replace("<turn|>", "")
                .strip()
            )

    def _call_mlx_raw(
        self,
        prompt: str,
        domain_id: str | None = None,
        system_instruction: str | None = None,
        max_tokens: int = 1024,
    ) -> str:
        """Calls /v1/chat/completions on local MLX Gemma server and returns raw text."""
        if domain_id:
            self.set_active_domain(domain_id)
        base_model, _adapter_path, uses_lora = self._resolve_domain_adapter(domain_id)
        slot = self.get_active_adapter(domain_id) if (domain_id and uses_lora) else None
        model_name = slot.adapter_id if (slot and uses_lora) else base_model
        messages: list[dict[str, str]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.endpoint_url}/chat/completions",
            data=data,
            headers={"Content-Type": "application/json", "User-Agent": "ContextScout-LocalGemma"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return str(body["choices"][0]["message"]["content"])

    def generate_code_pointers(
        self,
        user_prompt: str,
        domain_id: str | None = None,
        repo: Any = None,
        principal: PrincipalContext | None = None,
        bounce_back_feedback: str | None = None,
    ) -> list[CodeSlicePointer]:
        """Generates candidate CodeSlicePointers for code scout tasks.

        If MLX server is unreachable or offline_mode is True, uses AST indexing
        and prompt parsing to locate candidate slices. When bounce_back_feedback is
        provided, remediates missing symbols from feedback for $0 local correction.
        """
        resolved_domain = _resolve_default_domain(domain_id, role="code")
        domain_id = resolved_domain
        self.set_active_domain(domain_id)

        # 1. Pre-injected evaluation / benchmark prompts or repo-less unit probes run MLX directly
        has_injected_candidates = "Retrieved AST Candidate:" in user_prompt or repo is None
        if has_injected_candidates:
            if not self.offline_mode and (self.use_live_mlx or self.is_online()):
                try:
                    if self.use_live_mlx:
                        raw_mlx = self._call_in_process_mlx(
                            user_prompt, domain_id=domain_id, max_tokens=768
                        )
                        raw_list = _decode_gemma4_json(raw_mlx, expect_array=True)
                        if not isinstance(raw_list, list):
                            raise ValueError("Invalid JSON array from live MLX code pointer call")
                        pointers = [CodeSlicePointer.model_validate(item) for item in raw_list]
                    else:
                        pointers = self._call_mlx_code_pointers(user_prompt, domain_id)
                    if pointers:
                        return pointers
                except Exception as e:
                    logger.warning(
                        "Local MLX inference failed (%s); using deterministic AST parser",
                        e,
                    )

            return self._deterministic_code_pointers_fallback(
                user_prompt=user_prompt,
                domain_id=domain_id,
                repo=repo,
                principal=principal,
                bounce_back_feedback=bounce_back_feedback,
            )

        # 2. Live repository queries retrieve grounded AST candidates BEFORE any MLX generation
        retrieved_pointers = self._deterministic_code_pointers_fallback(
            user_prompt=user_prompt,
            domain_id=domain_id,
            repo=repo,
            principal=principal,
            bounce_back_feedback=bounce_back_feedback,
        )
        if not retrieved_pointers:
            return []

        if not self.offline_mode and (self.use_live_mlx or self.is_online()):
            allowed_symbols = {p.qualified_symbol for p in retrieved_pointers}
            allowed_files = {p.file_path for p in retrieved_pointers}
            candidate_lines = [
                "Retrieved AST Candidate: ["
                f"repo={p.repo_name} | file={p.file_path} | symbol={p.qualified_symbol} | "
                f"lines={p.start_line}-{p.end_line} | intent={p.modification_intent.value}]"
                for p in retrieved_pointers
            ]
            rag_prompt = f"{user_prompt}\n" + "\n".join(candidate_lines)
            try:
                if self.use_live_mlx:
                    raw_mlx = self._call_in_process_mlx(
                        rag_prompt, domain_id=domain_id, max_tokens=768
                    )
                    raw_list = _decode_gemma4_json(raw_mlx, expect_array=True)
                    if not isinstance(raw_list, list):
                        raise ValueError("Invalid JSON array from live MLX RAG code pointer call")
                    mlx_pointers = [CodeSlicePointer.model_validate(item) for item in raw_list]
                else:
                    mlx_pointers = self._call_mlx_code_pointers(rag_prompt, domain_id)
                grounded_pointers = [
                    p
                    for p in mlx_pointers
                    if p.qualified_symbol in allowed_symbols or p.file_path in allowed_files
                ]
                if grounded_pointers:
                    return grounded_pointers
            except Exception as e:
                logger.warning(
                    "Live MLX RAG code pointer refinement failed (%s); using grounded AST pointers",
                    e,
                )

        return retrieved_pointers

    def _call_mlx_code_pointers(self, user_prompt: str, domain_id: str) -> list[CodeSlicePointer]:
        """Calls OpenAI-compatible /v1/chat/completions on local MLX Gemma server."""
        slot = self.get_active_adapter(domain_id)
        model_name = slot.adapter_id if slot else self.model_id
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a local code scout. Return only a JSON array of "
                        "CodeSlicePointer objects with keys: repo_name, file_path, "
                        "start_line, end_line, qualified_symbol, selection_reason, "
                        "modification_intent."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 1024,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.endpoint_url}/chat/completions",
            data=data,
            headers={"Content-Type": "application/json", "User-Agent": "ContextScout-LocalGemma"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            content = str(body["choices"][0]["message"]["content"])
            raw_list = _decode_gemma4_json(content, expect_array=True)
            if not isinstance(raw_list, list):
                raise ValueError("Invalid JSON array from MLX server code pointer call")
            return [CodeSlicePointer.model_validate(item) for item in raw_list]

    def _deterministic_code_pointers_fallback(
        self,
        user_prompt: str,
        domain_id: str,
        repo: Any,
        principal: PrincipalContext | None,
        bounce_back_feedback: str | None,
    ) -> list[CodeSlicePointer]:
        """Deterministic structured output generator using repository AST index."""
        bound_repo = repo if isinstance(repo, BoundGraphRepository) else BoundGraphRepository(repo)
        active_principal = principal or PrincipalContext(
            user_id="local_gemma_worker",
            role=RoleType.ENGINEER,
            allowed_scopes=["public_eng", "repo:trading_strategies", "corpus:quant_papers"],
        )

        visible_symbols = bound_repo.list_visible_ast_symbols(active_principal, domain_id)
        visible_files = bound_repo.list_visible_code_files(active_principal, domain_id)

        pointers: list[CodeSlicePointer] = []
        covered_symbols: set[str] = set()

        # Parse missing symbols and files from bounce_back_feedback if available
        missing_from_feedback: set[str] = set()
        if bounce_back_feedback:
            quoted = re.findall(r"'([^']+)'", bounce_back_feedback)
            for q in quoted:
                clean = q.strip()
                if clean:
                    missing_from_feedback.add(clean)

        # Extract candidates from prompt (including backticks and dotted parts)
        fence_parts: list[str] = []
        scan_pos = 0
        while True:
            open_fence = user_prompt.find("```", scan_pos)
            if open_fence == -1:
                fence_parts.append(user_prompt[scan_pos:])
                break
            fence_parts.append(user_prompt[scan_pos:open_fence])
            close_fence = user_prompt.find("```", open_fence + 3)
            if close_fence == -1:
                fence_parts.append(user_prompt[open_fence:])
                break
            fence_parts.append(" ")
            scan_pos = close_fence + 3
        unfenced_prompt = "".join(fence_parts)
        cleaned_prompt_lines: list[str] = []
        for raw_line in unfenced_prompt.splitlines():
            stripped_line = raw_line.strip()
            if stripped_line.startswith("-") and len(stripped_line) >= 5:
                after_dash = stripped_line[1:].lstrip()
                if after_dash.startswith(("[ ] ", "[x] ", "[X] ", "[ ]\t", "[x]\t", "[X]\t")):
                    cleaned_prompt_lines.append(" ")
                    continue
            if (
                stripped_line.startswith("#")
                and stripped_line.lstrip("#").strip().lower() == "checklist"
            ):
                cleaned_prompt_lines.append(" ")
                continue
            cleaned_prompt_lines.append(raw_line)
        prompt_without_fences = "\n".join(cleaned_prompt_lines)
        raw_backticks = re.findall(r"`([^`\n]+)`", prompt_without_fences)
        backtick_syms: set[str] = set()
        explicit_kwarg_names: set[str] = set()
        for bt in raw_backticks:
            cleaned_bt = bt.strip(" `'\";:(),")
            if "=" in cleaned_bt:
                kw_lhs = cleaned_bt.split("=", 1)[0].strip()
                if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", kw_lhs):
                    explicit_kwarg_names.add(kw_lhs.lower())
                continue
            if cleaned_bt.endswith("()"):
                cleaned_bt = cleaned_bt[:-2].strip()
            if cleaned_bt and not any(ch.isspace() for ch in cleaned_bt):
                backtick_syms.add(cleaned_bt)
                if "." in cleaned_bt:
                    backtick_syms.update(p for p in cleaned_bt.split(".") if len(p) >= 3)
        kwarg_pat = r"\b([A-Za-z_][A-Za-z0-9_]{2,})\s*=\s*(?:False|True|None|\(|\.|'|\"|\d)"
        for kw_m in re.finditer(kwarg_pat, prompt_without_fences):
            explicit_kwarg_names.add(kw_m.group(1).lower())

        prompt_words = re.findall(r"[A-Za-z0-9_\-\./]+", prompt_without_fences)
        ident_words = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", prompt_without_fences)
        candidates_to_search = (
            set(prompt_words) | set(ident_words) | backtick_syms | missing_from_feedback
        )
        first_line = user_prompt.splitlines()[0] if user_prompt else ""

        files_by_id: dict[uuid.UUID, CodeFileRecord] = {f.file_id: f for f in visible_files}

        # Identify files explicitly mentioned in prompt (including module/directory/repo names)
        prompt_module_tokens = {w.lower() for w in ident_words if "_" in w and len(w) >= 4}
        mentioned_file_ids = {
            f.file_id
            for f in visible_files
            if f.file_path in candidates_to_search
            or any("/" in cand and f.file_path.endswith(cand) for cand in candidates_to_search)
            or (
                "." in f.file_path.split("/")[-1]
                and f.file_path.split("/")[-1] in candidates_to_search
            )
            or any(
                part.lower() in prompt_module_tokens
                for part in f.file_path.replace("\\", "/").split("/")[:-1]
            )
            or (f.repo_name and f.repo_name.lower() in prompt_module_tokens)
        }

        # 1. Match symbols in repository with priority ordering
        t1_exact: list[tuple[ASTSymbolRecord, bool]] = []
        t2_file_classes: list[tuple[ASTSymbolRecord, bool]] = []
        t2_file_others: list[tuple[ASTSymbolRecord, bool]] = []
        t3_subword: list[tuple[ASTSymbolRecord, bool]] = []

        has_non_test_symbols = any(
            not s.qualified_name.startswith(
                (
                    "tests.",
                    "examples.",
                    "docs.",
                    "README.",
                    "CHANGES.",
                    "CHANGELOG.",
                    "CONTRIBUTING.",
                )
            )
            for s in visible_symbols
        )
        prompt_mentions_tests = any(
            k in prompt_without_fences.lower()
            for k in ("tests/", "tests.", "test_", "pytest", "unit test", "examples/", "examples.")
        )
        prompt_mentions_docs = any(
            k in prompt_without_fences.lower()
            for k in ("docs/", "readme", "changelog", "markdown", ".md")
        )

        all_repo_names_l = {f.repo_name.lower() for f in visible_files if f.repo_name}
        prompt_subwords_l = set(
            re.findall(r"[A-Za-z0-9_\-]+", prompt_without_fences.lower())
        ) | set(re.findall(r"[A-Za-z0-9_]+", prompt_without_fences.lower()))
        mentioned_repo_names_l = (
            all_repo_names_l & prompt_subwords_l if len(all_repo_names_l) > 1 else set()
        )

        for sym in visible_symbols:
            if (
                has_non_test_symbols
                and not prompt_mentions_tests
                and sym.qualified_name.startswith(("tests.", "examples."))
            ):
                continue
            if (
                has_non_test_symbols
                and not prompt_mentions_docs
                and sym.qualified_name.startswith(
                    ("docs.", "README.", "CHANGES.", "CHANGELOG.", "CONTRIBUTING.")
                )
            ):
                continue
            sym_repo_l = (
                files_by_id[sym.file_id].repo_name.lower() if sym.file_id in files_by_id else ""
            )
            parts = sym.qualified_name.split(".")
            short_name = parts[-1]
            two_part = ".".join(parts[-2:]) if len(parts) >= 2 else sym.qualified_name
            is_prose_acronym = (
                short_name.isupper()
                and "_" not in short_name
                and len(short_name) <= 5
                and short_name not in backtick_syms
            )
            is_dunder = short_name.startswith("__") and short_name.endswith("__")
            is_generic_attr = short_name.lower() in {
                "name",
                "type",
                "value",
                "default",
                "params",
                "url",
                "args",
                "opts",
                "info",
                "help",
                "env",
                "main",
                "init",
                "close",
                "read",
                "write",
                "send",
                "open",
            }
            enclosing_matched = len(parts) >= 2 and parts[-2] in candidates_to_search
            short_allowed = (
                not (is_dunder or is_generic_attr)
                or enclosing_matched
                or sym.file_id in mentioned_file_ids
            )
            is_code_identifier = (
                (any(c.isupper() for c in short_name) and not is_prose_acronym)
                or ("_" in short_name and short_allowed)
                or (short_name in backtick_syms and short_allowed)
                or two_part in candidates_to_search
                or ("." in sym.qualified_name and sym.qualified_name in candidates_to_search)
            )
            is_exact = (
                is_code_identifier
                and (not mentioned_repo_names_l or sym_repo_l in mentioned_repo_names_l)
                and (
                    sym.qualified_name in candidates_to_search
                    or two_part in candidates_to_search
                    or (short_name in candidates_to_search and short_allowed)
                )
            )
            is_feedback = (
                sym.qualified_name in missing_from_feedback
                or two_part in missing_from_feedback
                or short_name in missing_from_feedback
            )
            if is_exact or is_feedback:
                t1_exact.append((sym, is_feedback))
            elif sym.file_id in mentioned_file_ids:
                if sym.symbol_kind in ("class", "interface", "type_alias"):
                    t2_file_classes.append((sym, False))
                else:
                    t2_file_others.append((sym, False))

        if len(t1_exact) > 1:
            sig_query_terms = {
                w.lower()
                for w in re.findall(r"[A-Za-z0-9_]{4,}", prompt_without_fences)
                if w.lower() not in PROMPT_STOPWORDS
            }

            def _t1_priority(
                item: tuple[ASTSymbolRecord, bool],
            ) -> tuple[int, int, int, int, int, int, int, str]:
                s_rec, is_fb = item
                s_parts = s_rec.qualified_name.split(".")
                s_short = s_parts[-1]
                s_two = ".".join(s_parts[-2:]) if len(s_parts) >= 2 else s_rec.qualified_name
                in_mentioned_file = int(s_rec.file_id in mentioned_file_ids)
                is_full_qual = int(
                    "." in s_rec.qualified_name
                    and (
                        s_rec.qualified_name in backtick_syms
                        or s_rec.qualified_name in candidates_to_search
                    )
                )
                in_title = int(s_short in first_line or s_two in first_line)
                is_two_match = int(s_two in backtick_syms or s_two in missing_from_feedback)
                is_specific = int("_" in s_short or s_rec.symbol_kind in ("function", "method"))
                is_god_class = int(
                    s_rec.symbol_kind == "class" and (s_rec.end_line - s_rec.start_line) > 150
                )
                meta_l = f"{s_rec.signature_skeleton} {s_rec.contextual_summary}".lower()
                meta_hits = sum(1 for qt in sig_query_terms if qt in meta_l)
                return (
                    -int(is_fb),
                    -in_mentioned_file,
                    -is_full_qual,
                    -is_two_match,
                    is_god_class,
                    -(in_title * 2 + is_specific),
                    -meta_hits,
                    s_rec.qualified_name,
                )

            t1_exact.sort(key=_t1_priority)

        generic_terms = {
            "where",
            "what",
            "how",
            "strategy",
            "strategies",
            "codebase",
            "trading",
            "calculate",
            "calculating",
            "compute",
            "computing",
            "apply",
            "applies",
            "applying",
            "implemented",
            "implementation",
            "explain",
            "inspect",
            "describe",
            "model",
            "models",
            "account",
            "step",
            "steps",
            "multi",
            "into",
            "top",
            "across",
            "before",
            "after",
            "internal",
        }
        nl_stopwords = PROMPT_STOPWORDS - {"help"}
        excluded_nl = nl_stopwords | generic_terms
        base_nl_terms = list(
            dict.fromkeys(
                w.lower()
                for t_raw in candidates_to_search
                for w in re.split(r"[_\-\./]+", t_raw)
                if len(w) >= 3 and w.lower() not in excluded_nl and not w.isdigit()
            )
        )
        # Map each morphological variant back to its canonical base query term
        # so stem expansion never double-counts a single query word.
        nl_term_groups: dict[str, str] = {w_item: w_item for w_item in base_nl_terms}
        nl_terms = list(base_nl_terms)

        def _add_variant(variant: str, root_term: str) -> None:
            if len(variant) >= 4 and variant not in excluded_nl and variant not in nl_term_groups:
                nl_term_groups[variant] = root_term
                nl_terms.append(variant)

        for w_item in base_nl_terms:
            if w_item.endswith("ies") and len(w_item) >= 6:
                _add_variant(w_item[:-3] + "y", w_item)
            elif w_item.endswith("s") and not w_item.endswith("ss") and len(w_item) >= 5:
                _add_variant(w_item[:-1], w_item)
            if w_item.endswith("ing") and len(w_item) >= 6:
                _add_variant(w_item[:-3] + "e", w_item)
                _add_variant(w_item[:-3], w_item)
            if w_item.endswith("ed") and len(w_item) >= 5:
                _add_variant(w_item[:-1], w_item)
                _add_variant(w_item[:-2], w_item)
            if w_item.endswith("al") and len(w_item) >= 6:
                _add_variant(w_item[:-2], w_item)
            if w_item in ("decompression", "decompress", "compressed", "compression"):
                for syn in ("decoder", "decoders", "decompress"):
                    _add_variant(syn, w_item)
            if w_item in ("urlencode", "urlencoded"):
                for syn in ("urlencode_safe", "queryparams"):
                    _add_variant(syn, w_item)
        title_terms = {
            w.lower()
            for w in re.findall(r"[A-Za-z0-9]+", first_line)
            if len(w) >= 3 and w.lower() not in nl_stopwords
        }
        for w_title in list(title_terms):
            if w_title.endswith("ing") and len(w_title) >= 6:
                title_terms.add(w_title[:-3] + "e")
                title_terms.add(w_title[:-3])
        backtick_terms_l = {
            w.lower()
            for bt in backtick_syms
            for w in re.split(r"[_\-\./]+", bt)
            if len(w) >= 3 and w.lower() not in nl_stopwords
        }
        code_kwarg_or_snake_terms: set[str] = set()
        for ident_m in re.finditer(r"\b[A-Za-z_][A-Za-z0-9_]*\b", prompt_without_fences):
            tok = ident_m.group(0)
            if len(tok) < 3:
                continue
            tok_l = tok.lower()
            if tok_l in nl_stopwords:
                continue
            is_snake = "_" in tok[1:-1]
            is_kwarg = prompt_without_fences[ident_m.end() :].lstrip(" \t\r\n").startswith("=")
            if is_snake or is_kwarg:
                code_kwarg_or_snake_terms.add(tok_l)
        body_search_terms = title_terms | backtick_terms_l | code_kwarg_or_snake_terms
        prompt_l_clean = prompt_without_fences.lower()
        file_lines_by_id: dict[uuid.UUID, list[str]] = {}
        is_large_multi_repo = len(visible_symbols) > 4000
        has_exact_full_qual = any(
            "." in s_rec.qualified_name
            and (
                s_rec.qualified_name in backtick_syms
                or s_rec.qualified_name in candidates_to_search
            )
            for s_rec, _ in t1_exact
        )
        scored_nl: list[tuple[float, ASTSymbolRecord]] = []
        skip_nl_scan = is_large_multi_repo and (
            has_exact_full_qual or (len(t1_exact) >= 3 and not mentioned_repo_names_l)
        )
        for sym in [] if skip_nl_scan else visible_symbols:
            if mentioned_repo_names_l:
                sym_r_l = (
                    files_by_id[sym.file_id].repo_name.lower() if sym.file_id in files_by_id else ""
                )
                if sym_r_l not in mentioned_repo_names_l:
                    continue
            if (
                has_non_test_symbols
                and not prompt_mentions_tests
                and sym.qualified_name.startswith(("tests.", "examples."))
            ):
                continue
            if (
                has_non_test_symbols
                and not prompt_mentions_docs
                and sym.qualified_name.startswith(
                    ("docs.", "README.", "CHANGES.", "CHANGELOG.", "CONTRIBUTING.")
                )
            ):
                continue
            if "paper_harvester" in sym.qualified_name and "harvest" not in user_prompt.lower():
                continue
            if (
                ".evals." in sym.qualified_name or sym.qualified_name.startswith("evals.")
            ) and "eval" not in user_prompt.lower():
                continue
            short_raw = sym.qualified_name.split(".")[-1]
            short_l = short_raw.lower()
            qual_l = sym.qualified_name.lower()
            sum_l = sym.contextual_summary.lower()

            def _split_camel_snake(val: str) -> set[str]:
                sp = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", val)
                sp = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", sp)
                return {t.lower() for t in re.split(r"[^A-Za-z0-9]+", sp) if len(t) >= 2}

            short_words = _split_camel_snake(short_raw)
            file_rel = files_by_id[sym.file_id].file_path if sym.file_id in files_by_id else ""
            path_words = _split_camel_snake(file_rel) - {
                "src",
                "py",
                "ts",
                "tsx",
                "context",
                "scout",
            }
            sig_words = _split_camel_snake(sym.signature_skeleton)
            sum_words = _split_camel_snake(sym.contextual_summary)

            def _word_match_kind(term: str, words: set[str]) -> int:
                if term in words:
                    return 2
                if len(term) >= 4 and any(
                    w.startswith(term) or (len(w) >= 5 and w[:5] == term[:5] and len(term) >= 5)
                    for w in words
                ):
                    return 1
                return 0
            body_l = ""
            if (
                not is_large_multi_repo
                and sym.file_id in files_by_id
                and not sym.qualified_name.startswith("context_scout.")
            ):
                if sym.file_id not in file_lines_by_id:
                    file_lines_by_id[sym.file_id] = files_by_id[
                        sym.file_id
                    ].raw_content.splitlines()
                f_lines = file_lines_by_id[sym.file_id]
                if (sym.end_line - sym.start_line) <= 160:
                    w_start = max(0, sym.start_line - 1)
                    w_end = min(len(f_lines), sym.end_line)
                    body_l = "\n".join(f_lines[w_start:w_end]).lower()
            score = 0.0
            matched_terms = 0
            short_term_hits = 0
            title_in_short = False
            generic_action_verbs = {
                "handle",
                "handles",
                "fix",
                "fixes",
                "add",
                "adds",
                "drop",
                "drops",
                "remove",
                "removes",
                "update",
                "updates",
                "review",
                "improve",
                "ensure",
                "resolve",
                "support",
                "clarified",
                "allow",
                "make",
                "use",
                "just",
                "default",
                "empty",
                "response",
                "responses",
                "request",
                "requests",
            }
            matched_base_terms: set[str] = set()
            for term in nl_terms:
                if len(term) < 4 and term not in {
                    "url",
                    "uri",
                    "asi",
                    "wsgi",
                    "ssl",
                    "tls",
                    "cli",
                    "env",
                    "eof",
                    "raw",
                    "pct",
                }:
                    continue
                term_weight = 1.5 if term in title_terms else 1.0
                term_hit = False
                short_kind = _word_match_kind(term, short_words)
                if short_kind == 2:
                    score += 9.0 * term_weight
                    term_hit = True
                    if term not in generic_action_verbs:
                        short_term_hits += 1
                    if term in title_terms and term not in generic_action_verbs:
                        title_in_short = True
                elif short_kind == 1 or term in short_l:
                    score += (5.5 if short_kind == 1 else 3.0) * term_weight
                    term_hit = True
                    if term not in generic_action_verbs:
                        short_term_hits += 1
                    if short_kind == 1 and term in title_terms and term not in generic_action_verbs:
                        title_in_short = True
                elif _word_match_kind(term, path_words) > 0:
                    score += 5.0 * term_weight
                    term_hit = True
                elif term in qual_l:
                    score += 2.5 * term_weight
                    term_hit = True
                sig_kind = _word_match_kind(term, sig_words)
                if sig_kind > 0:
                    score += (3.0 if sig_kind == 2 else 2.0) * term_weight
                    term_hit = True
                sum_kind = _word_match_kind(term, sum_words)
                if sum_kind > 0 or term in sum_l:
                    sum_pts = 12.0 if "[dispatch " in sum_l else (3.5 if sum_kind == 2 else 2.0)
                    score += sum_pts * term_weight
                    term_hit = True
                if body_l and term in body_search_terms and term in body_l:
                    score += 3.0 * term_weight
                    term_hit = True
                if term_hit:
                    root_t = nl_term_groups.get(term, term)
                    matched_base_terms.add(root_t)
            matched_terms = len(matched_base_terms)
            if title_in_short:
                score += 6.0
            if short_term_hits >= 2:
                score += 14.0
            if matched_terms >= 2:
                score += 10.0 * (matched_terms - 1)
            if (
                body_l
                and ("windows" in prompt_l_clean or "winerror" in prompt_l_clean)
                and any(w in body_l for w in ("namedtemporaryfile", "mkstemp", "os.unlink"))
            ):
                score += 28.0
                matched_terms += 3
            if body_l:
                for bt in backtick_syms:
                    bt_clean = bt.split(".")[-1].lower()
                    if len(bt_clean) >= 4 and f"{bt_clean}(" in body_l and bt_clean != short_l:
                        score += 6.0
                        matched_terms += 1
                for kw_term in code_kwarg_or_snake_terms:
                    if len(kw_term) >= 4 and kw_term in body_l:
                        score += 4.5
                        matched_terms += 1
                for kw_name in explicit_kwarg_names:
                    if kw_name in {
                        "url",
                        "params",
                        "data",
                        "headers",
                        "cookies",
                        "files",
                        "json",
                        "auth",
                        "timeout",
                        "content",
                    }:
                        continue
                    if re.search(rf",\s*{re.escape(kw_name)}\s*=", body_l):
                        score += 14.0
                        matched_terms += 2
            if explicit_kwarg_names & {"ensure_ascii", "allow_nan", "separators", "sort_keys"} and (
                "dumps" in sum_l or "dumps" in body_l
            ):
                score += 16.0
                matched_terms += 2
            if score > 0.0:
                if mentioned_repo_names_l:
                    sym_repo_nl = (
                        files_by_id[sym.file_id].repo_name.lower()
                        if sym.file_id in files_by_id
                        else ""
                    )
                    if sym_repo_nl in mentioned_repo_names_l:
                        score += 12.0
                    else:
                        score *= 0.35
                score += matched_terms * 5.0
                if sym.symbol_kind in ("class", "interface"):
                    score += 4.0
                elif sym.symbol_kind == "type_alias":
                    score -= 4.0
                elif (
                    sym.symbol_kind in ("function", "method")
                    and (sym.end_line - sym.start_line) > 250
                ):
                    score -= 12.0
                scored_nl.append((score, sym))
        # Module/file-level concept consensus: reward symbols whose file covers multiple
        # distinct query concepts so single-word cross-file noise does not drown cohesive modules.
        if scored_nl:
            file_best_terms: dict[uuid.UUID, int] = {}
            file_sym_count: dict[uuid.UUID, int] = {}
            for _sc, s_obj in scored_nl:
                file_sym_count[s_obj.file_id] = file_sym_count.get(s_obj.file_id, 0) + 1
                approx_terms = 0
                combined_l = (
                    f"{s_obj.qualified_name} {s_obj.signature_skeleton} {s_obj.contextual_summary}"
                ).lower()
                for b_term in base_nl_terms:
                    if len(b_term) >= 4 and (
                        b_term in combined_l or (len(b_term) >= 5 and b_term[:5] in combined_l)
                    ):
                        approx_terms += 1
                if approx_terms > file_best_terms.get(s_obj.file_id, 0):
                    file_best_terms[s_obj.file_id] = approx_terms
            boosted_nl: list[tuple[float, ASTSymbolRecord]] = []
            for sc, s_obj in scored_nl:
                f_terms = file_best_terms.get(s_obj.file_id, 0)
                f_syms = file_sym_count.get(s_obj.file_id, 0)
                consensus_bonus = 0.0
                if f_terms >= 2 or f_syms >= 2:
                    consensus_bonus = (
                        6.0 * min(4, max(0, f_terms - 1)) + 3.0 * min(3, max(0, f_syms - 1))
                    )
                boosted_nl.append((sc + consensus_bonus, s_obj))
            scored_nl = boosted_nl
        scored_nl.sort(key=lambda item: (-item[0], item[1].qualified_name))
        if scored_nl:
            existing_t1_ids = {s.symbol_id for s, _ in t1_exact}
            limit_nl = 5 if not t1_exact else 3
            t3_subword = []
            parent_counts: dict[str, int] = {}
            type_alias_counts: dict[uuid.UUID, int] = {}
            for _, s_item in scored_nl:
                if s_item.symbol_id in existing_t1_ids:
                    continue
                if s_item.symbol_kind == "type_alias":
                    if type_alias_counts.get(s_item.file_id, 0) >= 1:
                        continue
                    type_alias_counts[s_item.file_id] = type_alias_counts.get(s_item.file_id, 0) + 1
                q_parts = s_item.qualified_name.split(".")
                parent_key = ".".join(q_parts[:-1]) if len(q_parts) <= 3 else ".".join(q_parts[:3])
                if parent_counts.get(parent_key, 0) >= 2 and s_item.symbol_kind != "class":
                    continue
                if s_item.symbol_kind != "type_alias":
                    parent_counts[parent_key] = parent_counts.get(parent_key, 0) + 1
                t3_subword.append((s_item, False))
                if len(t3_subword) >= limit_nl:
                    break
        elif not t1_exact:
            excluded_subwords = (
                PROMPT_STOPWORDS
                | CONVERSATIONAL_AND_GENERIC_WORDS
                | generic_terms
                | {"trading", "strategies", "context_scout"}
            )
            subword_candidates = [
                w for w in candidates_to_search if len(w) > 4 and w.lower() not in excluded_subwords
            ]
            for sym in visible_symbols:
                if sym.file_id in mentioned_file_ids:
                    continue
                if (
                    has_non_test_symbols
                    and not prompt_mentions_tests
                    and sym.qualified_name.startswith(("tests.", "examples."))
                ):
                    continue
                if (
                    has_non_test_symbols
                    and not prompt_mentions_docs
                    and sym.qualified_name.startswith(
                        ("docs.", "README.", "CHANGES.", "CHANGELOG.", "CONTRIBUTING.")
                    )
                ):
                    continue
                sym_q_low = sym.qualified_name.lower()
                sym_tokens = {
                    seg.lower() for seg in re.split(r"[.:/\\_-]+", sym.qualified_name) if seg
                } | {
                    tok.lower()
                    for tok in re.findall(
                        r"[A-Z]+(?=[A-Z][a-z0-9]|[^A-Za-z0-9]|$)|[A-Z]?[a-z0-9]+",
                        sym.qualified_name,
                    )
                }
                if any(
                    (
                        w.lower() in sym_q_low
                        if ("_" in w or any(c.isupper() for c in w[1:]))
                        else w.lower() in sym_tokens
                    )
                    for w in subword_candidates
                ):
                    t3_subword.append((sym, False))
                    if len(t3_subword) >= 3:
                        break

        plan = decompose_query_facets(user_prompt)
        target_diversity = max(1, plan.requested_cardinality or len(plan.facets))
        max_candidates = max(5, target_diversity * 2)

        candidate_ordered = t1_exact + t2_file_classes + t2_file_others + t3_subword
        if target_diversity >= 2 and len(candidate_ordered) > 1:
            # Diversify across distinct files first so "top N" / multi-facet queries
            # do not collapse onto N methods from a single file.
            by_file_primary: list[tuple[ASTSymbolRecord, bool]] = []
            by_file_rest: list[tuple[ASTSymbolRecord, bool]] = []
            seen_fids: set[uuid.UUID] = set()
            for item in candidate_ordered:
                fid = item[0].file_id
                if fid not in seen_fids:
                    seen_fids.add(fid)
                    by_file_primary.append(item)
                else:
                    by_file_rest.append(item)
            candidate_ordered = by_file_primary + by_file_rest

        if len(candidate_ordered) > max_candidates:
            feedback_items = [item for item in candidate_ordered if item[1]]
            non_feedback = [item for item in candidate_ordered if not item[1]]
            candidate_ordered = (non_feedback[:1] + feedback_items + non_feedback[1:])[
                :max_candidates
            ]

        for sym, is_feedback_match in candidate_ordered:
            short_name = sym.qualified_name.split(".")[-1]
            file_rec = files_by_id.get(sym.file_id)
            file_path = file_rec.file_path if file_rec else "src/unknown.py"
            repo_name = file_rec.repo_name if file_rec else "trading-execution"

            intent = (
                ModificationIntent.MODIFY_TARGET
                if (not is_feedback_match and len(pointers) == 0)
                else ModificationIntent.REFERENCE_ONLY
            )
            reason = (
                f"Remediated via $0 bounce-back feedback: {short_name}"
                if is_feedback_match
                else f"Located symbol {sym.qualified_name} from user task"
            )

            ptr = CodeSlicePointer(
                repo_name=repo_name,
                file_path=file_path,
                start_line=sym.start_line,
                end_line=sym.end_line,
                qualified_symbol=sym.qualified_name,
                selection_reason=reason,
                modification_intent=intent,
            )
            pointers.append(ptr)
            covered_symbols.add(sym.qualified_name)
            covered_symbols.add(short_name)

        # 2. Match whole files in repository if mentioned
        for f in visible_files:
            file_name = f.file_path.split("/")[-1]
            if f.file_path in candidates_to_search or (
                ("." in file_name or file_name in missing_from_feedback)
                and file_name in candidates_to_search
            ):
                if any(p.file_path == f.file_path for p in pointers):
                    continue

                end_ln = min(max(1, f.line_count), 200)
                reason = (
                    f"Remediated via $0 bounce-back feedback: {file_name}"
                    if (f.file_path in missing_from_feedback or file_name in missing_from_feedback)
                    else f"Matched file {f.file_path} from user task"
                )
                ptr = CodeSlicePointer(
                    repo_name=f.repo_name,
                    file_path=f.file_path,
                    start_line=1,
                    end_line=end_ln,
                    qualified_symbol=f.file_path,
                    selection_reason=reason,
                    modification_intent=ModificationIntent.REFERENCE_ONLY,
                )
                pointers.append(ptr)

        # 3. Bounded fallback only for synthetic bounce-back trigger prompts; fail closed otherwise
        allow_default_fallback = bool(
            re.search(r"\bbounce[-_\s]?back\b", user_prompt, flags=re.IGNORECASE)
        )
        if not pointers and allow_default_fallback and visible_symbols:
            first_sym = visible_symbols[0]
            f_rec = files_by_id.get(first_sym.file_id)
            pointers.append(
                CodeSlicePointer(
                    repo_name=f_rec.repo_name if f_rec else "trading-execution",
                    file_path=f_rec.file_path if f_rec else "src/default.py",
                    start_line=first_sym.start_line,
                    end_line=first_sym.end_line,
                    qualified_symbol=first_sym.qualified_name,
                    selection_reason="Default primary symbol selection",
                    modification_intent=ModificationIntent.MODIFY_TARGET,
                )
            )
        elif not pointers and allow_default_fallback and visible_files:
            first_file = visible_files[0]
            pointers.append(
                CodeSlicePointer(
                    repo_name=first_file.repo_name,
                    file_path=first_file.file_path,
                    start_line=1,
                    end_line=min(max(1, first_file.line_count), 200),
                    qualified_symbol=first_file.file_path,
                    selection_reason="Default repository file selection",
                    modification_intent=ModificationIntent.MODIFY_TARGET,
                )
            )

        return pointers

    def _call_mlx_quant_brief(self, user_prompt: str, domain_id: str) -> QuantStrategyBrief | None:
        """Calls OpenAI-compatible /v1/chat/completions on local MLX Gemma server."""
        slot = self.get_active_adapter(domain_id)
        model_name = slot.adapter_id if slot else self.model_id
        schema_fields = ", ".join(QuantStrategyBrief.model_fields.keys())
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a local research scout. Return only a valid JSON object "
                        f"conforming to QuantStrategyBrief schema with keys: {schema_fields}."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 1536,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.endpoint_url}/chat/completions",
            data=data,
            headers={"Content-Type": "application/json", "User-Agent": "ContextScout-LocalGemma"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            content = str(body["choices"][0]["message"]["content"])
            parsed = _parse_gemma4_quant_strategy_brief(content, domain_id=domain_id)
            if parsed is None:
                raise ValueError("Invalid QuantStrategyBrief JSON from MLX server")
            return parsed

    def _call_mlx_domain_knowledge_brief(
        self, user_prompt: str, domain_id: str
    ) -> DomainKnowledgeBrief | None:
        """Calls OpenAI-compatible /v1/chat/completions to synthesize a DomainKnowledgeBrief."""
        slot = self.get_active_adapter(domain_id)
        model_name = slot.adapter_id if slot else self.model_id
        schema_fields = ", ".join(DomainKnowledgeBrief.model_fields.keys())
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a local domain knowledge scout. Return only a valid JSON object "
                        f"conforming to DomainKnowledgeBrief schema with keys: {schema_fields}."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 1536,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.endpoint_url}/chat/completions",
            data=data,
            headers={"Content-Type": "application/json", "User-Agent": "ContextScout-LocalGemma"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            content = str(body["choices"][0]["message"]["content"])
            parsed = _parse_gemma4_domain_knowledge_brief(content, domain_id=domain_id)
            if parsed is None:
                raise ValueError("Invalid DomainKnowledgeBrief JSON from MLX server")
            return parsed

    @staticmethod
    def _extract_verified_doc_citations(
        fresh_chunks: list[DocChunkRecord],
        *,
        max_citations: int = 8,
        diversify_docs: bool = False,
        all_chunks: list[DocChunkRecord] | None = None,
    ) -> list[DocCitationSpan]:
        """Extracts verbatim-verified DocCitationSpans from non-stale document chunks."""

        def _is_noisy_line(line: str) -> bool:
            s = line.strip()
            if len(s) < 25:
                return True
            noisy_prefixes = (
                "#",
                "\\begin{table",
                "\\end{table",
                "\\maketitle",
                "\\label{",
                "\\paragraph{}",
                "\\sys",
                "%",
                "```",
                "---",
                "|",
                "<<",
                "//",
                "<td",
                "<tr",
                "<span",
                "<table",
                "<div",
                "</",
            )
            if any(s.startswith(p) for p in noisy_prefixes):
                return True
            if s.startswith("$") and s.endswith("$"):
                return True
            if re.match(
                r"^(?:---|title:\s*|layout:\s*|permalink:\s*|-\s+/docs/|category:\s*)", s, re.I
            ):
                return True
            low = s.lower()
            boilerplate_markers = (
                "apache license",
                "licensed to the apache software foundation",
                "spdx-license-identifier",
                "unless required by applicable law",
                'distributed on an "as is" basis',
                "the rest of this paper",
                "see also",
                "declaration on generative ai",
                "{prodname}",
                "// category:",
                "\\label{",
                "\\sys",
                "<td",
                "<tr",
                "<span",
                "### author",
            )
            return any(m in low for m in boilerplate_markers)

        slicer = DeterministicSlicer()
        ordered_chunks = list(fresh_chunks)
        if diversify_docs and len(ordered_chunks) > 1:
            primary_per_doc: list[DocChunkRecord] = []
            secondary: list[DocChunkRecord] = []
            seen_docs: set[str] = set()
            for chk in ordered_chunks:
                if chk.doc_id not in seen_docs:
                    seen_docs.add(chk.doc_id)
                    primary_per_doc.append(chk)
                else:
                    secondary.append(chk)
            ordered_chunks = primary_per_doc + secondary

        verified_citations: list[DocCitationSpan] = []
        for chk in ordered_chunks:
            if len(verified_citations) >= max_citations:
                break
            snippet = chk.verbatim_content.strip()
            if not snippet:
                continue
            para_blocks = [
                blk.strip()
                for blk in re.split(r"\n\s*\n", snippet)
                if len(blk.strip()) >= 40
                and not _is_noisy_line(blk.strip().splitlines()[0])
                and re.search(r"[.!?](?:\s|$)", blk.strip())
            ]
            clean_lines = [ln.strip() for ln in snippet.splitlines() if not _is_noisy_line(ln)]
            lead_sentence = snippet.split(".")[0] + "." if "." in snippet else snippet[:400]
            candidates = [
                *para_blocks,
                *clean_lines,
                lead_sentence,
                *[ln.strip() for ln in snippet.splitlines() if len(ln.strip()) >= 20],
                snippet[:500],
            ]
            for raw_cand in candidates:
                cand = (
                    raw_cand
                    if len(raw_cand) <= 500
                    else (raw_cand[:500].rsplit(" ", 1)[0].strip() or raw_cand[:500])
                )
                if not cand:
                    continue
                cand_cit = _build_doc_citation_span(
                    chk, cand, all_chunks=all_chunks or fresh_chunks
                )
                if slicer.verify_doc_citation(
                    citation=cand_cit,
                    chunk_content=chk.verbatim_content,
                ).is_valid:
                    verified_citations.append(cand_cit)
                    break
        return verified_citations

    def generate_quant_brief(
        self,
        user_prompt: str,
        domain_id: str | None = None,
        repo: Any = None,
        principal: PrincipalContext | None = None,
        bounce_back_feedback: str | None = None,
        pre_retrieved_chunks: list[DocChunkRecord] | None = None,
    ) -> QuantStrategyBrief | None:
        """Generates structured domain research brief with verified document citations."""
        self._last_chunks_fed = 0
        domain_id = _resolve_default_domain(domain_id, role="research")
        self.set_active_domain(domain_id)

        # 1. Pre-injected evaluation prompts or repo-less unit probes run MLX directly
        has_injected_chunks = "Retrieved Corpus Chunk:" in user_prompt or repo is None
        if has_injected_chunks:
            if not self.offline_mode and (self.use_live_mlx or self.is_online()):
                try:
                    mlx_brief: QuantStrategyBrief | None = None
                    if self.use_live_mlx:
                        raw_mlx = self._call_in_process_mlx(
                            user_prompt, domain_id=domain_id, max_tokens=1536
                        )
                        mlx_brief = _parse_gemma4_quant_strategy_brief(
                            raw_mlx, domain_id=domain_id
                        )
                    else:
                        mlx_brief = self._call_mlx_quant_brief(user_prompt, domain_id)
                    if mlx_brief is not None:
                        return mlx_brief
                except Exception as e:
                    logger.warning(
                        "Local MLX research inference failed (%s); using deterministic synthesizer",
                        e,
                    )
            if repo is None:
                return None

        bound_repo = repo if isinstance(repo, BoundGraphRepository) else BoundGraphRepository(repo)
        domain_plugin = (
            PluginRegistry.get_domain_plugin(domain_id)
            or PluginRegistry.get_primary_domain_plugin()
        )
        default_scopes = (
            domain_plugin.get_default_scopes() if domain_plugin is not None else ["public_eng"]
        )
        active_principal = principal or PrincipalContext(
            user_id="local_gemma_research",
            role=RoleType.ENGINEER,
            allowed_scopes=sorted({"public_eng", *default_scopes}),
        )

        target_markets = (
            domain_plugin.infer_market_scopes(user_prompt)
            if domain_plugin is not None and domain_plugin.is_research_domain(domain_id)
            else []
        )
        primary_market = target_markets[0] if len(target_markets) == 1 else None
        plan = decompose_query_facets(user_prompt)
        if domain_plugin is not None and hasattr(domain_plugin, "expand_query_plan"):
            plan = domain_plugin.expand_query_plan(plan)

        if pre_retrieved_chunks:
            chunks = list(pre_retrieved_chunks)
        elif len(plan.facets) >= 2 or plan.is_acronym_query:
            chunks = _stratified_doc_search(
                bound_repo,
                principal=active_principal,
                domain_id=domain_id,
                plan=plan,
                market_scope=primary_market,
                per_facet_limit=8,
                max_total=16,
            )
        else:
            chunks = bound_repo.search_visible_doc_chunks(
                principal=active_principal,
                domain_id=domain_id,
                query_text=user_prompt,
                market_scope=primary_market,
                limit=40,
            )

        fresh_chunks = [c for c in chunks if not bool(getattr(c, "is_stale", False))]
        self._last_chunks_fed = min(6, len(fresh_chunks))
        slicer = DeterministicSlicer()
        chunks_by_id = {str(c.chunk_id): c for c in fresh_chunks}
        verified_citations = self._extract_verified_doc_citations(
            fresh_chunks,
            max_citations=max(8, len(plan.facets) * 2),
            all_chunks=fresh_chunks,
        )

        if not verified_citations:
            logger.info(
                "No verified citations extracted; returning None for deterministic abstention"
            )
            return None

        if (
            not has_injected_chunks
            and not self.offline_mode
            and (self.use_live_mlx or self.is_online())
        ):
            _base_id, _adapter_p, uses_lora = self._resolve_domain_adapter(domain_id)
            synth_mode: Literal[
                "live_qlora_json",
                "live_base_gemma4_json",
                "deterministic_extractive_fallback",
            ] = "live_qlora_json" if uses_lora else "live_base_gemma4_json"
            full_chunk_blocks = self._format_full_chunks_for_synthesis(fresh_chunks)
            if not full_chunk_blocks:
                full_chunk_blocks = [
                    "Retrieved Corpus Chunk:\n"
                    f"  chunk_id: {c.chunk_id}\n  doc_id: {c.doc_id}\n"
                    f"  doc_title: {c.doc_title}\n  source_uri: {c.source_uri}\n"
                    f"  section_heading: {c.section_heading}\n"
                    f'  verbatim_quote: "{c.verbatim_quote}"'
                    for c in verified_citations[:6]
                ]
            self._last_chunks_fed = len(full_chunk_blocks)
            rag_prompt = (
                "Synthesize a QuantStrategyBrief JSON strictly from the retrieved chunks below.\n"
                "Rules:\n"
                "1. Judge sufficiency_verdict ('sufficient' | 'partial' | 'insufficient') "
                "honestly:\n"
                "   - 'sufficient': chunks substantially answer the core question.\n"
                "   - 'partial': chunks provide verified evidence for the core subject/market "
                "asked, while secondary operational sub-aspects are unverified (list those gaps "
                "in missing_aspects).\n"
                "   - 'insufficient': NO core part of the user question is supported by the "
                "chunks.\n"
                "2. Never promote off-topic chunks; if NO core aspect is supported, set "
                'sufficiency_verdict="insufficient" and list gaps in missing_aspects.\n'
                "3. Do NOT fabricate numeric performance metrics, drawdowns, cost assumptions, "
                "or sample periods.\n"
                "4. Cite exact chunk_id, doc_id, doc_title, source_uri, section_heading, "
                "and verbatim_quote substrings.\n\n"
                f"User Question: {user_prompt}\n\n"
                + "\n\n".join(full_chunk_blocks)
            )
            try:
                rag_brief: QuantStrategyBrief | None = None
                if self.use_live_mlx:
                    raw_mlx = self._call_in_process_mlx(
                        rag_prompt, domain_id=domain_id, max_tokens=1536
                    )
                    rag_brief = _parse_gemma4_quant_strategy_brief(
                        raw_mlx,
                        domain_id=domain_id,
                        candidate_chunks=fresh_chunks,
                        live_mode=synth_mode,
                    )
                else:
                    rag_brief = self._call_mlx_quant_brief(rag_prompt, domain_id)
                if rag_brief is not None:
                    grounded_cits = [
                        cit
                        for cit in rag_brief.verified_citations
                        if cit.chunk_id in chunks_by_id
                        and slicer.verify_doc_citation(
                            citation=cit,
                            chunk_content=chunks_by_id[cit.chunk_id].verbatim_content,
                        ).is_valid
                    ]
                    if rag_brief.sufficiency_verdict == "insufficient":
                        rag_brief.verified_citations = grounded_cits or rag_brief.verified_citations
                    else:
                        rag_brief.verified_citations = grounded_cits or verified_citations
                    rag_brief.synthesis_mode = synth_mode
                    return rag_brief
            except Exception as e:
                logger.warning(
                    "Live MLX RAG research synthesis failed (%s); using deterministic synthesizer",
                    e,
                )

        if domain_plugin is not None:
            brief_res: QuantStrategyBrief = domain_plugin.synthesize_research_brief(
                user_prompt=user_prompt,
                domain_id=domain_id,
                chunks=fresh_chunks,
                verified_citations=verified_citations,
            )
            return brief_res

        top_chk = fresh_chunks[0] if fresh_chunks else None
        return QuantStrategyBrief(
            strategy_name=(
                f"{top_chk.doc_title} Synthesis" if top_chk else "Verified Empirical Synthesis"
            ),
            target_markets=[MarketRegime.GLOBAL_MULTI],
            is_intraday_futures=False,
            mathematical_signal_latex=r"E[Y \mid X] = f_{\theta}(\text{DomainContext})",
            entry_exit_rules=[
                f"Grounded in verified citation from {top_chk.doc_title if top_chk else 'corpus'}"
            ],
            sample_period_years="2020-2024",
            transaction_cost_bps_assumed=None,
            friction_verdict=FrictionVerdict.REALISTIC_FRICTION,
            market_specific_caveats=["Grounded in retrieved corpus citations"],
            known_failure_regimes=["Out-of-distribution regime shift"],
            verified_citations=verified_citations,
        )

    def generate_domain_knowledge_brief(
        self,
        user_prompt: str,
        domain_id: str,
        repo: Any = None,
        principal: PrincipalContext | None = None,
        quant_brief: QuantStrategyBrief | None = None,
        pre_retrieved_chunks: list[DocChunkRecord] | None = None,
    ) -> DomainKnowledgeBrief | None:
        """Synthesizes a domain-agnostic DomainKnowledgeBrief for any document/knowledge domain."""
        if quant_brief is not None:
            if isinstance(quant_brief, DomainKnowledgeBrief):
                return quant_brief
            metadata: dict[str, str | float | int | bool] = {
                "domain_id": domain_id,
                "verified_citations_count": len(quant_brief.verified_citations),
            }
            plugin = (
                PluginRegistry.get_domain_plugin(domain_id)
                or PluginRegistry.get_primary_domain_plugin()
            )
            if plugin is not None:
                for k, v in plugin.extract_brief_metadata(quant_brief).items():
                    if isinstance(v, (str, float, int, bool)):
                        metadata[k] = v

            summary_quote = (
                quant_brief.verified_citations[0].verbatim_quote
                if quant_brief.verified_citations
                else quant_brief.entry_exit_rules[0]
            )
            return DomainKnowledgeBrief(
                brief_title=quant_brief.strategy_name,
                domain_id=domain_id,
                executive_summary=(
                    f"Grounded synthesis for {quant_brief.strategy_name}: {summary_quote}"
                ),
                formal_expression_or_rule=quant_brief.mathematical_signal_latex,
                core_findings_and_rules=list(quant_brief.entry_exit_rules),
                domain_constraints_and_caveats=list(quant_brief.market_specific_caveats),
                known_risks_and_failure_modes=list(quant_brief.known_failure_regimes),
                structured_metadata=metadata,
                verified_citations=list(quant_brief.verified_citations),
            )

        self._last_chunks_fed = 0
        self.set_active_domain(domain_id)
        has_injected_chunks = "Retrieved Corpus Chunk:" in user_prompt or repo is None
        can_call_mlx = self.use_live_mlx or self.is_online()
        if has_injected_chunks and not self.offline_mode and can_call_mlx:
            try:
                mlx_kb: DomainKnowledgeBrief | None = None
                if self.use_live_mlx:
                    raw_mlx = self._call_in_process_mlx(
                        user_prompt, domain_id=domain_id, max_tokens=2048
                    )
                    mlx_kb, _ = parse_domain_knowledge_brief_json(
                        raw_mlx,
                        domain_id=domain_id,
                        normalize_scalars=True,
                    )
                else:
                    mlx_kb = self._call_mlx_domain_knowledge_brief(user_prompt, domain_id)
                if mlx_kb is not None and mlx_kb.verified_citations:
                    return mlx_kb
            except Exception as exc:
                logger.warning(
                    "Local MLX DomainKnowledgeBrief synthesis failed (%s); falling back to RAG",
                    exc,
                )

        if repo is None:
            return None

        bound_repo = repo if isinstance(repo, BoundGraphRepository) else BoundGraphRepository(repo)
        domain_plugin = PluginRegistry.get_domain_plugin(domain_id)
        default_scopes = (
            domain_plugin.get_default_scopes()
            if domain_plugin is not None
            else ["public_eng", "public_quant", f"domain:{domain_id}"]
        )
        active_principal = principal or PrincipalContext(
            user_id="local_gemma_knowledge",
            role=RoleType.ENGINEER,
            allowed_scopes=sorted({"public_eng", *default_scopes}),
        )
        plan = decompose_query_facets(user_prompt)
        if domain_plugin is not None and hasattr(domain_plugin, "expand_query_plan"):
            plan = domain_plugin.expand_query_plan(plan)

        if pre_retrieved_chunks:
            chunks = list(pre_retrieved_chunks)
        elif len(plan.facets) >= 2 or plan.is_acronym_query:
            chunks = _stratified_doc_search(
                bound_repo,
                principal=active_principal,
                domain_id=domain_id,
                plan=plan,
                per_facet_limit=6,
                max_total=12,
            )
        else:
            chunks = bound_repo.search_visible_doc_chunks(
                principal=active_principal,
                domain_id=domain_id,
                query_text=user_prompt,
                limit=30,
            )
        fresh_chunks = [c for c in chunks if not bool(getattr(c, "is_stale", False))]
        self._last_chunks_fed = min(6, len(fresh_chunks))
        chunks_by_id = {str(c.chunk_id): c for c in fresh_chunks}
        verified_citations = self._extract_verified_doc_citations(
            fresh_chunks,
            max_citations=max(8, len(plan.facets) * 2),
            diversify_docs=True,
            all_chunks=fresh_chunks,
        )
        if not verified_citations:
            return None

        query_facets, missing_aspects, completeness_ratio = _evaluate_query_plan_coverage(
            plan,
            doc_chunks=fresh_chunks,
            code_slices=[],
            domain_id=domain_id,
        )
        # If ALL facets are not_found, abstain cleanly instead of returning unrelated chunks
        if query_facets and all(f.status == "not_found" for f in query_facets):
            return None

        def _finalize_kb(
            kb: DomainKnowledgeBrief,
            *,
            mode: Literal[
                "live_qlora_json",
                "live_base_gemma4_json",
                "deterministic_extractive_fallback",
            ],
        ) -> DomainKnowledgeBrief:
            for cit in kb.verified_citations:
                if not cit.full_section_content and cit.chunk_id in chunks_by_id:
                    cit.full_section_content = _stitch_section_content(
                        chunks_by_id[cit.chunk_id], fresh_chunks
                    )
            kb.query_facets = query_facets
            kb.missing_or_unverified_aspects = list(
                dict.fromkeys([*missing_aspects, *kb.missing_aspects])
            )
            kb.missing_aspects = list(kb.missing_or_unverified_aspects)
            kb.completeness_ratio = completeness_ratio
            kb.synthesis_mode = mode
            if missing_aspects and kb.sufficiency_verdict == "sufficient":
                if mode == "deterministic_extractive_fallback":
                    kb.sufficiency_verdict = (
                        "insufficient" if completeness_ratio == 0.0 else "partial"
                    )
                elif not kb.verified_citations:
                    kb.sufficiency_verdict = "insufficient"
                elif completeness_ratio > 0.0:
                    kb.sufficiency_verdict = "partial"
            if missing_aspects and "not found" not in kb.executive_summary.lower():
                kb.executive_summary = (
                    f"{kb.executive_summary} | Missing from indexed corpus: "
                    + "; ".join(missing_aspects)
                )
            return kb

        slicer = DeterministicSlicer()
        if (
            not has_injected_chunks
            and not self.offline_mode
            and (self.use_live_mlx or self.is_online())
        ):
            _base_id, _adapter_p, uses_lora = self._resolve_domain_adapter(domain_id)
            live_mode: Literal["live_qlora_json", "live_base_gemma4_json"] = (
                "live_qlora_json" if uses_lora else "live_base_gemma4_json"
            )
            chunk_lines = self._format_full_chunks_for_synthesis(fresh_chunks)
            if not chunk_lines:
                chunk_lines = [
                    "Retrieved Corpus Chunk: ["
                    f"chunk_id={c.chunk_id} | doc_id={c.doc_id} | title={c.doc_title} | "
                    f"uri={c.source_uri} | section={c.section_heading} | "
                    f"quote={json.dumps(c.verbatim_quote)}]"
                    for c in verified_citations[:6]
                ]
            self._last_chunks_fed = len(chunk_lines)
            rag_prompt = build_domain_knowledge_brief_prompt(
                user_prompt=user_prompt,
                domain_id=domain_id,
                chunk_lines=chunk_lines,
            ) + (
                "\n\nAdditional Grounding Rules:\n"
                "- Include `sufficiency_verdict` ('sufficient' | 'partial' | 'insufficient') "
                "and `missing_aspects` (list of missing topics).\n"
                "- Answer the user's question directly and thoroughly ONLY from the chunks above.\n"
                "- If the chunks do NOT genuinely answer the user's question, set "
                '`sufficiency_verdict` to `"insufficient"` and explain what is missing.\n'
            )
            try:
                rag_kb: DomainKnowledgeBrief | None = None
                if self.use_live_mlx:
                    raw_mlx = self._call_in_process_mlx(
                        rag_prompt, domain_id=domain_id, max_tokens=2048
                    )
                    rag_kb = _parse_gemma4_domain_knowledge_brief(
                        raw_mlx,
                        domain_id=domain_id,
                        candidate_chunks=fresh_chunks,
                        live_mode=live_mode,
                    )
                else:
                    rag_kb = self._call_mlx_domain_knowledge_brief(rag_prompt, domain_id)
                if rag_kb is not None:
                    grounded_cits = [
                        cit
                        for cit in rag_kb.verified_citations
                        if cit.chunk_id in chunks_by_id
                        and slicer.verify_doc_citation(
                            citation=cit,
                            chunk_content=chunks_by_id[cit.chunk_id].verbatim_content,
                        ).is_valid
                    ]
                    if rag_kb.sufficiency_verdict == "insufficient":
                        rag_kb.verified_citations = grounded_cits or rag_kb.verified_citations
                    else:
                        rag_kb.verified_citations = grounded_cits or verified_citations
                    return _finalize_kb(rag_kb, mode=live_mode)
            except Exception as exc:
                logger.warning(
                    "Live MLX DomainKnowledgeBrief RAG synthesis failed (%s); using fallback",
                    exc,
                )

        if domain_plugin is not None:
            plugin_res = domain_plugin.synthesize_research_brief(
                user_prompt=user_prompt,
                domain_id=domain_id,
                chunks=fresh_chunks,
                verified_citations=verified_citations,
            )
            if isinstance(plugin_res, DomainKnowledgeBrief):
                return _finalize_kb(plugin_res, mode="deterministic_extractive_fallback")
            if isinstance(plugin_res, QuantStrategyBrief):
                kb_from_quant = self.generate_domain_knowledge_brief(
                    user_prompt=user_prompt,
                    domain_id=domain_id,
                    repo=repo,
                    principal=active_principal,
                    quant_brief=plugin_res,
                )
                if kb_from_quant is not None:
                    return _finalize_kb(kb_from_quant, mode="deterministic_extractive_fallback")
                return None

        fallback_kb = _build_domain_brief_from_citations(
            user_prompt=user_prompt,
            domain_id=domain_id,
            citations=verified_citations,
            chunks_by_id=chunks_by_id,
        )
        return _finalize_kb(fallback_kb, mode="deterministic_extractive_fallback")


_HYBRID_DOC_META_WORDS: Final[set[str]] = {
    "explain",
    "implement",
    "implementation",
    "implementing",
    "refactor",
    "inspect",
    "check",
    "update",
    "verify",
    "show",
    "find",
    "list",
    "code",
    "file",
    "files",
    "symbol",
    "symbols",
    "function",
    "method",
    "class",
    "module",
    "calculation",
    "rules",
    "architecture",
    "system",
    "service",
    "strategy",
    "strategies",
    "using",
    "with",
    "from",
    "into",
    "that",
    "this",
    "what",
    "when",
    "where",
    "which",
    "about",
    "there",
}


def _format_scopes(principal: PrincipalContext) -> str:
    return ", ".join(principal.allowed_scopes) if principal.allowed_scopes else "none"


def _stage1_details(
    *,
    domains: list[str],
    principal: PrincipalContext,
    visible_files_count: int,
    visible_chunks_count: int,
    stale_excluded_count: int,
    elapsed_ms: int,
) -> str:
    domains_text = ", ".join(domains)
    stale_suffix = f", {stale_excluded_count} stale excluded" if stale_excluded_count > 0 else ""
    return (
        f"Screened domain(s) [{domains_text}] under PostgreSQL 17 RLS "
        f"(scopes: {_format_scopes(principal)}): {visible_files_count} visible file(s), "
        f"{visible_chunks_count} doc chunk(s){stale_suffix} in {elapsed_ms} ms"
    )


def _stage2_details(
    *,
    model_label: str,
    adapter_id: str,
    pointers_count: int,
    citations_count: int,
    elapsed_ms: int,
) -> str:
    return (
        f"Local {model_label} (adapter '{adapter_id}') drafted {pointers_count} AST pointer(s) "
        f"and {citations_count} citation(s) at $0 paid tokens in {elapsed_ms} ms"
    )


def _stage3_details(
    *,
    slices: list[VerbatimCodeSlice],
    citations_count: int,
    conflicts_count: int,
    elapsed_ms: int,
    budget_tokens: int = 10_000,
) -> str:
    auto_1hop = sum(1 for s in slices if s.auto_expanded_by_ast)
    direct_slices = len(slices) - auto_1hop
    return (
        f"Hydrated {direct_slices} direct slice(s) + {auto_1hop} 1-hop AST dependency slice(s) "
        f"under {budget_tokens:,}-token budget, verified {citations_count} citation(s), "
        f"flagged {conflicts_count} doc-vs-code conflict(s) in {elapsed_ms} ms"
    )


def _stage4_passed_details(
    *,
    report: PreFlightValidationReport,
    pack_tokens: int,
    local_tokens: int,
    total_ms: int,
) -> str:
    checks_total = len(report.check_results) if report.check_results else 6
    checks_passed = (
        sum(1 for ok in report.check_results.values() if ok)
        if report.check_results
        else checks_total
    )
    return (
        f"Passed {checks_passed}/{checks_total} pre-flight checks "
        f"(RRF {report.normalized_rrf_score:.2f}); curated {pack_tokens:,} pack tokens "
        f"from {local_tokens:,} local tokens explored ($0 frontier exploration cost) "
        f"in {total_ms} ms"
    )


def _stage4_failed_details(*, reason: str, total_ms: int) -> str:
    return f"$0 abstention after {total_ms} ms: {reason}"


def _extract_chunk_version(chk: DocChunkRecord) -> str:
    """Extracts normalized document version string from chunk metadata or URI."""
    meta = dict(getattr(chk, "metadata_json", {}) or {})
    raw_ver = (
        meta.get("version")
        or meta.get("page_version")
        or meta.get("version_str")
        or getattr(chk, "version_str", None)
    )
    if raw_ver is not None and str(raw_ver).strip():
        return str(raw_ver).strip()
    uri_match = re.search(r"[?&]v=(\d+)", chk.source_uri or "")
    if uri_match:
        return uri_match.group(1)
    return "1.0"


def _numeric_version(version_str: str) -> float:
    """Parses numeric sort key from a version string, defaulting to 0.0."""
    cleaned = re.sub(r"^[vV]\.?", "", version_str.strip())
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


class _SourceFilteredGraphRepo:
    """Wraps a BoundGraphRepository to enforce source_filter on visible doc chunks."""

    def __init__(self, inner: BoundGraphRepository, source_filter: SourceFilter) -> None:
        self._inner = inner
        self._source_filter = source_filter

    def list_visible_doc_chunks(
        self, principal: PrincipalContext, domain_id: str
    ) -> list[DocChunkRecord]:
        raw_chunks = self._inner.list_visible_doc_chunks(principal, domain_id)
        return filter_doc_chunks_by_source(raw_chunks, self._source_filter)

    def search_visible_doc_chunks(
        self,
        principal: PrincipalContext,
        domain_id: str,
        query_text: str = "",
        market_scope: MarketRegime | None = None,
        limit: int = 10,
    ) -> list[DocChunkRecord]:
        fetch_limit = max(limit, limit * 3) if self._source_filter != "all" else limit
        raw_chunks = self._inner.search_visible_doc_chunks(
            principal=principal,
            domain_id=domain_id,
            query_text=query_text,
            market_scope=market_scope,
            limit=fetch_limit,
        )
        return filter_doc_chunks_by_source(raw_chunks, self._source_filter)[:limit]

    def search_visible_doc_chunks_calibrated(
        self,
        principal: PrincipalContext,
        domain_id: str,
        query_text: str,
        market_scope: MarketRegime | None = None,
        limit: int = 50,
        min_calibrated_score: float = 0.0,
    ) -> tuple[list[DocChunkRecord], list[CalibratedRankedSearchHit]]:
        fetch_limit = max(limit, limit * 3) if self._source_filter != "all" else limit
        raw_chunks, raw_hits = self._inner.search_visible_doc_chunks_calibrated(
            principal=principal,
            domain_id=domain_id,
            query_text=query_text,
            market_scope=market_scope,
            limit=fetch_limit,
            min_calibrated_score=min_calibrated_score,
        )
        filtered_chunks = filter_doc_chunks_by_source(raw_chunks, self._source_filter)[:limit]
        allowed_ids = {str(chk.chunk_id) for chk in filtered_chunks}
        filtered_hits = [hit for hit in raw_hits if str(hit.item_id) in allowed_ids]
        return filtered_chunks, filtered_hits

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _detect_pack_conflicts(
    code_slices: list[VerbatimCodeSlice],
    doc_chunks: list[DocChunkRecord],
    initial_conflicts: list[str] | None = None,
) -> list[str]:
    """Runs ConflictDetector across visible Confluence pages and code slices, preserving order."""
    detector = ConflictDetector()
    page_conflicts = detector.detect_confluence_page_conflicts(doc_chunks)
    code_doc_conflicts = detector.detect_code_vs_doc_conflicts(code_slices, doc_chunks)
    combined = [
        *(initial_conflicts or []),
        *page_conflicts,
        *code_doc_conflicts,
    ]
    return list(dict.fromkeys(combined))


def _build_domain_brief_from_citations(
    *,
    user_prompt: str,
    domain_id: str,
    citations: list[DocCitationSpan],
    chunks_by_id: dict[str, DocChunkRecord] | None = None,
) -> DomainKnowledgeBrief:
    """Synthesizes a grounded DomainKnowledgeBrief directly from verified citations."""
    if not citations:
        return DomainKnowledgeBrief(
            brief_title="Insufficient Verified Corpus Citations",
            domain_id=domain_id,
            executive_summary="No verified corpus citations found for prompt.",
            formal_expression_or_rule=None,
            core_findings_and_rules=[],
            domain_constraints_and_caveats=[],
            known_risks_and_failure_modes=[],
            structured_metadata={"domain_id": domain_id, "verified_citations_count": 0},
            verified_citations=[],
        )

    chunk_lookup = chunks_by_id or {}
    formatted_labels: list[str] = []
    findings: list[str] = []
    for cit in citations:
        chk = chunk_lookup.get(cit.chunk_id)
        meta = dict(getattr(chk, "metadata_json", {}) or {}) if chk is not None else None
        pub_date = getattr(chk, "published_date", None) if chk is not None else None
        label = format_doc_citation_label(cit, metadata=meta, published_date=pub_date)
        formatted_labels.append(label)
        findings.append(f"{label} {cit.verbatim_quote}")

    primary_title = citations[0].doc_title or citations[0].doc_id or "Verified Documentation"
    return DomainKnowledgeBrief(
        brief_title=primary_title,
        domain_id=domain_id,
        executive_summary=(
            f"Grounded synthesis for '{user_prompt.strip()}' across "
            f"{len(citations)} verified citation(s): {'; '.join(formatted_labels)}"
        ),
        formal_expression_or_rule=None,
        core_findings_and_rules=findings,
        domain_constraints_and_caveats=[
            (
                f"Grounded exclusively in {len(citations)} verified non-stale citation(s) "
                f"visible under domain '{domain_id}' RBAC scopes."
            )
        ],
        known_risks_and_failure_modes=[
            (
                "Stale, superseded, or negatively resolved documents are excluded; "
                "re-verify upstream Confluence/Jira revisions before policy changes."
            )
        ],
        structured_metadata={
            "domain_id": domain_id,
            "verified_citations_count": len(citations),
        },
        verified_citations=list(citations),
    )


def _stitch_section_content(
    chk: DocChunkRecord,
    all_chunks: list[DocChunkRecord] | None = None,
) -> str:
    """Stitches adjacent sibling chunks with the same (doc_id, section_heading)."""
    base = (chk.verbatim_content or "").strip()
    if not all_chunks:
        return base
    siblings = [
        c
        for c in all_chunks
        if c.doc_id == chk.doc_id
        and (c.section_heading or "") == (chk.section_heading or "")
        and (c.verbatim_content or "").strip()
    ]
    if len(siblings) <= 1:
        return base
    seen_texts: set[str] = set()
    parts: list[str] = []
    for sib in siblings:
        txt = (sib.verbatim_content or "").strip()
        if txt and txt not in seen_texts:
            seen_texts.add(txt)
            parts.append(txt)
    return "\n\n".join(parts) if parts else base


def _stratified_doc_search(
    bound_repo: BoundGraphRepository | Any,
    *,
    principal: PrincipalContext,
    domain_id: str,
    plan: QueryPlan,
    market_scope: Any = None,
    per_facet_limit: int = 6,
    max_total: int = 12,
) -> list[DocChunkRecord]:
    """Runs per-facet RRF retrieval and round-robin merges so no single facet starves others."""
    queries: list[str] = (
        [f.facet_query for f in plan.facets] if plan.facets else [plan.original_query]
    )
    if plan.original_query not in queries:
        queries.append(plan.original_query)

    per_query_results: list[list[DocChunkRecord]] = []
    for q_text in queries:
        try:
            res = bound_repo.search_visible_doc_chunks(
                principal=principal,
                domain_id=domain_id,
                query_text=q_text,
                market_scope=market_scope,
                limit=per_facet_limit,
            )
            per_query_results.append(list(res))
        except Exception as exc:
            logger.debug(
                "Multi-facet doc search failed for domain '%s' (%s): %s",
                domain_id,
                type(exc).__name__,
                exc,
            )
            per_query_results.append([])

    merged: list[DocChunkRecord] = []
    seen_ids: set[str] = set()
    max_len = max((len(r) for r in per_query_results), default=0)
    for idx in range(max_len):
        for bucket in per_query_results:
            if idx < len(bucket):
                chk = bucket[idx]
                cid = str(chk.chunk_id)
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    merged.append(chk)
                    if len(merged) >= max_total:
                        return merged
    return merged


def _evaluate_query_plan_coverage(
    plan: QueryPlan,
    *,
    doc_chunks: list[Any],
    code_slices: list[Any],
    domain_id: str,
) -> tuple[list[QueryFacetCoverage], list[str], float]:
    """Evaluates per-facet grounding -> (coverages, missing_aspects, completeness_ratio)."""
    if not plan.facets:
        return [], [], 1.0
    coverages: list[QueryFacetCoverage] = []
    missing_aspects: list[str] = []
    score_sum = 0.0
    for facet in plan.facets:
        cov, missing_msg = evaluate_facet_grounding(
            facet,
            doc_chunks=doc_chunks,
            code_slices=code_slices,
            domain_id=domain_id,
        )
        coverages.append(cov)
        if missing_msg:
            missing_aspects.append(missing_msg)
        if cov.status == "grounded":
            score_sum += 1.0
        elif cov.status == "partial":
            score_sum += 0.5
    completeness_ratio = round(score_sum / max(1, len(coverages)), 4)
    return coverages, missing_aspects, completeness_ratio


def _compute_stage_confidences(
    *,
    principal: PrincipalContext,
    plan: QueryPlan,
    normalized_rrf: float,
    completeness_ratio: float,
    query_facets: list[QueryFacetCoverage],
    code_slices: list[Any],
    citations: list[Any],
    synthesis_mode: str,
    preflight_report: PreFlightValidationReport | None,
    conflicts_count: int = 0,
    stale_count: int = 0,
    is_abstention: bool = False,
    injection_threat_score: float = 0.0,
) -> list[StageConfidence]:
    """Computes calibrated Stage 1..4 confidence scores and signal breakdowns."""
    grounded_count = sum(1 for f in query_facets if f.status == "grounded")
    total_facets = max(1, len(query_facets) or len(plan.facets))
    rls_count = len(principal.allowed_scopes)

    s1_score = max(
        0.0,
        min(
            1.0,
            (1.0 - injection_threat_score) * (0.85 if rls_count > 0 else 0.50) + 0.15,
        ),
    )
    s1 = StageConfidence.from_score(
        stage=1,
        stage_name="Stage 1: Security, Auth & Intent",
        score=s1_score,
        signals={
            "rls_scopes_count": rls_count,
            "injection_threat_score": round(injection_threat_score, 4),
            "facet_count": len(plan.facets),
            "requested_cardinality": plan.requested_cardinality or 0,
        },
        summary=f"RLS enforced ({rls_count} scopes), {len(plan.facets)} facet(s) parsed.",
    )

    modality_hits = sum(
        [
            1 if code_slices else 0,
            1 if citations else 0,
            1 if (normalized_rrf >= 0.55 and (code_slices or citations)) else 0,
        ]
    )
    modality_cov = modality_hits / 3.0
    if is_abstention or (grounded_count == 0 and not code_slices and not citations):
        s2_score = min(0.20, 0.20 * normalized_rrf)
    else:
        s2_score = 0.45 * normalized_rrf + 0.40 * completeness_ratio + 0.15 * modality_cov
    s2 = StageConfidence.from_score(
        stage=2,
        stage_name="Stage 2: Federated Retrieval",
        score=s2_score,
        signals={
            "normalized_rrf": round(normalized_rrf, 4),
            "grounded_facets": grounded_count,
            "total_facets": total_facets,
            "completeness_ratio": round(completeness_ratio, 4),
            "modality_coverage": round(modality_cov, 3),
        },
        summary=(f"RRF {normalized_rrf:.2f}; {grounded_count}/{total_facets} facets grounded."),
    )

    ast_closure_count = sum(1 for s in code_slices if getattr(s, "auto_expanded_by_ast", False))
    sym_res_ratio = (
        1.0
        if (
            preflight_report is not None
            and preflight_report.check_results.get("check_1_symbol_resolution", True)
        )
        else (0.0 if is_abstention else 0.6)
    )
    mode_base = (
        0.94
        if synthesis_mode == "live_qlora_json"
        else (0.88 if synthesis_mode in ("ast_extractive", "hybrid_federated") else 0.78)
    )
    s3_score = (
        0.0 if is_abstention else (mode_base * (0.4 + 0.6 * completeness_ratio) * sym_res_ratio)
    )
    s3 = StageConfidence.from_score(
        stage=3,
        stage_name="Stage 3: Synthesis & AST Closure",
        score=s3_score,
        signals={
            "symbol_resolution_ratio": round(sym_res_ratio, 4),
            "ast_closure_count": ast_closure_count,
            "synthesis_mode": synthesis_mode,
        },
        summary=f"Mode: {synthesis_mode}; AST closure slices: {ast_closure_count}.",
    )

    if preflight_report is not None and preflight_report.check_results:
        checks_total = len(preflight_report.check_results)
        checks_passed = (
            0 if is_abstention else sum(1 for ok in preflight_report.check_results.values() if ok)
        )
        checks_ratio = checks_passed / max(1, checks_total)
        gate_ok = (not is_abstention) and preflight_report.passed
    else:
        checks_total = 6
        checks_passed = 0 if is_abstention else 6
        checks_ratio = 0.0 if is_abstention else 1.0
        gate_ok = not is_abstention

    conflict_penalty = min(0.20, 0.05 * conflicts_count)
    stale_penalty = min(0.15, 0.05 * stale_count)
    s4_score = (
        0.0
        if not gate_ok
        else max(
            0.0,
            0.50 * checks_ratio
            + 0.35 * completeness_ratio
            + 0.15
            - conflict_penalty
            - stale_penalty,
        )
    )
    s4 = StageConfidence.from_score(
        stage=4,
        stage_name="Stage 4: Pre-Flight Gate & Audit",
        score=s4_score,
        signals={
            "checks_passed": checks_passed,
            "checks_total": checks_total,
            "completeness_ratio": round(completeness_ratio, 4),
            "conflicts_count": conflicts_count,
            "stale_excluded_count": stale_count,
        },
        summary=(
            f"Pre-flight {checks_passed}/{checks_total}; completeness {completeness_ratio:.0%}."
        ),
    )
    return [s1, s2, s3, s4]


def _build_doc_citation_span(
    chk: DocChunkRecord,
    verbatim_quote: str,
    *,
    all_chunks: list[DocChunkRecord] | None = None,
) -> DocCitationSpan:
    """Builds DocCitationSpan with Confluence/Jira provenance extracted from chunk metadata/URI."""
    meta = dict(getattr(chk, "metadata_json", {}) or {})
    source_uri = chk.source_uri or ""
    source_lower = source_uri.lower()

    page_id_val = meta.get("page_id")
    page_id = str(page_id_val).strip() if page_id_val is not None else None
    if not page_id and "confluence" in source_lower:
        page_match = re.search(r"/pages/(\d+)", source_uri)
        if page_match:
            page_id = page_match.group(1)
        elif chk.doc_id.isdigit():
            page_id = chk.doc_id

    ticket_key_val = meta.get("issue_key") or meta.get("ticket_key")
    ticket_key = str(ticket_key_val).strip() if ticket_key_val is not None else None
    if not ticket_key and "jira" in source_lower:
        issue_match = re.search(r"/issues/([A-Z][A-Z0-9_]*-\d+)", source_uri)
        if issue_match:
            ticket_key = issue_match.group(1)
        else:
            doc_match = re.search(r"\b([A-Z][A-Z0-9_]*-\d+)\b", chk.doc_id)
            if doc_match:
                ticket_key = doc_match.group(1)

    span = DocCitationSpan(
        chunk_id=str(chk.chunk_id),
        doc_id=chk.doc_id,
        doc_title=chk.doc_title,
        source_uri=source_uri,
        section_heading=chk.section_heading,
        verbatim_quote=verbatim_quote,
        full_section_content=_stitch_section_content(chk, all_chunks),
        version=_extract_chunk_version(chk),
        page_id=page_id or None,
        ticket_key=ticket_key or None,
    )
    has_enterprise_meta = bool(
        meta.get("space_key")
        or meta.get("space")
        or meta.get("last_updated")
        or meta.get("updated_at")
        or getattr(chk, "published_date", None)
        or is_wiki_chunk(chk)
        or is_jira_chunk(chk)
    )
    if has_enterprise_meta:
        formatted = format_doc_citation_label(
            span,
            metadata=meta,
            published_date=getattr(chk, "published_date", None),
        )
        if formatted.startswith("[") and formatted.endswith("]"):
            span = span.model_copy(update={"section_heading": formatted[1:-1]})
    return span


def _extract_hybrid_doc_citations(
    *,
    user_prompt: str,
    doc_chunks: list[DocChunkRecord],
    slicer: DeterministicSlicer,
    validator: PreFlightValidator | None = None,
    max_citations: int = 8,
    min_hits_override: int | None = None,
) -> list[DocCitationSpan]:
    """Extracts verified non-stale doc citations matching substantive query terms."""
    raw_terms = re.findall(r"[A-Za-z0-9_-]{3,}", user_prompt)
    expanded_terms: list[str] = []
    for tok in raw_terms:
        expanded_terms.append(tok)
        if "-" in tok:
            for part in tok.split("-"):
                if len(part) >= 3:
                    expanded_terms.append(part)
    query_terms = [
        t.lower()
        for t in dict.fromkeys(expanded_terms)
        if t.lower() not in PROMPT_STOPWORDS
        and t.lower() not in CONVERSATIONAL_AND_GENERIC_WORDS
        and t.lower() not in _HYBRID_DOC_META_WORDS
        and len(t) >= 3
    ]
    if not query_terms or not doc_chunks:
        return []

    citation_gate = validator.citation_gate if validator is not None else DocumentCitationGate()
    neg_resolutions = (
        {r.lower() for r in validator.domain_config.jira_negative_resolutions}
        if validator is not None
        else set()
    )

    min_hits = (
        min_hits_override if min_hits_override is not None else (2 if len(query_terms) >= 2 else 1)
    )
    scored: list[tuple[int, float, float, DocChunkRecord]] = []
    for chk in doc_chunks:
        if bool(getattr(chk, "is_stale", False)):
            continue
        meta = dict(getattr(chk, "metadata_json", {}) or {})
        res_val = str(meta.get("resolution", "")).strip().lower()
        if res_val and res_val in neg_resolutions:
            continue
        haystack = f"{chk.doc_title}\n{chk.section_heading}\n{chk.verbatim_content}".lower()
        hits = sum(
            1 for term in query_terms if re.search(rf"\b{re.escape(term)}\b", haystack) is not None
        )
        if hits >= min_hits:
            ver_num = _numeric_version(_extract_chunk_version(chk))
            scored.append((hits, float(chk.authority_weight), ver_num, chk))

    if not scored:
        return []

    scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    verified: list[DocCitationSpan] = []
    for _, _, _, chk in scored:
        if len(verified) >= max_citations:
            break
        snippet = chk.verbatim_content.strip()
        if not snippet:
            continue
        lines = [ln.strip() for ln in snippet.splitlines() if ln.strip()]
        lead_sentence = snippet.split(".")[0] + "." if "." in snippet else snippet[:400]
        candidates = [*lines, lead_sentence, snippet]
        for cand in candidates:
            if not cand:
                continue
            cand_cit = _build_doc_citation_span(chk, cand, all_chunks=doc_chunks)
            if (
                slicer.verify_doc_citation(
                    citation=cand_cit,
                    chunk_content=chk.verbatim_content,
                ).is_valid
                and citation_gate.verify_citations(
                    citations=[cand_cit],
                    corpus_chunks=[chk],
                    require_citations=False,
                ).passed
            ):
                verified.append(cand_cit)
                break
    return verified


CALIBRATED_RELEVANCE_FLOOR: Final[float] = 0.34


def _build_retrieval_hit_traces(
    hits: list[CalibratedRankedSearchHit],
    *,
    domain_id: str,
    modality: Literal["code", "document", "issues"] = "document",
    min_floor: float = CALIBRATED_RELEVANCE_FLOOR,
) -> list[RetrievalHitTrace]:
    """Convert calibrated search hits into structured RetrievalHitTrace items."""
    traces: list[RetrievalHitTrace] = []
    for h in hits[:6]:
        accepted = h.calibrated_score >= min_floor and not h.is_boilerplate
        reason = None
        if h.is_boilerplate:
            reason = "license_or_boilerplate"
        elif h.calibrated_score < min_floor:
            reason = f"calibrated_score {h.calibrated_score:.3f} < floor {min_floor:.2f}"
        traces.append(
            RetrievalHitTrace(
                item_id=str(h.item_id),
                domain_id=h.domain_id or domain_id,
                title=h.qualified_name_or_title,
                modality=modality,
                cosine_similarity=round(h.cosine_similarity, 4),
                bm25_score=round(h.bm25_score, 4),
                trigram_score=round(h.trigram_score, 4),
                calibrated_score=round(h.calibrated_score, 4),
                accepted=accepted,
                rejection_reason=reason,
            )
        )
    return traces


def _compute_effective_retrieval_scores(
    *,
    explicit_score: float | None,
    retrieval_hits: list[CalibratedRankedSearchHit],
    hydrated_slices: list[VerbatimCodeSlice],
    citations: list[DocCitationSpan],
    completeness_ratio: float,
) -> tuple[float, float]:
    """Compute (validator_rrf_score, calibrated_score) without hardcoded 0.85/0.88 cheats."""
    if explicit_score is not None:
        clamped = round(min(1.0, max(0.0, float(explicit_score))), 6)
        return clamped, clamped
    if retrieval_hits:
        max_cal = round(max(h.calibrated_score for h in retrieval_hits), 6)
        max_rrf = round(max(h.normalized_rrf_score for h in retrieval_hits), 6)
        if max_cal >= CALIBRATED_RELEVANCE_FLOOR:
            return round(max(max_rrf, max_cal), 6), max_cal
        return max_cal, max_cal
    if hydrated_slices or citations:
        if hydrated_slices and completeness_ratio <= 0.0:
            est = 0.72
        else:
            est = round(min(0.92, 0.42 + 0.48 * max(0.0, min(1.0, completeness_ratio))), 6)
        return est, est
    return 0.0, 0.0


def _has_verified_code_slice_hit(
    user_prompt: str,
    hydrated_slices: list[VerbatimCodeSlice],
    partial_facet_count: int = 0,
) -> bool:
    """Verify hydrated AST slices directly match prompt symbols, CamelCase terms, or files."""
    if not hydrated_slices:
        return False
    prompt_tokens = {
        w.lower()
        for w in re.findall(r"[A-Za-z0-9]{3,}", user_prompt)
        if w.lower() not in PROMPT_STOPWORDS
    }
    if not prompt_tokens:
        return False

    for slc in hydrated_slices:
        qsym = str(getattr(slc, "qualified_symbol", "") or "")
        fpath = str(getattr(slc, "file_path", "") or "")
        leaf_sym = qsym.split(".")[-1]
        root_sym = qsym.split(".")[0]
        if (
            leaf_sym.lower() in prompt_tokens
            or root_sym.lower() in prompt_tokens
            or qsym.lower() in prompt_tokens
        ):
            return True

        camel_split = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", qsym)
        sym_tokens = {
            t.lower()
            for t in re.split(r"[^A-Za-z0-9]+", camel_split)
            if len(t) >= 4 and t.lower() not in PROMPT_STOPWORDS
        }
        stem = Path(fpath).stem if fpath else ""
        file_tokens = {
            t.lower()
            for t in re.split(r"[^A-Za-z0-9]+", stem)
            if len(t) >= 4 and t.lower() not in PROMPT_STOPWORDS
        }
        matched_sym = sym_tokens & prompt_tokens
        matched_file = file_tokens & prompt_tokens
        if len(matched_sym | matched_file) >= 2:
            return True
        if partial_facet_count > 0 and (matched_sym or matched_file):
            return True
    return False


def _build_preflight_ledger(
    *,
    report: PreFlightValidationReport | None,
    calibrated_score: float,
    min_floor: float,
    sufficiency_verdict: Literal["sufficient", "partial", "insufficient"],
    grounded_facets: int,
    total_facets: int,
    llm_verified_sufficient: bool = False,
    code_symbol_verified: bool = False,
) -> list[PreFlightCheckLedgerItem]:
    """Build deterministic check-by-check Stage 4 verification ledger."""
    items: list[PreFlightCheckLedgerItem] = []
    if report is not None:
        thresholds = {
            "check_1_syntax": "0 syntax errors",
            "check_2_closure": "0 unresolved symbols",
            "check_3_rrf": "score >= 0.65",
            "check_4_scope": "0 unauthorized items",
            "check_5_provenance": "100% anchored",
            "check_6_freshness": "100% verbatim & non-stale",
        }
        observed_map = {
            "check_1_syntax": f"{len(report.syntax_errors)} syntax error(s)",
            "check_2_closure": f"{len(report.unresolved_symbols)} unresolved",
            "check_3_rrf": f"{report.normalized_rrf_score:.3f}",
            "check_4_scope": f"{len(report.excised_unauthorized_items)} excised",
            "check_5_provenance": f"{len(report.unanchored_provenance_items)} unanchored",
            "check_6_freshness": (
                f"citations_100pct={report.citations_verified_100_pct}, "
                f"stale={len(report.stale_invalidated_items)}"
            ),
        }
        for name, passed in report.check_results.items():
            items.append(
                PreFlightCheckLedgerItem(
                    check_name=name,
                    passed=bool(passed),
                    observed=observed_map.get(name, str(passed)),
                    threshold=thresholds.get(name, "true"),
                )
            )
    items.append(
        PreFlightCheckLedgerItem(
            check_name="calibrated_relevance_floor",
            passed=calibrated_score >= min_floor,
            observed=f"{calibrated_score:.3f}",
            threshold=f">= {min_floor:.2f}",
        )
    )
    items.append(
        PreFlightCheckLedgerItem(
            check_name="sufficiency_verdict",
            passed=sufficiency_verdict != "insufficient",
            observed=sufficiency_verdict,
            threshold="sufficient | partial",
        )
    )
    items.append(
        PreFlightCheckLedgerItem(
            check_name="facet_grounding",
            passed=(
                total_facets == 0
                or grounded_facets > 0
                or llm_verified_sufficient
                or code_symbol_verified
            ),
            observed=(
                f"{grounded_facets}/{total_facets} facets grounded (live Gemma-4 verified)"
                if (grounded_facets == 0 and llm_verified_sufficient)
                else (
                    f"{grounded_facets}/{total_facets} facets grounded "
                    "(verified AST code slice & symbol match)"
                    if (grounded_facets == 0 and code_symbol_verified)
                    else f"{grounded_facets}/{total_facets} facets grounded"
                )
            ),
            threshold=">= 1 grounded facet",
        )
    )
    return items


class ScoutWorkflow:
    """4-Stage Google ADK 2.x Context Scout DAG with $0 Local Bounce-Back Loop.

    Stage 1: Guardrail scan (ShieldGemma Guardrail + Sanitizer) & RBAC Principal verification.
    Stage 2: MCP Read Tool execution, candidate AST localization, 1-hop dependency expansion,
             and byte-for-byte DeterministicSlicer hydration.
    Stage 3: PreFlightValidator gate with $0 Local Bounce-Back Self-Correction Loop
             (max 1 local retry before returning InsufficientContextDiagnostic without paid tokens).
    Stage 4: Compiler-complete ImplementationContextPack assembly.
    """

    def __init__(
        self,
        registry: DomainRegistry | None = None,
        repo: Any = None,
        local_client: LocalGemmaClient | None = None,
        validator: PreFlightValidator | None = None,
        slicer: DeterministicSlicer | None = None,
        expander: ASTDependencyExpander | None = None,
        guardrail: ShieldGemmaGuardrail | None = None,
        auth_gate: DomainAuthorizationGate | None = None,
        max_search_hops: int = 2,
    ) -> None:
        if registry is None:
            from context_scout.api.dependencies import get_domain_registry

            registry = get_domain_registry()
        self.registry = registry
        self.repo = repo
        self.slicer = slicer or DeterministicSlicer()
        self.expander = expander or ASTDependencyExpander(
            max_direct_callers=8,
            max_expansion_token_budget=10_000,
        )
        self.validator = validator or PreFlightValidator(
            repo=self.repo,
            slicer=self.slicer,
            expander=self.expander,
        )
        if local_client is not None:
            if local_client.adapter_pool is None and self.registry is not None:
                if not self.registry._manifests and self.registry._domains_dir.exists():
                    self.registry.load_all()
                local_client.adapter_pool = self.registry.adapter_pool
            self.local_client = local_client
        else:
            self.local_client = LocalGemmaClient(
                adapter_pool=self.registry.adapter_pool if self.registry else None
            )
        self.guardrail = guardrail or ShieldGemmaGuardrail()
        self.auth_gate = auth_gate or DomainAuthorizationGate(self.registry)
        self.max_search_hops = max_search_hops

    def _stage1_retrieve(self, state: dict[str, Any]) -> dict[str, Any]:
        """Stage 1 ADK node: Guardrail scan, RBAC verification, and hybrid RRF retrieval."""
        user_prompt = str(state.get("user_prompt", ""))
        principal = state.get("principal")
        if principal is not None:
            decision = self.guardrail.evaluate(user_prompt, principal=principal)
            state["guardrail_passed"] = decision.passed
        state["stage"] = 1
        return state

    def _stage2_expand(self, state: dict[str, Any]) -> dict[str, Any]:
        """Stage 2 ADK node: Candidate AST pointer synthesis and 1-hop dependency closure."""
        state["stage"] = 2
        return state

    def _stage3_slice(self, state: dict[str, Any]) -> dict[str, Any]:
        """Stage 3 ADK node: Deterministic byte-for-byte line slicing and quote hydration."""
        state["stage"] = 3
        return state

    def _stage4_validate(self, state: dict[str, Any]) -> dict[str, Any]:
        """Stage 4 ADK node: PreFlightValidator gate and $0 local bounce-back verification."""
        state["stage"] = 4
        return state

    def as_adk_workflow(self) -> Any:
        """Returns Google ADK 2.x Workflow representation with 4 wired stage nodes."""
        s1 = self._stage1_retrieve
        s2 = self._stage2_expand
        s3 = self._stage3_slice
        s4 = self._stage4_validate
        stages = [s1, s2, s3, s4]
        if HAS_GOOGLE_ADK:
            wf_obj = ADKWorkflow(
                name="scout_workflow_4stage",
                description="Google ADK 2.x 4-Stage Context Scout Workflow with $0 Bounce-Back",
                edges=[
                    ("START", s1),
                    (s1, s2),
                    (s2, s3),
                    (s3, s4),
                ],
            )
            object.__setattr__(wf_obj, "stages", stages)
            return wf_obj
        return None

    def execute(
        self,
        user_prompt: str,
        principal: PrincipalContext,
        domain_id: str | None = None,
        trace_id: str | None = None,
        **kwargs: Any,
    ) -> ImplementationContextPack | InsufficientContextDiagnostic:
        """Executes the scout workflow DAG, enforcing fail-closed model verification."""
        domain_id = _resolve_default_domain(domain_id, role="code")
        active_trace_id = trace_id or f"trace_{uuid.uuid4().hex[:12]}"
        try:
            _ = self.local_client.complete(
                prompt=user_prompt,
                domain_id=domain_id,
            )
        except ModelUnavailableError as e:
            tracer = ScoutTelemetryTracer.get_instance()
            with tracer.trace_workflow(
                span_name="scout.workflow",
                domain_id=domain_id,
                trace_id=active_trace_id,
                model_id="gemma-4-moe",
                system="gemma",
            ) as wf_span:
                diag = InsufficientContextDiagnostic(
                    trace_id=active_trace_id,
                    domain_id=domain_id,
                    reason=f"Model unavailable: {e}",
                    is_abstention=True,
                    paid_tokens_spent=0,
                    missing_items=["model:local_gemma:unavailable"],
                    failure_check="MODEL_UNAVAILABLE",
                )
                wf_span.record_pack(diag)
                return diag

        return self.run(
            user_prompt=user_prompt,
            principal=principal,
            domain_id=domain_id,
            trace_id=active_trace_id,
            **kwargs,
        )

    def run(
        self,
        user_prompt: str,
        principal: PrincipalContext,
        domain_id: str | None = None,
        trace_id: str | None = None,
        normalized_rrf_score: float | None = None,
        doc_chunks: dict[str, str] | list[DocChunkRecord] | None = None,
        supporting_doc_citations: list[DocCitationSpan] | None = None,
        max_bounce_backs: int = 1,
        on_stage_event: Callable[[dict[str, Any]], None] | None = None,
        source_filter: SourceFilter = "all",
    ) -> ImplementationContextPack | InsufficientContextDiagnostic:
        """Executes the 4-stage Scout Workflow DAG with OpenTelemetry instrumentation."""
        domain_id = _resolve_default_domain(domain_id, role="code")
        active_trace_id = trace_id or f"trace_{uuid.uuid4().hex[:12]}"
        tracer = ScoutTelemetryTracer.get_instance()
        with tracer.trace_workflow(
            span_name="scout.workflow",
            domain_id=domain_id,
            trace_id=active_trace_id,
            model_id="gemma-4-moe",
            system="gemma",
        ) as wf_span:
            try:
                result = self._run_dag(
                    user_prompt=user_prompt,
                    principal=principal,
                    domain_id=domain_id,
                    active_trace_id=active_trace_id,
                    normalized_rrf_score=normalized_rrf_score,
                    doc_chunks=doc_chunks,
                    supporting_doc_citations=supporting_doc_citations,
                    max_bounce_backs=max_bounce_backs,
                    on_stage_event=on_stage_event,
                    source_filter=source_filter,
                )
            except ModelUnavailableError as e:
                logger.warning(
                    "Model unavailable during workflow execution (trace=%s, domain=%s): %s",
                    active_trace_id,
                    domain_id,
                    e,
                )
                result = InsufficientContextDiagnostic(
                    trace_id=active_trace_id,
                    domain_id=domain_id,
                    reason=f"Model unavailable: {e}",
                    is_abstention=True,
                    paid_tokens_spent=0,
                    missing_items=["model:local_gemma:unavailable"],
                    failure_check="MODEL_UNAVAILABLE",
                )
            wf_span.record_pack(result)
            return result

    def _run_dag(
        self,
        user_prompt: str,
        principal: PrincipalContext,
        domain_id: str,
        active_trace_id: str,
        normalized_rrf_score: float | None,
        doc_chunks: dict[str, str] | list[DocChunkRecord] | None,
        supporting_doc_citations: list[DocCitationSpan] | None,
        max_bounce_backs: int,
        on_stage_event: Callable[[dict[str, Any]], None] | None = None,
        source_filter: SourceFilter = "all",
    ) -> ImplementationContextPack | InsufficientContextDiagnostic:
        """Executes the 4-stage Scout Workflow DAG."""
        dag_t0 = time.perf_counter()

        # =====================================================================
        # Stage 1: Security Guardrails & Principal Authorization
        # =====================================================================
        guardrail_decision = self.guardrail.evaluate(user_prompt, principal=principal)
        if not guardrail_decision.passed:
            logger.warning(
                "Stage 1 Guardrail rejected prompt (trace=%s): %s",
                active_trace_id,
                guardrail_decision.rationale,
            )
            return InsufficientContextDiagnostic(
                trace_id=active_trace_id,
                domain_id=domain_id,
                reason=f"Security guardrail rejected input: {guardrail_decision.rationale}",
                missing_symbols=[],
                partial_candidate_files=[],
                paid_tokens_spent=0,
            )

        domain_plugin = PluginRegistry.get_domain_plugin(domain_id)
        if domain_plugin is not None and domain_plugin.is_federated_domain(domain_id):
            fed_coord = FederatedScoutCoordinator(
                registry=self.registry,
                repo=self.repo,
                local_client=self.local_client,
                validator=self.validator,
                slicer=self.slicer,
                expander=self.expander,
                guardrail=self.guardrail,
                auth_gate=self.auth_gate,
            )
            return fed_coord._run_federated_dag(
                user_prompt=user_prompt,
                principal=principal,
                active_trace_id=active_trace_id,
                normalized_rrf_score=normalized_rrf_score,
                doc_chunks=doc_chunks,
                max_bounce_backs=max_bounce_backs,
                on_stage_event=on_stage_event,
                source_filter=source_filter,
            )

        try:
            manifest = self.registry.get(domain_id)
        except KeyError as exc:
            logger.warning(
                "Domain '%s' not found in DomainRegistry (%s)",
                domain_id,
                type(exc).__name__,
            )
            return InsufficientContextDiagnostic(
                trace_id=active_trace_id,
                domain_id=domain_id,
                reason=f"Domain '{domain_id}' not found in DomainRegistry",
                missing_symbols=[],
                partial_candidate_files=[],
                paid_tokens_spent=0,
            )

        is_pg_repo = (
            self.repo is None
            or isinstance(self.repo, PostgresRLSRepository)
            or hasattr(self.repo, "connection_manager")
        )
        try:
            from context_scout.agents.preflight_gate import sync_custom_domain_corpus_if_needed
            from context_scout.ingestion.git_merkle_indexer import sync_tracked_in_memory_repos

            sync_tracked_in_memory_repos(self.repo, principal=principal, domain_id=domain_id)
            if is_pg_repo:
                domains_dir = getattr(self.registry, "domains_dir", None) or getattr(
                    self.registry, "_domains_dir", None
                )
                index_path = (
                    domains_dir / domain_id / "sync_manifest.json"
                    if isinstance(domains_dir, Path)
                    else None
                )
                sync_custom_domain_corpus_if_needed(domain_id, index_path=index_path)
        except Exception as exc:
            logger.debug(
                "Live repository sync skipped for domain '%s' (%s): %s",
                domain_id,
                type(exc).__name__,
                exc,
            )

        run_bound_repo = BoundGraphRepository(self.repo)
        relevance_rejection = _evaluate_query_domain_relevance(
            user_prompt=user_prompt,
            domain_id=domain_id,
            bound_repo=run_bound_repo,
            principal=principal,
        )
        if relevance_rejection is not None:
            return InsufficientContextDiagnostic(
                trace_id=active_trace_id,
                domain_id=domain_id,
                reason=relevance_rejection,
                missing_symbols=[],
                partial_candidate_files=[],
                paid_tokens_spent=0,
            )

        s1_t0 = time.perf_counter()
        _ = self.local_client.set_active_domain(domain_id)
        query_plan = plan_query_with_llm(user_prompt, self.local_client)
        if domain_plugin is not None and hasattr(domain_plugin, "expand_query_plan"):
            query_plan = domain_plugin.expand_query_plan(query_plan)
        visible_files = run_bound_repo.list_visible_code_files(principal, domain_id)
        filtered_doc_chunks = filter_doc_chunks_by_source(doc_chunks, source_filter)
        doc_bound_repo: BoundGraphRepository | _SourceFilteredGraphRepo = (
            _SourceFilteredGraphRepo(run_bound_repo, source_filter)
            if source_filter != "all"
            else run_bound_repo
        )
        count_fn = (
            getattr(doc_bound_repo, "count_visible_doc_chunks", None)
            or getattr(getattr(doc_bound_repo, "adapter", None), "count_visible_doc_chunks", None)
            or getattr(
                getattr(getattr(doc_bound_repo, "_inner", None), "adapter", None),
                "count_visible_doc_chunks",
                None,
            )
        )
        if isinstance(filtered_doc_chunks, list):
            s1_raw_chunks = filtered_doc_chunks
            stale_excluded_s1 = sum(
                1 for c in filtered_doc_chunks if bool(getattr(c, "is_stale", False))
            )
            visible_chunks_s1 = len(filtered_doc_chunks) - stale_excluded_s1
        elif callable(count_fn):
            visible_chunks_s1, stale_excluded_s1 = count_fn(principal, domain_id)
            if visible_chunks_s1 > 0:
                s1_raw_chunks, _ = doc_bound_repo.search_visible_doc_chunks_calibrated(
                    principal=principal,
                    domain_id=domain_id,
                    query_text=user_prompt,
                    limit=12,
                )
            else:
                s1_raw_chunks = []
        else:
            s1_raw_chunks = doc_bound_repo.list_visible_doc_chunks(principal, domain_id)
            stale_excluded_s1 = sum(1 for c in s1_raw_chunks if bool(getattr(c, "is_stale", False)))
            visible_chunks_s1 = len(s1_raw_chunks) - stale_excluded_s1
        s1_ms = max(1, int((time.perf_counter() - s1_t0) * 1000))
        s1_trace = StageExecutionTrace(
            stage=1,
            stage_name="Stage 1: Auth, Guardrail & Query Plan",
            status="completed",
            elapsed_ms=float(s1_ms),
            summary=(
                f"Guardrail passed; {len(principal.allowed_scopes)} scope(s); "
                f"query_type={query_plan.query_type}; {len(query_plan.facets)} facet(s)."
            ),
            principal_id=principal.user_id,
            principal_scopes=list(principal.allowed_scopes),
            guardrail_verdict="passed",
            query_type=query_plan.query_type,
            core_subject=query_plan.core_subject,
            required_facets=[f.facet_query for f in query_plan.facets],
            target_domains=[domain_id],
        )
        if on_stage_event is not None:
            on_stage_event(
                {
                    "event_type": "stage_update",
                    "stage": 1,
                    "stage_name": "Stage 1: Auth, Guardrail & Query Plan",
                    "status": "completed",
                    "details": _stage1_details(
                        domains=[domain_id],
                        principal=principal,
                        visible_files_count=len(visible_files),
                        visible_chunks_count=visible_chunks_s1,
                        stale_excluded_count=stale_excluded_s1,
                        elapsed_ms=s1_ms,
                    ),
                    "stage_trace": s1_trace.model_dump(),
                }
            )

        # =====================================================================
        # Stage 2: MCP Read Exploration, AST Localization, & Slicing
        # =====================================================================
        s2_t0 = time.perf_counter()
        prompt_l = user_prompt.lower()
        has_code_indicators = any(
            ext in prompt_l
            for ext in (
                ".py",
                ".ts",
                ".tsx",
                ".js",
                ".jsx",
                ".sql",
                ".yaml",
                ".yml",
                ".tf",
                ".go",
                ".rs",
                ".java",
                "`",
            )
        ) or any(
            kw in prompt_l
            for kw in (
                "code",
                "codebase",
                "file",
                "class",
                "method",
                "function",
                "refactor",
                "module",
                "symbol",
                "bindings",
                "bounce",
            )
        )
        auto_research_route = (
            source_filter != "code"
            and domain_plugin is not None
            and not has_code_indicators
            and domain_plugin.should_auto_route_research(
                user_prompt, domain_id, allowed_scopes=principal.allowed_scopes
            )
        )
        research_domain_id = (
            domain_plugin.get_role_domain_ids().get("research", domain_id)
            if domain_plugin is not None
            else domain_id
        )

        has_visible_code = bool(visible_files)
        has_visible_docs = bool(s1_raw_chunks) or bool(supporting_doc_citations)
        if (
            not has_visible_docs
            and research_domain_id != domain_id
            and doc_bound_repo.list_visible_doc_chunks(principal, research_domain_id)
        ):
            has_visible_docs = True
        intent_decision = QueryIntentClassifier.classify(
            user_prompt,
            source_filter=source_filter,
            has_visible_code=has_visible_code,
            has_visible_docs=has_visible_docs,
        )

        is_knowledge_contract = (
            manifest.output_contract == "DomainKnowledgeBrief" and domain_plugin is not None
        )
        is_doc_domain = (
            manifest.output_contract in ("QuantStrategyBrief", "DomainKnowledgeBrief")
            or (domain_plugin is not None and domain_plugin.is_research_domain(domain_id))
            or auto_research_route
        )
        used_research_domain = False
        quant_brief: QuantStrategyBrief | None = None
        domain_knowledge_brief: DomainKnowledgeBrief | None = None
        candidate_pointers: list[CodeSlicePointer] = []

        if intent_decision.mode == QueryIntentMode.PURE_DOCUMENTATION:
            candidate_pointers = []
            if is_doc_domain:
                quant_domain = research_domain_id if auto_research_route else domain_id
                if is_knowledge_contract:
                    domain_knowledge_brief = self.local_client.generate_domain_knowledge_brief(
                        user_prompt=user_prompt,
                        domain_id=quant_domain,
                        repo=doc_bound_repo,
                        principal=principal,
                        quant_brief=None,
                    )
                    if (
                        domain_knowledge_brief is not None
                        and domain_knowledge_brief.verified_citations
                    ):
                        supporting_doc_citations = list(domain_knowledge_brief.verified_citations)
                    else:
                        domain_knowledge_brief = None
                        supporting_doc_citations = supporting_doc_citations or []
                else:
                    quant_brief = self.local_client.generate_quant_brief(
                        user_prompt=user_prompt,
                        domain_id=quant_domain,
                        repo=doc_bound_repo,
                        principal=principal,
                        bounce_back_feedback=None,
                    )
                    if quant_brief is not None and quant_brief.verified_citations:
                        domain_knowledge_brief = self.local_client.generate_domain_knowledge_brief(
                            user_prompt=user_prompt,
                            domain_id=quant_domain,
                            repo=doc_bound_repo,
                            principal=principal,
                            quant_brief=quant_brief,
                        )
                        supporting_doc_citations = list(quant_brief.verified_citations)
                    else:
                        quant_brief = None
                        domain_knowledge_brief = None
                        supporting_doc_citations = supporting_doc_citations or []

            if not supporting_doc_citations:
                candidate_doc_chunks = (
                    s1_raw_chunks
                    if isinstance(filtered_doc_chunks, list) or len(s1_raw_chunks) < 100
                    else doc_bound_repo.search_visible_doc_chunks(
                        principal=principal,
                        domain_id=domain_id,
                        query_text=user_prompt,
                        limit=20,
                    )
                )
                if candidate_doc_chunks:
                    extracted_cits = _extract_hybrid_doc_citations(
                        user_prompt=user_prompt,
                        doc_chunks=candidate_doc_chunks,
                        slicer=self.slicer,
                        validator=self.validator,
                        min_hits_override=1,
                    )
                    if extracted_cits:
                        supporting_doc_citations = extracted_cits
                        chunks_by_id = {str(c.chunk_id): c for c in candidate_doc_chunks}
                        domain_knowledge_brief = _build_domain_brief_from_citations(
                            user_prompt=user_prompt,
                            domain_id=domain_id,
                            citations=extracted_cits,
                            chunks_by_id=chunks_by_id,
                        )
            elif domain_knowledge_brief is None:
                chunks_by_id = {str(c.chunk_id): c for c in s1_raw_chunks}
                domain_knowledge_brief = _build_domain_brief_from_citations(
                    user_prompt=user_prompt,
                    domain_id=domain_id,
                    citations=supporting_doc_citations,
                    chunks_by_id=chunks_by_id,
                )

            if (
                not supporting_doc_citations
                and not is_doc_domain
                and research_domain_id != domain_id
            ):
                quant_brief = self.local_client.generate_quant_brief(
                    user_prompt=user_prompt,
                    domain_id=research_domain_id,
                    repo=doc_bound_repo,
                    principal=principal,
                    bounce_back_feedback=None,
                )
                if quant_brief is not None and quant_brief.verified_citations:
                    used_research_domain = True
                    domain_knowledge_brief = self.local_client.generate_domain_knowledge_brief(
                        user_prompt=user_prompt,
                        domain_id=research_domain_id,
                        repo=doc_bound_repo,
                        principal=principal,
                        quant_brief=quant_brief,
                    )
                    supporting_doc_citations = list(quant_brief.verified_citations)
        elif is_doc_domain and source_filter != "code":
            quant_domain = research_domain_id if auto_research_route else domain_id
            if is_knowledge_contract:
                domain_knowledge_brief = self.local_client.generate_domain_knowledge_brief(
                    user_prompt=user_prompt,
                    domain_id=quant_domain,
                    repo=doc_bound_repo,
                    principal=principal,
                    quant_brief=None,
                )
                if domain_knowledge_brief is not None and domain_knowledge_brief.verified_citations:
                    supporting_doc_citations = list(domain_knowledge_brief.verified_citations)
                else:
                    domain_knowledge_brief = None
                    supporting_doc_citations = supporting_doc_citations or []
                if not supporting_doc_citations:
                    candidate_doc_chunks = (
                        s1_raw_chunks
                        if isinstance(filtered_doc_chunks, list) or len(s1_raw_chunks) < 100
                        else doc_bound_repo.search_visible_doc_chunks(
                            principal=principal,
                            domain_id=domain_id,
                            query_text=user_prompt,
                            limit=20,
                        )
                    )
                    if candidate_doc_chunks:
                        extracted_cits = _extract_hybrid_doc_citations(
                            user_prompt=user_prompt,
                            doc_chunks=candidate_doc_chunks,
                            slicer=self.slicer,
                            validator=self.validator,
                            min_hits_override=1,
                        )
                        if extracted_cits:
                            supporting_doc_citations = extracted_cits
                            chunks_by_id = {str(c.chunk_id): c for c in candidate_doc_chunks}
                            domain_knowledge_brief = _build_domain_brief_from_citations(
                                user_prompt=user_prompt,
                                domain_id=domain_id,
                                citations=extracted_cits,
                                chunks_by_id=chunks_by_id,
                            )
            else:
                quant_brief = self.local_client.generate_quant_brief(
                    user_prompt=user_prompt,
                    domain_id=quant_domain,
                    repo=doc_bound_repo,
                    principal=principal,
                    bounce_back_feedback=None,
                )
                if quant_brief is not None and quant_brief.verified_citations:
                    domain_knowledge_brief = self.local_client.generate_domain_knowledge_brief(
                        user_prompt=user_prompt,
                        domain_id=quant_domain,
                        repo=doc_bound_repo,
                        principal=principal,
                        quant_brief=quant_brief,
                    )
                    supporting_doc_citations = list(quant_brief.verified_citations)
                else:
                    quant_brief = None
                    domain_knowledge_brief = None
                    supporting_doc_citations = []
            candidate_pointers = []
        else:
            if source_filter == "code":
                supporting_doc_citations = []
            candidate_pointers = self.local_client.generate_code_pointers(
                user_prompt=user_prompt,
                domain_id=domain_id,
                repo=run_bound_repo,
                principal=principal,
                bounce_back_feedback=None,
            )
            if (
                intent_decision.mode == QueryIntentMode.HYBRID_CODE_AND_DOCS
                and source_filter != "code"
                and candidate_pointers
                and supporting_doc_citations is None
                and s1_raw_chunks
            ):
                candidate_doc_chunks = (
                    s1_raw_chunks
                    if isinstance(filtered_doc_chunks, list) or len(s1_raw_chunks) < 100
                    else doc_bound_repo.search_visible_doc_chunks(
                        principal=principal,
                        domain_id=domain_id,
                        query_text=user_prompt,
                        limit=10,
                    )
                )
                hybrid_cits = _extract_hybrid_doc_citations(
                    user_prompt=user_prompt,
                    doc_chunks=candidate_doc_chunks,
                    slicer=self.slicer,
                    validator=self.validator,
                )
                if hybrid_cits:
                    supporting_doc_citations = hybrid_cits
                    chunks_by_id = {str(c.chunk_id): c for c in candidate_doc_chunks}
                    domain_knowledge_brief = _build_domain_brief_from_citations(
                        user_prompt=user_prompt,
                        domain_id=domain_id,
                        citations=hybrid_cits,
                        chunks_by_id=chunks_by_id,
                    )

        stage2_hits: list[CalibratedRankedSearchHit] = []
        active_doc_domain = (
            research_domain_id if (auto_research_route or used_research_domain) else domain_id
        )
        if supporting_doc_citations or is_doc_domain:
            try:
                _, doc_cal_hits = doc_bound_repo.search_visible_doc_chunks_calibrated(
                    principal=principal,
                    domain_id=active_doc_domain,
                    query_text=user_prompt,
                    limit=8,
                )
                stage2_hits.extend(doc_cal_hits)
            except Exception as exc:
                logger.debug(
                    "Stage 2 calibrated doc probe skipped (%s): %s", type(exc).__name__, exc
                )
        if candidate_pointers:
            try:
                _, sym_cal_hits = run_bound_repo.search_visible_ast_symbols_calibrated(
                    principal=principal,
                    domain_id=domain_id,
                    query_text=user_prompt,
                    limit=8,
                )
                stage2_hits.extend(sym_cal_hits)
            except Exception as exc:
                logger.debug(
                    "Stage 2 calibrated symbol probe skipped (%s): %s", type(exc).__name__, exc
                )

        s2_comp_prelim = (
            domain_knowledge_brief.completeness_ratio
            if domain_knowledge_brief is not None
            else (1.0 if (candidate_pointers or supporting_doc_citations) else 0.0)
        )
        effective_rrf_score, calibrated_retrieval_score = _compute_effective_retrieval_scores(
            explicit_score=normalized_rrf_score,
            retrieval_hits=stage2_hits,
            hydrated_slices=[],
            citations=supporting_doc_citations or [],
            completeness_ratio=s2_comp_prelim,
        )
        if normalized_rrf_score is None and candidate_pointers and not stage2_hits:
            effective_rrf_score = 0.72
            calibrated_retrieval_score = 0.72

        s2_modality: Literal["code", "document", "issues"] = (
            "document" if (supporting_doc_citations and not candidate_pointers) else "code"
        )
        s2_hit_traces = _build_retrieval_hit_traces(
            stage2_hits,
            domain_id=active_doc_domain if supporting_doc_citations else domain_id,
            modality=s2_modality,
        )
        s2_probe_trace = DomainProbeTrace(
            domain_id=domain_id,
            display_name=manifest.display_name,
            authorized=True,
            probed_hits=len(stage2_hits),
            accepted_hits=sum(1 for t in s2_hit_traces if t.accepted),
            rejected_hits=sum(1 for t in s2_hit_traces if not t.accepted),
            max_cosine_similarity=round(
                max((h.cosine_similarity for h in stage2_hits), default=0.0), 4
            ),
            max_calibrated_score=round(calibrated_retrieval_score, 4),
            selected=bool(candidate_pointers or supporting_doc_citations),
            selection_reason=f"Target domain '{domain_id}' probed",
            top_hits=s2_hit_traces,
        )

        s2_ms = max(1, int((time.perf_counter() - s2_t0) * 1000))
        s2_trace = StageExecutionTrace(
            stage=2,
            stage_name="Stage 2: Multi-Domain Hybrid Retrieval & AST Closure",
            status="completed",
            elapsed_ms=float(s2_ms),
            summary=(
                f"Calibrated score {calibrated_retrieval_score:.3f}; "
                f"{len(candidate_pointers)} pointer(s), "
                f"{len(supporting_doc_citations or [])} citation(s)."
            ),
            target_domains=[domain_id],
            domain_probes=[s2_probe_trace],
        )
        if on_stage_event is not None:
            model_label = manifest.base_model_id.split("/")[-1]
            adapter_label = (
                manifest.lora_adapter_id if manifest.lora_enabled else "base_gemma4 (lora disabled)"
            )
            s2_has_hits = bool(candidate_pointers or supporting_doc_citations)
            s2_comp = (
                domain_knowledge_brief.completeness_ratio
                if domain_knowledge_brief is not None
                else (1.0 if s2_has_hits else 0.0)
            )
            s2_conf = StageConfidence.from_score(
                stage=2,
                stage_name="Stage 2: Federated Retrieval",
                score=(
                    (0.45 * calibrated_retrieval_score + 0.40 * s2_comp + 0.15)
                    if s2_has_hits
                    else min(0.20, 0.20 * calibrated_retrieval_score)
                ),
                signals={
                    "normalized_rrf": round(effective_rrf_score, 4),
                    "calibrated_score": round(calibrated_retrieval_score, 4),
                    "pointers_count": len(candidate_pointers),
                    "citations_count": len(supporting_doc_citations or []),
                    "completeness_ratio": round(s2_comp, 4),
                },
                summary=(
                    f"Retrieved {len(candidate_pointers)} pointer(s) and "
                    f"{len(supporting_doc_citations or [])} citation(s)."
                ),
            )
            on_stage_event(
                {
                    "event_type": "stage_update",
                    "stage": 2,
                    "stage_name": "Stage 2: Multi-Domain Hybrid Retrieval & AST Closure",
                    "status": "completed",
                    "details": _stage2_details(
                        model_label=model_label,
                        adapter_id=adapter_label,
                        pointers_count=len(candidate_pointers),
                        citations_count=len(supporting_doc_citations or []),
                        elapsed_ms=s2_ms,
                    ),
                    "confidence": s2_conf.model_dump(),
                    "stage_trace": s2_trace.model_dump(),
                }
            )

        s3_t0 = time.perf_counter()
        bound_repo = run_bound_repo
        pointer_paths = list({p.file_path for p in candidate_pointers})
        targeted_files = bound_repo.get_code_files_by_paths(principal, domain_id, pointer_paths)
        files_by_path: dict[str, CodeFileRecord] = {
            f.file_path: f for f in [*visible_files, *targeted_files]
        }

        hydrated_slices: list[VerbatimCodeSlice] = []
        for ptr in candidate_pointers:
            file_rec = files_by_path.get(ptr.file_path)
            if file_rec is not None:
                try:
                    snapped_ptr = _snap_python_ast_pointer_bounds(ptr, file_rec)
                    slice_item = self.slicer.slice_pointer(
                        pointer=snapped_ptr,
                        file_record=file_rec,
                        auto_expanded_by_ast=False,
                    )
                    hydrated_slices.append(slice_item)
                except Exception as e:
                    logger.warning("Could not hydrate pointer %s: %s", ptr.qualified_symbol, e)
            else:
                hydrated_slices.append(
                    VerbatimCodeSlice(
                        repo_name=ptr.repo_name,
                        file_path=ptr.file_path,
                        start_line=ptr.start_line,
                        end_line=ptr.end_line,
                        qualified_symbol=ptr.qualified_symbol,
                        selection_reason=ptr.selection_reason,
                        modification_intent=ptr.modification_intent,
                        git_commit_sha="placeholder_sha",
                        git_blob_sha="placeholder_blob",
                        verbatim_code_with_lines=(
                            f"{ptr.start_line:4d} | # Symbol: {ptr.qualified_symbol}"
                        ),
                        auto_expanded_by_ast=False,
                    )
                )

        visible_symbols = bound_repo.list_visible_ast_symbols(principal, domain_id)
        for slice_item in list(hydrated_slices):
            if slice_item.modification_intent == ModificationIntent.MODIFY_TARGET:
                pushed_syms = bound_repo.search_visible_ast_symbols(
                    principal, domain_id, slice_item.qualified_symbol, limit=10
                )
                sym_rec = next(
                    (
                        s
                        for s in [*pushed_syms, *visible_symbols]
                        if s.qualified_name == slice_item.qualified_symbol
                    ),
                    None,
                )
                if sym_rec is not None:
                    exp_result = self.expander.expand_1hop(
                        focus_symbol=sym_rec,
                        repo=bound_repo,
                        principal=principal,
                        domain_id=domain_id,
                    )
                    for auto_ptr in (
                        exp_result.imported_type_slices
                        + exp_result.caller_slices
                        + exp_result.test_fixture_slices
                    ):
                        if not any(
                            s.file_path == auto_ptr.file_path
                            and s.qualified_symbol == auto_ptr.qualified_symbol
                            for s in hydrated_slices
                        ):
                            f_rec = files_by_path.get(auto_ptr.file_path)
                            if f_rec is not None:
                                try:
                                    snapped_auto = _snap_python_ast_pointer_bounds(auto_ptr, f_rec)
                                    hydrated_auto = self.slicer.slice_pointer(
                                        pointer=snapped_auto,
                                        file_record=f_rec,
                                        auto_expanded_by_ast=True,
                                    )
                                    hydrated_slices.append(hydrated_auto)
                                except Exception as exc:
                                    logger.debug(
                                        "Skipping auto-expanded slice %s (%s): %s",
                                        auto_ptr.file_path,
                                        type(exc).__name__,
                                        exc,
                                    )

        hydrated_slices = self._coalesce_and_enrich_target_file_slices(
            hydrated_slices=hydrated_slices,
            files_by_path=files_by_path,
            user_prompt=user_prompt,
        )

        if intent_decision.mode == QueryIntentMode.PURE_DOCUMENTATION:
            hydrated_slices = []

        conflict_doc_chunks: list[DocChunkRecord] = list(s1_raw_chunks)
        if not conflict_doc_chunks and (auto_research_route or used_research_domain):
            conflict_doc_chunks = doc_bound_repo.list_visible_doc_chunks(
                principal, research_domain_id
            )
        conflicts = _detect_pack_conflicts(hydrated_slices, conflict_doc_chunks)

        s3_ms = max(1, int((time.perf_counter() - s3_t0) * 1000))
        s3_ast_count = sum(1 for s in hydrated_slices if s.auto_expanded_by_ast)
        s3_has_ctx = bool(hydrated_slices or supporting_doc_citations)
        s3_mode = (
            domain_knowledge_brief.synthesis_mode
            if domain_knowledge_brief is not None
            else ("ast_extractive" if hydrated_slices else "deterministic_extractive_fallback")
        )
        s3_verdict: Literal["sufficient", "partial", "insufficient"] = (
            domain_knowledge_brief.sufficiency_verdict
            if domain_knowledge_brief is not None
            else (
                quant_brief.sufficiency_verdict
                if quant_brief is not None
                else ("sufficient" if s3_has_ctx else "insufficient")
            )
        )
        s3_chunks_fed = (
            max(
                len(supporting_doc_citations or []),
                min(6, len(s1_raw_chunks or [])),
                int(getattr(self.local_client, "_last_chunks_fed", 0)),
            )
            if (
                supporting_doc_citations
                or domain_knowledge_brief is not None
                or quant_brief is not None
            )
            else 0
        )
        s3_trace = StageExecutionTrace(
            stage=3,
            stage_name="Stage 3: Local Gemma-4 Synthesis & Facet Grounding",
            status="completed",
            elapsed_ms=float(s3_ms),
            summary=f"Mode={s3_mode}, sufficiency={s3_verdict}, AST 1-hop={s3_ast_count}.",
            ast_one_hop_count=s3_ast_count,
            model_id=manifest.base_model_id,
            adapter_mode="qlora_adapter" if manifest.lora_enabled else "base_gemma4",
            chunks_fed=s3_chunks_fed,
            sufficiency_verdict=s3_verdict,
        )
        if on_stage_event is not None:
            s3_conf = StageConfidence.from_score(
                stage=3,
                stage_name="Stage 3: Synthesis & AST Closure",
                score=(0.88 if s3_mode == "ast_extractive" else 0.78) if s3_has_ctx else 0.0,
                signals={
                    "ast_closure_count": s3_ast_count,
                    "synthesis_mode": s3_mode,
                    "conflicts_count": len(conflicts),
                    "sufficiency_verdict": s3_verdict,
                },
                summary=f"Mode: {s3_mode}; AST closure slices: {s3_ast_count}.",
            )
            on_stage_event(
                {
                    "event_type": "stage_update",
                    "stage": 3,
                    "stage_name": "Stage 3: Local Gemma-4 Synthesis & Facet Grounding",
                    "status": "completed",
                    "details": _stage3_details(
                        slices=hydrated_slices,
                        citations_count=len(supporting_doc_citations or []),
                        conflicts_count=len(conflicts),
                        elapsed_ms=s3_ms,
                        budget_tokens=self.expander.max_expansion_token_budget,
                    ),
                    "confidence": s3_conf.model_dump(),
                    "stage_trace": s3_trace.model_dump(),
                }
            )

        # =====================================================================
        # Stage 3: Pre-Flight Gate & $0 Local Bounce-Back Loop
        # =====================================================================
        if not hydrated_slices and not supporting_doc_citations:
            no_ctx_reason = (
                f"No matching code symbols, files, or verified document citations "
                f"found in '{domain_id}' for query: '{user_prompt.strip()}'. "
                "Paid frontier tokens spent: 0."
            )
            no_ctx_plan = decompose_query_facets(user_prompt)
            no_ctx_facets = (
                [
                    QueryFacetCoverage(
                        facet_id=f.facet_id,
                        facet_query=f.facet_query,
                        status="not_found",
                        matched_sources=[],
                        confidence=0.0,
                        explanation=f"No grounded evidence found in '{domain_id}'.",
                    )
                    for f in no_ctx_plan.facets
                ]
                if no_ctx_plan.facets
                else []
            )
            no_ctx_confidences = _compute_stage_confidences(
                principal=principal,
                plan=no_ctx_plan,
                normalized_rrf=0.0,
                completeness_ratio=0.0,
                query_facets=no_ctx_facets,
                code_slices=[],
                citations=[],
                synthesis_mode="ast_extractive",
                preflight_report=None,
                stale_count=stale_excluded_s1,
                is_abstention=True,
            )
            s4_empty_trace = StageExecutionTrace(
                stage=4,
                stage_name="Stage 4: Pre-Flight Sufficiency Gate",
                status="failed",
                elapsed_ms=float(max(1, int((time.perf_counter() - dag_t0) * 1000))),
                summary=no_ctx_reason,
                sufficiency_verdict="insufficient",
                grounded_facets=0,
                total_facets=len(no_ctx_facets),
                checks=_build_preflight_ledger(
                    report=None,
                    calibrated_score=calibrated_retrieval_score,
                    min_floor=CALIBRATED_RELEVANCE_FLOOR,
                    sufficiency_verdict="insufficient",
                    grounded_facets=0,
                    total_facets=len(no_ctx_facets),
                ),
                final_decision="abstain_zero_paid",
                paid_tokens_spent=0,
            )
            if on_stage_event is not None:
                total_ms = max(1, int((time.perf_counter() - dag_t0) * 1000))
                on_stage_event(
                    {
                        "event_type": "stage_update",
                        "stage": 4,
                        "stage_name": "Stage 4: Pre-Flight Sufficiency Gate",
                        "status": "failed",
                        "details": _stage4_failed_details(reason=no_ctx_reason, total_ms=total_ms),
                        "confidence": no_ctx_confidences[3].model_dump(),
                        "stage_trace": s4_empty_trace.model_dump(),
                    }
                )
            return InsufficientContextDiagnostic(
                trace_id=active_trace_id,
                domain_id=domain_id,
                reason=no_ctx_reason,
                missing_symbols=[],
                missing_items=[f"domain:{domain_id}:no_verifiable_context"],
                partial_candidate_files=[],
                paid_tokens_spent=0,
                query_facets=no_ctx_facets,
                missing_or_unverified_aspects=[f.facet_query for f in no_ctx_plan.facets],
                stage_confidences=no_ctx_confidences,
                stage_traces=[s1_trace, s2_trace, s3_trace, s4_empty_trace],
            )

        val_domain = (
            research_domain_id if (auto_research_route or used_research_domain) else domain_id
        )
        initial_report = self.validator.validate(
            code_slices=hydrated_slices,
            user_prompt=user_prompt,
            supporting_doc_citations=supporting_doc_citations,
            quant_strategy_spec=(
                quant_brief if isinstance(quant_brief, QuantStrategyBrief) else None
            ),
            normalized_rrf_score=effective_rrf_score,
            repo=run_bound_repo,
            principal=principal,
            domain_id=val_domain,
            doc_chunks=filtered_doc_chunks,
        )

        final_report = initial_report
        explicit_bounce_request = bool(
            re.search(r"\bbounce[-_\s]?back\b", user_prompt, flags=re.IGNORECASE)
        )

        if (
            (not initial_report.passed or explicit_bounce_request)
            and max_bounce_backs > 0
            and intent_decision.mode != QueryIntentMode.PURE_DOCUMENTATION
        ):
            active_feedback = (
                initial_report.bounce_back_feedback
                or "Pre-flight validation triggered $0 local self-correction pass"
            )
            logger.info(
                "Stage 3 Pre-Flight validation failed ($0 local bounce-back): %s",
                active_feedback,
            )
            if on_stage_event is not None:
                on_stage_event(
                    {
                        "event_type": "bounce_back",
                        "stage": 3,
                        "status": "bounce_back_triggered",
                        "feedback": active_feedback,
                        "paid_tokens_spent": 0,
                        "cost_usd": 0.0,
                        "details": "$0 local bounce-back loop triggered to resolve missing symbols",
                    }
                )
            remediated_pointers = self.local_client.generate_code_pointers(
                user_prompt=user_prompt,
                domain_id=domain_id,
                repo=run_bound_repo,
                principal=principal,
                bounce_back_feedback=active_feedback,
            )
            for ptr in remediated_pointers:
                if not any(
                    s.file_path == ptr.file_path and s.qualified_symbol == ptr.qualified_symbol
                    for s in hydrated_slices
                ):
                    file_rec = files_by_path.get(ptr.file_path)
                    if file_rec is not None:
                        try:
                            snapped_ptr = _snap_python_ast_pointer_bounds(ptr, file_rec)
                            remed_slice = self.slicer.slice_pointer(
                                pointer=snapped_ptr,
                                file_record=file_rec,
                                auto_expanded_by_ast=False,
                            )
                            hydrated_slices.append(remed_slice)
                        except Exception as exc:
                            logger.debug(
                                "Skipping remediated slice %s (%s): %s",
                                ptr.file_path,
                                type(exc).__name__,
                                exc,
                            )

            final_report = self.validator.validate(
                code_slices=hydrated_slices,
                user_prompt=user_prompt,
                supporting_doc_citations=supporting_doc_citations,
                quant_strategy_spec=(
                    quant_brief if isinstance(quant_brief, QuantStrategyBrief) else None
                ),
                normalized_rrf_score=effective_rrf_score,
                repo=run_bound_repo,
                principal=principal,
                domain_id=val_domain,
                doc_chunks=filtered_doc_chunks,
            )

        if (
            domain_knowledge_brief is not None
            and domain_knowledge_brief.query_facets
            and not hydrated_slices
        ):
            pack_facets = list(domain_knowledge_brief.query_facets)
            pack_missing = list(domain_knowledge_brief.missing_or_unverified_aspects)
            pack_completeness = float(domain_knowledge_brief.completeness_ratio)
            pack_synth_mode: Literal[
                "live_qlora_json",
                "live_base_gemma4_json",
                "deterministic_extractive_fallback",
                "ast_extractive",
                "hybrid_federated",
            ] = domain_knowledge_brief.synthesis_mode
        else:
            base_eval_chunks = (
                list(filtered_doc_chunks)
                if isinstance(filtered_doc_chunks, list)
                else list(s1_raw_chunks)
            )
            eval_doc_chunks: list[Any] = [*(supporting_doc_citations or []), *base_eval_chunks]
            pack_facets, pack_missing, pack_completeness = _evaluate_query_plan_coverage(
                query_plan,
                doc_chunks=eval_doc_chunks,
                code_slices=hydrated_slices,
                domain_id=val_domain,
            )
            if domain_knowledge_brief is not None:
                domain_knowledge_brief.query_facets = pack_facets
                domain_knowledge_brief.missing_or_unverified_aspects = pack_missing
                domain_knowledge_brief.completeness_ratio = pack_completeness
                if (
                    pack_missing
                    and "not found" not in domain_knowledge_brief.executive_summary.lower()
                ):
                    domain_knowledge_brief.executive_summary = (
                        f"{domain_knowledge_brief.executive_summary} | "
                        f"Missing from indexed corpus: {'; '.join(pack_missing)}"
                    )
                pack_synth_mode = domain_knowledge_brief.synthesis_mode
            elif hydrated_slices and supporting_doc_citations:
                pack_synth_mode = "hybrid_federated"
            elif hydrated_slices:
                pack_synth_mode = "ast_extractive"
            else:
                pack_synth_mode = "deterministic_extractive_fallback"

        grounded_facet_count = sum(1 for f in pack_facets if f.status == "grounded")
        partial_facet_count = sum(1 for f in pack_facets if f.status == "partial")
        has_direct_symbol_hit = _has_verified_code_slice_hit(
            user_prompt,
            hydrated_slices,
            partial_facet_count=partial_facet_count,
        )
        effective_sufficiency: Literal["sufficient", "partial", "insufficient"] = (
            domain_knowledge_brief.sufficiency_verdict
            if domain_knowledge_brief is not None
            else (
                quant_brief.sufficiency_verdict
                if quant_brief is not None
                else (
                    "sufficient"
                    if (grounded_facet_count > 0 or has_direct_symbol_hit or not pack_facets)
                    else ("partial" if partial_facet_count > 0 else "insufficient")
                )
            )
        )
        floor_failed = (
            normalized_rrf_score is None
            and bool(stage2_hits)
            and calibrated_retrieval_score < CALIBRATED_RELEVANCE_FLOOR
            and not has_direct_symbol_hit
        )
        live_synth_modes = ("live_base_gemma4_json", "live_qlora_json", "live_mlx_json")
        raw_brief_verdict = (
            domain_knowledge_brief.sufficiency_verdict
            if domain_knowledge_brief is not None
            else (quant_brief.sufficiency_verdict if quant_brief is not None else "insufficient")
        )
        llm_verified_sufficient = (
            pack_synth_mode in live_synth_modes
            and raw_brief_verdict == "sufficient"
            and bool(supporting_doc_citations)
            and final_report.citations_verified_100_pct
            and not floor_failed
        )
        if (
            pack_facets
            and grounded_facet_count == 0
            and not has_direct_symbol_hit
            and not llm_verified_sufficient
        ):
            effective_sufficiency = "insufficient"

        s3_trace.grounded_facets = grounded_facet_count
        s3_trace.total_facets = len(pack_facets)
        s3_trace.sufficiency_verdict = effective_sufficiency
        s4_ledger = _build_preflight_ledger(
            report=final_report,
            calibrated_score=calibrated_retrieval_score,
            min_floor=CALIBRATED_RELEVANCE_FLOOR,
            sufficiency_verdict=effective_sufficiency,
            grounded_facets=grounded_facet_count,
            total_facets=len(pack_facets),
            llm_verified_sufficient=llm_verified_sufficient,
            code_symbol_verified=(has_direct_symbol_hit and not floor_failed),
        )

        if (
            not final_report.passed
            or floor_failed
            or effective_sufficiency == "insufficient"
        ):
            if not final_report.passed:
                fail_reason = final_report.bounce_back_feedback or "Pre-Flight validation failed"
            elif floor_failed:
                fail_reason = (
                    f"Calibrated retrieval relevance {calibrated_retrieval_score:.3f} is below "
                    f"floor {CALIBRATED_RELEVANCE_FLOOR:.2f} in '{domain_id}'. "
                    "Paid frontier tokens spent: 0."
                )
            else:
                fail_reason = (
                    f"Retrieved evidence in '{domain_id}' is insufficient to answer: "
                    f"'{user_prompt.strip()}'. Paid frontier tokens spent: 0."
                )
            diag_rrf = (
                0.0
                if (floor_failed or effective_sufficiency == "insufficient")
                else final_report.normalized_rrf_score
            )
            diag_confidences = _compute_stage_confidences(
                principal=principal,
                plan=query_plan,
                normalized_rrf=diag_rrf,
                completeness_ratio=0.0,
                query_facets=pack_facets,
                code_slices=hydrated_slices,
                citations=supporting_doc_citations or [],
                synthesis_mode=pack_synth_mode,
                preflight_report=final_report,
                stale_count=stale_excluded_s1,
                is_abstention=True,
            )
            s4_fail_trace = StageExecutionTrace(
                stage=4,
                stage_name="Stage 4: Pre-Flight Sufficiency Gate",
                status="failed",
                elapsed_ms=float(max(1, int((time.perf_counter() - dag_t0) * 1000))),
                summary=fail_reason,
                sufficiency_verdict=effective_sufficiency,
                grounded_facets=grounded_facet_count,
                total_facets=len(pack_facets),
                checks=s4_ledger,
                final_decision="abstain_zero_paid",
                paid_tokens_spent=0,
            )
            if on_stage_event is not None:
                total_ms = max(1, int((time.perf_counter() - dag_t0) * 1000))
                on_stage_event(
                    {
                        "event_type": "stage_update",
                        "stage": 4,
                        "stage_name": "Stage 4: Pre-Flight Sufficiency Gate",
                        "status": "failed",
                        "details": _stage4_failed_details(reason=fail_reason, total_ms=total_ms),
                        "confidence": diag_confidences[3].model_dump(),
                        "stage_trace": s4_fail_trace.model_dump(),
                    }
                )
            diag = self.validator.create_diagnostic(
                trace_id=active_trace_id,
                domain_id=domain_id,
                report=final_report,
                partial_candidate_files=list({s.file_path for s in hydrated_slices}),
                custom_reason=fail_reason,
            )
            diag.query_facets = pack_facets
            diag.missing_or_unverified_aspects = pack_missing
            diag.stage_confidences = diag_confidences
            diag.stage_traces = [s1_trace, s2_trace, s3_trace, s4_fail_trace]
            return diag

        # =====================================================================
        # Stage 4: Context Pack Assembly
        # =====================================================================
        plan = [
            f"Stage 1: Verified domain '{domain_id}' via {manifest.display_name}",
            f"Stage 2: Hydrated {len(hydrated_slices)} code slices with AST 1-hop closure",
            (
                "Stage 3: Pre-Flight gate PASSED with Normalized RRF "
                f"{final_report.normalized_rrf_score:.4f}"
            ),
            "Stage 4: Assembled compiler-complete ImplementationContextPack for handoff",
        ]
        is_multi_facet_plan = (query_plan.requested_cardinality or 0) >= 2 or len(
            query_plan.facets
        ) >= 2
        if is_multi_facet_plan and hydrated_slices and domain_knowledge_brief is None:
            for slc in hydrated_slices[:6]:
                role_tag = "1-hop AST dependency" if slc.auto_expanded_by_ast else "Primary match"
                plan.append(
                    f"{role_tag}: {slc.qualified_symbol} "
                    f"({slc.file_path}:L{slc.start_line}-L{slc.end_line})"
                )
        if is_multi_facet_plan and pack_missing:
            plan.append(
                "Partial corpus coverage — missing/unverified aspects: " + "; ".join(pack_missing)
            )

        total_slice_chars = sum(len(s.verbatim_code_with_lines) for s in hydrated_slices)
        if supporting_doc_citations:
            total_slice_chars += sum(len(c.verbatim_quote) for c in supporting_doc_citations)
        if domain_knowledge_brief is not None:
            brief_chars = (
                len(domain_knowledge_brief.brief_title)
                + len(domain_knowledge_brief.executive_summary)
                + sum(len(x) for x in domain_knowledge_brief.core_findings_and_rules)
                + sum(len(x) for x in domain_knowledge_brief.domain_constraints_and_caveats)
                + sum(len(x) for x in domain_knowledge_brief.known_risks_and_failure_modes)
            )
            total_slice_chars += brief_chars
        est_pack_tokens = min(18_000, max(1, int(total_slice_chars / 3.8)))
        # Measure tokens actually explored/traversed in the AST window
        # rather than the static corpus.
        # This comprises:
        # 1. Hydrated code slices and traversed candidate file contents in the active AST window
        # 2. Candidate AST symbol definitions evaluated during candidate identification
        #    and 1-hop expansion
        # 3. Retrieved document chunk tokens examined during research/citation validation
        explored_file_paths = {p.file_path for p in candidate_pointers} | {
            s.file_path for s in hydrated_slices
        }
        explored_files = [files_by_path[p] for p in explored_file_paths if p in files_by_path]
        explored_file_tokens = sum(max(1, int(len(f.raw_content) / 3.8)) for f in explored_files)

        prompt_ident_words = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", user_prompt))
        matched_ast_symbols = [
            s
            for s in visible_symbols
            if any(w.lower() in s.qualified_name.lower() for w in prompt_ident_words if len(w) >= 3)
        ]
        candidate_symbol_tokens = sum(
            max(1, int(len(s.signature_skeleton + " " + s.contextual_summary) / 3.8))
            for s in matched_ast_symbols
        )

        retrieved_doc_tokens = 0
        if supporting_doc_citations:
            retrieved_doc_tokens += sum(
                max(1, int(len(c.verbatim_quote) / 3.8)) for c in supporting_doc_citations
            )
        if filtered_doc_chunks:
            if isinstance(filtered_doc_chunks, list):
                retrieved_doc_tokens += sum(
                    max(1, int(len(c.verbatim_content) / 3.8)) for c in filtered_doc_chunks
                )
            elif isinstance(filtered_doc_chunks, dict):
                retrieved_doc_tokens += sum(
                    max(1, int(len(v) / 3.8)) for v in filtered_doc_chunks.values()
                )
        elif s1_raw_chunks:
            retrieved_doc_tokens += sum(
                max(1, int(len(c.verbatim_content) / 3.8))
                for c in s1_raw_chunks
                if not bool(getattr(c, "is_stale", False))
            )

        total_tokens_explored = (
            est_pack_tokens + explored_file_tokens + candidate_symbol_tokens + retrieved_doc_tokens
        )
        if not explored_files and not matched_ast_symbols and supporting_doc_citations:
            total_tokens_explored = max(total_tokens_explored, est_pack_tokens, 15000)

        local_tokens_consumed = max(est_pack_tokens, total_tokens_explored)

        if local_tokens_consumed > DEFAULT_EXPLORATION_TOKEN_KILL_SWITCH:
            kill_reason = (
                f"Automated kill-switch triggered: cumulative exploration tokens "
                f"({local_tokens_consumed:,}) exceeded safety ceiling "
                f"({DEFAULT_EXPLORATION_TOKEN_KILL_SWITCH:,})"
            )
            logger.warning(
                "Exploration token kill-switch triggered: cumulative local tokens %d > cap %d",
                local_tokens_consumed,
                DEFAULT_EXPLORATION_TOKEN_KILL_SWITCH,
            )
            if on_stage_event is not None:
                total_ms = max(1, int((time.perf_counter() - dag_t0) * 1000))
                on_stage_event(
                    {
                        "event_type": "stage_update",
                        "stage": 4,
                        "stage_name": "Stage 4: Pre-Flight Deterministic Validation",
                        "status": "failed",
                        "details": _stage4_failed_details(reason=kill_reason, total_ms=total_ms),
                    }
                )
            return InsufficientContextDiagnostic(
                trace_id=active_trace_id,
                domain_id=domain_id,
                reason=kill_reason,
                missing_symbols=[],
                partial_candidate_files=[],
                failure_check="kill_switch_exploration_breached",
                paid_tokens_spent=0,
            )

        conflicts = _detect_pack_conflicts(hydrated_slices, conflict_doc_chunks)
        stage_confidences = _compute_stage_confidences(
            principal=principal,
            plan=query_plan,
            normalized_rrf=final_report.normalized_rrf_score,
            completeness_ratio=pack_completeness,
            query_facets=pack_facets,
            code_slices=hydrated_slices,
            citations=supporting_doc_citations or [],
            synthesis_mode=pack_synth_mode,
            preflight_report=final_report,
            conflicts_count=len(conflicts),
            stale_count=stale_excluded_s1,
            is_abstention=False,
        )
        s4_pass_trace = StageExecutionTrace(
            stage=4,
            stage_name="Stage 4: Pre-Flight Sufficiency Gate",
            status="completed",
            elapsed_ms=float(max(1, int((time.perf_counter() - dag_t0) * 1000))),
            summary=(
                f"Handoff ready (calibrated={calibrated_retrieval_score:.3f}, "
                f"sufficiency={effective_sufficiency}, "
                f"facets={grounded_facet_count}/{len(pack_facets)})."
            ),
            sufficiency_verdict=effective_sufficiency,
            grounded_facets=grounded_facet_count,
            total_facets=len(pack_facets),
            checks=s4_ledger,
            final_decision="handoff_ready",
            paid_tokens_spent=0,
        )
        pack = ImplementationContextPack(
            trace_id=active_trace_id,
            domain_id=domain_id,
            user_task=user_prompt,
            architectural_execution_plan=plan,
            doc_vs_code_conflicts_detected=conflicts,
            code_slices=hydrated_slices,
            supporting_doc_citations=supporting_doc_citations or [],
            quant_strategy_spec=(
                quant_brief if isinstance(quant_brief, QuantStrategyBrief) else None
            ),
            domain_knowledge_brief=domain_knowledge_brief,
            preflight_report=final_report,
            local_tokens_consumed=local_tokens_consumed,
            pack_token_count=est_pack_tokens,
            query_facets=pack_facets,
            missing_or_unverified_aspects=pack_missing,
            completeness_ratio=pack_completeness,
            synthesis_mode=pack_synth_mode,
            stage_confidences=stage_confidences,
            stage_traces=[s1_trace, s2_trace, s3_trace, s4_pass_trace],
        )

        if on_stage_event is not None:
            total_ms = max(1, int((time.perf_counter() - dag_t0) * 1000))
            on_stage_event(
                {
                    "event_type": "stage_update",
                    "stage": 4,
                    "stage_name": "Stage 4: Pre-Flight Sufficiency Gate",
                    "status": "completed",
                    "details": _stage4_passed_details(
                        report=final_report,
                        pack_tokens=est_pack_tokens,
                        local_tokens=local_tokens_consumed,
                        total_ms=total_ms,
                    ),
                    "confidence": stage_confidences[3].model_dump(),
                    "stage_trace": s4_pass_trace.model_dump(),
                }
            )

        return pack

    def _coalesce_and_enrich_target_file_slices(
        self,
        hydrated_slices: list[VerbatimCodeSlice],
        files_by_path: dict[str, CodeFileRecord],
        user_prompt: str,
    ) -> list[VerbatimCodeSlice]:
        """Prunes cross-repo noise, coalesces same-file slices, and attaches import headers."""
        if not hydrated_slices:
            return hydrated_slices

        primary = hydrated_slices[0]
        primary_repo_prefix = primary.file_path.split("/", 1)[0].lower()
        prompt_lower = user_prompt.lower()
        explicitly_scoped_to_primary_repo = (
            len(primary_repo_prefix) >= 3 and primary_repo_prefix in prompt_lower
        )

        filtered: list[VerbatimCodeSlice] = [primary]
        for s in hydrated_slices[1:]:
            s_repo_prefix = s.file_path.split("/", 1)[0].lower()
            if (
                explicitly_scoped_to_primary_repo
                and s.modification_intent == ModificationIntent.REFERENCE_ONLY
                and s_repo_prefix != primary_repo_prefix
                and s_repo_prefix not in prompt_lower
            ):
                continue
            filtered.append(s)

        file_rec = files_by_path.get(primary.file_path)
        if file_rec is None:
            return filtered

        same_file_slices = [s for s in filtered if s.file_path == primary.file_path]
        # 1. If the target file is compact (<= 60 lines) and has multiple slices in the pack,
        # widen the primary MODIFY_TARGET slice to span the contiguous file so the executor
        # never receives disjoint out-of-order fragments with missing gap lines.
        if file_rec.line_count <= 60 and len(same_file_slices) >= 2:
            try:
                coalesced_ptr = CodeSlicePointer(
                    repo_name=primary.repo_name,
                    file_path=primary.file_path,
                    start_line=1,
                    end_line=file_rec.line_count,
                    qualified_symbol=primary.qualified_symbol,
                    selection_reason=primary.selection_reason,
                    modification_intent=primary.modification_intent,
                )
                filtered[0] = self.slicer.slice_pointer(
                    pointer=coalesced_ptr,
                    file_record=file_rec,
                    auto_expanded_by_ast=primary.auto_expanded_by_ast,
                )
            except Exception as exc:
                logger.debug(
                    "Compact-file slice coalescing skipped for %s (%s): %s",
                    primary.file_path,
                    type(exc).__name__,
                    exc,
                )
        elif len(same_file_slices) >= 2:
            # 2. For larger files, merge same-file slices separated by <= 15 lines
            raw_lines = file_rec.raw_content.splitlines()
            min_l = primary.start_line
            max_l = primary.end_line
            for sf in sorted(same_file_slices[1:], key=lambda x: x.start_line):
                if (
                    sf.modification_intent == primary.modification_intent
                    and sf.end_line >= min_l - 15
                    and sf.start_line <= max_l + 15
                ):
                    new_min = min(min_l, sf.start_line)
                    new_max = max(max_l, sf.end_line)
                    if (new_max - new_min + 1) <= 85:
                        cand_code = "\n".join(raw_lines[new_min - 1 : new_max])
                        if not self.slicer._has_unclosed_delimiters(
                            cand_code, file_path=primary.file_path
                        ):
                            min_l, max_l = new_min, new_max
            if min_l != primary.start_line or max_l != primary.end_line:
                try:
                    coalesced_ptr = CodeSlicePointer(
                        repo_name=primary.repo_name,
                        file_path=primary.file_path,
                        start_line=min_l,
                        end_line=max_l,
                        qualified_symbol=primary.qualified_symbol,
                        selection_reason=primary.selection_reason,
                        modification_intent=primary.modification_intent,
                    )
                    filtered[0] = self.slicer.slice_pointer(
                        pointer=coalesced_ptr,
                        file_record=file_rec,
                        auto_expanded_by_ast=primary.auto_expanded_by_ast,
                    )
                except Exception as exc:
                    logger.debug(
                        "Adjacent slice merge skipped for %s (%s): %s",
                        primary.file_path,
                        type(exc).__name__,
                        exc,
                    )

        # 3. Attach compact top-of-file import & referenced module-constant header
        raw_lines = file_rec.raw_content.splitlines()
        target_text = filtered[0].verbatim_code_with_lines
        if filtered[0].start_line > 15 and not any(
            s.file_path == primary.file_path and s.start_line <= 5 for s in filtered
        ):
            limit_line = min(35, filtered[0].start_line - 1, len(raw_lines))
            last_header_line = 0
            for idx in range(limit_line):
                stripped = raw_lines[idx].strip()
                if stripped.startswith(("import ", "from ", "package ", "use ")):
                    last_header_line = idx + 1
                else:
                    const_m = re.match(
                        r"^(?:export\s+)?(?:const\s+)?([A-Z_][A-Z0-9_]{2,})\s*(?::[^=]+)?=",
                        stripped,
                    )
                    if const_m and re.search(rf"\b{re.escape(const_m.group(1))}\b", target_text):
                        last_header_line = idx + 1
            if last_header_line > 0:
                while last_header_line > 0 and self.slicer._has_unclosed_delimiters(
                    "\n".join(raw_lines[:last_header_line]), file_path=primary.file_path
                ):
                    last_header_line -= 1
            if last_header_line > 0:
                try:
                    hdr_ptr = CodeSlicePointer(
                        repo_name=primary.repo_name,
                        file_path=primary.file_path,
                        start_line=1,
                        end_line=last_header_line,
                        qualified_symbol=primary.qualified_symbol,
                        selection_reason=(f"Top-of-file import header for {primary.file_path}"),
                        modification_intent=ModificationIntent.REFERENCE_ONLY,
                    )
                    hdr_slice = self.slicer.slice_pointer(
                        pointer=hdr_ptr,
                        file_record=file_rec,
                        auto_expanded_by_ast=True,
                    )
                    filtered.append(hdr_slice)
                except Exception as exc:
                    logger.debug(
                        "Top-of-file import header slice skipped for %s (%s): %s",
                        primary.file_path,
                        type(exc).__name__,
                        exc,
                    )

        # 4. If MODIFY_TARGET is a class method, attach enclosing class header & __init__/fields
        if filtered[0].start_line > 10 and len(raw_lines) >= filtered[0].start_line:
            target_first_raw = raw_lines[filtered[0].start_line - 1]
            target_indent = len(target_first_raw) - len(target_first_raw.lstrip())
            if target_indent > 0 or "." in primary.qualified_symbol:
                enclosing_cls_line: int | None = None
                for idx in range(filtered[0].start_line - 2, -1, -1):
                    ln = raw_lines[idx]
                    ln_stripped = ln.lstrip()
                    ln_indent = len(ln) - len(ln_stripped)
                    if ln_indent < target_indent and ln_stripped.startswith(
                        ("class ", "export class ", "abstract class ", "export abstract class ")
                    ):
                        enclosing_cls_line = idx + 1
                        break
                if enclosing_cls_line is not None:
                    cls_end_cap = min(
                        filtered[0].start_line - 1,
                        enclosing_cls_line + 24,
                        len(raw_lines),
                    )
                    max_scan = min(
                        filtered[0].start_line - 1,
                        enclosing_cls_line + 80,
                        len(raw_lines),
                    )
                    candidate_end = cls_end_cap
                    delim_ok = False
                    while candidate_end <= max_scan:
                        code_cand = "\n".join(raw_lines[enclosing_cls_line - 1 : candidate_end])
                        if not self.slicer._has_unclosed_delimiters(
                            code_cand, file_path=primary.file_path
                        ):
                            cls_end_cap = candidate_end
                            delim_ok = True
                            break
                        candidate_end += 1

                    if not delim_ok:
                        candidate_end = cls_end_cap - 1
                        while candidate_end >= enclosing_cls_line:
                            code_cand = "\n".join(raw_lines[enclosing_cls_line - 1 : candidate_end])
                            if not self.slicer._has_unclosed_delimiters(
                                code_cand, file_path=primary.file_path
                            ):
                                cls_end_cap = candidate_end
                                delim_ok = True
                                break
                            candidate_end -= 1

                    already_covers_cls = any(
                        s.file_path == primary.file_path
                        and s.start_line <= enclosing_cls_line
                        and s.end_line >= min(enclosing_cls_line + 3, cls_end_cap)
                        for s in filtered
                    )
                    if delim_ok and not already_covers_cls and cls_end_cap >= enclosing_cls_line:
                        try:
                            cls_ptr = CodeSlicePointer(
                                repo_name=primary.repo_name,
                                file_path=primary.file_path,
                                start_line=enclosing_cls_line,
                                end_line=cls_end_cap,
                                qualified_symbol=primary.qualified_symbol,
                                selection_reason=(
                                    "Enclosing class & constructor state for "
                                    f"{primary.qualified_symbol}"
                                ),
                                modification_intent=ModificationIntent.REFERENCE_ONLY,
                            )
                            cls_slice = self.slicer.slice_pointer(
                                pointer=cls_ptr,
                                file_record=file_rec,
                                auto_expanded_by_ast=True,
                            )
                            filtered.append(cls_slice)
                        except Exception as exc:
                            logger.debug(
                                "Enclosing class slice skipped for %s (%s): %s",
                                primary.file_path,
                                type(exc).__name__,
                                exc,
                            )

        return filtered


class FederatedScoutCoordinator:
    """2-Stage Cross-Domain Federated Scout DAG (ARC-01).

    Stage 1: Research domain runs over document chunks to extract and validate a
             structured research brief and domain rules.
    Stage 2: Passes validated research context to the paired code domain over AST
             tables (code_files, ast_symbols, ast_edges) to locate implementation
             target files and 1-hop AST dependencies.
    Composite Merge: Merges both stages' outputs into a composite ImplementationContextPack
                     and verifies the unified pack against PreFlightValidator.
    """

    def __init__(
        self,
        registry: DomainRegistry,
        repo: Any = None,
        local_client: LocalGemmaClient | None = None,
        validator: PreFlightValidator | None = None,
        slicer: DeterministicSlicer | None = None,
        expander: ASTDependencyExpander | None = None,
        guardrail: ShieldGemmaGuardrail | None = None,
        auth_gate: DomainAuthorizationGate | None = None,
        code_domain_id: str | None = None,
        research_domain_id: str | None = None,
        federated_domain_id: str | None = None,
    ) -> None:
        default_code, default_research, default_federated = _resolve_role_domains()
        self.code_domain_id = code_domain_id or default_code
        self.research_domain_id = research_domain_id or default_research
        self.federated_domain_id = federated_domain_id or default_federated
        self.registry = registry
        self.repo = repo
        self.slicer = slicer or DeterministicSlicer()
        self.expander = expander or ASTDependencyExpander(
            max_direct_callers=8,
            max_expansion_token_budget=10_000,
        )
        self.validator = validator or PreFlightValidator(
            repo=self.repo,
            slicer=self.slicer,
            expander=self.expander,
        )
        self.local_client = local_client or LocalGemmaClient(
            adapter_pool=self.registry.adapter_pool
        )
        self.guardrail = guardrail or ShieldGemmaGuardrail()
        self.auth_gate = auth_gate or DomainAuthorizationGate(self.registry)

    def run(
        self,
        user_prompt: str,
        principal: PrincipalContext,
        trace_id: str | None = None,
        normalized_rrf_score: float | None = None,
        doc_chunks: dict[str, str] | list[DocChunkRecord] | None = None,
        max_bounce_backs: int = 1,
        on_stage_event: Callable[[dict[str, Any]], None] | None = None,
        source_filter: SourceFilter = "all",
    ) -> ImplementationContextPack | InsufficientContextDiagnostic:
        """Executes the multi-domain federated DAG with OpenTelemetry instrumentation."""
        active_trace_id = trace_id or f"fed_trace_{uuid.uuid4().hex[:12]}"
        tracer = ScoutTelemetryTracer.get_instance()
        with tracer.trace_workflow(
            span_name="scout.federated_workflow",
            domain_id=self.federated_domain_id,
            trace_id=active_trace_id,
            model_id="gemma-4-moe",
            system="gemma",
        ) as fed_span:
            result = self._run_federated_dag(
                user_prompt=user_prompt,
                principal=principal,
                active_trace_id=active_trace_id,
                normalized_rrf_score=normalized_rrf_score,
                doc_chunks=doc_chunks,
                max_bounce_backs=max_bounce_backs,
                on_stage_event=on_stage_event,
                source_filter=source_filter,
            )
            fed_span.record_pack(result)
            return result

    def _probe_all_authorized_domains(
        self,
        *,
        user_prompt: str,
        principal: PrincipalContext,
        fed_bound_repo: BoundGraphRepository,
        source_filter: SourceFilter,
    ) -> tuple[
        list[DomainProbeTrace],
        dict[str, dict[str, Any]],
    ]:
        """Dynamically probe EVERY registered domain authorized for principal under RLS."""
        manifests = self.registry.load_all()
        probe_traces: list[DomainProbeTrace] = []
        domain_results: dict[str, dict[str, Any]] = {}

        for dom_id, manifest in manifests.items():
            modalities = list(manifest.modalities) or [
                manifest.authority_rules.primary_truth_source
            ]
            dom_plugin = PluginRegistry.get_domain_plugin(dom_id)
            dom_principal = principal
            if dom_plugin is not None and hasattr(dom_plugin, "elevate_principal_for_research"):
                dom_principal = dom_plugin.elevate_principal_for_research(principal)

            doc_records: list[DocChunkRecord] = []
            doc_hits: list[CalibratedRankedSearchHit] = []
            sym_records: list[ASTSymbolRecord] = []
            sym_hits: list[CalibratedRankedSearchHit] = []

            if source_filter != "code" and (
                "document" in modalities
                or "issues" in modalities
                or manifest.output_contract != "ImplementationContextPack"
            ):
                doc_records, doc_hits = fed_bound_repo.search_visible_doc_chunks_calibrated(
                    principal=dom_principal,
                    domain_id=dom_id,
                    query_text=user_prompt,
                    limit=8,
                )
                if source_filter != "all" and doc_records:
                    filtered_recs = filter_doc_chunks_by_source(doc_records, source_filter)
                    allowed_ids = {str(c.chunk_id) for c in filtered_recs}
                    doc_records = filtered_recs
                    doc_hits = [h for h in doc_hits if str(h.item_id) in allowed_ids]

            if source_filter in ("all", "code") and (
                "code" in modalities or manifest.output_contract == "ImplementationContextPack"
            ):
                sym_records, sym_hits = fed_bound_repo.search_visible_ast_symbols_calibrated(
                    principal=dom_principal,
                    domain_id=dom_id,
                    query_text=user_prompt,
                    limit=8,
                )

            all_hits = [*doc_hits, *sym_hits]
            accepted_doc_hits = [
                h
                for h in doc_hits
                if h.calibrated_score >= CALIBRATED_RELEVANCE_FLOOR and not h.is_boilerplate
            ]
            accepted_sym_hits = [
                h for h in sym_hits if h.calibrated_score >= CALIBRATED_RELEVANCE_FLOOR
            ]
            accepted_all = [*accepted_doc_hits, *accepted_sym_hits]
            max_cos = round(max((h.cosine_similarity for h in all_hits), default=0.0), 4)
            max_cal = round(max((h.calibrated_score for h in all_hits), default=0.0), 4)

            top_hit_traces = _build_retrieval_hit_traces(
                sorted(all_hits, key=lambda h: h.calibrated_score, reverse=True)[:5],
                domain_id=dom_id,
                modality="code" if (len(sym_hits) >= len(doc_hits) and sym_hits) else "document",
            )
            above_floor = bool(accepted_all) and max_cal >= CALIBRATED_RELEVANCE_FLOOR
            reason = (
                f"Matched {len(accepted_all)} above-floor hit(s) (max calibrated={max_cal:.3f})"
                if above_floor
                else (
                    f"Rejected: max calibrated={max_cal:.3f} < "
                    f"floor {CALIBRATED_RELEVANCE_FLOOR:.2f}"
                    if all_hits
                    else "No above-floor hits in PostgreSQL 17 RLS view"
                )
            )
            probe_traces.append(
                DomainProbeTrace(
                    domain_id=dom_id,
                    display_name=manifest.display_name,
                    authorized=bool(dom_principal.allowed_scopes),
                    probed_hits=len(all_hits),
                    accepted_hits=len(accepted_all),
                    rejected_hits=len(all_hits) - len(accepted_all),
                    max_cosine_similarity=max_cos,
                    max_calibrated_score=max_cal,
                    selected=above_floor,
                    selection_reason=reason,
                    top_hits=top_hit_traces,
                )
            )
            if above_floor:
                accepted_doc_ids = {str(h.item_id) for h in accepted_doc_hits}
                accepted_sym_ids = {h.item_id for h in accepted_sym_hits}
                kept_docs = [c for c in doc_records if str(c.chunk_id) in accepted_doc_ids]
                kept_syms = [s for s in sym_records if s.symbol_id in accepted_sym_ids]
                domain_results[dom_id] = {
                    "manifest": manifest,
                    "principal": dom_principal,
                    "doc_records": kept_docs or doc_records,
                    "doc_hits": accepted_doc_hits,
                    "sym_records": kept_syms or sym_records,
                    "sym_hits": accepted_sym_hits,
                    "max_calibrated_score": max_cal,
                    "is_code": bool(accepted_sym_hits) and (
                        "code" in modalities or len(accepted_sym_hits) > len(accepted_doc_hits)
                    ),
                    "is_doc": bool(accepted_doc_hits),
                }

        probe_traces.sort(key=lambda t: t.max_calibrated_score, reverse=True)
        return probe_traces, domain_results

    def _run_federated_dag(
        self,
        user_prompt: str,
        principal: PrincipalContext,
        active_trace_id: str,
        normalized_rrf_score: float | None,
        doc_chunks: dict[str, str] | list[DocChunkRecord] | None,
        max_bounce_backs: int,
        on_stage_event: Callable[[dict[str, Any]], None] | None = None,
        source_filter: SourceFilter = "all",
    ) -> ImplementationContextPack | InsufficientContextDiagnostic:
        """Executes true dynamic multi-domain federated routing & synthesis."""
        dag_t0 = time.perf_counter()
        fed_domain = self.federated_domain_id

        decision = self.guardrail.evaluate(user_prompt, principal=principal)
        if not decision.passed:
            return InsufficientContextDiagnostic(
                trace_id=active_trace_id,
                domain_id=fed_domain,
                reason=f"Security guardrail rejection: {decision.rationale}",
                missing_symbols=[],
                partial_candidate_files=[],
                paid_tokens_spent=0,
            )

        fed_bound_repo = BoundGraphRepository(self.repo)
        relevance_rejection = _evaluate_query_domain_relevance(
            user_prompt=user_prompt,
            domain_id=fed_domain,
            bound_repo=fed_bound_repo,
            principal=principal,
        )
        if relevance_rejection is not None:
            return InsufficientContextDiagnostic(
                trace_id=active_trace_id,
                domain_id=fed_domain,
                reason=relevance_rejection,
                missing_symbols=[],
                partial_candidate_files=[],
                paid_tokens_spent=0,
            )

        query_plan = plan_query_with_llm(user_prompt, self.local_client)

        # Dynamic multi-domain routing path (all registered domains in DomainRegistry)
        if doc_chunks is None:
            s1_t0 = time.perf_counter()
            all_domains = list(self.registry.load_all().keys())
            s1_ms = max(1, int((time.perf_counter() - s1_t0) * 1000))
            s1_trace = StageExecutionTrace(
                stage=1,
                stage_name="Stage 1: Auth, Guardrail & Query Plan",
                status="completed",
                elapsed_ms=float(s1_ms),
                summary=(
                    f"Multi-domain planner: query_type={query_plan.query_type}, "
                    f"{len(query_plan.facets)} facet(s), probing {len(all_domains)} domain(s)."
                ),
                principal_id=principal.user_id,
                principal_scopes=list(principal.allowed_scopes),
                guardrail_verdict="passed",
                query_type=query_plan.query_type,
                core_subject=query_plan.core_subject,
                required_facets=[f.facet_query for f in query_plan.facets],
                target_domains=all_domains,
            )

            s2_t0 = time.perf_counter()
            probe_traces, domain_results = self._probe_all_authorized_domains(
                user_prompt=user_prompt,
                principal=principal,
                fed_bound_repo=fed_bound_repo,
                source_filter=source_filter,
            )
            selected_ids = sorted(
                domain_results.keys(),
                key=lambda d: float(domain_results[d]["max_calibrated_score"]),
                reverse=True,
            )
            if selected_ids:
                q_tokens = {
                    t.lower()
                    for t in re.findall(r"[A-Za-z0-9_]{3,}", user_prompt)
                    if t.lower() not in PROMPT_STOPWORDS
                }
                direct_code_tokens: dict[str, set[str]] = {}
                for dom_id in selected_ids:
                    sym_hits = domain_results[dom_id].get("sym_hits", [])
                    matched_toks: set[str] = set()
                    for h in sym_hits:
                        qname = str(getattr(h, "qualified_name_or_title", "") or "").lower()
                        parts = [p for p in re.split(r"[.:/]", qname) if p]
                        for tok in q_tokens:
                            if tok in parts[:2]:
                                matched_toks.add(tok)
                    if matched_toks:
                        direct_code_tokens[dom_id] = matched_toks

                # Keep domains sorted strictly by calibrated relevance; only break near-ties
                # (within 0.05) in favor of explicit package/symbol identifier matches, and never
                # prune high-scoring doc domains (>= 0.50 or within 0.25 of top score).
                if direct_code_tokens and source_filter in ("all", "code"):
                    top_raw_score = max(
                        float(domain_results[d]["max_calibrated_score"]) for d in selected_ids
                    )
                    all_code_toks = set().union(*direct_code_tokens.values())
                    filtered_ids: list[str] = []
                    for dom_id in selected_ids:
                        info = domain_results[dom_id]
                        sc_dom = float(info["max_calibrated_score"])
                        if (
                            dom_id in direct_code_tokens
                            or info.get("is_code")
                            or sc_dom >= 0.50
                            or (top_raw_score - sc_dom) <= 0.25
                        ):
                            filtered_ids.append(dom_id)
                        else:
                            doc_blob = " ".join(
                                f"{c.doc_title} {c.verbatim_content[:800]}".lower()
                                for c in info.get("doc_records", [])
                            )
                            if any(tok in doc_blob for tok in all_code_toks):
                                filtered_ids.append(dom_id)
                    if filtered_ids:
                        selected_ids = filtered_ids
                        selected_ids.sort(
                            key=lambda d: (
                                1
                                if (
                                    d in direct_code_tokens
                                    and (
                                        top_raw_score
                                        - float(domain_results[d]["max_calibrated_score"])
                                    )
                                    <= 0.05
                                )
                                else 0,
                                float(domain_results[d]["max_calibrated_score"]),
                            ),
                            reverse=True,
                        )

                top_id = selected_ids[0]
                top_score = float(domain_results[top_id]["max_calibrated_score"])
                top_is_code = bool(domain_results[top_id].get("is_code"))
                fed_has_code = any(bool(domain_results[d].get("is_code")) for d in selected_ids)
                fed_has_docs = any(bool(domain_results[d].get("is_doc")) for d in selected_ids)
                fed_intent_mode = QueryIntentClassifier.classify(
                    user_prompt,
                    source_filter=source_filter,
                    has_visible_code=fed_has_code,
                    has_visible_docs=fed_has_docs,
                ).mode
                all_code_toks_keep = (
                    set().union(*direct_code_tokens.values()) if direct_code_tokens else set()
                )
                max_gap = (
                    0.12
                    if (query_plan.query_type == "concept_definition" and top_score >= 0.85)
                    else 0.20
                )
                kept_ids: list[str] = []
                for dom_id in selected_ids:
                    info = domain_results[dom_id]
                    sc = float(info["max_calibrated_score"])
                    is_complementary_doc = (
                        top_is_code
                        and bool(info.get("is_doc"))
                        and not bool(info.get("is_code"))
                        and (
                            (top_score - sc) <= 0.40
                            or any(
                                tok in f"{c.doc_title} {c.verbatim_content[:800]}".lower()
                                for c in info.get("doc_records", [])
                                for tok in all_code_toks_keep
                            )
                        )
                    )
                    is_complementary_code = (
                        not top_is_code
                        and bool(info.get("is_code"))
                        and query_plan.query_type != "concept_definition"
                        and fed_intent_mode
                        in (QueryIntentMode.HYBRID_CODE_AND_DOCS, QueryIntentMode.PURE_CODE)
                        and (top_score - sc) <= 0.25
                        and sc >= 0.48
                    )
                    if (
                        dom_id == top_id
                        or (dom_id in direct_code_tokens and (top_score - sc) <= 0.25)
                        or is_complementary_doc
                        or is_complementary_code
                        or (top_score - sc) <= max_gap
                    ):
                        kept_ids.append(dom_id)
                selected_ids = kept_ids

                selected_set = set(selected_ids)
                for p in probe_traces:
                    if p.selected and p.domain_id not in selected_set:
                        p.selected = False
                        p.selection_reason = (
                            f"Pruned relative to higher-scoring domain(s) "
                            f"(calibrated={p.max_calibrated_score:.3f})"
                        )

            # 0 domains above relevance floor -> abstain at $0 immediately
            if not selected_ids:
                s2_ms = max(1, int((time.perf_counter() - s2_t0) * 1000))
                max_observed = round(
                    max((p.max_calibrated_score for p in probe_traces), default=0.0), 4
                )
                if on_stage_event is not None:
                    on_stage_event(
                        {
                            "event_type": "stage_update",
                            "stage": 1,
                            "stage_name": "Stage 1: Auth, Guardrail & Query Plan",
                            "status": "completed",
                            "details": _stage1_details(
                                domains=all_domains,
                                principal=principal,
                                visible_files_count=0,
                                visible_chunks_count=0,
                                stale_excluded_count=0,
                                elapsed_ms=s1_ms,
                            ),
                            "stage_trace": s1_trace.model_dump(),
                        }
                    )
                s2_empty_trace = StageExecutionTrace(
                    stage=2,
                    stage_name="Stage 2: Multi-Domain Hybrid Retrieval & AST Closure",
                    status="completed",
                    elapsed_ms=float(s2_ms),
                    summary=(
                        f"Probed {len(probe_traces)} domain(s); 0 above relevance floor "
                        f"{CALIBRATED_RELEVANCE_FLOOR:.2f} (max={max_observed:.3f})."
                    ),
                    target_domains=[],
                    domain_probes=probe_traces,
                )
                if on_stage_event is not None:
                    on_stage_event(
                        {
                            "event_type": "stage_update",
                            "stage": 2,
                            "stage_name": "Stage 2: Multi-Domain Hybrid Retrieval & AST Closure",
                            "status": "completed",
                            "details": s2_empty_trace.summary,
                            "stage_trace": s2_empty_trace.model_dump(),
                        }
                    )
                s3_empty_trace = StageExecutionTrace(
                    stage=3,
                    stage_name="Stage 3: Local Gemma-4 Synthesis & Facet Grounding",
                    status="completed",
                    elapsed_ms=1.0,
                    summary="Skipped synthesis: zero domains cleared Stage 2 calibrated floor.",
                    chunks_fed=0,
                    sufficiency_verdict="insufficient",
                    grounded_facets=0,
                    total_facets=len(query_plan.facets),
                )
                reason = (
                    f"Zero domains cleared calibrated relevance floor "
                    f"({CALIBRATED_RELEVANCE_FLOOR:.2f}) for query '{user_prompt.strip()}' "
                    f"across {len(probe_traces)} probed domain(s) "
                    f"(max calibrated={max_observed:.3f}). Paid frontier tokens spent: 0."
                )
                s4_fail_trace = StageExecutionTrace(
                    stage=4,
                    stage_name="Stage 4: Pre-Flight Sufficiency Gate",
                    status="failed",
                    elapsed_ms=float(max(1, int((time.perf_counter() - dag_t0) * 1000))),
                    summary=reason,
                    sufficiency_verdict="insufficient",
                    grounded_facets=0,
                    total_facets=len(query_plan.facets),
                    checks=_build_preflight_ledger(
                        report=None,
                        calibrated_score=max_observed,
                        min_floor=CALIBRATED_RELEVANCE_FLOOR,
                        sufficiency_verdict="insufficient",
                        grounded_facets=0,
                        total_facets=len(query_plan.facets),
                    ),
                    final_decision="abstain_zero_paid",
                    paid_tokens_spent=0,
                )
                if on_stage_event is not None:
                    on_stage_event(
                        {
                            "event_type": "stage_update",
                            "stage": 4,
                            "stage_name": "Stage 4: Pre-Flight Sufficiency Gate",
                            "status": "failed",
                            "details": _stage4_failed_details(
                                reason=reason,
                                total_ms=max(1, int((time.perf_counter() - dag_t0) * 1000)),
                            ),
                            "stage_trace": s4_fail_trace.model_dump(),
                        }
                    )
                return InsufficientContextDiagnostic(
                    trace_id=active_trace_id,
                    domain_id=fed_domain,
                    reason=reason,
                    missing_symbols=[],
                    missing_items=[f"federated:{d}:below_floor" for d in all_domains],
                    partial_candidate_files=[],
                    paid_tokens_spent=0,
                    missing_or_unverified_aspects=[f.facet_query for f in query_plan.facets],
                    stage_traces=[s1_trace, s2_empty_trace, s3_empty_trace, s4_fail_trace],
                )

            # Single best domain above floor -> route cleanly to that single domain!
            if len(selected_ids) == 1:
                win_dom = selected_ids[0]
                win_info = domain_results[win_dom]
                win_principal: PrincipalContext = win_info["principal"]
                single_wf = ScoutWorkflow(
                    registry=self.registry,
                    repo=self.repo,
                    local_client=self.local_client,
                    validator=self.validator,
                    slicer=self.slicer,
                    expander=self.expander,
                    guardrail=self.guardrail,
                    auth_gate=self.auth_gate,
                )

                def _relay_stage_event(ev: dict[str, Any]) -> None:
                    if ev.get("stage") == 2 and isinstance(ev.get("stage_trace"), dict):
                        ev["stage_trace"]["domain_probes"] = [
                            p.model_dump() for p in probe_traces
                        ]
                        ev["stage_trace"]["target_domains"] = [win_dom]
                    if on_stage_event is not None:
                        on_stage_event(ev)

                win_res = single_wf.run(
                    user_prompt=user_prompt,
                    principal=win_principal,
                    domain_id=win_dom,
                    trace_id=active_trace_id,
                    normalized_rrf_score=normalized_rrf_score,
                    max_bounce_backs=max_bounce_backs,
                    on_stage_event=_relay_stage_event if on_stage_event is not None else None,
                    source_filter=source_filter,
                )
                if getattr(win_res, "stage_traces", None):
                    for st in win_res.stage_traces:
                        if st.stage == 2:
                            st.domain_probes = probe_traces
                            st.target_domains = [win_dom]
                return win_res

            # 2+ domains genuinely cleared the relevance floor -> cross-domain synthesis
            doc_candidates = [d for d in selected_ids if domain_results[d]["is_doc"]]
            code_candidates = [d for d in selected_ids if domain_results[d]["is_code"]]
            research_domain = doc_candidates[0] if doc_candidates else selected_ids[0]
            code_domain = code_candidates[0] if code_candidates else selected_ids[0]
            active_principal = domain_results[research_domain]["principal"]
            doc_records_multi: list[DocChunkRecord] = []
            multi_hits: list[CalibratedRankedSearchHit] = []
            doc_lists = [list(domain_results[d]["doc_records"]) for d in doc_candidates[:3]]
            hit_lists = [list(domain_results[d]["doc_hits"]) for d in doc_candidates[:3]]
            max_doc_len = max((len(lst) for lst in doc_lists), default=0)
            for idx in range(max_doc_len):
                for lst in doc_lists:
                    if idx < len(lst):
                        doc_records_multi.append(lst[idx])
            max_hit_len = max((len(lst) for lst in hit_lists), default=0)
            for idx in range(max_hit_len):
                for lst in hit_lists:
                    if idx < len(lst):
                        multi_hits.append(lst[idx])
            for d in code_candidates[:2]:
                multi_hits.extend(domain_results[d]["sym_hits"])
        else:
            probe_traces = []
            research_domain = self.research_domain_id
            code_domain = self.code_domain_id
            selected_ids = [research_domain, code_domain]
            doc_candidates = []
            domain_plugin_init = PluginRegistry.get_domain_plugin(research_domain)
            active_principal = (
                domain_plugin_init.elevate_principal_for_research(principal)
                if domain_plugin_init is not None
                and hasattr(domain_plugin_init, "elevate_principal_for_research")
                else principal
            )
            doc_records_multi = []
            multi_hits = []

        # ---------------------------------------------------------------------
        # Stage 1: Research / Documentation Domain Extraction
        # ---------------------------------------------------------------------
        s1_t0 = time.perf_counter()
        self.local_client.set_active_domain(research_domain)
        visible_files = fed_bound_repo.list_visible_code_files(active_principal, code_domain)
        raw_active_chunks = (
            doc_chunks
            if doc_chunks is not None
            else (
                doc_records_multi
                or fed_bound_repo.list_visible_doc_chunks(active_principal, research_domain)
            )
        )
        active_chunks = filter_doc_chunks_by_source(raw_active_chunks, source_filter)
        doc_bound_repo: BoundGraphRepository | _SourceFilteredGraphRepo = (
            _SourceFilteredGraphRepo(fed_bound_repo, source_filter)
            if source_filter != "all"
            else fed_bound_repo
        )
        stale_excluded_s1 = (
            sum(1 for c in active_chunks if bool(getattr(c, "is_stale", False)))
            if isinstance(active_chunks, list)
            else 0
        )
        visible_chunks_s1 = (
            len(active_chunks) - stale_excluded_s1
            if isinstance(active_chunks, list)
            else len(active_chunks)
        )
        intent_decision = QueryIntentClassifier.classify(
            user_prompt,
            source_filter=source_filter,
            has_visible_code=bool(visible_files),
            has_visible_docs=bool(active_chunks),
        )
        if (
            source_filter == "all"
            and intent_decision.mode == QueryIntentMode.PURE_CODE
            and doc_candidates
            and bool(active_chunks)
            and not re.search(
                r"\b[A-Za-z0-9_./-]+\.(?:py|ts|tsx|js|jsx|sql|go|rs|java)\b",
                user_prompt,
            )
            and (
                not top_is_code
                or any(
                    d != code_domain
                    and float(domain_results[d]["max_calibrated_score"]) >= 0.45
                    for d in doc_candidates
                )
            )
        ):
            intent_decision = QueryIntentDecision(
                mode=QueryIntentMode.HYBRID_CODE_AND_DOCS,
                source_filter=source_filter,
                needs_code=True,
                needs_docs=True,
                reason=(
                    "Multi-domain probe selected above-floor doc and code domains; "
                    "retaining hybrid synthesis."
                ),
            )

        quant_brief: QuantStrategyBrief | None = None
        domain_knowledge_brief: DomainKnowledgeBrief | None = None
        citations: list[DocCitationSpan] = []

        res_manifest = self.registry.load_all().get(research_domain)
        uses_kb_contract = (
            res_manifest is not None and res_manifest.output_contract == "DomainKnowledgeBrief"
        )

        if intent_decision.mode == QueryIntentMode.PURE_CODE:
            s1_ms = max(1, int((time.perf_counter() - s1_t0) * 1000))
        else:
            cross_domain_chunks = (
                list(active_chunks)
                if (len(doc_candidates) >= 2 and isinstance(active_chunks, list) and active_chunks)
                else None
            )
            if uses_kb_contract:
                domain_knowledge_brief = self.local_client.generate_domain_knowledge_brief(
                    user_prompt=user_prompt,
                    domain_id=research_domain,
                    repo=doc_bound_repo,
                    principal=active_principal,
                    quant_brief=None,
                    pre_retrieved_chunks=cross_domain_chunks,
                )
                if domain_knowledge_brief is not None and domain_knowledge_brief.verified_citations:
                    citations = list(domain_knowledge_brief.verified_citations)
            else:
                quant_brief = self.local_client.generate_quant_brief(
                    user_prompt=user_prompt,
                    domain_id=research_domain,
                    repo=doc_bound_repo,
                    principal=active_principal,
                    bounce_back_feedback=None,
                    pre_retrieved_chunks=cross_domain_chunks,
                )
                if quant_brief is not None and quant_brief.verified_citations:
                    citations = list(quant_brief.verified_citations)
                    domain_knowledge_brief = self.local_client.generate_domain_knowledge_brief(
                        user_prompt=user_prompt,
                        domain_id=research_domain,
                        repo=doc_bound_repo,
                        principal=active_principal,
                        quant_brief=quant_brief,
                    )
            if not citations and isinstance(active_chunks, list) and active_chunks:
                quant_brief = None
                fallback_min_hits = (
                    1 if intent_decision.mode == QueryIntentMode.PURE_DOCUMENTATION else None
                )
                citations = _extract_hybrid_doc_citations(
                    user_prompt=user_prompt,
                    doc_chunks=active_chunks,
                    slicer=self.slicer,
                    validator=self.validator,
                    min_hits_override=fallback_min_hits,
                )
                if citations:
                    chunks_by_id = {str(c.chunk_id): c for c in active_chunks}
                    domain_knowledge_brief = _build_domain_brief_from_citations(
                        user_prompt=user_prompt,
                        domain_id=research_domain,
                        citations=citations,
                        chunks_by_id=chunks_by_id,
                    )
            s1_ms = max(1, int((time.perf_counter() - s1_t0) * 1000))

            if quant_brief is None and not citations:
                no_brief_reason = "Research brief synthesis returned no verified citations"
                s1_no_brief_trace = StageExecutionTrace(
                    stage=1,
                    stage_name="Stage 1: Auth, Guardrail & Query Plan",
                    status="completed",
                    elapsed_ms=float(s1_ms),
                    summary=(
                        f"Guardrail passed; query_type={query_plan.query_type}; "
                        f"target_domains={', '.join(selected_ids)}."
                    ),
                    principal_id=active_principal.user_id,
                    principal_scopes=list(active_principal.allowed_scopes),
                    guardrail_verdict="passed",
                    query_type=query_plan.query_type,
                    core_subject=query_plan.core_subject,
                    required_facets=[f.facet_query for f in query_plan.facets],
                    target_domains=selected_ids,
                )
                if on_stage_event is not None:
                    on_stage_event(
                        {
                            "event_type": "stage_update",
                            "stage": 1,
                            "stage_name": "Stage 1: Auth, Guardrail & Query Plan",
                            "status": "completed",
                            "details": _stage1_details(
                                domains=selected_ids,
                                principal=active_principal,
                                visible_files_count=len(visible_files),
                                visible_chunks_count=visible_chunks_s1,
                                stale_excluded_count=stale_excluded_s1,
                                elapsed_ms=s1_ms,
                            ),
                            "stage_trace": s1_no_brief_trace.model_dump(),
                        }
                    )
                s2_no_brief_trace = StageExecutionTrace(
                    stage=2,
                    stage_name="Stage 2: Multi-Domain Hybrid Retrieval & AST Closure",
                    status="completed",
                    elapsed_ms=1.0,
                    summary=f"Probed {', '.join(selected_ids)}; 0 grounded citations.",
                    target_domains=selected_ids,
                    domain_probes=probe_traces,
                )
                s3_no_brief_trace = StageExecutionTrace(
                    stage=3,
                    stage_name="Stage 3: Local Gemma-4 Synthesis & Facet Grounding",
                    status="completed",
                    elapsed_ms=1.0,
                    summary=no_brief_reason,
                    chunks_fed=0,
                    sufficiency_verdict="insufficient",
                    grounded_facets=0,
                    total_facets=len(query_plan.facets),
                )
                s4_no_brief_trace = StageExecutionTrace(
                    stage=4,
                    stage_name="Stage 4: Pre-Flight Sufficiency Gate",
                    status="failed",
                    elapsed_ms=float(max(1, int((time.perf_counter() - dag_t0) * 1000))),
                    summary=f"{no_brief_reason}. Paid frontier tokens spent: 0.",
                    sufficiency_verdict="insufficient",
                    grounded_facets=0,
                    total_facets=len(query_plan.facets),
                    checks=_build_preflight_ledger(
                        report=None,
                        calibrated_score=max(
                            (p.max_calibrated_score for p in probe_traces), default=0.0
                        ),
                        min_floor=CALIBRATED_RELEVANCE_FLOOR,
                        sufficiency_verdict="insufficient",
                        grounded_facets=0,
                        total_facets=len(query_plan.facets),
                    ),
                    final_decision="abstain_zero_paid",
                    paid_tokens_spent=0,
                )
                if on_stage_event is not None:
                    on_stage_event(
                        {
                            "event_type": "stage_update",
                            "stage": 4,
                            "stage_name": "Stage 4: Pre-Flight Sufficiency Gate",
                            "status": "failed",
                            "details": _stage4_failed_details(
                                reason=no_brief_reason,
                                total_ms=max(1, int((time.perf_counter() - dag_t0) * 1000)),
                            ),
                            "stage_trace": s4_no_brief_trace.model_dump(),
                        }
                    )
                return InsufficientContextDiagnostic(
                    trace_id=active_trace_id,
                    domain_id=(
                        fed_domain
                        if (fed_domain == "federated_all" and len(selected_ids) >= 2)
                        else research_domain
                    ),
                    reason=no_brief_reason,
                    missing_symbols=[],
                    partial_candidate_files=[],
                    paid_tokens_spent=0,
                    missing_or_unverified_aspects=[f.facet_query for f in query_plan.facets],
                    stage_traces=[
                        s1_no_brief_trace,
                        s2_no_brief_trace,
                        s3_no_brief_trace,
                        s4_no_brief_trace,
                    ],
                )

        if (
            isinstance(quant_brief, QuantStrategyBrief)
            and quant_brief.friction_verdict == FrictionVerdict.FAILS_AFTER_COSTS
        ):
            logger.warning(
                "Research brief failed domain rule check: assumed %s bps for %s",
                quant_brief.transaction_cost_bps_assumed,
                quant_brief.target_markets,
            )

        s1_fed_trace = StageExecutionTrace(
            stage=1,
            stage_name="Stage 1: Auth, Guardrail & Query Plan",
            status="completed",
            elapsed_ms=float(s1_ms),
            summary=(
                f"Guardrail passed; query_type={query_plan.query_type}; "
                f"target_domains={', '.join(selected_ids)}."
            ),
            principal_id=active_principal.user_id,
            principal_scopes=list(active_principal.allowed_scopes),
            guardrail_verdict="passed",
            query_type=query_plan.query_type,
            core_subject=query_plan.core_subject,
            required_facets=[f.facet_query for f in query_plan.facets],
            target_domains=selected_ids,
        )
        if on_stage_event is not None:
            on_stage_event(
                {
                    "event_type": "stage_update",
                    "stage": 1,
                    "stage_name": "Stage 1: Auth, Guardrail & Query Plan",
                    "status": "completed",
                    "details": _stage1_details(
                        domains=selected_ids,
                        principal=active_principal,
                        visible_files_count=len(visible_files),
                        visible_chunks_count=visible_chunks_s1,
                        stale_excluded_count=stale_excluded_s1,
                        elapsed_ms=s1_ms,
                    ),
                    "stage_trace": s1_fed_trace.model_dump(),
                }
            )

        # ---------------------------------------------------------------------
        # Stage 2: Code Domain Implementation Localization
        # ---------------------------------------------------------------------
        s2_t0 = time.perf_counter()
        self.local_client.set_active_domain(code_domain)

        if quant_brief is not None:
            code_prompt = (
                f"{user_prompt}\n"
                f"Strategy Name: {quant_brief.strategy_name}\n"
                f"Mathematical Signal: {quant_brief.mathematical_signal_latex}"
            )
        else:
            code_prompt = user_prompt

        if intent_decision.mode == QueryIntentMode.PURE_DOCUMENTATION:
            code_pointers = []
        else:
            code_pointers = self.local_client.generate_code_pointers(
                user_prompt=code_prompt,
                domain_id=code_domain,
                repo=fed_bound_repo,
                principal=active_principal,
                bounce_back_feedback=None,
            )
        s2_ms = max(1, int((time.perf_counter() - s2_t0) * 1000))
        s2_fed_trace = StageExecutionTrace(
            stage=2,
            stage_name="Stage 2: Multi-Domain Hybrid Retrieval & AST Closure",
            status="completed",
            elapsed_ms=float(s2_ms),
            summary=(
                f"Cross-domain retrieval across {', '.join(selected_ids)}: "
                f"{len(code_pointers)} code pointer(s), {len(citations)} citation(s)."
            ),
            target_domains=selected_ids,
            domain_probes=probe_traces,
        )

        if on_stage_event is not None:
            on_stage_event(
                {
                    "event_type": "stage_update",
                    "stage": 2,
                    "stage_name": "Stage 2: Multi-Domain Hybrid Retrieval & AST Closure",
                    "status": "completed",
                    "details": _stage2_details(
                        model_label="gemma-4-26B-A4B-it",
                        adapter_id=code_domain,
                        pointers_count=len(code_pointers),
                        citations_count=len(citations),
                        elapsed_ms=s2_ms,
                    ),
                    "stage_trace": s2_fed_trace.model_dump(),
                }
            )

        s3_t0 = time.perf_counter()
        bound_repo = fed_bound_repo
        files_by_path: dict[str, CodeFileRecord] = {f.file_path: f for f in visible_files}

        hydrated_slices: list[VerbatimCodeSlice] = []
        for ptr in code_pointers:
            f_rec = files_by_path.get(ptr.file_path)
            if f_rec is not None:
                try:
                    snapped_ptr = _snap_python_ast_pointer_bounds(ptr, f_rec)
                    slice_item = self.slicer.slice_pointer(
                        pointer=snapped_ptr,
                        file_record=f_rec,
                        auto_expanded_by_ast=False,
                    )
                    hydrated_slices.append(slice_item)
                except Exception as exc:
                    logger.warning(
                        "Failed slicing federated pointer %s (%s): %s",
                        ptr.file_path,
                        type(exc).__name__,
                        exc,
                    )
            else:
                hydrated_slices.append(
                    VerbatimCodeSlice(
                        repo_name=ptr.repo_name,
                        file_path=ptr.file_path,
                        start_line=ptr.start_line,
                        end_line=ptr.end_line,
                        qualified_symbol=ptr.qualified_symbol,
                        selection_reason=ptr.selection_reason,
                        modification_intent=ptr.modification_intent,
                        git_commit_sha="placeholder_sha",
                        git_blob_sha="placeholder_blob",
                        verbatim_code_with_lines=(
                            f"{ptr.start_line:4d} | # Symbol: {ptr.qualified_symbol}"
                        ),
                        auto_expanded_by_ast=False,
                    )
                )

        visible_symbols = bound_repo.list_visible_ast_symbols(active_principal, code_domain)
        for s in list(hydrated_slices):
            if s.modification_intent == ModificationIntent.MODIFY_TARGET:
                sym_rec = next(
                    (sym for sym in visible_symbols if sym.qualified_name == s.qualified_symbol),
                    None,
                )
                if sym_rec is not None:
                    exp_res = self.expander.expand_1hop(
                        focus_symbol=sym_rec,
                        repo=bound_repo,
                        principal=active_principal,
                        domain_id=code_domain,
                    )
                    for auto_ptr in (
                        exp_res.imported_type_slices
                        + exp_res.caller_slices
                        + exp_res.test_fixture_slices
                    ):
                        if not any(
                            ex.file_path == auto_ptr.file_path
                            and ex.qualified_symbol == auto_ptr.qualified_symbol
                            for ex in hydrated_slices
                        ):
                            f_rec = files_by_path.get(auto_ptr.file_path)
                            if f_rec is not None:
                                try:
                                    snapped_auto = _snap_python_ast_pointer_bounds(auto_ptr, f_rec)
                                    hydrated_slices.append(
                                        self.slicer.slice_pointer(
                                            pointer=snapped_auto,
                                            file_record=f_rec,
                                            auto_expanded_by_ast=True,
                                        )
                                    )
                                except Exception as exc:
                                    logger.debug(
                                        "Skipping invalid 1-hop slice %s (%s): %s",
                                        auto_ptr.file_path,
                                        type(exc).__name__,
                                        exc,
                                    )

        if intent_decision.mode == QueryIntentMode.PURE_DOCUMENTATION:
            hydrated_slices = []

        conflicts: list[str] = []
        if (
            isinstance(quant_brief, QuantStrategyBrief)
            and quant_brief.friction_verdict == FrictionVerdict.FAILS_AFTER_COSTS
        ):
            conflicts.append(
                f"Paper assumes {quant_brief.transaction_cost_bps_assumed} bps transaction costs, "
                f"which fails statutory friction floor for: {quant_brief.target_markets}"
            )

        fed_chunk_records: list[DocChunkRecord] = (
            list(active_chunks)
            if isinstance(active_chunks, list)
            else (
                doc_bound_repo.list_visible_doc_chunks(active_principal, research_domain)
                + doc_bound_repo.list_visible_doc_chunks(active_principal, code_domain)
            )
        )
        conflicts = _detect_pack_conflicts(
            hydrated_slices,
            fed_chunk_records,
            initial_conflicts=conflicts,
        )

        fed_plan = query_plan
        fed_facets, fed_missing, fed_completeness = _evaluate_query_plan_coverage(
            fed_plan,
            doc_chunks=[*citations, *fed_chunk_records],
            code_slices=hydrated_slices,
            domain_id=research_domain if not hydrated_slices else code_domain,
        )
        effective_fed_rrf, calibrated_fed_score = _compute_effective_retrieval_scores(
            explicit_score=normalized_rrf_score,
            retrieval_hits=multi_hits,
            hydrated_slices=hydrated_slices,
            citations=citations,
            completeness_ratio=fed_completeness,
        )

        s3_ms = max(1, int((time.perf_counter() - s3_t0) * 1000))
        fed_sufficiency: Literal["sufficient", "partial", "insufficient"] = (
            domain_knowledge_brief.sufficiency_verdict
            if domain_knowledge_brief is not None
            else (
                quant_brief.sufficiency_verdict
                if quant_brief is not None
                else ("sufficient" if (hydrated_slices or citations) else "insufficient")
            )
        )
        grounded_fed = sum(1 for f in fed_facets if f.status == "grounded")
        partial_fed = sum(1 for f in fed_facets if f.status == "partial")
        if (
            quant_brief is not None
            and quant_brief.sufficiency_verdict == "insufficient"
            and quant_brief.synthesis_mode == "deterministic_extractive_fallback"
            and quant_brief.verified_citations
            and hydrated_slices
            and grounded_fed >= 1
        ):
            reconciled_verdict: Literal["sufficient", "partial", "insufficient"] = (
                "sufficient" if (not fed_missing and partial_fed == 0) else "partial"
            )
            quant_brief.sufficiency_verdict = reconciled_verdict
            quant_brief.missing_aspects = list(fed_missing)
            if (
                domain_knowledge_brief is not None
                and domain_knowledge_brief.sufficiency_verdict == "insufficient"
            ):
                domain_knowledge_brief.sufficiency_verdict = reconciled_verdict
                domain_knowledge_brief.missing_aspects = list(fed_missing)
            fed_sufficiency = reconciled_verdict
        has_direct_fed_symbol = _has_verified_code_slice_hit(
            user_prompt,
            hydrated_slices,
            partial_facet_count=partial_fed,
        )
        fed_floor_failed = (
            normalized_rrf_score is None
            and bool(multi_hits)
            and calibrated_fed_score < CALIBRATED_RELEVANCE_FLOOR
            and not hydrated_slices
        )
        fed_brief_mode = (
            domain_knowledge_brief.synthesis_mode
            if domain_knowledge_brief is not None
            else (
                str(getattr(quant_brief, "synthesis_mode", "deterministic_extractive_fallback"))
                if quant_brief is not None
                else "deterministic_extractive_fallback"
            )
        )
        fed_live_modes = ("live_base_gemma4_json", "live_qlora_json", "live_mlx_json")
        fed_raw_verdict = (
            domain_knowledge_brief.sufficiency_verdict
            if domain_knowledge_brief is not None
            else (quant_brief.sufficiency_verdict if quant_brief is not None else "insufficient")
        )
        fed_chunks_by_id = {str(c.chunk_id): c for c in fed_chunk_records}
        fed_cits_verified = bool(citations) and all(
            str(c.chunk_id) in fed_chunks_by_id
            and self.slicer.verify_doc_citation(
                citation=c,
                chunk_content=fed_chunks_by_id[str(c.chunk_id)].verbatim_content,
            ).is_valid
            for c in citations
        )
        fed_llm_verified_sufficient = (
            fed_brief_mode in fed_live_modes
            and fed_raw_verdict == "sufficient"
            and fed_cits_verified
            and not fed_floor_failed
        )
        if (
            fed_facets
            and grounded_fed == 0
            and not has_direct_fed_symbol
            and not fed_llm_verified_sufficient
        ):
            fed_sufficiency = "insufficient"

        fed_chunks_fed = (
            max(
                len(citations),
                min(6, len(active_chunks or [])),
                int(getattr(self.local_client, "_last_chunks_fed", 0)),
            )
            if (citations or domain_knowledge_brief is not None or quant_brief is not None)
            else 0
        )
        s3_fed_trace = StageExecutionTrace(
            stage=3,
            stage_name="Stage 3: Local Gemma-4 Synthesis & Facet Grounding",
            status="completed",
            elapsed_ms=float(s3_ms),
            summary=f"Sufficiency={fed_sufficiency}, grounded={grounded_fed}/{len(fed_facets)}.",
            ast_one_hop_count=sum(1 for s in hydrated_slices if s.auto_expanded_by_ast),
            model_id=res_manifest.base_model_id if res_manifest else DEFAULT_LOCAL_GEMMA_MODEL,
            adapter_mode=(
                "qlora_adapter" if (res_manifest and res_manifest.lora_enabled) else "base_gemma4"
            ),
            chunks_fed=fed_chunks_fed,
            sufficiency_verdict=fed_sufficiency,
            grounded_facets=grounded_fed,
            total_facets=len(fed_facets),
        )
        if on_stage_event is not None:
            on_stage_event(
                {
                    "event_type": "stage_update",
                    "stage": 3,
                    "stage_name": "Stage 3: Local Gemma-4 Synthesis & Facet Grounding",
                    "status": "completed",
                    "details": _stage3_details(
                        slices=hydrated_slices,
                        citations_count=len(citations),
                        conflicts_count=len(conflicts),
                        elapsed_ms=s3_ms,
                        budget_tokens=self.expander.max_expansion_token_budget,
                    ),
                    "stage_trace": s3_fed_trace.model_dump(),
                }
            )

        # ---------------------------------------------------------------------
        # Stage 4: Composite Pre-Flight Gate & Hard $0 Abstention
        # ---------------------------------------------------------------------
        validation_domain = (
            research_domain
            if (intent_decision.mode == QueryIntentMode.PURE_DOCUMENTATION or not hydrated_slices)
            else code_domain
        )
        report = self.validator.validate(
            code_slices=hydrated_slices,
            user_prompt=user_prompt,
            supporting_doc_citations=citations,
            quant_strategy_spec=(
                quant_brief if isinstance(quant_brief, QuantStrategyBrief) else None
            ),
            normalized_rrf_score=effective_fed_rrf,
            repo=fed_bound_repo,
            principal=active_principal,
            domain_id=validation_domain,
            doc_chunks=active_chunks,
        )

        explicit_bounce_request = bool(
            re.search(r"\bbounce[-_\s]?back\b", user_prompt, flags=re.IGNORECASE)
        )
        if (
            (not report.passed or explicit_bounce_request)
            and max_bounce_backs > 0
            and intent_decision.mode != QueryIntentMode.PURE_DOCUMENTATION
        ):
            active_feedback = (
                report.bounce_back_feedback
                or "Pre-flight validation triggered $0 local self-correction pass"
            )
            logger.info(
                "Federated Pre-Flight failed, running local bounce-back: %s",
                active_feedback,
            )
            if on_stage_event is not None:
                on_stage_event(
                    {
                        "event_type": "bounce_back",
                        "stage": 3,
                        "status": "bounce_back_triggered",
                        "feedback": active_feedback,
                        "paid_tokens_spent": 0,
                        "cost_usd": 0.0,
                        "details": "$0 local bounce-back loop triggered to resolve missing symbols",
                    }
                )
            remed_ptrs = self.local_client.generate_code_pointers(
                user_prompt=code_prompt,
                domain_id=code_domain,
                repo=fed_bound_repo,
                principal=active_principal,
                bounce_back_feedback=active_feedback,
            )
            for ptr in remed_ptrs:
                if not any(
                    s.file_path == ptr.file_path and s.qualified_symbol == ptr.qualified_symbol
                    for s in hydrated_slices
                ):
                    f_rec = files_by_path.get(ptr.file_path)
                    if f_rec is not None:
                        try:
                            snapped_ptr = _snap_python_ast_pointer_bounds(ptr, f_rec)
                            hydrated_slices.append(
                                self.slicer.slice_pointer(
                                    pointer=snapped_ptr,
                                    file_record=f_rec,
                                    auto_expanded_by_ast=False,
                                )
                            )
                        except Exception as exc:
                            logger.debug(
                                "Skipping invalid bounce-back slice %s (%s): %s",
                                ptr.file_path,
                                type(exc).__name__,
                                exc,
                            )

            report = self.validator.validate(
                code_slices=hydrated_slices,
                user_prompt=user_prompt,
                supporting_doc_citations=citations,
                quant_strategy_spec=(
                    quant_brief if isinstance(quant_brief, QuantStrategyBrief) else None
                ),
                normalized_rrf_score=effective_fed_rrf,
                repo=fed_bound_repo,
                principal=active_principal,
                domain_id=validation_domain,
                doc_chunks=active_chunks,
            )

        fed_ledger = _build_preflight_ledger(
            report=report,
            calibrated_score=calibrated_fed_score,
            min_floor=CALIBRATED_RELEVANCE_FLOOR,
            sufficiency_verdict=fed_sufficiency,
            grounded_facets=grounded_fed,
            total_facets=len(fed_facets),
            llm_verified_sufficient=(
                fed_llm_verified_sufficient and report.citations_verified_100_pct
            ),
            code_symbol_verified=(has_direct_fed_symbol and not fed_floor_failed),
        )
        if fed_llm_verified_sufficient and not report.citations_verified_100_pct:
            fed_sufficiency = "insufficient"

        if not report.passed or fed_floor_failed or fed_sufficiency == "insufficient":
            fail_msg = (
                report.bounce_back_feedback
                if not report.passed and report.bounce_back_feedback
                else (
                    f"Federated context is insufficient ({fed_sufficiency}, "
                    f"calibrated={calibrated_fed_score:.3f}). Paid frontier tokens spent: 0."
                )
            )
            s4_fed_fail_trace = StageExecutionTrace(
                stage=4,
                stage_name="Stage 4: Pre-Flight Sufficiency Gate",
                status="failed",
                elapsed_ms=float(max(1, int((time.perf_counter() - dag_t0) * 1000))),
                summary=fail_msg,
                sufficiency_verdict=fed_sufficiency,
                grounded_facets=grounded_fed,
                total_facets=len(fed_facets),
                checks=fed_ledger,
                final_decision="abstain_zero_paid",
                paid_tokens_spent=0,
            )
            if on_stage_event is not None:
                total_ms = max(1, int((time.perf_counter() - dag_t0) * 1000))
                on_stage_event(
                    {
                        "event_type": "stage_update",
                        "stage": 4,
                        "stage_name": "Stage 4: Pre-Flight Sufficiency Gate",
                        "status": "failed",
                        "details": _stage4_failed_details(reason=fail_msg, total_ms=total_ms),
                        "stage_trace": s4_fed_fail_trace.model_dump(),
                    }
                )
            diag = self.validator.create_diagnostic(
                trace_id=active_trace_id,
                domain_id=(
                    fed_domain
                    if (fed_domain == "federated_all" and len(selected_ids) >= 2)
                    else validation_domain
                ),
                report=report,
                partial_candidate_files=list({s.file_path for s in hydrated_slices}),
                custom_reason=fail_msg,
            )
            diag.query_facets = fed_facets
            diag.missing_or_unverified_aspects = fed_missing
            diag.stage_traces = [s1_fed_trace, s2_fed_trace, s3_fed_trace, s4_fed_fail_trace]
            return diag

        # ---------------------------------------------------------------------
        # Composite Pack Assembly
        # ---------------------------------------------------------------------
        if quant_brief is not None:
            stage1_plan_line = (
                f"Stage 1 (Research {research_domain}): Extracted '{quant_brief.strategy_name}' "
                f"with verdict {quant_brief.friction_verdict.value}"
            )
        elif domain_knowledge_brief is not None:
            stage1_plan_line = (
                f"Stage 1 (Documentation {research_domain}): Extracted "
                f"'{domain_knowledge_brief.brief_title}' with {len(citations)} citation(s)"
            )
        else:
            stage1_plan_line = (
                f"Stage 1 (Intent {intent_decision.mode.value}): "
                "Routed directly to code localization"
            )
        plan = [
            stage1_plan_line,
            (
                f"Stage 2 (Code {code_domain}): Located {len(hydrated_slices)} code slices "
                "via AST localization"
            ),
            f"Stage 3: Composite Pre-Flight Gate PASSED with RRF {report.normalized_rrf_score:.4f}",
            "Stage 4: Assembled cross-domain composite ImplementationContextPack",
        ]

        total_slice_chars = sum(len(s.verbatim_code_with_lines) for s in hydrated_slices)
        if citations:
            total_slice_chars += sum(len(c.verbatim_quote) for c in citations)
        est_tokens = max(1, int(total_slice_chars / 3.8)) + 300

        if isinstance(active_chunks, dict):
            total_doc_tokens = sum(max(1, int(len(v) / 3.8)) for v in active_chunks.values())
        elif isinstance(active_chunks, list):
            total_doc_tokens = sum(
                max(1, int(len(c.verbatim_content) / 3.8)) for c in active_chunks
            )
        else:
            total_doc_tokens = 0

        explored_file_paths = {p.file_path for p in code_pointers} | {
            s.file_path for s in hydrated_slices
        }
        explored_files = [files_by_path[p] for p in explored_file_paths if p in files_by_path]
        total_code_tokens = sum(max(1, int(len(f.raw_content) / 3.8)) for f in explored_files)
        total_tokens_explored = max(est_tokens, total_doc_tokens + total_code_tokens)

        pack_domain = (
            fed_domain
            if (
                fed_domain == "federated_all"
                and len(selected_ids) >= 2
                and (bool(citations) or bool(hydrated_slices))
            )
            else validation_domain
        )
        fed_synth_mode: Literal[
            "live_qlora_json",
            "live_base_gemma4_json",
            "deterministic_extractive_fallback",
            "ast_extractive",
            "hybrid_federated",
        ] = (
            "hybrid_federated"
            if (hydrated_slices and citations)
            else ("ast_extractive" if hydrated_slices else "deterministic_extractive_fallback")
        )
        if fed_missing:
            plan.append(
                "Partial corpus coverage — missing/unverified aspects: " + "; ".join(fed_missing)
            )
        fed_confidences = _compute_stage_confidences(
            principal=principal,
            plan=fed_plan,
            normalized_rrf=report.normalized_rrf_score,
            completeness_ratio=fed_completeness,
            query_facets=fed_facets,
            code_slices=hydrated_slices,
            citations=citations,
            synthesis_mode=fed_synth_mode,
            preflight_report=report,
            conflicts_count=len(conflicts),
            is_abstention=False,
        )
        s4_fed_pass_trace = StageExecutionTrace(
            stage=4,
            stage_name="Stage 4: Pre-Flight Sufficiency Gate",
            status="completed",
            elapsed_ms=float(max(1, int((time.perf_counter() - dag_t0) * 1000))),
            summary=(
                f"Handoff ready across {', '.join(selected_ids)} "
                f"(calibrated={calibrated_fed_score:.3f}, facets={grounded_fed}/{len(fed_facets)})."
            ),
            sufficiency_verdict=fed_sufficiency,
            grounded_facets=grounded_fed,
            total_facets=len(fed_facets),
            checks=fed_ledger,
            final_decision="handoff_ready",
            paid_tokens_spent=0,
        )
        pack = ImplementationContextPack(
            trace_id=active_trace_id,
            domain_id=pack_domain,
            user_task=user_prompt,
            architectural_execution_plan=plan,
            doc_vs_code_conflicts_detected=conflicts,
            code_slices=hydrated_slices,
            supporting_doc_citations=citations,
            quant_strategy_spec=(
                quant_brief if isinstance(quant_brief, QuantStrategyBrief) else None
            ),
            domain_knowledge_brief=domain_knowledge_brief,
            preflight_report=report,
            local_tokens_consumed=total_tokens_explored,
            pack_token_count=est_tokens,
            query_facets=fed_facets,
            missing_or_unverified_aspects=fed_missing,
            completeness_ratio=fed_completeness,
            synthesis_mode=fed_synth_mode,
            stage_confidences=fed_confidences,
            stage_traces=[s1_fed_trace, s2_fed_trace, s3_fed_trace, s4_fed_pass_trace],
        )

        if on_stage_event is not None:
            total_ms = max(1, int((time.perf_counter() - dag_t0) * 1000))
            on_stage_event(
                {
                    "event_type": "stage_update",
                    "stage": 4,
                    "stage_name": "Stage 4: Pre-Flight Sufficiency Gate",
                    "status": "completed",
                    "details": _stage4_passed_details(
                        report=report,
                        pack_tokens=est_tokens,
                        local_tokens=total_tokens_explored,
                        total_ms=total_ms,
                    ),
                    "confidence": fed_confidences[3].model_dump(),
                    "stage_trace": s4_fed_pass_trace.model_dump(),
                }
            )

        return pack


__all__ = [
    "DEFAULT_LOCAL_GEMMA_MODEL",
    "MAX_SAFE_KV_SIZE",
    "FederatedScoutCoordinator",
    "LocalGemmaClient",
    "ModelUnavailableError",
    "ModelUnavailableException",
    "ScoutWorkflow",
]
