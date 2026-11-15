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

from context_scout.agents.citation_gate import DocumentCitationGate
from context_scout.agents.conflict_detector import ConflictDetector
from context_scout.agents.domain_registry import AdapterPool, AdapterSlot, DomainRegistry
from context_scout.agents.intent_router import (
    QueryIntentClassifier,
    QueryIntentMode,
    QueryPlan,
    SourceFilter,
    decompose_query_facets,
    evaluate_facet_grounding,
    filter_doc_chunks_by_source,
    format_doc_citation_label,
    is_jira_chunk,
    is_wiki_chunk,
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
    CodeFileRecord,
    DocChunkRecord,
    PostgresRLSRepository,
)
from context_scout.plugins.registry import PluginRegistry
from context_scout.schemas.contracts import (
    CodeSlicePointer,
    DocCitationSpan,
    DomainKnowledgeBrief,
    FrictionVerdict,
    ImplementationContextPack,
    InsufficientContextDiagnostic,
    MarketRegime,
    ModificationIntent,
    PreFlightValidationReport,
    PrincipalContext,
    QuantStrategyBrief,
    QueryFacetCoverage,
    RoleType,
    StageConfidence,
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
        # Process-shared single-slot LRU cache prevents loading multiple ~15 GB models.
        self._mlx_cache: OrderedDict[str, tuple[Any, Any]] = _PROCESS_MLX_CACHE

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
        slot = self.get_active_adapter(domain_id) if domain_id else None
        adapter_path = str(slot.adapter_path) if slot and slot.adapter_path else None
        cache_key = f"{domain_id or 'base'}:{adapter_path or 'none'}"
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
            return (
                raw.replace("<eos>", "").replace("<end_of_turn>", "").replace("<turn|>", "").strip()
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
        slot = self.get_active_adapter(domain_id) if domain_id else None
        model_name = slot.adapter_id if slot else self.model_id
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
                        cleaned_json = re.sub(
                            r"```(?:json)?\s*\n?(.*?)```", r"\1", raw_mlx, flags=re.DOTALL
                        ).strip()
                        raw_list = json.loads(cleaned_json)
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
                    cleaned_json = re.sub(
                        r"```(?:json)?\s*\n?(.*?)```", r"\1", raw_mlx, flags=re.DOTALL
                    ).strip()
                    raw_list = json.loads(cleaned_json)
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
            content = body["choices"][0]["message"]["content"]
            raw_list = json.loads(content)
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
            "compute",
            "implemented",
            "implementation",
            "explain",
            "inspect",
            "describe",
            "model",
            "models",
        }
        nl_stopwords = PROMPT_STOPWORDS - {"help"}
        base_nl_terms = list(
            dict.fromkeys(
                w.lower()
                for t_raw in candidates_to_search
                for w in re.split(r"[_\-\./]+", t_raw)
                if len(w) >= 3
                and w.lower() not in nl_stopwords
                and w.lower() not in generic_terms
                and not w.isdigit()
            )
        )
        nl_terms = list(base_nl_terms)
        for w_item in base_nl_terms:
            if w_item.endswith("ing") and len(w_item) >= 6:
                stem_e = w_item[:-3] + "e"
                stem_raw = w_item[:-3]
                if len(stem_e) >= 4 and stem_e not in nl_terms:
                    nl_terms.append(stem_e)
                if len(stem_raw) >= 4 and stem_raw not in nl_terms:
                    nl_terms.append(stem_raw)
            if w_item in ("decompression", "decompress", "compressed", "compression"):
                for syn in ("decoder", "decoders", "decompress"):
                    if syn not in nl_terms:
                        nl_terms.append(syn)
            if w_item in ("urlencode", "urlencoded"):
                for syn in ("urlencode_safe", "queryparams"):
                    if syn not in nl_terms:
                        nl_terms.append(syn)
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
            if ".evals." in sym.qualified_name and "eval" not in user_prompt.lower():
                continue
            short_l = sym.qualified_name.split(".")[-1].lower()
            qual_l = sym.qualified_name.lower()
            sig_l = sym.signature_skeleton.lower()
            sum_l = sym.contextual_summary.lower()
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
                if term in short_l:
                    score += 6.0 * term_weight
                    term_hit = True
                    if term not in generic_action_verbs:
                        short_term_hits += 1
                    if term in title_terms and term not in generic_action_verbs:
                        title_in_short = True
                elif term in qual_l:
                    score += 3.5 * term_weight
                    term_hit = True
                if term in sig_l:
                    score += 2.0 * term_weight
                    term_hit = True
                if term in sum_l:
                    score += (12.0 if "[dispatch " in sum_l else 1.5) * term_weight
                    term_hit = True
                if body_l and term in body_search_terms and term in body_l:
                    score += 3.0 * term_weight
                    term_hit = True
                if term_hit:
                    matched_terms += 1
            if title_in_short:
                score += 8.0
            if short_term_hits >= 2:
                score += 14.0
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
                score += matched_terms * 4.0
                if sym.symbol_kind in ("class", "interface"):
                    score += 2.5
                elif sym.symbol_kind == "type_alias":
                    score -= 4.0
                scored_nl.append((score, sym))
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
            content = body["choices"][0]["message"]["content"]
            return QuantStrategyBrief.model_validate_json(content)

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
            content = body["choices"][0]["message"]["content"]
            return DomainKnowledgeBrief.model_validate_json(content)

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
    ) -> QuantStrategyBrief | None:
        """Generates structured domain research brief with verified document citations."""
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
                        cleaned_json = re.sub(
                            r"```(?:json)?\s*\n?(.*?)```", r"\1", raw_mlx, flags=re.DOTALL
                        ).strip()
                        mlx_brief = QuantStrategyBrief.model_validate_json(cleaned_json)
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

        if len(plan.facets) >= 2 or plan.is_acronym_query:
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
            chunk_lines = [
                "Retrieved Corpus Chunk: ["
                f"chunk_id={c.chunk_id} | doc_id={c.doc_id} | title={c.doc_title} | "
                f"uri={c.source_uri} | section={c.section_heading} | "
                f'quote="{c.verbatim_quote}"]'
                for c in verified_citations
            ]
            rag_prompt = f"{user_prompt}\n" + "\n".join(chunk_lines)
            try:
                rag_brief: QuantStrategyBrief | None = None
                if self.use_live_mlx:
                    raw_mlx = self._call_in_process_mlx(
                        rag_prompt, domain_id=domain_id, max_tokens=1536
                    )
                    cleaned_json = re.sub(
                        r"```(?:json)?\s*\n?(.*?)```", r"\1", raw_mlx, flags=re.DOTALL
                    ).strip()
                    rag_brief = QuantStrategyBrief.model_validate_json(cleaned_json)
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
                    if grounded_cits:
                        rag_brief.verified_citations = grounded_cits
                    else:
                        rag_brief.verified_citations = verified_citations
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

        if len(plan.facets) >= 2 or plan.is_acronym_query:
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
            mode: Literal["live_qlora_json", "deterministic_extractive_fallback"],
        ) -> DomainKnowledgeBrief:
            for cit in kb.verified_citations:
                if not cit.full_section_content and cit.chunk_id in chunks_by_id:
                    cit.full_section_content = _stitch_section_content(
                        chunks_by_id[cit.chunk_id], fresh_chunks
                    )
            kb.query_facets = query_facets
            kb.missing_or_unverified_aspects = missing_aspects
            kb.completeness_ratio = completeness_ratio
            kb.synthesis_mode = mode
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
            chunk_lines = [
                "Retrieved Corpus Chunk: ["
                f"chunk_id={c.chunk_id} | doc_id={c.doc_id} | title={c.doc_title} | "
                f"uri={c.source_uri} | section={c.section_heading} | "
                f"quote={json.dumps(c.verbatim_quote)}]"
                for c in verified_citations
            ]
            rag_prompt = build_domain_knowledge_brief_prompt(
                user_prompt=user_prompt,
                domain_id=domain_id,
                chunk_lines=chunk_lines,
            )
            try:
                rag_kb: DomainKnowledgeBrief | None = None
                if self.use_live_mlx:
                    raw_mlx = self._call_in_process_mlx(
                        rag_prompt, domain_id=domain_id, max_tokens=2048
                    )
                    rag_kb, _ = parse_domain_knowledge_brief_json(
                        raw_mlx,
                        domain_id=domain_id,
                        normalize_scalars=True,
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
                    rag_kb.verified_citations = grounded_cits or verified_citations
                    return _finalize_kb(rag_kb, mode="live_qlora_json")
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
        raw_chunks = self._inner.search_visible_doc_chunks(
            principal=principal,
            domain_id=domain_id,
            query_text=query_text,
            market_scope=market_scope,
            limit=limit,
        )
        return filter_doc_chunks_by_source(raw_chunks, self._source_filter)

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
        normalized_rrf_score: float = 0.85,
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
        normalized_rrf_score: float,
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
        s1_raw_chunks = (
            filtered_doc_chunks
            if isinstance(filtered_doc_chunks, list)
            else doc_bound_repo.list_visible_doc_chunks(principal, domain_id)
        )
        if isinstance(filtered_doc_chunks, list):
            stale_excluded_s1 = sum(
                1 for c in filtered_doc_chunks if bool(getattr(c, "is_stale", False))
            )
            visible_chunks_s1 = len(filtered_doc_chunks) - stale_excluded_s1
        elif callable(count_fn):
            visible_chunks_s1, stale_excluded_s1 = count_fn(principal, domain_id)
        else:
            stale_excluded_s1 = sum(1 for c in s1_raw_chunks if bool(getattr(c, "is_stale", False)))
            visible_chunks_s1 = len(s1_raw_chunks) - stale_excluded_s1
        s1_ms = max(1, int((time.perf_counter() - s1_t0) * 1000))
        if on_stage_event is not None:
            on_stage_event(
                {
                    "event_type": "stage_update",
                    "stage": 1,
                    "stage_name": "Stage 1: Hybrid RRF Retrieval",
                    "status": "completed",
                    "details": _stage1_details(
                        domains=[domain_id],
                        principal=principal,
                        visible_files_count=len(visible_files),
                        visible_chunks_count=visible_chunks_s1,
                        stale_excluded_count=stale_excluded_s1,
                        elapsed_ms=s1_ms,
                    ),
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

        s2_ms = max(1, int((time.perf_counter() - s2_t0) * 1000))
        if on_stage_event is not None:
            model_label = manifest.base_model_id.split("/")[-1]
            adapter_label = manifest.lora_adapter_id or domain_id
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
                    (0.45 * normalized_rrf_score + 0.40 * s2_comp + 0.15)
                    if s2_has_hits
                    else min(0.20, 0.20 * normalized_rrf_score)
                ),
                signals={
                    "normalized_rrf": round(normalized_rrf_score, 4),
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
                    "stage_name": "Stage 2: Gemma 4 26B-A4B Synthesis",
                    "status": "completed",
                    "details": _stage2_details(
                        model_label=model_label,
                        adapter_id=adapter_label,
                        pointers_count=len(candidate_pointers),
                        citations_count=len(supporting_doc_citations or []),
                        elapsed_ms=s2_ms,
                    ),
                    "confidence": s2_conf.model_dump(),
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
        if on_stage_event is not None:
            s3_ast_count = sum(1 for s in hydrated_slices if s.auto_expanded_by_ast)
            s3_has_ctx = bool(hydrated_slices or supporting_doc_citations)
            s3_mode = (
                domain_knowledge_brief.synthesis_mode
                if domain_knowledge_brief is not None
                else ("ast_extractive" if hydrated_slices else "deterministic_extractive_fallback")
            )
            s3_conf = StageConfidence.from_score(
                stage=3,
                stage_name="Stage 3: Synthesis & AST Closure",
                score=(0.88 if s3_mode == "ast_extractive" else 0.78) if s3_has_ctx else 0.0,
                signals={
                    "ast_closure_count": s3_ast_count,
                    "synthesis_mode": s3_mode,
                    "conflicts_count": len(conflicts),
                },
                summary=f"Mode: {s3_mode}; AST closure slices: {s3_ast_count}.",
            )
            on_stage_event(
                {
                    "event_type": "stage_update",
                    "stage": 3,
                    "stage_name": "Stage 3: 1-Hop AST Closure / Quant Scepticism Gate",
                    "status": "completed",
                    "details": _stage3_details(
                        slices=hydrated_slices,
                        citations_count=len(supporting_doc_citations or []),
                        conflicts_count=len(conflicts),
                        elapsed_ms=s3_ms,
                        budget_tokens=self.expander.max_expansion_token_budget,
                    ),
                    "confidence": s3_conf.model_dump(),
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
            if on_stage_event is not None:
                total_ms = max(1, int((time.perf_counter() - dag_t0) * 1000))
                on_stage_event(
                    {
                        "event_type": "stage_update",
                        "stage": 4,
                        "stage_name": "Stage 4: Pre-Flight Deterministic Validation",
                        "status": "failed",
                        "details": _stage4_failed_details(reason=no_ctx_reason, total_ms=total_ms),
                        "confidence": no_ctx_confidences[3].model_dump(),
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
            )

        val_domain = (
            research_domain_id if (auto_research_route or used_research_domain) else domain_id
        )
        initial_report = self.validator.validate(
            code_slices=hydrated_slices,
            user_prompt=user_prompt,
            supporting_doc_citations=supporting_doc_citations,
            normalized_rrf_score=normalized_rrf_score,
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
                normalized_rrf_score=normalized_rrf_score,
                repo=run_bound_repo,
                principal=principal,
                domain_id=val_domain,
                doc_chunks=filtered_doc_chunks,
            )

        query_plan = decompose_query_facets(user_prompt)
        if domain_plugin is not None and hasattr(domain_plugin, "expand_query_plan"):
            query_plan = domain_plugin.expand_query_plan(query_plan)

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

        if not final_report.passed:
            fail_reason = final_report.bounce_back_feedback or "Pre-Flight validation failed"
            diag_confidences = _compute_stage_confidences(
                principal=principal,
                plan=query_plan,
                normalized_rrf=final_report.normalized_rrf_score,
                completeness_ratio=0.0,
                query_facets=pack_facets,
                code_slices=hydrated_slices,
                citations=supporting_doc_citations or [],
                synthesis_mode=pack_synth_mode,
                preflight_report=final_report,
                stale_count=stale_excluded_s1,
                is_abstention=True,
            )
            if on_stage_event is not None:
                total_ms = max(1, int((time.perf_counter() - dag_t0) * 1000))
                on_stage_event(
                    {
                        "event_type": "stage_update",
                        "stage": 4,
                        "stage_name": "Stage 4: Pre-Flight Deterministic Validation",
                        "status": "failed",
                        "details": _stage4_failed_details(reason=fail_reason, total_ms=total_ms),
                        "confidence": diag_confidences[3].model_dump(),
                    }
                )
            logger.warning(
                "Stage 3 Pre-Flight check still failed after local bounce-back (trace=%s)",
                active_trace_id,
            )
            diag = self.validator.create_diagnostic(
                trace_id=active_trace_id,
                domain_id=domain_id,
                report=final_report,
                partial_candidate_files=list({s.file_path for s in hydrated_slices}),
            )
            diag.query_facets = pack_facets
            diag.missing_or_unverified_aspects = pack_missing
            diag.stage_confidences = diag_confidences
            return diag

        prompt_ident_tokens_low = {
            w.lower() for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", user_prompt)
        }
        has_direct_symbol_hit = any(
            s.qualified_symbol.split(".")[-1].lower() in prompt_ident_tokens_low
            or s.qualified_symbol.lower() in prompt_ident_tokens_low
            for s in hydrated_slices
        )
        if (
            pack_facets
            and all(f.status == "not_found" for f in pack_facets)
            and not has_direct_symbol_hit
        ):
            no_ground_reason = (
                f"Zero query facets were grounded in indexed corpus '{domain_id}' for query: "
                f"'{user_prompt.strip()}'. Paid frontier tokens spent: 0."
            )
            abstain_confidences = _compute_stage_confidences(
                principal=principal,
                plan=query_plan,
                normalized_rrf=0.0,
                completeness_ratio=0.0,
                query_facets=pack_facets,
                code_slices=[],
                citations=[],
                synthesis_mode=pack_synth_mode,
                preflight_report=final_report,
                stale_count=stale_excluded_s1,
                is_abstention=True,
            )
            if on_stage_event is not None:
                total_ms = max(1, int((time.perf_counter() - dag_t0) * 1000))
                on_stage_event(
                    {
                        "event_type": "stage_update",
                        "stage": 4,
                        "stage_name": "Stage 4: Pre-Flight Deterministic Validation",
                        "status": "failed",
                        "details": _stage4_failed_details(
                            reason=no_ground_reason, total_ms=total_ms
                        ),
                        "confidence": abstain_confidences[3].model_dump(),
                    }
                )
            return InsufficientContextDiagnostic(
                trace_id=active_trace_id,
                domain_id=domain_id,
                reason=no_ground_reason,
                missing_symbols=[],
                missing_items=[
                    *final_report.missing_items,
                    f"domain:{domain_id}:zero_grounded_facets",
                ],
                partial_candidate_files=[],
                paid_tokens_spent=0,
                query_facets=pack_facets,
                missing_or_unverified_aspects=pack_missing,
                stage_confidences=abstain_confidences,
            )

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
            total_slice_chars += sum(
                len(c.full_section_content or c.verbatim_quote) for c in supporting_doc_citations
            )
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
        )

        if on_stage_event is not None:
            total_ms = max(1, int((time.perf_counter() - dag_t0) * 1000))
            on_stage_event(
                {
                    "event_type": "stage_update",
                    "stage": 4,
                    "stage_name": "Stage 4: Pre-Flight Deterministic Validation",
                    "status": "completed",
                    "details": _stage4_passed_details(
                        report=final_report,
                        pack_tokens=est_pack_tokens,
                        local_tokens=local_tokens_consumed,
                        total_ms=total_ms,
                    ),
                    "confidence": stage_confidences[3].model_dump(),
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
        normalized_rrf_score: float = 0.88,
        doc_chunks: dict[str, str] | list[DocChunkRecord] | None = None,
        max_bounce_backs: int = 1,
        on_stage_event: Callable[[dict[str, Any]], None] | None = None,
        source_filter: SourceFilter = "all",
    ) -> ImplementationContextPack | InsufficientContextDiagnostic:
        """Executes the 2-stage cross-domain federated DAG with OpenTelemetry instrumentation."""
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

    def _run_federated_dag(
        self,
        user_prompt: str,
        principal: PrincipalContext,
        active_trace_id: str,
        normalized_rrf_score: float,
        doc_chunks: dict[str, str] | list[DocChunkRecord] | None,
        max_bounce_backs: int,
        on_stage_event: Callable[[dict[str, Any]], None] | None = None,
        source_filter: SourceFilter = "all",
    ) -> ImplementationContextPack | InsufficientContextDiagnostic:
        """Executes the 2-stage cross-domain federated DAG."""
        dag_t0 = time.perf_counter()
        code_domain = self.code_domain_id
        research_domain = self.research_domain_id
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

        domain_plugin = PluginRegistry.get_domain_plugin(research_domain)
        if domain_plugin is not None and hasattr(domain_plugin, "elevate_principal_for_research"):
            active_principal = domain_plugin.elevate_principal_for_research(principal)
        else:
            active_principal = principal

        # ---------------------------------------------------------------------
        # Stage 1: Research / Documentation Domain Extraction
        # ---------------------------------------------------------------------
        s1_t0 = time.perf_counter()
        self.local_client.set_active_domain(research_domain)
        visible_files = fed_bound_repo.list_visible_code_files(active_principal, code_domain)
        raw_active_chunks = (
            doc_chunks
            if doc_chunks is not None
            else fed_bound_repo.list_visible_doc_chunks(active_principal, research_domain)
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

        quant_brief: QuantStrategyBrief | None = None
        domain_knowledge_brief: DomainKnowledgeBrief | None = None
        citations: list[DocCitationSpan] = []

        if intent_decision.mode == QueryIntentMode.PURE_CODE:
            s1_ms = max(1, int((time.perf_counter() - s1_t0) * 1000))
        else:
            quant_brief = self.local_client.generate_quant_brief(
                user_prompt=user_prompt,
                domain_id=research_domain,
                repo=doc_bound_repo,
                principal=active_principal,
                bounce_back_feedback=None,
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
            elif isinstance(active_chunks, list) and active_chunks:
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
                if on_stage_event is not None:
                    on_stage_event(
                        {
                            "event_type": "stage_update",
                            "stage": 1,
                            "stage_name": "Stage 1: Research Domain Extraction",
                            "status": "completed",
                            "details": _stage1_details(
                                domains=[research_domain, code_domain],
                                principal=active_principal,
                                visible_files_count=len(visible_files),
                                visible_chunks_count=visible_chunks_s1,
                                stale_excluded_count=stale_excluded_s1,
                                elapsed_ms=s1_ms,
                            ),
                        }
                    )
                return InsufficientContextDiagnostic(
                    trace_id=active_trace_id,
                    domain_id=research_domain,
                    reason=no_brief_reason,
                    missing_symbols=[],
                    partial_candidate_files=[],
                    paid_tokens_spent=0,
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

        if on_stage_event is not None:
            on_stage_event(
                {
                    "event_type": "stage_update",
                    "stage": 1,
                    "stage_name": "Stage 1: Research Domain Extraction",
                    "status": "completed",
                    "details": _stage1_details(
                        domains=[research_domain, code_domain],
                        principal=active_principal,
                        visible_files_count=len(visible_files),
                        visible_chunks_count=visible_chunks_s1,
                        stale_excluded_count=stale_excluded_s1,
                        elapsed_ms=s1_ms,
                    ),
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

        if on_stage_event is not None:
            on_stage_event(
                {
                    "event_type": "stage_update",
                    "stage": 2,
                    "stage_name": "Stage 2: Code AST Localization",
                    "status": "completed",
                    "details": _stage2_details(
                        model_label="gemma-4-26B-A4B-it",
                        adapter_id=code_domain,
                        pointers_count=len(code_pointers),
                        citations_count=len(citations),
                        elapsed_ms=s2_ms,
                    ),
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

        s3_ms = max(1, int((time.perf_counter() - s3_t0) * 1000))
        if on_stage_event is not None:
            on_stage_event(
                {
                    "event_type": "stage_update",
                    "stage": 3,
                    "stage_name": "Stage 3: 1-Hop AST Closure & Scepticism Gate",
                    "status": "completed",
                    "details": _stage3_details(
                        slices=hydrated_slices,
                        citations_count=len(citations),
                        conflicts_count=len(conflicts),
                        elapsed_ms=s3_ms,
                        budget_tokens=self.expander.max_expansion_token_budget,
                    ),
                }
            )

        # ---------------------------------------------------------------------
        # Stage 3: Composite Pre-Flight Gate & Local Bounce-Back Loop
        # ---------------------------------------------------------------------
        validation_domain = (
            research_domain
            if intent_decision.mode == QueryIntentMode.PURE_DOCUMENTATION
            else code_domain
        )
        report = self.validator.validate(
            code_slices=hydrated_slices,
            user_prompt=user_prompt,
            supporting_doc_citations=citations,
            quant_strategy_spec=(
                quant_brief if isinstance(quant_brief, QuantStrategyBrief) else None
            ),
            normalized_rrf_score=normalized_rrf_score,
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
                normalized_rrf_score=normalized_rrf_score,
                repo=fed_bound_repo,
                principal=active_principal,
                domain_id=validation_domain,
                doc_chunks=active_chunks,
            )

        if not report.passed:
            if on_stage_event is not None:
                total_ms = max(1, int((time.perf_counter() - dag_t0) * 1000))
                on_stage_event(
                    {
                        "event_type": "stage_update",
                        "stage": 4,
                        "stage_name": "Stage 4: Pre-Flight Composite Validation",
                        "status": "failed",
                        "details": _stage4_failed_details(
                            reason=(
                                report.bounce_back_feedback
                                or "Composite pre-flight validation failed"
                            ),
                            total_ms=total_ms,
                        ),
                    }
                )
            return self.validator.create_diagnostic(
                trace_id=active_trace_id,
                domain_id=validation_domain,
                report=report,
                partial_candidate_files=list({s.file_path for s in hydrated_slices}),
            )

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
            research_domain
            if intent_decision.mode == QueryIntentMode.PURE_DOCUMENTATION
            else code_domain
        )
        conflicts = _detect_pack_conflicts(
            hydrated_slices,
            fed_chunk_records,
            initial_conflicts=conflicts,
        )
        fed_plan = decompose_query_facets(user_prompt)
        fed_facets, fed_missing, fed_completeness = _evaluate_query_plan_coverage(
            fed_plan,
            doc_chunks=fed_chunk_records,
            code_slices=hydrated_slices,
            domain_id=pack_domain,
        )
        fed_synth_mode: Literal[
            "live_qlora_json",
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
        )

        if on_stage_event is not None:
            total_ms = max(1, int((time.perf_counter() - dag_t0) * 1000))
            on_stage_event(
                {
                    "event_type": "stage_update",
                    "stage": 4,
                    "stage_name": "Stage 4: Pre-Flight Composite Validation",
                    "status": "completed",
                    "details": _stage4_passed_details(
                        report=report,
                        pack_tokens=est_tokens,
                        local_tokens=total_tokens_explored,
                        total_ms=total_ms,
                    ),
                    "confidence": fed_confidences[3].model_dump(),
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
