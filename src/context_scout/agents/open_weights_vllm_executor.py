"""Multi-architecture Open-Weights vLLM/SGLang & Apple Silicon MLX-LM Executor (M4-01).

Implements ``LLMBackendSPI`` for self-hosted GKE/K8s GPU ``vLLM``/``SGLang`` clusters and
local Apple Silicon ``mlx-lm`` development across ``google/gemma-4-*``,
``mlx-community/gemma-4-*``, and BYO open-weights model families (``Qwen3``, ``Qwen3-Coder``,
``Devstral``/``Mistral``, ``DeepSeek-V3``/``R1``, ``Llama 4``, ``Phi-4``, ``Nemotron-H``,
``IBM Granite-3.3``).

Enforces:
- Supply-chain governance (``.safetensors``-only, ``trust_remote_code=False``).
- Dynamic ``max_model_len`` discovery via ``GET /v1/models``.
- Vendor-neutral chat formatting via ``AutoTokenizer.apply_chat_template`` (zero hardcoded
  Gemma ``<start_of_turn>`` tags).
- Reasoning trace separation delegated to ``vLLM --reasoning-parser``.
- Constrained JSON generation via ``xgrammar`` ``structural_tag`` schemas.
- Multi-tenant / multi-ACL KV prefix-cache salting via ``sha256(tenant_id + sorted_scopes)``.
- ``OWASP ASI06`` cross-ACL ``training_scope_ceiling`` subset routing via ``AdapterRouter``.
- ``vLLM`` multi-LoRA safety (``rank in {8, 16, 32}``, ``use_rslora=True``,
  ``use_dora=False``, frozen MoE router gates and Mamba SSM states).
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import os
import re
import struct
import threading
import uuid
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal
from uuid import UUID

import httpx
from jinja2.sandbox import ImmutableSandboxedEnvironment
from pydantic import BaseModel, ValidationError

from context_scout.agents.llm_backend import (
    AdapterRouteDecision,
    AdapterRouter,
    GPUOverloadError,
    GPUPriorityClass,
    LLMBackendSPI,
    RedisGPUQuotaManager,
    compute_adaptive_complexity_budget,
)
from context_scout.config import ModelGovernancePolicy
from context_scout.schemas.contracts import (
    InsufficientContextDiagnostic,
    PrincipalContext,
)

logger = logging.getLogger(__name__)

RuntimeBackendKind = Literal["vllm", "sglang", "mlx-lm", "auto"]
ModelTopologyKind = Literal["dense", "moe", "hybrid_mamba_moe"]

DEFAULT_VLLM_ENDPOINT: Final[str] = "http://127.0.0.1:8000/v1"
DEFAULT_STRUCTURAL_BEGIN_TAG: Final[str] = "<json_output>"
DEFAULT_STRUCTURAL_END_TAG: Final[str] = "</json_output>"
VALID_VLLM_LORA_RANKS: Final[frozenset[int]] = frozenset({8, 16, 32})

# Module name substrings that MUST remain frozen in LoRA cartridges (vLLM multi-LoRA safety):
# - MoE router / expert gating weights crash or destabilize vLLM Punica multi-LoRA kernels
# - Mamba-2 / hybrid SSM state-space parameters lack multi-LoRA kernel support in vLLM
FROZEN_MOE_ROUTER_PATTERNS: Final[tuple[str, ...]] = (
    ".router.",
    ".gate.weight",
    ".gate.lora_",
    ".shared_expert_gate.",
    ".block_sparse_moe.gate",
    ".mlp.gate.",
)
FROZEN_MAMBA_SSM_PATTERNS: Final[tuple[str, ...]] = (
    ".mamba.",
    ".ssm.",
    ".a_log",
    ".dt_proj.",
    ".conv1d.",
    ".d_param",
)

_INLINE_THINK_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*(?:<think>(?P<think>.*?)</think>|<\|thought\|>(?P<thought>.*?)(?:<\|end_thought\|>|<thought_end>))\s*",
    re.DOTALL,
)


class StructuredOutputValidationError(ValueError):
    """Raised fail-closed when model output violates the enforced JSON schema."""


class LoRACompatibilityError(ValueError):
    """Raised fail-closed when a LoRA cartridge violates vLLM multi-LoRA invariants."""


class OpenWeightsBackendUnavailableError(RuntimeError):
    """Raised fail-closed when vLLM/SGLang/mlx-lm is unreachable or returns an error."""

    def __init__(
        self,
        message: str,
        *,
        diagnostic: InsufficientContextDiagnostic | None = None,
        paid_tokens_spent: int = 0,
    ) -> None:
        super().__init__(message)
        self.diagnostic = diagnostic
        self.paid_tokens_spent = paid_tokens_spent


@dataclass(frozen=True)
class ModelArchitectureProfile:
    """Architecture, reasoning-parser, and frozen-module profile for a governed model."""

    model_id: str
    family: str
    topology: ModelTopologyKind
    reasoning_parser: str | None
    supports_xgrammar_structural_tag: bool = True
    frozen_module_patterns: tuple[str, ...] = ()


@dataclass(frozen=True)
class ParsedModelCompletion:
    """Structured result separating reasoning trace from verified content."""

    content: str
    reasoning_content: str = ""
    routed_model: str = ""
    lora_request: str | None = None
    cache_salt: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    route_decision: AdapterRouteDecision | None = None


@dataclass(frozen=True)
class VLLMLoRADryRunReport:
    """Verification report from a 1-step vLLM multi-LoRA compatibility check."""

    adapter_id: str
    base_model_id: str
    rank: int
    use_rslora: bool
    use_dora: bool
    safetensors_tensor_count: int
    frozen_router_and_ssm_verified: bool
    live_endpoint_probed: bool
    passed: bool
    detail: str = ""


def detect_architecture_profile(
    model_id: str,
    *,
    reasoning_parser_override: str | None = None,
) -> ModelArchitectureProfile:
    """Classifies governed local/BYO model IDs into topology and reasoning-parser profiles."""
    cleaned = model_id.strip()
    low = cleaned.lower()

    family = "generic-open-weights"
    topology: ModelTopologyKind = "dense"
    default_parser: str | None = None

    if "gemma-4" in low:
        family = "gemma-4"
        topology = "moe" if ("a4b" in low or "moe" in low) else "dense"
        default_parser = "gemma4"
    elif "qwen3-coder" in low:
        family = "Qwen3-Coder"
        topology = "moe" if re.search(r"-a\d+b", low) else "dense"
        default_parser = "qwen3"
    elif "qwen3" in low:
        family = "Qwen3"
        topology = "moe" if re.search(r"-a\d+b", low) else "dense"
        default_parser = "qwen3"
    elif "devstral" in low:
        family = "Devstral"
        topology = "dense"
        default_parser = "mistral"
    elif "mistral" in low or "mixtral" in low:
        family = "Mistral"
        topology = "moe" if "mixtral" in low else "dense"
        default_parser = "mistral"
    elif "deepseek-r1" in low:
        family = "DeepSeek-R1"
        topology = "moe"
        default_parser = "deepseek_r1"
    elif "deepseek-v3" in low:
        family = "DeepSeek-V3"
        topology = "moe"
        default_parser = "deepseek_r1"
    elif "llama-4" in low:
        family = "Llama-4"
        topology = "moe"
        default_parser = None
    elif "phi-4" in low:
        family = "Phi-4"
        topology = "dense"
        default_parser = "deepseek_r1" if "reasoning" in low else None
    elif "nemotron-h" in low:
        family = "Nemotron-H"
        topology = "hybrid_mamba_moe"
        default_parser = "nemotron"
    elif "granite-3.3" in low or "granite" in low:
        family = "IBM Granite-3.3"
        topology = "hybrid_mamba_moe" if "mamba" in low else "dense"
        default_parser = "granite"

    frozen: list[str] = []
    if topology in {"moe", "hybrid_mamba_moe"}:
        frozen.extend(FROZEN_MOE_ROUTER_PATTERNS)
    if topology == "hybrid_mamba_moe":
        frozen.extend(FROZEN_MAMBA_SSM_PATTERNS)

    resolved_parser = (
        reasoning_parser_override.strip()
        if reasoning_parser_override is not None and reasoning_parser_override.strip()
        else default_parser
    )
    if resolved_parser == "none":
        resolved_parser = None

    return ModelArchitectureProfile(
        model_id=cleaned,
        family=family,
        topology=topology,
        reasoning_parser=resolved_parser,
        supports_xgrammar_structural_tag=True,
        frozen_module_patterns=tuple(frozen),
    )


def compute_prefix_cache_salt(
    principal: PrincipalContext | None = None,
    *,
    tenant_id: UUID | str | None = None,
    allowed_scopes: Sequence[str] | None = None,
) -> str:
    """Computes ``sha256(tenant_id + sorted_scopes)`` to isolate KV prefix caches across ACLs."""
    resolved_tenant: str | None = None
    resolved_scopes: list[str] = []

    if principal is not None:
        resolved_tenant = str(principal.tenant_id).strip()
        resolved_scopes = sorted({s.strip() for s in principal.allowed_scopes if s and s.strip()})
    elif tenant_id is not None:
        resolved_tenant = str(UUID(str(tenant_id).strip()))
        resolved_scopes = sorted(
            {s.strip() for s in (allowed_scopes or ()) if s and str(s).strip()}
        )

    if not resolved_tenant:
        # Unattributed calls receive a one-time random salt so they never share tenant KV blocks.
        return hashlib.sha256(f"ephemeral:{uuid.uuid4()}".encode()).hexdigest()

    canonical_payload = json.dumps(
        {"tenant_id": resolved_tenant, "sorted_scopes": resolved_scopes},
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()


def inspect_safetensors_header(data_or_path: bytes | Path) -> dict[str, Any]:
    """Parses and validates a ``.safetensors`` binary header without pickle execution."""
    if isinstance(data_or_path, Path):
        if data_or_path.suffix.lower() != ".safetensors":
            raise LoRACompatibilityError(
                f"Expected .safetensors weight file, got '{data_or_path.name}'"
            )
        raw = data_or_path.read_bytes()
    else:
        raw = data_or_path

    if len(raw) < 8:
        raise LoRACompatibilityError("Corrupt .safetensors payload: shorter than 8-byte header.")
    header_len = struct.unpack("<Q", raw[:8])[0]
    # Guard against adversarial multi-GB header allocation bombs (max 64 MiB metadata header)
    if header_len == 0 or header_len > 64 * 1024 * 1024 or 8 + header_len > len(raw):
        raise LoRACompatibilityError(
            f"Invalid .safetensors header length ({header_len}) for payload of {len(raw)} bytes."
        )
    try:
        header = json.loads(raw[8 : 8 + header_len].decode("utf-8"))
    except Exception as exc:
        raise LoRACompatibilityError(f"Malformed UTF-8 JSON in .safetensors header: {exc}") from exc
    if not isinstance(header, dict):
        raise LoRACompatibilityError("Invalid .safetensors header: root must be a JSON object.")

    tensor_entries = {k: v for k, v in header.items() if k != "__metadata__"}
    if not tensor_entries:
        raise LoRACompatibilityError("Invalid .safetensors archive: contains zero tensor entries.")
    return header


def validate_lora_safetensors_for_vllm(
    weights_data_or_path: bytes | Path,
    adapter_config: Mapping[str, Any],
    *,
    base_model_id: str | None = None,
) -> tuple[int, tuple[str, ...]]:
    """Validates vLLM multi-LoRA invariants on ``adapter_config.json`` & ``.safetensors``."""
    if adapter_config.get("trust_remote_code") is True or "auto_map" in adapter_config:
        raise LoRACompatibilityError(
            "LoRA adapter_config prohibits trust_remote_code=True and auto_map."
        )
    if adapter_config.get("use_dora") is True:
        raise LoRACompatibilityError("use_dora=True is incompatible with vLLM multi-LoRA kernels.")
    if adapter_config.get("use_rslora") is not True:
        raise LoRACompatibilityError(
            "use_rslora=True is required for rank-stabilized vLLM multi-LoRA serving."
        )

    raw_rank = adapter_config.get("r", adapter_config.get("rank", 0))
    try:
        rank = int(raw_rank)
    except (TypeError, ValueError) as exc:
        raise LoRACompatibilityError(f"Invalid LoRA rank value: {raw_rank!r}") from exc
    if rank not in VALID_VLLM_LORA_RANKS:
        raise LoRACompatibilityError(
            f"LoRA rank {rank} violates vLLM multi-LoRA policy; expected one of "
            f"{sorted(VALID_VLLM_LORA_RANKS)} (r <= 32)."
        )

    header = inspect_safetensors_header(weights_data_or_path)
    tensor_names = tuple(sorted(k for k in header if k != "__metadata__"))

    forbidden_patterns = (*FROZEN_MOE_ROUTER_PATTERNS, *FROZEN_MAMBA_SSM_PATTERNS)
    if base_model_id:
        profile = detect_architecture_profile(base_model_id)
        if profile.frozen_module_patterns:
            forbidden_patterns = tuple(
                dict.fromkeys((*forbidden_patterns, *profile.frozen_module_patterns))
            )

    violating_tensors: list[str] = []
    for t_name in tensor_names:
        t_low = t_name.lower()
        if any(pat.lower() in t_low for pat in forbidden_patterns):
            violating_tensors.append(t_name)

    if violating_tensors:
        raise LoRACompatibilityError(
            "LoRA .safetensors modifies frozen MoE router gate or Mamba SSM state parameters: "
            f"{violating_tensors[:5]}"
        )

    return rank, tensor_names


def _validate_json_against_schema(
    instance: Any, schema: Mapping[str, Any], path: str = "$"
) -> None:
    """Lightweight deterministic JSON Schema validator for xgrammar structural_tag outputs."""
    expected_type = schema.get("type")
    if isinstance(expected_type, str):
        type_ok = (
            (expected_type == "object" and isinstance(instance, dict))
            or (expected_type == "array" and isinstance(instance, list))
            or (expected_type == "string" and isinstance(instance, str))
            or (
                expected_type == "integer"
                and isinstance(instance, int)
                and not isinstance(instance, bool)
            )
            or (
                expected_type == "number"
                and isinstance(instance, (int, float))
                and not isinstance(instance, bool)
            )
            or (expected_type == "boolean" and isinstance(instance, bool))
            or (expected_type == "null" and instance is None)
        )
        if not type_ok:
            raise StructuredOutputValidationError(
                f"Schema type mismatch at {path}: expected '{expected_type}', "
                f"got {type(instance).__name__}"
            )

    if "enum" in schema:
        allowed_enum = schema["enum"]
        if isinstance(allowed_enum, Sequence) and instance not in allowed_enum:
            raise StructuredOutputValidationError(
                f"Value {instance!r} at {path} not in enum {list(allowed_enum)!r}"
            )

    if isinstance(instance, dict):
        required = schema.get("required") or ()
        for req_key in required:
            if req_key not in instance:
                raise StructuredOutputValidationError(
                    f"Missing required JSON property '{req_key}' at {path}"
                )
        props = schema.get("properties") or {}
        if schema.get("additionalProperties") is False and isinstance(props, Mapping):
            extras = sorted(set(instance) - set(props))
            if extras:
                raise StructuredOutputValidationError(
                    f"Unexpected additionalProperties {extras} at {path}"
                )
        if isinstance(props, Mapping):
            for key, sub_schema in props.items():
                if key in instance and isinstance(sub_schema, Mapping):
                    _validate_json_against_schema(instance[key], sub_schema, path=f"{path}.{key}")
    elif isinstance(instance, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for idx, elem in enumerate(instance):
                _validate_json_against_schema(elem, item_schema, path=f"{path}[{idx}]")


class ModelArchitectureAdapter:
    """Adapts governed open-weights architectures for vLLM/SGLang/mlx-lm without tag lock-in."""

    def __init__(
        self,
        model_id: str = "google/gemma-4-26B-A4B-it",
        *,
        default_max_model_len: int = 32_768,
        reasoning_parser: str | None = None,
        tokenizer: Any | None = None,
        tokenizer_path: str | Path | None = None,
        chat_template: str | None = None,
        local_files_only: bool = True,
        governance_policy: ModelGovernancePolicy | None = None,
    ) -> None:
        self.policy = governance_policy or ModelGovernancePolicy.default()
        self.model_id = self.policy.validate_local_model_id(model_id)
        if default_max_model_len < 512:
            raise ValueError(f"default_max_model_len must be >= 512, got {default_max_model_len}")
        self._default_max_model_len = int(default_max_model_len)
        self._discovered_max_model_len: int | None = None
        self._profile = detect_architecture_profile(
            self.model_id, reasoning_parser_override=reasoning_parser
        )
        self._tokenizer = tokenizer
        self._tokenizer_path = Path(tokenizer_path) if tokenizer_path is not None else None
        self._chat_template = chat_template
        self._local_files_only = local_files_only
        self._tokenizer_attempted = tokenizer is not None
        self._lock = threading.RLock()
        self._jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)

    @property
    def profile(self) -> ModelArchitectureProfile:
        return self._profile

    @property
    def reasoning_parser(self) -> str | None:
        return self._profile.reasoning_parser

    @property
    def max_model_len(self) -> int:
        with self._lock:
            return (
                self._discovered_max_model_len
                if self._discovered_max_model_len is not None
                else self._default_max_model_len
            )

    @staticmethod
    def normalize_v1_base_url(endpoint_url: str) -> str:
        cleaned = endpoint_url.strip().rstrip("/")
        if not cleaned:
            raise ValueError("vLLM/SGLang endpoint_url cannot be empty.")
        if cleaned.endswith("/v1"):
            return cleaned
        return f"{cleaned}/v1"

    def discover_max_model_len(
        self,
        endpoint_url: str,
        *,
        http_client: httpx.Client | None = None,
        api_key: str | None = None,
        timeout_seconds: float = 5.0,
        force_refresh: bool = False,
    ) -> int:
        """Discovers ``max_model_len`` dynamically via ``GET /v1/models``."""
        with self._lock:
            if self._discovered_max_model_len is not None and not force_refresh:
                return self._discovered_max_model_len

        v1_base = self.normalize_v1_base_url(endpoint_url)
        headers: dict[str, str] = {"Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        owns_client = http_client is None
        client = http_client or httpx.Client(timeout=timeout_seconds)
        try:
            resp = client.get(f"{v1_base}/models", headers=headers)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            logger.debug(
                "Could not discover max_model_len from %s/models (%s): %s; using default %d",
                v1_base,
                type(exc).__name__,
                exc,
                self._default_max_model_len,
            )
            return self.max_model_len
        finally:
            if owns_client:
                client.close()

        items = payload.get("data") if isinstance(payload, Mapping) else None
        if not isinstance(items, list) or not items:
            return self.max_model_len

        target_low = self.model_id.lower()
        target_tail = target_low.split("/")[-1]
        matched_entry: Mapping[str, Any] | None = None
        for entry in items:
            if not isinstance(entry, Mapping):
                continue
            candidates = (
                str(entry.get("id") or "").strip().lower(),
                str(entry.get("root") or "").strip().lower(),
                str(entry.get("parent") or "").strip().lower(),
            )
            if any(
                cand in (target_low, target_tail) or cand.endswith(f"/{target_tail}")
                for cand in candidates
                if cand
            ):
                if (
                    entry.get("max_model_len") is not None
                    or entry.get("context_length") is not None
                    or entry.get("max_context_len") is not None
                ):
                    matched_entry = entry
                    break
                if matched_entry is None:
                    matched_entry = entry

        if matched_entry is None:
            for entry in items:
                if isinstance(entry, Mapping) and (
                    entry.get("max_model_len") is not None
                    or entry.get("context_length") is not None
                    or entry.get("max_context_len") is not None
                ):
                    matched_entry = entry
                    break
            if matched_entry is None and isinstance(items[0], Mapping):
                matched_entry = items[0]

        if matched_entry is not None:
            raw_len = (
                matched_entry.get("max_model_len")
                or matched_entry.get("context_length")
                or matched_entry.get("max_context_len")
                or matched_entry.get("max_ context_len")
            )
            if raw_len is not None:
                try:
                    parsed_len = int(raw_len)
                    if parsed_len >= 512:
                        with self._lock:
                            self._discovered_max_model_len = parsed_len
                        return parsed_len
                except (TypeError, ValueError):
                    pass

        return self.max_model_len

    def _resolve_tokenizer(self) -> Any | None:
        with self._lock:
            if self._tokenizer is not None:
                return self._tokenizer
            if self._tokenizer_attempted:
                return None
            self._tokenizer_attempted = True

            source = (
                str(self._tokenizer_path) if self._tokenizer_path is not None else self.model_id
            )
            if self._tokenizer_path is not None and self._tokenizer_path.is_dir():
                cfg_file = self._tokenizer_path / "config.json"
                if cfg_file.is_file():
                    cfg_json = json.loads(cfg_file.read_text(encoding="utf-8"))
                    if isinstance(cfg_json, Mapping) and (
                        cfg_json.get("trust_remote_code") is True or "auto_map" in cfg_json
                    ):
                        raise ValueError(
                            "ModelGovernancePolicy prohibits trust_remote_code=True or auto_map "
                            f"in tokenizer directory '{self._tokenizer_path}'."
                        )
                tok_cfg_file = self._tokenizer_path / "tokenizer_config.json"
                if tok_cfg_file.is_file() and self._chat_template is None:
                    tok_cfg = json.loads(tok_cfg_file.read_text(encoding="utf-8"))
                    if isinstance(tok_cfg, Mapping) and isinstance(
                        tok_cfg.get("chat_template"), str
                    ):
                        self._chat_template = tok_cfg["chat_template"]

            try:
                from transformers import AutoTokenizer

                loaded = AutoTokenizer.from_pretrained(
                    source,
                    trust_remote_code=False,
                    local_files_only=self._local_files_only,
                )
                self._tokenizer = loaded
                return loaded
            except Exception as exc:
                logger.debug(
                    "Offline AutoTokenizer.from_pretrained(%r, trust_remote_code=False) "
                    "not cached locally (%s): %s",
                    source,
                    type(exc).__name__,
                    exc,
                )
                return None

    def format_chat_messages(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        tokenizer: Any | None = None,
        add_generation_prompt: bool = True,
        enable_thinking: bool | None = None,
    ) -> str:
        """Formats chat turns via ``AutoTokenizer.apply_chat_template`` (never raw Gemma tags)."""
        normalized = [
            {"role": str(m.get("role", "user")), "content": str(m.get("content", ""))}
            for m in messages
        ]
        active_tok = tokenizer if tokenizer is not None else self._resolve_tokenizer()
        if active_tok is not None and callable(getattr(active_tok, "apply_chat_template", None)):
            kwargs: dict[str, Any] = {
                "tokenize": False,
                "add_generation_prompt": add_generation_prompt,
            }
            if enable_thinking is not None:
                kwargs["enable_thinking"] = enable_thinking
            try:
                rendered = active_tok.apply_chat_template(normalized, **kwargs)
                if isinstance(rendered, str) and rendered:
                    return rendered
            except TypeError:
                kwargs.pop("enable_thinking", None)
                rendered = active_tok.apply_chat_template(normalized, **kwargs)
                if isinstance(rendered, str) and rendered:
                    return rendered

        if self._chat_template:
            compiled = self._jinja_env.from_string(self._chat_template)
            rendered_jinja = compiled.render(
                messages=normalized,
                add_generation_prompt=add_generation_prompt,
                bos_token="",
                eos_token="",
                enable_thinking=enable_thinking,
            )
            if rendered_jinja:
                return str(rendered_jinja)

        return self.policy.format_chat_prompt(
            normalized,
            tokenizer=None,
            add_generation_prompt=add_generation_prompt,
        )

    @staticmethod
    def build_xgrammar_structural_tag_format(
        json_schema: Mapping[str, Any] | type[BaseModel],
        *,
        begin_tag: str = DEFAULT_STRUCTURAL_BEGIN_TAG,
        end_tag: str = DEFAULT_STRUCTURAL_END_TAG,
        triggers: Sequence[str] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Builds ``xgrammar`` ``structural_tag`` ``response_format`` and ``extra_body``."""
        if isinstance(json_schema, type) and issubclass(json_schema, BaseModel):
            resolved_schema: dict[str, Any] = json_schema.model_json_schema()
        elif isinstance(json_schema, Mapping):
            resolved_schema = dict(json_schema)
        else:
            raise TypeError("json_schema must be a JSON Schema mapping or Pydantic BaseModel class")

        if not begin_tag.strip() or not end_tag.strip():
            raise ValueError("structural_tag begin_tag and end_tag must be non-empty.")

        effective_triggers = list(triggers) if triggers else [begin_tag]
        response_format: dict[str, Any] = {
            "type": "structural_tag",
            "structures": [
                {
                    "begin": begin_tag,
                    "schema": resolved_schema,
                    "end": end_tag,
                }
            ],
            "triggers": effective_triggers,
        }
        extra_body: dict[str, Any] = {}
        return response_format, extra_body

    @staticmethod
    def extract_and_validate_structural_json(
        raw_content: str,
        json_schema: Mapping[str, Any] | type[BaseModel],
        *,
        begin_tag: str = DEFAULT_STRUCTURAL_BEGIN_TAG,
        end_tag: str = DEFAULT_STRUCTURAL_END_TAG,
    ) -> dict[str, Any]:
        """Extracts ``structural_tag`` JSON payload and validates schema fail-closed."""
        cleaned = raw_content.strip()
        if not cleaned:
            raise StructuredOutputValidationError(
                "Empty model completion for structured JSON output."
            )

        candidate = cleaned
        start_idx = cleaned.find(begin_tag)
        if start_idx != -1:
            inner_start = start_idx + len(begin_tag)
            end_idx = cleaned.find(end_tag, inner_start)
            if end_idx == -1:
                raise StructuredOutputValidationError(
                    f"Missing closing structural_tag '{end_tag}' in model completion."
                )
            candidate = cleaned[inner_start:end_idx].strip()

        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise StructuredOutputValidationError(
                f"Invalid JSON inside structural_tag output: {exc}"
            ) from exc

        if not isinstance(parsed, dict):
            raise StructuredOutputValidationError(
                f"Expected JSON object from structural_tag, got {type(parsed).__name__}."
            )

        if isinstance(json_schema, type) and issubclass(json_schema, BaseModel):
            try:
                validated_model = json_schema.model_validate(parsed)
            except ValidationError as exc:
                raise StructuredOutputValidationError(
                    f"Pydantic schema validation failed for structural_tag output: {exc}"
                ) from exc
            return validated_model.model_dump(mode="json")

        _validate_json_against_schema(parsed, json_schema)
        return parsed

    def extract_completion_and_reasoning(
        self,
        response_payload: Mapping[str, Any],
        *,
        parse_inline_fallback: bool = False,
    ) -> tuple[str, str, int, int, int]:
        """Extracts ``(content, reasoning_content, prompt_toks, completion_toks, total_toks)``."""
        choices = response_payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise OpenWeightsBackendUnavailableError(
                "Malformed /v1/chat/completions payload: missing non-empty 'choices' list."
            )
        first = choices[0]
        if not isinstance(first, Mapping):
            raise OpenWeightsBackendUnavailableError("Malformed choice item in completion payload.")

        msg = first.get("message")
        content = ""
        reasoning = ""
        if isinstance(msg, Mapping):
            raw_content = msg.get("content")
            content = str(raw_content) if raw_content is not None else ""
            raw_reasoning = (
                msg.get("reasoning_content")
                if msg.get("reasoning_content") is not None
                else msg.get("reasoning")
            )
            reasoning = str(raw_reasoning) if raw_reasoning is not None else ""
        elif first.get("text") is not None:
            content = str(first.get("text") or "")

        if not reasoning and parse_inline_fallback and content:
            m = _INLINE_THINK_RE.match(content)
            if m:
                reasoning = (m.group("think") or m.group("thought") or "").strip()
                content = content[m.end() :]

        usage = response_payload.get("usage")
        p_tok = 0
        c_tok = 0
        t_tok = 0
        if isinstance(usage, Mapping):
            p_tok = int(usage.get("prompt_tokens", 0) or 0)
            c_tok = int(usage.get("completion_tokens", 0) or 0)
            t_tok = int(usage.get("total_tokens", 0) or (p_tok + c_tok))

        return content, reasoning, p_tok, c_tok, t_tok

    def recommended_vllm_serve_args(
        self,
        *,
        max_lora_rank: int = 32,
        max_loras: int = 16,
    ) -> list[str]:
        """Returns hardened vLLM launch flags for GKE/K8s multi-LoRA serving."""
        if max_lora_rank not in VALID_VLLM_LORA_RANKS:
            raise ValueError(f"max_lora_rank must be in {sorted(VALID_VLLM_LORA_RANKS)}")
        args = [
            "--model",
            self.model_id,
            "--load-format",
            "safetensors",
            "--structured-outputs-config.backend",
            "xgrammar",
            "--enable-prefix-caching",
            "--enable-lora",
            "--max-lora-rank",
            str(max_lora_rank),
            "--max-loras",
            str(max_loras),
        ]
        if self._profile.reasoning_parser:
            args.extend(
                [
                    "--reasoning-parser",
                    self._profile.reasoning_parser,
                    "--structured-outputs-config.enable_in_reasoning=True",
                ]
            )
        return args


_MLX_MODEL_CACHE_LOCK = threading.Lock()
_MLX_MODEL_CACHE: OrderedDict[tuple[str, str], tuple[Any, Any]] = OrderedDict()


class OpenWeightsVLLMExecutor:
    """Self-hosted ``LLMBackendSPI`` for GKE/K8s ``vLLM``/``SGLang`` & Apple Silicon ``mlx-lm``."""

    def __init__(
        self,
        model_id: str = "google/gemma-4-26B-A4B-it",
        *,
        endpoint_url: str = DEFAULT_VLLM_ENDPOINT,
        runtime_backend: RuntimeBackendKind = "vllm",
        default_max_model_len: int = 32_768,
        reasoning_parser: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
        adapter_router: AdapterRouter | None = None,
        gpu_quota_manager: RedisGPUQuotaManager | None = None,
        architecture_adapter: ModelArchitectureAdapter | None = None,
        http_client: httpx.Client | None = None,
        mlx_loader_fn: Callable[..., tuple[Any, Any]] | None = None,
        mlx_generate_fn: Callable[..., str] | None = None,
        governance_policy: ModelGovernancePolicy | None = None,
        auto_discover_max_len: bool = True,
    ) -> None:
        self.policy = governance_policy or ModelGovernancePolicy.default()
        validated_model = self.policy.validate_local_model_id(model_id)
        self._runtime_backend: RuntimeBackendKind = runtime_backend
        self.endpoint_url = ModelArchitectureAdapter.normalize_v1_base_url(endpoint_url)
        self.api_key = api_key or os.environ.get("SCOUT_VLLM_API_KEY")
        self.timeout_seconds = float(timeout_seconds)
        self.architecture = architecture_adapter or ModelArchitectureAdapter(
            validated_model,
            default_max_model_len=default_max_model_len,
            reasoning_parser=reasoning_parser,
            governance_policy=self.policy,
        )
        self.adapter_router = adapter_router or AdapterRouter(
            default_base_model_id=validated_model,
        )
        self.gpu_quota_manager = gpu_quota_manager
        self._http_client = http_client or httpx.Client(timeout=self.timeout_seconds)
        self._owns_http_client = http_client is None
        self._mlx_loader_fn = mlx_loader_fn
        self._mlx_generate_fn = mlx_generate_fn
        self._auto_discover_max_len = auto_discover_max_len
        self._last_completion_var: contextvars.ContextVar[ParsedModelCompletion | None] = (
            contextvars.ContextVar(
                f"open_weights_last_completion_{id(self)}",
                default=None,
            )
        )

    @property
    def last_completion(self) -> ParsedModelCompletion | None:
        return self._last_completion_var.get()

    @last_completion.setter
    def last_completion(self, value: ParsedModelCompletion | None) -> None:
        self._last_completion_var.set(value)

    @property
    def model_id(self) -> str:
        return self.architecture.model_id

    @property
    def runtime_backend(self) -> RuntimeBackendKind:
        return self._runtime_backend

    @property
    def max_model_len(self) -> int:
        if self._auto_discover_max_len and self._runtime_backend in {"vllm", "sglang", "auto"}:
            return self.architecture.discover_max_model_len(
                self.endpoint_url,
                http_client=self._http_client,
                api_key=self.api_key,
                timeout_seconds=min(self.timeout_seconds, 5.0),
            )
        return self.architecture.max_model_len

    def close(self) -> None:
        if self._owns_http_client:
            self._http_client.close()

    def _auth_headers(self, cache_salt: str | None = None) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if cache_salt:
            headers["X-Prefix-Cache-Salt"] = cache_salt
        return headers

    def is_online(self) -> bool:
        """Probes backend readiness without fabricating completions."""
        if self._runtime_backend == "mlx-lm":
            if self._mlx_loader_fn is not None and self._mlx_generate_fn is not None:
                return True
            try:
                import mlx_lm  # noqa: F401

                return os.environ.get("CONTEXT_SCOUT_USE_LIVE_MLX", "").strip() == "1"
            except ImportError:
                return False

        try:
            resp = self._http_client.get(
                f"{self.endpoint_url}/models",
                headers=self._auth_headers(),
                timeout=min(self.timeout_seconds, 3.0),
            )
            if resp.status_code == 200:
                if self._auto_discover_max_len:
                    self.architecture.discover_max_model_len(
                        self.endpoint_url,
                        http_client=self._http_client,
                        api_key=self.api_key,
                    )
                return True
        except Exception as exc:
            logger.debug(
                "vLLM/SGLang readiness probe failed at %s/models (%s): %s",
                self.endpoint_url,
                type(exc).__name__,
                exc,
            )

        if self._runtime_backend == "auto":
            return self._mlx_loader_fn is not None and self._mlx_generate_fn is not None
        return False

    def compute_query_budget(
        self,
        user_prompt: str,
        *,
        is_cross_repo: bool = False,
    ) -> tuple[str, int]:
        """Computes Adaptive Complexity Budget clamped to ``0.60 * max_model_len``."""
        tier, budget = compute_adaptive_complexity_budget(
            user_prompt,
            max_model_len=self.max_model_len,
            is_cross_repo=is_cross_repo,
        )
        return tier, budget

    @staticmethod
    def _ensure_structural_tag_instruction(
        messages: Sequence[Mapping[str, str]],
        *,
        begin_tag: str,
        end_tag: str,
    ) -> list[dict[str, str]]:
        normalized = [
            {"role": str(m.get("role", "user")), "content": str(m.get("content", ""))}
            for m in messages
        ]
        if any(begin_tag in m["content"] for m in normalized):
            return normalized
        directive = (
            f"\nEmit the final JSON object strictly enclosed between {begin_tag} and {end_tag}."
        )
        for msg in normalized:
            if msg["role"] == "system":
                msg["content"] = f"{msg['content']}{directive}"
                return normalized
        if normalized:
            normalized[-1]["content"] = f"{normalized[-1]['content']}{directive}"
        else:
            normalized.append({"role": "user", "content": directive.strip()})
        return normalized

    @staticmethod
    def _release_mlx_metal_cache() -> None:
        try:
            import mlx.core as mx

            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
            elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
                mx.metal.clear_cache()
        except Exception as exc:
            logger.debug("Ignoring MLX Metal cache release error (%s)", type(exc).__name__)

    def _validate_local_mlx_directory(self, path_str: str) -> None:
        cand = Path(path_str)
        if not cand.exists() or not cand.is_dir():
            return
        all_files = [p for p in cand.iterdir() if p.is_file()]
        disallowed_archives = [
            p.name for p in all_files if p.suffix.lower() in {".npz", ".gguf", ".onnx"}
        ]
        if disallowed_archives:
            raise LoRACompatibilityError(
                f"ModelGovernancePolicy requires .safetensors weights only; rejected: "
                f"{disallowed_archives}"
            )

        cfg_file = cand / "config.json"
        cfg_payload: Mapping[str, Any] | None = None
        if cfg_file.is_file():
            raw_cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
            if isinstance(raw_cfg, Mapping):
                cfg_payload = raw_cfg

        adapter_cfg_file = cand / "adapter_config.json"
        adapter_cfg_payload: Mapping[str, Any] | None = None
        if adapter_cfg_file.is_file():
            raw_acfg = json.loads(adapter_cfg_file.read_text(encoding="utf-8"))
            if not isinstance(raw_acfg, Mapping):
                raise LoRACompatibilityError("adapter_config.json must contain a JSON object.")
            adapter_cfg_payload = raw_acfg
            self.policy.validate_weights_safety(
                all_files,
                trust_remote_code=False,
                config_payload=adapter_cfg_payload,
            )
            st_files = sorted(cand.glob("*.safetensors"))
            if not st_files:
                raise LoRACompatibilityError(
                    f"Missing .safetensors weights in LoRA directory '{cand}'."
                )
            for st_file in st_files:
                validate_lora_safetensors_for_vllm(
                    st_file, adapter_cfg_payload, base_model_id=self.model_id
                )
            return

        weight_candidates = [
            p
            for p in all_files
            if p.suffix.lower()
            in {".safetensors", ".bin", ".pt", ".pth", ".pkl", ".pickle", ".h5", ".ckpt"}
        ]
        if weight_candidates or cfg_payload is not None:
            self.policy.validate_weights_safety(
                all_files,
                trust_remote_code=False,
                config_payload=cfg_payload,
            )

    def _complete_via_mlx_lm(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        route_decision: AdapterRouteDecision,
        cache_salt: str,
        max_tokens: int,
        json_schema: Mapping[str, Any] | type[BaseModel] | None = None,
        begin_tag: str = DEFAULT_STRUCTURAL_BEGIN_TAG,
        end_tag: str = DEFAULT_STRUCTURAL_END_TAG,
        enable_thinking: bool | None = None,
    ) -> ParsedModelCompletion:
        target_model = route_decision.base_model_id or self.model_id
        self.policy.validate_local_model_id(target_model)
        self._validate_local_mlx_directory(target_model)

        from context_scout.agents.llm_backend import detect_adapter_weight_format

        adapter_path: str | None = None
        if route_decision.routed_to_lora and route_decision.lora_adapter_path:
            weight_fmt = detect_adapter_weight_format(route_decision.lora_adapter_path)
            if weight_fmt != "mlx_lm":
                logger.warning(
                    "Adapter at %s has format '%s' (incompatible with mlx-lm); "
                    "serving clean base model %s",
                    route_decision.lora_adapter_path,
                    weight_fmt,
                    target_model,
                )
            else:
                self._validate_local_mlx_directory(route_decision.lora_adapter_path)
                adapter_path = route_decision.lora_adapter_path

        load_fn: Callable[..., Any] | None = self._mlx_loader_fn
        gen_fn: Callable[..., Any] | None = self._mlx_generate_fn
        if load_fn is None or gen_fn is None:
            try:
                import mlx_lm

                if load_fn is None:
                    load_fn = mlx_lm.load
                if gen_fn is None:
                    gen_fn = mlx_lm.generate
            except ImportError as exc:
                raise self._build_unavailable_error(
                    domain_id=route_decision.domain_id,
                    detail=f"mlx-lm package unavailable: {exc}",
                ) from exc
        assert load_fn is not None and gen_fn is not None

        cache_key = (target_model, adapter_path or "")
        with _MLX_MODEL_CACHE_LOCK:
            if cache_key in _MLX_MODEL_CACHE:
                _MLX_MODEL_CACHE.move_to_end(cache_key)
                model, tokenizer = _MLX_MODEL_CACHE[cache_key]
            else:
                if len(_MLX_MODEL_CACHE) >= 1:
                    _MLX_MODEL_CACHE.clear()
                    self._release_mlx_metal_cache()
                if adapter_path:
                    loaded = load_fn(
                        target_model,
                        adapter_path=adapter_path,
                        tokenizer_config={"trust_remote_code": False},
                    )
                else:
                    loaded = load_fn(
                        target_model,
                        tokenizer_config={"trust_remote_code": False},
                    )
                model, tokenizer = loaded[0], loaded[1]
                _MLX_MODEL_CACHE[cache_key] = (model, tokenizer)

        effective_messages: Sequence[Mapping[str, str]] = (
            self._ensure_structural_tag_instruction(messages, begin_tag=begin_tag, end_tag=end_tag)
            if json_schema is not None
            else messages
        )
        formatted_prompt = self.architecture.format_chat_messages(
            effective_messages,
            tokenizer=tokenizer,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        gen_kwargs: dict[str, Any] = {
            "prompt": formatted_prompt,
            "max_tokens": max_tokens,
        }
        if self._mlx_generate_fn is None:
            try:
                from mlx_lm.sample_utils import make_sampler

                gen_kwargs["sampler"] = make_sampler(temp=0.0)
            except Exception as exc:
                logger.debug("mlx_lm.sample_utils unavailable (%s)", type(exc).__name__)
        try:
            raw_out = str(gen_fn(model, tokenizer, **gen_kwargs))
        except TypeError:
            raw_out = str(
                gen_fn(
                    model,
                    tokenizer,
                    prompt=formatted_prompt,
                    max_tokens=max_tokens,
                )
            )
        content, reasoning, p_tok, c_tok, t_tok = (
            self.architecture.extract_completion_and_reasoning(
                {"choices": [{"message": {"content": raw_out}}]},
                parse_inline_fallback=True,
            )
        )
        if json_schema is not None:
            validated_obj = self.architecture.extract_and_validate_structural_json(
                content,
                json_schema,
                begin_tag=begin_tag,
                end_tag=end_tag,
            )
            content = json.dumps(validated_obj, separators=(",", ":"), sort_keys=True)

        parsed = ParsedModelCompletion(
            content=content,
            reasoning_content=reasoning,
            routed_model=route_decision.lora_request or target_model,
            lora_request=route_decision.lora_request if route_decision.routed_to_lora else None,
            cache_salt=cache_salt,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            total_tokens=t_tok,
            route_decision=route_decision,
        )
        self.last_completion = parsed
        return parsed

    def _complete_via_vllm_http(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        route_decision: AdapterRouteDecision,
        cache_salt: str,
        max_tokens: int,
        temperature: float = 0.0,
        json_schema: Mapping[str, Any] | type[BaseModel] | None = None,
        begin_tag: str = DEFAULT_STRUCTURAL_BEGIN_TAG,
        end_tag: str = DEFAULT_STRUCTURAL_END_TAG,
        enable_thinking: bool | None = None,
    ) -> ParsedModelCompletion:
        if self._auto_discover_max_len:
            self.architecture.discover_max_model_len(
                self.endpoint_url,
                http_client=self._http_client,
                api_key=self.api_key,
            )

        routed_model = (
            route_decision.lora_request
            if (route_decision.routed_to_lora and route_decision.lora_request)
            else self.model_id
        )
        normalized_msgs = (
            self._ensure_structural_tag_instruction(messages, begin_tag=begin_tag, end_tag=end_tag)
            if json_schema is not None
            else [
                {"role": str(m.get("role", "user")), "content": str(m.get("content", ""))}
                for m in messages
            ]
        )
        # Render via AutoTokenizer.apply_chat_template to guarantee template compatibility check
        formatted_prompt = self.architecture.format_chat_messages(
            normalized_msgs,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

        extra_body: dict[str, Any] = {
            "cache_salt": cache_salt,
        }
        body: dict[str, Any] = {
            "model": routed_model,
            "messages": normalized_msgs,
            "max_tokens": max(1, int(max_tokens)),
            "temperature": float(temperature),
            "cache_salt": cache_salt,
        }
        if enable_thinking is not None:
            tpl_kwargs = {"enable_thinking": enable_thinking}
            body["chat_template_kwargs"] = tpl_kwargs
            extra_body["chat_template_kwargs"] = tpl_kwargs

        if json_schema is not None:
            response_format, xg_extra = self.architecture.build_xgrammar_structural_tag_format(
                json_schema,
                begin_tag=begin_tag,
                end_tag=end_tag,
            )
            body["response_format"] = response_format
            body.update(xg_extra)
            extra_body.update(xg_extra)

        body["extra_body"] = extra_body
        body["metadata"] = {
            "formatted_prompt_sha256": hashlib.sha256(formatted_prompt.encode("utf-8")).hexdigest(),
            "lora_routed": str(route_decision.routed_to_lora).lower(),
        }

        try:
            resp = self._http_client.post(
                f"{self.endpoint_url}/chat/completions",
                headers=self._auth_headers(cache_salt=cache_salt),
                json=body,
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            raise self._build_unavailable_error(
                domain_id=route_decision.domain_id,
                detail=(
                    f"{self._runtime_backend} HTTP inference failed ({type(exc).__name__}): {exc}"
                ),
            ) from exc

        if not isinstance(payload, Mapping):
            raise self._build_unavailable_error(
                domain_id=route_decision.domain_id,
                detail="Non-JSON object returned by /v1/chat/completions.",
            )

        content, reasoning, p_tok, c_tok, t_tok = (
            self.architecture.extract_completion_and_reasoning(
                payload,
                parse_inline_fallback=False,
            )
        )
        if json_schema is not None:
            validated_obj = self.architecture.extract_and_validate_structural_json(
                content,
                json_schema,
                begin_tag=begin_tag,
                end_tag=end_tag,
            )
            content = json.dumps(validated_obj, separators=(",", ":"), sort_keys=True)

        parsed = ParsedModelCompletion(
            content=content,
            reasoning_content=reasoning,
            routed_model=routed_model,
            lora_request=route_decision.lora_request if route_decision.routed_to_lora else None,
            cache_salt=cache_salt,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            total_tokens=t_tok,
            route_decision=route_decision,
        )
        self.last_completion = parsed
        return parsed

    def _build_unavailable_error(
        self,
        *,
        domain_id: str,
        detail: str,
    ) -> OpenWeightsBackendUnavailableError:
        diag = InsufficientContextDiagnostic(
            trace_id=str(uuid.uuid4()),
            domain_id=domain_id or "none",
            reason=(
                f"Open-weights model '{self.model_id}' ({self._runtime_backend}) is offline or "
                f"failed: {detail}. Paid tokens spent: 0."
            ),
            is_abstention=True,
            paid_tokens_spent=0,
            missing_items=[f"model:{self.model_id}:unavailable"],
            failure_check="MODEL_UNAVAILABLE",
        )
        return OpenWeightsBackendUnavailableError(
            diag.reason,
            diagnostic=diag,
            paid_tokens_spent=0,
        )

    def complete_messages(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        domain_id: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        principal: PrincipalContext | None = None,
        priority: GPUPriorityClass = GPUPriorityClass.P0,
        json_schema: Mapping[str, Any] | type[BaseModel] | None = None,
        begin_tag: str = DEFAULT_STRUCTURAL_BEGIN_TAG,
        end_tag: str = DEFAULT_STRUCTURAL_END_TAG,
        enable_thinking: bool | None = None,
    ) -> ParsedModelCompletion:
        """Executes ACL-routed, prefix-cache-salted, optional ``xgrammar`` completion."""
        route_decision = self.adapter_router.route(
            domain_id,
            principal,
            verify_weights=True,
        )
        cache_salt = compute_prefix_cache_salt(principal)

        def _dispatch() -> ParsedModelCompletion:
            if self._runtime_backend == "mlx-lm":
                return self._complete_via_mlx_lm(
                    messages,
                    route_decision=route_decision,
                    cache_salt=cache_salt,
                    max_tokens=max_tokens,
                    json_schema=json_schema,
                    begin_tag=begin_tag,
                    end_tag=end_tag,
                    enable_thinking=enable_thinking,
                )
            if self._runtime_backend == "auto":
                try:
                    return self._complete_via_vllm_http(
                        messages,
                        route_decision=route_decision,
                        cache_salt=cache_salt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        json_schema=json_schema,
                        begin_tag=begin_tag,
                        end_tag=end_tag,
                        enable_thinking=enable_thinking,
                    )
                except OpenWeightsBackendUnavailableError:
                    if self._mlx_loader_fn is not None or self._mlx_generate_fn is not None:
                        return self._complete_via_mlx_lm(
                            messages,
                            route_decision=route_decision,
                            cache_salt=cache_salt,
                            max_tokens=max_tokens,
                            json_schema=json_schema,
                            begin_tag=begin_tag,
                            end_tag=end_tag,
                            enable_thinking=enable_thinking,
                        )
                    raise
            return self._complete_via_vllm_http(
                messages,
                route_decision=route_decision,
                cache_salt=cache_salt,
                max_tokens=max_tokens,
                temperature=temperature,
                json_schema=json_schema,
                begin_tag=begin_tag,
                end_tag=end_tag,
                enable_thinking=enable_thinking,
            )

        if self.gpu_quota_manager is not None and principal is not None:
            with self.gpu_quota_manager.lease(
                principal,
                priority=priority,
                estimated_tokens=max(256, int(max_tokens)),
            ) as gpu_lease:
                result = _dispatch()
                if result.total_tokens > 0:
                    gpu_lease.record_actual_tokens(result.total_tokens)
                return result

        return _dispatch()

    def complete_structured(
        self,
        prompt: str,
        json_schema: Mapping[str, Any] | type[BaseModel],
        *,
        domain_id: str | None = None,
        system_instruction: str | None = None,
        max_tokens: int = 1024,
        principal: PrincipalContext | None = None,
        priority: GPUPriorityClass = GPUPriorityClass.P0,
        begin_tag: str = DEFAULT_STRUCTURAL_BEGIN_TAG,
        end_tag: str = DEFAULT_STRUCTURAL_END_TAG,
        enable_thinking: bool | None = None,
    ) -> dict[str, Any]:
        """Executes ``xgrammar`` ``structural_tag`` constrained decoding and returns parsed JSON."""
        messages: list[dict[str, str]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        parsed = self.complete_messages(
            messages,
            domain_id=domain_id,
            max_tokens=max_tokens,
            principal=principal,
            priority=priority,
            json_schema=json_schema,
            begin_tag=begin_tag,
            end_tag=end_tag,
            enable_thinking=enable_thinking,
        )
        return dict(json.loads(parsed.content))

    def complete(
        self,
        prompt: str,
        domain_id: str | None = None,
        system_instruction: str | None = None,
        max_tokens: int = 1024,
        *,
        principal: PrincipalContext | None = None,
        priority: GPUPriorityClass = GPUPriorityClass.P0,
    ) -> str:
        """Implements ``LLMBackendSPI.complete`` with zero heuristic fallback."""
        messages: list[dict[str, str]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        parsed = self.complete_messages(
            messages,
            domain_id=domain_id,
            max_tokens=max_tokens,
            principal=principal,
            priority=priority,
        )
        return parsed.content

    def run_vllm_lora_dry_run_probe(
        self,
        *,
        adapter_id: str,
        adapter_dir: Path,
        admin_endpoint_url: str = "http://127.0.0.1:8001/v1",
        probe_live_vllm: bool = False,
    ) -> VLLMLoRADryRunReport:
        """Runs a 1-step vLLM multi-LoRA compatibility dry-run probe."""
        cfg_file = adapter_dir / "adapter_config.json"
        if not cfg_file.is_file():
            raise LoRACompatibilityError(f"Missing adapter_config.json in {adapter_dir}")
        st_candidates = sorted(adapter_dir.glob("*.safetensors"))
        self.policy.validate_weights_safety(
            [p for p in adapter_dir.iterdir() if p.is_file()],
            trust_remote_code=False,
        )
        adapter_cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
        if not isinstance(adapter_cfg, Mapping):
            raise LoRACompatibilityError("adapter_config.json must contain a JSON object")

        rank = 0
        all_tensors: list[str] = []
        for st_file in st_candidates:
            shard_rank, shard_names = validate_lora_safetensors_for_vllm(
                st_file,
                adapter_cfg,
                base_model_id=self.model_id,
            )
            rank = shard_rank
            all_tensors.extend(shard_names)
        tensor_names = tuple(sorted(set(all_tensors)))

        if probe_live_vllm:
            admin_v1 = ModelArchitectureAdapter.normalize_v1_base_url(admin_endpoint_url)
            dry_name = f"dryrun_{adapter_id}_{uuid.uuid4().hex[:6]}"
            load_resp = self._http_client.post(
                f"{admin_v1}/load_lora_adapter",
                headers=self._auth_headers(),
                json={"lora_name": dry_name, "lora_path": str(adapter_dir.resolve())},
            )
            load_resp.raise_for_status()
            try:
                step_resp = self._http_client.post(
                    f"{self.endpoint_url}/chat/completions",
                    headers=self._auth_headers(),
                    json={
                        "model": dry_name,
                        "messages": [{"role": "user", "content": "ping"}],
                        "max_tokens": 1,
                        "temperature": 0.0,
                    },
                )
                step_resp.raise_for_status()
            finally:
                self._http_client.post(
                    f"{admin_v1}/unload_lora_adapter",
                    headers=self._auth_headers(),
                    json={"lora_name": dry_name},
                )

        return VLLMLoRADryRunReport(
            adapter_id=adapter_id,
            base_model_id=self.model_id,
            rank=rank,
            use_rslora=True,
            use_dora=False,
            safetensors_tensor_count=len(tensor_names),
            frozen_router_and_ssm_verified=True,
            live_endpoint_probed=probe_live_vllm,
            passed=True,
            detail=(
                "Verified rank<=32, use_rslora=True, use_dora=False, and frozen MoE/SSM tensors."
            ),
        )


__all__ = [
    "DEFAULT_STRUCTURAL_BEGIN_TAG",
    "DEFAULT_STRUCTURAL_END_TAG",
    "DEFAULT_VLLM_ENDPOINT",
    "FROZEN_MAMBA_SSM_PATTERNS",
    "FROZEN_MOE_ROUTER_PATTERNS",
    "VALID_VLLM_LORA_RANKS",
    "GPUOverloadError",
    "LLMBackendSPI",
    "LoRACompatibilityError",
    "ModelArchitectureAdapter",
    "ModelArchitectureProfile",
    "OpenWeightsBackendUnavailableError",
    "OpenWeightsVLLMExecutor",
    "ParsedModelCompletion",
    "RuntimeBackendKind",
    "StructuredOutputValidationError",
    "VLLMLoRADryRunReport",
    "compute_prefix_cache_salt",
    "detect_architecture_profile",
    "inspect_safetensors_header",
    "validate_lora_safetensors_for_vllm",
]
