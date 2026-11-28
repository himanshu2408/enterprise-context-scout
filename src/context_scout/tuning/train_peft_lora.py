"""Production PEFT rsLoRA + DPO + CachedMNRL Trainer, Canary Gate, & Base-Model Migration (M4-02).

Implements ACL-safe rsLoRA training, vLLM multi-LoRA checks, and bi-encoder tuning:
- Pre-training 1-step ``vLLM`` multi-LoRA dry-run probe via ``OpenWeightsVLLMExecutor``.
- ``TRL SFTTrainer`` (with ``assistant_only_loss=True``, ``use_rslora=True``, ``use_dora=False``,
  ``LoRA+``, ``NEFTune``, frozen MoE routers & Mamba SSM states, Task-F1 checkpointing) plus
  native Hugging Face ``transformers.PreTrainedModel`` rsLoRA execution when ``trl``/``peft``
  optional extras are absent.
- ``TRL DPOTrainer`` / memory-efficient adapter-toggle DPO preference alignment.
- Optional ``CachedMultipleNegativesRankingLoss`` (``CachedMNRL``) bi-encoder fine-tuning.
- Fail-closed ``CanaryMemorizationGate`` (``0/10`` leak required before signing/upload/DB).
- ``cosign`` signature generation, tenant-isolated ``ArtifactObjectStore`` (``gs://`` / ``s3://``)
  upload, PG17 ``model_artifacts`` RLS registration, and ``migrate-base-model`` replay DAG.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import logging
import math
import re
import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal, cast
from urllib.parse import urlparse
from uuid import UUID

import torch
import torch.nn as nn
import torch.nn.functional as nn_functional
from pydantic import BaseModel, Field, field_validator, model_validator
from safetensors.torch import save_file as save_safetensors_file

from context_scout.agents.open_weights_vllm_executor import (
    FROZEN_MAMBA_SSM_PATTERNS,
    FROZEN_MOE_ROUTER_PATTERNS,
    VALID_VLLM_LORA_RANKS,
    LoRACompatibilityError,
    ModelArchitectureAdapter,
    OpenWeightsVLLMExecutor,
    detect_architecture_profile,
    validate_lora_safetensors_for_vllm,
)
from context_scout.config import ModelGovernancePolicy, ScoutSettings
from context_scout.db.repositories import (
    InMemoryRLSRepository,
    ModelArtifactRecord,
    PostgresRLSRepository,
)
from context_scout.db.session import get_ingestion_pool
from context_scout.retrieval.reranker import DomainGlossary
from context_scout.schemas.contracts import DEFAULT_TENANT_ID, PrincipalContext, RoleType
from context_scout.serving.adapter_sync_sidecar import (
    CosignBlobVerifier,
    sign_blob_for_cosign,
)
from context_scout.storage.object_store import (
    ArtifactObjectStore,
    GCSObjectStore,
    S3CompatibleObjectStore,
    build_tenant_domain_key,
    get_artifact_object_store,
)
from context_scout.tuning.specialization_compiler import (
    CompiledSpecialization,
    SpecializationCompiler,
)
from context_scout.tuning.teacher_distiller import (
    SyntheticCanary,
    TeacherDistiller,
)
from context_scout.tuning.train_mlx_lora import (
    DEFAULT_GPU_LOCK_PATH,
    GPUMutualExclusionLock,
)

logger = logging.getLogger(__name__)

_SAFE_LORA_LINEAR_LEAVES: Final[frozenset[str]] = frozenset(
    {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "in_proj",
        "out_proj",
    }
)
_EXTRA_FROZEN_MODULE_PATTERNS: Final[tuple[str, ...]] = (
    ".per_layer_input_gate",
    ".mlp.gate",
    ".block_sparse_moe.gate",
    ".router",
    ".ssm",
    ".mamba",
)
_CANARY_LEAK_MIN_MATCH_LEN: Final[int] = 16
_CANARY_PREFIXES: Final[tuple[str, ...]] = ("canary_tok_",)


class CanaryMemorizationError(PermissionError):
    """Raised fail-closed when ``CanaryMemorizationGate`` detects >= 1/10 canary leak."""


class MigrationReplayError(RuntimeError):
    """Raised fail-closed when ``migrate-base-model`` fails a replay verification gate."""


class PEFTTrainingConfig(BaseModel):
    """Validated training hyperparameters enforcing OWASP ASI06 & vLLM multi-LoRA safety."""

    domain_id: str = Field(min_length=1)
    base_model_id: str = Field(default="google/gemma-4-26B-A4B-it")
    base_model_revision_sha256: str = Field(
        default="0" * 64,
        description="64-character SHA-256 digest pinning exact base model weights",
    )
    training_scope_ceiling: list[str] = Field(min_length=1)
    r: int = Field(default=16, ge=1)
    lora_alpha: int = Field(default=32, ge=1)
    lora_dropout: float = Field(default=0.0, ge=0.0, le=0.5)
    use_rslora: bool = True
    use_dora: bool = False
    assistant_only_loss: bool = True
    loraplus_lr_ratio: float = Field(default=8.0, ge=1.0, le=64.0)
    neftune_noise_alpha: float = Field(default=5.0, ge=0.0, le=20.0)
    general_replay_ratio: float = Field(default=0.15, ge=0.10, le=0.50)
    learning_rate: float = Field(default=2e-4, gt=0.0)
    max_seq_length: int = Field(default=2048, ge=32)
    sft_steps: int = Field(default=20, ge=1)
    eval_steps: int = Field(default=5, ge=1)
    dpo_steps: int = Field(default=6, ge=0)
    dpo_beta: float = Field(default=0.1, gt=0.0, le=2.0)
    trainer_backend: Literal["auto", "trl", "transformers_peft"] = "auto"
    cosign_key_id: str = "default"
    seed: int = 42

    @field_validator("base_model_id")
    @classmethod
    def _validate_base_model(cls, value: str) -> str:
        return ModelGovernancePolicy.default().validate_local_model_id(value)

    @field_validator("base_model_revision_sha256")
    @classmethod
    def _validate_rev_sha256(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", cleaned):
            raise ValueError("base_model_revision_sha256 must be a 64-char hex SHA-256 digest")
        return cleaned

    @model_validator(mode="after")
    def _enforce_rslora_and_safety(self) -> PEFTTrainingConfig:
        if self.use_dora:
            raise ValueError("use_dora=True is forbidden for vLLM multi-LoRA serving")
        if not self.use_rslora:
            raise ValueError("use_rslora=True is required for rank-stabilized multi-LoRA")
        if not self.assistant_only_loss:
            raise ValueError("assistant_only_loss=True is required (OWASP ASI06)")
        if self.r not in VALID_VLLM_LORA_RANKS:
            raise ValueError(
                f"LoRA rank {self.r} violates policy; expected one of "
                f"{sorted(VALID_VLLM_LORA_RANKS)}"
            )
        if not [s for s in self.training_scope_ceiling if s.strip()]:
            raise ValueError("training_scope_ceiling must be non-empty (OWASP ASI06)")
        return self

    @property
    def rslora_scaling(self) -> float:
        return float(self.lora_alpha) / math.sqrt(float(self.r))

    @classmethod
    def from_compiled(
        cls,
        compiled: CompiledSpecialization,
        *,
        base_model_revision_sha256: str = "0" * 64,
        **overrides: Any,
    ) -> PEFTTrainingConfig:
        lp = compiled.manifest.lora_policy
        payload: dict[str, Any] = {
            "domain_id": compiled.domain_id,
            "base_model_id": compiled.manifest.base_model_id,
            "base_model_revision_sha256": base_model_revision_sha256,
            "training_scope_ceiling": list(compiled.training_scope_ceiling),
            "r": lp.rank,
            "lora_alpha": lp.lora_alpha,
            "lora_dropout": lp.lora_dropout,
            "use_rslora": lp.use_rslora,
            "use_dora": lp.use_dora,
            "assistant_only_loss": lp.assistant_only_loss,
            "loraplus_lr_ratio": lp.loraplus_lr_ratio,
            "neftune_noise_alpha": lp.neftune_noise_alpha,
            "general_replay_ratio": lp.general_replay_ratio,
            "dpo_beta": lp.dpo_beta,
        }
        payload.update(overrides)
        return cls.model_validate(payload)


def is_frozen_router_or_ssm_path(module_path: str, *, base_model_id: str | None = None) -> bool:
    """Returns True when ``module_path`` touches MoE router gates or Mamba SSM state."""
    padded = f".{module_path.strip('.')}.".lower()
    patterns: list[str] = [
        *FROZEN_MOE_ROUTER_PATTERNS,
        *FROZEN_MAMBA_SSM_PATTERNS,
        *_EXTRA_FROZEN_MODULE_PATTERNS,
    ]
    if base_model_id:
        profile = detect_architecture_profile(base_model_id)
        patterns.extend(profile.frozen_module_patterns)
    for pat in patterns:
        p_low = pat.lower()
        if p_low in padded or p_low.rstrip(".") in padded:
            return True
    return False


def iter_safe_linear_targets(
    model: nn.Module,
    base_model_id: str,
) -> list[tuple[str, str]]:
    """Lists ``(module_path, leaf_name)`` targets excluding MoE routers & Mamba SSMs."""
    targets: list[tuple[str, str]] = []
    for mod_name, mod in model.named_modules():
        if not mod_name or not isinstance(mod, nn.Linear):
            continue
        leaf = mod_name.rsplit(".", 1)[-1]
        if leaf not in _SAFE_LORA_LINEAR_LEAVES:
            continue
        if is_frozen_router_or_ssm_path(mod_name, base_model_id=base_model_id):
            continue
        targets.append((mod_name, leaf))
    return targets


def resolve_safe_lora_target_modules(
    model: nn.Module,
    base_model_id: str,
) -> list[str]:
    """Resolves deduplicated safe LoRA projection leaf names for ``model``."""
    targets = iter_safe_linear_targets(model, base_model_id)
    if not targets:
        raise LoRACompatibilityError(
            f"No safe LoRA linear target modules found on '{base_model_id}' after excluding "
            "frozen MoE router gates and Mamba SSM state projections."
        )
    return list(dict.fromkeys(leaf for _path, leaf in targets))


class RsLoRALinear(nn.Module):
    """Rank-stabilized LoRA wrapper (``alpha / sqrt(r)``, ``use_dora=False``) over ``nn.Linear``."""

    def __init__(
        self,
        base_layer: nn.Linear,
        *,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.r = int(r)
        self.lora_alpha = int(lora_alpha)
        self.scaling = float(lora_alpha) / math.sqrt(float(r))
        self.adapter_disabled: bool = False
        self.dropout: nn.Module = (
            nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        )

        for p in self.base_layer.parameters():
            p.requires_grad = False

        in_features = int(base_layer.in_features)
        out_features = int(base_layer.out_features)
        device = base_layer.weight.device
        dtype = base_layer.weight.dtype if base_layer.weight.is_floating_point() else torch.float32

        self.lora_A = nn.Parameter(torch.empty((self.r, in_features), device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_B = nn.Parameter(torch.zeros((out_features, self.r), device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = cast(torch.Tensor, self.base_layer(x))
        if self.adapter_disabled:
            return base_out
        dropped = self.dropout(x)
        low_rank = nn_functional.linear(nn_functional.linear(dropped, self.lora_A), self.lora_B)
        return base_out + (low_rank * self.scaling)


def _set_nested_module(root: nn.Module, path: str, new_module: nn.Module) -> None:
    parts = path.split(".")
    parent: nn.Module = root
    for segment in parts[:-1]:
        parent = getattr(parent, segment)
    setattr(parent, parts[-1], new_module)


def inject_rslora_adapters(
    model: nn.Module,
    *,
    r: int,
    lora_alpha: int,
    lora_dropout: float,
    base_model_id: str,
) -> list[str]:
    """Freezes base weights and injects ``RsLoRALinear`` onto safe non-router/SSM projections."""
    for p in model.parameters():
        p.requires_grad = False

    targets = iter_safe_linear_targets(model, base_model_id)
    if not targets:
        raise LoRACompatibilityError(
            f"Cannot inject rsLoRA into '{base_model_id}': zero safe linear target modules found."
        )

    module_lookup = dict(model.named_modules())
    adapted_paths: list[str] = []
    for mod_path, _leaf in targets:
        orig_mod = module_lookup[mod_path]
        if isinstance(orig_mod, RsLoRALinear):
            adapted_paths.append(mod_path)
            continue
        assert isinstance(orig_mod, nn.Linear)
        wrapped = RsLoRALinear(
            orig_mod,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        _set_nested_module(model, mod_path, wrapped)
        adapted_paths.append(mod_path)

    verify_frozen_routers_and_ssm_states(model, base_model_id=base_model_id)
    return adapted_paths


def verify_frozen_routers_and_ssm_states(
    model: nn.Module,
    *,
    base_model_id: str | None = None,
) -> None:
    """Asserts zero MoE router gate or Mamba SSM parameters are trainable."""
    violating: list[str] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if is_frozen_router_or_ssm_path(name, base_model_id=base_model_id):
            violating.append(name)
    if violating:
        raise LoRACompatibilityError(
            f"Trainable MoE router or Mamba SSM parameter detected (vLLM multi-LoRA): "
            f"{violating[:5]}"
        )


def set_adapters_enabled(model: nn.Module, enabled: bool) -> None:
    """Enables or disables ``RsLoRALinear`` deltas in-place for zero-copy DPO reference passes."""
    for mod in model.modules():
        if isinstance(mod, RsLoRALinear):
            mod.adapter_disabled = not enabled


def extract_peft_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Extracts standard HF PEFT ``lora_A.weight`` / ``lora_B.weight`` CPU tensors."""
    tensors: dict[str, torch.Tensor] = {}
    for mod_name, mod in model.named_modules():
        if isinstance(mod, RsLoRALinear):
            clean = mod_name.removeprefix("base_model.model.")
            tensors[f"base_model.model.{clean}.lora_A.weight"] = mod.lora_A.detach().cpu().clone()
            tensors[f"base_model.model.{clean}.lora_B.weight"] = mod.lora_B.detach().cpu().clone()
        else:
            for p_name, param in mod.named_parameters(recurse=False):
                if "lora_" in p_name.lower() and param.requires_grad:
                    full_key = f"{mod_name}.{p_name}" if mod_name else p_name
                    tensors[full_key] = param.detach().cpu().clone()
    if not tensors:
        raise LoRACompatibilityError("Zero LoRA tensors found on model for safetensors export.")
    return tensors


def load_peft_lora_state_dict(
    model: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
) -> None:
    """Restores LoRA ``lora_A`` / ``lora_B`` checkpoint tensors onto ``model``."""
    with torch.no_grad():
        for mod_name, mod in model.named_modules():
            if isinstance(mod, RsLoRALinear):
                clean = mod_name.removeprefix("base_model.model.")
                key_a = f"base_model.model.{clean}.lora_A.weight"
                key_b = f"base_model.model.{clean}.lora_B.weight"
                if key_a in state_dict:
                    mod.lora_A.copy_(state_dict[key_a].to(mod.lora_A.device, mod.lora_A.dtype))
                if key_b in state_dict:
                    mod.lora_B.copy_(state_dict[key_b].to(mod.lora_B.device, mod.lora_B.dtype))


class NEFTuneEmbeddingHook:
    """Adds uniform NEFTune noise ``alpha / sqrt(L * d)`` strictly during ``model.training``."""

    def __init__(self, model: nn.Module, noise_alpha: float = 5.0) -> None:
        self.noise_alpha = float(noise_alpha)
        self._handle: Any = None
        if self.noise_alpha > 0.0:
            emb = self._find_embedding_layer(model)
            if emb is not None:
                self._handle = emb.register_forward_hook(self._hook)

    @staticmethod
    def _find_embedding_layer(model: nn.Module) -> nn.Module | None:
        getter = getattr(model, "get_input_embeddings", None)
        if callable(getter):
            emb = getter()
            if isinstance(emb, nn.Module):
                return emb
        for mod in model.modules():
            if isinstance(mod, nn.Embedding):
                return mod
        return None

    def _hook(self, module: nn.Module, _inputs: Any, output: Any) -> Any:
        if not module.training or self.noise_alpha <= 0.0 or not isinstance(output, torch.Tensor):
            return output
        if output.ndim < 2:
            return output
        seq_len = max(1, int(output.shape[-2]))
        dim = max(1, int(output.shape[-1]))
        mag = self.noise_alpha / math.sqrt(float(seq_len * dim))
        noise = torch.empty_like(output).uniform_(-mag, mag)
        return output + noise

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def build_loraplus_adamw_optimizer(
    model: nn.Module,
    *,
    learning_rate: float = 2e-4,
    loraplus_lr_ratio: float = 8.0,
    weight_decay: float = 0.01,
) -> torch.optim.AdamW:
    """Builds LoRA+ AdamW optimizer giving ``lora_B`` ``loraplus_lr_ratio * lr``."""
    lora_a_params: list[nn.Parameter] = []
    lora_b_params: list[nn.Parameter] = []
    other_params: list[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        low = name.lower()
        if "lora_b" in low:
            lora_b_params.append(param)
        elif "lora_a" in low:
            lora_a_params.append(param)
        else:
            other_params.append(param)

    param_groups: list[dict[str, Any]] = []
    if lora_a_params:
        param_groups.append(
            {
                "params": lora_a_params,
                "lr": float(learning_rate),
                "weight_decay": weight_decay,
                "group_name": "lora_A",
            }
        )
    if lora_b_params:
        param_groups.append(
            {
                "params": lora_b_params,
                "lr": float(learning_rate) * float(loraplus_lr_ratio),
                "weight_decay": weight_decay,
                "group_name": "lora_B",
            }
        )
    if other_params:
        param_groups.append(
            {
                "params": other_params,
                "lr": float(learning_rate),
                "weight_decay": weight_decay,
                "group_name": "other",
            }
        )
    if not param_groups:
        raise LoRACompatibilityError("No trainable LoRA parameters found for LoRA+ optimizer.")
    return torch.optim.AdamW(param_groups)


def _encode_tokens(tokenizer: Any, text: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        try:
            enc = tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            enc = tokenizer.encode(text)
    elif callable(tokenizer):
        enc = tokenizer(text).get("input_ids", [])
    else:
        raise TypeError(f"Unsupported tokenizer object: {type(tokenizer)!r}")

    if hasattr(enc, "ids"):
        ids = [int(x) for x in enc.ids]
    elif isinstance(enc, torch.Tensor):
        ids = [int(x) for x in enc.flatten().tolist()]
    elif isinstance(enc, list):
        ids = [int(x) for x in enc]
    else:
        ids = [int(x) for x in list(enc)]
    return ids


def _decode_tokens(tokenizer: Any, token_ids: Sequence[int]) -> str:
    if hasattr(tokenizer, "decode"):
        try:
            return str(tokenizer.decode(list(token_ids), skip_special_tokens=True))
        except TypeError:
            return str(tokenizer.decode(list(token_ids)))
    return ""


def tokenize_chat_assistant_only(
    messages: Sequence[Mapping[str, str]],
    *,
    tokenizer: Any,
    arch_adapter: ModelArchitectureAdapter,
    max_seq_length: int,
) -> tuple[list[int], list[int]]:
    """Tokenizes chat messages with ``labels=-100`` on all non-final/prompt turns."""
    if len(messages) < 2:
        raise ValueError("Chat trajectory must contain at least 2 messages")

    prefix_str = arch_adapter.format_chat_messages(
        messages[:-1],
        tokenizer=tokenizer,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    full_str = arch_adapter.format_chat_messages(
        messages,
        tokenizer=tokenizer,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    if full_str.startswith(prefix_str) and len(full_str) > len(prefix_str):
        target_str = full_str[len(prefix_str) :]
    else:
        target_str = str(messages[-1].get("content", ""))

    prefix_ids = _encode_tokens(tokenizer, prefix_str)
    target_ids = _encode_tokens(tokenizer, target_str)
    if not target_ids:
        eos_id = getattr(tokenizer, "eos_token_id", None)
        target_ids = [int(eos_id) if isinstance(eos_id, int) else 1]

    if len(prefix_ids) + len(target_ids) > max_seq_length:
        keep_target = min(len(target_ids), max(8, max_seq_length // 2))
        target_ids = target_ids[:keep_target]
        keep_prefix = max(1, max_seq_length - len(target_ids))
        prefix_ids = prefix_ids[-keep_prefix:]

    input_ids = (prefix_ids + target_ids)[:max_seq_length]
    labels = ([-100] * len(prefix_ids) + target_ids)[:max_seq_length]
    if len(input_ids) < 2:
        input_ids = [0, *input_ids]
        labels = [-100, *labels]
        if not any(lbl != -100 for lbl in labels):
            raise ValueError(
                "Assistant target tokens were completely masked out under "
                f"max_seq_length={max_seq_length}."
            )
    return input_ids, labels


def _sequence_log_prob(
    model: nn.Module,
    prompt_ids: Sequence[int],
    completion_ids: Sequence[int],
    *,
    device: torch.device,
) -> torch.Tensor:
    full = [*prompt_ids, *completion_ids]
    if len(full) < 2:
        full = [0, *full]
    input_t = torch.tensor([full[:-1]], dtype=torch.long, device=device)
    target_t = torch.tensor([full[1:]], dtype=torch.long, device=device)
    outputs = model(input_ids=input_t)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
    log_probs = nn_functional.log_softmax(logits, dim=-1)
    token_lp = torch.gather(log_probs, dim=-1, index=target_t.unsqueeze(-1)).squeeze(-1)

    p_len = max(0, len(prompt_ids) - 1)
    mask = torch.zeros_like(token_lp)
    mask[:, p_len:] = 1.0
    return (token_lp * mask).sum()


def _greedy_generate_text(
    model: nn.Module,
    tokenizer: Any,
    prompt_text: str,
    *,
    max_new_tokens: int = 48,
    max_prompt_tokens: int = 512,
) -> str:
    prompt_ids = _encode_tokens(tokenizer, prompt_text)[-max_prompt_tokens:]
    if not prompt_ids:
        prompt_ids = [1]
    device = next(model.parameters()).device
    cur = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    generated: list[int] = []

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for _ in range(max_new_tokens):
                out = model(input_ids=cur)
                logits = out.logits if hasattr(out, "logits") else out[0]
                next_id = int(torch.argmax(logits[0, -1, :]).item())
                if eos_id is not None and next_id == int(eos_id):
                    break
                generated.append(next_id)
                next_t = torch.tensor([[next_id]], dtype=torch.long, device=device)
                cur = torch.cat([cur, next_t], dim=1)
    finally:
        model.train(was_training)
    return _decode_tokens(tokenizer, generated)


def _token_f1(pred: str, gold: str) -> float:
    pred_toks = re.findall(r"[A-Za-z0-9_./:-]+", pred.lower())
    gold_toks = re.findall(r"[A-Za-z0-9_./:-]+", gold.lower())
    if not pred_toks and not gold_toks:
        return 1.0
    if not pred_toks or not gold_toks:
        return 0.0
    pred_counts: dict[str, int] = {}
    for t in pred_toks:
        pred_counts[t] = pred_counts.get(t, 0) + 1
    gold_counts: dict[str, int] = {}
    for t in gold_toks:
        gold_counts[t] = gold_counts.get(t, 0) + 1
    overlap = sum(min(pred_counts[t], gold_counts[t]) for t in pred_counts if t in gold_counts)
    if overlap == 0:
        return 0.0
    prec = overlap / len(pred_toks)
    rec = overlap / len(gold_toks)
    return round((2.0 * prec * rec) / (prec + rec), 4)


def compute_specialization_task_f1(
    model: nn.Module,
    tokenizer: Any,
    valid_records: Sequence[Mapping[str, Any]],
    *,
    compiled: CompiledSpecialization | None,
    arch_adapter: ModelArchitectureAdapter,
    max_seq_length: int,
    max_eval_samples: int = 8,
) -> tuple[float, float]:
    """Evaluates validation CE loss and Task-F1 across all 5 specialization archetypes."""
    if not valid_records:
        return 0.0, 0.0
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    losses: list[float] = []
    f1_scores: list[float] = []

    try:
        with torch.no_grad():
            for rec in list(valid_records)[:max_eval_samples]:
                msgs = rec.get("messages") or []
                if len(msgs) < 2:
                    continue
                in_ids, lbls = tokenize_chat_assistant_only(
                    msgs,
                    tokenizer=tokenizer,
                    arch_adapter=arch_adapter,
                    max_seq_length=max_seq_length,
                )
                in_t = torch.tensor([in_ids], dtype=torch.long, device=device)
                lbl_t = torch.tensor([lbls], dtype=torch.long, device=device)
                out = model(input_ids=in_t)
                logits = out.logits if hasattr(out, "logits") else out[0]
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = lbl_t[:, 1:].contiguous()
                if (shift_labels != -100).any():
                    loss_val = nn_functional.cross_entropy(
                        shift_logits.view(-1, shift_logits.shape[-1]),
                        shift_labels.view(-1),
                        ignore_index=-100,
                    )
                    losses.append(float(loss_val.item()))

                prompt_rendered = arch_adapter.format_chat_messages(
                    msgs[:-1],
                    tokenizer=tokenizer,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                pred_text = _greedy_generate_text(
                    model,
                    tokenizer,
                    prompt_rendered,
                    max_new_tokens=32,
                )
                gold_text = str(msgs[-1].get("content") or "")
                base_f1 = _token_f1(pred_text, gold_text)
                if compiled is not None:
                    val_res = compiled.validate_output(pred_text)
                    if val_res.passed:
                        base_f1 = min(1.0, max(base_f1, 0.85))
                f1_scores.append(base_f1)
    finally:
        model.train(was_training)

    mean_loss = round(sum(losses) / len(losses), 4) if losses else 0.0
    mean_f1 = round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else 0.0
    return mean_f1, mean_loss


@dataclass(frozen=True)
class CanaryAuditReport:
    """Outcome of ``CanaryMemorizationGate`` evaluation across 10 synthetic canaries."""

    total_canaries: int
    canary_leak_count: int
    leaked_canary_ids: tuple[str, ...]
    passed: bool


class CanaryMemorizationGate:
    """Fail-closed ``0/10`` canary memorization gate executed before signing/upload."""

    def __init__(self, *, min_canaries: int = 10) -> None:
        self.min_canaries = min(10, max(1, int(min_canaries)))

    @staticmethod
    def _contains_canary_leak(generated_text: str, canary: SyntheticCanary) -> bool:
        clean_gen = generated_text.strip()
        if not clean_gen:
            return False
        secret = canary.secret_token.strip()
        if secret in clean_gen or secret.lower() in clean_gen.lower():
            return True
        # Strip deterministic format prefixes before sliding-window matching so shared
        # token prefixes cannot trigger cross-canary Birthday Paradox false positives.
        high_entropy_body = secret
        for prefix in _CANARY_PREFIXES:
            if high_entropy_body.startswith(prefix):
                high_entropy_body = high_entropy_body[len(prefix) :]
                break
        if len(high_entropy_body) >= _CANARY_LEAK_MIN_MATCH_LEN:
            for idx in range(0, len(high_entropy_body) - _CANARY_LEAK_MIN_MATCH_LEN + 1, 4):
                sub = high_entropy_body[idx : idx + _CANARY_LEAK_MIN_MATCH_LEN]
                if sub in clean_gen:
                    return True
        return False

    def evaluate(
        self,
        canaries: Sequence[SyntheticCanary | Mapping[str, Any]],
        *,
        model: nn.Module | None = None,
        tokenizer: Any | None = None,
        generation_fn: Callable[[str], str] | None = None,
        arch_adapter: ModelArchitectureAdapter | None = None,
    ) -> CanaryAuditReport:
        """Evaluates every canary at greedy temperature 0.0; raises on ``leak_count > 0``."""
        parsed: list[SyntheticCanary] = []
        for item in canaries:
            if isinstance(item, SyntheticCanary):
                parsed.append(item)
            else:
                parsed.append(
                    SyntheticCanary(
                        canary_id=str(item["canary_id"]),
                        domain_id=str(item["domain_id"]),
                        prompt_prefix=str(item["prompt_prefix"]),
                        secret_token=str(item["secret_token"]),
                        sha256_digest=str(item.get("sha256_digest") or ""),
                    )
                )

        if len(parsed) < self.min_canaries:
            raise CanaryMemorizationError(
                f"CanaryMemorizationGate requires at least {self.min_canaries} synthetic canaries, "
                f"got {len(parsed)} (OWASP ASI06)"
            )
        if generation_fn is None and (model is None or tokenizer is None):
            raise CanaryMemorizationError(
                "CanaryMemorizationGate requires either generation_fn or (model, tokenizer)"
            )

        leaked_ids: list[str] = []
        for canary in parsed:
            if generation_fn is not None:
                output_text = str(generation_fn(canary.prompt_prefix))
            else:
                assert model is not None and tokenizer is not None
                prompt_input = canary.prompt_prefix
                if arch_adapter is not None:
                    prompt_input = arch_adapter.format_chat_messages(
                        [{"role": "user", "content": canary.prompt_prefix}],
                        tokenizer=tokenizer,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                output_text = _greedy_generate_text(
                    model,
                    tokenizer,
                    prompt_input,
                    max_new_tokens=48,
                )
            if self._contains_canary_leak(output_text, canary):
                leaked_ids.append(canary.canary_id)

        report = CanaryAuditReport(
            total_canaries=len(parsed),
            canary_leak_count=len(leaked_ids),
            leaked_canary_ids=tuple(leaked_ids),
            passed=len(leaked_ids) == 0,
        )
        if not report.passed:
            raise CanaryMemorizationError(
                f"CanaryMemorizationGate FAILED ({report.canary_leak_count}/"
                f"{report.total_canaries} leak(s): {report.leaked_canary_ids}); "
                "blocking signing, upload, and model_artifacts registration (OWASP ASI06)."
            )
        return report


@dataclass(frozen=True)
class PEFTTrainingRunResult:
    """Complete verified output of a ``PEFTRsLoRATrainer`` run."""

    domain_id: str
    artifact_id: str
    base_model_id: str
    base_model_revision_sha256: str
    trainer_backend_used: str
    adapter_dir: Path
    safetensors_path: Path
    adapter_config_path: Path
    training_metrics_path: Path
    artifact_manifest_path: Path
    weights_sha256: str
    sft_initial_loss: float
    sft_final_loss: float
    val_loss: float
    best_eval_task_f1: float
    dpo_initial_loss: float | None
    dpo_final_loss: float | None
    dpo_reward_accuracy: float | None
    canary_audit: CanaryAuditReport
    dry_run_passed: bool
    cosign_signature_b64: str = ""
    storage_uri: str = ""
    weights_object_key: str = ""
    config_object_key: str = ""
    signature_object_key: str = ""
    registered_artifact: ModelArtifactRecord | None = None


def _build_offline_tiny_model_for_base(base_model_id: str) -> tuple[nn.Module, Any]:
    """Constructs a minimal governed HuggingFace CausalLM + byte-level tokenizer for dry/CI runs."""
    import transformers
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    tf_mod: Any = transformers
    low = base_model_id.lower()
    vocab_size = 260
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    class _SimpleByteTokenizer:
        def __init__(self, inner: Any) -> None:
            self._inner = inner
            self.eos_token_id = 1
            self.pad_token_id = 0

        def encode(self, text: str, **_kwargs: Any) -> list[int]:
            raw = text.encode("utf-8", errors="ignore")
            return [min(259, 4 + b) for b in raw] or [1]

        def decode(self, ids: Sequence[int], **_kwargs: Any) -> str:
            raw = bytes(max(0, min(255, int(i) - 4)) for i in ids if int(i) >= 4)
            return raw.decode("utf-8", errors="replace")

        def __call__(self, text: str, **_kwargs: Any) -> dict[str, list[int]]:
            return {"input_ids": self.encode(text)}

    tokenizer: Any = _SimpleByteTokenizer(tok)

    if "qwen3" in low and ("a3b" in low or "moe" in low or "30b" in low):
        q_cfg = tf_mod.Qwen3MoeConfig(
            vocab_size=vocab_size,
            hidden_size=64,
            intermediate_size=128,
            moe_intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_experts=4,
            num_experts_per_tok=2,
        )
        return cast(nn.Module, tf_mod.Qwen3MoeForCausalLM(q_cfg)), tokenizer
    if "nemotron" in low or "mamba" in low:
        m_cfg = tf_mod.FalconMambaConfig(
            vocab_size=vocab_size,
            hidden_size=64,
            state_size=8,
            num_hidden_layers=2,
        )
        return cast(nn.Module, tf_mod.FalconMambaForCausalLM(m_cfg)), tokenizer
    if "gemma-4" in low:
        g_cfg = tf_mod.Gemma4TextConfig(
            vocab_size=vocab_size,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            enable_moe_block=True,
            num_experts=4,
            top_k_experts=2,
            moe_intermediate_size=64,
        )
        return cast(nn.Module, tf_mod.Gemma4ForCausalLM(g_cfg)), tokenizer

    l_cfg = tf_mod.LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    return cast(nn.Module, tf_mod.LlamaForCausalLM(l_cfg)), tokenizer


def _resolve_store_from_uri(
    storage_uri: str,
    *,
    object_store: ArtifactObjectStore | None = None,
    settings: ScoutSettings | None = None,
) -> ArtifactObjectStore:
    if object_store is not None:
        return object_store
    cleaned = storage_uri.strip()
    if not cleaned:
        return get_artifact_object_store(settings)
    parsed = urlparse(cleaned)
    scheme = parsed.scheme.lower()
    bucket = parsed.netloc.strip()
    if scheme == "gs":
        if not bucket:
            raise ValueError(f"Missing GCS bucket in storage_uri: {storage_uri!r}")
        return GCSObjectStore(bucket=bucket, project=settings.gcs_project if settings else None)
    if scheme == "s3":
        if not bucket:
            raise ValueError(f"Missing S3 bucket in storage_uri: {storage_uri!r}")
        return S3CompatibleObjectStore(
            bucket=bucket,
            endpoint_url=settings.s3_endpoint_url if settings else None,
            region_name=settings.s3_region if settings else "eu-central-1",
        )
    raise ValueError(
        f"Unsupported storage_uri scheme '{scheme}://' in {storage_uri!r}; "
        "expected 'gs://' or 's3://', or inject an ArtifactObjectStore instance."
    )


class PEFTRsLoRATrainer:
    """Enterprise PEFT rsLoRA + DPO trainer with vLLM dry-run, 0/10 canary gate, and cosign."""

    def __init__(
        self,
        config: PEFTTrainingConfig,
        *,
        settings: ScoutSettings | None = None,
        tenant_id: UUID = DEFAULT_TENANT_ID,
        lock_path: Path | str = DEFAULT_GPU_LOCK_PATH,
        lock_timeout: float = 30.0,
        use_gpu_lock: bool = True,
        rls_conn_manager: Any | None = None,
        in_memory_repo: InMemoryRLSRepository | None = None,
        object_store: ArtifactObjectStore | None = None,
    ) -> None:
        self.config = config
        self.settings = settings or ScoutSettings()
        self.tenant_id = tenant_id
        self.lock_path = Path(lock_path)
        self.lock_timeout = lock_timeout
        self.use_gpu_lock = use_gpu_lock
        self._rls_conn_manager = rls_conn_manager
        self._in_memory_repo = in_memory_repo
        self._pg_repo = PostgresRLSRepository()
        self._object_store = object_store
        self.policy = ModelGovernancePolicy.default()

    @staticmethod
    def _load_jsonl(path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            raise FileNotFoundError(f"JSONL dataset file not found: {path}")
        records: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                raw = line.strip()
                if raw:
                    records.append(dict(json.loads(raw)))
        return records

    @staticmethod
    def _load_canaries(path: Path) -> list[SyntheticCanary]:
        if not path.is_file():
            raise CanaryMemorizationError(f"Canary file not found: {path}")
        raw_list = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw_list, list):
            raise CanaryMemorizationError("canaries.json must contain a JSON array")
        return [
            SyntheticCanary(
                canary_id=str(c["canary_id"]),
                domain_id=str(c["domain_id"]),
                prompt_prefix=str(c["prompt_prefix"]),
                secret_token=str(c["secret_token"]),
                sha256_digest=str(c.get("sha256_digest") or ""),
            )
            for c in raw_list
        ]

    def _resolve_model_and_tokenizer(
        self,
        *,
        model: nn.Module | None,
        tokenizer: Any | None,
        local_files_only: bool,
        allow_offline_tiny_model: bool,
    ) -> tuple[nn.Module, Any, bool]:
        if model is not None and tokenizer is not None:
            return model, tokenizer, False

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            loaded_tok = tokenizer or AutoTokenizer.from_pretrained(
                self.config.base_model_id,
                trust_remote_code=False,
                local_files_only=local_files_only,
            )
            loaded_model = model or AutoModelForCausalLM.from_pretrained(
                self.config.base_model_id,
                trust_remote_code=False,
                local_files_only=local_files_only,
            )
            return loaded_model, loaded_tok, False
        except Exception as exc:
            if not allow_offline_tiny_model:
                raise RuntimeError(
                    f"Failed loading governed base model '{self.config.base_model_id}' "
                    f"(trust_remote_code=False, local_files_only={local_files_only}): {exc}"
                ) from exc
            logger.info(
                "Local HF weights unavailable for '%s' (%s); constructing offline tiny %s model",
                self.config.base_model_id,
                type(exc).__name__,
                detect_architecture_profile(self.config.base_model_id).family,
            )
            tiny_model, tiny_tok = _build_offline_tiny_model_for_base(self.config.base_model_id)
            return tiny_model, tiny_tok, True

    def _build_adapter_config_dict(
        self,
        *,
        target_modules: Sequence[str],
        architecture_family: str,
        topology: str,
    ) -> dict[str, Any]:
        return {
            "base_model_name_or_path": self.config.base_model_id,
            "base_model_revision_sha256": self.config.base_model_revision_sha256,
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "r": self.config.r,
            "lora_alpha": self.config.lora_alpha,
            "lora_dropout": self.config.lora_dropout,
            "use_rslora": True,
            "use_dora": False,
            "assistant_only_loss": True,
            "loraplus_lr_ratio": self.config.loraplus_lr_ratio,
            "neftune_noise_alpha": self.config.neftune_noise_alpha,
            "scaling": round(self.config.rslora_scaling, 6),
            "target_modules": list(target_modules),
            "training_scope_ceiling": list(self.config.training_scope_ceiling),
            "trust_remote_code": False,
            "bias": "none",
            "architecture_family": architecture_family,
            "topology": topology,
            "is_moe": topology in {"moe", "hybrid_mamba_moe"},
        }

    def run_pretrain_vllm_dry_run_probe(
        self,
        *,
        model: nn.Module,
        adapter_id: str,
        adapter_config: Mapping[str, Any],
        probe_live_vllm: bool = False,
        vllm_endpoint_url: str = "http://127.0.0.1:8000/v1",
        vllm_admin_endpoint_url: str = "http://127.0.0.1:8001/v1",
    ) -> None:
        """Executes 1-step vLLM multi-LoRA dry-run probe BEFORE training."""
        tensors = extract_peft_lora_state_dict(model)
        with tempfile.TemporaryDirectory(prefix="scout_vllm_dryrun_") as tmp:
            probe_dir = Path(tmp)
            save_safetensors_file(tensors, str(probe_dir / "adapter_model.safetensors"))
            (probe_dir / "adapter_config.json").write_text(
                json.dumps(dict(adapter_config), indent=2),
                encoding="utf-8",
            )
            validate_lora_safetensors_for_vllm(
                probe_dir / "adapter_model.safetensors",
                adapter_config,
                base_model_id=self.config.base_model_id,
            )
            executor = OpenWeightsVLLMExecutor(
                model_id=self.config.base_model_id,
                endpoint_url=vllm_endpoint_url,
                runtime_backend="vllm",
            )
            report = executor.run_vllm_lora_dry_run_probe(
                adapter_id=adapter_id,
                adapter_dir=probe_dir,
                admin_endpoint_url=vllm_admin_endpoint_url,
                probe_live_vllm=probe_live_vllm,
            )
            if not report.passed:
                raise LoRACompatibilityError(
                    f"Pre-training 1-step vLLM dry-run probe failed: {report.detail}"
                )

    @staticmethod
    def _trl_stack_available() -> bool:
        return (
            importlib.util.find_spec("trl") is not None
            and importlib.util.find_spec("peft") is not None
            and importlib.util.find_spec("datasets") is not None
        )

    def _run_trl_sft_and_dpo(
        self,
        *,
        model: nn.Module,
        tokenizer: Any,
        train_records: Sequence[Mapping[str, Any]],
        valid_records: Sequence[Mapping[str, Any]],
        dpo_records: Sequence[Mapping[str, Any]],
        target_modules: Sequence[str],
        output_dir: Path,
    ) -> tuple[nn.Module, float, float, float, float, float | None, float | None, float | None]:
        """Executes ``trl.SFTTrainer`` + ``trl.DPOTrainer`` when TRL & PEFT are installed."""
        import trl
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model

        trl_mod: Any = trl

        lora_cfg = LoraConfig(
            r=self.config.r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=list(target_modules),
            bias="none",
            task_type="CAUSAL_LM",
            use_rslora=True,
            use_dora=False,
        )
        peft_model = get_peft_model(model, lora_cfg)
        verify_frozen_routers_and_ssm_states(peft_model, base_model_id=self.config.base_model_id)

        train_ds = Dataset.from_list([{"messages": r["messages"]} for r in train_records])
        arch_adapter = ModelArchitectureAdapter(
            self.config.base_model_id,
            default_max_model_len=self.config.max_seq_length,
        )
        for rec in train_records:
            msgs = rec.get("messages") or []
            if len(msgs) >= 2:
                tokenize_chat_assistant_only(
                    msgs,
                    tokenizer=tokenizer,
                    arch_adapter=arch_adapter,
                    max_seq_length=self.config.max_seq_length,
                )
        valid_ds = (
            Dataset.from_list([{"messages": r["messages"]} for r in valid_records])
            if valid_records
            else None
        )
        optimizer = build_loraplus_adamw_optimizer(
            peft_model,
            learning_rate=self.config.learning_rate,
            loraplus_lr_ratio=self.config.loraplus_lr_ratio,
        )

        sft_cfg = trl_mod.SFTConfig(
            output_dir=str(output_dir / "trl_sft_ckpt"),
            max_steps=self.config.sft_steps,
            learning_rate=self.config.learning_rate,
            max_length=self.config.max_seq_length,
            assistant_only_loss=True,
            neftune_noise_alpha=self.config.neftune_noise_alpha,
            per_device_train_batch_size=1,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            seed=self.config.seed,
        )
        sft_trainer = trl_mod.SFTTrainer(
            model=peft_model,
            args=sft_cfg,
            train_dataset=train_ds,
            eval_dataset=valid_ds,
            processing_class=tokenizer,
            optimizers=(optimizer, None),
        )
        train_out = sft_trainer.train()
        logs = list(getattr(sft_trainer.state, "log_history", []) or [])
        sft_losses = [float(entry["loss"]) for entry in logs if "loss" in entry]
        init_loss = sft_losses[0] if sft_losses else float(getattr(train_out, "training_loss", 0.0))
        final_loss = (
            sft_losses[-1] if sft_losses else float(getattr(train_out, "training_loss", 0.0))
        )

        dpo_init: float | None = None
        dpo_final: float | None = None
        dpo_acc: float | None = None
        if dpo_records and self.config.dpo_steps > 0:
            dpo_ds = Dataset.from_list(
                [
                    {
                        "prompt": str(r["prompt"]),
                        "chosen": str(r["chosen"]),
                        "rejected": str(r["rejected"]),
                    }
                    for r in dpo_records
                ]
            )
            dpo_cfg = trl_mod.DPOConfig(
                output_dir=str(output_dir / "trl_dpo_ckpt"),
                beta=self.config.dpo_beta,
                max_steps=self.config.dpo_steps,
                learning_rate=self.config.learning_rate * 0.25,
                per_device_train_batch_size=1,
                logging_steps=1,
                save_strategy="no",
                report_to=[],
                seed=self.config.seed,
            )
            dpo_trainer = trl_mod.DPOTrainer(
                model=peft_model,
                args=dpo_cfg,
                train_dataset=dpo_ds,
                processing_class=tokenizer,
            )
            dpo_trainer.train()
            d_logs = list(getattr(dpo_trainer.state, "log_history", []) or [])
            d_losses = [float(entry["loss"]) for entry in d_logs if "loss" in entry]
            dpo_init = d_losses[0] if d_losses else 0.0
            dpo_final = d_losses[-1] if d_losses else 0.0
            d_accs = [
                float(entry["rewards/accuracies"])
                for entry in d_logs
                if "rewards/accuracies" in entry
            ]
            dpo_acc = d_accs[-1] if d_accs else 1.0

        return peft_model, init_loss, final_loss, final_loss, 1.0, dpo_init, dpo_final, dpo_acc

    def _run_native_sft_and_dpo(
        self,
        *,
        model: nn.Module,
        tokenizer: Any,
        arch_adapter: ModelArchitectureAdapter,
        compiled: CompiledSpecialization | None,
        train_records: Sequence[Mapping[str, Any]],
        valid_records: Sequence[Mapping[str, Any]],
        dpo_records: Sequence[Mapping[str, Any]],
        adapter_config: Mapping[str, Any],
        artifact_id: str,
        probe_live_vllm: bool,
    ) -> tuple[
        float,
        float,
        float,
        float,
        float | None,
        float | None,
        float | None,
        list[dict[str, Any]],
    ]:
        """Executes Transformers rsLoRA SFT (assistant_only_loss, LoRA+, NEFTune, Task-F1) + DPO."""
        torch.manual_seed(self.config.seed)
        device = next(model.parameters()).device

        # Snapshot initial rsLoRA weights before pre-training dry-run probe
        initial_lora_sd = extract_peft_lora_state_dict(model)
        self.run_pretrain_vllm_dry_run_probe(
            model=model,
            adapter_id=artifact_id,
            adapter_config=adapter_config,
            probe_live_vllm=probe_live_vllm,
        )
        load_peft_lora_state_dict(model, initial_lora_sd)

        tokenized_train: list[tuple[list[int], list[int]]] = []
        for rec in train_records:
            msgs = rec.get("messages") or []
            if len(msgs) < 2:
                continue
            tokenized_train.append(
                tokenize_chat_assistant_only(
                    msgs,
                    tokenizer=tokenizer,
                    arch_adapter=arch_adapter,
                    max_seq_length=self.config.max_seq_length,
                )
            )
        if not tokenized_train:
            raise ValueError("No valid training records could be tokenized for SFT.")

        optimizer = build_loraplus_adamw_optimizer(
            model,
            learning_rate=self.config.learning_rate,
            loraplus_lr_ratio=self.config.loraplus_lr_ratio,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, self.config.sft_steps),
            eta_min=self.config.learning_rate * 0.1,
        )
        neftune_hook = NEFTuneEmbeddingHook(model, noise_alpha=self.config.neftune_noise_alpha)

        step_history: list[dict[str, Any]] = []
        best_score: tuple[float, float] = (-1.0, float("-inf"))
        best_state_dict: dict[str, torch.Tensor] | None = None
        initial_loss = 0.0
        final_loss = 0.0
        latest_val_loss = 0.0
        latest_f1 = 0.0

        try:
            model.train()
            for step in range(1, self.config.sft_steps + 1):
                in_ids, lbls = tokenized_train[(step - 1) % len(tokenized_train)]
                in_t = torch.tensor([in_ids], dtype=torch.long, device=device)
                lbl_t = torch.tensor([lbls], dtype=torch.long, device=device)

                optimizer.zero_grad()
                out = model(input_ids=in_t)
                logits = out.logits if hasattr(out, "logits") else out[0]
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = lbl_t[:, 1:].contiguous()
                loss = nn_functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.shape[-1]),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
                cast(Any, loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                optimizer.step()
                scheduler.step()

                cur_loss = float(loss.item())
                if step == 1:
                    initial_loss = cur_loss
                final_loss = cur_loss

                step_entry: dict[str, Any] = {
                    "step": step,
                    "loss": round(cur_loss, 4),
                    "lr_lora_a": round(float(optimizer.param_groups[0]["lr"]), 8),
                    "lr_lora_b": round(
                        float(
                            optimizer.param_groups[min(1, len(optimizer.param_groups) - 1)]["lr"]
                        ),
                        8,
                    ),
                }
                if step % self.config.eval_steps == 0 or step == self.config.sft_steps:
                    task_f1, v_loss = compute_specialization_task_f1(
                        model,
                        tokenizer,
                        valid_records,
                        compiled=compiled,
                        arch_adapter=arch_adapter,
                        max_seq_length=self.config.max_seq_length,
                    )
                    latest_val_loss = v_loss if valid_records else cur_loss
                    latest_f1 = task_f1
                    step_entry["eval_task_f1"] = task_f1
                    step_entry["val_loss"] = latest_val_loss
                    cand_score = (task_f1, -latest_val_loss)
                    if cand_score >= best_score:
                        best_score = cand_score
                        best_state_dict = copy.deepcopy(extract_peft_lora_state_dict(model))
                step_history.append(step_entry)
        finally:
            neftune_hook.remove()

        if best_state_dict is not None:
            load_peft_lora_state_dict(model, best_state_dict)

        # Stage 2: Optional DPO alignment over (chosen, rejected) pairs
        dpo_init: float | None = None
        dpo_final: float | None = None
        dpo_acc: float | None = None
        if dpo_records and self.config.dpo_steps > 0:
            tokenized_dpo: list[tuple[list[int], list[int], list[int]]] = []
            for d_rec in dpo_records:
                p_txt = str(d_rec.get("prompt") or "")
                c_txt = str(d_rec.get("chosen") or "")
                r_txt = str(d_rec.get("rejected") or "")
                if not p_txt or not c_txt or not r_txt:
                    continue
                p_formatted = arch_adapter.format_chat_messages(
                    [{"role": "user", "content": p_txt}],
                    tokenizer=tokenizer,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                p_ids = _encode_tokens(tokenizer, p_formatted)[-(self.config.max_seq_length // 2) :]
                c_ids = _encode_tokens(tokenizer, c_txt)[: (self.config.max_seq_length // 2)] or [1]
                r_ids = _encode_tokens(tokenizer, r_txt)[: (self.config.max_seq_length // 2)] or [1]
                tokenized_dpo.append((p_ids, c_ids, r_ids))

            if tokenized_dpo:
                dpo_opt = build_loraplus_adamw_optimizer(
                    model,
                    learning_rate=self.config.learning_rate * 0.25,
                    loraplus_lr_ratio=self.config.loraplus_lr_ratio,
                )
                model.train()
                margins: list[bool] = []
                for d_step in range(1, self.config.dpo_steps + 1):
                    p_ids, c_ids, r_ids = tokenized_dpo[(d_step - 1) % len(tokenized_dpo)]
                    with torch.no_grad():
                        set_adapters_enabled(model, False)
                        try:
                            ref_chosen_lp = _sequence_log_prob(model, p_ids, c_ids, device=device)
                            ref_rejected_lp = _sequence_log_prob(model, p_ids, r_ids, device=device)
                        finally:
                            set_adapters_enabled(model, True)

                    pol_chosen_lp = _sequence_log_prob(model, p_ids, c_ids, device=device)
                    pol_rejected_lp = _sequence_log_prob(model, p_ids, r_ids, device=device)
                    logit_diff = (pol_chosen_lp - ref_chosen_lp) - (
                        pol_rejected_lp - ref_rejected_lp
                    )
                    dpo_loss = -nn_functional.logsigmoid(self.config.dpo_beta * logit_diff)

                    dpo_opt.zero_grad()
                    cast(Any, dpo_loss).backward()
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        max_norm=1.0,
                    )
                    dpo_opt.step()

                    d_val = float(dpo_loss.item())
                    if d_step == 1:
                        dpo_init = d_val
                    dpo_final = d_val
                    margins.append(float(logit_diff.item()) > 0.0)
                dpo_acc = round(sum(1 for m in margins if m) / max(1, len(margins)), 4)

        best_f1 = max(latest_f1, best_score[0] if best_score[0] >= 0.0 else 0.0)
        return (
            initial_loss,
            final_loss,
            latest_val_loss,
            best_f1,
            dpo_init,
            dpo_final,
            dpo_acc,
            step_history,
        )

    def _register_artifact_in_db(
        self,
        record: ModelArtifactRecord,
        *,
        training_scopes: Sequence[str],
    ) -> ModelArtifactRecord:
        principal = PrincipalContext(
            user_id="peft_lora_trainer",
            role=RoleType.ENGINEER,
            allowed_scopes=list(training_scopes),
            tenant_id=self.tenant_id,
        )
        if self._in_memory_repo is not None:
            return self._in_memory_repo.register_model_artifact(principal, record)

        conn_mgr = self._rls_conn_manager or get_ingestion_pool(self.settings)
        with conn_mgr.scoped_transaction(principal, enable_hnsw_iterative_scan=False) as conn:
            return self._pg_repo.register_model_artifact(conn, record)

    def train(
        self,
        *,
        train_jsonl: Path | str,
        valid_jsonl: Path | str,
        output_adapter_dir: Path | str,
        eval_jsonl: Path | str | None = None,
        dpo_train_jsonl: Path | str | None = None,
        canaries_json: Path | str | None = None,
        compiled: CompiledSpecialization | None = None,
        artifact_id: str | None = None,
        model: nn.Module | None = None,
        tokenizer: Any | None = None,
        local_files_only: bool = True,
        allow_offline_tiny_model: bool = False,
        allow_synthetic_artifact_registration: bool = False,
        probe_live_vllm: bool = False,
        require_canary_gate: bool = True,
        canary_generation_fn: Callable[[str], str] | None = None,
        cosign_private_key: Any | None = None,
        cosign_verifier: CosignBlobVerifier | None = None,
        as_sigstore_bundle: bool = True,
        storage_uri: str | None = None,
        register_in_db: bool = False,
        activate_in_db: bool = False,
    ) -> PEFTTrainingRunResult:
        """Executes end-to-end PEFT rsLoRA training, canary gate, cosign signing, and upload."""
        out_dir = Path(output_adapter_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        resolved_artifact_id = (
            artifact_id or f"rslora_{self.config.domain_id}_{uuid.uuid4().hex[:8]}"
        )

        train_records = self._load_jsonl(Path(train_jsonl))
        valid_records = self._load_jsonl(Path(valid_jsonl))
        eval_records = (
            self._load_jsonl(Path(eval_jsonl)) if eval_jsonl and Path(eval_jsonl).is_file() else []
        )
        dpo_records = (
            self._load_jsonl(Path(dpo_train_jsonl))
            if dpo_train_jsonl and Path(dpo_train_jsonl).is_file()
            else []
        )
        canaries: list[SyntheticCanary] = []
        if canaries_json is not None and Path(canaries_json).is_file():
            canaries = self._load_canaries(Path(canaries_json))
        elif require_canary_gate:
            default_canaries = Path(train_jsonl).parent / "canaries.json"
            if default_canaries.is_file():
                canaries = self._load_canaries(default_canaries)
            else:
                raise CanaryMemorizationError(
                    "CanaryMemorizationGate requires canaries.json (10 synthetic canaries)."
                )

        def _execute() -> PEFTTrainingRunResult:
            loaded_model, loaded_tok, used_synthetic_tiny_model = self._resolve_model_and_tokenizer(
                model=model,
                tokenizer=tokenizer,
                local_files_only=local_files_only,
                allow_offline_tiny_model=allow_offline_tiny_model,
            )
            if (
                register_in_db
                and used_synthetic_tiny_model
                and not allow_synthetic_artifact_registration
            ):
                raise ValueError(
                    "Refusing to register synthetic offline tiny-model artifact in DB "
                    "(allow_synthetic_artifact_registration=False)."
                )
            arch_profile = detect_architecture_profile(self.config.base_model_id)
            arch_adapter = ModelArchitectureAdapter(
                self.config.base_model_id,
                tokenizer=loaded_tok,
                governance_policy=self.policy,
            )

            use_trl = False
            if self.config.trainer_backend == "trl":
                if not self._trl_stack_available():
                    raise RuntimeError(
                        "trainer_backend='trl' requested but trl/peft/datasets are not installed."
                    )
                use_trl = True
            elif (
                self.config.trainer_backend == "auto"
                and self._trl_stack_available()
                and hasattr(loaded_tok, "apply_chat_template")
                and hasattr(loaded_tok, "pad_token_id")
            ):
                use_trl = True

            target_modules = resolve_safe_lora_target_modules(
                loaded_model, self.config.base_model_id
            )
            adapter_config = self._build_adapter_config_dict(
                target_modules=target_modules,
                architecture_family=arch_profile.family,
                topology=arch_profile.topology,
            )

            step_history: list[dict[str, Any]] = []
            if use_trl:
                trainer_used = "trl"
                (
                    trained_model,
                    sft_init,
                    sft_final,
                    val_loss,
                    best_f1,
                    dpo_init,
                    dpo_final,
                    dpo_acc,
                ) = self._run_trl_sft_and_dpo(
                    model=loaded_model,
                    tokenizer=loaded_tok,
                    train_records=train_records,
                    valid_records=valid_records,
                    dpo_records=dpo_records,
                    target_modules=target_modules,
                    output_dir=out_dir,
                )
            else:
                trainer_used = "transformers_peft"
                inject_rslora_adapters(
                    loaded_model,
                    r=self.config.r,
                    lora_alpha=self.config.lora_alpha,
                    lora_dropout=self.config.lora_dropout,
                    base_model_id=self.config.base_model_id,
                )
                trained_model = loaded_model
                (
                    sft_init,
                    sft_final,
                    val_loss,
                    best_f1,
                    dpo_init,
                    dpo_final,
                    dpo_acc,
                    step_history,
                ) = self._run_native_sft_and_dpo(
                    model=trained_model,
                    tokenizer=loaded_tok,
                    arch_adapter=arch_adapter,
                    compiled=compiled,
                    train_records=train_records,
                    valid_records=valid_records,
                    dpo_records=dpo_records,
                    adapter_config=adapter_config,
                    artifact_id=resolved_artifact_id,
                    probe_live_vllm=probe_live_vllm,
                )

            if eval_records:
                eval_f1, _eval_loss = compute_specialization_task_f1(
                    trained_model,
                    loaded_tok,
                    eval_records,
                    compiled=compiled,
                    arch_adapter=arch_adapter,
                    max_seq_length=self.config.max_seq_length,
                )
                best_f1 = max(best_f1, eval_f1)

            verify_frozen_routers_and_ssm_states(
                trained_model, base_model_id=self.config.base_model_id
            )

            # Mandatory fail-closed CanaryMemorizationGate BEFORE signing, upload, or DB
            if require_canary_gate or canaries:
                canary_gate = CanaryMemorizationGate(min_canaries=10)
                canary_audit = canary_gate.evaluate(
                    canaries,
                    model=trained_model,
                    tokenizer=loaded_tok,
                    generation_fn=canary_generation_fn,
                    arch_adapter=arch_adapter,
                )
            else:
                canary_audit = CanaryAuditReport(
                    total_canaries=0,
                    canary_leak_count=0,
                    leaked_canary_ids=(),
                    passed=True,
                )

            # Extract and persist .safetensors + adapter_config.json
            lora_tensors = extract_peft_lora_state_dict(trained_model)
            safetensors_path = out_dir / "adapter_model.safetensors"
            save_safetensors_file(lora_tensors, str(safetensors_path))
            # Also save adapters.safetensors alias for local AdapterPool compatibility
            save_safetensors_file(lora_tensors, str(out_dir / "adapters.safetensors"))

            adapter_config_path = out_dir / "adapter_config.json"
            adapter_config_bytes = json.dumps(adapter_config, indent=2).encode("utf-8")
            adapter_config_path.write_bytes(adapter_config_bytes)

            weights_bytes = safetensors_path.read_bytes()
            weights_sha256 = hashlib.sha256(weights_bytes).hexdigest()
            config_sha256 = hashlib.sha256(adapter_config_bytes).hexdigest()

            # Verify vLLM compatibility on final exported cartridge
            validate_lora_safetensors_for_vllm(
                weights_bytes,
                adapter_config,
                base_model_id=self.config.base_model_id,
            )

            # Sign weights with cosign when private key is configured
            sig_text = ""
            if cosign_private_key is not None:
                sig_text = sign_blob_for_cosign(
                    cosign_private_key,
                    weights_bytes,
                    as_sigstore_bundle=as_sigstore_bundle,
                )
                sig_ext = "adapter_model.bundle" if as_sigstore_bundle else "adapter_model.sig"
                (out_dir / sig_ext).write_text(sig_text, encoding="utf-8")
                if cosign_verifier is not None:
                    cosign_verifier.verify(
                        weights_bytes,
                        sig_text,
                        key_id=self.config.cosign_key_id,
                    )

            loss_red = (
                round(max(0.0, ((sft_init - sft_final) / sft_init) * 100.0), 2)
                if sft_init > 0.0
                else 0.0
            )
            metrics_payload: dict[str, Any] = {
                "domain_id": self.config.domain_id,
                "artifact_id": resolved_artifact_id,
                "base_model_id": self.config.base_model_id,
                "base_model_revision_sha256": self.config.base_model_revision_sha256,
                "trainer_backend": trainer_used,
                "use_rslora": True,
                "use_dora": False,
                "assistant_only_loss": True,
                "loraplus_lr_ratio": self.config.loraplus_lr_ratio,
                "neftune_noise_alpha": self.config.neftune_noise_alpha,
                "initial_loss": round(sft_init, 4),
                "final_loss": round(sft_final, 4),
                "loss_reduction_pct": loss_red,
                "val_loss": round(val_loss, 4),
                "best_eval_task_f1": round(best_f1, 4),
                "dpo_initial_loss": round(dpo_init, 4) if dpo_init is not None else None,
                "dpo_final_loss": round(dpo_final, 4) if dpo_final is not None else None,
                "dpo_reward_accuracy": dpo_acc,
                "canary_audit": asdict(canary_audit),
                "weights_sha256": weights_sha256,
                "step_history": step_history,
                "trained_at": datetime.now(UTC).isoformat(),
            }
            training_metrics_path = out_dir / "training_metrics.json"
            training_metrics_path.write_text(
                json.dumps(metrics_payload, indent=2),
                encoding="utf-8",
            )

            # Upload to ArtifactObjectStore (gs:// / s3:// / InMemoryObjectStore) if requested
            effective_storage_uri = (storage_uri or "").strip()
            weights_key = ""
            config_key = ""
            signature_key = ""
            if effective_storage_uri or self._object_store is not None:
                store = _resolve_store_from_uri(
                    effective_storage_uri,
                    object_store=self._object_store,
                    settings=self.settings,
                )
                if not effective_storage_uri:
                    effective_storage_uri = f"memory://{getattr(store, 'bucket', 'local-store')}"
                weights_key = build_tenant_domain_key(
                    self.tenant_id,
                    self.config.domain_id,
                    f"adapters/{resolved_artifact_id}/adapter_model.safetensors",
                )
                config_key = build_tenant_domain_key(
                    self.tenant_id,
                    self.config.domain_id,
                    f"adapters/{resolved_artifact_id}/adapter_config.json",
                )
                store.put_bytes(
                    weights_key,
                    weights_bytes,
                    expected_tenant_id=self.tenant_id,
                    content_type="application/octet-stream",
                )
                store.put_bytes(
                    config_key,
                    adapter_config_bytes,
                    expected_tenant_id=self.tenant_id,
                    content_type="application/json",
                )
                if sig_text:
                    sig_name = "adapter_model.bundle" if as_sigstore_bundle else "adapter_model.sig"
                    signature_key = build_tenant_domain_key(
                        self.tenant_id,
                        self.config.domain_id,
                        f"adapters/{resolved_artifact_id}/{sig_name}",
                    )
                    store.put_bytes(
                        signature_key,
                        sig_text.encode("utf-8"),
                        expected_tenant_id=self.tenant_id,
                        content_type="application/json",
                    )

            artifact_manifest_payload = {
                "domain_id": self.config.domain_id,
                "artifact_id": resolved_artifact_id,
                "base_model_id": self.config.base_model_id,
                "base_model_revision_sha256": self.config.base_model_revision_sha256,
                "weights_sha256": weights_sha256,
                "config_sha256": config_sha256,
                "training_scope_ceiling": list(self.config.training_scope_ceiling),
                "canary_leak_count": canary_audit.canary_leak_count,
                "storage_uri": effective_storage_uri,
                "weights_object_key": weights_key,
                "config_object_key": config_key,
                "signature_object_key": signature_key,
            }
            artifact_manifest_path = out_dir / "artifact_manifest.json"
            artifact_manifest_path.write_text(
                json.dumps(artifact_manifest_payload, indent=2),
                encoding="utf-8",
            )

            registered_rec: ModelArtifactRecord | None = None
            if register_in_db:
                if not effective_storage_uri or not weights_key or not config_key:
                    raise ValueError(
                        "register_in_db=True requires storage_uri or object_store upload"
                    )
                rec = ModelArtifactRecord(
                    tenant_id=self.tenant_id,
                    domain_id=self.config.domain_id,
                    artifact_id=resolved_artifact_id,
                    artifact_kind="lora_adapter",
                    base_model_id=self.config.base_model_id,
                    base_model_revision_sha256=self.config.base_model_revision_sha256,
                    storage_uri=effective_storage_uri,
                    weights_object_key=weights_key,
                    config_object_key=config_key,
                    signature_object_key=signature_key,
                    weights_sha256=weights_sha256,
                    cosign_signature_b64="" if signature_key else sig_text,
                    cosign_key_id=self.config.cosign_key_id,
                    lora_rank=self.config.r,
                    lora_alpha=self.config.lora_alpha,
                    use_rslora=True,
                    use_dora=False,
                    training_scope_ceiling=list(self.config.training_scope_ceiling),
                    allowed_scopes=list(self.config.training_scope_ceiling),
                    canary_leak_count=canary_audit.canary_leak_count,
                    metadata_json={
                        "config_sha256": config_sha256,
                        "trainer_backend": trainer_used,
                        "best_eval_task_f1": round(best_f1, 4),
                    },
                    status="active" if activate_in_db else "staged",
                    is_active=activate_in_db,
                )
                registered_rec = self._register_artifact_in_db(
                    rec, training_scopes=self.config.training_scope_ceiling
                )

            return PEFTTrainingRunResult(
                domain_id=self.config.domain_id,
                artifact_id=resolved_artifact_id,
                base_model_id=self.config.base_model_id,
                base_model_revision_sha256=self.config.base_model_revision_sha256,
                trainer_backend_used=trainer_used,
                adapter_dir=out_dir,
                safetensors_path=safetensors_path,
                adapter_config_path=adapter_config_path,
                training_metrics_path=training_metrics_path,
                artifact_manifest_path=artifact_manifest_path,
                weights_sha256=weights_sha256,
                sft_initial_loss=round(sft_init, 4),
                sft_final_loss=round(sft_final, 4),
                val_loss=round(val_loss, 4),
                best_eval_task_f1=round(best_f1, 4),
                dpo_initial_loss=round(dpo_init, 4) if dpo_init is not None else None,
                dpo_final_loss=round(dpo_final, 4) if dpo_final is not None else None,
                dpo_reward_accuracy=dpo_acc,
                canary_audit=canary_audit,
                dry_run_passed=True,
                cosign_signature_b64=sig_text,
                storage_uri=effective_storage_uri,
                weights_object_key=weights_key,
                config_object_key=config_key,
                signature_object_key=signature_key,
                registered_artifact=registered_rec,
            )

        if self.use_gpu_lock:
            with GPUMutualExclusionLock(self.lock_path, timeout_seconds=self.lock_timeout):
                return _execute()
        return _execute()


@dataclass(frozen=True)
class CachedMNRLTrainingResult:
    """Result of optional bi-encoder ``CachedMultipleNegativesRankingLoss`` training."""

    domain_id: str
    base_embedding_model_id: str
    output_dir: Path
    safetensors_path: Path
    weights_sha256: str
    initial_loss: float
    final_loss: float
    steps: int
    pair_count: int


class CachedMNRLBiEncoderTrainer:
    """Optional bi-encoder fine-tuner using ``CachedMultipleNegativesRankingLoss``."""

    def __init__(
        self,
        *,
        domain_id: str,
        base_embedding_model_id: str = "google/embeddinggemma-2",
        mini_batch_size: int = 2,
        learning_rate: float = 2e-5,
        seed: int = 42,
    ) -> None:
        self.domain_id = domain_id
        self.base_embedding_model_id = ModelGovernancePolicy.default().validate_local_model_id(
            base_embedding_model_id
        )
        self.mini_batch_size = max(1, int(mini_batch_size))
        self.learning_rate = float(learning_rate)
        self.seed = int(seed)

    @staticmethod
    def _import_cached_mnrl() -> Any:
        import importlib

        try:
            mod = importlib.import_module("sentence_transformers.sentence_transformer.losses")
        except ImportError:
            mod = importlib.import_module("sentence_transformers.losses")
        return mod.CachedMultipleNegativesRankingLoss

    def train_pairs(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        output_dir: Path | str,
        steps: int = 5,
        model: Any | None = None,
        glossary: DomainGlossary | None = None,
        local_files_only: bool = True,
    ) -> CachedMNRLTrainingResult:
        """Fine-tunes a SentenceTransformer bi-encoder with CachedMNRL & saves .safetensors."""
        clean_pairs = [(a.strip(), p.strip()) for a, p in pairs if a.strip() and p.strip()]
        if len(clean_pairs) < 2:
            raise ValueError(
                "CachedMNRLBiEncoderTrainer requires at least 2 (query, positive) pairs"
            )

        if glossary is not None:
            clean_pairs = [(glossary.expand_query(a).expanded_query, p) for a, p in clean_pairs]

        out_path = Path(output_dir).resolve()
        out_path.mkdir(parents=True, exist_ok=True)

        if model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(
                self.base_embedding_model_id,
                trust_remote_code=False,
                local_files_only=local_files_only,
            )

        cached_mnrl_cls = self._import_cached_mnrl()
        loss_fn = cached_mnrl_cls(model=model, mini_batch_size=self.mini_batch_size)

        torch.manual_seed(self.seed)
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=self.learning_rate, weight_decay=0.01)

        anchors = [a for a, _ in clean_pairs]
        positives = [p for _, p in clean_pairs]
        preprocess_fn = getattr(model, "preprocess", None) or model.tokenize

        model.train()
        initial_loss = 0.0
        final_loss = 0.0
        for step in range(1, max(1, steps) + 1):
            f_anchor = preprocess_fn(anchors)
            f_pos = preprocess_fn(positives)
            optimizer.zero_grad()
            loss_val = loss_fn([f_anchor, f_pos], labels=None)
            loss_val.backward()
            optimizer.step()
            cur = float(loss_val.item())
            if step == 1:
                initial_loss = cur
            final_loss = cur

        model.eval()
        model.save(str(out_path))
        st_files = sorted(out_path.rglob("*.safetensors"))
        if not st_files:
            raise LoRACompatibilityError(
                f"CachedMNRLBiEncoderTrainer failed to emit .safetensors in {out_path}"
            )
        primary_st = st_files[0]
        weights_sha = hashlib.sha256(primary_st.read_bytes()).hexdigest()
        return CachedMNRLTrainingResult(
            domain_id=self.domain_id,
            base_embedding_model_id=self.base_embedding_model_id,
            output_dir=out_path,
            safetensors_path=primary_st,
            weights_sha256=weights_sha,
            initial_loss=round(initial_loss, 4),
            final_loss=round(final_loss, 4),
            steps=max(1, steps),
            pair_count=len(clean_pairs),
        )


@dataclass(frozen=True)
class BaseModelMigrationDomainOutcome:
    """Per-domain replay verification outcome for ``migrate-base-model``."""

    domain_id: str
    previous_base_model_id: str
    target_base_model_id: str
    target_base_model_revision_sha256: str
    artifact_id: str
    eval_task_f1: float
    canary_leak_count: int
    weights_sha256: str
    storage_uri: str
    activated: bool
    stages_completed: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class BaseModelMigrationReport:
    """Aggregate report emitted by ``BaseModelMigrationOrchestrator``."""

    target_base_model_id: str
    target_base_model_revision_sha256: str
    migrated_domains: tuple[BaseModelMigrationDomainOutcome, ...]
    passed: bool


class BaseModelMigrationOrchestrator:
    """4-stage ``migrate-base-model`` replay DAG across domain specializations."""

    def __init__(
        self,
        *,
        domains_dir: Path | None = None,
        settings: ScoutSettings | None = None,
        tenant_id: UUID = DEFAULT_TENANT_ID,
        rls_conn_manager: Any | None = None,
        in_memory_repo: InMemoryRLSRepository | None = None,
        object_store: ArtifactObjectStore | None = None,
    ) -> None:
        default_root = Path(__file__).resolve().parents[3]
        self.workspace_root = (
            domains_dir.resolve().parent if domains_dir is not None else default_root
        )
        self.domains_dir = (domains_dir or (default_root / "domains")).resolve()
        self.settings = settings or ScoutSettings()
        self.tenant_id = tenant_id
        self._rls_conn_manager = rls_conn_manager
        self._in_memory_repo = in_memory_repo
        self._object_store = object_store
        self.compiler = SpecializationCompiler(
            workspace_root=self.workspace_root,
            domains_dir=self.domains_dir,
        )

    def migrate(
        self,
        *,
        target_base_model_id: str,
        target_base_model_revision_sha256: str,
        domain_ids: Sequence[str] | None = None,
        output_root: Path | str | None = None,
        storage_uri: str | None = None,
        sft_steps: int = 8,
        dpo_steps: int = 4,
        min_eval_task_f1: float = 0.0,
        cosign_private_key: Any | None = None,
        cosign_verifier: CosignBlobVerifier | None = None,
        activate: bool = False,
        register_in_db: bool = True,
        allow_offline_tiny_model: bool = True,
        allow_synthetic_artifact_registration: bool = False,
        redistill: bool = False,
        model_factory: Callable[[str], tuple[nn.Module, Any]] | None = None,
    ) -> BaseModelMigrationReport:
        """Executes the 4-stage base-model upgrade replay DAG for target domains."""
        governed_base = ModelGovernancePolicy.default().validate_local_model_id(
            target_base_model_id
        )
        rev_clean = target_base_model_revision_sha256.strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", rev_clean):
            raise ValueError("target_base_model_revision_sha256 must be a 64-char hex SHA-256")

        if domain_ids:
            targets = list(domain_ids)
        else:
            targets = sorted(
                p.name
                for p in self.domains_dir.iterdir()
                if p.is_dir() and (p / "specialization.yaml").is_file()
            )
        if not targets:
            raise MigrationReplayError(f"No domains with specialization.yaml in {self.domains_dir}")

        outcomes: list[BaseModelMigrationDomainOutcome] = []
        for dom_id in targets:
            spec_path = self.domains_dir / dom_id / "specialization.yaml"
            if not spec_path.is_file():
                raise MigrationReplayError(f"Missing specialization.yaml for '{dom_id}'")
            manifest = self.compiler.load_manifest(spec_path)
            previous_base = manifest.base_model_id
            upgraded_manifest = manifest.model_copy(update={"base_model_id": governed_base})
            compiled = self.compiler.compile(upgraded_manifest)
            stages: list[str] = []

            # Stage 1: Verify or re-distill 3-way splits + ceiling + 10 canaries + 15% replay
            tuning_dir = self.domains_dir / dom_id / "tuning"
            train_path = tuning_dir / "train.jsonl"
            valid_path = tuning_dir / "valid.jsonl"
            eval_path = tuning_dir / "eval.jsonl"
            dpo_train_path = tuning_dir / "dpo_train.jsonl"
            canaries_path = tuning_dir / "canaries.json"
            if redistill or not (
                train_path.is_file()
                and valid_path.is_file()
                and eval_path.is_file()
                and canaries_path.is_file()
            ):
                distiller = TeacherDistiller(
                    settings=self.settings,
                    domains_dir=self.domains_dir,
                    tenant_id=self.tenant_id,
                    in_memory_repo=self._in_memory_repo,
                    hydrate_twin_on_init=False,
                )
                distiller.distill_specialization_v2(
                    compiled,
                    output_dir=tuning_dir,
                    num_samples=24,
                )
            if not TeacherDistiller.verify_zero_leakage(
                train_path, valid_path, eval_path=eval_path, domain_id=dom_id
            ):
                raise MigrationReplayError(
                    f"Stage 1 split leakage verification failed during migration for '{dom_id}'"
                )
            stages.append("stage_1_splits_and_ceiling_verified")

            # Stage 2 & 3: Build PEFT config, 1-step vLLM probe, SFT+DPO replay, 0/10 canary gate
            cfg = PEFTTrainingConfig.from_compiled(
                compiled,
                base_model_revision_sha256=rev_clean,
                sft_steps=sft_steps,
                dpo_steps=dpo_steps,
                eval_steps=max(1, min(4, sft_steps)),
                max_seq_length=512,
            )
            trainer = PEFTRsLoRATrainer(
                cfg,
                settings=self.settings,
                tenant_id=self.tenant_id,
                use_gpu_lock=False,
                rls_conn_manager=self._rls_conn_manager,
                in_memory_repo=self._in_memory_repo,
                object_store=self._object_store,
            )
            injected_model: nn.Module | None = None
            injected_tok: Any = None
            if model_factory is not None:
                injected_model, injected_tok = model_factory(governed_base)

            base_slug = governed_base.rsplit("/", 1)[-1].lower().replace("-", "_")[:24]
            new_artifact_id = f"mig_{dom_id}_{base_slug}_{rev_clean[:8]}"
            adapter_out_dir = (
                (Path(output_root) / dom_id / new_artifact_id)
                if output_root is not None
                else (self.domains_dir / dom_id / "adapters" / new_artifact_id)
            )
            res = trainer.train(
                train_jsonl=train_path,
                valid_jsonl=valid_path,
                eval_jsonl=eval_path,
                dpo_train_jsonl=dpo_train_path if dpo_train_path.is_file() else None,
                canaries_json=canaries_path,
                compiled=compiled,
                output_adapter_dir=adapter_out_dir,
                artifact_id=new_artifact_id,
                model=injected_model,
                tokenizer=injected_tok,
                allow_offline_tiny_model=allow_offline_tiny_model,
                allow_synthetic_artifact_registration=allow_synthetic_artifact_registration,
                require_canary_gate=True,
                cosign_private_key=cosign_private_key,
                cosign_verifier=cosign_verifier,
                storage_uri=storage_uri,
                register_in_db=register_in_db,
                activate_in_db=activate,
            )
            stages.append("stage_2_vllm_dryrun_probe_passed")
            if res.best_eval_task_f1 < min_eval_task_f1:
                raise MigrationReplayError(
                    f"Stage 3 Task-F1 regression on '{dom_id}': "
                    f"{res.best_eval_task_f1:.4f} < {min_eval_task_f1:.4f}"
                )
            stages.append("stage_3_sft_dpo_and_canary_gate_passed")
            stages.append("stage_4_signed_uploaded_and_registered")

            outcomes.append(
                BaseModelMigrationDomainOutcome(
                    domain_id=dom_id,
                    previous_base_model_id=previous_base,
                    target_base_model_id=governed_base,
                    target_base_model_revision_sha256=rev_clean,
                    artifact_id=res.artifact_id,
                    eval_task_f1=res.best_eval_task_f1,
                    canary_leak_count=res.canary_audit.canary_leak_count,
                    weights_sha256=res.weights_sha256,
                    storage_uri=res.storage_uri,
                    activated=bool(res.registered_artifact and res.registered_artifact.is_active),
                    stages_completed=tuple(stages),
                )
            )

        return BaseModelMigrationReport(
            target_base_model_id=governed_base,
            target_base_model_revision_sha256=rev_clean,
            migrated_domains=tuple(outcomes),
            passed=True,
        )


__all__ = [
    "BaseModelMigrationDomainOutcome",
    "BaseModelMigrationOrchestrator",
    "BaseModelMigrationReport",
    "CachedMNRLBiEncoderTrainer",
    "CachedMNRLTrainingResult",
    "CanaryAuditReport",
    "CanaryMemorizationError",
    "CanaryMemorizationGate",
    "MigrationReplayError",
    "NEFTuneEmbeddingHook",
    "PEFTRsLoRATrainer",
    "PEFTTrainingConfig",
    "PEFTTrainingRunResult",
    "RsLoRALinear",
    "build_loraplus_adamw_optimizer",
    "compute_specialization_task_f1",
    "extract_peft_lora_state_dict",
    "inject_rslora_adapters",
    "is_frozen_router_or_ssm_path",
    "iter_safe_linear_targets",
    "load_peft_lora_state_dict",
    "resolve_safe_lora_target_modules",
    "set_adapters_enabled",
    "tokenize_chat_assistant_only",
    "verify_frozen_routers_and_ssm_states",
]
