"""LLMBackendSPI, ACL AdapterRouter (OWASP ASI06), Redis GPU Quotas, & Complexity Budgets."""

from __future__ import annotations

import contextvars
import json
import logging
import re
import struct
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol, runtime_checkable

import yaml

from context_scout.schemas.contracts import DomainPackManifest, PrincipalContext
from context_scout.security.permission_mirror import _StdlibRESP2Client

if TYPE_CHECKING:
    from context_scout.agents.domain_registry import AdapterPool
    from context_scout.agents.intent_router import QueryPlan

logger = logging.getLogger(__name__)

SINGLE_SYMBOL_TOKEN_CAP: Final[int] = 2_500
FEATURE_TOKEN_CAP: Final[int] = 8_000
CROSS_REPO_TOKEN_CAP: Final[int] = 18_000
MAX_MODEL_LEN_CLAMP_RATIO: Final[float] = 0.60

_CROSS_REPO_HINTS: Final[tuple[str, ...]] = (
    "cross-repo",
    "cross repo",
    "across repos",
    "across repositories",
    "across services",
    "federated",
    "end-to-end",
    "end to end",
    "architecture",
    "multi-service",
    "distributed",
)
_FEATURE_BROAD_HINTS: Final[tuple[str, ...]] = (
    "refactor",
    "migrate",
    "implement",
    "feature",
    "compare",
    "versus",
    " vs ",
    "integration",
    "pipeline",
    "workflow",
    " and ",
    "verify against",
)
_VALID_LORA_RANKS: Final[frozenset[int]] = frozenset({8, 16, 32})
_ACTIVE_GPU_LEASES: Final[contextvars.ContextVar[frozenset[tuple[int, str, str]]]] = (
    contextvars.ContextVar("_active_gpu_leases", default=frozenset())
)


class GPUPriorityClass(StrEnum):
    """GPU inference scheduling priority tiers (`P0` preempts `P2`)."""

    P0 = "P0"  # Interactive Graph 1 scout queries
    P1 = "P1"  # Graph 2 exhaustive MapReduce queries
    P2 = "P2"  # Background indexing / distillation / batch jobs


QuotaScope = Literal[
    "global",
    "tenant",
    "team",
    "user",
    "preempted_by_p0",
    "priority_shed",
]


class GPUOverloadError(RuntimeError):
    """Raised fail-closed when GPU fair-share quota or priority headroom is exhausted."""

    def __init__(
        self,
        message: str,
        *,
        scope: QuotaScope,
        priority: GPUPriorityClass,
        retry_after_seconds: float = 2.0,
    ) -> None:
        super().__init__(message)
        self.scope = scope
        self.priority = priority
        self.retry_after_seconds = retry_after_seconds


@dataclass
class GPUQuotaLease:
    """Active GPU lease held during an inference call."""

    manager: RedisGPUQuotaManager
    principal: PrincipalContext
    priority: GPUPriorityClass
    estimated_tokens: int
    window_bucket: int
    actual_tokens: int | None = None
    used_redis: bool = False

    def record_actual_tokens(self, tokens: int) -> None:
        """Records observed completion token usage for quota window reconciliation."""
        self.actual_tokens = max(0, int(tokens))


class RedisGPUQuotaManager:
    """Redis/Memorystore + in-memory `(tenant, team, user)` GPU fair-share quota manager."""

    def __init__(
        self,
        *,
        redis_url: str | None = None,
        redis_client: Any | None = None,
        max_global_inflight: int = 16,
        max_tenant_inflight: int = 8,
        max_team_inflight: int = 6,
        max_user_inflight: int = 4,
        max_tenant_tokens_per_min: int = 500_000,
        max_team_tokens_per_min: int = 250_000,
        max_user_tokens_per_min: int = 100_000,
        window_seconds: int = 60,
        time_fn: Callable[[], float] | None = None,
        key_prefix: str = "scout:gpu",
    ) -> None:
        self.max_global_inflight = max(1, int(max_global_inflight))
        self.max_tenant_inflight = max(1, int(max_tenant_inflight))
        self.max_team_inflight = max(1, int(max_team_inflight))
        self.max_user_inflight = max(1, int(max_user_inflight))
        self.max_tenant_tokens_per_min = max(1, int(max_tenant_tokens_per_min))
        self.max_team_tokens_per_min = max(1, int(max_team_tokens_per_min))
        self.max_user_tokens_per_min = max(1, int(max_user_tokens_per_min))
        self.window_seconds = max(1, int(window_seconds))
        self._time_fn: Callable[[], float] = time_fn or time.time
        self._key_prefix = key_prefix.strip(":") or "scout:gpu"
        self._lock = threading.RLock()
        self._memory_counters: dict[str, tuple[int, float]] = {}
        self._redis: Any | None = redis_client
        self._owns_redis = False

        if self._redis is None and redis_url and redis_url.strip():
            try:
                self._redis = _StdlibRESP2Client(redis_url.strip())
                self._owns_redis = True
            except Exception as exc:
                logger.warning(
                    "Unable to initialize RedisGPUQuotaManager RESP2 client (%s): %s; "
                    "using local fair-share counters.",
                    type(exc).__name__,
                    exc,
                )
                self._redis = None

    def close(self) -> None:
        if self._owns_redis and self._redis is not None and hasattr(self._redis, "close"):
            try:
                self._redis.close()
            except Exception as exc:
                logger.debug("Ignoring error closing Redis quota client (%s)", type(exc).__name__)

    def _get_memory_counter(self, key: str, now: float) -> int:
        entry = self._memory_counters.get(key)
        if entry is None:
            return 0
        val, expires_at = entry
        if now >= expires_at:
            self._memory_counters.pop(key, None)
            return 0
        return max(0, val)

    def _add_memory_counter(self, key: str, delta: int, ttl_seconds: int, now: float) -> int:
        cur_mem = self._get_memory_counter(key, now)
        nxt_mem = max(0, cur_mem + delta)
        if nxt_mem == 0:
            self._memory_counters.pop(key, None)
        else:
            self._memory_counters[key] = (nxt_mem, now + max(1, ttl_seconds))
        return nxt_mem

    def _get_counter(self, key: str, now: float) -> int:
        if self._redis is not None:
            try:
                raw = self._redis.get(key)
                return max(0, int(raw)) if raw is not None else 0
            except Exception as exc:
                logger.debug(
                    "Redis quota get fallback for %s (%s): %s",
                    key,
                    type(exc).__name__,
                    exc,
                )
        return self._get_memory_counter(key, now)

    def _add_counter(self, key: str, delta: int, ttl_seconds: int, now: float) -> tuple[int, bool]:
        if self._redis is not None:
            try:
                if hasattr(self._redis, "incrby"):
                    new_val = int(self._redis.incrby(key, delta))
                    if new_val <= 0 and delta < 0:
                        if hasattr(self._redis, "delete"):
                            self._redis.delete(key)
                        elif hasattr(self._redis, "setex"):
                            self._redis.setex(key, max(1, ttl_seconds), "0")
                        return 0, True
                    if new_val < 0:
                        if hasattr(self._redis, "setex"):
                            self._redis.setex(key, max(1, ttl_seconds), "0")
                        return 0, True
                    if hasattr(self._redis, "expire"):
                        self._redis.expire(key, max(1, ttl_seconds))
                    return max(0, new_val), True
                cur_raw = self._redis.get(key)
                cur = max(0, int(cur_raw)) if cur_raw is not None else 0
                nxt = max(0, cur + delta)
                if hasattr(self._redis, "setex"):
                    self._redis.setex(key, max(1, ttl_seconds), str(nxt))
                return nxt, True
            except Exception as exc:
                logger.debug(
                    "Redis quota incr fallback for %s (%s): %s",
                    key,
                    type(exc).__name__,
                    exc,
                )
        return self._add_memory_counter(key, delta, ttl_seconds, now), False

    def _keys_for(
        self,
        principal: PrincipalContext,
        priority: GPUPriorityClass,
        window_bucket: int,
    ) -> dict[str, str]:
        tid = str(principal.tenant_id)
        team = principal.resolved_team_id()
        uid = principal.user_id.strip()
        pfx = self._key_prefix
        return {
            "global": f"{pfx}:inflight:global",
            "priority": f"{pfx}:inflight:prio:{priority.value}",
            "p0": f"{pfx}:inflight:prio:{GPUPriorityClass.P0.value}",
            "tenant_inf": f"{pfx}:inflight:t:{tid}",
            "team_inf": f"{pfx}:inflight:t:{tid}:tm:{team}",
            "user_inf": f"{pfx}:inflight:t:{tid}:tm:{team}:u:{uid}",
            "tenant_tok": f"{pfx}:tok:{window_bucket}:t:{tid}",
            "team_tok": f"{pfx}:tok:{window_bucket}:t:{tid}:tm:{team}",
            "user_tok": f"{pfx}:tok:{window_bucket}:t:{tid}:tm:{team}:u:{uid}",
        }

    def acquire(
        self,
        principal: PrincipalContext,
        *,
        priority: GPUPriorityClass = GPUPriorityClass.P0,
        estimated_tokens: int = 1024,
    ) -> GPUQuotaLease:
        est = max(1, int(estimated_tokens))
        with self._lock:
            now = self._time_fn()
            window_bucket = int(now // self.window_seconds)
            keys = self._keys_for(principal, priority, window_bucket)

            cur_global = self._get_counter(keys["global"], now)
            cur_p0 = self._get_counter(keys["p0"], now)

            if priority == GPUPriorityClass.P2:
                if cur_p0 > 0:
                    raise GPUOverloadError(
                        "P2 background GPU request preempted by active P0 interactive traffic.",
                        scope="preempted_by_p0",
                        priority=priority,
                        retry_after_seconds=2.0,
                    )
                p2_cap = max(1, int(self.max_global_inflight * 0.50))
                if cur_global >= p2_cap:
                    raise GPUOverloadError(
                        f"P2 GPU request shed at {cur_global}/{p2_cap} global concurrency cap.",
                        scope="priority_shed",
                        priority=priority,
                        retry_after_seconds=3.0,
                    )
            elif priority == GPUPriorityClass.P1:
                p1_cap = max(1, int(self.max_global_inflight * 0.80))
                if cur_global >= p1_cap:
                    raise GPUOverloadError(
                        f"P1 MapReduce GPU request shed at {cur_global}/{p1_cap} concurrency cap.",
                        scope="priority_shed",
                        priority=priority,
                        retry_after_seconds=2.0,
                    )

            if cur_global >= self.max_global_inflight:
                raise GPUOverloadError(
                    "Global GPU concurrency limit reached "
                    f"({cur_global}/{self.max_global_inflight}).",
                    scope="global",
                    priority=priority,
                    retry_after_seconds=2.0,
                )

            cur_t_inf = self._get_counter(keys["tenant_inf"], now)
            cur_t_tok = self._get_counter(keys["tenant_tok"], now)
            if cur_t_inf >= self.max_tenant_inflight or (
                cur_t_tok + est > self.max_tenant_tokens_per_min
            ):
                raise GPUOverloadError(
                    f"Tenant '{principal.tenant_id}' GPU fair-share quota exhausted.",
                    scope="tenant",
                    priority=priority,
                    retry_after_seconds=5.0,
                )

            cur_tm_inf = self._get_counter(keys["team_inf"], now)
            cur_tm_tok = self._get_counter(keys["team_tok"], now)
            if cur_tm_inf >= self.max_team_inflight or (
                cur_tm_tok + est > self.max_team_tokens_per_min
            ):
                raise GPUOverloadError(
                    f"Team '{principal.resolved_team_id()}' GPU fair-share quota exhausted.",
                    scope="team",
                    priority=priority,
                    retry_after_seconds=5.0,
                )

            cur_u_inf = self._get_counter(keys["user_inf"], now)
            cur_u_tok = self._get_counter(keys["user_tok"], now)
            if cur_u_inf >= self.max_user_inflight or (
                cur_u_tok + est > self.max_user_tokens_per_min
            ):
                raise GPUOverloadError(
                    f"User '{principal.user_id}' GPU fair-share quota exhausted.",
                    scope="user",
                    priority=priority,
                    retry_after_seconds=5.0,
                )

            inf_ttl = max(30, self.window_seconds * 2)
            tok_ttl = max(60, self.window_seconds * 2)
            used_redis_any = False
            for k in ("global", "priority", "tenant_inf", "team_inf", "user_inf"):
                _, hit_redis = self._add_counter(keys[k], 1, inf_ttl, now)
                used_redis_any = used_redis_any or hit_redis
            for k in ("tenant_tok", "team_tok", "user_tok"):
                _, hit_redis = self._add_counter(keys[k], est, tok_ttl, now)
                used_redis_any = used_redis_any or hit_redis

            return GPUQuotaLease(
                manager=self,
                principal=principal,
                priority=priority,
                estimated_tokens=est,
                window_bucket=window_bucket,
                used_redis=used_redis_any,
            )

    def release(self, lease: GPUQuotaLease) -> None:
        with self._lock:
            now = self._time_fn()
            keys = self._keys_for(lease.principal, lease.priority, lease.window_bucket)
            inf_ttl = max(30, self.window_seconds * 2)
            tok_ttl = max(60, self.window_seconds * 2)
            for k in ("global", "priority", "tenant_inf", "team_inf", "user_inf"):
                if lease.used_redis:
                    self._add_counter(keys[k], -1, inf_ttl, now)
                else:
                    self._add_memory_counter(keys[k], -1, inf_ttl, now)
            if lease.actual_tokens is not None:
                delta = lease.actual_tokens - lease.estimated_tokens
                if delta != 0:
                    for k in ("tenant_tok", "team_tok", "user_tok"):
                        if lease.used_redis:
                            self._add_counter(keys[k], delta, tok_ttl, now)
                        else:
                            self._add_memory_counter(keys[k], delta, tok_ttl, now)

    @contextmanager
    def lease(
        self,
        principal: PrincipalContext,
        *,
        priority: GPUPriorityClass = GPUPriorityClass.P0,
        estimated_tokens: int = 1024,
    ) -> Iterator[GPUQuotaLease]:
        lease_key = (id(self), str(principal.tenant_id), principal.user_id.strip())
        current = _ACTIVE_GPU_LEASES.get()
        if lease_key in current:
            yield GPUQuotaLease(
                manager=self,
                principal=principal,
                priority=priority,
                estimated_tokens=0,
                window_bucket=int(self._time_fn() // self.window_seconds),
                actual_tokens=0,
            )
            return

        active = self.acquire(
            principal,
            priority=priority,
            estimated_tokens=estimated_tokens,
        )
        token = _ACTIVE_GPU_LEASES.set(current | {lease_key})
        try:
            yield active
        finally:
            _ACTIVE_GPU_LEASES.reset(token)
            self.release(active)

    def snapshot(
        self,
        principal: PrincipalContext,
        *,
        priority: GPUPriorityClass = GPUPriorityClass.P0,
    ) -> dict[str, int]:
        with self._lock:
            now = self._time_fn()
            window_bucket = int(now // self.window_seconds)
            keys = self._keys_for(principal, priority, window_bucket)
            return {name: self._get_counter(key, now) for name, key in keys.items()}


@dataclass(frozen=True)
class AdapterRouteDecision:
    """Result of AdapterRouter evaluation against training_scope_ceiling (OWASP ASI06)."""

    domain_id: str
    base_model_id: str
    routed_to_lora: bool
    lora_adapter_id: str | None = None
    lora_adapter_path: str | None = None
    lora_request: str | None = None
    training_scope_ceiling: tuple[str, ...] = ()
    reason: str = "routed_to_base"


AdapterWeightFormat = Literal["mlx_lm", "hf_peft", "unknown", "missing", "corrupt"]


def detect_adapter_weight_format(
    adapter_path: Path | str | None,
    *,
    base_dir: Path | None = None,
) -> AdapterWeightFormat:
    """Inspects Safetensors v0 header keys to distinguish mlx-lm vs HF PEFT adapters.

    Reads only the lightweight JSON header prefix (never loading tensor buffers):
    - ``mlx_lm``: lowercase ``.lora_a`` / ``.lora_b`` keys (``language_model.model...lora_a``)
    - ``hf_peft``: keys use ``.lora_A.`` / ``.lora_B.`` or ``base_model.model.``
    - ``missing``: no ``adapters.safetensors`` / ``adapter_model.safetensors`` exists
    - ``corrupt``: file exists but is empty or has a malformed Safetensors v0 header
    """
    if not adapter_path:
        return "missing"
    cand = Path(adapter_path)
    if not cand.is_absolute() and base_dir is not None:
        cand = (base_dir / cand).resolve()
    if cand.is_file():
        st_candidates = [cand]
    elif cand.is_dir():
        st_candidates = [
            cand / name
            for name in ("adapters.safetensors", "adapter_model.safetensors")
            if (cand / name).is_file()
        ]
    else:
        return "missing"

    if not st_candidates:
        return "missing"

    st_file = st_candidates[0]
    try:
        file_size = st_file.stat().st_size
        if file_size <= 8:
            return "corrupt"
        with st_file.open("rb") as fh:
            prefix = fh.read(8)
            if len(prefix) < 8:
                return "corrupt"
            header_len = int(struct.unpack("<Q", prefix)[0])
            if header_len <= 0 or header_len > min(file_size - 8, 16_777_216):
                return "corrupt"
            header_obj = json.loads(fh.read(header_len).decode("utf-8"))
        if not isinstance(header_obj, Mapping):
            return "corrupt"
        tensor_keys = [str(k) for k in header_obj if k != "__metadata__"]
        if not tensor_keys:
            return "corrupt"
        has_mlx = any(k.endswith(".lora_a") or k.endswith(".lora_b") for k in tensor_keys)
        has_peft = any(
            ".lora_A." in k or ".lora_B." in k or k.startswith("base_model.model.")
            for k in tensor_keys
        )
        if has_mlx and not has_peft:
            return "mlx_lm"
        if has_peft:
            return "hf_peft"
        return "unknown"
    except Exception as exc:
        logger.warning(
            "Failed reading safetensors header at %s (%s): %s",
            st_file,
            type(exc).__name__,
            exc,
        )
        return "corrupt"


class AdapterRouter:
    """Enforces `set(training_scope_ceiling).issubset(set(principal.allowed_scopes))` (ASI06)."""

    def __init__(
        self,
        *,
        domains_dir: Path | None = None,
        adapter_pool: AdapterPool | None = None,
        default_base_model_id: str = "google/gemma-4-26B-A4B-it",
        spec_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self._domains_dir = domains_dir or (Path(__file__).resolve().parents[3] / "domains")
        self._adapter_pool = adapter_pool
        self._default_base_model_id = default_base_model_id
        self._spec_overrides: dict[str, dict[str, Any]] = {}
        self._tenant_spec_overrides: dict[tuple[str, str], dict[str, Any]] = {}
        for k, v in (spec_overrides or {}).items():
            raw_tid = str(v.get("tenant_id") or "").strip()
            if raw_tid:
                self._tenant_spec_overrides[(raw_tid, str(k))] = dict(v)
            else:
                self._spec_overrides[str(k)] = dict(v)
        self._lock = threading.RLock()

    def register_specialization(
        self,
        domain_id: str,
        spec: Mapping[str, Any],
        *,
        tenant_id: Any | None = None,
    ) -> None:
        resolved_tid = str(tenant_id or spec.get("tenant_id") or "").strip()
        with self._lock:
            if resolved_tid:
                entry = dict(spec)
                entry["tenant_id"] = resolved_tid
                self._tenant_spec_overrides[(resolved_tid, domain_id)] = entry
            else:
                self._spec_overrides[domain_id] = dict(spec)

    def unregister_specialization(
        self,
        domain_id: str,
        *,
        tenant_id: Any | None = None,
    ) -> bool:
        resolved_tid = str(tenant_id or "").strip()
        with self._lock:
            if resolved_tid:
                removed = self._tenant_spec_overrides.pop((resolved_tid, domain_id), None)
                return removed is not None
            removed = self._spec_overrides.pop(domain_id, None)
            return removed is not None

    def _load_specialization(
        self,
        domain_id: str,
        principal: PrincipalContext | None = None,
    ) -> dict[str, Any] | None:
        with self._lock:
            if principal is not None:
                tenant_key = (str(principal.tenant_id), domain_id)
                if tenant_key in self._tenant_spec_overrides:
                    return dict(self._tenant_spec_overrides[tenant_key])
            if domain_id in self._spec_overrides:
                return dict(self._spec_overrides[domain_id])
        spec_path = self._domains_dir / domain_id / "specialization.yaml"
        if not spec_path.is_file():
            return None
        try:
            raw = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
            return dict(raw) if isinstance(raw, Mapping) else None
        except Exception as exc:
            logger.warning(
                "Failed loading specialization.yaml for '%s' (%s): %s",
                domain_id,
                type(exc).__name__,
                exc,
            )
            return None

    def _verify_adapter_weights(
        self,
        domain_id: str,
        adapter_path: str | None,
        principal: PrincipalContext | None,
    ) -> bool:
        if self._adapter_pool is not None:
            try:
                res = self._adapter_pool.verify_adapter(
                    domain_id,
                    tenant_id=principal.tenant_id if principal is not None else None,  # type: ignore[arg-type]
                )
                if bool(res.get("is_valid", False)):
                    return True
            except TypeError:
                try:
                    res = self._adapter_pool.verify_adapter(domain_id)
                    if bool(res.get("is_valid", False)):
                        return True
                except Exception as exc:
                    logger.debug(
                        "AdapterPool.verify_adapter failed for '%s' (%s): %s",
                        domain_id,
                        type(exc).__name__,
                        exc,
                    )
            except Exception as exc:
                logger.debug(
                    "AdapterPool.verify_adapter failed for '%s' (%s): %s",
                    domain_id,
                    type(exc).__name__,
                    exc,
                )
        if not adapter_path:
            return False
        cand = Path(adapter_path)
        if not cand.is_absolute():
            cand = self._domains_dir.parent / adapter_path
        return cand.exists() and (
            (cand / "adapters.safetensors").is_file()
            or (cand / "adapter_model.safetensors").is_file()
            or (cand / "adapter_config.json").is_file()
        )

    def route(
        self,
        domain_id: str | None,
        principal: PrincipalContext | None,
        *,
        manifest: DomainPackManifest | None = None,
        verify_weights: bool = True,
        runtime_backend: Literal["any", "mlx_lm", "vllm"] = "any",
    ) -> AdapterRouteDecision:
        """Routes exclusively to frozen base model unless ALL ACL & LoRA checks pass."""
        base_model = manifest.base_model_id if manifest is not None else self._default_base_model_id
        if not domain_id:
            return AdapterRouteDecision(
                domain_id="none",
                base_model_id=base_model,
                routed_to_lora=False,
                reason="no_domain_id",
            )

        if manifest is not None and not manifest.lora_enabled:
            return AdapterRouteDecision(
                domain_id=domain_id,
                base_model_id=base_model,
                routed_to_lora=False,
                reason="lora_disabled_in_manifest",
            )

        spec = self._load_specialization(domain_id, principal)
        if spec is None:
            return AdapterRouteDecision(
                domain_id=domain_id,
                base_model_id=base_model,
                routed_to_lora=False,
                reason="missing_specialization_manifest",
            )

        spec_base = str(spec.get("base_model_id") or base_model).strip() or base_model
        spec_tenant = str(spec.get("tenant_id") or "").strip()
        if spec_tenant and (principal is None or str(principal.tenant_id) != spec_tenant):
            return AdapterRouteDecision(
                domain_id=domain_id,
                base_model_id=spec_base,
                routed_to_lora=False,
                reason="cross_tenant_adapter_blocked",
            )

        lora_policy = spec.get("lora_policy") or {}
        if not isinstance(lora_policy, Mapping) or not bool(lora_policy.get("enabled", False)):
            return AdapterRouteDecision(
                domain_id=domain_id,
                base_model_id=spec_base,
                routed_to_lora=False,
                reason="lora_disabled_in_policy",
            )

        if lora_policy.get("use_dora") is True:
            return AdapterRouteDecision(
                domain_id=domain_id,
                base_model_id=spec_base,
                routed_to_lora=False,
                reason="invalid_lora_policy_use_dora",
            )

        if lora_policy.get("use_rslora") is not True:
            return AdapterRouteDecision(
                domain_id=domain_id,
                base_model_id=spec_base,
                routed_to_lora=False,
                reason="invalid_lora_policy_requires_rslora",
            )

        try:
            rank_val = int(lora_policy.get("rank", 0))
        except (TypeError, ValueError):
            rank_val = 0
        if rank_val not in _VALID_LORA_RANKS:
            return AdapterRouteDecision(
                domain_id=domain_id,
                base_model_id=spec_base,
                routed_to_lora=False,
                reason=f"invalid_lora_rank_{rank_val}",
            )

        raw_ceiling = spec.get("training_scope_ceiling") or []
        ceiling = tuple(str(s).strip() for s in raw_ceiling if str(s).strip())
        if not ceiling:
            return AdapterRouteDecision(
                domain_id=domain_id,
                base_model_id=spec_base,
                routed_to_lora=False,
                reason="missing_training_scope_ceiling",
            )

        if principal is None:
            return AdapterRouteDecision(
                domain_id=domain_id,
                base_model_id=spec_base,
                routed_to_lora=False,
                training_scope_ceiling=ceiling,
                reason="missing_principal",
            )

        principal_scopes = set(principal.allowed_scopes)
        if not set(ceiling).issubset(principal_scopes):
            return AdapterRouteDecision(
                domain_id=domain_id,
                base_model_id=spec_base,
                routed_to_lora=False,
                training_scope_ceiling=ceiling,
                reason="scope_ceiling_exceeded",
            )

        adapter_id = (
            manifest.lora_adapter_id
            if manifest is not None
            else str(spec.get("lora_adapter_id") or f"lora_{domain_id}")
        )
        adapter_path = (
            manifest.lora_adapter_path
            if manifest is not None
            else str(spec.get("lora_adapter_path") or f"domains/{domain_id}/adapters/{adapter_id}")
        )
        if self._adapter_pool is not None:
            try:
                slot = self._adapter_pool.get_slot(domain_id)
                adapter_id = slot.adapter_id or adapter_id
                adapter_path = slot.adapter_path or adapter_path
            except KeyError as exc:
                logger.debug(
                    "AdapterPool slot lookup miss for '%s' (%s)",
                    domain_id,
                    type(exc).__name__,
                )

        if verify_weights and not self._verify_adapter_weights(domain_id, adapter_path, principal):
            return AdapterRouteDecision(
                domain_id=domain_id,
                base_model_id=spec_base,
                routed_to_lora=False,
                training_scope_ceiling=ceiling,
                reason="no_trained_adapter",
            )

        if verify_weights and runtime_backend in {"mlx_lm", "vllm"}:
            weight_fmt = detect_adapter_weight_format(
                adapter_path, base_dir=self._domains_dir.parent
            )
            if weight_fmt == "missing":
                return AdapterRouteDecision(
                    domain_id=domain_id,
                    base_model_id=spec_base,
                    routed_to_lora=False,
                    training_scope_ceiling=ceiling,
                    reason="no_trained_adapter",
                )
            if weight_fmt == "corrupt":
                return AdapterRouteDecision(
                    domain_id=domain_id,
                    base_model_id=spec_base,
                    routed_to_lora=False,
                    training_scope_ceiling=ceiling,
                    reason="corrupt_adapter_weights",
                )
            if runtime_backend == "mlx_lm" and weight_fmt != "mlx_lm":
                return AdapterRouteDecision(
                    domain_id=domain_id,
                    base_model_id=spec_base,
                    routed_to_lora=False,
                    training_scope_ceiling=ceiling,
                    reason=f"incompatible_adapter_format_{weight_fmt}",
                )
            if runtime_backend == "vllm" and weight_fmt == "mlx_lm":
                return AdapterRouteDecision(
                    domain_id=domain_id,
                    base_model_id=spec_base,
                    routed_to_lora=False,
                    training_scope_ceiling=ceiling,
                    reason="incompatible_adapter_format_mlx_lm",
                )

        return AdapterRouteDecision(
            domain_id=domain_id,
            base_model_id=spec_base,
            routed_to_lora=True,
            lora_adapter_id=adapter_id,
            lora_adapter_path=adapter_path,
            lora_request=adapter_id,
            training_scope_ceiling=ceiling,
            reason="authorized_lora_subset",
        )


ComplexityTier = Literal["single_symbol", "feature", "cross_repo"]


def compute_adaptive_complexity_budget(
    user_prompt: str,
    query_plan: QueryPlan | None = None,
    *,
    max_model_len: int = 32_768,
    is_cross_repo: bool = False,
) -> tuple[ComplexityTier, int]:
    """Selects complexity budget tier (<=2.5K / <=8K / <=18K) clamped to 0.60 * max_model_len."""
    prompt_low = user_prompt.lower()
    facet_count = len(query_plan.facets) if query_plan is not None else 1
    cardinality = (query_plan.requested_cardinality or 1) if query_plan is not None else 1
    qtype = query_plan.query_type if query_plan is not None else "general"

    if (
        is_cross_repo
        or qtype in {"multi_hop_research_to_code", "architecture_comparison"}
        or facet_count >= 3
        or cardinality >= 3
        or any(hint in prompt_low for hint in _CROSS_REPO_HINTS)
    ):
        tier: ComplexityTier = "cross_repo"
        tier_cap = CROSS_REPO_TOKEN_CAP
    else:
        ident_hits = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z0-9_]+\b", user_prompt)
        camel_or_snake = re.findall(
            r"\b(?:[A-Z][a-z0-9]+[A-Z][A-Za-z0-9]*|[a-z0-9]+_[a-z0-9_]+)\b",
            user_prompt,
        )
        word_count = len(re.findall(r"\S+", user_prompt))
        is_broad_feature = (
            facet_count >= 2
            or cardinality >= 2
            or any(hint in prompt_low for hint in _FEATURE_BROAD_HINTS)
            or word_count > 18
        )
        if not is_broad_feature and (ident_hits or camel_or_snake or word_count <= 14):
            tier = "single_symbol"
            tier_cap = SINGLE_SYMBOL_TOKEN_CAP
        else:
            tier = "feature"
            tier_cap = FEATURE_TOKEN_CAP

    clamped_max_model = max(256, int(MAX_MODEL_LEN_CLAMP_RATIO * max(1, int(max_model_len))))
    effective_budget = min(tier_cap, clamped_max_model)
    return tier, effective_budget


@runtime_checkable
class LLMBackendSPI(Protocol):
    """Protocol implemented by local MLX, vLLM/SGLang, and test LLM inference backends."""

    @property
    def model_id(self) -> str: ...

    @property
    def max_model_len(self) -> int: ...

    def is_online(self) -> bool: ...

    def complete(
        self,
        prompt: str,
        domain_id: str | None = None,
        system_instruction: str | None = None,
        max_tokens: int = 1024,
        *,
        principal: PrincipalContext | None = None,
        priority: GPUPriorityClass = GPUPriorityClass.P0,
    ) -> str: ...


__all__ = [
    "CROSS_REPO_TOKEN_CAP",
    "FEATURE_TOKEN_CAP",
    "MAX_MODEL_LEN_CLAMP_RATIO",
    "SINGLE_SYMBOL_TOKEN_CAP",
    "AdapterRouteDecision",
    "AdapterRouter",
    "AdapterWeightFormat",
    "ComplexityTier",
    "GPUOverloadError",
    "GPUPriorityClass",
    "GPUQuotaLease",
    "LLMBackendSPI",
    "QuotaScope",
    "RedisGPUQuotaManager",
    "compute_adaptive_complexity_budget",
    "detect_adapter_weight_format",
]
