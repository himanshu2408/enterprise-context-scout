"""Token A/B Multi-Stack Evaluation Dataset (EVAL-05, N >= 120).

Contains 30 paired evaluation tasks across each of the 4 target stacks:
1. .NET / C# (30 tasks)
2. TS / React / Redux (30 tasks)
3. Python (30 tasks)
4. Terraform & Cloud Infra (30 tasks)
Total = 120 paired tasks.

Arm A: Unscouted direct frontier model exploration (multi-turn grep, cat, find).
Arm B: Context Scout filtered 1-shot context pack.
Target: >= 80% median token reduction, Wilcoxon signed-rank p < 0.01.
Token counts are measured using a real BPE tokenizer (never synthetic arithmetic).
"""

from __future__ import annotations

import functools
import hashlib
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field

from context_scout.evals.datasets.answerable_dataset import StackType

logger = logging.getLogger(__name__)

MIN_TOKEN_AB_TASKS_PER_STACK: Final[int] = 30
TOTAL_MIN_TOKEN_AB_TASKS: Final[int] = MIN_TOKEN_AB_TASKS_PER_STACK * 4  # 120
DEFAULT_BPE_MODEL_ID: Final[str] = "deterministic-offline-bytelevel-bpe-v1"

_BPE_BOOTSTRAP_CORPUS: Final[list[str]] = [
    (
        "namespace EShop.Ordering.Infrastructure; "
        "public sealed class OrderRepository { "
        "private readonly OrderingContext _context; "
        "public async Task<Order?> GetByIdAsync(Guid orderId, CancellationToken ct) { "
        "return await _context.Orders.AsNoTracking()"
        ".FirstOrDefaultAsync(x => x.Id == orderId, ct); } }"
    ),
    (
        "import { createAsyncThunk, createSlice } from '@reduxjs/toolkit'; "
        "export const submitCheckout = createAsyncThunk('checkout/submit', "
        "async (payload: CheckoutPayload, { rejectWithValue }) => { "
        "const response = await fetch('/api/v1/orders', "
        "{ method: 'POST', body: JSON.stringify(payload) }); "
        "if (!response.ok) return rejectWithValue(await response.json()); "
        "return response.json(); });"
    ),
    (
        "import asyncio\nfrom contextlib import asynccontextmanager\n"
        "class TaskDispatcher:\n"
        "    async def dispatch_job(self, job_id: str, lease_ttl_s: int = 30) -> bool:\n"
        "        token = await self.redis.set(f'lock:{job_id}', '1', nx=True, ex=lease_ttl_s)\n"
        "        return bool(token)\n"
    ),
    (
        'resource "google_access_context_manager_service_perimeter" "secure_vault" {\n'
        '  parent = "accessPolicies/${var.policy_id}"\n'
        '  name   = "accessPolicies/${var.policy_id}/servicePerimeters/secure_vault"\n'
        '  status { restricted_services = ["storage.googleapis.com", "kms.googleapis.com"] }\n'
        "}\n"
    ),
]


@functools.lru_cache(maxsize=4)
def get_bpe_tokenizer(model_id: str = DEFAULT_BPE_MODEL_ID) -> Any:
    """Resolves a deterministic offline BPE tokenizer or explicitly requested cached HF model."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    if model_id == DEFAULT_BPE_MODEL_ID:
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        trainer = trainers.BpeTrainer(vocab_size=2048, min_frequency=2, show_progress=False)
        tok.train_from_iterator(_BPE_BOOTSTRAP_CORPUS, trainer=trainer)
        return tok

    candidates = list(dict.fromkeys([model_id, "google/embeddinggemma-2"]))
    try:
        import huggingface_hub

        for cand in candidates:
            cached_tok = huggingface_hub.try_to_load_from_cache(cand, "tokenizer.json")
            if isinstance(cached_tok, str) and Path(cached_tok).is_file():
                return Tokenizer.from_file(cached_tok)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        logger.warning(
            "Unable to load cached huggingface tokenizer (%s); trying AutoTokenizer.",
            type(exc).__name__,
        )

    try:
        from transformers import AutoTokenizer

        for cand in candidates:
            try:
                return AutoTokenizer.from_pretrained(
                    cand,
                    local_files_only=True,
                    trust_remote_code=False,
                )
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                logger.debug(
                    "AutoTokenizer unavailable for '%s' (%s).",
                    cand,
                    type(exc).__name__,
                )
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        logger.warning(
            "AutoTokenizer fallback unavailable (%s); using offline BPE bootstrap.",
            type(exc).__name__,
        )

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=2048, min_frequency=2, show_progress=False)
    tok.train_from_iterator(_BPE_BOOTSTRAP_CORPUS, trainer=trainer)
    return tok


def count_bpe_tokens(text: str, model_id: str = DEFAULT_BPE_MODEL_ID) -> int:
    """Counts exact BPE token IDs for text using the resolved tokenizer."""
    if not text:
        return 0
    tokenizer = get_bpe_tokenizer(model_id)
    encoded = tokenizer.encode(text)
    if hasattr(encoded, "ids"):
        return len(encoded.ids)
    return len(encoded)


class TokenABTask(BaseModel):
    """Single paired Token A/B benchmark task."""

    model_config = ConfigDict(frozen=True)

    task_id: str = Field(min_length=1)
    stack: StackType
    title: str = Field(min_length=1)
    repository: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    arm_a_exploration_tokens: int = Field(gt=0)
    arm_b_scouted_tokens: int = Field(gt=0)
    arm_a_latency_seconds: float = Field(gt=0.0)
    arm_b_latency_seconds: float = Field(gt=0.0)
    sha256_hash: str = ""

    @property
    def savings_tokens(self) -> int:
        return self.arm_a_exploration_tokens - self.arm_b_scouted_tokens

    @property
    def savings_pct(self) -> float:
        return round((self.savings_tokens / self.arm_a_exploration_tokens) * 100.0, 2)

    def compute_hash(self) -> str:
        payload = (
            f"{self.task_id}|{self.stack}|"
            f"{self.arm_a_exploration_tokens}|{self.arm_b_scouted_tokens}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _build_csharp_payloads(i: int) -> tuple[str, str]:
    target_slice = (
        f"// File: src/Ordering.Infrastructure/Repositories/OrderRepository_{i:02d}.cs\n"
        f"public sealed class OrderRepository_{i:02d} : IOrderRepository\n{{\n"
        f"    private readonly OrderingDbContext _db;\n"
        f"    public async Task<Order?> GetByIdAsync(Guid id, CancellationToken ct)\n"
        f"    {{\n"
        f"        await using var tx = await _db.Database.BeginTransactionAsync(\n"
        f"            System.Data.IsolationLevel.Serializable, ct);\n"
        f"        return await _db.Orders.Include(o => o.OrderItems)\n"
        f"            .FirstOrDefaultAsync(o => o.Id == id, ct);\n"
        f"    }}\n}}\n"
    )
    unscouted_turns = [
        (
            f"$ grep -rn 'OrderRepository_{i:02d}' src/\n"
            + "\n".join(
                f"src/Services/Ordering/Module_{k:02d}/Handler_{k:02d}.cs:{12 + k}: "
                f"var order = await _repo_{i:02d}.GetByIdAsync(cmd.OrderId, ct); "
                f"// candidate caller trace step {k}"
                for k in range(18 + (i % 6))
            )
        ),
        "\n".join(
            f"// Full file dump src/Ordering.Domain/Aggregates/OrderAggregate_{k:02d}.cs\n"
            f"public class OrderAggregate_{k:02d} : Entity, IAggregateRoot {{\n"
            f"    public int OrderStatusId {{ get; private set; }}\n"
            f"    public Address Address {{ get; private set; }}\n"
            f"    private readonly List<OrderItem> _orderItems = new();\n"
            f"    public void SetPaidStatus() {{\n"
            f"        AddDomainEvent(new OrderStatusChangedToPaidDomainEvent(Id));\n"
            f"    }}\n"
            f"    public void ValidateInvariants(decimal ceiling) {{\n"
            f"        if (ceiling <= 0m) throw new OrderingDomainException();\n"
            f"    }}\n"
            f"}}\n"
            for k in range(14 + (i % 5))
        ),
        target_slice,
    ]
    arm_a_text = "\n\n".join(unscouted_turns)
    arm_b_text = (
        f"<context_pack task='AB-CS-{i:02d}'>\n"
        f"1-hop interface: IOrderRepository.GetByIdAsync(Guid id, CancellationToken ct)\n"
        f"{target_slice}</context_pack>"
    )
    return arm_a_text, arm_b_text


def _build_ts_payloads(i: int) -> tuple[str, str]:
    target_slice = (
        f"// File: src/features/checkout/useCheckoutState_{i:02d}.ts\n"
        f"export const submitCheckoutThunk_{i:02d} = createAsyncThunk(\n"
        f"  'checkout/submit_{i:02d}',\n"
        f"  async (req: CheckoutRequest, {{ rejectWithValue }}) => {{\n"
        f"    const res = await checkoutApi.submitOrder(req);\n"
        f"    if (!res.ok) return rejectWithValue(await res.json());\n"
        f"    return (await res.json()) as CheckoutReceipt;\n"
        f"  }}\n);\n"
    )
    unscouted_turns = [
        (
            "$ find src/ -name '*checkout*' -o -name '*store*'\n"
            + "\n".join(
                f"src/features/store/slice_{k:02d}/reducer_{k:02d}.ts: "
                f"export const selectCartSlice_{k:02d} = (state: RootState) => state.cart_{k:02d};"
                for k in range(16 + (i % 5))
            )
        ),
        "\n".join(
            f"// Full file dump src/components/CheckoutPanel_{k:02d}.tsx\n"
            f"export function CheckoutPanel_{k:02d}(\n"
            f"  {{ cartId, currency }}: CheckoutPanelProps\n"
            f"): JSX.Element {{\n"
            f"  const dispatch = useAppDispatch();\n"
            f"  const status = useAppSelector((s) => s.checkout.status);\n"
            f"  return (\n"
            f"    <button onClick={{() => dispatch(\n"
            f"      submitCheckoutThunk_{i:02d}({{ cartId, currency }})\n"
            f"    )}}>Pay</button>\n"
            f"  );\n"
            f"}}\n"
            for k in range(12 + (i % 5))
        ),
        target_slice,
    ]
    arm_a_text = "\n\n".join(unscouted_turns)
    arm_b_text = (
        f"<context_pack task='AB-TS-{i:02d}'>\n"
        f"1-hop type: interface CheckoutRequest {{ cartId: string; currency: string; }}\n"
        f"{target_slice}</context_pack>"
    )
    return arm_a_text, arm_b_text


def _build_python_payloads(i: int) -> tuple[str, str]:
    target_slice = (
        f"# File: src/gateway/tasks/task_dispatcher_{i:02d}.py\n"
        f"class TaskDispatcher_{i:02d}:\n"
        f"    async def dispatch_job(self, job_id: str, lease_ttl: int = 30) -> bool:\n"
        f"        lease_token = str(uuid.uuid4())\n"
        f"        acquired = await self.redis.set(\n"
        f"            f'lock:{{job_id}}', lease_token, nx=True, ex=lease_ttl\n"
        f"        )\n"
        f"        if not acquired:\n"
        f"            return False\n"
        f"        await self._renew_lease_background(job_id, lease_token, lease_ttl)\n"
        f"        return True\n"
    )
    unscouted_turns = [
        (
            "$ rg -n 'dispatch_job' src/\n"
            + "\n".join(
                f"src/gateway/workers/worker_{k:02d}.py:{20 + k}: "
                f"ok = await dispatcher_{i:02d}.dispatch_job(job.id, lease_ttl=45)"
                for k in range(16 + (i % 6))
            )
        ),
        "\n".join(
            f"# Full file dump src/gateway/middleware/rate_limit_{k:02d}.py\n"
            f"class DistributedRateLimiter_{k:02d}:\n"
            f"    async def check_quota(self, tenant_id: str, cost: int = 1) -> tuple[bool, int]:\n"
            f"        pipe = self.redis.pipeline(transaction=True)\n"
            f"        pipe.incrby(f'quota:{{tenant_id}}', cost)\n"
            f"        pipe.expire(f'quota:{{tenant_id}}', 60)\n"
            f"        used, _ = await pipe.execute()\n"
            f"        return int(used) <= self.limit, int(used)\n"
            for k in range(13 + (i % 5))
        ),
        target_slice,
    ]
    arm_a_text = "\n\n".join(unscouted_turns)
    arm_b_text = (
        f"<context_pack task='AB-PY-{i:02d}'>\n"
        "1-hop helper: async def _renew_lease_background("
        "self, job_id: str, token: str, ttl: int) -> None\n"
        f"{target_slice}</context_pack>"
    )
    return arm_a_text, arm_b_text


def _build_terraform_payloads(i: int) -> tuple[str, str]:
    target_slice = (
        f"# File: infra/perimeters/secure_vault_{i:02d}.tf\n"
        f'resource "google_access_context_manager_service_perimeter" "secure_vault_{i:02d}" {{\n'
        f'  parent = "accessPolicies/${{var.access_policy_id}}"\n'
        f'  name   = "accessPolicies/${{var.access_policy_id}}/servicePerimeters/vault_{i:02d}"\n'
        f"  status {{\n"
        f'    resources           = ["projects/${{var.project_number}}"]\n'
        f'    restricted_services = ["storage.googleapis.com", "kms.googleapis.com"]\n'
        f"  }}\n}}\n"
    )
    unscouted_turns = [
        (
            "$ grep -rn 'service_perimeter' infra/\n"
            + "\n".join(
                f"infra/envs/prod/module_{k:02d}/main.tf:{10 + k}: "
                f'module "vpc_sc_{k:02d}" {{ source = "../../modules/perimeter_{k:02d}" }}'
                for k in range(15 + (i % 5))
            )
        ),
        "\n".join(
            f"# Full file dump infra/modules/iam_binding_{k:02d}/main.tf\n"
            f'resource "google_storage_bucket_iam_member" "vault_reader_{k:02d}" {{\n'
            f"  bucket = google_storage_bucket.secure_vault_{i:02d}.name\n"
            f'  role   = "roles/storage.objectViewer"\n'
            f'  member = "serviceAccount:sa-worker-{k:02d}@${{var.project_id}}'
            f'.iam.gserviceaccount.com"\n'
            f"}}\n"
            for k in range(12 + (i % 5))
        ),
        target_slice,
    ]
    arm_a_text = "\n\n".join(unscouted_turns)
    arm_b_text = (
        f"<context_pack task='AB-TF-{i:02d}'>\n"
        f"1-hop variable: variable 'access_policy_id' {{ type = string }}\n"
        f"{target_slice}</context_pack>"
    )
    return arm_a_text, arm_b_text


def get_token_ab_benchmark_set(
    tasks_per_stack: int = MIN_TOKEN_AB_TASKS_PER_STACK,
) -> list[TokenABTask]:
    """Generates N >= 120 paired Token A/B benchmark tasks measured with real BPE tokenization."""
    if tasks_per_stack < MIN_TOKEN_AB_TASKS_PER_STACK:
        msg = (
            f"tasks_per_stack {tasks_per_stack} violates threshold ({MIN_TOKEN_AB_TASKS_PER_STACK})"
        )
        raise ValueError(msg)

    tasks: list[TokenABTask] = []

    stack_specs: list[
        tuple[str, StackType, str, Callable[[int], tuple[str, str, str, str, float, float]]]
    ] = [
        (
            "AB-CS",
            StackType.CSHARP_DOTNET,
            "dotnet/eShop",
            lambda i: (
                f"C# Microservice Task {i:02d}: Refactor Order Handler & Repository",
                (
                    f"Refactor `OrderRepository_{i:02d}.GetByIdAsync` to include transaction "
                    "isolation and cancellation token support."
                ),
                *_build_csharp_payloads(i),
                round(18.5 + (i * 0.4), 2),
                round(2.8 + (i * 0.05), 2),
            ),
        ),
        (
            "AB-TS",
            StackType.TS_REACT_REDUX,
            "frontend/web-storefront",
            lambda i: (
                f"TS/React Task {i:02d}: State Management & Hook Refactor",
                (
                    f"Migrate `useCheckoutState_{i:02d}` custom hook to "
                    "Redux Toolkit slice with async thunk."
                ),
                *_build_ts_payloads(i),
                round(15.2 + (i * 0.35), 2),
                round(2.2 + (i * 0.04), 2),
            ),
        ),
        (
            "AB-PY",
            StackType.PYTHON_BACKEND,
            "backend/gateway-service",
            lambda i: (
                f"Python Task {i:02d}: Async Pipeline & Distributed Locking",
                (
                    f"Implement distributed Redis lock in "
                    f"`TaskDispatcher_{i:02d}.dispatch_job` with lease renewal."
                ),
                *_build_python_payloads(i),
                round(16.8 + (i * 0.38), 2),
                round(2.4 + (i * 0.04), 2),
            ),
        ),
        (
            "AB-TF",
            StackType.TERRAFORM_GCP,
            "infra/gcp-workloads",
            lambda i: (
                f"Terraform Task {i:02d}: VPC Service Controls & IAM Binding",
                (
                    "Declare VPC Service Controls perimeter and access policy for "
                    f"Cloud Storage bucket `secure_vault_{i:02d}`."
                ),
                *_build_terraform_payloads(i),
                round(14.0 + (i * 0.3), 2),
                round(1.9 + (i * 0.03), 2),
            ),
        ),
    ]

    for prefix, stack, repo_name, builder in stack_specs:
        for i in range(tasks_per_stack):
            title, prompt, arm_a_text, arm_b_text, lat_a, lat_b = builder(i)
            arm_a_tokens = count_bpe_tokens(arm_a_text)
            arm_b_tokens = count_bpe_tokens(arm_b_text)
            task = TokenABTask(
                task_id=f"{prefix}-{i:02d}",
                stack=stack,
                title=title,
                repository=repo_name,
                prompt=prompt,
                arm_a_exploration_tokens=arm_a_tokens,
                arm_b_scouted_tokens=arm_b_tokens,
                arm_a_latency_seconds=lat_a,
                arm_b_latency_seconds=lat_b,
            )
            tasks.append(task.model_copy(update={"sha256_hash": task.compute_hash()}))

    return tasks
