"""Answerable Grounded Benchmark Evaluation Dataset (EVAL-04, N >= 300).

Sample size N=300 across 4 enterprise stacks (.NET/C#, TS/React, Python, Terraform).
Target: >= 95% pass rate on ImplementationContextPack generation, Recall@5 >= 90%.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

MIN_ANSWERABLE_SAMPLE_SIZE: Final[int] = 300


class StackType(StrEnum):
    CSHARP_DOTNET = "CSHARP_DOTNET"
    TS_REACT_REDUX = "TS_REACT_REDUX"
    PYTHON_BACKEND = "PYTHON_BACKEND"
    TERRAFORM_GCP = "TERRAFORM_GCP"


class AnswerableTask(BaseModel):
    """Single answerable grounded evaluation task."""

    model_config = ConfigDict(frozen=True)

    task_id: str = Field(min_length=1)
    stack: StackType
    prompt: str = Field(min_length=1)
    target_repo: str = Field(min_length=1)
    target_file: str = Field(min_length=1)
    target_symbols: list[str] = Field(min_length=1)
    expected_doc_citations: list[str] = Field(default_factory=list)
    max_token_budget: int = 4000
    is_answerable: bool = True
    sha256_hash: str = ""

    def compute_hash(self) -> str:
        syms = ",".join(self.target_symbols)
        payload = f"{self.task_id}|{self.stack}|{self.prompt}|{self.target_file}|{syms}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_answerable_benchmark_set(
    sample_size: int = MIN_ANSWERABLE_SAMPLE_SIZE,
) -> list[AnswerableTask]:
    """Generates N >= 300 deterministic answerable grounded benchmark tasks across 4 stacks."""
    if sample_size < MIN_ANSWERABLE_SAMPLE_SIZE:
        msg = f"Sample size {sample_size} violates threshold (N >= {MIN_ANSWERABLE_SAMPLE_SIZE})"
        raise ValueError(msg)

    tasks: list[AnswerableTask] = []

    # 1. C# / .NET Tasks (75 tasks)
    for i in range(75):
        task = AnswerableTask(
            task_id=f"ANS-CS-{i:03d}",
            stack=StackType.CSHARP_DOTNET,
            prompt=(
                f"Add Polly retry policy with exponential backoff to "
                f"`OrderCheckoutHandler_{i:02d}.HandleOrderAsync`."
            ),
            target_repo="dotnet/eShop",
            target_file=(
                f"src/Services/Ordering/Ordering.API/Application/"
                f"Commands/OrderCheckoutHandler_{i:02d}.cs"
            ),
            target_symbols=[f"OrderCheckoutHandler_{i:02d}", "HandleOrderAsync", "OrderRepository"],
            expected_doc_citations=[f"conf_eshop_ordering_architecture_v{1 + (i % 3)}"],
        )
        tasks.append(task.model_copy(update={"sha256_hash": task.compute_hash()}))

    # 2. TS / React / Redux Tasks (75 tasks)
    for i in range(75):
        task = AnswerableTask(
            task_id=f"ANS-TS-{i:03d}",
            stack=StackType.TS_REACT_REDUX,
            prompt=(
                f"Add optimistic UI update reducer to `cartSlice_{i:02d}` "
                f"for item quantity updates."
            ),
            target_repo="frontend/web-storefront",
            target_file=f"src/features/cart/cartSlice_{i:02d}.ts",
            target_symbols=[f"cartSlice_{i:02d}", "updateQuantity", "selectCartTotal"],
            expected_doc_citations=[f"conf_storefront_state_mgmt_v{1 + (i % 2)}"],
        )
        tasks.append(task.model_copy(update={"sha256_hash": task.compute_hash()}))

    # 3. Python Backend Tasks (75 tasks)
    for i in range(75):
        task = AnswerableTask(
            task_id=f"ANS-PY-{i:03d}",
            stack=StackType.PYTHON_BACKEND,
            prompt=(
                f"Implement idempotent Redis token bucket rate limiting on "
                f"`FastAPIProxy_{i:02d}.dispatch`."
            ),
            target_repo="backend/gateway-service",
            target_file=f"src/gateway/proxy_{i:02d}.py",
            target_symbols=[f"FastAPIProxy_{i:02d}", "dispatch", "TokenBucketLimiter"],
            expected_doc_citations=[f"jira_GATEWAY_{100 + i}"],
        )
        tasks.append(task.model_copy(update={"sha256_hash": task.compute_hash()}))

    # 4. Terraform & GCP Infra Tasks (75 tasks)
    for i in range(75):
        task = AnswerableTask(
            task_id=f"ANS-TF-{i:03d}",
            stack=StackType.TERRAFORM_GCP,
            prompt=(
                f"Provision private Cloud SQL PostgreSQL 17 cluster with Cloud KMS "
                f"encryption in module `db_cluster_{i:02d}`."
            ),
            target_repo="infra/gcp-workloads",
            target_file=f"modules/storage/cloud_sql_{i:02d}.tf",
            target_symbols=[
                f"google_sql_database_instance.postgres_{i:02d}",
                "google_kms_crypto_key.db_key",
            ],
            expected_doc_citations=[f"rfc_cloud_security_infra_v{1 + (i % 4)}"],
        )
        tasks.append(task.model_copy(update={"sha256_hash": task.compute_hash()}))

    # Extra tasks if sample_size > 300
    for i in range(len(tasks), sample_size):
        task = AnswerableTask(
            task_id=f"ANS-EXTRA-{i:03d}",
            stack=StackType.PYTHON_BACKEND,
            prompt=f"Add structured logging span to `WorkerProcess_{i:03d}.run`.",
            target_repo="backend/worker-cluster",
            target_file=f"src/worker/proc_{i:03d}.py",
            target_symbols=[f"WorkerProcess_{i:03d}", "run"],
        )
        tasks.append(task.model_copy(update={"sha256_hash": task.compute_hash()}))

    return tasks
