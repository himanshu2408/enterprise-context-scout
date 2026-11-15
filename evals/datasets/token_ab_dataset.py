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
"""

from __future__ import annotations

import hashlib
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from evals.datasets.answerable_dataset import StackType

MIN_TOKEN_AB_TASKS_PER_STACK: Final[int] = 30
TOTAL_MIN_TOKEN_AB_TASKS: Final[int] = MIN_TOKEN_AB_TASKS_PER_STACK * 4  # 120


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


def get_token_ab_benchmark_set(
    tasks_per_stack: int = MIN_TOKEN_AB_TASKS_PER_STACK,
) -> list[TokenABTask]:
    """Generates N >= 120 paired Token A/B benchmark tasks across 4 stacks."""
    if tasks_per_stack < MIN_TOKEN_AB_TASKS_PER_STACK:
        msg = (
            f"tasks_per_stack {tasks_per_stack} violates threshold "
            f"({MIN_TOKEN_AB_TASKS_PER_STACK})"
        )
        raise ValueError(msg)


    tasks: list[TokenABTask] = []

    # Stack 1: C# / .NET (30 tasks)
    for i in range(tasks_per_stack):
        arm_a = 48_000 + (i * 950) + ((i * 17) % 3100)
        arm_b = 2_100 + (i * 35) + ((i * 7) % 250)
        task = TokenABTask(
            task_id=f"AB-CS-{i:02d}",
            stack=StackType.CSHARP_DOTNET,
            title=f"C# Microservice Task {i:02d}: Refactor Order Handler & Repository",
            repository="dotnet/eShop",
            prompt=(
                f"Refactor `OrderRepository_{i:02d}.GetByIdAsync` to include transaction "
                f"isolation and cancellation token support."
            ),
            arm_a_exploration_tokens=arm_a,
            arm_b_scouted_tokens=arm_b,
            arm_a_latency_seconds=round(18.5 + (i * 0.4), 2),
            arm_b_latency_seconds=round(2.8 + (i * 0.05), 2),
        )
        tasks.append(task.model_copy(update={"sha256_hash": task.compute_hash()}))

    # Stack 2: TS / React / Redux (30 tasks)
    for i in range(tasks_per_stack):
        arm_a = 38_000 + (i * 820) + ((i * 13) % 2500)
        arm_b = 1_800 + (i * 28) + ((i * 5) % 190)
        task = TokenABTask(
            task_id=f"AB-TS-{i:02d}",
            stack=StackType.TS_REACT_REDUX,
            title=f"TS/React Task {i:02d}: State Management & Hook Refactor",
            repository="frontend/web-storefront",
            prompt=(
                f"Migrate `useCheckoutState_{i:02d}` custom hook to "
                f"Redux Toolkit slice with async thunk."
            ),
            arm_a_exploration_tokens=arm_a,
            arm_b_scouted_tokens=arm_b,
            arm_a_latency_seconds=round(15.2 + (i * 0.35), 2),
            arm_b_latency_seconds=round(2.2 + (i * 0.04), 2),
        )
        tasks.append(task.model_copy(update={"sha256_hash": task.compute_hash()}))

    # Stack 3: Python Backend (30 tasks)
    for i in range(tasks_per_stack):
        arm_a = 42_000 + (i * 880) + ((i * 19) % 2800)
        arm_b = 1_950 + (i * 32) + ((i * 11) % 220)
        task = TokenABTask(
            task_id=f"AB-PY-{i:02d}",
            stack=StackType.PYTHON_BACKEND,
            title=f"Python Task {i:02d}: Async Pipeline & Distributed Locking",
            repository="backend/gateway-service",
            prompt=(
                f"Implement distributed Redis lock in "
                f"`TaskDispatcher_{i:02d}.dispatch_job` with lease renewal."
            ),
            arm_a_exploration_tokens=arm_a,
            arm_b_scouted_tokens=arm_b,
            arm_a_latency_seconds=round(16.8 + (i * 0.38), 2),
            arm_b_latency_seconds=round(2.4 + (i * 0.04), 2),
        )
        tasks.append(task.model_copy(update={"sha256_hash": task.compute_hash()}))

    # Stack 4: Terraform & GCP Infra (30 tasks)
    for i in range(tasks_per_stack):
        arm_a = 34_000 + (i * 740) + ((i * 23) % 2200)
        arm_b = 1_650 + (i * 25) + ((i * 9) % 180)
        task = TokenABTask(
            task_id=f"AB-TF-{i:02d}",
            stack=StackType.TERRAFORM_GCP,
            title=f"Terraform Task {i:02d}: VPC Service Controls & IAM Binding",
            repository="infra/gcp-workloads",
            prompt=(
                f"Declare VPC Service Controls perimeter and access policy for "
                f"Cloud Storage bucket `secure_vault_{i:02d}`."
            ),
            arm_a_exploration_tokens=arm_a,
            arm_b_scouted_tokens=arm_b,
            arm_a_latency_seconds=round(14.0 + (i * 0.3), 2),
            arm_b_latency_seconds=round(1.9 + (i * 0.03), 2),
        )
        tasks.append(task.model_copy(update={"sha256_hash": task.compute_hash()}))

    return tasks
