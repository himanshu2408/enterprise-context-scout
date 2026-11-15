"""Same-Name Collision Benchmark Evaluation Dataset (EVAL-03, N >= 100).

Statistically justified sample size N=100 guaranteeing that observing 0 cross-repo leaks
proves true collision leakage error rate <= 3.0% at 95% confidence under Rule of Three.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

MIN_COLLISION_SAMPLE_SIZE: Final[int] = 100


class CollisionKind(StrEnum):
    IDENTICAL_CLASS_NAME = "IDENTICAL_CLASS_NAME"
    IDENTICAL_CONFIG_FILE_PATH = "IDENTICAL_CONFIG_FILE_PATH"
    IDENTICAL_FUNCTION_NAME = "IDENTICAL_FUNCTION_NAME"


class CollisionTask(BaseModel):
    """Single same-name collision test case."""

    model_config = ConfigDict(frozen=True)

    task_id: str = Field(min_length=1)
    collision_kind: CollisionKind
    bare_symbol_or_path: str = Field(min_length=1)
    competing_repos: list[str] = Field(min_length=2)
    repo_qualified_ids: list[str] = Field(min_length=2)
    unqualified_prompt: str = Field(min_length=1)
    disambiguated_prompt: str = Field(min_length=1)
    requires_qualification: bool = True
    sha256_hash: str = ""

    def compute_hash(self) -> str:
        repos = ",".join(self.competing_repos)
        payload = f"{self.task_id}|{self.collision_kind}|{self.bare_symbol_or_path}|{repos}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_same_name_collision_set(
    sample_size: int = MIN_COLLISION_SAMPLE_SIZE,
) -> list[CollisionTask]:
    """Generates N >= 100 deterministic same-name collision benchmark tasks."""
    if sample_size < MIN_COLLISION_SAMPLE_SIZE:
        msg = f"Sample size {sample_size} violates threshold (N >= {MIN_COLLISION_SAMPLE_SIZE})"
        raise ValueError(msg)

    tasks: list[CollisionTask] = []

    # 1. Identical Class Names Across Repos (40 items)
    class_names = [
        "UserRepository",
        "PaymentService",
        "AccountManager",
        "AuditLogger",
        "ConfigLoader",
        "EventDispatcher",
        "TokenValidator",
        "OrderProcessor",
        "NotificationClient",
        "CacheManager",
    ]
    for i in range(40):
        cls_name = class_names[i % len(class_names)] + f"_{i // len(class_names)}"
        repo_a = f"enterprise/service-auth-{i:02d}"
        repo_b = f"enterprise/service-billing-{i:02d}"
        qualified_a = f"{repo_a}::core::services::{cls_name}"
        qualified_b = f"{repo_b}::core::services::{cls_name}"
        task = CollisionTask(
            task_id=f"COLLIDE-CLS-{i:03d}",
            collision_kind=CollisionKind.IDENTICAL_CLASS_NAME,
            bare_symbol_or_path=cls_name,
            competing_repos=[repo_a, repo_b],
            repo_qualified_ids=[qualified_a, qualified_b],
            unqualified_prompt=f"Refactor class `{cls_name}` to use connection pooling.",
            disambiguated_prompt=(
                f"Refactor class `{cls_name}` in `{repo_a}` to use connection pooling."
            ),
        )
        tasks.append(task.model_copy(update={"sha256_hash": task.compute_hash()}))

    # 2. Identical Config File Paths (30 items)
    config_paths = [
        "src/config/settings.py",
        "appsettings.json",
        "config.yaml",
        "src/constants.ts",
        "terraform.tfvars",
        "src/env.py",
    ]
    for i in range(30):
        cfg_path = config_paths[i % len(config_paths)]
        repo_a = f"enterprise/frontend-app-{i:02d}"
        repo_b = f"enterprise/backend-api-{i:02d}"
        qualified_a = f"{repo_a}::{cfg_path}"
        qualified_b = f"{repo_b}::{cfg_path}"
        task = CollisionTask(
            task_id=f"COLLIDE-PATH-{i:03d}",
            collision_kind=CollisionKind.IDENTICAL_CONFIG_FILE_PATH,
            bare_symbol_or_path=cfg_path,
            competing_repos=[repo_a, repo_b],
            repo_qualified_ids=[qualified_a, qualified_b],
            unqualified_prompt=f"Update database connection parameters in `{cfg_path}`.",
            disambiguated_prompt=(
                f"Update database connection parameters in `{repo_b}::{cfg_path}`."
            ),
        )
        tasks.append(task.model_copy(update={"sha256_hash": task.compute_hash()}))

    # 3. Identical Function Names Across Repos (30 items)
    fn_names = [
        "initialize_client",
        "format_error_response",
        "process_incoming_event",
        "build_auth_headers",
        "calculate_tax_rate",
        "sanitize_input_string",
    ]
    for i in range(30):
        fn_name = fn_names[i % len(fn_names)] + f"_{i // len(fn_names)}"
        repo_a = f"enterprise/ingestion-worker-{i:02d}"
        repo_b = f"enterprise/export-worker-{i:02d}"
        qualified_a = f"{repo_a}::utils::{fn_name}"
        qualified_b = f"{repo_b}::utils::{fn_name}"
        task = CollisionTask(
            task_id=f"COLLIDE-FN-{i:03d}",
            collision_kind=CollisionKind.IDENTICAL_FUNCTION_NAME,
            bare_symbol_or_path=fn_name,
            competing_repos=[repo_a, repo_b],
            repo_qualified_ids=[qualified_a, qualified_b],
            unqualified_prompt=f"Add telemetry spans to function `{fn_name}()`.",
            disambiguated_prompt=f"Add telemetry spans to `{repo_a}::utils::{fn_name}()`.",
        )
        tasks.append(task.model_copy(update={"sha256_hash": task.compute_hash()}))

    # Extra tasks if sample_size > 100
    for i in range(len(tasks), sample_size):
        extra_name = f"DuplicateHelper_{i:03d}"
        repo_a = f"enterprise/svc-x-{i:02d}"
        repo_b = f"enterprise/svc-y-{i:02d}"
        task = CollisionTask(
            task_id=f"COLLIDE-EXTRA-{i:03d}",
            collision_kind=CollisionKind.IDENTICAL_FUNCTION_NAME,
            bare_symbol_or_path=extra_name,
            competing_repos=[repo_a, repo_b],
            repo_qualified_ids=[f"{repo_a}::{extra_name}", f"{repo_b}::{extra_name}"],
            unqualified_prompt=f"Optimize `{extra_name}()` loop.",
            disambiguated_prompt=f"Optimize `{repo_a}::{extra_name}()` loop.",
        )
        tasks.append(task.model_copy(update={"sha256_hash": task.compute_hash()}))

    return tasks
