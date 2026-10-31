"""Teacher-Student Distillation & QLoRA offline tuning engine."""

from context_scout.tuning.benchmark import (
    BenchmarkSampleResult,
    QLoRABenchmarkRunner,
    run_benchmark,
)
from context_scout.tuning.teacher_distiller import DistillationStats, TeacherDistiller
from context_scout.tuning.train_mlx_lora import (
    DEFAULT_DENSE_MODEL_ID,
    DEFAULT_GPU_LOCK_PATH,
    DEFAULT_MOE_MODEL_ID,
    FROZEN_MODULES_MOE,
    TARGET_MODULES_MOE,
    GPUContentionError,
    GPUMutualExclusionLock,
    LocalQLoRATrainer,
    ModelAcquisitionResult,
    ModelVerificationResult,
    QLoRAHyperparameters,
    QLoRAMemoryProfile,
    TrainingMetrics,
    TrainingRunResult,
    acquire_hf_model,
    compute_qlora_memory_budget,
    verify_model_weights,
)
from context_scout.tuning.train_vertex_peft import (
    VertexPEFTConfig,
    VertexPEFTTrainer,
    VertexPEFTValidationResult,
)

__all__ = [
    "DEFAULT_DENSE_MODEL_ID",
    "DEFAULT_GPU_LOCK_PATH",
    "DEFAULT_MOE_MODEL_ID",
    "FROZEN_MODULES_MOE",
    "TARGET_MODULES_MOE",
    "BenchmarkSampleResult",
    "DistillationStats",
    "GPUContentionError",
    "GPUMutualExclusionLock",
    "LocalQLoRATrainer",
    "ModelAcquisitionResult",
    "ModelVerificationResult",
    "QLoRABenchmarkRunner",
    "QLoRAHyperparameters",
    "QLoRAMemoryProfile",
    "TeacherDistiller",
    "TrainingMetrics",
    "TrainingRunResult",
    "VertexPEFTConfig",
    "VertexPEFTTrainer",
    "VertexPEFTValidationResult",
    "acquire_hf_model",
    "compute_qlora_memory_budget",
    "run_benchmark",
    "verify_model_weights",
]
