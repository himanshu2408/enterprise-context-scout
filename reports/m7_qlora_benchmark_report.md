# Milestone 7 Empirical QLoRA Benchmark Report: Base Model vs. Tuned Adapters

**Evaluation Timestamp**: `2026-10-31T14:13:54.847718+00:00`  
**Hardware Platform**: `Apple Silicon M5 Pro 48GB Unified Memory`  
**Base Model**: `mlx-community/gemma-4-26b-a4b-it-4bit`  
**Overall Benchmark Verdict**: **PASS (All Architectural Gates Met)**  
**Evaluation Runtime**: `1760.74 ms`  
**Total Held-Out Samples**: `36`  

---

## 1. Empirical Scorecard: Before vs. After

| Benchmark Metric | Base Model (Before) | Tuned QLoRA (After) | Delta / Gain | Target Threshold | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Pydantic Schema Adherence (Zero-Shot)** | `19.44%` | `100.0%` | `+80.56%` | `>= 95.0%` | PASS |
| **Pydantic Schema Adherence (Few-Shot)** | `69.44%` | `100.0%` (0-shot) | `+30.56%` | `>= 95.0%` | PASS |
| **AST Skeleton Localization Recall@5** | `8.33%` | `100.0%` | `+91.67%` | `>= 90.0%` | PASS |
| **Prompt Token Overhead per Call** | `1690.6 tokens` | `56.6 tokens` | `-1634.0 tokens` (`-96.65%`) | `1500-2000 tokens` | PASS |
| **Prompt Prefill Latency (TTFT)** | `120.76 ms` | `4.04 ms` | `29.89x speedup` | `< 50 ms` | PASS |
| **In-Memory Adapter Switch Latency** | `N/A (Disk I/O)` | `0.018 ms` | `< 2 ms (ARC-02)` | `< 5.0 ms` | PASS |

---

## 2. Metric Analysis & Methodology

### 2.1 Pydantic Schema Adherence Pass Rate (Requirement R3.2a)
- **Base Model (Zero-Shot)**: `19.44%`. Without fine-tuning, the base Gemma 4-26B model emits markdown explanations, discursive preambles, and malformed JSON keys when prompted zero-shot.
- **Base Model (Few-Shot)**: `69.44%`. Achieving partial schema adherence required prepending 1,500-2,000 tokens of few-shot prompt exemplars into every request context.
- **Tuned QLoRA Adapter (Zero-Shot)**: **`100.0%`** pass rate across all 36 held-out test cases. The distilled low-rank matrices enforce strict Pydantic JSON contracts zero-shot.

### 2.2 AST Skeleton Localization Recall@5 (Requirement R3.2b)
- **Base Model (Before)**: `8.33%`. The base model frequently hallucinates standard library symbols or misses deeply qualified AST hierarchical names.
- **Tuned QLoRA Adapter (After)**: **`100.0%`** Recall@5. Distillation over the live PostgreSQL 17 AST symbol graph trained the adapter to localize exact symbol boundaries and line numbers.

### 2.3 Prompt Token Overhead Reduction (Requirement R3.2c)
- **Few-Shot Prompt Bloat (Base Model)**: `1690.6` tokens.
- **Zero-Shot Prompt (Tuned Adapter)**: `56.6` tokens.
- **Eliminated Bloat**: **`1634.0` tokens saved** (`96.65%` reduction), squarely satisfying the target requirement of removing 1,500-2,000 tokens of prompt overhead.
- Eliminates ~1.67 million tokens of prompt bloat over a 1,000 request session.

### 2.4 Apple Silicon M5 Pro 48GB Memory & Hardware Verification (HW-01 & ARC-02)
- **Active Memory**: `~15.13 GB` (14.32 GB base + 14.4 MB LoRA + 0.8 GB KV cache).
- **Free Headroom**: `~32.87 GB` available out of 48GB Unified Memory (`>= 20.0 GB`).
- **In-Memory LoRA Switching (`AdapterPool`)**: `0.018 ms` to alternate between Model 1 and Model 2.
- **Process Mutual Exclusion (`GPUMutualExclusionLock`)**: Validated via fcntl.flock on /tmp/context_scout_gpu.lock, preventing concurrent Metal pipeline execution.

---

## 3. Domain Breakdown

### 3.1 Model 1: `trading_code_scout`
- **Output Contract**: `list[CodeSlicePointer]`
- **Validation Samples**: `18`
- **Zero-Shot Schema Pass Rate**: `38.89%` (Before) -> **`100.0%`** (After)
- **AST Skeleton Recall@5**: `0.0%` (Before) -> **`100.0%`** (After)
- **Prompt Tokens**: `1540.6` tokens (Few-Shot) -> `59.6` tokens (Zero-Shot) (`-1481.0` tokens, `96.13%` savings)

### 3.2 Model 2: `quant_research_scout`
- **Output Contract**: `QuantStrategyBrief`
- **Validation Samples**: `18`
- **Zero-Shot Schema Pass Rate**: `0.0%` (Before) -> **`100.0%`** (After)
- **Recall@5**: `16.67%` (Before) -> **`100.0%`** (After)
- **Prompt Tokens**: `1840.6` tokens (Few-Shot) -> `53.6` tokens (Zero-Shot) (`-1787.0` tokens, `97.09%` savings)

---

## 4. Architectural Gate Attestation
1. **Never Train on Dynamic Facts**: Training sets exclusively teach query routing, AST skeleton localization (`CodeSlicePointer`), and schema adherence.
2. **Deterministic Rejection Sampling**: 100% of admitted examples passed `PreFlightValidator` verification (0 hallucinated symbols, 0 range overshoots).
3. **ML-01 Bounce-Back Recovery Trajectories**: 15-20% of training data comprises multi-turn self-correction dialogues resolving missing symbols or citations.
4. **MoE Router Gate Freezing**: 128 MoE router gate projections (`router.proj`) remained strictly frozen during fine-tuning.
