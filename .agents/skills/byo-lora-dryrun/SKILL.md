---
name: byo-lora-dryrun
description: Validates BYO open-weights LLM specialization manifests, training_scope_ceiling ACL safety, 3-way train/valid/eval SHA-256 disjointness, rsLoRA settings, and vLLM multi-LoRA compatibility. Use when working on domain packs, distillation, or LoRA training.
---
# BYO Open-Weights LLM & LoRA Specialization Gate (`/byo-lora-dryrun`)

```bash
uv run python .agents/skills/byo-lora-dryrun/scripts/check_splits_and_ceiling.py
```
Verify `use_dora=False`, `use_rslora=True`, `rank <= 32`, non-empty
`training_scope_ceiling`, `assistant_only_loss=True`, 15% replay, and `0/10` canary leak.
