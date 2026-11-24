---
trigger: glob
description: "BYO open-weights LLM governance, safetensors-only loading, ACL training_scope_ceiling routing, rsLoRA/LoRA+, and vLLM multi-LoRA safety."
globs: "**/src/context_scout/tuning/**, **/src/context_scout/agents/**, **/domains/**, **/config/**"
---
# BYO Open-Weights LLM & Multi-LoRA Specialization Rules

1. **Supply-Chain Safety:** Load model and adapter weights strictly from `.safetensors`
   with `trust_remote_code=False`. Default `allow_external_cloud_egress=False`.
2. **Cross-ACL Weight Isolation (`OWASP ASI06`):** Every LoRA manifest MUST declare
   `training_scope_ceiling: list[str]`. Route to a LoRA cartridge ONLY when
   `set(training_scope_ceiling).issubset(set(principal.allowed_scopes))`; otherwise route
   to the frozen base model (`lora_request=None`).
3. **vLLM Multi-LoRA Compatibility (`vllm#12174`):** Enforce `rank in {8, 16, 32}`,
   `use_rslora=True`, and `use_dora=False`. Freeze MoE router gates and Mamba SSM states.
   Run a 1-step `vLLM` load dry-run before full GPU training.
4. **Chat & Reasoning Templates:** Never hardcode Gemma `<start_of_turn>` tags. Always
   format via `AutoTokenizer.apply_chat_template` and `vLLM --reasoning-parser`.
5. **Leakage-Free Training:** Use `TRL SFTTrainer` with `assistant_only_loss=True`, 15%
   general replay buffer, repo/time-disjoint `80/10/10` splits, and
   `CanaryMemorizationGate` (`0/10` canary extraction).
