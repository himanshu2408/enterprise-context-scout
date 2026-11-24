---
name: gate-verify
description: Runs ruff, mypy, pytest, web-ui lint/typecheck/vitest/build, and repo hygiene gates in compact summary mode (<80 lines). Use after code edits, before commits, or when verifying milestone gates.
---
# Compact Gate Verification Skill (`/gate-verify`)

Full unskipped gate suite (`--scope all`, mandatory from M0-03 onward):
```bash
bash .agents/skills/gate-verify/scripts/run_compact_gates.sh --scope all
```

Fast pre-M0-03 gate suite (`--scope fast`, skips 5-min 17GB live MLX smoke load):
```bash
bash .agents/skills/gate-verify/scripts/run_compact_gates.sh --scope fast
```

Targeted pytest run in compact mode:
```bash
bash .agents/skills/gate-verify/scripts/run_compact_gates.sh --scope py --pytest-target tests/test_domain_neutrality_guard.py
```
