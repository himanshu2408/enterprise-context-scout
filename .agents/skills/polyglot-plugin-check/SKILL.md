---
name: polyglot-plugin-check
description: Verifies Tree-sitter and SCIP language plugins, fail-closed syntax error abstention, cross-boundary contract edges, and O(n) anti-ReDoS bounds. Use when adding or modifying AST extractors, SCIP indexers, or language plugins.
---
# Polyglot AST & SCIP Plugin Verification (`/polyglot-plugin-check`)

1. Confirm `tree-sitter` or SCIP backs the plugin (`low_confidence_fallback=True` on any
   heuristic fallback).
2. Assert corrupted syntax returns `([], [])` with `has_parse_error=True, abstained=True`.
3. Assert decoy tokens inside comments/heredocs/raw-strings produce zero ghost symbols.
4. Assert `< 250 ms` linear-time completion on 100 KB pathological inputs.
