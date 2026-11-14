"""Resolve local gitignored corpus directories across main checkout and git worktrees."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


def resolve_local_data_dir(relative_path: str) -> Path:
    """Resolve a gitignored local corpus directory without silently skipping.

    Checks cwd, the worktree root, and the primary repository root (via
    ``git rev-parse --git-common-dir``) so tests execute reliably from both the
    main checkout and linked git worktrees. Fails loudly if missing.
    """
    candidates: list[Path] = [
        Path(relative_path),
        Path(__file__).resolve().parents[2] / relative_path,
    ]
    try:
        common_dir = subprocess.check_output(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if common_dir:
            common_path = Path(common_dir)
            if not common_path.is_absolute():
                common_path = (Path(__file__).resolve().parents[2] / common_path).resolve()
            candidates.append(common_path.parent / relative_path)
    except (OSError, subprocess.SubprocessError):
        pass

    for candidate in candidates:
        if candidate.exists():
            return candidate

    pytest.fail(f"Required local test corpus directory missing: {relative_path}")
