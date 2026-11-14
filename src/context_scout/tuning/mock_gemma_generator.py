"""Deprecated module.

All simulation generators and test doubles have been relocated to
tests/fixtures/mock_generators.py per the Zero-Simulation Integrity Mandate.
Production code under src/ contains zero simulation generators.
"""

from __future__ import annotations


def __getattr__(name: str) -> None:
    raise ImportError(
        f"'{name}' has been excised from production code per the Zero-Simulation "
        "Integrity Mandate. For testing, import test doubles from tests.fixtures.mock_generators."
    )
