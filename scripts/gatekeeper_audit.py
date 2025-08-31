#!/usr/bin/env python3
"""Lightweight repository auditor used by CI Gatekeeper."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict


def collect_evidence(root: Path) -> Dict[str, int]:
    """Collect simple repository metrics as evidence."""
    py_files = list(root.rglob("*.py"))
    test_files = list((root / "tests").rglob("test_*.py")) if (root / "tests").exists() else []
    return {
        "python_files": len(py_files),
        "test_files": len(test_files),
    }


def score(evidence: Dict[str, int]) -> int:
    """Compute a naive rubric score out of 100."""
    base = 50
    if evidence.get("test_files"):
        base += 25
    if evidence.get("python_files", 0) > 20:
        base += 25
    return base


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    evidence = collect_evidence(repo)
    result = {"evidence": evidence, "score": score(evidence)}
    json.dump(result, sys.stdout, indent=2)


if __name__ == "__main__":  # pragma: no cover - script entry
    main()
