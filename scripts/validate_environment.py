#!/usr/bin/env python3

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


REQUIRED_MODULES = [
    "gudhi",
    "matplotlib",
    "networkx",
    "numpy",
    "pandas",
    "persim",
    "ripser",
]


def main() -> int:
    missing = []
    for module_name in REQUIRED_MODULES:
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            missing.append(module_name)

    if missing:
        print("Missing modules:", ", ".join(missing))
        return 1

    print("Environment validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
