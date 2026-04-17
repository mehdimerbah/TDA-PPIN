#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tda_ppin.synthetic import SyntheticConfig, run_synthetic_sanity_workflow


def main() -> int:
    result = run_synthetic_sanity_workflow(SyntheticConfig())
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
