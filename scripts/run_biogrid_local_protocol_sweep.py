#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tda_ppin.experiments import LocalProtocolSweepConfig
from tda_ppin.workflows import run_biogrid_local_protocol_sweep_workflow


def main() -> int:
    config = LocalProtocolSweepConfig(
        local_positive_proteins=100,
        local_negative_proteins=100,
        neighborhood_node_caps=(10, 20, 30, 50),
        filtration="weighted_shortest_path",
        repeats=4,
    )
    report = run_biogrid_local_protocol_sweep_workflow(config)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
