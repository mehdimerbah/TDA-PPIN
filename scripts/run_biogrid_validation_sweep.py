#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tda_ppin.experiments import DecisionStageConfig
from tda_ppin.workflows import run_biogrid_decision_stage_workflow


def main() -> int:
    config = DecisionStageConfig(
        repeats=4,
        matched_random_per_complex=2,
        max_random_attempts=120,
        max_real_complexes=100,
        max_complex_size=25,
        global_ph_node_cap=120,
        subgraph_ph_node_cap=60,
    )
    report = run_biogrid_decision_stage_workflow(
        config,
        run_name="biogrid_validation_sweep",
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
