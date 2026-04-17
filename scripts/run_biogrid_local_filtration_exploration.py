#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tda_ppin.experiments import ExtensionExplorationConfig
from tda_ppin.workflows import run_biogrid_local_filtration_exploration_workflow


def main() -> int:
    config = ExtensionExplorationConfig(
        local_positive_proteins=100,
        local_negative_proteins=100,
        local_neighborhood_node_cap=30,
        local_repeats=3,
        filtration_repeats=3,
        filtration_max_real_complexes=40,
        filtration_matched_random_per_complex=1,
        filtration_max_random_attempts=80,
        filtration_max_complex_size=25,
        filtration_subgraph_ph_node_cap=60,
    )
    report = run_biogrid_local_filtration_exploration_workflow(config)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
