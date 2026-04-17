# TDA-PPIN

Reproducible exploratory research workflows for studying whether Topological Data Analysis, and persistent homology in particular, captures biologically meaningful structure in protein-protein interaction networks.

## Canonical Research Question
Can persistent homology derived from weighted PPI networks recover biological signal related to known protein complexes, and does it add value beyond simpler graph-based summaries?

## Current Repository Shape
- `src/tda_ppin`: reusable loaders, graph construction, persistent homology, plotting, and evaluation helpers.
- `scripts/run_biogrid_reference.py`: canonical BioGrid + CORUM reference workflow.
- `scripts/run_biogrid_decision_stage.py`: decision-stage experiment suite for evaluating research value.
- `scripts/run_biogrid_validation_sweep.py`: larger grouped validation sweep using stricter controls.
- `scripts/run_biogrid_local_filtration_exploration.py`: local-neighborhood PH and alternative-filtration exploration workflow.
- `scripts/run_synthetic_sanity.py`: deterministic synthetic sanity check for debugging assumptions.
- `scripts/validate_environment.py`: verifies the required Python modules are installed.
- `results/`: generated figures, processed tables, and run reports.
- `tda/`: legacy notebooks and older exploratory scripts kept as reference material.
- `preprocessing/`: original exploratory notebook for dataset inspection and earlier preprocessing notes.

## Canonical Workflow
The canonical workflow is the human `BioGrid + CORUM` path:

1. Load `data/Human_PPI_Network.txt` as a weighted PPI network using `SemSim`.
2. Load `data/CORUM_Human_Complexes.txt` as the biological reference set.
3. Build the weighted adjacency matrix and convert it to a correlation-distance matrix with `distance = 1 - SemSim`.
4. Run persistent homology with Ripser and GUDHI.
5. Save degree distributions, persistence diagrams, barcodes, persistence landscapes, and a structured run report under `results/`.
6. Compare these outputs against simple graph baselines before attempting predictive modeling.

## Setup
Create a Python environment with the project dependencies:

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
python3 scripts/validate_environment.py
```

## Running The Canonical Workflow
Run the BioGrid + CORUM reference experiment:

```bash
python3 scripts/run_biogrid_reference.py
```

This writes reproducible artifacts under:
- `results/figures/biogrid_reference/`
- `results/processed/biogrid_reference/`
- `results/reports/biogrid_reference/`

Run the synthetic sanity check:

```bash
python3 scripts/run_synthetic_sanity.py
```

Run the decision-stage research evaluation:

```bash
python3 scripts/run_biogrid_decision_stage.py
```

This writes machine-readable experiment outputs under:
- `results/processed/biogrid_decision_stage/`
- `results/reports/biogrid_decision_stage/`

Run the larger validation sweep:

```bash
python3 scripts/run_biogrid_validation_sweep.py
```

This writes machine-readable experiment outputs under:
- `results/processed/biogrid_validation_sweep/`
- `results/reports/biogrid_validation_sweep/`

Run the local-neighborhood and filtration exploration:

```bash
python3 scripts/run_biogrid_local_filtration_exploration.py
```

This writes machine-readable experiment outputs under:
- `results/processed/biogrid_local_filtration_exploration/`
- `results/reports/biogrid_local_filtration_exploration/`

## Research Value: What To Look At Next
The current repo now supports the “up to current results” stage and frames the next research stage explicitly:

- test whether persistence-derived summaries align with known CORUM complex structure,
- compare PH-derived summaries against simple graph baselines such as degree and local connectivity,
- run null controls such as shuffled complex labels before claiming biological signal,
- only move to prediction once there is evidence that PH contributes signal beyond trivial graph structure.

The decision-stage runner operationalizes that into:
- protein-level baseline evaluation,
- global PH stability under perturbations,
- real-complex versus matched-random subgraph experiments,
- null-control checks,
- an explicit `go` / `conditional_go` / `no_go` recommendation.

The detailed workflow and next-step framing are documented in [docs/research_workflow.md](docs/research_workflow.md).
