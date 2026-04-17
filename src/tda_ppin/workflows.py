from __future__ import annotations

from datetime import UTC, datetime

from .evaluation import summarize_complex_coverage, summarize_degree_baseline
from .io import load_biogrid_ppi, load_corum_complexes, write_json, write_table
from .paths import ensure_directory, get_repo_paths
from .ph import adjacency_and_corr_distance, build_weighted_graph, run_persistent_homology
from .plotting import save_degree_distribution


def run_biogrid_reference_workflow() -> dict[str, object]:
    paths = get_repo_paths()
    ppi_path = paths.data_dir / "Human_PPI_Network.txt"
    complexes_path = paths.data_dir / "CORUM_Human_Complexes.txt"

    ppi_df = load_biogrid_ppi(ppi_path)
    complexes = load_corum_complexes(complexes_path)
    graph = build_weighted_graph(ppi_df)
    adjacency, corr_distance = adjacency_and_corr_distance(graph)

    run_name = "biogrid_reference"
    figure_dir = ensure_directory(paths.figures_dir / run_name)
    processed_dir = ensure_directory(paths.processed_dir / run_name)
    report_dir = ensure_directory(paths.reports_dir / run_name)

    save_degree_distribution(
        list(dict(graph.degree()).values()),
        figure_dir / "degree_distribution.png",
        "BioGrid PPI Node Degree Distribution",
    )

    ph_artifacts = run_persistent_homology(adjacency, corr_distance, figure_dir, run_name)

    coverage = summarize_complex_coverage(graph, complexes)
    degree_baseline = summarize_degree_baseline(graph, complexes)
    write_table(degree_baseline, processed_dir / "degree_baseline.csv")

    report = {
        "run_name": run_name,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "inputs": {
            "ppi": str(ppi_path.relative_to(paths.repo_root)),
            "complexes": str(complexes_path.relative_to(paths.repo_root)),
        },
        "graph": {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
        },
        "filtration": {
            "type": "weighted adjacency converted to correlation distance",
            "formula": "distance = 1 - SemSim",
            "ripser_maxdim": 3,
            "gudhi_max_dimension": 3,
            "gudhi_max_edge_length": 1.0,
        },
        "coverage": coverage,
        "rips_complex_summary": ph_artifacts.rips_complex_summary,
        "outputs": {
            "figures_dir": str(figure_dir.relative_to(paths.repo_root)),
            "processed_dir": str(processed_dir.relative_to(paths.repo_root)),
            "degree_baseline": str((processed_dir / "degree_baseline.csv").relative_to(paths.repo_root)),
        },
        "next_research_questions": [
            "Compare persistence-derived summaries against degree and local density baselines.",
            "Measure whether proteins participating in known CORUM complexes occupy distinctive topological regimes.",
            "Test null controls with shuffled complex labels before attempting predictive modeling.",
        ],
    }
    write_json(report, report_dir / "run_report.json")
    return report
