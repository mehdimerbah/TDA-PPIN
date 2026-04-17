from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gudhi as gd
import gudhi.representations
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import ripser

from .plotting import save_barcode, save_landscape, save_ripser_diagrams


@dataclass
class PHArtifacts:
    adjacency_matrix: np.ndarray
    correlation_distance_matrix: np.ndarray
    ripser_diagrams: list[np.ndarray]
    gudhi_persistence: list[tuple[int, tuple[float, float]]]
    landscape_dim1: np.ndarray | None
    rips_complex_summary: dict[str, int]


def build_weighted_graph(ppi_df: pd.DataFrame) -> nx.Graph:
    return nx.from_pandas_edgelist(
        ppi_df,
        source="ProteinA",
        target="ProteinB",
        edge_attr="SemSim",
    )


def adjacency_and_corr_distance(graph: nx.Graph) -> tuple[np.ndarray, np.ndarray]:
    adjacency = nx.adjacency_matrix(graph, weight="SemSim").toarray()
    np.fill_diagonal(adjacency, 1.0)
    corr_distance = 1 - adjacency
    return adjacency, corr_distance


def compute_ripser_diagrams(corr_distance_matrix: np.ndarray, maxdim: int = 3) -> list[np.ndarray]:
    return ripser.ripser(
        corr_distance_matrix,
        distance_matrix=True,
        maxdim=maxdim,
    )["dgms"]


def summarize_ripser_diagrams(diagrams: list[np.ndarray]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for dimension in range(3):
        diag = diagrams[dimension] if dimension < len(diagrams) else np.empty((0, 2))
        count = float(len(diag))
        if count == 0:
            summary[f"ph_dim{dimension}_count"] = 0.0
            summary[f"ph_dim{dimension}_finite_count"] = 0.0
            summary[f"ph_dim{dimension}_mean_lifetime"] = 0.0
            summary[f"ph_dim{dimension}_max_lifetime"] = 0.0
            summary[f"ph_dim{dimension}_total_persistence"] = 0.0
            continue

        births = diag[:, 0]
        deaths = diag[:, 1]
        lifetimes = deaths - births
        finite_lifetimes = lifetimes[np.isfinite(lifetimes)]
        summary[f"ph_dim{dimension}_count"] = count
        summary[f"ph_dim{dimension}_finite_count"] = float(len(finite_lifetimes))
        summary[f"ph_dim{dimension}_mean_lifetime"] = (
            float(np.mean(finite_lifetimes)) if len(finite_lifetimes) else 0.0
        )
        summary[f"ph_dim{dimension}_max_lifetime"] = (
            float(np.max(finite_lifetimes)) if len(finite_lifetimes) else 0.0
        )
        summary[f"ph_dim{dimension}_total_persistence"] = (
            float(np.sum(finite_lifetimes)) if len(finite_lifetimes) else 0.0
        )

    if len(diagrams) > 1 and len(diagrams[1]) > 0:
        landscape = gd.representations.Landscape(num_landscapes=5).fit_transform([diagrams[1]])
        summary["ph_dim1_landscape_l1"] = float(np.abs(landscape).sum())
        summary["ph_dim1_landscape_max"] = float(np.max(landscape))
    else:
        summary["ph_dim1_landscape_l1"] = 0.0
        summary["ph_dim1_landscape_max"] = 0.0
    return summary


def summarize_graph_persistent_homology(
    graph: nx.Graph,
    *,
    maxdim: int = 3,
) -> dict[str, float]:
    if graph.number_of_nodes() <= 1:
        summary = summarize_ripser_diagrams([])
        summary["num_nodes"] = float(graph.number_of_nodes())
        summary["num_edges"] = float(graph.number_of_edges())
        summary["distance_matrix_mean"] = 0.0
        summary["distance_matrix_std"] = 0.0
        return summary

    _, corr_distance = adjacency_and_corr_distance(graph)
    diagrams = compute_ripser_diagrams(corr_distance, maxdim=maxdim)
    summary = summarize_ripser_diagrams(diagrams)
    summary["num_nodes"] = float(graph.number_of_nodes())
    summary["num_edges"] = float(graph.number_of_edges())
    summary["distance_matrix_mean"] = float(np.mean(corr_distance))
    summary["distance_matrix_std"] = float(np.std(corr_distance))
    return summary


def sample_graph_for_ph(
    graph: nx.Graph,
    *,
    max_nodes: int,
    seed: int = 7,
    preserve_hubs: int = 10,
) -> nx.Graph:
    if graph.number_of_nodes() <= max_nodes:
        return graph.copy()

    rng = np.random.default_rng(seed)
    degree_sorted = sorted(graph.degree(), key=lambda item: item[1], reverse=True)
    hub_nodes = [node for node, _ in degree_sorted[: min(preserve_hubs, max_nodes)]]
    remaining_slots = max_nodes - len(hub_nodes)
    remaining_nodes = [node for node in graph.nodes if node not in hub_nodes]
    sampled_nodes = hub_nodes
    if remaining_slots > 0:
        sampled_nodes.extend(rng.choice(remaining_nodes, size=remaining_slots, replace=False).tolist())
    return graph.subgraph(sampled_nodes).copy()


def run_persistent_homology(
    adjacency_matrix: np.ndarray,
    corr_distance_matrix: np.ndarray,
    figure_dir: Path,
    prefix: str,
) -> PHArtifacts:
    ripser_diagrams = compute_ripser_diagrams(corr_distance_matrix, maxdim=3)
    save_ripser_diagrams(ripser_diagrams, figure_dir, f"{prefix}_ripser")
    for dimension in range(min(3, len(ripser_diagrams))):
        save_barcode(
            ripser_diagrams,
            dimension,
            figure_dir / f"{prefix}_barcode_dim{dimension}.png",
        )

    rips_complex = gd.RipsComplex(distance_matrix=corr_distance_matrix, max_edge_length=1.0)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=3)
    persistence = simplex_tree.persistence(min_persistence=-1, persistence_dim_max=True)

    plt.figure()
    gd.plot_persistence_diagram(
        persistence,
        max_intervals=4000000,
        title=f"{prefix} GUDHI Rips persistence",
    )
    plt.tight_layout()
    plt.savefig(figure_dir / f"{prefix}_gudhi_rips_diagram.png")
    plt.close()

    landscape_dim1 = _compute_landscape(ripser_diagrams, figure_dir, prefix)
    return PHArtifacts(
        adjacency_matrix=adjacency_matrix,
        correlation_distance_matrix=corr_distance_matrix,
        ripser_diagrams=ripser_diagrams,
        gudhi_persistence=persistence,
        landscape_dim1=landscape_dim1,
        rips_complex_summary={
            "dimension": simplex_tree.dimension(),
            "num_simplices": simplex_tree.num_simplices(),
            "num_vertices": simplex_tree.num_vertices(),
        },
    )


def _compute_landscape(
    ripser_diagrams: list[np.ndarray],
    figure_dir: Path,
    prefix: str,
) -> np.ndarray | None:
    if len(ripser_diagrams) < 2 or len(ripser_diagrams[1]) == 0:
        return None

    landscape = gd.representations.Landscape(num_landscapes=5).fit_transform(
        [ripser_diagrams[1]]
    )
    save_landscape(landscape, figure_dir / f"{prefix}_landscape_dim1.png", "Landscape Dim 1")
    return landscape
