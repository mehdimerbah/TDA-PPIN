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
