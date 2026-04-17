from __future__ import annotations

from dataclasses import asdict, dataclass
from random import Random
from statistics import mean

import networkx as nx

from .evaluation import summarize_degree_baseline


@dataclass(frozen=True)
class SyntheticConfig:
    num_proteins: int = 100
    edge_probability: float = 0.2
    random_seed: int = 7
    complex_sizes: tuple[int, int, int] = (10, 7, 5)


def build_synthetic_graph(config: SyntheticConfig) -> tuple[nx.Graph, list[list[str]]]:
    rng = Random(config.random_seed)
    graph = nx.Graph()
    proteins = [f"Protein_{index}" for index in range(1, config.num_proteins + 1)]
    graph.add_nodes_from(proteins)

    for index, protein_a in enumerate(proteins):
        for protein_b in proteins[index + 1 :]:
            if rng.random() < config.edge_probability:
                graph.add_edge(protein_a, protein_b, SemSim=rng.uniform(0.0, 0.7))

    complexes = [rng.sample(proteins, size) for size in config.complex_sizes]
    for complex_members in complexes:
        for index, protein_a in enumerate(complex_members):
            for protein_b in complex_members[index + 1 :]:
                if graph.has_edge(protein_a, protein_b):
                    weight = graph[protein_a][protein_b]["SemSim"]
                    graph[protein_a][protein_b]["SemSim"] = 0.95 if weight >= 0.5 else weight * 2
    return graph, complexes


def run_synthetic_sanity_workflow(config: SyntheticConfig) -> dict[str, object]:
    graph, complexes = build_synthetic_graph(config)
    degree_summary = summarize_degree_baseline(graph, complexes)
    return {
        "config": asdict(config),
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "mean_degree": mean(dict(graph.degree()).values()) if graph.number_of_nodes() else 0.0,
        "complexes": complexes,
        "degree_baseline_preview": degree_summary.head(10).to_dict(orient="records"),
    }
