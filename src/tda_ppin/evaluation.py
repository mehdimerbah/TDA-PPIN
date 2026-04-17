from __future__ import annotations

from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd

from .ph import sample_graph_for_ph, summarize_graph_persistent_homology
from .stats import binary_auroc


def _safe_weighted_clustering(graph: nx.Graph) -> dict[str, float]:
    if graph.number_of_edges() == 0:
        return {node: 0.0 for node in graph.nodes}

    max_weight = max(float(data.get("SemSim", 1.0)) for _, _, data in graph.edges(data=True))
    if max_weight <= 0:
        return {node: 0.0 for node in graph.nodes}
    return nx.clustering(graph, weight="SemSim")


def summarize_complex_coverage(
    graph: nx.Graph,
    complexes: list[list[str]],
) -> dict[str, float]:
    graph_nodes = set(graph.nodes())
    complex_sizes = [len(complex_members) for complex_members in complexes]
    covered_complexes = [members for members in complexes if set(members).issubset(graph_nodes)]
    member_overlap = sum(len(set(members) & graph_nodes) for members in complexes)
    total_members = sum(len(members) for members in complexes)

    return {
        "num_complexes": float(len(complexes)),
        "mean_complex_size": float(sum(complex_sizes) / len(complex_sizes)) if complex_sizes else 0.0,
        "fraction_complexes_fully_covered": float(len(covered_complexes) / len(complexes)) if complexes else 0.0,
        "fraction_complex_members_observed": float(member_overlap / total_members) if total_members else 0.0,
    }


def summarize_degree_baseline(graph: nx.Graph, complexes: list[list[str]]) -> pd.DataFrame:
    degrees = dict(graph.degree())
    records = []
    membership = Counter(node for complex_members in complexes for node in complex_members)
    for node, degree in degrees.items():
        records.append(
            {
                "protein": node,
                "degree": degree,
                "complex_membership_count": membership.get(node, 0),
            }
        )
    return pd.DataFrame.from_records(records).sort_values(
        by=["complex_membership_count", "degree"],
        ascending=False,
    )


def build_membership_lookup(complexes: list[list[str]]) -> Counter:
    return Counter(node for complex_members in complexes for node in complex_members)


def build_protein_feature_table(graph: nx.Graph, complexes: list[list[str]]) -> pd.DataFrame:
    membership = build_membership_lookup(complexes)
    rows: list[dict[str, float | int | str]] = []
    clustering = _safe_weighted_clustering(graph)
    for node in graph.nodes:
        ego_graph = nx.ego_graph(graph, node)
        strength = sum(edge_data.get("SemSim", 1.0) for _, _, edge_data in graph.edges(node, data=True))
        rows.append(
            {
                "protein": node,
                "degree": int(graph.degree(node)),
                "strength": float(strength),
                "clustering": float(clustering.get(node, 0.0)),
                "ego_nodes": int(ego_graph.number_of_nodes()),
                "ego_edges": int(ego_graph.number_of_edges()),
                "local_density": float(nx.density(ego_graph)) if ego_graph.number_of_nodes() > 1 else 0.0,
                "complex_membership_count": int(membership.get(node, 0)),
                "protein_in_any_complex": int(membership.get(node, 0) > 0),
            }
        )
    return pd.DataFrame.from_records(rows).sort_values(by="protein").reset_index(drop=True)


def build_bounded_ego_subgraph(
    graph: nx.Graph,
    protein: str,
    *,
    max_nodes: int,
) -> nx.Graph:
    ego_graph = nx.ego_graph(graph, protein)
    if ego_graph.number_of_nodes() <= max_nodes:
        return ego_graph.copy()

    weighted_neighbors = sorted(
        (
            (neighbor, float(graph[protein][neighbor].get("SemSim", 0.0)))
            for neighbor in graph.neighbors(protein)
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    selected_neighbors = [neighbor for neighbor, _ in weighted_neighbors[: max_nodes - 1]]
    selected_nodes = [protein, *selected_neighbors]
    return graph.subgraph(selected_nodes).copy()


def build_subgraph_feature_record(
    graph: nx.Graph,
    members: list[str],
    *,
    subgraph_id: str,
    source: str,
    label: int,
    ph_node_cap: int | None = None,
    ph_sampling_seed: int = 7,
    filtration: str = "correlation_distance",
    ph_feature_prefix: str | None = None,
) -> dict[str, float | int | str] | None:
    unique_members = [node for node in dict.fromkeys(members) if node in graph]
    if len(unique_members) == 0:
        return None

    subgraph = graph.subgraph(unique_members).copy()
    if subgraph.number_of_nodes() == 0:
        return None

    degrees = dict(subgraph.degree())
    strength_values = [
        sum(edge_data.get("SemSim", 1.0) for _, _, edge_data in subgraph.edges(node, data=True))
        for node in subgraph.nodes
    ]
    ph_graph = (
        sample_graph_for_ph(subgraph, max_nodes=ph_node_cap, seed=ph_sampling_seed)
        if ph_node_cap is not None and subgraph.number_of_nodes() > ph_node_cap
        else subgraph
    )
    ph_summary = summarize_graph_persistent_homology(ph_graph, filtration=filtration)
    if ph_feature_prefix is not None:
        ph_summary = {
            f"{ph_feature_prefix}_{key}" if key.startswith("ph_") else key: value
            for key, value in ph_summary.items()
        }

    record: dict[str, float | int | str] = {
        "subgraph_id": subgraph_id,
        "source": source,
        "label": label,
        "num_nodes": int(subgraph.number_of_nodes()),
        "num_edges": int(subgraph.number_of_edges()),
        "density": float(nx.density(subgraph)) if subgraph.number_of_nodes() > 1 else 0.0,
        "mean_degree": float(np.mean(list(degrees.values()))) if degrees else 0.0,
        "mean_strength": float(np.mean(strength_values)) if strength_values else 0.0,
        "mean_clustering": float(np.mean(list(_safe_weighted_clustering(subgraph).values()))),
        "num_connected_components": float(nx.number_connected_components(subgraph)),
        "ph_num_nodes_used": int(ph_graph.number_of_nodes()),
    }
    record.update(ph_summary)
    return record


def build_complex_subgraph_feature_table(
    graph: nx.Graph,
    complexes: list[list[str]],
    *,
    source_label: str,
    min_complex_size: int = 3,
    ph_node_cap: int | None = None,
    ph_sampling_seed: int = 7,
    filtration: str = "correlation_distance",
    ph_feature_prefix: str | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for index, complex_members in enumerate(complexes):
        unique_members = [node for node in dict.fromkeys(complex_members) if node in graph]
        if len(unique_members) < min_complex_size:
            continue
        record = build_subgraph_feature_record(
            graph,
            unique_members,
            subgraph_id=f"{source_label}_{index}",
            source=source_label,
            label=1 if source_label == "real_complex" else 0,
            ph_node_cap=ph_node_cap,
            ph_sampling_seed=ph_sampling_seed + index,
            filtration=filtration,
            ph_feature_prefix=ph_feature_prefix,
        )
        if record is not None:
            rows.append(record)
    return pd.DataFrame.from_records(rows)


def univariate_signal_summary(
    feature_table: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
) -> pd.DataFrame:
    labels = feature_table[label_column].to_numpy()
    positive_mask = labels == 1
    negative_mask = labels == 0
    rows = []
    for feature_column in feature_columns:
        values = feature_table[feature_column].fillna(0.0).to_numpy(dtype=float)
        auroc = binary_auroc(labels, values)
        rows.append(
            {
                "feature": feature_column,
                "positive_mean": float(np.mean(values[positive_mask])) if positive_mask.any() else 0.0,
                "negative_mean": float(np.mean(values[negative_mask])) if negative_mask.any() else 0.0,
                "mean_difference": float(
                    (np.mean(values[positive_mask]) if positive_mask.any() else 0.0)
                    - (np.mean(values[negative_mask]) if negative_mask.any() else 0.0)
                ),
                "auroc": float(auroc),
                "rank_biserial": float(2 * auroc - 1),
            }
        )
    return pd.DataFrame.from_records(rows).sort_values(by="auroc", ascending=False).reset_index(drop=True)
