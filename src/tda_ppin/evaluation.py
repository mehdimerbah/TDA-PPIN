from __future__ import annotations

from collections import Counter

import networkx as nx
import pandas as pd


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
    membership = Counter(node for complex_members in complexes for node in complex_members)
    records = []
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
