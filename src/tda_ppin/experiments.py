from __future__ import annotations

from dataclasses import asdict, dataclass

import networkx as nx
import numpy as np
import pandas as pd

from .evaluation import (
    build_bounded_ego_subgraph,
    build_complex_subgraph_feature_table,
    build_local_neighborhood_subgraph,
    build_membership_lookup,
    build_protein_feature_table,
    build_subgraph_feature_record,
    univariate_signal_summary,
)
from .ph import sample_graph_for_ph, summarize_graph_persistent_homology
from .stats import cosine_similarity, repeated_group_split_logistic_metrics, repeated_split_logistic_metrics


@dataclass(frozen=True)
class DecisionStageConfig:
    random_seed: int = 7
    repeats: int = 3
    test_fraction: float = 0.25
    matched_random_per_complex: int = 3
    max_random_attempts: int = 100
    min_complex_size: int = 3
    max_complex_size: int = 40
    max_real_complexes: int = 250
    node_subsample_fraction: float = 0.9
    edge_subsample_fraction: float = 0.9
    hub_removal_fraction: float = 0.01
    weight_jitter_std: float = 0.05
    density_tolerance: float = 0.15
    global_ph_node_cap: int = 250
    subgraph_ph_node_cap: int = 120
    random_walk_restart_prob: float = 0.15


@dataclass(frozen=True)
class ExtensionExplorationConfig:
    random_seed: int = 7
    local_positive_proteins: int = 100
    local_negative_proteins: int = 100
    local_neighborhood_node_cap: int = 30
    local_ph_maxdim: int = 2
    local_repeats: int = 3
    local_test_fraction: float = 0.25
    filtrations: tuple[str, ...] = (
        "correlation_distance",
        "hop_distance",
        "weighted_shortest_path",
    )
    filtration_repeats: int = 3
    filtration_test_fraction: float = 0.25
    filtration_max_real_complexes: int = 40
    filtration_matched_random_per_complex: int = 1
    filtration_max_random_attempts: int = 80
    filtration_min_complex_size: int = 3
    filtration_max_complex_size: int = 25
    filtration_subgraph_ph_node_cap: int = 60
    filtration_random_walk_restart_prob: float = 0.15


@dataclass(frozen=True)
class LocalProtocolSweepConfig:
    random_seed: int = 7
    local_positive_proteins: int = 150
    local_negative_proteins: int = 150
    protocols: tuple[str, ...] = (
        "top_weighted_neighbors",
        "ego_radius_1",
        "ego_radius_2",
        "seed_expansion",
    )
    neighborhood_node_caps: tuple[int, ...] = (10, 20, 30, 50)
    filtration: str = "weighted_shortest_path"
    ph_maxdim: int = 2
    repeats: int = 4
    test_fraction: float = 0.25


def run_data_sanity_experiment(graph: nx.Graph, complexes: list[list[str]]) -> dict[str, float]:
    membership = build_membership_lookup(complexes)
    covered_complexes = [
        complex_members for complex_members in complexes if set(complex_members).issubset(graph.nodes)
    ]
    degrees = dict(graph.degree())
    in_complex_degrees = [degrees[node] for node in graph.nodes if membership.get(node, 0) > 0]
    out_complex_degrees = [degrees[node] for node in graph.nodes if membership.get(node, 0) == 0]
    complex_sizes = [len(complex_members) for complex_members in complexes]

    return {
        "num_graph_nodes": float(graph.number_of_nodes()),
        "num_graph_edges": float(graph.number_of_edges()),
        "num_complexes": float(len(complexes)),
        "num_fully_covered_complexes": float(len(covered_complexes)),
        "fraction_fully_covered_complexes": (
            float(len(covered_complexes) / len(complexes)) if complexes else 0.0
        ),
        "mean_complex_size": float(np.mean(complex_sizes)) if complex_sizes else 0.0,
        "median_complex_size": float(np.median(complex_sizes)) if complex_sizes else 0.0,
        "mean_degree_in_complex": float(np.mean(in_complex_degrees)) if in_complex_degrees else 0.0,
        "mean_degree_outside_complex": float(np.mean(out_complex_degrees)) if out_complex_degrees else 0.0,
    }


def run_protein_baseline_experiment(
    graph: nx.Graph,
    complexes: list[list[str]],
    config: DecisionStageConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    feature_table = build_protein_feature_table(graph, complexes)
    feature_columns = [
        "degree",
        "strength",
        "clustering",
        "ego_nodes",
        "ego_edges",
        "local_density",
    ]
    univariate = univariate_signal_summary(feature_table, feature_columns, "protein_in_any_complex")
    model_metrics = repeated_split_logistic_metrics(
        feature_table[feature_columns].fillna(0.0).to_numpy(),
        feature_table["protein_in_any_complex"].to_numpy(),
        repeats=config.repeats,
        test_fraction=config.test_fraction,
        seed=config.random_seed,
    )
    report = {
        "feature_columns": feature_columns,
        "num_positive_proteins": int(feature_table["protein_in_any_complex"].sum()),
        "num_negative_proteins": int(len(feature_table) - feature_table["protein_in_any_complex"].sum()),
        "baseline_model": model_metrics,
    }
    return feature_table, univariate, report


def run_global_ph_characterization_experiment(
    graph: nx.Graph,
    config: DecisionStageConfig,
) -> tuple[pd.DataFrame, dict[str, object]]:
    sampled_original_graph = sample_graph_for_ph(
        graph,
        max_nodes=config.global_ph_node_cap,
        seed=config.random_seed,
    )
    original = summarize_graph_persistent_homology(sampled_original_graph)
    summary_keys = sorted(key for key in original if key.startswith("ph_"))
    reference_vector = np.array([original[key] for key in summary_keys], dtype=float)
    perturbation_rows = [
        {
            "perturbation": "original",
            **original,
            "cosine_to_original": 1.0,
            "sampled_nodes": int(sampled_original_graph.number_of_nodes()),
        }
    ]

    perturbations = [
        ("weight_jitter", _apply_weight_jitter(graph, config)),
        ("edge_subsample", _subsample_edges(graph, config)),
        ("node_subsample", _subsample_nodes(graph, config)),
        ("hub_removal", _remove_hubs(graph, config)),
    ]
    for perturbation_name, perturbed_graph in perturbations:
        sampled_graph = sample_graph_for_ph(
            perturbed_graph,
            max_nodes=config.global_ph_node_cap,
            seed=config.random_seed,
        )
        summary = summarize_graph_persistent_homology(sampled_graph)
        candidate_vector = np.array([summary.get(key, 0.0) for key in summary_keys], dtype=float)
        perturbation_rows.append(
            {
                "perturbation": perturbation_name,
                **summary,
                "cosine_to_original": cosine_similarity(reference_vector, candidate_vector),
                "sampled_nodes": int(sampled_graph.number_of_nodes()),
            }
        )

    table = pd.DataFrame.from_records(perturbation_rows)
    report = {
        "summary_keys": summary_keys,
        "global_ph_node_cap": config.global_ph_node_cap,
        "mean_cosine_to_original": float(table.loc[table["perturbation"] != "original", "cosine_to_original"].mean()),
        "min_cosine_to_original": float(table.loc[table["perturbation"] != "original", "cosine_to_original"].min()),
    }
    return table, report


def run_complex_vs_random_experiment(
    graph: nx.Graph,
    complexes: list[list[str]],
    config: DecisionStageConfig,
) -> tuple[pd.DataFrame, dict[str, object]]:
    random_table = _matched_random_subgraph_table(graph, complexes, config)
    eligible_complexes = _select_eligible_complexes(graph, complexes, config)
    real_table = build_complex_subgraph_feature_table(
        graph,
        eligible_complexes,
        source_label="real_complex",
        min_complex_size=config.min_complex_size,
        ph_node_cap=config.subgraph_ph_node_cap,
        ph_sampling_seed=config.random_seed,
    )
    real_table["complex_group"] = [f"group_{index}" for index in range(len(real_table))]
    real_table["size_bin"] = real_table["num_nodes"].apply(_size_bin)
    combined_table = pd.concat([real_table, random_table], ignore_index=True)
    group_values = combined_table["complex_group"].to_numpy()

    baseline_columns = [
        "num_nodes",
        "num_edges",
        "density",
        "mean_degree",
        "mean_strength",
        "mean_clustering",
        "num_connected_components",
    ]
    ph_columns = sorted(column for column in combined_table.columns if column.startswith("ph_"))
    combined_columns = baseline_columns + ph_columns

    report = {
        "num_real_subgraphs": int((combined_table["label"] == 1).sum()),
        "num_random_subgraphs": int((combined_table["label"] == 0).sum()),
        "max_real_complexes": config.max_real_complexes,
        "max_complex_size": config.max_complex_size,
        "baseline_model": repeated_group_split_logistic_metrics(
            combined_table[baseline_columns].fillna(0.0).to_numpy(),
            combined_table["label"].to_numpy(),
            group_values,
            repeats=config.repeats,
            test_fraction=config.test_fraction,
            seed=config.random_seed,
        ),
        "ph_model": repeated_group_split_logistic_metrics(
            combined_table[ph_columns].fillna(0.0).to_numpy(),
            combined_table["label"].to_numpy(),
            group_values,
            repeats=config.repeats,
            test_fraction=config.test_fraction,
            seed=config.random_seed + 1,
        ),
        "combined_model": repeated_group_split_logistic_metrics(
            combined_table[combined_columns].fillna(0.0).to_numpy(),
            combined_table["label"].to_numpy(),
            group_values,
            repeats=config.repeats,
            test_fraction=config.test_fraction,
            seed=config.random_seed + 2,
        ),
        "baseline_columns": baseline_columns,
        "ph_columns": ph_columns,
    }
    return combined_table, report


def run_null_control_experiment(
    protein_feature_table: pd.DataFrame,
    complex_feature_table: pd.DataFrame,
    config: DecisionStageConfig,
) -> dict[str, object]:
    rng = np.random.default_rng(config.random_seed)
    protein_labels = protein_feature_table["protein_in_any_complex"].to_numpy().copy()
    rng.shuffle(protein_labels)
    protein_null = repeated_split_logistic_metrics(
        protein_feature_table[
            ["degree", "strength", "clustering", "ego_nodes", "ego_edges", "local_density"]
        ].fillna(0.0).to_numpy(),
        protein_labels,
        repeats=config.repeats,
        test_fraction=config.test_fraction,
        seed=config.random_seed + 3,
    )

    complex_labels = _shuffle_labels_within_bins(
        complex_feature_table["label"].to_numpy(),
        complex_feature_table["size_bin"].to_numpy(),
        rng,
    )
    feature_columns = [
        column
        for column in complex_feature_table.columns
        if column not in {"subgraph_id", "source", "label", "complex_group", "size_bin"}
    ]
    complex_null = repeated_group_split_logistic_metrics(
        complex_feature_table[feature_columns].fillna(0.0).to_numpy(),
        complex_labels,
        complex_feature_table["complex_group"].to_numpy(),
        repeats=config.repeats,
        test_fraction=config.test_fraction,
        seed=config.random_seed + 4,
    )
    return {
        "protein_label_shuffle": protein_null,
        "complex_label_shuffle": complex_null,
    }


def synthesize_decision(
    data_sanity: dict[str, float],
    global_ph_report: dict[str, object],
    complex_report: dict[str, object],
    null_report: dict[str, object],
) -> dict[str, object]:
    coverage_ok = data_sanity["fraction_fully_covered_complexes"] >= 0.25
    stability = float(global_ph_report["mean_cosine_to_original"])
    baseline_auc = float(complex_report["baseline_model"]["mean_auroc"])
    ph_auc = float(complex_report["ph_model"]["mean_auroc"])
    combined_auc = float(complex_report["combined_model"]["mean_auroc"])
    null_auc = float(null_report["complex_label_shuffle"]["mean_auroc"])

    if coverage_ok and stability >= 0.90 and combined_auc >= baseline_auc + 0.05 and ph_auc >= baseline_auc - 0.02 and abs(null_auc - 0.5) <= 0.08:
        verdict = "go"
    elif coverage_ok and stability >= 0.80 and combined_auc >= baseline_auc + 0.02 and abs(null_auc - 0.5) <= 0.12:
        verdict = "conditional_go"
    else:
        verdict = "no_go"

    return {
        "verdict": verdict,
        "coverage_ok": coverage_ok,
        "mean_cosine_to_original": stability,
        "baseline_complex_auroc": baseline_auc,
        "ph_complex_auroc": ph_auc,
        "combined_complex_auroc": combined_auc,
        "null_complex_auroc": null_auc,
    }


def run_local_neighborhood_ph_experiment(
    graph: nx.Graph,
    protein_feature_table: pd.DataFrame,
    config: ExtensionExplorationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    sampled_feature_table = _sample_protein_panel(protein_feature_table, config)
    baseline_columns = [
        "degree",
        "strength",
        "clustering",
        "ego_nodes",
        "ego_edges",
        "local_density",
    ]
    rows: list[dict[str, float | int | str]] = []
    for index, row in sampled_feature_table.reset_index(drop=True).iterrows():
        protein = str(row["protein"])
        neighborhood = build_bounded_ego_subgraph(
            graph,
            protein,
            max_nodes=config.local_neighborhood_node_cap,
        )
        record = row.to_dict()
        record["local_nodes_used"] = int(neighborhood.number_of_nodes())
        record["local_edges_used"] = int(neighborhood.number_of_edges())
        for filtration in config.filtrations:
            ph_summary = summarize_graph_persistent_homology(
                neighborhood,
                maxdim=config.local_ph_maxdim,
                filtration=filtration,
            )
            for key, value in ph_summary.items():
                if key.startswith("ph_"):
                    record[f"local_{filtration}_{key}"] = value
        rows.append(record)

    local_feature_table = pd.DataFrame.from_records(rows)
    labels = local_feature_table["protein_in_any_complex"].to_numpy()
    metric_rows = []

    baseline_metrics = repeated_split_logistic_metrics(
        local_feature_table[baseline_columns].fillna(0.0).to_numpy(),
        labels,
        repeats=config.local_repeats,
        test_fraction=config.local_test_fraction,
        seed=config.random_seed,
    )
    metric_rows.append(
        {
            "experiment": "local_neighborhood",
            "filtration": "baseline_only",
            "model": "baseline_only",
            "mean_auroc": float(baseline_metrics["mean_auroc"]),
            "mean_average_precision": float(baseline_metrics["mean_average_precision"]),
        }
    )

    report = {
        "num_sampled_proteins": int(len(local_feature_table)),
        "num_positive_proteins": int(local_feature_table["protein_in_any_complex"].sum()),
        "num_negative_proteins": int(len(local_feature_table) - local_feature_table["protein_in_any_complex"].sum()),
        "baseline_model": baseline_metrics,
        "filtrations": {},
    }

    for filtration in config.filtrations:
        ph_columns = sorted(
            column for column in local_feature_table.columns if column.startswith(f"local_{filtration}_ph_")
        )
        ph_only_metrics = repeated_split_logistic_metrics(
            local_feature_table[ph_columns].fillna(0.0).to_numpy(),
            labels,
            repeats=config.local_repeats,
            test_fraction=config.local_test_fraction,
            seed=config.random_seed + len(metric_rows),
        )
        combined_metrics = repeated_split_logistic_metrics(
            local_feature_table[baseline_columns + ph_columns].fillna(0.0).to_numpy(),
            labels,
            repeats=config.local_repeats,
            test_fraction=config.local_test_fraction,
            seed=config.random_seed + len(metric_rows) + 50,
        )
        shuffled_labels = labels.copy()
        np.random.default_rng(config.random_seed + len(metric_rows) + 100).shuffle(shuffled_labels)
        null_metrics = repeated_split_logistic_metrics(
            local_feature_table[baseline_columns + ph_columns].fillna(0.0).to_numpy(),
            shuffled_labels,
            repeats=config.local_repeats,
            test_fraction=config.local_test_fraction,
            seed=config.random_seed + len(metric_rows) + 150,
        )

        report["filtrations"][filtration] = {
            "ph_only": ph_only_metrics,
            "baseline_plus_ph": combined_metrics,
            "label_shuffle_null": null_metrics,
            "num_ph_features": len(ph_columns),
        }
        metric_rows.append(
            {
                "experiment": "local_neighborhood",
                "filtration": filtration,
                "model": "ph_only",
                "mean_auroc": float(ph_only_metrics["mean_auroc"]),
                "mean_average_precision": float(ph_only_metrics["mean_average_precision"]),
            }
        )
        metric_rows.append(
            {
                "experiment": "local_neighborhood",
                "filtration": filtration,
                "model": "baseline_plus_ph",
                "mean_auroc": float(combined_metrics["mean_auroc"]),
                "mean_average_precision": float(combined_metrics["mean_average_precision"]),
            }
        )
        metric_rows.append(
            {
                "experiment": "local_neighborhood",
                "filtration": filtration,
                "model": "label_shuffle_null",
                "mean_auroc": float(null_metrics["mean_auroc"]),
                "mean_average_precision": float(null_metrics["mean_average_precision"]),
            }
        )

    metric_table = pd.DataFrame.from_records(metric_rows)
    return local_feature_table, metric_table, report


def run_filtration_comparison_experiment(
    graph: nx.Graph,
    complexes: list[list[str]],
    config: ExtensionExplorationConfig,
) -> tuple[pd.DataFrame, dict[str, object]]:
    decision_like_config = DecisionStageConfig(
        random_seed=config.random_seed,
        repeats=config.filtration_repeats,
        test_fraction=config.filtration_test_fraction,
        matched_random_per_complex=config.filtration_matched_random_per_complex,
        max_random_attempts=config.filtration_max_random_attempts,
        min_complex_size=config.filtration_min_complex_size,
        max_complex_size=config.filtration_max_complex_size,
        max_real_complexes=config.filtration_max_real_complexes,
        subgraph_ph_node_cap=config.filtration_subgraph_ph_node_cap,
        random_walk_restart_prob=config.filtration_random_walk_restart_prob,
    )
    instance_specs = _build_complex_instance_specs(graph, complexes, decision_like_config)
    baseline_columns = [
        "num_nodes",
        "num_edges",
        "density",
        "mean_degree",
        "mean_strength",
        "mean_clustering",
        "num_connected_components",
    ]
    metric_rows = []
    report = {
        "num_instance_rows": int(len(instance_specs)),
        "num_real_complexes": int(sum(1 for spec in instance_specs if spec["label"] == 1)),
        "num_random_controls": int(sum(1 for spec in instance_specs if spec["label"] == 0)),
        "filtrations": {},
    }

    for filtration in config.filtrations:
        table = _build_complex_instance_feature_table(
            graph,
            instance_specs,
            decision_like_config,
            filtration=filtration,
        )
        groups = table["complex_group"].to_numpy()
        labels = table["label"].to_numpy()
        ph_columns = sorted(column for column in table.columns if column.startswith(f"{filtration}_ph_"))
        ph_metrics = repeated_group_split_logistic_metrics(
            table[ph_columns].fillna(0.0).to_numpy(),
            labels,
            groups,
            repeats=config.filtration_repeats,
            test_fraction=config.filtration_test_fraction,
            seed=config.random_seed + len(metric_rows),
        )
        combined_metrics = repeated_group_split_logistic_metrics(
            table[baseline_columns + ph_columns].fillna(0.0).to_numpy(),
            labels,
            groups,
            repeats=config.filtration_repeats,
            test_fraction=config.filtration_test_fraction,
            seed=config.random_seed + len(metric_rows) + 30,
        )
        null_labels = _shuffle_labels_within_bins(labels, table["size_bin"].to_numpy(), np.random.default_rng(config.random_seed + len(metric_rows) + 60))
        null_metrics = repeated_group_split_logistic_metrics(
            table[baseline_columns + ph_columns].fillna(0.0).to_numpy(),
            null_labels,
            groups,
            repeats=config.filtration_repeats,
            test_fraction=config.filtration_test_fraction,
            seed=config.random_seed + len(metric_rows) + 90,
        )

        metric_rows.append(
            {
                "experiment": "filtration_comparison",
                "filtration": filtration,
                "model": "ph_only",
                "mean_auroc": float(ph_metrics["mean_auroc"]),
                "mean_average_precision": float(ph_metrics["mean_average_precision"]),
            }
        )
        metric_rows.append(
            {
                "experiment": "filtration_comparison",
                "filtration": filtration,
                "model": "baseline_plus_ph",
                "mean_auroc": float(combined_metrics["mean_auroc"]),
                "mean_average_precision": float(combined_metrics["mean_average_precision"]),
            }
        )
        metric_rows.append(
            {
                "experiment": "filtration_comparison",
                "filtration": filtration,
                "model": "label_shuffle_null",
                "mean_auroc": float(null_metrics["mean_auroc"]),
                "mean_average_precision": float(null_metrics["mean_average_precision"]),
            }
        )
        report["filtrations"][filtration] = {
            "ph_only": ph_metrics,
            "baseline_plus_ph": combined_metrics,
            "label_shuffle_null": null_metrics,
            "num_ph_features": len(ph_columns),
        }

    metric_table = pd.DataFrame.from_records(metric_rows)
    return metric_table, report


def run_local_protocol_sweep_experiment(
    graph: nx.Graph,
    protein_feature_table: pd.DataFrame,
    config: LocalProtocolSweepConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    sampled_feature_table = _sample_protein_panel(protein_feature_table, config)
    baseline_columns = [
        "degree",
        "strength",
        "clustering",
        "ego_nodes",
        "ego_edges",
        "local_density",
    ]
    baseline_metrics = repeated_split_logistic_metrics(
        sampled_feature_table[baseline_columns].fillna(0.0).to_numpy(),
        sampled_feature_table["protein_in_any_complex"].to_numpy(),
        repeats=config.repeats,
        test_fraction=config.test_fraction,
        seed=config.random_seed,
    )

    feature_tables: list[pd.DataFrame] = []
    metric_rows: list[dict[str, float | int | str]] = [
        {
            "experiment": "local_protocol_sweep",
            "protocol": "baseline_only",
            "max_nodes": 0,
            "filtration": config.filtration,
            "model": "baseline_only",
            "mean_auroc": float(baseline_metrics["mean_auroc"]),
            "mean_average_precision": float(baseline_metrics["mean_average_precision"]),
            "auroc_lift_vs_baseline": 0.0,
        }
    ]
    report: dict[str, object] = {
        "num_sampled_proteins": int(len(sampled_feature_table)),
        "num_positive_proteins": int(sampled_feature_table["protein_in_any_complex"].sum()),
        "num_negative_proteins": int(len(sampled_feature_table) - sampled_feature_table["protein_in_any_complex"].sum()),
        "baseline_model": baseline_metrics,
        "filtration": config.filtration,
        "protocols": {},
    }
    best_setting: dict[str, object] | None = None

    for protocol in config.protocols:
        for max_nodes in config.neighborhood_node_caps:
            rows: list[dict[str, float | int | str]] = []
            for index, row in sampled_feature_table.reset_index(drop=True).iterrows():
                protein = str(row["protein"])
                neighborhood = build_local_neighborhood_subgraph(
                    graph,
                    protein,
                    method=protocol,
                    max_nodes=max_nodes,
                )
                record = row.to_dict()
                record["protocol"] = protocol
                record["max_nodes"] = int(max_nodes)
                record["filtration"] = config.filtration
                record["local_nodes_used"] = int(neighborhood.number_of_nodes())
                record["local_edges_used"] = int(neighborhood.number_of_edges())
                ph_summary = summarize_graph_persistent_homology(
                    neighborhood,
                    maxdim=config.ph_maxdim,
                    filtration=config.filtration,
                )
                for key, value in ph_summary.items():
                    if key.startswith("ph_"):
                        record[f"local_{key}"] = value
                rows.append(record)

            setting_table = pd.DataFrame.from_records(rows)
            feature_tables.append(setting_table)
            ph_columns = sorted(column for column in setting_table.columns if column.startswith("local_ph_"))
            labels = setting_table["protein_in_any_complex"].to_numpy()
            ph_metrics = repeated_split_logistic_metrics(
                setting_table[ph_columns].fillna(0.0).to_numpy(),
                labels,
                repeats=config.repeats,
                test_fraction=config.test_fraction,
                seed=config.random_seed + len(metric_rows),
            )
            combined_metrics = repeated_split_logistic_metrics(
                setting_table[baseline_columns + ph_columns].fillna(0.0).to_numpy(),
                labels,
                repeats=config.repeats,
                test_fraction=config.test_fraction,
                seed=config.random_seed + len(metric_rows) + 40,
            )
            shuffled_labels = labels.copy()
            np.random.default_rng(config.random_seed + len(metric_rows) + 80).shuffle(shuffled_labels)
            null_metrics = repeated_split_logistic_metrics(
                setting_table[baseline_columns + ph_columns].fillna(0.0).to_numpy(),
                shuffled_labels,
                repeats=config.repeats,
                test_fraction=config.test_fraction,
                seed=config.random_seed + len(metric_rows) + 120,
            )

            combined_lift = float(combined_metrics["mean_auroc"]) - float(baseline_metrics["mean_auroc"])
            key = f"{protocol}__nodes_{max_nodes}"
            report["protocols"][key] = {
                "protocol": protocol,
                "max_nodes": int(max_nodes),
                "ph_only": ph_metrics,
                "baseline_plus_ph": combined_metrics,
                "label_shuffle_null": null_metrics,
                "num_ph_features": len(ph_columns),
                "auroc_lift_vs_baseline": combined_lift,
            }
            metric_rows.append(
                {
                    "experiment": "local_protocol_sweep",
                    "protocol": protocol,
                    "max_nodes": int(max_nodes),
                    "filtration": config.filtration,
                    "model": "ph_only",
                    "mean_auroc": float(ph_metrics["mean_auroc"]),
                    "mean_average_precision": float(ph_metrics["mean_average_precision"]),
                    "auroc_lift_vs_baseline": float(ph_metrics["mean_auroc"]) - float(baseline_metrics["mean_auroc"]),
                }
            )
            metric_rows.append(
                {
                    "experiment": "local_protocol_sweep",
                    "protocol": protocol,
                    "max_nodes": int(max_nodes),
                    "filtration": config.filtration,
                    "model": "baseline_plus_ph",
                    "mean_auroc": float(combined_metrics["mean_auroc"]),
                    "mean_average_precision": float(combined_metrics["mean_average_precision"]),
                    "auroc_lift_vs_baseline": combined_lift,
                }
            )
            metric_rows.append(
                {
                    "experiment": "local_protocol_sweep",
                    "protocol": protocol,
                    "max_nodes": int(max_nodes),
                    "filtration": config.filtration,
                    "model": "label_shuffle_null",
                    "mean_auroc": float(null_metrics["mean_auroc"]),
                    "mean_average_precision": float(null_metrics["mean_average_precision"]),
                    "auroc_lift_vs_baseline": float(null_metrics["mean_auroc"]) - float(baseline_metrics["mean_auroc"]),
                }
            )

            if best_setting is None or combined_lift > float(best_setting["auroc_lift_vs_baseline"]):
                best_setting = {
                    "protocol": protocol,
                    "max_nodes": int(max_nodes),
                    "filtration": config.filtration,
                    "auroc_lift_vs_baseline": combined_lift,
                    "combined_mean_auroc": float(combined_metrics["mean_auroc"]),
                    "null_mean_auroc": float(null_metrics["mean_auroc"]),
                }

    report["best_setting"] = best_setting
    return (
        pd.concat(feature_tables, ignore_index=True),
        pd.DataFrame.from_records(metric_rows),
        report,
    )


def _matched_random_subgraph_table(
    graph: nx.Graph,
    complexes: list[list[str]],
    config: DecisionStageConfig,
) -> pd.DataFrame:
    rng = np.random.default_rng(config.random_seed)
    degree_lookup = dict(graph.degree())
    graph_nodes = np.array(list(graph.nodes()))
    records: list[dict[str, float | int | str]] = []

    eligible_complexes = [
        complex_members
        for complex_members in complexes
        if (
            len(complex_members) >= config.min_complex_size
            and len(complex_members) <= config.max_complex_size
            and set(complex_members).issubset(graph.nodes)
        )
    ]
    eligible_complexes = eligible_complexes[: config.max_real_complexes]
    for index, complex_members in enumerate(eligible_complexes):
        real_subgraph = graph.subgraph(complex_members).copy()
        target_density = nx.density(real_subgraph) if real_subgraph.number_of_nodes() > 1 else 0.0
        target_mean_degree = float(np.mean([degree_lookup[node] for node in complex_members]))
        real_strengths = [
            sum(edge_data.get("SemSim", 1.0) for _, _, edge_data in real_subgraph.edges(node, data=True))
            for node in real_subgraph.nodes
        ]
        target_mean_strength = float(np.mean(real_strengths)) if real_strengths else 0.0
        for control_index in range(config.matched_random_per_complex):
            sampled_nodes, match_score = _sample_matched_random_nodes(
                graph,
                len(complex_members),
                target_density,
                target_mean_degree,
                target_mean_strength,
                set(complex_members),
                config,
                rng,
            )
            sampled_subgraph = graph.subgraph(sampled_nodes).copy()
            record = build_complex_subgraph_feature_table(
                sampled_subgraph,
                [list(sampled_nodes)],
                source_label="matched_random",
                min_complex_size=config.min_complex_size,
                ph_node_cap=config.subgraph_ph_node_cap,
                ph_sampling_seed=config.random_seed + 1000,
            ).iloc[0].to_dict()
            record["subgraph_id"] = f"matched_random_{index}_{control_index}"
            record["source"] = "matched_random"
            record["label"] = 0
            record["complex_group"] = f"group_{index}"
            record["size_bin"] = _size_bin(len(sampled_nodes))
            record["match_score"] = float(match_score)
            record["target_density"] = float(target_density)
            record["target_mean_degree"] = float(target_mean_degree)
            record["target_mean_strength"] = float(target_mean_strength)
            records.append(record)
    return pd.DataFrame.from_records(records)


def _build_complex_instance_specs(
    graph: nx.Graph,
    complexes: list[list[str]],
    config: DecisionStageConfig,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(config.random_seed)
    degree_lookup = dict(graph.degree())
    instance_specs: list[dict[str, object]] = []
    eligible_complexes = _select_eligible_complexes(graph, complexes, config)

    for index, complex_members in enumerate(eligible_complexes):
        unique_members = [node for node in dict.fromkeys(complex_members) if node in graph]
        if len(unique_members) < config.min_complex_size:
            continue

        real_subgraph = graph.subgraph(unique_members).copy()
        target_density = nx.density(real_subgraph) if real_subgraph.number_of_nodes() > 1 else 0.0
        target_mean_degree = float(np.mean([degree_lookup[node] for node in unique_members]))
        real_strengths = [
            sum(edge_data.get("SemSim", 1.0) for _, _, edge_data in real_subgraph.edges(node, data=True))
            for node in real_subgraph.nodes
        ]
        target_mean_strength = float(np.mean(real_strengths)) if real_strengths else 0.0
        group_name = f"group_{index}"
        size_bin = _size_bin(len(unique_members))

        instance_specs.append(
            {
                "group": group_name,
                "source": "real_complex",
                "label": 1,
                "nodes": unique_members,
                "size_bin": size_bin,
                "match_score": np.nan,
                "target_density": target_density,
                "target_mean_degree": target_mean_degree,
                "target_mean_strength": target_mean_strength,
            }
        )

        for control_index in range(config.matched_random_per_complex):
            sampled_nodes, match_score = _sample_matched_random_nodes(
                graph,
                len(unique_members),
                target_density,
                target_mean_degree,
                target_mean_strength,
                set(unique_members),
                config,
                rng,
            )
            instance_specs.append(
                {
                    "group": group_name,
                    "source": "matched_random",
                    "label": 0,
                    "nodes": sampled_nodes,
                    "size_bin": size_bin,
                    "match_score": float(match_score),
                    "target_density": target_density,
                    "target_mean_degree": target_mean_degree,
                    "target_mean_strength": target_mean_strength,
                    "control_index": control_index,
                }
            )
    return instance_specs


def _build_complex_instance_feature_table(
    graph: nx.Graph,
    instance_specs: list[dict[str, object]],
    config: DecisionStageConfig,
    *,
    filtration: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, spec in enumerate(instance_specs):
        record = build_subgraph_feature_record(
            graph,
            list(spec["nodes"]),
            subgraph_id=f"{spec['source']}_{index}",
            source=str(spec["source"]),
            label=int(spec["label"]),
            ph_node_cap=config.subgraph_ph_node_cap,
            ph_sampling_seed=config.random_seed + index,
            filtration=filtration,
            ph_feature_prefix=filtration,
        )
        if record is None:
            continue
        record["complex_group"] = str(spec["group"])
        record["size_bin"] = str(spec["size_bin"])
        record["match_score"] = float(spec.get("match_score", np.nan))
        record["target_density"] = float(spec.get("target_density", np.nan))
        record["target_mean_degree"] = float(spec.get("target_mean_degree", np.nan))
        record["target_mean_strength"] = float(spec.get("target_mean_strength", np.nan))
        rows.append(record)
    return pd.DataFrame.from_records(rows)


def _sample_matched_random_nodes(
    graph: nx.Graph,
    target_size: int,
    target_density: float,
    target_mean_degree: float,
    target_mean_strength: float,
    excluded_nodes: set[str],
    config: DecisionStageConfig,
    rng: np.random.Generator,
) -> tuple[list[str], float]:
    degree_lookup = dict(graph.degree())
    graph_nodes = [node for node in graph.nodes if node not in excluded_nodes]
    best_nodes = graph_nodes[:target_size]
    best_score = float("inf")
    for _ in range(config.max_random_attempts):
        candidate_nodes = _sample_connected_candidate(
            graph,
            graph_nodes,
            target_size,
            config,
            rng,
        )
        candidate_subgraph = graph.subgraph(candidate_nodes)
        density = nx.density(candidate_subgraph) if candidate_subgraph.number_of_nodes() > 1 else 0.0
        mean_degree = float(np.mean([degree_lookup[node] for node in candidate_nodes]))
        strengths = [
            sum(edge_data.get("SemSim", 1.0) for _, _, edge_data in candidate_subgraph.edges(node, data=True))
            for node in candidate_subgraph.nodes
        ]
        mean_strength = float(np.mean(strengths)) if strengths else 0.0
        score = (
            abs(density - target_density)
            + abs(mean_degree - target_mean_degree) / max(target_mean_degree, 1.0)
            + abs(mean_strength - target_mean_strength) / max(target_mean_strength, 1.0)
        )
        if score < best_score:
            best_score = score
            best_nodes = candidate_nodes
        if (
            abs(density - target_density) <= config.density_tolerance
            and abs(mean_degree - target_mean_degree) / max(target_mean_degree, 1.0) <= 0.35
        ):
            return candidate_nodes, score
    return best_nodes, best_score


def _sample_connected_candidate(
    graph: nx.Graph,
    graph_nodes: list[str],
    target_size: int,
    config: DecisionStageConfig,
    rng: np.random.Generator,
) -> list[str]:
    start_node = str(rng.choice(graph_nodes))
    selected = [start_node]
    selected_set = {start_node}
    frontier = list(graph.neighbors(start_node))

    while len(selected) < target_size:
        if frontier and rng.random() > config.random_walk_restart_prob:
            next_node = str(rng.choice(frontier))
        else:
            next_node = str(rng.choice(graph_nodes))

        if next_node in selected_set:
            frontier = [node for node in frontier if node not in selected_set]
            continue

        selected.append(next_node)
        selected_set.add(next_node)
        frontier.extend(neighbor for neighbor in graph.neighbors(next_node) if neighbor not in selected_set)
        frontier = list(dict.fromkeys(frontier))

        if len(selected_set) == len(graph_nodes):
            break

    if len(selected) < target_size:
        remaining_nodes = [node for node in graph_nodes if node not in selected_set]
        if remaining_nodes:
            fill = rng.choice(remaining_nodes, size=min(target_size - len(selected), len(remaining_nodes)), replace=False)
            selected.extend(str(node) for node in fill)
    return selected[:target_size]


def _select_eligible_complexes(
    graph: nx.Graph,
    complexes: list[list[str]],
    config: DecisionStageConfig,
) -> list[list[str]]:
    eligible = [
        complex_members
        for complex_members in complexes
        if (
            len(complex_members) >= config.min_complex_size
            and len(complex_members) <= config.max_complex_size
            and set(complex_members).issubset(graph.nodes)
        )
    ]
    return eligible[: config.max_real_complexes]


def _size_bin(size: int | float) -> str:
    size = int(size)
    if size <= 4:
        return "small"
    if size <= 8:
        return "medium"
    return "large"


def _shuffle_labels_within_bins(
    labels: np.ndarray,
    size_bins: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    shuffled = np.asarray(labels).copy()
    for size_bin in np.unique(size_bins):
        indices = np.where(size_bins == size_bin)[0]
        if len(indices) > 1:
            shuffled[indices] = shuffled[rng.permutation(indices)]
    return shuffled


def _sample_protein_panel(
    protein_feature_table: pd.DataFrame,
    config: ExtensionExplorationConfig | LocalProtocolSweepConfig,
) -> pd.DataFrame:
    rng = np.random.default_rng(config.random_seed)
    positive = protein_feature_table[protein_feature_table["protein_in_any_complex"] == 1]
    negative = protein_feature_table[protein_feature_table["protein_in_any_complex"] == 0]

    n_positive = min(config.local_positive_proteins, len(positive))
    n_negative = min(config.local_negative_proteins, len(negative))
    positive_sample = positive.iloc[rng.choice(len(positive), size=n_positive, replace=False)]
    negative_sample = negative.iloc[rng.choice(len(negative), size=n_negative, replace=False)]
    return (
        pd.concat([positive_sample, negative_sample], ignore_index=True)
        .sort_values(by="protein")
        .reset_index(drop=True)
    )


def _apply_weight_jitter(graph: nx.Graph, config: DecisionStageConfig) -> nx.Graph:
    rng = np.random.default_rng(config.random_seed + 10)
    perturbed = graph.copy()
    for node_a, node_b, data in perturbed.edges(data=True):
        jittered = float(np.clip(data.get("SemSim", 0.0) + rng.normal(0.0, config.weight_jitter_std), 0.0, 1.0))
        data["SemSim"] = jittered
    return perturbed


def _subsample_edges(graph: nx.Graph, config: DecisionStageConfig) -> nx.Graph:
    rng = np.random.default_rng(config.random_seed + 11)
    sampled = nx.Graph()
    sampled.add_nodes_from(graph.nodes(data=True))
    edges = list(graph.edges(data=True))
    n_keep = max(1, int(round(len(edges) * config.edge_subsample_fraction)))
    for edge_index in rng.choice(len(edges), size=n_keep, replace=False):
        node_a, node_b, data = edges[int(edge_index)]
        sampled.add_edge(node_a, node_b, **data)
    return sampled


def _subsample_nodes(graph: nx.Graph, config: DecisionStageConfig) -> nx.Graph:
    rng = np.random.default_rng(config.random_seed + 12)
    nodes = np.array(list(graph.nodes()))
    n_keep = max(2, int(round(len(nodes) * config.node_subsample_fraction)))
    keep_nodes = rng.choice(nodes, size=n_keep, replace=False)
    return graph.subgraph(keep_nodes).copy()


def _remove_hubs(graph: nx.Graph, config: DecisionStageConfig) -> nx.Graph:
    degrees = sorted(graph.degree(), key=lambda item: item[1], reverse=True)
    n_remove = max(1, int(round(len(degrees) * config.hub_removal_fraction)))
    remove_nodes = [node for node, _ in degrees[:n_remove]]
    perturbed = graph.copy()
    perturbed.remove_nodes_from(remove_nodes)
    return perturbed


def config_to_dict(
    config: DecisionStageConfig | ExtensionExplorationConfig | LocalProtocolSweepConfig,
) -> dict[str, object]:
    return asdict(config)
