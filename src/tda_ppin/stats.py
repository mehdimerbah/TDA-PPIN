from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def binary_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    scores = np.asarray(scores, dtype=float)
    positives = y_true == 1
    negatives = y_true == 0
    n_pos = int(np.sum(positives))
    n_neg = int(np.sum(negatives))
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    pos_rank_sum = float(np.sum(ranks[positives]))
    return (pos_rank_sum - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    scores = np.asarray(scores, dtype=float)
    positives = int(np.sum(y_true == 1))
    if positives == 0:
        return 0.0

    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y_true[order]
    cumulative_hits = np.cumsum(y_sorted == 1)
    precision = cumulative_hits / (np.arange(len(y_sorted)) + 1.0)
    return float(np.sum(precision * (y_sorted == 1)) / positives)


@dataclass
class LogisticModel:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: float


def fit_logistic_regression(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    learning_rate: float = 0.1,
    steps: int = 400,
    l2: float = 1e-3,
) -> LogisticModel:
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels, dtype=float)
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale == 0] = 1.0
    x = (features - mean) / scale

    weights = np.zeros(x.shape[1], dtype=float)
    bias = 0.0
    for _ in range(steps):
        logits = x @ weights + bias
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))
        error = probabilities - labels
        grad_w = (x.T @ error) / len(x) + l2 * weights
        grad_b = float(np.mean(error))
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b

    return LogisticModel(mean=mean, scale=scale, weights=weights, bias=bias)


def predict_logistic(model: LogisticModel, features: np.ndarray) -> np.ndarray:
    x = (np.asarray(features, dtype=float) - model.mean) / model.scale
    logits = x @ model.weights + model.bias
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))


def repeated_split_logistic_metrics(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    repeats: int = 5,
    test_fraction: float = 0.25,
    seed: int = 7,
) -> dict[str, float | list[float]]:
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels, dtype=float)
    rng = np.random.default_rng(seed)
    aurocs: list[float] = []
    aps: list[float] = []

    n_samples = len(labels)
    n_test = max(1, int(round(n_samples * test_fraction)))
    for _ in range(repeats):
        indices = rng.permutation(n_samples)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        if len(train_indices) == 0 or len(np.unique(labels[train_indices])) < 2:
            continue
        model = fit_logistic_regression(features[train_indices], labels[train_indices])
        probabilities = predict_logistic(model, features[test_indices])
        aurocs.append(binary_auroc(labels[test_indices], probabilities))
        aps.append(average_precision(labels[test_indices], probabilities))

    return {
        "mean_auroc": float(np.mean(aurocs)) if aurocs else 0.5,
        "mean_average_precision": float(np.mean(aps)) if aps else 0.0,
        "auroc_values": [float(value) for value in aurocs],
        "average_precision_values": [float(value) for value in aps],
    }


def repeated_group_split_logistic_metrics(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    repeats: int = 5,
    test_fraction: float = 0.25,
    seed: int = 7,
) -> dict[str, float | list[float]]:
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels, dtype=float)
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    aurocs: list[float] = []
    aps: list[float] = []

    unique_groups = np.unique(groups)
    n_test_groups = max(1, int(round(len(unique_groups) * test_fraction)))
    for _ in range(repeats):
        shuffled_groups = rng.permutation(unique_groups)
        test_groups = set(shuffled_groups[:n_test_groups].tolist())
        test_mask = np.array([group in test_groups for group in groups], dtype=bool)
        train_mask = ~test_mask
        if not train_mask.any() or len(np.unique(labels[train_mask])) < 2 or len(np.unique(labels[test_mask])) < 2:
            continue
        model = fit_logistic_regression(features[train_mask], labels[train_mask])
        probabilities = predict_logistic(model, features[test_mask])
        aurocs.append(binary_auroc(labels[test_mask], probabilities))
        aps.append(average_precision(labels[test_mask], probabilities))

    return {
        "mean_auroc": float(np.mean(aurocs)) if aurocs else 0.5,
        "mean_average_precision": float(np.mean(aps)) if aps else 0.0,
        "auroc_values": [float(value) for value in aurocs],
        "average_precision_values": [float(value) for value in aps],
    }


def cosine_similarity(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref_norm = np.linalg.norm(reference)
    cand_norm = np.linalg.norm(candidate)
    if ref_norm == 0 or cand_norm == 0:
        return 0.0
    return float(np.dot(reference, candidate) / (ref_norm * cand_norm))
