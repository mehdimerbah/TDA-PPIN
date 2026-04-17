from __future__ import annotations

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
