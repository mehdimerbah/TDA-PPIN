from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from persim import plot_diagrams


def save_degree_distribution(degrees: list[int], output_path: Path, title: str) -> None:
    values = {}
    for degree in degrees:
        values[degree] = values.get(degree, 0) + 1

    x, y = zip(*sorted(values.items()))
    plt.figure(figsize=(8, 5))
    plt.plot(x, y)
    plt.xlabel("Node Degree")
    plt.ylabel("Protein Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_ripser_diagrams(diagrams: list[np.ndarray], output_dir: Path, prefix: str) -> None:
    plt.figure()
    plot_diagrams(diagrams)
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_all.png")
    plt.close()

    for dimension in range(min(3, len(diagrams))):
        plt.figure()
        plot_diagrams(diagrams, plot_only=[dimension], title=f"{prefix} Dimension {dimension}")
        plt.tight_layout()
        plt.savefig(output_dir / f"{prefix}_dim{dimension}.png")
        plt.close()


def save_barcode(diagrams: list[np.ndarray], dimension: int, output_path: Path) -> None:
    diag_dim = diagrams[dimension]
    birth = diag_dim[:, 0]
    death = diag_dim[:, 1].copy()
    finite_bars = death[death != np.inf]
    inf_end = 2 * max(finite_bars) if len(finite_bars) > 0 else 2
    death[death == np.inf] = inf_end

    plt.figure(figsize=(10, 5))
    for index, (start, stop) in enumerate(zip(birth, death)):
        color = "k" if stop == inf_end else "b"
        plt.plot([start, stop], [index, index], color=color, lw=2)
    plt.title(f"Barcode Dimension {dimension}")
    plt.xlabel("Filtration Value")
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_landscape(landscape: np.ndarray, output_path: Path, title: str) -> None:
    plt.figure(figsize=(10, 5))
    for row in landscape:
        plt.plot(row)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
