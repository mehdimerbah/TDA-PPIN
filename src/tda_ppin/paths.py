from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RepoPaths:
    repo_root: Path
    data_dir: Path
    legacy_dir: Path
    results_dir: Path
    figures_dir: Path
    processed_dir: Path
    reports_dir: Path


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_repo_paths() -> RepoPaths:
    repo_root = get_repo_root()
    return RepoPaths(
        repo_root=repo_root,
        data_dir=repo_root / "data",
        legacy_dir=repo_root / "tda",
        results_dir=repo_root / "results",
        figures_dir=repo_root / "results" / "figures",
        processed_dir=repo_root / "results" / "processed",
        reports_dir=repo_root / "results" / "reports",
    )


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
