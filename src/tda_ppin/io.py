from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_biogrid_ppi(ppi_path: Path) -> pd.DataFrame:
    ppi_df = pd.read_table(ppi_path, header=None)
    ppi_df.columns = ["ProteinA", "ProteinB", "SemSim"]
    return ppi_df


def load_corum_complexes(complexes_path: Path) -> list[list[str]]:
    complexes: list[list[str]] = []
    with complexes_path.open() as handle:
        for line in handle:
            proteins = [token for token in line.strip().split("\t") if token]
            if proteins:
                complexes.append(proteins)
    return complexes


def write_json(data: dict, output_path: Path) -> None:
    import json

    output_path.write_text(json.dumps(data, indent=2, sort_keys=True))


def write_table(dataframe: pd.DataFrame, output_path: Path) -> None:
    dataframe.to_csv(output_path, index=False)
