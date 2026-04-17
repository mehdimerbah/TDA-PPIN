import sys
import unittest
from pathlib import Path
from unittest import SkipTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import networkx as nx
    import numpy as np

    from tda_ppin.evaluation import build_bounded_ego_subgraph, build_local_neighborhood_subgraph
    from tda_ppin.ph import graph_to_distance_matrix
except ModuleNotFoundError as exc:
    raise SkipTest(f"Scientific Python dependencies are unavailable: {exc}") from exc


class TestPHExplorationHelpers(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = nx.Graph()
        self.graph.add_edge("A", "B", SemSim=0.9)
        self.graph.add_edge("A", "C", SemSim=0.8)
        self.graph.add_edge("A", "D", SemSim=0.2)
        self.graph.add_edge("B", "C", SemSim=0.7)
        self.graph.add_edge("C", "E", SemSim=0.85)
        self.graph.add_edge("D", "F", SemSim=0.95)

    def test_bounded_ego_subgraph_keeps_strongest_neighbors(self) -> None:
        subgraph = build_bounded_ego_subgraph(self.graph, "A", max_nodes=3)
        self.assertEqual(set(subgraph.nodes()), {"A", "B", "C"})

    def test_local_neighborhood_methods_keep_seed_and_respect_cap(self) -> None:
        for method in ("top_weighted_neighbors", "ego_radius_1", "ego_radius_2", "seed_expansion"):
            subgraph = build_local_neighborhood_subgraph(self.graph, "A", method=method, max_nodes=4)
            self.assertIn("A", subgraph.nodes())
            self.assertLessEqual(subgraph.number_of_nodes(), 4)

    def test_seed_expansion_prefers_strong_frontier(self) -> None:
        subgraph = build_local_neighborhood_subgraph(
            self.graph,
            "A",
            method="seed_expansion",
            max_nodes=4,
        )
        self.assertEqual(set(subgraph.nodes()), {"A", "B", "C", "E"})

    def test_graph_to_distance_matrix_supports_all_filtrations(self) -> None:
        for filtration in ("correlation_distance", "hop_distance", "weighted_shortest_path"):
            matrix = graph_to_distance_matrix(self.graph, filtration=filtration)
            self.assertEqual(matrix.shape, (6, 6))
            self.assertTrue(np.allclose(np.diag(matrix), 0.0))


if __name__ == "__main__":
    unittest.main()
