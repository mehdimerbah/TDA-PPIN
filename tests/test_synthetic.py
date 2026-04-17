import sys
import unittest
from pathlib import Path
from unittest import SkipTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from tda_ppin.synthetic import SyntheticConfig, build_synthetic_graph
except ModuleNotFoundError as exc:
    raise SkipTest(f"Scientific Python dependencies are unavailable: {exc}") from exc


class TestSyntheticWorkflow(unittest.TestCase):
    def test_build_synthetic_graph_is_seeded_and_shaped(self) -> None:
        graph, complexes = build_synthetic_graph(SyntheticConfig(random_seed=11))
        self.assertEqual(graph.number_of_nodes(), 100)
        self.assertEqual(len(complexes), 3)
        self.assertEqual([len(complex_members) for complex_members in complexes], [10, 7, 5])


if __name__ == "__main__":
    unittest.main()
