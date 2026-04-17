import sys
import unittest
from pathlib import Path
from unittest import SkipTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import networkx as nx

    from tda_ppin.evaluation import summarize_complex_coverage
except ModuleNotFoundError as exc:
    raise SkipTest(f"Scientific Python dependencies are unavailable: {exc}") from exc


class TestEvaluation(unittest.TestCase):
    def test_summarize_complex_coverage_handles_partial_overlap(self) -> None:
        graph = nx.Graph()
        graph.add_nodes_from(["A", "B", "C"])
        complexes = [["A", "B"], ["A", "D"]]
        summary = summarize_complex_coverage(graph, complexes)
        self.assertEqual(summary["num_complexes"], 2.0)
        self.assertEqual(summary["fraction_complexes_fully_covered"], 0.5)
        self.assertEqual(summary["fraction_complex_members_observed"], 0.75)


if __name__ == "__main__":
    unittest.main()
