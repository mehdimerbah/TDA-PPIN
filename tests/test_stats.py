import sys
import unittest
from pathlib import Path
from unittest import SkipTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import numpy as np

    from tda_ppin.stats import average_precision, binary_auroc
except ModuleNotFoundError as exc:
    raise SkipTest(f"Scientific Python dependencies are unavailable: {exc}") from exc


class TestStats(unittest.TestCase):
    def test_binary_auroc_perfect_ranking(self) -> None:
        y_true = np.array([0, 0, 1, 1], dtype=float)
        scores = np.array([0.1, 0.2, 0.8, 0.9], dtype=float)
        self.assertEqual(binary_auroc(y_true, scores), 1.0)

    def test_average_precision_perfect_ranking(self) -> None:
        y_true = np.array([1, 0, 1, 0], dtype=float)
        scores = np.array([0.9, 0.4, 0.8, 0.1], dtype=float)
        self.assertEqual(average_precision(y_true, scores), 1.0)


if __name__ == "__main__":
    unittest.main()
