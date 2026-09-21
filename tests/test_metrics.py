import math
import unittest

from dohnuts.metrics import fit_temperatures, summarize


class MetricTests(unittest.TestCase):
    def test_twenty_latency_samples_do_not_report_maximum_as_p95(self):
        from dohnuts.experiment import latency_stats

        result = latency_stats(list(range(1, 21)), 1)
        self.assertAlmostEqual(result["p95_ms"], 19.05)

    def test_known_binary_distribution(self):
        rows = [
            {"type": "noul", "logits": [0.0, math.log(4)], "target": t}
            for t in [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]
        ]
        result = summarize(rows)
        self.assertAlmostEqual(result["accuracy"], 0.75)
        self.assertAlmostEqual(result["ece_15"], 0.05)
        self.assertAlmostEqual(result["nll"], -(0.75 * math.log(0.8) + 0.25 * math.log(0.2)))
        self.assertAlmostEqual(result["brier_sum"], 0.38)
        self.assertAlmostEqual(result["brier_per_candidate"], 0.19)

    def test_temperature_fits_soft_targets_without_changing_order(self):
        rows = [{"type": "noul", "logits": [0.0, 4.0], "target": [0.25, 0.75]} for _ in range(20)]
        temperatures = fit_temperatures(rows)
        self.assertAlmostEqual(temperatures["noul"], 4 / math.log(3), delta=0.02)
        self.assertLess(summarize(rows, temperatures)["nll"], summarize(rows)["nll"])

    def test_ordinal_distance_and_soft_accuracy(self):
        result = summarize(
            [{"type": "score", "logits": [-100.0, -100.0, 0.0], "target": [1.0, 0.0, 0.0]}]
        )
        self.assertAlmostEqual(result["rps"], 1.0)
        self.assertAlmostEqual(result["score_mae"], 2.0)
        self.assertEqual(result["soft_accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
