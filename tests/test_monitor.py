import unittest

from src.monitor import distribution_drift, forecast_metrics


class MonitoringTests(unittest.TestCase):
    def test_wasserstein_distance(self):
        self.assertAlmostEqual(
            distribution_drift([0, 1, 2], [0, 1, 2]),
            0.0,
        )
        self.assertGreater(
            distribution_drift([0, 1, 2], [10, 11, 12]),
            0,
        )

    def test_forecast_metrics(self):
        result = forecast_metrics([10, 20], [12, 18])
        self.assertAlmostEqual(result["mae"], 2.0)
        self.assertGreater(result["rmse"], 0)


if __name__ == "__main__":
    unittest.main()
