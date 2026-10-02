import unittest
import pandas as pd

from src.ingest import normalize_transactions, aggregate_daily


class IngestionTests(unittest.TestCase):
    def test_normalization_and_revenue(self):
        raw = pd.DataFrame(
            {
                "date": ["2020-01-01", "bad"],
                "country": ["UK", None],
                "price": [10, 5],
                "quantity": [2, 1],
            }
        )
        out = normalize_transactions(raw)

        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["revenue"], 20)

    def test_aggregation(self):
        raw = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2020-01-01", "2020-01-01"]
                ),
                "country": ["UK", "FR"],
                "price": [10, 5],
                "quantity": [2, 3],
                "revenue": [20, 15],
            }
        )
        out = aggregate_daily(raw)

        self.assertIn("ALL", set(out.country))
        self.assertEqual(
            float(out[out.country == "ALL"].revenue.iloc[0]),
            35,
        )


if __name__ == "__main__":
    unittest.main()
