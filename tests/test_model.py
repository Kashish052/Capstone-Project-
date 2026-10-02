import unittest
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from src.model import make_features, train_one


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=80, freq="D"),
                "country": ["Test"] * 80,
                "revenue": [100 + i * 2 for i in range(80)],
            }
        )

    def test_feature_engineering(self):
        out = make_features(self.df)
        self.assertTrue(
            {"trend", "sin_year", "cos_year"}.issubset(out.columns)
        )
        self.assertEqual(len(out), 80)

    def test_model_trains_and_predicts(self):
        model = train_one(
            self.df,
            ExtraTreesRegressor(n_estimators=10, random_state=1),
        )
        X = make_features(self.df).iloc[-5:]
        features = [
            "trend",
            "dayofweek",
            "dayofmonth",
            "month",
            "dayofyear",
            "sin_year",
            "cos_year",
        ]
        pred = model.predict(X[features])

        self.assertEqual(len(pred), 5)
        self.assertTrue((pred >= 0).all())


if __name__ == "__main__":
    unittest.main()
