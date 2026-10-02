"""Model training and forecasting utilities."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "daily_revenue.csv"
MODEL_DIR = BASE_DIR / "models"
FEATURES = [
    "trend", "dayofweek", "dayofmonth", "month",
    "dayofyear", "sin_year", "cos_year"
]


def make_features(df):
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    origin = out["date"].min()
    out["trend"] = (out["date"] - origin).dt.days.astype(float)
    out["dayofweek"] = out["date"].dt.dayofweek
    out["dayofmonth"] = out["date"].dt.day
    out["month"] = out["date"].dt.month
    out["dayofyear"] = out["date"].dt.dayofyear
    out["sin_year"] = np.sin(2 * np.pi * out["dayofyear"] / 365.25)
    out["cos_year"] = np.cos(2 * np.pi * out["dayofyear"] / 365.25)
    return out


def train_one(df, estimator):
    feat = make_features(df)
    estimator.fit(feat[FEATURES], feat["revenue"])
    return estimator


def evaluate_models(df, test_days=60):
    results = []
    for country, group in df.groupby("country"):
        group = group.sort_values("date").copy()
        if len(group) <= test_days + 30:
            continue

        train = group.iloc[:-test_days]
        test = group.iloc[-test_days:]
        train_f = make_features(train)
        test_f = make_features(
            pd.concat([train.tail(1), test], ignore_index=True)
        ).iloc[1:]

        estimators = [
            (
                "RandomForest",
                RandomForestRegressor(
                    n_estimators=160,
                    random_state=42,
                    min_samples_leaf=2,
                    n_jobs=-1,
                ),
            ),
            (
                "ExtraTrees",
                ExtraTreesRegressor(
                    n_estimators=160,
                    random_state=42,
                    min_samples_leaf=2,
                    n_jobs=-1,
                ),
            ),
        ]

        for name, est in estimators:
            est.fit(train_f[FEATURES], train_f["revenue"])
            pred = est.predict(test_f[FEATURES])
            results.append(
                {
                    "country": country,
                    "model": name,
                    "MAE": mean_absolute_error(test["revenue"], pred),
                    "RMSE": mean_squared_error(test["revenue"], pred) ** 0.5,
                }
            )

    return pd.DataFrame(results)


def train_all(data_path=DATA_PATH):
    df = pd.read_csv(data_path, parse_dates=["date"])
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    metadata = {
        "features": FEATURES,
        "origin": str(df["date"].min()),
    }

    for country, group in df.groupby("country"):
        if len(group) < 60:
            continue

        model = ExtraTreesRegressor(
            n_estimators=200,
            random_state=42,
            min_samples_leaf=2,
            n_jobs=-1,
        )
        model = train_one(group, model)

        safe = country.replace(" ", "_").replace("/", "_")
        joblib.dump(
            {
                "model": model,
                "origin": str(group["date"].min()),
                "features": FEATURES,
            },
            MODEL_DIR / f"{safe}.joblib",
        )

    joblib.dump(metadata, MODEL_DIR / "metadata.joblib")
    return evaluate_models(df)


def forecast_daily(country, start_date, duration=30):
    safe = country.replace(" ", "_").replace("/", "_")
    artifact_path = MODEL_DIR / f"{safe}.joblib"

    if not artifact_path.exists():
        raise ValueError(f"Unknown country: {country}")

    artifact = joblib.load(artifact_path)
    dates = pd.date_range(
        pd.Timestamp(start_date),
        periods=int(duration),
        freq="D",
    )

    features = make_features(
        pd.DataFrame(
            {
                "date": dates,
                "revenue": 0.0,
                "country": country,
            }
        )
    )

    origin = pd.Timestamp(artifact["origin"])
    features["trend"] = (features["date"] - origin).dt.days.astype(float)
    predictions = artifact["model"].predict(features[FEATURES])

    return pd.DataFrame(
        {
            "date": dates,
            "prediction": np.maximum(predictions, 0),
        }
    )


def baseline_forecast(
    daily_df,
    country,
    start_date,
    duration=30,
    window=30,
):
    subset = daily_df[daily_df["country"] == country].sort_values("date")
    start = pd.Timestamp(start_date)
    history = subset[subset["date"] < start]["revenue"]

    if history.empty:
        raise ValueError("No historical data before requested date")

    value = float(history.tail(window).mean())

    return pd.DataFrame(
        {
            "date": pd.date_range(start, periods=int(duration), freq="D"),
            "prediction": value,
        }
    )


if __name__ == "__main__":
    result = train_all()
    result.to_csv(BASE_DIR / "reports" / "model_comparison.csv", index=False)
    print(result.groupby("model")[["MAE", "RMSE"]].mean())
