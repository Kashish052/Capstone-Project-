"""Generate EDA and model-vs-baseline visualizations."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
from .model import evaluate_models, make_features, FEATURES

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "daily_revenue.csv"
REPORT_DIR = BASE_DIR / "reports" / "figures"


def run_eda():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    global_df = df[df.country == "ALL"].sort_values("date")

    plt.figure(figsize=(10, 5))
    plt.plot(global_df.date, global_df.revenue)
    plt.title("Global Daily Revenue")
    plt.xlabel("Date")
    plt.ylabel("Revenue")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "global_revenue_timeseries.png", dpi=140)
    plt.close()

    country_summary = (
        df[df.country != "ALL"]
        .groupby("country")["revenue"]
        .sum()
        .sort_values()
    )
    plt.figure(figsize=(10, 5))
    country_summary.plot(kind="barh")
    plt.title("Revenue by Country")
    plt.xlabel("Revenue")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "revenue_by_country.png", dpi=140)
    plt.close()

    features = make_features(global_df)
    corr = features[
        ["revenue", "dayofweek", "dayofmonth", "month", "dayofyear"]
    ].corr()

    plt.figure(figsize=(7, 5))
    plt.imshow(corr, aspect="auto")
    plt.xticks(
        range(len(corr.columns)),
        corr.columns,
        rotation=45,
        ha="right",
    )
    plt.yticks(range(len(corr.index)), corr.index)
    plt.colorbar(label="Correlation")
    plt.title("Revenue Feature Correlations")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "feature_correlations.png", dpi=140)
    plt.close()

    comparison = evaluate_models(df)
    comparison.to_csv(
        BASE_DIR / "reports" / "model_comparison.csv",
        index=False,
    )

    summary = comparison.groupby("model")[["MAE", "RMSE"]].mean()
    plt.figure(figsize=(8, 5))
    summary["RMSE"].plot(kind="bar")
    plt.title("Model Comparison: RMSE")
    plt.ylabel("RMSE")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "model_comparison.png", dpi=140)
    plt.close()

    # Final model vs 30-day moving-average baseline.
    holdout = global_df.tail(60).copy()
    train = global_df.iloc[:-60].copy()

    train_f = make_features(train)
    holdout_f = make_features(
        pd.concat([train.tail(1), holdout], ignore_index=True)
    ).iloc[1:]

    final_model = ExtraTreesRegressor(
        n_estimators=200,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=-1,
    )
    final_model.fit(train_f[FEATURES], train_f["revenue"])
    model_pred = final_model.predict(holdout_f[FEATURES])

    baseline_value = float(train["revenue"].tail(30).mean())

    plt.figure(figsize=(11, 5))
    plt.plot(holdout["date"], holdout["revenue"], label="Actual")
    plt.plot(holdout["date"], model_pred, label="Extra Trees")
    plt.plot(
        holdout["date"],
        [baseline_value] * len(holdout),
        label="30-day baseline",
    )
    plt.title("Final Model vs Baseline on Holdout")
    plt.xlabel("Date")
    plt.ylabel("Revenue")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "model_vs_baseline.png", dpi=140)
    plt.close()

    return comparison


if __name__ == "__main__":
    print(run_eda())
