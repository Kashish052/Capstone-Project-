"""Simple production monitoring utilities."""
from pathlib import Path
import json
import numpy as np
from scipy.stats import wasserstein_distance

BASE_DIR = Path(__file__).resolve().parents[1]


def distribution_drift(reference, current):
    """Measure distribution drift with Wasserstein distance."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref, cur = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
    if len(ref) == 0 or len(cur) == 0:
        raise ValueError("Both distributions must contain numeric observations")
    return float(wasserstein_distance(ref, cur))


def forecast_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mae = float(np.mean(np.abs(actual - predicted)))
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    return {"mae": mae, "rmse": rmse}


def write_monitoring_report(metrics, path=BASE_DIR / "reports" / "monitoring.json"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    return path
