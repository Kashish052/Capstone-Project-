"""Create a deterministic local fixture so the project is reproducible offline."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"

COUNTRIES = {
    "United Kingdom": 1.00,
    "Australia": 0.68,
    "Germany": 0.78,
    "France": 0.72,
    "Netherlands": 0.55,
    "Spain": 0.50,
}


def create_fixture(path=RAW_DIR / "transactions_sample.json", seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2017-11-01", "2019-12-31", freq="D")
    rows = []
    invoice = 100000

    for country, scale in COUNTRIES.items():
        for i, date in enumerate(dates):
            seasonal = 1 + 0.20 * np.sin(2 * np.pi * date.dayofyear / 365.25)
            weekly = 1.0 + (0.10 if date.dayofweek in (4, 5) else -0.03)
            trend = 1 + 0.00025 * i
            target = 9000 * scale * seasonal * weekly * trend
            n = max(8, int(rng.poisson(22)))

            values = np.maximum(
                0.25,
                rng.lognormal(np.log(target / n / 20), 0.55, n),
            )
            quantities = rng.integers(1, 5, n)
            prices = values / quantities

            for price, qty in zip(prices, quantities):
                rows.append(
                    {
                        "date": date.strftime("%Y-%m-%d"),
                        "country": country,
                        "price": round(float(price), 2),
                        "quantity": int(qty),
                        "invoice": str(invoice),
                    }
                )
                invoice += 1

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh)

    return path


if __name__ == "__main__":
    print(create_fixture())
