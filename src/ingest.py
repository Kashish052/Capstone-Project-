"""Data ingestion and daily revenue aggregation."""
from pathlib import Path
import json
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def load_json_files(data_dir=RAW_DIR):
    """Load transaction JSON files and return one normalized DataFrame."""
    rows = []
    for path in sorted(Path(data_dir).glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as fh:
                obj = json.load(fh)
            if isinstance(obj, dict):
                obj = obj.get("data", obj.get("records", []))
            if isinstance(obj, list):
                rows.extend(obj)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read {path}: {exc}") from exc

    if not rows:
        raise FileNotFoundError(f"No JSON transaction files found in {data_dir}")

    df = pd.DataFrame(rows)
    required = {"date", "country", "price", "quantity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required fields: {sorted(missing)}")
    return normalize_transactions(df)


def normalize_transactions(df):
    """Normalize dates/numerics and remove unusable transactions."""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out["quantity"] = pd.to_numeric(out["quantity"], errors="coerce")
    out["country"] = out["country"].fillna("Unknown").astype(str).str.strip()
    out = out.dropna(subset=["date", "price", "quantity"])
    out = out[out["price"] >= 0]
    out = out[out["quantity"] > 0]
    out["revenue"] = out["price"] * out["quantity"]
    return out.reset_index(drop=True)


def aggregate_daily(df):
    """Aggregate transactions by day/country plus global totals."""
    daily_country = (
        df.groupby(["date", "country"], as_index=False)["revenue"]
        .sum()
        .sort_values(["country", "date"])
    )
    global_daily = (
        df.groupby("date", as_index=False)["revenue"]
        .sum()
        .assign(country="ALL")
    )
    result = pd.concat([daily_country, global_daily], ignore_index=True)
    return result.sort_values(["country", "date"]).reset_index(drop=True)


def ingest(output_path=PROCESSED_DIR / "daily_revenue.csv"):
    """Automatable ingestion entry point."""
    df = load_json_files()
    daily = aggregate_daily(df)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output_path, index=False)
    return daily


if __name__ == "__main__":
    result = ingest()
    print(f"Wrote {len(result):,} daily country rows")
