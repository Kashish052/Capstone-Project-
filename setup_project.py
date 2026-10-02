"""Prepare data, train models, and generate reproducible reports."""
from pathlib import Path

from src.generate_sample_data import create_fixture
from src.ingest import ingest
from src.model import train_all
from src.eda import run_eda


BASE_DIR = Path(__file__).resolve().parent


def main():
    create_fixture()
    ingest()

    results = train_all()
    reports = BASE_DIR / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    results.to_csv(reports / "model_comparison.csv", index=False)

    run_eda()
    print("Project setup complete.")


if __name__ == "__main__":
    main()
