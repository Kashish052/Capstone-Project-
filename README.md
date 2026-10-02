# AI Production Capstone – Revenue Forecasting

Production-style machine learning capstone for country-level and global daily revenue forecasting.

## What this project demonstrates

This repository is organized around the Coursera peer-review rubric:

- Flask API with `/`, `/health`, `/predict`, and `/logs`
- Unit tests for API, model, ingestion, logging, and monitoring
- One command for the full test suite: `python run_tests.py`
- Windows-safe isolated test logging
- Automated transaction ingestion and daily aggregation
- Random Forest vs Extra Trees model comparison
- 30-day moving-average baseline
- EDA visualizations
- Final model vs baseline visualization
- Performance monitoring with forecast metrics and Wasserstein drift
- Dockerized application
- GitHub Actions CI for reproducible tests

## Project structure

```text
Capstone-Project-/
├── src/
│   ├── api.py
│   ├── eda.py
│   ├── generate_sample_data.py
│   ├── ingest.py
│   ├── log.py
│   ├── model.py
│   └── monitor.py
├── tests/
│   ├── test_api.py
│   ├── test_ingest.py
│   ├── test_logging.py
│   ├── test_model.py
│   └── test_monitor.py
├── Dockerfile
├── requirements.txt
├── run_tests.py
└── setup_project.py
```

## Run locally

From the repository root:

```bash
pip install -r requirements.txt
python setup_project.py
python run_tests.py
python app.py
```

`setup_project.py` creates a deterministic sample transaction fixture, performs ingestion, trains country/global models, and generates the EDA/model-comparison figures.

## API

Open:

```text
http://127.0.0.1:8080/
```

Health check:

```text
http://127.0.0.1:8080/health
```

Global forecast:

```text
http://127.0.0.1:8080/predict?date=2019-12-01&duration=30
```

Country forecast:

```text
http://127.0.0.1:8080/predict?date=2019-12-01&duration=30&country=Australia
```

The prediction endpoint returns the requested forecast window, total predicted revenue, baseline revenue, and measured request latency.

## Docker

```bash
docker build -t aavail-capstone .
docker run --rm -p 8080:8080 aavail-capstone
```

## Data note

The repository deliberately uses a deterministic sample-data generator so reviewers can reproduce the workflow without an external download. In a course environment where the official AAVAIL dataset is supplied, replace the generated raw fixture with the course-provided input while keeping the same ingestion interface.

## Academic-use note

This implementation is an original production-oriented capstone scaffold built to exercise the peer-review requirements. It should be adapted and verified against the exact dataset and instructions provided by the course before submission.
