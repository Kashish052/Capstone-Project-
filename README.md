# AI Production Capstone – Revenue Forecasting

Production-style machine learning capstone for country-level and global daily revenue forecasting.

## Rubric coverage
- Flask API with `/`, `/health`, and `/predict`
- Unit tests for API, model, ingestion, logging, and monitoring
- Single test runner: `python run_tests.py`
- Isolated test logging directories
- Automated data ingestion
- Random Forest and Extra Trees model comparison
- 30-day moving-average baseline
- EDA visualizations and model-vs-baseline visualization
- Performance monitoring using distribution drift checks
- Docker containerization

## Run locally
```bash
pip install -r requirements.txt
python setup_project.py
python run_tests.py
python app.py
```

API examples:
```text
http://127.0.0.1:8080/
http://127.0.0.1:8080/health
http://127.0.0.1:8080/predict?date=2019-12-01&duration=30
http://127.0.0.1:8080/predict?date=2019-12-01&duration=30&country=Australia
```

## Docker
```bash
docker build -t aavail-capstone .
docker run --rm -p 8080:8080 aavail-capstone
```

The repository includes deterministic sample data so it can be executed without external downloads. Replace the sample fixture with the course-provided dataset when required by the course environment.
