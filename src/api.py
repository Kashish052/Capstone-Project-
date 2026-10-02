"""Flask prediction and logging API."""
from pathlib import Path
import time
import pandas as pd
from flask import Flask, jsonify, request
from .model import forecast_daily, baseline_forecast
from .log import get_logger

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "daily_revenue.csv"
app = Flask(__name__)
logger = get_logger()


def _ensure_duration(value):
    try:
        duration = int(value)
        if duration < 1 or duration > 365:
            raise ValueError
        return duration
    except (TypeError, ValueError):
        raise ValueError("duration must be an integer between 1 and 365")


def predict(date, duration=30, country=None):
    duration = _ensure_duration(duration)
    daily = pd.read_csv(DATA_PATH, parse_dates=["date"])
    selected = country or "ALL"
    pred = forecast_daily(selected, date, duration)
    baseline = baseline_forecast(daily, selected, date, duration)
    return {
        "country": selected,
        "start_date": str(pd.Timestamp(date).date()),
        "duration": duration,
        "predicted_revenue": round(float(pred["prediction"].sum()), 2),
        "baseline_revenue": round(float(baseline["prediction"].sum()), 2),
        "daily_predictions": [
            {"date": str(row.date.date()), "revenue": round(float(row.prediction), 2)}
            for row in pred.itertuples()
        ],
    }


@app.get("/")
def home():
    return jsonify({
        "service": "AAVAIL Revenue Forecast API",
        "status": "running",
        "predict_endpoint": "/predict"
    })


@app.route("/predict", methods=["GET", "POST"])
def predict_endpoint():
    started = time.perf_counter()
    date = request.args.get("date")
    if not date:
        return jsonify({"error": "date query parameter is required"}), 400
    try:
        result = predict(date, request.args.get("duration", 30), request.args.get("country"))
        logger.info(
            "prediction country=%s date=%s duration=%s",
            result["country"], date, result["duration"]
        )
        result["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
        return jsonify(result)
    except (ValueError, FileNotFoundError) as exc:
        logger.warning("prediction_error=%s", exc)
        return jsonify({"error": str(exc)}), 400


@app.post("/logs")
def logs_endpoint():
    log_type = request.args.get("type", "application")
    logger.info("log_check type=%s", log_type)
    return jsonify({"status": "ok", "type": log_type})


@app.get("/health")
def health():
    return jsonify({"status": "healthy"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
