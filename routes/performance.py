"""
Model Performance Dashboard Routes.
"""

import os
import json
from flask import Blueprint, render_template, jsonify
from database.db import get_model_metrics

performance_bp = Blueprint("performance", __name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS_JSON_PATH = os.path.join(BASE_DIR, "models", "metrics.json")
EMOTION_METRICS_PATH = os.path.join(BASE_DIR, "models", "emotion_metrics.json")


@performance_bp.route("/performance", methods=["GET"])
def performance_page():
    """Render model performance comparison and detailed classification metrics."""
    metrics_list = get_model_metrics()

    detailed_metrics = {}
    if os.path.exists(METRICS_JSON_PATH):
        with open(METRICS_JSON_PATH, "r", encoding="utf-8") as f:
            detailed_metrics = json.load(f)

    emotion_metrics = {}
    if os.path.exists(EMOTION_METRICS_PATH):
        with open(EMOTION_METRICS_PATH, "r", encoding="utf-8") as f:
            emotion_metrics = json.load(f)

    return render_template(
        "performance.html",
        metrics_list=metrics_list,
        detailed_metrics=detailed_metrics,
        emotion_metrics=emotion_metrics
    )


@performance_bp.route("/api/metrics", methods=["GET"])
def api_metrics():
    """Return JSON evaluation metrics."""
    if os.path.exists(METRICS_JSON_PATH):
        with open(METRICS_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data)
    return jsonify({})
