"""
Single Text Sentiment Analyzer Routes.
"""

import logging
from flask import Blueprint, render_template, request, jsonify
from ml.predict import predict_sentiment, ModelManager
from database.db import save_analysis_record

logger = logging.getLogger(__name__)
analyzer_bp = Blueprint("analyzer", __name__)


@analyzer_bp.route("/analyze", methods=["GET"])
def analyze_page():
    """Render single-text sentiment analysis interface."""
    models = ModelManager.available_models()
    return render_template("analyzer.html", models=models)


@analyzer_bp.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Predict sentiment for single text input.
    Validates input, evaluates using selected ML pipeline,
    and records result into SQLite analysis history.
    """
    if not request.is_json:
        return jsonify({"status": "error", "message": "Expected JSON payload."}), 400

    data = request.get_json()
    text = data.get("text", "")
    model_name = data.get("model_name", "Logistic Regression")

    available = ModelManager.available_models()
    if model_name not in available:
        return jsonify({
            "status": "error",
            "message": f"Invalid model '{model_name}'. Choose from: {available}"
        }), 400

    try:
        result = predict_sentiment(text, model_name=model_name)
        if result["status"] == "error":
            return jsonify(result), 400

        # Persist prediction in database
        record_id = save_analysis_record(
            input_text=result["text"],
            sentiment=result["sentiment"],
            model_name=result["model_name"],
            positive_score=result["positive_score"],
            negative_score=result["negative_score"],
            neutral_score=result["neutral_score"],
            analysis_type="single"
        )
        result["record_id"] = record_id
        return jsonify(result), 200

    except Exception as e:
        logger.exception("Error during sentiment prediction")
        return jsonify({"status": "error", "message": f"Prediction failed: {str(e)}"}), 500
