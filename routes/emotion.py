"""
Emotion Detection Capability Routes.
"""

import logging
from flask import Blueprint, render_template, request, jsonify
from ml.predict import predict_emotion
from database.db import save_emotion_record, get_emotion_history

logger = logging.getLogger(__name__)
emotion_bp = Blueprint("emotion", __name__)


@emotion_bp.route("/emotion", methods=["GET"])
def emotion_page():
    """Render separate emotion detection interface."""
    recent_emotions = get_emotion_history(limit=10)
    return render_template("emotion.html", recent_emotions=recent_emotions)


@emotion_bp.route("/api/emotion", methods=["POST"])
def api_predict_emotion():
    """Predict emotion for input text and persist in emotion history."""
    if not request.is_json:
        return jsonify({"status": "error", "message": "Expected JSON payload."}), 400

    data = request.get_json()
    text = data.get("text", "")

    try:
        result = predict_emotion(text)
        if result["status"] == "error":
            return jsonify(result), 400

        # Save to emotion_history
        rec_id = save_emotion_record(
            input_text=result["text"],
            emotion=result["emotion"],
            confidence=result["confidence"],
            scores_dict=result["scores"]
        )
        result["record_id"] = rec_id
        return jsonify(result), 200

    except Exception as e:
        logger.exception("Emotion prediction failed")
        return jsonify({"status": "error", "message": f"Emotion analysis failed: {str(e)}"}), 500
