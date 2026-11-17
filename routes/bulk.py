"""
Bulk CSV Upload and Batch Processing Routes.
"""

import io
import os
import logging
import pandas as pd
from flask import Blueprint, render_template, request, jsonify, current_app
from werkzeug.utils import secure_filename

from ml.predict import predict_batch_sentiment, ModelManager
from database.db import (
    create_batch_job,
    save_batch_results,
    get_batch_job,
    get_batch_results,
    get_all_batch_jobs
)

logger = logging.getLogger(__name__)
bulk_bp = Blueprint("bulk", __name__)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in current_app.config["ALLOWED_EXTENSIONS"]


@bulk_bp.route("/bulk", methods=["GET"])
def bulk_page():
    """Render bulk CSV analysis interface."""
    models = ModelManager.available_models()
    recent_batches = get_all_batch_jobs(limit=10)
    return render_template("bulk.html", models=models, recent_batches=recent_batches)


@bulk_bp.route("/api/bulk-analyze", methods=["POST"])
def api_bulk_analyze():
    """
    Handle CSV upload, validate data, execute batch sentiment inference,
    and persist results in SQLite.
    """
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded."}), 400

    file = request.files["file"]
    if not file or file.filename.strip() == "":
        return jsonify({"status": "error", "message": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"status": "error", "message": "Invalid file format. Only CSV files (.csv) are supported."}), 400

    filename = secure_filename(file.filename)
    model_name = request.form.get("model_name", "Logistic Regression")
    column_name = request.form.get("column_name", "").strip()

    available = ModelManager.available_models()
    if model_name not in available:
        return jsonify({"status": "error", "message": f"Invalid model. Choose from: {available}"}), 400

    try:
        # Read uploaded CSV
        content = file.read().decode("utf-8", errors="replace")
        if not content.strip():
            return jsonify({"status": "error", "message": "Uploaded CSV file is empty."}), 400

        df = pd.read_csv(io.StringIO(content))
        if df.empty:
            return jsonify({"status": "error", "message": "Uploaded CSV contains no rows."}), 400

        max_rows = current_app.config.get("MAX_BATCH_ROWS", 5000)
        if len(df) > max_rows:
            return jsonify({
                "status": "error",
                "message": f"CSV has {len(df)} rows, exceeding the maximum allowed limit of {max_rows} rows."
            }), 400

        # Determine target text column
        if column_name and column_name in df.columns:
            target_col = column_name
        else:
            # Auto-detect plausible text column names
            candidates = ["text", "review", "comment", "content", "tweet", "message", "sentence"]
            detected = None
            for c in df.columns:
                if str(c).lower().strip() in candidates:
                    detected = c
                    break
            if not detected:
                # Default to first object/string column
                str_cols = df.select_dtypes(include=["object"]).columns
                if len(str_cols) > 0:
                    detected = str_cols[0]
                else:
                    detected = df.columns[0]
            target_col = detected

        texts = df[target_col].astype(str).tolist()

        # Run vectorized batch inference
        batch_predictions = predict_batch_sentiment(texts, model_name=model_name)
        if not batch_predictions or batch_predictions[0].get("status") == "error":
            return jsonify({"status": "error", "message": "Could not extract valid text rows from CSV."}), 400

        total_records = len(texts)
        processed_records = len(batch_predictions)

        # Store in database
        batch_id = create_batch_job(
            filename=filename,
            total_records=total_records,
            processed_records=processed_records,
            model_name=model_name,
            status="completed"
        )

        # Attach model name to result records
        for item in batch_predictions:
            item["model_name"] = model_name

        save_batch_results(batch_id, batch_predictions)

        # Calculate summary statistics
        counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
        for item in batch_predictions:
            s = item.get("sentiment")
            if s in counts:
                counts[s] += 1

        return jsonify({
            "status": "success",
            "batch_id": batch_id,
            "filename": filename,
            "total_records": total_records,
            "processed_records": processed_records,
            "column_used": target_col,
            "model_name": model_name,
            "summary": counts,
            "results": batch_predictions[:100]  # First 100 for UI table preview
        }), 200

    except Exception as e:
        logger.exception("Failed to process bulk CSV")
        return jsonify({"status": "error", "message": f"Failed to process CSV file: {str(e)}"}), 500


@bulk_bp.route("/api/batch/<int:batch_id>", methods=["GET"])
def api_get_batch(batch_id: int):
    """Retrieve full batch job details and results."""
    batch = get_batch_job(batch_id)
    if not batch:
        return jsonify({"status": "error", "message": "Batch not found."}), 404

    results = get_batch_results(batch_id)
    return jsonify({
        "status": "success",
        "batch": batch,
        "results": results
    })
