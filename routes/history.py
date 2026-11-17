"""
Analysis History Routes and Management Endpoints.
"""

from flask import Blueprint, render_template, request, jsonify
from database.db import get_analysis_history, delete_analysis_record
from ml.predict import ModelManager

history_bp = Blueprint("history", __name__)


@history_bp.route("/history", methods=["GET"])
def history_page():
    """Render analysis history page with search and filter controls."""
    models = ModelManager.available_models()
    return render_template("history.html", models=models)


@history_bp.route("/api/history", methods=["GET"])
def api_get_history():
    """Retrieve paginated analysis history."""
    try:
        page = max(1, int(request.args.get("page", 1)))
        per_page = min(100, max(1, int(request.args.get("per_page", 20))))
    except ValueError:
        page = 1
        per_page = 20

    sentiment_filter = request.args.get("sentiment", "all")
    model_filter = request.args.get("model", "all")
    search_query = request.args.get("q", "")

    records, total = get_analysis_history(
        page=page,
        per_page=per_page,
        sentiment_filter=sentiment_filter,
        model_filter=model_filter,
        search_query=search_query
    )

    total_pages = (total + per_page - 1) // per_page if total > 0 else 1

    return jsonify({
        "status": "success",
        "records": records,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": total_pages
    })


@history_bp.route("/api/history/<int:record_id>", methods=["DELETE"])
def api_delete_history(record_id: int):
    """Delete a history record by ID."""
    deleted = delete_analysis_record(record_id)
    if deleted:
        return jsonify({"status": "success", "message": f"Record {record_id} deleted."}), 200
    return jsonify({"status": "error", "message": "Record not found."}), 404
