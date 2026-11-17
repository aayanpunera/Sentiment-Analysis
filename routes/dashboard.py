"""
Dashboard Routes and Analytics Endpoints.
"""

from flask import Blueprint, render_template, jsonify
from database.db import get_dashboard_stats, get_model_metrics

dashboard_bp = Blueprint("dashboard", __name__)


@dashboard_bp.route("/")
def index():
    """Render main analytics dashboard with live metrics."""
    stats = get_dashboard_stats()
    metrics = get_model_metrics()
    return render_template("dashboard.html", stats=stats, metrics=metrics)


@dashboard_bp.route("/api/stats")
def api_stats():
    """Return JSON statistics for Chart.js live rendering."""
    stats = get_dashboard_stats()
    return jsonify(stats)
