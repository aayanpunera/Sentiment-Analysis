"""
Sentiment Analysis Web Application.
College Major Project - Main Application Factory.
"""

import os
import logging
from flask import Flask, render_template, jsonify, request
from config import Config
from database.db import init_db

# Configure application logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s"
)
logger = logging.getLogger(__name__)


def create_app(config_class=Config):
    """Application factory for Sentiment Analysis web application."""
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.from_object(config_class)

    # Ensure runtime directories exist
    os.makedirs(app.config.get("EXPORTS_DIR", "exports"), exist_ok=True)
    os.makedirs(app.config.get("UPLOADS_DIR", "uploads"), exist_ok=True)
    os.makedirs(os.path.dirname(app.config.get("DATABASE_PATH", "database/sentiment.db")), exist_ok=True)

    # Initialize SQLite schema
    with app.app_context():
        init_db(app.config.get("DATABASE_PATH"))

    # Register Blueprints
    from routes.dashboard import dashboard_bp
    from routes.analyzer import analyzer_bp
    from routes.bulk import bulk_bp
    from routes.history import history_bp
    from routes.performance import performance_bp
    from routes.emotion import emotion_bp
    from routes.reports import reports_bp

    app.register_blueprint(dashboard_bp)
    app.register_blueprint(analyzer_bp)
    app.register_blueprint(bulk_bp)
    app.register_blueprint(history_bp)
    app.register_blueprint(performance_bp)
    app.register_blueprint(emotion_bp)
    app.register_blueprint(reports_bp)

    # Global Error Handlers
    @app.errorhandler(404)
    def handle_not_found(e):
        if request.path.startswith("/api/"):
            return jsonify({"status": "error", "message": "Resource not found."}), 404
        return render_template("base.html", error_message="Page Not Found (404)"), 404

    @app.errorhandler(500)
    def handle_server_error(e):
        logger.exception("Unhandled server error encountered")
        if request.path.startswith("/api/"):
            return jsonify({"status": "error", "message": "Internal server error occurred."}), 500
        return render_template("base.html", error_message="An internal server error occurred."), 500

    logger.info("Application initialized successfully with all Blueprints registered.")
    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
