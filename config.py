"""
Application Configuration.

Loads settings from environment variables or uses safe production-ready defaults.
"""

import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-sentiment-secret-key-purple-analytics-2026")
    DATABASE_PATH = os.environ.get("DATABASE_PATH", os.path.join(BASE_DIR, "database", "sentiment.db"))
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    EXPORTS_DIR = os.path.join(BASE_DIR, "exports")
    UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")

    # CSV Upload security limits
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload
    ALLOWED_EXTENSIONS = {"csv"}
    MAX_BATCH_ROWS = 5000  # Safe cap on batch rows per file

    # Static assets
    STATIC_FOLDER = os.path.join(BASE_DIR, "static")
    TEMPLATES_FOLDER = os.path.join(BASE_DIR, "templates")


class TestConfig(Config):
    TESTING = True
    DATABASE_PATH = ":memory:"
