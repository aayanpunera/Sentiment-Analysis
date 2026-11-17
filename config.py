"""
Application Configuration.

Loads settings from environment variables or uses safe production-ready defaults.
Automatically configures paths for local development and Vercel serverless environments.
"""

import os
import tempfile
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Detect Vercel serverless environment
IS_VERCEL = bool(os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))

# Serverless environments have a strictly read-only filesystem except /tmp
TMP_DIR = tempfile.gettempdir()
if IS_VERCEL:
    os.environ["MPLCONFIGDIR"] = os.path.join(TMP_DIR, "matplotlib")


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-sentiment-secret-key-purple-analytics-2026")
    
    # Database path: /tmp on Vercel, local file in database/ for standard execution
    if IS_VERCEL:
        DATABASE_PATH = os.environ.get("DATABASE_PATH", os.path.join(TMP_DIR, "sentiment.db"))
        EXPORTS_DIR = os.path.join(TMP_DIR, "exports")
        UPLOADS_DIR = os.path.join(TMP_DIR, "uploads")
    else:
        DATABASE_PATH = os.environ.get("DATABASE_PATH", os.path.join(BASE_DIR, "database", "sentiment.db"))
        EXPORTS_DIR = os.path.join(BASE_DIR, "exports")
        UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")

    MODELS_DIR = os.path.join(BASE_DIR, "models")

    # CSV Upload security limits
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload
    ALLOWED_EXTENSIONS = {"csv"}
    MAX_BATCH_ROWS = 5000  # Safe cap on batch rows per file

    # Static assets and template folders
    STATIC_FOLDER = os.path.join(BASE_DIR, "static")
    TEMPLATES_FOLDER = os.path.join(BASE_DIR, "templates")


class TestConfig(Config):
    TESTING = True
    DATABASE_PATH = ":memory:"
