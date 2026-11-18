"""
Vercel Serverless Function Entry Point.

Exposes the Flask WSGI application instance `app` with VercelPathFixMiddleware
to ensure clean URL path routing and static file handling on Vercel.
"""

import os
import sys

# Ensure project root directory is on Python search path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app import app


class VercelPathFixMiddleware:
    """
    WSGI middleware to resolve PATH_INFO issues when deployed on Vercel.
    Corrects rewritten paths while preserving actual requested routes.
    """

    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        # 1. Prefer Vercel's x-matched-path header
        matched_path = environ.get("HTTP_X_MATCHED_PATH") or environ.get("HTTP_X_VERCEL_MATCHED_PATH")

        if matched_path:
            # If matched_path starts with /api/index, strip it to get the real route
            if matched_path.startswith("/api/index"):
                stripped = matched_path[len("/api/index"):]
                environ["PATH_INFO"] = stripped if (stripped.startswith("/") or not stripped) else "/" + stripped
            else:
                environ["PATH_INFO"] = matched_path
        else:
            path_info = environ.get("PATH_INFO", "")
            if path_info.startswith("/api/index"):
                stripped = path_info[len("/api/index"):]
                environ["PATH_INFO"] = stripped if (stripped.startswith("/") or not stripped) else "/" + stripped
            elif path_info.startswith("/api.py"):
                stripped = path_info[len("/api.py"):]
                environ["PATH_INFO"] = stripped if (stripped.startswith("/") or not stripped) else "/" + stripped
            elif path_info == "/api" or path_info == "/api/":
                environ["PATH_INFO"] = "/"
            elif path_info.startswith("/api/"):
                real_apis = (
                    "/api/predict",
                    "/api/bulk-analyze",
                    "/api/history",
                    "/api/metrics",
                    "/api/emotion",
                    "/api/stats",
                    "/api/batch"
                )
                if not any(path_info.startswith(ra) for ra in real_apis):
                    stripped = path_info[len("/api"):]
                    environ["PATH_INFO"] = stripped if (stripped.startswith("/") or not stripped) else "/" + stripped

        if not environ.get("PATH_INFO") or environ["PATH_INFO"] == "":
            environ["PATH_INFO"] = "/"

        return self.wsgi_app(environ, start_response)


# Wrap Flask's WSGI application with the path fix middleware
app.wsgi_app = VercelPathFixMiddleware(app.wsgi_app)
