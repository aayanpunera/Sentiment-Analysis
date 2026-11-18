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
        # Prefer original requested URI if provided by proxy headers
        orig_uri = environ.get("HTTP_X_FORWARDED_URI") or environ.get("HTTP_X_ORIGINAL_URI") or environ.get("RAW_URI")
        if orig_uri and not orig_uri.startswith("/api/index"):
            environ["PATH_INFO"] = orig_uri.split("?")[0]
        else:
            path_info = environ.get("PATH_INFO", "")
            if path_info.startswith("/api/index"):
                stripped = path_info[len("/api/index"):]
                environ["PATH_INFO"] = stripped if (stripped.startswith("/") or not stripped) else "/" + stripped
            elif path_info.startswith("/api.py"):
                stripped = path_info[len("/api.py"):]
                environ["PATH_INFO"] = stripped if (stripped.startswith("/") or not stripped) else "/" + stripped

        if not environ.get("PATH_INFO") or environ["PATH_INFO"] == "":
            environ["PATH_INFO"] = "/"

        return self.wsgi_app(environ, start_response)


# Wrap Flask's WSGI application with the path fix middleware
app.wsgi_app = VercelPathFixMiddleware(app.wsgi_app)
