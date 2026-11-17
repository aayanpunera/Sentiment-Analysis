"""
Vercel Serverless Function Entry Point.

Exposes the Flask WSGI application instance `app` for Vercel deployment.
"""

import os
import sys

# Ensure project root directory is on Python search path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app import app
