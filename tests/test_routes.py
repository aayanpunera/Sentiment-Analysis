"""
End-to-End Route and API Integration Tests for the Flask Web Application.
"""

import io
import pytest
from app import create_app
from config import Config
from database.db import init_db


class TestRoutesConfig(Config):
    TESTING = True
    DATABASE_PATH = "database/test_routes.db"


@pytest.fixture(scope="module")
def app():
    """Create test application and initialize test database."""
    test_app = create_app(TestRoutesConfig)
    with test_app.app_context():
        init_db(TestRoutesConfig.DATABASE_PATH)
    yield test_app


@pytest.fixture(scope="module")
def client(app):
    """Flask test client."""
    return app.test_client()


def test_dashboard_route(client):
    """Test dashboard index page rendering."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Analytics Overview" in response.data
    assert b"Total Texts Analyzed" in response.data


def test_analyzer_page(client):
    """Test analyzer page rendering."""
    response = client.get("/analyze")
    assert response.status_code == 200
    assert b"Single Text Analyzer" in response.data
    assert b"Logistic Regression" in response.data


def test_api_predict_success(client):
    """Test API single text prediction with valid input."""
    payload = {
        "text": "This application works exceptionally well! Outstanding job.",
        "model_name": "Logistic Regression"
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["sentiment"] == "Positive"
    assert "positive_score" in data
    assert "negative_score" in data
    assert "neutral_score" in data
    assert data["record_id"] > 0


def test_api_predict_empty_text(client):
    """Test API error handling for empty text."""
    payload = {"text": "   ", "model_name": "Logistic Regression"}
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 400
    data = response.get_json()
    assert data["status"] == "error"
    assert "blank" in data["message"]


def test_api_predict_invalid_model(client):
    """Test API error handling for unsupported model."""
    payload = {"text": "Some text", "model_name": "NonExistentModel123"}
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 400
    data = response.get_json()
    assert "Invalid model" in data["message"]


def test_bulk_page(client):
    """Test bulk page rendering."""
    response = client.get("/bulk")
    assert response.status_code == 200
    assert b"Bulk CSV Dataset Analyzer" in response.data


def test_api_bulk_analyze(client):
    """Test bulk CSV upload and analysis."""
    csv_content = (
        "review\n"
        "\"Fantastic product! Loved it so much.\"\n"
        "\"Terrible quality, broke on day one.\"\n"
        "\"Package was delivered at 2pm on Monday.\"\n"
    )
    data = {
        "file": (io.BytesIO(csv_content.encode("utf-8")), "test_reviews.csv"),
        "model_name": "Logistic Regression"
    }
    response = client.post("/api/bulk-analyze", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    result = response.get_json()
    assert result["status"] == "success"
    assert result["total_records"] == 3
    assert result["processed_records"] == 3
    assert "Positive" in result["summary"]


def test_api_bulk_analyze_no_file(client):
    """Test bulk upload error with no file provided."""
    response = client.post("/api/bulk-analyze", data={}, content_type="multipart/form-data")
    assert response.status_code == 400


def test_history_page_and_api(client):
    """Test history page and history retrieval endpoint."""
    page_resp = client.get("/history")
    assert page_resp.status_code == 200
    assert b"Analysis History" in page_resp.data

    api_resp = client.get("/api/history?page=1&per_page=10")
    assert api_resp.status_code == 200
    data = api_resp.get_json()
    assert data["status"] == "success"
    assert data["total"] > 0
    assert len(data["records"]) > 0


def test_performance_page_and_api(client):
    """Test model performance page and API metrics."""
    page_resp = client.get("/performance")
    assert page_resp.status_code == 200
    assert b"Algorithm Performance" in page_resp.data

    api_resp = client.get("/api/metrics")
    assert api_resp.status_code == 200
    data = api_resp.get_json()
    assert "Logistic Regression" in data


def test_emotion_page_and_api(client):
    """Test emotion detection page and API."""
    page_resp = client.get("/emotion")
    assert page_resp.status_code == 200
    assert b"Emotion Classification" in page_resp.data

    payload = {"text": "I am so happy and delighted with this victory!"}
    api_resp = client.post("/api/emotion", json=payload)
    assert api_resp.status_code == 200
    data = api_resp.get_json()
    assert data["status"] == "success"
    assert "emotion" in data
    assert "scores" in data


def test_reports_page_and_exports(client):
    """Test reports page, CSV export, and PDF export."""
    page_resp = client.get("/reports")
    assert page_resp.status_code == 200
    assert b"Reports &amp; Data Exports" in page_resp.data

    # Test CSV export
    csv_resp = client.get("/reports/csv")
    assert csv_resp.status_code == 200
    assert "text/csv" in csv_resp.content_type
    assert b"Input_Text" in csv_resp.data

    # Test PDF export
    pdf_resp = client.get("/reports/pdf")
    assert pdf_resp.status_code == 200
    assert "application/pdf" in pdf_resp.content_type
    # PDF starts with %PDF header
    assert pdf_resp.data.startswith(b"%PDF")


def test_404_error_handling(client):
    """Test 404 response for HTML and API requests."""
    html_resp = client.get("/nonexistent-page-xyz")
    assert html_resp.status_code == 404

    api_resp = client.get("/api/nonexistent-endpoint")
    assert api_resp.status_code == 404
    data = api_resp.get_json()
    assert data["status"] == "error"
