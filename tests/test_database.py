"""
Unit tests for SQLite database operations, schema integrity, and analytics queries.
"""

import os
import pytest
from database.db import (
    init_db,
    save_analysis_record,
    get_analysis_history,
    delete_analysis_record,
    get_dashboard_stats,
    create_batch_job,
    save_batch_results,
    get_batch_job,
    get_batch_results,
    save_emotion_record,
    get_emotion_history,
    get_model_metrics
)


@pytest.fixture
def temp_db(tmp_path):
    """Create and initialize a clean test SQLite database."""
    db_file = str(tmp_path / "test_sentiment.db")
    init_db(db_file)
    return db_file


def test_init_db_and_metrics(temp_db):
    """Ensure database initializes and syncs model metrics."""
    metrics = get_model_metrics(temp_db)
    assert len(metrics) >= 3
    model_names = [m["model_name"] for m in metrics]
    assert "Logistic Regression" in model_names
    assert "Multinomial Naive Bayes" in model_names
    assert "Linear SVM" in model_names


def test_save_and_get_analysis_history(temp_db):
    """Test saving a single record and retrieving it."""
    rec_id = save_analysis_record(
        input_text="Great experience with customer service!",
        sentiment="Positive",
        model_name="Logistic Regression",
        positive_score=0.88,
        negative_score=0.04,
        neutral_score=0.08,
        db_path=temp_db
    )
    assert rec_id > 0

    records, total = get_analysis_history(page=1, per_page=10, db_path=temp_db)
    assert total == 1
    assert len(records) == 1
    assert records[0]["sentiment"] == "Positive"
    assert records[0]["positive_score"] == 0.88


def test_history_filtering_and_search(temp_db):
    """Test filtering by sentiment, model, and search query."""
    save_analysis_record("I loved the fast shipping", "Positive", "Logistic Regression", db_path=temp_db)
    save_analysis_record("Defective item, returned it", "Negative", "Linear SVM", db_path=temp_db)
    save_analysis_record("Normal package arrived", "Neutral", "Multinomial Naive Bayes", db_path=temp_db)

    # Filter by sentiment
    pos_recs, pos_total = get_analysis_history(sentiment_filter="Positive", db_path=temp_db)
    assert pos_total == 1
    assert pos_recs[0]["sentiment"] == "Positive"

    # Search keyword
    search_recs, search_total = get_analysis_history(search_query="defective", db_path=temp_db)
    assert search_total == 1
    assert "Defective" in search_recs[0]["input_text"]


def test_delete_analysis_record(temp_db):
    """Test deleting an analysis history record."""
    rec_id = save_analysis_record("To be deleted", "Neutral", "Logistic Regression", db_path=temp_db)
    records, total = get_analysis_history(db_path=temp_db)
    assert total == 1

    deleted = delete_analysis_record(rec_id, db_path=temp_db)
    assert deleted is True

    records_after, total_after = get_analysis_history(db_path=temp_db)
    assert total_after == 0


def test_dashboard_stats(temp_db):
    """Test calculation of live dashboard analytics."""
    stats_empty = get_dashboard_stats(temp_db)
    assert stats_empty["total_analyzed"] == 0
    assert stats_empty["positive_count"] == 0

    save_analysis_record("Awesome!", "Positive", "Logistic Regression", db_path=temp_db)
    save_analysis_record("Awful!", "Negative", "Logistic Regression", db_path=temp_db)
    save_analysis_record("Okay.", "Neutral", "Logistic Regression", db_path=temp_db)

    stats = get_dashboard_stats(temp_db)
    assert stats["total_analyzed"] == 3
    assert stats["positive_count"] == 1
    assert stats["negative_count"] == 1
    assert stats["neutral_count"] == 1
    assert len(stats["recent_activity"]) == 3


def test_batch_job_and_results(temp_db):
    """Test batch job tracking and bulk results persistence."""
    batch_id = create_batch_job(
        filename="reviews.csv",
        total_records=2,
        processed_records=2,
        model_name="Logistic Regression",
        db_path=temp_db
    )
    assert batch_id > 0

    results = [
        {"text": "Sample 1", "sentiment": "Positive", "positive_score": 0.9, "negative_score": 0.05, "neutral_score": 0.05},
        {"text": "Sample 2", "sentiment": "Negative", "positive_score": 0.1, "negative_score": 0.8, "neutral_score": 0.1}
    ]
    save_batch_results(batch_id, results, db_path=temp_db)

    batch = get_batch_job(batch_id, db_path=temp_db)
    assert batch["filename"] == "reviews.csv"
    assert batch["total_records"] == 2

    batch_rows = get_batch_results(batch_id, db_path=temp_db)
    assert len(batch_rows) == 2
    assert batch_rows[0]["sentiment"] == "Positive"


def test_emotion_history(temp_db):
    """Test emotion history storage and retrieval."""
    emo_id = save_emotion_record(
        input_text="I am so thrilled!",
        emotion="Happiness",
        confidence=0.92,
        scores_dict={"Happiness": 0.92, "Sadness": 0.01},
        db_path=temp_db
    )
    assert emo_id > 0

    history = get_emotion_history(limit=10, db_path=temp_db)
    assert len(history) == 1
    assert history[0]["emotion"] == "Happiness"
    assert history[0]["scores"]["Happiness"] == 0.92
