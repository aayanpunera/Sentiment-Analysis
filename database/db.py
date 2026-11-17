"""
Database Access Layer for SQLite3.

Handles schema initialization, transaction safety, foreign key enforcement,
parameterized queries, and persistence for analysis history, batches, and model metrics.
"""

import os
import sqlite3
import json
import logging
import shutil
import tempfile
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(BASE_DIR, "database")
IS_VERCEL = bool(os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))

if IS_VERCEL:
    DEFAULT_DB_PATH = os.path.join(tempfile.gettempdir(), "sentiment.db")
else:
    DEFAULT_DB_PATH = os.path.join(DB_DIR, "sentiment.db")

SCHEMA_PATH = os.path.join(DB_DIR, "schema.sql")
METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics.json")
PREPACKAGED_DB_PATH = os.path.join(DB_DIR, "sentiment.db")


def _connect(path: str) -> sqlite3.Connection:
    """Internal helper to create connection and set pragmas."""
    conn = sqlite3.connect(path, timeout=10.0)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn


def get_db_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """
    Establish a connection with foreign key constraints enabled
    and row_factory configured for dictionary-style column access.
    """
    path = db_path or os.environ.get("DATABASE_PATH", DEFAULT_DB_PATH)

    # In serverless environments, ensure DB exists in /tmp
    if IS_VERCEL and path != ":memory:" and not os.path.exists(path):
        init_db(path)

    return _connect(path)


def init_db(db_path: Optional[str] = None):
    """
    Initialize SQLite database schema and seed model performance metrics.
    Gracefully handles Vercel serverless /tmp seeding.
    """
    path = db_path or os.environ.get("DATABASE_PATH", DEFAULT_DB_PATH)

    try:
        dir_name = os.path.dirname(os.path.abspath(path))
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
    except OSError:
        pass

    # If running on Vercel and pre-packaged db exists in the repo, copy it to /tmp
    if IS_VERCEL and path != PREPACKAGED_DB_PATH and os.path.exists(PREPACKAGED_DB_PATH) and not os.path.exists(path):
        try:
            logger.info(f"Copying pre-packaged database from {PREPACKAGED_DB_PATH} to {path}...")
            shutil.copy2(PREPACKAGED_DB_PATH, path)
            return
        except Exception as e:
            logger.warning(f"Failed to copy pre-packaged db: {e}. Falling back to fresh schema initialization.")

    with _connect(path) as conn:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            conn.executescript(f.read())
        conn.commit()

    logger.info(f"Initialized database schema at {path}")
    sync_model_metrics_from_json(path)


def sync_model_metrics_from_json(db_path: Optional[str] = None):
    """
    Sync actual evaluation metrics from models/metrics.json into model_metrics table.
    """
    if not os.path.exists(METRICS_PATH):
        logger.warning(f"Metrics file not found at {METRICS_PATH}. Skipping metric synchronization.")
        return

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)

    with get_db_connection(db_path) as conn:
        for model_name, data in metrics_data.items():
            conn.execute(
                """
                INSERT INTO model_metrics (
                    model_name, accuracy, precision, recall, f1_score,
                    training_samples, testing_samples, trained_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(model_name) DO UPDATE SET
                    accuracy=excluded.accuracy,
                    precision=excluded.precision,
                    recall=excluded.recall,
                    f1_score=excluded.f1_score,
                    training_samples=excluded.training_samples,
                    testing_samples=excluded.testing_samples,
                    trained_at=excluded.trained_at
                """,
                (
                    model_name,
                    data["accuracy"],
                    data["precision"],
                    data["recall"],
                    data["f1_score"],
                    data.get("training_samples", 0),
                    data.get("testing_samples", 0),
                    data.get("trained_at", "")
                )
            )
        conn.commit()
    logger.info("Successfully synced model metrics to SQLite.")


def save_analysis_record(
    input_text: str,
    sentiment: str,
    model_name: str,
    positive_score: Optional[float] = None,
    negative_score: Optional[float] = None,
    neutral_score: Optional[float] = None,
    emotion: Optional[str] = None,
    analysis_type: str = "single",
    batch_id: Optional[int] = None,
    db_path: Optional[str] = None
) -> int:
    """Save an analysis record into analysis_history table."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO analysis_history (
                input_text, sentiment, model_name, positive_score,
                negative_score, neutral_score, emotion, analysis_type, batch_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                input_text, sentiment, model_name, positive_score,
                negative_score, neutral_score, emotion, analysis_type, batch_id
            )
        )
        conn.commit()
        return cursor.lastrowid


def get_analysis_history(
    page: int = 1,
    per_page: int = 20,
    sentiment_filter: Optional[str] = None,
    model_filter: Optional[str] = None,
    search_query: Optional[str] = None,
    db_path: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Retrieve paginated analysis history with optional filters and keyword search.
    Returns: (records: list, total_count: int)
    """
    offset = max(0, (page - 1) * per_page)
    where_clauses = []
    params = []

    if sentiment_filter and sentiment_filter.lower() != "all":
        where_clauses.append("sentiment = ?")
        params.append(sentiment_filter)

    if model_filter and model_filter.lower() != "all":
        where_clauses.append("model_name = ?")
        params.append(model_filter)

    if search_query and search_query.strip():
        where_clauses.append("input_text LIKE ?")
        params.append(f"%{search_query.strip()}%")

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    count_sql = f"SELECT COUNT(*) AS total FROM analysis_history {where_sql}"
    query_sql = f"""
        SELECT id, input_text, sentiment, model_name,
               positive_score, negative_score, neutral_score,
               emotion, analysis_type, batch_id, created_at
        FROM analysis_history
        {where_sql}
        ORDER BY created_at DESC, id DESC
        LIMIT ? OFFSET ?
    """

    with get_db_connection(db_path) as conn:
        total = conn.execute(count_sql, params).fetchone()["total"]
        rows = conn.execute(query_sql, params + [per_page, offset]).fetchall()
        records = [dict(r) for r in rows]

    return records, total


def delete_analysis_record(record_id: int, db_path: Optional[str] = None) -> bool:
    """Delete a single analysis history record by ID."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM analysis_history WHERE id = ?", (record_id,))
        conn.commit()
        return cursor.rowcount > 0


def get_dashboard_stats(db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Calculate live analytical statistics for the dashboard directly from SQLite.
    Never returns hardcoded or fabricated statistics.
    """
    with get_db_connection(db_path) as conn:
        # Total counts
        total_row = conn.execute("SELECT COUNT(*) AS cnt FROM analysis_history").fetchone()
        total_analyzed = total_row["cnt"] if total_row else 0

        # Distribution
        dist_rows = conn.execute(
            """
            SELECT sentiment, COUNT(*) AS count
            FROM analysis_history
            GROUP BY sentiment
            """
        ).fetchall()

        dist = {"Positive": 0, "Negative": 0, "Neutral": 0}
        for r in dist_rows:
            if r["sentiment"] in dist:
                dist[r["sentiment"]] = r["count"]

        # Recent 5 activity records
        recent_rows = conn.execute(
            """
            SELECT id, input_text, sentiment, model_name, created_at
            FROM analysis_history
            ORDER BY created_at DESC, id DESC
            LIMIT 5
            """
        ).fetchall()
        recent_activity = [dict(r) for r in recent_rows]

        # Daily trend (last 7 days of activity)
        trend_rows = conn.execute(
            """
            SELECT DATE(created_at) as date,
                   SUM(CASE WHEN sentiment = 'Positive' THEN 1 ELSE 0 END) as positive,
                   SUM(CASE WHEN sentiment = 'Negative' THEN 1 ELSE 0 END) as negative,
                   SUM(CASE WHEN sentiment = 'Neutral' THEN 1 ELSE 0 END) as neutral
            FROM analysis_history
            WHERE created_at >= DATE('now', '-7 days')
            GROUP BY DATE(created_at)
            ORDER BY date ASC
            """
        ).fetchall()
        trend_data = [dict(r) for r in trend_rows]

        # Total batches processed
        batch_count_row = conn.execute("SELECT COUNT(*) as cnt FROM batch_jobs").fetchone()
        total_batches = batch_count_row["cnt"] if batch_count_row else 0

    return {
        "total_analyzed": total_analyzed,
        "positive_count": dist["Positive"],
        "negative_count": dist["Negative"],
        "neutral_count": dist["Neutral"],
        "sentiment_distribution": dist,
        "recent_activity": recent_activity,
        "sentiment_trends": trend_data,
        "total_batches": total_batches
    }


def create_batch_job(
    filename: str,
    total_records: int,
    processed_records: int,
    model_name: str,
    status: str = "completed",
    db_path: Optional[str] = None
) -> int:
    """Create a new batch job tracking record."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO batch_jobs (filename, total_records, processed_records, model_name, status)
            VALUES (?, ?, ?, ?, ?)
            """,
            (filename, total_records, processed_records, model_name, status)
        )
        conn.commit()
        return cursor.lastrowid


def save_batch_results(
    batch_id: int,
    results: List[Dict[str, Any]],
    db_path: Optional[str] = None
):
    """
    Save bulk analysis results into batch_results and link to analysis_history.
    Executed inside a single database transaction.
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        for r in results:
            cursor.execute(
                """
                INSERT INTO batch_results (
                    batch_id, input_text, sentiment,
                    positive_score, negative_score, neutral_score
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    batch_id,
                    r["text"],
                    r["sentiment"],
                    r.get("positive_score"),
                    r.get("negative_score"),
                    r.get("neutral_score")
                )
            )
            # Also store in analysis_history for unified historical analytics
            cursor.execute(
                """
                INSERT INTO analysis_history (
                    input_text, sentiment, model_name,
                    positive_score, negative_score, neutral_score,
                    analysis_type, batch_id
                ) VALUES (?, ?, ?, ?, ?, ?, 'bulk', ?)
                """,
                (
                    r["text"],
                    r["sentiment"],
                    r.get("model_name", "Logistic Regression"),
                    r.get("positive_score"),
                    r.get("negative_score"),
                    r.get("neutral_score"),
                    batch_id
                )
            )
        conn.commit()


def get_batch_job(batch_id: int, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieve batch job details by ID."""
    with get_db_connection(db_path) as conn:
        row = conn.execute("SELECT * FROM batch_jobs WHERE id = ?", (batch_id,)).fetchone()
        return dict(row) if row else None


def get_batch_results(batch_id: int, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve all prediction records belonging to a batch."""
    with get_db_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, batch_id, input_text, sentiment,
                   positive_score, negative_score, neutral_score
            FROM batch_results
            WHERE batch_id = ?
            ORDER BY id ASC
            """,
            (batch_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_all_batch_jobs(limit: int = 50, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve list of recent batch jobs."""
    with get_db_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, filename, total_records, processed_records,
                   model_name, status, created_at
            FROM batch_jobs
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def save_emotion_record(
    input_text: str,
    emotion: str,
    confidence: float,
    scores_dict: Dict[str, float],
    db_path: Optional[str] = None
) -> int:
    """Save an emotion prediction record into emotion_history."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO emotion_history (input_text, emotion, confidence, scores_json)
            VALUES (?, ?, ?, ?)
            """,
            (input_text, emotion, confidence, json.dumps(scores_dict))
        )
        conn.commit()
        return cursor.lastrowid


def get_emotion_history(limit: int = 50, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve recent emotion predictions."""
    with get_db_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, input_text, emotion, confidence, scores_json, created_at
            FROM emotion_history
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (limit,)
        ).fetchall()
        records = []
        for r in rows:
            rec = dict(r)
            try:
                rec["scores"] = json.loads(rec["scores_json"])
            except Exception:
                rec["scores"] = {}
            records.append(rec)
        return records


def get_model_metrics(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve stored evaluation metrics for all trained models."""
    with get_db_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, model_name, accuracy, precision, recall, f1_score,
                   training_samples, testing_samples, trained_at
            FROM model_metrics
            ORDER BY f1_score DESC
            """
        ).fetchall()
        return [dict(r) for r in rows]
