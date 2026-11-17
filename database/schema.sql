-- Sentiment Analysis Application Database Schema
-- SQLite3 with Foreign Keys Enabled

PRAGMA foreign_keys = ON;

-- 1. Batch jobs tracking
CREATE TABLE IF NOT EXISTS batch_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    total_records INTEGER NOT NULL,
    processed_records INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'completed',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 2. Analysis history for single and bulk text predictions
CREATE TABLE IF NOT EXISTS analysis_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    input_text TEXT NOT NULL,
    sentiment TEXT NOT NULL,
    model_name TEXT NOT NULL,
    positive_score REAL,
    negative_score REAL,
    neutral_score REAL,
    emotion TEXT,
    analysis_type TEXT NOT NULL DEFAULT 'single',
    batch_id INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (batch_id) REFERENCES batch_jobs (id) ON DELETE SET NULL
);

-- 3. Detailed batch results
CREATE TABLE IF NOT EXISTS batch_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id INTEGER NOT NULL,
    input_text TEXT NOT NULL,
    sentiment TEXT NOT NULL,
    positive_score REAL,
    negative_score REAL,
    neutral_score REAL,
    FOREIGN KEY (batch_id) REFERENCES batch_jobs (id) ON DELETE CASCADE
);

-- 4. Persisted model evaluation metrics
CREATE TABLE IF NOT EXISTS model_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL UNIQUE,
    accuracy REAL NOT NULL,
    precision REAL NOT NULL,
    recall REAL NOT NULL,
    f1_score REAL NOT NULL,
    training_samples INTEGER NOT NULL,
    testing_samples INTEGER NOT NULL,
    trained_at DATETIME NOT NULL
);

-- 5. Dedicated emotion history table
CREATE TABLE IF NOT EXISTS emotion_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    input_text TEXT NOT NULL,
    emotion TEXT NOT NULL,
    confidence REAL,
    scores_json TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_history_sentiment ON analysis_history (sentiment);
CREATE INDEX IF NOT EXISTS idx_history_model ON analysis_history (model_name);
CREATE INDEX IF NOT EXISTS idx_history_created_at ON analysis_history (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_batch_results_batch_id ON batch_results (batch_id);
CREATE INDEX IF NOT EXISTS idx_emotion_created_at ON emotion_history (created_at DESC);
