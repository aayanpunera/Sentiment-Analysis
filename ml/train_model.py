"""
Model Training and Evaluation Pipeline.

Trains, evaluates, and serializes:
  1. Multinomial Naive Bayes
  2. Logistic Regression
  3. Linear SVM (Calibrated for probability estimation)
  4. Emotion Detection Model (GoEmotions 7-class taxonomy)

Uses real benchmark data (TweetEval Sentiment and GoEmotions),
ensures zero data leakage via scikit-learn Pipelines, evaluates on held-out test splits,
and exports model artifacts and performance charts.
"""

import os
import time
from datetime import datetime
import logging
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

from ml.preprocessing import build_tfidf_vectorizer, INT_TO_LABEL
from ml.evaluate import compute_metrics, generate_comparison_chart, save_metrics_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
STATIC_IMG_DIR = os.path.join(BASE_DIR, "static", "images")


def load_sentiment_data():
    """Load TweetEval train and test splits."""
    train_path = os.path.join(DATASET_DIR, "sentiment_train.csv")
    test_path = os.path.join(DATASET_DIR, "sentiment_test.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Sentiment datasets not found in dataset/ directory. Run download_datasets.py first.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Ensure clean text and drop any empty rows
    train_df = train_df.dropna(subset=["text", "label"])
    test_df = test_df.dropna(subset=["text", "label"])

    X_train, y_train = train_df["text"].tolist(), train_df["label"].tolist()
    X_test, y_test = test_df["text"].tolist(), test_df["label"].tolist()

    logger.info(f"Loaded Sentiment Data: {len(X_train)} train samples, {len(X_test)} test samples.")
    return X_train, y_train, X_test, y_test


def build_sentiment_pipelines():
    """Create the 3 candidate model pipelines with TF-IDF vectorization."""
    return {
        "Multinomial Naive Bayes": Pipeline([
            ("tfidf", build_tfidf_vectorizer(ngram_range=(1, 2), max_features=25000)),
            ("clf", MultinomialNB(alpha=0.5))
        ]),
        "Logistic Regression": Pipeline([
            ("tfidf", build_tfidf_vectorizer(ngram_range=(1, 2), max_features=25000)),
            ("clf", LogisticRegression(max_iter=1000, random_state=42, C=1.0, class_weight="balanced"))
        ]),
        "Linear SVM": Pipeline([
            ("tfidf", build_tfidf_vectorizer(ngram_range=(1, 2), max_features=25000)),
            ("clf", CalibratedClassifierCV(
                estimator=LinearSVC(random_state=42, dual=False, max_iter=2000, C=1.0),
                cv=3
            ))
        ])
    }


def train_sentiment_models():
    """Train, evaluate, and save sentiment models."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(STATIC_IMG_DIR, exist_ok=True)

    X_train, y_train, X_test, y_test = load_sentiment_data()
    target_names = [INT_TO_LABEL[i] for i in sorted(INT_TO_LABEL.keys())]

    pipelines = build_sentiment_pipelines()
    all_metrics = {}

    artifact_filenames = {
        "Multinomial Naive Bayes": "sentiment_nb.joblib",
        "Logistic Regression": "sentiment_lr.joblib",
        "Linear SVM": "sentiment_svm.joblib"
    }

    for name, pipeline in pipelines.items():
        logger.info(f"Training {name}...")
        start_time = time.time()
        pipeline.fit(X_train, y_train)
        train_duration = round(time.time() - start_time, 2)
        logger.info(f"Training {name} completed in {train_duration}s.")

        logger.info(f"Evaluating {name} on held-out test set ({len(X_test)} samples)...")
        y_pred = pipeline.predict(X_test)
        metrics = compute_metrics(y_test, y_pred, target_names)
        metrics["training_samples"] = len(X_train)
        metrics["testing_samples"] = len(X_test)
        metrics["train_duration_sec"] = train_duration
        metrics["trained_at"] = datetime.now().isoformat()

        logger.info(f"{name} Results -> Accuracy: {metrics['accuracy']}, Macro F1: {metrics['f1_score']}")
        all_metrics[name] = metrics

        # Save pipeline artifact
        artifact_path = os.path.join(MODELS_DIR, artifact_filenames[name])
        joblib.dump(pipeline, artifact_path)
        logger.info(f"Saved {name} pipeline to {artifact_path}")

    # Save metrics JSON
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    save_metrics_json(all_metrics, metrics_path)

    # Generate and save comparison chart
    chart_path = os.path.join(MODELS_DIR, "comparison_chart.png")
    static_chart_path = os.path.join(STATIC_IMG_DIR, "comparison_chart.png")
    generate_comparison_chart(all_metrics, chart_path)
    generate_comparison_chart(all_metrics, static_chart_path)

    return all_metrics


def train_emotion_model():
    """Train and evaluate the dedicated emotion detection model using GoEmotions."""
    emotion_csv = os.path.join(DATASET_DIR, "emotion_dataset.csv")
    if not os.path.exists(emotion_csv):
        logger.warning("Emotion dataset not found. Skipping emotion training.")
        return None

    df = pd.read_csv(emotion_csv).dropna(subset=["text", "emotion"])
    logger.info(f"Loaded Emotion Dataset: {len(df)} records across {df['emotion'].nunique()} categories.")

    X = df["text"].tolist()
    y = df["emotion"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    categories = sorted(list(set(y)))

    emotion_pipeline = Pipeline([
        ("tfidf", build_tfidf_vectorizer(ngram_range=(1, 2), max_features=25000)),
        ("clf", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"))
    ])

    logger.info("Training Emotion Classifier...")
    start_time = time.time()
    emotion_pipeline.fit(X_train, y_train)
    duration = round(time.time() - start_time, 2)
    logger.info(f"Emotion classifier trained in {duration}s.")

    y_pred = emotion_pipeline.predict(X_test)
    y_test_idx = [categories.index(lbl) for lbl in y_test]
    y_pred_idx = [categories.index(lbl) for lbl in y_pred]

    metrics = compute_metrics(y_test_idx, y_pred_idx, categories)
    metrics["training_samples"] = len(X_train)
    metrics["testing_samples"] = len(X_test)
    metrics["train_duration_sec"] = duration
    metrics["trained_at"] = datetime.now().isoformat()
    metrics["categories"] = categories

    logger.info(f"Emotion Classifier -> Accuracy: {metrics['accuracy']}, Macro F1: {metrics['f1_score']}")

    # Save artifact
    emo_path = os.path.join(MODELS_DIR, "emotion_model.joblib")
    joblib.dump(emotion_pipeline, emo_path)
    logger.info(f"Saved emotion pipeline to {emo_path}")

    # Save metrics
    emo_metrics_path = os.path.join(MODELS_DIR, "emotion_metrics.json")
    save_metrics_json(metrics, emo_metrics_path)

    return metrics


if __name__ == "__main__":
    logger.info("Starting model training pipeline...")
    train_sentiment_models()
    train_emotion_model()
    logger.info("All model training and evaluations completed successfully.")
