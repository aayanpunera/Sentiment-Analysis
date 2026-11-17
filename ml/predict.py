"""
Prediction Service and Model Manager.

Loads cached scikit-learn model pipelines and generates real-time predictions
for single-text and batch inputs with calibrated confidence estimates.
"""

import os
import logging
from typing import Dict, Any, List, Optional
import joblib

from ml.preprocessing import validate_text_input, INT_TO_LABEL, EMOTION_LABELS

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_FILE_MAP = {
    "Multinomial Naive Bayes": "sentiment_nb.joblib",
    "Logistic Regression": "sentiment_lr.joblib",
    "Linear SVM": "sentiment_svm.joblib"
}

DEFAULT_MODEL = "Logistic Regression"


class ModelManager:
    """
    Manages loading and memory caching of trained pipelines.
    Guarantees models are not re-read from disk on every HTTP request.
    """
    _cache: Dict[str, Any] = {}
    _emotion_model: Optional[Any] = None

    @classmethod
    def get_sentiment_model(cls, model_name: str = DEFAULT_MODEL):
        if model_name not in MODEL_FILE_MAP:
            raise ValueError(
                f"Unknown model '{model_name}'. Available models: {list(MODEL_FILE_MAP.keys())}"
            )

        if model_name not in cls._cache:
            artifact_name = MODEL_FILE_MAP[model_name]
            artifact_path = os.path.join(MODELS_DIR, artifact_name)
            if not os.path.exists(artifact_path):
                raise FileNotFoundError(
                    f"Model artifact '{artifact_name}' not found at {artifact_path}. Please train models first."
                )
            logger.info(f"Loading {model_name} from {artifact_path} into memory cache...")
            cls._cache[model_name] = joblib.load(artifact_path)

        return cls._cache[model_name]

    @classmethod
    def get_emotion_model(cls):
        if cls._emotion_model is None:
            artifact_path = os.path.join(MODELS_DIR, "emotion_model.joblib")
            if not os.path.exists(artifact_path):
                raise FileNotFoundError(
                    f"Emotion model artifact not found at {artifact_path}. Please train emotion model first."
                )
            logger.info("Loading emotion model into memory cache...")
            cls._emotion_model = joblib.load(artifact_path)
        return cls._emotion_model

    @classmethod
    def available_models(cls) -> List[str]:
        return list(MODEL_FILE_MAP.keys())


def predict_sentiment(text: str, model_name: str = DEFAULT_MODEL) -> Dict[str, Any]:
    """
    Predict sentiment for a single input text using the specified model.

    Returns structured dictionary with label, class probabilities, and explanation.
    """
    is_valid, err_msg = validate_text_input(text)
    if not is_valid:
        return {
            "status": "error",
            "message": err_msg
        }

    pipeline = ModelManager.get_sentiment_model(model_name)

    # Scikit-learn expects an iterable of documents
    inputs = [text]

    # Predict class index
    pred_idx = int(pipeline.predict(inputs)[0])
    sentiment = INT_TO_LABEL.get(pred_idx, "Neutral")

    # Extract class probability scores
    positive_score = 0.0
    negative_score = 0.0
    neutral_score = 0.0

    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(inputs)[0]
        # In TweetEval: 0=Negative, 1=Neutral, 2=Positive
        classes = list(pipeline.classes_)
        for cls_idx, prob in zip(classes, probabilities):
            if cls_idx == 0:
                negative_score = round(float(prob), 4)
            elif cls_idx == 1:
                neutral_score = round(float(prob), 4)
            elif cls_idx == 2:
                positive_score = round(float(prob), 4)

    # Overall confidence is the probability of the predicted class
    confidence_map = {
        "Positive": positive_score,
        "Negative": negative_score,
        "Neutral": neutral_score
    }
    confidence = confidence_map.get(sentiment, 0.0)

    return {
        "status": "success",
        "text": text,
        "model_name": model_name,
        "sentiment": sentiment,
        "positive_score": positive_score,
        "negative_score": negative_score,
        "neutral_score": neutral_score,
        "confidence": confidence,
        "confidence_percent": round(confidence * 100, 1),
        "disclaimer": "Model confidence scores are statistical estimates, not absolute guarantees."
    }


def predict_batch_sentiment(texts: List[str], model_name: str = DEFAULT_MODEL) -> List[Dict[str, Any]]:
    """
    Vectorized batch inference for bulk CSV reviews.
    """
    pipeline = ModelManager.get_sentiment_model(model_name)
    results = []

    # Filter and validate texts
    valid_indices = []
    cleaned_texts = []
    for idx, t in enumerate(texts):
        valid, _ = validate_text_input(t)
        if valid:
            valid_indices.append(idx)
            cleaned_texts.append(t)

    if not cleaned_texts:
        return [{"status": "error", "message": "No valid text rows found to analyze."}]

    # Predict in one vectorized call for optimal performance
    predictions = pipeline.predict(cleaned_texts)

    has_proba = hasattr(pipeline, "predict_proba")
    probas = pipeline.predict_proba(cleaned_texts) if has_proba else None
    classes = list(pipeline.classes_) if has_proba else []

    for i, original_idx in enumerate(valid_indices):
        pred_idx = int(predictions[i])
        sentiment = INT_TO_LABEL.get(pred_idx, "Neutral")

        neg_sc, neu_sc, pos_sc = 0.0, 0.0, 0.0
        if probas is not None:
            for cls_idx, prob in zip(classes, probas[i]):
                if cls_idx == 0:
                    neg_sc = round(float(prob), 4)
                elif cls_idx == 1:
                    neu_sc = round(float(prob), 4)
                elif cls_idx == 2:
                    pos_sc = round(float(prob), 4)

        conf_map = {"Positive": pos_sc, "Negative": neg_sc, "Neutral": neu_sc}
        confidence = conf_map.get(sentiment, 0.0)

        results.append({
            "index": original_idx,
            "text": cleaned_texts[i],
            "sentiment": sentiment,
            "positive_score": pos_sc,
            "negative_score": neg_sc,
            "neutral_score": neu_sc,
            "confidence": confidence,
            "confidence_percent": round(confidence * 100, 1)
        })

    return results


def predict_emotion(text: str) -> Dict[str, Any]:
    """
    Predict emotion for an input text using the trained GoEmotions classifier.
    """
    is_valid, err_msg = validate_text_input(text)
    if not is_valid:
        return {
            "status": "error",
            "message": err_msg
        }

    pipeline = ModelManager.get_emotion_model()
    pred_label = str(pipeline.predict([text])[0])

    scores = {}
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba([text])[0]
        for emo_cls, p in zip(pipeline.classes_, probs):
            scores[str(emo_cls)] = round(float(p), 4)

    confidence = scores.get(pred_label, 0.0)

    return {
        "status": "success",
        "text": text,
        "emotion": pred_label,
        "confidence": confidence,
        "confidence_percent": round(confidence * 100, 1),
        "scores": scores,
        "disclaimer": "Emotion detection maps complex affective nuances to broad psychological categories."
    }
