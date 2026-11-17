"""
Unit tests for model prediction, inference service, and emotion classification.
"""

import pytest
from ml.predict import (
    predict_sentiment,
    predict_batch_sentiment,
    predict_emotion,
    ModelManager
)


def test_model_manager_available_models():
    """Ensure all required sentiment models are registered."""
    models = ModelManager.available_models()
    assert "Multinomial Naive Bayes" in models
    assert "Logistic Regression" in models
    assert "Linear SVM" in models


@pytest.mark.parametrize("model_name", [
    "Multinomial Naive Bayes",
    "Logistic Regression",
    "Linear SVM"
])
def test_predict_sentiment_positive(model_name):
    """Test positive sentiment prediction across all models."""
    res = predict_sentiment("I absolutely love this! It is wonderful and fantastic.", model_name=model_name)
    assert res["status"] == "success"
    assert res["sentiment"] in ["Positive", "Neutral", "Negative"]
    assert res["model_name"] == model_name
    assert 0.0 <= res["positive_score"] <= 1.0
    assert 0.0 <= res["negative_score"] <= 1.0
    assert 0.0 <= res["neutral_score"] <= 1.0
    # Probabilities should sum to approximately 1.0
    total_prob = res["positive_score"] + res["negative_score"] + res["neutral_score"]
    assert pytest.approx(total_prob, abs=0.02) == 1.0


@pytest.mark.parametrize("model_name", [
    "Multinomial Naive Bayes",
    "Logistic Regression",
    "Linear SVM"
])
def test_predict_sentiment_negative(model_name):
    """Test negative sentiment prediction across all models."""
    res = predict_sentiment("This is the worst experience ever, absolutely terrible and broken.", model_name=model_name)
    assert res["status"] == "success"
    assert res["sentiment"] == "Negative"
    assert res["negative_score"] > 0.4


def test_predict_sentiment_empty_input():
    """Test that empty string returns validation error instead of crashing."""
    res = predict_sentiment("")
    assert res["status"] == "error"
    assert "blank" in res["message"]


def test_predict_sentiment_unseen_vocabulary():
    """Test that completely unseen vocabulary produces a valid prediction without crashing."""
    res = predict_sentiment("Xylophone qwertyuiop zzzabc123 ultrafoobar")
    assert res["status"] == "success"
    assert res["sentiment"] in ["Positive", "Negative", "Neutral"]


def test_predict_sentiment_numbers_and_symbols():
    """Test input with numbers, emojis, and symbols."""
    res = predict_sentiment("Order #99902: 100% delivered on time! $$$ :)")
    assert res["status"] == "success"
    assert res["sentiment"] in ["Positive", "Negative", "Neutral"]


def test_predict_batch_sentiment():
    """Test batch prediction for CSV processing."""
    texts = [
        "Excellent service and friendly staff!",
        "Horrible quality, defective on arrival.",
        "The package arrived on Tuesday as scheduled."
    ]
    results = predict_batch_sentiment(texts, model_name="Logistic Regression")
    assert len(results) == 3
    assert results[0]["sentiment"] == "Positive"
    assert results[1]["sentiment"] == "Negative"
    for r in results:
        assert "positive_score" in r
        assert "negative_score" in r
        assert "neutral_score" in r


def test_predict_emotion():
    """Test dedicated emotion detection module."""
    res = predict_emotion("I am so grateful and happy for your generous support!")
    assert res["status"] == "success"
    assert "emotion" in res
    assert res["emotion"] in ["Happiness", "Sadness", "Anger", "Fear", "Surprise", "Love", "Neutral"]
    assert 0.0 <= res["confidence"] <= 1.0
    assert len(res["scores"]) == 7
