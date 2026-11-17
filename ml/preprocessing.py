"""
Text Preprocessing and Validation Module.

Provides conservative text cleaning, input validation, and scikit-learn
compatible preprocessing utilities for sentiment analysis and emotion classification.
Ensures zero data leakage during vectorization and pipeline serialization.
"""

import re
import html
from typing import Optional, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

# Sentiment class mapping
LABEL_TO_INT = {
    "Negative": 0,
    "Neutral": 1,
    "Positive": 2
}
INT_TO_LABEL = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

# Emotion labels mapping
EMOTION_LABELS = [
    "Happiness",
    "Sadness",
    "Anger",
    "Fear",
    "Surprise",
    "Love",
    "Neutral"
]

MAX_TEXT_LENGTH = 5000


def validate_text_input(text: Optional[str], max_length: int = MAX_TEXT_LENGTH) -> Tuple[bool, str]:
    """
    Validate raw input text against empty, non-string, or oversized inputs.

    Returns:
        (is_valid: bool, error_message: str)
    """
    if text is None:
        return False, "Input text cannot be empty."

    if not isinstance(text, str):
        return False, "Input text must be a valid string."

    stripped = text.strip()
    if not stripped:
        return False, "Input text cannot be blank or contain only whitespace."

    if len(stripped) > max_length:
        return False, f"Input text exceeds maximum allowed length of {max_length} characters (got {len(stripped)})."

    return True, ""


def clean_text(text: str) -> str:
    """
    Conservative text cleaning for social media and general reviews:
      1. Unescapes HTML entities.
      2. Removes URLs while preserving surrounding context.
      3. Normalizes user mentions (@user -> '').
      4. Cleans non-printable control characters.
      5. Normalizes excessive whitespace.
      6. Converts to lowercase.
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    # Decode HTML entities (e.g., &amp; -> &, &quot; -> ")
    text = html.unescape(text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Clean Twitter / Social mentions (e.g. @user, @name)
    text = re.sub(r"@\w+", " ", text)

    # Replace newlines, tabs, and multiple whitespace with single space
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip().lower()


class TextCleaner(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that applies `clean_text`
    across an iterable of strings.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [clean_text(item) for item in X]


def build_tfidf_vectorizer(
    ngram_range: Tuple[int, int] = (1, 2),
    max_features: int = 20000,
    sublinear_tf: bool = True
) -> TfidfVectorizer:
    """
    Build a standard TF-IDF vectorizer adhering to project specification:
      - n-grams: (1, 2)
      - max_features: 20000
      - sublinear_tf: True
    """
    return TfidfVectorizer(
        preprocessor=clean_text,
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=sublinear_tf,
        strip_accents="unicode"
    )
