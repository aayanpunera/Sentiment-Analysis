"""
Unit tests for text preprocessing, validation, and TF-IDF pipeline components.
"""

import pytest
from ml.preprocessing import (
    clean_text,
    validate_text_input,
    TextCleaner,
    build_tfidf_vectorizer,
    LABEL_TO_INT,
    INT_TO_LABEL
)


def test_validate_text_input_valid():
    """Ensure standard sentences pass validation."""
    valid, msg = validate_text_input("This product is really amazing!")
    assert valid is True
    assert msg == ""


def test_validate_text_input_empty():
    """Ensure empty or blank strings fail validation with helpful message."""
    valid, msg = validate_text_input("")
    assert valid is False
    assert "cannot be blank" in msg

    valid, msg = validate_text_input("   \t\n  ")
    assert valid is False
    assert "cannot be blank" in msg

    valid, msg = validate_text_input(None)
    assert valid is False
    assert "cannot be empty" in msg


def test_validate_text_input_oversized():
    """Ensure strings exceeding max length fail validation."""
    long_text = "word " * 1200  # > 5000 chars
    valid, msg = validate_text_input(long_text, max_length=5000)
    assert valid is False
    assert "exceeds maximum allowed length" in msg


def test_clean_text_html_entities():
    """Test unescaping of HTML entities."""
    raw = "Rock &amp; Roll is &lt;awesome&gt; &quot;yes&quot;"
    cleaned = clean_text(raw)
    assert "&amp;" not in cleaned
    assert "rock & roll is <awesome> \"yes\"" in cleaned


def test_clean_text_urls_and_mentions():
    """Test stripping of URLs and user handles."""
    raw = "Check this out @steve https://example.com/test and www.greatsite.org!"
    cleaned = clean_text(raw)
    assert "@steve" not in cleaned
    assert "http" not in cleaned
    assert "www." not in cleaned
    assert "check this out" in cleaned


def test_clean_text_whitespace_normalization():
    """Test reduction of duplicate spaces, tabs, and newlines."""
    raw = "Lots   of \t\t spaces   and \n\n newlines."
    cleaned = clean_text(raw)
    assert cleaned == "lots of spaces and newlines."


def test_clean_text_special_characters_and_numbers():
    """Test processing of numbers and special characters without crash."""
    raw = "Order #12345: cost was $99.99 for 10 items!!! :) <3"
    cleaned = clean_text(raw)
    assert "12345" in cleaned
    assert "$99.99" in cleaned


def test_text_cleaner_transformer():
    """Test scikit-learn compatible transformer."""
    transformer = TextCleaner()
    docs = [
        "First @user test http://foo.bar",
        "Second &amp; final test!"
    ]
    transformed = transformer.transform(docs)
    assert len(transformed) == 2
    assert "@user" not in transformed[0]
    assert "&amp;" not in transformed[1]


def test_tfidf_vectorizer_fit_and_unseen():
    """Ensure TF-IDF vectorizer fits on training text and transforms unseen vocab."""
    corpus = [
        "Great quality and fast delivery.",
        "Terrible experience, very bad.",
        "Average product, it was okay."
    ]
    vec = build_tfidf_vectorizer(ngram_range=(1, 2), max_features=100)
    X = vec.fit_transform(corpus)
    assert X.shape[0] == 3
    assert X.shape[1] > 0

    # Unseen vocabulary should transform safely without error
    unseen = ["Completely new unexpected vocabulary words zzzxyz"]
    X_unseen = vec.transform(unseen)
    assert X_unseen.shape[0] == 1


def test_label_mappings():
    """Ensure bidirectional label mappings are consistent."""
    for label, idx in LABEL_TO_INT.items():
        assert INT_TO_LABEL[idx] == label
