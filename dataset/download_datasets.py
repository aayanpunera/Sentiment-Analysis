"""
Dataset Download & Preparation Script for Sentiment Analysis and Emotion Detection.

This script fetches the official TweetEval (sentiment configuration) benchmark dataset
from the Cardiff NLP repository and a structured subset of GoEmotions.
It formats and stores:
  - dataset/sentiment_train.csv
  - dataset/sentiment_val.csv
  - dataset/sentiment_test.csv
  - dataset/sentiment_dataset.csv (combined full dataset with split metadata)
  - dataset/emotion_dataset.csv
"""

import os
import urllib.request
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TWEETEVAL_BASE_URL = "https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment"
LABEL_MAPPING = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

EMOTION_SOURCE_URL = "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/train.tsv"

# Broad 7 emotion categories mapping as per project specification:
# Happiness, Sadness, Anger, Fear, Surprise, Love, Neutral
GOEMOTIONS_TAXONOMY = {
    "admiration": "Happiness",
    "amusement": "Happiness",
    "approval": "Happiness",
    "caring": "Love",
    "desire": "Love",
    "excitement": "Happiness",
    "gratitude": "Happiness",
    "joy": "Happiness",
    "love": "Love",
    "optimism": "Happiness",
    "pride": "Happiness",
    "relief": "Happiness",
    "anger": "Anger",
    "annoyance": "Anger",
    "disapproval": "Anger",
    "disgust": "Anger",
    "embarrassment": "Sadness",
    "fear": "Fear",
    "nervousness": "Fear",
    "grief": "Sadness",
    "remorse": "Sadness",
    "sadness": "Sadness",
    "disappointment": "Sadness",
    "surprise": "Surprise",
    "realization": "Surprise",
    "curiosity": "Surprise",
    "confusion": "Surprise",
    "neutral": "Neutral"
}


def download_file(url: str, dest_path: str):
    """Download a file with user-agent headers to avoid rate limiting."""
    logger.info(f"Downloading {url} -> {dest_path}")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "SentimentAnalysisProject/1.0"}
    )
    with urllib.request.urlopen(req, timeout=30) as response, open(dest_path, "wb") as out_file:
        out_file.write(response.read())


def load_tweeteval_split(base_dir: str, split_name: str) -> pd.DataFrame:
    """Download and load TweetEval text and label files for a specific split."""
    text_url = f"{TWEETEVAL_BASE_URL}/{split_name}_text.txt"
    label_url = f"{TWEETEVAL_BASE_URL}/{split_name}_labels.txt"

    text_file = os.path.join(base_dir, f"{split_name}_text.txt")
    label_file = os.path.join(base_dir, f"{split_name}_labels.txt")

    if not os.path.exists(text_file):
        download_file(text_url, text_file)
    if not os.path.exists(label_file):
        download_file(label_url, label_file)

    with open(text_file, "r", encoding="utf-8", errors="replace") as f:
        texts = [line.strip() for line in f]

    with open(label_file, "r", encoding="utf-8") as f:
        labels = [int(line.strip()) for line in f]

    if len(texts) != len(labels):
        raise ValueError(f"Mismatch in split {split_name}: {len(texts)} texts vs {len(labels)} labels")

    df = pd.DataFrame({
        "text": texts,
        "label": labels,
        "sentiment": [LABEL_MAPPING[lbl] for lbl in labels],
        "split": split_name
    })

    # Drop completely empty texts
    df = df[df["text"].str.strip() != ""]
    return df


def prepare_sentiment_dataset(output_dir: str):
    """Acquire and prepare the full TweetEval sentiment dataset."""
    os.makedirs(output_dir, exist_ok=True)
    raw_dir = os.path.join(output_dir, "raw_tweeteval")
    os.makedirs(raw_dir, exist_ok=True)

    logger.info("Fetching TweetEval splits (train, val, test)...")
    train_df = load_tweeteval_split(raw_dir, "train")
    val_df = load_tweeteval_split(raw_dir, "val")
    test_df = load_tweeteval_split(raw_dir, "test")

    logger.info(f"TweetEval Train samples: {len(train_df)}")
    logger.info(f"TweetEval Val samples:   {len(val_df)}")
    logger.info(f"TweetEval Test samples:  {len(test_df)}")

    # Save individual splits
    train_df.to_csv(os.path.join(output_dir, "sentiment_train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "sentiment_val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "sentiment_test.csv"), index=False)

    # Combined dataset
    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    combined_path = os.path.join(output_dir, "sentiment_dataset.csv")
    combined_df.to_csv(combined_path, index=False)
    logger.info(f"Saved complete sentiment dataset to {combined_path} ({len(combined_df)} records)")

    # Print class distributions
    logger.info("Sentiment class distribution (Full):")
    for sentiment, count in combined_df["sentiment"].value_counts().items():
        pct = (count / len(combined_df)) * 100
        logger.info(f"  {sentiment}: {count} ({pct:.2f}%)")

    return combined_df


def prepare_emotion_dataset(output_dir: str):
    """
    Acquire and prepare GoEmotions subset mapped to 7 target categories.
    Categories: Happiness, Sadness, Anger, Fear, Surprise, Love, Neutral
    """
    raw_file = os.path.join(output_dir, "raw_goemotions_train.tsv")
    if not os.path.exists(raw_file):
        try:
            download_file(EMOTION_SOURCE_URL, raw_file)
        except Exception as e:
            logger.warning(f"Could not directly download GoEmotions TSV from GitHub: {e}")
            return None

    # Load GoEmotions headers from google-research specification
    emotion_columns = [
        "admiration", "amusement", "anger", "annoyance", "approval", "caring",
        "confusion", "curiosity", "desire", "disappointment", "disapproval",
        "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
        "joy", "love", "nervousness", "optimism", "pride", "realization",
        "relief", "remorse", "sadness", "surprise", "neutral"
    ]
    columns = ["text", "labels", "id"]

    try:
        df = pd.read_csv(raw_file, sep="\t", header=None, names=columns, quoting=3)
    except Exception as e:
        logger.error(f"Failed to read GoEmotions TSV: {e}")
        return None

    mapped_records = []
    for _, row in df.iterrows():
        text = str(row["text"]).strip()
        if not text or len(text) < 3:
            continue
        try:
            # labels are comma-separated integer indices matching emotion_columns
            raw_label_indices = [int(idx.strip()) for idx in str(row["labels"]).split(",") if idx.strip().isdigit()]
        except Exception:
            continue

        emotions_found = set()
        for idx in raw_label_indices:
            if 0 <= idx < len(emotion_columns):
                raw_emo = emotion_columns[idx]
                target_emo = GOEMOTIONS_TAXONOMY.get(raw_emo)
                if target_emo:
                    emotions_found.add(target_emo)

        # Single distinct broad emotion mapping per sample
        if len(emotions_found) == 1:
            mapped_records.append({"text": text, "emotion": list(emotions_found)[0]})

    emotion_df = pd.DataFrame(mapped_records)
    logger.info(f"Processed GoEmotions samples: {len(emotion_df)}")
    logger.info("Emotion distribution:")
    for emo, cnt in emotion_df["emotion"].value_counts().items():
        logger.info(f"  {emo}: {cnt} ({(cnt/len(emotion_df))*100:.2f}%)")

    out_path = os.path.join(output_dir, "emotion_dataset.csv")
    emotion_df.to_csv(out_path, index=False)
    logger.info(f"Saved emotion dataset to {out_path}")
    return emotion_df


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    logger.info("Starting dataset acquisition...")
    prepare_sentiment_dataset(current_dir)
    prepare_emotion_dataset(current_dir)
    logger.info("Dataset preparation complete.")
