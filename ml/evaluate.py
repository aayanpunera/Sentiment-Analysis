"""
Evaluation and Metrics Visualization Utilities.

Calculates standard classification metrics (accuracy, precision, recall, f1)
and generates confusion matrices and comparison charts.
"""

import json
import logging
from typing import Dict, Any, List
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    target_names: List[str]
) -> Dict[str, Any]:
    """
    Compute comprehensive metrics adhering to the project specification.
    """
    acc = float(accuracy_score(y_true, y_pred))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    cm = confusion_matrix(y_true, y_pred).tolist()
    clf_report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )

    return {
        "accuracy": round(acc, 4),
        "precision": round(macro_prec, 4),
        "recall": round(macro_rec, 4),
        "f1_score": round(macro_f1, 4),
        "confusion_matrix": cm,
        "classification_report": clf_report
    }


def generate_comparison_chart(
    all_metrics: Dict[str, Dict[str, Any]],
    output_path: str
):
    """
    Generate and save a bar chart comparing Accuracy, Precision, Recall, and F1 across models.
    """
    models = list(all_metrics.keys())
    metric_keys = ["accuracy", "precision", "recall", "f1_score"]
    metric_labels = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]

    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#7C3AED", "#A78BFA", "#38BDF8", "#34D399"]

    for i, (key, label, color) in enumerate(zip(metric_keys, metric_labels, colors)):
        values = [all_metrics[m][key] for m in models]
        offset = (i - 1.5) * width
        rects = ax.bar(x + offset, values, width, label=label, color=color, edgecolor="#4C1D95", linewidth=0.5)
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.2f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=11, fontweight="bold")
    ax.set_title("Sentiment Model Performance Comparison (Held-Out Test Set)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info(f"Saved comparison chart to {output_path}")


def save_metrics_json(metrics: Dict[str, Any], output_path: str):
    """Persist evaluation metrics dictionary to JSON."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {output_path}")
