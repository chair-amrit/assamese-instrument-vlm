"""
Plotting utilities for evaluation and analysis.

These functions centralize the plots already used in the project:
- training vs. validation loss
- instrument-wise LAVE
- instrument-wise cosine similarity
- concept-wise LAVE
- concept-wise cosine similarity
- Qwen2.5-VL vs. PaliGemma baseline comparison
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd


def _save_and_show(
    output_path: str | Path | None = None,
    *,
    dpi: int = 300,
) -> None:
    """Save the current figure if requested and display it."""
    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")

    plt.show()
    plt.close()


def plot_loss_curve(
    train_steps: Sequence[float],
    train_loss: Sequence[float],
    eval_steps: Sequence[float],
    eval_loss: Sequence[float],
    *,
    output_path: str | Path | None = None,
    figsize: tuple[float, float] = (8, 5),
    dpi: int = 300,
) -> None:
    """
    Plot training and validation loss against training step.

    This reproduces the training-vs-validation loss plot used in the
    original evaluation workflow.
    """
    plt.figure(figsize=figsize)

    plt.plot(
        train_steps,
        train_loss,
        label="Training Loss",
    )

    plt.plot(
        eval_steps,
        eval_loss,
        label="Validation Loss",
    )

    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)

    _save_and_show(output_path, dpi=dpi)


def plot_instrument_metric(
    instrument_scores: pd.DataFrame,
    *,
    metric: str,
    ylabel: str | None = None,
    title: str | None = None,
    output_path: str | Path | None = None,
    figsize: tuple[float, float] = (8, 5),
    dpi: int = 300,
) -> None:
    """
    Plot an evaluation metric by instrument.

    Expected DataFrame structure:
        index   -> instrument
        metric  -> metric column

    Supported project metrics include:
        - lave_score
        - cosine_similarity
    """
    if metric not in instrument_scores.columns:
        raise ValueError(
            f"Metric '{metric}' was not found in the DataFrame."
        )

    if ylabel is None:
        ylabel = _metric_label(metric)

    if title is None:
        title = f"{_metric_name(metric)} by Instrument"

    plt.figure(figsize=figsize)

    instrument_scores[metric].plot(kind="bar")

    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y")

    _save_and_show(output_path, dpi=dpi)


def plot_concept_metric(
    concept_scores: pd.DataFrame,
    *,
    metric: str,
    ylabel: str | None = None,
    title: str | None = None,
    output_path: str | Path | None = None,
    figsize: tuple[float, float] = (10, 5),
    dpi: int = 300,
) -> None:
    """
    Plot an evaluation metric by question concept.

    Expected DataFrame structure:
        index   -> concept
        metric  -> metric column

    Supported project metrics include:
        - lave_score
        - cosine_similarity
    """
    if metric not in concept_scores.columns:
        raise ValueError(
            f"Metric '{metric}' was not found in the DataFrame."
        )

    if ylabel is None:
        ylabel = _metric_label(metric)

    if title is None:
        title = f"{_metric_name(metric)} by Question Concept"

    plt.figure(figsize=figsize)

    concept_scores[metric].plot(kind="bar")

    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y")

    _save_and_show(output_path, dpi=dpi)


def plot_baseline_comparison(
    *,
    qwen_lave: float,
    qwen_cosine: float,
    paligemma_lave: float,
    paligemma_cosine: float,
    output_path: str | Path | None = None,
    figsize: tuple[float, float] = (6, 5),
    dpi: int = 300,
) -> None:
    """
    Plot Qwen2.5-VL against the PaliGemma baseline.

    Parameters correspond to the two evaluation metrics used in the
    existing evaluation workflow.
    """
    comparison = pd.DataFrame(
        {
            "PaliGemma": [
                paligemma_lave,
                paligemma_cosine,
            ],
            "Qwen2.5-VL": [
                qwen_lave,
                qwen_cosine,
            ],
        },
        index=[
            "LAVE",
            "Cosine Similarity",
        ],
    )

    plt.figure(figsize=figsize)

    comparison.plot(kind="bar")

    plt.ylabel("Score")
    plt.title("Qwen2.5-VL vs PaliGemma")
    plt.grid(axis="y")

    _save_and_show(output_path, dpi=dpi)


def _metric_name(metric: str) -> str:
    """Return a human-readable metric name."""
    names = {
        "lave_score": "LAVE",
        "cosine_similarity": "Cosine Similarity",
    }

    return names.get(metric, metric.replace("_", " ").title())


def _metric_label(metric: str) -> str:
    """Return the y-axis label used for a metric."""
    labels = {
        "lave_score": "Average LAVE",
        "cosine_similarity": "Average Cosine Similarity",
    }

    return labels.get(metric, metric.replace("_", " ").title())