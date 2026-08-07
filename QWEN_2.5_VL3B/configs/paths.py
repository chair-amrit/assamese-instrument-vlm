"""
Project paths.

Update this file if the dataset or output directories change.
"""

import os

# Dataset Root
DATASET_ROOT = "dataset"

IMAGE_ROOT = os.path.join(
    DATASET_ROOT,
    "dataset_32images",
)

# Dataset Files
TRAIN_JSON = os.path.join(
    DATASET_ROOT,
    "train.jsonl",
)

VAL_JSON = os.path.join(
    DATASET_ROOT,
    "validation.jsonl",
)

TEST_JSON = os.path.join(
    DATASET_ROOT,
    "test.jsonl",
)

FINAL_VQA_CSV = os.path.join(
    DATASET_ROOT,
    "final_vqa.csv",
)

# Model Output
OUTPUT_DIR = "models/qwen2.5vl_lora"

ADAPTER_PATH = os.path.join(
    OUTPUT_DIR,
    "adapter",
)

# Evaluation Output
RESULTS_DIR = "results"

PREDICTIONS_JSON = os.path.join(
    RESULTS_DIR,
    "test_predictions.json",
)