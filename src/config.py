from __future__ import annotations

import re
from pathlib import Path


SEED = 42
MODEL_NAME = "microsoft/deberta-v3-large"
DATASET_NAME = "GonzaloA/fake_news"
MAX_LEN = 256

OUTPUT_DIR = Path("./model")
RESULTS_DIR = Path("./results")
TRAINING_PLOT_PATH = RESULTS_DIR / "training_curve.png"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"
TEST_METRICS_PATH = RESULTS_DIR / "test_metrics.json"

TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
EARLY_STOPPING_PATIENCE = 2

LABELS = ["FAKE", "REAL"]
ID2LABEL = {0: "FAKE", 1: "REAL"}
LABEL2ID = {"FAKE": 0, "REAL": 1}

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"\s+")
