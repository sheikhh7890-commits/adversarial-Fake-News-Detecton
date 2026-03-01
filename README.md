# Fake News Detection with DeBERTa v3 Large

This repository contains a local fake news detection pipeline built with PyTorch, Hugging Face Transformers, and the Hugging Face `datasets` API. The training setup is tuned for a single 12GB GPU such as an RTX 4070 and uses `microsoft/deberta-v3-large` for binary classification.

The pipeline trains on:

```python
from datasets import load_dataset
ds = load_dataset("GonzaloA/fake_news")
```

Everything runs locally. The project does not use the Hugging Face Hub for uploads and does not require API tokens.

## What the Pipeline Does

- Loads `GonzaloA/fake_news` with the Hugging Face datasets API
- Uses the dataset's existing `train`, `validation`, and `test` splits
- Combines `title` and `content/text` into one model input
- Cleans text with minimal preprocessing
- Normalizes labels to binary `0/1`
- Drops empty or invalid rows
- Tokenizes with `microsoft/deberta-v3-large` using `max_length=256`
- Trains with Hugging Face `Trainer`
- Uses RTX 4070-safe settings:
  - `fp16=True`
  - `batch_size=4`
  - `gradient_accumulation_steps=4`
  - `gradient_checkpointing=True`
- Evaluates with:
  - accuracy
  - precision
  - recall
  - F1-score
  - confusion matrix
- Saves model, tokenizer, metrics, and plots locally
- Runs a sample inference step after training

## Project Structure

```text
.
|-- src/
|   |-- __init__.py
|   |-- config.py
|   |-- data.py
|   |-- main.py
|   |-- model.py
|   `-- train_eval.py
|-- docs/
|-- models/
|-- notebooks/
|-- results/
|-- main.py
|-- requirements.txt
`-- README.md
```

## Module Guide

- `src/config.py`
  Stores model name, dataset name, hyperparameters, label mappings, output paths, and regex patterns.

- `src/data.py`
  Loads the dataset, cleans and combines `title` plus `content/text`, normalizes labels, filters invalid rows, prints split statistics, and tokenizes the splits.

- `src/model.py`
  Creates the tokenizer and DeBERTa classification model and provides the prediction helper for new text.

- `src/train_eval.py`
  Builds the `Trainer`, computes metrics, runs training and evaluation, plots the confusion matrix and training curve, and saves metrics and model artifacts.

- `src/main.py`
  Runs the full pipeline:
  1. set seeds
  2. load and prepare dataset
  3. tokenize
  4. create model
  5. train
  6. evaluate
  7. save outputs
  8. run inference

- `main.py`
  Thin root-level entry point so the project can still be started with `python main.py`.

## Training Configuration

The default configuration is tuned for a single GPU with 12GB VRAM:

```python
MODEL_NAME = "microsoft/deberta-v3-large"
MAX_LEN = 256
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
EARLY_STOPPING_PATIENCE = 2
```

These values live in [src/config.py](C:/Users/sheik/OneDrive/Desktop/PROJECT%20FINAL/New%20folder/src/config.py).

## Installation

Install the current dependencies:

```powershell
pip install -r requirements.txt
```

You also need:

- a CUDA-compatible PyTorch build
- `sentencepiece` for DeBERTa v3 tokenization

If `sentencepiece` is missing:

```powershell
pip install sentencepiece
```

If PyTorch is not installed with CUDA support, install the correct build for your system before running training.

## How to Run

Start the full pipeline with:

```powershell
python main.py
```

This will:

1. download the dataset locally through `datasets`
2. download the DeBERTa model locally
3. train the classifier
4. evaluate on the test set
5. save all artifacts locally
6. print an example prediction

Example console output:

```text
Prediction: FAKE
```

## Saved Outputs

After a successful run, artifacts are stored locally in:

- `./model`
  - trained model
  - tokenizer
  - trainer checkpoint data

- `./results`
  - `test_metrics.json`
  - `confusion_matrix.png`
  - `training_curve.png`

## Notes

- Dynamic padding is used to reduce wasted memory.
- The tokenizer truncates strictly at `256` tokens.
- The training code is configured to avoid settings that are likely to exceed 12GB VRAM.
- `notebooks/` is kept in the repository for experimentation history, but the main code path now lives under `src/`.
