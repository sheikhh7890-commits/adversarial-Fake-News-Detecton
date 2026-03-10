from __future__ import annotations

import random

import numpy as np
import torch
from transformers import set_seed

from .config import OUTPUT_DIR, RESULTS_DIR, SEED
from .data import load_and_prepare_dataset, print_split_stats, tokenize_dataset
from .model import create_model, create_tokenizer, predict_text
from .train_eval import create_trainer, train_and_evaluate


def main() -> None:
    set_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ds = load_and_prepare_dataset()
    print_split_stats(ds)

    tokenizer = create_tokenizer()
    tokenized, data_collator = tokenize_dataset(ds, tokenizer)
    model = create_model()
    trainer = create_trainer(model, tokenizer, tokenized, data_collator)
    trained_model = train_and_evaluate(trainer, tokenizer, tokenized)

    raw_example = ds["test"][0]["text"]
    prediction = predict_text(trained_model, tokenizer, raw_example)
    print(f"Prediction: {prediction['label_name']}")


if __name__ == "__main__":
    main()
