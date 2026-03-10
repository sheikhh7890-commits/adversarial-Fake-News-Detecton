from __future__ import annotations

import html

import pandas as pd
from datasets import load_dataset
from transformers import DataCollatorWithPadding

from .config import DATASET_NAME, HTML_TAG_PATTERN, MAX_LEN, URL_PATTERN, WHITESPACE_PATTERN


def clean_text(text: object) -> str:
    if text is None:
        return ""
    text = html.unescape(str(text))
    text = URL_PATTERN.sub(" ", text)
    text = HTML_TAG_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def normalize_label(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and not pd.isna(value):
        value = int(value)
        if value in (0, 1):
            return value
    value = str(value).strip().lower()
    if value in {"0", "fake", "false"}:
        return 0
    if value in {"1", "real", "true"}:
        return 1
    return None


def build_text(example: dict[str, object]) -> dict[str, object]:
    title = clean_text(example.get("title", ""))
    content = clean_text(example.get("content", example.get("text", "")))
    if title and content:
        example["text"] = f"{title} [SEP] {content}"
    elif title:
        example["text"] = title
    else:
        example["text"] = content
    example["labels"] = normalize_label(example.get("value"))
    return example


def is_valid(example: dict[str, object]) -> bool:
    return example["labels"] is not None and len(example["text"]) > 0


def load_and_prepare_dataset():
    ds = load_dataset(DATASET_NAME)
    ds = ds.map(build_text)
    ds = ds.filter(is_valid)
    keep_columns = {"text", "labels"}
    ds = ds.remove_columns([column for column in ds["train"].column_names if column not in keep_columns])
    return ds


def print_split_stats(ds) -> None:
    for split in ds:
        counts = pd.Series(ds[split]["labels"]).value_counts().sort_index().to_dict()
        print(split, len(ds[split]), counts)


def tokenize_dataset(ds, tokenizer):
    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, list[int]]:
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN)

    tokenized = ds.map(tokenize_batch, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return tokenized, data_collator
