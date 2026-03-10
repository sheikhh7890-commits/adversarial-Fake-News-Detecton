from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import ID2LABEL, LABEL2ID, MAX_LEN, MODEL_NAME
from .data import clean_text


def create_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)


def create_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    return model


@torch.inference_mode()
def predict_text(model, tokenizer, text: str) -> dict[str, float | str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    encoded = tokenizer(
        clean_text(text),
        truncation=True,
        max_length=MAX_LEN,
        padding=True,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    logits = model(**encoded).logits
    probs = torch.softmax(logits, dim=-1).squeeze(0)
    label_id = int(torch.argmax(probs).item())
    return {
        "label_id": label_id,
        "label_name": ID2LABEL[label_id],
        "fake_probability": float(probs[0].item()),
        "real_probability": float(probs[1].item()),
    }
