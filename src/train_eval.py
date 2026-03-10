from __future__ import annotations

import json
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, get_linear_schedule_with_warmup

from .config import (
    CONFUSION_MATRIX_PATH,
    EARLY_STOPPING_PATIENCE,
    EVAL_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    LABELS,
    LEARNING_RATE,
    NUM_EPOCHS,
    OUTPUT_DIR,
    SEED,
    TEST_METRICS_PATH,
    TRAIN_BATCH_SIZE,
    TRAINING_PLOT_PATH,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)


def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }


def create_trainer(model, tokenizer, tokenized, data_collator):
    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        fp16=torch.cuda.is_available(),
        seed=SEED,
        report_to="none",
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_len = len(tokenized["train"])
    steps_per_epoch = math.ceil(train_len / (args.per_device_train_batch_size * args.gradient_accumulation_steps))
    num_training_steps = steps_per_epoch * int(args.num_train_epochs)
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    return Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
        optimizers=(optimizer, scheduler),
    )


def plot_confusion_matrix(labels: np.ndarray, preds: np.ndarray) -> None:
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks(np.arange(2), LABELS)
    plt.yticks(np.arange(2), LABELS)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=200, bbox_inches="tight")
    plt.close()


def plot_training_curve(trainer: Trainer) -> None:
    train_loss_history = [
        entry["loss"]
        for entry in trainer.state.log_history
        if "loss" in entry and "eval_loss" not in entry
    ]
    train_loss_steps = [
        entry["step"]
        for entry in trainer.state.log_history
        if "loss" in entry and "eval_loss" not in entry
    ]
    eval_f1_history = [entry["eval_f1"] for entry in trainer.state.log_history if "eval_f1" in entry]
    eval_f1_steps = [entry["step"] for entry in trainer.state.log_history if "eval_f1" in entry]

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss_steps, train_loss_history, marker="o")
    plt.title("Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(eval_f1_steps, eval_f1_history, marker="o")
    plt.title("Validation F1")
    plt.xlabel("Step")
    plt.ylabel("F1")

    plt.tight_layout()
    plt.savefig(TRAINING_PLOT_PATH, dpi=200, bbox_inches="tight")
    plt.close()


def train_and_evaluate(trainer: Trainer, tokenizer, tokenized):
    train_out = trainer.train()
    print(f"Training complete! Loss: {train_out.metrics['train_loss']:.4f}")

    val_metrics = trainer.evaluate(tokenized["validation"])
    test_metrics = trainer.evaluate(tokenized["test"])
    print(f"\nValidation F1: {val_metrics['eval_f1']:.4f}")
    print(f"Test F1: {test_metrics['eval_f1']:.4f}")
    print(f"Test Accuracy: {test_metrics['eval_accuracy']:.4f}")

    pred = trainer.predict(tokenized["test"])
    preds = np.argmax(pred.predictions, axis=-1)
    print("\n" + classification_report(pred.label_ids, preds, target_names=LABELS, digits=4))

    plot_confusion_matrix(pred.label_ids, preds)
    plot_training_curve(trainer)

    TEST_METRICS_PATH.write_text(
        json.dumps(
            {
                "accuracy": test_metrics["eval_accuracy"],
                "precision": test_metrics["eval_precision"],
                "recall": test_metrics["eval_recall"],
                "f1": test_metrics["eval_f1"],
                "confusion_matrix": confusion_matrix(pred.label_ids, preds).tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"Model saved to {OUTPUT_DIR}")

    return trainer.model
