from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.config import (
    TRAIN_FILE,
    VAL_FILE,
    TEXT_COLUMN,
    LABEL_COLUMN,
    OUTPUTS_DIR,
    MODELS_DIR,
)
from src.preprocess import clean_text


MODEL_NAME = "distilbert-base-uncased"

DISTILBERT_MODEL_DIR = MODELS_DIR / "distilbert_rumor_model"
DISTILBERT_METRICS_FILE = OUTPUTS_DIR / "distilbert_metrics.txt"
DISTILBERT_PREDICTIONS_FILE = OUTPUTS_DIR / "val_predictions_distilbert.csv"

MAX_LENGTH = 128
RANDOM_SEED = 42

NUM_EPOCHS = 5
LEARNING_RATE = 3e-5
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32

def set_random_seed(seed: int = RANDOM_SEED) -> None:
    """
    固定随机种子，让实验结果尽量可复现。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


class RumorDataset(Dataset):
    """
    将文本和标签封装成 PyTorch Dataset，供 Trainer 使用。
    """

    def __init__(self, texts, labels, tokenizer, max_length: int = MAX_LENGTH):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = str(self.texts[index])
        label = int(self.labels[index])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }

        return item


def load_dataset(path: Path) -> pd.DataFrame:
    """
    读取数据，并使用项目已有 clean_text 做预处理。
    """
    data = pd.read_csv(path)

    required_columns = [TEXT_COLUMN, LABEL_COLUMN]
    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns in {path}: {missing_columns}")

    data = data.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN]).copy()
    data["clean_text"] = data[TEXT_COLUMN].map(clean_text)
    data[LABEL_COLUMN] = data[LABEL_COLUMN].astype(int)

    return data


def compute_metrics(eval_pred):
    """
    计算 accuracy / precision / recall / f1。
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main() -> None:
    set_random_seed()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DISTILBERT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    train_df = load_dataset(TRAIN_FILE)
    val_df = load_dataset(VAL_FILE)

    print("Train size:", len(train_df))
    print("Val size:", len(val_df))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = RumorDataset(
        texts=train_df["clean_text"],
        labels=train_df[LABEL_COLUMN],
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )

    val_dataset = RumorDataset(
        texts=val_df["clean_text"],
        labels=val_df[LABEL_COLUMN],
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )

    training_args = TrainingArguments(
        output_dir=str(DISTILBERT_MODEL_DIR / "checkpoints"),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        fp16=torch.cuda.is_available(),
        seed=RANDOM_SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    eval_result = trainer.evaluate()
    print("Evaluation result:")
    print(eval_result)

    prediction_output = trainer.predict(val_dataset)
    logits = prediction_output.predictions
    predictions = np.argmax(logits, axis=1)

    y_true = val_df[LABEL_COLUMN].to_numpy()
    accuracy = accuracy_score(y_true, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        predictions,
        average="binary",
        zero_division=0,
    )
    report = classification_report(y_true, predictions, digits=4)

    result_df = val_df.copy()
    result_df["prediction"] = predictions

    output_columns = [
        column
        for column in ["id", TEXT_COLUMN, LABEL_COLUMN, "prediction", "event"]
        if column in result_df.columns
    ]
    result_df[output_columns].to_csv(DISTILBERT_PREDICTIONS_FILE, index=False)

    metrics_text = (
        "DistilBERT Fine-tuning Result\n\n"
        f"Model name: {MODEL_NAME}\n"
        f"Max length: {MAX_LENGTH}\n"
        f"Epochs: {NUM_EPOCHS}\n"
        f"Learning rate: {LEARNING_RATE}\n"
        f"Train batch size: {TRAIN_BATCH_SIZE}\n"
        f"Eval batch size: {EVAL_BATCH_SIZE}\n\n"
        f"Accuracy: {accuracy:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall: {recall:.4f}\n"
        f"F1: {f1:.4f}\n\n"
        f"{report}\n"
    )

    DISTILBERT_METRICS_FILE.write_text(metrics_text, encoding="utf-8")

    trainer.save_model(str(DISTILBERT_MODEL_DIR))
    tokenizer.save_pretrained(str(DISTILBERT_MODEL_DIR))

    print(metrics_text)
    print(f"Saved DistilBERT model to: {DISTILBERT_MODEL_DIR}")
    print(f"Saved predictions to: {DISTILBERT_PREDICTIONS_FILE}")
    print(f"Saved metrics to: {DISTILBERT_METRICS_FILE}")


if __name__ == "__main__":
    main()