from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.config import (
    BERT_METRICS_FILE,
    BERT_MODEL_DIR,
    BERT_VAL_PREDICTIONS_FILE,
    LABEL_COLUMN,
    MODELS_DIR,
    OUTPUTS_DIR,
    TEXT_COLUMN,
    TRAIN_FILE,
    VAL_FILE,
)

DEFAULT_MODEL_NAME = "bert-base-uncased"
EXPERIMENT_RESULTS_FILE = OUTPUTS_DIR / "transformer_experiments.csv"
BEST_EXPERIMENT_FILE = OUTPUTS_DIR / "transformer_best_experiment.txt"
BASELINE_ACCURACY = 0.8429


@dataclass
class RumorSample:
    text: str
    label: int


@dataclass
class ExperimentConfig:
    model_name: str
    epochs: int
    batch_size: int
    eval_batch_size: int
    grad_accumulation: int
    max_length: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    output_dir: Path


class RumorDataset(Dataset):
    def __init__(self, samples: list[RumorSample], tokenizer, max_length: int) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        encoded = self.tokenizer(
            sample.text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(sample.label, dtype=torch.long)
        return item


def load_dataset(path) -> pd.DataFrame:
    data = pd.read_csv(path)
    required_columns = [TEXT_COLUMN, LABEL_COLUMN]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in {path}: {missing_columns}")
    data = data.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN]).copy()
    data[TEXT_COLUMN] = data[TEXT_COLUMN].astype(str)
    data[LABEL_COLUMN] = data[LABEL_COLUMN].astype(int)
    return data


def build_samples(dataframe: pd.DataFrame) -> list[RumorSample]:
    return [
        RumorSample(text=row[TEXT_COLUMN], label=int(row[LABEL_COLUMN]))
        for _, row in dataframe.iterrows()
    ]


def compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="binary", zero_division=0)
    return {"accuracy": accuracy, "f1": f1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a BERT rumor classifier.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Hugging Face model name or local model directory")
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Per-device batch size")
    parser.add_argument("--eval-batch-size", type=int, default=16, help="Per-device eval batch size")
    parser.add_argument("--grad-accumulation", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--max-length", type=int, default=256, help="Maximum token length")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--sweep", action="store_true", help="Run a small experiment sweep")
    return parser.parse_args()


def load_pretrained_components(model_name: str):
    model_source = Path(model_name).expanduser()
    if model_source.exists():
        source = str(model_source.resolve())
        try:
            tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(
                source,
                num_labels=2,
                local_files_only=True,
            )
        except OSError as error:
            raise RuntimeError(
                f"Failed to load local pretrained model from: {source}. "
                "Make sure the directory contains the tokenizer and model files saved by Hugging Face."
            ) from error
        return tokenizer, model

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    except OSError as error:
        raise RuntimeError(
            "Failed to load the pretrained model. Check your network connection to Hugging Face, "
            "or pass a valid local model directory with --model-name."
        ) from error
    return tokenizer, model


def build_training_args(config: ExperimentConfig) -> TrainingArguments:
    use_fp16 = torch.cuda.is_available()
    return TrainingArguments(
        output_dir=str(config.output_dir),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.grad_accumulation,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_strategy="epoch",
        report_to="none",
        fp16=use_fp16,
    )


def run_experiment(config: ExperimentConfig, train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict[str, Any]:
    tokenizer, model = load_pretrained_components(config.model_name)

    train_dataset = RumorDataset(build_samples(train_df), tokenizer, config.max_length)
    val_dataset = RumorDataset(build_samples(val_df), tokenizer, config.max_length)

    trainer = Trainer(
        model=model,
        args=build_training_args(config),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    predictions_output = trainer.predict(val_dataset)
    predictions = predictions_output.predictions.argmax(axis=-1)
    accuracy = accuracy_score(val_df[LABEL_COLUMN], predictions)
    f1 = f1_score(val_df[LABEL_COLUMN], predictions, average="binary", zero_division=0)
    report = classification_report(val_df[LABEL_COLUMN], predictions, digits=4)

    return {
        "trainer": trainer,
        "tokenizer": tokenizer,
        "predictions": predictions,
        "accuracy": accuracy,
        "f1": f1,
        "report": report,
    }


def save_best_artifacts(config: ExperimentConfig, result: dict[str, Any], val_df: pd.DataFrame) -> None:
    trainer = result["trainer"]
    tokenizer = result["tokenizer"]
    predictions = result["predictions"]
    accuracy = result["accuracy"]
    report = result["report"]

    trainer.save_model(str(BERT_MODEL_DIR))
    tokenizer.save_pretrained(str(BERT_MODEL_DIR))

    result_df = val_df.copy()
    result_df["prediction"] = predictions
    result_df.to_csv(BERT_VAL_PREDICTIONS_FILE, index=False)

    BERT_METRICS_FILE.write_text(
        (
            f"Model: {config.model_name}\n"
            f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}\n"
            f"Validation accuracy: {accuracy:.4f}\n\n"
            f"{report}\n"
        ),
        encoding="utf-8",
    )


def run_single(args: argparse.Namespace) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_dataset(TRAIN_FILE)
    val_df = load_dataset(VAL_FILE)

    config = ExperimentConfig(
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        grad_accumulation=args.grad_accumulation,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        output_dir=BERT_MODEL_DIR / "checkpoints",
    )

    result = run_experiment(config, train_df, val_df)
    save_best_artifacts(config, result, val_df)

    print(f"Validation accuracy: {result['accuracy']:.4f}")
    print(result["report"])
    print(f"Saved BERT model to: {BERT_MODEL_DIR}")
    print(f"Saved validation predictions to: {BERT_VAL_PREDICTIONS_FILE}")
    print(f"Saved metrics to: {BERT_METRICS_FILE}")


def run_sweep() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_dataset(TRAIN_FILE)
    val_df = load_dataset(VAL_FILE)

    model_names = [
        "bert-base-uncased",
        "FacebookAI/roberta-base",
    ]
    learning_rates = [2e-5, 1.5e-5]
    epochs_list = [4, 5]
    max_lengths = [256]

    results = []
    best_result: dict[str, Any] | None = None
    best_config: ExperimentConfig | None = None
    experiment_id = 0

    for model_name, learning_rate, epochs, max_length in product(
        model_names,
        learning_rates,
        epochs_list,
        max_lengths,
    ):
        experiment_id += 1
        config = ExperimentConfig(
            model_name=model_name,
            epochs=epochs,
            batch_size=8,
            eval_batch_size=16,
            grad_accumulation=2,
            max_length=max_length,
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.1,
            output_dir=MODELS_DIR / f"experiment_{experiment_id}",
        )

        print("=" * 80)
        print(f"Experiment {experiment_id}")
        print(config)

        result = run_experiment(config, train_df, val_df)

        row = {
            "experiment_id": experiment_id,
            "model_name": model_name,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "max_length": max_length,
            "batch_size": config.batch_size,
            "eval_batch_size": config.eval_batch_size,
            "grad_accumulation": config.grad_accumulation,
            "accuracy": result["accuracy"],
            "f1": result["f1"],
            "improvement_over_baseline": result["accuracy"] - BASELINE_ACCURACY,
        }
        results.append(row)

        print(f"Accuracy: {result['accuracy']:.4f}")
        print(f"F1: {result['f1']:.4f}")

        if best_result is None or result["accuracy"] > best_result["accuracy"]:
            best_result = result
            best_config = config

    result_df = pd.DataFrame(results).sort_values(by=["accuracy", "f1"], ascending=False)
    result_df.to_csv(EXPERIMENT_RESULTS_FILE, index=False)

    if best_result is None or best_config is None:
        raise RuntimeError("No experiment results were produced.")

    save_best_artifacts(best_config, best_result, val_df)

    BEST_EXPERIMENT_FILE.write_text(
        (
            f"Best model: {best_config.model_name}\n"
            f"Epochs: {best_config.epochs}\n"
            f"Learning rate: {best_config.learning_rate}\n"
            f"Max length: {best_config.max_length}\n"
            f"Validation accuracy: {best_result['accuracy']:.4f}\n"
            f"Validation F1: {best_result['f1']:.4f}\n"
            f"Improvement over baseline: {best_result['accuracy'] - BASELINE_ACCURACY:.4f}\n"
        ),
        encoding="utf-8",
    )

    print("\nBest experiment summary:")
    print(BEST_EXPERIMENT_FILE.read_text(encoding="utf-8"))
    print(f"Saved sweep results to: {EXPERIMENT_RESULTS_FILE}")


def main() -> None:
    args = parse_args()
    if args.sweep:
        run_sweep()
        return
    run_single(args)


if __name__ == "__main__":
    main()
