from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, set_seed

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
from src.train_bert import RumorDataset, RumorSample

CURRENT_BEST_ACCURACY = 0.8827930174563591
RESULTS_FILE = OUTPUTS_DIR / "cuda_transformer_optimization.csv"
BEST_FILE = OUTPUTS_DIR / "cuda_transformer_optimization_best.txt"


@dataclass
class Candidate:
    name: str
    model_name: str
    epochs: int
    learning_rate: float
    max_length: int
    batch_size: int = 8
    eval_batch_size: int = 16
    grad_accumulation: int = 2
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = 42


def load_dataset(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    data = data.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN]).copy()
    data[TEXT_COLUMN] = data[TEXT_COLUMN].astype(str)
    data[LABEL_COLUMN] = data[LABEL_COLUMN].astype(int)
    return data


def build_samples(dataframe: pd.DataFrame) -> list[RumorSample]:
    return [RumorSample(text=row[TEXT_COLUMN], label=int(row[LABEL_COLUMN])) for _, row in dataframe.iterrows()]


def compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="binary", zero_division=0),
    }


def candidate_list() -> list[Candidate]:
    return [
        Candidate("roberta_lr2e5_len160_ep4", "FacebookAI/roberta-base", 4, 2e-5, 160, seed=42),
        Candidate("roberta_lr1e5_len160_ep4", "FacebookAI/roberta-base", 4, 1e-5, 160, seed=42),
        Candidate("roberta_lr2e5_len128_ep4", "FacebookAI/roberta-base", 4, 2e-5, 128, seed=42),
        Candidate("roberta_lr1e5_len128_ep5", "FacebookAI/roberta-base", 5, 1e-5, 128, seed=42),
        Candidate("continue_best_lr5e6_ep2", str(BERT_MODEL_DIR), 2, 5e-6, 160, seed=42),
        Candidate("continue_best_lr1e6_ep2", str(BERT_MODEL_DIR), 2, 1e-6, 160, seed=42),
        Candidate("continue_best_lr5e6_len256_ep2", str(BERT_MODEL_DIR), 2, 5e-6, 256, seed=42),
        Candidate("continue_best_lr1e6_len256_ep2", str(BERT_MODEL_DIR), 2, 1e-6, 256, seed=42),
        Candidate("roberta_lr2e5_len256_ep4_seed7", "FacebookAI/roberta-base", 4, 2e-5, 256, seed=7),
        Candidate("roberta_lr2e5_len256_ep4_seed123", "FacebookAI/roberta-base", 4, 2e-5, 256, seed=123),
        Candidate("roberta_lr2e5_len256_ep4_seed2024", "FacebookAI/roberta-base", 4, 2e-5, 256, seed=2024),
        Candidate("roberta_lr15e6_len256_ep4_seed42", "FacebookAI/roberta-base", 4, 1.5e-5, 256, seed=42),
        Candidate("roberta_lr25e6_len256_ep4_seed42", "FacebookAI/roberta-base", 4, 2.5e-5, 256, seed=42),
    ]


def run_candidate(candidate: Candidate, train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict[str, Any]:
    set_seed(candidate.seed)
    output_dir = MODELS_DIR / "cuda_optimization" / candidate.name
    tokenizer = AutoTokenizer.from_pretrained(candidate.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(candidate.model_name, num_labels=2)

    train_dataset = RumorDataset(build_samples(train_df), tokenizer, candidate.max_length)
    val_dataset = RumorDataset(build_samples(val_df), tokenizer, candidate.max_length)

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=candidate.epochs,
        per_device_train_batch_size=candidate.batch_size,
        per_device_eval_batch_size=candidate.eval_batch_size,
        gradient_accumulation_steps=candidate.grad_accumulation,
        learning_rate=candidate.learning_rate,
        weight_decay=candidate.weight_decay,
        warmup_ratio=candidate.warmup_ratio,
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
        fp16=torch.cuda.is_available(),
        seed=candidate.seed,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    prediction_output = trainer.predict(val_dataset)
    predictions = prediction_output.predictions.argmax(axis=-1)
    accuracy = accuracy_score(val_df[LABEL_COLUMN], predictions)
    f1 = f1_score(val_df[LABEL_COLUMN], predictions, average="binary", zero_division=0)
    report = classification_report(val_df[LABEL_COLUMN], predictions, digits=4)

    row = asdict(candidate)
    row.update(
        {
            "accuracy": accuracy,
            "f1": f1,
            "improvement_over_current_best": accuracy - CURRENT_BEST_ACCURACY,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "output_dir": str(output_dir),
        }
    )
    predictions_df = val_df.copy()
    predictions_df["prediction"] = predictions
    predictions_df.to_csv(OUTPUTS_DIR / f"{candidate.name}_val_predictions.csv", index=False)
    (OUTPUTS_DIR / f"{candidate.name}_metrics.txt").write_text(
        f"Candidate: {candidate.name}\nModel: {candidate.model_name}\nDevice: {row['device']}\nValidation accuracy: {accuracy:.4f}\nValidation F1: {f1:.4f}\n\n{report}\n",
        encoding="utf-8",
    )
    return {"row": row, "trainer": trainer, "tokenizer": tokenizer, "predictions": predictions, "report": report}


def save_official_best(candidate: Candidate, result: dict[str, Any], val_df: pd.DataFrame) -> None:
    result["trainer"].save_model(str(BERT_MODEL_DIR))
    result["tokenizer"].save_pretrained(str(BERT_MODEL_DIR))

    predictions_df = val_df.copy()
    predictions_df["prediction"] = result["predictions"]
    predictions_df.to_csv(BERT_VAL_PREDICTIONS_FILE, index=False)

    row = result["row"]
    BERT_METRICS_FILE.write_text(
        f"Model: {candidate.model_name}\nCandidate: {candidate.name}\nDevice: {row['device']}\nValidation accuracy: {row['accuracy']:.4f}\nValidation F1: {row['f1']:.4f}\n\n{result['report']}\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CUDA transformer optimization candidates.")
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N candidates")
    parser.add_argument("--only", nargs="*", default=None, help="Run only candidates with these names")
    parser.add_argument("--adopt", action="store_true", help="Replace official BERT artifacts if a candidate improves accuracy")
    return parser.parse_args()


def load_existing_results() -> list[dict[str, Any]]:
    if not RESULTS_FILE.exists():
        return []
    return pd.read_csv(RESULTS_FILE).to_dict("records")


def main() -> None:
    args = parse_args()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_dataset(TRAIN_FILE)
    val_df = load_dataset(VAL_FILE)
    candidates = candidate_list()
    if args.only:
        requested_names = set(args.only)
        candidates = [candidate for candidate in candidates if candidate.name in requested_names]
        missing_names = requested_names - {candidate.name for candidate in candidates}
        if missing_names:
            raise ValueError(f"Unknown candidate names: {', '.join(sorted(missing_names))}")
    if args.limit > 0:
        candidates = candidates[: args.limit]

    results: list[dict[str, Any]] = load_existing_results()
    best_result: dict[str, Any] | None = None
    best_candidate: Candidate | None = None

    for candidate in candidates:
        print("=" * 80)
        print(candidate)
        result = run_candidate(candidate, train_df, val_df)
        results = [row for row in results if row.get("name") != candidate.name]
        results.append(result["row"])
        print(f"Accuracy: {result['row']['accuracy']:.4f}")
        print(f"F1: {result['row']['f1']:.4f}")
        if best_result is None or result["row"]["accuracy"] > best_result["row"]["accuracy"]:
            best_result = result
            best_candidate = candidate
        pd.DataFrame(results).sort_values(by=["accuracy", "f1"], ascending=False).to_csv(RESULTS_FILE, index=False)

    if best_result is None or best_candidate is None:
        raise RuntimeError("No candidates were run.")

    best_row = best_result["row"]
    improved = best_row["accuracy"] > CURRENT_BEST_ACCURACY
    BEST_FILE.write_text(
        f"Best candidate: {best_candidate.name}\nModel: {best_candidate.model_name}\nValidation accuracy: {best_row['accuracy']:.4f}\nValidation F1: {best_row['f1']:.4f}\nImprovement over current best: {best_row['improvement_over_current_best']:.4f}\nAdopted: {bool(args.adopt and improved)}\n",
        encoding="utf-8",
    )
    if args.adopt and improved:
        save_official_best(best_candidate, best_result, val_df)

    print(BEST_FILE.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
