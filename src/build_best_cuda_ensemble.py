from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.config import LABEL_COLUMN, OUTPUTS_DIR

ENSEMBLE_MEMBERS = [
    "val_predictions_best_joint_all_models.csv",
    "continue_best_lr5e6_ep2_val_predictions.csv",
    "roberta_lr2e5_len256_ep4_seed123_val_predictions.csv",
]
OUTPUT_PREDICTIONS_FILE = OUTPUTS_DIR / "best_cuda_ensemble_val_predictions.csv"
OUTPUT_METRICS_FILE = OUTPUTS_DIR / "best_cuda_ensemble_metrics.txt"


def prediction_column(dataframe: pd.DataFrame) -> str:
    if "prediction" in dataframe.columns:
        return "prediction"
    if "predicted_label" in dataframe.columns:
        return "predicted_label"
    raise ValueError("Prediction file must contain prediction or predicted_label column.")


def main() -> None:
    frames = [pd.read_csv(OUTPUTS_DIR / member) for member in ENSEMBLE_MEMBERS]
    labels = frames[0][LABEL_COLUMN].to_numpy()
    prediction_sum = sum(frame[prediction_column(frame)].to_numpy() for frame in frames)
    predictions = (prediction_sum >= 2).astype(int)

    output = frames[0].copy()
    output["prediction"] = predictions
    output.to_csv(OUTPUT_PREDICTIONS_FILE, index=False)

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="binary", zero_division=0)
    report = classification_report(labels, predictions, digits=4)
    OUTPUT_METRICS_FILE.write_text(
        "Best CUDA validation ensemble\n"
        f"Models: {', '.join(member.removesuffix('_val_predictions.csv').removesuffix('.csv') for member in ENSEMBLE_MEMBERS)}\n"
        f"Validation accuracy: {accuracy:.4f}\n"
        f"Validation F1: {f1:.4f}\n\n"
        f"{report}\n",
        encoding="utf-8",
    )
    print(OUTPUT_METRICS_FILE.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
