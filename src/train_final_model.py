from __future__ import annotations

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.config import (
    TRAIN_FILE,
    VAL_FILE,
    TEXT_COLUMN,
    LABEL_COLUMN,
    ID_COLUMN,
    EVENT_COLUMN,
    MODELS_DIR,
    OUTPUTS_DIR,
)
from src.preprocess import clean_text


FINAL_MODEL_FILE = MODELS_DIR / "final_classifier_pipeline.joblib"
FINAL_PREDICTIONS_FILE = OUTPUTS_DIR / "val_predictions_final_model.csv"
FINAL_METRICS_FILE = OUTPUTS_DIR / "final_model_metrics.txt"


def load_dataset(path) -> pd.DataFrame:
    """
    读取数据，并使用项目统一的 clean_text 进行文本预处理。
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


def build_final_pipeline() -> Pipeline:
    """
    构造最终分类模型。

    当前实验得到的最优传统模型为：
        original preprocess
        + TF-IDF
        + Linear SVM

    最优参数：
        TF-IDF:
            ngram_range=(1,3)
            min_df=1
            max_df=0.9
            sublinear_tf=True

        Linear SVM:
            C=2.0
            class_weight=balanced
    """
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 3),
                    min_df=1,
                    max_df=0.9,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LinearSVC(
                    C=2.0,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def save_predictions(val_df: pd.DataFrame, predictions) -> None:
    """
    保存最终模型在验证集上的预测结果。
    """
    result_df = val_df.copy()
    result_df["prediction"] = predictions
    result_df["is_correct"] = result_df[LABEL_COLUMN] == result_df["prediction"]

    output_columns = [
        column
        for column in [
            ID_COLUMN,
            TEXT_COLUMN,
            LABEL_COLUMN,
            "prediction",
            "is_correct",
            EVENT_COLUMN,
        ]
        if column in result_df.columns
    ]

    result_df[output_columns].to_csv(FINAL_PREDICTIONS_FILE, index=False)


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_dataset(TRAIN_FILE)
    val_df = load_dataset(VAL_FILE)

    pipeline = build_final_pipeline()

    pipeline.fit(
        train_df["clean_text"],
        train_df[LABEL_COLUMN],
    )

    predictions = pipeline.predict(val_df["clean_text"])

    accuracy = accuracy_score(val_df[LABEL_COLUMN], predictions)
    report = classification_report(
        val_df[LABEL_COLUMN],
        predictions,
        digits=4,
    )

    joblib.dump(pipeline, FINAL_MODEL_FILE)
    save_predictions(val_df, predictions)

    metrics_text = (
        "Final Classifier Model\n\n"
        "Model:\n"
        "original preprocess + TF-IDF + Linear SVM\n\n"
        "TF-IDF parameters:\n"
        "ngram_range=(1,3)\n"
        "min_df=1\n"
        "max_df=0.9\n"
        "sublinear_tf=True\n\n"
        "Linear SVM parameters:\n"
        "C=2.0\n"
        "class_weight=balanced\n"
        "random_state=42\n\n"
        f"Validation accuracy: {accuracy:.4f}\n\n"
        f"{report}\n"
    )

    FINAL_METRICS_FILE.write_text(metrics_text, encoding="utf-8")

    print(metrics_text)
    print(f"Saved final classifier model to: {FINAL_MODEL_FILE}")
    print(f"Saved final validation predictions to: {FINAL_PREDICTIONS_FILE}")
    print(f"Saved final metrics to: {FINAL_METRICS_FILE}")


if __name__ == "__main__":
    main()