from __future__ import annotations

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from src.config import (
    TRAIN_FILE,
    VAL_FILE,
    TEXT_COLUMN,
    LABEL_COLUMN,
    EVENT_COLUMN,
    ID_COLUMN,
    OUTPUTS_DIR,
)
from src.preprocess import clean_text


ERROR_CASES_FILE = OUTPUTS_DIR / "error_cases_best_joint.csv"
ERROR_SUMMARY_FILE = OUTPUTS_DIR / "error_summary_best_joint.txt"
EVENT_ACCURACY_FILE = OUTPUTS_DIR / "event_accuracy_best_joint.csv"
PREDICTIONS_FILE = OUTPUTS_DIR / "val_predictions_best_joint_for_error_analysis.csv"


def load_dataset(path) -> pd.DataFrame:
    """
    读取数据，并使用项目原始 clean_text 进行文本预处理。
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

    return data


def build_best_pipeline() -> Pipeline:
    """
    构造当前联合搜索得到的最佳模型。

    best model:
        original preprocess
        + TF-IDF:
            ngram_range=(1,3)
            min_df=1
            max_df=0.9
            sublinear_tf=True
        + Linear SVM:
            C=2.0
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


def save_event_accuracy(result_df: pd.DataFrame) -> pd.DataFrame:
    """
    按 event 统计每个事件上的准确率和错误数量。
    """
    if EVENT_COLUMN not in result_df.columns:
        return pd.DataFrame()

    event_summary = (
        result_df.groupby(EVENT_COLUMN)
        .agg(
            sample_count=(LABEL_COLUMN, "size"),
            correct_count=("is_correct", "sum"),
            error_count=("is_correct", lambda x: int((~x).sum())),
            accuracy=("is_correct", "mean"),
        )
        .reset_index()
        .sort_values(by=["accuracy", "sample_count"], ascending=[True, False])
    )

    event_summary.to_csv(EVENT_ACCURACY_FILE, index=False)
    return event_summary


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_dataset(TRAIN_FILE)
    val_df = load_dataset(VAL_FILE)

    pipeline = build_best_pipeline()
    pipeline.fit(train_df["clean_text"], train_df[LABEL_COLUMN])

    predictions = pipeline.predict(val_df["clean_text"])

    result_df = val_df.copy()
    result_df["prediction"] = predictions
    result_df["is_correct"] = result_df[LABEL_COLUMN] == result_df["prediction"]

    accuracy = accuracy_score(result_df[LABEL_COLUMN], result_df["prediction"])
    report = classification_report(
        result_df[LABEL_COLUMN],
        result_df["prediction"],
        digits=4,
    )

    cm = confusion_matrix(
        result_df[LABEL_COLUMN],
        result_df["prediction"],
        labels=[0, 1],
    )

    # confusion matrix:
    # rows = true labels, columns = predicted labels
    true_0_pred_0 = int(cm[0][0])
    true_0_pred_1 = int(cm[0][1])
    true_1_pred_0 = int(cm[1][0])
    true_1_pred_1 = int(cm[1][1])

    error_df = result_df[result_df["is_correct"] == False].copy()

    # 标记错误类型
    def get_error_type(row) -> str:
        true_label = int(row[LABEL_COLUMN])
        pred_label = int(row["prediction"])

        if true_label == 0 and pred_label == 1:
            return "false_positive_non_rumor_as_rumor"

        if true_label == 1 and pred_label == 0:
            return "false_negative_rumor_as_non_rumor"

        return "unknown"

    error_df["error_type"] = error_df.apply(get_error_type, axis=1)

    # 保存完整预测结果
    prediction_columns = [
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
    result_df[prediction_columns].to_csv(PREDICTIONS_FILE, index=False)

    # 保存错误样例
    error_columns = [
        column
        for column in [
            ID_COLUMN,
            TEXT_COLUMN,
            LABEL_COLUMN,
            "prediction",
            "error_type",
            EVENT_COLUMN,
        ]
        if column in error_df.columns
    ]
    error_df[error_columns].to_csv(ERROR_CASES_FILE, index=False)

    # 按 event 统计准确率
    event_summary = save_event_accuracy(result_df)

    total_count = len(result_df)
    error_count = len(error_df)
    false_positive_count = int(
        ((result_df[LABEL_COLUMN] == 0) & (result_df["prediction"] == 1)).sum()
    )
    false_negative_count = int(
        ((result_df[LABEL_COLUMN] == 1) & (result_df["prediction"] == 0)).sum()
    )

    summary_text = (
        "Error Analysis for Best Joint Model\n\n"
        "Model:\n"
        "original preprocess + TF-IDF + Linear SVM\n\n"
        "TF-IDF parameters:\n"
        "ngram_range=(1,3)\n"
        "min_df=1\n"
        "max_df=0.9\n"
        "sublinear_tf=True\n\n"
        "Linear SVM parameters:\n"
        "C=2.0\n"
        "class_weight=balanced\n\n"
        f"Total validation samples: {total_count}\n"
        f"Correct predictions: {total_count - error_count}\n"
        f"Error predictions: {error_count}\n"
        f"Accuracy: {accuracy:.4f}\n\n"
        "Confusion Matrix:\n"
        "Rows = true labels, columns = predicted labels\n"
        f"true 0 predicted 0: {true_0_pred_0}\n"
        f"true 0 predicted 1: {true_0_pred_1}\n"
        f"true 1 predicted 0: {true_1_pred_0}\n"
        f"true 1 predicted 1: {true_1_pred_1}\n\n"
        "Error Types:\n"
        f"False Positive, true non-rumor predicted as rumor: {false_positive_count}\n"
        f"False Negative, true rumor predicted as non-rumor: {false_negative_count}\n\n"
        "Classification Report:\n"
        f"{report}\n"
    )

    if not event_summary.empty:
        summary_text += "\nLowest event-level accuracy:\n"
        summary_text += event_summary.head(10).to_string(index=False)
        summary_text += "\n"

    ERROR_SUMMARY_FILE.write_text(summary_text, encoding="utf-8")

    print(summary_text)

    print(f"Saved full validation predictions to: {PREDICTIONS_FILE}")
    print(f"Saved error cases to: {ERROR_CASES_FILE}")
    print(f"Saved event accuracy to: {EVENT_ACCURACY_FILE}")
    print(f"Saved error summary to: {ERROR_SUMMARY_FILE}")


if __name__ == "__main__":
    main()