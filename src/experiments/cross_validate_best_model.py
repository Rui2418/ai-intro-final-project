from __future__ import annotations

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.config import (
    TRAIN_FILE,
    TEXT_COLUMN,
    LABEL_COLUMN,
    OUTPUTS_DIR,
)
from src.preprocess import clean_text


CV_RESULT_FILE = OUTPUTS_DIR / "cross_validation_results.csv"
CV_SUMMARY_FILE = OUTPUTS_DIR / "cross_validation_summary.txt"


def load_dataset(path) -> pd.DataFrame:
    """
    读取 train.csv，并使用项目原始 clean_text 进行预处理。
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
    构造联合搜索得到的当前最佳模型：

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


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    data = load_dataset(TRAIN_FILE)

    X = data["clean_text"]
    y = data[LABEL_COLUMN]

    # 5 折分层交叉验证：
    # 分层的意思是每一折中 0/1 标签比例尽量接近原始数据。
    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    fold_results = []

    for fold_id, (train_index, dev_index) in enumerate(skf.split(X, y), start=1):
        print("=" * 80)
        print(f"Fold {fold_id}")
        print("=" * 80)

        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]

        X_dev = X.iloc[dev_index]
        y_dev = y.iloc[dev_index]

        pipeline = build_best_pipeline()
        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_dev)

        accuracy = accuracy_score(y_dev, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_dev,
            predictions,
            average="binary",
            zero_division=0,
        )

        result = {
            "fold": fold_id,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "train_size": len(train_index),
            "dev_size": len(dev_index),
        }

        fold_results.append(result)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1: {f1:.4f}")

    result_df = pd.DataFrame(fold_results)
    result_df.to_csv(CV_RESULT_FILE, index=False)

    summary = {
        "accuracy_mean": result_df["accuracy"].mean(),
        "accuracy_std": result_df["accuracy"].std(),
        "precision_mean": result_df["precision"].mean(),
        "precision_std": result_df["precision"].std(),
        "recall_mean": result_df["recall"].mean(),
        "recall_std": result_df["recall"].std(),
        "f1_mean": result_df["f1"].mean(),
        "f1_std": result_df["f1"].std(),
    }

    summary_text = (
        "5-Fold Cross Validation for Best Joint Model\n\n"
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
        f"Accuracy:  mean={summary['accuracy_mean']:.4f}, "
        f"std={summary['accuracy_std']:.4f}\n"
        f"Precision: mean={summary['precision_mean']:.4f}, "
        f"std={summary['precision_std']:.4f}\n"
        f"Recall:    mean={summary['recall_mean']:.4f}, "
        f"std={summary['recall_std']:.4f}\n"
        f"F1:        mean={summary['f1_mean']:.4f}, "
        f"std={summary['f1_std']:.4f}\n\n"
        "Fold details:\n"
        f"{result_df.to_string(index=False)}\n"
    )

    CV_SUMMARY_FILE.write_text(summary_text, encoding="utf-8")

    print("\nCross validation summary:")
    print(summary_text)

    print(f"Saved cross validation results to: {CV_RESULT_FILE}")
    print(f"Saved cross validation summary to: {CV_SUMMARY_FILE}")


if __name__ == "__main__":
    main()