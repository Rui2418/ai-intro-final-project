from __future__ import annotations

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)
from sklearn.pipeline import Pipeline

from src.config import (
    TRAIN_FILE,
    VAL_FILE,
    TEXT_COLUMN,
    LABEL_COLUMN,
    OUTPUTS_DIR,
    MODELS_DIR,
)
from src.preprocess import clean_text


COMPARISON_FILE = OUTPUTS_DIR / "model_comparison.csv"
SVM_PREDICTIONS_FILE = OUTPUTS_DIR / "val_predictions_svm.csv"
NB_PREDICTIONS_FILE = OUTPUTS_DIR / "val_predictions_naive_bayes.csv"
BEST_PREDICTIONS_FILE = OUTPUTS_DIR / "val_predictions_best.csv"
BEST_MODEL_FILE = MODELS_DIR / "best_classifier_pipeline.joblib"


def load_dataset(path) -> pd.DataFrame:
    """
    读取数据，并进行基础清洗。

    输入:
        path: train.csv 或 val.csv 的路径

    输出:
        带有 clean_text 列的 DataFrame
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


def build_tfidf_vectorizer() -> TfidfVectorizer:
    """
    构造 TF-IDF 文本向量器。

    ngram_range=(1, 2):
        同时使用单个词和连续两个词作为特征。

    min_df=2:
        只保留至少在 2 条文本中出现过的词，过滤极少见词。

    max_df=0.95:
        过滤在 95% 以上文本中都出现的过于常见词。

    sublinear_tf=True:
        对词频做 log 缩放，降低高频词的过强影响。
    """
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )


def build_models() -> list[tuple[str, Pipeline]]:
    """
    构造多个待对比的模型。

    每个模型都是:
        TF-IDF 文本向量化 + 分类器
    """
    models = [
        (
            "TF-IDF + Logistic Regression",
            Pipeline(
                steps=[
                    ("tfidf", build_tfidf_vectorizer()),
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=1000,
                            class_weight="balanced",
                            random_state=42,
                        ),
                    ),
                ]
            ),
        ),
        (
            "TF-IDF + Linear SVM",
            Pipeline(
                steps=[
                    ("tfidf", build_tfidf_vectorizer()),
                    (
                        "classifier",
                        LinearSVC(
                            class_weight="balanced",
                            random_state=42,
                        ),
                    ),
                ]
            ),
        ),
        (
            "TF-IDF + Naive Bayes",
            Pipeline(
                steps=[
                    ("tfidf", build_tfidf_vectorizer()),
                    ("classifier", MultinomialNB()),
                ]
            ),
        ),
    ]

    return models


def evaluate_model(
    name: str,
    model: Pipeline,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> dict:
    """
    训练并评估一个模型。
    """
    X_train = train_df["clean_text"]
    y_train = train_df[LABEL_COLUMN]

    X_val = val_df["clean_text"]
    y_val = val_df[LABEL_COLUMN]

    print("=" * 80)
    print(f"Training model: {name}")
    print("=" * 80)

    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    accuracy = accuracy_score(y_val, predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_val,
        predictions,
        average="binary",
        zero_division=0,
    )

    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_val, predictions, digits=4))

    return {
        "model": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predictions": predictions,
        "model_object": model,
    }


def save_prediction_file(
    val_df: pd.DataFrame,
    predictions,
    output_file,
) -> None:
    """
    保存某个模型在验证集上的预测结果。
    """
    result_df = val_df.copy()
    result_df["prediction"] = predictions

    columns = [
        column
        for column in ["id", TEXT_COLUMN, LABEL_COLUMN, "prediction", "event"]
        if column in result_df.columns
    ]

    result_df[columns].to_csv(output_file, index=False)


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_dataset(TRAIN_FILE)
    val_df = load_dataset(VAL_FILE)

    models = build_models()
    comparison_results = []
    best_result = None

    for name, model in models:
        result = evaluate_model(name, model, train_df, val_df)

        comparison_results.append(
            {
                "model": result["model"],
                "accuracy": result["accuracy"],
                "precision": result["precision"],
                "recall": result["recall"],
                "f1": result["f1"],
            }
        )

        if best_result is None or result["accuracy"] > best_result["accuracy"]:
            best_result = result

        if name == "TF-IDF + Linear SVM":
            save_prediction_file(
                val_df,
                result["predictions"],
                SVM_PREDICTIONS_FILE,
            )

        if name == "TF-IDF + Naive Bayes":
            save_prediction_file(
                val_df,
                result["predictions"],
                NB_PREDICTIONS_FILE,
            )

    comparison_df = pd.DataFrame(comparison_results)
    comparison_df = comparison_df.sort_values(by="accuracy", ascending=False)

    comparison_df.to_csv(COMPARISON_FILE, index=False)

    print("\nFinal model comparison:")
    print(comparison_df)

    print(f"\nSaved comparison result to: {COMPARISON_FILE}")
    print(f"Saved SVM predictions to: {SVM_PREDICTIONS_FILE}")
    print(f"Saved Naive Bayes predictions to: {NB_PREDICTIONS_FILE}")

    if best_result is not None:
        joblib.dump(best_result["model_object"], BEST_MODEL_FILE)
        save_prediction_file(
            val_df,
            best_result["predictions"],
            BEST_PREDICTIONS_FILE,
        )

        print(
            f"Saved best model ({best_result['model']}) "
            f"to: {BEST_MODEL_FILE}"
        )
        print(f"Saved best model predictions to: {BEST_PREDICTIONS_FILE}")


if __name__ == "__main__":
    main()