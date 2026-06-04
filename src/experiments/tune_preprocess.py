from __future__ import annotations

import html
import re
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
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


PREPROCESS_RESULT_FILE = OUTPUTS_DIR / "preprocess_tuning_results.csv"
BEST_PREPROCESS_MODEL_FILE = MODELS_DIR / "best_preprocess_naive_bayes_pipeline.joblib"
BEST_PREPROCESS_PREDICTIONS_FILE = OUTPUTS_DIR / "val_predictions_best_preprocess.csv"
BEST_PREPROCESS_METRICS_FILE = OUTPUTS_DIR / "best_preprocess_metrics.txt"


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#(\w+)")
PUNCT_PATTERN = re.compile(r"[^\w\s<>]")
WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_basic(text: str) -> str:
    """
    基础规范化：
    1. 非字符串转为空串
    2. HTML 反转义
    3. 去首尾空格
    4. 转小写
    """
    if not isinstance(text, str):
        return ""

    text = html.unescape(text)
    text = text.strip().lower()
    return text


def clean_original(text: str) -> str:
    """
    策略 1：使用项目原始 clean_text。
    """
    return clean_text(text)


def clean_remove_url_user(text: str) -> str:
    """
    策略 2：直接删除 URL 和 @用户。
    """
    text = normalize_basic(text)
    text = URL_PATTERN.sub(" ", text)
    text = MENTION_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def clean_hashtag_split(text: str) -> str:
    """
    策略 3：
    保留 hashtag 的词内容。
    例如 #BreakingNews -> breakingnews

    URL 和 @用户仍然替换为特殊标记。
    """
    text = normalize_basic(text)
    text = URL_PATTERN.sub(" <url> ", text)
    text = MENTION_PATTERN.sub(" <user> ", text)
    text = HASHTAG_PATTERN.sub(r" \1 ", text)
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def clean_remove_punctuation(text: str) -> str:
    """
    策略 4：
    在原有 URL、@用户特殊标记基础上，去除大部分标点符号。
    """
    text = normalize_basic(text)
    text = URL_PATTERN.sub(" <url> ", text)
    text = MENTION_PATTERN.sub(" <user> ", text)
    text = HASHTAG_PATTERN.sub(r" \1 ", text)
    text = PUNCT_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def build_pipeline() -> Pipeline:
    """
    使用当前调参得到的最佳 TF-IDF + Naive Bayes 参数。
    """
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.9,
                    sublinear_tf=False,
                ),
            ),
            (
                "classifier",
                MultinomialNB(alpha=1.0),
            ),
        ]
    )


def load_dataset(path, preprocess_func) -> pd.DataFrame:
    """
    读取数据，并使用指定预处理函数生成 clean_text。
    """
    data = pd.read_csv(path)

    required_columns = [TEXT_COLUMN, LABEL_COLUMN]
    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns in {path}: {missing_columns}")

    data = data.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN]).copy()
    data["clean_text"] = data[TEXT_COLUMN].map(preprocess_func)

    return data


def evaluate_preprocess_strategy(
    strategy_name: str,
    preprocess_func,
) -> dict:
    """
    对一种预处理策略进行训练和验证。
    """
    print("=" * 80)
    print(f"Evaluating preprocess strategy: {strategy_name}")
    print("=" * 80)

    train_df = load_dataset(TRAIN_FILE, preprocess_func)
    val_df = load_dataset(VAL_FILE, preprocess_func)

    X_train = train_df["clean_text"]
    y_train = train_df[LABEL_COLUMN]

    X_val = val_df["clean_text"]
    y_val = val_df[LABEL_COLUMN]

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_val)

    accuracy = accuracy_score(y_val, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_val,
        predictions,
        average="binary",
        zero_division=0,
    )
    report = classification_report(y_val, predictions, digits=4)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    print(report)

    return {
        "strategy": strategy_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pipeline": pipeline,
        "predictions": predictions,
        "report": report,
        "val_df": val_df,
    }


def save_prediction_file(
    val_df: pd.DataFrame,
    predictions,
    output_file,
) -> None:
    """
    保存最佳预处理策略对应的验证集预测结果。
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

    strategies = [
        ("original", clean_original),
        ("remove_url_user", clean_remove_url_user),
        ("hashtag_split", clean_hashtag_split),
        ("remove_punctuation", clean_remove_punctuation),
    ]

    results = []
    best_result = None

    for strategy_name, preprocess_func in strategies:
        result = evaluate_preprocess_strategy(strategy_name, preprocess_func)

        results.append(
            {
                "strategy": result["strategy"],
                "accuracy": result["accuracy"],
                "precision": result["precision"],
                "recall": result["recall"],
                "f1": result["f1"],
            }
        )

        if best_result is None or result["accuracy"] > best_result["accuracy"]:
            best_result = result

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values(by="accuracy", ascending=False)
    result_df.to_csv(PREPROCESS_RESULT_FILE, index=False)

    print("\nFinal preprocess tuning results:")
    print(result_df)

    print(f"\nSaved preprocess tuning results to: {PREPROCESS_RESULT_FILE}")

    if best_result is not None:
        joblib.dump(best_result["pipeline"], BEST_PREPROCESS_MODEL_FILE)
        save_prediction_file(
            best_result["val_df"],
            best_result["predictions"],
            BEST_PREPROCESS_PREDICTIONS_FILE,
        )

        metrics_text = (
            "Best preprocess strategy for TF-IDF + Naive Bayes\n"
            f"strategy: {best_result['strategy']}\n\n"
            f"Accuracy: {best_result['accuracy']:.4f}\n"
            f"Precision: {best_result['precision']:.4f}\n"
            f"Recall: {best_result['recall']:.4f}\n"
            f"F1: {best_result['f1']:.4f}\n\n"
            f"{best_result['report']}\n"
        )

        BEST_PREPROCESS_METRICS_FILE.write_text(metrics_text, encoding="utf-8")

        print("\nBest preprocess strategy:")
        print(metrics_text)

        print(f"Saved best preprocess model to: {BEST_PREPROCESS_MODEL_FILE}")
        print(f"Saved best preprocess predictions to: {BEST_PREPROCESS_PREDICTIONS_FILE}")
        print(f"Saved best preprocess metrics to: {BEST_PREPROCESS_METRICS_FILE}")


if __name__ == "__main__":
    main()