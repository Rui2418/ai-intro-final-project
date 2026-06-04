from __future__ import annotations

import html
import re
import joblib
import pandas as pd

from itertools import product

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


JOINT_ALL_RESULT_FILE = OUTPUTS_DIR / "joint_search_all_models_results.csv"
BEST_JOINT_ALL_MODEL_FILE = MODELS_DIR / "best_joint_all_models_pipeline.joblib"
BEST_JOINT_ALL_PREDICTIONS_FILE = OUTPUTS_DIR / "val_predictions_best_joint_all_models.csv"
BEST_JOINT_ALL_METRICS_FILE = OUTPUTS_DIR / "best_joint_all_models_metrics.txt"


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
    策略 3：保留 hashtag 内容。
    例如 #BreakingNews -> breakingnews
    """
    text = normalize_basic(text)
    text = URL_PATTERN.sub(" <url> ", text)
    text = MENTION_PATTERN.sub(" <user> ", text)
    text = HASHTAG_PATTERN.sub(r" \1 ", text)
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def clean_remove_punctuation(text: str) -> str:
    """
    策略 4：在保留 URL / USER 特殊标记基础上去除大部分标点。
    """
    text = normalize_basic(text)
    text = URL_PATTERN.sub(" <url> ", text)
    text = MENTION_PATTERN.sub(" <user> ", text)
    text = HASHTAG_PATTERN.sub(r" \1 ", text)
    text = PUNCT_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


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


def build_classifier(classifier_name: str, classifier_param: float):
    """
    根据分类器名称和参数构造分类器。

    Logistic Regression:
        classifier_param 表示 C

    Linear SVM:
        classifier_param 表示 C

    Naive Bayes:
        classifier_param 表示 alpha
    """
    if classifier_name == "logistic_regression":
        return LogisticRegression(
            C=classifier_param,
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )

    if classifier_name == "linear_svm":
        return LinearSVC(
            C=classifier_param,
            class_weight="balanced",
            random_state=42,
        )

    if classifier_name == "naive_bayes":
        return MultinomialNB(alpha=classifier_param)

    raise ValueError(f"Unknown classifier: {classifier_name}")


def build_pipeline(
    ngram_range: tuple[int, int],
    min_df: int,
    max_df: float,
    sublinear_tf: bool,
    classifier_name: str,
    classifier_param: float,
) -> Pipeline:
    """
    构造完整 Pipeline:
        TF-IDF + classifier
    """
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=ngram_range,
                    min_df=min_df,
                    max_df=max_df,
                    sublinear_tf=sublinear_tf,
                ),
            ),
            (
                "classifier",
                build_classifier(classifier_name, classifier_param),
            ),
        ]
    )


def evaluate_pipeline(
    pipeline: Pipeline,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> tuple[float, float, float, float, object, str]:
    """
    训练并评估一个 pipeline。
    """
    X_train = train_df["clean_text"]
    y_train = train_df[LABEL_COLUMN]

    X_val = val_df["clean_text"]
    y_val = val_df[LABEL_COLUMN]

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

    return accuracy, precision, recall, f1, predictions, report


def save_prediction_file(
    val_df: pd.DataFrame,
    predictions,
    output_file,
) -> None:
    """
    保存最佳模型在验证集上的预测结果。
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

    # 当前已经得到的分阶段最优结果，用来做对照
    previous_best_accuracy = 0.8653

    preprocess_strategies = [
        ("original", clean_original),
        ("remove_url_user", clean_remove_url_user),
        ("hashtag_split", clean_hashtag_split),
        ("remove_punctuation", clean_remove_punctuation),
    ]

    # 轻量搜索空间，避免实验数量过大
    ngram_ranges = [(1, 2), (1, 3)]
    min_dfs = [1, 2]
    max_dfs = [0.9, 0.95]
    sublinear_tfs = [True, False]

    classifier_spaces = {
        "logistic_regression": [0.5, 1.0, 2.0],
        "linear_svm": [0.5, 1.0, 2.0],
        "naive_bayes": [0.5, 1.0],
    }

    total_experiments = (
        len(preprocess_strategies)
        * len(ngram_ranges)
        * len(min_dfs)
        * len(max_dfs)
        * len(sublinear_tfs)
        * sum(len(params) for params in classifier_spaces.values())
    )

    experiment_id = 0
    results = []
    best_result = None

    for strategy_name, preprocess_func in preprocess_strategies:
        # 同一种预处理策略下，清洗一次即可
        train_df = load_dataset(TRAIN_FILE, preprocess_func)
        val_df = load_dataset(VAL_FILE, preprocess_func)

        for ngram_range, min_df, max_df, sublinear_tf in product(
            ngram_ranges,
            min_dfs,
            max_dfs,
            sublinear_tfs,
        ):
            for classifier_name, classifier_params in classifier_spaces.items():
                for classifier_param in classifier_params:
                    experiment_id += 1

                    print("=" * 80)
                    print(f"Experiment {experiment_id}/{total_experiments}")
                    print(
                        f"strategy={strategy_name}, "
                        f"ngram_range={ngram_range}, "
                        f"min_df={min_df}, "
                        f"max_df={max_df}, "
                        f"sublinear_tf={sublinear_tf}, "
                        f"classifier={classifier_name}, "
                        f"classifier_param={classifier_param}"
                    )

                    pipeline = build_pipeline(
                        ngram_range=ngram_range,
                        min_df=min_df,
                        max_df=max_df,
                        sublinear_tf=sublinear_tf,
                        classifier_name=classifier_name,
                        classifier_param=classifier_param,
                    )

                    (
                        accuracy,
                        precision,
                        recall,
                        f1,
                        predictions,
                        report,
                    ) = evaluate_pipeline(
                        pipeline,
                        train_df,
                        val_df,
                    )

                    print(f"Accuracy: {accuracy:.4f}")
                    print(f"Precision: {precision:.4f}")
                    print(f"Recall: {recall:.4f}")
                    print(f"F1: {f1:.4f}")

                    result = {
                        "strategy": strategy_name,
                        "ngram_range": str(ngram_range),
                        "min_df": min_df,
                        "max_df": max_df,
                        "sublinear_tf": sublinear_tf,
                        "classifier": classifier_name,
                        "classifier_param": classifier_param,
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                    }

                    results.append(result)

                    if best_result is None or accuracy > best_result["accuracy"]:
                        best_result = {
                            **result,
                            "pipeline": pipeline,
                            "predictions": predictions,
                            "report": report,
                            "val_df": val_df,
                        }

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values(by="accuracy", ascending=False)
    result_df.to_csv(JOINT_ALL_RESULT_FILE, index=False)

    print("\nFinal joint search over all models:")
    print(result_df.head(15))

    print(f"\nSaved joint search results to: {JOINT_ALL_RESULT_FILE}")

    if best_result is not None:
        joblib.dump(best_result["pipeline"], BEST_JOINT_ALL_MODEL_FILE)

        save_prediction_file(
            best_result["val_df"],
            best_result["predictions"],
            BEST_JOINT_ALL_PREDICTIONS_FILE,
        )

        improvement = best_result["accuracy"] - previous_best_accuracy

        metrics_text = (
            "Best joint search model over preprocessing, TF-IDF, and classifiers\n"
            f"strategy: {best_result['strategy']}\n"
            f"ngram_range: {best_result['ngram_range']}\n"
            f"min_df: {best_result['min_df']}\n"
            f"max_df: {best_result['max_df']}\n"
            f"sublinear_tf: {best_result['sublinear_tf']}\n"
            f"classifier: {best_result['classifier']}\n"
            f"classifier_param: {best_result['classifier_param']}\n\n"
            f"Accuracy: {best_result['accuracy']:.4f}\n"
            f"Precision: {best_result['precision']:.4f}\n"
            f"Recall: {best_result['recall']:.4f}\n"
            f"F1: {best_result['f1']:.4f}\n"
            f"Improvement over previous staged best: {improvement:.4f}\n\n"
            f"{best_result['report']}\n"
        )

        BEST_JOINT_ALL_METRICS_FILE.write_text(metrics_text, encoding="utf-8")

        print("\nBest joint search model:")
        print(metrics_text)

        print(f"Saved best joint all-model pipeline to: {BEST_JOINT_ALL_MODEL_FILE}")
        print(f"Saved best joint all-model predictions to: {BEST_JOINT_ALL_PREDICTIONS_FILE}")
        print(f"Saved best joint all-model metrics to: {BEST_JOINT_ALL_METRICS_FILE}")


if __name__ == "__main__":
    main()