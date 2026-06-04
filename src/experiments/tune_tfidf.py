from __future__ import annotations

import joblib
import pandas as pd

from itertools import product

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


TUNING_RESULT_FILE = OUTPUTS_DIR / "tfidf_tuning_results.csv"
BEST_TUNED_MODEL_FILE = MODELS_DIR / "best_tuned_naive_bayes_pipeline.joblib"
BEST_TUNED_PREDICTIONS_FILE = OUTPUTS_DIR / "val_predictions_best_tuned.csv"
BEST_TUNED_METRICS_FILE = OUTPUTS_DIR / "best_tuned_metrics.txt"


def load_dataset(path) -> pd.DataFrame:
    """
    读取数据，并对文本进行清洗。
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


def build_pipeline(
    ngram_range: tuple[int, int],
    min_df: int,
    max_df: float,
    sublinear_tf: bool,
    alpha: float,
) -> Pipeline:
    """
    构造 TF-IDF + Naive Bayes 模型。

    alpha 是 Naive Bayes 的平滑参数：
    - alpha 越大，模型越保守
    - alpha 越小，模型越依赖训练数据中的词频统计
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
                MultinomialNB(alpha=alpha),
            ),
        ]
    )


def evaluate_pipeline(
    pipeline: Pipeline,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> tuple[float, float, float, float, object, str]:
    """
    训练并评估一个模型，返回 accuracy、precision、recall、f1、预测结果和详细报告。
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
    保存最佳调参模型在验证集上的预测结果。
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

    # 当前最优基准：TF-IDF + Naive Bayes accuracy = 0.8628
    baseline_accuracy = 0.8628

    # 调参搜索空间
    ngram_ranges = [(1, 1), (1, 2), (1, 3)]
    min_dfs = [1, 2, 3]
    max_dfs = [0.9, 0.95, 1.0]
    sublinear_tfs = [True, False]
    alphas = [0.1, 0.5, 1.0]

    results = []
    best_result = None

    total_experiments = (
        len(ngram_ranges)
        * len(min_dfs)
        * len(max_dfs)
        * len(sublinear_tfs)
        * len(alphas)
    )

    experiment_id = 0

    for ngram_range, min_df, max_df, sublinear_tf, alpha in product(
        ngram_ranges,
        min_dfs,
        max_dfs,
        sublinear_tfs,
        alphas,
    ):
        experiment_id += 1

        print("=" * 80)
        print(f"Experiment {experiment_id}/{total_experiments}")
        print(
            f"ngram_range={ngram_range}, "
            f"min_df={min_df}, "
            f"max_df={max_df}, "
            f"sublinear_tf={sublinear_tf}, "
            f"alpha={alpha}"
        )

        pipeline = build_pipeline(
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            alpha=alpha,
        )

        accuracy, precision, recall, f1, predictions, report = evaluate_pipeline(
            pipeline,
            train_df,
            val_df,
        )

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1: {f1:.4f}")

        result = {
            "ngram_range": str(ngram_range),
            "min_df": min_df,
            "max_df": max_df,
            "sublinear_tf": sublinear_tf,
            "alpha": alpha,
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
            }

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values(by="accuracy", ascending=False)
    result_df.to_csv(TUNING_RESULT_FILE, index=False)

    print("\nFinal TF-IDF tuning results:")
    print(result_df.head(10))

    print(f"\nSaved tuning results to: {TUNING_RESULT_FILE}")

    if best_result is not None:
        joblib.dump(best_result["pipeline"], BEST_TUNED_MODEL_FILE)
        save_prediction_file(
            val_df,
            best_result["predictions"],
            BEST_TUNED_PREDICTIONS_FILE,
        )

        improvement = best_result["accuracy"] - baseline_accuracy

        metrics_text = (
            "Best tuned TF-IDF + Naive Bayes model\n"
            f"ngram_range: {best_result['ngram_range']}\n"
            f"min_df: {best_result['min_df']}\n"
            f"max_df: {best_result['max_df']}\n"
            f"sublinear_tf: {best_result['sublinear_tf']}\n"
            f"alpha: {best_result['alpha']}\n\n"
            f"Accuracy: {best_result['accuracy']:.4f}\n"
            f"Precision: {best_result['precision']:.4f}\n"
            f"Recall: {best_result['recall']:.4f}\n"
            f"F1: {best_result['f1']:.4f}\n"
            f"Improvement over previous Naive Bayes baseline: {improvement:.4f}\n\n"
            f"{best_result['report']}\n"
        )

        BEST_TUNED_METRICS_FILE.write_text(metrics_text, encoding="utf-8")

        print("\nBest tuned model:")
        print(metrics_text)

        print(f"Saved best tuned model to: {BEST_TUNED_MODEL_FILE}")
        print(f"Saved best tuned predictions to: {BEST_TUNED_PREDICTIONS_FILE}")
        print(f"Saved best tuned metrics to: {BEST_TUNED_METRICS_FILE}")


if __name__ == "__main__":
    main()