from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import BERT_MODEL_DIR, MODEL_FILE, TEXT_COLUMN
from src.explain import generate_explanation
from src.preprocess import clean_text


def load_baseline_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_FILE}. Run `python -m src.train_baseline` first."
        )
    return joblib.load(MODEL_FILE)


class BertPredictor:
    def __init__(self, model_dir: Path) -> None:
        if not model_dir.exists():
            raise FileNotFoundError(
                f"BERT model directory not found: {model_dir}. Run `python -m src.train_bert` first."
            )
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def predict(self, texts: list[str]) -> list[int]:
        encoded = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = self.model(**encoded).logits
        return logits.argmax(dim=-1).cpu().tolist()


def predict_with_baseline(text: str) -> int:
    model = load_baseline_model()
    cleaned_text = clean_text(text)
    return int(model.predict([cleaned_text])[0])


def predict_with_bert(text: str) -> int:
    predictor = BertPredictor(BERT_MODEL_DIR)
    return int(predictor.predict([text])[0])


def predict_text(text: str, model_type: str = "bert") -> tuple[int, str]:
    if model_type == "baseline":
        label = predict_with_baseline(text)
    elif model_type == "bert":
        label = predict_with_bert(text)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    reason = generate_explanation(text, label)
    return label, reason


def predict_dataframe(dataframe: pd.DataFrame, model_type: str = "bert") -> pd.DataFrame:
    result = dataframe.copy()
    result[TEXT_COLUMN] = result[TEXT_COLUMN].fillna("").astype(str)

    if model_type == "baseline":
        model = load_baseline_model()
        cleaned_texts = result[TEXT_COLUMN].map(clean_text)
        predictions = model.predict(cleaned_texts)
    elif model_type == "bert":
        predictor = BertPredictor(BERT_MODEL_DIR)
        predictions = predictor.predict(result[TEXT_COLUMN].tolist())
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    result["prediction"] = predictions
    result["reason"] = [
        generate_explanation(text, int(label))
        for text, label in zip(result[TEXT_COLUMN], result["prediction"])
    ]
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict rumor labels for one CSV file.")
    parser.add_argument("--input", required=True, help="Input CSV path with a text column")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument(
        "--model-type",
        choices=["baseline", "bert"],
        default="bert",
        help="Model family to use for prediction",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.input)
    result = predict_dataframe(data, model_type=args.model_type)
    result.to_csv(args.output, index=False)
    print(f"Saved predictions to: {args.output}")


if __name__ == "__main__":
    main()
