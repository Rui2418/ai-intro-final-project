from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "rumer2026"
TRAIN_FILE = DATA_DIR / "train.csv"
VAL_FILE = DATA_DIR / "val.csv"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"
EVENT_COLUMN = "event"
ID_COLUMN = "id"

MODEL_FILE = MODELS_DIR / "baseline_pipeline.joblib"
BERT_MODEL_DIR = MODELS_DIR / "bert_classifier"
BERT_METRICS_FILE = OUTPUTS_DIR / "bert_metrics.txt"
BERT_VAL_PREDICTIONS_FILE = OUTPUTS_DIR / "bert_val_predictions.csv"
VAL_PREDICTIONS_FILE = OUTPUTS_DIR / "val_predictions.csv"
METRICS_FILE = OUTPUTS_DIR / "metrics.txt"
EXAMPLES_FILE = OUTPUTS_DIR / "examples.csv"
