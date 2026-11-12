"""Training and prediction helpers for the sentiment classifier."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "sentiment_dataset.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "sentiment_model.joblib"


def build_model() -> Pipeline:
    """Create a reproducible TF-IDF + Logistic Regression pipeline."""
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2))),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def load_dataset(data_path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load and validate the CSV dataset."""
    data = pd.read_csv(data_path)
    required_columns = {"text", "label"}
    if not required_columns.issubset(data.columns):
        raise ValueError("Dataset must contain 'text' and 'label' columns.")
    data = data.dropna(subset=["text", "label"])
    data["text"] = data["text"].astype(str).str.strip()
    data["label"] = data["label"].astype(str).str.strip().str.lower()
    data = data[data["text"] != ""]
    if set(data["label"].unique()) != {"positive", "negative"}:
        raise ValueError("Labels must contain exactly 'positive' and 'negative'.")
    if data["label"].value_counts().min() < 2:
        raise ValueError("At least two examples per class are required.")
    return data


def train_and_evaluate(data_path: Path = DEFAULT_DATA_PATH):
    """Train the model and return it with simple holdout metrics."""
    data = load_dataset(data_path)
    x_train, x_test, y_train, y_test = train_test_split(
        data["text"],
        data["label"],
        test_size=0.25,
        random_state=42,
        stratify=data["label"],
    )
    model = build_model()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "report": classification_report(y_test, predictions, zero_division=0),
        "test_size": len(y_test),
        "dataset_size": len(data),
    }
    return model, metrics


def save_model(model: Pipeline, model_path: Path = DEFAULT_MODEL_PATH) -> None:
    """Save a trained model to disk."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


def load_model(model_path: Path = DEFAULT_MODEL_PATH) -> Pipeline:
    """Load a previously trained model."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run 'python train.py' first."
        )
    return joblib.load(model_path)


def predict_sentiment(text: str, model: Pipeline) -> tuple[str, float]:
    """Return the predicted label and its model confidence."""
    cleaned_text = text.strip()
    if not cleaned_text:
        raise ValueError("Please enter a non-empty sentence.")
    probabilities = model.predict_proba([cleaned_text])[0]
    best_index = probabilities.argmax()
    label = str(model.classes_[best_index])
    confidence = float(probabilities[best_index])
    return label, confidence
