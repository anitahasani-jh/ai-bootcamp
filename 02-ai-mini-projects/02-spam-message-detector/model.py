"""Training and inference helpers for the SMS spam detector."""
from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_model() -> Pipeline:
    """Create an untrained TF-IDF + logistic-regression pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True, strip_accents="unicode")),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])


def train_model(messages, labels) -> Pipeline:
    """Fit and return a model using message text and spam/ham labels."""
    model = build_model()
    model.fit(messages, labels)
    return model


def predict_message(model: Pipeline, message: str) -> dict:
    """Return the predicted label and class probabilities for one message."""
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Message must be a non-empty string.")
    label = model.predict([message])[0]
    probabilities = model.predict_proba([message])[0]
    scores = dict(zip(model.classes_, probabilities))
    return {
        "label": str(label),
        "spam_probability": float(scores.get("spam", 0.0)),
        "ham_probability": float(scores.get("ham", 0.0)),
    }


def save_model(model: Pipeline, path: str | Path) -> None:
    """Save a trained model to disk."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination)


def load_model(path: str | Path) -> Pipeline:
    """Load a previously saved model."""
    return joblib.load(path)
