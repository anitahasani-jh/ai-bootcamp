"""Train the spam detector and report a held-out evaluation."""
from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from model import train_model, save_model


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "sms_messages.csv"
MODEL_PATH = ROOT / "models" / "spam_detector.joblib"


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    required = {"message", "label"}
    if not required.issubset(data.columns):
        raise ValueError(f"Dataset must contain columns: {sorted(required)}")
    data = data.dropna(subset=["message", "label"])
    data["label"] = data["label"].str.lower().str.strip()
    data = data[data["label"].isin(["spam", "ham"])]

    X_train, X_test, y_train, y_test = train_test_split(
        data["message"].astype(str),
        data["label"],
        test_size=0.2,
        random_state=42,
        stratify=data["label"],
    )
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)

    print("Evaluation on the held-out split")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix (labels: ham, spam):")
    print(confusion_matrix(y_test, predictions, labels=["ham", "spam"]))

    save_model(model, MODEL_PATH)
    print(f"\nSaved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
