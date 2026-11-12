"""Predict whether a message looks like spam or a normal message."""
import argparse
from pathlib import Path

from model import load_model, predict_message


MODEL_PATH = Path(__file__).resolve().parent / "models" / "spam_detector.joblib"


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a message as spam or ham.")
    parser.add_argument("message", help="The message text to classify.")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit("Model not found. Run 'python train.py' first.")

    model = load_model(MODEL_PATH)
    result = predict_message(model, args.message)
    print(f"Prediction: {result['label'].upper()}")
    print(f"Spam probability: {result['spam_probability']:.1%}")
    print(f"Normal-message probability: {result['ham_probability']:.1%}")


if __name__ == "__main__":
    main()
