"""Predict whether an English sentence has positive or negative sentiment."""

import argparse

from model import load_model, predict_sentiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict the sentiment of a sentence."
    )
    parser.add_argument("text", nargs="?", help="The sentence to analyze")
    args = parser.parse_args()
    text = args.text if args.text is not None else input("Enter an English sentence: ")

    try:
        model = load_model()
        label, confidence = predict_sentiment(text, model)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    print(f"Sentiment: {label.upper()}")
    print(f"Model confidence: {confidence:.1%}")
    print("Note: confidence is not a guarantee that the prediction is correct.")


if __name__ == "__main__":
    main()
