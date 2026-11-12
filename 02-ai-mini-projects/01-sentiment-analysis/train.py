"""Train and evaluate the sentiment-analysis model."""

from model import save_model, train_and_evaluate


def main() -> None:
    model, metrics = train_and_evaluate()
    save_model(model)
    print("Sentiment model trained successfully.")
    print(f"Dataset size: {metrics['dataset_size']}")
    print(f"Test samples: {metrics['test_size']}")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print("\nClassification report:")
    print(metrics["report"])
    print("Model saved to models/sentiment_model.joblib")


if __name__ == "__main__":
    main()
