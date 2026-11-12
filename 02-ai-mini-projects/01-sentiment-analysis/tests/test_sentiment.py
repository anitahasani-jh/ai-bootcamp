"""Basic tests for data loading and prediction."""

import unittest

from model import load_dataset, predict_sentiment, train_and_evaluate


class SentimentAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_dataset()
        cls.model, cls.metrics = train_and_evaluate()

    def test_dataset_has_both_labels(self):
        self.assertEqual(set(self.data["label"].unique()), {"positive", "negative"})

    def test_dataset_has_enough_examples(self):
        self.assertGreaterEqual(len(self.data), 40)

    def test_positive_sentence(self):
        label, confidence = predict_sentiment(
            "I love this wonderful product", self.model
        )
        self.assertEqual(label, "positive")
        self.assertGreaterEqual(confidence, 0.0)

    def test_negative_sentence(self):
        label, confidence = predict_sentiment(
            "This is awful and disappointing", self.model
        )
        self.assertEqual(label, "negative")
        self.assertGreaterEqual(confidence, 0.0)

    def test_empty_text_is_rejected(self):
        with self.assertRaises(ValueError):
            predict_sentiment("   ", self.model)


if __name__ == "__main__":
    unittest.main()
