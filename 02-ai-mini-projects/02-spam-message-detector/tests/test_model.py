import unittest

from model import predict_message, train_model


class SpamDetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        messages = [
            "Hey, are we still meeting after class?",
            "I will call you when I get home.",
            "Please send me the notes from today's lecture.",
            "Are you free for lunch tomorrow?",
            "Thanks for helping me with the assignment.",
            "Your package will arrive this afternoon.",
            "Congratulations! Claim your cash prize now.",
            "URGENT: You have won a free gift. Click now.",
            "Get a limited offer and claim your reward today.",
            "You are selected for a cash bonus. Call now.",
            "Win a free voucher by claiming this special prize.",
            "Exclusive deal! Click the link to claim your offer.",
        ]
        labels = [
            "ham", "ham", "ham", "ham", "ham", "ham",
            "spam", "spam", "spam", "spam", "spam", "spam",
        ]
        cls.model = train_model(messages, labels)

    def test_predicts_obvious_spam(self):
        result = predict_message(self.model, "Congratulations, claim your free cash prize now!")
        self.assertEqual(result["label"], "spam")

    def test_predicts_normal_message(self):
        result = predict_message(self.model, "Are we still meeting at the library after class?")
        self.assertEqual(result["label"], "ham")

    def test_probabilities_are_between_zero_and_one(self):
        result = predict_message(self.model, "Please call me after class.")
        self.assertGreaterEqual(result["spam_probability"], 0.0)
        self.assertLessEqual(result["spam_probability"], 1.0)
        self.assertAlmostEqual(
            result["spam_probability"] + result["ham_probability"], 1.0, places=6
        )

    def test_empty_message_is_rejected(self):
        with self.assertRaises(ValueError):
            predict_message(self.model, "   ")


if __name__ == "__main__":
    unittest.main()
