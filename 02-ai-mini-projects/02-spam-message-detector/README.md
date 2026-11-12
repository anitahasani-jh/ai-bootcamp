# SMS Spam Message Detector

A beginner-friendly machine-learning project that classifies a text message as **spam** or **ham** (a normal message).

The project uses:
- **TF-IDF** to convert message text into numerical features.
- **Logistic Regression** to learn patterns associated with spam and normal messages.
- **scikit-learn** to build a reusable machine-learning pipeline.

## Project structure

```text
02-spam-message-detector/
├── data/
│   └── sms_messages.csv
├── models/
│   └── (created after training; model files are not committed)
├── tests/
│   └── test_model.py
├── .gitignore
├── model.py
├── predict.py
├── requirements.txt
└── train.py
```

## Setup

Python 3.10 or newer is recommended.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Train the model

From this project directory, run:

```bash
python train.py
```

The script reads `data/sms_messages.csv`, creates a reproducible train/test split, prints a classification report and confusion matrix, then saves the trained pipeline to `models/spam_detector.joblib`.

## Classify a message

After training, run:

```bash
python predict.py "Congratulations! You have won a free prize. Claim now."
```

Try a normal message too:

```bash
python predict.py "Hi, are we still meeting at the library?"
```

## Run tests

```bash
python -m unittest discover -s tests -v
```

The tests cover clear example messages, probability output, and empty-input validation.

## Dataset and limitations

The included CSV is a **small synthetic teaching dataset** created for learning and demonstration. It is not a representative collection of real SMS messages. As a result, the model's predictions and evaluation scores should not be treated as reliable real-world spam detection.

For a stronger project, replace the sample data with a properly licensed, documented public SMS spam dataset, inspect class balance, and evaluate on data that was not used during training. Avoid entering private or sensitive messages.

## What you can learn

1. How text is cleaned and represented with TF-IDF.
2. How a scikit-learn Pipeline combines preprocessing and a classifier.
3. How to split data into training and test sets.
4. How to read precision, recall, F1-score, and a confusion matrix.
5. How to save and reuse a trained model.

## License

This project is available under the MIT License. See [LICENSE](LICENSE).
