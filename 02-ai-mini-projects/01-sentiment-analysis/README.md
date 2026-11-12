# Sentiment Analysis with Python

A beginner-friendly Natural Language Processing (NLP) project that classifies an English sentence as **positive** or **negative** using TF-IDF and Logistic Regression.

## What you'll learn
- Load and validate text data with Pandas
- Convert text into numerical features with TF-IDF
- Train a Logistic Regression classifier
- Evaluate a model on a stratified holdout set
- Save and load a trained model with Joblib
- Write basic automated tests with `unittest`

## Project structure
```text
01-sentiment-analysis/
├── data/
│   └── sentiment_dataset.csv
├── models/                  # Created after training; model file is not committed
├── tests/
│   └── test_sentiment.py
├── .gitignore
├── LICENSE
├── model.py
├── predict.py
├── requirements.txt
├── train.py
└── README.md
```

## Requirements
- Python 3.10 or newer
- pip

## Setup (Windows CMD)
Open CMD in this project folder, then run:

```bash
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train the model
```bash
python train.py
```

The script prints the dataset size, holdout accuracy, and a classification report. It saves the trained model to `models/sentiment_model.joblib`.

## Predict a sentence
```bash
python predict.py "I really enjoyed this product"
```

Or run `python predict.py` without an argument to enter a sentence interactively.

Example output:
```text
Sentiment: POSITIVE
Model confidence: <percentage>
Note: confidence is not a guarantee that the prediction is correct.
```
The confidence value can differ depending on the sentence and installed library versions.

Train the model before making predictions. The generated model file is excluded from Git because it can be recreated from the included code and dataset.

## Run tests
```bash
python -m unittest discover -s tests -v
```

## Dataset and limitations
The CSV contains a small educational corpus: examples generated from curated positive/negative word lists and sentence templates. This makes it useful for learning the workflow, but it is **not** a representative real-world review dataset. A random holdout score on this template-based corpus can be much higher than performance on real customer messages. The model may misclassify sarcasm, mixed opinions, unfamiliar wording, and non-English text. Do not use this demo model for important decisions.

## Technologies
Python · Pandas · scikit-learn · TF-IDF · Logistic Regression · Joblib

## License
MIT. See [LICENSE](LICENSE).
