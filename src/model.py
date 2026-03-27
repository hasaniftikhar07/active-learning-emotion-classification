"""
Baseline model for emotion classification.

This module defines functions to build a baseline text classification model using
scikit‑learn.  A simple pipeline with TF‑IDF vectorisation followed by logistic
regression is provided as a starting point.  Additional models or architectures
can be defined here.
"""
from typing import Any, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


def build_model() -> Pipeline:
    """Build and return a baseline classification model.

    The model consists of a TF‑IDF vectoriser followed by a logistic regression
    classifier.  Hyperparameters are chosen for generality; feel free to adjust
    them or replace the pipeline with your own model.

    Returns
    -------
    sklearn.pipeline.Pipeline
        A scikit‑learn pipeline ready for training and inference.
    """
    vectoriser = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
    classifier = LogisticRegression(max_iter=1000)
    return Pipeline([
        ("vectoriser", vectoriser),
        ("classifier", classifier),
    ])


def train_model(model: Pipeline, X_train, y_train) -> Pipeline:
    """Train the model on the provided data and return the fitted model."""
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: Pipeline, X_test, y_test) -> Tuple[float, str]:
    """Evaluate the model and return accuracy and a classification report."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=3)
    return acc, report
