import pandas as pd
import pytest

from sentiment_classifier import SentimentClassifier


@pytest.fixture
def toy_dataset():
    positive = [
        "Profits soared this quarter beating all expectations.",
        "The company posted record revenue and strong growth.",
        "Shares rallied after outstanding earnings results.",
        "Analysts praised the impressive quarterly performance.",
        "Sales grew significantly boosting investor confidence.",
        "The firm reported excellent profit margins this year.",
    ] * 5
    negative = [
        "Profits collapsed this quarter missing all expectations.",
        "The company posted a massive loss and steep decline.",
        "Shares plunged after disastrous earnings results.",
        "Analysts criticized the terrible quarterly performance.",
        "Sales shrank significantly hurting investor confidence.",
        "The firm reported huge losses this year.",
    ] * 5
    neutral = [
        "The company will report earnings next Tuesday.",
        "The annual shareholder meeting is scheduled for March.",
        "The firm operates in the technology sector.",
        "The board appointed a new chief financial officer.",
        "The conference call is planned for next week.",
        "The company is headquartered in New York.",
    ] * 5
    texts = positive + negative + neutral
    labels = ["positive"] * len(positive) + ["negative"] * len(negative) + ["neutral"] * len(neutral)
    return pd.DataFrame({"text": texts, "label": labels})


def test_train_runs_and_returns_metrics(toy_dataset):
    clf = SentimentClassifier()
    metrics = clf.train(toy_dataset)
    assert 0.0 <= metrics["model_accuracy"] <= 1.0
    assert "baseline_accuracy" in metrics
    assert metrics["n_train"] + metrics["n_test"] == len(toy_dataset)


def test_predicts_clear_positive_example_correctly(toy_dataset):
    clf = SentimentClassifier()
    clf.train(toy_dataset)
    result = clf.predict(["Profits soared and revenue grew, beating every expectation."])[0]
    assert result["label"] == "positive"
    assert 0.0 <= result["confidence"] <= 1.0


def test_model_beats_or_matches_majority_baseline_on_easy_toy_data(toy_dataset):
    clf = SentimentClassifier()
    metrics = clf.train(toy_dataset)
    # On this trivially separable toy dataset the model should be at least
    # as good as always predicting the majority class.
    assert metrics["model_accuracy"] >= metrics["baseline_accuracy"]
