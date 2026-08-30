"""
Module 4: SentimentClassifier

TF-IDF + Logistic Regression sentiment classifier trained on Financial
PhraseBank. Reports accuracy/F1 against a majority-class baseline.
"""
from __future__ import annotations

import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import config

logger = logging.getLogger(__name__)

MODEL_PATH = config.MODELS_DIR / "sentiment_pipeline.joblib"


class SentimentClassifier:
    LABELS = ["negative", "neutral", "positive"]

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.pipeline: Pipeline | None = None
        self.baseline: DummyClassifier | None = None
        self.metrics_: dict | None = None

    def build_pipeline(self) -> Pipeline:
        return Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words="english",
                min_df=2,
            )),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=self.random_state,
            )),
        ])

    def train(self, df: pd.DataFrame, test_size: float = 0.2) -> dict:
        """df must have columns 'text' and 'label'. Returns metrics dict."""
        # .tolist() rather than .values: pandas' pyarrow-backed string arrays
        # (default since pandas 3.0) aren't indexable the way sklearn's
        # train_test_split expects, and raise inside _safe_indexing.
        texts = df["text"].astype(str).tolist()
        labels = df["label"].astype(str).tolist()
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels,
            test_size=test_size, random_state=self.random_state, stratify=labels,
        )

        self.pipeline = self.build_pipeline()
        self.pipeline.fit(X_train, y_train)
        preds = self.pipeline.predict(X_test)

        self.baseline = DummyClassifier(strategy="most_frequent", random_state=self.random_state)
        self.baseline.fit(X_train, y_train)
        baseline_preds = self.baseline.predict(X_test)

        metrics = {
            "n_train": len(X_train),
            "n_test": len(X_test),
            "model_accuracy": accuracy_score(y_test, preds),
            "model_f1_macro": f1_score(y_test, preds, average="macro"),
            "baseline_accuracy": accuracy_score(y_test, baseline_preds),
            "baseline_f1_macro": f1_score(y_test, baseline_preds, average="macro"),
            # scikit-learn's own inline type hint claims zero_division: str,
            # but its actual accepted values (and default) include the int 0 -
            # confirmed via inspect.signature(classification_report) at
            # runtime. This is an upstream stub inaccuracy, not a real bug.
            "classification_report": classification_report(y_test, preds, zero_division=0),  # type: ignore[arg-type]
            "label_distribution": df["label"].value_counts().to_dict(),
        }
        self.metrics_ = metrics
        return metrics

    def predict(self, texts: list[str]) -> list[dict]:
        """Return list of {text, label, confidence} for each input text."""
        if self.pipeline is None:
            self.load()
        assert self.pipeline is not None  # self.load() always sets it or raises
        probs = self.pipeline.predict_proba(texts)
        labels = self.pipeline.classes_
        results = []
        for text, p in zip(texts, probs):
            best_idx = int(np.argmax(p))
            results.append({
                "text": text,
                "label": labels[best_idx],
                "confidence": float(p[best_idx]),
                "all_probs": dict(zip(labels, p.tolist())),
            })
        return results

    def save(self, path=MODEL_PATH):
        joblib.dump(self.pipeline, path)
        logger.info("Saved sentiment pipeline to %s", path)

    def load(self, path=MODEL_PATH):
        self.pipeline = joblib.load(path)
        logger.info("Loaded sentiment pipeline from %s", path)
        return self


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from news_ingestion import NewsIngestion

    ingest = NewsIngestion()
    df = ingest.load_phrasebank()
    print(f"Training on {len(df)} PhraseBank rows (source={ingest.phrasebank_source_used})")

    clf = SentimentClassifier()
    metrics = clf.train(df)
    print(f"Model accuracy: {metrics['model_accuracy']:.4f} (baseline: {metrics['baseline_accuracy']:.4f})")
    print(f"Model F1-macro: {metrics['model_f1_macro']:.4f} (baseline: {metrics['baseline_f1_macro']:.4f})")
    print(metrics["classification_report"])

    clf.save()

    examples = [
        "The company reported record profits and strong revenue growth this quarter.",
        "The firm announced massive layoffs after posting a huge quarterly loss.",
    ]
    for r in clf.predict(examples):
        print(r["label"], r["confidence"], "-", r["text"][:60])
