import pandas as pd

from news_ingestion import NewsIngestion


def test_normalize_phrasebank_maps_labels_and_dedupes():
    raw = pd.DataFrame({
        "text": ["Profits rose sharply.", "Profits rose sharply.", "Losses widened.", "  "],
        "label": ["Positive", "positive", "NEGATIVE", "neutral"],
    })
    out = NewsIngestion._normalize_phrasebank(raw)
    assert set(out.columns) == {"text", "label", "label_id"}
    assert len(out) == 2  # dedup + blank text dropped
    assert out.loc[out["text"] == "Profits rose sharply.", "label_id"].iloc[0] == 2
    assert out.loc[out["text"] == "Losses widened.", "label_id"].iloc[0] == 0


def test_normalize_phrasebank_drops_unknown_labels():
    raw = pd.DataFrame({"text": ["A sentence."], "label": ["mixed"]})
    out = NewsIngestion._normalize_phrasebank(raw)
    assert out.empty


def test_normalize_fnspid_record_handles_alternate_column_names():
    record = {"Stock_symbol": "aapl", "Article_title": "Apple beats estimates", "Article": "body text", "Date": "2023-01-05"}
    row = NewsIngestion._normalize_fnspid_record(record)
    assert row is not None
    assert row["ticker"] == "AAPL"
    assert row["headline"] == "Apple beats estimates"
    assert row["date"] == "2023-01-05"


def test_normalize_fnspid_record_returns_none_for_malformed():
    assert NewsIngestion._normalize_fnspid_record({"foo": "bar"}) is None
    assert NewsIngestion._normalize_fnspid_record({"ticker": "AAPL"}) is None  # no headline/body
