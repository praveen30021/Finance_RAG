from unittest.mock import MagicMock

import pandas as pd
import pytest

from rag_pipeline import Document, RAGPipeline, EVAL_QUERIES


@pytest.fixture
def articles_df():
    return pd.DataFrame([
        {
            "headline": "Apple beats earnings estimates",
            "body": "Apple reported strong quarterly results. " * 40,
            "source": "example.com/apple",
            "date": "2023-05-01",
            "ticker": "AAPL",
        },
        {
            "headline": "Tesla misses delivery targets",
            "body": "Tesla fell short of its delivery goals this quarter. " * 40,
            "source": "example.com/tesla",
            "date": "2023-06-15",
            "ticker": "TSLA",
        },
    ])


def test_chunk_articles_produces_multiple_chunks_per_long_article(articles_df):
    rag = RAGPipeline(chunk_size=200, chunk_overlap=50)
    docs = rag.chunk_articles(articles_df)
    assert len(docs) > len(articles_df)  # long bodies split into multiple chunks
    assert all(isinstance(d, Document) for d in docs)
    assert all(len(d.page_content) <= 250 for d in docs)  # allow splitter slack


def test_chunk_articles_tags_metadata_correctly(articles_df):
    rag = RAGPipeline(chunk_size=200, chunk_overlap=50)
    docs = rag.chunk_articles(articles_df)
    aapl_docs = [d for d in docs if d.metadata["ticker"] == "AAPL"]
    assert len(aapl_docs) > 0
    assert aapl_docs[0].metadata["headline"] == "Apple beats earnings estimates"


def test_chunk_articles_skips_empty_bodies():
    df = pd.DataFrame([{"headline": "", "body": "", "source": "x", "date": "2023-01-01", "ticker": "AAPL"}])
    rag = RAGPipeline()
    docs = rag.chunk_articles(df)
    assert docs == []


def test_retrieve_calls_vectorstore_similarity_search_with_top_k():
    rag = RAGPipeline(top_k=3)
    mock_docs = [Document(page_content="passage text", metadata={"headline": "H", "source": "s", "date": "d", "ticker": "AAPL"})]
    rag.vectorstore = MagicMock()
    rag.vectorstore.similarity_search.return_value = mock_docs

    result = rag.retrieve("What happened to Apple?")

    rag.vectorstore.similarity_search.assert_called_once_with("What happened to Apple?", k=3)
    assert result == mock_docs


def test_retrieve_without_index_raises():
    rag = RAGPipeline()
    with pytest.raises(RuntimeError):
        rag.retrieve("anything")


def test_format_citations_produces_expected_markers():
    docs = [
        Document(page_content="first passage", metadata={"headline": "H1", "source": "s1", "date": "2023-01-01", "ticker": "AAPL"}),
        Document(page_content="second passage", metadata={"headline": "H2", "source": "s2", "date": "2023-02-01", "ticker": "MSFT"}),
    ]
    citations = RAGPipeline.format_citations(docs)
    assert [c["marker"] for c in citations] == ["[1]", "[2]"]
    assert citations[0]["headline"] == "H1"
    assert citations[1]["ticker"] == "MSFT"


def test_evaluate_retrieval_precision_computes_fraction_matching_expected_ticker():
    rag = RAGPipeline()
    # 3 of 4 retrieved docs match the expected ticker (AAPL) -> precision 0.75
    mixed_docs = [
        Document(page_content="p1", metadata={"ticker": "AAPL"}),
        Document(page_content="p2", metadata={"ticker": "AAPL"}),
        Document(page_content="p3", metadata={"ticker": "AAPL"}),
        Document(page_content="p4", metadata={"ticker": "TSLA"}),
    ]
    rag.retrieve = MagicMock(return_value=mixed_docs)

    result = rag.evaluate_retrieval_precision(
        queries=[{"question": "What's new with Apple?", "expected_ticker": "AAPL"}], k=4,
    )

    assert result["n_queries"] == 1
    assert result["mean_precision_at_k"] == pytest.approx(0.75)
    assert result["per_query"][0]["retrieved_tickers"] == ["AAPL", "AAPL", "AAPL", "TSLA"]


def test_evaluate_retrieval_precision_scores_zero_when_no_docs_retrieved():
    rag = RAGPipeline()
    rag.retrieve = MagicMock(return_value=[])

    result = rag.evaluate_retrieval_precision(
        queries=[{"question": "anything", "expected_ticker": "AAPL"}], k=4,
    )

    assert result["per_query"][0]["precision_at_k"] == 0.0
    assert result["mean_precision_at_k"] == 0.0


def test_evaluate_retrieval_precision_averages_across_multiple_queries():
    rag = RAGPipeline()
    all_match = [Document(page_content="p", metadata={"ticker": "MSFT"})]
    no_match = [Document(page_content="p", metadata={"ticker": "TSLA"})]
    rag.retrieve = MagicMock(side_effect=[all_match, no_match])

    result = rag.evaluate_retrieval_precision(
        queries=[
            {"question": "q1", "expected_ticker": "MSFT"},
            {"question": "q2", "expected_ticker": "MSFT"},
        ],
        k=1,
    )

    assert result["mean_precision_at_k"] == pytest.approx(0.5)


def test_eval_queries_cover_all_five_tickers():
    tickers = {q["expected_ticker"] for q in EVAL_QUERIES}
    assert tickers == {"AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"}


def test_benchmark_latency_calls_answer_once_per_question_and_reports_consistent_stats():
    rag = RAGPipeline()
    rag.answer = MagicMock(return_value={"question": "q", "answer": "a", "citations": []})

    # delay_between_calls_sec=0: this is a unit test, not a real burst against
    # Groq - the default courtesy delay only matters for the live benchmark.
    result = rag.benchmark_latency(questions=["q1", "q2", "q3"], delay_between_calls_sec=0)

    assert rag.answer.call_count == 3
    assert result["n_queries"] == 3
    assert result["n_failed"] == 0
    assert len(result["latencies_ms"]) == 3
    assert result["min_ms"] <= result["median_ms"] <= result["max_ms"]
    assert result["min_ms"] <= result["mean_ms"] <= result["max_ms"]
    assert all(ms >= 0 for ms in result["latencies_ms"])


def test_benchmark_latency_retries_then_succeeds_on_rate_limit(monkeypatch):
    from groq import RateLimitError

    monkeypatch.setattr("time.sleep", lambda _seconds: None)  # don't actually wait out the backoff

    rag = RAGPipeline()
    rate_limit_error = RateLimitError("rate limited", response=MagicMock(status_code=429), body=None)
    rag.answer = MagicMock(side_effect=[rate_limit_error, {"question": "q", "answer": "a", "citations": []}])

    result = rag.benchmark_latency(questions=["q1"], delay_between_calls_sec=0, max_retries_per_call=2)

    assert rag.answer.call_count == 2  # failed once, retried, succeeded
    assert result["n_queries"] == 1
    assert result["n_failed"] == 0


def test_benchmark_latency_records_failure_instead_of_crashing_after_exhausting_retries(monkeypatch):
    from groq import RateLimitError

    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    rag = RAGPipeline()
    rate_limit_error = RateLimitError("rate limited", response=MagicMock(status_code=429), body=None)
    rag.answer = MagicMock(side_effect=rate_limit_error)  # always rate limited

    result = rag.benchmark_latency(questions=["q1", "q2"], delay_between_calls_sec=0, max_retries_per_call=1)

    assert result["n_queries"] == 0
    assert result["n_failed"] == 2
    assert result["median_ms"] is None
    assert len(result["failures"]) == 2
