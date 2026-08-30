"""
CLI entry point to run each pipeline standalone (non-Streamlit).

Usage:
    python main.py --run stock
    python main.py --run forecast
    python main.py --run sentiment
    python main.py --run rag
    python main.py --run eval
    python main.py --run all
"""
from __future__ import annotations

import os

# Must run before numpy/scipy/statsmodels are imported anywhere. Repeated
# ARIMA re-fitting inside ForecastEngine's walk-forward loop spawns a fresh
# BLAS thread pool per fit; on Windows this reliably exhausts memory after a
# few hundred fits ("OpenBLAS error: Memory allocation still failed after 10
# retries") even though the arrays involved are tiny. Pinning every BLAS
# backend to 1 thread avoids the thread-pool churn entirely - each ARIMA fit
# is small enough that single-threaded BLAS costs nothing in wall time.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import logging

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("main")


def run_stock(force_refresh: bool = False):
    from data_pipeline import StockDataPipeline

    pipeline = StockDataPipeline()
    data = pipeline.run(force_refresh=force_refresh)
    print("\n=== Stock Data Pipeline ===")
    for ticker, df in data.items():
        print(f"{ticker}: {len(df)} feature rows, source={pipeline.source_used[ticker]}")
    print(f"Feature count (engineered): {pipeline.feature_count}")
    return data


def run_forecast(force_refresh: bool = False):
    from forecasting import ForecastEngine

    data = run_stock(force_refresh=force_refresh)
    engine = ForecastEngine()
    series_map = {t: df["Close"] for t, df in data.items()}
    summary = engine.evaluate_all(series_map)
    print("\n=== Forecast Engine (ARIMA vs Naive baseline, walk-forward CV) ===")
    print(summary.to_string())
    summary.to_csv(config.PROCESSED_DIR / "forecast_summary.csv")
    return summary


def run_sentiment(force_refresh: bool = False):
    from news_ingestion import NewsIngestion
    from sentiment_classifier import SentimentClassifier

    ingest = NewsIngestion()
    df = ingest.load_phrasebank(force_refresh=force_refresh)
    print(f"\n=== Sentiment Classifier ===")
    print(f"Financial PhraseBank: {len(df)} rows (source={ingest.phrasebank_source_used})")
    print(df["label"].value_counts().to_string())

    clf = SentimentClassifier()
    metrics = clf.train(df)
    clf.save()

    print(f"\nModel accuracy: {metrics['model_accuracy']:.4f} vs baseline: {metrics['baseline_accuracy']:.4f}")
    print(f"Model F1-macro: {metrics['model_f1_macro']:.4f} vs baseline: {metrics['baseline_f1_macro']:.4f}")
    print(metrics["classification_report"])

    with open(config.PROCESSED_DIR / "sentiment_metrics.json", "w") as f:
        json.dump({k: v for k, v in metrics.items() if k != "classification_report"}, f, indent=2, default=str)

    return metrics


def run_rag(force_refresh: bool = False, question: str | None = None):
    from news_ingestion import NewsIngestion
    from rag_pipeline import RAGPipeline

    ingest = NewsIngestion()
    articles = ingest.load_fnspid_filtered(force_refresh=force_refresh)
    print(f"\n=== RAG Pipeline ===")
    print(f"FNSPID filtered articles: {len(articles)} (source={ingest.fnspid_source_used})")

    if articles.empty:
        print("No articles matched the ticker/date filter - cannot build RAG index. "
              "See README for known FNSPID access limitations.")
        return None

    rag = RAGPipeline()
    rag.build_index(articles, force_rebuild=force_refresh)
    print(f"Indexed passage count: {rag.indexed_passage_count}")

    if question:
        result = rag.answer(question)
        print(f"\nQ: {question}")
        print(f"A: {result['answer']}")
        for c in result["citations"]:
            print(f"  {c['marker']} {c['headline']} ({c['date']})")
        return result
    return rag


def run_eval(force_refresh: bool = False):
    """Retrieval precision@k + end-to-end latency benchmark for the RAG pipeline.

    Requires a built index and a working GROQ_API_KEY (latency includes the
    real Groq generation call, not just retrieval).
    """
    rag = run_rag(force_refresh)
    if rag is None:
        print("Cannot run eval - no RAG index available.")
        return None

    print("\n=== RAG Retrieval Precision@k (ticker-match proxy, see EVAL_QUERIES docstring) ===")
    precision = rag.evaluate_retrieval_precision()
    print(f"Mean precision@{precision['k']} across {precision['n_queries']} hand-labeled queries: "
          f"{precision['mean_precision_at_k']:.4f}")
    for q in precision["per_query"]:
        print(f"  [{q['precision_at_k']:.2f}] {q['question']} -> {q['retrieved_tickers']}")

    print("\n=== RAG End-to-End Latency Benchmark (retrieval + Groq generation) ===")
    latency = rag.benchmark_latency()
    print(f"n={latency['n_queries']}  median={latency['median_ms']:.0f}ms  mean={latency['mean_ms']:.0f}ms  "
          f"p95={latency['p95_ms']:.0f}ms  min={latency['min_ms']:.0f}ms  max={latency['max_ms']:.0f}ms")

    with open(config.PROCESSED_DIR / "rag_eval_metrics.json", "w") as f:
        json.dump({"precision": precision, "latency": latency}, f, indent=2, default=str)

    return {"precision": precision, "latency": latency}


def main():
    parser = argparse.ArgumentParser(description="Finance ML/RAG project CLI")
    parser.add_argument(
        "--run", choices=["stock", "forecast", "sentiment", "rag", "eval", "all"],
        required=True, help="Which pipeline to run",
    )
    parser.add_argument("--force-refresh", action="store_true", help="Ignore local caches and refetch")
    parser.add_argument("--question", type=str, default=None, help="Question to ask the RAG pipeline (--run rag)")
    args = parser.parse_args()

    if args.run == "stock":
        run_stock(args.force_refresh)
    elif args.run == "forecast":
        run_forecast(args.force_refresh)
    elif args.run == "sentiment":
        run_sentiment(args.force_refresh)
    elif args.run == "rag":
        run_rag(args.force_refresh, question=args.question)
    elif args.run == "eval":
        run_eval(args.force_refresh)
    elif args.run == "all":
        run_forecast(args.force_refresh)
        run_sentiment(args.force_refresh)
        run_rag(args.force_refresh)


if __name__ == "__main__":
    main()
