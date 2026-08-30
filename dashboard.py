"""
Module 6: DashboardApp (Streamlit)

Three tabs: Forecast / Sentiment / Chat, sharing the pipelines built in the
other modules. Run with: streamlit run dashboard.py
"""
from __future__ import annotations

import os

# See main.py for why: repeated ARIMA re-fitting in ForecastEngine's
# walk-forward loop exhausts Windows OpenBLAS thread pools unless every
# BLAS backend is pinned to 1 thread before numpy/scipy are imported.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import pandas as pd
import streamlit as st

import config

st.set_page_config(page_title="Finance ML + RAG Research Assistant", layout="wide")


# ---------------------------------------------------------------------------
# Cached resource loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def get_stock_data():
    from data_pipeline import StockDataPipeline

    pipeline = StockDataPipeline()
    return pipeline.run(), pipeline.source_used


@st.cache_resource
def get_forecast_summary(_series_map: dict):
    from forecasting import ForecastEngine

    engine = ForecastEngine()
    summary = engine.evaluate_all(_series_map)
    return summary, engine._last_details


@st.cache_resource
def get_sentiment_classifier():
    from sentiment_classifier import SentimentClassifier, MODEL_PATH

    clf = SentimentClassifier()
    if MODEL_PATH.exists():
        clf.load()
    else:
        from news_ingestion import NewsIngestion

        ingest = NewsIngestion()
        df = ingest.load_phrasebank()
        clf.train(df)
        clf.save()
    return clf


@st.cache_resource
def get_rag_pipeline():
    from rag_pipeline import RAGPipeline
    from news_ingestion import NewsIngestion

    rag = RAGPipeline()
    index_ready = (config.FAISS_INDEX_DIR / "index.faiss").exists()
    if index_ready:
        rag.build_index(pd.DataFrame())  # will load existing index, articles arg unused
    else:
        ingest = NewsIngestion()
        articles = ingest.load_fnspid_filtered()
        if not articles.empty:
            rag.build_index(articles)
        else:
            return None
    return rag


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
def main():
    st.title("Financial Forecasting, Sentiment & RAG Research Assistant")
    st.caption(
        "ARIMA forecasting + TF-IDF/LogReg sentiment + RAG chatbot over financial news "
        "(LangChain + FAISS + Groq). All numbers shown are computed live from "
        "real public data - nothing is hardcoded."
    )

    tab_forecast, tab_sentiment, tab_chat = st.tabs(["📈 Forecast", "📰 Sentiment", "💬 Chat"])

    # ---------------- Forecast tab ----------------
    with tab_forecast:
        st.subheader("ARIMA Forecast vs Naive Baseline (walk-forward CV)")
        ticker = st.selectbox("Ticker", config.TICKERS, key="forecast_ticker")

        if st.button("Run forecast", key="run_forecast_btn"):
            with st.spinner("Fetching data and running walk-forward ARIMA..."):
                data, sources = get_stock_data()
                series_map = {t: df["Close"] for t, df in data.items()}
                summary, details = get_forecast_summary(series_map)

            st.write(f"Data source used for {ticker}: **{sources[ticker]}**")
            st.dataframe(summary)

            det = details[ticker]
            plot_df = pd.DataFrame({
                "date": det["dates"],
                "actual": det["actuals"],
                "arima_pred": det["arima_preds"],
                "naive_pred": det["naive_preds"],
            }).set_index("date")
            st.line_chart(plot_df)

            col1, col2 = st.columns(2)
            col1.metric(f"{ticker} ARIMA MAPE", f"{summary.loc[ticker, 'arima_mape']:.2f}%")
            col2.metric(f"{ticker} Naive MAPE", f"{summary.loc[ticker, 'naive_mape']:.2f}%")

    # ---------------- Sentiment tab ----------------
    with tab_sentiment:
        st.subheader("Financial Headline Sentiment (TF-IDF + Logistic Regression)")
        st.write("Trained on Financial PhraseBank (Malo et al., 2014).")

        text = st.text_area(
            "Enter a financial headline or sentence",
            value="The company reported record profits and strong revenue growth this quarter.",
        )
        if st.button("Predict sentiment", key="predict_sentiment_btn"):
            clf = get_sentiment_classifier()
            result = clf.predict([text])[0]
            st.write(f"**Predicted label:** {result['label']}  (confidence: {result['confidence']:.2%})")
            st.bar_chart(pd.Series(result["all_probs"]))

    # ---------------- Chat tab ----------------
    with tab_chat:
        st.subheader("RAG Chatbot over Financial News (FNSPID + FAISS + Groq)")
        if not config.GROQ_API_KEY:
            st.warning(
                "GROQ_API_KEY is not set. Add a free key from "
                "https://console.groq.com/keys to your .env file to enable answer generation. "
                "Retrieval will still work without it."
            )

        question = st.text_input("Ask a question about the indexed articles")
        if st.button("Ask", key="ask_rag_btn") and question:
            with st.spinner("Retrieving and generating..."):
                rag = get_rag_pipeline()
                if rag is None:
                    st.error("No RAG index available - the FNSPID filter returned no articles. See README.")
                else:
                    try:
                        result = rag.answer(question)
                        st.markdown(f"**Answer:** {result['answer']}")
                        st.markdown("**Sources:**")
                        for c in result["citations"]:
                            st.markdown(f"- {c['marker']} *{c['headline']}* ({c['ticker']}, {c['date']})")
                            st.caption(c["snippet"])
                    except RuntimeError as exc:
                        st.error(str(exc))


if __name__ == "__main__":
    main()
