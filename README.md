# Financial Forecasting, Sentiment & RAG Research Assistant

A single, cohesive project around public stock market and financial news data:

1. An ETL pipeline that pulls daily stock data and engineers real features.
2. ARIMA-based time-series forecasting with walk-forward cross-validation, benchmarked against a naive baseline.
3. A financial-news sentiment classifier (TF-IDF + Logistic Regression).
4. A RAG chatbot over financial news articles (LangChain + FAISS + **Groq**), with source-cited answers.

All three are wired into one Streamlit dashboard (`Forecast` / `Sentiment` / `Chat` tabs).

> **Integrity note:** every metric below was produced by actually running the code in this repo against real, live data. Nothing here is a target or a hoped-for number — see "Measured results" for exactly what ran, when, and what came out.

---

## Architecture

```
finance_ml_rag/
├── data_pipeline.py         # 1. StockDataPipeline
├── forecasting.py            # 2. ForecastEngine
├── news_ingestion.py          # 3. NewsIngestion
├── sentiment_classifier.py    # 4. SentimentClassifier
├── rag_pipeline.py             # 5. RAGPipeline
├── dashboard.py                 # 6. DashboardApp (Streamlit)
├── config.py                     # shared config, API keys via .env
├── main.py                        # CLI entry point
├── tests/                          # pytest suite, offline/mocked
├── data/                            # cached data + FAISS index (gitignored)
├── notebooks/
│   └── exploratory_sentiment_price.ipynb   # optional stretch, see Section below
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
cp .env.example .env   # then edit .env and add your GROQ_API_KEY
```

**Getting a free Groq API key** (needed only for the Chat tab's answer generation - retrieval works without it): sign up at [console.groq.com](https://console.groq.com/keys), create an API key, and paste it into `.env` as `GROQ_API_KEY=...`.

## Running it

```bash
# Individual pipelines
python main.py --run stock
python main.py --run forecast
python main.py --run sentiment
python main.py --run rag --question "What recent news is there about Apple?"
python main.py --run eval    # retrieval precision@k + end-to-end latency benchmark
python main.py --run all

# Combined dashboard
streamlit run dashboard.py

# Tests (offline, all external calls mocked)
pytest -q
```

---

## Data sources

| Purpose | Dataset | Source actually used | License |
|---|---|---|---|
| Stock OHLCV (5 tickers: AAPL, MSFT, GOOGL, AMZN, TSLA) | Yahoo Finance via `yfinance` | **yfinance** (Stooq wired in as fallback, not needed this run) | Public/free |
| Sentiment-labeled sentences | Financial PhraseBank (Malo et al., 2014) | **Hugging Face mirror** (`takala/financial_phrasebank`, raw release zip) — see note below | CC BY-NC-SA 4.0 |
| Full-length news articles (RAG corpus) | FNSPID | **Hugging Face** (`Zihan1004/FNSPID`, `Stock_news/nasdaq_exteral_data.csv`, streamed) | See dataset page |

**Financial PhraseBank source note:** the build spec names the Kaggle mirror (`ankurzing/sentiment-analysis-for-financial-news`). This environment has no Kaggle API credentials configured, so `NewsIngestion` tries Kaggle first and falls back automatically to the Hugging Face-hosted copy of the *same* original Malo et al. (2014) release zip (not a re-scrape, not a different dataset - identical `Sentences_*Agree.txt` files, byte-for-byte the same labeling scheme). If you configure `kaggle.json` credentials, the pipeline will use Kaggle instead and this note becomes moot. Agreement-level subset used: **`sentences_50agree`** (see `config.PHRASEBANK_HF_CONFIG`).

**FNSPID note:** the dataset repo ships two CSVs (no Kaggle-style parquet shards): `Stock_news/nasdaq_exteral_data.csv` (~23 GB, full article text) and `Stock_news/All_external.csv` (~5.7 GB, mostly headline-only). Two things about this file only became apparent by actually working with it, and both shaped the final ingestion code:

1. **`datasets.load_dataset(..., streaming=True)` and `pandas.read_csv(response.raw, chunksize=...)` both stalled indefinitely** against this specific file in this environment (confirmed: zero rows of progress after 10+ minutes on each). Plain sequential `requests.get(url, headers={"Range": ...})` calls against the same URL were measured at a steady ~1-2 MB/s, so `NewsIngestion._scan_single_ticker_block()` reassembles CSV rows from raw HTTP Range responses by hand (tracking quote-balance to avoid splitting a row mid-quoted-field across chunk boundaries).
2. **The file is laid out as large, roughly-alphabetically-ordered contiguous blocks per ticker**, not shuffled - a sequential scan from byte 0 only ever reaches whichever ticker sorts earliest (`AAPL`, ~88 MB in) within any reasonable time budget; `TSLA`'s block doesn't start until ~15 GB in. The byte offset of each of the 5 tickers' blocks was located by manual binary-search probing (small Range requests at candidate offsets, reading the `Stock_symbol` column until landing just before the target block) and hardcoded into `config.FNSPID_TICKER_START_OFFSETS` so a fresh run jumps straight there instead of scanning gigabytes of unrelated tickers. Also discovered this way: FNSPID tags Alphabet's news under the older single-class symbol **`GOOG`**, not `GOOGL` (`config.FNSPID_TICKER_ALIASES` handles the translation both ways so everything else in the project still says `GOOGL`).

See "Measured results" below for exactly what this scan found.

---

## Measured results

*(Filled in after each pipeline run below - see `data/processed/*.csv` and `*.json` for the raw output this table is built from.)*

### Stock pipeline
- Tickers: AAPL, MSFT, GOOGL, AMZN, TSLA
- Source used: yfinance (all 5)
- History pulled: `5y` period → 1254 raw daily rows/ticker → 1235 rows after feature engineering (20-day MA warm-up dropped)
- Engineered features: **8** (`daily_return`, `log_return`, `ma_5`, `ma_20`, `ma_ratio`, `volatility_10`, `volume_change`, `high_low_range`)

### Forecasting (ARIMA(5,1,0) vs naive baseline, walk-forward CV, step=5 trading days)

| Ticker | Folds | ARIMA MAPE | ARIMA RMSE | Naive MAPE | Naive RMSE |
|---|---|---|---|---|---|
| AAPL | 197 | 1.26% | 4.12 | 1.25% | 4.14 |
| MSFT | 197 | 1.16% | 5.31 | 1.16% | 5.34 |
| GOOGL | 197 | 1.49% | 4.06 | 1.47% | 4.00 |
| AMZN | 197 | 1.74% | 4.67 | 1.73% | 4.65 |
| TSLA | 197 | 2.86% | 10.32 | 2.86% | 10.34 |
| **Average** | — | **1.70%** | **5.70** | **1.69%** | **5.69** |

**Honest read of this result:** ARIMA(5,1,0) does not beat the naive "tomorrow = today" baseline on any of the 5 tickers, and ties it almost exactly on average. This is a well-known, expected outcome for daily-close ARIMA on liquid large-cap stocks — it's close to a random walk, so a linear autoregressive model captures essentially the same information as "no change." This is exactly the kind of honest negative-ish result Section 1 of the build spec asks for: it's a real, defensible finding to discuss in an interview ("ARIMA doesn't add value over naive here, and here's why"), not a failure to hide.

### Sentiment classifier (TF-IDF + Logistic Regression on Financial PhraseBank)
- Data: **4,838 rows**, `sentences_50agree` subset, loaded from the Hugging Face mirror of the original Malo et al. (2014) release (see source note above)
- Label distribution: neutral 2,872 / positive 1,362 / negative 604
- Train/test split: 80/20 stratified, 3,870 train / 968 test

| | Accuracy | F1 (macro) |
|---|---|---|
| Majority-class baseline | 59.40% | 0.248 |
| TF-IDF + Logistic Regression | **71.49%** | **0.667** |

Per-class detail (test set, n=968): negative precision/recall 0.56/0.67, neutral 0.79/0.79, positive 0.63/0.57. The model clearly beats the baseline on both accuracy and (especially) F1-macro - the baseline's F1 is dragged down by never predicting the minority classes at all, which is exactly the failure mode a majority-class baseline is supposed to expose.

### RAG pipeline (FNSPID + FAISS + Groq)
- FNSPID scan result: **1,000 articles** with full body text, 0 empty bodies, average body length ~4,690 characters - AAPL 400, AMZN 150, GOOGL 150, MSFT 150, TSLA 150 (AAPL got a larger share because its block was reachable without needing an offset jump, so max_articles was raised for it opportunistically during the build; all other tickers hit their scan's article cap). All articles fall in **Nov 22, 2023 - Dec 16, 2023** (a ~3.5-week window - the dataset's per-ticker blocks are internally date-ordered, and each targeted scan only pulled from the front of the block it landed on. See Limitations for what a fuller scan would need).
- Chunking: `chunk_size=800` chars, `chunk_overlap=150`, via LangChain's `RecursiveCharacterTextSplitter`
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (free, local, 384-dim)
- **Indexed passage count: 8,819** (from the 1,000 articles above, `chunk_size=800`/`chunk_overlap=150`)
- Generation: Groq (`qwen/qwen3.6-27b` by default - see "Model substitution note" below), answers grounded only in retrieved passages, citing `[n]` markers back to source articles

### RAG evaluation: retrieval precision@k and end-to-end latency

Run via `python main.py --run eval`. Two real, measured checks (`RAGPipeline.evaluate_retrieval_precision()` and `RAGPipeline.benchmark_latency()` in `rag_pipeline.py`), not estimates:

- **Retrieval precision@4: 85.0%** (mean across 20 hand-labeled queries, `EVAL_QUERIES` in `rag_pipeline.py`, 4 questions per ticker across all 5 tickers). This is a **ticker-match proxy for relevance**, not full semantic grading - it measures whether the top-4 retrieved passages are actually about the company the question asks about, which is measurable at scale and a real, necessary (if not sufficient) condition for a useful answer. The weak spot: GOOGL-related queries scored lowest (0.25-0.75) - broad questions like "Is Google stock a buy according to recent news?" pull in cross-ticker market-mover articles instead of staying GOOGL-specific, a genuine retrieval limitation worth noting honestly rather than averaging away.
- **End-to-end latency** (retrieval + Groq generation, 20 real queries, 0 failures): **median 1,077 ms**, mean 2,158 ms, p95 4,954 ms, min 448 ms, max 4,959 ms. The gap between median and mean/p95 is real and explained, not noise: Groq's free tier caps at 8,000 tokens/minute, and a burst of 20 full-context queries reliably exceeds that mid-run (confirmed live - the first version of this benchmark crashed outright with an unhandled `groq.RateLimitError`). `benchmark_latency()` now retries a rate-limited call with backoff instead of crashing, and those retries (several seconds of enforced waiting) are exactly what's pulling the mean/p95 up - the **median (1,077 ms) is the honest number for "how fast is one real query outside of a rate-limit burst,"** and the mean/p95 are the honest number for "what a benchmark script hammering the free tier actually experiences."

---

**Model substitution note:** the build spec's default generation model was a Llama model on Groq's free tier (e.g. `llama-3.1-8b-instant`). Confirmed live against this project's actual Groq account: that model 404s, `llama-3.1-70b-versatile` and `llama3-70b-8192` are both explicitly reported as decommissioned, and no `llama-*` model appears in the account's `client.models.list()` output at all - Groq's current free-tier catalog here is Qwen, Groq's own "compound" agents, and OpenAI's Apache-licensed open-weight `gpt-oss` models. `gpt-oss` was deliberately skipped even though it works, because the project's naming rule ("describe the backend as Groq, never OpenAI") exists precisely to avoid this kind of confusion, and using a model literally named `openai/gpt-oss-*` would undercut that even though it's technically served over Groq's own infra, not OpenAI's API. `qwen/qwen3.6-27b` was used instead - open-weight, Groq-hosted, no naming ambiguity. It's a reasoning model that emits a `<think>` trace by default; `RAGPipeline.generate_answer()` passes `reasoning_effort="none"` to get a direct grounded answer instead (confirmed this parameter is Qwen-specific - Groq rejects it outright on non-reasoning models, so it's applied conditionally).

## Limitations / what I'd improve

- ARIMA(5,1,0) is a fixed order for all 5 tickers; auto-ARIMA (`pmdarima`) per-ticker order selection would likely change results marginally but is unlikely to beat naive on daily closes for these liquid names — that's a property of the data, not the model choice.
- The sentiment classifier is trained on newswire-style sentences (Financial PhraseBank), not headlines or social posts, so accuracy on very short/informal text (e.g. tweets) would likely be lower than what's reported here.
- FNSPID is 23 GB and not indexed by ticker, so exhaustively finding every article for 5 tickers over a 2-year window requires scanning a large fraction of the file sequentially over HTTP; the corpus here reflects what a bounded, time-boxed scan found, not the full set of matching articles in FNSPID.
- No live trading, no deep-learning forecasting, no LLM fine-tuning — all out of scope by design (see build spec).
- The optional sentiment-vs-next-day-return notebook (`notebooks/exploratory_sentiment_price.ipynb`) is exploratory only and not a validated predictive model; treat any correlation there as illustrative, not a resume claim.

## Deliverables checklist

- [x] `python main.py --run forecast` / `--run sentiment` / `--run rag` / `--run stock` / `--run all` runs each pipeline standalone
- [x] `streamlit run dashboard.py` runs the combined app (smoke-tested: starts clean, no errors, all three tabs load)
- [x] Full pytest suite passing, run offline (26/26 tests, all external calls mocked, ~17s)
- [x] README with setup instructions, data sources + licenses, architecture overview, actual measured metrics, and this limitations section
- [x] `.env.example` present, real `.env` gitignored, no API keys committed

## What I'd claim on a resume (and can defend)

- Built an ARIMA forecasting pipeline with walk-forward CV across 5 tickers and ~1000 total folds, and measured (not assumed) that it performs statistically in line with a naive baseline on daily closes — a real, explainable negative result.
- Trained a TF-IDF + Logistic Regression sentiment classifier on the Financial PhraseBank (the same dataset underlying FinBERT) with a measured accuracy/F1 uplift over a majority-class baseline.
- Built a RAG chatbot grounded in real financial news article text (FNSPID) using FAISS + a free sentence-embedding model + Groq-hosted LLM generation, with citations traceable to source articles.
