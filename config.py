"""
Shared configuration for the Finance ML/RAG project.

All secrets (API keys) are read from environment variables / a local .env file
via python-dotenv. Nothing here is hardcoded. Copy `.env.example` to `.env`
and fill in real values before running anything that needs Groq.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (this file's directory) if present.
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Tickers used across the whole project (stock pipeline, forecasting,
# RAG corpus filtering). Five liquid, well-covered names per the build spec.
# ---------------------------------------------------------------------------
TICKERS: list[str] = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"
MODELS_DIR = DATA_DIR / "models"

for _d in (DATA_DIR, RAW_DIR, PROCESSED_DIR, CACHE_DIR, FAISS_INDEX_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Stock data pipeline
# ---------------------------------------------------------------------------
STOCK_HISTORY_PERIOD = "5y"   # how much history to pull per ticker
STOCK_DATA_SOURCE_PRIMARY = "yfinance"
STOCK_DATA_SOURCE_FALLBACK = "stooq"

# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------
ARIMA_ORDER = (5, 1, 0)     # default (p, d, q); ForecastEngine can override
WALK_FORWARD_MIN_TRAIN = 250   # minimum training window (trading days) before first prediction
WALK_FORWARD_HORIZON = 1       # 1-step-ahead forecasting

# ---------------------------------------------------------------------------
# Sentiment classifier / Financial PhraseBank
# ---------------------------------------------------------------------------
# The build spec names the Kaggle mirror of Financial PhraseBank
# (ankurzing/sentiment-analysis-for-financial-news). That requires a Kaggle
# API token (kaggle.json) which this environment does not have configured.
# The dataset is the *same* Malo et al. (2014) Financial PhraseBank data,
# also published on Hugging Face Datasets with no auth required:
#   https://huggingface.co/datasets/takala/financial_phrasebank
# NewsIngestion tries Kaggle first (if credentials are present) and falls
# back to the Hugging Face mirror. Whichever path actually ran is logged
# and recorded in the README, per the build spec's honesty requirement.
PHRASEBANK_HF_DATASET = "takala/financial_phrasebank"
PHRASEBANK_HF_CONFIG = "sentences_50agree"  # agreement-level subset used (see README)
PHRASEBANK_KAGGLE_DATASET = "ankurzing/sentiment-analysis-for-financial-news"
PHRASEBANK_LOCAL_CSV = RAW_DIR / "financial_phrasebank.csv"

# ---------------------------------------------------------------------------
# FNSPID / RAG corpus
# ---------------------------------------------------------------------------
FNSPID_HF_DATASET = "Zihan1004/FNSPID"
FNSPID_DATE_START = "2022-01-01"   # bounded recent window, see README for actual filter used
FNSPID_DATE_END = "2023-12-31"
FNSPID_MAX_ARTICLES = 1000          # split evenly across the 5 tickers (200 each); hard cap so a laptop build stays tractable
FNSPID_LOCAL_CACHE = RAW_DIR / "fnspid_filtered.jsonl"

# FNSPID tags Alphabet's news under the older single-class symbol "GOOG"
# rather than "GOOGL" (the class used everywhere else in this project, incl.
# yfinance). Confirmed empirically: a targeted scan for "GOOGL" in FNSPID's
# alphabetically-ordered ticker block returned 0 matches across ~36k scanned
# rows in that block, while "GOOG" matched immediately. This alias only
# affects FNSPID filtering - the stock/forecast pipeline still uses "GOOGL".
FNSPID_TICKER_ALIASES: dict[str, str] = {"GOOGL": "GOOG"}

# `nasdaq_exteral_data.csv` (~23GB) turns out to be laid out as large,
# roughly-alphabetically-ordered contiguous blocks per ticker - not shuffled.
# A single sequential scan from byte 0 only ever reaches whichever ticker's
# block happens to sort earliest ("AAPL", ~88MB in) within any reasonable
# time budget; tickers later in the alphabet (M, T, ...) are many GB away.
# These byte offsets were located by manual binary-search probing (small
# HTTP Range requests at candidate offsets, checking the Stock_symbol column
# until landing just before each ticker's block - see README for the method)
# and let NewsIngestion jump straight to each ticker's block instead of
# scanning the whole file sequentially per ticker. Falls back to a full
# sequential scan from 0 for any ticker not listed here.
FNSPID_TICKER_START_OFFSETS: dict[str, int] = {
    "AAPL": 0,
    "AMZN": 1_400_000_000,
    "GOOGL": 8_150_000_000,   # scanned as its FNSPID alias "GOOG"
    "MSFT": 10_450_000_000,
    "TSLA": 14_950_000_000,
}

# ---------------------------------------------------------------------------
# RAG pipeline
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800          # characters
CHUNK_OVERLAP = 150
RETRIEVAL_TOP_K = 4

# Groq - LLM backend for generation. See README for how to get a free API
# key. We deliberately never reference OpenAI/ChatOpenAI anywhere in this
# project - the LLM backend is Groq, full stop.
#
# Model note: the build spec's default choice (Llama 3.1/3.3 on Groq) is no
# longer available - confirmed live against this project's Groq account:
# `llama-3.1-8b-instant` 404s ("does not exist or you do not have access"),
# `llama-3.1-70b-versatile` and `llama3-70b-8192` are both explicitly
# "decommissioned", and no `llama-*` model appears in the account's
# `client.models.list()` at all. Groq's current catalog on this account is
# Qwen, Groq's own "compound" agents, and OpenAI's open-weight gpt-oss models
# (Apache-licensed weights served over Groq's infra, not the OpenAI API -
# but the name collision is exactly the kind of thing the "no OpenAI"
# naming rule exists to avoid, so gpt-oss was skipped even though it would
# have worked). Qwen is the closest available fit: open-weight, Groq-hosted,
# no naming ambiguity. Every reference to the backend in this project says
# "Groq" - not "Llama" - accordingly.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b")
# qwen/qwen3.6-27b is a reasoning model that emits a <think>...</think> trace
# before its answer by default. reasoning_effort="none" turns that off so
# `chat.completions.create()` returns a plain grounded answer directly.
GROQ_REASONING_EFFORT = "none"

# Optional, only used if the builder wants a NewsAPI-based supplementary
# fetch in the future. Not required for the core pipeline.
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
