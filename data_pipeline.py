"""
Module 1: StockDataPipeline

Fetches daily OHLCV data for the configured tickers (yfinance primary,
Stooq fallback), cleans it, validates it, and engineers a small set of
real features used downstream by ForecastEngine.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


class StockDataPipeline:
    """Fetch, clean, validate, and feature-engineer daily OHLCV data."""

    def __init__(
        self,
        tickers: list[str] | None = None,
        period: str = config.STOCK_HISTORY_PERIOD,
        cache_dir: Path = config.RAW_DIR,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
    ):
        self.tickers = tickers or list(config.TICKERS)
        self.period = period
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        # Track which source actually served each ticker (for honest README reporting)
        self.source_used: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Fetching
    # ------------------------------------------------------------------
    def _cache_path(self, ticker: str) -> Path:
        return self.cache_dir / f"{ticker}_ohlcv.csv"

    def fetch_ticker(self, ticker: str, force_refresh: bool = False) -> pd.DataFrame:
        """Fetch OHLCV for one ticker, yfinance first, Stooq fallback, disk-cached."""
        cache_path = self._cache_path(ticker)
        if cache_path.exists() and not force_refresh:
            df = pd.read_csv(cache_path, index_col=0)
            df.index = pd.to_datetime(df.index, utc=True, format="mixed")
            df.index.name = "Date"
            if not df.empty:
                logger.info("Loaded %s from cache (%d rows)", ticker, len(df))
                self.source_used.setdefault(ticker, "cache")
                return df

        df = self._fetch_yfinance(ticker)
        if df is not None and not df.empty:
            self.source_used[ticker] = "yfinance"
        else:
            logger.warning("yfinance failed for %s, falling back to Stooq", ticker)
            df = self._fetch_stooq(ticker)
            if df is not None and not df.empty:
                self.source_used[ticker] = "stooq"

        if df is None or df.empty:
            raise RuntimeError(f"Could not fetch OHLCV data for {ticker} from any source")

        df.to_csv(cache_path)
        return df

    def _fetch_yfinance(self, ticker: str) -> pd.DataFrame | None:
        import yfinance as yf

        for attempt in range(1, self.max_retries + 1):
            try:
                # yfinance ships no type stubs, so its return value is Unknown to
                # the type checker; the explicit annotation stops that
                # imprecision from leaking into every reassignment below.
                data: pd.DataFrame | None = yf.Ticker(ticker).history(period=self.period, auto_adjust=False)
                if data is None or data.empty:
                    raise ValueError("empty dataframe returned")
                data = data[[c for c in REQUIRED_COLUMNS if c in data.columns]]
                data.index.name = "Date"
                return data
            except Exception as exc:  # broad: yfinance raises many different error types
                logger.warning(
                    "yfinance attempt %d/%d for %s failed: %s",
                    attempt, self.max_retries, ticker, exc,
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff ** attempt)
        return None

    def _fetch_stooq(self, ticker: str) -> pd.DataFrame | None:
        """Fallback via pandas-datareader's Stooq reader (no API key needed)."""
        try:
            import pandas_datareader.data as web

            # Same reasoning as the yfinance branch above: pandas_datareader
            # has no type stubs either, so pin the declared type explicitly.
            data: pd.DataFrame = web.DataReader(ticker, "stooq")
            data = data.sort_index()  # stooq returns newest-first
            data = data[[c for c in REQUIRED_COLUMNS if c in data.columns]]
            data.index.name = "Date"
            return data
        except Exception as exc:
            logger.error("Stooq fallback failed for %s: %s", ticker, exc)
            return None

    def fetch_all(self, force_refresh: bool = False) -> dict[str, pd.DataFrame]:
        return {t: self.fetch_ticker(t, force_refresh=force_refresh) for t in self.tickers}

    # ------------------------------------------------------------------
    # Cleaning
    # ------------------------------------------------------------------
    @staticmethod
    def clean(df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows missing a Close price; forward-fill isolated gaps in other columns.

        Rule (documented): trading calendars already exclude weekends/holidays
        for yfinance/Stooq output, so "missing days" here means NaN cells
        within the fetched range (e.g. a halted-trading day). We forward-fill
        OHLC, including Close, on the assumption the price carried over from
        the prior session, and set Volume to 0 for those rows. Only rows that
        still have NaN after forward-fill (i.e. no prior data exists yet -
        the very start of a series) are dropped.
        """
        # Explicit annotation: boolean-array indexing (`out[~mask]`) is one of
        # DataFrame.__getitem__'s many overloads, and the stub resolution for
        # it is imprecise enough that leaving `out`'s type to be re-inferred
        # at each reassignment below cascades into false-positive errors on
        # every later pandas call. Pinning it once here stops that.
        out: pd.DataFrame = df.copy()
        out = out[~out.index.duplicated(keep="last")]
        out = out.sort_index()
        out[["Open", "High", "Low", "Close"]] = out[["Open", "High", "Low", "Close"]].ffill()
        out["Volume"] = out["Volume"].fillna(0)
        out = out.dropna(subset=["Open", "High", "Low", "Close"])
        return out

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @staticmethod
    def validate(df: pd.DataFrame) -> tuple[bool, list[str]]:
        """Schema + sanity checks. Returns (is_valid, list_of_problems)."""
        problems: list[str] = []

        missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing_cols:
            problems.append(f"missing columns: {missing_cols}")
            return False, problems  # can't check further without the columns

        if not isinstance(df.index, pd.DatetimeIndex):
            problems.append("index is not a DatetimeIndex")

        for col in ["Open", "High", "Low", "Close"]:
            if (df[col] < 0).any():
                problems.append(f"negative values found in {col}")

        if (df["Volume"] < 0).any():
            problems.append("negative values found in Volume")

        if (df["High"] < df["Low"]).any():
            problems.append("High < Low on at least one row")

        if df.isna().any().any():
            problems.append("NaN values remain after cleaning")

        return (len(problems) == 0), problems

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------
    @staticmethod
    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add real, useful features. Returns a new DataFrame with NaN warm-up rows dropped.

        Features (counted honestly, no padding):
          1. daily_return        - pct change in Close
          2. log_return          - log pct change in Close
          3. ma_5                - 5-day simple moving average of Close
          4. ma_20                - 20-day simple moving average of Close
          5. ma_ratio             - ma_5 / ma_20 (short/long trend ratio)
          6. volatility_10        - 10-day rolling std of daily_return
          7. volume_change        - pct change in Volume
          8. high_low_range       - (High - Low) / Close, intraday range
        """
        out = df.copy()
        out["daily_return"] = out["Close"].pct_change()
        # Equivalent to np.log(Close).diff() (log(P_t) - log(P_t-1) == log(P_t/P_t-1)),
        # but this form's result is used only as a plain assignment - no further
        # pandas method chained on np.log()'s output, which numpy's stubs type as
        # a bare ndarray (they don't know Series survives ufuncs, though it does
        # at runtime) and would otherwise flag a false-positive attribute error.
        out["log_return"] = np.log(out["Close"] / out["Close"].shift(1))
        out["ma_5"] = out["Close"].rolling(window=5).mean()
        out["ma_20"] = out["Close"].rolling(window=20).mean()
        out["ma_ratio"] = out["ma_5"] / out["ma_20"]
        out["volatility_10"] = out["daily_return"].rolling(window=10).std()
        out["volume_change"] = out["Volume"].pct_change()
        out["high_low_range"] = (out["High"] - out["Low"]) / out["Close"]

        feature_cols = [
            "daily_return", "log_return", "ma_5", "ma_20",
            "ma_ratio", "volatility_10", "volume_change", "high_low_range",
        ]
        out = out.dropna(subset=feature_cols)
        return out

    @property
    def feature_count(self) -> int:
        return 8

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def run(self, force_refresh: bool = False) -> dict[str, pd.DataFrame]:
        """Full pipeline: fetch -> clean -> validate -> feature-engineer, per ticker."""
        results: dict[str, pd.DataFrame] = {}
        for ticker in self.tickers:
            raw = self.fetch_ticker(ticker, force_refresh=force_refresh)
            cleaned = self.clean(raw)
            is_valid, problems = self.validate(cleaned)
            if not is_valid:
                raise ValueError(f"{ticker} failed validation: {problems}")
            featured = self.engineer_features(cleaned)
            featured.to_csv(config.PROCESSED_DIR / f"{ticker}_features.csv")
            results[ticker] = featured
            logger.info(
                "%s: %d raw rows -> %d feature rows, source=%s",
                ticker, len(raw), len(featured), self.source_used.get(ticker, "unknown"),
            )
        return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pipeline = StockDataPipeline()
    data = pipeline.run()
    for ticker, df in data.items():
        print(f"{ticker}: {len(df)} rows, source={pipeline.source_used[ticker]}")
        print(df.tail(3))
