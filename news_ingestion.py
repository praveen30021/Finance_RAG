"""
Module 3: NewsIngestion

Two responsibilities:
  1. Load Financial PhraseBank (sentiment-labeled sentences) and normalize
     to a (text, label) schema for SentimentClassifier.
  2. Stream and filter FNSPID (full-length financial news articles) down
     to the configured tickers + date range, normalized to
     (headline, body, source, date, ticker) for RAGPipeline.

Both are real, named, public datasets (see README for citations/licenses).
"""
from __future__ import annotations

import json
import logging

import pandas as pd

import config

logger = logging.getLogger(__name__)

LABEL_MAP: dict[str | int, int] = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
    0: 0, 1: 1, 2: 2,
}


class NewsIngestion:
    def __init__(self):
        self.phrasebank_source_used: str | None = None
        self.fnspid_source_used: str | None = None

    # ------------------------------------------------------------------
    # 1. Financial PhraseBank -> sentiment schema
    # ------------------------------------------------------------------
    def load_phrasebank(self, force_refresh: bool = False) -> pd.DataFrame:
        """Return DataFrame with columns: text, label (str), label_id (int).

        Tries Kaggle first (if kaggle.json credentials are configured),
        then falls back to the Hugging Face mirror (no auth required).
        Whichever path actually serves the data is recorded in
        self.phrasebank_source_used for the README.
        """
        if config.PHRASEBANK_LOCAL_CSV.exists() and not force_refresh:
            df = pd.read_csv(config.PHRASEBANK_LOCAL_CSV)
            self.phrasebank_source_used = "cache"
            logger.info("Loaded Financial PhraseBank from cache (%d rows)", len(df))
            return self._normalize_phrasebank(df)

        df = self._load_phrasebank_kaggle()
        if df is not None:
            self.phrasebank_source_used = "kaggle"
        else:
            df = self._load_phrasebank_huggingface()
            self.phrasebank_source_used = "huggingface"

        if df is None or df.empty:
            raise RuntimeError("Could not load Financial PhraseBank from Kaggle or Hugging Face")

        df.to_csv(config.PHRASEBANK_LOCAL_CSV, index=False)
        return self._normalize_phrasebank(df)

    def _load_phrasebank_kaggle(self) -> pd.DataFrame | None:
        try:
            import kaggle  # type: ignore[import-not-found]  # optional dep, raises OSError if kaggle.json isn't configured
        except Exception:
            logger.info("Kaggle credentials not configured; skipping Kaggle source")
            return None
        try:
            import zipfile
            import io

            api = kaggle.api
            api.authenticate()
            dest = config.RAW_DIR / "phrasebank_kaggle"
            dest.mkdir(exist_ok=True)
            api.dataset_download_files(config.PHRASEBANK_KAGGLE_DATASET, path=str(dest), unzip=True)
            # The Kaggle release ships several .txt files, one per agreement level.
            candidates = sorted(dest.glob("*AllAgree*.txt")) or sorted(dest.glob("*.txt"))
            if not candidates:
                return None
            rows = []
            for path in candidates:
                with open(path, encoding="latin-1") as f:
                    for line in f:
                        line = line.strip()
                        if "@" not in line:
                            continue
                        text, label = line.rsplit("@", 1)
                        rows.append({"text": text.strip(), "label": label.strip()})
            return pd.DataFrame(rows)
        except Exception as exc:
            logger.warning("Kaggle download failed (%s); falling back to Hugging Face", exc)
            return None

    def _load_phrasebank_huggingface(self) -> pd.DataFrame:
        """Download the raw Financial PhraseBank release from its HF dataset repo.

        `datasets.load_dataset(...)` can't be used here: the takala/financial_phrasebank
        repo only ships a legacy Python loading script, and the installed `datasets`
        version (5.x) has dropped loading-script support entirely. The repo does host
        the original Malo et al. (2014) release zip directly, so we pull that file via
        huggingface_hub and parse it ourselves - same raw data, no auth required.
        """
        import zipfile

        from huggingface_hub import hf_hub_download

        zip_path = hf_hub_download(
            repo_id=config.PHRASEBANK_HF_DATASET,
            repo_type="dataset",
            filename="data/FinancialPhraseBank-v1.0.zip",
        )

        agreement_filename = {
            "sentences_allagree": "Sentences_AllAgree.txt",
            "sentences_75agree": "Sentences_75Agree.txt",
            "sentences_66agree": "Sentences_66Agree.txt",
            "sentences_50agree": "Sentences_50Agree.txt",
        }.get(config.PHRASEBANK_HF_CONFIG, "Sentences_50Agree.txt")

        rows = []
        with zipfile.ZipFile(zip_path) as zf:
            member = next(n for n in zf.namelist() if n.endswith(agreement_filename))
            with zf.open(member) as f:
                for raw_line in f:
                    line = raw_line.decode("latin-1").strip()
                    if "@" not in line:
                        continue
                    text, label = line.rsplit("@", 1)
                    rows.append({"text": text.strip(), "label": label.strip()})

        return pd.DataFrame(rows)

    @staticmethod
    def _normalize_phrasebank(df: pd.DataFrame) -> pd.DataFrame:
        # See the identical annotation in data_pipeline.clean() for why:
        # boolean-array indexing below otherwise cascades into false-positive
        # type errors on every later pandas call in this method.
        out: pd.DataFrame = df.copy()
        out["label"] = out["label"].astype(str).str.lower().str.strip()
        out = out[out["label"].isin(["negative", "neutral", "positive"])]
        out["label_id"] = out["label"].map(LABEL_MAP)
        out = out.dropna(subset=["text", "label_id"])
        out["text"] = out["text"].astype(str).str.strip()
        out = out[out["text"].str.len() > 0]
        out = out.drop_duplicates(subset=["text"]).reset_index(drop=True)
        return out[["text", "label", "label_id"]]

    # ------------------------------------------------------------------
    # 2. FNSPID -> RAG corpus schema
    # ------------------------------------------------------------------
    def load_fnspid_filtered(
        self,
        tickers: list[str] | None = None,
        date_start: str = config.FNSPID_DATE_START,
        date_end: str = config.FNSPID_DATE_END,
        max_articles: int = config.FNSPID_MAX_ARTICLES,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Stream FNSPID from Hugging Face and filter to tickers + date range.

        Returns DataFrame with columns: headline, body, source, date, ticker.
        Never downloads the full 15.7M-row dataset - uses streaming mode and
        stops once max_articles matching rows are collected.
        """
        tickers = tickers or list(config.TICKERS)

        if config.FNSPID_LOCAL_CACHE.exists() and not force_refresh:
            rows = [json.loads(line) for line in open(config.FNSPID_LOCAL_CACHE, encoding="utf-8")]
            self.fnspid_source_used = "cache"
            logger.info("Loaded FNSPID filtered subset from cache (%d rows)", len(rows))
            return pd.DataFrame(rows)

        self.fnspid_source_used = "huggingface_range_scan"
        rows = self._scan_fnspid_via_range_requests(
            tickers=tickers, date_start=date_start, date_end=date_end, max_articles=max_articles,
        )

        with open(config.FNSPID_LOCAL_CACHE, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        return pd.DataFrame(rows)

    @staticmethod
    def _find_safe_csv_split(buf: str) -> int:
        """Largest prefix index of `buf` that ends after a newline outside any quoted field.

        Used to know how much of a raw byte chunk is safe to hand to csv.reader
        without risking a row that's been cut mid-quoted-field by the chunk boundary.
        """
        quotes = 0
        last_safe = 0
        for i, ch in enumerate(buf):
            if ch == '"':
                quotes += 1
            elif ch == "\n" and quotes % 2 == 0:
                last_safe = i + 1
        return last_safe

    def _scan_fnspid_via_range_requests(
        self,
        tickers: list[str],
        date_start: str,
        date_end: str,
        max_articles: int,
        chunk_bytes: int = 8_000_000,
        time_budget_sec: int = 20 * 60,
    ) -> list[dict]:
        """Stream FNSPID's raw CSV via sequential HTTP Range requests, one ticker at a time.

        Why not `datasets.load_dataset(..., streaming=True)` or
        `pandas.read_csv(response.raw, chunksize=...)`: both were tried first
        and both stalled indefinitely against this specific ~23GB file in this
        environment (zero rows/zero bytes of progress after 10+ minutes),
        while plain sequential `requests.get(..., headers={"Range": ...})`
        calls against the same URL were measured at a steady ~1-2 MB/s.

        Why per-ticker, not one pass over all tickers: `nasdaq_exteral_data.csv`
        turns out to be laid out as large, roughly-alphabetically-ordered
        contiguous blocks per ticker rather than shuffled. A single sequential
        scan from byte 0 only reaches whichever ticker's block sorts earliest
        within any reasonable time budget - tickers later in the alphabet are
        many GB away and would never be reached. Each ticker instead jumps to
        its pre-located block offset (config.FNSPID_TICKER_START_OFFSETS,
        found by manual binary-search probing - see README) and scans forward
        from there; a ticker with no known offset falls back to byte 0.

        Time/row-count bounded per ticker: stops at its share of
        `max_articles` or `time_budget_sec` (split evenly across tickers),
        whichever comes first, and reports both counts honestly rather than
        pretending the scan was exhaustive.
        """
        import csv

        csv.field_size_limit(2_000_000)  # FNSPID article-body fields exceed the 131072-char default

        # Reverse-alias map: e.g. "GOOG" (as tagged in FNSPID) -> "GOOGL" (this
        # project's canonical symbol everywhere else). See config.FNSPID_TICKER_ALIASES.
        per_ticker_budget = max(30, time_budget_sec // max(1, len(tickers)))
        per_ticker_cap = max(1, max_articles // max(1, len(tickers)))

        all_matches: list[dict] = []
        for ticker in tickers:
            fnspid_symbol = config.FNSPID_TICKER_ALIASES.get(ticker, ticker)
            start_offset = config.FNSPID_TICKER_START_OFFSETS.get(ticker, 0)
            matches = self._scan_single_ticker_block(
                fnspid_symbol=fnspid_symbol,
                canonical_ticker=ticker,
                start_offset=start_offset,
                date_start=date_start,
                date_end=date_end,
                max_articles=per_ticker_cap,
                chunk_bytes=chunk_bytes,
                time_budget_sec=per_ticker_budget,
            )
            logger.info("FNSPID scan: %s -> %d matching articles", ticker, len(matches))
            all_matches.extend(matches)

        return all_matches

    def _scan_single_ticker_block(
        self,
        fnspid_symbol: str,
        canonical_ticker: str,
        start_offset: int,
        date_start: str,
        date_end: str,
        max_articles: int,
        chunk_bytes: int,
        time_budget_sec: int,
    ) -> list[dict]:
        import csv
        import io
        import time

        import requests

        url = (
            f"https://huggingface.co/datasets/{config.FNSPID_HF_DATASET}"
            "/resolve/main/Stock_news/nasdaq_exteral_data.csv"
        )
        header = ["Unnamed: 0", "Date", "Article_title", "Stock_symbol", "Url", "Publisher",
                  "Author", "Article", "Lsa_summary", "Luhn_summary", "Textrank_summary", "Lexrank_summary"]

        session = requests.Session()
        offset = start_offset
        buffer = ""
        skipped_first_partial_line = start_offset == 0  # header row only exists at true offset 0
        matches: list[dict] = []
        rows_scanned = 0
        t0 = time.time()

        while True:
            elapsed = time.time() - t0
            if elapsed > time_budget_sec or len(matches) >= max_articles:
                break

            resp = session.get(url, headers={"Range": f"bytes={offset}-{offset + chunk_bytes - 1}"}, timeout=60)
            if resp.status_code not in (200, 206):
                logger.warning("FNSPID scan (%s): HTTP %d, stopping", fnspid_symbol, resp.status_code)
                break
            chunk = resp.content
            if not chunk:
                break
            offset += len(chunk)
            buffer += chunk.decode("utf-8", errors="ignore")

            if not skipped_first_partial_line:
                first_nl = buffer.find("\n")
                if first_nl < 0:
                    continue
                buffer = buffer[first_nl + 1:]
                skipped_first_partial_line = True

            split_at = self._find_safe_csv_split(buffer)
            if split_at == 0:
                continue
            ready, buffer = buffer[:split_at], buffer[split_at:]

            csv_rows = list(csv.reader(io.StringIO(ready)))
            if start_offset == 0 and rows_scanned == 0 and csv_rows and csv_rows[0][:2] == header[:2]:
                csv_rows = csv_rows[1:]  # drop the real header row

            for raw_row in csv_rows:
                if len(raw_row) != len(header):
                    continue
                rows_scanned += 1
                rec = dict(zip(header, raw_row))
                row = self._normalize_fnspid_record(rec)
                if row is None or row["ticker"] != fnspid_symbol:
                    continue
                row["ticker"] = canonical_ticker
                date_str = (row["date"] or "")[:10]
                if not (date_start <= date_str <= date_end):
                    continue
                matches.append(row)
                if len(matches) >= max_articles:
                    break

        logger.info(
            "FNSPID scan (%s): %.2f GB, %d rows scanned, %d matches, %ds elapsed",
            fnspid_symbol, offset / 1e9, rows_scanned, len(matches), time.time() - t0,
        )
        return matches

    @staticmethod
    def _normalize_fnspid_record(record: dict) -> dict | None:
        """Best-effort normalization; FNSPID column names vary by release."""
        def pick(*keys):
            for k in keys:
                if k in record and record[k] not in (None, ""):
                    return record[k]
            return None

        ticker = pick("Stock_symbol", "stock_symbol", "ticker", "Symbol")
        headline = pick("Article_title", "title", "headline", "Headline")
        body = pick("Article", "article", "text", "Text", "content")
        date = pick("Date", "date", "Datetime")
        source = pick("Url", "url", "source", "Source") or "FNSPID"

        if not ticker or not (headline or body):
            return None

        return {
            "ticker": str(ticker).upper().strip(),
            "headline": str(headline or "").strip(),
            "body": str(body or "").strip(),
            "date": str(date) if date is not None else None,
            "source": str(source),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingest = NewsIngestion()

    pb = ingest.load_phrasebank()
    print(f"Financial PhraseBank: {len(pb)} rows, source={ingest.phrasebank_source_used}")
    print(pb["label"].value_counts())

    fn = ingest.load_fnspid_filtered()
    print(f"FNSPID filtered: {len(fn)} rows, source={ingest.fnspid_source_used}")
