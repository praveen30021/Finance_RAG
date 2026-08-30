"""
Module 2: ForecastEngine

ARIMA forecasting with walk-forward cross-validation, benchmarked against
a naive ("tomorrow = today") baseline. Reports MAPE and RMSE per ticker.
"""
from __future__ import annotations

import os

# See main.py for the full explanation: pin BLAS threads to 1 before numpy
# is imported, so a standalone `python forecasting.py` run is also protected
# against Windows OpenBLAS thread-pool exhaustion during walk-forward CV.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import logging
import warnings

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


class ForecastEngine:
    """Walk-forward ARIMA forecasting with a naive baseline for comparison."""

    def __init__(
        self,
        order: tuple[int, int, int] = config.ARIMA_ORDER,
        min_train: int = config.WALK_FORWARD_MIN_TRAIN,
        horizon: int = config.WALK_FORWARD_HORIZON,
        step: int = 5,
    ):
        self.order = order
        self.min_train = min_train
        self.horizon = horizon
        # Refitting ARIMA at every single trading day over ~5y of history
        # (~1000+ folds/ticker) is prohibitively slow for a laptop build.
        # `step` subsamples the expanding window (still genuine walk-forward:
        # train-only-on-the-past, no look-ahead) so a full run finishes in
        # minutes instead of hours. step=1 recovers the exhaustive version.
        self.step = max(1, step)

    # ------------------------------------------------------------------
    # Walk-forward split
    # ------------------------------------------------------------------
    def walk_forward_folds(self, n: int) -> list[tuple[int, int]]:
        """Return list of (train_end_idx, test_idx) index pairs.

        Expanding window: train on series[0:train_end_idx], predict series[test_idx],
        where test_idx == train_end_idx (next point after the training window).
        """
        folds = []
        if n <= self.min_train:
            return folds
        for train_end in range(self.min_train, n - self.horizon + 1, self.step):
            test_idx = train_end
            folds.append((train_end, test_idx))
        return folds

    # ------------------------------------------------------------------
    # Naive baseline
    # ------------------------------------------------------------------
    @staticmethod
    def naive_baseline_predict(series: pd.Series, train_end: int) -> float:
        """Predict tomorrow = today: last observed value in the training window."""
        return float(series.iloc[train_end - 1])

    # ------------------------------------------------------------------
    # ARIMA one-step forecast
    # ------------------------------------------------------------------
    def _arima_predict(self, train_series: pd.Series) -> float:
        from statsmodels.tsa.arima.model import ARIMA

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(train_series.values, order=self.order)
            fitted = model.fit()
            forecast = fitted.forecast(steps=self.horizon)
        return float(forecast[-1])

    # ------------------------------------------------------------------
    # Walk-forward evaluation for one ticker's Close series
    # ------------------------------------------------------------------
    def evaluate_ticker(self, series: pd.Series, ticker: str = "") -> dict:
        series = series.dropna()
        n = len(series)
        folds = self.walk_forward_folds(n)
        if not folds:
            raise ValueError(
                f"Not enough data ({n} rows) for min_train={self.min_train}"
            )

        arima_preds, naive_preds, actuals, dates = [], [], [], []
        index = series.index

        for train_end, test_idx in folds:
            train_series = series.iloc[:train_end]
            actual = float(series.iloc[test_idx])
            naive_pred = self.naive_baseline_predict(series, train_end)
            try:
                arima_pred = self._arima_predict(train_series)
            except Exception as exc:
                logger.warning("%s: ARIMA fit failed at fold train_end=%d: %s", ticker, train_end, exc)
                arima_pred = naive_pred  # degrade gracefully rather than crash the whole eval

            arima_preds.append(arima_pred)
            naive_preds.append(naive_pred)
            actuals.append(actual)
            dates.append(index[test_idx])

        actuals_arr = np.array(actuals)
        arima_arr = np.array(arima_preds)
        naive_arr = np.array(naive_preds)

        result = {
            "ticker": ticker,
            "n_folds": len(folds),
            "arima_mape": _mape(actuals_arr, arima_arr),
            "arima_rmse": _rmse(actuals_arr, arima_arr),
            "naive_mape": _mape(actuals_arr, naive_arr),
            "naive_rmse": _rmse(actuals_arr, naive_arr),
            "dates": dates,
            "actuals": actuals,
            "arima_preds": arima_preds,
            "naive_preds": naive_preds,
        }
        return result

    def evaluate_all(self, ticker_series: dict[str, pd.Series]) -> pd.DataFrame:
        rows = []
        self._last_details: dict[str, dict] = {}
        for ticker, series in ticker_series.items():
            res = self.evaluate_ticker(series, ticker=ticker)
            self._last_details[ticker] = res
            rows.append({
                "ticker": ticker,
                "n_folds": res["n_folds"],
                "arima_mape": res["arima_mape"],
                "arima_rmse": res["arima_rmse"],
                "naive_mape": res["naive_mape"],
                "naive_rmse": res["naive_rmse"],
            })
        summary = pd.DataFrame(rows).set_index("ticker")
        avg_row = summary[["arima_mape", "arima_rmse", "naive_mape", "naive_rmse"]].mean()
        summary.loc["AVERAGE"] = [np.nan, *avg_row.values]
        return summary


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO)

    from data_pipeline import StockDataPipeline

    pipeline = StockDataPipeline()
    data = pipeline.run()
    engine = ForecastEngine()
    series_map = {t: df["Close"] for t, df in data.items()}
    summary = engine.evaluate_all(series_map)
    print(summary)
    summary.to_csv(config.PROCESSED_DIR / "forecast_summary.csv")
