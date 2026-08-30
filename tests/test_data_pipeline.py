import numpy as np
import pandas as pd
import pytest

from data_pipeline import StockDataPipeline


def make_ohlcv(n=30, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    df = pd.DataFrame({
        "Open": close + rng.normal(0, 0.1, n),
        "High": close + np.abs(rng.normal(0, 0.5, n)),
        "Low": close - np.abs(rng.normal(0, 0.5, n)),
        "Close": close,
        "Volume": rng.integers(1000, 5000, n).astype(float),
    }, index=dates)
    df.index.name = "Date"
    return df


def test_validate_passes_on_clean_data():
    df = make_ohlcv()
    ok, problems = StockDataPipeline.validate(df)
    assert ok, problems


def test_validate_flags_negative_prices():
    df = make_ohlcv()
    df.loc[df.index[0], "Close"] = -5
    ok, problems = StockDataPipeline.validate(df)
    assert not ok
    assert any("negative" in p for p in problems)


def test_clean_forward_fills_missing_values():
    df = make_ohlcv()
    df.loc[df.index[5], ["Open", "High", "Low", "Close"]] = np.nan
    cleaned = StockDataPipeline.clean(df)
    assert not cleaned.isna().any().any()
    # forward-filled value should equal the prior row's Close
    prior_close = df["Close"].iloc[4]
    assert cleaned["Close"].iloc[5] == pytest.approx(prior_close)


def test_engineer_features_produces_expected_columns():
    df = make_ohlcv(n=40)
    featured = StockDataPipeline.engineer_features(df)
    expected_cols = {
        "daily_return", "log_return", "ma_5", "ma_20",
        "ma_ratio", "volatility_10", "volume_change", "high_low_range",
    }
    assert expected_cols.issubset(set(featured.columns))
    assert not featured[list(expected_cols)].isna().to_numpy().any()
    assert len(featured) < len(df)  # warm-up rows dropped


def test_feature_count_matches_engineered_columns():
    pipeline = StockDataPipeline()
    assert pipeline.feature_count == 8
