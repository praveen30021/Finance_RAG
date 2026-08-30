"""
Minimal smoke tests: the dashboard module imports cleanly and its helper
functions behave sanely on sample data. Streamlit's own runtime (widgets,
session state) isn't exercised here - that needs `streamlit run`, not pytest.
"""
import importlib

import pandas as pd


def test_dashboard_module_imports():
    module = importlib.import_module("dashboard")
    assert hasattr(module, "main")
    assert hasattr(module, "get_stock_data")
    assert hasattr(module, "get_forecast_summary")
    assert hasattr(module, "get_sentiment_classifier")
    assert hasattr(module, "get_rag_pipeline")


def test_get_forecast_summary_underlying_function_on_sample_series():
    from forecasting import ForecastEngine

    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    series = pd.Series(range(100, 130), index=dates, dtype=float)
    engine = ForecastEngine(min_train=10, step=5)
    summary = engine.evaluate_all({"TEST": series})
    assert "arima_mape" in summary.columns
    assert "TEST" in summary.index
