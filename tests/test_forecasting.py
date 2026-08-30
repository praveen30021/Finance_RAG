import numpy as np
import pandas as pd

from forecasting import ForecastEngine, _mape, _rmse


def test_mape_simple_case():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 220.0])
    assert abs(_mape(y_true, y_pred) - 10.0) < 1e-9


def test_rmse_simple_case():
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([3.0, 4.0])
    assert abs(_rmse(y_true, y_pred) - 3.5355339059327378) < 1e-6


def test_walk_forward_folds_boundaries():
    engine = ForecastEngine(min_train=10, step=1)
    folds = engine.walk_forward_folds(15)
    # expect train_end from 10..14 (since range stops before n - horizon + 1 = 15)
    assert folds[0] == (10, 10)
    assert folds[-1] == (14, 14)
    assert len(folds) == 5


def test_walk_forward_folds_respects_step():
    engine = ForecastEngine(min_train=10, step=3)
    folds = engine.walk_forward_folds(20)
    train_ends = [f[0] for f in folds]
    assert train_ends == list(range(10, 20, 3))


def test_walk_forward_folds_empty_when_insufficient_data():
    engine = ForecastEngine(min_train=250)
    assert engine.walk_forward_folds(100) == []


def test_naive_baseline_predict_is_last_training_value():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    pred = ForecastEngine.naive_baseline_predict(series, train_end=3)
    assert pred == series.iloc[2]
