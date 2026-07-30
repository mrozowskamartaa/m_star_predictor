import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")           # headless: plots must build without a display
import matplotlib.pyplot as plt
import pytest

import plots
from experiment import Split, Prediction


def _mock_prediction(label, n_cases=3, n_time=8, n_feat=2, feature_names=None):
    truth = torch.randn(n_cases, n_time)
    pred = truth + torch.randn(n_cases, n_time) * 0.1
    feats = torch.randn(n_cases, n_time, n_feat)
    names = feature_names or [f"feat{j}" for j in range(n_feat)]
    split = Split(truth=truth, pred=pred, n_time=n_time, n_cases=n_cases,
                  features=feats, feature_names=names)
    return Prediction(label=label, val=split, feature_names=names)


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def test_scatter_state_and_tendency():
    a, b = _mock_prediction("A"), _mock_prediction("B")
    plots.scatter(a, b, quantity="state")
    plots.scatter(a, b, quantity="tendency", lim=(-3, 3))


def test_case_timeseries_with_features():
    a = _mock_prediction("A", feature_names=["u_star", "bl_tke"])
    b = _mock_prediction("B", feature_names=["u_star", "bl_tke"])
    figs = plots.case_timeseries([a, b], cases=[0, 1], features=["u_star"])
    assert len(figs) == 2


def test_case_timeseries_tendency():
    a = _mock_prediction("A")
    figs = plots.case_timeseries(a, cases=[0], quantity="tendency")
    assert len(figs) == 1


def test_rmse_table():
    a, b = _mock_prediction("A"), _mock_prediction("B")
    df = plots.rmse_table(a, b)
    assert set(df["label"]) == {"A", "B"}
    assert "state" in df.columns and "tendency" in df.columns


def test_hovmoller_profile():
    names = [f"NN_{d:.2f}" for d in np.linspace(0, 1, 5)] + ["u_star"]
    pred = _mock_prediction("prof", n_feat=6, feature_names=names)
    ax = plots.hovmoller(pred, case=0, feature="NN", log=False)
    assert ax is not None


def test_hovmoller_missing_feature_raises():
    pred = _mock_prediction("A", feature_names=["u_star", "bl_tke"])
    with pytest.raises(ValueError, match="No profile columns"):
        plots.hovmoller(pred, case=0, feature="NN")
