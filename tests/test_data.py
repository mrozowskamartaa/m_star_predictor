import numpy as np
import xarray as xr
import torch
import pytest

from data import (
    assemble_dataset, DatasetMaker, FeatureSpec, FeatureSelector,
    compute_norm_stats, calculate_f, buoyancy_flux,
)

N_CASES, N_TIME, N_DEPTH = 3, 6, 4


@pytest.fixture
def ds():
    # M[c, t] = c*100 + t  -> lag arithmetic is trivial to check by eye
    M = (np.arange(N_CASES)[:, None] * 100 + np.arange(N_TIME)[None, :]).astype(float)
    forcing = np.tile(np.arange(N_TIME, dtype=float), (N_CASES, 1))     # (case, time)
    const = np.array([10.0, 20.0, 30.0])                               # (case,) constant
    NN = np.arange(N_CASES * N_TIME * N_DEPTH, dtype=float).reshape(N_CASES, N_TIME, N_DEPTH)
    depth = np.linspace(0, 1, N_DEPTH)
    return assemble_dataset(
        scalars=dict(M=M, forcing=forcing, fconst=const),
        profiles=dict(NN=NN), depth=depth,
        attrs=dict(dataset_name="mock", kind="transient"),
    )


# --------------------------------------------------------------------------- #
# assembly / IO
# --------------------------------------------------------------------------- #
def test_assemble_dims_and_attrs(ds):
    assert ds.M.dims == ("case", "time")
    assert ds.fconst.dims == ("case",)
    assert ds.NN.dims == ("case", "time", "depth")
    assert ds.attrs["dataset_name"] == "mock"


def test_save_load_roundtrip(ds, tmp_path):
    path = tmp_path / "mock.nc"
    ds.to_netcdf(path)
    loaded = DatasetMaker.load(str(path))
    xr.testing.assert_equal(loaded[["M", "forcing"]], ds[["M", "forcing"]])
    assert loaded.attrs["kind"] == "transient"


# --------------------------------------------------------------------------- #
# lag arithmetic (the M_i-1 / M_i+1 replacement)
# --------------------------------------------------------------------------- #
def test_lag_window_and_alignment(ds):
    sel = FeatureSelector(
        ar_features=[FeatureSpec("M", lag=1)],       # M_i-1
        forcings=[],
        target=FeatureSpec("M", lag=-1),             # M_i+1
        mode="sequence",
    )
    res = sel.select(ds)
    # T=6, lags {1,-1} -> lo=1, hi=5, n_time=4
    assert res.n_time == 4
    assert res.X.shape == (N_CASES, 4, 1)
    assert res.y.shape == (N_CASES, 4, 1)
    M = ds.M.values
    for c in range(N_CASES):
        for tau in range(4):
            assert res.X[c, tau, 0].item() == M[c, tau]          # M_i-1
            assert res.y[c, tau, 0].item() == M[c, tau + 2]      # M_i+1


def test_column_order_ar_first(ds):
    sel = FeatureSelector(
        ar_features=[FeatureSpec("M", lag=1)],
        forcings=[FeatureSpec("forcing")],
        target=FeatureSpec("M"),
        mode="sequence",
    )
    res = sel.select(ds)
    assert res.n_ar == 1
    assert res.feature_names[0] == "M_lag1"       # ar channel first
    assert res.feature_names[1] == "forcing"


def test_parallel_flatten_shape(ds):
    sel = FeatureSelector([FeatureSpec("M", lag=1)], [FeatureSpec("forcing")],
                          FeatureSpec("M"), mode="parallel")
    res = sel.select(ds)
    # T=6, lags {1,0} -> lo=1, hi=6, n_time=5
    assert res.n_time == 5
    assert res.X.shape == (N_CASES * 5, 2)
    assert res.y.shape == (N_CASES * 5, 1)


def test_per_case_constant_broadcast(ds):
    sel = FeatureSelector([FeatureSpec("M", lag=1)], [FeatureSpec("fconst")],
                          FeatureSpec("M"), mode="sequence")
    res = sel.select(ds)
    fcol = res.feature_names.index("fconst")
    for c in range(N_CASES):
        assert torch.all(res.X[c, :, fcol] == ds.fconst.values[c])


def test_profile_expands_to_depth_columns(ds):
    sel = FeatureSelector([FeatureSpec("M", lag=1)], [FeatureSpec("NN")],
                          FeatureSpec("M"), mode="sequence")
    res = sel.select(ds)
    nn_cols = [n for n in res.feature_names if n.startswith("NN_")]
    assert len(nn_cols) == N_DEPTH
    assert res.X.shape[-1] == 1 + N_DEPTH


def test_transform_applied(ds):
    sel = FeatureSelector([FeatureSpec("M", lag=1)],
                          [FeatureSpec("forcing", transform="log1p")],
                          FeatureSpec("M"), mode="sequence")
    res = sel.select(ds)
    fcol = res.feature_names.index("forcing")
    # lags {1, 0} -> lo=1, hi=6, n_time=5; forcing maps to original times 1:6
    expected = np.log1p(ds.forcing.values[:, 1:6])
    np.testing.assert_allclose(res.X[:, :, fcol].numpy(), expected, rtol=1e-6)


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def test_zero_std_forcing_dropped(ds):
    ds2 = ds.copy()
    ds2["dead"] = (("case", "time"), np.full((N_CASES, N_TIME), 5.0))
    sel = FeatureSelector([FeatureSpec("M", lag=1)],
                          [FeatureSpec("forcing"), FeatureSpec("dead")],
                          FeatureSpec("M"), mode="sequence")
    with pytest.warns(UserWarning, match="zero-variance"):
        res = sel.select(ds2)
    assert "dead" in res.dropped
    assert "dead" not in res.feature_names


def test_zero_std_ar_raises(ds):
    ds2 = ds.copy()
    ds2["dead"] = (("case", "time"), np.full((N_CASES, N_TIME), 5.0))
    sel = FeatureSelector([FeatureSpec("dead")], [FeatureSpec("forcing")],
                          FeatureSpec("M"), mode="sequence")
    with pytest.raises(ValueError, match="zero variance"):
        sel.select(ds2)


def test_nan_in_features_raises_when_strict(ds):
    ds2 = ds.copy()
    m = ds2.forcing.values.copy()
    m[0, 3] = np.nan                                   # inside the valid window
    ds2["forcing"] = (("case", "time"), m)
    sel = FeatureSelector([FeatureSpec("M", lag=1)], [FeatureSpec("forcing")],
                          FeatureSpec("M"), mode="sequence", strict_nan=True)
    with pytest.raises(ValueError, match="Non-finite"):
        sel.select(ds2)


def test_compute_norm_stats(ds):
    sel = FeatureSelector([FeatureSpec("M", lag=1)], [FeatureSpec("forcing")],
                          FeatureSpec("M"), mode="parallel")
    res = sel.select(ds)
    mean, std = compute_norm_stats(res.X.numpy())
    assert mean.shape == (2,)
    assert np.all(std > 0)


# --------------------------------------------------------------------------- #
# physics helpers
# --------------------------------------------------------------------------- #
def test_calculate_f_equator_and_sign():
    assert calculate_f(0.0) == pytest.approx(0.0)
    assert calculate_f(45.0) > 0
    assert calculate_f(-45.0) < 0


def test_buoyancy_flux_sign():
    assert buoyancy_flux(100.0) > 0     # heating -> positive B
