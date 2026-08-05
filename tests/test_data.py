import numpy as np
import xarray as xr
import torch
import pytest

from data import (
    assemble_dataset, DatasetMaker, FeatureSpec, FeatureSelector,
    compute_norm_stats, calculate_f, buoyancy_flux,
    VarSpec, velocity_work_terms, TransientDatasetMaker,
    ProfileTransientDatasetMaker, MeanDatasetMaker,
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


def test_target_tendency_is_next_minus_current(ds):
    # M[c,t]=c*100+t  ->  dM = M_{i+1}-M_i == 1 everywhere. Target is the next state
    # (lag=-1); inputs at time i pair with the increment over [i, i+1].
    sel = FeatureSelector(
        ar_features=[], forcings=[FeatureSpec("forcing")],
        target=FeatureSpec("M", lag=-1), mode="sequence", target_tendency=True,
    )
    res = sel.select(ds)
    assert res.n_time == N_TIME - 1                       # one future step consumed
    assert res.X.shape == (N_CASES, N_TIME - 1, 1)
    assert res.y.shape == (N_CASES, N_TIME - 1, 1)
    assert res.target_names == ["dM_i+1"]
    assert torch.allclose(res.y, torch.ones_like(res.y))
    # y_state is the full state trajectory (length n_time+1) for reconstruction
    assert res.y_state.shape == (N_CASES, N_TIME, 1)
    assert torch.allclose(res.y_state[:, :, 0], torch.tensor(ds.M.values).float())
    # inputs at time i align to the increment over [i, i+1]
    assert torch.allclose(res.X[:, :, 0], torch.tensor(ds.forcing.values[:, :-1]).float())


def test_target_tendency_parallel_flattens(ds):
    sel = FeatureSelector([], [FeatureSpec("forcing")], FeatureSpec("M", lag=-1),
                          mode="parallel", target_tendency=True)
    res = sel.select(ds)
    assert res.X.shape == (N_CASES * (N_TIME - 1), 1)
    assert res.y.shape == (N_CASES * (N_TIME - 1), 1)
    assert res.y_state.shape == (N_CASES, N_TIME, 1)     # kept per-case for integration


def test_column_order_ar_first(ds):
    sel = FeatureSelector(
        ar_features=[FeatureSpec("M", lag=1)],
        forcings=[FeatureSpec("forcing")],
        target=FeatureSpec("M"),
        mode="sequence",
    )
    res = sel.select(ds)
    assert res.n_ar == 1
    assert res.feature_names[0] == "M_i-1"        # ar channel first
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


# --------------------------------------------------------------------------- #
# derived stress-work terms
# --------------------------------------------------------------------------- #
def test_work_along_tau_is_abs_tau_u():
    """The identity the old surf_mag/surf_angle/surf_ekin trio computed the long way."""
    u, v, tau = np.array([[3.0, -3.0]]), np.array([[4.0, 4.0]]), np.array([[2.0, 2.0]])
    t = velocity_work_terms(u, v, tau, "surf")
    np.testing.assert_allclose(t["surf_work"], tau * 5.0)
    np.testing.assert_allclose(t["surf_work_along_tau"], np.abs(tau * u))
    np.testing.assert_allclose(t["surf_veer"], np.arccos(u / 5.0))


def test_veer_signed_option_keeps_sign_of_v():
    u, v, tau = np.array([[1.0, 1.0]]), np.array([[1.0, -1.0]]), np.array([[1.0, 1.0]])
    unsigned = velocity_work_terms(u, v, tau, "s")["s_veer"]
    signed = velocity_work_terms(u, v, tau, "s", signed_veer=True)["s_veer"]
    assert unsigned[0, 0] == pytest.approx(unsigned[0, 1])      # arccos loses it
    assert signed[0, 0] == pytest.approx(-signed[0, 1])         # arctan2 keeps it


def test_zero_stress_stays_finite():
    """The old formulation divided tau by tau, giving 0/0 where tau == 0."""
    z = np.zeros((1, 1))
    terms = velocity_work_terms(z, z, z, "s")
    assert all(np.all(np.isfinite(a)) for a in terms.values())


# --------------------------------------------------------------------------- #
# DatasetMaker var specs: required / optional / user-supplied
# --------------------------------------------------------------------------- #
NAME = "toy"


def _write(d, suffix, key, arr, depth=None):
    dims = ("case", "time") if arr.ndim == 2 else ("case", "time", "sigma_depth")
    ds = xr.Dataset({key: (dims, arr)})
    if depth is not None:
        ds = ds.assign_coords(sigma_depth=depth)
    ds.to_netcdf(d / f"{NAME}_{suffix}")


@pytest.fixture
def raw(tmp_path):
    """A synthetic raw-file tree; factory lets each test drop or perturb files."""
    def build(with_surf=True, eps_depth=N_DEPTH):
        import json
        rng = np.random.default_rng(0)
        s = lambda: rng.normal(size=(N_CASES, N_TIME))
        d = tmp_path / f"raw_{with_surf}_{eps_depth}"
        d.mkdir(exist_ok=True)
        cases = {f"c{i}": dict(heat_flux=-10.0, tx=0.01 * (i + 1), lat=45.0)
                 for i in range(N_CASES)}
        (d / f"{NAME}_training_set_cases.json").write_text(json.dumps(cases))
        for suffix, key in [("M.nc", "M"), ("bl.nc", "bl"), ("tke_bl.nc", "bl"),
                            ("SS_bl_avg.nc", "SS_bl_mean"), ("NN_bl_avg.nc", "NN_bl_mean"),
                            ("wb_dataset.nc", "wb"), ("u_bl.nc", "u_bl"), ("v_bl.nc", "v_bl")]:
            _write(d, suffix, key, s())
        if with_surf:
            _write(d, "u_surf.nc", "u_surf", s())
            _write(d, "v_surf.nc", "v_surf", s())
        for suffix, key, n in [("S2_sigma_interp.nc", "SS", N_DEPTH),
                               ("N2_sigma_interp.nc", "NN", N_DEPTH),
                               ("tke_sigma_interp.nc", "tke", N_DEPTH),
                               ("eps_sigma_interp.nc", "eps", eps_depth)]:
            _write(d, suffix, key, rng.normal(size=(N_CASES, N_TIME, n)),
                   depth=np.linspace(0, -1, n))
        np.save(d / f"{NAME}_mean_Ms.npy", rng.normal(size=N_CASES))
        np.save(d / f"{NAME}_mean_bls.npy", rng.normal(size=N_CASES))
        return d
    return build


def test_transient_derives_both_velocity_levels(raw):
    ds = TransientDatasetMaker(str(raw()), NAME).build()
    for level in ("surf", "bl"):
        assert {f"{level}_work", f"{level}_veer", f"{level}_work_along_tau"} <= set(ds.data_vars)
    assert ds.attrs["vars_skipped"] == ""


def test_missing_optional_file_skips_instead_of_breaking(raw):
    """The point of the exercise: no u_surf.nc must not be fatal."""
    with pytest.warns(UserWarning, match="skipped"):
        ds = TransientDatasetMaker(str(raw(with_surf=False)), NAME).build()
    assert not any(v.startswith("surf_") for v in ds.data_vars)
    assert "bl_work" in ds.data_vars              # the other pair still derived
    assert ds.attrs["vars_skipped"] == "u_surf,v_surf"


def test_missing_required_file_raises(raw):
    d = raw()
    (d / f"{NAME}_M.nc").unlink()
    with pytest.raises(FileNotFoundError, match="requires 'M'"):
        TransientDatasetMaker(str(d), NAME).build()


def test_extra_vars_appends_without_subclassing(raw):
    d = raw()
    _write(d, "my_diag.nc", "raw_key", np.ones((N_CASES, N_TIME)))
    ds = TransientDatasetMaker(str(d), NAME,
                               extra_vars=[("my_diag.nc", "raw_key", "my_diag")]).build()
    assert "my_diag" in ds.data_vars


def test_wrong_in_file_key_lists_available(raw):
    d = raw()
    _write(d, "my_diag.nc", "raw_key", np.ones((N_CASES, N_TIME)))
    sel = [VarSpec("my_diag.nc", "typo", "x", required=True)]
    with pytest.raises(KeyError, match="raw_key"):
        TransientDatasetMaker(str(d), NAME, extra_vars=sel).build()


def test_duplicate_output_names_rejected(raw):
    with pytest.raises(ValueError, match="Duplicate output names"):
        TransientDatasetMaker(str(raw()), NAME, extra_vars=[("o.nc", "x", "M")])


def test_sources_attr_records_provenance(raw):
    ds = TransientDatasetMaker(str(raw()), NAME).build()
    assert "M<-M.nc:M" in ds.attrs["sources"]
    assert "SS_bl<-SS_bl_avg.nc:SS_bl_mean" in ds.attrs["sources"]


# --------------------------------------------------------------------------- #
# profiles: arbitrary sets, shared depth axis
# --------------------------------------------------------------------------- #
def test_profile_dataset_builds_on_common_grid(raw):
    ds = ProfileTransientDatasetMaker(str(raw(with_surf=False)), NAME).build()
    assert ds.SS.dims == ("case", "time", "depth")
    assert ds.attrs["n_depth"] == N_DEPTH
    assert ds.attrs["method"] == "interp"


def test_profile_depth_mismatch_raises(raw):
    """Previously the last file read silently won the depth coordinate."""
    d = raw(with_surf=False, eps_depth=N_DEPTH + 2)
    with pytest.raises(ValueError, match="common grid"):
        ProfileTransientDatasetMaker(str(d), NAME).build()


def test_arbitrary_profile_set(raw):
    ds = ProfileTransientDatasetMaker(
        str(raw(with_surf=False)), NAME,
        profile_vars=[("N2_sigma_{method}.nc", "NN", "buoyancy_freq_sq")]).build()
    assert "buoyancy_freq_sq" in ds.data_vars and "SS" not in ds.data_vars


def test_invalid_method_warns(raw):
    with pytest.warns(UserWarning, match="not among the currently valid"):
        ProfileTransientDatasetMaker(str(raw(with_surf=False)), NAME, method="raw")


def test_tau_min_mask_drops_cases(raw):
    d = raw(with_surf=False)
    ds = ProfileTransientDatasetMaker(str(d), NAME, tau_min=0.015).build()
    assert ds.sizes["case"] == 2          # tx = .01, .02, .03 -> keeps .02, .03
    assert ds.attrs["tau_min"] == 0.015


def test_mean_maker_reads_npy(raw):
    ds = MeanDatasetMaker(str(raw(with_surf=False)), NAME).build()
    assert ds.M.dims == ("case",) and ds.attrs["n_cases"] == N_CASES
