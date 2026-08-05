import numpy as np
import xarray as xr
import torch
from torch import nn, optim
import torch.utils.data as Data
import pytest

from data import assemble_dataset, FeatureSpec, FeatureSelector, compute_norm_stats
from model import FCNN, ParallelPredictor, _step
from metrics import normalize, real_units
from experiment import (
    PredictorConfig, Registry, Split, Prediction, restore,
    persistence_prediction, epbl_prediction, integrate_tendency,
)


def _mock_ds(n_cases=4, n_time=10, seed=0):
    rng = np.random.default_rng(seed)
    forcing = rng.normal(size=(n_cases, n_time)) * 0.1
    M = np.cumsum(forcing, axis=1) + rng.normal(size=(n_cases, n_time)) * 0.01
    fconst = np.linspace(1e-4, 2e-4, n_cases)
    return assemble_dataset(
        scalars=dict(M=M, forcing=forcing, f=fconst),
        attrs=dict(dataset_name="mock", kind="transient"),
    )


def _base_config(**over):
    defaults = dict(
        run_id="predictor_test", mode="parallel",
        dataset_path="", val_dataset_path="",
        ar_features=[dict(name="M", lag=1, transform=None)],
        forcings=[dict(name="forcing", lag=0, transform=None)],
        target=[dict(name="M", lag=0, transform=None)],
        feature_names=["M_i-1", "forcing"],
        network="fcnn", n_neurons_list=[8], activation="tanh",
        n_ar=1, predict_tendency=False, k=4,
        learning_rate=1e-2, weight_decay=0.0, loss="mse_loss", n_epochs=5,
        random_state=0, train_batch_size=8, test_batch_size=8,
        feature_mean=[0.0, 0.0], feature_std=[1.0, 1.0],
        target_mean=[0.0], target_std=[1.0],
        train_loss_trajectory=[1.0], test_loss_trajectory=[1.0],
        n_time=9, n_time_val=9, n_cases=4, n_cases_val=4, rmse=0.1,
    )
    defaults.update(over)
    return PredictorConfig(**defaults)


# --------------------------------------------------------------------------- #
# Config + Registry
# --------------------------------------------------------------------------- #
def test_config_save_load_roundtrip(tmp_path):
    cfg = _base_config()
    cfg.save(str(tmp_path))
    loaded = PredictorConfig.load(str(tmp_path))
    assert loaded == cfg
    assert loaded.target_transform is None
    assert loaded.selector_mode == "parallel"


def test_config_builds_network_and_selector():
    cfg = _base_config()
    net = cfg.build_network()
    assert isinstance(net, FCNN)
    assert net(torch.randn(3, 2)).shape == (3, 1)
    assert isinstance(cfg.to_selector(), FeatureSelector)


def test_registry_lists_and_tabulates(tmp_path):
    for i, rid in enumerate(["predictor_1", "predictor_2"]):
        _base_config(run_id=rid, rmse=0.1 * i).save(str(tmp_path / rid))
    reg = Registry(str(tmp_path))
    assert reg.run_ids() == ["predictor_1", "predictor_2"]
    table = reg.table()
    assert list(table["run_id"]) == ["predictor_1", "predictor_2"]
    assert "features" in table.columns


# --------------------------------------------------------------------------- #
# Split views: state vs tendency
# --------------------------------------------------------------------------- #
def test_split_state_and_tendency():
    truth = torch.tensor([[0.0, 1.0, 3.0]])
    pred = torch.tensor([[0.0, 1.5, 3.5]])
    s = Split(truth=truth, pred=pred, n_time=3, n_cases=1)
    t, p = s.series("state")
    torch.testing.assert_close(p, pred)
    tt, tp = s.series("tendency")
    torch.testing.assert_close(tt, torch.tensor([[1.0, 2.0]]))     # diff of truth
    torch.testing.assert_close(tp, torch.tensor([[1.5, 2.0]]))     # diff of pred
    assert s.rmse("state") >= 0.0


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def test_persistence_prediction():
    ds = _mock_ds()
    pred = persistence_prediction(ds)
    s = pred.val
    assert s.pred.shape == (4, 9)
    torch.testing.assert_close(s.pred[:, 0], torch.tensor(ds.M.values[:, 0]).float())
    torch.testing.assert_close(s.truth[:, 0], torch.tensor(ds.M.values[:, 1]).float())


def test_epbl_prediction_neutral_is_finite():
    # all-neutral (B=0) exercises only the neutral branch -> finite algebraic M
    ds = assemble_dataset(scalars=dict(
        M=np.full((2, 5), 1e-5), bl_tke=np.full((2, 5), 50.0),
        wb=np.zeros((2, 5)), B=np.zeros(2),
        f=np.array([1e-4, 1.2e-4]), u_star=np.array([0.01, 0.02]),
    ))
    pred = epbl_prediction(ds)
    assert pred.val.pred.shape == (2, 5)
    assert torch.all(torch.isfinite(pred.val.pred))


# --------------------------------------------------------------------------- #
# Tendency target: dM training + cumulative-sum state reconstruction
# --------------------------------------------------------------------------- #
def test_integrate_tendency_cumsum():
    dM = torch.tensor([[[1.0], [2.0], [3.0]]])         # (1, 3, 1)
    m0 = torch.tensor([[[10.0]]])
    out = integrate_tendency(dM, m0)
    assert out.shape == (1, 4, 1)                       # one longer than the increments
    torch.testing.assert_close(out[0, :, 0], torch.tensor([10.0, 11.0, 13.0, 16.0]))


def test_target_tendency_and_predict_tendency_are_exclusive():
    with pytest.raises(ValueError):
        _base_config(target_tendency=True, predict_tendency=True)


def test_tendency_restore_reconstructs_by_cumsum(tmp_path):
    torch.manual_seed(0)
    ds = _mock_ds()
    ds_path = tmp_path / "mock.nc"
    ds.to_netcdf(ds_path)

    sel = FeatureSelector([], [FeatureSpec("forcing")], FeatureSpec("M", lag=-1),
                          mode="parallel", target_tendency=True)
    res = sel.select(ds)
    fmean, fstd = compute_norm_stats(res.X.numpy())
    fmean, fstd = torch.tensor(fmean).float(), torch.tensor(fstd).float()
    tmean, tstd = res.y.mean(0), res.y.std(0)           # scaled by std(dM), not std(M)
    Xn = normalize(res.X, fmean, fstd)

    net = FCNN(input_size=len(res.feature_names), n_neurons_list=[8], activation=nn.Tanh())
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    torch.save(net.state_dict(), run_dir / "weights.pt")
    cfg = _base_config(
        run_id="run", dataset_path=str(ds_path), val_dataset_path=str(ds_path),
        ar_features=[], forcings=[dict(name="forcing", lag=0, transform=None)],
        target=[dict(name="M", lag=-1, transform=None)], feature_names=res.feature_names,
        n_ar=0, predict_tendency=False, target_tendency=True,
        feature_mean=fmean.tolist(), feature_std=fstd.tolist(),
        target_mean=tmean.tolist(), target_std=tstd.tolist(),
        n_time=res.n_time, n_time_val=res.n_time, n_cases=res.n_cases, n_cases_val=res.n_cases,
    )
    cfg.save(str(run_dir))

    prediction = restore(str(run_dir))
    s = prediction.val

    net.eval()
    with torch.no_grad():
        dM_real = real_units(_step(net, Xn, 0, False), tmean, tstd).reshape(
            res.n_cases, res.n_time, -1)
    expected = integrate_tendency(dM_real, res.y_state[:, :1, :])[..., 0]

    torch.testing.assert_close(s.pred, expected, rtol=1e-5, atol=1e-6)
    assert s.pred.shape == (res.n_cases, res.n_time + 1)        # states, one past the dMs
    torch.testing.assert_close(s.truth[:, 0], res.y_state[:, 0, 0])   # true initial condition
    torch.testing.assert_close(s.truth[:, -1],
                               torch.tensor(ds.M.values[:, -1]).float())


# --------------------------------------------------------------------------- #
# Full integration: train -> save -> restore reproduces the forward pass
# --------------------------------------------------------------------------- #
def test_train_save_restore_roundtrip(tmp_path):
    torch.manual_seed(0)
    ds = _mock_ds()
    ds_path = tmp_path / "mock.nc"
    ds.to_netcdf(ds_path)

    sel = FeatureSelector([FeatureSpec("M", lag=1)], [FeatureSpec("forcing")],
                          FeatureSpec("M"), mode="parallel")
    res = sel.select(ds)
    fmean, fstd = compute_norm_stats(res.X.numpy())
    fmean, fstd = torch.tensor(fmean).float(), torch.tensor(fstd).float()
    tmean, tstd = res.y.mean(0), res.y.std(0)

    Xn = normalize(res.X, fmean, fstd)
    yn = normalize(res.y, tmean, tstd)
    loader = Data.DataLoader(Data.TensorDataset(Xn, yn), batch_size=8, shuffle=True)

    net = FCNN(input_size=len(res.feature_names), n_neurons_list=[8], activation=nn.Tanh())
    pr = ParallelPredictor(net, nn.MSELoss(), optim.Adam(net.parameters(), lr=1e-2),
                           device="cpu", n_ar=1, predict_tendency=False)
    for _ in pr.fit(loader, loader, n_epochs=5):
        pass

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    torch.save(net.state_dict(), run_dir / "weights.pt")
    cfg = _base_config(
        run_id="run", dataset_path=str(ds_path), val_dataset_path=str(ds_path),
        feature_names=res.feature_names,
        feature_mean=fmean.tolist(), feature_std=fstd.tolist(),
        target_mean=tmean.tolist(), target_std=tstd.tolist(),
        n_time=res.n_time, n_time_val=res.n_time, n_cases=res.n_cases, n_cases_val=res.n_cases,
    )
    cfg.save(str(run_dir))

    prediction = restore(str(run_dir))

    # restored prediction must equal a direct forward pass with the same weights
    net.eval()
    with torch.no_grad():
        direct = real_units(_step(net, Xn, 1, False), tmean, tstd).reshape(res.n_cases, res.n_time)
    torch.testing.assert_close(prediction.val.pred, direct, rtol=1e-5, atol=1e-6)
    assert prediction.val.truth.shape == (res.n_cases, res.n_time)
    assert prediction.val.features.shape == (res.n_cases, res.n_time, len(res.feature_names))
