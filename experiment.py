"""Experiment config, registry, and the Prediction object (TASK 2 + 3).

* ``PredictorConfig`` fully describes one reproducible run (data + feature spec +
  architecture + training + normalization stats + results). A single config
  covers both modes -- ``mode``/``predict_tendency``/``n_ar``/``k`` are just
  fields, so there is no separate autoregressive config.
* ``Registry`` lists/loads runs under a root directory for cross-run comparison.
* ``Prediction`` is the object every analysis/plot function consumes. It carries
  state truth+pred per split and exposes a tendency view (diff along time), so a
  neural predictor, a persistence baseline and ePBL are all the same type.
* Restoring a *mean* predictor (or any other NN baseline) is just ``restore()``
  on that run -- it is not a special case.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from typing import List, Optional

import numpy as np
import xarray as xr
import torch
from torch import nn

from metrics import normalize, real_units, invert_transform, compute_rmse
from model import FCNN, LinearRegression, _step, rollout
from data import FeatureSelector, FeatureSpec

CONFIG_FILE = "config.json"
WEIGHTS_FILE = "weights.pt"


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def _spec_dicts(specs):
    return [dict(name=s.name, lag=s.lag, transform=s.transform) for s in specs]


def _specs(dicts):
    return [FeatureSpec(**d) for d in dicts]


@dataclass
class PredictorConfig:
    run_id: str
    mode: str                       # "parallel" | "autoregressive"
    dataset_path: str
    val_dataset_path: str
    ar_features: List[dict]         # each: {name, lag, transform}
    forcings: List[dict]
    target: List[dict]
    feature_names: List[str]        # final, expanded, post zero-std drop
    network: str                    # "fcnn" | "linear"
    n_neurons_list: List[int]
    activation: str                 # "relu" | "tanh"
    n_ar: int
    predict_tendency: bool
    k: int
    learning_rate: float
    weight_decay: float
    loss: str
    n_epochs: int
    random_state: int
    train_batch_size: int
    test_batch_size: int
    feature_mean: List[float]
    feature_std: List[float]
    target_mean: List[float]
    target_std: List[float]
    train_loss_trajectory: List[float]
    test_loss_trajectory: List[float]
    n_time: int
    n_time_val: int
    n_cases: int
    n_cases_val: int
    rmse: float
    notes: str = ""
    target_tendency: bool = False   # train on dM, reconstruct state by cumulative sum

    def __post_init__(self):
        if self.target_tendency and self.predict_tendency:
            raise ValueError(
                "target_tendency (dM target, integrated afterwards) and "
                "predict_tendency (residual skip in _step) are mutually exclusive; "
                "set predict_tendency=False for a dM target."
            )

    # -- convenience -------------------------------------------------------- #
    @property
    def target_transform(self):
        return self.target[0].get("transform")

    @property
    def selector_mode(self):
        return "sequence" if self.mode == "autoregressive" else "parallel"

    def to_selector(self, mode=None):
        return FeatureSelector(
            ar_features=_specs(self.ar_features),
            forcings=_specs(self.forcings),
            target=_specs(self.target),
            mode=mode or self.selector_mode,
            drop_zero_std=False,     # columns already fixed at train time
            target_tendency=self.target_tendency,
        )

    def build_network(self):
        act = {"relu": nn.ReLU(), "tanh": nn.Tanh()}[self.activation]
        n_in = len(self.feature_names)
        n_out = len(self.target)
        if self.network == "fcnn":
            return FCNN(n_in, self.n_neurons_list, act, output_size=n_out)
        if self.network == "linear":
            return LinearRegression(n_in, n_out)
        raise ValueError(f"Unknown network {self.network!r}.")

    # -- IO ----------------------------------------------------------------- #
    def save(self, run_dir):
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, CONFIG_FILE), "w") as f:
            json.dump(asdict(self), f, indent=1)

    @classmethod
    def load(cls, run_dir):
        with open(os.path.join(run_dir, CONFIG_FILE)) as f:
            return cls(**json.load(f))


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
class Registry:
    """Lists and loads runs under a root directory (e.g. m_star_predictors/)."""

    SCALAR_FIELDS = ("run_id", "mode", "network", "activation", "predict_tendency",
                     "n_ar", "k", "learning_rate", "weight_decay", "n_epochs", "rmse")

    def __init__(self, root):
        self.root = root

    def run_ids(self):
        return sorted(d for d in os.listdir(self.root)
                      if os.path.exists(os.path.join(self.root, d, CONFIG_FILE)))

    def path(self, run_id):
        return os.path.join(self.root, run_id)

    def load(self, run_id):
        return PredictorConfig.load(self.path(run_id))

    def table(self):
        import pandas as pd
        rows = []
        for rid in self.run_ids():
            c = self.load(rid)
            rows.append({f: getattr(c, f) for f in self.SCALAR_FIELDS}
                        | {"features": ",".join(c.feature_names)})
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Prediction object
# --------------------------------------------------------------------------- #
@dataclass
class Split:
    truth: torch.Tensor            # (n_cases, n_time) state, real units
    pred: torch.Tensor             # (n_cases, n_time)
    n_time: int
    n_cases: int
    features: Optional[torch.Tensor] = None       # (n_cases, n_time, n_feat), real units
    feature_names: Optional[List[str]] = None
    case_meta: Optional[dict] = None

    def series(self, quantity="state"):
        """Return (truth, pred) as (n_cases, n_time[-1]) for the requested quantity."""
        if quantity == "state":
            return self.truth, self.pred
        if quantity == "tendency":
            return self.truth[:, 1:] - self.truth[:, :-1], self.pred[:, 1:] - self.pred[:, :-1]
        raise ValueError("quantity must be 'state' or 'tendency'.")

    def flat(self, quantity="state"):
        t, p = self.series(quantity)
        return p.reshape(-1), t.reshape(-1)

    def rmse(self, quantity="state"):
        p, t = self.flat(quantity)
        return compute_rmse(p, t).item()


@dataclass
class Prediction:
    label: str
    train: Optional[Split] = None
    val: Optional[Split] = None
    feature_names: Optional[List[str]] = None
    mode: str = "parallel"
    config: Optional[PredictorConfig] = None

    def split(self, name="val"):
        return getattr(self, name)


# --------------------------------------------------------------------------- #
# Restore a trained predictor
# --------------------------------------------------------------------------- #
def integrate_tendency(dM, m0):
    """Free-running state reconstruction: M[t] = m0 + sum_{j<t} dM[j].

    dM: (n_cases, n_time, n_tar) predicted increments; m0: (n_cases, 1, n_tar)
    initial state. Returns states (n_cases, n_time + 1, n_tar) -- errors in dM
    accumulate, so drift is visible.
    """
    zero = torch.zeros_like(dM[:, :1, :])
    return m0 + torch.cat([zero, torch.cumsum(dM, dim=1)], dim=1)


def _predict_split(cfg, path, net, device):
    ds = xr.open_dataset(path)
    res = cfg.to_selector().select(ds)
    fmean = torch.tensor(cfg.feature_mean)
    fstd = torch.tensor(cfg.feature_std)
    tmean = torch.tensor(cfg.target_mean)
    tstd = torch.tensor(cfg.target_std)

    Xn = normalize(res.X, fmean, fstd)
    with torch.no_grad():
        if cfg.mode == "autoregressive":
            out = rollout(net, Xn.to(device), cfg.n_ar, cfg.predict_tendency).cpu()
        else:
            out = _step(net, Xn.to(device), cfg.n_ar, cfg.predict_tendency).cpu()

    nc = res.n_cases
    if cfg.target_tendency:
        # network predicts dM (real units); integrate from the true initial state
        dM = real_units(out, tmean, tstd).reshape(nc, res.n_time, -1)
        state = res.y_state                                   # (nc, L, n_tar), real state
        pred = integrate_tendency(dM, state[:, :1, :])
        return Split(truth=state[..., 0], pred=pred[..., 0],
                     n_time=state.shape[1], n_cases=nc)

    pred = invert_transform(real_units(out, tmean, tstd), cfg.target_transform)
    truth = invert_transform(res.y, cfg.target_transform)

    nt = res.n_time
    return Split(truth=truth.reshape(nc, nt), pred=pred.reshape(nc, nt),
                 n_time=nt, n_cases=nc,
                 features=res.X.reshape(nc, nt, -1), feature_names=res.feature_names)


def restore(run_dir, device="cpu") -> Prediction:
    cfg = PredictorConfig.load(run_dir)
    net = cfg.build_network()
    net.load_state_dict(torch.load(os.path.join(run_dir, WEIGHTS_FILE),
                                   map_location=device))
    net.eval()
    return Prediction(
        label=cfg.run_id,
        train=_predict_split(cfg, cfg.dataset_path, net, device),
        val=_predict_split(cfg, cfg.val_dataset_path, net, device),
        feature_names=cfg.feature_names, mode=cfg.mode, config=cfg,
    )


# --------------------------------------------------------------------------- #
# Non-NN baselines (mean / other NN baselines are just restore() on their run)
# --------------------------------------------------------------------------- #
def _reshape_like(ds, values):
    nc, nt = ds.sizes["case"], ds.sizes["time"]
    return torch.tensor(np.asarray(values).reshape(nc, nt)).float()


def persistence_prediction(ds, target="M", label="persistence") -> Prediction:
    """Prediction that just repeats the previous value: pred[t] = truth[t-1]."""
    M = torch.tensor(ds[target].values).float()
    truth = M[:, 1:]
    pred = M[:, :-1]
    split = Split(truth=truth, pred=pred, n_time=truth.shape[1], n_cases=truth.shape[0])
    return Prediction(label=label, val=split)


def epbl_prediction(ds, neutral_mode="Nb", H="bl_tke", label="ePBL") -> Prediction:
    """Algebraic ePBL M from stored columns (see epbl_basic.calculate_M)."""
    from epbl_basic import calculate_M

    def col(name):
        da = ds[name]
        if "time" not in da.dims:                      # broadcast per-case constant
            da = da.expand_dims(time=ds.sizes["time"]).transpose("case", "time")
        return da.transpose("case", "time").values.reshape(-1)

    M_alg = calculate_M(H=col(H), f=col("f"), u_star=col("u_star"),
                        B=col("B"), wb=-col("wb"), neutral_mode=neutral_mode)
    split = Split(truth=_reshape_like(ds, ds["M"].transpose("case", "time").values),
                  pred=_reshape_like(ds, M_alg),
                  n_time=ds.sizes["time"], n_cases=ds.sizes["case"])
    return Prediction(label=label, val=split)
