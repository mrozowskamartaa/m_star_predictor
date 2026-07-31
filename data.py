"""Dataset compilation (TASK 1).

Two responsibilities, deliberately split:

* ``DatasetMaker`` reads raw GOTM output into an xarray ``Dataset`` on the
  canonical grid ``(case, time)`` for scalars and ``(case, time, depth)`` for
  profiles, carrying metadata in ``.attrs``, and saves it as netCDF. One maker
  per dataset *kind* (mean / transient / full-profile transient). Instantiating
  and calling these in ``make_datasets.ipynb`` is the tractable, appendable
  record of which datasets exist.

* ``FeatureSelector`` turns a stored Dataset into training tensors. Features and
  targets are chosen by name and *lagged along the time axis* (``lag=+1`` gives
  M_i-1, ``lag=-1`` gives M_i+1) -- so lagged quantities are derived, never
  stored. Columns are ordered ``[ar..., forcing...]`` (the first ``n_ar`` are the
  autoregressive state channels the model feeds back). Profiles expand to one
  column per depth. The result is the canonical ``(case, time, feature)`` tensor,
  flattened to ``(sample, feature)`` for parallel mode.
"""
from __future__ import annotations

import os
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import numpy as np
import xarray as xr

from metrics import apply_transform

try:
    import torch
except ImportError:
    torch = None


# --------------------------------------------------------------------------- #
# Physics helpers (previously duplicated in every compile_features notebook)
# --------------------------------------------------------------------------- #
OMEGA = 2 * np.pi / 24 / 60 / 60
RHO0, G, ALPHA, CP = 1027.0, 9.81, 2e-4, 4e3


def calculate_f(latitude: float) -> float:
    return 2 * OMEGA * np.sin(np.pi * np.asarray(latitude) / 180)


def calculate_T(latitude: float) -> float:
    return 2 * np.pi / calculate_f(latitude)


def compute_u_star(
        tau: float,
        rho: float = RHO0
    ) -> float:
    return (np.asarray(tau) / rho) ** 0.5


def buoyancy_flux(
        heat_flux: Union[np.ndarray, float], 
        rho: float = RHO0, 
        g: float = G, 
        alpha: float = ALPHA, 
        cp: float = CP
    ) -> Union[np.ndarray, float]:
    return g * alpha * np.asarray(heat_flux) / (rho * cp)


def case_forcings(case_dict: dict) -> dict:
    """Per-case scalar forcings from a GOTM training-set case dictionary."""
    hf = np.array([c["heat_flux"] for c in case_dict.values()])
    tau = np.array([c["tx"] for c in case_dict.values()])
    lat = np.array([c["lat"] for c in case_dict.values()])
    return dict(
        heat_flux=hf, tau=tau, lat=lat,
        f=calculate_f(lat), u_star=compute_u_star(tau), B=buoyancy_flux(hf),
    )


# --------------------------------------------------------------------------- #
# Dataset assembly / IO
# --------------------------------------------------------------------------- #
def assemble_dataset(
        scalars: dict, 
        profiles: dict = None, 
        depth: np.ndarray = None, 
        attrs: dict = None
    ) -> xr.Dataset:
    """Build an xr.Dataset from named arrays.

    scalars:  name -> array of shape (case,) or (case, time)
    profiles: name -> array of shape (case, time, depth)
    depth:    1-D depth coordinate (required iff profiles given)
    attrs:    metadata
    """
    data_vars = {}
    for name, arr in scalars.items():
        arr = np.asarray(arr)
        dims = ("case",) if arr.ndim == 1 else ("case", "time")
        data_vars[name] = (dims, arr)
    if profiles:
        if depth is None:
            raise ValueError("depth coordinate required when profiles are given.")
        for name, arr in profiles.items():
            data_vars[name] = (("case", "time", "depth"), np.asarray(arr))
    ds = xr.Dataset(data_vars)
    if depth is not None:
        ds = ds.assign_coords(depth=depth)
    ds.attrs.update(attrs or {})
    return ds


class DatasetMaker(ABC):
    """Reads raw GOTM output for one dataset into a metadata-carrying Dataset."""

    kind = "abstract"

    def __init__(self, data_dir: str, dataset_name: str):
        self.data_dir = data_dir
        self.dataset_name = dataset_name

    def _path(self, suffix):
        return os.path.join(self.data_dir, f"{self.dataset_name}_{suffix}")

    @abstractmethod
    def build(self) -> xr.Dataset:
        """Return the assembled Dataset (subclass reads its raw files here)."""

    def _base_attrs(self):
        return dict(dataset_name=self.dataset_name, kind=self.kind,
                    rho0=RHO0, g=G, alpha=ALPHA, cp=CP)

    def save(self, out_dir: Optional[str] = None) -> str:
        ds = self.build()
        out_dir = out_dir or self.data_dir
        path = os.path.join(out_dir, f"{self.dataset_name}_{self.kind}.nc")
        ds.to_netcdf(path)
        return path

    @staticmethod
    def load(path: str) -> xr.Dataset:
        return xr.open_dataset(path)


class TransientDatasetMaker(DatasetMaker):
    """(case, time) scalar dataset -- the standard transient case (see
    compile_features.ipynb). Stores the state M and forcings; lagged quantities
    (M_i-1, M_i+1, ...) are NOT stored -- FeatureSelector derives them."""

    kind = "transient"

    def build(self) -> xr.Dataset:
        import json
        with open(self._path("training_set_cases.json")) as f:
            cases = json.load(f)
        forc = case_forcings(cases)

        def var(suffix, key):
            return xr.open_dataset(self._path(suffix))[key].values

        M = var("M.nc", "M")
        n_cases, n_time = M.shape
        bl_tke_raw = xr.open_dataset(self._path("tke_bl.nc"))
        bl_tke = -bl_tke_raw.where(np.isfinite(bl_tke_raw.bl), -1.0).bl.values
        u = var("u_surf.nc", "u_surf")
        v = var("v_surf.nc", "v_surf")
        tau = np.repeat(forc["tau"][:, None], n_time, axis=1)
        surf_mag = tau * np.sqrt(u ** 2 + v ** 2)
        surf_angle = np.nan_to_num(np.arccos(u * tau / (np.sqrt(u ** 2 + v ** 2) * tau)))
        surf_ekin = np.abs(surf_mag * np.cos(surf_angle))

        scalars = dict(
            M=M, bl_rh18=var("bl.nc", "bl"), bl_tke=bl_tke,
            u_bl=var("u_bl.nc", "u_bl"), v_bl=var("v_bl.nc", "v_bl"),
            SS_bl=var("SS_bl_avg.nc", "SS_bl_mean"),
            NN_bl=var("NN_bl_avg.nc", "NN_bl_mean"),
            wb=var("wb_dataset.nc", "wb"),
            surf_mag=surf_mag, surf_angle=surf_angle, surf_ekin=surf_ekin,
            # per-case constants (case,) -- broadcast by FeatureSelector
            f=forc["f"], u_star=forc["u_star"], B=forc["B"],
        )
        return assemble_dataset(scalars, attrs=dict(**self._base_attrs(),
                                                    n_cases=n_cases, n_time=n_time))


class ProfileTransientDatasetMaker(DatasetMaker):
    """(case, time) scalars + (case, time, depth) profiles (see
    compile_features_N2_S2_profiles.ipynb)."""

    kind = "profile_transient"

    def __init__(self, data_dir, dataset_name, method="interp", tau_min=0.005):
        super().__init__(data_dir, dataset_name)
        self.method = method
        self.tau_min = tau_min

    def build(self) -> xr.Dataset:
        import json
        with open(self._path("training_set_cases.json")) as f:
            cases = json.load(f)
        forc = case_forcings(cases)
        mask = forc["tau"] > self.tau_min

        M = xr.open_dataset(self._path("M.nc")).M.values[mask]
        n_cases, n_time = M.shape
        bl_tke_raw = xr.open_dataset(self._path("tke_bl.nc"))
        bl_tke = -bl_tke_raw.where(np.isfinite(bl_tke_raw.bl), -1.0).bl.values[mask]

        scalars = dict(
            M=M, bl_rh18=xr.open_dataset(self._path("bl.nc")).bl.values[mask],
            bl_tke=bl_tke, wb=xr.open_dataset(self._path("wb_dataset.nc")).wb.values[mask],
            f=forc["f"][mask], u_star=forc["u_star"][mask], B=forc["B"][mask],
        )
        profiles, depth = {}, None
        for suffix, key, out in [(f"S2_sigma_{self.method}.nc", "SS", "SS"),
                                 (f"N2_sigma_{self.method}.nc", "NN", "NN"),
                                 (f"tke_sigma_{self.method}.nc", "tke", "tke"),
                                 (f"eps_sigma_{self.method}.nc", "eps", "eps")]:
            pd_ = xr.open_dataset(self._path(suffix))
            profiles[out] = pd_[key].values[mask]
            depth = pd_.sigma_depth.values
        return assemble_dataset(scalars, profiles, depth,
                                attrs=dict(**self._base_attrs(), method=self.method,
                                           n_cases=n_cases, n_time=n_time))


class MeanDatasetMaker(DatasetMaker):
    """One value per case: time-mean quantities (mean_vals=True path)."""

    kind = "mean"

    def build(self) -> xr.Dataset:
        import json
        with open(self._path("training_set_cases.json")) as f:
            cases = json.load(f)
        forc = case_forcings(cases)
        scalars = dict(
            M=np.load(self._path("mean_Ms.npy")),
            bl_rh18=np.load(self._path("mean_bls.npy")),
            f=forc["f"], u_star=forc["u_star"], B=forc["B"],
        )
        return assemble_dataset(scalars, attrs=dict(**self._base_attrs(),
                                                    n_cases=len(forc["f"])))


# --------------------------------------------------------------------------- #
# Feature selection
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FeatureSpec:
    """A model input/target column: a variable, a time lag, an optional transform.

    lag > 0 is a past value (lag=1 -> x_i-1); lag < 0 is a future value
    (lag=-1 -> x_i+1); lag = 0 is the value at the target time.
    """
    name: str
    lag: int = 0
    transform: Optional[str] = None

    @property
    def label(self) -> str:
        if self.lag > 0:
            return f"{self.name}_i-{self.lag}"
        if self.lag < 0:
            return f"{self.name}_i+{-self.lag}"
        return self.name


def _spec(s):
    return s if isinstance(s, FeatureSpec) else FeatureSpec(s)


@dataclass
class SelectionResult:
    X: "torch.Tensor"
    y: "torch.Tensor"
    feature_names: list
    target_names: list
    n_ar: int
    n_time: int
    n_cases: int
    mode: str
    dropped: list = field(default_factory=list)


class FeatureSelector:
    """Select, lag, transform, validate and shape features/targets from a Dataset.

    ar_features   : autoregressive state channels (fed back during rollout), placed
                    first. For a next-step model use e.g. FeatureSpec("M", lag=1).
    forcings      : remaining input columns.
    target        : FeatureSpec (or list). For a next-step target, lag=-1.
    mode          : "parallel" -> (sample, feature); "autoregressive"/"sequence"
                    -> (case, time, feature).
    drop_zero_std : drop non-ar columns with zero variance (they can't be
                    normalized); a zero-variance ar/target column is an error.
    """

    def __init__(self, ar_features, forcings, target,
                 mode="parallel", drop_zero_std=True, strict_nan=True):
        self.ar = [_spec(s) for s in ar_features]
        self.forcings = [_spec(s) for s in forcings]
        self.targets = [_spec(s) for s in (target if isinstance(target, (list, tuple)) else [target])]
        self.mode = mode
        self.drop_zero_std = drop_zero_std
        self.strict_nan = strict_nan

    # -- time-lag window shared by features and targets --------------------- #
    def _window(self, n_time):
        lags = [s.lag for s in self.ar + self.forcings + self.targets]
        lo = max([0] + [l for l in lags if l > 0])
        hi = n_time + min([0] + [l for l in lags if l < 0])
        if hi <= lo:
            raise ValueError("Lag choices leave no valid timesteps.")
        return lo, hi

    def _columns(self, ds, specs, lo, hi):
        """Return list of (label, array(case, n_time)) for the given specs."""
        cols = []
        for s in specs:
            da = ds[s.name]
            has_time = "time" in da.dims
            if s.lag and not has_time:
                warnings.warn(f"lag={s.lag} ignored for time-less variable {s.name!r}.")
            if has_time and s.lag:
                da = da.shift(time=s.lag)
            if has_time:
                da = da.transpose("case", "time", ...).isel(time=slice(lo, hi))
                arr = da.values
            else:
                # per-case constant: broadcast across the trimmed window
                arr = np.repeat(np.asarray(da.transpose("case", ...).values)[:, None],
                                hi - lo, axis=1)
            arr = apply_transform(arr, s.transform)
            if arr.ndim == 3:                                   # profile -> depth cols
                depth = ds[ds[s.name].dims[-1]].values
                for d in range(arr.shape[-1]):
                    cols.append((f"{s.label}_{depth[d]:.4g}", arr[:, :, d]))
            else:
                cols.append((s.label, arr))
        return cols

    def select(self, ds) -> SelectionResult:
        if torch is None:
            raise ImportError("torch is required to build training tensors.")
        n_time_full = ds.sizes["time"] if "time" in ds.sizes else 1
        n_cases = ds.sizes["case"]
        lo, hi = self._window(n_time_full)

        ar_cols = self._columns(ds, self.ar, lo, hi)
        forcing_cols = self._columns(ds, self.forcings, lo, hi)
        target_cols = self._columns(ds, self.targets, lo, hi)
        n_ar = len(ar_cols)

        feat_cols = ar_cols + forcing_cols
        X = np.stack([c[1] for c in feat_cols], axis=-1)        # (case, n_time, n_feat)
        Y = np.stack([c[1] for c in target_cols], axis=-1)
        feat_names = [c[0] for c in feat_cols]
        target_names = [c[0] for c in target_cols]

        X, feat_names, n_ar, dropped = self._validate(X, feat_names, n_ar, Y, target_names)

        n_time = hi - lo
        if self.mode == "parallel":
            X = X.reshape(-1, X.shape[-1])
            Y = Y.reshape(-1, Y.shape[-1])

        return SelectionResult(
            X=torch.tensor(X).float(), y=torch.tensor(Y).float(),
            feature_names=feat_names, target_names=target_names,
            n_ar=n_ar, n_time=n_time, n_cases=n_cases, mode=self.mode, dropped=dropped,
        )

    def _validate(self, X, names, n_ar, Y, target_names):
        flat = X.reshape(-1, X.shape[-1])
        # NaN / Inf
        bad = ~np.isfinite(flat)
        if bad.any():
            cols = [names[i] for i in np.where(bad.any(axis=0))[0]]
            msg = f"Non-finite values in feature columns: {cols}"
            if self.strict_nan:
                raise ValueError(msg)
            warnings.warn(msg)
        if not np.all(np.isfinite(Y)):
            tcols = [target_names[i] for i in np.where(~np.isfinite(Y).reshape(-1, Y.shape[-1]).all(axis=0))[0]]
            raise ValueError(f"Non-finite values in target columns: {tcols}")
        # zero variance
        std = flat.std(axis=0)
        zero = np.where(std == 0)[0]
        dropped = []
        if len(zero):
            for i in zero:
                if i < n_ar:
                    raise ValueError(f"Autoregressive column {names[i]!r} has zero variance.")
            if self.drop_zero_std:
                keep = [i for i in range(len(names)) if i not in zero]
                dropped = [names[i] for i in zero]
                warnings.warn(f"Dropping zero-variance columns: {dropped}")
                X = X[..., keep]
                names = [names[i] for i in keep]
            else:
                warnings.warn(f"Zero-variance columns kept: {[names[i] for i in zero]}")
        return X, names, n_ar, dropped


def compute_norm_stats(X):
    """Per-feature mean/std over the sample axis (train stats; reuse for val)."""
    flat = X.reshape(-1, X.shape[-1])
    return flat.mean(0), flat.std(0)
