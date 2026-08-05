"""Dataset compilation (TASK 1).

Two responsibilities, deliberately split:

* ``DatasetMaker`` reads raw GOTM output into an xarray ``Dataset`` on the
  canonical grid ``(case, time)`` for scalars and ``(case, time, depth)`` for
  profiles, carrying metadata in ``.attrs``, and saves it as netCDF. One maker
  per dataset *kind* (mean / transient / full-profile transient). Instantiating
  and calling these in ``make_datasets.ipynb`` is the tractable, appendable
  record of which datasets exist.

  Which variables a maker reads is *data*, not code: each is a ``VarSpec``, and
  every maker takes ``extra_vars=`` to append more without subclassing. See
  "Naming conventions" below and the per-class docstrings.

* ``FeatureSelector`` turns a stored Dataset into training tensors. Features and
  targets are chosen by name and *lagged along the time axis* (``lag=+1`` gives
  M_i-1, ``lag=-1`` gives M_i+1) -- so lagged quantities are derived, never
  stored. Columns are ordered ``[ar..., forcing...]`` (the first ``n_ar`` are the
  autoregressive state channels the model feeds back). Profiles expand to one
  column per depth. The result is the canonical ``(case, time, feature)`` tensor,
  flattened to ``(sample, feature)`` for parallel mode.

Naming conventions
------------------
Three distinct names are involved in reading one raw quantity, and they are
*not* interchangeable -- a ``VarSpec`` carries all three:

======================  =========================================  ==================
``VarSpec`` field       meaning                                    example
======================  =========================================  ==================
``suffix``              file, read as ``{dataset_name}_{suffix}``  ``SS_bl_avg.nc``
``key``                 variable name *inside* that file           ``SS_bl_mean``
``out``                 name in the assembled Dataset              ``SS_bl``
======================  =========================================  ==================

``out`` defaults to ``key``. Profile suffixes may contain a ``{method}``
placeholder, filled in from the maker's ``method`` argument.

Derived (never read, always computed) quantities follow ``{level}_{quantity}``,
where ``level`` is ``surf`` (surface) or ``bl`` (boundary-layer average):

* ``{level}_work``            -- ``tau * |u_vec|``, the rate wind stress does
  work against the current [W m-2]. (Was ``surf_mag``, which named a work rate
  as if it were a magnitude.)
* ``{level}_work_along_tau``  -- ``|tau * u|``, that work rate projected on the
  stress direction [W m-2]. (Was ``surf_ekin``; it is not a kinetic energy, and
  it is algebraically determined by the other two -- expect collinearity.)
* ``{level}_veer``            -- angle between the current and the stress [rad].
  Unsigned in ``[0, pi]`` by default; ``signed_veer=True`` uses ``arctan2`` and
  keeps the sign of ``v``. (Was ``surf_angle``.)

``tau`` is the x-directed stress ``tx``, so "angle to the x-axis" and "angle to
the stress" are the same angle.
"""
from __future__ import annotations

import os
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

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


def velocity_work_terms(
        u: np.ndarray,
        v: np.ndarray,
        tau: np.ndarray,
        prefix: str,
        signed_veer: bool = False,
    ) -> dict:
    """Stress-work terms for one velocity pair, named ``{prefix}_{quantity}``.

    See "Naming conventions" in the module docstring. ``tau`` is x-directed, so
    the veer angle is measured from the +x axis. Note ``work_along_tau`` is
    exactly ``|tau * u|``: the older formulation reached it via
    ``|work * cos(veer)|``, in which ``tau`` cancelled, leaving only a 0/0
    hazard where ``tau == 0``.
    """
    u, v, tau = np.asarray(u), np.asarray(v), np.asarray(tau)
    speed = np.sqrt(u ** 2 + v ** 2)
    veer = (np.arctan2(v, u) if signed_veer
            else np.arccos(np.divide(u, speed, out=np.zeros_like(speed), where=speed > 0)))
    return {
        f"{prefix}_work": tau * speed,
        f"{prefix}_veer": np.nan_to_num(veer),
        f"{prefix}_work_along_tau": np.abs(tau * u),
    }


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
# Variable specifications -- what a maker reads, as data rather than code
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class VarSpec:
    """One raw file -> one variable in the assembled Dataset.

    suffix   : file suffix; the file read is ``{dataset_name}_{suffix}``. May
               contain a ``{method}`` placeholder (profile datasets).
    key      : variable name inside that file.
    out      : name in the assembled Dataset (defaults to ``key``).
    required : a missing file raises ``FileNotFoundError``. Optional specs are
               skipped with a warning and recorded in ``ds.attrs['vars_skipped']``.
    post     : ``callable(xr.Dataset) -> array``, for quantities that are not a
               plain read (sign flips, fill values). Overrides ``key`` lookup.
    """
    suffix: str
    key: str
    out: Optional[str] = None
    required: bool = False
    post: Optional[Callable] = None

    @property
    def name(self) -> str:
        return self.out or self.key


def _varspec(s) -> VarSpec:
    """Coerce VarSpec | (suffix, key[, out]) | {..} -> VarSpec."""
    if isinstance(s, VarSpec):
        return s
    if isinstance(s, dict):
        return VarSpec(**s)
    return VarSpec(*s)


def _bl_tke(raw: xr.Dataset) -> np.ndarray:
    """TKE-derived boundary-layer depth: sign-flipped, non-finite filled with -1."""
    return -raw.where(np.isfinite(raw.bl), -1.0).bl.values


# The state variables that *define* a transient m* dataset. Absent -> error.
CORE_TRANSIENT_VARS: Tuple[VarSpec, ...] = (
    VarSpec("M.nc", "M", required=True),
    VarSpec("bl.nc", "bl", "bl_rh18", required=True),
    VarSpec("tke_bl.nc", "bl", "bl_tke", required=True, post=_bl_tke),
)

# Candidate features: read when present, skipped with a warning when not.
# Move a line into CORE_TRANSIENT_VARS (or pass required=True) to insist on it.
OPTIONAL_TRANSIENT_VARS: Tuple[VarSpec, ...] = (
    VarSpec("SS_bl_avg.nc", "SS_bl_mean", "SS_bl"),
    VarSpec("NN_bl_avg.nc", "NN_bl_mean", "NN_bl"),
    VarSpec("wb_dataset.nc", "wb"),
    VarSpec("u_bl.nc", "u_bl"),
    VarSpec("v_bl.nc", "v_bl"),
    VarSpec("u_surf.nc", "u_surf"),
    VarSpec("v_surf.nc", "v_surf"),
)

# (level, u name, v name). Derived terms are added only when both are present.
VELOCITY_PAIRS: Tuple[Tuple[str, str, str], ...] = (
    ("surf", "u_surf", "v_surf"),
    ("bl", "u_bl", "v_bl"),
)

# Profiles on the common sigma grid. Only method='interp' is currently valid,
# but the raw filenames still carry the method, so it stays in the pattern.
CORE_PROFILE_VARS: Tuple[VarSpec, ...] = (
    VarSpec("S2_sigma_{method}.nc", "SS", required=True),
    VarSpec("N2_sigma_{method}.nc", "NN", required=True),
    VarSpec("tke_sigma_{method}.nc", "tke"),
    VarSpec("eps_sigma_{method}.nc", "eps"),
)

CORE_MEAN_VARS: Tuple[VarSpec, ...] = (
    VarSpec("mean_Ms.npy", "M", required=True),
    VarSpec("mean_bls.npy", "bl_rh18", required=True),
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
        if arr.ndim not in (1, 2):
            raise ValueError(
                f"scalar {name!r} has {arr.ndim} dims; expected (case,) or "
                f"(case, time). Pass 3-D (case, time, depth) arrays as profiles."
            )
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
    """Reads raw GOTM output for one dataset into a metadata-carrying Dataset.

    Subclasses declare a core ``VarSpec`` table; callers append to it with
    ``extra_vars=`` or replace it wholesale with ``vars=``. Reading is uniform:
    required specs must exist, optional ones are skipped with a warning, and
    every variable read is recorded in ``ds.attrs['sources']``.
    """

    kind = "abstract"
    core_vars: Tuple[VarSpec, ...] = ()

    def __init__(self, data_dir: str, dataset_name: str,
                 vars: Optional[Sequence] = None, extra_vars: Sequence = ()):
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        base = self.core_vars if vars is None else vars
        self.vars = [_varspec(s) for s in (*base, *extra_vars)]
        dupes = {s.name for s in self.vars if
                 sum(o.name == s.name for o in self.vars) > 1}
        if dupes:
            raise ValueError(f"Duplicate output names in var specs: {sorted(dupes)}")

    def _path(self, suffix):
        return os.path.join(self.data_dir, f"{self.dataset_name}_{suffix}")

    def _cases(self) -> dict:
        import json
        with open(self._path("training_set_cases.json")) as f:
            return json.load(f)

    def _read(self, specs, fmt=None, depth_coord=None):
        """Read specs -> (data, depths, skipped, sources).

        ``depths`` is empty unless ``depth_coord`` is given, in which case it
        maps variable name -> that file's depth coordinate.
        """
        data, depths, skipped, sources = {}, {}, [], []
        for s in specs:
            suffix = s.suffix.format(**(fmt or {}))
            path = self._path(suffix)
            if not os.path.exists(path):
                if s.required:
                    raise FileNotFoundError(
                        f"{self.kind} dataset {self.dataset_name!r} requires "
                        f"{s.name!r} from {suffix!r}; expected at {path}"
                    )
                skipped.append(s.name)
                continue
            if suffix.endswith(".npy"):
                data[s.name] = np.asarray(np.load(path))
            else:
                raw = xr.open_dataset(path)
                if s.post is None and s.key not in raw:
                    raise KeyError(
                        f"{path}: no variable {s.key!r} (found {list(raw.data_vars)})"
                    )
                data[s.name] = np.asarray(s.post(raw) if s.post else raw[s.key].values)
                if depth_coord is not None:
                    if depth_coord not in raw.coords and depth_coord not in raw:
                        raise KeyError(
                            f"{path}: no depth coordinate {depth_coord!r} "
                            f"(found {list(raw.coords)})"
                        )
                    depths[s.name] = np.asarray(raw[depth_coord].values)
            sources.append(f"{s.name}<-{suffix}:{s.key}")
        if skipped:
            warnings.warn(
                f"{self.dataset_name}: optional variables not found, skipped: {skipped}"
            )
        return data, depths, skipped, sources

    @staticmethod
    def _check_shapes(data, ref="M"):
        """Every variable must agree with the reference on (case, time)."""
        want = data[ref].shape[:2]
        for name, arr in data.items():
            if arr.shape[:2] != want:
                raise ValueError(
                    f"{name!r} has shape {arr.shape}, incompatible with {ref!r} "
                    f"{data[ref].shape} on the (case, time) axes."
                )

    @staticmethod
    def _check_depths(depths):
        """All profiles must share one depth axis (assemble_dataset has one)."""
        if not depths:
            return None
        ref_name, ref = next(iter(depths.items()))
        for name, d in depths.items():
            if d.shape != ref.shape or not np.allclose(d, ref):
                raise ValueError(
                    f"Profile {name!r} has a depth axis of length {d.shape[0]} "
                    f"that does not match {ref_name!r} (length {ref.shape[0]}). "
                    f"All profiles must be on a common grid."
                )
        return ref

    @abstractmethod
    def build(self) -> xr.Dataset:
        """Return the assembled Dataset (subclass reads its raw files here)."""

    def _base_attrs(self, skipped=(), sources=()):
        return dict(dataset_name=self.dataset_name, kind=self.kind,
                    rho0=RHO0, g=G, alpha=ALPHA, cp=CP,
                    vars_skipped=",".join(skipped), sources="; ".join(sources))

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
    (M_i-1, M_i+1, ...) are NOT stored -- FeatureSelector derives them.

    Variables come from ``CORE_TRANSIENT_VARS`` (required) plus
    ``OPTIONAL_TRANSIENT_VARS`` (read when the file exists, skipped with a
    warning otherwise), and each is named per the module-docstring convention.
    Add your own without subclassing::

        TransientDatasetMaker(data_dir, name,
                              extra_vars=[("my_diag.nc", "raw_key", "my_diag")])

    Derived stress-work terms (``surf_work``, ``surf_veer``,
    ``surf_work_along_tau`` and the ``bl_*`` equivalents) are added per entry in
    ``VELOCITY_PAIRS``, but only where both velocity components were read.
    Per-case constants f, u_star and B are always present; FeatureSelector
    broadcasts them along time.
    """

    kind = "transient"
    core_vars = CORE_TRANSIENT_VARS + OPTIONAL_TRANSIENT_VARS

    def __init__(self, data_dir, dataset_name, vars=None, extra_vars=(),
                 signed_veer=False):
        super().__init__(data_dir, dataset_name, vars, extra_vars)
        self.signed_veer = signed_veer

    def build(self) -> xr.Dataset:
        forc = case_forcings(self._cases())
        scalars, _, skipped, sources = self._read(self.vars)
        self._check_shapes(scalars)
        n_cases, n_time = scalars["M"].shape

        tau = np.repeat(forc["tau"][:, None], n_time, axis=1)
        for level, u_name, v_name in VELOCITY_PAIRS:
            if u_name in scalars and v_name in scalars:
                scalars.update(velocity_work_terms(
                    scalars[u_name], scalars[v_name], tau, level, self.signed_veer))

        # per-case constants (case,) -- broadcast by FeatureSelector
        scalars.update(f=forc["f"], u_star=forc["u_star"], B=forc["B"])
        return assemble_dataset(scalars, attrs=dict(
            **self._base_attrs(skipped, sources), n_cases=n_cases, n_time=n_time))


class ProfileTransientDatasetMaker(DatasetMaker):
    """(case, time) scalars + (case, time, depth) profiles (see
    compile_features_N2_S2_profiles.ipynb).

    Any set of profiles may be passed via ``profile_vars=`` / ``extra_profiles=``;
    scalars work exactly as in TransientDatasetMaker. Every profile must lie on
    the *same* depth axis -- ``assemble_dataset`` carries a single depth
    coordinate -- and a mismatch raises rather than silently taking whichever
    file was read last. The resolved axis length is recorded as ``n_depth``.

    ``method`` selects the vertical treatment and is substituted into the
    ``{method}`` placeholder in profile suffixes, because the raw filenames
    carry it (``S2_sigma_interp.nc``). Only ``interp`` -- everything on a sigma
    coordinate within the boundary layer -- is currently valid; the other
    historical values (``raw``, ``shift``) are not.

    Cases with ``tau <= tau_min`` are dropped from every variable.
    """

    kind = "profile_transient"
    core_vars = CORE_TRANSIENT_VARS + (
        VarSpec("wb_dataset.nc", "wb"),
    )
    core_profiles = CORE_PROFILE_VARS
    valid_methods = ("interp",)

    def __init__(self, data_dir, dataset_name, method="interp", tau_min=0.005,
                 vars=None, extra_vars=(), profile_vars=None, extra_profiles=(),
                 depth_coord="sigma_depth"):
        super().__init__(data_dir, dataset_name, vars, extra_vars)
        if method not in self.valid_methods:
            warnings.warn(
                f"method={method!r} is not among the currently valid methods "
                f"{self.valid_methods}; profiles may not share a depth grid."
            )
        self.method = method
        self.tau_min = tau_min
        self.depth_coord = depth_coord
        base = self.core_profiles if profile_vars is None else profile_vars
        self.profile_vars = [_varspec(s) for s in (*base, *extra_profiles)]

    def build(self) -> xr.Dataset:
        forc = case_forcings(self._cases())
        mask = forc["tau"] > self.tau_min
        fmt = dict(method=self.method)

        scalars, _, skipped, sources = self._read(self.vars, fmt)
        profiles, depths, p_skipped, p_sources = self._read(
            self.profile_vars, fmt, depth_coord=self.depth_coord)
        self._check_shapes({**scalars, **profiles})
        depth = self._check_depths(depths)

        scalars = {k: v[mask] for k, v in scalars.items()}
        profiles = {k: v[mask] for k, v in profiles.items()}
        n_cases, n_time = scalars["M"].shape
        scalars.update(f=forc["f"][mask], u_star=forc["u_star"][mask],
                       B=forc["B"][mask])
        return assemble_dataset(scalars, profiles, depth, attrs=dict(
            **self._base_attrs([*skipped, *p_skipped], [*sources, *p_sources]),
            method=self.method, tau_min=self.tau_min, n_cases=n_cases,
            n_time=n_time, n_depth=0 if depth is None else len(depth)))


class MeanDatasetMaker(DatasetMaker):
    """One value per case: time-mean quantities (mean_vals=True path).

    Reads ``.npy`` rather than netCDF, so a spec's ``key`` is only a label --
    the filename alone identifies the array. Otherwise the ``vars=`` /
    ``extra_vars=`` contract is the same as the other makers.
    """

    kind = "mean"
    core_vars = CORE_MEAN_VARS

    def build(self) -> xr.Dataset:
        forc = case_forcings(self._cases())
        scalars, _, skipped, sources = self._read(self.vars)
        self._check_shapes(scalars)
        scalars.update(f=forc["f"], u_star=forc["u_star"], B=forc["B"])
        return assemble_dataset(scalars, attrs=dict(
            **self._base_attrs(skipped, sources), n_cases=len(forc["f"])))


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
    y_state: Optional["torch.Tensor"] = None   # aligned state trajectory (tendency targets)


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
    target_tendency : train on the increment dM = M_{i+1} - M_i instead of the
                    state. The target is the next state (same convention as AR, e.g.
                    FeatureSpec("M", lag=-1)); dM is target(lag) - target(lag+1). The
                    state is reconstructed later by cumulative sum (see
                    experiment.restore / integrate_tendency). ``y_state`` on the
                    result carries the state trajectory (real units, length n_time+1)
                    for that integration.
    """

    def __init__(self, ar_features, forcings, target,
                 mode="parallel", drop_zero_std=True, strict_nan=True,
                 target_tendency=False):
        self.ar = [_spec(s) for s in ar_features]
        self.forcings = [_spec(s) for s in forcings]
        self.targets = [_spec(s) for s in (target if isinstance(target, (list, tuple)) else [target])]
        self.mode = mode
        self.drop_zero_std = drop_zero_std
        self.strict_nan = strict_nan
        self.target_tendency = target_tendency

    # -- time-lag window shared by features and targets --------------------- #
    def _window(self, n_time):
        lags = [s.lag for s in self.ar + self.forcings + self.targets]
        if self.target_tendency:
            lags += [t.lag + 1 for t in self.targets]   # previous state for dM = y(lag) - y(lag+1)
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
        Y = np.stack([c[1] for c in target_cols], axis=-1)      # (case, n_time, n_tar)
        feat_names = [c[0] for c in feat_cols]
        target_names = [c[0] for c in target_cols]

        y_state = None
        if self.target_tendency:
            # Y (target at its lag) is the next state, e.g. M_{i+1} for lag=-1.
            # The current state is the same variable one step back (lag+1); their
            # difference is the increment dM_i, aligned to the inputs at time i.
            prev_cols = self._columns(ds, [replace(t, lag=t.lag + 1) for t in self.targets], lo, hi)
            prev = np.stack([c[1] for c in prev_cols], axis=-1)  # current state (case, L, n_tar)
            y_state = np.concatenate([prev, Y[:, -1:, :]], axis=1)  # states M_lo..M_hi (L+1)
            Y = Y - prev                                            # dM (case, L, n_tar)
            target_names = [f"d{n}" for n in target_names]

        X, feat_names, n_ar, dropped = self._validate(X, feat_names, n_ar, Y, target_names)

        n_time = X.shape[1]
        if self.mode == "parallel":
            X = X.reshape(-1, X.shape[-1])
            Y = Y.reshape(-1, Y.shape[-1])

        return SelectionResult(
            X=torch.tensor(X).float(), y=torch.tensor(Y).float(),
            feature_names=feat_names, target_names=target_names,
            n_ar=n_ar, n_time=n_time, n_cases=n_cases, mode=self.mode, dropped=dropped,
            y_state=None if y_state is None else torch.tensor(y_state).float(),
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
