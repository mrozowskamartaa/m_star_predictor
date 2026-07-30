"""Shared numeric primitives: normalization, feature/target transforms, RMSE.

Everything here is backend-agnostic: functions dispatch on the argument type so
they work on both numpy arrays and torch tensors. Features are typically numpy
(straight out of xarray); predictions are typically torch tensors.
"""
import numpy as np

try:
    import torch
except ImportError:  # torch is optional for the pure-numpy paths
    torch = None


# "log1p" is the safe log for zero-valued features (log(1 + x)); it replaces the
# old bare "log"/"log10" applied to data containing zeros.
TRANSFORMS = ("sqrt", "log10", "log1p")


def _lib(x):
    """Return the array library (torch or numpy) matching x."""
    if torch is not None and torch.is_tensor(x):
        return torch
    return np


def apply_transform(x, kind):
    """Forward feature/target transform. kind None/"" is the identity."""
    if kind in (None, "", "none"):
        return x
    lib = _lib(x)
    if kind == "sqrt":
        return lib.sqrt(x)
    if kind == "log10":
        return lib.log10(x)
    if kind == "log1p":
        return lib.log1p(x)
    raise ValueError(f"Unknown transform {kind!r}; expected one of {TRANSFORMS}.")


def invert_transform(x, kind):
    """Inverse of apply_transform, used to map predictions back to real units."""
    if kind in (None, "", "none"):
        return x
    lib = _lib(x)
    if kind == "sqrt":
        return x ** 2
    if kind == "log10":
        return 10.0 ** x
    if kind == "log1p":
        return lib.expm1(x)
    raise ValueError(f"Unknown transform {kind!r}; expected one of {TRANSFORMS}.")


def normalize(x, mean, std):
    """Standardize to zero mean / unit std."""
    return (x - mean) / std


def real_units(x, mean, std):
    """Undo normalize()."""
    return x * std + mean


def compute_rmse(pred, true, mask=None):
    """Root-mean-square error. Optional boolean mask selects the elements used."""
    lib = _lib(pred)
    err = (pred - true) ** 2
    if mask is not None:
        err = err[mask]
    return lib.sqrt(err.mean())
