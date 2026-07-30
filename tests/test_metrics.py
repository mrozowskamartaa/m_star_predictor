import numpy as np
import torch
import pytest

from metrics import (
    apply_transform, invert_transform, normalize, real_units, compute_rmse,
    TRANSFORMS,
)


@pytest.mark.parametrize("kind", TRANSFORMS)
@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_transform_roundtrip(kind, backend):
    # strictly positive so log10 is defined; includes small values
    x = np.array([0.5, 1.0, 3.0, 100.0])
    x = torch.tensor(x) if backend == "torch" else x
    back = invert_transform(apply_transform(x, kind), kind)
    np.testing.assert_allclose(np.asarray(back), [0.5, 1.0, 3.0, 100.0], rtol=1e-6)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_log1p_handles_zeros(backend):
    x = np.array([0.0, 0.0, 1.0, 5.0])
    x = torch.tensor(x) if backend == "torch" else x
    out = apply_transform(x, "log1p")
    assert np.all(np.isfinite(np.asarray(out)))
    np.testing.assert_allclose(np.asarray(out)[0], 0.0, atol=1e-12)


def test_identity_transform():
    x = np.array([0.0, -1.0, 2.0])
    for kind in (None, "", "none"):
        np.testing.assert_array_equal(apply_transform(x, kind), x)
        np.testing.assert_array_equal(invert_transform(x, kind), x)


def test_unknown_transform_raises():
    with pytest.raises(ValueError):
        apply_transform(np.array([1.0]), "cube")


def test_normalize_roundtrip():
    x = torch.randn(50, 3)
    mean, std = x.mean(0), x.std(0)
    back = real_units(normalize(x, mean, std), mean, std)
    torch.testing.assert_close(back, x, rtol=1e-5, atol=1e-5)


def test_compute_rmse_known_value():
    pred = torch.tensor([1.0, 2.0, 3.0])
    true = torch.tensor([1.0, 2.0, 0.0])   # single error of 3 -> mse 3, rmse sqrt(3)
    assert compute_rmse(pred, true).item() == pytest.approx(3.0 ** 0.5)


def test_compute_rmse_mask():
    pred = torch.tensor([1.0, 2.0, 100.0])
    true = torch.tensor([1.0, 2.0, 0.0])
    mask = torch.tensor([True, True, False])
    assert compute_rmse(pred, true, mask=mask).item() == pytest.approx(0.0)
