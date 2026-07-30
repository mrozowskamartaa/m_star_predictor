import numpy as np
import torch
from torch import nn, optim
import torch.utils.data as Data
import pytest

from model import (
    LinearRegression, FCNN, _step, rollout,
    ParallelPredictor, AutoregressivePredictor,
)


class ConstantNet(nn.Module):
    """Ignores its input and returns a constant vector of length n_ar."""
    def __init__(self, c, n_ar=1):
        super().__init__()
        self.register_buffer("c", torch.full((n_ar,), float(c)))

    def forward(self, x):
        return self.c.expand(x.shape[0], -1)


# --------------------------------------------------------------------------- #
# _step: state vs tendency arithmetic
# --------------------------------------------------------------------------- #
def test_step_state_mode_returns_network_output():
    net = ConstantNet(2.0, n_ar=1)
    x = torch.tensor([[5.0, 9.0], [1.0, 0.0]])   # state channel = col 0
    out = _step(net, x, n_ar=1, predict_tendency=False)
    torch.testing.assert_close(out, torch.tensor([[2.0], [2.0]]))


def test_step_tendency_mode_adds_previous_state():
    net = ConstantNet(2.0, n_ar=1)
    x = torch.tensor([[5.0, 9.0], [1.0, 0.0]])
    out = _step(net, x, n_ar=1, predict_tendency=True)   # prev_state + increment
    torch.testing.assert_close(out, torch.tensor([[7.0], [3.0]]))


# --------------------------------------------------------------------------- #
# rollout: feedback correctness (closed-form)
# --------------------------------------------------------------------------- #
def test_rollout_constant_tendency_accumulates():
    # predict_tendency=True with constant increment c -> state_t = init + (t+1)*c
    c, init, T = 0.5, 3.0, 5
    net = ConstantNet(c, n_ar=1)
    seq = torch.zeros(2, T, 3)          # F = 1 state + 2 forcings
    seq[:, 0, 0] = init
    out = rollout(net, seq, n_ar=1, predict_tendency=True)
    expected = init + c * torch.arange(1, T + 1).float()
    torch.testing.assert_close(out[0, :, 0], expected)
    torch.testing.assert_close(out[1, :, 0], expected)


def test_rollout_state_mode_is_constant():
    # predict_tendency=False with constant net -> every step equals c
    net = ConstantNet(7.0, n_ar=1)
    seq = torch.randn(3, 4, 2)
    out = rollout(net, seq, n_ar=1, predict_tendency=False)
    torch.testing.assert_close(out, torch.full((3, 4, 1), 7.0))


def test_rollout_shape_multi_ar():
    net = ConstantNet(1.0, n_ar=2)
    seq = torch.randn(6, 8, 5)          # F = 2 state + 3 forcings
    out = rollout(net, seq, n_ar=2, predict_tendency=False)
    assert out.shape == (6, 8, 2)


def test_rollout_feeds_prediction_not_truth():
    # a net that reads its own state channel: out = state + 1 (tendency=False).
    class Increment(nn.Module):
        def forward(self, x):
            return x[:, :1] + 1.0
    net = Increment()
    seq = torch.zeros(1, 3, 1)          # only a state channel, init 0
    out = rollout(net, seq, n_ar=1, predict_tendency=False)
    # fed back: 0 -> 1 -> 2 -> 3
    torch.testing.assert_close(out[0, :, 0], torch.tensor([1.0, 2.0, 3.0]))


# --------------------------------------------------------------------------- #
# FCNN architecture: hidden layers are independent (the bug fix)
# --------------------------------------------------------------------------- #
def test_fcnn_hidden_layers_are_distinct():
    net = FCNN(input_size=3, n_neurons_list=[4, 4, 4])
    linears = [m for m in net.network if isinstance(m, nn.Linear)]
    # input + 2 hidden transitions + output = 4 Linear layers, all distinct objects
    assert len(linears) == 4
    assert len({id(m) for m in linears}) == 4


def test_fcnn_forward_shape():
    net = FCNN(input_size=3, n_neurons_list=[8, 8])
    assert net(torch.randn(10, 3)).shape == (10, 1)


# --------------------------------------------------------------------------- #
# Predictors actually learn (smoke tests for wiring)
# --------------------------------------------------------------------------- #
def test_parallel_predictor_reduces_loss():
    torch.manual_seed(0)
    N = 512
    x = torch.randn(N, 3)
    y = (x @ torch.tensor([[1.0], [-2.0], [0.5]]) + 0.3)   # linear target
    loader = Data.DataLoader(Data.TensorDataset(x, y), batch_size=64, shuffle=True)

    net = LinearRegression(input_size=3)
    pred = ParallelPredictor(net, nn.MSELoss(), optim.Adam(net.parameters(), lr=0.05),
                             device="cpu", n_ar=1, predict_tendency=False)
    losses = [t for _, t, _ in pred.fit(loader, loader, n_epochs=40)]
    assert losses[-1][-1] < losses[0][0] * 0.1


def test_autoregressive_predictor_reduces_loss():
    # synthetic dynamical system: M[t+1] = M[t] + 0.5*forcing[t]  (a tendency model)
    torch.manual_seed(0)
    n_cases, T = 32, 24
    forcing = torch.randn(n_cases, T, 1) * 0.1
    M = torch.zeros(n_cases, T, 1)
    for t in range(1, T):
        M[:, t] = M[:, t - 1] + 0.5 * forcing[:, t - 1]
    # canonical (case, time, feature): col 0 = state M, col 1 = forcing
    X = torch.cat([M, forcing], dim=-1)
    Y = M                                     # target is the state sequence
    loader = Data.DataLoader(Data.TensorDataset(X, Y), batch_size=8, shuffle=True)

    net = FCNN(input_size=2, n_neurons_list=[16, 16], activation=nn.Tanh())
    pred = AutoregressivePredictor(net, nn.MSELoss(),
                                   optim.Adam(net.parameters(), lr=1e-2),
                                   device="cpu", n_ar=1, predict_tendency=True, k=4)
    losses = [t for _, t, _ in pred.fit(loader, loader, n_epochs=60)]
    assert losses[-1][-1] < losses[0][0]


def test_parallel_and_ar_share_step_for_seq_len_one():
    # a single-timestep rollout must equal one parallel _step (same net, same math)
    torch.manual_seed(1)
    net = FCNN(input_size=4, n_neurons_list=[8])
    x = torch.randn(5, 4)
    parallel_out = _step(net, x, n_ar=1, predict_tendency=True)
    ar_out = rollout(net, x.unsqueeze(1), n_ar=1, predict_tendency=True)[:, 0, :]
    torch.testing.assert_close(parallel_out, ar_out)
