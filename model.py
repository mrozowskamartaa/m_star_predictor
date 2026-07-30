"""Networks and the two rollout strategies that share one step function.

Design
------
The whole codebase runs on a single canonical tensor shape ``(case, time, feature)``
in which the *first ``n_ar`` feature columns are the autoregressive state channels*
(e.g. M) and the rest are forcings. Given that convention:

* ``_step`` is the one forward pass shared by both predictors. ``predict_tendency``
  is a residual flag on that step (the net predicts an increment that is added to
  the previous state) -- meaningful in both modes, so state-vs-tendency is no
  longer a data-layout decision.
* ``ParallelPredictor`` is *teacher forcing*: every sample is independent and the
  true previous state is fed in. It operates on flattened ``(sample, feature)``.
* ``AutoregressivePredictor`` is a *rollout*: the predicted state is fed back.
  It operates on ``(case, time, feature)``, trains on k-step windows and tests on
  the full free rollout.
"""
import time
from abc import ABC, abstractmethod

import torch
from torch import nn


# --------------------------------------------------------------------------- #
# Networks
# --------------------------------------------------------------------------- #
class LinearRegression(nn.Module):
    def __init__(self, input_size=1, output_size=1):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


class FCNN(nn.Module):
    """Fully-connected net with independent hidden layers.

    This is the corrected architecture: one distinct ``nn.Linear`` per hidden
    layer (the old ``FCNN`` reused a single hidden module, so its hidden layers
    shared weights -- that was the bug behind the FCNN vs FCNN_corrected re-runs).
    """

    def __init__(self, input_size=1, n_neurons_list=(16, 16, 16),
                 activation=nn.ReLU(), output_size=1):
        super().__init__()
        n_neurons_list = list(n_neurons_list)
        layers = [nn.Linear(input_size, n_neurons_list[0]), activation]
        for a, b in zip(n_neurons_list[:-1], n_neurons_list[1:]):
            layers += [nn.Linear(a, b), activation]
        layers.append(nn.Linear(n_neurons_list[-1], output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# --------------------------------------------------------------------------- #
# Shared step + rollout
# --------------------------------------------------------------------------- #
def _step(network, x, n_ar, predict_tendency):
    """One forward pass. x is (batch, n_ar + n_forcings); returns (batch, n_ar).

    With predict_tendency the network output is an increment added to the current
    state channels x[:, :n_ar]; otherwise it is the next state directly.
    """
    out = network(x)
    return x[:, :n_ar] + out if predict_tendency else out


def rollout(network, seq, n_ar=1, predict_tendency=False):
    """Free autoregressive rollout.

    seq: (batch, seq_len, n_ar + n_forcings). The state channels of seq[:, 0] seed
    the rollout; forcings are read from seq at each step and predicted states are
    fed back. Returns predicted states (batch, seq_len, n_ar).
    """
    seq_len = seq.shape[1]
    x = seq[:, 0, :]
    preds = []
    for t in range(seq_len):
        y = _step(network, x, n_ar, predict_tendency)
        preds.append(y)
        if t + 1 < seq_len:
            x = torch.cat([y, seq[:, t + 1, n_ar:]], dim=-1)
    return torch.stack(preds, dim=1)


# --------------------------------------------------------------------------- #
# Predictors
# --------------------------------------------------------------------------- #
class Predictor(ABC):
    def __init__(self, network, criterion, optimizer, device,
                 n_ar=1, predict_tendency=False):
        self.network = network
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.n_ar = n_ar
        self.predict_tendency = predict_tendency

    @abstractmethod
    def train_epoch(self, loader):
        ...

    @abstractmethod
    def test_epoch(self, loader):
        ...

    def fit(self, train_loader, test_loader, n_epochs):
        """Train + validate. Yields (epoch, train_losses, test_losses) each epoch."""
        train_losses, test_losses = [], []
        start = time.time()
        try:
            for epoch in range(1, n_epochs + 1):
                train_losses.append(self.train_epoch(train_loader))
                test_losses.append(self.test_epoch(test_loader))
                print(f"Epoch {epoch}: train {train_losses[-1]:.3e}; "
                      f"test {test_losses[-1]:.3e}.")
                yield epoch, train_losses, test_losses
        finally:
            print(f"Training completed in {int(time.time() - start)} seconds.")


class ParallelPredictor(Predictor):
    """Teacher forcing on flattened (sample, feature) data."""

    def train_epoch(self, loader):
        self.network.to(self.device).train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            pred = _step(self.network, xb, self.n_ar, self.predict_tendency)
            loss = self.criterion(pred, yb)
            total += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return total / len(loader)

    def test_epoch(self, loader):
        self.network.eval()
        total = 0.0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                pred = _step(self.network, xb, self.n_ar, self.predict_tendency)
                total += self.criterion(pred, yb).item()
        return total / len(loader)


class AutoregressivePredictor(Predictor):
    """Rollout on (case, time, feature) data: k-step windows to train, full rollout
    to test."""

    def __init__(self, *args, k=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k

    def train_epoch(self, loader):
        self.network.to(self.device).train()
        total, n_batches = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            B, T, F = xb.shape
            n_win = T // self.k
            T_trunc = n_win * self.k                    # drop the tail if T % k != 0
            xw = xb[:, :T_trunc, :].reshape(B * n_win, self.k, F)
            yw = yb[:, :T_trunc, :].reshape(B * n_win, self.k, self.n_ar)

            x = xw[:, 0, :]
            loss = torch.tensor(0.0, device=self.device)
            for j in range(self.k):
                y = _step(self.network, x, self.n_ar, self.predict_tendency)
                loss = loss + self.criterion(y, yw[:, j, :])
                if j + 1 < self.k:
                    x = torch.cat([y, xw[:, j + 1, self.n_ar:]], dim=-1)

            self.optimizer.zero_grad()
            (loss / self.k).backward()
            self.optimizer.step()
            total += (loss / self.k).item()
            n_batches += 1
        return total / n_batches

    def test_epoch(self, loader):
        self.network.to(self.device).eval()
        total = 0.0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                pred = rollout(self.network, xb, self.n_ar, self.predict_tendency)
                total += self.criterion(pred, yb).item()
        return total / len(loader)
