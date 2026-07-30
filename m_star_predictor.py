import torch
from torch import nn
from torch.nn.modules.loss import MSELoss
import time


def train_model(network, criterion, loader, optimizer, device):
    """Train the network for one epoch"""
    network.to(device)
    network.train()

    train_loss = 0
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        # Get predictions
        if len(batch_x.shape) == 1:
            # This if block is needed to add a dummy dimension if our inputs are 1D
            # (where each number is a different sample)
            prediction = torch.squeeze(network(torch.unsqueeze(batch_x, 1)))
        else:
            prediction = network(batch_x)

        # Compute the loss
        loss = criterion(prediction, batch_y)
        train_loss += loss.item()

        # Clear the gradients
        optimizer.zero_grad()

        # Backpropagation to compute the gradients and update the weights
        loss.backward()
        optimizer.step()

    return train_loss / len(loader)


def test_model(network, criterion, loader, device):
    """Test the network"""
    network.eval()  # Evaluation mode (important when having dropout layers)

    test_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            # Get predictions
            if len(batch_x.shape) == 1:
                # This if block is needed to add a dummy dimension if our inputs are 1D
                # (where each number is a different sample)
                prediction = torch.squeeze(network(torch.unsqueeze(batch_x, 1)))
            else:
                prediction = network(batch_x)

            # Compute the loss
            loss = criterion(prediction, batch_y)
            test_loss += loss.item()

        # Get an average loss for the entire dataset
        test_loss /= len(loader)

    return test_loss


def fit_model(network, criterion, optimizer, train_loader, test_loader, n_epochs, device):
    """Train and validate the network. Yields (epoch, train_losses, test_losses) each epoch."""
    train_losses, test_losses = [], []
    start_time = time.time()

    try:
        for epoch in range(1, n_epochs + 1):
            train_loss = train_model(network, criterion, train_loader, optimizer, device)
            test_loss = test_model(network, criterion, test_loader, device)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            print(f"Epoch {epoch} completed. Train loss: {train_loss}; test loss: {test_loss}.")

            yield epoch, train_losses, test_losses

    finally:
        end_time = time.time()
        print(f"Training completed in {int(end_time - start_time)} seconds.")


def _step(network, x, n_ar, predict_tendency):
    """One forward pass. Returns the next state M (batch, n_ar)."""
    out = network(x)
    return x[:, :n_ar] + out if predict_tendency else out


def train_autoregressive(network, criterion, loader, optimizer, device,
                         k=4, n_ar=1, predict_tendency=False):
    network.to(device)
    network.train()
    total_loss, n_batches = 0.0, 0

    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        B, T, F = batch_x.shape
        n_win = T // k
        T_trunc = n_win * k                                   # drop the tail if T % k != 0

        xw = batch_x[:, :T_trunc, :].reshape(B * n_win, k, F)
        yw = batch_y[:, :T_trunc, :].reshape(B * n_win, k, n_ar)

        x = xw[:, 0, :]
        loss = torch.tensor(0.0, device=device)
        for j in range(k):
            y = _step(network, x, n_ar, predict_tendency)
            loss = loss + criterion(y, yw[:, j, :])           # scalar each time
            if j + 1 < k:
                x = torch.cat([y, xw[:, j + 1, n_ar:]], dim=-1)

        optimizer.zero_grad()
        (loss / k).backward()
        optimizer.step()

        total_loss += (loss / k).item()                       # mean per-step MSE, per batch
        n_batches += 1

    return total_loss / n_batches


def predict_autoregressive(network, batch_x, n_ar=1, predict_tendency=False):
    """Free rollout. batch_x: (batch, seq_len, n_ar + n_forcings)."""
    seq_len = batch_x.shape[1]
    x = batch_x[:, 0, :]
    preds = []

    for t in range(seq_len):
        y = _step(network, x, n_ar, predict_tendency)
        preds.append(y)
        if t + 1 < seq_len:
            x = torch.cat([y, batch_x[:, t + 1, n_ar:]], dim=-1)

    return torch.stack(preds, dim=1)      # (batch, seq_len, n_ar) -- states, not tendencies


def test_autoregressive(network, criterion, loader, device, n_ar=1, predict_tendency=False):
    network.to(device)
    network.eval()
    test_loss = 0.0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            prediction = predict_autoregressive(network, batch_x, n_ar, predict_tendency)
            test_loss += criterion(prediction, batch_y).item()

    return test_loss / len(loader)


def fit_autoregressive(network, criterion, optimizer, train_loader, test_loader, 
                       n_epochs, device, k=4, n_ar=1, predict_tendency=False):
    train_losses, test_losses = [], []
    start_time = time.time()

    try:
        for epoch in range(1, n_epochs + 1):
            train_loss = train_autoregressive(network, criterion, train_loader, optimizer, device, k, n_ar, predict_tendency)
            test_loss = test_autoregressive(network, criterion, test_loader, device, n_ar, predict_tendency)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            print(f"Epoch {epoch} completed. Train loss: {train_loss}; test loss: {test_loss}.")

            yield epoch, train_losses, test_losses

    finally:
        end_time = time.time()
        print(f"Training completed in {int(end_time - start_time)} seconds.")


class LinearRegression(nn.Module):
    def __init__(self, input_size=1):
        super().__init__()
        self.linear1 = nn.Linear(input_size, 1)

    def forward(self, x):
        x = self.linear1(x)
        return x
    

class simple_FCNN(nn.Module):
    def __init__(self, input_size=1):
        super().__init__()
        self.linear1 = nn.Linear(input_size, 8)
        self.linear3 = nn.Linear(8, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear3(x)
        return x
    

class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, y, forcings, n_ar=1):
        # y:        (n_cases, n_steps, n_ar)
        # forcings: (n_cases, n_steps, n_forcings)
        x_ar = y[:, :-1, :]         # y[t-1]  -> AR channels
        x_f  = forcings[:, 1:, :]   # f[t]    -> forcing channels, aligned to target
        self.x = torch.cat([x_ar, x_f], dim=-1)   # (n_cases, n_steps-1, n_ar + n_forcings)
        self.y = y[:, 1:, :]                      # (n_cases, n_steps-1, n_ar)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class FCNN(nn.Module):
    def __init__(
            self, 
            input_size=1,
            n_hidden_layers=1,
            n_neurons=16,
            activation=nn.ReLU()
    ):
        super().__init__()

        self.n_hidden_layers = n_hidden_layers

        self.linear_in = nn.Linear(input_size, n_neurons)
        self.linear_hidden = nn.Linear(n_neurons, n_neurons)
        self.linear_out = nn.Linear(n_neurons, 1)

        self.activation = activation

    def forward(self, x):
        x = self.activation(self.linear_in(x))

        if self.n_hidden_layers == 1:
            x = self.activation(self.linear_hidden(x))
        elif self.n_hidden_layers == 2:
            x = self.activation(self.linear_hidden(x))
            x = self.activation(self.linear_hidden(x))
        elif self.n_hidden_layers == 3:
            x = self.activation(self.linear_hidden(x))
            x = self.activation(self.linear_hidden(x))
            x = self.activation(self.linear_hidden(x))
        else:
            raise ValueError("Too many hidden layers (try 1, 2 or 3).")

        x = self.linear_out(x)
        return x


class FCNN_corrected(nn.Module):
    def __init__(
            self,
            input_size=1,
            n_neurons_list=[16, 16, 16],
            activation=nn.ReLU()
    ):
        super().__init__()

        layers = []
        layers.append(nn.Linear(input_size, n_neurons_list[0]))
        layers.append(activation)

        for i in range(len(n_neurons_list)-1):
            layers.append(nn.Linear(n_neurons_list[i], n_neurons_list[i+1]))
            layers.append(activation)

        layers.append(nn.Linear(n_neurons_list[-1], 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

