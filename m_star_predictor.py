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


def train_autoregressive(network, criterion, loader, optimizer, device, k=4, n_ar=1, modified_mse=False):
    network.to(device)
    network.train()

    total_loss, n_steps_total = 0.0, 0

    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        seq_len = batch_x.shape[1]

        x = batch_x[:, 0, 1:] if modified_mse else batch_x[:, 0, :]     # true IC + forcing at step 0
        optimizer.zero_grad()
        chunk_loss = torch.tensor(0.0, device=device)
        steps_in_chunk = 0

        for t in range(seq_len):
            y = network(x)
            if modified_mse:                                   # (batch, n_ar)
                chunk_loss = chunk_loss + criterion(y, batch_x[:, t, :], 0, 1)
            else:
                chunk_loss = chunk_loss + criterion(y, batch_y[:, t, :])
            steps_in_chunk += 1

            if (t + 1) % k == 0 or t == seq_len - 1:
                (chunk_loss / steps_in_chunk).backward()
                optimizer.step()                            # backward -> step -> zero grad needs to be understood
                optimizer.zero_grad()

                total_loss += chunk_loss.item()
                n_steps_total += steps_in_chunk

                chunk_loss = torch.tensor(0.0, device=device)
                steps_in_chunk = 0
                y = y.detach()                               # cut graph between chunks

            if t + 1 < seq_len:
                x = torch.cat([y, batch_x[:, t + 1, n_ar:]], dim=-1)    # here is the choice of state vs past vs future for other features

    return total_loss / n_steps_total


def predict_autoregressive(network, batch_x, n_ar=1):
    """Free rollout over the full sequence. batch_x: (batch, seq_len, n_features)."""
    seq_len = batch_x.shape[1]
    x = batch_x[:, 0, :]
    preds = []

    for t in range(seq_len):
        y = network(x)
        preds.append(y)
        if t + 1 < seq_len:
            x = torch.cat([y, batch_x[:, t + 1, n_ar:]], dim=-1)

    return torch.stack(preds, dim=1)      # (batch, seq_len, n_ar)


def test_autoregressive(network, criterion, loader, device, n_ar=1, modified_mse=False):
    network.to(device)
    network.eval()

    test_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            if modified_mse:
                prediction = predict_autoregressive(network, batch_x[:, :, 1:], n_ar=n_ar)
                test_loss += criterion(prediction, batch_x, 0).item()
            else:
                prediction = predict_autoregressive(network, batch_x, n_ar=n_ar)
                test_loss += criterion(prediction, batch_y).item()

    return test_loss / len(loader)


def fit_autoregressive(network, criterion, optimizer, train_loader, test_loader, 
                       n_epochs, device, k=4, n_ar=1, modified_mse=False):
    train_losses, test_losses = [], []
    start_time = time.time()

    try:
        for epoch in range(1, n_epochs + 1):
            train_loss = train_autoregressive(network, criterion, train_loader, optimizer, device, k, n_ar, modified_mse)
            test_loss = test_autoregressive(network, criterion, test_loader, device, n_ar, modified_mse)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            print(f"Epoch {epoch} completed. Train loss: {train_loss}; test loss: {test_loss}.")

            yield epoch, train_losses, test_losses

    finally:
        end_time = time.time()
        print(f"Training completed in {int(end_time - start_time)} seconds.")


class MSELoss_new(MSELoss):
    def forward(
            self, 
            dM: torch.Tensor,
            x: torch.Tensor,
            i_target: int = 0,
            i_M: int = 1
    ) -> torch.Tensor:
        target = x[:, :, i_target]
        input = x[:, :, i_M] + dM
        return ((input - target) ** 2).mean()


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

