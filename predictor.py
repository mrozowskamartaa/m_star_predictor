import torch
from torch import nn
import time
from abc import abstractmethod, ABC


def _step(
        network: nn.Module, 
        x, 
        n_ar, 
        predict_tendency
    ):
    out = network(x)
    return x[:, :n_ar] + out if predict_tendency else out


def predict_autoregressive(
        network: nn.Module, 
        batch_x, 
        n_ar=1, 
        predict_tendency=False
    ):
    """Free rollout. batch_x: (batch, seq_len, n_ar + n_forcings)."""
    seq_len = batch_x.shape[1]
    x = batch_x[:, 0, :]
    preds = []

    for t in range(seq_len):
        y = _step(network, x, n_ar, predict_tendency)
        preds.append(y)
        if t + 1 < seq_len:
            x = torch.cat([y, batch_x[:, t + 1, n_ar:]], dim=-1)

    return torch.stack(preds, dim=1)


class Predictor(ABC):
    def __init__(
        self,
        network: nn.Module,
        criterion,
        optimizer,
        device,
        n_epochs: int,
        n_ar: int = 1,
        predict_tendency: bool = False
    ):
        self.network = network
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.n_epochs = n_epochs
        self.n_ar = n_ar
        self.predict_tendency = predict_tendency


    @abstractmethod
    def train_model(
        self,
        train_loader
    ):
        return ...


    @abstractmethod
    def test_model(
        self,
        test_loader
    ):
        return ...


    def fit_model(
            self,
            train_loader,
            test_loader
    ):
        train_losses, test_losses = [], []
        start_time = time.time()

        try:
            for epoch in range(1, self.n_epochs + 1):
                train_loss = self.train_model(train_loader=train_loader)
                test_loss = self.test_model(test_loader=test_loader)
                train_losses.append(train_loss)
                test_losses.append(test_loss)
                print(f"Epoch {epoch} completed. Train loss: {train_loss}; test loss: {test_loss}.")

                yield epoch, train_losses, test_losses

        finally:
            end_time = time.time()
            print(f"Training completed in {int(end_time - start_time)} seconds.")
            return


class ParallelPredictor(Predictor):


    def train_model(
            self, 
            train_loader
    ):
        self.network.to(self.device)
        self.network.train()

        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            if len(batch_x.shape) == 1:
                prediction = torch.squeeze(self.network(torch.unsqueeze(batch_x, 1)))
            else:
                prediction = self.network(batch_x)

            loss = self.criterion(prediction, batch_y)
            train_loss += loss.item()

            self.optimizer.zero_grad()

            loss.backward()
            self.optimizer.step()

        return train_loss / len(train_loader)


    def test_model(
            self,
            test_loader
    ):
        self.network.eval()

        test_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                if len(batch_x.shape) == 1:
                    prediction = torch.squeeze(self.network(torch.unsqueeze(batch_x, 1)))
                else:
                    prediction = self.network(batch_x)

                loss = self.criterion(prediction, batch_y)
                test_loss += loss.item()

            test_loss /= len(test_loader)

        return test_loss


class AutoregressivePredictor(Predictor):
    def __init__(
            self,
            k: int = 4,
            n_ar: int = 1,
            predict_tendency: bool = False
    ):
        self.k = k
        self.n_ar = n_ar
        self.predict_tendency = predict_tendency


class LinearRegression(nn.Module):
    def __init__(self, input_size=1):
        super().__init__()
        self.linear1 = nn.Linear(input_size, 1)

    def forward(self, x):
        x = self.linear1(x)
        return x


class FCNN(nn.Module):
    def __init__(
            self,
            input_size: int = 1,
            n_neurons_list: list[int] = [16, 16, 16],
            activation = nn.ReLU()
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