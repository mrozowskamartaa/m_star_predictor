import torch
from torch import nn
import time


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