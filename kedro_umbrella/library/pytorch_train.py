import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass


logger = logging.getLogger(__name__)

def _make_deterministic(random_state):
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    random.seed(random_state)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@dataclass
class InputStats:
    size: int
    mean: np.ndarray
    std: np.ndarray

# Define the neural network architecture
class Regressor(nn.Module):
    def __init__(self, input_stats: InputStats, hidden_sizes, output_size, use_lstm=False):
        super(Regressor, self).__init__()
        self.input_stats = input_stats
        self.use_lstm = use_lstm
        if self.use_lstm:
            self.lstm = nn.LSTM(input_size=self.input_stats.size,
                    hidden_size=hidden_sizes[0], num_layers=len(hidden_sizes),
                    batch_first=True)
            self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        else:
            self.hidden_layers = nn.ModuleList()
            self.hidden_layers.append(nn.Linear(self.input_stats.size, hidden_sizes[0]))
            for i in range(len(hidden_sizes) - 1):
                self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
    
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        if self.use_lstm:
            x = x.unsqueeze(0)  # Add batch dimension for LSTM
            x, _ = self.lstm(x)
            x = x.squeeze(0)  # Remove batch dimension
        else:
            for layer in self.hidden_layers:
                x = torch.relu(layer(x))
        x = self.output_layer(x)
        return x
    
    def infer(self, x):
        self.train(False)
        with torch.no_grad():
            return self(x).detach().numpy()
    

    
    

def pytorch_trainer(X, Y, parameters):
    """
    Trains a PyTorch model using the provided input data and parameters.

    Args:
        X (numpy.ndarray or torch.Tensor): Input features with shape (n_samples, n_features).
        Y (numpy.ndarray or torch.Tensor): Target values with shape (n_samples, n_targets).
        parameters (dict): Dictionary containing training parameters:
            - "random_state" (int, optional): Seed for random number generation.
            - "hidden_layer_sizes" (tuple, optional): Sizes of hidden layers. Default is (50, 50).
            - "max_iter" (int, optional): Maximum number of training iterations. Default is 50000.
            - "learning_rate" (float, optional): Learning rate. Default is 0.001.
            - "use_lstm" (bool, optional): Whether to use LSTM layers. Default is False.
    
    Returns:
        torch.nn.Module: Trained PyTorch model with gradients disabled. It can be used to make predictions on the trained model. 
    """

    # Params
    _make_deterministic(parameters.get("random_state", None))
    hidden_sizes = parameters.get("hidden_layer_sizes", (50, 50))
    max_iter=parameters.get("max_iter", 50000)
    learning_rate = parameters.get("learning_rate", 0.001)
    use_lstm = parameters.get("use_lstm", False)

    # Convert data to torch tensors if not already
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    if not isinstance(Y, torch.Tensor):
        Y = torch.tensor(Y, dtype=torch.float32)

    # Init model
    input_stats = InputStats(
        size=X.shape[1],
        mean=np.array([X.mean().abs().floor().numpy()] * X.shape[1]),
        std=np.array([X.std().abs().floor().numpy()] * X.shape[1])
    )
    model = Regressor(input_stats, hidden_sizes, Y.shape[1], use_lstm=use_lstm)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), 
                           lr=learning_rate,
                           weight_decay=0.0001
                           )

    logger.info(f"Training model {model} with parameters: {parameters}")

    # Training loop
    max_iter = max_iter
    model.train(True)
    for epoch in range(max_iter):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()
    
        if (epoch + 1) % 1000 == 0:
            logger.info(f'Epoch [{epoch + 1}/{max_iter}], Loss: {loss.item():.4f}')
    
    model.train(False)
    # Disable gradients for the entire model
    for param in model.parameters():
        param.requires_grad = False
    return model
