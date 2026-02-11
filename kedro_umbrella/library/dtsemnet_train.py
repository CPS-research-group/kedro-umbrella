import logging

import numpy as np
import torch

from kedro_umbrella.library._dtsemnet import DTSemNet
from kedro_umbrella.library.utils import _make_deterministic

logger = logging.getLogger(__name__)


def _evaluate(model: DTSemNet, dataloader, loss_fn):
    """Evaluation helper used during training."""
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        total_samples = 0
        for data, target in dataloader:
            output = model(data)
            total_loss += loss_fn(output, target).item() * len(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total_samples += len(data)
    model.train()
    return total_loss / total_samples, correct / total_samples


def dtsemnet_trainer(
    X: np.ndarray | torch.Tensor, Y: np.ndarray | torch.Tensor, parameters: dict
) -> torch.nn.Module:
    # Basic validation
    config = parameters
    assert "n_attributes" in config and "n_classes" in config

    import torch
    import torch.nn as nn
    from torch import optim
    from torch.utils.data import DataLoader, TensorDataset

    # Params
    _make_deterministic(config.get("random_state", None))
    max_iter = parameters.get("max_iter", 50)

    # Init model
    model = DTSemNet(
        in_dim=config["n_attributes"],
        out_dim=config["n_classes"],
        height=config["height"],
        is_regression=False,
        over_param=[],
        wt_init=False,
        custom_leaf=None,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    if config["lr_scheduler"]:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=config["lr_scheduler_gamma"]
        )
    else:
        scheduler = None
    logger.info(f"Training model {model} with parameters: {parameters}")

    # Convert data to torch tensors if not already
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    if not isinstance(Y, torch.Tensor):
        Y = torch.tensor(Y, dtype=torch.long)
    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    # Training loop
    for epoch in range(max_iter):
        model.train()
        total_loss = 0.0
        total_batches = 0
        for data, target in dataloader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            # L1 regularization
            if config.get("use_L1"):
                l1_lambda = config.get("lamda_L1", 0.0)
                l1_reg = torch.tensor(0.0, requires_grad=False)
                for p in model.parameters():
                    l1_reg += torch.norm(p, p=1)
                loss = loss + l1_lambda * l1_reg

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_batches += 1

        if scheduler:
            scheduler.step()
        avg_loss = total_loss / max(1, total_batches)
        loss, acc = _evaluate(model, dataloader, criterion)
        logger.info(
            f"Epoch {epoch + 1}/{max_iter} Avg loss: {avg_loss:.4f} Acc: {acc:.4f}"
        )

    model.train(False)
    # Disable gradients for the entire model
    for param in model.parameters():
        param.requires_grad = False
    return model
