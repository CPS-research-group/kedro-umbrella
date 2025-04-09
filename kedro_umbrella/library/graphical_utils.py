import matplotlib.pyplot as plt
from typing import Literal
import numpy as np
import torch
import os
import logging

logger = logging.getLogger(__name__)


def plot_2D3D(
    X: np.ndarray | torch.Tensor,
    Y: np.ndarray | torch.Tensor,
    Z: np.ndarray | torch.Tensor,
    x_label: str = "x",
    y_label: str = "y",
    z_label: str = "u(x,y)",
    output_path: str = "./data/07_model_output",
    file_name: str = "model_output",
    file_type: Literal["png", "jpg", "jpeg"] = "png",
):
    if isinstance(X, torch.Tensor):
        X = X.detach().numpy()
    if isinstance(Y, torch.Tensor):
        Y = Y.detach().numpy()
    if isinstance(Z, torch.Tensor):
        Z = Z.detach().numpy()

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    cm = ax1.contourf(X, Y, Z, 20, cmap="viridis")
    fig.colorbar(cm, ax=ax1)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)

    ax1.set_aspect("equal")

    ax2 = fig.add_subplot(122, projection="3d")
    ax2.plot_surface(X, Y, Z, cmap="viridis")
    ax2.set_xlabel(x_label)
    ax2.set_ylabel(y_label)
    ax2.set_zlabel(z_label)

    fig.tight_layout()

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    plt.savefig(f"{output_path}/{file_name}.{file_type}")
    logger.info(f"Plot saved at {output_path}/{file_name}.{file_type}")

    plt.close()
