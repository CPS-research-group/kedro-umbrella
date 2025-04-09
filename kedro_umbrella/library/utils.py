import logging
import os
from typing import Any, Dict, Tuple
import numpy as np
import torch
from sklearn.metrics import root_mean_squared_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import mat73
import scipy.io as sio

import matplotlib.pyplot as plt
import yaml
import os

DECIMAL_PRECISION = 8 # based on machine precision vals

logger = logging.getLogger(__name__)

class ReportDir:
    REPORT_DIR_PREFIX = "data/08_reporting/"

    def __init__(self, node_name: str):
        self.node_name = node_name
        self.report_dir = ReportDir.REPORT_DIR_PREFIX + node_name
        if not os.path.exists(self.report_dir):
            os.makedirs(self.report_dir)

    def get(self):
        return self.report_dir


def _round_float(value):
    from decimal import Decimal
    dec = round(Decimal(value), DECIMAL_PRECISION)
    return float(dec)

def score(Y_test, Y_pred, parameters: Dict[str, Any] = {}):
    """
    Compute and log various regression metrics, generate plots, and save results to files.
    
    Args: 
        Y_test : array-like or torch.Tensor
            The ground truth target values.
        Y_pred : array-like or torch.Tensor
            The predicted target values.
        parameters : dict, optional
            Additional parameters for the function. Expected keys:
            - '_node_name': str, required for determining the report directory.
            - 'plot': str, optional, specifies the type of plot to generate. 
            Can be 'regression' or 'time_series'.
    
    Returns:
        A tuple containing the normalized root mean squared error (NRMSE) and the R-squared (r2) score.
    
    Typing partition: 
        P1 = {Y_test, Y_pred}

    Notes:
        - Generates and saves regression scatter and residual plots if `parameters['plot']` is 'regression'.
        - Generates and saves time series plots if `parameters['plot']` is 'time_series'.
        - Saves the computed metrics (MSE, RMSE, NRMSE, r2) to a YAML file in the report directory.
    """

    def _regression_plot(Y_test, Y_pred):
        logger.info(f"Generating regression scatter and residual plot "
                    f"'{REPORT_DIR}/actual_vs_pred.png'")
        # Scatter plot of actual vs. predicted values
        plt.figure()
        plt.scatter(Y_test, Y_pred, alpha=0.5)
        plt.title("Actual vs. Predicted Values")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.plot(
            [Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], "k--", lw=4
        )  # Diagonal line
        plt.tight_layout()
        plt.savefig(f"{REPORT_DIR}/actual_vs_pred.png")

        # Residual plot
        residuals = Y_pred - Y_test
        plt.figure()
        plt.scatter(Y_test, residuals, alpha=0.5)
        plt.title("Residuals vs. Actual Values")
        plt.xlabel("Actual Values")
        plt.ylabel("Residuals")
        plt.tight_layout()
        plt.axhline(y=0, color="k", linestyle="--")  # Horizontal line at 0
        plt.savefig(f"{REPORT_DIR}/residual_plot.png")

    def _time_series_plot(Y_test, Y_pred):
        # Test and pred
        plt.figure()
        plt.plot(Y_test, label="Test")
        plt.plot(Y_pred, label="Prediction")
        plt.title("Test & predictions")
        plt.xlabel("Sample")
        plt.ylabel("Value")
        plt.tight_layout()
        plt.legend()
        plt.savefig(f"{REPORT_DIR}/test_pred.png")

    REPORT_DIR = ReportDir(parameters['_node_name']).get()

    # compute score values
    Y_test_ = Y_test.numpy() if isinstance(Y_test, torch.Tensor) else Y_test
    Y_pred_ = Y_pred.numpy() if isinstance(Y_pred, torch.Tensor) else Y_pred
    mse = mean_squared_error(Y_test_, Y_pred_)
    rmse = root_mean_squared_error(Y_test_, Y_pred_)
    nrmse = rmse / (np.max(Y_test_) - np.min(Y_test_))
    r2 = r2_score(Y_test_, Y_pred_)
    logger.info(f"MSE = {mse}, NRMSE = {nrmse}, r2 = {r2}")

    # dump score YAML
    score = {
        "mse": _round_float(mse),
        "rmse": _round_float(rmse),
        "nrmse": _round_float(nrmse),
        "r2": _round_float(r2),
    }
    logger.info(f"Saving scores to '{REPORT_DIR}/score.yml'")
    with open(f"{REPORT_DIR}/score.yml", "w") as file:
        yaml.dump(score, file)

    # dump plot
    if parameters.get("plot", None) == "regression":
        _regression_plot(Y_test, Y_pred)
    if parameters.get("plot", None) == "time_series":
        _time_series_plot(Y_test, Y_pred)

    return score["nrmse"], score["r2"]


def load_mat(
    parameters: Dict[str, Any]
) -> Tuple[np.ndarray | list[np.ndarray], np.ndarray | list[np.ndarray]]:
    """
    Load data from a .mat file.

    Args:
        parameters: Additional parameters.

    Returns:
        Tuple of input features and output data.
    """
    data = None
    try:
        data = mat73.loadmat(parameters["data_path"])
    except TypeError as e:
        if "is not a MATLAB 7.3 file" in str(e):
            data = sio.loadmat(parameters["data_path"])
        else:
            logger.error(
                f"Error loading .mat file: {parameters['data_path']}\n{type(e)}: {e}"
            )  # Error will not propagate to the below logger
            raise e
    except Exception as e:
        logger.error(
            f"Error loading .mat file: {parameters['data_path']}\n{type(e)}: {e}"
        )
        raise e

    input_key: str | list[str] = parameters.get("input_key", "input")
    output_key: str | list[str] = parameters.get("output_key", "output")

    X = None
    Y = None

    if isinstance(input_key, str):
        X = data[input_key]
    else:
        X = [data[key] for key in input_key]
    if isinstance(output_key, str):
        Y = data[output_key]
    else:
        Y = [data[key] for key in output_key]

    return X, Y


def split_data(
    X: np.ndarray,
    Y: np.ndarray,
    parameters: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and testing sets.

    Args:
        X: Input features.
        Y: Output data.
        parameters (Dict[str, Any]): Additional parameters for splitting the data.
            - "random_state" (int, optional): The seed used by the random number generator for shuffling the data.
            - "split_time" (int, optional): The index at which to split the data into training and testing sets. If provided, the data will be split at this index (use for splitting based on time).
            - "train_size" (float or int, optional): If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split. If int, represents the absolute number of train samples. If None, it will be set to 0.75.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing the training and testing sets:
            - X_train (np.ndarray): Training input features.
            - X_test (np.ndarray): Testing input features.
            - Y_train (np.ndarray): Training output data.
            - Y_test (np.ndarray): Testing output data.

    Typing partition: 
        P1 = {X, X_train, X_test}; P2 = {Y, Y_train, Y_test}
    """
    # Use provided split time or random state
    split_time = parameters.get("split_time", None)
    if split_time:
        X_train, X_test = X[:split_time], X[split_time:]
        Y_train, Y_test = Y[:split_time], Y[split_time:]
        return X_train, X_test, Y_train, Y_test

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        random_state=parameters.get("random_state", None),
        train_size=parameters.get("train_size", None),
    )
    return X_train, X_test, Y_train, Y_test


def load_device(parameters) -> torch.device:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    logger.info(f"Loaded device: {device}")

    return device

def difference(a, b):
    """
    Calculate the element-wise difference between two arrays.
    
    Parameters:
        a (array-like): The first input array.
        b (array-like): The second input array.

    Returns:
        numpy.ndarray: The element-wise difference (diff) of the input arrays.

    Typing partition: 
        P1 = {a, b, diff}
    """
    return np.subtract(a, b)
