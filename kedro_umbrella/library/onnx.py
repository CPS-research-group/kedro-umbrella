from pathlib import PurePosixPath
from typing import Any, Dict

import fsspec
import numpy as np

from kedro.io import AbstractDataset
from kedro.io.core import get_filepath_str, get_protocol_and_path

from kedro_umbrella.library.pytorch_train import Regressor

import torch

try:
    import onnx
except ImportError as e:
    raise ImportError("The 'onnx' library is not installed. Please install it to use ONNXModel.") from e

class ONNXModel(AbstractDataset[Regressor, Regressor]):
    def __init__(self, filepath: str):
        """Creates a new instance of ONNXDataset to save data in ONNX format.

        Args:
            filepath: The location of the ONNX file to save data.
        """
        protocol, path = get_protocol_and_path(filepath)
        self._protocol = protocol
        self._filepath = PurePosixPath(path)
        self._fs = fsspec.filesystem(self._protocol)

    def _load(self) -> Regressor:
        """Loading from ONNX is not supported in this implementation."""
        raise NotImplementedError("Loading from ONNX is not supported in this implementation.")

    def _save(self, model: Regressor) -> None:
        """Saves a PyTorch model to an ONNX file.

        Args:
            data: A dictionary containing the PyTorch model and input example.
                  Expected keys are 'model' (torch.nn.Module) and 'example_input' (torch.Tensor).
        """
        self._model = model
        if not isinstance(model, torch.nn.Module):
            raise TypeError("The model must be an instance of torch.nn.Module.")
        example_input = torch.randn(1, model.input_size)
        save_path = get_filepath_str(self._filepath, self._protocol)
        with self._fs.open(save_path, "wb") as f:
            torch.onnx.export(
                model,
                example_input,
                f,
                export_params=True
            )

    def _describe(self) -> Dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset."""
        return dict(filepath=self._filepath, protocol=self._protocol)