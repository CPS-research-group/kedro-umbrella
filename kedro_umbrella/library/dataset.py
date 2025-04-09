from pathlib import PurePosixPath
from typing import Any, Dict

import fsspec
import numpy as np
import h5py

from kedro.io import AbstractDataset
from kedro.io.core import get_filepath_str, get_protocol_and_path


class H5Dataset(AbstractDataset[np.ndarray, np.ndarray]):
    def __init__(self, filepath: str):
        """Creates a new instance of H5Dataset to load data in H5 (matlab) format.

        Args:
            filepath: The location of the image file to load / save data.
        """
        protocol, path = get_protocol_and_path(filepath)
        self._protocol = protocol
        self._filepath = PurePosixPath(path)
        self._fs = fsspec.filesystem(self._protocol)

    def _load(self) -> np.ndarray:
        """Loads data from the H5 file. 

        Returns:
            Data from the H5 file as a numpy array
        """
        load_path = get_filepath_str(self._filepath, self._protocol)
        return h5py.File(load_path, 'r')

    def _save(self, data: np.ndarray) -> None:
        assert 0 


    def _describe(self) -> Dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset."""
        return dict(filepath=self._filepath, protocol=self._protocol)