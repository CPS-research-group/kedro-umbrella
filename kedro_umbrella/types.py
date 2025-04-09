""" This module defines the Data Types and Function Types.
"""

import warnings

class DataType:
    """Represents the numerical values to process (physical measurements, images, etc.).
    The type is defined by a numerical ID which will be created based on the port traversal order.

    A data port will have a type given by default by a number in $[1, m]$, where $m$ is the number of data types. 
    """
    def __init__(self, id: int):
        assert id > 0
        self._id = id

    def is_data(self) -> bool:
        return True
    
    def __repr__(self) -> str:
        return f"DataType({self._id})"

class FunctionType(DataType):
    """Represents the trained regression models (for traditional AI), the physics
    model, etc. Given some input data (in_0, in_1, ..., in_n), the Function F, will output some new data (out_0, out_1, ..., out_n). 

    (in_0, in_1, ..., in_n) -> F -> (out_0, out_1, ..., out_n)
    
    The type of Function is thus the corresponding types for the inputs and outputs.
    """

    def __init__(self, in_type: list[DataType], out_type: list[DataType], id : int):
        super().__init__(id)
        assert isinstance(in_type, list)
        assert isinstance(out_type, list)
        self.in_type = in_type
        self.out_type = out_type

    def is_function(self) -> bool:
        return True
    
    @property
    def num_inputs(self) -> int:
        return len(self.in_type)
    
    @property
    def num_outputs(self) -> int:
        return len(self.out_type)
    
    def __repr__(self) -> str:
        return f"FunctionType({self.in_type}, {self.out_type})"

"""
TypeCatalog is a dictionary that maps the port name to the port type.
    port name is the output port name, guaranteed to be unique by Kedro
"""
class TypeCatalog:
    def __init__(self):
        self._catalog : dict [str, DataType] = {}
        self._last_id = 0

    def make_data(self) -> DataType:
        self._last_id += 1
        return DataType(self._last_id)

    def add_data(self, port_name: str) -> DataType:
        self._last_id += 1
        data = DataType(self._last_id)
        self._add(port_name, data)
        return data

    def add_function(self, port_name: str, 
                     in_type: DataType | list[DataType],
                     out_type: DataType | list[DataType]):
        self._last_id += 1
        if isinstance(in_type, DataType):
            in_type = [in_type]
        if isinstance(out_type, DataType):
            out_type = [out_type]
        self._add(port_name, FunctionType(in_type, out_type, self._last_id))

    def _add(self, port_name: str, port_type: DataType):
        # only add if not present
        if port_name in self._catalog:
            warnings.warn(f"Port name {port_name} already in catalog")
            #raise ValueError(f"Port name {port_name} already in catalog")
        self._catalog[port_name] = port_type

    def get_or_create(self, port_name: str) -> DataType:
        if port_name not in self._catalog:
            self.add_data(port_name)
        return self._catalog[port_name]

    def __setitem__(self, port_name: str, port_type: DataType):
        self._add(port_name, port_type)
    
    def __getitem__(self, port_name: str) -> DataType:
        return self._catalog[port_name]
    
    def __contains__(self, port_name: str) -> bool:
        return port_name in self._catalog
    
    def __repr__(self) -> str:
        return str(self._catalog)