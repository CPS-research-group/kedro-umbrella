"""This module allow to create Processor nodes as part of Kedro pipelines.
"""

import re
from typing import Any, Iterable, Callable

from kedro.pipeline.node import Node, _to_list, _get_readable_func_name
from kedro_umbrella.types import *
from kedro.pipeline.modular_pipeline import _is_parameter
from warnings import warn

class Processor(Node):
    """``Processor`` is for typical data processing (including applying functions learned by other boxes). The Processor function is either provided as the first input (as obtained via Coder and Trainer blocks) or is a "fixed (pre-defined) function". 
    """
    

    def __init__(
        self,
        inputs: str | list[str],
        outputs: str | list[str] | dict[str, str],
        *,
        name: str = None,
        tags: str | Iterable[str] | None = None,
        confirms: str | list[str] | None = None,
        namespace: str = None,
        func: Callable = None,
        inputs_type: dict [str, str] = None,
    ):
        """Create a processor in the pipeline. The function to be called is either the first input (from previous Box) or a fixed function provided by 'func'.
        Remaining inputs and/or outputs are the data.

        Args:
            inputs: The name or the list of the names of variables used as
                inputs to the function. The __first input__ is special and is
                the function to be called (from a "Code" or "Trainer"). 
                The number of names should match
                the number of arguments in the definition of the provided
                function. When dict[str, str] is provided, variable names
                will be mapped to function argument names.
            outputs: The name or the list of the names of variables used
                as outputs to the function. The number of names should match
                the number of outputs returned by the provided function.
                When dict[str, str] is provided, variable names will be mapped
                to the named outputs the function returns.
            name: Optional processor name to be used when displaying the processor in
                logs or any other visualisations.
            tags: Optional set of tags to be applied to the processor.
            confirms: Optional name or the list of the names of the datasets
                that should be confirmed. This will result in calling
                ``confirm()`` method of the corresponding data set instance.
                Specified dataset names do not necessarily need to be present
                in the processor ``inputs`` or ``outputs``.
            namespace: Optional processor namespace.
            func: Optional fixed function.
            inputs_type: Optional input type annotation.

        Raises:
            ValueError: Raised in the following cases:
                a) When the provided arguments do not conform to
                the format suggested by the type hint of the argument.
                b) When the processor produces multiple outputs with the same name.
                c) When an input has the same name as an output.
                d) When the given processor name violates the requirements:
                it must contain only letters, digits, hyphens, underscores
                and/or fullstops.

        """ 
        if not inputs:
            raise ValueError(_process_error_message("it must have some 'inputs'."))
        
        if inputs and not isinstance(inputs, (str, list)):
            raise ValueError(
                _process_error_message(
                    f"'inputs' type must be one of [String, List], "
                    f"not '{type(inputs).__name__}'."
                )
            )
        
        if outputs and not isinstance(outputs, (str, list, dict)):
            raise ValueError(
                _process_error_message(
                    f"'outputs' type must be one of [String, List, Dict, None], "
                    f"not '{type(outputs).__name__}'."
                )
            )
        if func:
            if not callable(func):
                raise ValueError(
                    _process_error_message(
                        f"'func' must be a callable, not '{type(func).__name__}'."
                    )
                )
            self._fixed_func = True
            self._func = func
        else:
            self._fixed_func = False
            self._func = self._extract_function(inputs)
        self._inputs = inputs
        self._outputs = outputs
        if name and (not isinstance(name, str) or not re.match(r"[\w\.-]+$", name)):
            raise ValueError(
                _process_error_message(
                    f"'{name}' is not a valid processor name. It must contain only "
                    f"letters, digits, hyphens, underscores and/or fullstops."
                )
            )
        self._name = name
        self._namespace = namespace
        self._tags = set(_to_list(tags))

        self._validate_unique_outputs()
        self._validate_inputs_dif_than_outputs()
        self._confirms = confirms
        self._param_num = self._count_param()
        self._inputs_type = inputs_type
    
    def _count_param(self):
        # str case
        if isinstance(self._inputs, str) and _is_parameter(self._inputs):
            return 1
        # list case
        cnt = 0 
        for input in self._inputs:
            if _is_parameter(input):
                cnt += 1
        return cnt
    
    def __str__(self):
        def _set_to_str(xset):
            return f"[{','.join(xset)}]"

        out_str = _set_to_str(self.outputs) if self._outputs else "None"
        in_str = _set_to_str(self.inputs) if self._inputs else "None"

        prefix = self._name + ": " if self._name else ""
        return prefix + f"{self._func_name}({in_str}) -> {out_str}"

    def __repr__(self):  # pragma: no cover
        return (
            f"Processor({self._func_name}, {repr(self._inputs)}, {repr(self._outputs)}, "
            f"{repr(self._name)})"
        )
    
    @property
    def _func_name(self) -> str:
        if not self._fixed_func:
            return self._func
        # fixed func
        name = _get_readable_func_name(self._func)
        return name

    def _copy(self, **overwrite_params):
        """
        Helper function to copy the processor, replacing some values.
        """
        params = {
            "inputs": self._inputs,
            "outputs": self._outputs,
            "name": self._name,
            "namespace": self._namespace,
            "tags": self._tags,
            "confirms": self._confirms,
            # XXX hacky way to reset _func when fixed function is used
            "func": None if isinstance(self._func, str) else self._func,
            "inputs_type": self._inputs_type
        }
        params.update(overwrite_params)
        return Processor(**params)

    @staticmethod
    def _extract_function(inputs: str | list[str]):
        if isinstance(inputs, str):
            return inputs
        if isinstance(inputs, list):
            return inputs[0]
    
    @property  
    def num_inputs(self):
        if self._fixed_func:
            return len(self.inputs) - self._param_num
        else: 
            # -1 for fixed function 
            return len(self.inputs) - 1 - self._param_num

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Run this processor using the provided inputs and return its results
        in a dictionary.

        Args:
            inputs: Dictionary of inputs as specified at the creation of
                the processor.

        Raises:
            ValueError: In the following cases:
                a) The processor function inputs are incompatible with the processor
                input definition.
                Example 1: processor definition input is a list of 2
                DataFrames, whereas only 1 was provided or 2 different ones
                were provided.
                b) The processor function outputs are incompatible with the processor
                output definition.
                Example 1: processor function definition is a dictionary,
                whereas function returns a list.
                Example 2: processor definition output is a list of 5
                strings, whereas the function returns a list of 4 objects.
            Exception: Any exception thrown during execution of the processor.

        Returns:
            All produced processor outputs are returned in a dictionary, where the
            keys are defined by the processor outputs.

        """
        assert len(inputs) >= 1, "Invalid input size"
        self._logger.info("Running processor: %s", str(self))
        if not isinstance(inputs, dict):
            raise ValueError(
                f"Processor.run() expects a dictionary, "
                f"but got {type(inputs)} instead."
            )
        outputs = None
        inputs1 = inputs.copy()
        self._func = self.get_real_func(inputs1)
        self._inputs : list[str] = list(inputs1.keys())

        self._validate_inputs(self._func, self._inputs)
        try:
            inputs1 = {} if inputs1 is None else inputs1
            if not self._inputs:
                outputs = self._run_with_no_inputs(inputs1)
            elif isinstance(self._inputs, list):
                self.add_internal_inputs(inputs1)
                outputs = self._run_with_list(inputs1, self._inputs)

            return self._outputs_to_dictionary(outputs)

        # purposely catch all exceptions
        except Exception as exc:
            self._logger.error(
                "Processor '%s' failed with error: \n%s", str(self), str(exc)
            )
            raise exc

    def add_internal_inputs(self, inputs):
        '''
        Add extra name params.
        Include a 'node_name' parameter w/ the node name
        '''
        def need_node_name():
            if not hasattr(self._func, "__name__"):
                return False
            if self._func.__name__ in ["score", "sensitivity_analysis"]:
                return True
            return False
        def add_node_name():
            if not need_node_name():
                return
            # add parameter if not there 
            if not _is_parameter(self._inputs[-1]):
                self._inputs.append('parameters')
            
            if self._inputs[-1] in inputs:
                # there are already some user-declared param
                param = inputs[self._inputs[-1]]
                param['_node_name'] = self.name
            else:
                # no user-declared param
                inputs[self._inputs[-1]] = {'_node_name': self.name}
        add_node_name()
            
    def get_real_func(self, inputs1):
        # fixed function
        if self._fixed_func:
            return self._func
        
        # function as input
        k = next(iter(inputs1))
        real_func = inputs1.pop(k)
        if not callable(real_func):
            raise ValueError(
                _process_error_message(
                    f"First 'input' must be a callable, not '{type(real_func).__name__}'."
                )
            )
        assert callable(real_func), "Expected func is not callable."
        return real_func

    def check(self, types: TypeCatalog) -> None:
            self._logger.info("Checking process: %s", self)
            inputs = self.inputs
            outputs = self.outputs

            in_it = iter(inputs)
            if self._fixed_func:
                # fixed function
                func_name = self._func.__name__
                make_fixed_type(self, types)
                func_type = types[func_name]
                assert type(func_type) is FunctionType
            else:
                # non-fixed function: first input is the function
                func_name = next(in_it)
                func_type = types[func_name]
                if not type(func_type) is FunctionType:
                    warn(f"In process {self}: function expected as first input '{func_name}'")
                    return
            
            # ensure consistency func_type with the data
            if func_type.num_inputs != self.num_inputs:
                raise ValueError(
                    f"In process {self}: inconsistent number of inputs, "
                    f"expected {func_type.num_inputs}, got {self.num_inputs}")
            if func_type.num_outputs != len(outputs):
                raise ValueError(
                    f"In process {self}: inconsistent number of outputs, "
                    f"expected {func_type.num_outputs}, got {len(outputs)}")
            
            # remaining input are data
            i = 0 
            for in_name in in_it:
                if _is_parameter(in_name):
                    continue
                in_type = types[in_name]
                if not type(in_type) is DataType:
                    warn(f"In process {self}: data expected for input '{in_name}'")
                if func_type.in_type[i] != in_type:
                    warn(f"In process {self}: function '{func_name}' and "
                         f"data '{in_name}' are not compatible.")
                i += 1
            
            # outputs: check consistency or propagate type from function
            for i in range(len(outputs)):
                if outputs[i] in types:
                    out_type = types[outputs[i]]
                    if func_type.out_type[i] != out_type:
                        warn(f"In process {self}: inconsistent output type")
                else:
                    types[outputs[i]] = func_type.out_type[i]

def _process_error_message(msg) -> str:
    return (
        f"Invalid Processor definition: {msg}\n"
        f"Format should be: processor(inputs, outputs)"
    )


def processor(
    inputs: str | list[str],
    outputs: str | list[str] | dict[str, str],
    *,
    name: str = None,
    tags: str | Iterable[str] | None = None,
    confirms: str | list[str] | None = None,
    namespace: str = None,
    func: Callable = None,
    inputs_type: dict [str, str] = None,
) -> Processor:
    """Create a processor in the pipeline. The function to be called is the first input.
    Remaining inputs and/or outputs are the data.

    Args:
        inputs: The name or the list of the names of variables used as
            inputs to the function. The __first input__ is special and is
            the function to be called (from a "Code"). The number of names should match
            the number of arguments in the definition of the provided
            function. When dict[str, str] is provided, variable names
            will be mapped to function argument names.
        outputs: The name or the list of the names of variables used as outputs
            to the function. The number of names should match the number of
            outputs returned by the provided function. When dict[str, str]
            is provided, variable names will be mapped to the named outputs the
            function returns.
        name: Optional processor name to be used when displaying the processor in logs or
            any other visualisations.
        tags: Optional set of tags to be applied to the processor.
        confirms: Optional name or the list of the names of the datasets
            that should be confirmed. This will result in calling ``confirm()``
            method of the corresponding data set instance. Specified dataset
            names do not necessarily need to be present in the processor ``inputs``
            or ``outputs``.
        namespace: Optional processor namespace.
        func: Optional fixed function.

    Returns:
        A Processor object with mapped inputs, outputs and function.
    """
    return Processor(
        inputs,
        outputs,
        name=name,
        tags=tags,
        confirms=confirms,
        namespace=namespace,
        func=func,
        inputs_type = inputs_type,
    )


# XXX BAD LOCATION??
def make_fixed_type(node: Processor, types: TypeCatalog):
    func_name = node._func.__name__
    if func_name == "split_data":
        # P1 = {X, X_train, X_test} for first type
        # P2 = {Y, Y_train, Y_test} for second type
        assert len(node.inputs) == 3
        X = types.get_or_create(node.inputs[0])
        Y = types.get_or_create(node.inputs[1])
        types.add_function(
            # X, Y -> X_train, X_test, Y_train, Y_test
            func_name, [X, Y], [X, X, Y, Y])
        return
    elif func_name == "score":
        # P1 = {Y_test, Y_pred}, P2 = {nrmse}, P3 = {r2}
        assert 2 <= len(node.inputs) <= 3
        assert len(node.outputs) == 2
        Y_test = types[node.inputs[0]]
        Y_pred = types[node.inputs[1]]
        if Y_test != Y_pred:
            warn(f"The type of test ({Y_test}) and prediction ({Y_pred}) do not match")
        nrmse = types.add_data(node.outputs[0])
        r2 = types.add_data(node.outputs[1])
        types.add_function(
            func_name, [Y_test, Y_pred], [nrmse, r2])
        return
    else:
        # simple case where all outputs have a new data type
        inputs = [types.get_or_create(i) for i in node.inputs]
        outputs = [types.add_data(o) for o in node.outputs]
        types.add_function(func_name, inputs, outputs)