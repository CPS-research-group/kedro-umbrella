"""This module allow to create Trainer nodes as part of Kedro pipelines.
"""

from typing import Any, Callable, Iterable

from kedro.pipeline.node import Node
from kedro_umbrella.types import *
from kedro.pipeline.modular_pipeline import _is_parameter


class Trainer(Node):
    """``Trainer`` is an extension of Node to generate function as output
    run user-provided functions as part of Kedro pipelines. A Trainer represents learning a function with supervised ML. 
    """

    def __init__(
        self,
        func: Callable,
        inputs: list[str] | dict[str, str],
        outputs: str | list[str],
        *,
        name: str = None,
        tags: str | Iterable[str] | None = None,
        confirms: str | list[str] | None = None,
        namespace: str = None,
        numX: int = 1
    ):
        """Create a trainer in the pipeline by providing a function to be called
        along with variable names for inputs and/or outputs.

        A Trainer has len(inputs) >= 2 inputs and one output.
        - The input are the supervised $(X, Y)$ pairs.
        - The number of X values is `numX` with `numX` < len(inputs).
        - The number of Y values is given by the remaining inputs: `len(inputs) - numX`.
        - The output is a function that predicts Y given X.
        - The algorithm to learn the prediction function is provided in func.
        Args:
            func: A function that corresponds to the trainer logic.
                The function should have at least one input or output.
            numX: The number of inputs that are to be trea
            inputs: The name or the list of the names of variables used as
                inputs to the function. The number of names should match
                the number of arguments in the definition of the provided
                function. When dict[str, str] is provided, variable names
                will be mapped to function argument names.
            outputs: The name or the list of the names of variables used
                as outputs to the function. The number of names should match
                the number of outputs returned by the provided function.
                When dict[str, str] is provided, variable names will be mapped
                to the named outputs the function returns.
            name: Optional trainer name to be used when displaying the trainer in
                logs or any other visualisations.
            tags: Optional set of tags to be applied to the trainer.
            confirms: Optional name or the list of the names of the datasets
                that should be confirmed. This will result in calling
                ``confirm()`` method of the corresponding data set instance.
                Specified dataset names do not necessarily need to be present
                in the trainer ``inputs`` or ``outputs``.
            namespace: Optional trainer namespace.

        Raises:
            ValueError: Raised in the following cases:
                a) When the provided arguments do not conform to
                the format suggested by the type hint of the argument.
                b) When the trainer produces multiple outputs with the same name.
                c) When an input has the same name as an output.
                d) When the given trainer name violates the requirements:
                it must contain only letters, digits, hyphens, underscores
                and/or fullstops.
                e) the number of inputs is less than 2 (at least X and Y required)
                f) numX > len(inputs) - 1. At least one Y input is required. 

        """
        if not isinstance(inputs, (list, dict)):
            raise ValueError(f"Invalid input type")
        if not isinstance(outputs, (str, list)):
            raise ValueError(
                    f"'outputs' type must be one of [String, List], "
                    f"not '{type(outputs).__name__}'."
            )
        if len(inputs) < 2:
            raise ValueError(f"At least two inputs required, found {len(inputs)}")

        super().__init__(
            func,
            inputs,
            outputs,
            name=name,
            tags=tags,
            confirms=confirms,
            namespace=namespace,
        )
        if numX > len(inputs) - 1:
            raise ValueError(
                f"numX={numX} must be <= {len(inputs) - 1}"
                " (at least one Y is required)"
            )
        self.numX = numX
        self.numY = len(inputs) - numX

    def __repr__(self):  # pragma: no cover
        return (
            f"Trainer({self._func_name}, {repr(self._inputs)}, {repr(self._outputs)}, "
            f"{repr(self._name)})"
        )

    def _copy(self, **overwrite_params):
        """
        Helper function to copy the trainer, replacing some values.
        """
        params = {
            "func": self._func,
            "inputs": self._inputs,
            "outputs": self._outputs,
            "name": self._name,
            "namespace": self._namespace,
            "tags": self._tags,
            "confirms": self._confirms,
            "numX": self.numX,
        }
        params.update(overwrite_params)
        return Trainer(**params)

    def run(self, inputs: dict[str, Any] = None) -> dict[str, Any]:
        """Run this node using the provided inputs and return its results
        in a dictionary.

        Args:
            inputs: Dictionary of inputs as specified at the creation of
                the node.

        Raises:
            ValueError: In the following cases:
                a) The node function inputs are incompatible with the node
                input definition.
                Example 1: node definition input is a list of 2
                DataFrames, whereas only 1 was provided or 2 different ones
                were provided.
                b) The node function outputs are incompatible with the node
                output definition.
                Example 1: node function definition is a dictionary,
                whereas function returns a list.
                Example 2: node definition output is a list of 5
                strings, whereas the function returns a list of 4 objects.
                c) The output of the Trainer is not callable. 
            Exception: Any exception thrown during execution of the node.

        Returns:
            All produced node outputs are returned in a dictionary, where the
            keys are defined by the node outputs.

        """
        self._logger.info("Running trainer: %s", str(self))
        outputs = None

        if not isinstance(inputs, dict):
            raise ValueError(
                f"Trainer.run() expects a dictionary, "
                f"but got {type(inputs)} instead"
            )

        try:
            inputs = {} if inputs is None else inputs
            if isinstance(self._inputs, list):
                outputs = self._run_with_list(inputs, self._inputs)
            elif isinstance(self._inputs, dict):
                outputs = self._run_with_dict(inputs, self._inputs)

            outputs = self._outputs_to_dictionary(outputs)
            for out in outputs:
                # check dict values are callable
                if not callable(outputs[out]):
                    raise ValueError(
                        f"Trainer expected callable output but got {type(outputs[out])} instead!"
                    )
            return outputs

        # purposely catch all exceptions
        except Exception as exc:
            self._logger.error("Tran '%s' failed with error: \n%s", str(self), str(exc))
            raise exc

    def check(self, types: TypeCatalog) -> None:
        from warnings import warn

        def check_input(self, inputs, msg):
            in_types = []
            for input in inputs:
                if _is_parameter(input):
                    continue
                the_type = types[input]
                if not type(the_type) is DataType:
                    warn(f"In train {self}: Data expected as {msg} input '{input}'")
                in_types.append(the_type)
            return in_types

        self._logger.info("Checking trainer: %s", self)
        inputs = self.inputs
        outputs = self.outputs

        X_inputs = inputs[: self.numX]
        X_types = check_input(self, X_inputs, "X")
        Y_inputs = inputs[self.numX :]
        Y_types = check_input(self, Y_inputs, "Y")
        out_name = next(iter(outputs))
        types.add_function(out_name, X_types, Y_types)


def trainer(
    func: Callable,
    inputs: list[str] | dict[str, str],
    outputs: str,
    *,
    name: str = None,
    tags: str | Iterable[str] | None = None,
    confirms: str | list[str] | None = None,
    namespace: str = None,
    numX: int = 1,
) -> Trainer:
    """Create a trainer in the pipeline by providing a function to be called
    along with variable names for inputs and/or outputs.

    Args:
        func: A function that corresponds to the trainer logic. The function
            should have at least one input or output.
        inputs: The name or the list of the names of variables used as inputs
            to the function. The number of names should match the number of
            arguments in the definition of the provided function. When
            dict[str, str] is provided, variable names will be mapped to
            function argument names.
        outputs: The name or the list of the names of variables used as outputs
            to the function. The number of names should match the number of
            outputs returned by the provided function. When dict[str, str]
            is provided, variable names will be mapped to the named outputs the
            function returns.
        name: Optional trainer name to be used when displaying the trainer in logs or
            any other visualisations.
        tags: Optional set of tags to be applied to the trainer.
        confirms: Optional name or the list of the names of the datasets
            that should be confirmed. This will result in calling ``confirm()``
            method of the corresponding data set instance. Specified dataset
            names do not necessarily need to be present in the trainer ``inputs``
            or ``outputs``.
        namespace: Optional trainer namespace.

    Returns:
        A Trainer object with mapped inputs, outputs and function.
    """
    return Trainer(
        func,
        inputs,
        outputs,
        name=name,
        tags=tags,
        confirms=confirms,
        namespace=namespace,
        numX=numX,
    )
