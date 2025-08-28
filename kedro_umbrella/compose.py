"""This module allow to create Composer nodes as part of Kedro pipelines.
"""

from typing import Any, Callable, Iterable

from kedro.pipeline.node import Node
from kedro_umbrella.types import *
from kedro.pipeline.modular_pipeline import _is_parameter


class ComposedFunction:
    def __init__(self, *functions: Callable):
        if not functions:
            raise ValueError("ComposedFunctions requires at least one function")
        # store a flat list of callables
        self.functions: list[Callable] = list(functions)

    def __call__(self, input_):
        result = input_
        for function in self.functions:
            result = function(result)
        return result

def the_composer(*functions: Callable) -> Callable:
    return ComposedFunction(*functions)

class Composer(Node):
    """``Composer`` is an extension of Node to compose function 
    """

    def __init__(
        self,
        inputs: list[str] | dict[str, str],
        outputs: str,
        *,
        name: str = None,
        tags: str | Iterable[str] | None = None,
        confirms: str | list[str] | None = None,
        namespace: str = None,
    ):
        """Create a Composer in the pipeline by providing the list of functions to be composed. 
        A Composer will generate a new function that performs the composition of all the input functions by calling them in the passed order.

        Example:
            - inputs = [f1, f2, f3]
            The new function will call f1, f2, f3 in this order with the given composition operator *. Let 'in' be the input when the function is called. The result is:
                r1 = f_1(in)
                r2 = r1 * f2(r1)
                r3 = r2 * f3(f2)
            Composition "*" operator supported are: plus, minus and plain. If 'plain' is used we've
                r1 = f1(in)
                r2 = f2(r1)
                r3 = f3(r2)
            Or r3 = f3(f2(f1(in)))

            For typing, we need to check:
                - #inputs(f2) == #outputs(f1) # same with all subsequent
                - all inputs are functions
            
        Args:
            inputs: The functions to be used as input
            outputs: The composed function 
            operator: The composition operator
            name: Optional Composer name to be used when displaying the Composer in
                logs or any other visualisations.
            tags: Optional set of tags to be applied to the Composer.
            confirms: Optional name or the list of the names of the datasets
                that should be confirmed. This will result in calling
                ``confirm()`` method of the corresponding data set instance.
                Specified dataset names do not necessarily need to be present
                in the Composer ``inputs`` or ``outputs``.
            namespace: Optional Composer namespace.

        Raises:
            ValueError: Raised in the following cases:
                a) When the provided arguments do not conform to
                the format suggested by the type hint of the argument.
                b) When the Composer produces multiple outputs with the same name.
                c) When an input has the same name as an output.
                d) When the given Composer name violates the requirements:
                it must contain only letters, digits, hyphens, underscores
                and/or fullstops.

        """
        if not isinstance(inputs, (list, dict)):
            raise ValueError(f"Invalid input type")
        if not isinstance(outputs, str):
            raise ValueError(
                    f"'outputs' type must be one a String, "
                    f"not '{type(outputs).__name__}'."
            )
        if len(inputs) < 2:
            raise ValueError(f"At least two inputs required, found {len(inputs)}")

        super().__init__(
            the_composer,
            inputs,
            outputs,
            name=name,
            tags=tags,
            confirms=confirms,
            namespace=namespace,
        )

    def __repr__(self):  # pragma: no cover
        return (
            f"Composer({self._func_name}, {repr(self._inputs)}, {repr(self._outputs)}, "
            f"{repr(self._name)})"
        )

    def _copy(self, **overwrite_params):
        """
        Helper function to copy the Composer, replacing some values.
        """
        params = {
            "inputs": self._inputs,
            "outputs": self._outputs,
            "name": self._name,
            "namespace": self._namespace,
            "tags": self._tags,
            "confirms": self._confirms,
        }
        params.update(overwrite_params)
        return Composer(**params)

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
            Exception: Any exception thrown during execution of the node.

        Returns:
            All produced node outputs are returned in a dictionary, where the
            keys are defined by the node outputs.

        """
        self._logger.info("Running Composer: %s", str(self))
        outputs = None

        if not isinstance(inputs, dict):
            raise ValueError(
                f"Composer.run() expects a dictionary, "
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
                        f"Composer expected callable output but got {type(outputs[out])} instead!"
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
                if not type(the_type) is FunctionType:
                    warn(f"In Composer {self}: Function expected as {msg} input '{input}'")
                in_types.append(the_type)
            return in_types

        self._logger.info("Checking Composer: %s", self)
        inputs = check_input(self.inputs)
        # Result is F = f1 -> f2 -> f3 
        # input of required by f1 and output as f3
        in_type = inputs[0]
        out_type = inputs[-1]
        out_name = next(iter(self.outputs))
        types.add_function(out_name, in_type, out_type)


def composer(
    inputs: list[str] | dict[str, str],
    outputs: str,
    *,
    name: str = None,
    tags: str | Iterable[str] | None = None,
    confirms: str | list[str] | None = None,
    namespace: str = None,
) -> Composer:
    """Create a Composer in the pipeline by providing the list of functions to be composed. 
    A Composer will generate a new function that performs the composition of all the input functions by calling them in the passed order.

    Example:
        - inputs = [f1, f2, f3]
        The new function will call f1, f2, f3 in this order with the given composition operator *. Let 'in' be the input when the function is called. The result is:
            r1 = f_1(in)
            r2 = r1 * f2(r1)
            r3 = r2 * f3(f2)
        Composition "*" operator supported are: plus, minus and plain. If 'plain' is used we've
            r1 = f1(in)
            r2 = f2(r1)
            r3 = f3(r2)
        Or r3 = f3(f2(f1(in)))

        For typing, we need to check:
            - #inputs(f2) == #outputs(f1) # same with all subsequent
            - all inputs are functions
        
    Args:
        inputs: The functions to be used as input
        outputs: The composed function 
        operator: The composition operator
        name: Optional Composer name to be used when displaying the Composer in
            logs or any other visualisations.
        tags: Optional set of tags to be applied to the Composer.
        confirms: Optional name or the list of the names of the datasets
            that should be confirmed. This will result in calling
            ``confirm()`` method of the corresponding data set instance.
            Specified dataset names do not necessarily need to be present
            in the Composer ``inputs`` or ``outputs``.
        namespace: Optional Composer namespace.

    Raises:
        ValueError: Raised in the following cases:
            a) When the provided arguments do not conform to
            the format suggested by the type hint of the argument.
            b) When the Composer produces multiple outputs with the same name.
            c) When an input has the same name as an output.
            d) When the given Composer name violates the requirements:
            it must contain only letters, digits, hyphens, underscores
            and/or fullstops.

    """
    return Composer(
        inputs,
        outputs,
        name=name,
        tags=tags,
        confirms=confirms,
        namespace=namespace,
    )
