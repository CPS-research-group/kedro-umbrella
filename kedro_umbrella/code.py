"""This module allow to create Coder nodes as part of Kedro pipelines.
"""

from typing import Any, Callable, Iterable

from kedro.pipeline.node import Node
from kedro.pipeline.modular_pipeline import _is_parameter
from kedro_umbrella.types import *


class Coder(Node):
    """``Coder`` is an extension of Node to generate function as output
    run user-provided functions as part of Kedro pipelines.

    A Coder block is used to represent learning a reduced basis (or unsupervised clustering) and the associated projection and inverse projection.
     
    It should take as input the data from which to compute the reduced basis and it can return:
        - two functions representing the encoder and decoder functions (i.e., the projection and inverse projection onto the reduced basis, respectively). 
        - one function representing the encoder function, decoder is not available. 
    """

    def __init__(self,
        func: Callable,
        inputs: str | list[str] | dict[str, str],
        outputs: str | list[str] | dict[str, str],
        *,
        name: str = None,
        tags: str | Iterable[str] | None = None,
        confirms: str | list[str] | None = None,
        namespace: str = None):
        """ Create a Coder in the pipeline by providing a function to be called along with variable names for inputs and/or outputs.

        Args:
            func: A function that corresponds to the node logic.
                The function should have at least one input or output.
            inputs: The name or the list of the names of variables used as
                inputs to the function. The number of names should match
                the number of arguments in the definition of the provided
                function. When dict[str, str] is provided, variable names
                will be mapped to function argument names.
            outputs: The name or the list of the names of variables used
                as outputs of the function. The number of names should match
                the number of outputs returned by the provided function.
                When dict[str, str] is provided, variable names will be mapped
                to the named outputs the function returns.
            name: Optional node name to be used when displaying the node in
                logs or any other visualisations. Valid node name must contain
                only letters, digits, hyphens, underscores and/or fullstops.
            tags: Optional set of tags to be applied to the node. Valid node tag must
                contain only letters, digits, hyphens, underscores and/or fullstops.
            confirms: Optional name or the list of the names of the datasets
                that should be confirmed. This will result in calling
                ``confirm()`` method of the corresponding dataset instance.
                Specified dataset names do not necessarily need to be present
                in the node ``inputs`` or ``outputs``.
            namespace: Optional node namespace.

        Raises:
            ValueError: Raised in the following cases:
                a) When the provided arguments do not conform to
                the format suggested by the type hint of the argument.
                b) When the node produces multiple outputs with the same name.
                c) When an input has the same name as an output.
                d) When the given node name violates the requirements:
                it must contain only letters, digits, hyphens, underscores
                and/or fullstops.
                e) When the Coder does not have 1 or 2 outputs. 
        """


        if not isinstance(inputs, (str, list, dict)):
            raise ValueError(f"Invalid input type")
        if not isinstance(outputs, (str, list, dict)):
            raise ValueError(f"Invalid output type")
        if isinstance(outputs, (list, dict)) and not (1 <= len(outputs) <= 2):
            raise ValueError(f"Expected 1 or 2 outputs, found {len(outputs)}")

        super().__init__(func,
                inputs,
                outputs,
                name=name,
                tags=tags,
                confirms=confirms,
                namespace=namespace)


    def _copy(self, **overwrite_params):
        """
        Helper function to copy the coder, replacing some values.
        """
        params = {
            "func": self._func,
            "inputs": self._inputs,
            "outputs": self._outputs,
            "name": self._name,
            "namespace": self._namespace,
            "tags": self._tags,
            "confirms": self._confirms,
        }
        params.update(overwrite_params)
        return Coder(**params)

    def __repr__(self):  # pragma: no cover
        return (
            f"Coder({self._func_name}, {repr(self._inputs)}, {repr(self._outputs)}, "
            f"{repr(self._name)})"
        )

    def run(self, inputs: dict[str, Any] = None) -> dict[str, Any]:
        """Run this node using the provided inputs and return its results
        in a dictionary.

        Args:
            inputs: Dictionary of inputs as specified at the creation of
                the node.

        Raises:
            ValueError: In the following cases:
                a) The Coder output are not callable functions.

                b) The node function inputs are incompatible with the node
                input definition.
                Example 1: node definition input is a list of 2
                DataFrames, whereas only 1 was provided or 2 different ones
                were provided.
                a) The node function outputs are incompatible with the node
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
        self._logger.info("Running coder: %s", str(self))

        outputs = None

        if not isinstance(inputs, dict):
            raise ValueError(
                f"Coder.run() expects a dictionary, "
                f"but got {type(inputs)} instead"
            )

        try:
            inputs = {} if inputs is None else inputs
            if isinstance(self._inputs, str):
                outputs = self._run_with_one_input(inputs, self._inputs)
            elif isinstance(self._inputs, list):
                outputs = self._run_with_list(inputs, self._inputs)
            elif isinstance(self._inputs, dict):
                outputs = self._run_with_dict(inputs, self._inputs)

            outputs = self._outputs_to_dictionary(outputs)
            for out in outputs:
                # check dict values are callable
                if not callable(outputs[out]):
                    raise ValueError(
                        f"Coder expected callable outputs but got {type(outputs[out])} instead!"
                    )
            return outputs

        # purposely catch all exceptions
        except Exception as exc:
            self._logger.error("Coder '%s' failed with error: \n%s", str(self), str(exc))
            raise exc

    def check(self, types: TypeCatalog) -> None:
        from warnings import warn

        self._logger.info("Checking coder: %s", self)
        inputs = self.inputs
        outputs = self.outputs

        # check the inputs are Data
        in_types = []
        for input in inputs:
            if _is_parameter(input):
                continue
            in_type = types[input]
            if not type(in_type) is DataType:
                warn(f"In coder {self}: input {input} is not data")
                return
            in_types.append(in_type)

        # propagate the output types
        out_types = types.make_data()
        # F1: in_type -> out_type
        out_it = iter(outputs)
        first_func = next(out_it)
        types.add_function(first_func, in_types, out_types)
        # F2: out_type -> in_type (optional)
        if len(outputs) == 1:
            return
        second_func = next(out_it)
        types.add_function(second_func, out_types, in_types)


def coder(
    func: Callable,
    inputs: str | list[str] | dict[str, str],
    outputs: str | list[str] | dict[str, str],
    *,
    name: str = None,
    tags: str | Iterable[str] | None = None,
    confirms: str | list[str] | None = None,
    namespace: str = None,
) -> Coder:
    """Create a Coder in the pipeline by providing a function to be called
    along with variable names for inputs and/or outputs.

    Args:
        func: A function that corresponds to the coder logic. The function
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
        name: Optional coder name to be used when displaying the coder in logs or
            any other visualisations.
        tags: Optional set of tags to be applied to the coder.
        confirms: Optional name or the list of the names of the datasets
            that should be confirmed. This will result in calling ``confirm()``
            method of the corresponding data set instance. Specified dataset
            names do not necessarily need to be present in the coder ``inputs``
            or ``outputs``.
        namespace: Optional coder namespace.

    Returns:
        A Coder object with mapped inputs, outputs and function.
    """
    return Coder(
        func,
        inputs,
        outputs,
        name=name,
        tags=tags,
        confirms=confirms,
        namespace=namespace,
    )
