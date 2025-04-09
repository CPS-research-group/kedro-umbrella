"""``SequentialChecker`` is an ``AbstractRunner`` implementation. It can check
the pipeline and ensure process/code/train are used consistently.
"""

from collections import Counter
from itertools import chain
from typing import Any

from kedro.io import AbstractDataset, DataCatalog, MemoryDataset
from kedro.pipeline import Pipeline
from kedro.pipeline.node import Node
from kedro.runner.runner import AbstractRunner
from pluggy import PluginManager

from kedro_umbrella import Coder, Processor, Trainer
from kedro_umbrella.types import TypeCatalog, DataType, FunctionType
from kedro.pipeline.modular_pipeline import _is_parameter

def warn_on_fail(condition, message=""):
    from warnings import warn
    if not condition:
        warn(message)

class SequentialChecker(AbstractRunner):
    """``SequentialRunner`` is an ``AbstractRunner`` implementation. It can
    be used to run the ``Pipeline`` in a sequential manner using a
    topological sort of provided nodes.
    """

    def __init__(self, types : TypeCatalog, is_async: bool = False):
        """Instantiates the runner classs.

        Args:
            is_async: If True, the node inputs and outputs are loaded and saved
                asynchronously with threads. Defaults to False.

        """
        # check if the types are already serialized
        self._types : TypeCatalog = types
        import os; import pickle
        if os.path.exists("types.pickle") and (os.getenv("UMBRL_LOAD_TYPES") != None):
            with open("types.pickle", "rb") as f:
                self._types = pickle.load(f)

        super().__init__(is_async=is_async)

    def create_default_data_set(self, ds_name: str) -> AbstractDataset:
        """Factory method for creating the default data set for the runner.

        Args:
            ds_name: Name of the missing data set

        Returns:
            An instance of an implementation of AbstractDataset to be used
            for all unregistered data sets.

        """
        return MemoryDataset()

    def run(
        self,
        pipeline: Pipeline,
        catalog: DataCatalog,
        hook_manager: PluginManager = None,
        session_id: str = None,
    ) -> None:
        """Run the ``Pipeline`` using the datasets provided by ``catalog``
        and save results back to the same objects.

        Args:
            pipeline: The ``Pipeline`` to run.
            catalog: The ``DataCatalog`` from which to fetch data.
            hook_manager: The ``PluginManager`` to activate hooks.
            session_id: The id of the session.

        Raises:
            ValueError: Raised when ``Pipeline`` inputs cannot be satisfied.

        Returns:
            Any node outputs that cannot be processed by the ``DataCatalog``.
            These are returned in a dictionary, where the keys are defined
            by the node outputs.

        """
        catalog = catalog.shallow_copy()

        # Check which datasets used in the pipeline are in the catalog or match
        # a pattern in the catalog
        registered_ds = [ds for ds in pipeline.datasets() if ds in catalog]

        # Check if there are any input datasets that aren't in the catalog and
        # don't match a pattern in the catalog.
        unsatisfied = pipeline.inputs() - set(registered_ds)

        if unsatisfied:
            raise ValueError(
                f"Pipeline input(s) {unsatisfied} not found in the DataCatalog"
            )

        # Check if there's any output datasets that aren't in the catalog and don't match a pattern
        # in the catalog.
        unregistered_ds = pipeline.datasets() - set(registered_ds)

        # Create a default dataset for unregistered datasets
        for ds_name in unregistered_ds:
            catalog.add(ds_name, self.create_default_data_set(ds_name))

        if self._is_async:
            self._logger.info(
                "Asynchronous mode is enabled for loading and saving data"
            )
        self._run(pipeline, catalog, hook_manager, session_id)

        self._logger.info("Pipeline execution completed successfully.")

    def _run(
        self,
        pipeline: Pipeline,
        catalog: DataCatalog,
        hook_manager: PluginManager,
        session_id: str = None,
    ) -> None:
        """The method implementing sequential pipeline running.

        Args:
            pipeline: The ``Pipeline`` to run.
            catalog: The ``DataCatalog`` from which to fetch data.
            hook_manager: The ``PluginManager`` to activate hooks.
            session_id: The id of the session.

        Raises:
            Exception: in case of any downstream node failure.
        """
        import pickle
        import os
        nodes = pipeline.nodes
        done_nodes = set()

        load_counts = Counter(chain.from_iterable(n.inputs for n in nodes))
        for exec_index, node in enumerate(nodes):
            try:
                self.backtrack_if_needed(pipeline, node)
                self.check_node(node)
                # run_node(node, catalog, hook_manager, self._is_async, session_id)
                done_nodes.add(node)
            except Exception:
                self._suggest_resume_scenario(pipeline, done_nodes, catalog)
                raise

            # decrement load counts and release any data sets we've finished with
            for data_set in node.inputs:
                load_counts[data_set] -= 1
                if load_counts[data_set] < 1 and data_set not in pipeline.inputs():
                    catalog.release(data_set)
            for data_set in node.outputs:
                if load_counts[data_set] < 1 and data_set not in pipeline.outputs():
                    catalog.release(data_set)

            self._logger.info(
                "Completed %d out of %d tasks", exec_index + 1, len(nodes)
            )
        # serialize the types as pickle
        with open("types.pickle", "wb") as f:
            pickle.dump(self._types, f)
        
    def backtrack_if_needed(
            self,
            pipeline: Pipeline,
            node: Node
    ):
        if type(node) is not Processor: 
            return
        if not node._inputs_type:
            return
        for annot in node._inputs_type.items():
            cur_name = annot[0]; new_name = annot[1]
            cur_name_orig_id = self._types[cur_name]

            self._types[cur_name] = self._types[new_name]

            # find what is generating cur_name
            # replace cur_name w/ new_name
            out_pipe = pipeline.to_outputs(cur_name)
            for node in out_pipe.nodes:
                for out in node.outputs:
                    type_ = self._types[out]
                    if type(type_) is FunctionType:
                        # replace elements in type.out_type w/ cur_name_orig_id
                        # with the self._types[new_name]
                        if cur_name_orig_id in type_.out_type:
                            type_.out_type = [
                                self._types[new_name] 
                                if t == cur_name_orig_id else t for t in type_.out_type]
                            self._types[out] = type_
                        elif cur_name_orig_id in type_.in_type:
                            type_.in_type = [
                                self._types[new_name] 
                                if t == cur_name_orig_id else t for t in type_.in_type]
                            self._types[out] = type_
                    else:
                        self._types[out] = self._types[new_name]
                        
    def check_node(
        self,
        node: Node
    ):
        types = self._types

        if type(node) is Node:
            self._logger.warning(f"Found Kedro node {node} in Builder pipeline")
            self._logger.info("Checking node: %s", node)
            for data in node.inputs + node.outputs:
                if _is_parameter(data):
                    continue
                if data in types:
                    warn_on_fail(
                        type(types[data]) is DataType,
                        f"In node {node}: Data expected on input {data}",
                    )
                else:
                    types.add_data(data)
        else:
            node.check(types)
            return