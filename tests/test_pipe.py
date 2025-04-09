# pylint: disable=unused-argument

import pytest

from kedro_umbrella import coder, processor, trainer
from kedro_umbrella.types import TypeCatalog, DataType, FunctionType

from kedro.pipeline import Pipeline, pipeline

from kedro.io import DataCatalog
from kedro_umbrella.checker import SequentialChecker

def one_in_one_out(arg):
    return arg

def two_in_one_out(arg1, arg2):
    return arg1 + arg2

def three_in_one_out(arg1, arg2, arg3):
    return arg1 + arg2 + arg3

def test_pipe():
    # COMMON SETUP
    types : TypeCatalog = TypeCatalog()
    types.add_data("X1")
    types.add_data("X2")
    types.add_data("Y")

    # declare pipe
    pipe = pipeline(
        [
            coder(
                func = one_in_one_out, 
                name = "code",
                inputs = "X1",
                outputs = ["X1_enc", "X1_dec"]
            ),
            processor(
                name = "enc_proc",
                inputs = ["X1_enc", "X1"],
                outputs = "X1_n"
            ),
            trainer(
                func = three_in_one_out,
                name = "trainer",
                inputs = ["X1", "X2", "Y"],
                numX = 2,
                outputs = "predict"
            ),
            processor(
                name = "tr_proc",
                inputs_type = {"X1_n": "X1"},
                inputs = ["predict", "X1_n", "X2"],
                outputs = "Ypred"
            ),
        ]
    )

    catalog = DataCatalog()
    catalog.add_feed_dict(
        {
            "X1": 1,
            "X2": 2,
            "Y": 10,
        }
    )
    
    SequentialChecker(types).run(pipe, catalog)

    # check that all ids for X1_n have been replace by the id for X1

test_pipe() 