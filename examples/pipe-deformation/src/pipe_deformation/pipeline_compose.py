"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, pipeline
from kedro_umbrella import coder, processor, trainer, composer
from kedro_umbrella.library import *

# pro => doesn't need to code the node +- => one still need to think in advance
# about wanting to output some valid function that can be called further ahead
# dis => longer pipeline


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            # TRAINING PIPELINE
            processor(
                func=split_data,
                name="split_data",
                inputs=["displ", "eps", "params:split_data"],
                outputs=["X_train", "X_test", "Y_train", "Y_test"],
            ),
            coder(
                func=xform_data,
                name="xform_X",
                inputs=["X_train", "params:xform_X"],
                outputs=["X_xform", "X_inv_xform"],
            ),
            coder(
                func=xform_data,
                name="xform_Y",
                inputs=["Y_train", "params:xform_Y"],
                outputs=["Y_xform", "Y_inv_xform"],
            ),
            processor(
                name="reduce_X", inputs=["X_xform", "X_train"], outputs="X_train_red"
            ),
            processor(
                name="reduce_Y", inputs=["Y_xform", "Y_train"], outputs="Y_train_red"
            ),
            trainer(
                func=basic_trainer,
                name="trainer",
                inputs=["X_train_red", "Y_train_red", "params:trainer"],
                outputs="regressor",
            ),
            # TESTING PIPELINE
            composer(
                inputs=["X_xform", "regressor", "Y_inv_xform"],
                outputs="comp_model"
            ),
            processor(inputs=["comp_model", "X_test"], outputs="Y_pred"),
            processor(
                func=score,
                name="score",
                inputs=["Y_test", "Y_pred", "params:score"],
                outputs=["nrmse", "r2"],
            ),
        ]
    )
