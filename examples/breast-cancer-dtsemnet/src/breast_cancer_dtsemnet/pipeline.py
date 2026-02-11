"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, pipeline
from kedro_umbrella import coder, processor, trainer
from kedro_umbrella.library import *

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            # TRAINING PIPELINE
            processor(
                func=split_data,
                inputs=["diagnostic", "cancer_type", "params:split_data"],
                outputs=["X_train", "X_test", "Y_train", "Y_test"],
            ),
            coder(
                func=xform_data,
                name="xform_X",
                inputs=["X_train", "params:xform_X"],
                outputs=["X_xform", "X_inv_xform"],
            ),
            processor(
                name="reduce_X", inputs=["X_xform", "X_train"], outputs="X_train_red"
            ),
            trainer(
                func=dtsemnet_trainer,
                name="trainer",
                inputs=["X_train_red", "Y_train", "params:trainer"],
                outputs="model",
            ),
            # TESTING PIPELINE
            processor(
                name="f_test_red",
                inputs=["X_xform", "X_test"],
                outputs="X_test_red",
            ),
            processor(
                name="f_test_pred",
                inputs=["model", "X_test_red"],
                outputs="Y_pred",
            ),
            processor(
                func=score,
                name="score",
                inputs=["Y_test", "Y_pred", "params:score"],
                outputs=["nrmse", "r2"],
            ),
        ]
    )
