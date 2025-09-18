"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, pipeline
from kedro_umbrella import coder, processor, trainer
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
                name="t_xform_X",
                inputs=["X_train", "params:t_xform_X"],
                outputs=["X_xform", "X_inv_xform"],
            ),
            coder(
                func=xform_data,
                name="t_xform_Y",
                inputs=["Y_train", "params:t_xform_Y"],
                outputs=["Y_xform", "Y_inv_xform"],
            ),
            processor(
                name="reduce_X", inputs=["X_xform", "X_train"], outputs="X_train_red"
            ),
            processor(
                name="reduce_Y", inputs=["Y_xform", "Y_train"], outputs="Y_train_red"
            ),
            trainer(
                func=pytorch_trainer,
                name="t_trainer",
                inputs=["X_train_red", "Y_train_red", "params:t_trainer"],
                outputs = "model"
            ),
            # # TESTING PIPELINE
            # processor(name = "X_test_red", inputs=["X_xform", "X_test"], outputs="X_test_red"),
            # processor(name = "Y_pred_red", 
            #           inputs=["infer", "X_test_red"],
            #           outputs="Y_pred_red"),
            # processor(
            #     name = "Y_pred", inputs=["Y_inv_xform", "Y_pred_red"], outputs="Y_pred"),
            # processor(
            #     func=score,
            #     name="score",
            #     inputs=["Y_test", "Y_pred", "params:score"],
            #     outputs=["nrmse", "r2"],
            # ),
            processor(
                func = sensitivity_analysis,
                name = "sensitivity",
                inputs=["model", "params:sensitivity"],
                outputs = "top_samples"
            )
        ]
    )
