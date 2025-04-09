from kedro.pipeline import Pipeline, node, pipeline
from kedro_umbrella import coder, processor, trainer
from kedro_umbrella.library import *
from .nodes import *

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            # Data Retrieval
            node(
                func=get_data,
                inputs = "params:get_data",
                outputs=["X", "Y"],
                name="get_data",
            ),
            # Data Splitting
            processor(
                func=split_data,
                inputs=["X", "Y", "params:split_data"],
                outputs=["X_train", "X_test", "Y_train", "Y_test"],
                name="split_data",
            ),
            # Data Transformation
            coder(
                func=xform_data,
                inputs=["X_train", "params:xform_data"],
                outputs=["X_train_scaler", "X_train_inv_scaler"],
                name="xform_X_train",
            ),
            coder(
                func=xform_data,
                inputs=["Y_train", "params:xform_data"],
                outputs=["Y_train_scaler", "Y_train_inv_scaler"],
                name="xform_Y_train",
            ),
            processor(
                name="reduce_X_train", inputs=["X_train_scaler", "X_train"], outputs="X_train_red"
            ),
            # Model Training
            trainer(
                func=basic_trainer,
                inputs=["X_train_red", "Y_train", "params:basic_trainer"],
                outputs="regressor",
                name="train_model",
            ),
            # TESTING PIPELINE
            processor(
                inputs=["X_train_scaler", "X_test"],
                outputs="X_test_xform",
                name="xform_X_test"
            ),
            processor(
                inputs=["regressor", "X_test_xform"],
                outputs="Y_pred_xform",
                name="predict"
            ),
            # Evaluation
            processor(
                func=score,
                name="score",
                inputs=["Y_test", "Y_pred_xform", "params:score"],
                outputs=["mse", "r2"],
            )
        ]
    )
