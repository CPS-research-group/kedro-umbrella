from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn import svm
import numpy as np
import logging

logger = logging.getLogger(__name__)

class MLPRegressorUmbrl(MLPRegressor):
    def predict(self, X):
        y_pred = super().predict(X)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        return y_pred

def _train_model(X, Y, parameters):
    model_type = parameters["model"]
    random_state = parameters.get("random_state", None)

    if model_type == "mlp":
        model = MLPRegressorUmbrl(
            hidden_layer_sizes=parameters.get("hidden_layer_sizes", (50, 50)),
            max_iter=parameters.get("max_iter", 50000),
            random_state=random_state,
        )
    elif model_type == "mlp1":
        model = MLPRegressorUmbrl(
            max_iter=parameters.get("max_iter", 10000),
            hidden_layer_sizes=parameters.get("hidden_layer_sizes", (100, 100, 100)),
            activation=parameters.get("activation", "relu"),
            solver=parameters.get("solver", "lbfgs"),
            random_state=random_state,
        )
    elif model_type == "mlp2":
        model = MLPRegressorUmbrl(
            max_iter=parameters.get("max_iter", 10000),
            hidden_layer_sizes=parameters.get("hidden_layer_sizes", (50, 50)),
            activation=parameters.get("activation", "relu"),
            learning_rate_init=parameters.get("learning_rate_init", 0.001),
            random_state=random_state,
        )
    elif model_type == "lr":
        model = LinearRegression()
    elif model_type == "dt":
        model = DecisionTreeRegressor(
            max_depth=parameters.get("max_depth", None),
            min_samples_split=parameters.get("min_samples_split", 2),
            random_state=random_state,
        )
    elif model_type == "svr":
        model = svm.SVR(
            C=parameters.get("C", 1.0), epsilon=parameters.get("epsilon", 0.1)
        )
    # TODO potentially train many models and pick the best one here
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    if (Y.ndim == 2) and (Y.shape[1] == 1):
        Y = np.ravel(Y)
    return model.fit(X, Y)


def basic_trainer(X, Y, parameters):
    """
    Train a model with given data and parameters. This wraps the Scikit-Learn implementation for access in the Builder. 

    Args:
        X (pd.DataFrame or np.ndarray): Feature matrix.
        Y (pd.Series or np.ndarray): Target vector.
        parameters (dict): Dictionary containing model parameters and training options:
            - "random_state" (int, optional): Random state for reproducibility
            - "model" (str): Type of model to train. Supported values are "mlp", "dt", "svr". Additional parameter per model are available.
                - "mlp" is a multi-layer perceptron,
                - "dt" is a decision tree,
                - "svr" is support-vector regression.

    Additional model config:
        - if model == "mlp":
            - "hidden_layer_sizes" (tuple, optional): Sizes of hidden layers.
            - "learning_rate_init" (float, optional): Initial learning rate.
            - "max_iter" (int, optional): Maximum number of iterations.
        - if model == "dt":
            - "max_depth" (int, optional): Maximum depth of the decision tree.
            - "min_samples_split" (int, optional): Minimum number of samples required to split an internal node.
        - if model == "svr":
            - "C" (float, optional): Regularization parameter for SVR models.
            - "epsilon" (float, optional): Epsilon in the epsilon-SVR model.
        
    Returns:
        callable: A function that can be used to make predictions with the trained model.
    """

    return _train_model(X, Y, parameters).predict
